import math
from dataclasses import dataclass, field
from typing import Type, Optional

from omegaconf import II

from fairseq import utils
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics
from speech_translation.criterions.cascaded_st_cross_entropy import CascadedSTLabelSmoothedCrossEntropyCriterion
from speech_translation.criterions.named_label_smoothed_cross_entropy import NamedLabelSmoothedCrossEntropy


@dataclass
class JointCascadedSpeechTranslationCriterionConfig(FairseqDataclass):
    mt_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing of mt task, 0 means no label smoothing"},
    )
    asr_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing of asr task, 0 means no label smoothing"},
    )
    st_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing of st task, 0 means no label smoothing"},
    )
    st_weight: float = field(
        default=1.0,
        metadata={"help": "lambda for weighting the st task loss term vs. mt and asr task. only applies for joint training"}
    )
    auxiliary_task_weight_schedule: Optional[str] = field(
        default=None
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ensemble_training: bool = field(
        default=False
    )


@register_criterion(
    "joint_cascaded_st", dataclass=JointCascadedSpeechTranslationCriterionConfig
)
class JointCascadedSpeechTranslationCriterion(FairseqCriterion):

    mt_criterion_cls: Type[LabelSmoothedCrossEntropyCriterion]
    asr_criterion_cls: Type[LabelSmoothedCrossEntropyCriterion]

    def __init__(self, task, mt_label_smoothing, asr_label_smoothing, st_label_smoothing, st_weight,
                 auxiliary_task_weight_schedule, sentence_avg, report_accuracy, ensemble_training=False):
        super().__init__(task)

        self.st_criterion = CascadedSTLabelSmoothedCrossEntropyCriterion(task,
                                                                         sentence_avg=sentence_avg,
                                                                         label_smoothing=st_label_smoothing,
                                                                         report_accuracy=report_accuracy,
                                                                         ensemble_training=ensemble_training)
        self.mt_criterion = None
        self.asr_criterion = None

        self.st_weight = st_weight

        self.is_joint_training = task.is_joint_training

        if self.is_joint_training:
            self.mt_criterion = NamedLabelSmoothedCrossEntropy(task, sentence_avg, mt_label_smoothing, name='mt',
                                                               report_accuracy=report_accuracy)
            self.asr_criterion = NamedLabelSmoothedCrossEntropy(task, sentence_avg, asr_label_smoothing, name='asr',
                                                                report_accuracy=report_accuracy)

            self.auxiliary_task_weight_schedule = auxiliary_task_weight_schedule

            assert self.auxiliary_task_weight_schedule in [None, "inverse_sqrt"]

    def step_auxiliary_task_weight_schedule(self, update_num):
        if self.auxiliary_task_weight_schedule is None:
            return 1.0
        if self.auxiliary_task_weight_schedule == "inverse_sqrt":
            return 1.5 * ((update_num / 100 + 1) ** -0.5)
        else:
            raise NotImplementedError()

    def update_asr_loss(self, raw_loss, update_num):
        return self.step_auxiliary_task_weight_schedule(update_num) * raw_loss

    def update_mt_loss(self, raw_loss, update_num):
        return self.step_auxiliary_task_weight_schedule(update_num) * raw_loss

    def update_st_loss(self, raw_loss):
        return self.st_weight * raw_loss

    def forward(self, model, sample, reduce=True):
        raise NotImplementedError('This criterion should not be called directly.')

    @classmethod
    def reduce_metrics(cls, logging_outputs, do_mt_training=True, do_asr_training=True) -> None:
        """Aggregate logging outputs from data parallel training."""
        if any(key.startswith('mt_') or key.startswith('asr_') for key in logging_outputs[0].keys()):
            if do_mt_training:
                cls.reduce_metrics_named(logging_outputs, 'mt')
            if do_asr_training:
                cls.reduce_metrics_named(logging_outputs, 'asr')

        CascadedSTLabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)

    @classmethod
    def reduce_metrics_named(cls, logging_outputs, name: str) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get(name + "_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get(name + "_nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get(name + "_ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get(name + "_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            name + "_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            name + "_nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            name + "_ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get(name + "_total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar(name + "_total", total)
            n_correct = utils.item(
                sum(log.get(name + "_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar(name + "_n_correct", n_correct)
            metrics.log_derived(
                name + "_accuracy",
                lambda meters: round(
                    meters[name + "_n_correct"].sum * 100.0 / meters[name + "_total"].sum, 3
                )
                if meters[name + "_total"].sum > 0
                else float("nan"),
                )
