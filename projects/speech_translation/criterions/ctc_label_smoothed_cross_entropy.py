import math
import torch
from dataclasses import dataclass, field
import warnings

from torch.nn import CTCLoss

from fairseq import utils, metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, \
    LabelSmoothedCrossEntropyCriterionConfig


@dataclass
class CTCLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    ctc_weight: float = field(
        default=1.0,
        metadata={"help": "weighting of ctc objective"},
    )


@register_criterion(
    "ctc_label_smoothed_cross_entropy", dataclass=CTCLabelSmoothedCrossEntropyCriterionConfig
)
class CTCLabelSmoothedCrossEntropy(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ctc_weight, ignore_prefix_size=0, report_accuracy=False):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.ctc_loss = CTCLoss(blank=task.target_dictionary.pad(), zero_infinity=True)
        self.ctc_weight = ctc_weight
        assert 0 <= self.ctc_weight < 1, "CTC weight must be in interval [0,1)."

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        assert isinstance(net_output, tuple) and len(net_output) == 2, "Model must be i6_s2t with --ctc-loss set."
        enc_output, net_output = net_output

        ctc_target = sample.get('l2r_target', sample['target'])
        ctc_target_lengths = sample['target_lengths']
        if sample.get('prepend_direction', False):
            ctc_target = ctc_target[:, 1:]
            ctc_target_lengths = torch.maximum(ctc_target_lengths - 1, torch.zeros_like(ctc_target_lengths))

        ctc_loss = self.ctc_loss(enc_output['ctc_scores'],
                                 ctc_target,
                                 enc_output['input_lengths'],
                                 ctc_target_lengths)

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss = (1 - self.ctc_weight) * loss + self.ctc_weight * ctc_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ctc_loss": ctc_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super(CTCLabelSmoothedCrossEntropy, cls).reduce_metrics(logging_outputs)

        """Aggregate logging outputs from data parallel training."""
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
