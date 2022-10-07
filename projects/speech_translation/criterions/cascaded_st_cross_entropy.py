from dataclasses import dataclass, field
import typing

import torch

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig, LabelSmoothedCrossEntropyCriterion, \
    label_smoothed_nll_loss

from speech_translation.models.cascaded_st_model import CascadedSTModel

if typing.TYPE_CHECKING:
    from speech_translation.tasks.cascaded_speech_translation import CascadedSpeechTranslationTask

from speech_translation.typings import STCriterionSample


@dataclass
class CascadedSTLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    ensemble_training: bool = field(
        default=False
    )


@register_criterion(
    "cascaded_st_label_smoothed_cross_entropy", dataclass=CascadedSTLabelSmoothedCrossEntropyCriterionConfig
)
class CascadedSTLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    task: 'CascadedSpeechTranslationTask'

    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False, ensemble_training=False):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.ensemble_training = ensemble_training

    def forward(self, model: CascadedSTModel, sample: STCriterionSample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, nll_loss = self.compute_loss(model, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, sample['decoder_out'], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_accuracy(self, model, net_output, sample):
        decoder_out = sample['decoder_out']
        bsz, piv_beam_size, seq_len, num_classes = decoder_out[0].size()
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        target = target.view(bsz, seq_len).repeat_interleave(repeats=piv_beam_size, dim=0).view(-1)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def compute_loss(self, model: CascadedSTModel, sample: STCriterionSample, reduce=True, **kwargs):
        decoder_out = sample['decoder_out']
        bsz, piv_beam_size, seq_len, num_classes = decoder_out[0].size()
        lprobs, target = self.get_lprobs_and_target(model, decoder_out, sample)

        if self.ensemble_training:
            # RAG-Token
            lprobs = lprobs.view(bsz, piv_beam_size, seq_len, num_classes)
            asr_weights = sample['asr_score'].view(bsz, piv_beam_size, 1, 1)
            lprobs = torch.logsumexp(lprobs + asr_weights, dim=1).view(bsz, seq_len, num_classes)
            lprobs = lprobs.view(-1, num_classes)

            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=True,
            )
            return loss, nll_loss.sum()
        else:
            # RAG-Sequence
            target = target.view(bsz, seq_len).repeat_interleave(repeats=piv_beam_size, dim=0).view(-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=False,
            )
            loss = -loss.view(bsz, piv_beam_size, seq_len).sum(-1)

            loss = loss + sample['asr_score']
            loss = -torch.logsumexp(loss, dim=1).sum()

            return loss, nll_loss.sum()
