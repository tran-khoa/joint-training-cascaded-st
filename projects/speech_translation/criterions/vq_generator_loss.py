import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, \
    LabelSmoothedCrossEntropyCriterionConfig


@dataclass
class VQGeneratorLossConfig(LabelSmoothedCrossEntropyCriterionConfig):
    entropy_weight: float = field(default=1)


@register_criterion(
    "vq_generator", dataclass=VQGeneratorLossConfig
)
class VQGeneratorLoss(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, entropy_weight=1, report_accuracy=False):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.entropy_weight = entropy_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        dec_output, enc_output, code_probs = net_output

        loss, nll_loss = self.compute_loss(model, dec_output, sample, reduce=reduce)

        last_enc_output_norm = torch.norm(enc_output['encoder_states'][-1], dim=-1).mean()

        averaged_probability = code_probs.mean(0).mean(0)
        min_real = torch.finfo(code_probs.dtype).min
        code_entropy = (torch.clamp(averaged_probability.log2(), min=min_real) * averaged_probability).mean(-1)
        loss = loss + last_enc_output_norm + self.entropy_weight * code_entropy

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "encoder_l2": last_enc_output_norm,
            "code_entropy": code_entropy,
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
        super(VQGeneratorLoss, cls).reduce_metrics(logging_outputs)

        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        encoder_l2 = sum(log.get("encoder_l2", 0) for log in logging_outputs)
        code_entropy = sum(log.get("code_entropy", 0) for log in logging_outputs)

        metrics.log_scalar(
            "encoder_l2", encoder_l2, sample_size, round=3
        )
        metrics.log_scalar(
            "code_entropy", code_entropy, sample_size, round=6
        )
