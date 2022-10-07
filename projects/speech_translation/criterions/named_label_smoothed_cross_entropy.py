from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


class NamedLabelSmoothedCrossEntropy(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, name: str, ignore_prefix_size=0, report_accuracy=False):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.name = name

    def forward(self, model, sample, reduce=True):
        loss, sample_size, logging_output = super().forward(model, sample, reduce)
        logging_output = {f"{self.name}_{key}": val for key, val in logging_output.items()}
        return loss, sample_size, logging_output
