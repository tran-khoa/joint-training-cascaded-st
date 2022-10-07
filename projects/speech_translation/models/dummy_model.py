import torch

from fairseq.models.fairseq_encoder import FairseqEncoder


class DummyEncoder(FairseqEncoder):
    def __init__(self, tensor: torch.Tensor, actual_encoder):
        super().__init__(None)
        self.tensor = tensor
        self.actual_encoder = actual_encoder

    def forward(self, **kwargs):
        return self.tensor

    def reorder_encoder_out(self, encoder_out, new_order):
        if self.actual_encoder is None:
            raise NotImplementedError()
        return self.actual_encoder.reorder_encoder_out(encoder_out, new_order)
