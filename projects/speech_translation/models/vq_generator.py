from argparse import Namespace
from typing import Optional

import torch.nn
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.models.speech_to_text.utils import lengths_to_encoder_padding_mask
from speech_translation.models.relative_positioning_transformer import (
    RelativePositioningTransformerModel, relative_positioning_transformer_vaswani_base)
from speech_translation.models.relative_positioning_s2t_transformer import Conv1dSubsampler


CODEBOOK_PADDING_IDX = 0


@register_model("vq_generator")
class VQGenerator(RelativePositioningTransformerModel):

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)

        self.codebook = torch.nn.Embedding(cfg.codebook_size, cfg.codebook_dim, padding_idx=CODEBOOK_PADDING_IDX)
        self.time_reduction_kernels = cfg.time_reduction_kernels
        self.time_reduction_strides = cfg.time_reduction_strides

        self.time_reduction_conv = Conv1dSubsampler(
            in_channels=cfg.encoder_embed_dim,
            mid_channels=cfg.codebook_conv_dim,
            out_channels=cfg.codebook_dim,
            kernel_sizes=[int(k) for k in cfg.time_reduction_kernels.split(',')],
            strides=[int(s) for s in cfg.time_reduction_strides.split(',')]
        )
        self.gumbel_tau = cfg.gumbel_tau

    @staticmethod
    def add_args(parser):
        super(VQGenerator, VQGenerator).add_args(parser)

        parser.add_argument('--codebook-size', type=int, default=102400)
        parser.add_argument('--codebook-conv-dim', type=int, default=768)
        parser.add_argument('--codebook-dim', type=int, default=1024)
        parser.add_argument('--time-reduction-kernels', type=str, default="3,3")
        parser.add_argument('--time-reduction-strides', type=str, default="2,2")
        parser.add_argument('--gumbel-tau', type=float, default=1.)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=True
        )

        pre_code = encoder_out['encoder_out'][0]  # (T, B, D_model)
        pre_code, new_lengths = self.time_reduction_conv(pre_code.transpose(0, 1),
                                                         encoder_out['src_lengths'][0])
        pre_code = pre_code @ self.codebook.weight.T  # (T, B, num_codes)
        code_dist = F.gumbel_softmax(pre_code, tau=self.gumbel_tau, hard=False)  # (T, B, num_codes), one-hot

        # Straight-through estimation
        index = code_dist.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(pre_code, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        code = y_hard - code_dist.detach() + code_dist

        encoder_out['encoder_out'][0] = code @ self.codebook.weight
        encoder_out['src_lengths'][0] = new_lengths

        encoder_out['encoder_padding_mask'][0], _ = lengths_to_encoder_padding_mask(lengths=new_lengths, batch_first=True)

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=new_lengths,
            return_all_hiddens=False,
        )
        return decoder_out, encoder_out, code_dist

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        args = Namespace(**vars(args))
        args.encoder_embed_dim = args.codebook_dim

        return super(VQGenerator, cls).build_decoder(
            args, tgt_dict, embed_tokens
        )


@register_model_architecture("vq_generator", "vq_generator")
def vq_generator_base(args):
    relative_positioning_transformer_vaswani_base(args)
