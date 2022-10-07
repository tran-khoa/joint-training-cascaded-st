from argparse import Namespace
from dataclasses import dataclass, fields
from typing import Optional, Callable, Union, Any

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.models.transformer.transformer_config import TransformerConfig, EncDecBaseConfig, DecoderConfig, QuantNoiseConfig
from fairseq.modules import FairseqDropout, LayerNorm
from speech_translation.layers.attention import RelativePositioningMultiheadAttention
from speech_translation.layers.relative_positioning import RelativePositionalEncoding


@dataclass
class ConformerEncoderConfig(EncDecBaseConfig):
    conv_bias: bool = True
    depthwise_conv_kernel_size: int = 31


@dataclass
class ConformerConfig(TransformerConfig):
    encoder: ConformerEncoderConfig = ConformerEncoderConfig()

    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = ConformerEncoderConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, ConformerEncoderConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = args._asdict() if hasattr(args, '_asdict') else vars(args) if hasattr(args,
                                                                                              '__dict__') else {}  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args


class ConformerMultiHeadedSelfAttentionModule(nn.Module):

    def __init__(self, cfg: ConformerConfig):
        super().__init__()

        self.layer_norm = LayerNorm(cfg.encoder.embed_dim)
        self.attention = RelativePositioningMultiheadAttention(
            cfg.encoder.embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True
        )
        self.relpos = RelativePositionalEncoding(Namespace(relpos_clipping=16), cfg.encoder.embed_dim // cfg.encoder.attention_heads)
        self.dropout = FairseqDropout(cfg.dropout)

    def forward(self, x: torch.Tensor,
                encoder_padding_mask: Optional[Tensor],
                attn_mask: Optional[Tensor] = None):
        x = self.layer_norm(x)
        x, _ = self.attention(query=x,
                              key=x,
                              value=x,
                              key_padding_mask=encoder_padding_mask,
                              need_weights=False,
                              attn_mask=attn_mask,
                              position_encodings=self.relpos(x.size()[0]))
        x = self.dropout(x)
        return x


class ConformerConvolutionModule(nn.Module):
    """
    Adapted from NMTGMinor
    """

    def __init__(self, cfg: ConformerConfig):
        super(ConformerConvolutionModule, self).__init__()

        model_dim = cfg.encoder.embed_dim

        self.layer_norm = LayerNorm(model_dim)
        self.pointwise_conv1 = nn.Conv1d(model_dim, 2 * model_dim, kernel_size=1, stride=1, padding=0, bias=cfg.encoder.conv_bias)
        self.depthwise_conv = nn.Conv1d(model_dim, model_dim,
                                        kernel_size=cfg.encoder.depthwise_conv_kernel_size,
                                        stride=1,
                                        padding=(cfg.encoder.depthwise_conv_kernel_size - 1) // 2,
                                        groups=model_dim,
                                        bias=False)
        self.batch_norm = nn.BatchNorm1d(model_dim)
        self.pointwise_conv2 = nn.Conv1d(model_dim, model_dim, kernel_size=1, stride=1, padding=0, bias=cfg.encoder.conv_bias)
        self.dropout = nn.Dropout(cfg.dropout)

        self.glu_activation = nn.GLU(dim=1)
        self.swish_activation = nn.SiLU()

    def forward(self, x):
        x = self.layer_norm(x)

        # x: (T, N, D)
        x = x.permute(1, 2, 0)
        # -> x: (N, D, T)

        x = self.pointwise_conv1(x)
        x = self.glu_activation(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish_activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.permute(2, 0, 1)
        # -> x: (T, N, D)


class ConformerFeedForwardModule(nn.Module):
    def __init__(self, cfg: ConformerConfig):
        super(ConformerFeedForwardModule, self).__init__()

        self.layer_norm = LayerNorm(cfg.encoder.embed_dim)
        self.lin1 = nn.Linear(cfg.encoder.embed_dim, cfg.encoder.embed_dim * 4)
        self.swish_activation = nn.SiLU()
        self.dropout1 = FairseqDropout(cfg.dropout)
        self.lin2 = nn.Linear(cfg.encoder.embed_dim * 4, cfg.encoder.embed_dim)
        self.dropout2 = FairseqDropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        x = self.layer_norm(x)
        x = self.lin1(x)
        x = self.swish_activation(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.dropout2(x)
        return x


class ConformerEncoderLayer(nn.Module):
    cfg: ConformerConfig

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.cfg = ConformerConfig.from_namespace(args)

        self.conv_module = ConformerConvolutionModule(self.cfg)
        self.self_att_module = ConformerMultiHeadedSelfAttentionModule(self.cfg)
        self.ff_module1 = ConformerFeedForwardModule(self.cfg)
        self.ff_module2 = ConformerFeedForwardModule(self.cfg)
        self.layer_norm = LayerNorm(self.cfg.encoder.embed_dim)

    def residual_connection(self, x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor], scale_residual: float = 1., **kwargs) -> \
    Union[torch.Tensor,
          tuple[torch.Tensor, tuple[Any]]]:
        residual = x
        func_out = func(**kwargs) if kwargs else func(x)
        if isinstance(func_out, tuple):
            states = func_out[1:]
            func_out = func_out[0]
            return scale_residual * residual + func_out, states
        return scale_residual * residual + func_out

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        x = self.residual_connection(x, self.ff_module1, scale_residual=0.5)
        x = self.residual_connection(x, lambda _x: self.self_att_module(_x, encoder_padding_mask, attn_mask))

        x = self.residual_connection(x, self.conv_module)
        x = self.residual_connection(x, self.ff_module2, scale_residual=0.5)
        x = self.layer_norm(x)

        return x
