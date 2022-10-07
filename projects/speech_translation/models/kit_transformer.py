import logging
from pathlib import Path

from torch.nn import ModuleList

from fairseq import checkpoint_utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model_architecture, register_model
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from speech_translation.layers.transformer import StochasticTransformerDecoderLayer, StochasticTransformerEncoderLayer
from speech_translation.models.i6_s2t_transformer import RWTHS2TTransformerEncoder, RWTHS2TTransformerModel
from speech_translation.models.relative_positioning_s2t_transformer import \
    relative_positioning_s2t_transformer as relative_positioning_s2t_args, RelativePositioningS2TTransformerDecoder

logger = logging.getLogger(__name__)


class KITS2TTransformerDecoder(RelativePositioningS2TTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.stochastic_p = args.stochastic_p

    def build_decoder_layer(self, args, idx, no_encoder_attn=False):
        layer = StochasticTransformerDecoderLayer(args, idx, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class KITS2TTransformerEncoder(RWTHS2TTransformerEncoder):
    @staticmethod
    def build_layers(args):
        assert not getattr(args, 'conformer', False), 'KIT-style training not implemented with conformers yet'
        adapter_layers = getattr(args, "adapter_layers", 0)

        return ModuleList(
            [StochasticTransformerEncoderLayer(args, idx) for idx in range(args.encoder_layers + adapter_layers)]
        )


@register_model("kit_s2t_transformer")
class KITS2TTransformerModel(RWTHS2TTransformerModel):

    @classmethod
    def add_args(cls, parser):
        super(KITS2TTransformerModel, KITS2TTransformerModel).add_args(parser)
        parser.add_argument('--stochastic-p', type=float)

    @classmethod
    def build_encoder(cls, args, decoder_embed_tokens, **kwargs):
        return KITS2TTransformerEncoder(args, decoder_embed_tokens)

    @classmethod
    def build_decoder(cls, args, task, embed_tokens, **kwargs):
        return KITS2TTransformerDecoder(args, task.target_dictionary, embed_tokens)


@register_model_architecture("kit_s2t_transformer", "kit_s2t_transformer")
def kit_s2t_transformer(args):
    args.ctc_loss = getattr(args, "ctc_loss", False)
    args.subsampling = getattr(args, "subsampling", "stack")
    args.num_stack = getattr(args, "num_stack", 4)
    args.stochastic_p = getattr(args, "stochastic_p", 0.5)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.decoder_layers = getattr(args, "decoder_layers", 8)
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    relative_positioning_s2t_args(args)


@register_model_architecture("kit_s2t_transformer", "kit_s2t_transformer_alt")
def kit_s2t_transformer_alt(args):
    args.ctc_loss = getattr(args, "ctc_loss", False)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.stochastic_p = getattr(args, "stochastic_p", 0.3)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 8)
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    relative_positioning_s2t_args(args)


@register_model_architecture("kit_s2t_transformer", "fair_s2t_transformer")
def fair_s2t_transformer(args):
    args.ctc_loss = getattr(args, "ctc_loss", False)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.stochastic_p = getattr(args, "stochastic_p", 1)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    relative_positioning_s2t_args(args)

@register_model_architecture("kit_s2t_transformer", "fair_s2t_transformer_m")
def fair_s2t_transformer_m(args):
    args.ctc_loss = getattr(args, "ctc_loss", False)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 768)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.stochastic_p = getattr(args, "stochastic_p", 1)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    relative_positioning_s2t_args(args)


@register_model_architecture("kit_s2t_transformer", "fair_s2t_transformer_s")
def fair_s2t_transformer_s(args):
    args.ctc_loss = getattr(args, "ctc_loss", False)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 768)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.stochastic_p = getattr(args, "stochastic_p", 1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    relative_positioning_s2t_args(args)
