import argparse
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from fairseq.modules import PositionalEmbedding
from speech_translation.cli.checkpoints import load_pretrained_component_from_model
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, S2TTransformerModel, S2TTransformerEncoder, base_architecture
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
)
from speech_translation.layers.conformer import ConformerEncoderLayer
from speech_translation.layers.transformer import RelativePositioningTransformerEncoderLayer
from speech_translation.models.relative_positioning_transformer import RelativePositioningTransformerDecoder

logger = logging.getLogger(__name__)


class RelativePositioningS2TTransformerEncoder(S2TTransformerEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args):
        super(S2TTransformerEncoder, self).__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.embed_dropout_module = FairseqDropout(
            p=args.embedding_dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = self.build_subsampler(args)

        self.transformer_layers = self.build_layers(args)
        if args.encoder_normalize_before and not getattr(args, 'conformer', False):
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.is_conformer = getattr(args, 'conformer', False)

        self.use_relpos = args.use_relpos
        if self.use_relpos:
            logger.info("Using relative positional encodings in encoder.")
        else:
            logger.info("Using absolute positional encoodings in encoder.")
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

    @staticmethod
    def build_layers(args):
        cls = RelativePositioningTransformerEncoderLayer if not getattr(args, 'conformer', False) else ConformerEncoderLayer
        return nn.ModuleList(
            [cls(args) for _ in range(args.encoder_layers)]
        )

    @staticmethod
    def build_subsampler(args):
        return Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if not self.use_relpos:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
        x = self.embed_dropout_module(x)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class RelativePositioningS2TTransformerDecoder(RelativePositioningTransformerDecoder):

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


@register_model("relpos_s2t_transformer")
class RelativePositioningS2TTransformerModel(S2TTransformerModel):
    @classmethod
    def add_args(cls, parser):
        super(RelativePositioningS2TTransformerModel, RelativePositioningS2TTransformerModel).add_args(parser)

        # Relative Positional Encodings (Shaw et al., 2018)
        parser.add_argument('--use-relpos', action='store_true')
        parser.set_defaults(use_relpos=True)
        parser.add_argument('--relpos-clipping', type=int)

        parser.add_argument('--embedding-dropout', type=float)
        parser.add_argument('--conformer', action=argparse.BooleanOptionalAction)

        parser.add_argument('--load-pretrained-decoder-from', type=str)
        parser.add_argument('--pretrained-encoder-reset-ctc', action=argparse.BooleanOptionalAction)

    def __init__(self, encoder, decoder, args):
        super().__init__(encoder, decoder)
        self.args = args

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return nn.Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args)
        encoder = cls.load_pretrained_encoder(encoder, args)

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        decoder = cls.load_pretrained_decoder(decoder, args)
        return cls(encoder, decoder, args)

    @classmethod
    def load_pretrained_encoder(cls, encoder, args):
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped loading pretrained encoder weights because {pretraining_path} does not exist"
                )
            else:
                encoder = load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path, reset_ctc=args.pretrained_encoder_reset_ctc
                    if hasattr(args, "pretrained_encoder_reset_ctc") else False
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def load_pretrained_decoder(cls, decoder, args):
        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped loading pretrained decoder weights {pretraining_path} does not exist"
                )
            else:
                decoder = load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder

    @classmethod
    def build_encoder(cls, args, **kwargs):
        return RelativePositioningS2TTransformerEncoder(args)

    @classmethod
    def build_decoder(cls, args, task, embed_tokens, **kwargs):
        return RelativePositioningS2TTransformerDecoder(args, task.target_dictionary, embed_tokens)

    def build_submodel(self, new_args, task):
        args = Namespace(**self.args.__dict__)
        for k, v in new_args.items():
            setattr(args, k, v)
        return self.build_model(args, task)


@register_model_architecture(model_name="relpos_s2t_transformer", arch_name="relpos_s2t_transformer")
def relative_positioning_s2t_transformer(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.embedding_dropout = getattr(args, "embedding_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.use_relpos = getattr(args, 'use_relpos', True)
    args.relpos_clipping = getattr(args, 'relpos_clipping', 16)


@register_model_architecture("relpos_s2t_transformer", "relpos_s2t_transformer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    relative_positioning_s2t_transformer(args)


@register_model_architecture("relpos_s2t_transformer", "fair_s2t_transformer_deprecated")
def fair_s2t_transformer_deprecated(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.stochastic_p = getattr(args, "stochastic_p", 1)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    relative_positioning_s2t_transformer(args)


