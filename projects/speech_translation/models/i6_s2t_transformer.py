import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool1d, Embedding, Linear
from torchtyping import TensorType

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from speech_translation.cli.checkpoints import load_pretrained_component_from_model
from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import base_architecture, Conv1dSubsampler
from speech_translation.models.relative_positioning_s2t_transformer import RelativePositioningS2TTransformerModel, \
    RelativePositioningS2TTransformerEncoder, \
    relative_positioning_s2t_transformer as relative_positioning_s2t_args
from speech_translation.layers.conformer import ConformerEncoderLayer
from speech_translation.layers.transformer import RelativePositioningTransformerEncoderLayer

logger = logging.getLogger(__name__)


def extended_init_bert_params(module):
    init_bert_params(module)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            nn.init.xavier_uniform_(param.data, gain=1.0)
    return m


class RWTHLSTMPreEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.num_lstm_layers = args.num_lstm_layers
        self.lstm_dim = args.lstm_dim
        self.embed_dim = args.encoder_embed_dim

        self.lstm_pooling = None
        self.lstm_pooling_sizes = None
        if args.lstm_pooling is not None:
            self.lstm_pooling_sizes = [int(p) for p in args.lstm_pooling.split(',')]
            assert len(self.lstm_pooling_sizes) == args.num_lstm_layers, f"With {args.num_lstm_layers} BiLSTM layers, " \
                                                                         f"--lstm-pooling should contain {args.num_lstm_layers} comma-separated values."
            self.lstm_pooling = nn.ModuleList(
                MaxPool1d(kernel_size=p, ceil_mode=True) for p in self.lstm_pooling_sizes
            )

        self.lstm = nn.ModuleList(
            LSTM(
                input_size=args.input_feat_per_channel * args.input_channels if idx == 0 else 2 * args.lstm_dim,
                hidden_size=args.lstm_dim,
                num_layers=1,
                bidirectional=True,
            )
            for idx in range(args.num_lstm_layers)
        )
        self.lin_proj = nn.Linear(2 * self.lstm_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(p=args.lstm_dropout)

        self.input_left_pad = False  # TODO

    def forward(self,
                src_tokens: TensorType['batch', 'src_time', 'dim'],
                src_lengths: TensorType['batch']) -> tuple[TensorType['src_time', 'batch', 'dim'],
                                                           TensorType['batch']]:
        # src_tokens: # B x T x (C x D)
        # src_lengths: # B
        if self.input_left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )

        # src_tokens: # B x T x (C x D) -> T x B x C x D
        x = src_tokens.transpose(0, 1)

        # Possible TODO: optimize using PackedSequence

        bsz = src_tokens.size(0)

        state_size = self.num_lstm_layers, 2, bsz, self.lstm_dim

        h0 = src_tokens.new_zeros(*state_size)
        c0 = src_tokens.new_zeros(*state_size)

        for idx in range(self.num_lstm_layers):
            x, (_, _) = self.lstm[idx](x, (h0[idx], c0[idx]))  # T x B x 2 * LSTM_DIM
            if self.lstm_pooling is not None:
                x = self.lstm_pooling[idx](x.permute(1, 2, 0)).permute(2, 0, 1)
                src_lengths = torch.ceil(src_lengths / self.lstm_pooling_sizes[idx])

        x = self.dropout(x)

        return self.lin_proj(x), src_lengths.to(torch.long)


class StackSubsampler(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.num_stack = args.num_stack
        self.proj = nn.Linear(args.input_feat_per_channel * args.input_channels * self.num_stack, args.encoder_embed_dim, bias=False)

    def forward(self,
                src_tokens: TensorType['batch', 'src_time', 'dim'],
                src_lengths: TensorType['batch']) -> tuple[TensorType['src_time', 'batch', 'dim'],
                                                           TensorType['batch']]:
        batch_size = src_tokens.size(0)
        dim = src_tokens.size(2)

        pad = (self.num_stack - (src_tokens.size(1) % self.num_stack)) % self.num_stack

        x = src_tokens.transpose(1, 0)
        x = F.pad(x, pad=(0, 0, 0, 0, 0, pad))  # confusing torch syntax, reverse axis!
        x = x.view(-1, self.num_stack, batch_size, dim)  # ((T+r) // stack, stack, batch_size, dim)
        x = x.transpose(1, 0)  # (stack, (T+r)//stack, N, D)
        x = torch.cat(list(x), dim=-1)
        x = self.proj(x)

        return x, torch.ceil(src_lengths / self.num_stack).to(torch.long)


class RWTHS2TTransformerEncoder(RelativePositioningS2TTransformerEncoder):

    def __init__(self, args, decoder_embed_tokens):
        super(RWTHS2TTransformerEncoder, self).__init__(args)

        self.ctc_loss = args.ctc_loss
        self.ignore_ctc = getattr(args, "ignore_ctc", False)
        self.ctc_proj = None
        if self.ctc_loss:
            self.ctc_proj = Linear(args.encoder_embed_dim, decoder_embed_tokens.num_embeddings, bias=True)
            if self.ignore_ctc:
                self.ctc_proj.requires_grad_(False)

    @staticmethod
    def build_layers(args):
        adapter_layers = getattr(args, "adapter_layers", 0)

        cls = RelativePositioningTransformerEncoderLayer if not getattr(args, 'conformer',
                                                                        False) else ConformerEncoderLayer
        return nn.ModuleList(
            [cls(args) for _ in range(args.encoder_layers + adapter_layers)]
        )

    @staticmethod
    def build_subsampler(args):
        if args.subsampling == "lstm":
            return RWTHLSTMPreEncoder(args)
        elif args.subsampling == "conv":
            return Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        elif args.subsampling == "stack":
            return StackSubsampler(args)
        else:
            raise ValueError(f"Invalid subsampling method {args.subsampling}.")

    def _forward(self,
                 src_tokens: TensorType['batch', 'time', 'channels', 'dim'],
                 src_lengths: TensorType['batch'],
                 return_all_hiddens=False):
        x = src_tokens

        x, input_lengths = self.subsample(x, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
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
            "input_lengths": input_lengths,
            "ctc_scores": None if not self.ctc_loss or self.ignore_ctc else torch.log_softmax(self.ctc_proj(x), -1)
        }


@register_model("rwth_s2t_transformer")
class RWTHS2TTransformerModel(RelativePositioningS2TTransformerModel):

    @classmethod
    def add_args(cls, parser):
        super(RWTHS2TTransformerModel, RWTHS2TTransformerModel).add_args(parser)

        parser.add_argument('--lstm-dim', type=int)
        parser.add_argument('--num-lstm-layers', type=int)
        parser.add_argument('--lstm-dropout', type=float)
        parser.add_argument('--lstm-pooling', type=str)
        parser.add_argument('--ctc-loss', action='store_true', default=False)
        parser.add_argument('--ignore-ctc', action='store_true', default=False, help='keep ctc weights, but do not return encoder outputs')
        parser.add_argument('--subsampling', type=str, choices=("lstm", "conv", "stack"))
        parser.add_argument('--num-stack', type=int)
        parser.add_argument('--adapter-layers', type=int, default=0)
        parser.add_argument('--force-decoder-bert-init', action='store_true', default=False)

    def __init__(self, encoder, decoder, args):
        super().__init__(encoder, decoder, args)

        self.ctc_loss = args.ctc_loss
        self.ignore_ctc = getattr(args, 'ignore_ctc', False)

    @classmethod
    def load_pretrained_encoder(cls, encoder, args):
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped loading pretrained encoder weights because {pretraining_path} does not exist"
                )
            else:
                adapter_layers = getattr(args, "adapter_layers", 0)
                layer_params = None
                if adapter_layers > 0:
                    layer_params = [key for key in encoder.state_dict().keys() if key.startswith('transformer_layers')]
                    layer_params = [key for key in layer_params if int(key.split('.')[1]) >= args.encoder_layers]

                encoder = load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path, reset_ctc=args.pretrained_encoder_reset_ctc
                    if hasattr(args, "pretrained_encoder_reset_ctc") else False,
                    ignore_keys=layer_params
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args, decoder_embed_tokens)
        encoder = cls.load_pretrained_encoder(encoder, args)

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        decoder = cls.load_pretrained_decoder(decoder, args)

        if getattr(args, "force_decoder_bert_init", False):
            logger.info("Initializing decoder with BERT-style initializiation, "
                        "this will overwrite --load-pretrained-decoder-from.")
            decoder.apply(extended_init_bert_params)

        return cls(encoder, decoder, args)

    @classmethod
    def build_encoder(cls, args, decoder_embed_tokens, **kwargs):
        return RWTHS2TTransformerEncoder(args, decoder_embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if self.ctc_loss and not self.ignore_ctc:
            return encoder_out, decoder_out
        else:
            return decoder_out


@register_model_architecture("rwth_s2t_transformer", "rwth_s2t_transformer")
def rwth_s2t_transformer(args):
    args.ctc_loss = getattr(args, "ctc_loss", False)
    args.embedding_dropout = getattr(args, "embedding_dropout", 0.3)
    args.num_lstm_layers = getattr(args, "num_lstm_layers", 2)
    args.lstm_dim = getattr(args, "lstm_dim", 1024)
    args.lstm_dropout = getattr(args, "lstm_dropout", getattr(args, "dropout", 0.1))
    args.lstm_pooling = getattr(args, "lstm_pooling", "3,2")
    args.subsampling = getattr(args, "subsampling", "lstm")
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    relative_positioning_s2t_args(args)


@register_model_architecture("rwth_s2t_transformer", "espnet_transformer")
def espnet_transformer(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")  # https://arxiv.org/pdf/2107.00636.pdf
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.ctc_loss = getattr(args, "ctc_loss", True)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    relative_positioning_s2t_args(args)


@register_model_architecture("rwth_s2t_transformer", "fair_conformer")
def fair_conformer(args):
    args.conformer = getattr(args, "conformer", True)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")  # https://arxiv.org/pdf/2107.00636.pdf
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    relative_positioning_s2t_args(args)


@register_model_architecture("rwth_s2t_transformer", "espnet_conformer")
def espnet_conformer(args):
    args.conformer = getattr(args, "conformer", True)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")  # https://arxiv.org/pdf/2107.00636.pdf
    args.conv_channels = getattr(args, "conv_channels", 256)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.subsampling = getattr(args, "subsampling", "conv")
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    relative_positioning_s2t_args(args)
