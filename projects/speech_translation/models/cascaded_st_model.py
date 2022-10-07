import argparse
from pathlib import Path
from typing import Dict, Any, Literal, Optional

import logging
import torch.nn.functional as F
import torch
import yaml
from torch import Tensor
from torchtyping import TensorType

from fairseq import models
from fairseq.data.dictionary import Dictionary
from fairseq.models import BaseFairseqModel, register_model_architecture, register_model, ARCH_CONFIG_REGISTRY, \
    FairseqEncoderDecoderModel
from fairseq.models import FairseqEncoder
from fairseq.data.data_utils import lengths_to_padding_mask



logger = logging.getLogger(__name__)


class CascadedPseudoEncoder(FairseqEncoder):

    asr_model: BaseFairseqModel
    mt_encoder: FairseqEncoder

    def __init__(self,
                 args: argparse.Namespace,
                 asr_model: BaseFairseqModel,
                 mt_encoder: FairseqEncoder,
                 pivot_dictionary: Dictionary):
        """
        Handles ASR-Encoder -> ASR-Decoder -> MT-Encoder
        """
        super().__init__(pivot_dictionary)

        self.asr_model = asr_model
        self.mt_encoder = mt_encoder

        self.tight_integrated = getattr(args, "tight_integrated", False)
        self.tight_integrated_training_temp = getattr(args, "tight_integrated_training_temp", 1.0)
        self.tight_integrated_decoding_temp = getattr(args, "tight_integrated_decoding_temp", 1.5)

        if self.tight_integrated:
            logger.info(f"Pseudo-encoder will use tight integration, "
                        f"with training temperature {self.tight_integrated_training_temp} "
                        f"and decoding temperature {self.tight_integrated_decoding_temp}.")
            assert self.asr_model.decoder.dictionary == self.mt_encoder.dictionary, \
                "ASR and MT model must have same dictionary on transcript side."

        self.auxiliary_cross_attention = getattr(args, "auxiliary_cross_attention", False)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--tight-integrated', action=argparse.BooleanOptionalAction)
        parser.add_argument('--tight-integrated-training-temp', type=float, default=1.0)
        parser.add_argument('--tight-integrated-decoding-temp', type=float, default=1.5)

    def forward_asr_encoder(self, src_tokens, src_lengths):
        return self.asr_model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

    def forward(self, **kwargs):
        return self.forward_torchscript(kwargs)

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        return self.__forward(**net_input)

    def __forward(
            self,
            src_tokens: TensorType["batch", "time", "dim"],
            src_lengths: TensorType["batch"],
            asr_prev_output_tokens: TensorType["batch * beam", "piv_len"],
            asr_piv_tokens,
            asr_piv_lengths,
            mt_piv_tokens,
            mt_piv_lengths,
            r2l_mode=None,
            r2l_model=None,
            asr_encoder_out=None,
            asr_only: bool = False,
            asr_device: str = "cuda",
            mt_device: str = "cuda",
            length_normalize_asr_beam_scores: bool = True,
            **kwargs
    ):
        """
        Args:
            src_tokens:
            src_lengths:
            asr_prev_output_tokens:
            mt_piv_tokens:
            mt_piv_lengths:
            asr_encoder_out:
            **kwargs:

        Returns:

        """

        if asr_encoder_out is None:
            asr_encoder_out = self.forward_asr_encoder(src_tokens, src_lengths)

        batch_size = src_tokens.size(0)
        beam_size = asr_prev_output_tokens.size(0) // batch_size
        piv_max_length = asr_prev_output_tokens.size(-1)

        if r2l_model is not None:
            if r2l_mode == "gen":
                raise NotImplementedError()
            elif r2l_mode == "rescore":
                raise NotImplementedError()
            else:
                raise NotImplementedError()

        asr_decoder_out, _ = self.asr_model.decoder(
            prev_output_tokens=asr_prev_output_tokens,
            encoder_out=asr_encoder_out
        )  # (batch, tgt_len, vocab)

        # Obtain pivot scores of selected tokens
        raw_asr_scores = self.asr_model.get_normalized_probs((asr_decoder_out, {}), log_probs=True)
        asr_scores = torch.gather(raw_asr_scores, -1, asr_piv_tokens.unsqueeze(-1)).squeeze(-1)
        # Remove padding from scoring
        mask_pad = (asr_piv_tokens == self.asr_model.decoder.dictionary.pad())
        asr_scores[mask_pad] = 0

        if r2l_model is not None:
            if r2l_model == "gen":
                pass #TODO reverse r2l outputs, obtain target scores, etc.`

        asr_hyp_scores = asr_scores.sum(-1)
        if length_normalize_asr_beam_scores:
            asr_hyp_scores = asr_hyp_scores / asr_piv_lengths.view(-1)

        if asr_only:
            return asr_decoder_out, asr_hyp_scores

        if asr_device != mt_device:
            mt_piv_tokens = mt_piv_tokens.to(mt_device)
            mt_piv_lengths = mt_piv_lengths.to(mt_device)

        token_embeddings = None
        if self.tight_integrated:
            tight_integrated_temp = self.tight_integrated_training_temp if self.training else self.tight_integrated_decoding_temp
            if tight_integrated_temp > 0:
                tight_integrated_scores = torch.softmax(raw_asr_scores * tight_integrated_temp, dim=-1)
                token_embeddings = tight_integrated_scores @ self.mt_encoder.embed_tokens.weight
                # Should be taken care of by the transformer encoder anyways
                # token_embeddings = token_embeddings * (1 - mask_pad.unsqueeze(-1).type_as(token_embeddings))
                # TODO adapter layers?

        mt_encoder_out = self.mt_encoder(
            src_tokens=mt_piv_tokens,
            src_lengths=mt_piv_lengths,
            return_all_hiddens=False,
            token_embeddings=token_embeddings
        )

        mt_encoder_out['asr_score'] = asr_hyp_scores
        if self.auxiliary_cross_attention:
            mt_encoder_out['asr_encoder_out'] = asr_encoder_out
            mt_encoder_out['asr_encoder_padding_mask'] = lengths_to_padding_mask(src_lengths)

        return mt_encoder_out

    def max_positions(self):
        return self.asr_model.encoder.max_positions()


@register_model('cascaded_st')
class CascadedSTModel(FairseqEncoderDecoderModel):
    def __init__(self, args: argparse.Namespace, asr_model: BaseFairseqModel, mt_model: BaseFairseqModel, mt_args):
        encoder = CascadedPseudoEncoder(args, asr_model, mt_model.encoder, mt_model.encoder.dictionary)
        decoder = mt_model.decoder

        super().__init__(encoder, decoder)

        self.mt_model_cls = mt_model.__class__
        self.mt_args = mt_args

        self.freeze_components = getattr(args, 'freeze_components', None)
        if self.freeze_components:
            for comp in self.freeze_components.split(','):
                if comp == "asr_enc":
                    self.asr_model.encoder.requires_grad_(False)
                elif comp == "asr_dec":
                    self.asr_model.decoder.requires_grad_(False)
                elif comp == "mt_enc":
                    self.mt_model.encoder.requires_grad_(False)
                elif comp == "mt_dec":
                    self.mt_model.decoder.requires_grad_(False)
                else:
                    raise ValueError(f"Invalid component, cannot freeze {comp}.")
        self.auxiliary_cross_attention = getattr(args, "auxiliary_cross_attention", False)

    def update_freeze_component(self, freeze_components):
        self.asr_model.encoder.requires_grad_("asr_enc" not in freeze_components.split(','))
        self.asr_model.decoder.requires_grad_("asr_dec" not in freeze_components.split(','))
        self.mt_model.encoder.requires_grad_("mt_enc" not in freeze_components.split(','))
        self.mt_model.decoder.requires_grad_("mt_dec" not in freeze_components.split(','))

    @classmethod
    def build_model(cls, args, task):
        asr_config = cls.get_arch_conf(args.asr_arch, Path(args.asr_model_conf) if getattr(args, 'asr_model_conf', None) else None, args)
        mt_config = cls.get_arch_conf(args.mt_arch, Path(args.mt_model_conf) if getattr(args, 'mt_model_conf', None) else None, args)

        print(f"ASR model configuration: {str(asr_config)}")
        print(f"MT model configuration {str(mt_config)}")

        asr_model: BaseFairseqModel = models.build_model(asr_config, task.asr_task)
        mt_model: BaseFairseqModel = models.build_model(mt_config, task.mt_task)

        return CascadedSTModel(args, asr_model, mt_model, mt_config)

    @property
    def asr_model(self):
        return self.encoder.asr_model

    @property
    def mt_model(self):
        return self.mt_model_cls(args=self.mt_args,
                                 encoder=self.encoder.mt_encoder,
                                 decoder=self.decoder)

    def load_model_params(self, model: Literal['mt', 'asr'], params: Dict[str, Any]):
        if model == 'mt':
            self.mt_model.load_state_dict(params)
        else:
            self.asr_model.load_state_dict(params)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--asr-arch', type=str)
        parser.add_argument('--asr-model-conf', type=str)
        parser.add_argument('--mt-arch', type=str)
        parser.add_argument('--mt-model-conf', type=str)
        parser.add_argument('--freeze-components', type=str, default=None, help="comma separated, possible components"
                                                                                "asr_enc,asr_dec,mt_enc,mt_dec")
        parser.add_argument('--auxiliary-cross-attention', action='store_true')
        CascadedPseudoEncoder.add_args(parser)

    @staticmethod
    def get_arch_conf(arch: str, yaml_path: Optional[Path], global_args: argparse.Namespace) -> Any:

        # Update with global arguments
        config = argparse.Namespace(**global_args.__dict__)

        # Override with model config files
        if yaml_path is not None:
            if yaml_path.is_file():
                try:
                    with open(yaml_path) as f:
                        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
                        cfg_dict = {key.replace("-", "_"): val for key, val in cfg_dict.items()}
                        config.__dict__.update(cfg_dict)
                except Exception as e:
                    raise Exception(
                        f"Failed to load config from {yaml_path.as_posix()}: {e}"
                    )
            else:
                raise FileNotFoundError(f"{yaml_path.as_posix()} not found")

        # Populate unset keys with architecture defaults
        ARCH_CONFIG_REGISTRY[arch](config)

        # Set architecture name
        config._name = arch

        return config

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Tasks and criterions should use encoder and decoder separately.')


@register_model_architecture('cascaded_st', 'cascaded_st')
def cascaded_st_args(args):
    args.asr_arch = getattr(args, "asr_arch", 'relpos_s2t_transformer')
    args.mt_arch = getattr(args, "mt_arch", 'relpos_transformer')

