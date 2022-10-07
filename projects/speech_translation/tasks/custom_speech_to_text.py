# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torch

import logging
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any

from speech_translation.data.r2l_speech_to_text_dataset import R2LSpeechToTextDataset, L2R_TAG, R2L_TAG
from speech_translation.data.dictionary import CustomDictionary
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator, get_features_or_waveform, \
    SpeechToTextDataset
from fairseq.tasks import LegacyFairseqTask, register_task
from speech_translation.data.cached_speech_to_text_dataset import (
    S2TDataConfig,
    CachedSpeechToTextDataset,
    CachedSpeechToTextDatasetCreator,
)
from speech_translation.pretrain_schemes import PRETRAIN_SCHEMES
from speech_translation.tasks.has_pretraining import HasPretraining

logger = logging.getLogger(__name__)


@register_task("custom_speech_to_text")
class CustomSpeechToTextTask(LegacyFairseqTask, HasPretraining):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--pretrain-scheme",
            default=None
        )
        parser.add_argument(
            "--cache-manager",
            default=True,
            action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--r2l",
            default=False,
            action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            '--joint-l2r-r2l',
            default=False,
            action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            '--decode-direction',
            default="l2r",
            type=str
        )

    def __init__(self, args, tgt_dict):
        LegacyFairseqTask.__init__(self, args)
        HasPretraining.__init__(self, PRETRAIN_SCHEMES[args.pretrain_scheme] if args.pretrain_scheme else None)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)

        self.use_cache_manager = args.cache_manager
        self.r2l = args.r2l
        self.joint_l2r_r2l = args.joint_l2r_r2l
        self.decode_direction = args.decode_direction

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data) / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = CustomDictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "joint_l2r_r2l", False):
            logger.info("--joint-l2r-r2l set, will add direction tags...")
            tgt_dict.add_special_symbol(L2R_TAG)
            tgt_dict.add_special_symbol(R2L_TAG)

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)

        creator = SpeechToTextDatasetCreator
        if self.use_cache_manager and is_train_split:
            # Only cache in training
            creator = CachedSpeechToTextDatasetCreator

        self.datasets[split] = creator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

        if self.r2l or self.joint_l2r_r2l:
            self.datasets[split] = R2LSpeechToTextDataset(self.datasets[split], self.tgt_dict,
                                                          bidirectional=self.joint_l2r_r2l)

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(CustomSpeechToTextTask, self).build_model(args)

    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if CachedSpeechToTextDataset.is_lang_tag(s)
        }
        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}

        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids
        if self.joint_l2r_r2l:
            if self.decode_direction == "l2r":
                extra_gen_cls_kwargs["eos"] = self.tgt_dict.index(L2R_TAG)
            elif self.decode_direction == "r2l":
                extra_gen_cls_kwargs["eos"] = self.tgt_dict.index(R2L_TAG)
            else:
                raise ValueError(f"Unknown decoding direction {self.decode_direction}.")

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False, subepoch=None):
        if not self.joint_l2r_r2l:
            return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        else:
            sample['net_input'] = sample['r2l_net_input']
            r2l_loss, r2l_sample_size, r2l_logging_output = super().train_step(sample, model, criterion, optimizer,
                                                                               update_num, ignore_grad)
            sample['net_input'] = sample['l2r_net_input']
            l2r_loss, l2r_sample_size, l2r_logging_output = super().train_step(sample, model, criterion, optimizer,
                                                                               update_num, ignore_grad)

            return 0.5 * (r2l_loss + l2r_loss), \
                   r2l_sample_size + l2r_sample_size, \
                   {k: r2l_logging_output[k] + l2r_logging_output[k] for k in r2l_logging_output.keys()}

    def valid_step(self, sample, model, criterion):
        if not self.joint_l2r_r2l:
            return super().valid_step(sample, model, criterion)
        else:
            sample['net_input'] = sample['r2l_net_input']
            r2l_loss, r2l_sample_size, r2l_logging_output = super().valid_step(sample, model, criterion)
            sample['net_input'] = sample['l2r_net_input']
            l2r_loss, l2r_sample_size, l2r_logging_output = super().valid_step(sample, model, criterion)

            return 0.5 * (r2l_loss + l2r_loss), \
                   r2l_sample_size + l2r_sample_size, \
                   {k: r2l_logging_output[k] + l2r_logging_output[k] for k in r2l_logging_output.keys()}

    def state_dict(self):
        state_dict = LegacyFairseqTask.state_dict(self)
        state_dict |= HasPretraining.state_dict(self)
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        LegacyFairseqTask.load_state_dict(self, state_dict)
        HasPretraining.load_state_dict(self, state_dict)
