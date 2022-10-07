import logging
import os.path
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Iterator, Callable, Any
import math
import warnings

import torch
from omegaconf import II, DictConfig
from torchtyping import TensorType

from speech_translation.data.repeat_dataset import RepeatDataset
from fairseq import checkpoint_utils, metrics, utils, search
from fairseq.data import Dictionary, encoders, data_utils
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.dataclass.configs import GenerationConfig
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.sequence_generator import (
    SequenceGenerator, EnsembleModel
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.tasks.translation import load_langpair_dataset as load_mt_langpair_dataset, TranslationTask
from fairseq.utils import apply_to_sample
from speech_translation.models.dummy_model import DummyEncoder
from speech_translation.data import MultiModalRoundRobinZipDatasets
from speech_translation.data.cached_speech_to_text_dataset import CachedSpeechToTextDatasetCreator
from speech_translation.models.cascaded_st_model import CascadedSTModel
from speech_translation.models.weighted_ensemble import WeightedEnsembleModel
from speech_translation.typings import NetInput, JointSample, Sample, PivNetInput, STCriterionSample
from speech_translation.utils import move_eos_to_beginning

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig(FairseqDataclass):
    data: str = II("task.data_mt")
    source_lang: Optional[str] = II("task.source_lang")
    target_lang: Optional[str] = II("task.target_lang")
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=False,
        metadata={"help": "pad the source on the left. Observe per default, the source is right padded as well!"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
                    "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
                    "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    disable_normalize_mt_beam_scores: bool = field(
        default=False, metadata={"help": "beam scores sum up to 1, ignored in ensembling"}
    )


@dataclass
class ASRConfig(FairseqDataclass):
    data: str = II("task.data_asr")
    config_yaml: str = II("task.config_yaml_asr")
    max_source_positions: int = II("task.max_source_positions")
    max_target_positions: int = II("task.max_pivot_positions")


@dataclass
class ASRGenerationConfig(GenerationConfig):
    beam: int = II('task.pivot_beam_generate')
    nbest: int = II('task.pivot_beam')
    max_len: int = II('task.max_pivot_positions')
    max_len_b: int = II('task.max_pivot_positions')
    pivot_spm_model: Optional[str] = II('task.pivot_spm_model')
    disable_asr_length_norm_score: bool = field(
        default=False, metadata={"help": "disable length normalization of asr scores"}
    )
    disable_asr_norm_score_train: bool = field(
        default=False, metadata={"help": "disable normalization during training, such that sum_piv p(piv|src)=1"}
    )
    disable_asr_norm_score_generation: bool = field(
        default=False, metadata={"help": "disable normalization during translation, such that sum_piv p(piv|src)=1"}
    )
    diverse_pivots: bool = field(
        default=False
    )
    sampling: bool = II('task.asr_sampling')


@dataclass
class CascadedSpeechTranslationConfig(FairseqDataclass):
    asr_task_cfg: ASRConfig = ASRConfig()
    mt_task_cfg: TranslationConfig = TranslationConfig()

    data_st: Optional[str] = field(
        default=None,
        metadata={"help": "manifest root path"}
    )
    data_asr: Optional[str] = field(
        default=None,
        metadata={"help": "manifest root path"}
    )
    data_mt: Optional[str] = field(
        default=None,
        metadata={"help": "data-bin of mt data"}
    )
    config_yaml_st: Optional[str] = field(
        default='config.yaml',
        metadata={"help": "Configuration YAML filename (under manifest root)"}
    )
    config_yaml_asr: Optional[str] = field(
        default='config.yaml',
        metadata={"help": "Configuration YAML filename (under manifest root)"}
    )
    max_source_positions: int = field(
        default=6000,
        metadata={"help": "max number of tokens in the source sequence"}
    )
    max_pivot_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the pivot sequence"}
    )
    max_target_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"}
    )
    asr_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "path to asr model checkpoint"}
    )
    mt_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "path to mt model checkpoint"}
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    train_subset: str = II("dataset.train_subset")
    seed: int = II("common.seed")

    asr_sample_ratio: float = field(
        default=1
    )
    mt_sample_ratio: float = field(
        default=1
    )
    st_sample_ratio: float = field(
        default=1
    )

    asr_generation: ASRGenerationConfig = ASRGenerationConfig()

    pivot_beam: int = field(
        default=4
    )
    pivot_beam_generate: int = field(
        default=12
    )

    pivot_spm_model: Optional[str] = field(
        default=None,
        metadata={"help": "re-apply sentencepiece model to generated pivot"}
    )

    asr_mt_model_parallel: bool = field(
        default=False,
    )

    model_parallel_asr_device: str = field(
        default="cuda:1"
    )
    model_parallel_mt_device: str = field(
        default="cuda:0"
    )

    eval_task: ChoiceEnum(["asr", "mt", "st"]) = field(
        default="st"
    )
    print_beams: bool = field(
        default=False,
        metadata={"help": "Only used in generation. If set, will print all pivot beams and scores."}
    )
    cache_manager: bool = field(
        default=False
    )

    ensemble_decoding: bool = field(
        default=False
    )
    ensemble_decoding_weight_asr: bool = field(
        default=False
    )
    r2l_model_path: Optional[str] = field(default=None)
    r2l_mode: ChoiceEnum(["gen", "rescore"]) = field(
        default="gen"
    )
    alternated_training: bool = field(
        default=False
    )
    asr_sampling: bool = field(
        default=False,
        metadata={"help": "Use sampling instead of beam search for transcript generation."}
    )
    asr_eval_temp: float = field(
        default=1.0
    )


@register_task("cascaded_speech_translation", dataclass=CascadedSpeechTranslationConfig)
class CascadedSpeechTranslationTask(FairseqTask):
    cfg: CascadedSpeechTranslationConfig
    asr_task: SpeechToTextTask
    mt_task: TranslationTask

    def __init__(self, cfg: CascadedSpeechTranslationConfig, piv_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = piv_dict
        self.tgt_dict = tgt_dict

        self.st_data_cfg = S2TDataConfig(Path(cfg.data_st) / cfg.config_yaml_st)
        self.asr_data_cfg = S2TDataConfig(Path(cfg.data_asr) / cfg.config_yaml_asr) if cfg.data_asr else None
        self.mt_data_path = cfg.data_mt

        assert cfg.source_lang is not None and cfg.target_lang is not None
        self.source_lang = cfg.source_lang
        self.target_lang = cfg.target_lang

        self.asr_checkpoint = cfg.asr_checkpoint
        self.mt_checkpoint = cfg.mt_checkpoint

        # noinspection PyTypeChecker
        self.asr_task = SpeechToTextTask.setup_task(cfg.asr_task_cfg)
        self.mt_task = TranslationTask.setup_task(cfg.mt_task_cfg)

        # Infer joint training from given train subset
        self.is_joint_training = (len(cfg.train_subset.split(',')) == 3)
        if self.is_joint_training:
            print('INFO: Will perform joint-training of asr, mt and st criterions.')
        else:
            print('INFO: Will only train st criterion.')

        self.asr_generator = None

        self.asr_mt_model_parallel = cfg.asr_mt_model_parallel

        self.pivot_spm = None
        if self.cfg.asr_generation.pivot_spm_model is not None:
            import sentencepiece as spm

            self.pivot_spm = spm.SentencePieceProcessor()
            self.pivot_spm.Load(self.cfg.asr_generation.pivot_spm_model)

        self.eval_task = cfg.eval_task
        self.freeze_components = None
        self.alternated_training = cfg.alternated_training

        self.r2l_model = None
        self.asr_r2l_generator = None
        self.r2l_mode = cfg.r2l_mode

    @classmethod
    def setup_task(cls, cfg: CascadedSpeechTranslationConfig, **kwargs):
        st_cfg = S2TDataConfig(Path(cfg.data_st) / cfg.config_yaml_st)
        tgt_dict_path = Path(cfg.data_st) / st_cfg.vocab_filename
        if not tgt_dict_path.is_file():
            raise FileNotFoundError(f"Pivot dict not found: {tgt_dict_path.as_posix()}")
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())
        logger.info(
            f"target dictionary size ({st_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        assert cfg.data_mt or cfg.data_asr, "Either --data-asr or --data-mt must be set (this will determine the source vocabulary)!"
        if cfg:
            asr_cfg = S2TDataConfig(Path(cfg.data_asr) / cfg.config_yaml_asr)
            piv_dict_path = Path(cfg.data_asr) / asr_cfg.vocab_filename
            if not piv_dict_path.is_file():
                raise FileNotFoundError(f"Target dict not found: {piv_dict_path.as_posix()}")
            src_dict = Dictionary.load(piv_dict_path.as_posix())
            logger.info(
                f"source dictionary size ({asr_cfg.vocab_filename}): " f"{len(src_dict):,}"
            )
        else:
            src_dict = cls.load_dictionary(
                os.path.join(cfg.data_mt, "dict.{}.txt".format(cfg.source_lang))
            )

        return cls(cfg, src_dict, tgt_dict)

    def build_criterion(self, cfg: DictConfig):
        from fairseq import criterions

        if self.st_data_cfg.prepend_tgt_lang_tag and cfg.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        # noinspection PyTypeChecker
        return criterions.build_criterion(cfg, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        st_split = split
        asr_split = None
        mt_split = None

        # train split passed in order st, asr, mt or just st
        is_train_split = split.startswith('train')
        is_multi_split = False
        if is_train_split:
            splits = split.split(',')
            is_multi_split = (len(splits) > 1)

            if is_multi_split:
                st_split, asr_split, mt_split = splits

        creator = SpeechToTextDatasetCreator
        if self.cfg.cache_manager and is_train_split:
            # Only cache in training
            creator = CachedSpeechToTextDatasetCreator

        st_dataset = creator.from_tsv(
            self.cfg.data_st,
            self.st_data_cfg,
            st_split,
            self.tgt_dict,
            self.build_tokenizer(self.cfg, task='st'),
            self.build_bpe(self.cfg, task='st'),
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed,
        )

        if is_train_split and self.alternated_training:
            st_dataset = RepeatDataset(st_dataset, repeats=2)

        if not is_multi_split:
            self.datasets[split] = st_dataset
        else:
            asr_dataset = creator.from_tsv(
                self.cfg.data_asr,
                self.asr_data_cfg,
                asr_split,
                self.src_dict,
                self.build_tokenizer(self.cfg, task='asr'),
                self.build_bpe(self.cfg, task='asr'),
                is_train_split=is_train_split,
                epoch=epoch,
                seed=self.cfg.seed,
            )

            mt_dataset = load_mt_langpair_dataset(
                self.cfg.data_mt,
                mt_split,
                self.source_lang,
                self.src_dict,
                self.target_lang,
                self.target_dictionary,
                combine=combine,
                dataset_impl=self.cfg.mt_task_cfg.dataset_impl,
                upsample_primary=self.cfg.mt_task_cfg.upsample_primary,
                left_pad_source=self.cfg.mt_task_cfg.left_pad_source,
                left_pad_target=self.cfg.mt_task_cfg.left_pad_target,
                max_source_positions=self.cfg.max_pivot_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.mt_task_cfg.load_alignments,
                truncate_source=self.cfg.mt_task_cfg.truncate_source,
                num_buckets=self.cfg.mt_task_cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.mt_task_cfg.required_seq_len_multiple,
            )

            self.datasets[split] = MultiModalRoundRobinZipDatasets({
                'st': st_dataset,
                'asr': asr_dataset,
                'mt': mt_dataset
            })

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def asr_pivot_dictionary(self):
        return self.asr_task.target_dictionary

    @property
    def mt_pivot_dictionary(self):
        return self.mt_task.source_dictionary

    @property
    def source_dictionary(self):
        # There is no source dictionary (audio features)
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_model(self, cfg):
        cfg.input_feat_per_channel = self.asr_data_cfg.input_feat_per_channel
        cfg.input_channels = self.asr_data_cfg.input_channels

        # Create models
        model: CascadedSTModel = CascadedSTModel.build_model(cfg, self)
        self.freeze_components = getattr(cfg, "freeze_components", None)
        if self.freeze_components:
            assert not self.alternated_training, "--alternated-training and --freeze-components are mutually exclusive"

        def load_state(path: str):
            state = checkpoint_utils.load_checkpoint_to_cpu(path)
            return state['model']

        if self.mt_checkpoint is not None:
            logger.info(f'Loading MT model from {self.mt_checkpoint}...')
            model.load_model_params('mt', load_state(self.mt_checkpoint))
            # TODO: Currently only supports bilingual mt models

        # Create ASR generator
        if self.cfg.asr_generation.diverse_pivots:
            logger.info('Diverse beam search for pivots enabled.')
            self.cfg.asr_generation.diverse_beam_groups = self.cfg.asr_generation.beam

        self.asr_generator = self.asr_task.build_generator([model.asr_model], self.cfg.asr_generation,
                                                           extra_gen_cls_kwargs={
                                                               "max_len": self.cfg.max_pivot_positions})

        r2l_model_path = getattr(self.cfg, "r2l_model_path", None)
        if r2l_model_path:
            logger.info("Loading r2l model...")
            cfg.load_pretrained_encoder_from = r2l_model_path
            cfg.load_pretrained_decoder_from = r2l_model_path
            self.r2l_model = CascadedSTModel.build_model(cfg, self).asr_model
            self.asr_r2l_generator = self.asr_task.build_generator([self.r2l_model], self.cfg.asr_generation,
                                                                    extra_gen_cls_kwargs={
                                                                    "max_len": self.cfg.max_pivot_positions})
        return model

    def build_generator(
            self,
            models,
            cfg: GenerationConfig,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
            **kwargs
    ):
        if self.st_data_cfg.prepend_tgt_lang_tag and cfg.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}

        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

        return super().build_generator(
            models, cfg, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_mt_generator(
            self, models,
            max_in_out_shift: int = 1024,
            max_in_out_ratio: float = 0,
            beam: int = 5,
            min_len: int = 1,
            normalize_scores: bool = True,
            lenpen: int = 1,
            unkpen: int = 0,
            temperature: float = 1.0,
            match_source_len: bool = False,
            no_repeat_ngram_size: int = 0
    ):
        search_strategy = search.BeamSearch(self.target_dictionary)

        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=beam,
            max_len_a=max_in_out_ratio,
            max_len_b=max_in_out_shift,
            min_len=min_len,
            max_len=self.cfg.max_target_positions,
            normalize_scores=normalize_scores,
            len_penalty=lenpen,
            unk_penalty=unkpen,
            temperature=temperature,
            match_source_len=match_source_len,
            no_repeat_ngram_size=no_repeat_ngram_size,
            search_strategy=search_strategy,
        )

    def build_tokenizer(self, cfg, task='st'):
        if task == 'asr':
            conf = self.asr_data_cfg.pre_tokenizer
        else:
            conf = self.st_data_cfg.pre_tokenizer

        logger.info(f"pre-tokenizer: {conf}")
        return encoders.build_tokenizer(Namespace(**conf))

    def build_bpe(self, cfg, task='st'):
        if task == 'asr':
            conf = self.asr_data_cfg.bpe_tokenizer
        else:
            conf = self.st_data_cfg.bpe_tokenizer
        logger.info(f"tokenizer: {conf}")
        return encoders.build_bpe(Namespace(**conf))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.st_data_cfg, src_tokens, src_lengths
        )

    def compute_vbatch(self, batch_size: int, beam: int, spl: float, device) -> tuple[Callable,
                                                                                      tuple[torch.Tensor, ...]]:
        if spl < 0:
            max_v_bsz = batch_size
        elif spl == 0:
            max_v_bsz = max(1, batch_size // beam)
        else:
            max_v_bsz = max(1, int(batch_size / spl))

        idxs_tup = torch.arange(batch_size, device=device).split(max_v_bsz)

        def select_idxs_recursively(x: Any, idxs: torch.Tensor, expected_size: int, dim: int = 0) -> Any:
            """
            Selects tensor indices given by idxs recursively by axis 0,
            if the size of axis 0 corresponds to expected_size
            (should be changed once PyTorch NamedTensors are finalized)
            """

            if isinstance(x, (int, str, float, bool)):
                return x
            if isinstance(x, torch.Tensor):
                if x.size(dim) == expected_size:
                    return torch.index_select(x, dim, idxs)
                else:
                    return x
            if isinstance(x, list):
                return [select_idxs_recursively(y, idxs, expected_size, dim) for y in x]
            if isinstance(x, set):
                return set(select_idxs_recursively(y, idxs, expected_size, dim) for y in x)
            if isinstance(x, tuple):
                return tuple(select_idxs_recursively(y, idxs, expected_size, dim) for y in x)
            if isinstance(x, dict):
                return {k: select_idxs_recursively(v, idxs, expected_size, dim) for k, v in x.items()}
            if x is None:
                return None
            raise ValueError(f'select_idxs_recursively cannot handle type {type(x)}')

        def vbatch_iterator(**kwargs: Any) -> Iterator[tuple[int, ...]]:
            outputs = {}
            for i, idxs in enumerate(idxs_tup):
                v_bsz = len(idxs)
                for k, v in kwargs.items():
                    sel_dim, obj = v
                    outputs[k] = select_idxs_recursively(obj, idxs, expected_size=batch_size, dim=sel_dim)

                yield i, (v_bsz, outputs)

        return vbatch_iterator, idxs_tup

    def combine_st_logging_outputs(self, _global, _local):
        if _global is None:
            return _local

        for key in ('loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size'):
            _global[key] += _local[key]

        return _global

    def forward_st(self, st_sample: Sample, joint_model: CascadedSTModel, criterion, training: bool):
        batch_size = st_sample['id'].size(0)
        if self.asr_mt_model_parallel:
            st_sample = apply_to_sample(lambda x: x.to(self.cfg.model_parallel_asr_device), st_sample)
            joint_model.asr_model.to(self.cfg.model_parallel_asr_device)

        mt_piv_tokens, mt_piv_lengths, asr_piv_tokens, asr_piv_lengths, _ = self.generate_pivot_sequences(self.asr_generator, st_sample,
                                                                                                          joint_model.asr_model,
                                                                                                          is_training=True)  # (batch, beam, len), (batch, beam)
        v_beam_size = self.cfg.pivot_beam
        r2l_idxs = None
        if self.r2l_model is not None:
            if self.r2l_mode == "gen":
                r2l_mt_piv_tokens, r2l_mt_piv_lengths, r2l_asr_piv_tokens, r2l_asr_piv_lengths, _ = self.generate_pivot_sequences(
                        self.asr_r2l_generator, st_sample, self.r2l_model, is_training=True)
                mt_piv_tokens = torch.cat([mt_piv_tokens, r2l_mt_piv_tokens], dim=1)
                mt_piv_lengths = torch.cat([mt_piv_lengths, r2l_mt_piv_lengths], dim=1)
                asr_piv_tokens = torch.cat([asr_piv_tokens, r2l_asr_piv_tokens], dim=1)
                asr_piv_lengths = torch.cat([asr_piv_lengths, r2l_asr_piv_lengths], dim=1)

                # TODO add padding
                r2l_idxs = list(range(v_beam_size, v_beam_size * 2, 1))
                v_beam_size *= 2
            elif self.r2l_mode == "rescore":
                raise NotImplementedError()
            else:
                raise NotImplementedError()

        asr_encoder_out = joint_model.encoder.forward_asr_encoder(st_sample['net_input']['src_tokens'],
                                                                  st_sample['net_input']['src_lengths'])
        # (batch, time, dim)
        vbatch_iterator, v_idxs = self.compute_vbatch(batch_size, self.cfg.pivot_beam, spl=0,
                                                      device=mt_piv_tokens.device)
        num_vbatches = len(v_idxs)

        total_loss, total_sample_size, global_logging_outputs = 0, 0, None
        for vbatch_id, (v_bsz, vbatch) in vbatch_iterator(st_sample=(0, st_sample),
                                                          asr_encoder_out=(1, asr_encoder_out['encoder_out']),
                                                          asr_encoder_padding_mask=(
                                                                  0, asr_encoder_out['encoder_padding_mask']),
                                                          asr_encoder_embedding=(
                                                                  0, asr_encoder_out['encoder_embedding']),
                                                          asr_encoder_states=(1, asr_encoder_out['encoder_states']),
                                                          mt_piv_tokens=(0, mt_piv_tokens),
                                                          mt_piv_lengths=(0, mt_piv_lengths),
                                                          asr_piv_tokens=(0, asr_piv_tokens),
                                                          asr_piv_lengths=(0, asr_piv_lengths)
                                                          ):
            v_mt_piv_tokens = vbatch['mt_piv_tokens'].view(v_bsz * v_beam_size, -1)
            v_mt_piv_lengths = vbatch['mt_piv_lengths'].view(v_bsz * v_beam_size, -1)
            v_asr_piv_tokens = vbatch['asr_piv_tokens'].view(v_bsz * v_beam_size, -1)
            v_asr_piv_lengths = vbatch['asr_piv_lengths'].view(v_bsz * v_beam_size, -1)

            pseudo_encoder_input: PivNetInput = vbatch['st_sample']['net_input']
            pseudo_encoder_input.update({
                'mt_piv_tokens': v_mt_piv_tokens,
                'mt_piv_lengths': v_mt_piv_lengths,
                'asr_piv_tokens': v_asr_piv_tokens,
                'asr_piv_lengths': v_asr_piv_lengths,
                'asr_prev_output_tokens': move_eos_to_beginning(v_asr_piv_tokens,
                                                                eos_token=self.asr_pivot_dictionary.eos(),
                                                                pad_token=self.asr_pivot_dictionary.pad()),
                "length_normalize_asr_beam_scores": not self.cfg.asr_generation.disable_asr_length_norm_score,
                "r2l_idxs": r2l_idxs
            })

            if self.asr_mt_model_parallel:
                pseudo_encoder_input.update({
                    'asr_device': self.cfg.model_parallel_asr_device,
                    'mt_device': self.cfg.model_parallel_mt_device
                })

            v_asr_encoder_out = {
                "encoder_out": [x.repeat_interleave(repeats=v_beam_size, dim=1) for x in
                                vbatch['asr_encoder_out']],
                "encoder_padding_mask": [x.repeat_interleave(repeats=v_beam_size, dim=0) for x in
                                         vbatch['asr_encoder_padding_mask']],
                "encoder_embedding": [x.repeat_interleave(repeats=v_beam_size, dim=0) for x in
                                      vbatch['asr_encoder_embedding']],
                "encoder_states": [x.repeat_interleave(repeats=v_beam_size, dim=1) for x in
                                   vbatch['asr_encoder_states']],
            }

            pseudo_encoder_output = joint_model.encoder.forward(asr_encoder_out=v_asr_encoder_out,
                                                                **pseudo_encoder_input)  # (piv_len, batch * beam, dim)
            decoder_prev_output_tokens = vbatch['st_sample']['net_input']['prev_output_tokens'].repeat_interleave(
                v_beam_size, dim=0)

            if self.asr_mt_model_parallel:
                vbatch['st_sample']['net_input']['prev_output_tokens'] = vbatch['st_sample']['net_input'][
                    'prev_output_tokens'].to(self.cfg.model_parallel_mt_device)
                vbatch['st_sample']['id'] = vbatch['st_sample']['id'].to(self.cfg.model_parallel_mt_device)
                vbatch['st_sample']['target'] = vbatch['st_sample']['target'].to(self.cfg.model_parallel_mt_device)
                vbatch['st_sample']['target_lengths'] = vbatch['st_sample']['target_lengths'].to(
                    self.cfg.model_parallel_mt_device)
                pseudo_encoder_output['asr_score'] = pseudo_encoder_output['asr_score'].to(
                    self.cfg.model_parallel_mt_device)
                decoder_prev_output_tokens = decoder_prev_output_tokens.to(self.cfg.model_parallel_mt_device)

            decoder_output, decoder_extra = joint_model.decoder.forward(
                prev_output_tokens=decoder_prev_output_tokens,
                encoder_out=pseudo_encoder_output,
                return_all_hiddens=False,
                features_only=False,
            )
            decoder_output = decoder_output.view(v_bsz, self.cfg.pivot_beam, decoder_output.size(1),
                                                 decoder_output.size(2))

            asr_score = pseudo_encoder_output['asr_score'].view(v_bsz, self.cfg.pivot_beam)
            if not self.cfg.asr_generation.disable_asr_norm_score_train:
                # Normalize ASR scores (sum up to 1)
                asr_score = asr_score - asr_score.logsumexp(1, keepdim=True)

            criterion_sample = STCriterionSample(
                asr_score=asr_score,
                decoder_out=(decoder_output, decoder_extra),
                id=vbatch['st_sample']['id'],
                net_input={},
                ntokens=vbatch['st_sample']['target_lengths'].sum(),
                nsentences=v_bsz,
                target=vbatch['st_sample']['target'],
                target_lengths=vbatch['st_sample']['target_lengths'],
            )

            raw_loss, sample_size, logging_output = criterion.st_criterion(joint_model, criterion_sample)
            total_loss += raw_loss.detach()
            total_sample_size += sample_size
            global_logging_outputs = self.combine_st_logging_outputs(global_logging_outputs, logging_output)

            if training:
                weighted_loss = criterion.update_st_loss(raw_loss)
                weighted_loss.backward(retain_graph=vbatch_id < num_vbatches - 1)

        return total_loss, total_sample_size, global_logging_outputs

    def do_train_mt(self):
        return (
            self.is_joint_training
            and (
                not self.freeze_components
                or not all(x in self.freeze_components.split(',') for x in ['mt_enc', 'mt_dec'])
            )
        )

    def do_train_asr(self):
        return (
            self.is_joint_training
            and (
                not self.freeze_components
                or not all(x in self.freeze_components.split(',') for x in ['asr_enc', 'asr_dec'])
            )
        )

    def train_step(self, sample: Union[Sample, JointSample], model: CascadedSTModel, criterion, optimizer, update_num,
                   subepoch=None, ignore_grad=False, **kwargs):
        if ignore_grad:
            raise NotImplementedError()

        model.train()
        model.set_num_updates(update_num)
        with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
            if self.alternated_training:
                if update_num % 2 == 0:
                    model.update_freeze_component("asr_enc,asr_dec")
                else:
                    model.update_freeze_component("mt_enc,mt_dec")

            if self.is_joint_training:
                sample: JointSample
                st_loss, st_sample_size, st_logging_output = self.forward_st(sample['st'], model, criterion,
                                                                             training=True)
                asr_loss, mt_loss = 0, 0
                asr_sample_size, mt_sample_size = 0, 0
                asr_logging_output, mt_logging_output = {}, {}
                if self.do_train_mt():
                    mt_loss, mt_sample_size, mt_logging_output = self.mt_task.train_step(sample['mt'],
                                                                                         model.mt_model,
                                                                                         criterion.mt_criterion,
                                                                                         optimizer, update_num, ignore_grad)
                    mt_loss = criterion.update_mt_loss(mt_loss, update_num)
                if self.do_train_asr():
                    asr_loss, asr_sample_size, asr_logging_output = self.asr_task.train_step(sample['asr'],
                                                                                             model.asr_model,
                                                                                             criterion.asr_criterion,
                                                                                             optimizer, update_num,
                                                                                             ignore_grad)
                    asr_loss = criterion.update_asr_loss(asr_loss, update_num)

                loss = sum([asr_loss + mt_loss + st_loss])
                sample_size = sum([mt_sample_size, asr_sample_size, st_sample_size])
                logging_output = mt_logging_output | asr_logging_output | st_logging_output
            else:
                sample: Sample
                loss, sample_size, logging_output = self.forward_st(sample, model, criterion, training=True)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self.forward_st(sample, model, criterion, training=False)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        assert len(models) == 1, 'Ensembling not implemented'
        joint_model = models[0]
        joint_model.eval()

        with torch.no_grad():
            batch_size = sample['id'].size(0)
            piv_beam_size = self.cfg.pivot_beam
            final_hyp = [None for _ in range(batch_size)] if not self.cfg.ensemble_decoding else []

            mt_piv_tokens, mt_piv_lengths, asr_piv_tokens, asr_piv_lengths, asr_scores = self.generate_pivot_sequences(
                self.asr_generator,
                sample,
                joint_model.asr_model,
                is_training=False)  # (batch, beam, len), (batch, beam)

            asr_encoder_out = joint_model.encoder.forward_asr_encoder(sample['net_input']['src_tokens'],
                                                                      sample['net_input']['src_lengths'])
            # (batch, time, dim)
            vbatch_iterator, v_idxs = self.compute_vbatch(batch_size, self.cfg.pivot_beam, spl=0,
                                                          device=mt_piv_tokens.device)

            for vbatch_id, (v_bsz, vbatch) in vbatch_iterator(st_sample=(0, sample),
                                                              asr_encoder_out=(1, asr_encoder_out['encoder_out']),
                                                              asr_encoder_padding_mask=(
                                                                      0, asr_encoder_out['encoder_padding_mask']),
                                                              asr_encoder_embedding=(
                                                                      0, asr_encoder_out['encoder_embedding']),
                                                              asr_encoder_states=(1, asr_encoder_out['encoder_states']),
                                                              mt_piv_tokens=(0, mt_piv_tokens),
                                                              mt_piv_lengths=(0, mt_piv_lengths),
                                                              asr_piv_tokens=(0, asr_piv_tokens),
                                                              asr_piv_lengths=(0, asr_piv_lengths),
                                                              asr_scores=(0, asr_scores)):
                v_mt_piv_tokens = vbatch['mt_piv_tokens'].view(v_bsz * self.cfg.pivot_beam, -1)
                v_mt_piv_lengths = vbatch['mt_piv_lengths'].view(v_bsz * self.cfg.pivot_beam, -1)
                v_asr_piv_tokens = vbatch['asr_piv_tokens'].view(v_bsz * self.cfg.pivot_beam, -1)
                v_asr_piv_lengths = vbatch['asr_piv_lengths'].view(v_bsz * self.cfg.pivot_beam, -1)
                v_asr_scores = vbatch['asr_scores']

                v_asr_scores = v_asr_scores ** self.cfg.asr_eval_temp

                pseudo_encoder_input: PivNetInput = vbatch['st_sample']['net_input']
                pseudo_encoder_input.update({
                    'mt_piv_tokens': v_mt_piv_tokens,
                    'mt_piv_lengths': v_mt_piv_lengths,
                    'asr_piv_tokens': v_asr_piv_tokens,
                    'asr_piv_lengths': v_asr_piv_lengths,
                    'asr_prev_output_tokens': move_eos_to_beginning(v_asr_piv_tokens,
                                                                    eos_token=self.asr_pivot_dictionary.eos(),
                                                                    pad_token=self.asr_pivot_dictionary.pad()),
                    "normalize_asr_beam_scores": not self.cfg.asr_generation.disable_asr_length_norm_score
                })
                if self.asr_mt_model_parallel:
                    pseudo_encoder_input.update({
                        'asr_device': self.cfg.model_parallel_asr_device,
                        'mt_device': self.cfg.model_parallel_mt_device
                    })

                v_asr_encoder_out = {
                    "encoder_out": [x.repeat_interleave(repeats=self.cfg.pivot_beam, dim=1) for x in
                                    vbatch['asr_encoder_out']],
                    "encoder_padding_mask": [x.repeat_interleave(repeats=self.cfg.pivot_beam, dim=0) for x in
                                             vbatch['asr_encoder_padding_mask']],
                    "encoder_embedding": [x.repeat_interleave(repeats=self.cfg.pivot_beam, dim=0) for x in
                                          vbatch['asr_encoder_embedding']],
                    "encoder_states": [x.repeat_interleave(repeats=self.cfg.pivot_beam, dim=1) for x in
                                       vbatch['asr_encoder_states']],
                }

                pseudo_encoder_output = joint_model.encoder.forward(asr_encoder_out=v_asr_encoder_out,
                                                                    **pseudo_encoder_input)  # (piv_len, batch * beam, dim)

                if self.cfg.ensemble_decoding:
                    decoder_input = {'net_input': {
                        "src_tokens": vbatch['st_sample']['net_input']['src_tokens'],
                        "src_lengths": vbatch['st_sample']['net_input']['src_lengths'],
                    }}
                    pseudo_encoder_output_tensor = pseudo_encoder_output['encoder_out'][0]

                    pseudo_encoder_output['encoder_out'][0] = pseudo_encoder_output_tensor.view(
                        pseudo_encoder_output_tensor.size(0),
                        v_bsz,
                        piv_beam_size,
                        pseudo_encoder_output_tensor.size(-1)
                    ).permute(2, 0, 1, 3)  # (beam, piv_len, batch, dim)
                    pseudo_encoder_output['encoder_padding_mask'][0] = pseudo_encoder_output['encoder_padding_mask'][0].view(
                        v_bsz,
                        piv_beam_size,
                        -1
                    ).permute(1, 0, 2)
                    pseudo_encoder_output['encoder_embedding'][0] = pseudo_encoder_output['encoder_embedding'][0].view(
                        v_bsz,
                        piv_beam_size,
                        pseudo_encoder_output['encoder_embedding'][0].size(1),
                        pseudo_encoder_output['encoder_embedding'][0].size(2)
                    ).permute(1, 0, 2, 3)
                    pseudo_encoder_output['src_lengths'][0] = pseudo_encoder_output['src_lengths'][0].view(
                        v_bsz,
                        piv_beam_size,
                        -1
                    ).permute(1, 0, 2)

                    pseudo_encoder_splits = []
                    for piv_beam in range(piv_beam_size):
                        spl = {}
                        for key, val in pseudo_encoder_output.items():
                            if key not in ['encoder_out', 'encoder_padding_mask', 'encoder_embedding', 'src_lengths']:
                                spl[key] = val
                            else:
                                spl[key] = [val[0][piv_beam]]
                        pseudo_encoder_splits.append(spl)

                    generator.model = WeightedEnsembleModel(
                        [FairseqEncoderDecoderModel(encoder=DummyEncoder(spl,
                                                                         actual_encoder=joint_model.encoder.mt_encoder),
                                                    decoder=joint_model.decoder) for spl in pseudo_encoder_splits],
                        weights=v_asr_scores.exp()
                        if self.cfg.ensemble_decoding_weight_asr else torch.ones_like(v_asr_scores))
                else:
                    decoder_input = {'net_input': {
                        "src_tokens": vbatch['st_sample']['net_input']['src_tokens'].repeat_interleave(repeats=self.cfg.pivot_beam,
                                                                                          dim=0),
                        "src_lengths": vbatch['st_sample']['net_input']['src_lengths'].repeat_interleave(repeats=self.cfg.pivot_beam,
                                                                                            dim=0),
                    }}
                    generator_model = FairseqEncoderDecoderModel(encoder=DummyEncoder(pseudo_encoder_output,
                                                                                      actual_encoder=joint_model.encoder.mt_encoder),
                                                                 decoder=joint_model.decoder)
                    generator.model = EnsembleModel([generator_model])

                hyp = generator.generate(
                    None, decoder_input, prefix_tokens=prefix_tokens, constraints=constraints
                )  # (piv_beam_size * num_sentences, tgt_beam_size) dicts or (num_sentences, tgt_beam_size) in ensemble mode

                if self.cfg.ensemble_decoding:
                    final_hyp.extend(hyp)
                else:
                    for b_id, mt_sentence in enumerate(hyp):
                        sentence_id = math.floor(b_id / piv_beam_size)
                        global_sentence_id = vbatch_id * len(v_idxs[0]) + sentence_id
                        asr_beam_id = b_id % piv_beam_size

                        if not self.cfg.mt_task_cfg.disable_normalize_mt_beam_scores:
                            norm_term = torch.tensor([beam['score'] for beam in mt_sentence])
                            norm_term = float(torch.logsumexp(norm_term, 0))
                        else:
                            norm_term = 0

                        for tgt_beam in mt_sentence:
                            tgt_beam['srcpiv_score'] = v_asr_scores[sentence_id][asr_beam_id]
                            tgt_beam['pivtgt_score'] = tgt_beam['score'] - norm_term
                            tgt_beam['score'] = tgt_beam['srcpiv_score'] + tgt_beam['pivtgt_score']

                        if self.cfg.print_beams:
                            print(f'TsHyp-{sample["id"][global_sentence_id]}.{asr_beam_id}' +
                                  f'\t{math.exp(v_asr_scores[sentence_id, asr_beam_id]):.3f}' +
                                  f"\t " + " ".join(self.asr_pivot_dictionary[t]
                                                    for t in asr_piv_tokens[global_sentence_id][asr_beam_id]))
                            print(f'TlHyp-{sample["id"][global_sentence_id]}.{asr_beam_id}' +
                                  f"\t{math.exp(mt_sentence[0]['pivtgt_score']):.3f}|tot:{math.exp(mt_sentence[0]['score']):.3f}" +
                                  f"\t " + " ".join(self.tgt_dict[t] for t in mt_sentence[0]['tokens'])
                                  )

                        curr_score = mt_sentence[0]['score']
                        prev_score = -math.inf
                        if final_hyp[global_sentence_id] is not None:
                            prev_score = final_hyp[global_sentence_id][0]['score']

                        if curr_score > prev_score:
                            final_hyp[global_sentence_id] = mt_sentence

        return final_hyp

    def generate_pivot_sequences(self, generator, st_sample: Sample, asr_model: FairseqEncoderDecoderModel, is_training: bool) -> \
            tuple[
                TensorType['batch * beam', 'time'],
                TensorType['batch * beam'],
                TensorType['batch * beam', 'time'],
                TensorType['batch * beam'],
                Optional[TensorType['batch * beam']]]:
        asr_model.eval()

        # Technically not necessary, but cleaner.
        pseudo_sample = Sample(net_input=NetInput(src_tokens=st_sample['net_input']['src_tokens'],
                                                  src_lengths=st_sample['net_input']['src_lengths']))
        with torch.no_grad():
            generated_seqs = generator.generate([asr_model], pseudo_sample)
        batch_size = len(generated_seqs)
        # levels: batch size -> piv_beam_generate -> dict(tokens, score, attention, alignment, positional_scores)

        # prune to --pivot-beam items
        pruned_sequences = [sentence[:self.cfg.pivot_beam] for sentence in generated_seqs]

        if self.pivot_spm is not None:
            for seq in pruned_sequences:
                for beam in seq:
                    dec_seq = self.asr_pivot_dictionary.string(beam['tokens'], 'sentencepiece')
                    new_tokens = self.pivot_spm.Encode(
                        dec_seq, out_type=str
                    )[:self.cfg.asr_generation.max_len_b - 1]
                    new_tokens = [self.mt_pivot_dictionary.index(tok) for tok in new_tokens] + [
                        self.mt_pivot_dictionary.eos()]
                    beam['asr_tokens'] = beam['tokens']
                    beam['tokens'] = torch.tensor(new_tokens, dtype=beam['tokens'].dtype, device=beam['tokens'].device)

        max_piv_len = max(
            len(sentence[beam]['tokens']) for sentence in pruned_sequences for beam in range(self.cfg.pivot_beam)
        )

        piv_tokens = torch.cat([data_utils.collate_tokens([beam['tokens'] for beam in sentence],
                                                          self.mt_pivot_dictionary.pad(),
                                                          self.mt_pivot_dictionary.eos(),
                                                          left_pad=False,
                                                          move_eos_to_beginning=False,
                                                          pad_to_length=max_piv_len) for sentence in pruned_sequences],
                               dim=0).view(batch_size, self.cfg.pivot_beam, -1)
        piv_lengths = torch.tensor([len(beam['tokens']) for sentence in pruned_sequences for beam in sentence],
                                   device=piv_tokens.device).view(batch_size, self.cfg.pivot_beam)

        if self.pivot_spm is not None:
            max_asr_tokens_len = max(
                len(sentence[beam]['asr_tokens']) for sentence in pruned_sequences for beam in
                range(self.cfg.pivot_beam)
            )
            asr_target = torch.cat([data_utils.collate_tokens([beam['asr_tokens'] for beam in sentence],
                                                              self.asr_pivot_dictionary.pad(),
                                                              self.asr_pivot_dictionary.eos(),
                                                              left_pad=False,
                                                              move_eos_to_beginning=False,
                                                              pad_to_length=max_asr_tokens_len)
                                    for sentence in pruned_sequences], dim=0).view(batch_size, self.cfg.pivot_beam, -1)
            asr_lengths = torch.tensor([len(beam['asr_tokens']) for sentence in pruned_sequences for beam in sentence],
                                       device=asr_target.device).view(batch_size, self.cfg.pivot_beam)
        else:
            asr_target = piv_tokens
            asr_lengths = piv_lengths

        scores = None
        if not is_training:
            scores = torch.tensor([[beam['score'] for beam in sentence] for sentence in pruned_sequences],
                                  device=piv_tokens.device)
            if not self.cfg.asr_generation.disable_asr_norm_score_generation:
                scores -= scores.logsumexp(1, keepdim=True)
        else:
            asr_model.train()

        return piv_tokens, piv_lengths, asr_target, asr_lengths, scores

    def move_model_to_devices(self, model, criterion):
        if self.asr_mt_model_parallel:
            print(
                f'Model parallism activated, moving ASR model to {self.cfg.model_parallel_asr_device} and MT model to {self.cfg.model_parallel_mt_device}.')
            model.asr_model = model.asr_model.to(self.cfg.model_parallel_asr_device)
            model.mt_model = model.mt_model.to(self.cfg.model_parallel_mt_device)
            criterion = criterion.to(self.cfg.model_parallel_mt_device)
        else:
            model = model.to('cuda')
            criterion = criterion.to('cuda')

        return model, criterion

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs,
                                           do_mt_training=self.do_train_mt(),
                                           do_asr_training=self.do_train_asr())
