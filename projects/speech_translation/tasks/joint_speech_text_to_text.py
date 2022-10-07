import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import II

from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task, FairseqTask
from fairseq.tasks.translation import load_langpair_dataset

from speech_translation.data import MultiModalRoundRobinZipDatasets
from speech_translation.data.cached_speech_to_text_dataset import CachedSpeechToTextDatasetCreator
from speech_translation.data.transform_target_eos_dataset import TransformTargetEosDataset


logger = logging.getLogger(__name__)


@dataclass
class JointSpeechTextTranslationConfig(FairseqDataclass):
    data_speech: Optional[str] = field(
        default=None,
        metadata={"help": "manifest root path"}
    )
    data_text: Optional[str] = field(
        default=None,
        metadata={"help": "data-bin of mt data"}
    )
    speech_config_yaml: Optional[str] = field(
        default='config.yaml',
        metadata={"help": "Configuration YAML filename (under manifest root)"}
    )
    joint_vocabulary_path: Optional[str] = field(
        default=None
    )
    max_speech_source_positions: int = field(
        default=6000,
        metadata={"help": "max number of tokens in the source sequence"}
    )
    max_text_source_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"}
    )
    train_subset: str = II("dataset.train_subset")
    train_directions: Optional[str] = field(default=None)
    eval_direction: Optional[str] = field(default=None)
    seed: int = II("common.seed")
    speech_cache_manager: bool = field(
        default=False
    )

    decoder_langtok: bool = field(
        default=False
    )


def _lang_token(lang: str):
    return "__{}__".format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx


@register_task("joint_speech_text_to_text", dataclass=JointSpeechTextTranslationConfig)
class JointSpeechTextToTextTask(FairseqTask):

    cfg: JointSpeechTextTranslationConfig

    def __init__(self, cfg: JointSpeechTextTranslationConfig, joint_dict):
        super().__init__(cfg)
        self.joint_dict = joint_dict

        self.speech_data_cfg = S2TDataConfig(Path(cfg.data_speech) / cfg.speech_config_yaml)
        self.mt_data_path = cfg.data_text

        self.langs = set(lang for direction in self.cfg.train_directions.split(',') for lang in direction.split('-'))
        logger.info(f"Languages inferred: {', '.join(self.langs)}")

    @classmethod
    def setup_task(cls, cfg: JointSpeechTextTranslationConfig, **kwargs):
        assert cfg.joint_vocabulary_path is not None, "--joint-vocabulary-path must be set"
        joint_vocabulary_path = Path(cfg.joint_vocabulary_path)
        if not joint_vocabulary_path.is_file():
            raise FileNotFoundError(f"Joint vocabulary file not found: {joint_vocabulary_path.as_posix()}")
        joint_dict = Dictionary.load(joint_vocabulary_path.as_posix())

        if cfg.decoder_langtok:
            decoder_langs = set(
                direction.split('-')[1] for direction in cfg.train_directions.split(',') )

            for lang in decoder_langs:
                joint_dict.add_symbol(_lang_token(lang))

            logger.info(f"adding language tokens for {', '.join(decoder_langs)}")

        logger.info(
            f"joint dictionary size ({joint_vocabulary_path.name}): " f"{len(joint_dict):,}"
        )
        return cls(cfg, joint_dict)

    def load_dataset(
        self,
        split: str,
        epoch=1,
        combine: bool = False,
        **kwargs,
    ):
        subsets = []

        is_train_split = None

        for subset in split.split(','):
            if subset.startswith('text:'):
                subset = subset.removeprefix('text:')
                assert is_train_split is None or is_train_split == subset.startswith('train')
                is_train_split = subset.startswith('train')

                dataset = load_langpair_dataset(
                    self.cfg.data_text,
                    subset,
                    self.source_lang,
                    self.src_dict,
                    self.target_lang,
                    self.target_dictionary,
                    combine=combine,
                    dataset_impl=None,
                    upsample_primary=-1,
                    left_pad_source=False,
                    left_pad_target=False,
                    max_source_positions=self.cfg.max_text_source_positions,
                    max_target_positions=self.cfg.max_target_positions,
                    load_alignments=False,
                    truncate_source=False,
                    num_buckets=0,
                    shuffle=(split != "test"),
                    pad_to_multiple=1,
                )

                subsets.append(('text', dataset))
                logger.info(f"Added text training subset '{subset}'.")
            elif subset.startswith('speech:'):
                subset = subset.removeprefix('speech:')
                assert is_train_split is None or is_train_split == subset.startswith('train')
                is_train_split = subset.startswith('train')

                speech_data_creator = SpeechToTextDatasetCreator
                if self.use_cache_manager:
                    # Only cache in training
                    speech_data_creator = CachedSpeechToTextDatasetCreator
                dataset = speech_data_creator.from_tsv(
                    self.cfg.data_speech,
                    self.speech_data_cfg,
                    subset,
                    self.tgt_dict,
                    self.build_tokenizer(self.cfg),
                    self.build_bpe(self.cfg),
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.cfg.seed,
                )

                subsets.append(('speech', dataset))
                logger.info(f"Added speech training subset '{subset}'.")
            else:
                raise ValueError(f"Invalid subset {subset}, must be prefixed with 'text:' or 'speech:'.")

        if is_train_split:
            assert len(subsets) == len(self.cfg.train_directions.split(','))

            datasets = {}
            for (modality, dataset), direction in zip(subsets, self.cfg.train_directions.split(',')):
                if self.cfg.decoder_langtok:
                    dataset = TransformTargetEosDataset(
                        dataset,
                        tgt_bos=self.joint_dict.bos(),
                        new_tgt_bos=_lang_token_index(self.joint_dict, direction.split('-')[1])
                    )
                datasets[f"{modality}_{direction}"] = dataset
            self.datasets[split] = MultiModalRoundRobinZipDatasets(datasets)
        else:
            assert len(subsets) == 1
            modality, dataset = subsets[0]
            if self.cfg.decoder_langtok:
                dataset = TransformTargetEosDataset(
                    dataset,
                    tgt_bos=self.joint_dict.bos(),
                    new_tgt_bos=_lang_token_index(self.joint_dict, self.cfg.eval_direction.split('-')[1])
                )
            self.datasets[split] = dataset
