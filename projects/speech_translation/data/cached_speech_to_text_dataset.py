import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset, SpeechToTextDatasetItem, parse_path, get_waveform, \
    get_features_from_npy_or_audio, get_features_or_waveform_from_stored_zip, SpeechToTextDatasetCreator
import torch
from subprocess import Popen


logger = logging.getLogger(__name__)

TIMEOUT_DELTA = timedelta(minutes=30)


@dataclass
class CacheItem:
    src_path: str
    src_size: int
    time_started: datetime
    path: Optional[str] = None
    process: Optional[Popen] = None

    def __init__(self, src_path: str):
        self.src_path = src_path
        assert os.path.exists(self.src_path), f"{self.src_path} does not exist!"
        self.src_size = os.stat(self.src_path).st_size

        self.__init_process()

    def __init_process(self):
        self.process = Popen(['cf', self.src_path], stdout=subprocess.PIPE)
        self.time_started = datetime.utcnow()

    @property
    def is_valid(self) -> bool:
        return self.path is not None and os.path.exists(self.path) and os.stat(self.path).st_size == self.src_size

    def invalidate(self):
        self.path = None
        self.__init_process()

    def get_path(self) -> str:
        if not self.is_valid:
            if not self.process:
                self.__init_process()
            else:
                if (datetime.utcnow() - self.time_started) >= TIMEOUT_DELTA:
                    logger.warning("CacheManager process timed out, restarting...")
                    self.process.kill()
                    self.__init_process()

            status = self.process.poll()
            if status is None:
                return self.src_path
            if status == 0:
                self.path = self.process.communicate()[0].strip().decode("utf8")
                self.process = None

                if not os.path.isfile(self.path):
                    logging.warning(f'CacheManager returned invalid path {self.path} for {self.src_path}.')
                    self.path = None
                    self.__init_process()
                    return self.src_path
            else:
                self.__init_process()
                return self.src_path

        return self.path


class CacheManager:
    cache: dict[str, CacheItem]

    def __init__(self):
        self.cache = {}

    def query(self, q: str) -> str:
        if q not in self.cache:
            self.cache[q] = CacheItem(q)

        return self.cache[q].get_path()

    def invalidate(self, q: str):
        if q in self.cache:
            self.cache[q].invalidate()


def cached_get_features_or_waveform(
        path: str, cache_manager: CacheManager, need_waveform=False, use_sample_rate=None
):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        cache_manager: The cache manager
        need_waveform (bool): return waveform instead of features.
        use_sample_rate (int): change sample rate for the input wave file

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, slice_ptr = parse_path(path)
    _orig_path = _path
    _path = cache_manager.query(_path)

    # TODO: clean up code...
    if len(slice_ptr) == 0:
        try:
            if need_waveform:
                return get_waveform(
                    _path, always_2d=False, output_sample_rate=use_sample_rate
                )[0]
            return get_features_from_npy_or_audio(_path)
        except Exception as e:
            logging.error(f'Failed to load features from {_path}!')
            cache_manager.invalidate(_orig_path)

            if need_waveform:
                return get_waveform(
                    _orig_path, always_2d=False, output_sample_rate=use_sample_rate
                )[0]
            return get_features_from_npy_or_audio(_orig_path)
    elif len(slice_ptr) == 2:
        try:
            assert _path.endswith('.zip'), f"{_path} is not a zip."
            features_or_waveform = get_features_or_waveform_from_stored_zip(
                _path, slice_ptr[0], slice_ptr[1], need_waveform=need_waveform,
                use_sample_rate=use_sample_rate
            )
        except Exception as e:
            logging.error(f'Failed to load features from {_path} ({slice_ptr}!')
            cache_manager.invalidate(_orig_path)
            features_or_waveform = get_features_or_waveform_from_stored_zip(
                _orig_path, slice_ptr[0], slice_ptr[1], need_waveform=need_waveform,
                use_sample_rate=use_sample_rate
            )
    else:
        raise ValueError(f"Invalid path: {path}")

    return features_or_waveform


class CachedSpeechToTextDataset(SpeechToTextDataset):

    def __init__(self, split: str, is_train_split: bool, cfg: S2TDataConfig, audio_paths: List[str], n_frames: List[int],
                 src_texts: Optional[List[str]] = None, tgt_texts: Optional[List[str]] = None, speakers: Optional[List[str]] = None,
                 src_langs: Optional[List[str]] = None, tgt_langs: Optional[List[str]] = None, ids: Optional[List[str]] = None,
                 tgt_dict: Optional[Dictionary] = None, pre_tokenizer=None, bpe_tokenizer=None, n_frames_per_step=1, speaker_to_id=None):
        super().__init__(split, is_train_split, cfg, audio_paths, n_frames, src_texts, tgt_texts, speakers, src_langs, tgt_langs, ids,
                         tgt_dict, pre_tokenizer, bpe_tokenizer, n_frames_per_step, speaker_to_id)

        self.cache_manager = CacheManager()

    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
        source = cached_get_features_or_waveform(
            self.audio_paths[index],
            need_waveform=self.cfg.use_audio_input,
            use_sample_rate=self.cfg.use_sample_rate,
            cache_manager=self.cache_manager
        )
        if self.feature_transforms is not None:
            assert not self.cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()
        source = self.pack_frames(source)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        return SpeechToTextDatasetItem(
            index=index, source=source, target=target, speaker_id=speaker_id
        )


class CachedSpeechToTextDatasetCreator(SpeechToTextDatasetCreator):
    @classmethod
    def _from_list(
            cls,
            split_name: str,
            is_train_split,
            samples: List[Dict],
            cfg: S2TDataConfig,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id
    ) -> SpeechToTextDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        return CachedSpeechToTextDataset(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id
        )
