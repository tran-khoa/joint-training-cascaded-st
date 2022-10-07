from dataclasses import dataclass
from typing import Optional, List, Dict

import torch

from fairseq.data import (BaseWrapperDataset,
                          data_utils as fairseq_data_utils, Dictionary)
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetItem, SpeechToTextDataset, _collate_frames

L2R_TAG = "<l2r>"
R2L_TAG = "<r2l>"


@dataclass
class R2LSpeechToTextDatasetItem(SpeechToTextDatasetItem):
    l2r_target: Optional[torch.Tensor] = None

    @classmethod
    def from_item(cls, item: SpeechToTextDatasetItem):
        return cls(
            index=item.index,
            source=item.source,
            l2r_target=item.target,
            speaker_id=item.speaker_id
        )


class R2LSpeechToTextDataset(BaseWrapperDataset):
    dataset: SpeechToTextDataset

    def __init__(self, dataset, dictionary: Dictionary, bidirectional: bool = False):
        super().__init__(dataset)
        assert isinstance(self.dataset, SpeechToTextDataset)

        self.bidirectional = bidirectional
        self.dictionary = dictionary

    def __getitem__(self, index):
        item = self.dataset[index]

        assert isinstance(item, SpeechToTextDatasetItem)

        if item.target is None:
            return item

        item = R2LSpeechToTextDatasetItem.from_item(item)
        item.target = item.l2r_target.clone()

        if self.dataset.cfg.prepend_tgt_lang_tag:
            item.target[1:-1] = item.l2r_target[1:-1][::-1]
        else:
            item.target[:-1] = item.l2r_target[:-1].flip(0)

        if self.bidirectional:
            item.target[-1] = self.r2l_idx
            item.l2r_target[-1] = self.l2r_idx
        return item

    @property
    def l2r_idx(self) -> int:
        idx = self.dictionary.index(L2R_TAG)
        assert idx != self.dictionary.unk()
        return idx

    @property
    def r2l_idx(self) -> int:
        idx = self.dictionary.index(R2L_TAG)
        assert idx != self.dictionary.unk()
        return idx

    def collater(
            self, samples: List[R2LSpeechToTextDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.source for x in samples], self.dataset.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, l2r_target, target_lengths = None, None, None
        prev_output_tokens, l2r_prev_output_tokens = None, None
        ntokens = None
        if self.dataset.tgt_texts is not None:
            l2r_eos_index = self.l2r_idx if self.bidirectional else self.dictionary.eos()
            r2l_eos_index = self.r2l_idx if self.bidirectional else self.dictionary.eos()

            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.dataset.tgt_dict.pad(),
                r2l_eos_index,
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)

            l2r_target = fairseq_data_utils.collate_tokens(
                [x.l2r_target for x in samples],
                self.dataset.tgt_dict.pad(),
                l2r_eos_index,
                left_pad=False,
                move_eos_to_beginning=False,
            )
            l2r_target = l2r_target.index_select(0, order)

            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)

            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.dataset.tgt_dict.pad(),
                r2l_eos_index,
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            if self.bidirectional:
                l2r_prev_output_tokens = fairseq_data_utils.collate_tokens(
                    [x.l2r_target for x in samples],
                    self.dataset.tgt_dict.pad(),
                    l2r_eos_index,
                    left_pad=False,
                    move_eos_to_beginning=True,
                )

            ntokens = sum(x.target.size(0) for x in samples)

        speaker = None
        if self.dataset.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                    .index_select(0, order)
                    .view(-1, 1)
            )

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
        }

        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": speaker,
            "target": target,
            "l2r_target": l2r_target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "prepend_direction": self.bidirectional
        }
        if self.bidirectional:
            l2r_net_input = {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": l2r_prev_output_tokens,
            }
            out["r2l_net_input"] = net_input
            out["l2r_net_input"] = l2r_net_input

        if return_order:
            out["order"] = order
        return out
