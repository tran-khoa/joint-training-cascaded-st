# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.utils.data.dataloader import default_collate

from fairseq.data.fairseq_dataset import FairseqDataset


class RepeatDataset(FairseqDataset):

    def __init__(self, dataset, repeats):
        super(RepeatDataset, self).__init__()
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return self.repeats * len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx // self.repeats]

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        return self.dataset.size(idx // self.repeats)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        return getattr(self.dataset, attr, None)

    @property
    def sizes(self):
        if isinstance(self.dataset.sizes, np.ndarray):
            return self.dataset.sizes.repeat(self.repeats)
        else:
            assert isinstance(self.dataset.sizes, list)
            _dataset_sizes = []
            for s in self.dataset.sizes:
                _dataset_sizes.extend([s] * self.repeats)
            return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        if isinstance(self.sizes, np.ndarray) and len(self.sizes.shape) > 1:
            # special handling for concatenating lang_pair_datasets
            indices = np.arange(len(self))
            sizes = self.sizes
            tgt_sizes = (
                sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
            )
            src_sizes = (
                sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes
            )
            # sort by target length, then source length
            if tgt_sizes is not None:
                indices = indices[np.argsort(tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(src_sizes[indices], kind="mergesort")]
        else:
            return np.argsort(self.sizes)

    def prefetch(self, indices):
        if getattr(self.dataset, "supports_prefetch", False):
            self.dataset.prefetch(indices)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return self.dataset.can_reuse_epoch_itr_across_epochs

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
