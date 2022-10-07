# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class KITLRScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=4000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is cfg.lr"
        },
    )
    lr: List[float] = II("optimization.lr")
    lr_dim: int = field(
        default=512,
        metadata={
            "help": "specify model dimension used in architecture"
        },
    )


@register_lr_scheduler("kit", dataclass=KITLRScheduleConfig)
class KITSchedule(FairseqLRScheduler):

    def __init__(self, cfg: KITLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with inverse_sqrt."
                " Consider --lr-scheduler=fixed instead."
            )
        warmup_end_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
        if cfg.warmup_init_lr < 0:
            cfg.warmup_inift_lr = 0 if cfg.warmup_updates > 0 else warmup_end_lr

        # initial learning rate
        self.lr = cfg.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        self.lr = self.cfg.warmup_init_lr * (self.cfg.lr_dim ** -0.5) * min(num_updates ** -0.5 if num_updates > 0 else math.inf,
                                                                            num_updates * self.cfg.warmup_updates ** -1.5)
        self.optimizer.set_lr(self.lr)
        return self.lr
