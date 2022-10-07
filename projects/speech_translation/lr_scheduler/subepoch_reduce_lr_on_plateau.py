from dataclasses import dataclass, field

from fairseq.optim.lr_scheduler import register_lr_scheduler, FairseqLRScheduler
from fairseq.optim.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauLRSchedule, ReduceLROnPlateauLRScheduleConfig
import torch

from speech_translation.lr_scheduler.subepoch_lr_scheduler import SubepochLRScheduler, SubepochLRSchedulerConfig


@dataclass
class SubepochReduceLROnPlateauLRScheduleConfig(ReduceLROnPlateauLRScheduleConfig, SubepochLRSchedulerConfig):
    warmup_subepochs: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N subepochs"},
    )
    min_lr: float = field(
        default=0
    )


@register_lr_scheduler(
    "subepoch_reduce_lr_on_plateau", dataclass=SubepochReduceLROnPlateauLRScheduleConfig
)
class SubepochReduceLROnPlateau(SubepochLRScheduler, ReduceLROnPlateauLRSchedule):
    def __init__(self, cfg: SubepochReduceLROnPlateauLRScheduleConfig, optimizer):
        SubepochLRScheduler.__init__(self, cfg)
        FairseqLRScheduler.__init__(self, cfg, optimizer)

        if len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with reduce_lr_on_plateau."
                " Consider --lr-scheduler=fixed instead."
            )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer,
            patience=cfg.lr_patience,
            factor=cfg.lr_shrink,
            mode="max" if cfg.maximize_best_checkpoint_metric else "min",
            threshold=cfg.lr_threshold,
            min_lr=cfg.min_lr
        )
        warmup_end_lr = cfg.lr[0]
        # if no warm up, sets initial lr to be cfg.lr[0]
        if cfg.warmup_init_lr < 0:
            cfg.warmup_init_lr = 0 if cfg.warmup_updates > 0 or cfg.warmup_subepochs > 0 else warmup_end_lr

        # linearly warmup for the first cfg.warmup_updates
        assert not (cfg.warmup_updates > 0 and cfg.warmup_subepochs > 0)

        if cfg.warmup_updates > 0:
            self.lr_step = (warmup_end_lr - cfg.warmup_init_lr) / cfg.warmup_updates
        if cfg.warmup_subepochs > 0:
            self.lr_step = (warmup_end_lr - cfg.warmup_init_lr) / cfg.warmup_subepochs

        # this flag is either set from arg when no warm up, or set by
        # step_update() when warmup finishes
        self.warmup_end = True if cfg.warmup_updates <= 0 and cfg.warmup_subepochs <= 0 else False

        # initial learning rate
        # this self.lr is used only during init and/or warm up period
        self.lr = warmup_end_lr if self.warmup_end else cfg.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        if self.step_mode == 'epoch':
            return super().step(epoch, val_loss)
        else:
            return self.optimizer.get_lr()

    def step_subepoch(self, subepoch, val_loss=None):
        assert subepoch >= 0
        if self.cfg.warmup_subepochs > 0:
            if subepoch <= self.cfg.warmup_subepochs:
                self.lr = self.cfg.warmup_init_lr + subepoch * self.lr_step
                self.optimizer.set_lr(self.lr)
            else:
                if self.warmup_end is False:
                    self.warmup_end = True

        if self.warmup_end is True and self.step_mode == 'subepoch' and val_loss is not None:
            self.lr_scheduler.step(val_loss)

        # else do nothing
        return self.optimizer.get_lr()
