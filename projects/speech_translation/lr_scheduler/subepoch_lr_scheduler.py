from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from fairseq.dataclass import ChoiceEnum, FairseqDataclass

STEP_MODE_CHOICE = ChoiceEnum(['epoch', 'subepoch'])


@dataclass
class SubepochLRSchedulerConfig(FairseqDataclass):
    step_mode: STEP_MODE_CHOICE = field(
        default="subepoch",
        metadata={
            "help": "lr scheduler step will be executed after each {subepoch,epoch}"
        },
    )


class SubepochLRScheduler(ABC):

    def __init__(self, cfg, *args, **kwargs):
        assert hasattr(cfg, 'step_mode')
        self.step_mode = cfg.step_mode

    @abstractmethod
    def step_subepoch(self, subepoch: int, val_loss: float) -> float:
        raise NotImplementedError()
