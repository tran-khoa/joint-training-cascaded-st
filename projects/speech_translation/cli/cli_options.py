from dataclasses import dataclass, field

from fairseq.dataclass.configs import CheckpointConfig as FairseqCheckpointConfig
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.options import get_parser, add_dataset_args, add_distributed_training_args, add_model_args, add_optimization_args, add_ema_args


def add_checkpoint_args(parser):
    group = parser.add_argument_group("checkpoint")
    # fmt: off
    gen_parser_from_dataclass(group, CustomCheckpointConfig())
    # fmt: on
    return group


def get_training_parser(default_task="translation"):
    parser = get_parser("Trainer", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    add_ema_args(parser)
    return parser


@dataclass
class CustomCheckpointConfig(FairseqCheckpointConfig):
    epoch_split: int = field(
        default=-1, metadata={"help": "simulates RETURNN's subepoch feature. Epochs are split into `epoch_split` subepochs, "
                                      "after each we validate and create a checkpoint"
                                      "requires --*-interval-updates=-1"}
    )
    keep_last_subepochs: int = field(
        default=-1, metadata={"help": "keep last subepochs"}
    )
    validate_save_subepochs: int = field(
        default=1, metadata={"help": "validate and save every N subepochs"}
    )


CHECKPOINT_ADDED_PARAMS = set(CustomCheckpointConfig.__dataclass_fields__.keys()) - set(FairseqCheckpointConfig.__dataclass_fields__.keys())
