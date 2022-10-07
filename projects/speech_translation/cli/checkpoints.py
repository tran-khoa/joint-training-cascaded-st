# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from collections import OrderedDict
import logging
import os
from typing import Union, Optional, Collection

import numpy as np

from fairseq.checkpoint_utils import checkpoint_paths, load_checkpoint_to_cpu
from fairseq.data import data_utils
from fairseq.dataclass.configs import CheckpointConfig
from fairseq.file_io import PathManager
from fairseq.models import FairseqEncoder, FairseqDecoder

logger = logging.getLogger(__name__)


def save_checkpoint(cfg: CheckpointConfig, trainer, epoch_itr, val_loss, save_metadata=False, subepoch: int = -1):
    from fairseq import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return

    trainer.consolidate_optimizer()  # TODO(SS): do we need this if no_save_optimizer_state

    if not trainer.should_save_checkpoint_on_current_rank:
        if trainer.always_call_state_dict_during_save_checkpoint:
            trainer.state_dict()
        return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint_{}.{}{}.pt".format(epoch, subepoch, suffix)] = (
            subepoch > 0 and not cfg.no_epoch_checkpoints
    )
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
            end_of_epoch and not cfg.no_epoch_checkpoints and cfg.epoch_split == -1 and epoch % cfg.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
            not end_of_epoch
            and cfg.save_interval_updates > 0
            and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
            not hasattr(save_checkpoint, "best")
            or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and cfg.keep_best_checkpoints > 0:
        worst_best = getattr(save_checkpoint, "best", None)
        chkpts = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if len(chkpts) > 0:
            p = chkpts[-1] if cfg.maximize_best_checkpoint_metric else chkpts[0]
            worst_best = float(p.rsplit("_")[-1].replace("{}.pt".format(suffix), ""))
        # add random digits to resolve ties
        with data_utils.numpy_seed(epoch, updates, val_loss):
            rand_sfx = np.random.randint(0, cfg.keep_best_checkpoints)

        checkpoint_conds[
            "checkpoint.best_{}_{:.3f}{}{}.pt".format(
                cfg.best_checkpoint_metric,
                val_loss,
                rand_sfx,
                suffix
            )
        ] = worst_best is None or is_better(val_loss, worst_best)
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not cfg.no_last_checkpoints

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            if cfg.write_checkpoints_asynchronously:
                # TODO[ioPath]: Need to implement a delayed asynchronous
                # file copying/moving feature.
                logger.warning(
                    f"ioPath is not copying {checkpoints[0]} to {cp} "
                    "since async write mode is on."
                )
            else:
                assert PathManager.copy(
                    checkpoints[0], cp, overwrite=True
                ), f"Failed to copy {checkpoints[0]} to {cp}"

        write_timer.stop()
        logger.info(
            "Saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    if not end_of_epoch and cfg.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        if cfg.keep_interval_updates_pattern == -1:
            checkpoints = checkpoint_paths(
                cfg.save_dir, pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix)
            )
        else:
            checkpoints = checkpoint_paths(
                cfg.save_dir,
                pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix),
                keep_match=True,
            )
            checkpoints = [
                x[0]
                for x in checkpoints
                if x[1] % cfg.keep_interval_updates_pattern != 0
            ]

        for old_chk in checkpoints[cfg.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_last_subepochs > 0:
        checkpoints = checkpoint_paths(
            cfg.save_dir, pattern=r"checkpoint_\d+\.(\d+){}\.pt".format(suffix)
        )
        for old_chk in checkpoints[cfg.keep_last_subepochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            cfg.save_dir, pattern=r"checkpoint(\d+){}\.pt".format(suffix)
        )
        for old_chk in checkpoints[cfg.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if not cfg.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[cfg.keep_best_checkpoints:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder], checkpoint: str, reset_ctc: bool = False,
        ignore_keys: Optional[Collection[str]] = None
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            if reset_ctc and component_type == "encoder" and key.startswith("encoder.ctc_proj"):
                logger.info(f"Since --reset-ctc is set, will not load {key}.")
                continue

            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]

    missing_keys, unexpected_keys = component.load_state_dict(component_state_dict, strict=False)

    missing_keys = set(missing_keys)
    unexpected_keys = set(unexpected_keys)
    if missing_keys or unexpected_keys:
        logger.info(f"The following keys are missing or unexpected, which is acceptable under certain conditions:\n"
                    f"Missing:\n\t {', '.join(missing_keys)}\n\n "
                    f"Unexpected:\n\t {', '.join(unexpected_keys)}")
    if reset_ctc:
        missing_keys.discard("ctc_proj.weight")
        missing_keys.discard("ctc_proj.bias")
        unexpected_keys.discard("ctc_proj.weight")
        unexpected_keys.discard("ctc_proj.bias")
    if ignore_keys:
        for key in ignore_keys:
            missing_keys.discard(key)
            unexpected_keys.discard(key)

    if missing_keys or unexpected_keys:
        raise ValueError(f"The following keys are missing:\n\t {', '.join(missing_keys)}\n\n "
                         f"The following keys are unexpected:\n\t {', '.join(unexpected_keys)}")

    return component
