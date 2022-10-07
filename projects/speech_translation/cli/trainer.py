import contextlib
import logging
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.distributed import utils as distributed_utils, fsdp_wrap, fsdp_enable_wrap
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.nan_detector import NanDetector
from fairseq.trainer import Trainer, _catalog_shared_params, _get_module_by_path, _set_module_by_path

logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def __init__(self, cfg: FairseqConfig, task, model, criterion, quantizer=None):
        super().__init__(cfg, task, model, criterion, quantizer)

        self.do_subepoching = hasattr(cfg.checkpoint, "epoch_split") and cfg.checkpoint.epoch_split > 0
        self.subepoch = 1
        with Path(__file__).with_name('checkpoint_version').open('rt') as f:
            self.latest_checkpoint_version = int(f.read().strip())
        self.checkpoint_version = 0

        if hasattr(self.task, "move_model_to_devices"):
            self._model, self._criterion = self.task.move_model_to_devices(self._model, self._criterion)

    def set_model_and_criterion(self, model, criterion):
        shared_params = _catalog_shared_params(model)
        self._wrapped_model = None
        self._optimizer = None  # TODO: copy estimates?
        self._model = model
        self._criterion = criterion

        if not self.is_fsdp:
            if self.cfg.common.fp16:
                assert not self.cfg.common.amp, "Cannot use fp16 and AMP together"
                self._criterion = self._criterion.half()
                self._model = self._model.half()
            elif self.cfg.common.bf16:
                self._criterion = self._criterion.to(dtype=torch.bfloat16)
                self._model = self._model.to(dtype=torch.bfloat16)
            elif self.cfg.common.amp:
                self._amp_retries = 0
        if (
                not self.cfg.distributed_training.pipeline_model_parallel
                # the DistributedFairseqModel wrapper will handle moving to device,
                # so only handle cases which don't use the wrapper
                and not self.use_distributed_wrapper
                and not hasattr(self.task, "move_model_to_devices")
        ):
            self._criterion = self._criterion.to(device=self.device)
            self._model = self._model.to(device=self.device)
        if hasattr(self.task, "move_model_to_devices"):
            self._model, self._criterion = self.task.move_model_to_devices(self._model, self._criterion)

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    "detected shared parameter: {} <- {}".format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)

    @property
    def do_pretraining(self):
        return getattr(self.task, "do_pretraining", False)

    def step_subepoch(self):
        self.subepoch += 1

    def state_dict(self):
        state_dict = super().state_dict()
        if 'extra_state' not in state_dict:
            state_dict['extra_state'] = {}
        state_dict['extra_state']['subepoch'] = self.subepoch
        state_dict['extra_state']['checkpoint_version'] = self.latest_checkpoint_version
        return state_dict

    def run_checkpoint_migrations(self, extra_state):
        if self.checkpoint_version < 1:
            logger.info("Running checkpoint migrations for checkpoint version < 1")
            # Checkpoint version 0: there was an error where the subepoch was stepped after saving.
            if self.do_subepoching:
                self.step_subepoch()
                if 'val_loss' in extra_state:
                    self.lr_step_subepoch(valid_losses=[extra_state['val_loss']])
                else:
                    logger.warning('No validation loss found, could not step subepoch.')

    def build_model(self, overrides: dict[str, Any] = None):
        if overrides is None:
            overrides = {}
        model_cfg = Namespace(**(vars(self.cfg.model) | overrides))

        if self.cfg.distributed_training.ddp_backend == "fully_sharded":
            with fsdp_enable_wrap(self.cfg.distributed_training):
                model = fsdp_wrap(self.task.build_model(model_cfg))
        else:
            model = self.task.build_model(model_cfg)

        return model

    def build_criterion(self, overrides: dict[str, Any] = None):
        criterion_cfg = self.cfg.criterion.copy()
        if overrides is not None:
            criterion_cfg.update(overrides)
        return self.task.build_criterion(criterion_cfg)

    def load_checkpoint(
            self,
            filename,
            reset_optimizer=False,
            reset_lr_scheduler=False,
            optimizer_overrides=None,
            reset_meters=False,
            reset_subepochs=False
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        bexists = PathManager.isfile(filename)
        if bexists:
            load_on_all_ranks = (
                    self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                    # TPUs don't support broadcast yet, so load checkpoints
                    # on every worker for now
                    or self.tpu
                    # FSDP requires loading checkpoint shards on all ranks
                    or (self.is_fsdp and self.cfg.distributed_training.use_sharded_state)
                    or getattr(self.cfg.model, "base_layers", 0) > 0
            )

            if load_on_all_ranks or self.data_parallel_rank == 0:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    filename, load_on_all_ranks=load_on_all_ranks
                )
                last_optim_state = state.get("last_optimizer_state", None)

                # If doing zero_sharding, do not broadcast global optimizer
                # state. Later we will broadcast sharded states to each rank
                # to avoid memory from exploding.
                if (
                        not load_on_all_ranks
                        and self.cfg.distributed_training.zero_sharding == "os"
                        and "last_optimizer_state" in state
                        and is_distributed
                ):
                    state["last_optimizer_state"] = "SHARDED"
            else:
                last_optim_state = None
                state = None

            if is_distributed and not load_on_all_ranks:
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
                if self.data_parallel_rank > 0:
                    last_optim_state = state.get("last_optimizer_state", None)

            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                    last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), f"Criterion does not match; please reset the optimizer (--reset-optimizer). {last_optim['criterion_name']} vs {self.get_criterion().__class__.__name__}"
            assert (
                    last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), f"Optimizer does not match; please reset the optimizer (--reset-optimizer). {last_optim['optimizer_name']} vs {self.optimizer.__class__.__name__}"

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            if self.is_fsdp and not self.model.use_sharded_state:
                # if use_sharded_state, the last_optim_state is already sharded, skip this
                last_optim_state = self.model.get_shard_from_optim_state_dict(
                    last_optim_state
                )
            elif not load_on_all_ranks and is_distributed:
                last_optim_state = self.optimizer.broadcast_global_state_dict(
                    last_optim_state
                )

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            if self.do_subepoching:
                assert "subepoch" in extra_state, "Checkpoint did not store subepoch."
                if not reset_subepochs:
                    self.subepoch = extra_state['subepoch']
                else:
                    logger.info('Not restoring subepoch since we are in fine-tuning mode. Will fail with models that have not completed pretrain scheme.')

                logger.info(f"Checkpoint is at subepoch {self.subepoch}.")

                self.lr_step_subepoch()
            if 'checkpoint_version' in extra_state:
                self.checkpoint_version = extra_state['checkpoint_version']
                if self.checkpoint_version < self.latest_checkpoint_version:
                    logger.info(f'Loaded checkpoint has checkpoint version {self.checkpoint_version}, but latest version is {self.latest_checkpoint_version}.')

            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step(epoch)

            if (
                    itr_state.get("version", 1) >= 2
                    and itr_state["iterations_in_epoch"] == 0
            ):
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            if self.cfg.ema.store_ema:
                if "ema" not in extra_state:
                    logger.warn(
                        "EMA not found in checkpoint. But store_ema is True. "
                        "EMA is re-initialized from checkpoint."
                    )
                    self.ema.restore(state["model"], build_fp32_params=self.cfg.ema.ema_fp32)
                else:
                    logger.info(
                        "Loading EMA from checkpoint"
                    )
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)

                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info(
                                "Loading EMA fp32 params from checkpoint"
                            )
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info(
                                "Building EMA fp32 params from EMA model in checkpoint"
                            )
                            self.ema.build_fp32_params()

            self.run_checkpoint_migrations(extra_state)

            if self.do_pretraining:
                last_pretrain_args = state['task_state'].get('last_pretrain_args', None)
                if last_pretrain_args is not None:
                    checkpoint_model = self.build_model(last_pretrain_args[0])
                    checkpoint_criterion = self.build_criterion(last_pretrain_args[1])
                    self.set_model_and_criterion(checkpoint_model, checkpoint_criterion)

            # load model parameters after loading extra states!
            try:
                self.model.load_state_dict(
                    state["model"], strict=True, model_cfg=self.cfg.model
                )

                # save memory for later steps
                del state["model"]
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
                    del state["criterion"]

            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )

            # Only now we can load the optimizer parameters
            if last_optim_state is not None and not reset_optimizer:
                self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            if self.do_pretraining:
                self.pretrain_step(copy_param=True)
                model_overrides, criterion_overrides = self.task.handle_pretraining(self.cfg.model, self.cfg.criterion, update_num=self.get_num_updates(), subepoch=self.subepoch)
                if len(model_overrides) + len(criterion_overrides) > 0:
                    logger.info(f'Pre-training: Creating new model with {model_overrides} and new criterion with {criterion_overrides}.')
                    new_model = self.build_model(model_overrides)
                    new_criterion = self.build_criterion(criterion_overrides)
                    self.set_model_and_criterion(new_model, new_criterion)

            logger.info(
                "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

        else:
            logger.info("No existing checkpoint found {}".format(filename))
            if self.do_pretraining:
                self.pretrain_step(copy_param=False)

        return extra_state

    def lr_step_subepoch(self, valid_losses=None):
        """Update the learning rate after each subepoch"""
        if not self.do_subepoching:
            return self.get_lr()

        if valid_losses is None:
            valid_losses = [None]

        if hasattr(self.lr_scheduler, 'step_subepoch'):
            new_lr = self.lr_scheduler.step_subepoch(self.subepoch, valid_losses[0])
            if isinstance(new_lr, dict):
                for k, v in new_lr.items():
                    metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
                new_lr = new_lr.get("default", next(iter(new_lr.values())))
            else:
                metrics.log_scalar("lr", new_lr, weight=0, priority=300)
            return new_lr
        else:
            return self.get_lr()

    def pretrain_step(self, copy_param: bool = False):
        model_overrides, criterion_overrides = self.task.handle_pretraining(self.cfg.model, self.cfg.criterion,
                                                                            update_num=self.get_num_updates(), subepoch=self.subepoch)
        if len(model_overrides) + len(criterion_overrides) > 0:
            logger.info(f'Pre-training: Creating new model with {model_overrides} and new criterion with {criterion_overrides}.')
            new_model = self.build_model(model_overrides)
            new_criterion = self.build_criterion(criterion_overrides)
            old_model = self.model
            self.set_model_and_criterion(new_model, new_criterion)
            if copy_param:
                self.copy_params(old_model, new_model)

            # As the model changes, we also have to update the optimizer
            # which means we have to update the lr scheduler...
            # We reinitialize the optimizer states, which is not optimal
            # but consistent with RETURNN
            self._build_optimizer()

    @staticmethod
    def copy_params(old_model, new_model):
        new_model_params = dict(new_model.named_parameters())
        for name, old_param in old_model.named_parameters():
            if name not in new_model_params:
                continue
            pads = list(max(0, new_dim - old_dim) for old_dim, new_dim in zip(old_param.size(), new_model_params[name].size()))
            pads = tuple(i for tup in map(lambda x: (0, x), reversed(pads)) for i in tup)
            mask = torch.ones_like(old_param, dtype=torch.bool)
            mask = F.pad(mask, pads, value=0)
            new_model_params[name].data[mask] = old_param.detach().view(-1)

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""

        # Update model and criterion first
        if self.do_pretraining:
            self.pretrain_step(copy_param=True)

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()
        if self.do_subepoching:
            extra_kwargs["subepoch"] = self.subepoch

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):  # delayed update loop
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                    # The no_sync context manager results in increased memory
                    # usage with FSDP, since full-size gradients will be
                    # accumulated on each GPU. It's typically a better tradeoff
                    # to do the extra communication with FSDP.
                    and not self.is_fsdp
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                        **extra_kwargs,
                    )
                    del loss

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                    if self.cfg.distributed_training.distributed_world_size == 1:
                        return None
                else:
                    raise e

            if self.tpu and i < len(samples) - 1:
                # tpu-comment: every XLA operation before marking step is
                # appended to the IR graph, and processing too many batches
                # before marking step can lead to OOM errors.
                # To handle gradient accumulation use case, we explicitly
                # mark step here for every forward pass without a backward pass
                self._xla_markstep_and_send_to_cpu()

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self._sync_stats():
            train_time = self._local_cumulative_training_time()
            logging_outputs, (
                sample_size,
                ooms,
                total_train_time,
            ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms, train_time, ignore=is_dummy_batch
            )
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        overflow = False
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                # reduce gradients across workers
                self.optimizer.all_reduce_grads(self.model)
                if utils.has_parameters(self.criterion):
                    self.optimizer.all_reduce_grads(self.criterion)

            with torch.autograd.profiler.record_function("multiply-grads"):
                # multiply gradients by (data_parallel_size / sample_size) since
                # DDP normalizes by the number of data parallel workers for
                # improved fp16 precision.
                # Thus we get (sum_of_gradients / sample_size) at the end.
                # In case of fp16, this step also undoes loss scaling.
                # (Debugging note: Some optimizers perform this scaling on the
                # fly, so inspecting model.parameters() or optimizer.params may
                # still show the original, unscaled gradients.)
                numer = (
                    self.data_parallel_world_size
                    if not self.cfg.optimization.use_bmuf or self._sync_stats()
                    else 1
                )
                self.optimizer.multiply_grads(numer / (sample_size or 1.0))
                # Note: (sample_size or 1.0) handles the case of a zero gradient, in a
                # way that avoids CPU/device transfers in case sample_size is a GPU or
                # TPU object. The assumption is that the gradient itself is also 0.

            with torch.autograd.profiler.record_function("clip-grads"):
                # clip grads
                grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)

            # check that grad norms are consistent across workers
            # on tpu check tensor is slow
            if not self.tpu:
                if (
                    not self.cfg.optimization.use_bmuf
                    and self.cfg.distributed_training.ddp_backend != "slowmo"
                ):
                    self._check_grad_norms(grad_norm)
                if not torch.isfinite(grad_norm).all():
                    # in case of AMP, if gradients are Nan/Inf then
                    # optimizer step is still required
                    if self.cfg.common.amp:
                        overflow = True
                    else:
                        # check local gradnorm single GPU case, trigger NanDetector
                        raise FloatingPointError("gradients are Nan/Inf")

            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.task.optimizer_step(
                    self.optimizer, model=self.model, update_num=self.get_num_updates()
                )
                if self.cfg.common.amp and overflow:
                    if self._amp_retries == self.cfg.common.amp_batch_retries:
                        logger.info("AMP: skipping this batch.")
                        self._amp_retries = 0
                    else:
                        self._amp_retries += 1
                        return self.train_step(samples, raise_oom)  # recursion to feed in same batch

        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            self.zero_grad()
            with NanDetector(self.get_model()):
                for _, sample in enumerate(samples):
                    sample, _ = self._prepare_sample(sample)
                    self.task.train_step(
                        sample,
                        self.model,
                        self.criterion,
                        self.optimizer,
                        self.get_num_updates(),
                        ignore_grad=False,
                        **extra_kwargs,
                    )
            raise
        except OverflowError as e:
            overflow = True
            logger.info(
                f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}"
            )
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        # Some distributed wrappers (e.g., SlowMo) need access to the optimizer
        # after the step
        if hasattr(self.model, "perform_slowmo"):
            self.model.perform_slowmo(
                self.optimizer.optimizer, getattr(self.optimizer, "fp32_params", None)
            )

        logging_output = None
        if not overflow or self.cfg.distributed_training.ddp_backend == "slowmo":
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cfg.ema.store_ema:
                # Step EMA forward with new model.
                self.ema.step(
                    self.get_model(),
                    self.get_num_updates(),
                )
                metrics.log_scalar(
                    "ema_decay",
                    self.ema.get_decay(),
                    priority=10000,
                    round=5,
                    weight=0,
                )

            if self.tpu:
                import torch_xla.core.xla_model as xm

                # mark step on TPUs
                self._xla_markstep_and_send_to_cpu()

                # only log stats every log_interval steps
                # this causes wps to be misreported when log_interval > 1
                logging_output = {}
                if self.get_num_updates() % self.cfg.common.log_interval == 0:
                    # log memory usage
                    mem_info = xm.get_memory_info(self.device)
                    gb_free = mem_info["kb_free"] / 1024 / 1024
                    gb_total = mem_info["kb_total"] / 1024 / 1024
                    metrics.log_scalar(
                        "gb_free", gb_free, priority=1500, round=1, weight=0
                    )
                    metrics.log_scalar(
                        "gb_total", gb_total, priority=1600, round=1, weight=0
                    )
                    logging_outputs = self._xla_markstep_and_send_to_cpu(
                        logging_outputs
                    )
                    logging_output = self._reduce_and_log_stats(
                        logging_outputs, sample_size, grad_norm
                    )

                # log whenever there's an XLA compilation, since these
                # slow down training and may indicate opportunities for
                # optimization
                self._check_xla_compilation()
            else:
                if self.cuda and self.cuda_env is not None:
                    # log minimum free memory over the iteration
                    gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                    torch.cuda.reset_peak_memory_stats()
                    gb_free = self.cuda_env.total_memory_in_GB - gb_used
                    metrics.log_scalar(
                        "gb_free", gb_free, priority=1500, round=1, weight=0
                    )

                # log stats
                logging_output = self._reduce_and_log_stats(
                    logging_outputs, sample_size, grad_norm
                )

                # clear CUDA cache to reduce memory fragmentation
                if (
                    self.cuda
                    and self.cfg.common.empty_cache_freq > 0
                    and (
                        (self.get_num_updates() + self.cfg.common.empty_cache_freq - 1)
                        % self.cfg.common.empty_cache_freq
                    )
                    == 0
                ):
                    torch.cuda.empty_cache()

        if self.cfg.common.fp16 or self.cfg.common.amp:
            metrics.log_scalar(
                "loss_scale",
                (
                    self.optimizer.scaler.loss_scale
                    if self.cfg.common.fp16
                    else self.optimizer.scaler.get_scale()
                ),
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    def _build_optimizer(self):
        super()._build_optimizer()
        self.lr_step_subepoch()
