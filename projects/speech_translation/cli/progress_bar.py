import logging
from numbers import Number
import time
from typing import Optional

import wandb

from fairseq.logging.meters import AverageMeter
from fairseq.logging.progress_bar import BaseProgressBar, rename_logger, TensorboardProgressBarWrapper, \
    AzureMLProgressBarWrapper

logger = logging.getLogger(__name__)


class WandBProgressBarWrapper(BaseProgressBar):
    """Log to Weights & Biases."""

    def __init__(self, wrapped_bar, wandb_project, run_name=None):
        self.wrapped_bar = wrapped_bar
        if wandb is None:
            logger.warning("wandb not found, pip install wandb")
            return

        # reinit=False to ensure if wandb.init() is called multiple times
        # within one process it still references the same run
        # also: https://docs.wandb.ai/guides/track/launch#init-start-error
        num_tries = 0
        last_exc = None
        success = False
        while num_tries < 30:
            try:
                wandb.init(id=wandb.util.generate_id(),
                           project=wandb_project,
                           reinit=False,
                           resume="allow",
                           name=run_name,
                           settings=wandb.Settings(start_method="fork")
                           )
                success = True
                break
            except Exception as e:
                num_tries += 1
                last_exc = e
                print("Retrying")
                time.sleep(10)
        if not success:
            raise last_exc

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        if wandb is not None:
            wandb.config.update(config)
        self.wrapped_bar.update_config(config)

    def _log_to_wandb(self, stats, tag=None, step=None):
        if wandb is None:
            return
        if step is None:
            step = stats["num_updates"]

        prefix = "" if tag is None else tag + "/"

        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                wandb.log({prefix + key: stats[key].val}, step=step)
            elif isinstance(stats[key], Number):
                wandb.log({prefix + key: stats[key]}, step=step)


class ForceSimpleProgressBar(BaseProgressBar):
    """A minimal logger for non-TTY environments."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.i = None
        self.size = None

    def __iter__(self):
        self.size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.n):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        stats = self._format_stats(stats)
        postfix = self._str_commas(stats)
        with rename_logger(logger, tag):
            logger.info(
                "{}:  {:5d} / {:d} {}".format(
                    self.prefix, self.i + 1, self.size, postfix
                )
            )

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        with rename_logger(logger, tag):
            logger.info("{} | {}".format(self.prefix, postfix))


def simple_progress_bar(
    iterator,
    log_interval: int = 100,
    log_file: Optional[str] = None,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    tensorboard_logdir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    azureml_logging: Optional[bool] = False,
    **kwargs
):
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file)
        logger.addHandler(handler)

    bar = ForceSimpleProgressBar(iterator, epoch, prefix, log_interval)

    if tensorboard_logdir:
        try:
            # [FB only] custom wrapper for TensorBoard
            import palaas  # noqa
            from .fb_tbmf_wrapper import FbTbmfWrapper

            bar = FbTbmfWrapper(bar, log_interval)
        except ImportError:
            bar = TensorboardProgressBarWrapper(bar, tensorboard_logdir)

    if wandb_project:
        bar = WandBProgressBarWrapper(bar, wandb_project, run_name=wandb_run_name)

    if azureml_logging:
        bar = AzureMLProgressBarWrapper(bar)

    return bar
