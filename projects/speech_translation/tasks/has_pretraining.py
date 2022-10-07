from speech_translation.pretrain_schemes import PretrainScheme
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class HasPretraining:
    """
    Extended by a task that also handles layer-wise pretraining
    """
    def __init__(self, pretrain_scheme: Optional[PretrainScheme]):
        super().__init__()

        self.pretrain_scheme = pretrain_scheme
        self.last_pretrain_args = None

    @property
    def do_pretraining(self) -> bool:
        return self.pretrain_scheme is not None

    def handle_pretraining(self, model_cfg, criterion_cfg, update_num, subepoch) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.pretrain_scheme is not None:
            overrides = self.pretrain_scheme(model_cfg, criterion_cfg, self, update_num, subepoch)
            if overrides is None:
                # If the scheme returns None, we are done with pretraining
                logger.info('Pre-training completed.')
                self.last_pretrain_args = {}, {}
                self.pretrain_scheme = None
                return {}, {}
            if overrides == self.last_pretrain_args:
                return {}, {}
            else:
                self.last_pretrain_args = overrides
                return overrides

        return {}, {}

    def state_dict(self):
        return {"last_pretrain_args": self.last_pretrain_args}

    def load_state_dict(self, state_dict):
        self.last_pretrain_args = state_dict.get("last_pretrain_args", None)
