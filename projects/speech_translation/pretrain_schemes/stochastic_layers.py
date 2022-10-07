from argparse import Namespace
from typing import Optional, Any

from speech_translation.pretrain_schemes import register_pretrain_scheme


@register_pretrain_scheme('stochastic_layers')
def stochastic_layers(model_cfg: Namespace, criterion_cfg: Namespace, task, num_updates: int, subepoch: Optional[int] = None) -> Optional[
    tuple[dict[str, Any],
          dict[str, Any]]]:

    # requires subepochs
    assert subepoch is not None

    pretrain_subepochs = 10

    if subepoch > pretrain_subepochs:
        return None

    new_model_args = {
        'stochastic_p': 0.5
    }

    return new_model_args, {}
