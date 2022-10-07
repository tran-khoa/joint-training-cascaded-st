import os
import importlib
from argparse import Namespace
from typing import Callable, Optional, Any


PretrainScheme = Callable[[Namespace, Namespace, object, int, Optional[int]], Optional[tuple[dict[str, Any], dict[str, Any]]]]

PRETRAIN_SCHEMES: dict[str, PretrainScheme] = {}


def register_pretrain_scheme(name):
    def _register(func):
        PRETRAIN_SCHEMES[name] = func
        return func
    return _register


# automatically import any Python files in the criterions/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("speech_translation.pretrain_schemes." + file_name)
