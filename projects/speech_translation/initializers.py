import math
import warnings

import torch
from torch.nn.init import _calculate_correct_fan


def rwth_initializer(tensor: torch.Tensor, gain: float = math.sqrt(0.78)):
    """
    Equivalent to TensorFlow's tf.keras.initializers.VarianceScaling
    with mode=fan_in and with scale=gain^2.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, 'fan_in')
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
