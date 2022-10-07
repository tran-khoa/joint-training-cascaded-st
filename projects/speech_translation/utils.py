import torch

from torchtyping import TensorType


def move_eos_to_beginning(tensor: TensorType['batch', 'time'], eos_token: int, pad_token: int) -> TensorType['batch', 'time']:
    data = torch.where((tensor[:, :-1] == eos_token), pad_token, tensor[:, :-1])
    eos = tensor.new_full(size=(tensor.size(0), 1), fill_value=eos_token)
    return torch.cat((eos, data), dim=1)
