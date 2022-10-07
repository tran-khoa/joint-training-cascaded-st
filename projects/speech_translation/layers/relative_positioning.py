import argparse
from typing import Optional, Union

import torch
import torch.nn as nn


@torch.jit.script
def fused_index_matrix(index_matrix, offset: int, clipping: int):
    index_matrix = index_matrix - offset
    return (index_matrix - index_matrix.T).clamp(min=-clipping, max=clipping) + clipping


class RelativePositionalEncoding(nn.Module):
    """
    Relative positioning term as introduced by Shaw et al., 2018

    Usually added to Self-Attention using key_shift.
    Parts of the code are adapted from Tensor2Tensor (https://github.com/tensorflow/tensor2tensor).
    """
    def __init__(self, args: argparse.Namespace, n_out: int):
        super().__init__()

        self.n_out = n_out
        self.clipping = args.relpos_clipping  # TODO Namespace not necessary here...
        self.encoding_matrix = nn.Parameter(torch.empty(size=(2 * self.clipping + 1, n_out), requires_grad=True))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoding_matrix)

    def forward(self, num_keys: Union[int, torch.Tensor], offset: Union[int, torch.Tensor] = 0,
                num_queries: Optional[Union[int, torch.Tensor]] = None):
        """
        Usually, in training, the length should be the sequence length and offset should be 0.
        In auto-regressive decoding, offset is the current time step t and length should be t+1.

        Args:
            num_keys: scalar describing the length of the positional encoding
            offset: scalar describing the position of the first query encoded
            num_queries: number of queries, must be smaller or equal to num_keys (e.g. in decoding)

        Returns: Encoding tensor of shape (T_q, T_k, n_out) (corresponds to T_q, T_k, d_k, where T_q=T_k in standard self-attention)

        """
        if num_queries is not None:
            assert num_queries <= num_keys, f"num_queries ({num_queries}) must not be larger than num_keys ({num_keys})"

        device = self.encoding_matrix.device

        # create range vector
        indices = torch.arange(0, num_keys, dtype=torch.long, device=device).view(1, num_keys)

        # create index matrix
        indices = indices.repeat(1, num_keys).view(num_keys, num_keys)

        # compute index matrix
        indices = fused_index_matrix(indices, offset, self.clipping)

        # if num_queries given, only return last num_queries rows
        if num_queries is not None:
            indices = indices[-num_queries:, :]

        return self.encoding_matrix[indices]
