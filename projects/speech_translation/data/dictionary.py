import torch

from fairseq.data import Dictionary
from fairseq import utils
from fairseq.data import data_utils


class CustomDictionary(Dictionary):

    def __init__(self, *, bos="<s>", pad="<pad>", eos="</s>", unk="<unk>", extra_special_symbols=None):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.extra_special_symbol_indices = set()
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_special_symbol(s)

        self.nspecial = len(self.symbols)

    def add_special_symbol(self, word):
        self.add_symbol(word)
        self.extra_special_symbol_indices.add(self.index(word))

        self.nspecial = len([self.bos_index, self.pad_index, self.eos_index, self.unk_index]) + len(
            self.extra_special_symbol_indices)

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(
                    t,
                    bpe_symbol,
                    escape_unk,
                    extra_symbols_to_ignore,
                    include_eos=include_eos,
                )
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore = extra_symbols_to_ignore.union(self.extra_special_symbol_indices)
        if not include_eos:
            extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())

        sent = separator.join(
            token_string(i)
            for i in tensor
            if utils.item(i) not in extra_symbols_to_ignore
        )

        return data_utils.post_process(sent, bpe_symbol)
