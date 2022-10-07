import math
from typing import List, Dict, Optional

import torch
from torch import Tensor

from fairseq.sequence_generator import EnsembleModel


class WeightedEnsembleModel(EnsembleModel):
    def __init__(self, models, weights: Tensor):
        super().__init__(models)

        assert len(models) == weights.size(1)
        self.weights: Tensor = weights / weights.sum(-1, keepdim=True)  # (bsz, num_pivs)

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        beam_size = log_probs[0].size(0) / self.weights.size(0)
        assert float(beam_size).is_integer(), f"batch * tgt_beam = {log_probs[0].size(0)}, " \
                                              f"batch = {self.weights.size(0)}"
        weights = self.weights.T  # (piv_beam, batch)
        weights = weights.repeat_interleave(int(beam_size), dim=-1)  # (piv_beam, batch * beam)
        weights = weights.unsqueeze(-1)  # (piv_beam, batch * beam, 1)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0) + torch.log(weights), dim=0)

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn
