import random
from typing import Optional, Dict, List

import torch
from torch import Tensor

from fairseq.modules.transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .attention import RelativePositioningMultiheadAttention
from .relative_positioning import RelativePositionalEncoding


class RelativePositioningTransformerEncoderLayer(TransformerEncoderLayer):

    def __init__(self, args):
        super().__init__(args)
        self.use_relpos = args.use_relpos

        if self.use_relpos:
            self.relpos = RelativePositionalEncoding(args, self.embed_dim // args.encoder_attention_heads)

    def build_self_attention(self, embed_dim, args):
        if not args.use_relpos:
            return super().build_self_attention(embed_dim, args)
        else:
            return RelativePositioningMultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def forward_self_attention(self, x: torch.Tensor,
                               encoder_padding_mask: Optional[Tensor],
                               attn_mask: Optional[Tensor] = None):
        if self.use_relpos:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                position_encodings=self.relpos(x.size()[0]),
            )
        else:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
        x = self.dropout_module(x)
        return x

    def forward_self_attention_block(self, x: torch.Tensor, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = self.forward_self_attention(x, encoder_padding_mask, attn_mask)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        return x

    def forward_ffn(self, x: torch.Tensor):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

    def forward_ffn_block(self, x: torch.Tensor):
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.forward_ffn(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        x = self.forward_self_attention_block(x, encoder_padding_mask, attn_mask)
        x = self.forward_ffn_block(x)

        return x


class RelativePositioningTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, *nargs, **kwargs):
        super().__init__(args, *nargs, **kwargs)
        self.use_relpos = args.use_relpos

        if self.use_relpos:
            self.relpos_encoder = RelativePositionalEncoding(args, self.embed_dim // args.encoder_attention_heads)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        if not args.use_relpos:
            return super().build_self_attention(embed_dim, args, add_bias_kv, add_zero_attn)
        else:
            return RelativePositioningMultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    def forward_self_attention(self,
                               x,
                               encoder_out: Optional[torch.Tensor] = None,
                               encoder_padding_mask: Optional[torch.Tensor] = None,
                               incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                               prev_self_attn_state: Optional[List[torch.Tensor]] = None,
                               self_attn_mask: Optional[torch.Tensor] = None,
                               self_attn_padding_mask: Optional[torch.Tensor] = None, ):
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        if self.use_relpos:
            if incremental_state is None or "prev_key" not in _self_attn_input_buffer:
                # training mode, no offset needed, full input x given
                relpos = self.relpos_encoder(num_keys=x.size(0))
            else:
                # incremental inference mode, set offset to index of current step, length is 1
                _, _, offset, _ = _self_attn_input_buffer["prev_key"].size()  # coincides with the index of the current step
                num_keys = offset + 1

                relpos = self.relpos_encoder(num_keys, offset, num_queries=1)

            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_encodings=relpos
            )
            del relpos
        else:
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )

        x = self.dropout_module(x)
        return x, attn

    def forward_self_attention_block(self, x,
                                     encoder_out: Optional[torch.Tensor] = None,
                                     encoder_padding_mask: Optional[torch.Tensor] = None,
                                     incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                                     prev_self_attn_state: Optional[List[torch.Tensor]] = None,
                                     self_attn_mask: Optional[torch.Tensor] = None,
                                     self_attn_padding_mask: Optional[torch.Tensor] = None, ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn = self.forward_self_attention(x, encoder_out, encoder_padding_mask, incremental_state,
                                              prev_self_attn_state, self_attn_mask, self_attn_padding_mask)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        return x, attn

    def forward_cross_attention(self,
                                x,
                                encoder_out: Optional[torch.Tensor] = None,
                                encoder_padding_mask: Optional[torch.Tensor] = None,
                                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                                prev_attn_state: Optional[List[torch.Tensor]] = None,
                                need_attn: bool = False,
                                need_head_weights: bool = False, ):
        if prev_attn_state is not None:
            prev_key, prev_value = prev_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            assert incremental_state is not None
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )
        x = self.dropout_module(x)
        return x, attn

    def forward_cross_attention_block(self, x,
                                      encoder_out: Optional[torch.Tensor] = None,
                                      encoder_padding_mask: Optional[torch.Tensor] = None,
                                      incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                                      prev_attn_state: Optional[List[torch.Tensor]] = None,
                                      need_attn: bool = False,
                                      need_head_weights: bool = False, ):
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            x, attn = self.forward_cross_attention(x, encoder_out, encoder_padding_mask, incremental_state, prev_attn_state,
                                                   need_attn, need_head_weights)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        return x, attn

    def forward_ffn(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

    def forward_ffn_block(self, x):
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.forward_ffn(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_out:
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            self_attn_padding_mask:
            self_attn_mask:
            prev_attn_state:
            prev_self_attn_state:
            incremental_state:
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        x, _ = self.forward_self_attention_block(x, encoder_out, encoder_padding_mask, incremental_state,
                                                 prev_self_attn_state, self_attn_mask, self_attn_padding_mask)

        x, attn = self.forward_cross_attention_block(x, encoder_out, encoder_padding_mask, incremental_state, prev_attn_state,
                                                     need_attn, need_head_weights)

        x = self.forward_ffn_block(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


class StochasticTransformerEncoderLayer(RelativePositioningTransformerEncoderLayer):
    def __init__(self, args, layer_num):
        """ Adapted from https://github.com/quanpn90/NMTGMinor/blob/master/onmt/models/transformer_layers.py"""
        super().__init__(args)

        self.stochastic_p = args.stochastic_p
        assert 0 < self.stochastic_p <= 1
        self.stochastic_p_l = (layer_num + 1) / args.encoder_layers * (1 - self.stochastic_p)

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        compute_layer = True
        if self.stochastic_p < 1 and self.training:
            compute_layer = (random.random() >= self.stochastic_p_l)

        if compute_layer:
            return super().forward(x, encoder_padding_mask, attn_mask)
        else:
            return x

    def forward_ffn(self, x: torch.Tensor):
        x = super().forward_ffn(x)
        if self.stochastic_p < 1 and self.training:
            x = x / (1 - self.stochastic_p_l)
        return x

    def forward_self_attention(self, x: torch.Tensor, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        x = super().forward_self_attention(x, encoder_padding_mask, attn_mask)

        if self.stochastic_p < 1 and self.training:
            x = x / (1 - self.stochastic_p_l)
        return x


class StochasticTransformerDecoderLayer(RelativePositioningTransformerDecoderLayer):
    def __init__(self, args, layer_num, *nargs, **kwargs):
        super().__init__(args, *nargs, **kwargs)
        self.stochastic_p = args.stochastic_p
        assert 0 < self.stochastic_p <= 1
        self.stochastic_p_l = layer_num / args.encoder_layers * (1 - self.stochastic_p)

    def forward(self, x, encoder_out: Optional[torch.Tensor] = None, encoder_padding_mask: Optional[torch.Tensor] = None,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                prev_self_attn_state: Optional[List[torch.Tensor]] = None, prev_attn_state: Optional[List[torch.Tensor]] = None,
                self_attn_mask: Optional[torch.Tensor] = None, self_attn_padding_mask: Optional[torch.Tensor] = None,
                need_attn: bool = False, need_head_weights: bool = False):
        compute_layer = True
        if self.stochastic_p < 1 and self.training:
            compute_layer = (random.random() >= self.stochastic_p_l)

        if compute_layer:
            return super().forward(x, encoder_out, encoder_padding_mask, incremental_state, prev_self_attn_state, prev_attn_state,
                                   self_attn_mask, self_attn_padding_mask, need_attn, need_head_weights)
        else:
            return x, None, None

    def forward_self_attention(self, x, encoder_out: Optional[torch.Tensor] = None, encoder_padding_mask: Optional[torch.Tensor] = None,
                               incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                               prev_self_attn_state: Optional[List[torch.Tensor]] = None, self_attn_mask: Optional[torch.Tensor] = None,
                               self_attn_padding_mask: Optional[torch.Tensor] = None):
        x, attn = super().forward_self_attention(x, encoder_out, encoder_padding_mask, incremental_state, prev_self_attn_state,
                                                 self_attn_mask, self_attn_padding_mask)
        if self.stochastic_p < 1 and self.training:
            x = x / (1 - self.stochastic_p_l)
        return x, attn

    def forward_cross_attention(self, x, encoder_out: Optional[torch.Tensor] = None, encoder_padding_mask: Optional[torch.Tensor] = None,
                                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                                prev_attn_state: Optional[List[torch.Tensor]] = None, need_attn: bool = False,
                                need_head_weights: bool = False):
        x, attn = super().forward_cross_attention(x, encoder_out, encoder_padding_mask, incremental_state, prev_attn_state, need_attn,
                                                  need_head_weights)
        if self.stochastic_p < 1 and self.training:
            x = x / (1 - self.stochastic_p_l)
        return x, attn

    def forward_ffn(self, x):
        x = super().forward_ffn(x)
        if self.stochastic_p < 1 and self.training:
            x = x / (1 - self.stochastic_p_l)
        return x
