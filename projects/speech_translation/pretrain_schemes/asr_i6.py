from argparse import Namespace
from typing import Optional, Any

from speech_translation.pretrain_schemes import register_pretrain_scheme


@register_pretrain_scheme('asr_i6')
def asr_i6(model_cfg: Namespace, criterion_cfg: Namespace, task, num_updates: int, subepoch: Optional[int] = None) -> Optional[
    tuple[dict[str, Any],
          dict[str, Any]]]:

    # requires subepochs
    assert subepoch is not None

    # params: TODO move out
    pretrain_repetitions = 5
    initial_dim_factor = 0.5
    start_num_layers = 2

    current_repetition = (subepoch - 1) // pretrain_repetitions

    dim_keys = {"lstm_dim", "encoder_ffn_embed_dim", "decoder_ffn_embed_dim", "encoder_embed_dim", "decoder_output_dim",
                "decoder_embed_dim", "decoder_input_dim"}
    dropout_keys = ["dropout", "attention_dropout", "activation_dropout", "embedding_dropout"]

    att_num_heads = max(model_cfg.encoder_attention_heads, model_cfg.decoder_attention_heads)

    max_num_layers = max(model_cfg.encoder_layers, model_cfg.decoder_layers)
    num_layers = 2 ** current_repetition

    if num_layers > max_num_layers:
        # Pretraining finished
        return None

    # Continue pretraining
    new_criterion_args = {'label_smoothing': 0}
    new_model_args = {}

    for dropout in dropout_keys:
        new_model_args[dropout] = 0

    new_model_args['encoder_layers'] = min(num_layers, model_cfg.encoder_layers)
    new_model_args['decoder_layers'] = min(num_layers, model_cfg.decoder_layers)

    if max_num_layers > start_num_layers:
        grow_frac = 1.0 - float(max_num_layers - num_layers) / (max_num_layers - 1)
        dim_frac = initial_dim_factor + (1.0 - initial_dim_factor) * grow_frac

        #for dropout in dropout_keys:
        #    new_model_args[dropout] = getattr(model_cfg, dropout) * grow_frac
        for key in dim_keys:
            if not getattr(model_cfg, key, False):
                continue
            new_model_args[key] = int(getattr(model_cfg, key) * dim_frac / float(att_num_heads)) * att_num_heads

    return new_model_args, new_criterion_args
