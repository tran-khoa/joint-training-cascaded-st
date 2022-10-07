from typing import TypedDict, Union, Any
from torchtyping import TensorType


class NetInput(TypedDict, total=False):
    src_tokens: TensorType['batch', 'src_time']
    src_lengths: TensorType['batch']
    prev_output_tokens: TensorType['batch', 'time']


class PivNetInput(NetInput, total=False):
    asr_piv_tokens: TensorType['batch', 'piv_time']
    asr_piv_lengths: TensorType['batch']
    asr_prev_output_tokens: TensorType['batch', 'time']
    mt_piv_tokens: TensorType['batch', 'piv_time']
    mt_piv_lengths: TensorType['batch']


class Sample(TypedDict, total=False):
    id: TensorType['batch']
    net_input: Union[NetInput, PivNetInput]
    ntokens: int
    nsentences: int
    target: TensorType['batch', 'time']


class STCriterionSample(Sample, total=False):
    asr_score: TensorType['batch', 'beam']
    decoder_out: tuple[TensorType['batch', 'tgt_len', 'vocab'], Any]
    target_lengths: TensorType['batch']


class JointSample(TypedDict):
    asr: Sample
    mt: Sample
    st: Sample
