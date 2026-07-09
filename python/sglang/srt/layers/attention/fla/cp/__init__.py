# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Forward-only context-parallel primitives for SGLang's KDA prefill CP path,
# ported from flash-linear-attention (fla/ops/cp).

from sglang.srt.layers.attention.fla.cp.chunk_delta_h_cp import (
    chunk_gated_delta_rule_fwd_h_pre_process,
)
from sglang.srt.layers.attention.fla.cp.comm import (
    all_gather_into_tensor,
    conv_cp_send_recv_fwd,
)
from sglang.srt.layers.attention.fla.cp.context import FLACPContext

__all__ = [
    "FLACPContext",
    "all_gather_into_tensor",
    "conv_cp_send_recv_fwd",
    "chunk_gated_delta_rule_fwd_h_pre_process",
]
