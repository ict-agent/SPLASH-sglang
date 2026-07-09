# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Lightweight, inference-only CP context for SGLang's KDA prefill CP path.
# Unlike flash-linear-attention's ``build_cp_context`` (which re-derives a
# uniform ``total // world_size`` split and drops the remainder), the fields
# here are populated by the caller from SGLang's remainder-aware plain-split
# metadata (see ``kda_cp_utils.build_kda_fla_cp_context``).

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass
class FLACPContext:
    """Operator-level context parallel context (forward-only subset).

    Attributes:
        group: process group used for the all-gather / halo exchange.
        cu_seqlens: rank-local cumulative sequence lengths (int32, GPU),
            shape ``[num_local_segments + 1]``.
        cu_seqlens_cpu: same data on CPU (optional).
        is_first_rank: whether this rank owns the first token of its first
            local segment's sequence (i.e. no incoming cross-rank state).
        is_last_rank: whether this rank owns the last token of its last local
            segment's sequence (i.e. no outgoing cross-rank state).
        pre_num_ranks: number of previous ranks that contribute state to this
            rank's first local segment (0 when ``is_first_rank``).
        pre_num_conv_tokens: number of tokens the previous rank(s) hold that
            this rank needs as conv1d history for its first local segment.
    """

    group: "ProcessGroup | None" = None
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    is_first_rank: bool | None = None
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    pre_num_conv_tokens: int | None = None

    @property
    def num_seqs(self) -> int:
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    @property
    def is_cp_enabled(self) -> bool:
        return self.group is not None
