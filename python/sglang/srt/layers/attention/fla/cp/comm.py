# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported (forward-only subset) from flash-linear-attention:
#   fla/ops/cp/comm.py
# for use by SGLang's KDA prefill context-parallel path.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def all_gather_into_tensor(
    inp: torch.Tensor,
    out: torch.Tensor | None = None,
    group: "ProcessGroup | None" = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, "dist.Work | None"]:
    """All-gather a tensor across ranks.

    Args:
        inp: Input tensor to gather.
        out: Optional output tensor of shape ``[world_size, *inp.shape]``.
        group: Process group.
        async_op: Whether to perform an async operation.

    Returns:
        Tuple of (output tensor, handle if async_op else None).
    """
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def send_recv_fwd(
    send_tensor: torch.Tensor,
    group: "ProcessGroup",
    recv_from_prev: bool = True,
) -> torch.Tensor:
    """Forward-pass communication implemented with all-gather.

    Every rank participates in one all-gather; each rank then picks the slice
    from its previous (or next) rank. Ranks with no valid source get zeros.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    gathered, _ = all_gather_into_tensor(send_tensor, group=group, async_op=False)

    if recv_from_prev:
        if rank == 0:
            return torch.zeros_like(send_tensor)
        return gathered[rank - 1].clone()
    else:
        if rank == world_size - 1:
            return torch.zeros_like(send_tensor)
        return gathered[rank + 1].clone()


def conv_cp_send_recv_fwd(tails: torch.Tensor, group: "ProcessGroup") -> torch.Tensor:
    """Conv1d CP forward: each rank sends its ``W-1`` tail tokens and receives
    the previous rank's tail tokens (to use as conv ``initial_state``).

    Args:
        tails: ``[W-1, D]`` tail tokens from the current rank.
        group: Process group.

    Returns:
        heads: same shape as ``tails`` — previous rank's tail (zeros for rank 0).
    """
    return send_recv_fwd(tails, group, recv_from_prev=True)
