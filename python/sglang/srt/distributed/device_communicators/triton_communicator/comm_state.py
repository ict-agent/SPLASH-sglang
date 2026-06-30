# Adapted from https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py

"""Host-side comm state, launch-geometry config, and token-distribution
helpers shared by the NVIDIA multimem all_gather / reduce_scatter launchers."""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


_MULTIMEM_BLOCK_THREADS = 1024
_MULTIMEM_NUMEL_PER_THREAD = 8
_MULTIMEM_MIN_BLOCKS = 4
_MULTIMEM_MAX_BLOCKS = 32
_WARP_SIZE = 32


@dataclass
class TritonMultimemState:
    group: dist.ProcessGroup
    rank_in_group: int
    world_size: int
    device: torch.device
    max_numel: int = 0
    comm_buffer: torch.Tensor | None = None
    symm_mem_handle: object | None = None
    _last_collective_stream: torch.cuda.Stream | None = None
    _last_collective_event: torch.cuda.Event | None = None

    def validate_symm_mem_handle(self, symm_mem_handle):
        multicast_ptr = getattr(symm_mem_handle, "multicast_ptr", 0)
        if not multicast_ptr:
            raise RuntimeError(
                "Triton multimem communicator is unavailable: symmetric-memory "
                "rendezvous returned multicast_ptr=0."
            )
        if self.rank_in_group != symm_mem_handle.rank:
            raise RuntimeError(
                f"Mismatched rank id: state={self.rank_in_group}, "
                f"symm_mem={symm_mem_handle.rank}"
            )
        if self.world_size != symm_mem_handle.world_size:
            raise RuntimeError(
                f"Mismatched world size: state={self.world_size}, "
                f"symm_mem={symm_mem_handle.world_size}"
            )

    def get_symm_mem_handle(self):
        if self.symm_mem_handle is None:
            self.symm_mem_handle = symm_mem.rendezvous(
                self.comm_buffer, group=self.group
            )
            self.validate_symm_mem_handle(self.symm_mem_handle)
        return self.symm_mem_handle

    def wait_for_workspace(self):
        """Serialize launches that reuse the same signal pad and comm buffer.

        Same-stream launches are already ordered. For cross-stream reuse, record
        an event on the previous stream at the moment the next collective is
        about to start, so any caller work queued after the previous collective
        and before this call is also protected.
        """
        current_stream = torch.cuda.current_stream(self.device)
        last_stream = self._last_collective_stream
        if last_stream is None or _same_cuda_stream(last_stream, current_stream):
            return

        event = torch.cuda.Event()
        event.record(last_stream)
        current_stream.wait_event(event)
        # Keep the event alive until the wait has been enqueued and another
        # cross-stream handoff replaces it.
        self._last_collective_event = event

    def mark_workspace_used(self):
        self._last_collective_stream = torch.cuda.current_stream(self.device)

    def view_comm_buffer(self, rows: int, cols: int) -> torch.Tensor:
        """View the flat comm buffer as a contiguous ``[rows, cols]`` tensor."""
        numel = rows * cols
        assert (
            numel <= self.max_numel
        ), f"comm buffer too small: {rows}*{cols} > {self.max_numel}"
        return self.comm_buffer[:numel].view(rows, cols)


def _same_cuda_stream(lhs: torch.cuda.Stream, rhs: torch.cuda.Stream) -> bool:
    return lhs.cuda_stream == rhs.cuda_stream


def _validate_create_state_args(
    group: dist.ProcessGroup,
    rank_in_group: int,
    buffer_bytes: int,
) -> int:
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError(f"Expected dist.ProcessGroup, got {type(group)}")
    if buffer_bytes <= 0:
        raise ValueError(f"buffer_bytes must be positive, got {buffer_bytes}")
    world_size = group.size()
    if world_size & (world_size - 1) != 0:
        raise RuntimeError(
            "Triton multimem communicator currently requires a power-of-two "
            f"world size, got {world_size}."
        )
    if not 0 <= rank_in_group < world_size:
        raise RuntimeError(
            f"rank_in_group must be in [0, {world_size}), got {rank_in_group}."
        )
    return world_size


def _reserve_signal_pad(world_size: int):
    # blockwise_barrier indexes the pad at block_id * world_size + rank. Reserve
    # enough slots for the largest CTA grid any launcher can use.
    pad_bytes = _MULTIMEM_MAX_BLOCKS * world_size * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))


def get_symm_mem_handle(state: TritonMultimemState):
    return state.get_symm_mem_handle()


def wait_for_workspace(state: TritonMultimemState):
    state.wait_for_workspace()


def mark_workspace_used(state: TritonMultimemState):
    state.mark_workspace_used()


def get_launch_config(
    local_numel: int, num_blocks: int | None = None
) -> Tuple[int, int, int, int]:
    max_block_size = _MULTIMEM_BLOCK_THREADS
    bytes_per_thread = 16
    numel_per_thread = _MULTIMEM_NUMEL_PER_THREAD
    assert (
        local_numel % numel_per_thread == 0
    ), f"The number of elements must be {bytes_per_thread} bytes aligned"
    block_size = max_block_size
    num_warps = max_block_size // _WARP_SIZE
    # A payload-scaled count is passed when the caller has one; otherwise fall
    # back to the minimum grid.
    num_blocks = _MULTIMEM_MIN_BLOCKS if num_blocks is None else num_blocks
    assert (
        num_blocks <= _MULTIMEM_MAX_BLOCKS
    ), f"num_blocks={num_blocks} exceeds signal-pad capacity {_MULTIMEM_MAX_BLOCKS}"
    return num_blocks, block_size, num_warps, numel_per_thread


def get_even_token_distribution(
    state: TritonMultimemState, total_tokens_in_group: int
) -> List[int]:
    token_list_in_group = []
    for rank in range(state.world_size):
        num_tokens_per_rank = total_tokens_in_group // state.world_size + (
            1 if (rank < total_tokens_in_group % state.world_size) else 0
        )
        token_list_in_group.append(num_tokens_per_rank)
    return token_list_in_group


def get_token_partition(
    state: TritonMultimemState, token_list_in_group: List[int]
) -> Tuple[int, int, int]:
    total_num_tokens = sum(token_list_in_group)
    local_num_tokens = token_list_in_group[state.rank_in_group]
    local_token_offset = sum(token_list_in_group[: state.rank_in_group])
    return total_num_tokens, local_num_tokens, local_token_offset


def is_even_token_distribution(token_list_in_group: List[int]) -> bool:
    """True when every rank holds the same token count -- a plain rank-major
    split that torch's multimem all_gather / reduce_scatter ops can handle
    directly (they don't support the non-uniform token_list case)."""
    return len(set(token_list_in_group)) == 1


def fits_comm_buffer(state: TritonMultimemState, rows: int, cols: int) -> bool:
    """Whether a ``[rows, cols]`` collective fits the fixed comm buffer."""
    return rows * cols <= state.max_numel


def view_comm_buffer(
    state: TritonMultimemState,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """View the flat comm buffer as a contiguous ``[rows, cols]`` tensor.

    The buffer is a flat element budget; a collective re-views it to the exact
    shape it needs each call. Asserts the request fits ``state.max_numel``.
    """
    return state.view_comm_buffer(rows, cols)


def create_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    buffer_bytes: int,
    device: torch.device = None,
) -> TritonMultimemState:
    """Allocate a fixed-size symmetric-memory comm buffer shared by all the
    multimem collectives. NVIDIA / multimem only.

    ``buffer_bytes`` is the flat byte budget (e.g. 64 * 1024 * 1024 for 64 MiB);
    it is rounded down to a whole number of bf16 elements. Each collective views
    this flat buffer into the 2-D shape it needs and only requires that its
    gathered ``total_tokens * hidden`` fit within the budget.
    """
    world_size = _validate_create_state_args(group, rank_in_group, buffer_bytes)
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    max_numel = buffer_bytes // 2  # 2 bytes / bf16 element
    _reserve_signal_pad(world_size)
    free_gpu_memory_begin = get_available_gpu_memory(
        "cuda", torch.cuda.current_device()
    )
    # Allocate outside inference_mode so the persistent comm buffer is not an
    # inference tensor; this state is often lazily constructed during forward
    # (which may run under inference_mode). Pair with no_grad so we don't
    # accidentally re-enable autograd just to escape inference.
    with torch.inference_mode(False), torch.no_grad():
        comm_buffer = symm_mem.empty(
            (max_numel,), dtype=torch.bfloat16, device=device
        )
    free_gpu_memory_after = get_available_gpu_memory(
        "cuda", torch.cuda.current_device()
    )
    logger.info(
        "Triton multimem comm buffer allocated: %s GB",
        free_gpu_memory_begin - free_gpu_memory_after,
    )
    state = TritonMultimemState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=world_size,
        device=device,
        max_numel=max_numel,
        comm_buffer=comm_buffer,
    )
    state.symm_mem_handle = symm_mem.rendezvous(comm_buffer, group=group)
    state.validate_symm_mem_handle(state.symm_mem_handle)
    return state
