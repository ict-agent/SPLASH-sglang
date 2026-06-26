from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class StateType(str, enum.Enum):
    MAMBA = "mamba"
    SWA = "swa"
    NSA = "nsa"
    NSA_TAIL = "nsa_tail"  # NSA kpool tail: one per-request slot index.


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    # Side state is a list of components; entry ``i`` is one ordered group of
    # buffers (a "component") whose addressing is selected by ``state_types[i]``.
    # A request's per-component indices ride the wire as a parallel list-of-lists
    # so the two nodes never need to agree on a flat layout. Example layout for
    # an MTP model whose target and draft are both NSA+kpool:
    #
    #     state_types = [MAMBA, NSA, NSA_TAIL, NSA, NSA_TAIL]
    #                     mamba  tgt  tgt_tail  drf  drf_tail
    state_types: List[StateType]
    state_data_ptrs: List[List[int]]
    state_data_lens: List[List[int]]
    state_item_lens: List[List[int]]
    # Per-component, per-tensor TP slice dim, used when prefill/decode attn_tp
    # sizes differ (mamba head-shard transfer only).
    state_dim_per_tensor: List[List[int]]
    # Per-component, per-tensor packed sub-component sizes for the mamba slice
    # path (e.g. fused [Q|K|V]). Local-only: never sent on the wire.
    state_dim_components_per_tensor: List[List[List[int]]]
    ib_device: str
    ib_traffic_class: str
    gpu_id: int
    kv_head_num: int
    total_kv_head_num: int
    page_size: int
    # for pp prefill
    pp_rank: int
    prefill_start_layer: int
    # for system dp
    system_dp_rank: int


class KVPoll:
    Failed = 0
    Cancelled = 1
    Bootstrapping = 2
    WaitingForInput = 3
    Transferring = 4
    Success = 5


class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """Register prefill server info to the bootstrap server."""
        ...


class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[List[int]]] = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...


class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        prefill_dp_rank: int,
    ):
        """
        Resolve bootstrap metadata and mark the receiver ready for transfer metadata.
        """
        ...

    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[List[int]]] = None,
    ):
        """
        Notify the prefill server about the kv indices, aux index, and state_indices.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...

    def clear(self):
        """
        Clear any internal states.
        """
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        pass


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
