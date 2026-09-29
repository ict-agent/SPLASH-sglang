from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch


def _server_args():
    from sglang.srt.server_args import get_global_server_args

    return get_global_server_args()


def _getattr_default(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


class _ParallelCompat:
    def __getattr__(self, name: str) -> Any:
        args = _server_args()
        if hasattr(args, name):
            return getattr(args, name)
        raise AttributeError(name)

    @property
    def world_size(self) -> int:
        from sglang.srt.distributed import get_world_size

        return get_world_size()

    @property
    def world_rank(self) -> int:
        from sglang.srt.distributed import get_world_rank

        return get_world_rank()

    @property
    def tp_size(self) -> int:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()

    @property
    def tp_rank(self) -> int:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()

    @property
    def attn_tp_size(self) -> int:
        from sglang.srt.distributed import get_attn_tensor_model_parallel_world_size

        return get_attn_tensor_model_parallel_world_size()

    @property
    def attn_tp_rank(self) -> int:
        from sglang.srt.distributed import get_attn_tensor_model_parallel_rank

        return get_attn_tensor_model_parallel_rank()

    @property
    def attn_cp_size(self) -> int:
        from sglang.srt.distributed import get_attn_context_model_parallel_world_size

        return get_attn_context_model_parallel_world_size()

    @property
    def attn_cp_rank(self) -> int:
        from sglang.srt.distributed import get_attn_context_model_parallel_rank

        return get_attn_context_model_parallel_rank()

    @property
    def attn_dp_size(self) -> int:
        from sglang.srt.layers.dp_attention import get_attention_dp_size

        return get_attention_dp_size()

    @property
    def attn_dp_rank(self) -> int:
        from sglang.srt.layers.dp_attention import get_attention_dp_rank

        return get_attention_dp_rank()

    @property
    def moe_ep_size(self) -> int:
        from sglang.srt.distributed import get_moe_expert_parallel_world_size

        return get_moe_expert_parallel_world_size()

    @property
    def moe_ep_rank(self) -> int:
        from sglang.srt.distributed import get_moe_expert_parallel_rank

        return get_moe_expert_parallel_rank()

    @property
    def moe_dp_size(self) -> int:
        from sglang.srt.distributed import get_moe_data_parallel_world_size

        return get_moe_data_parallel_world_size()

    @property
    def moe_dp_rank(self) -> int:
        from sglang.srt.distributed import get_moe_data_parallel_rank

        return get_moe_data_parallel_rank()

    @property
    def moe_tp_size(self) -> int:
        from sglang.srt.distributed import get_moe_tensor_parallel_world_size

        return get_moe_tensor_parallel_world_size()

    @property
    def moe_tp_rank(self) -> int:
        from sglang.srt.distributed import get_moe_tensor_parallel_rank

        return get_moe_tensor_parallel_rank()

    @property
    def tp_group(self):
        from sglang.srt.distributed import get_tp_group

        return get_tp_group()

    @property
    def attn_tp_group(self):
        from sglang.srt.distributed import get_attn_tp_group

        return get_attn_tp_group()

    @property
    def attn_cp_group(self):
        from sglang.srt.distributed import get_attn_cp_group

        return get_attn_cp_group()

    @property
    def moe_ep_group(self):
        from sglang.srt.distributed import get_moe_ep_group

        return get_moe_ep_group()

    @property
    def moe_dp_group(self):
        from sglang.srt.distributed import get_moe_dp_group

        return get_moe_dp_group()

    @property
    def moe_tp_group(self):
        from sglang.srt.distributed import get_moe_tp_group

        return get_moe_tp_group()

    @property
    def dcp_enabled(self) -> bool:
        return int(_getattr_default(_server_args(), "dcp_size", 1) or 1) > 1

    @property
    def dcp_size(self) -> int:
        return int(_getattr_default(_server_args(), "dcp_size", 1) or 1)

    @property
    def dcp_rank(self) -> int:
        return 0

    @property
    def attn_dcp_size(self) -> int:
        return self.dcp_size if self.dcp_enabled else 1

    @property
    def attn_dcp_rank(self) -> int:
        return self.dcp_rank

    @property
    def dcp_replicate_q_proj(self) -> bool:
        return bool(_getattr_default(_server_args(), "dcp_replicate_q_proj", False))

    @property
    def dcp_comm_backend(self) -> str:
        return _getattr_default(_server_args(), "dcp_comm_backend", "all_gather")

    @property
    def dcp_group(self):
        return self.attn_cp_group


class _DpFlags(SimpleNamespace):
    enabled: bool = False
    buffer_hidden_size: int = 0
    buffer_dtype: Any = torch.bfloat16
    buffer_device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len_with_idle: bool = False
    use_world_group_for_gather: bool = False
    joiner_skip_all_gather: bool = False


class _Flags(SimpleNamespace):
    def __init__(self):
        super().__init__(dp=_DpFlags(), capture=SimpleNamespace())


class _ForwardState(SimpleNamespace):
    def __init__(self):
        super().__init__(is_extend_in_batch=False)

    def set(self, name: str, value: Any) -> None:
        setattr(self, name, value)


_PARALLEL = _ParallelCompat()
_FLAGS = _Flags()
_FORWARD = _ForwardState()
_RESOURCES = SimpleNamespace(tbo_event_pool={})
_STREAMS = {}


def get_parallel() -> _ParallelCompat:
    return _PARALLEL


def get_flags() -> _Flags:
    return _FLAGS


def get_forward() -> _ForwardState:
    return _FORWARD


def get_resources() -> SimpleNamespace:
    return _RESOURCES


def get_stream(name: str):
    stream = _STREAMS.get(name)
    if stream is None:
        stream = torch.cuda.Stream()
        _STREAMS[name] = stream
    return stream


def get_device() -> SimpleNamespace:
    return SimpleNamespace(device=_getattr_default(_server_args(), "device", "cuda"))


def get_exec() -> SimpleNamespace:
    args = _server_args()
    kernel = SimpleNamespace(
        flashinfer_mla_disable_ragged=_getattr_default(
            args, "flashinfer_mla_disable_ragged", False
        ),
        dsa_decode_backend=_getattr_default(
            args, "dsa_decode_backend", _getattr_default(args, "nsa_decode_backend", None)
        ),
        dsa_prefill_backend=_getattr_default(
            args,
            "dsa_prefill_backend",
            _getattr_default(args, "nsa_prefill_backend", None),
        ),
    )
    moe = SimpleNamespace(
        elastic_ep_backend=_getattr_default(args, "elastic_ep_backend", None),
        moe_a2a_backend=_getattr_default(args, "moe_a2a_backend", None),
    )
    return SimpleNamespace(kernel=kernel, moe=moe)


def configured_attn_cp_size() -> int:
    return int(_getattr_default(_server_args(), "attn_cp_size", 1) or 1)


def configured_moe_dp_size() -> int:
    return int(_getattr_default(_server_args(), "moe_dp_size", 1) or 1)
