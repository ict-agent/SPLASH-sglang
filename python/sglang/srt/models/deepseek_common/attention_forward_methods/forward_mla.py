from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
)
from sglang.srt.layers.mla_only_dp_transfer import (
    dp_packed_head_major_to_tp_head_shard,
    dp_packed_rows_to_tp_head_shard,
    dp_rows_to_tp_head_shard,
    tp_fused_q_head_shard_to_dp_rows,
    tp_two_head_shards_to_dp_rows,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_cpu,
    _is_cublas_ge_129,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _use_aiter,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

logger = logging.getLogger(__name__)
_logged_q_b_proj_move_reasons: set[str] = set()
_logged_o_proj_move_reasons: set[str] = set()


def _use_true_mla_only_dp(
    self: "DeepseekV2AttentionMLA", forward_batch: ForwardBatch
) -> bool:
    return bool(
        getattr(self, "mla_only_dp", False)
        and forward_batch.global_num_tokens_cpu is not None
        and forward_batch.global_dp_buffer_len is not None
    )


def _dp_gather_rows_like(
    local_rows: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Replicate owner-local DP rows into the global token order on every TP rank."""
    global_rows = local_rows.new_empty(
        (forward_batch.global_dp_buffer_len, *local_rows.shape[1:])
    )
    dp_gather_replicate(
        global_rows.flatten(1),
        local_rows.contiguous().flatten(1),
        forward_batch,
    )
    return global_rows


def _dp_gather_positions_once(
    positions: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Replicate position ids once per MLA-DP forward batch.

    MLA-DP applies RoPE before the TP-to-DP query transfer, so q uses global
    positions while k uses owner-local positions. The global position vector is
    identical for every layer in one forward pass; caching it avoids one tiny
    DP gather and allocation on every following MLA layer.
    """

    cached = getattr(forward_batch, "_mla_only_dp_global_positions", None)
    if (
        cached is not None
        and cached.shape[0] == forward_batch.global_dp_buffer_len
        and cached.device == positions.device
        and cached.dtype == positions.dtype
    ):
        return cached

    global_positions = _dp_gather_rows_like(
        positions.reshape(-1, 1), forward_batch
    ).reshape(-1)
    setattr(forward_batch, "_mla_only_dp_global_positions", global_positions)
    return global_positions


def _dp_scatter_rows_like(
    global_rows: torch.Tensor, local_rows: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Select this DP rank's owner-local rows from a global token buffer."""
    scattered = global_rows.new_empty((local_rows.shape[0], *global_rows.shape[1:]))
    dp_scatter(
        scattered.flatten(1),
        global_rows.contiguous().flatten(1),
        forward_batch,
    )
    return scattered


def _all_equal_int(values: list[int]) -> bool:
    if len(values) <= 1:
        return True
    first = values[0]
    for value in values:
        if value != first:
            return False
    return True


def _use_mla_only_dp_o_proj_reduce_scatter(
    attn_bmm_output: torch.Tensor,
    local_rows_like: torch.Tensor,
    forward_batch: ForwardBatch,
) -> bool:
    if os.getenv("SGLANG_MLA_ONLY_DP_ENABLE_O_PROJ_REDUCE_SCATTER") != "1":
        return False
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1 or forward_batch.global_num_tokens_cpu is None:
        return False
    token_counts = [int(count) for count in forward_batch.global_num_tokens_cpu]
    local_rows = local_rows_like.shape[0]
    can_reduce_scatterv = False
    if os.getenv("SGLANG_MLA_ONLY_DP_ENABLE_O_PROJ_REDUCE_SCATTERV") == "1":
        pynccl_comm = getattr(get_tp_group(), "pynccl_comm", None)
        can_reduce_scatterv = pynccl_comm is not None and not getattr(
            pynccl_comm, "disabled", False
        )
    return (
        len(token_counts) == tp_size
        and attn_bmm_output.shape[0] == sum(token_counts)
        and token_counts[get_tensor_model_parallel_rank()] == local_rows
        and (_all_equal_int(token_counts) or can_reduce_scatterv)
    )


def _mla_only_dp_o_proj_to_dp_rows(
    self: "DeepseekV2AttentionMLA",
    attn_bmm_output: torch.Tensor,
    local_rows_like: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    if _use_mla_only_dp_o_proj_reduce_scatter(
        attn_bmm_output, local_rows_like, forward_batch
    ):
        output_parallel, _ = self.o_proj(attn_bmm_output, skip_all_reduce=True)
        output = output_parallel.new_empty(
            (local_rows_like.shape[0], output_parallel.shape[1])
        )
        token_counts = [int(count) for count in forward_batch.global_num_tokens_cpu]
        if _all_equal_int(token_counts):
            dp_reduce_scatter_tensor(output, output_parallel.contiguous())
        else:
            get_tp_group().reduce_scatterv(
                output_parallel.contiguous(), output=output, sizes=token_counts
            )
        return output

    output, _ = self.o_proj(attn_bmm_output)
    return _dp_scatter_rows_like(output, local_rows_like, forward_batch)


def _return_output_with_topk(
    self: "DeepseekV2AttentionMLA",
    output: torch.Tensor,
    topk_indices: Optional[torch.Tensor],
):
    if self.next_skip_topk is None:
        return output

    # Return topk_indices for the next layer when enabling index cache.
    if not self.next_skip_topk:
        return output, None
    else:
        return output, topk_indices


def _tp_all_gather_head_shards(x: torch.Tensor) -> torch.Tensor:
    """Assemble full-head tensors from full tensor-parallel projection shards."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return x
    flat = x.contiguous().view(x.shape[0], -1)
    gathered = flat.new_empty((flat.shape[0] * tp_size, flat.shape[1]))
    get_tp_group().all_gather_into_tensor(gathered, flat)
    return (
        gathered.view(tp_size, x.shape[0], *x.shape[1:])
        .permute(1, 0, 2, 3)
        .reshape(x.shape[0], tp_size * x.shape[1], x.shape[2])
        .contiguous()
    )


def _tp_all_gather_head_weights(x: torch.Tensor) -> torch.Tensor:
    """Assemble local per-head weight shards into rank-major full-head weights."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return x
    flat = x.contiguous().view(-1)
    gathered = flat.new_empty((tp_size, flat.numel()))
    get_tp_group().all_gather_into_tensor(gathered, flat)
    return (
        gathered.view(tp_size, *x.shape)
        .reshape(tp_size * x.shape[0], *x.shape[1:])
        .contiguous()
    )


def _tp_all_gather_row_parallel_input_weights(x: torch.Tensor) -> torch.Tensor:
    """Assemble row-parallel input-dimension shards into a full weight tensor."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return x
    flat = x.contiguous().view(-1)
    gathered = flat.new_empty((tp_size, flat.numel()))
    get_tp_group().all_gather_into_tensor(gathered, flat)
    return (
        gathered.view(tp_size, x.shape[0], x.shape[1])
        .permute(1, 0, 2)
        .reshape(x.shape[0], tp_size * x.shape[1])
        .contiguous()
    )


def _is_fp8_weight(x: torch.Tensor) -> bool:
    return x.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _is_plain_mla_bmm_weight(x: torch.Tensor) -> bool:
    return x.dtype in (torch.float16, torch.bfloat16, torch.float32)


def _q_b_proj_block_size(self: "DeepseekV2AttentionMLA") -> Optional[list[int]]:
    q_b_proj = self.q_b_proj
    block_size = getattr(q_b_proj, "weight_block_size", None)
    if block_size is None:
        quant_method = getattr(q_b_proj, "quant_method", None)
        block_size = getattr(quant_method, "weight_block_size", None)
        if block_size is None:
            quant_config = getattr(quant_method, "quant_config", None)
            block_size = getattr(quant_config, "weight_block_size", None)
    if block_size is None:
        return None
    return [int(block_size[0]), int(block_size[1])]


def _ue8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    return (
        (scale.contiguous().view(-1).to(torch.int32) << 23)
        .view(torch.float32)
        .view_as(scale)
    )


def _q_b_proj_scale_for_cache(self: "DeepseekV2AttentionMLA") -> Optional[torch.Tensor]:
    weight_scale_inv = getattr(self.q_b_proj, "weight_scale_inv", None)
    if weight_scale_inv is not None:
        return weight_scale_inv
    return getattr(self.q_b_proj, "weight_scale", None)


def _o_proj_block_size(self: "DeepseekV2AttentionMLA") -> Optional[list[int]]:
    o_proj = self.o_proj
    block_size = getattr(o_proj, "weight_block_size", None)
    if block_size is None:
        quant_method = getattr(o_proj, "quant_method", None)
        block_size = getattr(quant_method, "weight_block_size", None)
        if block_size is None:
            quant_config = getattr(quant_method, "quant_config", None)
            block_size = getattr(quant_config, "weight_block_size", None)
    if block_size is None:
        return None
    return [int(block_size[0]), int(block_size[1])]


def _o_proj_scale_for_cache(self: "DeepseekV2AttentionMLA") -> Optional[torch.Tensor]:
    weight_scale_inv = getattr(self.o_proj, "weight_scale_inv", None)
    if weight_scale_inv is not None:
        return weight_scale_inv
    return getattr(self.o_proj, "weight_scale", None)


def _orient_q_b_proj_weight_for_linear(
    self: "DeepseekV2AttentionMLA", weight: torch.Tensor
) -> torch.Tensor:
    q_lora_rank = self.q_lora_rank or getattr(self.q_b_proj, "input_size", None)
    if q_lora_rank is None:
        return weight
    if weight.dim() != 2:
        return weight
    if weight.shape[1] == q_lora_rank:
        return weight.contiguous()
    if weight.shape[0] == q_lora_rank:
        return weight.t().contiguous()
    return weight.contiguous()


def _q_b_proj_fp8_scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8 or getattr(scale, "format_ue8m0", False):
        return _ue8m0_to_float32(scale)
    return scale.to(torch.float32)


def _dequantize_q_b_proj_weight_for_linear(
    self: "DeepseekV2AttentionMLA",
) -> Optional[torch.Tensor]:
    q_b_proj = self.q_b_proj
    weight = q_b_proj.weight

    if _is_plain_mla_bmm_weight(weight):
        return _orient_q_b_proj_weight_for_linear(self, weight)
    if not _is_fp8_weight(weight):
        return None

    weight_scale_inv = getattr(q_b_proj, "weight_scale_inv", None)
    block_size = _q_b_proj_block_size(self)
    if weight_scale_inv is not None and block_size is not None:
        scale = _q_b_proj_fp8_scale_to_float(weight_scale_inv)
        return _orient_q_b_proj_weight_for_linear(
            self, block_quant_dequant(weight, scale, block_size, torch.bfloat16)
        )

    weight_scale = getattr(q_b_proj, "weight_scale", None)
    if weight_scale is None:
        return None

    scale = _q_b_proj_fp8_scale_to_float(weight_scale)
    weight_f32 = weight.to(torch.float32)
    q_lora_rank = self.q_lora_rank or getattr(q_b_proj, "input_size", None)

    if scale.numel() == 1:
        dequant = weight_f32 * scale.reshape(())
    elif (
        weight.dim() == 2
        and q_lora_rank is not None
        and weight.shape[0] == q_lora_rank
    ):
        dequant = weight_f32 * scale.reshape(1, -1)
    else:
        dequant = weight_f32 * scale.reshape(-1, 1)

    return _orient_q_b_proj_weight_for_linear(self, dequant.to(torch.bfloat16))


def _mla_only_dp_q_b_proj_move_skip_reason(
    self: "DeepseekV2AttentionMLA",
    mla_only_dp_move_mla_bmm: bool,
    is_capture_mode: bool,
) -> Optional[str]:
    if os.getenv("SGLANG_MLA_ONLY_DP_MOVE_Q_B_PROJ") != "1":
        return "disabled"
    if not mla_only_dp_move_mla_bmm:
        return "requires SGLANG_MLA_ONLY_DP_MOVE_MLA_BMM=1"
    if self.q_lora_rank is None:
        return "requires MLA q_lora_rank"
    if not hasattr(self, "q_b_proj") or self.q_b_proj is None:
        return "missing q_b_proj"

    weight = self.q_b_proj.weight
    if _is_plain_mla_bmm_weight(weight):
        pass
    elif _is_fp8_weight(weight):
        quant_method = getattr(self.q_b_proj, "quant_method", None)
        scale = getattr(self.q_b_proj, "weight_scale_inv", None)
        has_block = (
            scale is not None
            and _q_b_proj_block_size(self) is not None
            and getattr(quant_method, "block_quant", False)
        )
        if not has_block:
            return "requires block-wise fp8 q_b_proj"
        if getattr(quant_method, "w8a8_block_fp8_linear", None) is None:
            return "missing block-wise fp8 linear kernel"
        if scale.dtype != torch.float32 or getattr(scale, "format_ue8m0", False):
            return f"unsupported q_b_proj scale format {scale.dtype}"
    else:
        return f"unsupported q_b_proj weight dtype {weight.dtype}"

    if is_capture_mode and not _has_mla_only_dp_full_q_b_proj_cache(self):
        return "full q_b_proj weight is not cached before CUDA graph capture"
    return None


def _log_mla_only_dp_q_b_proj_move_skip_once(reason: str) -> None:
    if torch.compiler.is_compiling():
        return
    if os.getenv("SGLANG_MLA_ONLY_DP_DEBUG") != "1":
        return
    if reason in ("disabled",) or reason in _logged_q_b_proj_move_reasons:
        return
    _logged_q_b_proj_move_reasons.add(reason)
    logger.warning("mla-only-dp q_b_proj movement disabled: %s", reason)


def _mla_only_dp_full_weight_cache_key(self: "DeepseekV2AttentionMLA"):
    return (
        self.w_kc.data_ptr(),
        self.w_vc.data_ptr(),
        self.w_kc._version,
        self.w_vc._version,
        self.w_kc.dtype,
        self.w_vc.dtype,
        self.w_kc.device,
        self.w_vc.device,
    )


def _mla_only_dp_full_q_b_proj_cache_key(self: "DeepseekV2AttentionMLA"):
    weight = self.q_b_proj.weight
    bias = self.q_b_proj.bias
    scale = _q_b_proj_scale_for_cache(self)
    return (
        weight.data_ptr(),
        weight._version,
        weight.dtype,
        tuple(weight.shape),
        weight.device,
        None if scale is None else scale.data_ptr(),
        None if scale is None else scale._version,
        None if scale is None else scale.dtype,
        None if scale is None else tuple(scale.shape),
        None if scale is None else getattr(scale, "format_ue8m0", False),
        tuple(_q_b_proj_block_size(self) or ()),
        None if bias is None else bias.data_ptr(),
        None if bias is None else bias._version,
    )


def _mla_only_dp_full_o_proj_cache_key(self: "DeepseekV2AttentionMLA"):
    weight = self.o_proj.weight
    bias = self.o_proj.bias
    scale = _o_proj_scale_for_cache(self)
    return (
        weight.data_ptr(),
        weight._version,
        weight.dtype,
        tuple(weight.shape),
        weight.device,
        None if scale is None else scale.data_ptr(),
        None if scale is None else scale._version,
        None if scale is None else scale.dtype,
        None if scale is None else tuple(scale.shape),
        None if scale is None else getattr(scale, "format_ue8m0", False),
        tuple(_o_proj_block_size(self) or ()),
        None if bias is None else bias.data_ptr(),
        None if bias is None else bias._version,
    )


def _has_mla_only_dp_full_weight_cache(self: "DeepseekV2AttentionMLA") -> bool:
    cache = getattr(self, "_mla_only_dp_full_mla_bmm_weight_cache", None)
    return cache is not None and cache[0] == _mla_only_dp_full_weight_cache_key(self)


def _has_mla_only_dp_full_q_b_proj_cache(self: "DeepseekV2AttentionMLA") -> bool:
    cache = getattr(self, "_mla_only_dp_full_q_b_proj_weight_cache", None)
    return cache is not None and cache[0] == _mla_only_dp_full_q_b_proj_cache_key(self)


def _has_mla_only_dp_full_o_proj_cache(self: "DeepseekV2AttentionMLA") -> bool:
    cache = getattr(self, "_mla_only_dp_full_o_proj_weight_cache", None)
    return cache is not None and cache[0] == _mla_only_dp_full_o_proj_cache_key(self)


def _get_mla_only_dp_full_mla_bmm_weights(
    self: "DeepseekV2AttentionMLA",
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = _mla_only_dp_full_weight_cache_key(self)
    cache = getattr(self, "_mla_only_dp_full_mla_bmm_weight_cache", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2]

    full_w_kc = _tp_all_gather_head_weights(self.w_kc)
    full_w_vc = _tp_all_gather_head_weights(self.w_vc)
    self._mla_only_dp_full_mla_bmm_weight_cache = (
        cache_key,
        full_w_kc,
        full_w_vc,
    )
    return full_w_kc, full_w_vc


def _get_mla_only_dp_full_q_b_proj_weight(
    self: "DeepseekV2AttentionMLA",
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], str]:
    cache_key = _mla_only_dp_full_q_b_proj_cache_key(self)
    cache = getattr(self, "_mla_only_dp_full_q_b_proj_weight_cache", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2], cache[3], cache[4]

    weight = self.q_b_proj.weight
    bias = self.q_b_proj.bias
    if _is_fp8_weight(weight):
        scale = getattr(self.q_b_proj, "weight_scale_inv", None)
        block_size = _q_b_proj_block_size(self)
        if (
            scale is None
            or block_size is None
            or scale.dtype != torch.float32
            or getattr(scale, "format_ue8m0", False)
        ):
            raise RuntimeError(
                "SGLANG_MLA_ONLY_DP_MOVE_Q_B_PROJ=1 does not support "
                f"q_b_proj fp8 scale format {None if scale is None else scale.dtype}."
            )
        full_weight = _tp_all_gather_head_weights(weight)
        full_scale = _tp_all_gather_head_weights(scale)
        mode = "fp8_block"
    else:
        local_weight = _dequantize_q_b_proj_weight_for_linear(self)
        if local_weight is None:
            raise RuntimeError(
                "SGLANG_MLA_ONLY_DP_MOVE_Q_B_PROJ=1 does not support "
                f"q_b_proj weight dtype {self.q_b_proj.weight.dtype}."
            )
        full_weight = _tp_all_gather_head_weights(local_weight)
        full_scale = None
        mode = "plain"

    full_bias = None if bias is None else _tp_all_gather_head_weights(bias)
    self._mla_only_dp_full_q_b_proj_weight_cache = (
        cache_key,
        full_weight,
        full_scale,
        full_bias,
        mode,
    )
    return full_weight, full_scale, full_bias, mode


def _mla_only_dp_apply_full_q_b_proj(
    self: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
) -> torch.Tensor:
    full_weight, full_scale, full_bias, mode = _get_mla_only_dp_full_q_b_proj_weight(
        self
    )
    if mode == "fp8_block":
        return self.q_b_proj.quant_method.w8a8_block_fp8_linear(
            input=q,
            weight=full_weight,
            block_size=_q_b_proj_block_size(self),
            weight_scale=full_scale,
            input_scale=None,
            bias=full_bias,
        )
    return torch.nn.functional.linear(q, full_weight, full_bias)

def _get_mla_only_dp_full_o_proj_weight(
    self: "DeepseekV2AttentionMLA",
) -> tuple[torch.Tensor, Optional[torch.Tensor], str]:
    cache_key = _mla_only_dp_full_o_proj_cache_key(self)
    cache = getattr(self, "_mla_only_dp_full_o_proj_weight_cache", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2], cache[3]

    weight = self.o_proj.weight
    scale = getattr(self.o_proj, "weight_scale_inv", None)
    block_size = _o_proj_block_size(self)
    if (
        _is_fp8_weight(weight)
        and scale is not None
        and block_size is not None
        and scale.dtype == torch.float32
        and not getattr(scale, "format_ue8m0", False)
    ):
        full_weight = _tp_all_gather_row_parallel_input_weights(weight)
        full_scale = _tp_all_gather_row_parallel_input_weights(scale)
        mode = "fp8_block"
    elif _is_plain_mla_bmm_weight(weight):
        full_weight = _tp_all_gather_row_parallel_input_weights(weight)
        full_scale = None
        mode = "plain"
    else:
        raise RuntimeError(
            "SGLANG_MLA_ONLY_DP_MOVE_O_PROJ=1 does not support "
            f"o_proj weight dtype {weight.dtype} or scale format."
        )

    self._mla_only_dp_full_o_proj_weight_cache = (
        cache_key,
        full_weight,
        full_scale,
        mode,
    )
    return full_weight, full_scale, mode


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _mla_only_dp_o_proj_move_skip_reason(
    self: "DeepseekV2AttentionMLA",
    mla_only_dp_move_mla_bmm: bool,
    is_capture_mode: bool,
) -> Optional[str]:
    if os.getenv("SGLANG_MLA_ONLY_DP_MOVE_O_PROJ") != "1":
        return "disabled"
    max_layers = _env_int("SGLANG_MLA_ONLY_DP_MOVE_O_PROJ_MAX_LAYERS", 0)
    stride = _env_int("SGLANG_MLA_ONLY_DP_MOVE_O_PROJ_LAYER_STRIDE", 1)
    offset = _env_int("SGLANG_MLA_ONLY_DP_MOVE_O_PROJ_LAYER_OFFSET", 0)
    layer_id = getattr(self, "layer_id", None)
    if max_layers > 0 and layer_id is not None and layer_id >= max_layers:
        return f"layer {layer_id} is outside max direct o_proj layers {max_layers}"
    if stride > 1 and layer_id is not None and layer_id % stride != offset:
        return (
            f"layer {layer_id} does not match direct o_proj layer stride "
            f"{stride} offset {offset}"
        )
    if not mla_only_dp_move_mla_bmm:
        return "requires SGLANG_MLA_ONLY_DP_MOVE_MLA_BMM=1"
    if not hasattr(self, "o_proj") or self.o_proj is None:
        return "missing o_proj"
    if self.o_proj.bias is not None or self.o_proj.skip_bias_add:
        return "bias handling is not supported"

    weight = self.o_proj.weight
    if _is_plain_mla_bmm_weight(weight):
        pass
    elif _is_fp8_weight(weight):
        quant_method = getattr(self.o_proj, "quant_method", None)
        scale = getattr(self.o_proj, "weight_scale_inv", None)
        if not getattr(quant_method, "block_quant", False):
            return "requires block-wise fp8 o_proj"
        if getattr(quant_method, "w8a8_block_fp8_linear", None) is None:
            return "missing block-wise fp8 linear kernel"
        if scale is None:
            return "missing o_proj weight_scale_inv"
        if scale.dtype != torch.float32 or getattr(scale, "format_ue8m0", False):
            return f"unsupported o_proj scale format {scale.dtype}"
        if _o_proj_block_size(self) is None:
            return "missing o_proj block size"
    else:
        return f"unsupported o_proj weight dtype {weight.dtype}"

    if is_capture_mode and not _has_mla_only_dp_full_o_proj_cache(self):
        return "full o_proj weight is not cached before CUDA graph capture"
    return None


def _log_mla_only_dp_o_proj_move_skip_once(reason: str) -> None:
    if torch.compiler.is_compiling():
        return
    if os.getenv("SGLANG_MLA_ONLY_DP_DEBUG") != "1":
        return
    if reason in ("disabled",) or reason in _logged_o_proj_move_reasons:
        return
    _logged_o_proj_move_reasons.add(reason)
    logger.warning("mla-only-dp o_proj movement disabled: %s", reason)


def _use_mla_only_dp_move_o_proj(
    self: "DeepseekV2AttentionMLA",
    mla_only_dp_move_mla_bmm: bool,
    is_capture_mode: bool,
) -> bool:
    reason = _mla_only_dp_o_proj_move_skip_reason(
        self, mla_only_dp_move_mla_bmm, is_capture_mode
    )
    if reason is not None:
        _log_mla_only_dp_o_proj_move_skip_once(reason)
        return False
    return True


def _mla_only_dp_apply_full_o_proj(
    self: "DeepseekV2AttentionMLA",
    local_v: torch.Tensor,
) -> torch.Tensor:
    flat_v = local_v.flatten(1, 2).contiguous()
    full_weight, full_scale, mode = _get_mla_only_dp_full_o_proj_weight(self)
    if mode == "fp8_block":
        return self.o_proj.quant_method.w8a8_block_fp8_linear(
            input=flat_v,
            weight=full_weight,
            block_size=_o_proj_block_size(self),
            weight_scale=full_scale,
            input_scale=None,
            bias=None,
        )
    return torch.nn.functional.linear(flat_v, full_weight, None)


def maybe_init_mla_only_dp_weight_caches(model: torch.nn.Module) -> None:
    if os.getenv("SGLANG_MLA_ONLY_DP_MOVE_MLA_BMM") != "1":
        return

    num_mla_bmm = 0
    num_o_proj = 0
    num_q_b_proj = 0
    for module in model.modules():
        if not getattr(module, "mla_only_dp", False):
            continue
        if _use_mla_only_dp_move_mla_bmm(module, True, False):
            _get_mla_only_dp_full_mla_bmm_weights(module)
            num_mla_bmm += 1
        if _use_mla_only_dp_move_o_proj(module, True, False):
            _get_mla_only_dp_full_o_proj_weight(module)
            num_o_proj += 1
        if _use_mla_only_dp_move_q_b_proj(module, True, False):
            _get_mla_only_dp_full_q_b_proj_weight(module)
            num_q_b_proj += 1

    if (
        (num_mla_bmm or num_o_proj or num_q_b_proj)
        and get_tensor_model_parallel_rank() == 0
    ):
        logger.info(
            "Initialized MLA-only-DP cached weights before graph capture: "
            "mla_bmm_layers=%d, o_proj_layers=%d, q_b_proj_layers=%d",
            num_mla_bmm,
            num_o_proj,
            num_q_b_proj,
        )


def _use_mla_only_dp_move_mla_bmm(
    self: "DeepseekV2AttentionMLA",
    mla_only_dp_state: bool,
    is_capture_mode: bool,
) -> bool:
    if os.getenv("SGLANG_MLA_ONLY_DP_MOVE_MLA_BMM") != "1":
        return False
    if not mla_only_dp_state or self.use_deep_gemm_bmm or _is_hip:
        return False
    if self.w_kc is None or self.w_vc is None:
        return False
    if not _is_plain_mla_bmm_weight(self.w_kc) or not _is_plain_mla_bmm_weight(
        self.w_vc
    ):
        return False
    if self.w_scale_k is not None or self.w_scale_v is not None:
        return False
    if not isinstance(self.w_scale, (float, int)) or float(self.w_scale) != 1.0:
        return False
    if is_capture_mode and not _has_mla_only_dp_full_weight_cache(self):
        return False
    return True


def _use_mla_only_dp_move_q_b_proj(
    self: "DeepseekV2AttentionMLA",
    mla_only_dp_move_mla_bmm: bool,
    is_capture_mode: bool,
) -> bool:
    reason = _mla_only_dp_q_b_proj_move_skip_reason(
        self, mla_only_dp_move_mla_bmm, is_capture_mode
    )
    if reason is not None:
        _log_mla_only_dp_q_b_proj_move_skip_once(reason)
        return False
    return True


def _use_mla_only_dp_tp_q_b_proj_prefill_only(
    forward_batch: ForwardBatch,
    is_capture_mode: bool,
) -> bool:
    return (
        os.getenv("SGLANG_MLA_ONLY_DP_TP_Q_B_PROJ_PREFILL_ONLY") == "1"
        and not is_capture_mode
        and forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_mixed()
    )


def _mla_only_dp_stream_v_bmm_min_rows() -> int:
    return _env_int("SGLANG_MLA_ONLY_DP_STREAM_V_BMM_MIN_ROWS", 8192)


def _use_mla_only_dp_stream_v_bmm(local_rows: int) -> bool:
    return (
        os.getenv("SGLANG_MLA_ONLY_DP_DISABLE_STREAM_V_BMM") != "1"
        and local_rows >= _mla_only_dp_stream_v_bmm_min_rows()
        and not is_in_piecewise_cuda_graph()
    )


def _use_mla_only_dp_head_major_v_bmm(local_rows: int) -> bool:
    return (
        os.getenv("SGLANG_MLA_ONLY_DP_ENABLE_HEAD_MAJOR_V_BMM") == "1"
        and local_rows >= _mla_only_dp_stream_v_bmm_min_rows()
        and not is_in_piecewise_cuda_graph()
    )


def _mla_only_dp_bmm_v_to_tp_head_shard(
    attn_output: torch.Tensor,
    full_w_vc: torch.Tensor,
    forward_batch: ForwardBatch,
    tp_size: int,
    num_local_heads: int,
    v_head_dim: int,
) -> torch.Tensor:
    local_rows = attn_output.shape[0]
    send_tp_shards = torch.empty(
        (tp_size, num_local_heads, local_rows, v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )
    torch.bmm(
        attn_output.transpose(0, 1),
        full_w_vc,
        out=send_tp_shards.view(tp_size * num_local_heads, local_rows, v_head_dim),
    )
    return dp_packed_head_major_to_tp_head_shard(send_tp_shards, forward_batch)


def _mla_only_dp_bmm_v_to_tp_send_shards(
    attn_output: torch.Tensor,
    full_w_vc: torch.Tensor,
    tp_size: int,
    num_local_heads: int,
    v_head_dim: int,
) -> torch.Tensor:
    local_rows = attn_output.shape[0]
    send_tp_shards = torch.empty(
        (tp_size, local_rows, num_local_heads, v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )
    attn_by_head = attn_output.transpose(0, 1)
    for dst_rank in range(tp_size):
        head_start = dst_rank * num_local_heads
        head_end = head_start + num_local_heads
        torch.bmm(
            attn_by_head[head_start:head_end],
            full_w_vc[head_start:head_end],
            out=send_tp_shards[dst_rank].transpose(0, 1),
        )
    return send_tp_shards


if _is_cuda:
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

    from sglang.srt.utils.custom_op import register_custom_op

    # TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
    # Wrap bmm_fp8 as a custom op so torch.compile does not trace into
    # torch.cuda.current_blas_handle() (which returns a non-Tensor).
    @register_custom_op(mutates_args=["out"])
    def _bmm_fp8_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)
        return out


if _use_aiter:
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla


class DeepseekMLAForwardMixin:

    def init_mla_forward(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )

    def forward_absorb_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        is_capture_mode = get_is_capture_mode()
        mla_only_dp_state = _use_true_mla_only_dp(self, forward_batch)
        mla_only_dp_move_mla_bmm = _use_mla_only_dp_move_mla_bmm(
            self, mla_only_dp_state, is_capture_mode
        )
        mla_only_dp_move_q_b_proj = _use_mla_only_dp_move_q_b_proj(
            self, mla_only_dp_move_mla_bmm, is_capture_mode
        )
        if (
            mla_only_dp_state
            and mla_only_dp_move_q_b_proj
            and _use_mla_only_dp_tp_q_b_proj_prefill_only(
                forward_batch, is_capture_mode
            )
        ):
            mla_only_dp_move_q_b_proj = False
        if mla_only_dp_state and self.q_lora_rank is None:
            raise NotImplementedError(
                "true --mla-only-dp currently requires MLA q_lora_rank."
            )

        q_lora = None
        topk_indices = None
        mla_only_dp_q_shard = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        q = self.q_a_layernorm(q)
                        k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_nsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and not mla_only_dp_state
                and is_capture_mode
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                if not self.skip_topk or (self.is_nextn and prev_topk_indices is None):
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    topk_indices = prev_topk_indices
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                if mla_only_dp_move_q_b_proj:
                    q = _mla_only_dp_apply_full_q_b_proj(self, q).view(
                        q.shape[0], self.num_heads, self.qk_head_dim
                    )
                else:
                    q_proj_input = (
                        _dp_gather_rows_like(q, forward_batch)
                        if mla_only_dp_state
                        else q
                    )
                    q = self.q_b_proj(q_proj_input)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                if q_lora is not None:
                    if not self.skip_topk or (self.is_nextn and prev_topk_indices is None):
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = prev_topk_indices
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if mla_only_dp_move_mla_bmm:
            q_nope_out = q_nope
            if (
                os.getenv("SGLANG_MLA_ONLY_DP_ENABLE_FUSED_Q_TRANSFER") == "1"
                and mla_only_dp_state
                and not mla_only_dp_move_q_b_proj
                and q.is_contiguous()
            ):
                mla_only_dp_q_shard = q
        elif self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if (_use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn) or (
                    get_is_capture_mode() and self.w_kc.dtype == torch.float8_e4m3fnuz
                ):
                    # fp8 Triton kernel: always on gfx950,
                    # cudagraph-only on gfx942 (hides launch overhead)
                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        elif self.w_kc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                q_nope_out = torch.bmm(
                    q_nope.to(torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(torch.bfloat16) * self.w_scale,
                )
            else:
                # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                    q_nope.transpose(0, 1),
                    (
                        torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                q_nope_out = bmm_fp8(
                    q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
                )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        if not mla_only_dp_move_mla_bmm:
            q_nope_out = q_nope_out.transpose(0, 1)

        skip_rope_for_nsa_tilelang_fused = self._skip_rope_for_nsa_tilelang_fused()
        if (
            self.rotary_emb is not None
            and (
                mla_only_dp_state
                or not self._fuse_rope_for_trtllm_mla(forward_batch)
            )
            and (not skip_rope_for_nsa_tilelang_fused)
            and (not _use_aiter or not _is_gfx95_supported or self.use_nsa)
        ):
            if mla_only_dp_state:
                if mla_only_dp_move_q_b_proj:
                    q_positions = positions[: q_pe.shape[0]]
                else:
                    q_positions = _dp_gather_positions_once(positions, forward_batch)
                if q_pe.shape[0] > 0:
                    dummy_k_for_q = torch.empty_strided(
                        q_pe.shape, q_pe.stride(), dtype=q_pe.dtype, device=q_pe.device
                    )
                    q_pe, _ = self.rotary_emb(q_positions, q_pe, dummy_k_for_q)
                if k_pe.shape[0] > 0:
                    dummy_q_for_k = torch.empty_strided(
                        k_pe.shape, k_pe.stride(), dtype=k_pe.dtype, device=k_pe.device
                    )
                    _, k_pe = self.rotary_emb(
                        positions[: k_pe.shape[0]], dummy_q_for_k, k_pe
                    )
            else:
                q_pe, k_pe = self.rotary_emb(positions[: q_pe.shape[0]], q_pe, k_pe)

        if mla_only_dp_q_shard is not None and q_pe.shape[0] > 0:
            mla_only_dp_q_shard[..., self.qk_nope_head_dim :].copy_(q_pe)

        if nsa_use_prefill_cp(forward_batch):
            if mla_only_dp_state:
                raise NotImplementedError(
                    "true --mla-only-dp does not yet support NSA prefill CP."
                )
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
            mla_only_dp_state,
            mla_only_dp_move_mla_bmm,
            mla_only_dp_move_q_b_proj,
            mla_only_dp_q_shard,
        )

    def forward_absorb_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        mla_only_dp_state=False,
        mla_only_dp_move_mla_bmm=False,
        mla_only_dp_move_q_b_proj=False,
        mla_only_dp_q_shard=None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        is_capture_mode = get_is_capture_mode()
        save_kv_cache = True

        if mla_only_dp_state:
            if mla_only_dp_move_q_b_proj:
                local_q_nope, local_q_pe = q_nope_out, q_pe
            elif mla_only_dp_q_shard is not None:
                local_q_nope, local_q_pe = tp_fused_q_head_shard_to_dp_rows(
                    mla_only_dp_q_shard,
                    k_nope,
                    self.qk_nope_head_dim,
                    forward_batch,
                )
            else:
                local_q_nope, local_q_pe = tp_two_head_shards_to_dp_rows(
                    q_nope_out, q_pe, k_nope, forward_batch
                )
            if mla_only_dp_move_mla_bmm:
                full_w_kc, full_w_vc = _get_mla_only_dp_full_mla_bmm_weights(self)
                if local_q_nope.shape[0] == 0:
                    local_q_nope = local_q_nope.new_empty(
                        (0, self.num_heads, self.kv_lora_rank)
                    )
                else:
                    projected_q_nope = torch.empty(
                        (
                            local_q_nope.shape[0],
                            self.num_heads,
                            self.kv_lora_rank,
                        ),
                        dtype=local_q_nope.dtype,
                        device=local_q_nope.device,
                    )
                    torch.bmm(
                        local_q_nope.transpose(0, 1),
                        full_w_kc,
                        out=projected_q_nope.transpose(0, 1),
                    )
                    local_q_nope = projected_q_nope

            if local_q_nope.shape[0] == 0:
                attn_output = torch.empty_like(local_q_nope)
            else:
                attn_output = self.attn_mqa(
                    local_q_nope,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=local_q_pe,
                    k_rope=k_pe,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
            attn_output = attn_output.view(
                attn_output.shape[0], self.num_heads, self.kv_lora_rank
            )
            if mla_only_dp_move_mla_bmm:
                if _use_mla_only_dp_move_o_proj(
                    self, mla_only_dp_move_mla_bmm, is_capture_mode
                ):
                    if attn_output.shape[0] == 0:
                        local_v = attn_output.new_empty(
                            (0, self.num_heads, self.v_head_dim)
                        )
                    else:
                        local_v = torch.empty(
                            (attn_output.shape[0], self.num_heads, self.v_head_dim),
                            dtype=attn_output.dtype,
                            device=attn_output.device,
                        )
                        torch.bmm(
                            attn_output.transpose(0, 1),
                            full_w_vc,
                            out=local_v.transpose(0, 1),
                        )
                    output = _mla_only_dp_apply_full_o_proj(self, local_v)
                    return _return_output_with_topk(self, output, topk_indices)
                elif attn_output.shape[0] == 0:
                    local_v = attn_output.new_empty(
                        (0, self.num_heads, self.v_head_dim)
                    )
                    attn_bmm_output = dp_rows_to_tp_head_shard(
                        local_v, forward_batch, self.num_local_heads
                    ).flatten(1, 2)
                elif _use_mla_only_dp_head_major_v_bmm(attn_output.shape[0]):
                    attn_bmm_output = _mla_only_dp_bmm_v_to_tp_head_shard(
                        attn_output,
                        full_w_vc,
                        forward_batch,
                        get_tensor_model_parallel_world_size(),
                        self.num_local_heads,
                        self.v_head_dim,
                    ).flatten(1, 2)
                elif _use_mla_only_dp_stream_v_bmm(attn_output.shape[0]):
                    send_tp_shards = _mla_only_dp_bmm_v_to_tp_send_shards(
                        attn_output,
                        full_w_vc,
                        get_tensor_model_parallel_world_size(),
                        self.num_local_heads,
                        self.v_head_dim,
                    )
                    attn_bmm_output = dp_packed_rows_to_tp_head_shard(
                        send_tp_shards, forward_batch
                    ).flatten(1, 2)
                elif is_in_piecewise_cuda_graph():
                    local_v = (
                        torch.bmm(attn_output.transpose(0, 1), full_w_vc)
                        .transpose(0, 1)
                        .contiguous()
                    )
                    attn_bmm_output = dp_rows_to_tp_head_shard(
                        local_v, forward_batch, self.num_local_heads
                    ).flatten(1, 2)
                else:
                    local_v = torch.empty(
                        (attn_output.shape[0], self.num_heads, self.v_head_dim),
                        dtype=attn_output.dtype,
                        device=attn_output.device,
                    )
                    torch.bmm(
                        attn_output.transpose(0, 1),
                        full_w_vc,
                        out=local_v.transpose(0, 1),
                    )
                    attn_bmm_output = dp_rows_to_tp_head_shard(
                        local_v, forward_batch, self.num_local_heads
                    ).flatten(1, 2)
                output = _mla_only_dp_o_proj_to_dp_rows(
                    self, attn_bmm_output, k_nope, forward_batch
                )
                return _return_output_with_topk(self, output, topk_indices)
            else:
                attn_output = dp_rows_to_tp_head_shard(
                    attn_output,
                    forward_batch,
                    self.num_local_heads,
                )
        elif self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            if self._skip_rope_for_nsa_tilelang_fused() and self.rotary_emb is not None:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )
                q_cat, _, k_pe_fused, _ = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )
                q_nope_fused = q_cat[..., : self.kv_lora_rank]
                q_pe_fused = q_cat[..., self.kv_lora_rank :]
                save_kv_cache = False
                if llama_4_scaling is not None:
                    q_nope_fused *= llama_4_scaling
                attn_output = self.attn_mqa(
                    q_nope_fused,
                    None,
                    None,
                    forward_batch,
                    q_rope=q_pe_fused,
                    k_rope=k_pe_fused,
                    save_kv_cache=save_kv_cache,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
            else:
                extra_args = {}
                if self._fuse_rope_for_trtllm_mla(forward_batch):
                    extra_args = {
                        "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                        "is_neox": self.rotary_emb.is_neox_style,
                        "llama_4_scaling": llama_4_scaling,
                    }
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
                )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        if mla_only_dp_state:
            output = _mla_only_dp_o_proj_to_dp_rows(
                self, attn_bmm_output, k_nope, forward_batch
            )
        else:
            output, _ = self.o_proj(attn_bmm_output)

        return _return_output_with_topk(self, output, topk_indices)

    def _fuse_rope_for_trtllm_mla(
        self: DeepseekV2AttentionMLA, forward_batch: ForwardBatch
    ) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        if self.current_attention_backend == "nsa":
            return (
                get_global_server_args().nsa_decode_backend == "trtllm"
                or get_global_server_args().nsa_prefill_backend == "trtllm"
            ) and forward_batch.attn_backend.kv_cache_dtype == torch.float8_e4m3fn

        return (
            self.current_attention_backend == "trtllm_mla"
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
        )

    def _skip_rope_for_nsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
        """
        Check if we should skip rope and use fused rope+cache path for TileLang NSA on gfx95.
        """
        server_args = get_global_server_args()
        return (
            _use_aiter_gfx95
            and self.current_attention_backend == "nsa"
            and (
                server_args.nsa_decode_backend == "tilelang"
                or server_args.nsa_prefill_backend == "tilelang"
            )
        )
