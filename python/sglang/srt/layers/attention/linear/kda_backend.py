from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kda_cp_utils import (
    KDAPrefillContextParallelMetadata,
    is_kda_prefill_cp_plain_split,
    kda_cp_owner_of_global_token,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.dp_attention import get_attention_cp_group
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

# KDA always uses the triton causal_conv1d_fn (no CUDA override).
# Only causal_conv1d_update needs platform-specific overrides for decode.
if is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu

    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_update_cpu

    causal_conv1d_update = causal_conv1d_update_cpu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class KDAKernelDispatcher:
    """Dispatches KDA kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonKDAKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            self.decode_kernel = CuteDSLKDAKernel()
        elif decode_backend.is_flash_kda():
            # FlashKDA has no decode kernel — fall back to Triton so that
            # `--linear-attn-backend flash_kda` (which fans out to both modes)
            # still works end-to-end.
            self.decode_kernel = triton_kernel
            rank0_log(
                "KDA decode backend 'flash_kda' has no decode kernel; "
                "falling back to TritonKDAKernel for decode."
            )
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA supports 'triton', 'cutedsl', or 'flash_kda' (decode "
                "auto-falls back to triton)."
            )

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_flash_kda():
            if not is_cuda():
                raise ValueError("KDA FlashKDA backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_flash import (
                FlashKDAKernel,
            )

            self.extend_kernel = FlashKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA supports 'triton' or 'flash_kda' for prefill."
            )

        # KDA verify kernel is only implemented in TritonKDAKernel
        # (FlashKDA/CuteDSL will raise NotImplementedError). Hard-bind
        # so verify works even when extend_backend is flash_kda.
        self.verify_kernel = triton_kernel
        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__}"
        )

    @property
    def extend_applies_gate_internally(self) -> bool:
        return self.extend_kernel.applies_gate_internally

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        safe_gate_lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            safe_gate_lower_bound=safe_gate_lower_bound,
            **kwargs,
        )


class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0][0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()

        # FlashKDA applies sigmoid(beta) internally with no post-sigmoid scale
        # hook, so it can't honour allow_neg_eigval=True (which requires
        # beta_scale=2.0). Fall back to Triton for extend in that case — same
        # pattern as the FlashKDA-decode → Triton fallback inside the
        # dispatcher. Read from hf_config so the rule is model-agnostic.
        allow_neg_eigval = bool(
            getattr(model_runner.model_config.hf_config, "linear_allow_neg_eigval", False)
        )
        if prefill_backend.is_flash_kda() and allow_neg_eigval:
            rank0_log(
                "KDA prefill backend 'flash_kda' is incompatible with "
                "linear_allow_neg_eigval=True; falling back to TritonKDAKernel "
                "for extend."
            )
            prefill_backend = LinearAttnKernelBackend.TRITON

        self.kernel_dispatcher = KDAKernelDispatcher(decode_backend, prefill_backend)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

    def _use_kda_prefill_cp(self, forward_batch: ForwardBatch) -> bool:
        return (
            is_kda_prefill_cp_plain_split()
            and forward_batch.kda_cp_metadata is not None
            and forward_batch.forward_mode.is_context_parallel_extend()
        )

    def _gather_full_kda_state(
        self,
        state: torch.Tensor,
        cache_indices: torch.Tensor,
        *,
        shard_dim: int,
        expected_dim: int,
    ) -> torch.Tensor:
        selected = state[cache_indices].contiguous()
        if selected.shape[shard_dim] == expected_dim:
            return selected

        cp_group = get_attention_cp_group()
        if (
            cp_group.world_size > 1
            and selected.shape[shard_dim] * cp_group.world_size == expected_dim
        ):
            return cp_group.all_gather(selected, dim=shard_dim)

        raise RuntimeError(
            "KDA-CP expected Prefix Cache state to be either full-head or "
            "sharded across the attention CP group, but got "
            f"shape={tuple(selected.shape)}, shard_dim={shard_dim}, "
            f"expected_dim={expected_dim}, cp_size={cp_group.world_size}."
        )

    def _prepare_kda_cp_states(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        metadata: KDAPrefillContextParallelMetadata,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_segments = len(metadata.local_seq_lens_cpu)
        local_cache_indices = torch.arange(
            num_segments, dtype=cache_indices.dtype, device=cache_indices.device
        )
        has_initial_state = torch.ones(
            num_segments, dtype=torch.bool, device=mixed_qkv.device
        )

        full_conv_by_req = self._gather_full_kda_state(
            conv_states,
            cache_indices,
            shard_dim=1,
            expected_dim=mixed_qkv.shape[-1],
        )
        full_ssm_by_req = self._gather_full_kda_state(
            ssm_states,
            cache_indices,
            shard_dim=1,
            expected_dim=layer.num_q_heads,
        )

        local_conv_states = mixed_qkv.new_zeros(
            (num_segments, mixed_qkv.shape[-1], conv_states.shape[-1])
        )
        local_ssm_states = ssm_states.new_zeros(
            (num_segments,) + tuple(full_ssm_by_req.shape[1:])
        )

        prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
        for local_idx, req_idx in enumerate(metadata.local_req_indices_cpu):
            if (
                metadata.local_req_extend_offsets_cpu[local_idx] == 0
                and prefix_lens_cpu[req_idx] > 0
            ):
                local_conv_states[local_idx].copy_(full_conv_by_req[req_idx])
                local_ssm_states[local_idx].copy_(full_ssm_by_req[req_idx])

        recv_idx = self._kda_cp_recv_segment_index(metadata)
        if recv_idx is not None:
            cp_group = get_attention_cp_group()
            prev_rank = kda_cp_owner_of_global_token(
                metadata.local_segment_global_starts_cpu[recv_idx] - 1,
                metadata.total_tokens,
                metadata.cp_size,
            )
            torch.distributed.recv(
                local_conv_states[recv_idx],
                src=cp_group.ranks[prev_rank],
                group=cp_group.device_group,
            )
            torch.distributed.recv(
                local_ssm_states[recv_idx],
                src=cp_group.ranks[prev_rank],
                group=cp_group.device_group,
            )

        return (
            local_conv_states,
            local_ssm_states,
            local_cache_indices,
            has_initial_state,
        )

    def _kda_cp_recv_segment_index(
        self, metadata: KDAPrefillContextParallelMetadata
    ) -> Optional[int]:
        for local_idx, (seg_start, req_start) in enumerate(
            zip(
                metadata.local_segment_global_starts_cpu,
                metadata.local_req_global_starts_cpu,
            )
        ):
            if seg_start > req_start:
                return local_idx
        return None

    def _kda_cp_send_segment_index(
        self, metadata: KDAPrefillContextParallelMetadata
    ) -> Optional[int]:
        for local_idx in range(len(metadata.local_seq_lens_cpu) - 1, -1, -1):
            if (
                metadata.local_segment_global_ends_cpu[local_idx]
                < metadata.local_req_global_ends_cpu[local_idx]
            ):
                return local_idx
        return None

    def _send_kda_cp_boundary_state(
        self,
        metadata: KDAPrefillContextParallelMetadata,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
    ) -> None:
        send_idx = self._kda_cp_send_segment_index(metadata)
        if send_idx is None:
            return

        cp_group = get_attention_cp_group()
        next_rank = kda_cp_owner_of_global_token(
            metadata.local_segment_global_ends_cpu[send_idx],
            metadata.total_tokens,
            metadata.cp_size,
        )
        torch.distributed.send(
            conv_states[send_idx].contiguous(),
            dst=cp_group.ranks[next_rank],
            group=cp_group.device_group,
        )
        torch.distributed.send(
            ssm_states[send_idx].contiguous(),
            dst=cp_group.ranks[next_rank],
            group=cp_group.device_group,
        )

    def _track_kda_cp_state_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        metadata: KDAPrefillContextParallelMetadata,
        runtime_conv_states: torch.Tensor,
        runtime_ssm_states: torch.Tensor,
        persistent_conv_states: torch.Tensor,
        persistent_ssm_states: torch.Tensor,
    ) -> None:
        if (
            forward_batch.mamba_track_mask is None
            or not forward_batch.mamba_track_mask.any()
        ):
            return

        cp_group = get_attention_cp_group()
        cp_rank = cp_group.rank_in_group
        req_starts = [0]
        for seq_len in forward_batch.extend_seq_lens_cpu[:-1]:
            req_starts.append(req_starts[-1] + int(seq_len))

        track_mask_cpu = forward_batch.mamba_track_mask.cpu().tolist()
        track_seqlens_cpu = forward_batch.mamba_track_seqlens.cpu().tolist()
        track_indices = forward_batch.mamba_track_indices
        prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

        conv_full_dim = layer.q_dim + layer.k_dim + layer.v_dim
        conv_state_len = persistent_conv_states.shape[-1]
        ssm_full_shape = (layer.num_q_heads, layer.head_v_dim, layer.head_k_dim)

        for req_idx, should_track in enumerate(track_mask_cpu):
            if not should_track:
                continue

            current_chunk_track_len = (
                int(track_seqlens_cpu[req_idx]) - int(prefix_lens_cpu[req_idx])
            )
            if current_chunk_track_len <= 0:
                continue

            req_start = req_starts[req_idx]
            req_end = req_start + int(forward_batch.extend_seq_lens_cpu[req_idx])
            boundary_end = req_start + current_chunk_track_len
            if boundary_end > req_end:
                continue

            owner_rank = kda_cp_owner_of_global_token(
                boundary_end - 1, metadata.total_tokens, metadata.cp_size
            )
            _, owner_end = self._plain_split_bounds_for_rank(
                metadata.total_tokens, owner_rank, metadata.cp_size
            )
            owner_segment_end = min(owner_end, req_end)

            # The minimal write-back path handles boundaries whose state is the
            # final state of one KDA-CP local segment. Branching points inside a
            # segment still need h-index based extraction, mirroring the
            # non-CP path's _init_track_ssm_indices logic.
            if boundary_end != owner_segment_end:
                continue

            full_conv_state = runtime_conv_states.new_empty(
                (conv_full_dim, conv_state_len)
            )
            full_ssm_state = runtime_ssm_states.new_empty(ssm_full_shape)

            if cp_rank == owner_rank:
                local_idx = self._find_kda_cp_segment(
                    metadata, req_idx=req_idx, segment_end=boundary_end
                )
                if local_idx is None:
                    raise RuntimeError(
                        "KDA-CP state owner could not find its local segment "
                        f"for req_idx={req_idx}, boundary_end={boundary_end}."
                    )
                full_conv_state.copy_(runtime_conv_states[local_idx])
                full_ssm_state.copy_(runtime_ssm_states[local_idx])

            cp_group.broadcast(full_conv_state, src=owner_rank)
            cp_group.broadcast(full_ssm_state, src=owner_rank)

            dst = track_indices[req_idx]
            conv_shard = self._local_state_shard(
                full_conv_state,
                local_dim=persistent_conv_states.shape[1],
                shard_dim=0,
                cp_rank=cp_rank,
                cp_size=cp_group.world_size,
            )
            ssm_shard = self._local_state_shard(
                full_ssm_state,
                local_dim=persistent_ssm_states.shape[1],
                shard_dim=0,
                cp_rank=cp_rank,
                cp_size=cp_group.world_size,
            )
            persistent_conv_states[dst].copy_(
                conv_shard.to(persistent_conv_states.dtype, copy=False)
            )
            persistent_ssm_states[dst].copy_(
                ssm_shard.to(persistent_ssm_states.dtype, copy=False)
            )

    def _writeback_kda_cp_final_states(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        metadata: KDAPrefillContextParallelMetadata,
        runtime_conv_states: torch.Tensor,
        runtime_ssm_states: torch.Tensor,
        persistent_cache_indices: torch.Tensor,
        persistent_conv_states: torch.Tensor,
        persistent_ssm_states: torch.Tensor,
    ) -> None:
        cp_group = get_attention_cp_group()
        cp_rank = cp_group.rank_in_group
        req_starts = [0]
        for seq_len in forward_batch.extend_seq_lens_cpu[:-1]:
            req_starts.append(req_starts[-1] + int(seq_len))

        conv_full_dim = layer.q_dim + layer.k_dim + layer.v_dim
        conv_state_len = persistent_conv_states.shape[-1]
        ssm_full_shape = (layer.num_q_heads, layer.head_v_dim, layer.head_k_dim)

        for req_idx, req_start in enumerate(req_starts):
            req_end = req_start + int(forward_batch.extend_seq_lens_cpu[req_idx])
            if req_end <= req_start:
                continue

            owner_rank = kda_cp_owner_of_global_token(
                req_end - 1, metadata.total_tokens, metadata.cp_size
            )
            full_conv_state = runtime_conv_states.new_empty(
                (conv_full_dim, conv_state_len)
            )
            full_ssm_state = runtime_ssm_states.new_empty(ssm_full_shape)

            if cp_rank == owner_rank:
                local_idx = self._find_kda_cp_segment(
                    metadata, req_idx=req_idx, segment_end=req_end
                )
                if local_idx is None:
                    raise RuntimeError(
                        "KDA-CP final-state owner could not find its local "
                        f"segment for req_idx={req_idx}, req_end={req_end}."
                    )
                full_conv_state.copy_(runtime_conv_states[local_idx])
                full_ssm_state.copy_(runtime_ssm_states[local_idx])

            cp_group.broadcast(full_conv_state, src=owner_rank)
            cp_group.broadcast(full_ssm_state, src=owner_rank)

            dst = persistent_cache_indices[req_idx]
            conv_shard = self._local_state_shard(
                full_conv_state,
                local_dim=persistent_conv_states.shape[1],
                shard_dim=0,
                cp_rank=cp_rank,
                cp_size=cp_group.world_size,
            )
            ssm_shard = self._local_state_shard(
                full_ssm_state,
                local_dim=persistent_ssm_states.shape[1],
                shard_dim=0,
                cp_rank=cp_rank,
                cp_size=cp_group.world_size,
            )
            persistent_conv_states[dst].copy_(
                conv_shard.to(persistent_conv_states.dtype, copy=False)
            )
            persistent_ssm_states[dst].copy_(
                ssm_shard.to(persistent_ssm_states.dtype, copy=False)
            )

    def _scatter_kda_cp_runtime_states_to_persistent(
        self,
        cache_indices: torch.Tensor,
        runtime_conv_states: torch.Tensor,
        runtime_ssm_states: torch.Tensor,
        persistent_conv_states: torch.Tensor,
        persistent_ssm_states: torch.Tensor,
    ) -> None:
        cp_group = get_attention_cp_group()
        conv_shard = self._local_state_shard(
            runtime_conv_states,
            local_dim=persistent_conv_states.shape[1],
            shard_dim=1,
            cp_rank=cp_group.rank_in_group,
            cp_size=cp_group.world_size,
        )
        ssm_shard = self._local_state_shard(
            runtime_ssm_states,
            local_dim=persistent_ssm_states.shape[1],
            shard_dim=1,
            cp_rank=cp_group.rank_in_group,
            cp_size=cp_group.world_size,
        )
        persistent_conv_states[cache_indices].copy_(
            conv_shard.to(persistent_conv_states.dtype, copy=False)
        )
        persistent_ssm_states[cache_indices].copy_(
            ssm_shard.to(persistent_ssm_states.dtype, copy=False)
        )

    @staticmethod
    def _plain_split_bounds_for_rank(
        total_tokens: int, cp_rank: int, cp_size: int
    ) -> tuple[int, int]:
        base = total_tokens // cp_size
        rem = total_tokens % cp_size
        start = cp_rank * base + min(cp_rank, rem)
        end = start + base + (1 if cp_rank < rem else 0)
        return start, end

    @staticmethod
    def _find_kda_cp_segment(
        metadata: KDAPrefillContextParallelMetadata,
        *,
        req_idx: int,
        segment_end: int,
    ) -> Optional[int]:
        for local_idx, (local_req_idx, local_segment_end) in enumerate(
            zip(
                metadata.local_req_indices_cpu,
                metadata.local_segment_global_ends_cpu,
            )
        ):
            if local_req_idx == req_idx and local_segment_end == segment_end:
                return local_idx
        return None

    @staticmethod
    def _local_state_shard(
        full_state: torch.Tensor,
        *,
        local_dim: int,
        shard_dim: int,
        cp_rank: int,
        cp_size: int,
    ) -> torch.Tensor:
        if full_state.shape[shard_dim] == local_dim:
            return full_state
        if full_state.shape[shard_dim] != local_dim * cp_size:
            raise RuntimeError(
                "KDA-CP state shard shape mismatch: "
                f"full_shape={tuple(full_state.shape)}, local_dim={local_dim}, "
                f"shard_dim={shard_dim}, cp_size={cp_size}."
            )
        start = cp_rank * local_dim
        return full_state.narrow(shard_dim, start, local_dim).contiguous()

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        core_attn_out = self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            beta_scale=getattr(layer, "beta_scale", 1.0),
            safe_gate=getattr(layer, "safe_gate", False),
            safe_gate_lower_bound=getattr(layer, "safe_gate_lower_bound", -5.0),
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        # Under input_scattered or dp-attention, the model passes [N_pad, ...]
        # tensors but query_start_loc still describes only [0, n_valid). The
        # Triton kernels (causal_conv1d / chunk_kda) only schedule thread blocks
        # for rows covered by query_start_loc, so output rows [n_valid, N_pad)
        # are never written and keep whatever bit pattern torch.empty() left in
        # GPU memory — often NaN/Inf. Strip to valid rows here, run the kernels
        # on a smaller tensor, then re-pad core_attn_out with zeros so the
        # caller sees the [N_pad, ...] shape it expects.
        n_total = mixed_qkv.shape[0]
        n_valid = forward_batch.extend_num_valid_tokens
        needs_repad = n_valid is not None and n_valid < n_total
        if needs_repad:
            mixed_qkv = mixed_qkv[:n_valid]
            a = a[:, :n_valid]
            b = b[:, :n_valid]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata
        query_start_loc = forward_metadata.query_start_loc
        seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        persistent_cache_indices = forward_metadata.mamba_cache_indices
        cache_indices = persistent_cache_indices

        metadata = None
        kda_cp_active = False
        kda_cp_replicated_state = False
        kda_cp_full_state_fallback = False
        if not is_target_verify:
            kda_cp_active = self._use_kda_prefill_cp(forward_batch)
            kda_cp_replicated_state = is_kda_prefill_cp_plain_split()
        if kda_cp_active:
            metadata = forward_batch.kda_cp_metadata
            query_start_loc = metadata.local_query_start_loc
            seq_lens_cpu = metadata.local_seq_lens_cpu

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        persistent_conv_states = mamba_cache_params.conv[0]
        persistent_ssm_states = mamba_cache_params.temporal
        conv_states = persistent_conv_states
        ssm_states = persistent_ssm_states

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            draft_token_num = forward_batch.spec_info.draft_token_num
            seq_len = mixed_qkv.shape[0]
            batch_size = seq_len // draft_token_num
            intermediate_state_indices = torch.arange(
                batch_size, dtype=torch.int32, device=cache_indices.device
            )
            mixed_qkv = mixed_qkv.view(batch_size, draft_token_num, -1).transpose(1, 2)
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                activation="silu",
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=mamba_cache_params.intermediate_conv_window[0],
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=forward_metadata.retrieve_next_token,
                retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
                retrieve_parent_token=forward_metadata.retrieve_parent_token,
            ).transpose(1, 2).reshape(seq_len, -1)
            q, k, v = mixed_qkv.split(splits, dim=-1)
        else:
            if kda_cp_active:
                assert metadata is not None
                (
                    conv_states,
                    ssm_states,
                    cache_indices,
                    has_initial_state,
                ) = self._prepare_kda_cp_states(
                    layer,
                    forward_batch,
                    metadata,
                    mixed_qkv,
                    conv_states,
                    ssm_states,
                    cache_indices,
                )
            elif kda_cp_replicated_state:
                kda_cp_full_state_fallback = True
                local_cache_indices = torch.arange(
                    cache_indices.shape[0],
                    dtype=cache_indices.dtype,
                    device=cache_indices.device,
                )
                conv_states = self._gather_full_kda_state(
                    persistent_conv_states,
                    cache_indices,
                    shard_dim=1,
                    expected_dim=mixed_qkv.shape[-1],
                )
                ssm_states = self._gather_full_kda_state(
                    persistent_ssm_states,
                    cache_indices,
                    shard_dim=1,
                    expected_dim=layer.num_q_heads,
                )
                cache_indices = local_cache_indices
                has_initial_state = forward_batch.extend_prefix_lens > 0
            else:
                has_initial_state = forward_batch.extend_prefix_lens > 0

            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_metadata.has_mamba_track_mask
                and not kda_cp_active
                and not kda_cp_full_state_fallback
            ):
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track.to(conv_states.dtype, copy=False)
                )

            q, k, v = mixed_qkv.split(splits, dim=0)
            q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
                splits, dim=0
            )
            q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
            if layer.bias is not None:
                q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None

            q = causal_conv1d_fn(
                q,
                q_conv_weight,
                q_bias,
                activation="silu",
                conv_states=q_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k,
                k_conv_weight,
                k_bias,
                activation="silu",
                conv_states=k_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v,
                v_conv_weight,
                v_bias,
                activation="silu",
                conv_states=v_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
            ).transpose(0, 1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=mamba_cache_params.intermediate_ssm,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=draft_token_num,
                retrieve_parent_token=forward_metadata.retrieve_parent_token,
                beta_scale=getattr(layer, "beta_scale", 1.0),
                safe_gate=getattr(layer, "safe_gate", False),
                safe_gate_lower_bound=getattr(layer, "safe_gate_lower_bound", -5.0),
            )
        else:
            # Apply KDA gate activation here unless the extend kernel does it
            # internally (e.g. FlashKDA). Model always passes raw g (a) and raw
            # beta logits (b); backend decides based on the dispatched kernel.
            if not self.kernel_dispatcher.extend_applies_gate_internally:
                from sglang.srt.layers.attention.fla.kda import fused_kda_gate

                a, b = fused_kda_gate(
                    a,
                    layer.A_log,
                    layer.head_k_dim,
                    g_bias=layer.dt_bias,
                    safe_gate=getattr(layer, "safe_gate", False),
                    lower_bound=getattr(layer, "safe_gate_lower_bound", -5.0),
                    beta=b,
                    beta_scale=getattr(layer, "beta_scale", 1.0),
                )

            core_attn_out, h = self.kernel_dispatcher.extend(
                q=q,
                k=k,
                v=v,
                g=a,
                beta=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                # Extra plumbing for FlashKDA; TritonKDAKernel ignores these via **kwargs.
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                seq_lens_cpu=seq_lens_cpu,
                safe_gate_lower_bound=getattr(layer, "safe_gate_lower_bound", -5.0),
            )

            if kda_cp_active:
                assert metadata is not None
                self._send_kda_cp_boundary_state(metadata, conv_states, ssm_states)
                self._writeback_kda_cp_final_states(
                    layer,
                    forward_batch,
                    metadata,
                    conv_states,
                    ssm_states,
                    persistent_cache_indices,
                    persistent_conv_states,
                    persistent_ssm_states,
                )
                self._track_kda_cp_state_extend(
                    layer,
                    forward_batch,
                    metadata,
                    conv_states,
                    ssm_states,
                    persistent_conv_states,
                    persistent_ssm_states,
                )
            elif kda_cp_full_state_fallback:
                self._scatter_kda_cp_runtime_states_to_persistent(
                    persistent_cache_indices,
                    conv_states,
                    ssm_states,
                    persistent_conv_states,
                    persistent_ssm_states,
                )
            elif h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        if needs_repad:
            # core_attn_out comes back from chunk_kda as [1, n_valid, h, d]
            # (token dim is dim 1 due to the unsqueeze(0) above). Re-pad along
            # the token dim so the caller's `.squeeze(0).flatten(-2)` gives
            # [n_total, h*d] as expected.
            full = core_attn_out.new_zeros(
                (
                    core_attn_out.shape[0],
                    n_total,
                    *core_attn_out.shape[2:],
                )
            )
            full[:, :n_valid] = core_attn_out
            core_attn_out = full

        return core_attn_out
