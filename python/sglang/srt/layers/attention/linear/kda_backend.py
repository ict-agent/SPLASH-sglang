from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kda_cp_utils import (
    KDAPrefillContextParallelMetadata,
    build_kda_fla_cp_context,
    is_kda_prefill_cp_plain_split,
    kda_cp_continuation_segment_index,
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
        # KDA prefill context-parallel is only implemented for the Triton
        # chunk_kda path (all-gather + merge). Bind it independently of the
        # configured extend backend so CP works even when extend=flash_kda.
        self.cp_kernel = triton_kernel
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

    def extend_cp(
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
        cp_context,
        continuation_h0=None,
        continuation_index=None,
        **kwargs,
    ) -> tuple:
        return self.cp_kernel.extend_cp(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            cp_context=cp_context,
            continuation_h0=continuation_h0,
            continuation_index=continuation_index,
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
        cp_context,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> tuple:
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

        # Prefix-cache seed for request-start segments (offset 0 with a prefix).
        # The continuation segment (offset > 0) is intentionally skipped: its
        # conv history comes from the cross-rank halo below and its SSM state
        # from the all-gather + merge inside chunk_kda_cp.
        prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
        for local_idx, req_idx in enumerate(metadata.local_req_indices_cpu):
            if (
                metadata.local_req_extend_offsets_cpu[local_idx] == 0
                and prefix_lens_cpu[req_idx] > 0
            ):
                local_conv_states[local_idx].copy_(full_conv_by_req[req_idx])
                local_ssm_states[local_idx].copy_(full_ssm_by_req[req_idx])

        # Cross-rank conv halo: all-gather each rank's W-1 tail tokens and use
        # the previous rank's tail as the continuation segment's conv history
        # (replaces the serial dist.recv of conv_states).
        continuation_index = kda_cp_continuation_segment_index(metadata)
        self._fill_kda_cp_conv_halo(
            cp_context, mixed_qkv, local_conv_states, continuation_index
        )

        # Prefix seed for the continued sequence's SSM merge chain (the merge is
        # done inside chunk_kda_cp; the seed is the whole sequence's prefix
        # state, identical on every rank thanks to _gather_full_kda_state).
        continuation_h0 = None
        if continuation_index is not None:
            cont_req_idx = metadata.local_req_indices_cpu[continuation_index]
            if prefix_lens_cpu[cont_req_idx] > 0:
                continuation_h0 = (
                    full_ssm_by_req[cont_req_idx].to(torch.float32).contiguous()
                )

        return (
            local_conv_states,
            local_ssm_states,
            local_cache_indices,
            has_initial_state,
            continuation_index,
            continuation_h0,
        )

    def _fill_kda_cp_conv_halo(
        self,
        cp_context,
        mixed_qkv: torch.Tensor,
        local_conv_states: torch.Tensor,
        continuation_index: Optional[int],
    ) -> None:
        """All-gather each rank's last ``W-1`` tokens and load the previous
        rank's tail as the continuation segment's conv1d history. Every rank
        must call the collective, even when it has no continuation segment."""
        from sglang.srt.layers.attention.fla.cp import all_gather_into_tensor

        state_len = local_conv_states.shape[-1]  # W - 1
        n = mixed_qkv.shape[0]
        tail = mixed_qkv.new_zeros(state_len, mixed_qkv.shape[-1])
        take = min(state_len, n)
        if take > 0:
            tail[state_len - take :] = mixed_qkv[n - take :]
        gathered, _ = all_gather_into_tensor(
            tail.contiguous(), group=cp_context.group
        )  # [cp_size, state_len, D]

        if continuation_index is None:
            return
        prev = get_attention_cp_group().rank_in_group - 1
        if prev < 0:
            return
        valid = min(state_len, int(cp_context.pre_num_conv_tokens or 0))
        if valid <= 0:
            return
        heads = gathered[prev]  # [state_len, D]
        local_conv_states[continuation_index][:, state_len - valid :] = (
            heads[state_len - valid :].transpose(0, 1).to(local_conv_states.dtype)
        )

    def _ensure_kda_cp_cpu_meta(
        self,
        forward_batch: ForwardBatch,
        metadata: KDAPrefillContextParallelMetadata,
    ) -> None:
        """Populate the per-forward CPU metadata cache on ``metadata`` the first
        time any KDA layer needs it, so the same lists (and the track_mask /
        track_seqlens D2H copies) are not rebuilt on every layer."""
        if metadata.cpu_req_starts is None:
            req_starts = [0]
            for seq_len in forward_batch.extend_seq_lens_cpu[:-1]:
                req_starts.append(req_starts[-1] + int(seq_len))
            metadata.cpu_req_starts = req_starts
        if metadata.cpu_track_mask is None and forward_batch.mamba_track_mask is not None:
            metadata.cpu_track_mask = forward_batch.mamba_track_mask.cpu().tolist()
            metadata.cpu_track_seqlens = (
                forward_batch.mamba_track_seqlens.cpu().tolist()
            )

    def _track_kda_cp_state_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        metadata: KDAPrefillContextParallelMetadata,
        runtime_conv_states: torch.Tensor,
        runtime_ssm_states: torch.Tensor,
        runtime_h: Optional[torch.Tensor],
        raw_mixed_qkv: torch.Tensor,
        track_q: torch.Tensor,
        track_k: torch.Tensor,
        track_v: torch.Tensor,
        track_g: torch.Tensor,
        track_beta: torch.Tensor,
        persistent_conv_states: torch.Tensor,
        persistent_ssm_states: torch.Tensor,
    ) -> None:
        """Write the mamba-radix prefix-cache snapshot for extra_buffer mode.

        The radix cache records each tracked request's state at the last
        complete FLA chunk boundary ``bc = req_start + (lens // 64) * 64``
        (``lens = track_seqlens - prefix``). The owner rank of ``bc`` provides:

        * SSM state -- if ``bc`` is the owner's segment end, the segment-final
          state; if ``bc`` lands on the owner's local FLA chunk grid, the
          intermediate state read from ``h`` at ``bc``; otherwise (a request that
          crosses a rank boundary with a non-64-aligned packed start, so its
          per-request chunk grid is offset from the owner's local grid) the state
          at the nearest local chunk boundary ``lb <= bc`` from ``h`` advanced by
          the remaining ``m = bc - lb < 64`` tokens via a short ``chunk_kda``
          recurrence. This makes every tracked request exact regardless of split
          alignment, mirroring the non-CP ``_init_track_ssm_indices`` result.
        * conv state -- the last ``W-1`` raw input tokens before ``bc``, sliced
          from the rank-local ``mixed_qkv``.

        ``track_{q,k,v,g,beta}`` are the rank-local post-conv / gated inputs
        (``[1, n, H, D]`` / ``[1, n, H]``) that fed the main kernel; the short
        recurrence replays the identical computation over the ``m``-token tail.
        """
        if (
            forward_batch.mamba_track_mask is None
            or not forward_batch.mamba_track_mask.any()
        ):
            return

        cp_group = get_attention_cp_group()
        cp_rank = cp_group.rank_in_group
        self._ensure_kda_cp_cpu_meta(forward_batch, metadata)
        req_starts = metadata.cpu_req_starts
        track_mask_cpu = metadata.cpu_track_mask
        track_seqlens_cpu = metadata.cpu_track_seqlens
        track_indices = forward_batch.mamba_track_indices
        prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

        conv_full_dim = layer.q_dim + layer.k_dim + layer.v_dim
        conv_state_len = persistent_conv_states.shape[-1]  # W - 1
        ssm_full_shape = (layer.num_q_heads, layer.head_v_dim, layer.head_k_dim)
        chunk = FLA_CHUNK_SIZE

        # Cumulative per-segment h chunk offsets (this rank's local segments).
        seg_h_offsets = [0]
        for L in metadata.local_seq_lens_cpu:
            seg_h_offsets.append(seg_h_offsets[-1] + (int(L) - 1) // chunk + 1)

        dst_req_idx = []
        owned = []
        for req_idx, should_track in enumerate(track_mask_cpu):
            if not should_track:
                continue

            lens = int(track_seqlens_cpu[req_idx]) - int(prefix_lens_cpu[req_idx])
            if lens <= 0:
                continue

            req_start = req_starts[req_idx]
            req_end = req_start + int(forward_batch.extend_seq_lens_cpu[req_idx])
            # Position (relative to req start) of the last complete FLA chunk
            # that the radix cache records this snapshot at.
            cache_offset = lens if (lens % chunk == 0) else (lens // chunk) * chunk
            bc = req_start + cache_offset
            if bc > req_end:
                continue

            owner_rank = kda_cp_owner_of_global_token(
                bc - 1, metadata.total_tokens, metadata.cp_size
            )

            i = len(dst_req_idx)
            dst_req_idx.append(req_idx)

            if cp_rank == owner_rank:
                local_idx = None
                for j, local_req_idx in enumerate(metadata.local_req_indices_cpu):
                    if local_req_idx == req_idx:
                        local_idx = j
                        break
                if local_idx is None:
                    raise RuntimeError(
                        "KDA-CP track: owner could not find its local segment "
                        f"for req_idx={req_idx}, bc={bc}."
                    )
                seg_start = metadata.local_segment_global_starts_cpu[local_idx]
                seg_len = int(metadata.local_seq_lens_cpu[local_idx])
                local_pos = bc - seg_start

                # --- SSM state at bc ---
                if local_pos == seg_len:
                    # Boundary is the segment end -> final recurrent state.
                    ssm_state = runtime_ssm_states[local_idx]
                else:
                    # Nearest local FLA chunk boundary lb <= bc (h has its state).
                    lb_offset = (local_pos // chunk) * chunk
                    h_index = seg_h_offsets[local_idx] + lb_offset // chunk
                    state_lb = runtime_h[0, h_index]  # [HV, V, K] at seg_start+lb_offset
                    m = local_pos - lb_offset
                    if m == 0:
                        ssm_state = state_lb
                    else:
                        # Short recurrence: advance state_lb by the m (< 64) tokens
                        # [lb, bc) to reach the per-request chunk boundary bc that
                        # is off the owner's local grid. Replays the same kernel
                        # math over just the tail.
                        from sglang.srt.layers.attention.fla.kda import chunk_kda

                        lb_local = seg_start + lb_offset - metadata.local_start
                        bc_local = bc - metadata.local_start
                        init = (
                            state_lb.to(runtime_ssm_states.dtype)
                            .unsqueeze(0)
                            .contiguous()
                            .clone()  # never alias runtime_h; chunk_kda writes init in place
                        )
                        sub_cu = torch.tensor(
                            [0, m], dtype=torch.int32, device=init.device
                        )
                        chunk_kda(
                            q=track_q[:, lb_local:bc_local],
                            k=track_k[:, lb_local:bc_local],
                            v=track_v[:, lb_local:bc_local],
                            g=track_g[:, lb_local:bc_local],
                            beta=track_beta[:, lb_local:bc_local],
                            initial_state=init,
                            initial_state_indices=torch.zeros(
                                1, dtype=torch.int32, device=init.device
                            ),
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=sub_cu,
                        )
                        ssm_state = init[0]

                # --- conv window at bc (last W-1 raw input tokens) ---
                bc_local = bc - metadata.local_start
                win = raw_mixed_qkv.new_zeros((conv_state_len, conv_full_dim))
                take = min(conv_state_len, bc_local)
                if take > 0:
                    win[conv_state_len - take :] = raw_mixed_qkv[
                        bc_local - take : bc_local
                    ]
                conv_win = win.transpose(0, 1)  # [conv_full_dim, W-1]

                owned.append((i, conv_win, ssm_state))

        dst_idx = track_indices[
            torch.tensor(dst_req_idx, device=track_indices.device, dtype=torch.long)
        ].to(device=persistent_conv_states.device, dtype=torch.long)
        self._coalesced_cp_state_writeback(
            dst_idx,
            owned,
            conv_full_dim,
            conv_state_len,
            ssm_full_shape,
            persistent_conv_states,
            persistent_ssm_states,
        )

    def _coalesced_cp_state_writeback(
        self,
        dst_idx: torch.Tensor,
        owned: list,
        conv_full_dim: int,
        conv_state_len: int,
        ssm_full_shape: tuple,
        persistent_conv_states: torch.Tensor,
        persistent_ssm_states: torch.Tensor,
    ) -> None:
        """Coalesce per-request conv+ssm state exchange into a single collective.

        ``dst_idx`` (identical on every rank) is the ``[n]`` LongTensor of
        persistent destination slots; ``owned`` holds ``(i, conv[D,W-1],
        ssm[HV,V,K])`` for the items this rank owns (full head/channel dim, ``i``
        indexing into ``dst_idx``). Every request has exactly one owner.

        Fully vectorized to avoid per-item ops: the owned states are stacked and
        cast once, scattered into the collective buffer with a single
        ``index_copy_``, and written back with a single ``index_copy_`` on
        ``dst_idx`` (no GPU-scalar indexing / D2H syncs).

        Fast path (state head/channel-sharded across the CP group, the usual
        case): one **reduce_scatter** delivers each rank exactly its shard of
        every item (owner lays its full state out as ``cp_size`` contiguous
        blocks by destination rank; others contribute zeros; SUM picks the
        owner's block). Fallback (full storage or dims not divisible by cp_size):
        one SUM **all-reduce** over the full-state buffer, then a batched narrow.
        """
        n = int(dst_idx.shape[0])
        if n == 0:
            return

        cp_group = get_attention_cp_group()
        cp_rank = cp_group.rank_in_group
        cp_size = cp_group.world_size
        device = persistent_conv_states.device

        conv_local_dim = persistent_conv_states.shape[1]
        ssm_local_dim = persistent_ssm_states.shape[1]
        ssm_tail = tuple(ssm_full_shape[1:])  # (V, K)
        ssm_tail_numel = 1
        for s in ssm_tail:
            ssm_tail_numel *= s
        conv_numel = conv_full_dim * conv_state_len
        ssm_numel = ssm_full_shape[0] * ssm_tail_numel

        conv_sharded = conv_full_dim == conv_local_dim * cp_size
        ssm_sharded = ssm_full_shape[0] == ssm_local_dim * cp_size

        # Batch this rank's owned states once (this rank owns ~n / cp_size items).
        num_owned = len(owned)
        if num_owned:
            owned_rows = torch.tensor(
                [i for (i, _, _) in owned], device=device, dtype=torch.long
            )
            conv_stack = torch.stack([c for (_, c, _) in owned]).to(torch.float32)
            ssm_stack = torch.stack([s for (_, _, s) in owned]).to(torch.float32)

        if cp_size > 1 and conv_sharded and ssm_sharded:
            # --- reduce_scatter fast path ---
            conv_shard_numel = conv_local_dim * conv_state_len
            ssm_shard_numel = ssm_local_dim * ssm_tail_numel
            chunk_numel = conv_shard_numel + ssm_shard_numel

            inp = torch.zeros(cp_size, n, chunk_numel, dtype=torch.float32, device=device)
            if num_owned:
                # Per owned item, split its full state into cp_size contiguous
                # blocks (block d -> rank d, matching _local_state_shard), then
                # lay out by destination: src[d, j, :] = item j's block d.
                conv_blocks = conv_stack.reshape(num_owned, cp_size, conv_shard_numel)
                ssm_blocks = ssm_stack.reshape(num_owned, cp_size, ssm_shard_numel)
                src = torch.cat(
                    [conv_blocks.transpose(0, 1), ssm_blocks.transpose(0, 1)], dim=-1
                ).contiguous()  # [cp_size, num_owned, chunk_numel]
                inp.index_copy_(1, owned_rows, src)

            out = torch.empty(n, chunk_numel, dtype=torch.float32, device=device)
            torch.distributed.reduce_scatter_tensor(
                out, inp, group=cp_group.device_group
            )

            conv_out = out[:, :conv_shard_numel].reshape(
                n, conv_local_dim, conv_state_len
            )
            ssm_out = out[:, conv_shard_numel:].reshape(n, ssm_local_dim, *ssm_tail)
        else:
            # --- all_reduce fallback (full storage, or non-divisible dims) ---
            buf = torch.zeros(
                n, conv_numel + ssm_numel, dtype=torch.float32, device=device
            )
            if num_owned:
                src = torch.cat(
                    [conv_stack.reshape(num_owned, -1), ssm_stack.reshape(num_owned, -1)],
                    dim=-1,
                ).contiguous()
                buf.index_copy_(0, owned_rows, src)

            torch.distributed.all_reduce(buf, group=cp_group.device_group)

            conv_full = buf[:, :conv_numel].reshape(n, conv_full_dim, conv_state_len)
            ssm_full = buf[:, conv_numel:].reshape(n, ssm_full_shape[0], *ssm_tail)
            conv_out = (
                conv_full.narrow(1, cp_rank * conv_local_dim, conv_local_dim)
                if conv_sharded
                else conv_full
            )
            ssm_out = (
                ssm_full.narrow(1, cp_rank * ssm_local_dim, ssm_local_dim)
                if ssm_sharded
                else ssm_full
            )

        persistent_conv_states.index_copy_(
            0, dst_idx, conv_out.to(persistent_conv_states.dtype).contiguous()
        )
        persistent_ssm_states.index_copy_(
            0, dst_idx, ssm_out.to(persistent_ssm_states.dtype).contiguous()
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
        self._ensure_kda_cp_cpu_meta(forward_batch, metadata)
        req_starts = metadata.cpu_req_starts

        conv_full_dim = layer.q_dim + layer.k_dim + layer.v_dim
        conv_state_len = persistent_conv_states.shape[-1]
        ssm_full_shape = (layer.num_q_heads, layer.head_v_dim, layer.head_k_dim)

        dst_req_idx = []
        owned = []
        for req_idx, req_start in enumerate(req_starts):
            req_end = req_start + int(forward_batch.extend_seq_lens_cpu[req_idx])
            if req_end <= req_start:
                continue

            i = len(dst_req_idx)
            dst_req_idx.append(req_idx)

            owner_rank = kda_cp_owner_of_global_token(
                req_end - 1, metadata.total_tokens, metadata.cp_size
            )
            if cp_rank == owner_rank:
                local_idx = self._find_kda_cp_segment(
                    metadata, req_idx=req_idx, segment_end=req_end
                )
                if local_idx is None:
                    raise RuntimeError(
                        "KDA-CP final-state owner could not find its local "
                        f"segment for req_idx={req_idx}, req_end={req_end}."
                    )
                owned.append(
                    (
                        i,
                        runtime_conv_states[local_idx],
                        runtime_ssm_states[local_idx],
                    )
                )

        dst_idx = persistent_cache_indices[
            torch.tensor(
                dst_req_idx, device=persistent_cache_indices.device, dtype=torch.long
            )
        ].to(device=persistent_conv_states.device, dtype=torch.long)
        self._coalesced_cp_state_writeback(
            dst_idx,
            owned,
            conv_full_dim,
            conv_state_len,
            ssm_full_shape,
            persistent_conv_states,
            persistent_ssm_states,
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
        cp_context = None
        continuation_index = None
        continuation_h0 = None
        if kda_cp_active:
            metadata = forward_batch.kda_cp_metadata
            query_start_loc = metadata.local_query_start_loc
            seq_lens_cpu = metadata.local_seq_lens_cpu
            cp_group = get_attention_cp_group()
            conv1d_kernel_size = self.conv_states_shape[-1] + 1
            cp_context = build_kda_fla_cp_context(
                metadata,
                cp_group.device_group,
                conv1d_kernel_size=conv1d_kernel_size,
            )

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
                    continuation_index,
                    continuation_h0,
                ) = self._prepare_kda_cp_states(
                    layer,
                    forward_batch,
                    metadata,
                    cp_context,
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

            # Keep the rank-local [n_valid, D] mixed_qkv (pre-transpose) for the
            # KDA-CP prefix-cache conv snapshot (see _track_kda_cp_state_extend).
            kda_cp_raw_mixed_qkv = mixed_qkv if kda_cp_active else None

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
            # The CP path always runs the Triton chunk_kda_cp kernel, which does
            # NOT apply the gate internally, so it must be gated here regardless
            # of the configured extend backend.
            if kda_cp_active or not self.kernel_dispatcher.extend_applies_gate_internally:
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

            if kda_cp_active:
                # The Triton KDA kernel writes its attention output IN PLACE over
                # the `v` buffer (chunk_gla_fwd_o_gk(o=v)) and normalizes q/k /
                # cumsums g on their contiguous inputs, so after extend_cp `v`
                # holds the output — not the post-conv value — and q/k/g may be
                # mutated too. The extra_buffer radix-cache track short recurrence
                # must replay chunk_kda over the ORIGINAL post-conv inputs, so
                # snapshot them here before the kernel clobbers `v`. (Also avoids
                # the track kernel writing its own o=v back into core_attn_out,
                # which aliases `v`.) Only needed when tracking is active.
                if forward_batch.mamba_track_mask is not None:
                    track_q, track_k = q.clone(), k.clone()
                    track_v, track_g, track_beta = v.clone(), a.clone(), b.clone()
                else:
                    track_q = track_k = track_v = track_g = track_beta = None
                core_attn_out, h = self.kernel_dispatcher.extend_cp(
                    q=q,
                    k=k,
                    v=v,
                    g=a,
                    beta=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    cp_context=cp_context,
                    continuation_h0=continuation_h0,
                    continuation_index=continuation_index,
                )
            else:
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
                    h,
                    kda_cp_raw_mixed_qkv,
                    track_q,
                    track_k,
                    track_v,
                    track_g,
                    track_beta,
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
