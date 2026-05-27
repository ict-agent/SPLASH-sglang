"""FlashKDA kernel for KDA extend/prefill (CUDA only).

FlashKDA API quirks handled inline:
    - Gate: FlashKDA applies ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
      internally; pass raw g (pre-activation).
    - Beta: FlashKDA applies sigmoid internally; pass raw beta logits.
    - L2 norm: FlashKDA applies q/k L2 normalization internally in its
      prepare kernel; only contiguity is enforced here.
    - State: FlashKDA writes directly into the sglang state pool via
      ``state_indices=cache_indices`` — no external gather/scatter.
    - Intermediate states: FlashKDA emits states at every 64-token boundary so
      radix-cache tracking (``_track_mamba_state_extend``) keeps working.

Decode is not implemented — the dispatcher routes decode to ``TritonKDAKernel``
even when FlashKDA is selected as the global linear attention backend.
"""

import flash_kda
import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class FlashKDAKernel(LinearAttnKernelBase):
    @property
    def applies_gate_internally(self) -> bool:
        return True

    def decode(self, *args, **kwargs):
        raise NotImplementedError(
            "FlashKDAKernel only supports extend; decode should route to "
            "TritonKDAKernel via KDAKernelDispatcher fallback."
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
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        seq_lens_cpu,
        safe_gate_lower_bound: float,
        **kwargs,
    ) -> tuple:
        # g and beta arrive RAW from the backend dispatcher (KDAAttnBackend
        # skips fused_kda_gate when the kernel sets applies_gate_internally=True).
        # FlashKDA applies lower_bound*sigmoid(exp(A_log)*(g+dt_bias)) and
        # sigmoid(beta) internally.
        #
        # Shape: g may arrive as [T, H*D] (fused qkvbfg path) or [1, T, H*D]
        # (non-fused); FlashKDA wants [1, T, H, D]. q is already 4D so head_dim
        # = q.shape[-1]. beta may arrive as [T, H] or [1, T, H]. All shape
        # adjustments here are metadata-only (unsqueeze/unflatten — no memcpy).
        head_dim = q.shape[-1]
        if g.dim() == 2:
            g = g.unsqueeze(0)
        if g.dim() == 3:
            g = g.unflatten(-1, (-1, head_dim))
        if beta.dim() == 2:
            beta = beta.unsqueeze(0)

        # FlashKDA wants bf16; raw projection output may be fp32 in some paths.
        # .to() is a no-op when dtype already matches.
        g = g.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)

        H, D = q.shape[2], q.shape[3]
        N = cache_indices.shape[0]
        device = q.device

        # FlashKDA requires int64 cu_seqlens; sglang's query_start_loc may be int32.
        cu_seqlens = query_start_loc
        if cu_seqlens.dtype != torch.int64:
            cu_seqlens = cu_seqlens.to(torch.int64)

        # CPU companion of cu_seqlens — lets FlashKDA's CP planner skip a D2H sync.
        cu_seqlens_cpu = torch.zeros(N + 1, dtype=torch.int64)
        cu_seqlens_cpu[1:] = torch.tensor(
            seq_lens_cpu, dtype=torch.int64
        ).cumsum(0)

        # Model stores A_log as [1, 1, local_H, 1] and dt_bias as [local_H * D];
        # FlashKDA requires A_log [H] and dt_bias [H, D].
        A_log = A_log.reshape(H)
        dt_bias = dt_bias.reshape(H, D)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        # Plain torch.empty — torch's caching allocator already pools repeated
        # same-size allocations. A grow-only Python-side pool would defeat
        # CUDA-graph capture (capture pins addresses; reallocating to a larger
        # buffer mid-replay points the graph at freed memory).
        total_chunks = sum((sl + 63) // 64 for sl in seq_lens_cpu)
        intermediate_h = torch.empty(
            (total_chunks, H, D, D), dtype=ssm_states.dtype, device=device
        )
        out = torch.empty_like(q)

        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            head_dim**-0.5,
            out,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=safe_gate_lower_bound,
            initial_state=ssm_states,
            final_state=ssm_states,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            intermediate_states=intermediate_h,
            state_indices=cache_indices,
            enable_cp=True,
        )

        # Return h with leading batch dim to match FLA convention [1, NT, H, V, K].
        return out, intermediate_h.unsqueeze(0)
