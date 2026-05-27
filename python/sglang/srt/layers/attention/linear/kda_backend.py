from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
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
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
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

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}"
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

        forward_metadata = self.forward_metadata
        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        mixed_qkv = mixed_qkv.transpose(0, 1)
        if forward_metadata.has_mamba_track_mask:
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
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
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
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
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
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

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
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            safe_gate_lower_bound=getattr(layer, "safe_gate_lower_bound", -5.0),
        )

        if h is not None:
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
