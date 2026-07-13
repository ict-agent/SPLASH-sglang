from typing import Callable, Optional

import torch

from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.layers.communicator import (
    AttentionInputs,
    LayerCommunicator,
    LayerScatterModes,
    get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MHCLayerCommunicator(LayerCommunicator):
    """Communication and residual-state handling for an mHC decoder layer."""

    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func=None,
        layer_id: int = -1,
        is_first_layer: bool = False,
        hc_mult: int = 1,
        hc_attn_pre: Optional[Callable] = None,
        hc_ffn_pre: Optional[Callable] = None,
        hc_post: Optional[Callable] = None,
        is_layer_sparse: bool = False,
    ):
        self.is_first_layer = is_first_layer
        self.hc_mult = hc_mult
        self.hc_attn_pre = hc_attn_pre
        self.hc_ffn_pre = hc_ffn_pre
        self.hc_post = hc_post
        self.is_layer_sparse = is_layer_sparse
        self._h_res = None
        self._h_post = None
        self._mlp_comm_kind = None
        self._a2a_scatter_chunks = None
        super().__init__(
            layer_scatter_modes=layer_scatter_modes,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            allow_reduce_scatter=allow_reduce_scatter,
            is_last_layer=is_last_layer,
            qkv_latent_func=qkv_latent_func,
            layer_id=layer_id,
        )

    def _post_init_communicate(self):
        pass

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        del residual, quant_format, post_residual_addition
        assert self.hc_attn_pre is not None

        if (
            self.is_first_layer
            and hidden_states.shape[-1] == self.input_layernorm.weight.shape[0]
        ):
            hidden_states = hidden_states.repeat(1, self.hc_mult)

        residual = hidden_states
        hidden_states, self._h_res, self._h_post, norm_fused = self.hc_attn_pre(
            hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
        )
        if not norm_fused and hidden_states.shape[0] != 0:
            hidden_states = self.input_layernorm(hidden_states)

        if self.qkv_latent_func is not None:
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(hidden_states, forward_batch, self.qkv_latent_func)
            )
        return hidden_states, residual

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        del cache
        assert self.hc_ffn_pre is not None
        assert self.hc_post is not None
        assert self._h_res is not None and self._h_post is not None

        if hidden_states.shape[0] != 0:
            hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.hc_post(
            hidden_states,
            residual,
            self._h_res,
            self._h_post,
        )

        residual = hidden_states
        hidden_states, self._h_res, self._h_post, norm_fused = self.hc_ffn_pre(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        if not norm_fused and hidden_states.shape[0] != 0:
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.prepare_mlp_input(
            hidden_states,
            forward_batch,
            self.is_layer_sparse,
        )
        return hidden_states, residual

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        assert self.hc_post is not None
        assert self._h_res is not None and self._h_post is not None

        hidden_states = self.restore_mlp_output(hidden_states, forward_batch)
        hidden_states = self.hc_post(
            hidden_states,
            residual,
            self._h_res,
            self._h_post,
        )
        if self.is_last_layer:
            hidden_states = hidden_states.unflatten(
                -1, (self.hc_mult, -1)
            ).mean(dim=-2)
        self._h_res = None
        self._h_post = None
        return hidden_states, None

    def prepare_mlp_input(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        is_sparse_layer: bool,
    ) -> torch.Tensor:
        self._mlp_comm_kind = None
        self._a2a_scatter_chunks = None
        if not is_sparse_layer:
            return hidden_states

        moe_a2a_backend = get_moe_a2a_backend()
        if get_attention_dp_size() > 1 and moe_a2a_backend.is_none():
            global_hidden_states = get_global_dp_buffer(get_tp_group())
            dp_gather_partial(global_hidden_states, hidden_states, forward_batch)
            self._mlp_comm_kind = "dp_gather"
            return global_hidden_states

        if (
            envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()
            and get_attention_tp_size() > 1
            and not moe_a2a_backend.is_none()
            and hidden_states.shape[0] > 0
            and hidden_states.shape[0] % get_attention_tp_size() == 0
        ):
            tp_size = get_attention_tp_size()
            tp_rank = get_attention_tp_rank()
            self._a2a_scatter_chunks = list(hidden_states.tensor_split(tp_size))
            self._mlp_comm_kind = "a2a_scatter"
            return self._a2a_scatter_chunks[tp_rank].contiguous()

        return hidden_states

    def restore_mlp_output(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self._mlp_comm_kind == "dp_gather":
            local_hidden_states = get_local_dp_buffer(get_tp_group())
            dp_scatter(local_hidden_states, hidden_states, forward_batch)
            hidden_states = local_hidden_states
        elif self._mlp_comm_kind == "a2a_scatter":
            assert self._a2a_scatter_chunks is not None
            gathered = [
                hidden_states.new_empty(chunk.shape)
                for chunk in self._a2a_scatter_chunks
            ]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        self._mlp_comm_kind = None
        self._a2a_scatter_chunks = None
        return hidden_states

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        del forward_batch
        return False

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        del forward_batch
        return False

    def maybe_prefetch_next_full_attention_kv(
        self,
        forward_batch: ForwardBatch,
        next_full_attention_layer_id: Optional[int],
    ):
        del forward_batch, next_full_attention_layer_id
        return None
