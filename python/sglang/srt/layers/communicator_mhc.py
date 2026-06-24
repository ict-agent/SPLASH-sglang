from typing import Optional

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
    """Communication helpers for a decoder layer whose residual is mHC state.

    mHC pre/post math stays in the decoder layer, matching DeepSeek V4. This
    class only handles attention output reduction and token-layout changes
    around MoE.
    """

    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func=None,
    ):
        self._mlp_comm_kind = None
        self._a2a_scatter_chunks = None
        super().__init__(
            layer_scatter_modes=layer_scatter_modes,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            allow_reduce_scatter=allow_reduce_scatter,
            is_last_layer=is_last_layer,
            qkv_latent_func=qkv_latent_func,
        )

    def _post_init_communicate(self):
        # mHC owns residual mixing and normalization, so the generic
        # residual-add/layernorm communication functions are not applicable.
        pass

    def prepare_attention_input(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        is_linear_attn: bool,
    ) -> torch.Tensor:
        del is_linear_attn
        if self.qkv_latent_func is not None:
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(hidden_states, forward_batch, self.qkv_latent_func)
            )
        return hidden_states

    def restore_attention_output(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        is_linear_attn: bool,
    ) -> torch.Tensor:
        del forward_batch, is_linear_attn
        return hidden_states

    def reduce_attention_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] != 0:
            hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states

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
