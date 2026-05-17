# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch

from sglang.srt.layers.attention.nsa.utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import (
    CommunicateContext,
    ScatterMode,
)
from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator, MHCState
from sglang.srt.layers.communicator_nsa_cp import (
    NSACPCommunicateSimpleFn,
    NSACPCommunicateSummableTensorPairFn,
    NSACPCommunicateWithAllReduceAndLayerNormFn,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_attention_tp_size,
    get_local_dp_buffer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MHCHybridNSACPLayerCommunicator(MHCLayerCommunicator):
    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert (
                self._context.attn_dp_size == 1
            ), f"dp_size should be 1 when moe_runner_backend is none"
        self._communicate_simple_fn = NSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = MHCHybridNSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_summable_tensor_pair_fn = MHCHybridNSACPCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )

    def prepare_attn(
        self,
        hidden_states,
        residual,
        forward_batch: ForwardBatch,
        quant_format: str = "",
        post_residual_addition=None,
    ):
        hidden_states, residual = super().prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format,
            post_residual_addition,
        )

        # KDA-style attention (no qkv_latent abstraction) has kernels that
        # need the full-context tensor (causal_conv1d / chunk_kda are sequence-
        # serial). Under CP, hidden_states arrives split across CP ranks; rerange-
        # gather it here, then prepare_mlp will scatter+rebuild it back.
        if (
            nsa_use_prefill_cp(forward_batch)
            and self.qkv_latent_func is None
        ):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                get_attention_tp_size(),
                forward_batch,
                torch.cuda.current_stream(),
            )

        return hidden_states, residual

    def prepare_mlp(
        self,
        hidden_states,
        residual,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        # Undo the CP gather done in prepare_attn for KDA layers: split the
        # full-context output back into the per-CP-rank slice the MLP path
        # expects.
        if (
            nsa_use_prefill_cp(forward_batch)
            and self.qkv_latent_func is None
        ):
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)

        return super().prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache,
        )


class MHCHybridNSACPCommunicateWithAllReduceAndLayerNormFn(
    NSACPCommunicateWithAllReduceAndLayerNormFn
):
    @staticmethod
    def _simple(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        mhc: MHCState,
    ):
        hidden_states, residual = mhc.attn_to_mlp(hidden_states, residual)
        if hidden_states.shape[0] != 0:
            hidden_states = layernorm(hidden_states)
        return hidden_states, residual

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
        mhc: MHCState,
    ):
        hidden_states, residual = mhc.attn_to_mlp(hidden_states, residual)
        if hidden_states.shape[0] != 0:
            hidden_states = layernorm(hidden_states)

        # for prefill: attn tp scattered -> full
        # for decode: attn tp full -> full
        if nsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            hidden_states, local_hidden_states = (
                get_local_dp_buffer(),
                hidden_states,
            )
            attn_cp_all_gather_into_tensor(
                hidden_states,
                local_hidden_states,
            )
        return hidden_states, residual


class MHCHybridNSACPCommunicateSummableTensorPairFn(
    NSACPCommunicateSummableTensorPairFn
):
    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if not is_last_layer:
            return hidden_states, None

        hidden_states = mhc.mlp_hc.contract_output(hidden_states)
        return hidden_states, None

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        # for prefill: full -> attn tp scattered
        # for decode: full -> attn tp full
        if nsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            input_hidden_states = hidden_states
            hidden_states = hidden_states.tensor_split(context.attn_cp_size)[
                context.attn_cp_rank
            ]
            attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)

        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if not is_last_layer:
            return hidden_states, None

        hidden_states = mhc.mlp_hc.contract_output(hidden_states)
        return hidden_states, None
