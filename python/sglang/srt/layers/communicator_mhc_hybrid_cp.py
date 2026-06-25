from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MHCHybridNSACPLayerCommunicator(MHCLayerCommunicator):
    """Communication for ModelNext's KDA/NSA hybrid CP layout.

    Cross-layer mHC state is block-contiguous ("plain"). KDA consumes that
    layout directly and performs its own CP gather/reduce-scatter. NSA consumes
    the round-robin CP layout, so only NSA inputs/outputs are converted here.
    """

    def prepare_mlp_input(
        self,
        hidden_states,
        forward_batch: ForwardBatch,
        is_sparse_layer: bool,
    ):
        if (
            is_sparse_layer
            and nsa_use_prefill_cp(forward_batch)
            and get_moe_a2a_backend().is_none()
        ):
            raise RuntimeError(
                "ModelNext NSA context parallelism requires an MoE all-to-all "
                "backend for sparse layers."
            )
        return hidden_states

    def restore_mlp_output(
        self,
        hidden_states,
        forward_batch: ForwardBatch,
    ):
        del forward_batch
        return hidden_states
