from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from sglang.srt.layers.mhc.functional import hc_contract, hc_expand, hc_post, hc_pre


class HyperConnection(nn.Module):
    """
    Multi hyper-connection module wrapping the functional ``hc_pre`` /
    ``hc_post`` API with learnable parameters.

    Layout (matches dsv4_mhc tilelang kernels):

        residual:  [S, N * hidden_size]  bf16
        layer_in:  [S, hidden_size]      bf16
        h_post:    [S, N]                fp32
        h_res:     [S, N * N]            fp32

    Parameters:

        mapping_proj: nn.Linear(N * hidden_size, mix_hc, bias=False, fp32)
        bias:  (mix_hc,)                 fp32  — [pre(N) | post(N) | comb(N*N)]
        scale: (3,)                      fp32  — [pre, post, comb] scales
        norm:  optional RMSNorm(N * hidden_size) — gated by mhc_no_norm_weight

    where ``mix_hc = N * (N + 2)``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int,
        layer_number: int,
        rms_norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
        sinkhorn_iterations: int = 20,
        post_mult_value: float = 2.0,
        mhc_no_norm_weight: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n = num_streams
        self.layer_number = layer_number
        self.rms_norm_eps = rms_norm_eps
        self.hc_eps = hc_eps
        self.sinkhorn_iterations = sinkhorn_iterations
        self.post_mult_value = post_mult_value
        self.mhc_no_norm_weight = mhc_no_norm_weight

        n = num_streams
        mix_hc = (2 + n) * n
        d_model = n * hidden_size

        self.d_model = d_model
        self.mix_hc = mix_hc

        if not mhc_no_norm_weight:
            self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps)

        self.mapping_proj = nn.Linear(d_model, mix_hc, bias=False, dtype=torch.float32)
        self.bias = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _check_input(self, hidden_states: Tensor):
        if hidden_states.dim() != 2:
            raise ValueError(
                "Expected hidden_states to have shape [S, N*C], "
                f"got {tuple(hidden_states.shape)}"
            )

        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected hidden_states.shape[-1] == {self.d_model}, "
                f"got {hidden_states.shape[-1]}"
            )

    def pre_forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Pre-forward stage.

        Returns:
            layer_input: [S, C]
            residual:    [S, N*C]
            h_res:       [S, N*N]
            h_post:      [S, N]
        """
        self._check_input(hidden_states)

        if residual is None:
            residual = hidden_states
        else:
            self._check_input(residual)

        layer_input, h_res, h_post = hc_pre(
            x=hidden_states,
            hc_fn=self.mapping_proj.weight,
            hc_scale=self.scale,
            hc_base=self.bias,
            hc_mult=self.n,
            rms_eps=self.rms_norm_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iterations,
            post_mult_value=self.post_mult_value,
            hc_norm_weight=None if self.mhc_no_norm_weight else self.norm.weight,
        )

        return layer_input, residual, h_res, h_post

    def post_forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        h_res: Tensor,
        h_post: Tensor,
    ) -> Tensor:
        """Post-forward stage producing the next [S, N*C] residual stream."""
        self._check_input(residual)
        return hc_post(
            x=hidden_states,
            residual=residual,
            h_post=h_post,
            h_res=h_res,
            hc_mult=self.n,
        )

    def expand_input(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                "Expected hidden_states to have shape [S, C], "
                f"got {tuple(hidden_states.shape)}"
            )

        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_states to have shape [S, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        return hc_expand(hidden_states, self.n)

    def contract_output(self, hidden_states: Tensor) -> Tensor:
        self._check_input(hidden_states)
        return hc_contract(hidden_states, self.n)
