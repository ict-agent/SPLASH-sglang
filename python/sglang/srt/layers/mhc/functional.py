from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from sglang.srt.environ import envs


def _mhc_pre_torch(
    residual: Tensor,
    fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pure-torch reference implementation of mhc_pre.

    Layout matches the tilelang ``hc_split_sinkhorn`` kernel: ``hc_base`` and
    ``fn`` are laid out as ``[pre(n) | post(n) | comb(n*n)]`` along the
    mix-channel axis.

        residual: (s, n, h)        bf16
        fn:       (mix_hc, n*h)    fp32
        hc_scale: (3,)             fp32
        hc_base:  (mix_hc,)        fp32
        Returns: (post=(s, n, 1), comb=(s, n, n), layer_input=(s, h))
    """
    s, n, h = residual.shape
    dtype = residual.dtype

    x_flat = residual.view(s, n * h).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, fn) * rsqrt  # (s, mix_hc)

    pre_raw = mixes[:, :n]
    post_raw = mixes[:, n : 2 * n]
    comb_raw = mixes[:, 2 * n :].view(s, n, n)
    pre_base = hc_base[:n]
    post_base = hc_base[n : 2 * n]
    comb_base = hc_base[2 * n :].view(n, n)

    pre = torch.sigmoid(pre_raw * hc_scale[0] + pre_base) + hc_pre_eps
    post = hc_post_mult_value * torch.sigmoid(post_raw * hc_scale[1] + post_base)
    comb = comb_raw * hc_scale[2] + comb_base

    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(dtype)
    return post.unsqueeze(-1), comb, layer_input


def _mhc_post_torch(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """Pure-torch reference implementation of mhc_post.

    x:               (s, h)         bf16
    residual:        (s, n, h)      bf16
    post_layer_mix:  (s, n, 1)      fp32
    comb_res_mix:    (s, n, n)      fp32
    Returns:         (s, n, h)      bf16
    """
    out = post_layer_mix * x.unsqueeze(1) + (
        comb_res_mix.unsqueeze(-1) * residual.unsqueeze(2)
    ).sum(dim=1)
    return out.type_as(x)


# Both dispatch functions ultimately call into JIT kernels (tilelang /
# tile_kernels) whose `__call__` is a TVM-FFI Function that torch._dynamo
# cannot trace through. Mark them as opaque so dynamo treats each call as a
# graph break — piecewise CUDA graph runner is designed to handle that.
@torch._dynamo.disable
def _mhc_pre_dispatch(
    residual: Tensor,
    fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run mhc_pre using the env-selected backend.

    Backend priority (torch > tilelang > tile_kernels):
      - SGLANG_OPT_USE_TORCH_MHC=True    -> pure-torch native fallback.
      - SGLANG_OPT_USE_TILELANG_MHC=True -> in-tree tilelang dsv4_mhc.
      - both False (default)             -> external tile_kernels package.

    Returns: (post_mix=(s, n, 1), comb_mix=(s, n, n), layer_input=(s, h)).
    """
    assert residual.dim() == 3, f"residual must be (s, n, h); got {residual.shape}"
    if envs.SGLANG_OPT_USE_TORCH_MHC.get():
        return _mhc_pre_torch(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
    if envs.SGLANG_OPT_USE_TILELANG_MHC.get():
        from sglang.srt.layers.mhc.dsv4_mhc import mhc_pre as _impl

        return _impl(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )

    from tile_kernels.modeling.mhc.functional import mhc_pre as _impl

    n = residual.shape[-2]
    layer_input, (post_mix, comb_mix) = _impl(
        residual=residual,
        fn=fn,
        scale=hc_scale,
        base=hc_base,
        norm_eps=rms_eps,
        mhc_mult=n,
        post_mult_value=hc_post_mult_value,
        pre_eps=hc_pre_eps,
        sinkhorn_eps=hc_sinkhorn_eps,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    return post_mix, comb_mix, layer_input


@torch._dynamo.disable
def _mhc_post_dispatch(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """Run mhc_post using the env-selected backend.

    Backend priority matches ``_mhc_pre_dispatch``. The external
    ``tile_kernels`` post kernel asserts a 4-D ``(b, s, n, h)`` form, so we
    re-introduce a leading ``b=1`` solely for that backend.

    Returns: (s, n, h)
    """
    assert x.dim() == 2 and residual.dim() == 3
    assert post_layer_mix.dim() == 3 and comb_res_mix.dim() == 3
    if envs.SGLANG_OPT_USE_TORCH_MHC.get():
        return _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    if envs.SGLANG_OPT_USE_TILELANG_MHC.get():
        from sglang.srt.layers.mhc.dsv4_mhc import mhc_post as _impl

        return _impl(x, residual, post_layer_mix, comb_res_mix)

    from tile_kernels.mhc.post_kernel import mhc_post_fwd

    s, n, h = residual.shape
    out = mhc_post_fwd(
        x.unsqueeze(0),
        residual.unsqueeze(0),
        post_layer_mix.unsqueeze(0),
        comb_res_mix.unsqueeze(0),
    )
    return out.view(s, n, h)


def hc_pre(
    x: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult_value: float = 2.0,
    hc_norm_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Multi hyper-connection pre-processing for one sublayer (attn or ffn).

    x:        [s, n * hidden_size] bf16  (flattened multi-stream residual)
    hc_fn:    [mix_hc, n * hidden_size] fp32
    hc_scale: [3] fp32
    hc_base:  [mix_hc] fp32
    hc_norm_weight: optional [n * hidden_size] fp32 RMSNorm weight; when
              provided, ``hc_fn`` is multiplied by it before the GEMM.
    Returns:  (layer_input [s, hidden_size],
               h_res       [s, n * n] fp32,
               h_post      [s, n] fp32)
    """
    s, total = x.shape
    hidden_size = total // hc_mult
    if x.numel() == 0:
        empty_layer_input = x.new_zeros((s, hidden_size))
        empty_h_res = torch.zeros(
            (s, hc_mult * hc_mult), device=x.device, dtype=torch.float32
        )
        empty_h_post = torch.zeros((s, hc_mult), device=x.device, dtype=torch.float32)
        return empty_layer_input, empty_h_res, empty_h_post

    fn = hc_fn if hc_norm_weight is None else hc_fn * hc_norm_weight
    residual_3d = x.view(s, hc_mult, hidden_size)
    post_mix, comb_mix, layer_input = _mhc_pre_dispatch(
        residual=residual_3d,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=post_mult_value,
        sinkhorn_repeat=sinkhorn_iters,
    )
    return (
        layer_input,
        comb_mix.reshape(s, hc_mult * hc_mult),
        post_mix.reshape(s, hc_mult),
    )


def hc_post(
    x: Tensor,
    residual: Tensor,
    h_post: Tensor,
    h_res: Tensor,
    hc_mult: int,
) -> Tensor:
    """Multi hyper-connection post-processing for one sublayer.

    x:        [s, hidden_size]
    residual: [s, n * hidden_size]
    h_post:   [s, n]
    h_res:    [s, n * n]
    Returns:  [s, n * hidden_size]
    """
    s, hidden_size = x.shape
    if s == 0:
        return x.new_zeros((s, hc_mult * hidden_size))
    residual = residual.view(s, hc_mult, hidden_size)
    h_post = h_post.view(s, hc_mult, 1)
    h_res = h_res.view(s, hc_mult, hc_mult)
    out = _mhc_post_dispatch(x, residual, h_post, h_res)
    return out.view(s, -1)


def hc_expand(x: Tensor, n: int) -> Tensor:
    """[s, hidden_size] -> [s, n * hidden_size] by replication."""
    return x.repeat(1, n)


def hc_contract(x: Tensor, n: int) -> Tensor:
    """[s, n * hidden_size] -> [s, hidden_size] by averaging."""
    return x.unflatten(-1, (n, -1)).mean(dim=-2)
