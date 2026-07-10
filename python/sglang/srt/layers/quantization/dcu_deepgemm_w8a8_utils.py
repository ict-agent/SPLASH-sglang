from __future__ import annotations

import torch


def _register_runtime_buffer(
    layer: torch.nn.Module, name: str, value: torch.Tensor
) -> None:
    if name in layer._buffers:
        layer._buffers[name] = value
        layer._non_persistent_buffers_set.add(name)
        return

    if hasattr(layer, name):
        delattr(layer, name)
    layer.register_buffer(name, value, persistent=False)


def prepare_w8a8_int8_deepgemm_weights(layer: torch.nn.Module) -> None:
    if getattr(layer, "_w8a8_int8_deepgemm_repacked", False):
        return
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return

    from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency

    w13_weight = layer.w13_weight
    w2_weight = layer.w2_weight

    layer._w8a8_int8_w13_weight_shape = tuple(w13_weight.shape)
    layer._w8a8_int8_w2_weight_shape = tuple(w2_weight.shape)

    with torch.no_grad():
        w13_weight_deepgemm = pack_int8_weight_enk_to_w6_low_latency(
            w13_weight
        ).detach()
        w2_weight_deepgemm = pack_int8_weight_enk_to_w6_low_latency(
            w2_weight
        ).detach()

    _register_runtime_buffer(layer, "w13_weight_deepgemm", w13_weight_deepgemm)
    _register_runtime_buffer(layer, "w2_weight_deepgemm", w2_weight_deepgemm)
    layer._w8a8_int8_deepgemm_repacked = True

    del layer.w13_weight
    del layer.w2_weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
