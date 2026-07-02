import itertools
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import (
    get_tp_group as get_tensor_model_parallel_group,
)
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.communicator import enable_moe_dense_fully_dp
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE as GLM4MoESparseMoeBlock
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.load_ckpt import (
    WrappedStorageReader,
    communicate_for_attn,
    count_tp_pp_ep,
    load_distcp,
    merge_ep,
    postprocess,
)

logger = logging.getLogger(__name__)


def is_ep_moe_enabled():
    server_args = get_global_server_args()
    return server_args.ep_size == server_args.tp_size


class _ConfigView:
    """Loader-local config: fields shadowed by ``common.pt`` win, the rest
    fall through to ``init_model.config``. Read-only — nothing is written
    back to the underlying model config."""

    def __init__(self, base, overrides):
        self.__dict__.update(overrides)
        self.__dict__["_base"] = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    @classmethod
    def from_checkpoint(cls, init_model, checkpoint_path):
        common_pt = os.path.join(checkpoint_path, "common.pt")
        if not os.path.exists(common_pt):
            return init_model.config
        a = torch.load(common_pt, weights_only=False)["args"]

        return cls(
            init_model.config,
            {
                "use_qk_norm": a.qk_layernorm or False,
                "post_self_attn_layernorm": getattr(
                    a, "post_self_attn_layernorm", False
                ),
                "post_mlp_layernorm": getattr(a, "post_mlp_layernorm", False),
            },
        )


class _UnpicklerWrapper(pickle.Unpickler):
    """Tolerant unpickler that swaps out megatron / glm classes the inference
    runtime doesn't have. Installed as ``pickle.Unpickler`` for the duration of
    a single ``load_megatron_weights`` call."""

    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if (
            mod_name.startswith("megatron")
            or mod_name.startswith("glm")
            or name == "DummyClass"
        ):
            return DummyClass
        return super().find_class(mod_name, name)


def _dict_access_multi(a_dict, keys):
    if len(keys) == 0:
        return a_dict
    return _dict_access_multi(a_dict[keys[0]], keys[1:])


def _has_keys(a_dict, keys):
    try:
        _dict_access_multi(a_dict, keys)
        return True
    except Exception:
        return False


def _merge_tensors(
    tp_sd: List[Dict],
    model_key: str,
    keys: List[str],
    original_tp: int,
    target_tp: int,
    current_tp: int,
    slice_dim: Optional[int] = None,
    merge_fn: Optional[Callable] = None,
    split_fn: Optional[Callable] = None,
):
    if target_tp <= original_tp:
        cnt = original_tp // target_tp
        offset = cnt * current_tp
        sd_list = [
            _dict_access_multi(tp_sd[i + offset], [model_key] + keys)
            for i in range(cnt)
        ]
        if original_tp == target_tp:
            return sd_list[0]
        if slice_dim is not None:
            return torch.cat(sd_list, dim=slice_dim)
        assert merge_fn is not None
        return merge_fn(sd_list)
    else:
        cnt = target_tp // original_tp
        tensor = _dict_access_multi(tp_sd[current_tp // cnt], [model_key] + keys)
        if slice_dim is not None:
            return torch.chunk(tensor, cnt, dim=slice_dim)[current_tp % cnt].clone()
        assert split_fn is not None
        return split_fn(tensor, cnt, current_tp % cnt)


_KEY_MAP_BASE = {
    "word_embeddings": {
        "mgt": ["language_model", "embedding", "word_embeddings", "weight"],
        "mcore": ["embedding.word_embeddings.weight"],
    },
    "input_layernorm": {
        "mgt": [
            "language_model",
            "encoder",
            "layers.{i}.input_layernorm.{attr}",
        ],
        "mcore": ["decoder.layers.{i}.self_attention.linear_qkv.layer_norm_{attr}"],
    },
    "standalone.input_layernorm": {
        "mcore": ["decoder.layers.{i}.input_layernorm.{attr}"]
    },
    "query_key_value": {
        "mgt": [
            "language_model",
            "encoder",
            "layers.{i}.self_attention.query_key_value.{attr}",
        ],
        "mcore": ["decoder.layers.{i}.self_attention.linear_qkv.{attr}"],
    },
    "q_layernorm": {
        "mcore": ["decoder.layers.{i}.self_attention.q_layernorm.{attr}"],
    },
    "k_layernorm": {
        "mcore": ["decoder.layers.{i}.self_attention.k_layernorm.{attr}"],
    },
    "q_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.linear_q_proj.{attr}"],
    },
    "q_a_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.linear_q_down_proj.{attr}"],
    },
    "q_b_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.linear_q_up_proj.{attr}"],
    },
    "kv_a_proj_with_mqa": {
        "mcore": ["decoder.layers.{i}.self_attention.linear_kv_down_proj.{attr}"],
    },
    "kv_b_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.linear_kv_up_proj.{attr}"],
    },
    "kv_a_layernorm": {
        "mcore": [
            "decoder.layers.{i}.self_attention.linear_kv_up_proj.layer_norm_{attr}"
        ],
    },
    "q_a_layernorm": {
        "mcore": [
            "decoder.layers.{i}.self_attention.linear_q_up_proj.layer_norm_{attr}"
        ],
    },
    "dsa_wq_b": {
        "mcore": ["decoder.layers.{i}.self_attention.wq_b.{attr}"],
    },
    "dsa_wk": {
        "mcore": ["decoder.layers.{i}.self_attention.wk.{attr}"],
    },
    "dsa_weights_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.weights_proj.{attr}"],
    },
    "dsa_k_norm": {
        "mcore": ["decoder.layers.{i}.self_attention.k_norm.{attr}"],
    },
    "dsa_index_kpool_compress_ape": {
        "mcore": ["decoder.layers.{i}.self_attention.index_kpool_compress_ape"],
    },
    "dsa_index_kpool_compress_gate": {
        "mcore": ["decoder.layers.{i}.self_attention.index_kpool_compress_gate"],
    },
    "dsa_index_kpool_compress_gate_weight": {
        "mcore": ["decoder.layers.{i}.self_attention.index_kpool_compress_gate.weight"],
    },
    "dense": {
        "mgt": [
            "language_model",
            "encoder",
            "layers.{i}.self_attention.dense.{attr}",
        ],
        "mcore": ["decoder.layers.{i}.self_attention.linear_proj.{attr}"],
    },
    "standalone.post_attention_layernorm": {
        "mgt": [
            "language_model",
            "encoder",
            "layers.{i}.post_attention_layernorm.{attr}",
        ],
        "mcore": ["decoder.layers.{i}.mlp.linear_fc1.layer_norm_{attr}"],
    },
    "post_attention_layernorm": {
        "mcore": ["decoder.layers.{i}.pre_mlp_layernorm.{attr}"]
    },
    "dense_h_to_4h": {
        "mgt": [
            "language_model",
            "encoder",
            "layers.{i}.mlp.dense_h_to_4h.{attr}",
        ],
        "mcore": ["decoder.layers.{i}.mlp.linear_fc1.{attr}"],
    },
    "dense_4h_to_h": {
        "mgt": [
            "language_model",
            "encoder",
            "layers.{i}.mlp.dense_4h_to_h.{attr}",
        ],
        "mcore": ["decoder.layers.{i}.mlp.linear_fc2.{attr}"],
    },
    "post_mlp_layernorm": {"mcore": ["decoder.layers.{i}.post_mlp_layernorm.{attr}"]},
    "post_self_attn_layernorm": {
        "mcore": ["decoder.layers.{i}.post_self_attn_layernorm.{attr}"]
    },
    "moe.router": {"mcore": ["decoder.layers.{i}.mlp.router.weight"]},
    "moe.router_bias": {"mcore": ["decoder.layers.{i}.mlp.router.expert_bias"]},
    "moe.dense_h_to_4h": {
        "mcore": ["decoder.layers.{i}.mlp.experts.linear_fc1.weight{j}"],
    },
    "moe.dense_4h_to_h": {
        "mcore": ["decoder.layers.{i}.mlp.experts.linear_fc2.weight{j}"],
    },
    "moe.shared_experts.dense_h_to_4h": {
        "mcore": ["decoder.layers.{i}.mlp.shared_experts.linear_fc1.{attr}"]
    },
    "moe.shared_experts.dense_4h_to_h": {
        "mcore": ["decoder.layers.{i}.mlp.shared_experts.linear_fc2.{attr}"]
    },
    "moe.weight1": {"mcore": ["decoder.layers.{i}.mlp.experts.weight1"]},
    "moe.weight2": {"mcore": ["decoder.layers.{i}.mlp.experts.weight2"]},
    "A_log": {
        "mcore": ["decoder.layers.{i}.self_attention.A_log"],
    },
    "dt_bias": {
        "mcore": ["decoder.layers.{i}.self_attention.dt_bias"],
    },
    "f_a_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.f_proj.0.weight"],
    },
    "f_b_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.f_proj.1.weight"],
    },
    "g_a_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.g_proj.0.weight"],
    },
    "g_b_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.g_proj.1.weight"],
    },
    "o_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.out_proj.weight"],
    },
    "b_proj": {
        "mcore": ["decoder.layers.{i}.self_attention.b_proj.weight"],
    },
    "output_layer": {
        "mgt": ["language_model", "output_layer", "weight"],
        "mcore": ["output_layer.weight"],
    },
    "final_layernorm": {
        "mgt": ["language_model", "encoder", "final_layernorm.{attr}"],
        "mcore": ["decoder.final_layernorm.weight"],
    },
    "self_attention_hyper_connection.mapping_proj.weight": {
        "mcore": [
            "decoder.layers.{i}.self_attention_hyper_connection.mapping_proj.weight"
        ]
    },
    "self_attention_hyper_connection.norm.weight": {
        "mcore": ["decoder.layers.{i}.self_attention_hyper_connection.norm_weight"]
    },
    "self_attention_hyper_connection.scale": {
        "mcore": ["decoder.layers.{i}.self_attention_hyper_connection.scale"]
    },
    "self_attention_hyper_connection.bias": {
        "mcore": ["decoder.layers.{i}.self_attention_hyper_connection.base"]
    },
    "mlp_hyper_connection.mapping_proj.weight": {
        "mcore": ["decoder.layers.{i}.mlp_hyper_connection.mapping_proj.weight"]
    },
    "mlp_hyper_connection.norm.weight": {
        "mcore": ["decoder.layers.{i}.mlp_hyper_connection.norm_weight"]
    },
    "mlp_hyper_connection.scale": {
        "mcore": ["decoder.layers.{i}.mlp_hyper_connection.scale"]
    },
    "mlp_hyper_connection.bias": {
        "mcore": ["decoder.layers.{i}.mlp_hyper_connection.base"]
    },
    "o_norm": {
        "mcore": ["decoder.layers.{i}.self_attention.o_norm.weight"],
    },
    **{
        f"{linear_key}_proj": {
            "mcore": [
                "decoder.layers.{i}" + f".self_attention.{linear_key}_proj.weight"
            ],
        }
        for linear_key in "qkv"
    },
    **{
        f"{linear_key}_conv1d": {
            "mcore": [
                "decoder.layers.{i}" + f".self_attention.{linear_key}_conv1d.weight"
            ],
        }
        for linear_key in "qkv"
    },
}


def _build_key_map(ifmtp: bool):
    """Return the per-call key map.

    For the base (non-MTP) loader, this is a copy of ``_KEY_MAP_BASE``.

    For the MTP loader, every ``decoder.layers.{i}.`` prefix is rewritten to
    ``mtp.layers.{i}.transformer_layer.`` so MLA / DSA / kpool / KDA dispatch
    is reused unchanged. Heads that live one level above the transformer
    layer (``eh_proj``, ``enorm``, ``hnorm``, ``final_layernorm``) and the
    global embedding / ``output_layer`` are then overridden explicitly.
    """
    key_map = {k: dict(v) for k, v in _KEY_MAP_BASE.items()}
    if not ifmtp:
        return key_map

    BASE_PREFIX = "decoder.layers.{i}."
    MTP_PREFIX = "mtp.layers.{i}.transformer_layer."
    for name, spec in list(key_map.items()):
        paths = spec.get("mcore")
        if not paths:
            continue
        spec["mcore"] = [
            (MTP_PREFIX + p[len(BASE_PREFIX) :]) if p.startswith(BASE_PREFIX) else p
            for p in paths
        ]

    key_map.update(
        {
            "word_embeddings": {"mcore": ["embedding.word_embeddings.weight"]},
            "eh_proj": {"mcore": ["mtp.layers.{i}.eh_proj.weight"]},
            "enorm": {"mcore": ["mtp.layers.{i}.enorm.weight"]},
            "hnorm": {"mcore": ["mtp.layers.{i}.hnorm.weight"]},
            "final_layernorm": {
                "mcore": ["mtp.layers.{i}.final_layernorm.weight"],
            },
            "output_layer": {"mcore": ["output_layer.weight"]},
        }
    )
    return key_map


def _read_torch_ep_ckpt(ckpt_dir, original_tp, original_pp, original_ep, cfg, tp):
    """Reader for the ``torch`` (legacy .pt) checkpoint format with EP > 1.

    Either reads one EP rank per TP rank (``ep_tp_transpose``) or reads all
    EP ranks and merges them via :func:`merge_ep`. Returns an
    ``mgt_sd[pp][tp]`` shaped list of state dicts.
    """
    mgt_sd = [[None for _ in range(original_tp)] for _ in range(original_pp)]
    if cfg.ep_tp_transpose:
        assert original_ep == get_tensor_model_parallel_world_size(), (
            f"EP and TP must match for `ep_tp_transpose` enabled, "
            f"EP={original_ep}, TP={get_tensor_model_parallel_world_size()} "
        )
        assert original_tp == 1
        logger.info("EP-TP Transpose enabled")
        for i, j in itertools.product(range(original_tp), range(original_pp)):
            ep_rank = tp
            if original_pp == 1:
                ckpt_file = (
                    ckpt_dir / f"mp_rank_{i:02d}_{ep_rank:03d}" / "model_optim_rng.pt"
                )
            else:
                ckpt_file = (
                    ckpt_dir
                    / f"mp_rank_{i:02d}_{j:03d}_{ep_rank:03d}"
                    / "model_optim_rng.pt"
                )
            try:
                ep_checkpoint = torch.load(ckpt_file, map_location="cpu")
            except Exception:
                logger.info(f"{ckpt_file} Error")
                raise
            mgt_sd[j][i] = ep_checkpoint
        return mgt_sd

    for i, j in itertools.product(range(original_tp), range(original_pp)):
        ep_checkpoints = []
        for k in range(original_ep):
            if original_pp == 1:
                ckpt_file = ckpt_dir / f"mp_rank_{i:02d}_{k:03d}" / "model_optim_rng.pt"
            else:
                ckpt_file = (
                    ckpt_dir / f"mp_rank_{i:02d}_{j:03d}_{k:03d}" / "model_optim_rng.pt"
                )
            logger.info(f"{ckpt_file} {os.path.exists(ckpt_file)}")
            ep_checkpoints.append(torch.load(ckpt_file, map_location="cpu"))
        for key in ep_checkpoints[0].keys():
            if key.startswith("model"):
                ep_checkpoints[0][key] = merge_ep(
                    [c[key] for c in ep_checkpoints],
                    cfg.n_routed_experts // original_ep,  # experts_per_rank
                    cfg.moe_ffn_hidden_size,
                )
        mgt_sd[j][i] = ep_checkpoints[0]
    return mgt_sd


def _read_torch_dist_ckpt(
    checkpoint_path, metadata, ifmtp, cfg, target_tp, tp, st_time
):
    """Reader for ``torch_dist`` distcp format. Builds an ``mgt_sd[1][1]`` of
    one merged state dict, plus a stashed ``args`` from ``common.pt`` if
    present (consumed by the orchestrator's bookkeeping at the end)."""
    mgt_sd = [[{}]]
    attn_state_dict, attn_keys, attn_dicts = load_distcp(
        checkpoint_path,
        metadata,
        ifattn=True,
        ifnextn=ifmtp,
        total_num_experts=cfg.n_routed_experts,
        layer=cfg.num_hidden_layers,
    )
    mgt_sd[0][0]["model"] = attn_state_dict
    common_pt = os.path.join(checkpoint_path, "common.pt")
    if os.path.exists(common_pt):
        mgt_sd[0][0]["args"] = torch.load(common_pt, weights_only=False)["args"]
    communicate_for_attn(mgt_sd[0][0]["model"], attn_keys, target_tp, tp)
    del attn_state_dict
    if torch.distributed.get_rank() == 0:
        logger.info(f"loading attention distcp time: {time.time() - st_time}")

    moe_dicts = {}
    if cfg.n_routed_experts is not None:
        moe_state_dict, _moe_keys, moe_dicts = load_distcp(
            checkpoint_path,
            metadata,
            ifattn=False,
            ifnextn=ifmtp,
            total_num_experts=cfg.n_routed_experts,
            layer=cfg.num_hidden_layers,
        )
        mgt_sd[0][0]["model"].update(moe_state_dict)
        if torch.distributed.get_rank() == 0:
            logger.info(f"loading distcp total time: {time.time() - st_time}")
    # TODO: postprocess before communicate_for_attn (load moe -> load attn -> postprocess -> communicate)
    postprocess(mgt_sd, attn_dicts, moe_dicts)

    # avoid cast to bf16 with tp
    hc_state_dict, _hc_keys, _hc_dicts = load_distcp(
        checkpoint_path,
        metadata,
        ifattn=False,
        ifnextn=ifmtp,
        total_num_experts=cfg.n_routed_experts,
        layer=cfg.num_hidden_layers,
        load_hc=True,
    )
    for key in hc_state_dict:
        mgt_sd[0][0]["model"][key] = hc_state_dict[key].view(
            mgt_sd[0][0]["model"][key].shape
        )
    return mgt_sd


def _read_torch_legacy_ckpt(
    ckpt_dir, original_tp, original_pp, original_pp_enabled, target_tp, tp, cnt
):
    """Reader for the plain ``torch`` checkpoint format (no EP).

    Each rank only loads the TP shards it owns; others are ``None`` and never
    accessed by the rest of the loader.
    """
    return [
        [
            (
                torch.load(
                    ckpt_dir
                    / (
                        f"mp_rank_{j:02d}_{i:03d}"
                        if original_pp_enabled
                        else f"mp_rank_{j:02d}"
                    )
                    / "model_optim_rng.pt",
                    map_location="cpu",
                    pickle_module=pickle,
                )
                if (j // cnt == tp if target_tp <= original_tp else j == tp // cnt)
                else None
            )
            for j in range(original_tp)
        ]
        for i in range(original_pp)
    ]


def _load_embedding(
    init_model,
    mgt_sd,
    get_keys,
    first_model_in_pp,
    original_tp,
    target_tp,
    tp,
    device,
    original_pp,
):
    """Copy ``word_embeddings`` from any PP rank that carries it. Raises
    ``KeyError`` if no PP rank had the key."""
    for pp in range(original_pp):
        try:
            init_model.model.embed_tokens = init_model.model.embed_tokens.to(device)
            init_model.model.embed_tokens.weight.copy_(
                _merge_tensors(
                    tp_sd=mgt_sd[pp],
                    model_key=first_model_in_pp,
                    keys=get_keys("word_embeddings"),
                    original_tp=original_tp,
                    target_tp=target_tp if not is_dp_attention_enabled() else 1,
                    current_tp=tp if not is_dp_attention_enabled() else 0,
                    slice_dim=0,
                ).to(device)
            )
        except KeyError:
            pass
        else:
            return
    raise KeyError("word_embeddings not found in checkpoint")


def _load_lm_head(
    init_model,
    cfg,
    mgt_sd,
    get_keys,
    last_model_in_pp,
    original_tp,
    target_tp,
    tp,
    device,
):
    init_model.lm_head = init_model.lm_head.to(device)
    if not getattr(cfg, "tie_word_embeddings", False):
        embedding_keys = get_keys("output_layer", attr="weight")
    else:
        embedding_keys = get_keys("word_embeddings")

    lm_head_tp_size = getattr(init_model.lm_head, "tp_size", target_tp)
    if lm_head_tp_size == 1:
        lm_target_tp, lm_current_tp = 1, 0
    else:
        lm_target_tp, lm_current_tp = target_tp, tp

    init_model.lm_head.weight.copy_(
        _merge_tensors(
            tp_sd=mgt_sd[-1],
            model_key=last_model_in_pp,
            keys=embedding_keys,
            original_tp=original_tp,
            target_tp=lm_target_tp,
            current_tp=lm_current_tp,
            slice_dim=0,
        ).to(device)
    )


def _load_mtp_heads(
    init_model, mgt_sd, get_keys, last_model_in_pp, first_rank_loaded, device
):
    """Copy MTP-only heads (hnorm, enorm, eh_proj, shared_head.norm)."""
    for name in ("hnorm", "enorm"):
        m = getattr(init_model.model, name)
        m = m.to(device)
        setattr(init_model.model, name, m)
        m.weight.copy_(
            _dict_access_multi(mgt_sd[0][0]["model"], get_keys(name, i=0)).to(device)
        )

    init_model.model.eh_proj = init_model.model.eh_proj.to(device)
    init_model.model.eh_proj.weight.copy_(
        _dict_access_multi(mgt_sd[0][0]["model"], get_keys("eh_proj", i=0)).to(device)
    )

    init_model.model.shared_head = init_model.model.shared_head.to(device)
    init_model.model.shared_head.norm = init_model.model.shared_head.norm.to(device)
    final_ln = _dict_access_multi(
        mgt_sd[-1][first_rank_loaded],
        [last_model_in_pp] + get_keys("final_layernorm", i=0),
    )
    init_model.model.shared_head.norm.weight.copy_(final_ln.to(device))
    if hasattr(init_model.model.shared_head.norm, "bias"):
        init_model.model.shared_head.norm.bias.copy_(final_ln)


def _load_final_layernorm(
    init_model, mgt_sd, get_keys, last_model_in_pp, first_rank_loaded, device
):
    init_model.model.norm = init_model.model.norm.to(device)
    init_model.model.norm.weight.copy_(
        _dict_access_multi(
            mgt_sd[-1][first_rank_loaded],
            [last_model_in_pp] + get_keys("final_layernorm", attr="weight"),
        ).to(device)
    )
    if hasattr(init_model.model.norm, "bias"):
        init_model.model.norm.bias.copy_(
            _dict_access_multi(
                mgt_sd[-1][first_rank_loaded],
                [last_model_in_pp] + get_keys("final_layernorm", attr="bias"),
            )
        )


def _record_consumed_train_samples(init_model, mgt_sd, ckpt_dir):
    """Persist megatron-side training counters to ``meta.json`` on rank 0.

    The ``args`` blob is only present for distcp checkpoints and may be
    missing for legacy ``torch`` ones; in that case both attributes are set
    to ``None`` to mirror the legacy behavior.
    """
    try:
        init_model.model.consumed_train_samples = mgt_sd[0][0][
            "args"
        ].consumed_train_samples
        init_model.model.consumed_train_tokens = (
            init_model.model.consumed_train_samples * mgt_sd[0][0]["args"].seq_length
        )
        if torch.distributed.get_rank() == 0:
            meta_file = ckpt_dir / "meta.json"
            with open(meta_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "consumed_train_samples": init_model.model.consumed_train_samples,
                            "consumed_train_tokens": init_model.model.consumed_train_tokens,
                        }
                    )
                )
    except Exception:
        init_model.model.consumed_train_samples = None
        init_model.model.consumed_train_tokens = None


# ---------------------------------------------------------------------------
# Per-layer loaders
# ---------------------------------------------------------------------------


class _LayerCtx:
    """Bag of per-layer state passed into ``_load_one_layer`` and friends.

    We collect everything as instance attributes so the helpers below can be
    plain functions rather than methods on a giant class — readability wins
    against either of: (a) 25-arg signatures, or (b) stuffing the loader into
    a class. The fields are written once at the call site and read-only after.
    """

    __slots__ = (
        "cfg",
        "init_model",
        "params_dict",
        "ifmtp",
        "original_tp",
        "original_ep",
        "target_tp",
        "tp",
        "mlp_tp_rank",
        "mlp_tp_size",
        "attn_tp_rank",
        "attn_tp_size",
        "mgt_sd",
        "model_key",
        "pp",
        "mgt_tp_0",
        "get_keys",
        "first_rank_loaded",
        "layer",
        "layer_offset",
        "i",
        "is_moe_layer",
        "is_linear_layer",
        "device",
    )

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    # --- shorthand helpers used pervasively by the per-branch builders ---

    def keys(self, name, **kw):
        """``get_keys(name, i=ctx.i, **kw)`` — defaults ``i`` to the current
        layer index. Pass ``i=`` explicitly to override (e.g., MTP heads use
        ``i=0``)."""
        kw.setdefault("i", self.i)
        return self.get_keys(name, **kw)

    def read(self, name, **kw):
        """Read a tensor from ``mgt_tp_0`` via the key map."""
        return _dict_access_multi(self.mgt_tp_0, self.keys(name, **kw))

    def merge(
        self,
        name,
        *,
        target_tp,
        current_tp,
        attr=None,
        slice_dim=None,
        merge_fn=None,
        split_fn=None,
    ):
        """``_merge_tensors`` with the boilerplate (``tp_sd``, ``model_key``,
        ``original_tp``) prefilled from this context."""
        kw = {} if attr is None else {"attr": attr}
        return _merge_tensors(
            tp_sd=self.mgt_sd[self.pp],
            model_key=self.model_key,
            keys=self.keys(name, **kw),
            original_tp=self.original_tp,
            target_tp=target_tp,
            current_tp=current_tp,
            slice_dim=slice_dim,
            merge_fn=merge_fn,
            split_fn=split_fn,
        )


def _merge_glu(sd_list, dim=0):
    return torch.cat(
        [sd.chunk(dim=dim, chunks=2)[0].clone() for sd in sd_list]
        + [sd.chunk(dim=dim, chunks=2)[1].clone() for sd in sd_list],
        dim=dim,
    )


def _split_glu(sd, cnt, idx, dim=0):
    return torch.cat(
        (
            sd.chunk(dim=dim, chunks=2)[0].chunk(cnt, dim=dim)[idx].clone(),
            sd.chunk(dim=dim, chunks=2)[1].chunk(cnt, dim=dim)[idx].clone(),
        ),
        dim=dim,
    )


def _build_norms_layer_sd(ctx, layer_sd):
    """Fill input_layernorm, post_attention_layernorm, mhc / qk-norm
    extras, and the optional post-attn / post-mlp norms."""
    cfg = ctx.cfg

    layer_sd["input_layernorm.weight"] = ctx.read(
        (
            "standalone.input_layernorm"
            if getattr(cfg, "mla", False)
            else "input_layernorm"
        ),
        attr="weight",
    )
    layer_sd["post_attention_layernorm.weight"] = ctx.read(
        (
            "post_attention_layernorm"
            if ctx.is_moe_layer
            else "standalone.post_attention_layernorm"
        ),
        attr="weight",
    )

    if getattr(cfg, "mhc", False):
        # mcore mhtk ckpt layout: per-prefix {mapping_proj.weight,
        # norm_weight, scale, base}. The sglang model owns these as
        # plain nn.Parameter on the layer with names hc_{attn,ffn}_{base,
        # scale, fn}. When the training-time config disabled the norm
        # (mhc_no_norm_weight=True), the ckpt still carries norm_weight
        # but it was NOT used in forward — so we must NOT fold it in.
        mhc_fold_norm = not getattr(cfg, "mhc_no_norm_weight", True)
        for hc_prefix, dst_prefix in (
            ("self_attention_hyper_connection", "hc_attn"),
            ("mlp_hyper_connection", "hc_ffn"),
        ):
            base = ctx.read(f"{hc_prefix}.bias")
            if base.numel() == 1:
                base = base.unsqueeze(0)
            layer_sd[f"{dst_prefix}_base"] = base

            scale = ctx.read(f"{hc_prefix}.scale")
            if scale.numel() == 1:
                scale = scale.unsqueeze(0)
            layer_sd[f"{dst_prefix}_scale"] = scale

            fn_weight = ctx.read(f"{hc_prefix}.mapping_proj.weight")
            if mhc_fold_norm:
                norm_keys = ctx.keys(f"{hc_prefix}.norm.weight")
                assert _has_keys(ctx.mgt_tp_0, norm_keys), (
                    f"[mHC fold] mhc_no_norm_weight=False but ckpt is "
                    f"missing {norm_keys}"
                )
                fn_weight = fn_weight * _dict_access_multi(ctx.mgt_tp_0, norm_keys)
            layer_sd[f"{dst_prefix}_fn"] = fn_weight

    if getattr(cfg, "use_qk_norm", False) and not getattr(cfg, "mla", False):
        layer_sd["self_attn.q_norm.weight"] = ctx.read("q_layernorm", attr="weight")
        layer_sd["self_attn.k_norm.weight"] = ctx.read("k_layernorm", attr="weight")
    if cfg.post_self_attn_layernorm:
        layer_sd["post_self_attn_layernorm.weight"] = ctx.read(
            "post_self_attn_layernorm", attr="weight"
        )
    if cfg.post_mlp_layernorm:
        layer_sd["post_mlp_layernorm.weight"] = ctx.read(
            "post_mlp_layernorm", attr="weight"
        )


def _build_kda_attn_sd(ctx, layer_sd):
    """KDA / linear-attention branch. Mirrors Glm5NextLinearAttention init."""
    from sglang.srt.environ import envs as _envs

    layer = ctx.layer
    i = ctx.i

    logger.info(f"{i} loading linear layer")
    do_fuse_qkvbfg = _envs.SGLANG_GLM5_NEXT_FUSE_QKVBFG.get()

    # KDA heads shard by CP under NSA prefill CP, otherwise by attn TP.
    if is_nsa_enable_prefill_cp():
        shard_size = get_attention_cp_size()
        shard_rank = get_attention_cp_rank()
    else:
        shard_size = ctx.attn_tp_size
        shard_rank = ctx.attn_tp_rank

    qkv_conv_weights = []
    for k in "qkv":
        conv_weight = ctx.merge(
            f"{k}_conv1d", target_tp=shard_size, current_tp=shard_rank, slice_dim=0
        )
        if hasattr(layer.self_attn, "qkv_conv1d"):
            qkv_conv_weights.append(conv_weight)
        else:
            layer_sd[f"self_attn.{k}_conv1d.weight"] = conv_weight
    if qkv_conv_weights:
        layer_sd["self_attn.qkv_conv1d.weight"] = torch.cat(qkv_conv_weights, dim=0)

    if do_fuse_qkvbfg:
        # Order must match Glm5NextForCausalLM._STACKED_PARAMS_MAPPING:
        # fused_qkvbfg_a_proj is q, k, v, b (column-parallel) then f_a, g_a (replicated).
        fused_a_parts = [
            ctx.merge(
                f"{k}_proj", target_tp=shard_size, current_tp=shard_rank, slice_dim=0
            )
            for k in ("q", "k", "v", "b")
        ] + [
            ctx.merge(f"{k}_proj", target_tp=1, current_tp=0, slice_dim=0)
            for k in ("f_a", "g_a")
        ]
        layer_sd["self_attn.fused_qkvbfg_a_proj.weight"] = torch.cat(
            fused_a_parts, dim=0
        )

        # fused_fg_b_proj: stack(f_b, g_b) along batch dim 0.
        f_b = ctx.merge(
            "f_b_proj", target_tp=shard_size, current_tp=shard_rank, slice_dim=0
        )
        g_b = ctx.merge(
            "g_b_proj", target_tp=shard_size, current_tp=shard_rank, slice_dim=0
        )
        layer_sd["self_attn.fused_fg_b_proj.weight"] = torch.stack([f_b, g_b], dim=0)

        layer_sd["self_attn.o_proj.weight"] = ctx.merge(
            "o_proj", target_tp=shard_size, current_tp=shard_rank, slice_dim=1
        )
    else:
        qkv_proj_weights = []
        for k in "qkv":
            proj_weight = ctx.merge(
                f"{k}_proj", target_tp=shard_size, current_tp=shard_rank, slice_dim=0
            )
            if hasattr(layer.self_attn, "qkv_proj"):
                qkv_proj_weights.append(proj_weight)
            else:
                layer_sd[f"self_attn.{k}_proj.weight"] = proj_weight
        if qkv_proj_weights:
            layer_sd["self_attn.qkv_proj.weight"] = torch.cat(qkv_proj_weights, dim=0)
        for k in "g_a g_b f_a f_b b o".split():
            replicated = k in ("f_a", "g_a")
            layer_sd[f"self_attn.{k}_proj.weight"] = ctx.merge(
                f"{k}_proj",
                target_tp=1 if replicated else shard_size,
                current_tp=0 if replicated else shard_rank,
                slice_dim=1 if k == "o" else 0,
            )

    layer_sd["self_attn.A_log"] = ctx.read("A_log").view(1, 1, -1, 1)
    if shard_size > 1:
        layer_sd["self_attn.A_log"] = torch.chunk(
            layer_sd["self_attn.A_log"], shard_size, dim=2
        )[shard_rank].clone()
    layer_sd["self_attn.dt_bias"] = ctx.read("dt_bias")
    if shard_size > 1:
        layer_sd["self_attn.dt_bias"] = torch.chunk(
            layer_sd["self_attn.dt_bias"], shard_size, dim=0
        )[shard_rank].clone()
    layer_sd["self_attn.o_norm.weight"] = ctx.read("o_norm")


def _build_mla_attn_sd(ctx, layer_sd):
    """MLA + DSA-indexer + kpool branch."""
    cfg = ctx.cfg
    mgt_sd, mgt_tp_0 = ctx.mgt_sd, ctx.mgt_tp_0
    model_key, pp, i = ctx.model_key, ctx.pp, ctx.i
    original_tp, target_tp, tp = ctx.original_tp, ctx.target_tp, ctx.tp
    first_rank_loaded = ctx.first_rank_loaded

    if is_nsa_enable_prefill_cp():
        dsa_tp_rank, dsa_tp_size = 0, 1
    else:
        dsa_tp_rank, dsa_tp_size = ctx.attn_tp_rank, ctx.attn_tp_size

    if cfg.q_lora_rank is None:
        layer_sd["self_attn.q_proj.weight"] = ctx.merge(
            "q_proj", attr="weight", target_tp=target_tp, current_tp=tp, slice_dim=0
        )
        layer_sd["self_attn.kv_a_proj_with_mqa.weight"] = _dict_access_multi(
            mgt_sd[pp][0],
            [model_key] + ctx.keys("kv_a_proj_with_mqa", attr="weight"),
        )
    else:
        q_a_proj_weight = torch.cat(
            [
                _dict_access_multi(
                    mgt_sd[pp][j + first_rank_loaded],
                    [model_key] + ctx.keys("q_a_proj", attr="weight"),
                )
                for j in range(original_tp)
            ]
        )
        kv_a_proj_with_mqa_weight = torch.cat(
            [
                _dict_access_multi(
                    mgt_sd[pp][j + first_rank_loaded],
                    [model_key] + ctx.keys("kv_a_proj_with_mqa", attr="weight"),
                )
                for j in range(original_tp)
            ]
        )
        fused_qkv_a_proj_with_mqa_weight = torch.cat(
            [q_a_proj_weight, kv_a_proj_with_mqa_weight]
        )
        if ctx.ifmtp:
            param_key = "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.weight"
        else:
            param_key = (
                f"model.layers.{ctx.layer_offset + i}"
                f".self_attn.fused_qkv_a_proj_with_mqa.weight"
            )
        param = ctx.params_dict[param_key]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, fused_qkv_a_proj_with_mqa_weight)
        layer_sd["self_attn.fused_qkv_a_proj_with_mqa.weight"] = param.data
        layer_sd["self_attn.q_a_layernorm.weight"] = ctx.read(
            "q_a_layernorm", attr="weight"
        )
        layer_sd["self_attn.q_b_proj.weight"] = ctx.merge(
            "q_b_proj",
            attr="weight",
            target_tp=dsa_tp_size,
            current_tp=dsa_tp_rank,
            slice_dim=0,
        )

    layer_sd["self_attn.kv_a_layernorm.weight"] = ctx.read(
        "kv_a_layernorm", attr="weight"
    )
    layer_sd["self_attn.kv_b_proj.weight"] = ctx.merge(
        "kv_b_proj",
        attr="weight",
        target_tp=dsa_tp_size,
        current_tp=dsa_tp_rank,
        slice_dim=0,
    )
    layer_sd["self_attn.o_proj.weight"] = ctx.merge(
        "dense",
        attr="weight",
        target_tp=dsa_tp_size,
        current_tp=dsa_tp_rank,
        slice_dim=1,
    )

    if getattr(cfg, "index_head_dim", None) is None:
        return

    # DSA indexer: rotate the rope half so it sits in the second half (the
    # mcore checkpoint stores it in the first half).
    wq_b = ctx.read("dsa_wq_b", attr="weight")
    wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
    wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
    layer_sd["self_attn.indexer.wq_b.weight"] = wq_b

    wk = ctx.read("dsa_wk", attr="weight")
    wk = torch.cat([wk[64:], wk[:64]], dim=0).view(-1, wk.shape[-1])
    layer_sd["self_attn.indexer.wk.weight"] = wk

    layer_sd["self_attn.indexer.weights_proj.weight"] = ctx.read(
        "dsa_weights_proj", attr="weight"
    )
    knorm_weight = ctx.read("dsa_k_norm", attr="weight")
    knorm_weight = torch.cat([knorm_weight[64:], knorm_weight[:64]], dim=0)
    layer_sd["self_attn.indexer.k_norm.weight"] = knorm_weight
    if getattr(cfg, "index_dsa_use_layernorm", False):
        knorm_bias = ctx.read("dsa_k_norm", attr="bias")
        knorm_bias = torch.cat([knorm_bias[64:], knorm_bias[:64]], dim=0)
        layer_sd["self_attn.indexer.k_norm.bias"] = knorm_bias

    # ``index_kpool > 1`` -> kpool enabled (Glm5NextConfig normalizes the
    # degenerate combo to 1).
    if getattr(cfg, "index_kpool", 1) > 1:
        ape = ctx.read("dsa_index_kpool_compress_ape")
        ape = torch.cat([ape[:, 64:], ape[:, :64]], dim=-1).contiguous()
        layer_sd["self_attn.indexer.index_kpool_compress_ape"] = ape

        gate_keys = ctx.keys("dsa_index_kpool_compress_gate")
        if not _has_keys(mgt_tp_0, gate_keys):
            gate_keys = ctx.keys("dsa_index_kpool_compress_gate_weight")
        gate = _dict_access_multi(mgt_tp_0, gate_keys)
        gate = torch.cat([gate[64:], gate[:64]], dim=0).contiguous()
        layer_sd["self_attn.indexer.index_kpool_compress_gate"] = gate


def _build_dense_attn_sd(ctx, layer_sd):
    """Plain (non-MLA, non-KDA) qkv-fused attention branch."""
    cfg = ctx.cfg
    layer = ctx.layer
    mgt_sd = ctx.mgt_sd
    model_key, pp, i = ctx.model_key, ctx.pp, ctx.i
    original_tp = ctx.original_tp
    first_rank_loaded = ctx.first_rank_loaded

    interleaved = getattr(cfg, "interleaved_qkv", True)

    def split_qkv_non_interleaved(sd):
        if layer.self_attention.multi_query_attention:
            head_d = layer.self_attention.head_dim
            return sd.split(
                [
                    cfg.num_attention_heads // original_tp * head_d,
                    cfg.num_key_value_heads // original_tp * head_d,
                    cfg.num_key_value_heads // original_tp * head_d,
                ],
                dim=0,
            )
        return sd.chunk(dim=0, chunks=3)

    def merge_qkv(sd_list):
        if interleaved:
            return torch.cat(sd_list, dim=0)
        q, k, v = [], [], []
        for sd in sd_list:
            q_, k_, v_ = split_qkv_non_interleaved(sd)
            q.append(q_.clone())
            k.append(k_.clone())
            v.append(v_.clone())
        return torch.cat(
            (torch.cat(q, dim=0), torch.cat(k, dim=0), torch.cat(v, dim=0)), dim=0
        )

    def interleaved_to_non(weight):
        weight = weight.view(cfg.num_key_value_heads, -1, cfg.hidden_size)
        q, k, v = weight.split(
            [
                cfg.num_attention_heads // cfg.num_key_value_heads * cfg.head_dim,
                cfg.head_dim,
                cfg.head_dim,
            ],
            dim=1,
        )
        return torch.cat(
            (
                q.reshape(-1, cfg.hidden_size),
                k.reshape(-1, cfg.hidden_size),
                v.reshape(-1, cfg.hidden_size),
            ),
            dim=0,
        )

    def interleaved_to_non_bias(weight):
        weight = weight.view(cfg.num_key_value_heads, -1)
        q, k, v = weight.split(
            [
                cfg.num_attention_heads // cfg.num_key_value_heads * cfg.head_dim,
                cfg.head_dim,
                cfg.head_dim,
            ],
            dim=1,
        )
        return torch.cat((q.reshape(-1), k.reshape(-1), v.reshape(-1)), dim=0)

    def load_qkv(attr, target_path):
        sd_list = [
            _dict_access_multi(
                mgt_sd[pp][j + first_rank_loaded],
                [model_key] + ctx.keys("query_key_value", attr=attr),
            )
            for j in range(original_tp)
        ]
        if interleaved:
            merged = (
                interleaved_to_non if attr == "weight" else interleaved_to_non_bias
            )(torch.cat(sd_list))
        else:
            merged = merge_qkv(sd_list)
        if ctx.ifmtp:
            param_key = f"model.decoder.self_attn.qkv_proj.{attr}"
        else:
            param_key = f"model.layers.{ctx.layer_offset + i}.self_attn.qkv_proj.{attr}"
        param = ctx.params_dict[param_key]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, merged)
        layer_sd[target_path] = param.data

    load_qkv("weight", "self_attn.qkv_proj.weight")
    if layer.self_attn.qkv_proj.bias is not None:
        load_qkv("bias", "self_attn.qkv_proj.bias")

    layer_sd["self_attn.o_proj.weight"] = ctx.merge(
        "dense",
        attr="weight",
        target_tp=ctx.attn_tp_size,
        current_tp=ctx.attn_tp_rank,
        slice_dim=1,
    )
    if layer.self_attn.o_proj.bias is not None:
        layer_sd["self_attn.o_proj.bias"] = ctx.read("dense", attr="bias")


def _build_dense_mlp_sd(ctx, layer_sd):
    """Dense (non-MoE) MLP block with GLU fusion."""
    layer = ctx.layer
    mlp_tp_rank, mlp_tp_size = ctx.mlp_tp_rank, ctx.mlp_tp_size

    layer_sd["mlp.gate_up_proj.weight"] = ctx.merge(
        "dense_h_to_4h",
        attr="weight",
        target_tp=mlp_tp_size,
        current_tp=mlp_tp_rank,
        merge_fn=_merge_glu,
        split_fn=_split_glu,
    )
    if layer.mlp.gate_up_proj.bias is not None:
        layer_sd["mlp.gate_up_proj.bias"] = ctx.merge(
            "dense_h_to_4h",
            attr="bias",
            target_tp=mlp_tp_size,
            current_tp=mlp_tp_rank,
            merge_fn=_merge_glu,
            split_fn=_split_glu,
        )
    layer_sd["mlp.down_proj.weight"] = ctx.merge(
        "dense_4h_to_h",
        attr="weight",
        target_tp=mlp_tp_size,
        current_tp=mlp_tp_rank,
        slice_dim=1,
    )
    if layer.mlp.down_proj.bias is not None:
        layer_sd["mlp.down_proj.bias"] = ctx.read("dense_4h_to_h", attr="bias")


def _build_moe_mlp_sd(ctx, layer_sd):
    """MoE MLP block: router + (optional) shared experts + grouped expert
    weights. Handles deepep / regular all-gather and disable_shared_experts_fusion.
    """
    cfg = ctx.cfg
    layer = ctx.layer
    mgt_sd, mgt_tp_0 = ctx.mgt_sd, ctx.mgt_tp_0
    model_key, pp, i = ctx.model_key, ctx.pp, ctx.i
    original_tp, original_ep, target_tp, tp = (
        ctx.original_tp,
        ctx.original_ep,
        ctx.target_tp,
        ctx.tp,
    )

    # ``is_ep`` controls how the routed-expert weights are sharded — under
    # deepep / EP-MoE the per-rank tensor is the full (un-sharded) weight.
    # Shared experts are different: their TP shard size is decided by the
    # model itself (see ``DeepseekV2MoE`` shared_experts construction —
    # tp_size is collapsed to 1 only for deepep / mooncake / nixl / mori /
    # ascend_fuseep / flashinfer / fp4-allgather backends, NOT just because
    # ``ep_size == tp_size``). Read the shard size off the live module so
    # the loader stays in lock-step with whatever the model picked.
    is_ep = get_moe_a2a_backend().is_deepep() or is_ep_moe_enabled()
    fuse_shared = not get_global_server_args().disable_shared_experts_fusion

    if hasattr(layer.mlp, "shared_experts"):
        shared_tp_size = getattr(
            layer.mlp.shared_experts.gate_up_proj, "tp_size", target_tp
        )
        shared_tp_rank = getattr(layer.mlp.shared_experts.gate_up_proj, "tp_rank", tp)
    else:
        shared_tp_size, shared_tp_rank = target_tp, tp

    layer_sd["mlp.gate.weight"] = ctx.read("moe.router")
    if getattr(cfg, "moe_router_enable_expert_bias", True):
        layer_sd["mlp.gate.e_score_correction_bias"] = ctx.read("moe.router_bias")

    if hasattr(layer.mlp, "shared_experts") and not fuse_shared:
        layer_sd["mlp.shared_experts.gate_up_proj.weight"] = ctx.merge(
            "moe.shared_experts.dense_h_to_4h",
            attr="weight",
            target_tp=shared_tp_size,
            current_tp=shared_tp_rank,
            merge_fn=_merge_glu,
            split_fn=_split_glu,
        )
        layer_sd["mlp.shared_experts.down_proj.weight"] = ctx.merge(
            "moe.shared_experts.dense_4h_to_h",
            attr="weight",
            target_tp=shared_tp_size,
            current_tp=shared_tp_rank,
            slice_dim=1,
        )

    if not isinstance(layer.mlp, GLM4MoESparseMoeBlock):
        raise ValueError(f"Unsupported expert type: {type(layer.mlp)}")

    assert original_tp <= target_tp
    if not is_ep:
        slot_count = cfg.n_routed_experts + (cfg.n_shared_experts if fuse_shared else 0)
        gate_up_list = [None] * slot_count
        down_list = [None] * slot_count

        pre_state_expert = cfg.n_routed_experts // original_ep
        rank_state_number = original_ep // target_tp if original_ep >= target_tp else 1
        ep_su = (
            pre_state_expert
            if original_ep >= target_tp
            else pre_state_expert // (target_tp // original_ep)
        )
        ep_offset = tp * ep_su % pre_state_expert
        world_size = get_tensor_model_parallel_world_size()
        tp_group = get_tensor_model_parallel_group().device_group

        for k in range(rank_state_number):
            assert original_ep * pre_state_expert >= target_tp
            sd_list = [
                ctx.read("moe.dense_h_to_4h", j=j)
                for j in range(ep_offset, ep_offset + ep_su)
            ]
            down_sd_list = [
                ctx.read("moe.dense_4h_to_h", j=j)
                for j in range(ep_offset, ep_offset + ep_su)
            ]
            for l in range(len(sd_list)):
                gate_ups = [
                    torch.empty_like(sd_list[l], device="cuda")
                    for _ in range(world_size)
                ]
                downs = [
                    torch.empty_like(down_sd_list[l], device="cuda")
                    for _ in range(world_size)
                ]
                torch.distributed.all_gather(
                    gate_ups, sd_list[l].cuda(), group=tp_group
                )
                torch.distributed.all_gather(
                    downs, down_sd_list[l].cuda(), group=tp_group
                )
                for rank_id in range(world_size):
                    expert_number = rank_id * ep_su + k * pre_state_expert + l
                    gate, up = gate_ups[rank_id].cpu().clone().chunk(2, dim=0)
                    gate_up_list[expert_number] = torch.cat(
                        [
                            gate.chunk(target_tp, dim=0)[tp],
                            up.chunk(target_tp, dim=0)[tp],
                        ],
                        dim=0,
                    )
                    down_list[expert_number] = (
                        downs[rank_id].cpu().clone().chunk(target_tp, dim=1)[tp]
                    )

        if fuse_shared:

            def gate_up_split_moe(sd, cnt, idx, dim=0):
                chunks_to_cat = []
                gate, up = sd.chunk(chunks=2, dim=dim)
                for k in range(cfg.n_shared_experts):
                    gate_chunk_k = gate.chunk(chunks=cfg.n_shared_experts, dim=dim)[k]
                    up_chunk_k = up.chunk(chunks=cfg.n_shared_experts, dim=dim)[k]
                    gate_chunk = gate_chunk_k.chunk(chunks=cnt, dim=dim)[idx].clone()
                    up_chunk = up_chunk_k.chunk(chunks=cnt, dim=dim)[idx].clone()
                    chunks_to_cat.append(torch.cat((gate_chunk, up_chunk), dim=dim))
                return chunks_to_cat

            def down_split_moe(sd, cnt, idx, dim=1):
                return [
                    sd.chunk(chunks=cfg.n_shared_experts, dim=dim)[k]
                    .chunk(chunks=cnt, dim=dim)[idx]
                    .clone()
                    for k in range(cfg.n_shared_experts)
                ]

            cnt = target_tp // original_tp
            tensor = _dict_access_multi(
                mgt_sd[pp][tp // cnt],
                [model_key]
                + ctx.keys("moe.shared_experts.dense_h_to_4h", attr="weight"),
            )
            gate_up_list[-cfg.n_shared_experts :] = gate_up_split_moe(
                tensor, cnt, tp % cnt
            )
            tensor = _dict_access_multi(
                mgt_sd[pp][tp // cnt],
                [model_key]
                + ctx.keys("moe.shared_experts.dense_4h_to_h", attr="weight"),
            )
            down_list[-cfg.n_shared_experts :] = down_split_moe(tensor, cnt, tp % cnt)
    else:
        gate_up_list, down_list = [], []
        per_rank = cfg.n_routed_experts // target_tp
        for j in range(per_rank * tp, per_rank * (tp + 1)):
            gate_up_list.append(ctx.read("moe.dense_h_to_4h", j=j))
            down_list.append(ctx.read("moe.dense_4h_to_h", j=j))
    layer_sd["mlp.experts.w13_weight"] = torch.stack(gate_up_list, dim=0)
    layer_sd["mlp.experts.w2_weight"] = torch.stack(down_list, dim=0)


def _apply_layer_sd(ctx, layer_sd):
    """Move ``layer_sd`` and the layer to ``device``, load it via
    ``load_state_dict``, filter benign duplicate-registration aliases, and
    bridge ``correction_bias`` for deepep MoE layers."""
    layer = ctx.layer
    device = ctx.device

    torch.cuda.empty_cache()
    layer = layer.to(device)
    # `RadixLinearAttention.conv_weights` / `.bias` are plain Python-attribute
    # views into `qkv_conv1d.weight` / `.bias`. Building layers on CPU and
    # only later moving them to GPU rebinds the underlying `.data`, so these
    # views go stale. Refresh them here so the workaround stays out of the
    # model code.
    sa = getattr(layer, "self_attn", None)
    if sa is not None and hasattr(sa, "qkv_conv1d") and hasattr(sa, "attn"):
        sa.attn.conv_weights = sa.qkv_conv1d.weight.squeeze(1)
        sa.attn.bias = sa.qkv_conv1d.bias
    for k in layer_sd:
        layer_sd[k] = layer_sd[k].to(device)
    missing_keys, unexpected_keys = layer.load_state_dict(layer_sd, strict=False)

    # A single nn.Parameter that lives on a parent module and is also passed
    # into a sub-module shows up under multiple paths in state_dict. layer_sd
    # only fills the canonical path, so load_state_dict reports the alias
    # paths as missing even though they share storage.
    path_to_pid = {
        path: id(param)
        for path, param in layer.named_parameters(remove_duplicate=False)
    }
    loaded_pids = {path_to_pid[k] for k in layer_sd if k in path_to_pid}
    missing_keys = [k for k in missing_keys if path_to_pid.get(k) not in loaded_pids]
    if missing_keys or unexpected_keys:
        logger.info(f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")
    if ctx.is_moe_layer and get_moe_a2a_backend().is_deepep():
        layer.mlp.correction_bias = layer.mlp.gate.e_score_correction_bias.data
        layer.mlp.correction_bias = layer.mlp.correction_bias.to(device)


def _load_one_layer(ctx):
    """Build the per-layer state dict and apply it to ``ctx.layer``."""
    layer_sd = {}
    _build_norms_layer_sd(ctx, layer_sd)

    if ctx.is_linear_layer:
        _build_kda_attn_sd(ctx, layer_sd)
    elif getattr(ctx.cfg, "mla", False):
        _build_mla_attn_sd(ctx, layer_sd)
    else:
        _build_dense_attn_sd(ctx, layer_sd)

    if ctx.is_moe_layer:
        _build_moe_mlp_sd(ctx, layer_sd)
    else:
        _build_dense_mlp_sd(ctx, layer_sd)

    _apply_layer_sd(ctx, layer_sd)


def _iterate_layers(
    init_model,
    cfg,
    params_dict,
    ifmtp,
    mgt_sd,
    get_keys,
    first_rank_loaded,
    original_tp,
    original_pp,
    original_ep,
    target_tp,
    tp,
    attn_tp_rank,
    attn_tp_size,
    mlp_tp_rank,
    mlp_tp_size,
    device,
):
    """Walk every transformer block found in ``mgt_sd``, build its layer
    state dict via the per-branch helpers, and apply it. Returns the total
    number of decoder layers that were loaded across PP ranks (the caller
    asserts this matches ``len(model.layers)`` on the non-MTP path)."""
    layer_offset = 0
    for model_key in sorted(mgt_sd[0][first_rank_loaded].keys()):
        if "model" not in model_key:
            continue
        for pp in range(original_pp):
            i = 0
            mgt_tp_0 = mgt_sd[pp][first_rank_loaded][model_key]
            while (
                _has_keys(mgt_tp_0, get_keys("input_layernorm", i=i, attr="weight"))
                or _has_keys(
                    mgt_tp_0,
                    get_keys("standalone.input_layernorm", i=i, attr="weight"),
                )
                or _has_keys(mgt_tp_0, get_keys("A_log", i=i))
            ):
                is_linear_layer = _has_keys(mgt_tp_0, get_keys("A_log", i=i))
                if ifmtp:
                    layer = init_model.model.decoder
                else:
                    layer = init_model.model.layers[layer_offset + i]
                is_moe_layer = _has_keys(
                    mgt_tp_0, get_keys("moe.router", i=i, attr="weight")
                )
                ctx = _LayerCtx(
                    cfg=cfg,
                    init_model=init_model,
                    params_dict=params_dict,
                    ifmtp=ifmtp,
                    original_tp=original_tp,
                    original_ep=original_ep,
                    target_tp=target_tp,
                    tp=tp,
                    mlp_tp_rank=mlp_tp_rank,
                    mlp_tp_size=mlp_tp_size,
                    attn_tp_rank=attn_tp_rank,
                    attn_tp_size=attn_tp_size,
                    mgt_sd=mgt_sd,
                    model_key=model_key,
                    pp=pp,
                    mgt_tp_0=mgt_tp_0,
                    get_keys=get_keys,
                    first_rank_loaded=first_rank_loaded,
                    layer=layer,
                    layer_offset=layer_offset,
                    i=i,
                    is_moe_layer=is_moe_layer,
                    is_linear_layer=is_linear_layer,
                    device=device,
                )
                _load_one_layer(ctx)
                i += 1
            layer_offset += i
    return layer_offset


@torch.no_grad()
def load_megatron_weights(
    init_model, checkpoint_path: str, params_dict=None, ifmtp: bool = False
):
    if ifmtp:
        logger.info("mtp weights loading!")
    # Install the megatron-tolerant unpickler BEFORE any torch.load call —
    # ``_ConfigView.from_checkpoint`` reads ``common.pt`` which references
    # megatron classes the inference runtime doesn't have.
    pickle.Unpickler = _UnpicklerWrapper

    cfg = _ConfigView.from_checkpoint(init_model, checkpoint_path)
    ckpt_dir = checkpoint_path

    checkpoint_format = "torch"
    if os.path.exists(os.path.join(checkpoint_path, ".metadata")):
        metadata = WrappedStorageReader(checkpoint_path).read_metadata()
        checkpoint_format = "torch_dist"
    st_time = time.time()

    ckpt_dir = Path(ckpt_dir)
    original_tp, original_pp, original_ep = 1, 1, 1
    if checkpoint_format == "torch":
        original_tp, original_pp, original_ep = count_tp_pp_ep(ckpt_dir)
    original_pp_enabled, original_ep_enabled = False, False
    if original_ep > 1:
        original_ep_enabled = True
    if original_pp > 1:
        original_pp_enabled = True

    target_tp = get_tensor_model_parallel_world_size()
    logger.info(f"Original TP={original_tp}, PP={original_pp}, EP={original_ep}")
    if target_tp <= original_tp:
        assert original_tp % target_tp == 0
        cnt = original_tp // target_tp
    else:
        assert target_tp % original_tp == 0
        cnt = target_tp // original_tp
    tp = get_tensor_model_parallel_rank()
    attn_tp_rank = get_attention_tp_rank()
    attn_tp_size = get_attention_tp_size()
    if enable_moe_dense_fully_dp():
        mlp_tp_rank, mlp_tp_size = 0, 1
    else:
        mlp_tp_rank, mlp_tp_size = tp, target_tp
    device = torch.cuda.current_device()

    if checkpoint_format == "torch" and original_ep_enabled:
        mgt_sd = _read_torch_ep_ckpt(
            ckpt_dir, original_tp, original_pp, original_ep, cfg, tp
        )
    elif checkpoint_format == "torch_dist":
        mgt_sd = _read_torch_dist_ckpt(
            checkpoint_path, metadata, ifmtp, cfg, target_tp, tp, st_time
        )
    else:
        mgt_sd = _read_torch_legacy_ckpt(
            ckpt_dir, original_tp, original_pp, original_pp_enabled, target_tp, tp, cnt
        )

    first_rank_loaded = cnt * tp if target_tp <= original_tp else tp // cnt
    impl = (
        "mgt"
        if "model" in mgt_sd[0][first_rank_loaded]
        and "language_model" in mgt_sd[0][first_rank_loaded]["model"]
        else "mcore"
    )

    key_map = _build_key_map(ifmtp)

    def get_keys(key, **kwargs):
        return key_map[key][impl][:-1] + [key_map[key][impl][-1].format(**kwargs)]

    vp_enabled = "model0" in mgt_sd[0][first_rank_loaded]
    first_model_in_pp = "model0" if vp_enabled else "model"
    last_model_in_pp = (
        f"model{sum(['model' in key for key in mgt_sd[0][first_rank_loaded].keys()]) - 1}"
        if vp_enabled
        else "model"
    )

    # Embedding
    _load_embedding(
        init_model,
        mgt_sd,
        get_keys,
        first_model_in_pp,
        original_tp,
        target_tp,
        tp,
        device,
        original_pp,
    )

    layer_offset = _iterate_layers(
        init_model,
        cfg,
        params_dict,
        ifmtp,
        mgt_sd,
        get_keys,
        first_rank_loaded,
        original_tp,
        original_pp,
        original_ep,
        target_tp,
        tp,
        attn_tp_rank,
        attn_tp_size,
        mlp_tp_rank,
        mlp_tp_size,
        device,
    )

    _load_lm_head(
        init_model,
        cfg,
        mgt_sd,
        get_keys,
        last_model_in_pp,
        original_tp,
        target_tp,
        tp,
        device,
    )

    if ifmtp:
        _load_mtp_heads(
            init_model, mgt_sd, get_keys, last_model_in_pp, first_rank_loaded, device
        )
        if torch.distributed.get_rank() == 0:
            logger.info(f"total loading time: {time.time() - st_time}")
    else:
        assert layer_offset == len(init_model.model.layers), (
            f"layer_offset: {layer_offset}, "
            f"len(self.layers): {len(init_model.model.layers)}"
        )
        _load_final_layernorm(
            init_model, mgt_sd, get_keys, last_model_in_pp, first_rank_loaded, device
        )
        if torch.distributed.get_rank() == 0:
            logger.info(f"total loading time: {time.time() - st_time}")
        _record_consumed_train_samples(init_model, mgt_sd, ckpt_dir)
