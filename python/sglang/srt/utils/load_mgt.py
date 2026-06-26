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


@torch.no_grad()
def load_megatron_weights(
    init_model, checkpoint_path: str, params_dict=None, ifmtp: bool = False
):
    if ifmtp:
        logger.info("mtp weights loading!")
    ckpt_dir = checkpoint_path

    class UnpicklerWrapper(pickle.Unpickler):
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

    pickle.Unpickler = UnpicklerWrapper

    checkpoint_format = "torch"
    if os.path.exists(os.path.join(checkpoint_path, ".metadata")):
        metadata = WrappedStorageReader(checkpoint_path).read_metadata()
        checkpoint_format = "torch_dist"
    st_time = time.time()

    def dict_access_multi(a_dict, keys):
        if len(keys) == 0:
            return a_dict
        return dict_access_multi(a_dict[keys[0]], keys[1:])

    def merge_tensors(
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
                dict_access_multi(tp_sd[i + offset], [model_key] + keys)
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
            tensor = dict_access_multi(tp_sd[current_tp // cnt], [model_key] + keys)
            if slice_dim is not None:
                return torch.chunk(tensor, cnt, dim=slice_dim)[current_tp % cnt].clone()
            assert split_fn is not None
            # print("merge tensor shape {}".format(tensor.shape))
            return split_fn(tensor, cnt, current_tp % cnt)

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
        mgt_sd = [[None for j in range(original_tp)] for i in range(original_pp)]
        if init_model.config.ep_tp_transpose:
            # Distributed read EP and use TP instead
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
                        ckpt_dir
                        / f"mp_rank_{i:02d}_{ep_rank:03d}"
                        / "model_optim_rng.pt"
                    )
                else:
                    ckpt_file = (
                        ckpt_dir
                        / f"mp_rank_{i:02d}_{j:03d}_{ep_rank:03d}"
                        / "model_optim_rng.pt"
                    )

                try:
                    ep_checkpoint = torch.load(ckpt_file, map_location="cpu")
                except:
                    ep_checkpoint = {}
                    logger.info(f"{ckpt_file} Error")
                    raise
                mgt_sd[j][i] = ep_checkpoint
        else:
            # Read and merge all ep_ranks
            for i, j in itertools.product(range(original_tp), range(original_pp)):
                ep_checkpoints = []
                for k in range(original_ep):
                    if original_pp == 1:
                        ckpt_file = (
                            ckpt_dir / f"mp_rank_{i:02d}_{k:03d}" / "model_optim_rng.pt"
                        )
                    else:
                        ckpt_file = (
                            ckpt_dir
                            / f"mp_rank_{i:02d}_{j:03d}_{k:03d}"
                            / "model_optim_rng.pt"
                        )
                    logger.info(f"{ckpt_file} {os.path.exists(ckpt_file)}")
                    ep_checkpoints.append(torch.load(ckpt_file, map_location="cpu"))
                for k in ep_checkpoints[0].keys():
                    if k.startswith("model"):
                        merged_checkpoint_key = merge_ep(
                            [c[k] for c in ep_checkpoints],
                            init_model.config.n_routed_experts
                            // original_ep,  # experts_per_rank
                            init_model.config.moe_ffn_hidden_size,
                        )
                        ep_checkpoints[0][k] = merged_checkpoint_key
                mgt_sd[j][i] = ep_checkpoints[0]
    elif checkpoint_format == "torch_dist":
        mgt_sd = [[{}]]
        attn_state_dict, attn_keys, attn_dicts = load_distcp(
            checkpoint_path,
            metadata,
            ifattn=True,
            ifnextn=ifmtp,
            total_num_experts=init_model.config.n_routed_experts,
            layer=init_model.config.num_hidden_layers,
        )
        mgt_sd[0][0]["model"] = attn_state_dict
        common_pt = os.path.join(checkpoint_path, "common.pt")
        if os.path.exists(common_pt):
            mgt_sd[0][0]["args"] = torch.load(common_pt, weights_only=False)["args"]
        # communicate for attention
        communicate_for_attn(mgt_sd[0][0]["model"], attn_keys, target_tp, tp)
        del attn_state_dict
        if torch.distributed.get_rank() == 0:
            logger.info(f"loading attention distcp time: {time.time() - st_time}")

        moe_dicts = {}
        if init_model.config.n_routed_experts is not None:
            moe_state_dict, moe_keys, moe_dicts = load_distcp(
                checkpoint_path,
                metadata,
                ifattn=False,
                ifnextn=ifmtp,
                total_num_experts=init_model.config.n_routed_experts,
                layer=init_model.config.num_hidden_layers,
            )
            mgt_sd[0][0]["model"].update(moe_state_dict)

            if torch.distributed.get_rank() == 0:
                logger.info(f"loading distcp total time: {time.time() - st_time}")
        # TODO: postprocess before comunicate_for_attn (load moe--->load attn--->postprocess--->communicate for attn)
        postprocess(mgt_sd, attn_dicts, moe_dicts)

        # avoid cast to bf16 with tp
        hc_state_dict, hc_keys, hc_dicts = load_distcp(
            checkpoint_path,
            metadata,
            ifattn=False,
            ifnextn=ifmtp,
            total_num_experts=init_model.config.n_routed_experts,
            layer=init_model.config.num_hidden_layers,
            load_hc=True,
        )
        for key in hc_state_dict:
            mgt_sd[0][0]["model"][key] = hc_state_dict[key].view(
                mgt_sd[0][0]["model"][key].shape
            )

        # if tp == 0:
        #     for key in mgt_sd[0][0]["model"].keys():
        #         print(key, mgt_sd[0][0]["model"][key].shape)
    else:
        mgt_sd = [
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

    first_rank_loaded = cnt * tp if target_tp <= original_tp else tp // cnt
    impl = (
        "mgt"
        if "model" in mgt_sd[0][first_rank_loaded]
        and "language_model" in mgt_sd[0][first_rank_loaded]["model"]
        else "mcore"
    )

    key_map = {
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
            "mcore": [
                "decoder.layers.{i}.self_attention.index_kpool_compress_gate.weight"
            ],
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
        "post_mlp_layernorm": {
            "mcore": ["decoder.layers.{i}.post_mlp_layernorm.{attr}"]
        },
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
    }

    for linear_key in "qkv":
        key_map[f"{linear_key}_proj"] = {
            "mcore": [
                "decoder.layers.{i}" + f".self_attention.{linear_key}_proj.weight"
            ],
        }
        key_map[f"{linear_key}_conv1d"] = {
            "mcore": [
                "decoder.layers.{i}" + f".self_attention.{linear_key}_conv1d.weight"
            ],
        }

    if ifmtp:
        key_map.update(
            {
                "word_embeddings": {
                    "mcore": ["embedding.word_embeddings.weight"],
                },
                "eh_proj": {"mcore": ["mtp.layers.{i}.eh_proj.weight"]},
                "moe.shared_experts.dense_h_to_4h": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.mlp.shared_experts.linear_fc1.{attr}"
                    ]
                },
                "moe.shared_experts.dense_4h_to_h": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.mlp.shared_experts.linear_fc2.{attr}"
                    ]
                },
                "query_key_value": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.self_attention.linear_qkv.{attr}"
                    ],
                },
                "dense": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.self_attention.linear_proj.{attr}"
                    ],
                },
                "moe.router": {
                    "mcore": ["mtp.layers.{i}.transformer_layer.mlp.router.weight"]
                },
                "enorm": {"mcore": ["mtp.layers.{i}.enorm.weight"]},
                "final_layernorm": {
                    "mcore": ["mtp.layers.{i}.final_layernorm.weight"],
                },
                "hnorm": {"mcore": ["mtp.layers.{i}.hnorm.weight"]},
                "post_attention_layernorm": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.pre_mlp_layernorm.{attr}"
                    ]
                },
                "input_layernorm": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.self_attention.linear_qkv.layer_norm_{attr}"
                    ]
                },
                "moe.router_bias": {
                    "mcore": ["mtp.layers.{i}.transformer_layer.mlp.router.expert_bias"]
                },
                "q_layernorm": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.self_attention.q_layernorm.{attr}"
                    ],
                },
                "k_layernorm": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.self_attention.k_layernorm.{attr}"
                    ],
                },
                "moe.dense_h_to_4h": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.mlp.experts.linear_fc1.weight{j}"
                    ],
                },
                "moe.dense_4h_to_h": {
                    "mcore": [
                        "mtp.layers.{i}.transformer_layer.mlp.experts.linear_fc2.weight{j}"
                    ],
                },
                "output_layer": {
                    "mcore": ["output_layer.weight"],
                },
            }
        )

    def get_keys(key, **kwargs):
        return key_map[key][impl][:-1] + [key_map[key][impl][-1].format(**kwargs)]

    def has_keys(a_dict, keys):
        try:
            dict_access_multi(a_dict, keys)
            return True
        except:
            return False

    vp_enabled = "model0" in mgt_sd[0][first_rank_loaded]
    first_model_in_pp = "model0" if vp_enabled else "model"
    last_model_in_pp = (
        f"model{sum(['model' in key for key in mgt_sd[0][first_rank_loaded].keys()]) - 1}"
        if vp_enabled
        else "model"
    )

    # Embedding
    for pp in range(original_pp):
        try:
            init_model.model.embed_tokens = init_model.model.embed_tokens.to(device)
            init_model.model.embed_tokens.weight.copy_(
                merge_tensors(
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
            break
    else:
        raise KeyError("word_embeddings not found in checkpoint")

    layer_offset = 0
    for model_key in sorted(mgt_sd[0][first_rank_loaded].keys()):
        # model_index = 0 if model_key == 'model' else model_key.split('model')[1]
        if "model" not in model_key:
            continue
        for pp in range(original_pp):
            i = 0
            mgt_tp_0 = mgt_sd[pp][first_rank_loaded][model_key]
            while (
                has_keys(mgt_tp_0, get_keys("input_layernorm", i=i, attr="weight"))
                or has_keys(
                    mgt_tp_0, get_keys("standalone.input_layernorm", i=i, attr="weight")
                )
                or has_keys(mgt_tp_0, get_keys("A_log", i=i))
            ):
                is_linear_layer = has_keys(mgt_tp_0, get_keys("A_log", i=i))
                if ifmtp:
                    layer = init_model.model.decoder
                else:
                    layer = init_model.model.layers[layer_offset + i]
                is_moe_layer = has_keys(
                    mgt_tp_0, get_keys("moe.router", i=i, attr="weight")
                )
                layer_sd = {
                    "input_layernorm.weight": dict_access_multi(
                        mgt_tp_0,
                        get_keys(
                            (
                                "standalone.input_layernorm"
                                if getattr(init_model.config, "mla", False)
                                else "input_layernorm"
                            ),
                            i=i,
                            attr="weight",
                        ),
                    ),
                    "post_attention_layernorm.weight": dict_access_multi(
                        mgt_tp_0,
                        get_keys(
                            (
                                "post_attention_layernorm"
                                if is_moe_layer
                                else "standalone.post_attention_layernorm"
                            ),
                            i=i,
                            attr="weight",
                        ),
                    ),
                }

                if getattr(init_model.config, "mhc", False):
                    # mcore mhtk ckpt layout: per-prefix {mapping_proj.weight,
                    # norm_weight, scale, base}. The sglang model owns these
                    # as plain nn.Parameter on the layer with names
                    # hc_{attn,ffn}_{base, scale, fn}. When the training-time
                    # config disabled the norm (mhc_no_norm_weight=True), the
                    # ckpt still carries norm_weight but it was NOT used in
                    # forward — so we must NOT fold it in. Only fold when the
                    # config says the norm is active.
                    mhc_fold_norm = not getattr(
                        init_model.config, "mhc_no_norm_weight", True
                    )
                    for hc_prefix, dst_prefix in (
                        ("self_attention_hyper_connection", "hc_attn"),
                        ("mlp_hyper_connection", "hc_ffn"),
                    ):
                        base = dict_access_multi(
                            mgt_tp_0, get_keys(f"{hc_prefix}.bias", i=i)
                        )
                        if base.numel() == 1:
                            base = base.unsqueeze(0)
                        layer_sd[f"{dst_prefix}_base"] = base

                        scale = dict_access_multi(
                            mgt_tp_0, get_keys(f"{hc_prefix}.scale", i=i)
                        )
                        if scale.numel() == 1:
                            scale = scale.unsqueeze(0)
                        layer_sd[f"{dst_prefix}_scale"] = scale

                        fn_weight = dict_access_multi(
                            mgt_tp_0, get_keys(f"{hc_prefix}.mapping_proj.weight", i=i)
                        )
                        if mhc_fold_norm:
                            norm_keys = get_keys(f"{hc_prefix}.norm.weight", i=i)
                            assert has_keys(mgt_tp_0, norm_keys), (
                                f"[mHC fold] mhc_no_norm_weight=False but ckpt is "
                                f"missing {norm_keys}"
                            )
                            norm_w = dict_access_multi(mgt_tp_0, norm_keys)
                            fn_weight = fn_weight * norm_w
                        layer_sd[f"{dst_prefix}_fn"] = fn_weight

                if getattr(init_model.config, "use_qk_norm", False) and not getattr(
                    init_model.config, "mla", False
                ):
                    layer_sd["self_attn.q_norm.weight"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys(
                            "q_layernorm",
                            i=i,
                            attr="weight",
                        ),
                    )
                    layer_sd["self_attn.k_norm.weight"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys(
                            "k_layernorm",
                            i=i,
                            attr="weight",
                        ),
                    )
                if init_model.config.post_self_attn_layernorm:
                    layer_sd["post_self_attn_layernorm.weight"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys("post_self_attn_layernorm", i=i, attr="weight"),
                    )
                if init_model.config.post_mlp_layernorm:
                    layer_sd["post_mlp_layernorm.weight"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys("post_mlp_layernorm", i=i, attr="weight"),
                    )

                def split_qkv_from_non_interleaved(sd):
                    if layer.self_attention.multi_query_attention:
                        return sd.split(
                            [
                                init_model.config.num_attention_heads
                                // original_tp
                                * layer.self_attention.head_dim,
                                init_model.config.num_key_value_heads
                                // original_tp
                                * layer.self_attention.head_dim,
                                init_model.config.num_key_value_heads
                                // original_tp
                                * layer.self_attention.head_dim,
                            ],
                            dim=0,
                        )
                    else:
                        return sd.chunk(dim=0, chunks=3)

                def merge_qkv(sd_list):
                    if getattr(init_model.config, "interleaved_qkv", True):
                        return torch.cat(sd_list, dim=0)
                    q, k, v = [], [], []
                    for sd in sd_list:
                        q_, k_, v_ = split_qkv_from_non_interleaved(sd)
                        q.append(q_.clone())
                        k.append(k_.clone())
                        v.append(v_.clone())
                    return torch.cat(
                        (
                            torch.cat(q, dim=0),
                            torch.cat(k, dim=0),
                            torch.cat(v, dim=0),
                        ),
                        dim=0,
                    )

                def transformer_interleaved_to_non_interleaved(weight):
                    weight = weight.view(
                        init_model.config.num_key_value_heads,
                        -1,
                        init_model.config.hidden_size,
                    )
                    q, k, v = weight.split(
                        [
                            (
                                init_model.config.num_attention_heads
                                // init_model.config.num_key_value_heads
                                * init_model.config.head_dim
                            ),
                            init_model.config.head_dim,
                            init_model.config.head_dim,
                        ],
                        dim=1,
                    )
                    q = q.reshape(-1, init_model.config.hidden_size)
                    k = k.reshape(-1, init_model.config.hidden_size)
                    v = v.reshape(-1, init_model.config.hidden_size)
                    return torch.cat((q, k, v), dim=0)

                def transformer_interleaved_to_non_interleaved_bias(weight):
                    weight = weight.view(init_model.config.num_key_value_heads, -1)
                    q, k, v = weight.split(
                        [
                            (
                                init_model.config.num_attention_heads
                                // init_model.config.num_key_value_heads
                                * init_model.config.head_dim
                            ),
                            init_model.config.head_dim,
                            init_model.config.head_dim,
                        ],
                        dim=1,
                    )
                    q = q.reshape(-1)
                    k = k.reshape(-1)
                    v = v.reshape(-1)
                    return torch.cat((q, k, v), dim=0)

                if is_linear_layer:
                    logger.info(f"{i} loading linear layer")
                    # Mirrors the model's fusion switch in
                    # Glm5NextLinearAttention.__init__: when fused, the
                    # state_dict must carry the merged param names
                    # (fused_qkvbfg_a_proj.weight, fused_fg_b_proj.weight)
                    # instead of the per-proj names.
                    from sglang.srt.environ import envs as _envs

                    do_fuse_qkvbfg = _envs.SGLANG_GLM5_NEXT_FUSE_QKVBFG.get()

                    # Mirror Glm5NextLinearAttention's head-shard choice:
                    # KDA heads are sharded by CP when NSA prefill CP is on,
                    # otherwise by attention TP. All KDA projections,
                    # qkv_conv1d, A_log and dt_bias must use the same shard.
                    if is_nsa_enable_prefill_cp():
                        kda_shard_size = get_attention_cp_size()
                        kda_shard_rank = get_attention_cp_rank()
                    else:
                        kda_shard_size = attn_tp_size
                        kda_shard_rank = attn_tp_rank

                    qkv_conv_weights = []
                    for linear_key in "qkv":
                        conv_weight = merge_tensors(
                            tp_sd=mgt_sd[pp],
                            model_key=model_key,
                            keys=get_keys(f"{linear_key}_conv1d", i=i),
                            original_tp=original_tp,
                            target_tp=kda_shard_size,
                            current_tp=kda_shard_rank,
                            merge_fn=None,
                            slice_dim=0,
                        )
                        if hasattr(layer.self_attn, "qkv_conv1d"):
                            qkv_conv_weights.append(conv_weight)
                        else:
                            layer_sd[f"self_attn.{linear_key}_conv1d.weight"] = (
                                conv_weight
                            )
                    if qkv_conv_weights:
                        layer_sd["self_attn.qkv_conv1d.weight"] = torch.cat(
                            qkv_conv_weights, dim=0
                        )

                    if do_fuse_qkvbfg:
                        # Order must match Glm5NextForCausalLM._STACKED_PARAMS_MAPPING
                        # for fused_qkvbfg_a_proj: q, k, v, b (column-parallel)
                        # then f_a, g_a (replicated).
                        fused_a_parts = []
                        for linear_key in ("q", "k", "v", "b"):
                            fused_a_parts.append(
                                merge_tensors(
                                    tp_sd=mgt_sd[pp],
                                    model_key=model_key,
                                    keys=get_keys(f"{linear_key}_proj", i=i),
                                    original_tp=original_tp,
                                    target_tp=kda_shard_size,
                                    current_tp=kda_shard_rank,
                                    merge_fn=None,
                                    slice_dim=0,
                                )
                            )
                        for linear_key in ("f_a", "g_a"):
                            fused_a_parts.append(
                                merge_tensors(
                                    tp_sd=mgt_sd[pp],
                                    model_key=model_key,
                                    keys=get_keys(f"{linear_key}_proj", i=i),
                                    original_tp=original_tp,
                                    target_tp=1,
                                    current_tp=0,
                                    merge_fn=None,
                                    slice_dim=0,
                                )
                            )
                        layer_sd["self_attn.fused_qkvbfg_a_proj.weight"] = torch.cat(
                            fused_a_parts, dim=0
                        )

                        # fused_fg_b_proj: stack(f_b, g_b) along batch dim 0,
                        # each column-parallel along its own output dim.
                        f_b = merge_tensors(
                            tp_sd=mgt_sd[pp],
                            model_key=model_key,
                            keys=get_keys("f_b_proj", i=i),
                            original_tp=original_tp,
                            target_tp=kda_shard_size,
                            current_tp=kda_shard_rank,
                            merge_fn=None,
                            slice_dim=0,
                        )
                        g_b = merge_tensors(
                            tp_sd=mgt_sd[pp],
                            model_key=model_key,
                            keys=get_keys("g_b_proj", i=i),
                            original_tp=original_tp,
                            target_tp=kda_shard_size,
                            current_tp=kda_shard_rank,
                            merge_fn=None,
                            slice_dim=0,
                        )
                        layer_sd["self_attn.fused_fg_b_proj.weight"] = torch.stack(
                            [f_b, g_b], dim=0
                        )

                        layer_sd["self_attn.o_proj.weight"] = merge_tensors(
                            tp_sd=mgt_sd[pp],
                            model_key=model_key,
                            keys=get_keys("o_proj", i=i),
                            original_tp=original_tp,
                            target_tp=kda_shard_size,
                            current_tp=kda_shard_rank,
                            merge_fn=None,
                            slice_dim=1,
                        )
                    else:
                        qkv_proj_weights = []
                        for linear_key in "qkv":
                            proj_weight = merge_tensors(
                                tp_sd=mgt_sd[pp],
                                model_key=model_key,
                                keys=get_keys(f"{linear_key}_proj", i=i),
                                original_tp=original_tp,
                                target_tp=kda_shard_size,
                                current_tp=kda_shard_rank,
                                merge_fn=None,
                                slice_dim=0,
                            )
                            if hasattr(layer.self_attn, "qkv_proj"):
                                qkv_proj_weights.append(proj_weight)
                            else:
                                layer_sd[f"self_attn.{linear_key}_proj.weight"] = (
                                    proj_weight
                                )
                        if qkv_proj_weights:
                            layer_sd["self_attn.qkv_proj.weight"] = torch.cat(
                                qkv_proj_weights, dim=0
                            )
                        for linear_key in "g_a g_b f_a f_b b o".split():
                            slice_dim = 1 if linear_key == "o" else 0
                            layer_sd[f"self_attn.{linear_key}_proj.weight"] = (
                                merge_tensors(
                                    tp_sd=mgt_sd[pp],
                                    model_key=model_key,
                                    keys=get_keys(f"{linear_key}_proj", i=i),
                                    original_tp=original_tp,
                                    target_tp=(
                                        kda_shard_size
                                        if linear_key not in ["f_a", "g_a"]
                                        else 1
                                    ),
                                    current_tp=(
                                        kda_shard_rank
                                        if linear_key not in ["f_a", "g_a"]
                                        else 0
                                    ),
                                    merge_fn=None,
                                    slice_dim=slice_dim,
                                )
                            )

                    layer_sd["self_attn.A_log"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys("A_log", i=i),
                    ).view(1, 1, -1, 1)
                    if kda_shard_size > 1:
                        layer_sd["self_attn.A_log"] = torch.chunk(
                            layer_sd["self_attn.A_log"], kda_shard_size, dim=2
                        )[kda_shard_rank].clone()
                    layer_sd["self_attn.dt_bias"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys("dt_bias", i=i),
                    )
                    if kda_shard_size > 1:
                        layer_sd["self_attn.dt_bias"] = torch.chunk(
                            layer_sd["self_attn.dt_bias"], kda_shard_size, dim=0
                        )[kda_shard_rank].clone()
                    layer_sd["self_attn.o_norm.weight"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys("o_norm", i=i),
                    )

                elif getattr(init_model.config, "mla", False):
                    if is_nsa_enable_prefill_cp():
                        dsa_tp_rank, dsa_tp_size = 0, 1
                    else:
                        dsa_tp_rank, dsa_tp_size = attn_tp_rank, attn_tp_size
                    if init_model.config.q_lora_rank is None:
                        layer_sd["self_attn.q_proj.weight"] = merge_tensors(
                            tp_sd=mgt_sd[pp],
                            model_key=model_key,
                            keys=get_keys("q_proj", i=i, attr="weight"),
                            original_tp=original_tp,
                            target_tp=target_tp,
                            current_tp=tp,
                            merge_fn=None,
                            slice_dim=0,
                        )
                        layer_sd["self_attn.kv_a_proj_with_mqa.weight"] = (
                            dict_access_multi(
                                mgt_sd[pp][0],
                                [model_key]
                                + get_keys("kv_a_proj_with_mqa", i=i, attr="weight"),
                            )
                        )
                    else:
                        q_a_proj_weight = torch.cat(
                            [
                                dict_access_multi(
                                    mgt_sd[pp][j + first_rank_loaded],
                                    [model_key]
                                    + get_keys("q_a_proj", i=i, attr="weight"),
                                )
                                for j in range(original_tp)
                            ]
                        )
                        kv_a_proj_with_mqa_weight = torch.cat(
                            [
                                dict_access_multi(
                                    mgt_sd[pp][j + first_rank_loaded],
                                    [model_key]
                                    + get_keys(
                                        "kv_a_proj_with_mqa", i=i, attr="weight"
                                    ),
                                )
                                for j in range(original_tp)
                            ]
                        )
                        fused_qkv_a_proj_with_mqa_weight = torch.cat(
                            [q_a_proj_weight, kv_a_proj_with_mqa_weight]
                        )
                        if ifmtp:
                            param = params_dict[
                                f"model.decoder.self_attn.fused_qkv_a_proj_with_mqa.weight"
                            ]
                        else:
                            param = params_dict[
                                f"model.layers.{layer_offset + i}.self_attn.fused_qkv_a_proj_with_mqa.weight"
                            ]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, fused_qkv_a_proj_with_mqa_weight)
                        layer_sd["self_attn.fused_qkv_a_proj_with_mqa.weight"] = (
                            param.data
                        )
                        layer_sd["self_attn.q_a_layernorm.weight"] = dict_access_multi(
                            mgt_tp_0,
                            get_keys("q_a_layernorm", i=i, attr="weight"),
                        )
                        layer_sd["self_attn.q_b_proj.weight"] = merge_tensors(
                            tp_sd=mgt_sd[pp],
                            model_key=model_key,
                            keys=get_keys("q_b_proj", i=i, attr="weight"),
                            original_tp=original_tp,
                            target_tp=dsa_tp_size,
                            current_tp=dsa_tp_rank,
                            merge_fn=None,
                            slice_dim=0,
                        )
                    layer_sd["self_attn.kv_a_layernorm.weight"] = dict_access_multi(
                        mgt_tp_0,
                        get_keys("kv_a_layernorm", i=i, attr="weight"),
                    )
                    layer_sd["self_attn.kv_b_proj.weight"] = merge_tensors(
                        tp_sd=mgt_sd[pp],
                        model_key=model_key,
                        keys=get_keys("kv_b_proj", i=i, attr="weight"),
                        original_tp=original_tp,
                        target_tp=dsa_tp_size,
                        current_tp=dsa_tp_rank,
                        merge_fn=None,
                        slice_dim=0,
                    )
                    layer_sd[f"self_attn.o_proj.weight"] = merge_tensors(
                        tp_sd=mgt_sd[pp],
                        model_key=model_key,
                        keys=get_keys("dense", i=i, attr="weight"),
                        original_tp=original_tp,
                        target_tp=dsa_tp_size,
                        current_tp=dsa_tp_rank,
                        slice_dim=1,
                    )
                    if getattr(init_model.config, "index_head_dim", None) is not None:
                        layer_sd["self_attn.indexer.wq_b.weight"] = dict_access_multi(
                            mgt_tp_0,
                            get_keys("dsa_wq_b", i=i, attr="weight"),
                        )
                        wq_b = layer_sd["self_attn.indexer.wq_b.weight"]
                        # our training use last half to rope, but in dsa they use first half
                        wq_b = wq_b.view(-1, 128, wq_b.shape[-1])  # hard code 128
                        wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(
                            -1, wq_b.shape[-1]
                        )
                        layer_sd["self_attn.indexer.wq_b.weight"] = wq_b

                        layer_sd["self_attn.indexer.wk.weight"] = dict_access_multi(
                            mgt_tp_0,
                            get_keys("dsa_wk", i=i, attr="weight"),
                        )
                        wk = layer_sd["self_attn.indexer.wk.weight"]
                        wk = torch.cat([wk[64:], wk[:64]], dim=0).view(-1, wk.shape[-1])
                        layer_sd["self_attn.indexer.wk.weight"] = wk

                        layer_sd["self_attn.indexer.weights_proj.weight"] = (
                            dict_access_multi(
                                mgt_tp_0,
                                get_keys("dsa_weights_proj", i=i, attr="weight"),
                            )
                        )
                        if getattr(init_model.config, "index_dsa_use_layernorm", False):
                            layer_sd["self_attn.indexer.k_norm.weight"] = (
                                dict_access_multi(
                                    mgt_tp_0,
                                    get_keys("dsa_k_norm", i=i, attr="weight"),
                                )
                            )
                            layer_sd["self_attn.indexer.k_norm.bias"] = (
                                dict_access_multi(
                                    mgt_tp_0,
                                    get_keys("dsa_k_norm", i=i, attr="bias"),
                                )
                            )
                        else:
                            layer_sd["self_attn.indexer.k_norm.weight"] = (
                                dict_access_multi(
                                    mgt_tp_0,
                                    get_keys("dsa_k_norm", i=i, attr="weight"),
                                )
                            )
                        knorm_weight = layer_sd["self_attn.indexer.k_norm.weight"]
                        knorm_weight = torch.cat(
                            [knorm_weight[64:], knorm_weight[:64]], dim=0
                        )
                        layer_sd["self_attn.indexer.k_norm.weight"] = knorm_weight
                        if getattr(init_model.config, "index_dsa_use_layernorm", False):
                            knorm_bias = layer_sd["self_attn.indexer.k_norm.bias"]
                            knorm_bias = torch.cat(
                                [knorm_bias[64:], knorm_bias[:64]], dim=0
                            )
                            layer_sd["self_attn.indexer.k_norm.bias"] = knorm_bias
                        # ``index_kpool > 1`` -> kpool enabled (Glm5NextConfig
                        # normalizes the degenerate combo to 1).
                        if getattr(init_model.config, "index_kpool", 1) > 1:
                            ape = dict_access_multi(
                                mgt_tp_0,
                                get_keys("dsa_index_kpool_compress_ape", i=i),
                            )
                            ape = torch.cat(
                                [ape[:, 64:], ape[:, :64]], dim=-1
                            ).contiguous()
                            layer_sd["self_attn.indexer.index_kpool_compress_ape"] = ape

                            gate_keys = get_keys("dsa_index_kpool_compress_gate", i=i)
                            if not has_keys(mgt_tp_0, gate_keys):
                                gate_keys = get_keys(
                                    "dsa_index_kpool_compress_gate_weight", i=i
                                )
                            gate = dict_access_multi(mgt_tp_0, gate_keys)
                            gate = torch.cat([gate[64:], gate[:64]], dim=0).contiguous()
                            layer_sd["self_attn.indexer.index_kpool_compress_gate"] = (
                                gate
                            )
                else:
                    keys = get_keys("query_key_value", i=i, attr="weight")
                    # if target_tp <= original_tp:
                    #     layer_sd["self_attention.qkv_proj.weight"] = merge_tensors(
                    #         tp_sd=mgt_sd[pp], model_key=model_key,
                    #         keys=get_keys('query_key_value', i=i, attr='weight'),
                    #         original_tp=original_tp,
                    #         target_tp=target_tp,
                    #         current_tp=tp,
                    #         merge_fn=merge_qkv,
                    #         split_fn=None
                    #     )
                    # else:
                    sd_list = [
                        dict_access_multi(
                            mgt_sd[pp][j + first_rank_loaded], [model_key] + keys
                        )
                        for j in range(original_tp)
                    ]
                    if getattr(init_model.config, "interleaved_qkv", True):
                        qkv_merged = transformer_interleaved_to_non_interleaved(
                            torch.cat(sd_list)
                        )
                    else:
                        qkv_merged = merge_qkv(sd_list)
                    if ifmtp:
                        param = params_dict[f"model.decoder.self_attn.qkv_proj.weight"]
                    else:
                        param = params_dict[
                            f"model.layers.{layer_offset + i}.self_attn.qkv_proj.weight"
                        ]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, qkv_merged)
                    layer_sd["self_attn.qkv_proj.weight"] = param.data

                    if layer.self_attn.qkv_proj.bias is not None:
                        # if target_tp <= original_tp:
                        #     layer_sd["self_attn.qkv_proj.bias"] = merge_tensors(
                        #         tp_sd=mgt_sd[pp], model_key=model_key,
                        #         keys=get_keys('query_key_value', i=i, attr='bias'),
                        #         original_tp=original_tp,
                        #         target_tp=target_tp,
                        #         current_tp=tp,
                        #         merge_fn=merge_qkv,
                        #         split_fn=None
                        #     )
                        # else:
                        keys = get_keys("query_key_value", i=i, attr="bias")
                        sd_list = [
                            dict_access_multi(
                                mgt_sd[pp][j + first_rank_loaded],
                                [model_key] + keys,
                            )
                            for j in range(original_tp)
                        ]
                        if getattr(init_model.config, "interleaved_qkv", True):
                            qkv_bias_merged = (
                                transformer_interleaved_to_non_interleaved_bias(
                                    torch.cat(sd_list)
                                )
                            )
                        else:
                            qkv_bias_merged = merge_qkv(sd_list)
                        if ifmtp:
                            param = params_dict[
                                f"model.decoder.self_attn.qkv_proj.bias"
                            ]
                        else:
                            param = params_dict[
                                f"model.layers.{layer_offset + i}.self_attn.qkv_proj.bias"
                            ]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, qkv_bias_merged)
                        layer_sd["self_attn.qkv_proj.bias"] = param.data

                    layer_sd[f"self_attn.o_proj.weight"] = merge_tensors(
                        tp_sd=mgt_sd[pp],
                        model_key=model_key,
                        keys=get_keys("dense", i=i, attr="weight"),
                        original_tp=original_tp,
                        target_tp=attn_tp_size,
                        current_tp=attn_tp_rank,
                        slice_dim=1,
                    )

                    if layer.self_attn.o_proj.bias is not None:
                        layer_sd[f"self_attn.o_proj.bias"] = dict_access_multi(
                            mgt_tp_0, get_keys("dense", i=i, attr="bias")
                        )

                def merge_glu(sd_list, dim=0):
                    return torch.cat(
                        [sd.chunk(dim=dim, chunks=2)[0].clone() for sd in sd_list]
                        + [sd.chunk(dim=dim, chunks=2)[1].clone() for sd in sd_list],
                        dim=dim,
                    )

                def split_glu(sd, cnt, idx, dim=0):
                    return torch.cat(
                        (
                            sd.chunk(dim=dim, chunks=2)[0]
                            .chunk(cnt, dim=dim)[idx]
                            .clone(),
                            sd.chunk(dim=dim, chunks=2)[1]
                            .chunk(cnt, dim=dim)[idx]
                            .clone(),
                        ),
                        dim=dim,
                    )

                if get_moe_a2a_backend().is_deepep() or is_ep_moe_enabled():
                    temp_target_tp = 1
                    temp_current_tp = 0
                else:
                    temp_target_tp = target_tp
                    temp_current_tp = tp

                if not is_moe_layer:
                    layer_sd[f"mlp.gate_up_proj.weight"] = merge_tensors(
                        tp_sd=mgt_sd[pp],
                        model_key=model_key,
                        keys=get_keys("dense_h_to_4h", i=i, attr="weight"),
                        original_tp=original_tp,
                        target_tp=mlp_tp_size,
                        current_tp=mlp_tp_rank,
                        merge_fn=merge_glu,
                        split_fn=split_glu,
                    )

                    if layer.mlp.gate_up_proj.bias is not None:
                        layer_sd["mlp.gate_up_proj.bias"] = merge_tensors(
                            tp_sd=mgt_sd[pp],
                            model_key=model_key,
                            keys=get_keys("dense_h_to_4h", i=i, attr="bias"),
                            original_tp=original_tp,
                            target_tp=mlp_tp_size,
                            current_tp=mlp_tp_rank,
                            merge_fn=merge_glu,
                            split_fn=split_glu,
                        )

                    layer_sd[f"mlp.down_proj.weight"] = merge_tensors(
                        tp_sd=mgt_sd[pp],
                        model_key=model_key,
                        keys=get_keys("dense_4h_to_h", i=i, attr="weight"),
                        original_tp=original_tp,
                        target_tp=mlp_tp_size,
                        current_tp=mlp_tp_rank,
                        slice_dim=1,
                    )

                    if layer.mlp.down_proj.bias is not None:
                        layer_sd[f"mlp.down_proj.bias"] = dict_access_multi(
                            mgt_tp_0, get_keys("dense_4h_to_h", i=i, attr="bias")
                        )
                else:
                    layer_sd[f"mlp.gate.weight"] = dict_access_multi(
                        mgt_tp_0, get_keys("moe.router", i=i)
                    )
                    if (
                        getattr(init_model.config, "moe_router_dtype", "float32")
                        == "float32"
                    ):
                        layer_sd[f"mlp.gate.weight"].float()
                    if getattr(
                        init_model.config, "moe_router_enable_expert_bias", True
                    ):
                        layer_sd[f"mlp.gate.e_score_correction_bias"] = (
                            dict_access_multi(
                                mgt_tp_0, get_keys("moe.router_bias", i=i)
                            )
                        )

                    if (
                        hasattr(layer.mlp, "shared_experts")
                        and get_global_server_args().disable_shared_experts_fusion
                    ):
                        layer_sd[f"mlp.shared_experts.gate_up_proj.weight"] = (
                            merge_tensors(
                                tp_sd=mgt_sd[pp],
                                model_key=model_key,
                                keys=get_keys(
                                    "moe.shared_experts.dense_h_to_4h",
                                    i=i,
                                    attr="weight",
                                ),
                                original_tp=original_tp,
                                target_tp=temp_target_tp,
                                current_tp=temp_current_tp,
                                merge_fn=merge_glu,
                                split_fn=split_glu,
                            )
                        )

                        layer_sd[f"mlp.shared_experts.down_proj.weight"] = (
                            merge_tensors(
                                tp_sd=mgt_sd[pp],
                                model_key=model_key,
                                keys=get_keys(
                                    "moe.shared_experts.dense_4h_to_h",
                                    i=i,
                                    attr="weight",
                                ),
                                original_tp=original_tp,
                                target_tp=temp_target_tp,
                                current_tp=temp_current_tp,
                                slice_dim=1,
                            )
                        )

                    if isinstance(
                        layer.mlp, GLM4MoESparseMoeBlock
                    ):  # Read Args and decide whether checkpoint use Grouped GEMM.
                        assert original_tp <= target_tp
                        if not (
                            get_moe_a2a_backend().is_deepep() or is_ep_moe_enabled()
                        ):
                            if get_global_server_args().disable_shared_experts_fusion:
                                gate_up_list, down_list = (
                                    [
                                        None
                                        for _ in range(
                                            init_model.config.n_routed_experts
                                        )
                                    ],
                                    [
                                        None
                                        for _ in range(
                                            init_model.config.n_routed_experts
                                        )
                                    ],
                                )
                            else:
                                gate_up_list, down_list = (
                                    [
                                        None
                                        for _ in range(
                                            init_model.config.n_routed_experts
                                            + init_model.config.n_shared_experts
                                        )
                                    ],
                                    [
                                        None
                                        for _ in range(
                                            init_model.config.n_routed_experts
                                            + init_model.config.n_shared_experts
                                        )
                                    ],
                                )
                            pre_state_expert = (
                                init_model.config.n_routed_experts // original_ep
                            )
                            rank_state_number = (
                                original_ep // target_tp
                                if original_ep >= target_tp
                                else 1
                            )
                            for k in range(rank_state_number):
                                assert original_ep * pre_state_expert >= target_tp
                                ep_su = (
                                    pre_state_expert
                                    if original_ep >= target_tp
                                    else pre_state_expert // (target_tp // original_ep)
                                )
                                ep_offset = tp * ep_su % pre_state_expert

                                sd_list = [
                                    dict_access_multi(
                                        mgt_tp_0,
                                        get_keys("moe.dense_h_to_4h", i=i, j=j),
                                    )
                                    for j in range(ep_offset, ep_offset + ep_su)
                                ]
                                down_sd_list = [
                                    dict_access_multi(
                                        mgt_tp_0,
                                        get_keys("moe.dense_4h_to_h", i=i, j=j),
                                    )
                                    for j in range(ep_offset, ep_offset + ep_su)
                                ]

                                for l in range(len(sd_list)):
                                    gate_ups = [
                                        torch.empty_like(sd_list[l], device="cuda")
                                        for _ in range(
                                            get_tensor_model_parallel_world_size()
                                        )
                                    ]
                                    downs = [
                                        torch.empty_like(down_sd_list[l], device="cuda")
                                        for _ in range(
                                            get_tensor_model_parallel_world_size()
                                        )
                                    ]

                                    torch.distributed.all_gather(
                                        gate_ups,
                                        sd_list[l].cuda(),
                                        group=get_tensor_model_parallel_group().device_group,
                                    )
                                    torch.distributed.all_gather(
                                        downs,
                                        down_sd_list[l].cuda(),
                                        group=get_tensor_model_parallel_group().device_group,
                                    )

                                    for rank_id in range(len(downs)):
                                        expert_number = (
                                            rank_id * ep_su + k * pre_state_expert + l
                                        )
                                        chunk_data = gate_ups[rank_id].cpu().clone()
                                        gate, up = chunk_data.chunk(2, dim=0)

                                        gate_up = torch.cat(
                                            [
                                                gate.chunk(target_tp, dim=0)[tp],
                                                up.chunk(target_tp, dim=0)[tp],
                                            ],
                                            dim=0,
                                        )
                                        down = (
                                            downs[rank_id]
                                            .cpu()
                                            .clone()
                                            .chunk(target_tp, dim=1)[tp]
                                        )

                                        gate_up_list[expert_number] = gate_up
                                        down_list[expert_number] = down

                            if (
                                not get_global_server_args().disable_shared_experts_fusion
                            ):

                                def gate_up_split_moe(sd, cnt, idx, dim=0):
                                    chunks_to_cat = []
                                    gate, up = sd.chunk(chunks=2, dim=dim)
                                    for k in range(init_model.config.n_shared_experts):
                                        gate_chunk_k = gate.chunk(
                                            chunks=init_model.config.n_shared_experts,
                                            dim=dim,
                                        )[k]
                                        up_chunk_k = up.chunk(
                                            chunks=init_model.config.n_shared_experts,
                                            dim=dim,
                                        )[k]
                                        gate_chunk = gate_chunk_k.chunk(
                                            chunks=cnt, dim=dim
                                        )[idx].clone()
                                        up_chunk = up_chunk_k.chunk(
                                            chunks=cnt, dim=dim
                                        )[idx].clone()
                                        chunks_to_cat.append(
                                            torch.cat((gate_chunk, up_chunk), dim=dim)
                                        )
                                    return chunks_to_cat

                                def down_split_moe(sd, cnt, idx, dim=1):
                                    chunks_to_cat = []
                                    for k in range(init_model.config.n_shared_experts):
                                        chunk_k = sd.chunk(
                                            chunks=init_model.config.n_shared_experts,
                                            dim=dim,
                                        )[k]
                                        chunks_to_cat.append(
                                            chunk_k.chunk(chunks=cnt, dim=dim)[
                                                idx
                                            ].clone()
                                        )
                                    return chunks_to_cat

                                cnt = target_tp // original_tp
                                tensor = dict_access_multi(
                                    mgt_sd[pp][tp // cnt],
                                    [model_key]
                                    + get_keys(
                                        "moe.shared_experts.dense_h_to_4h",
                                        i=i,
                                        attr="weight",
                                    ),
                                )

                                gate_up_list[-init_model.config.n_shared_experts :] = (
                                    gate_up_split_moe(tensor, cnt, tp % cnt)
                                )

                                tensor = dict_access_multi(
                                    mgt_sd[pp][tp // cnt],
                                    [model_key]
                                    + get_keys(
                                        "moe.shared_experts.dense_4h_to_h",
                                        i=i,
                                        attr="weight",
                                    ),
                                )
                                down_list[-init_model.config.n_shared_experts :] = (
                                    down_split_moe(tensor, cnt, tp % cnt)
                                )
                        else:
                            gate_up_list, down_list = [], []
                            for j in range(
                                init_model.config.n_routed_experts // target_tp * tp,
                                init_model.config.n_routed_experts
                                // target_tp
                                * (tp + 1),
                            ):
                                gate_up_list.append(
                                    dict_access_multi(
                                        mgt_tp_0,
                                        get_keys("moe.dense_h_to_4h", i=i, j=j),
                                    )
                                )
                                down_list.append(
                                    dict_access_multi(
                                        mgt_tp_0,
                                        get_keys("moe.dense_4h_to_h", i=i, j=j),
                                    )
                                )
                        layer_sd[f"mlp.experts.w13_weight"] = torch.stack(
                            gate_up_list, dim=0
                        )
                        layer_sd[f"mlp.experts.w2_weight"] = torch.stack(
                            down_list, dim=0
                        )

                    else:
                        raise ValueError(f"Unsupported expert type: {type(layer.mlp)}")

                torch.cuda.empty_cache()
                layer = layer.to(device)
                # `RadixLinearAttention.conv_weights` / `.bias` are plain
                # Python-attribute views into `qkv_conv1d.weight` / `.bias`.
                # Building layers on CPU and only later moving them to GPU
                # rebinds the underlying `.data`, so these views go stale.
                # The HF loader path doesn't hit this because it builds on
                # the target device. Refresh them here so the workaround
                # stays out of the model code.
                sa = getattr(layer, "self_attn", None)
                if sa is not None and hasattr(sa, "qkv_conv1d") and hasattr(sa, "attn"):
                    sa.attn.conv_weights = sa.qkv_conv1d.weight.squeeze(1)
                    sa.attn.bias = sa.qkv_conv1d.bias
                for k in layer_sd:
                    layer_sd[k] = layer_sd[k].to(device)
                missing_keys, unexpected_keys = layer.load_state_dict(
                    layer_sd, strict=False
                )
                # Filter benign duplicate-registration aliases. A single
                # nn.Parameter that lives on a parent module and is also
                # passed into a sub-module shows up under multiple paths in
                # state_dict. layer_sd only fills the canonical path, so
                # load_state_dict reports the other alias paths as missing
                # even though they share storage and are already loaded.
                path_to_pid = {
                    path: id(param)
                    for path, param in layer.named_parameters(remove_duplicate=False)
                }
                loaded_pids = {path_to_pid[k] for k in layer_sd if k in path_to_pid}
                missing_keys = [
                    k for k in missing_keys if path_to_pid.get(k) not in loaded_pids
                ]
                if 0 < len(missing_keys) or 0 < len(unexpected_keys):
                    logger.info(
                        f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}"
                    )
                if is_moe_layer and get_moe_a2a_backend().is_deepep():
                    layer.mlp.correction_bias = (
                        layer.mlp.gate.e_score_correction_bias.data
                    )
                    layer.mlp.correction_bias = layer.mlp.correction_bias.to(device)
                i += 1
            layer_offset += i

    init_model.lm_head = init_model.lm_head.to(device)
    if not getattr(init_model.config, "tie_word_embeddings", False):
        embedding_keys = get_keys("output_layer", attr="weight")
    else:
        embedding_keys = get_keys("word_embeddings")
    init_model.lm_head.weight.copy_(
        merge_tensors(
            tp_sd=mgt_sd[-1],
            model_key=last_model_in_pp,
            keys=embedding_keys,
            original_tp=original_tp,
            target_tp=target_tp,
            current_tp=tp,
            slice_dim=0,
        ).to(device)
    )

    if ifmtp:
        init_model.model.hnorm = init_model.model.hnorm.to(device)
        init_model.model.hnorm.weight.copy_(
            dict_access_multi(
                mgt_sd[0][0]["model"],
                get_keys(
                    "hnorm",
                    i=0,
                ),
            ).to(device)
        )
        init_model.model.enorm = init_model.model.enorm.to(device)
        init_model.model.enorm.weight.copy_(
            dict_access_multi(
                mgt_sd[0][0]["model"],
                get_keys(
                    "enorm",
                    i=0,
                ),
            ).to(device)
        )
        init_model.model.eh_proj = init_model.model.eh_proj.to(device)
        init_model.model.eh_proj.weight.copy_(
            dict_access_multi(
                mgt_sd[0][0]["model"],
                get_keys(
                    "eh_proj",
                    i=0,
                ),
            ).to(device)
        )

        init_model.model.shared_head = init_model.model.shared_head.to(device)
        init_model.model.shared_head.norm = init_model.model.shared_head.norm.to(device)
        init_model.model.shared_head.norm.weight.copy_(
            dict_access_multi(
                mgt_sd[-1][first_rank_loaded],
                [last_model_in_pp] + get_keys("final_layernorm", i=0),
            ).to(device)
        )
        if hasattr(init_model.model.shared_head.norm, "bias"):
            init_model.model.shared_head.norm.bias.copy_(
                dict_access_multi(
                    mgt_sd[-1][first_rank_loaded],
                    [last_model_in_pp] + get_keys("final_layernorm", i=0),
                )
            )
        if torch.distributed.get_rank() == 0:
            logger.info(f"total loading time: {time.time() - st_time}")

    if not ifmtp:
        assert layer_offset == len(
            init_model.model.layers
        ), f"layer_offset: {layer_offset}, len(self.layers): {len(init_model.model.layers)}"

        init_model.model.norm = init_model.model.norm.to(device)
        init_model.model.norm.weight.copy_(
            dict_access_multi(
                mgt_sd[-1][first_rank_loaded],
                [last_model_in_pp] + get_keys("final_layernorm", attr="weight"),
            ).to(device)
        )
        if hasattr(init_model.model.norm, "bias"):
            init_model.model.norm.bias.copy_(
                dict_access_multi(
                    mgt_sd[-1][first_rank_loaded],
                    [last_model_in_pp] + get_keys("final_layernorm", attr="bias"),
                )
            )
        if torch.distributed.get_rank() == 0:
            logger.info(f"total loading time: {time.time() - st_time}")
        # for key in params_dict.keys():
        #     if tp == 0:
        #         print("tp_rank", tp, " key: ", key, " number: ", params_dict[key], " sum: ", params_dict[key].sum(), " shape: ", params_dict[key].shape)

        try:
            init_model.model.consumed_train_samples = mgt_sd[0][0][
                "args"
            ].consumed_train_samples
            init_model.model.consumed_train_tokens = (
                init_model.model.consumed_train_samples
                * mgt_sd[0][0]["args"].seq_length
            )
            if torch.distributed.get_rank() == 0:
                meta_file = ckpt_dir / "meta.json"
                with open(meta_file, "w") as f:
                    meta = {}
                    meta["consumed_train_samples"] = (
                        init_model.model.consumed_train_samples
                    )
                    meta["consumed_train_tokens"] = (
                        init_model.model.consumed_train_tokens
                    )
                    f.write(json.dumps(meta))
        except:
            init_model.model.consumed_train_samples = None
            init_model.model.consumed_train_tokens = None
