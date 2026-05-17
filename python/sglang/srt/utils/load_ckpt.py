import math
import pickle
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import _create_read_item_for_tensor
from typing_extensions import override

from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import (
    get_tp_group as get_tensor_model_parallel_group,
)


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


def merge_ep(
    checkpoints: list[dict[str, Any]],
    experts_per_rank: int,
    expert_ffn_size: int,
):
    base_ckpt = checkpoints[0]
    for ckpt in checkpoints[1:]:
        for k, v in ckpt.items():
            if isinstance(v, torch.Tensor):
                if "experts.weight" in k:  # to be merged
                    if not isinstance(base_ckpt[k], list):
                        base_ckpt[k] = [base_ckpt[k]]
                    base_ckpt[k].append(v)
            else:
                pass

    for k in list(base_ckpt.keys()):
        if isinstance(base_ckpt[k], list):
            assert len(base_ckpt[k]) == len(
                checkpoints
            ), f"Length mismatch for {k}: {len(base_ckpt[k])} != {len(checkpoints)}"
            if k.endswith("experts.weight1"):
                w1s = [
                    w1.view(experts_per_rank, w1.shape[0], expert_ffn_size * 2)
                    for w1 in base_ckpt[k]
                ]
                base_ckpt[k] = torch.cat(w1s, dim=0).view(base_ckpt[k][0].shape[0], -1)
            elif k.endswith("experts.weight2"):
                w2s = [
                    w2.view(experts_per_rank, expert_ffn_size, w2.shape[1])
                    for w2 in base_ckpt[k]
                ]
                base_ckpt[k] = torch.cat(w2s, dim=0).view(-1, base_ckpt[k][0].shape[1])
    return base_ckpt


class ExpertDistributedStateDict(dict):
    def __init__(
        self,
        state_dict,
        expert_per_tank: int,
        moe_ffn_hidden_size: int,
        original_tp: int,
    ):
        self.group = get_tensor_model_parallel_group().device_group
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        assert expert_per_tank > 0
        self.expert_per_tank = expert_per_tank
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.original_tp = original_tp
        super().__init__(state_dict)

    def merge_ep(self, checkpoints: list[torch.Tensor], k: str):
        if k.endswith("experts.weight1"):
            w1s = [
                w1.view(self.expert_per_tank, w1.shape[0], self.moe_ffn_hidden_size * 2)
                for w1 in checkpoints
            ]
            return torch.cat(w1s, dim=0).view(checkpoints[0].shape[0], -1)
        elif k.endswith("experts.weight2"):
            w2s = [
                w2.view(self.expert_per_tank, self.moe_ffn_hidden_size, w2.shape[1])
                for w2 in checkpoints
            ]
            return torch.cat(w2s, dim=0).view(-1, checkpoints[0].shape[1])
        else:
            raise NotImplementedError(k)

    def __getitem__(self, k):
        original_tensor = super().__getitem__(k)

        if "experts.weight" not in k:
            return original_tensor

        ep_checkpoints: list[torch.Tensor] = [
            torch.empty_like(original_tensor, device="cuda")
            for _ in range(self.tp_size)
        ]
        original_tensor_cuda = original_tensor.cuda(non_blocking=True)
        torch.distributed.all_gather(
            ep_checkpoints, original_tensor_cuda, group=self.group
        )
        merged = self.merge_ep(ep_checkpoints, k)

        return merged


def count_tp_pp_ep(path: Path):
    tp_set = set()
    pp_set = set()
    ep_set = set()

    # Unified pattern that supports:
    # - mp_rank_00 (TP only)
    # - mp_rank_00_001 (TP + PP)
    # - mp_rank_00_001_002 (TP + PP + EP)
    pattern = re.compile(r"mp_rank_(\d{2})(?:_(\d{3}))?(?:_(\d{3}))?")

    for dir in path.iterdir():
        match = pattern.match(dir.name)
        if match:
            p_0, p_1, p_2 = match.groups()
            tp_set.add(p_0)
            if p_1 is not None:
                if p_2 is None:
                    # Two numbers: TP + PP
                    pp_set.add(p_1)
                else:
                    # Three numbers: TP + PP + EP
                    pp_set.add(p_1)
                    ep_set.add(p_2)

    # Always return 3 values
    tp_count = len(tp_set) if len(tp_set) > 0 else 1
    pp_count = len(pp_set) if len(pp_set) > 0 else 1
    ep_count = len(ep_set) if len(ep_set) > 0 else 1

    return tp_count, pp_count, ep_count


class RefineLoadPlanner(DefaultLoadPlanner):
    def __init__(self, local_plan) -> None:
        super().__init__()
        self.local_plan = local_plan

    @override
    def create_local_plan(self):
        return self.local_plan


def sort_attnkey(filename):
    match = re.search(r"layers\.(\d+)\.", filename[0])
    if match:
        return int(match.group(1))
    return -1


def sort_moekey(filename):
    layer_match = re.search(r"layers\.(\d+)\.", filename[0])
    layer_num = int(layer_match.group(1)) if layer_match else -1
    return (layer_num, filename[3])


def group_key_acquire(metadata, layer, ifattn, ifnextn, load_hc):
    attn_keys, moe_keys = [], [[], []]
    attn_dicts, moe_dicts = {}, {}
    hc_keys, hc_dicts = [], {}
    for key in metadata.state_dict_metadata:
        if not ifnextn:
            if "mtp" in key:
                continue
        else:
            if (
                "mtp" not in key
                and "output" not in key
                and "word_embeddings" not in key
            ):
                continue
        if (
            "optimizer" in key
            or "_extra_state" in key
            or not isinstance(
                metadata.state_dict_metadata[key], dist_cp.TensorStorageMetadata
            )
        ):
            continue

        if len(metadata.state_dict_metadata[key].chunks) != 1:
            flag = True
        else:
            flag = False

        if load_hc:
            sorted_chunks = sorted(
                metadata.state_dict_metadata[key].chunks, key=lambda x: tuple(x.offsets)
            )
            if (
                "hyper_connection" in key
                or "engram.multi_head_embedding.offsets" in key
                or "A_log" in key
                or "dt_bias" in key
            ):
                for idx, item in enumerate(sorted_chunks):
                    hc_keys.append(
                        [
                            key,
                            metadata.state_dict_metadata[key].properties.dtype,
                            item,
                            idx,
                            flag,
                            key,
                        ]
                    )
                    if flag:
                        dim = 0
                        for i, size in enumerate(item.offsets):
                            if size != 0:
                                dim = i
                                break
                        name = key if not flag else key + str(idx)
                        hc_dicts[name] = {"dim": dim}
            continue

        if ".experts." in key and not ifattn:  # moe
            sorted_chunks = sorted(
                metadata.state_dict_metadata[key].chunks, key=lambda x: tuple(x.offsets)
            )
            for idx, item in enumerate(sorted_chunks):
                if ".experts.linear_fc1" in key:
                    moe_keys[0].append(
                        [
                            key,
                            metadata.state_dict_metadata[key].properties.dtype,
                            item,
                            idx,
                            flag,
                            key,
                        ]
                    )
                elif ".experts.linear_fc2" in key:
                    moe_keys[1].append(
                        [
                            key,
                            metadata.state_dict_metadata[key].properties.dtype,
                            item,
                            idx,
                            flag,
                            key,
                        ]
                    )
                    if flag:
                        dim = 0
                        for i, size in enumerate(item.offsets):
                            if size != 0:
                                dim = i
                                break
                        name = key if not flag else key + str(idx)
                        moe_dicts[name] = {"dim": dim}
        elif ".experts." not in key and ifattn:  # attn
            sorted_chunks = sorted(
                metadata.state_dict_metadata[key].chunks, key=lambda x: tuple(x.offsets)
            )
            match = bool(re.search(r"layers\.\d+\.", key))
            if "layers." in key and not match:
                flag = False
                temp_offset, num = sorted_chunks[-1].offsets[0], 0
                for i in sorted_chunks:
                    if temp_offset == i.offsets[0]:
                        num += 1
                if num > 1:
                    flag = True
                for idx, item in enumerate(sorted_chunks):
                    attn_keys.append(
                        [
                            key,
                            metadata.state_dict_metadata[key].properties.dtype,
                            item,
                            idx % num,
                            flag,
                            key.replace("layers.", f"layers.{item.offsets[0]}."),
                        ]
                    )
                    if flag:
                        dim = 0
                        for i, size in enumerate(item.offsets):
                            if i == 0:
                                continue
                            if size != 0:
                                dim = i
                                break
                        name = (
                            key.replace("layers.", f"layers.{item.offsets[0]}.")
                            if not flag
                            else key.replace("layers.", f"layers.{item.offsets[0]}.")
                            + str(idx % num)
                        )
                        attn_dicts[name] = {"dim": dim}
            else:
                for idx, item in enumerate(sorted_chunks):
                    attn_keys.append(
                        [
                            key,
                            metadata.state_dict_metadata[key].properties.dtype,
                            item,
                            idx,
                            flag,
                            key,
                        ]
                    )
                    if flag:
                        dim = 0
                        for i, size in enumerate(item.offsets):
                            if size != 0:
                                dim = i
                                break
                        name = key if not flag else key + str(idx)
                        attn_dicts[name] = {"dim": dim}

    if load_hc:
        return hc_keys, hc_dicts
    if len(attn_keys) != 0:
        return attn_keys, attn_dicts
    else:
        return moe_keys, moe_dicts


def load_distcp(
    meta_file_path, metadata, ifattn, ifnextn, total_num_experts, layer, load_hc=False
):
    world_size, tp = (
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
    )
    keys, dicts = group_key_acquire(metadata, layer, ifattn, ifnextn, load_hc)
    large_tensor_keys = []

    # calculate the tensors should be loaded(layers for attn and tp for moe)
    assert total_num_experts is None or total_num_experts % world_size == 0
    if load_hc:
        part_keys = keys
    elif ifattn:
        keys = sorted(keys, key=sort_attnkey)
        for key in keys:
            if (
                "embedding.word_embeddings.weight" in key[0]
                or "output_layer.weight" in key[0]
            ):
                large_tensor_keys.append(key)
        for key in large_tensor_keys:
            keys.remove(key)

        cnt = len(keys) // world_size
        if tp + 1 <= len(keys) % world_size:
            attn_start = (cnt + 1) * tp
            attn_end = (cnt + 1) * (tp + 1)
        else:
            attn_start = (cnt + 1) * (len(keys) % world_size) + cnt * (
                tp - len(keys) % world_size
            )
            attn_end = (cnt + 1) * (len(keys) % world_size) + cnt * (
                tp - len(keys) % world_size + 1
            )

        part_keys = keys[attn_start:attn_end] + large_tensor_keys
    else:
        keys[0] = sorted(keys[0], key=sort_moekey)
        keys[1] = sorted(keys[1], key=sort_moekey)
        cnt = total_num_experts // world_size
        offset_fc1, offset_fc2, part_keys = 0, 0, []
        for i in range(layer):
            layer_match = re.search(r"layers\.(\d+)\.", keys[0][0][0])
            layer_num = int(layer_match.group(1))
            if i >= layer_num:
                part_keys += (
                    keys[0][offset_fc1 + 2 * cnt * tp : offset_fc1 + 2 * cnt * (tp + 1)]
                    + keys[1][offset_fc2 + cnt * tp : offset_fc2 + cnt * (tp + 1)]
                )
                offset_fc1 += total_num_experts * 2
                offset_fc2 += total_num_experts

    # load tensors with the order of files
    state_dict, requests = {}, []
    for item in part_keys:
        name = item[5] if not item[4] else item[5] + str(item[3])
        state_dict[name] = torch.empty(item[2].sizes, dtype=item[1])
        requests.append(
            _create_read_item_for_tensor(
                dest_index=MetadataIndex(
                    name, torch.Size([0 for _ in item[2].offsets]), 0
                ),
                dest_offsets=torch.Size([0 for _ in item[2].offsets]),
                storage_index=MetadataIndex(item[0], item[2].offsets, item[3]),
                storage_offsets=item[2].offsets,
                lengths=item[2].sizes,
            )
        )

    local_plan = LoadPlan(requests)
    planner = RefineLoadPlanner(local_plan)

    dist_cp.load(
        state_dict=state_dict,
        storage_reader=WrappedStorageReader(meta_file_path),
        planner=planner,
    )
    return state_dict, keys, dicts


def postprocess(mgt_sd, attn_keys_dict, moe_keys_dict):
    for pp_item in mgt_sd:
        tp_item, temp_item, temp_key = pp_item[0]["model"], {}, []
        # merge and rename keys
        for key in tp_item.keys():
            # merge "weight0" and "weight1" for moe_h_to_4h
            if ".experts.experts" in key and "linear_fc1" in key:
                match = re.search(r"(\d+)$", key)
                if match is None:
                    continue
                num = int(match.group(1))
                if num % 2 == 1:
                    relative_key = key[: -len(match.group(1))] + str(num - 1)
                    final_key = key[: -len(match.group(1))] + str((num - 1) // 2)
                    temp_item[final_key] = torch.cat(
                        (tp_item[relative_key], tp_item[key]), dim=1
                    ).squeeze(0)
                    temp_key.append(key)
                    temp_key.append(relative_key)

                    new_key = final_key.replace(".experts", "", 1)
                    temp_item[new_key] = temp_item[final_key]
                    temp_item.pop(final_key)

            elif ".experts.experts" in key and "linear_fc2" in key:
                new_key = key.replace(".experts", "", 1)
                temp_item[new_key] = tp_item[key].squeeze(0)
                temp_key.append(key)

            # merge "weight0~x" for attn weights/bias
            elif ".experts.experts" not in key:  # and "linear_fc1" in key:
                match = re.search(r"(\d+)$", key)
                if match is None:
                    continue
                base_key = key[: -len(match.group(1))]
                num = int(match.group(1))
                if num == 0:
                    i, dense_list = 0, []
                    while True:
                        current_key = base_key + str(i)
                        if current_key in tp_item:
                            dense_list.append(tp_item[current_key])
                            temp_key.append(current_key)
                            i += 1
                        else:
                            break
                    if dense_list and key in attn_keys_dict:
                        temp_item[base_key] = torch.cat(
                            dense_list, dim=attn_keys_dict[base_key + str(1)]["dim"]
                        )
                    elif dense_list and key in moe_keys_dict:
                        temp_item[base_key] = torch.cat(
                            dense_list, dim=moe_keys_dict[base_key + str(1)]["dim"]
                        )

        for key in temp_key:
            tp_item.pop(key)
        tp_item.update(temp_item)

        for key, value in tp_item.items():
            if value.shape[0] == 1:
                tp_item[key] = value.squeeze(0)


def _gather_by_dtype(mgt_sd_model, dtype_keys_per_tp, dtype, world_size, current_tp):
    """Run all_gather for a single dtype group within a chunk."""
    maxm = 0
    local_tensors = []
    for tp in range(world_size):
        temp = 0
        for item in dtype_keys_per_tp[tp]:
            temp += math.prod(item[2].sizes)
            if tp == current_tp:
                name = item[5] if not item[4] else item[5] + str(item[3])
                tensor = mgt_sd_model[name].contiguous().view(-1)
                local_tensors.append(tensor)
        maxm = max(maxm, temp)

    if maxm == 0:
        return

    # padding and gather
    local_concat = (
        torch.cat(local_tensors).to("cuda")
        if len(local_tensors) != 0
        else torch.zeros(1, dtype=dtype, device="cuda")
    )
    padded_local = torch.zeros(maxm, dtype=dtype, device="cuda")
    padded_local[: local_concat.numel()] = local_concat
    gathered = [
        torch.empty(maxm, dtype=dtype, device="cuda") for _ in range(world_size)
    ]
    dist.all_gather(
        gathered, padded_local, group=get_tensor_model_parallel_group().device_group
    )

    # reshape received tensors from other ranks
    for tp in range(world_size):
        if tp == current_tp:
            continue
        offset = 0
        for item in dtype_keys_per_tp[tp]:
            numel = math.prod(item[2].sizes)
            name = item[5] if not item[4] else item[5] + str(item[3])
            mgt_sd_model[name] = (
                gathered[tp][offset : offset + numel].view(item[2].sizes).cpu().clone()
            )
            offset += numel

    del gathered, padded_local, local_concat


def communicate_for_attn(mgt_sd_model, attn_keys, world_size, current_tp):
    # construct tensors for all_gather
    cnt = len(attn_keys) // world_size
    chunk_size = 10
    attn_start, attn_end = [], []
    for tp in range(world_size):
        if tp + 1 <= len(attn_keys) % world_size:
            attn_start.append((cnt + 1) * tp)
            attn_end.append((cnt + 1) * (tp + 1))
        else:
            attn_start.append(
                (cnt + 1) * (len(attn_keys) % world_size)
                + cnt * (tp - len(attn_keys) % world_size)
            )
            attn_end.append(
                (cnt + 1) * (len(attn_keys) % world_size)
                + cnt * (tp - len(attn_keys) % world_size + 1)
            )

    for chunk_start in range(attn_start[0], attn_end[0], chunk_size):
        # collect chunk_keys per tp and separate by dtype
        dtype_groups = {}  # dtype -> {tp: [keys]}
        for tp in range(world_size):
            chunk_keys = attn_keys[
                chunk_start
                + attn_start[tp] : min(
                    chunk_start + attn_start[tp] + chunk_size, attn_end[tp]
                )
            ]
            for item in chunk_keys:
                dt = item[1]
                if dt not in dtype_groups:
                    dtype_groups[dt] = {t: [] for t in range(world_size)}
                dtype_groups[dt][tp].append(item)

        # run separate all_gather for each dtype to avoid precision loss
        for dt, keys_per_tp in dtype_groups.items():
            _gather_by_dtype(mgt_sd_model, keys_per_tp, dt, world_size, current_tp)

        torch.cuda.empty_cache()
