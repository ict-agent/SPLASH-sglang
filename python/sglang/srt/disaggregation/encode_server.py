import asyncio
import concurrent.futures
import contextvars
import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response
from transformers import AutoProcessor

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_receiver import EmbeddingData
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    get_mooncake_transfer_engine,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqInput, ProfileReqType
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.multimodal.processors.glm4v import (
    _MM_SAMPLING_KEYS as glm_mm_sampling_keys,
    _split_mm_items as glm_split_mm_items,
    glm_budget_kwargs,
    glm_decode_frames_at,
    glm_max_image_tokens_from_configs,
    glm_sample_and_decode_sync,
    glm_sample_frame_indices,
    preprocess_video_frames_sync,
)
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.observability.metrics_collector import (
    create_encoder_metrics_collector,
)
from sglang.srt.observability.req_time_stats import EncoderReqTimeStats
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    add_prometheus_middleware,
    is_cuda,
    load_audio,
    load_image,
    load_video,
    random_uuid,
    set_prometheus_multiproc_dir,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 10

# Minimal 32x32 black PNG for health check dummy encode
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="

rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}

use_image_processor_gpu = (
    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
)

_GPU_ADMIT_POLL_S = 0.2
_GPU_ADMIT_IDLE_SLACK = 1.1
_VIDEO_ADMIT_BYTES_FACTOR = 2.0

# Request-local cross-Encoder video shard coordinates. Each Encoder service
# decodes its own contiguous temporal slice without a distributed collective.
_video_shard_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "video_shard_ctx", default=None
)


class MMError(Exception):
    def __init__(self, message, code=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BadRequestError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.BAD_REQUEST)


class InternalError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.INTERNAL_SERVER_ERROR)


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy."""

    def __init__(self, tensor):
        # Ensure tensor is on CPU and contiguous
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Keep tensor reference
        self.tensor = tensor
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __buffer__(self):
        data_ptr = self.tensor.data_ptr()
        total_bytes = self.tensor.numel() * self.tensor.element_size()
        c_obj = (ctypes.c_char * total_bytes).from_address(data_ptr)
        c_obj._keep_alive_ref = self
        return memoryview(c_obj)


def _convert(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return torch.tensor(np.array(data))
    elif isinstance(data, list) and isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        return data


_mm_grid_attrs = {
    # Kimi K2.5 HF processor uses grid_thws (see base_processor.ATTR_NAME_TO_MODALITY).
    Modality.IMAGE: ["image_grid_thw", "image_grid_hws", "grid_thws"],
    Modality.VIDEO: ["video_grid_thw"],
    Modality.AUDIO: ["audio_feature_lens_raw"],
}

_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


def _get_mm_grid_dim(mm_inputs, modality, model_type: Optional[str] = None):
    if modality == Modality.VIDEO and "_reported_grid" in mm_inputs:
        return mm_inputs["_reported_grid"]
    # Kimi K2.5 vision processor only emits `grid_thws`; prefer it over generic keys
    # so we never pick a mis-typed or stale `image_grid_hws` field from kwargs.
    attrs = _mm_grid_attrs[modality]
    if (model_type or "").lower() in [
        "kimi_k25",
        "kimi_vl",
    ] and modality == Modality.IMAGE:
        attrs = ("grid_thws", "image_grid_thw", "image_grid_hws")
    for attr in attrs:
        if attr in mm_inputs and mm_inputs[attr] is not None:
            return mm_inputs[attr]
    raise ValueError(f"Grid dim ({_mm_grid_attrs[modality]}) not found in {mm_inputs}")


def _get_mm_feature(mm_inputs, modality):
    for attr in _mm_feature_attrs[modality]:
        if attr in mm_inputs:
            return mm_inputs[attr]
    raise ValueError(
        f"Feature attrs ({_mm_feature_attrs[modality]}) not found in {mm_inputs}"
    )


def _build_mm_aux_data(mm_inputs):
    """
    Build auxiliary data for video modality.
    """
    aux_data = {
        "video_timestamps": mm_inputs.get("video_timestamps", None),
        "second_per_grid_ts": mm_inputs.get("second_per_grid_ts", None),
    }
    return aux_data


def _set_video_shard_context(request: dict, modality: Modality) -> bool:
    """Install this request's video shard coordinates in the current task."""
    num_shards = request.get("video_num_shards")
    if num_shards and modality == Modality.VIDEO:
        num_shards = int(num_shards)
        shard_idx = int(request.get("video_shard_idx", 0))
        if num_shards <= 0 or shard_idx < 0 or shard_idx >= num_shards:
            raise BadRequestError(
                f"Invalid video shard {shard_idx}/{num_shards}"
            )
        _video_shard_ctx.set((shard_idx, num_shards))
        return True
    _video_shard_ctx.set(None)
    return False


class MMEncoder:
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.profiler = EncoderProfiler(rank)
        self._load_mm_processor(server_args)
        self.metrics = create_encoder_metrics_collector(server_args, rank)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        self.device_module = torch.get_device_module(self.device)
        self.device_module.set_device(self.gpu_id)

        self.use_image_processor_gpu = (
            use_image_processor_gpu and not server_args.disable_fast_image_processor
        )
        self._build_vision_config(server_args.mm_process_config)

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )

        # Reserve decoded-frame memory against device headroom before starting
        # large parallel video decodes. This is per Encoder process/device.
        self._admit_lock = asyncio.Lock()
        try:
            self._admit_baseline_bytes = self.device_module.memory_allocated(
                self.gpu_id
            )
        except Exception:
            self._admit_baseline_bytes = 0
        self._admit_reserved_bytes = 0

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=envs.SGLANG_ENCODER_MM_LOAD_WORKERS.get(),
            # The current device is thread-local. Pin every preprocessing worker
            # to this encoder rank's device so GPU-backed processors do not all
            # fall back to device 0.
            initializer=lambda gid=self.gpu_id: torch.get_device_module(
                self.device
            ).set_device(gid),
        )
        # Keep model execution serialized, matching the event-loop behavior it
        # replaces, while moving ViT forward and the synchronous D2H copy off
        # the loop so health checks and other requests can keep progressing.
        self.gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            initializer=lambda gid=self.gpu_id: torch.get_device_module(
                self.device
            ).set_device(gid),
        )
        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )
        self.background_tasks: Set[asyncio.Task] = set()

        if self.server_args.enable_mm_global_cache:
            from sglang.srt.mem_cache.storage.mooncake_store.embedding_cache_controller import (
                EmbeddingCacheController,
            )

            hidden_dims = self._infer_embedding_dims()
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                server_args.tp_size,
                hidden_dims=hidden_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
            )
        else:
            self.mm_global_cache = None

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {self.server_args.encoder_transfer_backend}"
            )

            if self.server_args.encoder_transfer_backend == "mooncake":
                self.local_ip = get_local_ip_auto()

                self.engine = get_mooncake_transfer_engine()
                if self.engine is None:
                    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                        init_mooncake_transfer_engine,
                    )

                    self.engine = init_mooncake_transfer_engine(
                        hostname=self.local_ip,
                        gpu_id=self.gpu_id,
                        ib_device=(
                            self.server_args.disaggregation_ib_device
                            or self.server_args.mooncake_ib_device
                        ),
                    )

            self.embedding_to_send = dict()

        logger.info(f"rank {rank} init finish ")

    def _infer_embedding_dims(self) -> dict:
        """Infer per-modality embedding dimensions from hf_config at init time."""
        default = self.model_config.hidden_size
        hf_cfg = self.model_config.hf_config
        thinker_cfg = getattr(hf_cfg, "thinker_config", None)
        dims = {
            Modality.IMAGE: default,
            Modality.VIDEO: default,
            Modality.AUDIO: default,
        }

        vision_cfg = getattr(thinker_cfg, "vision_config", None) or getattr(
            hf_cfg, "vision_config", None
        )
        if vision_cfg is not None:
            out_hs = getattr(vision_cfg, "out_hidden_size", None)
            if out_hs is not None:
                ds = getattr(vision_cfg, "deepstack_visual_indexes", None)
                vis_dim = (
                    out_hs * (1 + len(ds))
                    if isinstance(ds, (list, tuple)) and ds
                    else out_hs
                )
                dims[Modality.IMAGE] = vis_dim
                dims[Modality.VIDEO] = vis_dim

        audio_cfg = getattr(thinker_cfg, "audio_config", None) or getattr(
            hf_cfg, "audio_config", None
        )
        if audio_cfg is not None:
            for attr in ("output_dim", "d_model"):
                val = getattr(audio_cfg, attr, None)
                if val and int(val) > 0:
                    dims[Modality.AUDIO] = int(val)
                    break

        logger.info(f"Global cache embedding dims: {dims}")
        return dims

    async def _sweep_stale_embeddings_loop(self):
        """Background task: reclaim embeddings that were encoded but never
        claimed by a /send (prefill LLM timed out / cancelled / crashed).

        In the mooncake backend an entry in embedding_to_send is normally freed
        the moment its /send arrives. If /send never comes, the entry -- and its
        host/GPU memory -- would leak until the process restarts. This sweeper
        drops any entry older than TTL. TTL is derived from the LLM's receive
        timeout (plus a margin) so it only ever collects true orphans: by the
        time TTL elapses the LLM has long since given up.
        """
        interval = envs.SGLANG_ENCODER_EMBEDDING_SWEEP_INTERVAL.get()
        ttl = envs.SGLANG_ENCODER_EMBEDDING_TTL.get()
        if ttl <= 0:
            # 0 => derive from the LLM recv timeout + one sweep interval margin.
            ttl = envs.SGLANG_ENCODER_RECV_TIMEOUT.get() + interval
        if interval <= 0:
            return  # sweeper disabled
        while True:
            try:
                await asyncio.sleep(interval)
                deadline = time.perf_counter() - ttl
                for req_id in list(self.embedding_to_send.keys()):
                    mm_data = self.embedding_to_send.get(req_id)
                    if (
                        mm_data is None
                        or getattr(mm_data, "created_at", 0) > deadline
                    ):
                        continue  # already freed, or still fresh
                    # Orphan: encoded but not claimed within TTL -> reclaim.
                    self.embedding_to_send.pop(req_id, None)
                    mm_data.embedding = None
                    logger.warning(
                        f"[embedding-sweeper] reclaimed orphan req_id={req_id} "
                        f"(no /send within TTL={ttl:.0f}s); "
                        f"remaining={len(self.embedding_to_send)}"
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[embedding-sweeper] unexpected error; continuing")

    def _build_vision_config(self, mm_process_config):
        """
        Validate vision config, used for image/video/audio.
        If not provided, keep default values.
        """
        self.vision_config = (
            mm_process_config.get("vision_config", {})
            if mm_process_config is not None
            else {}
        )
        for modality_str in ["image", "video", "audio"]:
            if not self.vision_config.get(modality_str, None):
                self.vision_config[modality_str] = {}
            if self.use_image_processor_gpu:
                self.vision_config[modality_str]["device"] = self.device

            if modality_str == "video":
                video_defaults = {"fps": 2.0, "max_frames": 768, "min_frames": 4}
                for k, v in video_defaults.items():
                    self.vision_config["video"].setdefault(k, v)

            if modality_str == "audio":
                if "return_attention_mask" not in self.vision_config["audio"]:
                    self.vision_config["audio"]["return_attention_mask"] = True
                if "padding" not in self.vision_config["audio"]:
                    if self.model_type == "qwen2_audio":
                        # For Qwen2Audio, use padding="max_length"
                        # (same as https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L93)
                        self.vision_config["audio"]["padding"] = "max_length"
                    else:
                        self.vision_config["audio"]["padding"] = True
                if "truncation" not in self.vision_config["audio"]:
                    # keep same logic as base_processor.py
                    if (
                        hasattr(self, "audio_processor")
                        and self.audio_processor is not None
                    ):
                        if self.audio_processor.__class__.__name__ in {
                            "Gemma3nProcessor",
                            "GlmAsrProcessor",
                            "Qwen2AudioProcessor",
                            "Qwen3OmniMoeProcessor",
                        }:
                            self.vision_config["audio"]["truncation"] = False

    def _load_mm_processor(self, server_args: ServerArgs):
        """
        Load image/video/audio processor separately,
        avoid issues with AutoProcessor not recognizing certain models
        """
        from transformers import AutoImageProcessor, AutoVideoProcessor

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load image processor: {e}")
            self.image_processor = None

        try:
            self.video_processor = AutoVideoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load video processor: {e}")
            self.video_processor = None

        try:
            # Note: AutoProcessor is used for audio processor
            _audio_proc = AutoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
            if not hasattr(_audio_proc, "feature_extractor"):
                logger.warning(
                    "Loaded AutoProcessor has no feature_extractor attribute, "
                    "audio processing will be unavailable."
                )
                self.audio_processor = None
            else:
                self.audio_processor = _audio_proc
        except Exception as e:
            logger.warning(f"Failed to load audio processor: {e}")
            self.audio_processor = None

    def _load_single_item(
        self,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.
        If data is precomputed, returns directly.
        Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data, self.use_image_processor_gpu)
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):
                    # Needed only when `img` is a PIL image
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, use_gpu=self.use_image_processor_gpu)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            if isinstance(data, (str, bytes, bytearray)):
                description = f"{type(data).__name__}(len={len(data)})"
            else:
                description = type(data).__name__
            raise RuntimeError(
                f"Error while loading data [{description}]: {e}"
            ) from e

    def submit_data_loading_tasks(self, items, modalities):
        futures = []
        task_info = []

        for data, modality in zip(items, modalities):
            if modality is not None:
                futures.append(
                    self.io_executor.submit(
                        self._load_single_item,
                        data,
                        modality,
                    )
                )
                task_info.append((modality, data))
        return futures, task_info

    def _get_feat_extract_output_lengths(self, feature_lens):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        # qwen2_audio/qwen2.5_omni
        if self.model_type in ["qwen2_audio", "qwen2_5_omni"]:
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1
        # qwen3_asr / qwen3_omni_moe (same audio encoder architecture)
        elif self.model_type in ["qwen3_asr", "qwen3_omni_moe"]:
            input_lengths_leave = feature_lens % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            output_lengths = (
                ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13
            )
            return output_lengths
        else:
            # fallback to original HF audio sample logic for other models
            logger.warning(
                f"Fallback to original HF audio sample logic for {self.model_type}"
            )
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1

    async def _shard_decode_single_video(
        self,
        vr,
        video_config,
        num_decode_workers,
        *,
        shard_idx,
        num_shards,
        video_processor_kwargs,
        precomputed_indices=None,
    ):
        """Decode one contiguous temporal shard of a single GLM-V video."""
        video_config = video_config or {}
        video_fps = vr.avg_fps
        total_num_frames = len(vr)
        duration = total_num_frames / video_fps if video_fps else 0

        global_indices = (
            precomputed_indices
            if precomputed_indices is not None
            else glm_sample_frame_indices(
                total_num_frames,
                video_fps,
                duration,
                target_fps=video_config.get("fps"),
                max_frame_count=video_config.get("max_frames"),
            )
        )
        # GLM-V groups every two sampled frames into one temporal unit. Split
        # only on unit boundaries so concatenating Encoder outputs is lossless.
        n_units = len(global_indices) // 2
        base, remainder = divmod(n_units, num_shards)
        shard_counts = [
            base + (1 if index < remainder else 0) for index in range(num_shards)
        ]
        start = sum(shard_counts[:shard_idx])
        count = shard_counts[shard_idx]
        local_frame_indices = list(
            global_indices[2 * start : 2 * (start + count)]
        )

        frames = None
        reserved = 0
        try:
            if local_frame_indices:
                if self._video_decode_uses_cuda(vr):
                    height, width = vr.frame_shape
                    estimate = len(local_frame_indices) * height * width * 3
                    admit_error, reserved = await self.await_gpu_bytes(estimate)
                    if admit_error is not None:
                        raise MMError(
                            admit_error, code=HTTPStatus.SERVICE_UNAVAILABLE
                        )
                loop = asyncio.get_running_loop()
                frames = await loop.run_in_executor(
                    self.io_executor,
                    glm_decode_frames_at,
                    vr,
                    local_frame_indices,
                    num_decode_workers,
                    video_config,
                )
        finally:
            self._release_admit_bytes(reserved)

        if frames is not None:
            videos = [frames]
        else:
            # _process_mm_items consumes the empty-shard marker before invoking
            # the HF processor, so no synthetic frame tensor is necessary.
            videos = []

        video_processor_kwargs["do_sample_frames"] = False
        video_processor_kwargs["return_metadata"] = True

        # Scale the token budget by shard ratio so every shard selects the same
        # spatial resolution as the whole video.
        n_global_frames = len(global_indices)
        n_local_frames = len(local_frame_indices)
        if n_local_frames > 0 and n_global_frames > 0:
            ratio = n_local_frames / n_global_frames
            user_budget = (
                video_config.get("max_image_tokens") if video_config else None
            )
            budget = (
                int(user_budget)
                if user_budget is not None
                else self.video_processor.max_image_tokens
            )
            video_processor_kwargs["max_image_tokens"] = max(
                1, int(budget * ratio)
            )

        video_processor_kwargs["_shard_meta"] = {
            "global_indices": global_indices,
            "fps": video_fps,
            "shard_idx": shard_idx,
            "start_unit": start,
            "count": count,
            "n_units": n_units,
        }
        return videos, video_processor_kwargs

    @staticmethod
    def _video_decode_uses_cuda(video) -> bool:
        device = getattr(video, "device", getattr(video, "_device", "cpu"))
        return is_cuda() and str(device).startswith("cuda")

    def _estimate_video_decode_bytes(self, video_items, video_configs) -> int:
        """Estimate the native RGB frame footprint for a video request."""
        total = 0
        for idx, video in enumerate(video_items):
            if not self._video_decode_uses_cuda(video):
                continue
            config = video_configs[idx] if idx < len(video_configs) else {}
            try:
                frame_count = len(video)
                fps = video.avg_fps
                duration = frame_count / fps if fps else 0
                indices = glm_sample_frame_indices(
                    frame_count,
                    fps,
                    duration,
                    target_fps=config.get("fps"),
                    max_frame_count=config.get("max_frames"),
                )
                height, width = video.frame_shape
                total += len(indices) * height * width * 3
            except Exception as exc:
                logger.warning(
                    "[video-admit] could not estimate item %d: %s; treating as 0",
                    idx,
                    exc,
                )
        return total

    async def await_gpu_bytes(self, need_bytes: int) -> Tuple[Optional[str], int]:
        """Wait until the current device can admit a parallel video decode."""
        if not is_cuda() or need_bytes <= 0:
            return None, 0
        need = int(need_bytes * _VIDEO_ADMIT_BYTES_FACTOR)
        idle_ceiling = self._admit_baseline_bytes * _GPU_ADMIT_IDLE_SLACK
        max_wait = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()
        deadline = time.monotonic() + max_wait
        async with self._admit_lock:
            while True:
                try:
                    free, total = self.device_module.mem_get_info(self.gpu_id)
                    allocated = self.device_module.memory_allocated(self.gpu_id)
                    cached = self.device_module.memory_reserved(self.gpu_id)
                except Exception:
                    # Devices without CUDA-style memory accounting keep the
                    # pre-existing unrestricted decode behavior.
                    return None, 0
                available = (
                    (cached - allocated) + free - self._admit_reserved_bytes
                )
                idle = (
                    allocated <= idle_ceiling and self._admit_reserved_bytes == 0
                )
                if total <= 0 or available >= need or idle:
                    self._admit_reserved_bytes += need
                    return None, need
                if time.monotonic() >= deadline:
                    return (
                        f"GPU busy: video decode needs ~{need / 1e9:.1f}GB "
                        f"(est {need_bytes / 1e9:.1f}GB "
                        f"x{_VIDEO_ADMIT_BYTES_FACTOR:g}), available "
                        f"{available / 1e9:.1f}GB of {total / 1e9:.1f}GB after "
                        f"{max_wait:.0f}s"
                    ), 0
                await asyncio.sleep(_GPU_ADMIT_POLL_S)

    def _release_admit_bytes(self, reserved: int) -> None:
        if reserved:
            self._admit_reserved_bytes = max(
                0, self._admit_reserved_bytes - reserved
            )

    @staticmethod
    def _close_video_decoders(video_items):
        for item in video_items or []:
            close = getattr(item, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("mm: failed to close video decoder")

    async def _flatten_and_load_videos(self, mm_items):
        if not isinstance(mm_items, (list, tuple)):
            mm_items = [mm_items]

        video_urls, video_configs = glm_split_mm_items(
            mm_items, glm_mm_sampling_keys
        )
        if video_urls is not None:
            mm_items = video_urls

        futures, _ = self.submit_data_loading_tasks(
            mm_items, [Modality.VIDEO] * len(mm_items)
        )
        async_futures = [asyncio.wrap_future(f) for f in futures]
        video_items = await asyncio.gather(*async_futures)

        try:
            video_processor_kwargs = {}
            if "qwen" in self.model_type:
                # for qwen-series model, do sample frames before preprocess
                video_processed = [
                    await preprocess_video(
                        video, video_config=self.vision_config.get("video", {})
                    )
                    for video in video_items
                ]
                videos, video_metadata = map(list, zip(*video_processed))
                video_processor_kwargs["do_sample_frames"] = False
                if video_metadata:
                    video_processor_kwargs["video_metadata"] = video_metadata
                return videos, video_processor_kwargs
            elif "glm" in self.model_type:
                budget_kwargs = glm_budget_kwargs(
                    self.video_processor,
                    user_max_image_tokens=glm_max_image_tokens_from_configs(
                        video_configs
                    ),
                    count=len(video_items),
                    split=True,
                )
                if budget_kwargs is not None:
                    video_processor_kwargs.update(budget_kwargs)
                framed = any(isinstance(video, list) for video in video_items)
                loop = asyncio.get_running_loop()
                if framed:
                    tasks = [
                        loop.run_in_executor(
                            self.io_executor,
                            preprocess_video_frames_sync,
                            video,
                        )
                        for video in video_items
                    ]
                    video_processed = await asyncio.gather(*tasks)
                    videos, video_metadata = map(list, zip(*video_processed))
                    video_processor_kwargs["do_sample_frames"] = False
                    video_processor_kwargs["return_metadata"] = True
                    if video_metadata:
                        video_processor_kwargs["video_metadata"] = video_metadata
                    return videos, video_processor_kwargs

                num_decode_workers = (
                    envs.SGLANG_ENCODER_GLM_VIDEO_DECODE_WORKERS.get()
                )
                shard = _video_shard_ctx.get()
                if shard is not None and len(video_items) == 1:
                    shard_idx, num_shards = shard
                    return await self._shard_decode_single_video(
                        video_items[0],
                        video_configs[0] if video_configs else {},
                        num_decode_workers,
                        shard_idx=shard_idx,
                        num_shards=num_shards,
                        video_processor_kwargs=video_processor_kwargs,
                    )

                estimate = self._estimate_video_decode_bytes(
                    video_items, video_configs
                )
                admit_error, reserved = await self.await_gpu_bytes(estimate)
                if admit_error is not None:
                    logger.warning("[video-admit] rejected: %s", admit_error)
                    raise MMError(
                        admit_error, code=HTTPStatus.SERVICE_UNAVAILABLE
                    )
                tasks = [
                    loop.run_in_executor(
                        self.io_executor,
                        glm_sample_and_decode_sync,
                        video,
                        num_decode_workers,
                        video_configs[idx] if idx < len(video_configs) else {},
                    )
                    for idx, video in enumerate(video_items)
                ]
                try:
                    video_processed = await asyncio.gather(*tasks)
                finally:
                    self._release_admit_bytes(reserved)
                videos, video_metadata = map(list, zip(*video_processed))
                video_processor_kwargs["do_sample_frames"] = False
                video_processor_kwargs["return_metadata"] = True
                if video_metadata:
                    video_processor_kwargs["video_metadata"] = video_metadata
                return videos, video_processor_kwargs
            else:
                raise NotImplementedError(
                    f"Video processing is not supported for {self.model_type} model."
                )
        finally:
            self._close_video_decoders(video_items)

    async def _flatten_and_load_data_by_modality(self, mm_items, modality):
        """
        Flatten mm_items structure, load multimodal data concurrently, and restore original structure.

        Returns:
            Same structure as load_mm_items would return, support for image/audio
        """
        # Handle single mm_item (not a list)
        if not isinstance(mm_items, (list, tuple)):
            futures, _ = self.submit_data_loading_tasks([mm_items], [modality])
            return await asyncio.wrap_future(futures[0])

        # Handle nested list (list of lists)
        if len(mm_items) > 0 and isinstance(mm_items[0], (list, tuple)):
            # Flatten nested structure
            flat_data = []
            flat_indices = []  # Track which group each item belongs to
            for group_idx, item_group in enumerate(mm_items):
                for item in item_group:
                    flat_data.append(item)
                    flat_indices.append(group_idx)

            # Submit all tasks concurrently
            futures, _ = self.submit_data_loading_tasks(
                flat_data, [modality] * len(flat_data)
            )

            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            results = await asyncio.gather(*async_futures)

            # Restore nested structure
            nested_results = [[] for _ in range(len(mm_items))]
            for idx, result in zip(flat_indices, results):
                nested_results[idx].append(result)

            return nested_results

        # Handle simple list
        else:
            futures, _ = self.submit_data_loading_tasks(
                mm_items, [modality] * len(mm_items)
            )
            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            return await asyncio.gather(*async_futures)

    def get_num_patches(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of raw patches (before merge/sampling). Used for pixel_values slicing."""
        if modality == Modality.AUDIO:
            return int(grid.item())
        else:
            return int(grid[0] * grid[1] * grid[2])

    def _kimi_tokens_from_patch_grid(self, grid: Union[torch.Tensor, List[int]]) -> int:
        """MoonViT + tpool: output len is (h//mh)*(w//mw); temporal dim is pooled (not t*h*w/merge^2)."""
        if isinstance(grid, torch.Tensor):
            flat = grid.flatten()
            _t, h, w = (int(x) for x in flat[:3].tolist())
        else:
            _t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        merge_h, merge_w = self.model_config.hf_config.vision_config.merge_kernel_size
        return (h * w) // (merge_h * merge_w)

    def get_num_tokens(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of tokens (after 2x2 merge). Used for mm_embedding slicing."""
        if modality == Modality.AUDIO:
            input_length = self.get_num_patches(grid, modality)
            return self._get_feat_extract_output_lengths(input_length)
        else:
            if (
                self.model_type in ["kimi_k25", "kimi_vl"]
                and modality == Modality.IMAGE
            ):
                return self._kimi_tokens_from_patch_grid(grid)
            merge_size = getattr(self.image_processor, "merge_size", 2)
            return self.get_num_patches(grid, modality) // (merge_size**2)

    def slice_embedding(
        self, mm_embedding: torch.Tensor, grid_thw: List, modality: Modality
    ) -> List[torch.Tensor]:
        """Slice a concatenated embedding tensor into individual image embeddings."""
        slices, offset = [], 0
        for grid in grid_thw:
            count = self.get_num_tokens(grid, modality)
            slices.append(mm_embedding[offset : offset + count])
            offset += count
        return slices

    def _calculate_hashes_from_features(
        self, mm_feature: torch.Tensor, grid_thw: List, modality: Modality
    ) -> List[str]:
        """CPU Task: Compute hashes based on processed feature patches."""
        hashes, offset = [], 0
        logger.info(f"{mm_feature.shape=} with {modality=}")
        for grid in grid_thw:
            num_patches = self.get_num_patches(grid, modality)
            feature_slice = mm_feature[offset : offset + num_patches]
            tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
            tmp_item.set_pad_value()
            hashes.append(tmp_item.hash)
            offset += num_patches
        return hashes

    async def _encode_missing(
        self,
        mm_feature: torch.Tensor,
        mm_inputs: dict,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        """
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)

        # 1. Slice mm_feature to get only the patches for missing mm items
        sub_feature_list = []
        offsets = [0]
        curr = 0
        for g in grid_thw:
            curr += self.get_num_patches(g, modality)
            offsets.append(curr)

        for idx in indices:
            sub_feature_list.append(mm_feature[offsets[idx] : offsets[idx + 1]])

        sub_feature = torch.cat(sub_feature_list, dim=0)

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": modality,
                "feature": _convert(sub_feature),
            }
        )

        for k, v in mm_inputs.items():
            if k in _mm_feature_attrs.get(modality, []):
                continue
            val = _convert(v)
            if k in _mm_grid_attrs.get(modality, []):
                mm_item.set(k, val[indices])
            else:
                mm_item.set(k, val)

        def _run_vit():
            with torch.inference_mode():
                embeddings = get_feature_fn([mm_item]).cpu()
                if embeddings.ndim != 2:
                    embeddings = embeddings.reshape(-1, embeddings.shape[-1])
                return embeddings

        loop = asyncio.get_running_loop()
        new_embeddings = await loop.run_in_executor(self.gpu_executor, _run_vit)

        sub_grids = [grid_thw[i] for i in indices]
        return self.slice_embedding(new_embeddings, sub_grids, modality)

    async def encode_with_global_cache(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        # mm_inputs: dict
        mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)
        modality_name = modality.name.lower()
        execution_path = "global_cache"
        if self.metrics is not None:
            self.metrics.observe_mm_items_per_request(num_items, modality_name)

        # Step 1: Rank 0 checks global cache and broadcasts hit/miss mask to all ranks.
        cache_tic = time.perf_counter()
        if self.rank == 0:
            if hashes is None:
                loop = asyncio.get_running_loop()
                mm_hashes = await loop.run_in_executor(
                    self.io_executor,
                    lambda: self._calculate_hashes_from_features(
                        mm_feature, grid_thw, modality
                    ),
                )
            else:
                mm_hashes = hashes
            exist_mask = await self.mm_global_cache.batch_is_exist(mm_hashes)
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mm_hashes = None
            mask_tensor = torch.zeros(num_items, dtype=torch.int32)

        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]
        if self.metrics is not None:
            item_tokens = [self.get_num_tokens(grid, modality) for grid in grid_thw]
            self.metrics.record_cache_tokens(
                sum(item_tokens[i] for i in hit_indices),
                sum(item_tokens),
                modality=modality_name,
            )
            self.metrics.record_cache_files(len(hit_indices), num_items, modality_name)
            self.metrics.observe_stage(
                modality_name,
                execution_path,
                "cache_lookup",
                time.perf_counter() - cache_tic,
            )

        # Step 2: All ranks run ViT together on cache-miss images.
        new_slices = []
        if missing_indices:
            vit_tic = time.perf_counter()
            new_slices = await self._encode_missing(
                mm_feature, mm_inputs, missing_indices, modality, get_feature_fn
            )
            if self.metrics is not None:
                self.metrics.observe_vit(
                    modality_name,
                    execution_path,
                    time.perf_counter() - vit_tic,
                )

        # Step 3: Rank 0 prefetches cache-hit embeddings from global cache.
        prefetch_status = torch.tensor([1], dtype=torch.int32)

        if self.rank == 0:
            if hit_indices:
                prefetch_tic = time.perf_counter()
                hit_hashes = [mm_hashes[i] for i in hit_indices]
                hit_tokens = [
                    self.get_num_tokens(grid_thw[i], modality) for i in hit_indices
                ]
                self.mm_global_cache.prefetch(req_id, hit_hashes, hit_tokens, modality)

                try:

                    async def _wait_prefetch():
                        while not self.mm_global_cache.check_prefetch_progress(req_id):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(_wait_prefetch(), timeout=60.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    prefetch_status[0] = 0
                finally:
                    if self.metrics is not None:
                        self.metrics.observe_stage(
                            modality_name,
                            execution_path,
                            "cache_prefetch",
                            time.perf_counter() - prefetch_tic,
                        )

        # Step 4: Broadcast prefetch result to all ranks so they stay in sync.
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                prefetch_status,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        # Step 5: If prefetch failed, all ranks fallback to ViT for the hit mm items.
        if prefetch_status.item() == 0 and hit_indices:
            logger.info(
                f"Req {req_id}: Prefetch failed, all ranks running ViT fallback "
                f"for {len(hit_indices)} mm items."
            )
            fallback_tic = time.perf_counter()
            fallback_slices = await self._encode_missing(
                mm_feature, mm_inputs, hit_indices, modality, get_feature_fn
            )
            if self.metrics is not None:
                self.metrics.observe_vit(
                    modality_name,
                    execution_path,
                    time.perf_counter() - fallback_tic,
                    stage="vit_fallback",
                )
        else:
            fallback_slices = None

        # Step 6: Rank 0 assembles final embedding and prepares for sending.
        if self.rank == 0:
            final_slices = [None] * num_items

            for i, idx in enumerate(missing_indices):
                final_slices[idx] = new_slices[i]

            # Fill in cache-hit embeddings (from prefetch or fallback)
            if prefetch_status.item() == 1 and hit_indices:
                cached_slices = self.mm_global_cache.get_embeddings(
                    [mm_hashes[i] for i in hit_indices]
                )
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = cached_slices[i]
            elif fallback_slices is not None:
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = fallback_slices[i]

            mm_embedding = torch.cat(final_slices, dim=0)
            if self.metrics is not None:
                self.metrics.observe_embedding(
                    modality_name, execution_path, int(mm_embedding.shape[0])
                )

            # Background insert: store newly computed embeddings into global cache.
            # Includes both original misses and fallback-recomputed hits.
            all_new_hashes = [mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [mm_hashes[i] for i in hit_indices]
                all_new_slices += list(fallback_slices)

            if all_new_hashes:

                async def _background_insert():
                    insert_tic = time.perf_counter()
                    try:
                        await asyncio.to_thread(
                            self.mm_global_cache.insert_batch,
                            all_new_hashes,
                            all_new_slices,
                        )
                    finally:
                        if self.metrics is not None:
                            self.metrics.observe_stage(
                                modality_name,
                                execution_path,
                                "cache_store",
                                time.perf_counter() - insert_tic,
                            )

                task = asyncio.create_task(_background_insert())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

            aux_data = _build_mm_aux_data(mm_inputs)
            self.embedding_to_send[req_id] = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                grid_thw,
                modality,
                mm_embedding,
                **aux_data,
            )
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        else:
            return (0, 0, 0, None, None)

    async def _flatten_and_load_audios(self, mm_items):
        """
        Flatten mm_items structure, load audios concurrently, and restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.AUDIO)

    async def _flatten_and_load_images(self, mm_items):
        """
        Flatten mm_items structure, load images concurrently, and restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.IMAGE)

    def _calculate_timestamps(self, indices, video_fps: float, merge_size: int = 2):
        """Calculate timestamps for video frames, used for qwen3_vl models."""
        # refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L255
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        # Frames are merged by merge_size, so we need to average the timestamps
        # between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    @staticmethod
    def _flatten_nested_items(items):
        if not isinstance(items, (list, tuple)):
            return [items]

        flat = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flat.extend(MMEncoder._flatten_nested_items(item))
            else:
                flat.append(item)
        return flat

    def _normalize_kimi_encoder_images(self, images):
        """Normalize Kimi image inputs for the image processor call."""
        from PIL import Image as PILImage

        def wrap_one(img):
            if isinstance(img, dict) and img.get("type") in ("image", "video_chunk"):
                return [img]
            if isinstance(img, PILImage.Image):
                return [{"type": "image", "image": img}]
            return [img]

        if not images:
            return images

        # Disagg may supply nested lists from grouped routing.
        images = self._flatten_nested_items(images)

        # Kimi-VL image processor expects a flat list of concrete images.
        if self.model_type == "kimi_vl":
            normalized = []
            for img in images:
                if (
                    isinstance(img, dict)
                    and img.get("type") == "image"
                    and "image" in img
                ):
                    inner = img["image"]
                    if isinstance(inner, (list, tuple)):
                        normalized.extend(self._flatten_nested_items(inner))
                    else:
                        normalized.append(inner)
                else:
                    normalized.append(img)
            return normalized

        # Kimi-K2.5 vision processor expects media dicts.
        normalized = []
        for img in images:
            wrapped = wrap_one(img)
            for media in wrapped:
                # Some pipelines may produce {"type": "image", "image": [PIL]}.
                # Split it into one media item per concrete image object.
                if (
                    isinstance(media, dict)
                    and media.get("type") == "image"
                    and isinstance(media.get("image"), (list, tuple))
                ):
                    for inner in self._flatten_nested_items(media["image"]):
                        normalized.append({**media, "image": inner})
                else:
                    normalized.append(media)

        return normalized

    async def _process_mm_items(self, mm_items, modality):
        if modality == Modality.IMAGE and self.image_processor:
            image_urls, image_configs = glm_split_mm_items(
                mm_items, glm_mm_sampling_keys
            )
            if image_urls is not None:
                mm_items = image_urls
            images = await self._flatten_and_load_images(mm_items)
            image_config = dict(self.vision_config.get("image", {}))
            if "glm" in self.model_type:
                budget = glm_budget_kwargs(
                    self.image_processor,
                    user_max_image_tokens=glm_max_image_tokens_from_configs(
                        image_configs
                    ),
                )
                if budget is not None:
                    image_config.update(budget)
            if self.model_type in ["kimi_k25", "kimi_vl"]:
                images = self._normalize_kimi_encoder_images(images)
            loop = asyncio.get_running_loop()
            processor_input = await loop.run_in_executor(
                self.io_executor,
                lambda: self.image_processor(images=images, **image_config),
            )
            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_image_feature
            else:
                get_feature_method = self.model.get_image_feature
        elif modality == Modality.VIDEO and self.video_processor:
            videos, video_processor_kwargs = await self._flatten_and_load_videos(
                mm_items
            )
            # Internal shard metadata is consumed here and must not be passed
            # to the Hugging Face processor.
            shard_meta = video_processor_kwargs.pop("_shard_meta", None)
            if shard_meta is not None and shard_meta.get("count", 0) == 0:
                if hasattr(self.model, "thinker"):
                    get_feature_method = self.model.thinker.get_video_feature
                else:
                    get_feature_method = self.model.get_video_feature
                return {"_empty_video_shard": True}, get_feature_method
            video_device = self.vision_config.get("video", {}).get("device")
            if video_device is not None and "device" not in video_processor_kwargs:
                video_processor_kwargs["device"] = video_device
            loop = asyncio.get_running_loop()
            processor_input = await loop.run_in_executor(
                self.io_executor,
                lambda: self.video_processor(
                    videos=videos, **video_processor_kwargs
                ),
            )
            # Get additional video metadata
            if (
                self.model_type
                in [
                    "qwen3_vl",
                    "qwen3_vl_moe",
                    "qwen3_5",
                    "qwen3_5_moe",
                    "intern_s2_preview",
                ]
                and video_processor_kwargs.get("video_metadata", None) is not None
            ):
                # For qwen3-vl/qwen3.5 models, we need to store the video timestamps
                video_metadata = video_processor_kwargs["video_metadata"]
                try:
                    merge_size = (
                        self.model_config.hf_config.vision_config.spatial_merge_size
                    )
                except (AttributeError, KeyError):
                    merge_size = 2  # Default merge_size

                video_timestamps = []
                for metadata in video_metadata:
                    video_fps = metadata.get("fps", None) or 24  # original video fps
                    frames_indices = metadata.get("frames_indices", None)
                    timestamps = self._calculate_timestamps(
                        frames_indices, video_fps, merge_size
                    )
                    video_timestamps.append(timestamps)
                processor_input["video_timestamps"] = video_timestamps
            elif "glm" in self.model_type:
                if shard_meta is not None:
                    # Shard 0 reports whole-video metadata once. Every Encoder
                    # still feeds its local grid into the ViT.
                    processor_input.pop("video_metadata", None)
                    local_grid = processor_input.get("video_grid_thw")
                    if shard_meta["shard_idx"] == 0:
                        global_indices = shard_meta["global_indices"]
                        fps = shard_meta["fps"]
                        global_timestamps = [i / fps for i in global_indices][::2]
                        processor_input["video_timestamps"] = [global_timestamps]
                        if local_grid is not None and len(local_grid) > 0:
                            height = int(local_grid[0][1])
                            width = int(local_grid[0][2])
                            processor_input["_reported_grid"] = torch.tensor(
                                [[shard_meta["n_units"], height, width]]
                            )
                    else:
                        processor_input["video_timestamps"] = None
                        processor_input["_reported_grid"] = None
                else:
                    video_metadata = processor_input.get("video_metadata", None)
                    video_timestamps = []
                    if video_metadata is not None:
                        for metadata in video_metadata:
                            ts = getattr(metadata, "timestamps", None)
                            if ts is None and isinstance(metadata, dict):
                                ts = metadata.get("timestamps", None)
                            if ts is None:
                                raise InternalError(
                                    "GLM-V video metadata missing timestamps: "
                                    f"{metadata}"
                                )
                            video_timestamps.append(list(ts)[::2])
                    processor_input["video_timestamps"] = video_timestamps
                    processor_input.pop("video_metadata", None)
            elif (
                self.model_type in ["qwen2_5_vl", "qwen2_5_omni", "qwen3_omni_moe"]
                and processor_input.get("video_grid_thw", None) is not None
            ):
                # For omni/qwen2_5_vl models, calculate second_per_grid_ts for rotary embedding
                video_grid_thw = processor_input["video_grid_thw"]
                try:
                    temporal_patch_size = self.video_processor.temporal_patch_size
                except AttributeError:
                    temporal_patch_size = 2  # Default temporal_patch_size
                # get sampled fps, default: 2
                fps_list = [
                    self.vision_config.get("video", {}).get("fps", None) or 2
                ] * len(video_grid_thw)
                second_per_grid_ts = [(temporal_patch_size / fps) for fps in fps_list]
                second_per_grid_ts_tensor = torch.tensor(
                    second_per_grid_ts, dtype=torch.float32
                )
                processor_input["second_per_grid_ts"] = second_per_grid_ts_tensor

            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_video_feature
            else:
                get_feature_method = self.model.get_video_feature
        elif modality == Modality.AUDIO and self.audio_processor:
            audios = await self._flatten_and_load_audios(mm_items)
            audio_config = self.vision_config.get("audio", {})
            processor_input = self.audio_processor.feature_extractor(
                audios, **audio_config
            )
            processor_input["feature_attention_mask"] = processor_input.pop(
                "attention_mask"
            )
            # convert to same format as image/video
            input_lengths = torch.tensor(
                processor_input["feature_attention_mask"].sum(-1), dtype=torch.long
            )
            processor_input["audio_feature_lens_raw"] = input_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)
            processor_input["audio_feature_lens"] = output_lengths
            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_audio_feature
            else:
                get_feature_method = self.model.get_audio_feature
        else:
            raise ValueError(
                f"Currently only support image, video and audio modalities, {modality} modality has no processor available."
            )

        return processor_input, get_feature_method

    async def _encode(self, mm_items, modality: Modality) -> torch.Tensor:
        modality_name = modality.name.lower()
        execution_path = "single"
        try:
            preprocess_tic = time.perf_counter()
            mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
            if self.metrics is not None:
                self.metrics.observe_stage(
                    modality_name,
                    execution_path,
                    "preprocess",
                    time.perf_counter() - preprocess_tic,
                )
                self.metrics.observe_mm_items_per_request(len(mm_items), modality_name)
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except MMError:
            raise
        except TimeoutError as e:
            raise MMError(str(e), code=HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")

        if isinstance(mm_inputs, dict) and mm_inputs.get("_empty_video_shard"):
            hidden_size = int(
                getattr(self.model_config.hf_config, "hidden_size", 1) or 1
            )
            empty = torch.zeros((0, hidden_size), dtype=torch.bfloat16)
            if self.metrics is not None:
                self.metrics.observe_embedding(modality_name, execution_path, 0)
            return None, empty, _build_mm_aux_data({})

        try:
            # support mm_cache
            mm_embedding = None
            mm_hash = None

            mm_item = MultimodalDataItem.from_dict(
                {
                    "modality": modality,
                    "feature": _convert(_get_mm_feature(mm_inputs, modality)),
                }
            )
            internal_keys = {"_reported_grid"}
            for k, v in mm_inputs.items():
                if k in _mm_feature_attrs[modality]:
                    continue
                if k in internal_keys:
                    continue
                mm_item.set(k, _convert(v))

            if self.server_args.enable_prefix_mm_cache:
                cache_tic = time.perf_counter()

                def _hash_mm_item():
                    mm_item.set_pad_value()
                    item_hash = mm_item.hash
                    return item_hash, MultiModalStaticCache.combine_hashes([item_hash])

                loop = asyncio.get_running_loop()
                item_hash, mm_hash = await loop.run_in_executor(
                    self.io_executor, _hash_mm_item
                )
                async with self.mm_cache_lock:
                    mm_cache = self.mm_cache.get([item_hash])
                    if mm_cache is not None:
                        mm_embedding = mm_cache.embedding
                if self.metrics is not None:
                    self.metrics.observe_stage(
                        modality_name,
                        execution_path,
                        "cache_lookup",
                        time.perf_counter() - cache_tic,
                    )

            if mm_embedding is None:
                vit_tic = time.perf_counter()

                def _run_vit():
                    with torch.inference_mode():
                        return get_feature_fn([mm_item]).cpu()

                loop = asyncio.get_running_loop()
                mm_embedding = await loop.run_in_executor(
                    self.gpu_executor, _run_vit
                )
                if self.metrics is not None:
                    self.metrics.observe_vit(
                        modality_name,
                        execution_path,
                        time.perf_counter() - vit_tic,
                    )
                if len(mm_embedding.shape) != 2:
                    mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

            if self.server_args.enable_prefix_mm_cache:
                async with self.mm_cache_lock:
                    entries_before = len(self.mm_cache)
                    already_present = self.mm_cache.has(mm_hash)
                    inserted = self.mm_cache.set(
                        mm_hash, EmbeddingResult(embedding=mm_embedding)
                    )
                    entries_after = len(self.mm_cache)
                if self.metrics is not None:
                    cache_hit = mm_cache is not None
                    total_tokens = int(mm_embedding.shape[0])
                    self.metrics.record_cache_tokens(
                        total_tokens if cache_hit else 0,
                        total_tokens,
                        modality=modality_name,
                    )
                    self.metrics.record_cache_files(
                        1 if cache_hit else 0, 1, modality=modality_name
                    )
                    added = 0 if already_present else (1 if inserted else 0)
                    self.metrics.inc_cache_evictions(
                        modality_name,
                        max(0, added - (entries_after - entries_before)),
                    )
                    self.metrics.set_cache_state(
                        self.mm_cache.current_size, entries_after
                    )
            if self.metrics is not None:
                self.metrics.observe_embedding(
                    modality_name, execution_path, int(mm_embedding.shape[0])
                )
            if self.profiler is not None:
                self.profiler.step()

            aux_data = _build_mm_aux_data(mm_inputs)
            return (
                _get_mm_grid_dim(mm_inputs, modality, self.model_type),
                mm_embedding,
                aux_data,
            )
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")

    async def _send(
        self,
        embedding: torch.Tensor,
        mm_data: EmbeddingData,
        session_id=None,
        buffer_address=None,
        prefill_host=None,
        embedding_port=None,
        url=None,
        meta_only=False,
    ):
        if self.server_args.encoder_transfer_backend == "mooncake" and not meta_only:
            reg, ret = 0, 0
            if embedding is not None and embedding.nbytes > 0:

                def _transfer_sync():
                    reg = self.engine.register(embedding.data_ptr(), embedding.nbytes)
                    if reg != 0:
                        return reg, -1
                    try:
                        ret = self.engine.transfer_sync(
                            session_id,
                            embedding.data_ptr(),
                            buffer_address,
                            embedding.nbytes,
                        )
                    finally:
                        self.engine.deregister(embedding.data_ptr())
                    return reg, ret

                loop = asyncio.get_running_loop()
                reg, ret = await loop.run_in_executor(self.executor, _transfer_sync)

            mm_data.embedding = None

            # RDMA write not confirmed: turn this into an error frame so the
            # language side fails the request loudly, instead of reading a
            # never-/partially-written buffer -> silent garbage output.
            if reg != 0 or ret != 0:
                logger.error(
                    "mooncake RDMA write failed for req_id=%s "
                    "(register=%s, transfer=%s)",
                    mm_data.req_id,
                    reg,
                    ret,
                )
                mm_data.error_msg = (
                    f"mooncake RDMA write failed (register={reg}, transfer={ret})"
                )
                mm_data.error_code = HTTPStatus.INTERNAL_SERVER_ERROR

        # Send ack/data
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        if meta_only:
            serialized_data = pickle.dumps(mm_data.copy_without_embedding())
            buffer = None
        elif self.server_args.encoder_transfer_backend == "mooncake":
            serialized_data = pickle.dumps(mm_data)
            buffer = None
        else:
            new_mm_data = mm_data.copy_without_embedding()
            if new_mm_data.error_msg is not None:
                buffer = None
                serialized_data = pickle.dumps(new_mm_data)
            else:
                embedding_tensor = TensorWrapper(mm_data.embedding)
                serialized_data = pickle.dumps(new_mm_data)
                buffer = embedding_tensor.__buffer__()

        # Use thread pool executor for parallel ZMQ send operations
        def send_with_socket():
            sock = self.sync_context.socket(zmq.PUSH)
            config_socket(sock, zmq.PUSH)
            try:
                sock.connect(endpoint)
                if buffer is not None:
                    sock.send_multipart([serialized_data, buffer], copy=False)
                else:
                    sock.send_multipart([serialized_data], copy=False)
            finally:
                sock.close()

        transfer_tic = time.perf_counter()
        outcome = "success"
        try:
            await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)
        except Exception:
            outcome = "error"
            raise
        finally:
            if self.metrics is not None and not meta_only:
                modality_name = getattr(mm_data.modality, "name", str(mm_data.modality)).lower()
                num_bytes = int(embedding.nbytes) if embedding is not None else 0
                self.metrics.observe_transfer_attempt(
                    modality_name,
                    time.perf_counter() - transfer_tic,
                    outcome,
                    num_bytes,
                )

    async def encode(self, mm_items, modality: Modality, req_id, num_parts, part_idx):
        try:
            grid_dim, mm_embedding, aux_data = await self._encode(mm_items, modality)

            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    grid_dim,
                    modality,
                    mm_embedding,
                    **aux_data,
                )
                self.embedding_to_send[req_id] = mm_data
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        except Exception as e:
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(f"Rank {self.rank} encode failed: {error_msg} {error_code = }")
            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    None,
                    modality,
                    error_msg=error_msg,
                    error_code=error_code,
                )
                self.embedding_to_send[req_id] = mm_data
                logger.debug(f"Created error EmbeddingData: {mm_data}")
            return 0, 0, 0, error_msg, error_code

    # For zmq_to_tokenizer zmq_to_scheduler and mooncake
    async def send(
        self,
        req_id,
        prefill_host,
        embedding_port,
        session_id=None,
        buffer_address=None,
        meta_only=False,
    ):
        mm_data: EmbeddingData = self.embedding_to_send[req_id]
        await self._send(
            mm_data.embedding,
            mm_data,
            session_id=session_id,
            buffer_address=buffer_address,
            prefill_host=prefill_host,
            embedding_port=embedding_port,
            meta_only=meta_only,
        )

    # For zmq_to_scheduler
    async def send_with_url(
        self,
        req_id,
    ):
        mm_data = self.embedding_to_send.get(req_id)
        if not mm_data:
            return
        sent_urls: Set[str] = set()
        all_tasks: List[Tuple[asyncio.Task, str]] = []
        start_time = asyncio.get_running_loop().time()
        timeout = self.send_timeout
        cond = await get_condition(req_id)

        try:
            while True:
                async with rid_lock:
                    current_targets = rid_to_receive_endpoint.get(req_id, set()).copy()
                    expected_count = rid_to_receive_count.get(req_id)

                new_targets = current_targets - sent_urls

                if new_targets:
                    logger.info(
                        f"Found {len(new_targets)} new endpoints for {req_id}. Starting tasks..."
                    )
                    for url in new_targets:
                        task = asyncio.create_task(
                            self._send(
                                mm_data.embedding,
                                mm_data,
                                url=url,
                            )
                        )
                        all_tasks.append((task, url))
                        sent_urls.add(url)  # Mark as handled immediately
                if expected_count is not None and len(sent_urls) >= expected_count:
                    logger.info(
                        f"All {expected_count} endpoints initiated for {req_id}. Breaking loop."
                    )
                    break
                remaining = timeout - (asyncio.get_running_loop().time() - start_time)
                if remaining <= 0:
                    logger.error(
                        f"[{req_id}] Timeout! Sent {len(sent_urls)}/{expected_count}"
                    )
                    break

                async with cond:
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue

            if all_tasks:
                logger.info(
                    f"Loop finished. Awaiting completion of {len(all_tasks)} sending tasks..."
                )
                tasks_only = [t[0] for t in all_tasks]
                results = await asyncio.gather(*tasks_only, return_exceptions=True)

                # Process results and log errors
                for i, result in enumerate(results):
                    url = all_tasks[i][1]  # Retrieve URL associated with the task
                    if isinstance(result, Exception):
                        logger.error(f"Failed to send to {url}: {result}")
                    else:
                        logger.debug(f"Successfully sent to {url}")

            logger.info(f"All tasks completed for req_id: {req_id}")

        finally:
            logger.info(f"Cleaning up resources for req_id {req_id}")
            async with rid_lock:
                rid_to_receive_endpoint.pop(req_id, None)
                rid_to_receive_count.pop(req_id, None)
            async with cond_dict_lock:
                rid_to_cond.pop(req_id, None)
            self.embedding_to_send.pop(req_id, None)

    async def get_embedding_port(self, prefill_url):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            response = await session.post(
                f"{prefill_url}/embedding_bootstrap",
                json={"embedding_port": None},
            )
            response_json = await response.json()
            return response_json["embedding_port"]


class EncoderProfiler:
    def __init__(self, rank: int):
        self.rank = rank
        self.profiler = None
        self.steps_left = None
        self.output_dir = None
        self.prefix = None
        self.profile_id = None

    def start(self, obj: ProfileReq):
        if self.profiler is not None:
            return False, "profiling already running"

        output_dir = obj.output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = obj.profile_prefix or "encoder"
        self.profile_id = str(time.time())

        activities = obj.activities or ["CPU", "GPU"]
        torch_activities = []
        if "CPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_memory = "MEM" in activities
        if not torch_activities and not profile_memory:
            return False, "no supported activities"

        self.profiler = torch.profiler.profile(
            activities=torch_activities,
            with_stack=True if obj.with_stack is None else obj.with_stack,
            record_shapes=False if obj.record_shapes is None else obj.record_shapes,
            profile_memory=profile_memory,
        )
        self.profiler.start()
        self.steps_left = obj.num_steps
        logger.info(
            f"Encoder profiling started. output_dir={self.output_dir} profile_id={self.profile_id}"
        )
        return True, None

    def step(self):
        if self.profiler is None:
            return
        self.profiler.step()
        if self.steps_left is not None:
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.stop()

    def stop(self):
        if self.profiler is None:
            return False, "profiling not running"
        self.profiler.stop()
        filename = f"{self.prefix}-rank{self.rank}-{self.profile_id}.trace.json"
        trace_path = os.path.join(self.output_dir, filename)
        self.profiler.export_chrome_trace(trace_path)
        logger.info("Encoder profiling saved to: %s", trace_path)
        self.profiler = None
        self.steps_left = None
        return True, None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Only rank 0 owns embedding_to_send; run the orphan-embedding sweeper
    # there to prevent a slow leak when /send never arrives.
    sweeper_task = None
    if (
        encoder is not None
        and getattr(encoder, "rank", 0) == 0
        and hasattr(encoder, "embedding_to_send")
    ):
        sweeper_task = asyncio.create_task(encoder._sweep_stale_embeddings_loop())
    try:
        yield
    finally:
        if sweeper_task is not None:
            sweeper_task.cancel()
            try:
                await sweeper_task
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(lifespan=_lifespan)
encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        if isinstance(request, ProfileReq):
            if request.type == ProfileReqType.START_PROFILE:
                if encoder.profiler is None:
                    encoder.profiler = EncoderProfiler(encoder.rank)
                encoder.profiler.start(request)
            else:
                encoder.profiler.stop()
        else:
            modality = Modality.from_str(request["modality"])
            is_video_shard = _set_video_shard_context(request, modality)
            if encoder.mm_global_cache is not None and not is_video_shard:
                await encoder.encode_with_global_cache(
                    mm_items=request["mm_items"],
                    modality=modality,
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes", None),
                )
            else:
                await encoder.encode(
                    mm_items=request["mm_items"],
                    modality=modality,
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                )


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def launch_server(server_args: ServerArgs):
    global encoder
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        add_prometheus_middleware(app)
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


async def get_condition(rid):
    async with cond_dict_lock:
        if rid not in rid_to_cond:
            rid_to_cond[rid] = asyncio.Condition()
        return rid_to_cond[rid]


@app.post("/encode")
async def handle_encode_request(request: dict):
    req_tic = time.perf_counter()
    req_id = request["req_id"]
    metrics_started = False
    encoder_time_stats = None
    modality_name = "unknown"
    execution_path = "single"
    outcome = "success"
    error_stage = "none"
    include_canonical_metrics = True
    try:

        def start_background_send(req_id):
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        # broadcast request
        request.update({"enter_time": time.time()})
        modality = Modality.from_str(request["modality"])
        is_video_shard = _set_video_shard_context(request, modality)
        modality_name = modality.name.lower()
        meta_only = (
            request.get("role") == "decode"
            and encoder.server_args.encoder_transfer_backend == "mooncake"
        )
        include_canonical_metrics = not meta_only
        if meta_only:
            execution_path = "meta_only"
        elif encoder.mm_global_cache is not None and not is_video_shard:
            execution_path = "global_cache"
        if encoder.metrics is not None:
            encoder.metrics.request_started(
                modality_name,
                execution_path,
                include_canonical=include_canonical_metrics,
            )
            metrics_started = True
            if include_canonical_metrics:
                encoder_time_stats = EncoderReqTimeStats(modality=modality_name)
                encoder_time_stats.set_metrics_collector(encoder.metrics)
                encoder_time_stats.set_mm_encode_start_time(req_tic)
        for socket in send_sockets:
            socket.send_pyobj(request)
        if encoder.mm_global_cache is not None and not is_video_shard:
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode_with_global_cache(
                    mm_items=request["mm_items"],
                    modality=modality,
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes", None),
                )
            )
        else:
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode(
                    mm_items=request["mm_items"],
                    modality=modality,
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                )
            )

        if error_msg:
            outcome = "error"
            error_stage = "encode"
            if encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                if request["embedding_port"] is None:
                    start_background_send(req_id)
                else:
                    for port in request["embedding_port"]:
                        await encoder.send(
                            req_id=req_id,
                            prefill_host=request["prefill_host"],
                            embedding_port=port,
                        )
            return ORJSONResponse(
                status_code=error_code,
                content={"status": "error", "message": error_msg, "req_id": req_id},
            )
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            if request.get("role") == "decode":
                await encoder.send(
                    req_id=req_id,
                    prefill_host=request["prefill_host"],
                    embedding_port=request["embedding_port"],
                    meta_only=True,
                )
                encoder.embedding_to_send.pop(req_id, None)
                return ORJSONResponse(content=None)
            del request["mm_items"]
            request.update(
                {
                    "embedding_size": nbytes,
                    "embedding_len": embedding_len,
                    "embedding_dim": embedding_dim,
                }
            )
            return ORJSONResponse(content=request)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
            logger.info(f"{request['embedding_port'] = }")
            if request["embedding_port"] is None:
                await encoder.send_with_url(
                    req_id=request["req_id"],
                )
            else:
                assert type(request["embedding_port"]) == list
                tasks = []
                for embedding_port in request["embedding_port"]:
                    tasks.append(
                        encoder.send(
                            req_id=request["req_id"],
                            prefill_host=request["prefill_host"],
                            embedding_port=embedding_port,
                        )
                    )
                await asyncio.gather(*tasks)
                encoder.embedding_to_send.pop(request["req_id"], None)
            return ORJSONResponse(content=None)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
            await encoder.send(
                req_id=request["req_id"],
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
            )
            encoder.embedding_to_send.pop(request["req_id"], None)
            return ORJSONResponse(content=None)
    except Exception as e:
        outcome = "error"
        error_stage = "encode"
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
        )
    finally:
        if encoder is not None and encoder.metrics is not None:
            if encoder_time_stats is not None:
                encoder_time_stats.set_mm_encode_end_time()
            if metrics_started:
                encoder.metrics.request_finished(
                    modality_name,
                    execution_path,
                    outcome,
                    error_stage,
                    include_canonical=include_canonical_metrics,
                )


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    encoder.embedding_to_send.pop(request["req_id"], None)
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    rid = request["req_id"]
    async with rid_lock:
        global rid_to_receive_endpoint
        if rid not in rid_to_receive_endpoint:
            rid_to_receive_endpoint[rid] = set()
            rid_to_receive_count[rid] = request["receive_count"]
        assert rid_to_receive_count[rid] == request["receive_count"]
        rid_to_receive_endpoint[rid].add(request["receive_url"])
    cond = await get_condition(rid)
    async with cond:
        cond.notify_all()


@app.get("/health")
@app.get("/health_generate")
async def health_generate():
    """
    Health check endpoint for the encoder server.
    Performs a dummy encode to verify the encoder is functional.
    Returns 200 if the encoder is healthy, 503 otherwise.
    """
    if encoder is None:
        return Response(status_code=503)

    # Skip the dummy encode when real requests are already in flight — the
    # ongoing traffic already proves liveness, matching the scheduler's
    # `is_fully_idle`-based health-check skip pattern.
    if encoder.embedding_to_send:
        return Response(status_code=200)

    # Pick the first available modality for the dummy encode
    if encoder.image_processor is not None:
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif encoder.audio_processor is not None:
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor available, fall back to liveness check only
        return Response(status_code=200)

    try:
        req_id = f"{HEALTH_CHECK_RID_PREFIX}_{time.time()}"

        dummy_request = {
            "mm_items": mm_items,
            "modality": modality.name,
            "req_id": req_id,
            "num_parts": 1,
            "part_idx": 0,
        }

        # Broadcast to other TP ranks so distributed ops stay in sync
        for socket in send_sockets:
            socket.send_pyobj(dummy_request)

        # Run encode on rank 0 with timeout
        _, _, _, error_msg, _ = await asyncio.wait_for(
            encoder.encode(
                mm_items=mm_items,
                modality=modality,
                req_id=req_id,
                num_parts=1,
                part_idx=0,
            ),
            timeout=HEALTH_CHECK_TIMEOUT,
        )

        # Clean up stored embedding
        encoder.embedding_to_send.pop(req_id, None)

        if error_msg:
            logger.error(f"Encoder health check failed: {error_msg}")
            return Response(status_code=503)

        return Response(status_code=200)

    except asyncio.TimeoutError:
        logger.error(f"Encoder health check timed out after {HEALTH_CHECK_TIMEOUT}s")
        return Response(status_code=503)
    except Exception as e:
        logger.error(f"Encoder health check failed: {e}")
        return Response(status_code=503)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = None
    if obj is None:
        req = ProfileReq(ProfileReqType.START_PROFILE)
    else:
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=obj.output_dir,
            start_step=obj.start_step,
            num_steps=obj.num_steps,
            activities=obj.activities,
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_by_stage=obj.profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=obj.merge_profiles,
            profile_prefix=obj.profile_prefix,
            profile_stages=obj.profile_stages,
        )
    for socket in send_sockets:
        socket.send_pyobj(req)
    if encoder.profiler is None:
        encoder.profiler = EncoderProfiler(encoder.rank)
    ok, msg = encoder.profiler.start(req)
    if ok:
        detail = (
            f"Start profiling. output_dir={encoder.profiler.output_dir} "
            f"profile_id={encoder.profiler.profile_id}\n"
        )
        return Response(content=detail, status_code=200)
    return Response(
        content=(msg or "Start profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(ProfileReqType.STOP_PROFILE)
    for socket in send_sockets:
        socket.send_pyobj(req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )
