import asyncio
import concurrent.futures
import contextlib
import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback
from collections import defaultdict
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
try:
    support_tvF = True
    from torchvision.transforms.v2 import functional as tvF
except ImportError as e:
    support_tvF = False
    print(f"Failed to import torchvision.transforms.v2 functional (tvF): {e}")


from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.encode_receiver import (
    EmbeddingData,
    RdmaRegRefcount,
    rdma_pool_enabled,
)
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
    _split_video_items as glm_split_video_items,
    glm_sample_and_decode_sync,
    glm_decode_frames_at,
    glm_sample_frame_indices,
    preprocess_video_frames_sync as glm_preprocess_video_frames_sync,
)
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    load_audio,
    load_image,
    load_video,
    random_uuid,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
rid_to_err_msg: Dict[str, str] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}

use_image_processor_gpu = (
    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
)

ENCODER_MAX_BATCH_SIZE = envs.SGLANG_ENCODER_MAX_BATCH_SIZE.get()
# Watchdog: max time to wait for a batched /encode result. Bounds HTTP latency
# if the batch worker stalls (NCCL hang, dead worker proc, etc.).
ENCODER_REQ_TIMEOUT = envs.SGLANG_ENCODER_REQ_TIMEOUT.get()
ENCODER_MAX_CONCURRENT_VIDEO = envs.SGLANG_ENCODER_MAX_CONCURRENT_VIDEO.get()

# Byte-based video admission (await_gpu_bytes):
_GPU_ADMIT_POLL_S = 0.2
_GPU_ADMIT_IDLE_SLACK = 1.1
_VIDEO_ADMIT_BYTES_FACTOR = 2.0


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
    Modality.IMAGE: ["image_grid_thw", "image_grid_hws"],
    Modality.VIDEO: ["video_grid_thw"],
    Modality.AUDIO: ["audio_feature_lens_raw"],
}

_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


def _get_mm_grid_dim(mm_inputs, modality):
    # DP-sharded decode: use global grid override if present. Scoped to VIDEO
    # so a mixed image+video request doesn't hand the video grid to the image
    # modality lookup.
    if modality == Modality.VIDEO and "_video_grid_thw_global" in mm_inputs:
        return mm_inputs["_video_grid_thw_global"]
    for attr in _mm_grid_attrs[modality]:
        if attr in mm_inputs:
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

        torch.get_device_module(self.device).set_device(self.gpu_id)

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

        # Byte-based video admission 
        self._admit_lock = asyncio.Lock()
        try:
            self._admit_baseline_bytes = torch.cuda.memory_allocated(self.gpu_id)
        except Exception:
            self._admit_baseline_bytes = 0
        self._admit_reserved_bytes = 0

        self.context = zmq.asyncio.Context(2)
        # send_with_url fans out one _send per receiving TP rank over this shared
        # context; give it tp_size IO threads (min 2) so the transfers overlap
        # instead of serializing on the default single thread.
        self.sync_context = zmq.Context(max(2, server_args.tp_size))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_ENCODER_MM_LOAD_WORKERS", 4))
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
                # Sender-side zero-copy registration: register the embedding
                # tensor in place (reference counted) instead of copying into a
                # pool. No extra buffer memory; the refcount prevents a
                # concurrent send's deregister from tearing down an MR still in
                # use (sender-side teardown race). Disabled (pool off, i.e.
                # SGLANG_MC_RDMA_POOL_MAX_MB/MAX_BUFFERS == 0) -> original
                # per-request register.
                self._use_rdma_pool = rdma_pool_enabled()
                self._rdma_reg = (
                    RdmaRegRefcount(self.engine) if self._use_rdma_pool else None
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
        claimed by a /send (prefill LLM timed out / crashed).

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
                    if mm_data is None or getattr(mm_data, "created_at", 0) > deadline:
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

        # torchvision "fast" processors accept torch.Tensor inputs, letting us
        # pre-convert PIL -> CHW uint8 in the parallel IO pool (see
        # `_load_single_item`) instead of the serial per-image `pil_to_tensor`
        # inside the HF processor call.
        from transformers.image_processing_utils_fast import BaseImageProcessorFast

        self.image_processor_is_fast = isinstance(
            self.image_processor, BaseImageProcessorFast
        )

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
                if support_tvF and self.image_processor_is_fast and not isinstance(img, torch.Tensor):
                    # Do the PIL->tensor conversion in the IO thread pool
                    # (GIL-releasing, so it scales across threads) rather than
                    # serially inside the HF fast processor. Byte-identical to HF.
                    img = tvF.pil_to_tensor(img)
                return img
            elif modality == Modality.VIDEO:
                # NOTE: In load_video(video_file, use_gpu=True) the second
                # positional argument is use_gpu, not frame_count_limit (load_video
                # does not consume frame_count_limit; frame sampling happens later in
                # preprocess_video). Previously frame_count_limit (=None) was wrongly
                # passed as use_gpu, forcing device to always be "cpu" and permanently
                # disabling the GPU video decode path. Now GPU torchcodec CUDA decoding
                # is explicitly enabled via use_image_processor_gpu.
                return load_video(data, use_gpu=self.use_image_processor_gpu)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

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
        # qwen3_omni_moe
        elif self.model_type == "qwen3_omni_moe":
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

    async def _dp_sharded_decode_single_video(
        self, vr, video_config, num_decode_workers, *, tp_rank, tp_size,
        video_processor_kwargs, precomputed_indices=None,
    ):
        """DP-sharded decode: each TP rank decodes only its assigned temporal
        units from a single video, so no rank holds all frames at once.

        Returns ``(videos, video_processor_kwargs)`` like ``_flatten_and_load_videos``.
        ``video_processor_kwargs`` carries an extra ``_dp_meta`` dict for
        ``_process_mm_items`` to build global metadata from.
        """
        video_config = video_config or {}
        video_fps = vr.avg_fps
        total_num_frames = len(vr)
        duration = total_num_frames / video_fps if video_fps else 0

        # Global sampling: deterministic, identical on all ranks.
        if precomputed_indices is not None:
            global_indices = precomputed_indices
        else:
            global_indices = glm_sample_frame_indices(
                total_num_frames,
                video_fps,
                duration,
                target_fps=video_config.get("fps"),
                max_frame_count=video_config.get("max_frames"),
            )
        n_units = len(global_indices) // 2  # temporal_patch_size=2

        # Contiguous split: rank r owns a contiguous block of temporal units
        # (remainder to the low ranks), so the ViT all_gather reassembles in
        # global order without any permutation.
        base, rem = divmod(n_units, tp_size)
        gpu_sample_counts = [base + (1 if r < rem else 0) for r in range(tp_size)]
        start = sum(gpu_sample_counts[:tp_rank])
        count = gpu_sample_counts[tp_rank]

        # Unit u -> frame pair (2u, 2u+1); a contiguous unit block is a
        # contiguous frame slice.
        local_frame_indices = list(global_indices[2 * start : 2 * (start + count)])

        # Rank-local admission + decode. Any failure here (admission rejection,
        # decode error) must NOT unwind this rank alone: every rank enters this
        # function symmetrically and peers go on to the ViT all_gather, so a lone
        # bail-out would deadlock them. Capture the error, agree on a global
        # outcome below, then all ranks proceed or all abort together.
        loop = asyncio.get_running_loop()
        frames = None
        reserved = 0
        local_err = None
        try:
            # Byte-based admission on this rank's share.
            if local_frame_indices:
                try:
                    h, w = vr.frame_shape
                    est_bytes = len(local_frame_indices) * h * w * 3
                except Exception:
                    est_bytes = 0
                admit_err, reserved = await self.await_gpu_bytes(est_bytes)
                if admit_err is not None:
                    logger.warning(f"[video-admit-dp] rejected: {admit_err}")
                    raise MMError(admit_err, code=HTTPStatus.SERVICE_UNAVAILABLE)

                # Decode only local frames.
                frames = await loop.run_in_executor(
                    self.io_executor,
                    glm_decode_frames_at,
                    vr,
                    local_frame_indices,
                    num_decode_workers,
                    video_config,
                )
        except Exception as e:
            local_err = e
            frames = None
        finally:
            self._release_admit_bytes(reserved)

        # Cross-rank agreement: MIN all-reduce of the per-rank success flag over
        # the attention-TP group. Any single failure (flag 0) aborts every rank,
        # so no peer is ever left waiting on the downstream ViT all_gather.
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        local_ok = 0 if local_err is not None else 1
        if tp_size > 1:
            flag = torch.tensor([local_ok], dtype=torch.int32)
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MIN,
                group=get_attention_tp_group().cpu_group,
            )
            global_ok = int(flag.item())
        else:
            global_ok = local_ok

        if global_ok == 0:
            if local_err is not None:
                # Re-raise this rank's own error, preserving its code (MMError
                # 503, TimeoutError, ...) which _encode already maps correctly.
                raise local_err
            # A peer failed; abort symmetrically. Retryable 503, no deadlock.
            raise MMError(
                "peer TP rank failed during DP-sharded video decode; "
                "aborting to keep ranks in sync",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        if frames is not None:
            videos = [frames]
        else:
            # Empty rank: HF processor still needs a placeholder.
            import numpy as np
            h, w = vr.frame_shape
            videos = [np.zeros((0, h, w, 3), dtype=np.uint8)]

        video_processor_kwargs["do_sample_frames"] = False
        video_processor_kwargs["return_metadata"] = True
        # Global metadata for _process_mm_items to rebuild timestamps/grid.
        video_processor_kwargs["_dp_meta"] = {
            "global_indices": global_indices,
            "fps": video_fps,
            "n_units": n_units,
            "gpu_sample_counts": gpu_sample_counts,
        }
        return videos, video_processor_kwargs

    def _estimate_video_decode_bytes(self, video_items, video_configs) -> int:
        """Estimate a request's raw GPU decode footprint (bytes): per video
        sampled_frames * H * W * 3 (native-resolution NHWC uint8)."""
        total = 0
        for idx, vr in enumerate(video_items):
            cfg = video_configs[idx] if idx < len(video_configs) else {}
            try:
                total_frames = len(vr)
                fps = vr.avg_fps
                duration = total_frames / fps if fps else 0
                indices = glm_sample_frame_indices(
                    total_frames,
                    fps,
                    duration,
                    target_fps=cfg.get("fps"),
                    max_frame_count=cfg.get("max_frames"),
                )
                h, w = vr.frame_shape
                total += len(indices) * h * w * 3
            except Exception as e:
                logger.warning(
                    f"[video-admit] could not estimate decode bytes for item "
                    f"{idx}: {e}; treating as 0"
                )
        return total

    async def await_gpu_bytes(self, need_bytes: int) -> Tuple[Optional[str], int]:
        """Hold a video request OUT of decode until the GPU has room for it.

        need_bytes (raw estimate) is scaled by _VIDEO_ADMIT_BYTES_FACTOR (~2x
        K-way transient). Returns (None, reserved) when admitted
        """
        if need_bytes <= 0:
            return None, 0
        need = int(need_bytes * _VIDEO_ADMIT_BYTES_FACTOR)
        idle_ceiling = self._admit_baseline_bytes * _GPU_ADMIT_IDLE_SLACK
        max_wait = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()
        deadline = time.monotonic() + max_wait
        async with self._admit_lock:
            while True:
                free, total = torch.cuda.mem_get_info(self.gpu_id)
                allocated = torch.cuda.memory_allocated(self.gpu_id)
                reserved = torch.cuda.memory_reserved(self.gpu_id)
                available = (
                    (reserved - allocated) + free - self._admit_reserved_bytes
                )
                idle = allocated <= idle_ceiling and self._admit_reserved_bytes == 0
                if total <= 0 or available >= need or idle:
                    self._admit_reserved_bytes += need
                    return None, need
                if time.monotonic() >= deadline:
                    return (
                        f"GPU busy: video decode needs ~{need / 1e9:.1f}GB "
                        f"(est {need_bytes / 1e9:.1f}GB x{_VIDEO_ADMIT_BYTES_FACTOR:g}), "
                        f"available {available / 1e9:.1f}GB of {total / 1e9:.1f}GB "
                        f"(free {free / 1e9:.1f} + cache "
                        f"{(reserved - allocated) / 1e9:.1f} - pending "
                        f"{self._admit_reserved_bytes / 1e9:.1f}) after "
                        f"{max_wait:.0f}s"
                    ), 0
                await asyncio.sleep(_GPU_ADMIT_POLL_S)

    def _release_admit_bytes(self, reserved: int) -> None:
        """Drop a reservation once its frames are decoded (now counted in
        memory_allocated, so keeping it would double-count)."""
        if reserved:
            self._admit_reserved_bytes = max(0, self._admit_reserved_bytes - reserved)

    async def _flatten_and_load_videos(self, mm_items):
        if not isinstance(mm_items, (list, tuple)):
            mm_items = [mm_items]

        # Per-video sampling overrides may arrive as inline dict items in
        # video_data, e.g. {"url": ..., "fps": 1, "max_frames": 64}. Split the
        # decodable URL/bytes (for _load_single_item) from the sampling config,
        # which is realigned by index and forwarded to the GLM sampler below.
        # Non-GLM/framed paths ignore configs.
        video_urls, video_configs = glm_split_video_items(mm_items)
        if video_urls is not None:
            mm_items = video_urls

        futures, _ = self.submit_data_loading_tasks(
            mm_items, [Modality.VIDEO] * len(mm_items)
        )
        async_futures = [asyncio.wrap_future(f) for f in futures]
        video_items = await asyncio.gather(*async_futures)

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
            # GLM4V EPD fast path: sample frame indices on the decode side (matching
            # the HF GLM video processor exactly via glm_sample_frame_indices) and
            # decode ONLY the sampled frames with K-way segmented parallelism, then
            # tell the HF processor NOT to re-sample. Timestamps stay identical
            # because we pass the sampled frames_indices + original fps in
            # video_metadata (VideoMetadata.timestamps = frames_indices / fps).
            # The frame decode (get_frames_at) is synchronous/blocking with no await
            # point — offload to the IO thread pool so concurrent requests overlap.
            framed = any(isinstance(video, list) for video in video_items)
            loop = asyncio.get_running_loop()
            if framed:
                # Frame-list inputs have no underlying decoder/source to
                # parallelize; keep the decode-all + HF-sample behavior unchanged.
                tasks = [
                    loop.run_in_executor(
                        self.io_executor, glm_preprocess_video_frames_sync, video
                    )
                    for video in video_items
                ]
                video_processed = await asyncio.gather(*tasks)
                videos, video_metadata = map(list, zip(*video_processed))
                video_processor_kwargs["do_sample_frames"] = True
                video_processor_kwargs["return_metadata"] = True
                if video_metadata:
                    video_processor_kwargs["video_metadata"] = video_metadata
                return videos, video_processor_kwargs

            # Decoded-video fast path: sampled + K-way parallel decode.
            num_decode_workers = int(
                os.environ.get("SGLANG_ENCODER_GLM_VIDEO_DECODE_WORKERS", "4")
            )

            # DP-sharded decode: each TP rank decodes only its temporal units,
            # then the ViT path all_gathers to rebuild the full embedding —
            # avoids every rank holding all frames (large-video OOM). GPU-decode
            # only, since that is where the per-rank VRAM pressure exists.
            from sglang.srt.layers.dp_attention import (
                get_attention_tp_rank,
                get_attention_tp_size,
            )

            tp_size = get_attention_tp_size()
            _DP_DECODE_MIN_FRAMES = envs.SGLANG_DP_DECODE_MIN_FRAMES.get()
            if (
                self.use_image_processor_gpu
                and self.server_args.mm_enable_dp_encoder
                and tp_size > 1
                and len(video_items) == 1
            ):
                vr = video_items[0]
                cfg = video_configs[0] if video_configs else {}
                sampled = glm_sample_frame_indices(
                    len(vr), vr.avg_fps,
                    len(vr) / vr.avg_fps if vr.avg_fps else 0,
                    target_fps=cfg.get("fps"),
                    max_frame_count=cfg.get("max_frames"),
                )
                if len(sampled) >= _DP_DECODE_MIN_FRAMES:
                    return await self._dp_sharded_decode_single_video(
                        vr, cfg, num_decode_workers,
                        tp_rank=get_attention_tp_rank(),
                        tp_size=tp_size,
                        video_processor_kwargs=video_processor_kwargs,
                        precomputed_indices=sampled,
                    )

            # Byte-based admission: wait for GPU headroom before decoding.
            est_bytes = self._estimate_video_decode_bytes(video_items, video_configs)
            admit_err, reserved = await self.await_gpu_bytes(est_bytes)
            if admit_err is not None:
                logger.warning(f"[video-admit] rejected: {admit_err}")
                raise MMError(admit_err, code=HTTPStatus.SERVICE_UNAVAILABLE)

            tasks = [
                loop.run_in_executor(
                    self.io_executor,
                    glm_sample_and_decode_sync,
                    video,
                    num_decode_workers,
                    # per-video sampling overrides (fps/max_frames/max_tokens_per_frame)
                    video_configs[idx] if idx < len(video_configs) else {},
                )
                for idx, video in enumerate(video_items)
            ]
            try:
                video_processed = await asyncio.gather(*tasks)
            finally:
                # frames now counted in memory_allocated -> drop the reservation
                self._release_admit_bytes(reserved)
            videos, video_metadata = map(list, zip(*video_processed))
            # We already sampled -> HF must NOT sample again, but must echo metadata.
            video_processor_kwargs["do_sample_frames"] = False
            video_processor_kwargs["return_metadata"] = True
            if video_metadata:
                video_processor_kwargs["video_metadata"] = video_metadata
            return videos, video_processor_kwargs
        else:
            raise NotImplementedError(
                f"Video processing is not supported for {self.model_type} model."
            )

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

    def get_num_tokens(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of tokens (after 2x2 merge). Used for mm_embedding slicing."""
        if modality == Modality.AUDIO:
            input_length = self.get_num_patches(grid, modality)
            return self._get_feat_extract_output_lengths(input_length)
        else:
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
        grid_thw = _get_mm_grid_dim(mm_inputs, modality)

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

        with torch.inference_mode():
            new_embeddings = get_feature_fn([mm_item]).cpu()
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

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
        grid_thw = _get_mm_grid_dim(mm_inputs, modality)
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)

        # Step 1: Rank 0 checks global cache and broadcasts hit/miss mask to all ranks.
        if self.rank == 0:
            if hashes is None:
                mm_hashes = self._calculate_hashes_from_features(
                    mm_feature, grid_thw, modality
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

        # Step 2: All ranks run ViT together on cache-miss images.
        new_slices = []
        if missing_indices:
            new_slices = await self._encode_missing(
                mm_feature, mm_inputs, missing_indices, modality, get_feature_fn
            )

        # Step 3: Rank 0 prefetches cache-hit embeddings from global cache.
        prefetch_status = torch.tensor([1], dtype=torch.int32)

        if self.rank == 0:
            if hit_indices:
                hit_hashes = [mm_hashes[i] for i in hit_indices]
                hit_tokens = [
                    self.get_num_tokens(grid_thw[i], modality) for i in hit_indices
                ]
                self.mm_global_cache.prefetch(req_id, hit_hashes, hit_tokens, modality)

                try:

                    async def _wait_prefetch():
                        while not self.mm_global_cache.check_prefetch_progress(req_id):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(
                        _wait_prefetch(),
                        timeout=envs.SGLANG_ENCODER_SEND_TIMEOUT.get(),
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    prefetch_status[0] = 0

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
            fallback_slices = await self._encode_missing(
                mm_feature, mm_inputs, hit_indices, modality, get_feature_fn
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

            # Background insert: store newly computed embeddings into global cache.
            # Includes both original misses and fallback-recomputed hits.
            all_new_hashes = [mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [mm_hashes[i] for i in hit_indices]
                all_new_slices += list(fallback_slices)

            if all_new_hashes:

                async def _background_insert():
                    await asyncio.to_thread(
                        self.mm_global_cache.insert_batch,
                        all_new_hashes,
                        all_new_slices,
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

    async def _process_mm_items(self, mm_items, modality):
        if modality == Modality.IMAGE and self.image_processor:
            images = await self._flatten_and_load_images(mm_items)
            image_config = self.vision_config.get("image", {})
            processor_input = self.image_processor(images=images, **image_config)
            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_image_feature
            else:
                get_feature_method = self.model.get_image_feature
        elif modality == Modality.VIDEO and self.video_processor:
            videos, video_processor_kwargs = await self._flatten_and_load_videos(
                mm_items
            )
            # Pop DP metadata before passing to HF processor (it's not an HF kwarg)
            dp_meta = video_processor_kwargs.pop("_dp_meta", None)
            processor_input = self.video_processor(
                videos=videos, **video_processor_kwargs
            )
            # Get additional video metadata
            if (
                self.model_type in ["qwen3_vl", "qwen3_vl_moe"]
                and video_processor_kwargs.get("video_metadata", None) is not None
            ):
                # For qwen3-vl models, we need to store the video timestamps
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
                if dp_meta is not None:
                    # DP-sharded: build GLOBAL timestamps from global indices;
                    # local processor metadata only has this rank's subset.
                    gidx = dp_meta["global_indices"]
                    gfps = dp_meta["fps"]
                    global_ts = [i / gfps for i in gidx][::2]
                    processor_input["video_timestamps"] = [global_ts]
                    processor_input.pop("video_metadata", None)
                    # DP sharding info for get_video_feature / ViT path.
                    processor_input["dp_decode_sharded"] = True
                    processor_input["dp_meta"] = dp_meta
                    # Override grid_thw with the global grid, not the subset.
                    local_grid = processor_input.get("video_grid_thw")
                    if local_grid is not None and len(local_grid) > 0:
                        h_val = int(local_grid[0][1])
                        w_val = int(local_grid[0][2])
                        processor_input["_video_grid_thw_global"] = torch.tensor(
                            [[dp_meta["n_units"], h_val, w_val]]
                        )
                else:
                    # GLM4V: the HF video processor sampled frames internally and
                    # returned the sampled metadata. Reproduce HF's per-frame
                    # timestamps (metadata.timestamps[::2]) so the language side can
                    # rebuild the same interleaved frame/timestamp token layout.
                    video_metadata = processor_input.get("video_metadata", None)
                    video_timestamps = []
                    if video_metadata is not None:
                        for metadata in video_metadata:
                            ts = getattr(metadata, "timestamps", None)
                            if ts is None and isinstance(metadata, dict):
                                ts = metadata.get("timestamps", None)
                            if ts is None:
                                raise InternalError(
                                    f"GLM4V video metadata missing timestamps: {metadata}"
                                )
                            video_timestamps.append(list(ts)[::2])
                    processor_input["video_timestamps"] = video_timestamps
                    # video_metadata is not transferable / needed downstream.
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

    async def _encode(
        self, mm_items, modality: Modality, meta_only: bool = False
    ) -> torch.Tensor:
        try:
            mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except MMError:
            raise  # preserve intentional codes (e.g. admission 503)
        except TimeoutError as e:
            # Stuck K-way decode: transient/contention, retryable -> 503 not 400.
            raise MMError(str(e), code=HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")
        if meta_only:
            return _get_mm_grid_dim(mm_inputs, modality), None, _build_mm_aux_data(
                mm_inputs
            )
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
            # Keys internal to DP sharding — don't convert with _convert
            _dp_internal_keys = {"dp_decode_sharded", "dp_meta", "_video_grid_thw_global"}
            is_dp_sharded = mm_inputs.get("dp_decode_sharded", False)
            for k, v in mm_inputs.items():
                if k in _mm_feature_attrs[modality]:
                    continue
                if k in _dp_internal_keys:
                    continue
                mm_item.set(k, _convert(v))

            # Thread DP sharding info onto mm_item so get_video_feature can see it
            if is_dp_sharded:
                mm_item.set("dp_decode_sharded", True)
                mm_item.set("dp_meta", mm_inputs["dp_meta"])

            if self.server_args.enable_prefix_mm_cache and not is_dp_sharded:
                mm_item.set_pad_value()
                mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
                async with self.mm_cache_lock:
                    mm_cache = self.mm_cache.get([mm_item.hash])
                    if mm_cache is not None:
                        mm_embedding = mm_cache.embedding

            if mm_embedding is None:
                with torch.inference_mode():
                    mm_embedding: torch.Tensor = get_feature_fn([mm_item])
                    mm_embedding = mm_embedding.cpu()
                if len(mm_embedding.shape) != 2:
                    mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

            if self.server_args.enable_prefix_mm_cache and not is_dp_sharded:
                async with self.mm_cache_lock:
                    self.mm_cache.set(mm_hash, EmbeddingResult(embedding=mm_embedding))
            if self.profiler is not None:
                self.profiler.step()

            aux_data = _build_mm_aux_data(mm_inputs)
            return _get_mm_grid_dim(mm_inputs, modality), mm_embedding, aux_data
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")
        finally:
            free, total = torch.cuda.mem_get_info(self.gpu_id)
            if total > 0 and (total - free) / total > 0.25:
                torch.cuda.empty_cache()

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
            if self._use_rdma_pool:
                # Zero-copy: register the embedding in place (reference
                # counted) and transfer directly -- no pool buffer, no copy.
                # The tensor must be contiguous so [data_ptr, data_ptr+nbytes)
                # covers the data.
                if not embedding.is_contiguous():
                    embedding = embedding.contiguous()
                n = embedding.nbytes
                src_addr = self._rdma_reg.acquire(embedding)
                ret = -1
                try:
                    ret = self.engine.transfer_sync(
                        session_id, src_addr, buffer_address, n
                    )
                finally:
                    # transfer_sync is synchronous; safe to drop the
                    # registration (refcounted) as soon as it returns.
                    self._rdma_reg.release(src_addr)
                reg = 0
            else:
                # Fallback (pool off): original per-request register +
                # deregister.
                reg = self.engine.register(
                    embedding.data_ptr(), embedding.nbytes
                )
                ret = -1
                if reg == 0:
                    ret = self.engine.transfer_sync(
                        session_id,
                        embedding.data_ptr(),
                        buffer_address,
                        embedding.nbytes,
                    )
                    self.engine.deregister(embedding.data_ptr())

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
            # Decode role: ship metadata only (grid_dim / shape / video_timestamps /
            # second_per_grid_ts). No RDMA transfer happened above, so there is no
            # embedding to reference. Reuse the error-path single-frame shape.
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

        await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)

    async def encode(
        self,
        mm_items,
        modality: Modality,
        req_id,
        num_parts,
        part_idx,
        meta_only: bool = False,
    ):
        try:
            grid_dim, mm_embedding, aux_data = await self._encode(
                mm_items, modality, meta_only=meta_only
            )

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
            if meta_only:
                return 0, 0, 0, None, None
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

    async def encode_request(
        self, req: dict, modality: Modality, meta_only: bool = False
    ):
        """Single-request encode dispatcher: picks cache vs no-cache path."""
        if self.mm_global_cache is not None and not meta_only:
            return await self.encode_with_global_cache(
                mm_items=req["mm_items"],
                modality=modality,
                req_id=req["req_id"],
                num_parts=req["num_parts"],
                part_idx=req["part_idx"],
                hashes=req.get("hashes"),
            )
        return await self.encode(
            mm_items=req["mm_items"],
            modality=modality,
            req_id=req["req_id"],
            num_parts=req["num_parts"],
            part_idx=req["part_idx"],
            meta_only=meta_only,
        )

    async def batch_encode(
        self, requests: List[dict], modality: Modality
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        """Cross-request encoder fusion (image only here).

        When --enable-prefix-mm-cache is set, items already in the local
        mm_cache are served from cache and only the cache-miss subset is fused
        through the ViT. The cache key scheme (combine_hashes([item.hash]))
        matches the per-request _encode path, so both paths share entries.
        """
        # items_per_req counts grid entries so per-request slicing of
        # grid_dim/final_slices stays aligned. For IMAGE on this branch each leaf
        # maps 1:1 to a grid (no Kimi-style tile expansion), so leaf count == grid
        # count per request.
        flat_items, items_per_req = [], []
        for req in requests:
            leaves = MMEncoder._flatten_nested_items(req["mm_items"])
            flat_items.extend(leaves)
            items_per_req.append(len(leaves))
        total = sum(items_per_req)

        try:
            mm_inputs, get_feat = await self._process_mm_items(flat_items, modality)
        except NotImplementedError as e:
            return self._batch_set_error(
                requests, modality, InternalError(f"Not implemented error: {e}")
            )
        except Exception as e:
            return self._batch_set_error(
                requests, modality, BadRequestError(f"Failed to process mm items: {e}")
            )

        try:
            mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
            grid_dim = _get_mm_grid_dim(mm_inputs, modality)
            if len(grid_dim) != total:
                return self._batch_set_error(
                    requests,
                    modality,
                    InternalError(
                        f"Grid count mismatch for {self.model_type}/"
                        f"{modality.name}: {len(flat_items)} leaves across "
                        f"{len(requests)} requests → expected {total} grids "
                        f"(per-req {items_per_req}), but processor produced "
                        f"{len(grid_dim)}."
                    ),
                )

            final_slices: List[Optional[torch.Tensor]] = [None] * total

            # Cache-aware fusion: serve items already in mm_cache from cache and
            # only run the ViT on the miss subset. Hashes are computed from the
            # processed feature patches (same scheme as the per-request _encode
            # path), so missing_indices is identical across TP ranks and the
            # _encode_missing collective stays shape-aligned.
            missing_indices = list(range(total))
            item_hashes: List[Optional[int]] = [None] * total
            if self.server_args.enable_prefix_mm_cache:
                item_hashes = self._calculate_hashes_from_features(
                    mm_feature, grid_dim, modality
                )
                missing_indices = []
                async with self.mm_cache_lock:
                    for idx, h in enumerate(item_hashes):
                        cached = self.mm_cache.get([h])
                        if cached is not None:
                            final_slices[idx] = cached.embedding
                        else:
                            missing_indices.append(idx)

            if missing_indices:
                new_slices = await self._encode_missing(
                    mm_feature,
                    mm_inputs,
                    missing_indices,
                    modality,
                    get_feat,
                )
                for slot, emb in zip(missing_indices, new_slices):
                    final_slices[slot] = emb

                if self.server_args.enable_prefix_mm_cache:
                    async with self.mm_cache_lock:
                        for slot, emb in zip(missing_indices, new_slices):
                            mm_hash = MultiModalStaticCache.combine_hashes(
                                [item_hashes[slot]]
                            )
                            self.mm_cache.set(
                                mm_hash, EmbeddingResult(embedding=emb)
                            )

            if self.profiler is not None:
                for _ in requests:
                    self.profiler.step()
            # No aux_data here: batch_encode only handles IMAGE, and
            # _build_mm_aux_data only extracts video-meta fields.
            results = []
            offset = 0
            for req, n in zip(requests, items_per_req):
                slices = final_slices[offset : offset + n]
                emb = slices[0] if n == 1 else torch.cat(slices, dim=0)
                if self.rank == 0:
                    self.embedding_to_send[req["req_id"]] = EmbeddingData(
                        req["req_id"],
                        req["num_parts"],
                        req["part_idx"],
                        grid_dim[offset : offset + n],
                        modality,
                        emb,
                    )
                results.append((emb.nbytes, emb.shape[0], emb.shape[1], None, None))
                offset += n
            return results
        except Exception as e:
            return self._batch_set_error(
                requests, modality, InternalError(f"Internal encoding error: {e}")
            )

    def _batch_set_error(
        self, requests: List[dict], modality: Modality, exc: Exception
    ) -> List[Tuple[int, int, int, str, int]]:
        code = getattr(exc, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
        msg = str(exc)
        logger.error(f"Rank {self.rank} batch_encode failed: {msg} {code = }")
        if self.rank == 0:
            for req in requests:
                self.embedding_to_send[req["req_id"]] = EmbeddingData(
                    req["req_id"],
                    req["num_parts"],
                    req["part_idx"],
                    None,
                    modality,
                    error_msg=msg,
                    error_code=code,
                )
        return [(0, 0, 0, msg, code)] * len(requests)

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
        mm_data: Optional[EmbeddingData] = self.embedding_to_send.get(req_id)
        if mm_data is None:
            logger.warning(
                f"send: no embedding for req_id={req_id} (already sent/reclaimed); skipping"
            )
            return
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


class PendingRequest:
    __slots__ = ("request", "future", "submit_time")

    def __init__(self, request: dict, loop: asyncio.AbstractEventLoop):
        self.request = request
        self.future: asyncio.Future = loop.create_future()
        self.submit_time = time.time()


# Only IMAGE is fused here. VIDEO can't fuse (per-video preprocess kwargs vary),
# and AUDIO on this branch keeps the per-request path.
_BATCHABLE_MODALITIES = {Modality.IMAGE}


class EncoderScheduler:
    """Aggregate concurrent /encode requests into bounded image batches."""

    def __init__(
        self,
        encoder: "MMEncoder",
        send_sockets: List[zmq.Socket],
        max_batch_size: int,
        request_timeout: float = ENCODER_REQ_TIMEOUT,
    ):
        self.encoder = encoder
        self.send_sockets = send_sockets
        self.max_batch_size = max(1, int(max_batch_size))
        self.request_timeout = max(1.0, float(request_timeout))
        self.pending_queue: "asyncio.Queue[PendingRequest]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())
            logger.info(
                f"EncoderScheduler started with max_batch_size={self.max_batch_size}"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        # Reject any requests still queued so their HTTP handlers don't hang.
        while True:
            try:
                pending = self.pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))

    async def submit(self, request: dict) -> Tuple:
        pending = PendingRequest(request, asyncio.get_running_loop())
        await self.pending_queue.put(pending)
        try:
            return await asyncio.wait_for(pending.future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            if not pending.future.done():
                pending.future.cancel()
            req_id = request.get("req_id")
            logger.error(
                f"EncoderScheduler.submit timed out after {self.request_timeout}s "
                f"for req_id={req_id}"
            )
            raise

    async def _collect_batch(self) -> List[PendingRequest]:
        batch = [await self.pending_queue.get()]
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _batch_worker(self) -> None:
        while True:
            batch: List[PendingRequest] = []
            try:
                batch = await self._collect_batch()
                groups: Dict[Modality, List[PendingRequest]] = defaultdict(list)
                for p in batch:
                    groups[
                        Modality.from_str(p.request.get("modality", "image"))
                    ].append(p)
                for modality, group in groups.items():
                    await self._dispatch_group(group, modality)
            except asyncio.CancelledError:
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(RuntimeError("EncoderScheduler stopped"))
                raise
            except Exception as e:
                logger.error(
                    f"Error in EncoderScheduler batch worker: {e}", exc_info=True
                )
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)

    @staticmethod
    def _validate_request_shape(req: dict) -> Optional[str]:
        # Cheap pre-broadcast checks: shape errors that don't require running
        # the HF processor. Once a request reaches TP workers they enter
        # batch_encode and expect to join its collectives — a malformed batch
        # that makes rank-0 bail mid-flight would deadlock the workers.
        if not isinstance(req, dict):
            return f"request is not a dict: {type(req).__name__}"
        if not req.get("req_id"):
            return "missing req_id"
        if not req.get("mm_items"):
            return "missing or empty mm_items"
        if "num_parts" not in req or "part_idx" not in req:
            return "missing num_parts / part_idx"
        h = req.get("hashes")
        if h is not None and not isinstance(h, (list, tuple, str, int, bytes)):
            return f"hashes must be list/scalar, got {type(h).__name__}"
        return None

    async def _dispatch_group(
        self, group: List[PendingRequest], modality: Modality
    ) -> None:
        # Video / non-batchable can't fuse.
        if modality not in _BATCHABLE_MODALITIES:
            await self._dispatch_per_request(group, modality)
            return

        # Drop structurally-bad requests before broadcasting; otherwise TP
        # workers would join batch_encode collectives that rank-0 has already
        # abandoned.
        valid: List[PendingRequest] = []
        for p in group:
            if p.future.done():
                logger.info(
                    f"Skipping cancelled req_id={p.request.get('req_id')} before batch broadcast"
                )
                continue
            err = self._validate_request_shape(p.request)
            if err is None:
                valid.append(p)
                continue
            logger.error(f"Dropping req_id={p.request.get('req_id')} from batch: {err}")
            if not p.future.done():
                p.future.set_exception(BadRequestError(err))
        if not valid:
            return
        group = valid

        requests = [p.request for p in group]
        start = time.time()
        for sock in self.send_sockets:
            sock.send_pyobj(
                {
                    "type": "batch_encode",
                    "modality": modality.name,
                    "requests": requests,
                    "enter_time": start,
                }
            )

        logger.info(f"Dispatching batch of {len(group)} {modality.name} requests")

        try:
            results = await self.encoder.batch_encode(requests, modality)
            if len(group) > 1:
                logger.info(
                    f"Batch of {len(group)} {modality.name} requests completed in "
                    f"{(time.time() - start) * 1000:.1f}ms"
                )
        except Exception as e:
            # batch_encode normally catches and returns errors via _batch_set_error.
            # If it raised, rank-0 may have skipped a collective broadcast, leaving
            # TP workers stuck. Don't try to recover — fail every pending future
            # and let the client retry. Re-broadcasting would risk a deadlock.
            logger.error(f"batch_encode raised: {e}", exc_info=True)
            for p in group:
                if not p.future.done():
                    p.future.set_exception(e)
            return

        if len(results) != len(group):
            err = RuntimeError(
                f"batch_encode returned {len(results)} results for {len(group)} requests"
            )
            logger.error(str(err))
            for p in group:
                if not p.future.done():
                    p.future.set_exception(err)
            return

        for p, result in zip(group, results):
            if not p.future.done():
                p.future.set_result(result)

    async def _dispatch_per_request(
        self,
        group: List[PendingRequest],
        modality: Modality,
    ) -> None:
        for p in group:
            req = p.request
            if p.future.done():
                logger.info(
                    f"Skipping cancelled req_id={req.get('req_id')} before per-request encode"
                )
                continue
            try:
                for sock in self.send_sockets:
                    sock.send_pyobj(req)
                result = await self.encoder.encode_request(req, modality)
                if not p.future.done():
                    p.future.set_result(result)
            except Exception as e:
                logger.error(
                    f"Per-request encode failed for req_id={req.get('req_id')}: {e}"
                )
                if not p.future.done():
                    p.future.set_exception(e)


encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []
encoder_scheduler: Optional[EncoderScheduler] = None
# Bounds concurrent in-flight video encodes on rank 0 (None = unlimited).
video_encode_gate: Optional[asyncio.Semaphore] = None


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    global encoder_scheduler, video_encode_gate
    sweeper_task = None
    if encoder is not None:
        encoder_scheduler = EncoderScheduler(
            encoder, send_sockets, max_batch_size=ENCODER_MAX_BATCH_SIZE
        )
        encoder_scheduler.start()
        if ENCODER_MAX_CONCURRENT_VIDEO > 0:
            video_encode_gate = asyncio.Semaphore(ENCODER_MAX_CONCURRENT_VIDEO)
        # Only rank 0 owns embedding_to_send; run the orphan-embedding sweeper
        # there to prevent a slow leak when /send never arrives.
        if getattr(encoder, "rank", 0) == 0 and hasattr(
            encoder, "embedding_to_send"
        ):
            sweeper_task = asyncio.create_task(
                encoder._sweep_stale_embeddings_loop()
            )
    try:
        yield
    finally:
        if sweeper_task is not None:
            sweeper_task.cancel()
            try:
                await sweeper_task
            except (asyncio.CancelledError, Exception):
                pass
        if encoder_scheduler is not None:
            await encoder_scheduler.stop()


app = FastAPI(lifespan=_lifespan)


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
        elif isinstance(request, dict) and request.get("type") == "batch_encode":
            await encoder.batch_encode(
                request["requests"],
                Modality.from_str(request["modality"]),
            )
        else:
            if encoder.mm_global_cache is not None:
                await encoder.encode_with_global_cache(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes", None),
                )
            else:
                await encoder.encode(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
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
    req_id = request["req_id"]
    try:

        def start_background_send(req_id):
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        # broadcast request
        request.update({"enter_time": time.time()})
        modality = Modality.from_str(request["modality"])
        meta_only = (
            request.get("role") == "decode"
            and encoder.server_args.encoder_transfer_backend == "mooncake"
        )
        if (
            encoder_scheduler is not None
            and encoder.mm_global_cache is None
            and modality in _BATCHABLE_MODALITIES
            and not meta_only
        ):
            # Batched path: EncoderScheduler accumulates concurrent requests and
            # broadcasts the fused batch_encode task to TP workers itself.
            try:
                nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                    await encoder_scheduler.submit(request)
                )
            except asyncio.TimeoutError:
                return ORJSONResponse(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT,
                    content={
                        "status": "error",
                        "message": "encoder batch timed out",
                        "req_id": req_id,
                    },
                )
        elif meta_only:
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode_request(request, modality, meta_only=True)
            )
        else:
            # Per-request path (global cache enabled, or non-image modality).
            if modality == Modality.VIDEO and video_encode_gate is not None:
                # Decode is parallel but ViT is serial, so unbounded video
                # concurrency piles up resident decoded frames on rank 0; bound
                # the in-flight count. Gate wraps the broadcast so TP workers are
                # paced in lockstep with rank 0.
                async with video_encode_gate:
                    for socket in send_sockets:
                        socket.send_pyobj(request)
                    nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                        await encoder.encode_request(request, modality)
                    )
            else:
                for socket in send_sockets:
                    socket.send_pyobj(request)
                nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                    await encoder.encode_request(request, modality)
                )

        if error_msg:
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
                # Decode instances run the LM in PREBUILT mode and never consume
                # embedding values. Push a meta-only ZMQ frame now and skip the
                # RDMA /send handshake entirely (no embedding_size in the reply).
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
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        rid_to_err_msg[req_id] = error_msg
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
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
    Returns 200 if the encoder is initialized and ready.
    """
    if encoder is None:
        return Response(status_code=503)
    return Response(status_code=200)


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
