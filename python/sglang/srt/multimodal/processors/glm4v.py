import concurrent.futures
import json
import math
from typing import List, Union

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.utils import logger

try:
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
except ImportError:
    Glm5NextForConditionalGeneration = None

try:
    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None


class Glm4vImageProcessor(SGLangBaseProcessor):
    models = [
        m
        for m in [
            Glm4vForConditionalGeneration,
            Glm4vMoeForConditionalGeneration,
            Glm5NextForConditionalGeneration,
            GlmOcrForConditionalGeneration,
        ]
        if m is not None
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        from sglang.srt.constrained.glm.escape import (
            get_global_escaped_special_tokens,
        )

        escaped_tokens = get_global_escaped_special_tokens()
        self.IMAGE_TOKEN = escaped_tokens.get("<|image|>")
        self.VIDEO_TOKEN = escaped_tokens.get("<|video|>")
        self.IMAGE_START_TOKEN = escaped_tokens.get("<|begin_of_image|>")
        self.IMAGE_END_TOKEN = escaped_tokens.get("<|end_of_image|>")
        self.VIDEO_START_TOKEN = escaped_tokens.get("<|begin_of_video|>")
        self.VIDEO_END_TOKEN = escaped_tokens.get("<|end_of_video|>")

        # Token IDs
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id
        self.IM_START_TOKEN_ID = hf_config.image_start_token_id
        self.IM_END_TOKEN_ID = hf_config.image_end_token_id
        self.VIDEO_START_TOKEN_ID = hf_config.video_start_token_id
        self.VIDEO_END_TOKEN_ID = hf_config.video_end_token_id

        # Vision config
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 112 * 112
        self.MAX_PIXELS = 30000 * 28 * 28 * 2

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            video_token=self.VIDEO_TOKEN,
            # Note: For GLM4v videos, it uses the video token before tokenization but uses image token after tokenization
            video_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    def compute_mrope_positions(self, input_ids, mm_items):
        image_grid_thw = None
        video_grid_thw = None
        for item in mm_items:
            if "image_grid_thw" in item.model_specific_data:
                image_grid_thw = item.model_specific_data["image_grid_thw"]
            if "video_grid_thw" in item.model_specific_data:
                video_grid_thw = item.model_specific_data["video_grid_thw"]

        import torch

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids_tensor)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids_tensor,
            hf_config=self.hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    def build_input_ids_with_timestamps(
        self,
        prompt: Union[str, List[int]],
        embeddings: torch.Tensor,
        img_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        video_timestamps: List[list],
    ):
        """Reconstruct GLM-V input_ids on the EPD language side."""
        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        spatial_merge_size = getattr(self, "spatial_merge_size", 1)
        image_start_token_id = getattr(self, "IMAGE_START_TOKEN_ID", None)
        image_end_token_id = getattr(self, "IMAGE_END_TOKEN_ID", None)

        input_ids = []
        offsets = []
        modality_list = []
        cur_idx = 0

        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))

        img_idx = 0
        video_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            modality_list.append(modality)
            assert cur_idx <= mm_start_idx
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])

            if modality == Modality.IMAGE:
                mm_token_num = int(
                    img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
                )
                mm_offset_start = len(input_ids)
                input_ids.extend([img_token_id] * mm_token_num)
                offsets.append((mm_offset_start, len(input_ids) - 1))
                img_idx += 1
            elif modality == Modality.VIDEO:
                curr_timestamps = video_timestamps[video_idx]
                num_frames = int(video_grid_thw[video_idx][0])
                frame_seqlen = int(video_grid_thw[video_idx][1:].prod()) // (
                    spatial_merge_size**2
                )
                for frame_idx in range(num_frames):
                    if image_start_token_id is not None:
                        input_ids.append(image_start_token_id)
                    mm_offset_start = len(input_ids)
                    input_ids.extend([img_token_id] * frame_seqlen)
                    offsets.append((mm_offset_start, len(input_ids) - 1))
                    if image_end_token_id is not None:
                        input_ids.append(image_end_token_id)
                    timestamp_sec = curr_timestamps[frame_idx]
                    timestamp_tokens = self._tokenizer.encode(
                        f"{int(timestamp_sec)}", add_special_tokens=False
                    )
                    input_ids.extend(timestamp_tokens)
                video_idx += 1
            else:
                logger.warning(f"{modality} modality is not supported for GLM-V EPD.")
                continue
            cur_idx = mm_start_idx + 2
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """EPD language side: rebuild mm_inputs from precomputed embeddings."""
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        video_timestamps = kwargs.get("video_timestamps", None)

        input_ids, offsets, modality_list = self.build_input_ids_with_timestamps(
            prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
        )
        assert all(isinstance(modality, Modality) for modality in modality_list)
        assert len(set(modality_list)) == 1, (
            f"GLM-V EPD only supports a single modality per request, "
            f"got {set(modality_list)}"
        )
        modality = modality_list[0]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids_tensor,
            hf_config=self.hf_config,
            image_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_items = [
            MultimodalDataItem(
                modality=modality,
                offsets=offsets,
                precomputed_embeddings=(
                    embeddings.get(modality) if embeddings else None
                ),
            )
        ]

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        video_urls, video_configs = _split_video_items(request_obj.video_data)

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_urls,
            multimodal_tokens=self.mm_tokens,
        )

        video_metadata = None
        if base_output.videos:
            videos_processed = [
                await sample_video_sglang(video, video_config=cfg)
                for video, cfg in zip(base_output.videos, video_configs)
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        if video_metadata is not None:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
                video_metadata=video_metadata,
                do_sample_frames=False,
            )
        else:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            attention_mask=getattr(ret, "attention_mask", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )


def _video_metadata(total_num_frames, fps, duration, frames_indices):
    return {
        "total_num_frames": int(total_num_frames),
        "fps": float(fps),
        "duration": float(duration),
        "video_backend": "decord",
        "frames_indices": list(frames_indices),
    }


def _split_video_items(video_data):
    _SAMPLING_KEYS = ("fps", "max_frames", "max_tokens_per_frame")
    if video_data is None:
        return None, []
    items = video_data if isinstance(video_data, list) else [video_data]

    urls, configs = [], []
    for item in items:
        if isinstance(item, dict) and "format" not in item and "url" in item:
            urls.append(item["url"])
            configs.append(
                {k: item[k] for k in _SAMPLING_KEYS if item.get(k) is not None}
            )
        else:
            urls.append(item)
            configs.append({})
    return urls, configs


def glm_sample_frame_indices(
    total_frames,
    fps,
    duration,
    *,
    target_fps=None,
    max_frame_count=None,
    temporal_patch_size=None,
):
    if temporal_patch_size is None:
        temporal_patch_size = envs.SGLANG_GLM_VIDEO_TEMPORAL_PATCH_SIZE.get()
    max_frame_idx = total_frames - 1
    if not duration:
        duration = (round(max_frame_idx / fps) + 1) if fps else 0
    if max_frame_count is None:
        max_frame_count = envs.SGLANG_GLM_VIDEO_MAX_FRAMES.get()

    effective_duration = min(duration, envs.SGLANG_GLM_VIDEO_MAX_DURATION.get())
    if target_fps is None:
        if effective_duration <= 30:
            target_fps = envs.SGLANG_GLM_VIDEO_FPS_SHORT.get()
        elif effective_duration <= 300:
            target_fps = envs.SGLANG_GLM_VIDEO_FPS_MEDIUM.get()
        else:
            target_fps = envs.SGLANG_GLM_VIDEO_FPS_LONG.get()
    extract_t = int(effective_duration * target_fps * temporal_patch_size)
    extract_t = min(extract_t, max_frame_count)

    duration_per_frame = 1 / fps
    timestamps = [i * duration_per_frame for i in range(total_frames)]
    max_second = int(duration)

    if total_frames < extract_t:
        frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
    else:
        frame_indices = []
        current_second = 0
        inv_fps = 1 / (temporal_patch_size * target_fps)
        for frame_index in range(total_frames):
            if timestamps[frame_index] >= current_second:
                current_second += inv_fps
                frame_indices.append(frame_index)
                if current_second >= max_second:
                    break

    if len(frame_indices) < extract_t:
        if len(frame_indices) == 0:
            start, end = 0, max(total_frames - 1, 0)
        else:
            start, end = frame_indices[0], frame_indices[-1]
        frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(frame_indices) > extract_t:
        frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

    seen, uniq = set(), []
    for idx in frame_indices:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)

    if len(uniq) & 1:
        uniq.append(uniq[-1])

    return uniq


def _resize_frames_to_max_tokens(frames, max_tokens_per_frame):
    import torchvision.transforms.functional as TF

    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(np.asarray(frames))
    nchw = frames.permute(0, 3, 1, 2)
    _, _, h, w = nchw.shape
    patch = envs.SGLANG_GLM_VIDEO_PATCH_SIZE.get()
    merge = envs.SGLANG_GLM_VIDEO_MERGE_SIZE.get()
    pixels_per_token = (patch * merge) ** 2
    factor = patch * merge * envs.SGLANG_GLM_VIDEO_PATCH_EXPAND_FACTOR.get()
    max_pixels = max(int(max_tokens_per_frame) * pixels_per_token, factor * factor)
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    if (h_bar, w_bar) != (h, w):
        nchw = TF.resize(
            nchw,
            [h_bar, w_bar],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
    return nchw.permute(0, 2, 3, 1).contiguous()


async def preprocess_video(vr):
    return preprocess_video_sync(vr)


async def sample_video_sglang(vr, video_config: dict = None):
    return sample_video_sglang_sync(vr, video_config)


def sample_video_sglang_sync(vr, video_config: dict = None):
    video_config = video_config or {}
    video_fps = vr.avg_fps
    total_num_frames = len(vr)
    duration = total_num_frames / video_fps if video_fps else 0

    indices = glm_sample_frame_indices(
        total_num_frames,
        video_fps,
        duration,
        target_fps=video_config.get("fps"),
        max_frame_count=video_config.get("max_frames"),
    )
    frames = vr.get_frames_at(indices)

    max_tokens_per_frame = video_config.get("max_tokens_per_frame")
    if max_tokens_per_frame is not None:
        frames = _resize_frames_to_max_tokens(frames, max_tokens_per_frame)

    metadata = _video_metadata(total_num_frames, video_fps, duration, indices)
    return frames, metadata


def _decode_indices_parallel(source, device, indices, num_workers):
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    n = len(indices)
    if num_workers <= 1 or n <= 1:
        vr = VideoDecoderWrapper(source, device=device)
        try:
            return vr.get_frames_at(list(indices))
        finally:
            vr.close()

    num_workers = min(num_workers, n)
    buckets = [list(range(w, n, num_workers)) for w in range(num_workers)]
    out = [None] * n

    def _work(bucket_positions):
        sub_indices = [indices[p] for p in bucket_positions]
        vr = VideoDecoderWrapper(source, device=device)
        try:
            frames = vr.get_frames_at(sub_indices)
        finally:
            vr.close()
        return bucket_positions, frames

    initializer = None
    if str(device).startswith("cuda"):
        device_module = torch.get_device_module("cuda")
        current_device = device_module.current_device()

        def set_decode_device():
            device_module.set_device(current_device)

        initializer = set_decode_device

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers, initializer=initializer
    ) as ex:
        for bucket_positions, frames in ex.map(_work, buckets):
            for local_i, global_pos in enumerate(bucket_positions):
                out[global_pos] = frames[local_i]

    if isinstance(out[0], torch.Tensor):
        return torch.stack(out, dim=0)
    return np.stack(out, axis=0)


def glm_sample_and_decode_sync(vr, num_decode_workers=4, video_config=None):
    video_config = video_config or {}
    video_fps = vr.avg_fps
    total_num_frames = len(vr)
    duration = total_num_frames / video_fps if video_fps else 0

    indices = glm_sample_frame_indices(
        total_num_frames,
        video_fps,
        duration,
        target_fps=video_config.get("fps"),
        max_frame_count=video_config.get("max_frames"),
    )

    if num_decode_workers and num_decode_workers > 1:
        frames = _decode_indices_parallel(
            vr._source,
            device=vr._device,
            indices=indices,
            num_workers=num_decode_workers,
        )
    else:
        frames = vr.get_frames_at(list(indices))

    max_tokens_per_frame = video_config.get("max_tokens_per_frame")
    if max_tokens_per_frame is not None:
        frames = _resize_frames_to_max_tokens(frames, max_tokens_per_frame)

    metadata = _video_metadata(total_num_frames, video_fps, duration, indices)
    return frames, metadata


def glm_decode_frames_at(vr, indices, num_decode_workers=4, video_config=None):
    """Decode only explicitly sampled frame indices for one video shard."""
    indices = list(indices)
    if not indices:
        return None
    video_config = video_config or {}
    if num_decode_workers and num_decode_workers > 1 and len(indices) > 1:
        frames = _decode_indices_parallel(
            vr._source,
            device=vr._device,
            indices=indices,
            num_workers=num_decode_workers,
        )
    else:
        frames = vr.get_frames_at(indices)

    max_tokens_per_frame = video_config.get("max_tokens_per_frame")
    if max_tokens_per_frame is not None:
        frames = _resize_frames_to_max_tokens(frames, max_tokens_per_frame)
    return frames


def preprocess_video_sync(vr):
    video_fps = vr.avg_fps
    total_num_frames = len(vr)
    duration = total_num_frames / video_fps if video_fps else 0

    indices = list(range(total_num_frames))
    frames = vr.get_frames_at(indices)

    metadata = _video_metadata(total_num_frames, video_fps, duration, indices)
    return frames, metadata


async def preprocess_video_frames(frame_list: List[dict]):
    return preprocess_video_frames_sync(frame_list)


def preprocess_video_frames_sync(frame_list: List[dict]):
    total_num_frames = len(frame_list)
    duration = 0
    if frame_list[0].get("detail") is not None:
        details = json.loads(frame_list[0]["detail"])
        duration = details.get("video_duration", 0)
    if duration == 0:
        duration = float(frame_list[-1]["timestamp"])

    indices = list(range(total_num_frames))
    images = [np.array(frame["frame_image"]) for frame in frame_list]
    fps = total_num_frames / duration if duration else 0

    metadata = _video_metadata(total_num_frames, fps, duration, indices)
    return images, metadata
