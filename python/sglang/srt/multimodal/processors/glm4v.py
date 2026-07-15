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

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

try:
    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None

def preprocess_video_frames_sync(frame_list: List[dict]):
    """Synchronous core of :func:`preprocess_video_frames`."""
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

async def preprocess_video_frames(frame_list: List[dict]):
    """Async wrapper around :func:`preprocess_video_frames_sync`.

    Kept async so callers can `await` it uniformly alongside
    :func:`sample_video_sglang`.
    """
    return preprocess_video_frames_sync(frame_list)


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

        # GLM-V specific tokens
        self.IMAGE_TOKEN = "<|image|>"
        self.VIDEO_TOKEN = "<|video|>"
        self.IMAGE_START_TOKEN = "<|begin_of_image|>"
        self.IMAGE_END_TOKEN = "<|end_of_image|>"
        self.VIDEO_START_TOKEN = "<|begin_of_video|>"
        self.VIDEO_END_TOKEN = "<|end_of_video|>"

        # Token IDs
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id
        # Aliases required by the base `get_mm_data` (EPD/encoder language side).
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

    def assign_mm_offsets(self, all_collected_items, input_ids, mm_tokens):
        """Assign GLM4V image/video offsets structurally from the token stream.

        GLM4V images and video frames both collapse to ``IM_TOKEN_ID`` after
        tokenization, so the base token-id offset logic cannot tell them apart.
        Given a correct ``input_ids`` (image bodies inside
        ``<|begin_of_image|>..<|end_of_image|>`` and video frames inside
        ``<|begin_of_video|>..<|end_of_video|>``), we split the maximal
        ``IM_TOKEN_ID`` runs by whether they fall inside a video span: runs
        outside a span are image bodies (one ``(start, end)`` per image), runs
        inside a span are video frames (one per frame). This is exactly what
        ``get_new_expanded_mm_items`` expects and needs no grids. Non-visual
        modalities (e.g. audio) defer to the base token-id offsets.
        """
        ids = (
            input_ids.tolist()
            if isinstance(input_ids, torch.Tensor)
            else list(input_ids)
        )
        im_token_id = self.IM_TOKEN_ID
        video_start_id = self.VIDEO_START_TOKEN_ID
        video_end_id = self.VIDEO_END_TOKEN_ID

        # Split maximal IM-token runs, tagging each by video-span membership.
        image_runs, video_runs = [], []
        in_video = False
        run_start = None
        for i, t in enumerate(ids):
            if t == im_token_id:
                if run_start is None:
                    run_start = i
                run_end = i
                continue
            if run_start is not None:
                (video_runs if in_video else image_runs).append((run_start, run_end))
                run_start = None
            if t == video_start_id:
                in_video = True
            elif t == video_end_id:
                in_video = False
        if run_start is not None:
            (video_runs if in_video else image_runs).append((run_start, run_end))

        for mm_item in all_collected_items:
            if mm_item.is_image():
                mm_item.offsets = image_runs
            elif mm_item.is_video():
                mm_item.offsets = video_runs
            else:
                # Non-visual modality (e.g. audio): fall back to token-id offsets.
                mm_token_id = mm_tokens.get_token_id_by_modality(mm_item.modality)
                if mm_token_id is None:
                    raise ValueError(
                        f"No token id found for modality: {mm_item.modality}"
                    )
                mm_item.offsets = self.get_mm_items_offset(
                    input_ids=input_ids,
                    mm_token_id=mm_token_id,
                )

    def build_input_ids_with_timestamps(
        self,
        prompt: Union[str, List[int]],
        embeddings: torch.Tensor,
        img_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        video_timestamps: List[list],
    ):
        """Reconstruct GLM4V input_ids on the EPD language side.

        GLM4V images and videos both collapse to ``IM_TOKEN_ID`` after
        tokenization, but the *original* prompt still distinguishes them via
        ``IM_TOKEN_ID`` vs ``VIDEO_TOKEN_ID``. Images expand to a flat run of
        image tokens; videos expand frame by frame, wrapping each frame in
        ``image_start``/``image_end`` and appending an integer-seconds timestamp
        text segment. This mirrors the HF ``Glm4vProcessor`` expansion
        (``<|begin_of_image|>`` + image tokens + ``<|end_of_image|>`` +
        ``int(timestamp_sec)``) so token counts and precomputed embeddings stay
        aligned with the encoder output.

        ``offsets`` are inclusive (start, end) positions of the image-token runs
        within the returned ``input_ids``; for video there is one offset per
        frame.
        """
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
            # Copy the prefix up to and including the vision-start placeholder.
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
                    # HF Glm4vProcessor.replace_frame_token_id appends int seconds.
                    timestamp_sec = curr_timestamps[frame_idx]
                    timestamp_tokens = self._tokenizer.encode(
                        f"{float(timestamp_sec):.1f} seconds", add_special_tokens=False
                    )
                    input_ids.extend(timestamp_tokens)
                video_idx += 1
            else:
                logger.warning(f"{modality} modality is not supported for GLM4V EPD.")
                continue
            cur_idx = mm_start_idx + 2  # jump past the vision-end placeholder
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    @staticmethod
    def _group_offsets_by_modality(offsets, modality_list, video_grid_thw):
        """Split the flat ``build_input_ids_with_timestamps`` offsets by modality.

        ``build_input_ids_with_timestamps`` returns one offset per image and one
        offset per video *frame*, while ``modality_list`` has a single entry per
        image and per video. This regroups the flat offsets into:
          - ``image_runs``: one ``(start, end)`` per image
          - ``video_frame_runs``: one ``(start, end)`` per frame, all videos
            concatenated
        matching the bundled-item contract ``get_new_expanded_mm_items`` expects
        (image offsets len == #images; video offsets len == total frames).
        """
        image_runs = []
        video_frame_runs = []
        off_idx = 0
        video_idx = 0
        for modality in modality_list:
            if modality == Modality.IMAGE:
                image_runs.append(offsets[off_idx])
                off_idx += 1
            elif modality == Modality.VIDEO:
                num_frames = int(video_grid_thw[video_idx][0])
                video_frame_runs.extend(offsets[off_idx : off_idx + num_frames])
                off_idx += num_frames
                video_idx += 1
        return image_runs, video_frame_runs

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """EPD language side: rebuild mm_inputs from precomputed embeddings.

        ``embeddings`` is ``{Modality: tensor}`` (assembled by the encoder
        receiver). GLM4V currently sends a single modality per request, so the
        reconstructed embeddings are sliced per multimodal item and attached as
        ``precomputed_embeddings``.
        """
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        video_timestamps = kwargs.get("video_timestamps", None)

        input_ids, offsets, modality_list = self.build_input_ids_with_timestamps(
            prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
        )
        assert all(isinstance(modality, Modality) for modality in modality_list)
        # GLM4V EPD currently sends a single modality per request. A video maps
        # to many per-frame offsets but a single modality entry, so we attach all
        # offsets and the full modality embedding to one item (mm_utils slices it
        # back out via the offsets in get_embedding_chunk).
        assert len(set(modality_list)) == 1, (
            f"GLM4V EPD only supports a single modality per request, "
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
                # Decode instances receive metadata only (empty embeddings dict):
                # they run the LM in PREBUILT mode and never consume the embedding
                # values, so leave precomputed_embeddings as None. Prefill instances
                # receive the real per-modality tensor.
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
        # Per-video sampling overrides may arrive as dict items in video_data,
        # e.g. {"url": ..., "fps": 1, "max_frames": 64}. Split the
        # decodable URL/bytes from the sampling config so load_mm_data only sees
        # the former; configs are realigned by index below.
        video_urls, video_configs = _split_video_items(request_obj.video_data)

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_urls,
            multimodal_tokens=self.mm_tokens,
        )

        video_metadata = None
        if base_output.videos:
            # GLM NOTE: original code
            # base_output.videos = request_obj.video_data
            framed = False
            for video in base_output.videos:
                if isinstance(video, list):
                    framed = True
            if framed:
                videos_processed = [
                    await preprocess_video_frames(video)
                        for video in base_output.videos
                ]
            else:
                videos_processed = [
                    await sample_video(video, video_config=cfg)
                        for video, cfg in zip(base_output.videos, video_configs)
                ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        # With the remote-code Glm4vProcessor emitting a correct token stream
        # (image bodies inside <|begin_of_image|>..<|end_of_image|>, video frames
        # inside <|begin_of_video|>..<|end_of_video|>), the base pipeline is enough:
        # our assign_mm_offsets override splits the shared IM_TOKEN runs into image
        # vs video by video-span membership, so no id rebuild is needed here.
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


# GLM video frame-sampling defaults, mirroring the HF GLM-4.6V video processor
# (``glm-v-utils/sglang_config/4-6v/video_processing_glm46v.py:sample_frames``).
# Values are read from ``SGLANG_GLM_VIDEO_*`` env vars at call time (see
# environ.py) so they can be tuned without code changes; with no per-request
# overrides the indices produced here are identical to the HF processor's output.


def _video_metadata(total_num_frames, fps, duration, frames_indices):
    """Build the video metadata dict shared by all GLM video preprocess paths."""
    return {
        "total_num_frames": int(total_num_frames),
        "fps": float(fps),
        "duration": float(duration),
        "video_backend": "decord",
        "frames_indices": list(frames_indices),
    }


def _split_video_items(video_data):
    """Split video_data items into (decodable_urls, per_video_configs).

    Each item is either a URL/bytes (no overrides) or a dict carrying a ``url``
    plus optional sampling overrides (``fps`` / ``max_frames`` /
    ``max_tokens_per_frame``). Dict items that are processor-output/precomputed
    embeddings (have a ``format`` key) are passed through untouched. Returns two
    index-aligned lists.
    """
    _SAMPLING_KEYS = ("fps", "max_frames", "max_tokens_per_frame")
    if video_data is None:
        return None, []
    items = video_data if isinstance(video_data, list) else [video_data]

    urls, configs = [], []
    for item in items:
        if isinstance(item, dict) and "format" not in item and "url" in item:
            urls.append(item["url"])
            configs.append({k: item[k] for k in _SAMPLING_KEYS if item.get(k) is not None})
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
    """Replicate ``Glm46VVideoProcessor.sample_frames`` inside sglang.

    Per-request overrides (``None`` -> GLM default, output then matches HF):
    - ``target_fps`` (request key ``fps``): bypass the dynamic-fps table.
    - ``max_frame_count`` (request key ``max_frames``): override
      ``MAX_FRAME_COUNT_DYNAMIC`` (640).
    """
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
    """Downscale each frame so its post-merge token count <= max_tokens_per_frame.

    Resizes to multiples of ``patch_size*merge_size*patch_expand_factor`` (=112)
    so the HF processor's own ``smart_resize`` is a no-op (idempotent) and only
    this per-video budget takes effect. ``frames`` is NHWC (numpy or torch);
    returns NHWC torch uint8.
    """
    import torchvision.transforms.functional as TF

    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(np.asarray(frames))
    # NHWC -> NCHW for torchvision resize
    nchw = frames.permute(0, 3, 1, 2)
    _, _, h, w = nchw.shape
    patch = envs.SGLANG_GLM_VIDEO_PATCH_SIZE.get()
    merge = envs.SGLANG_GLM_VIDEO_MERGE_SIZE.get()
    # tokens-per-frame -> pixels: each merged token covers (patch_size*merge_size)^2.
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
    return nchw.permute(0, 2, 3, 1).contiguous()  # back to NHWC


# GLM NOTE: restore preprocess_video which was removed since v0.5.6.
# adapted from https://github.com/huggingface/transformers/blob/369c99d0cea403b77bd0aef818527106453fd9fc/src/transformers/video_utils.py#L312
async def preprocess_video(vr):
    """Decode all frames from a decord VideoReader and return frames + metadata.

    Used by the EPD encoder server to feed GLM4V's HF video processor (which then
    samples frames internally). Returns ``(frames, metadata)`` where metadata
    carries ``fps``/``frames_indices`` so timestamps can be reproduced.

    Thin async wrapper; the heavy synchronous decode lives in
    :func:`preprocess_video_sync` so callers can offload it to a thread pool
    (the decode blocks the event loop otherwise — see encode_server).
    """
    return preprocess_video_sync(vr)


async def sample_video(vr, video_config: dict = None):
    """Sample frames inside sglang (non-EPD path) and return frames + metadata.

    Unlike :func:`preprocess_video` (decode-all, HF samples), this selects frame
    indices here so the caller can pass ``do_sample_frames=False`` downstream.
    With an empty ``video_config`` the indices match the HF GLM video processor
    exactly (baseline preserved). ``video_config`` may carry per-video overrides
    ``fps`` / ``max_frames`` / ``max_tokens_per_frame``.
    """
    return sample_video_sync(vr, video_config)


def sample_video_sync(vr, video_config: dict = None):
    """Synchronous core of :func:`sample_video` (blocking frame decode)."""
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
    frames = vr.get_frames_at(indices)  # NHWC uint8 (numpy on CPU)

    max_tokens_per_frame = video_config.get("max_tokens_per_frame")
    if max_tokens_per_frame is not None:
        frames = _resize_frames_to_max_tokens(frames, max_tokens_per_frame)

    metadata = _video_metadata(total_num_frames, video_fps, duration, indices)
    return frames, metadata



def _decode_indices_parallel(source, device, indices, num_workers):
    """Decode ``indices`` frames from ``source`` using ``num_workers`` independent
    VideoDecoderWrapper instances (K-way segmented parallel decode).
    """
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    n = len(indices)
    if num_workers <= 1 or n <= 1:
        vr = VideoDecoderWrapper(source, device=device)
        try:
            return vr.get_frames_at(list(indices))
        finally:
            vr.close()

    num_workers = min(num_workers, n)
    # Round-robin (interleaved) split: worker w handles positions w, w+K, w+2K...
    buckets = [list(range(w, n, num_workers)) for w in range(num_workers)]

    out = [None] * n  # per-frame slot, filled in global order

    def _work(bucket_positions):
        sub_indices = [indices[p] for p in bucket_positions]
        vr = VideoDecoderWrapper(source, device=device)
        try:
            frames = vr.get_frames_at(sub_indices)
        finally:
            vr.close()
        return bucket_positions, frames

    
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    futures = [ex.submit(_work, b) for b in buckets]
    timeout_s = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()
    try:
        done, not_done = concurrent.futures.wait(futures, timeout=timeout_s)
        if not_done:
            logger.warning(
                f"[video-decode] stuck: {len(not_done)}/{len(futures)} worker(s) "
                f"exceeded {timeout_s:.0f}s; dropping this request's "
                f"decoded sub-batches and failing it"
            )
            raise TimeoutError(
                f"video decode stuck: {len(not_done)}/{len(futures)} worker(s) "
                f"exceeded {timeout_s:.0f}s"
            )
        for f in futures:
            bucket_positions, frames = f.result()
            for local_i, global_pos in enumerate(bucket_positions):
                out[global_pos] = frames[local_i]
    finally:
        # NEVER wait: a `with` block / shutdown(wait=True) would hang on the
        # stuck worker. Cancel queued tasks; leave any running one detached.
        ex.shutdown(wait=False, cancel_futures=True)

    # Preserve device: GPU frames are CUDA tensors (keep on-device for a GPU HF
    # processor), CPU frames are numpy arrays.
    if isinstance(out[0], torch.Tensor):
        return torch.stack(out, dim=0)
    return np.stack(out, axis=0)


def glm_sample_and_decode_sync(vr, num_decode_workers=4, video_config=None):
    """EPD GLM fast path: sample frame indices on the decode side (matching the HF
    GLM video processor exactly), then decode ONLY those frames with K-way
    segmented parallelism.

    Returns ``(frames, metadata)`` where ``metadata["frames_indices"]`` is the
    SAMPLED indices and ``metadata["fps"]`` is preserved, so the downstream
    ``VideoMetadata.timestamps`` property ([idx / fps for idx in frames_indices])
    is byte-identical to the old decode-all + do_sample_frames=True path.

    The caller MUST pass ``do_sample_frames=False`` so the HF processor does not
    re-sample (it would otherwise overwrite frames_indices).

    ``video_config`` carries optional per-video overrides
    (``fps`` / ``max_frames`` / ``max_tokens_per_frame``) extracted from an inline
    ``video_url`` dict by :func:`_split_video_items`. With an empty config the
    sampled indices match the HF Glm46VVideoProcessor.sample_frames output exactly.
    """
    video_config = video_config or {}
    video_fps = vr.avg_fps
    total_num_frames = len(vr)
    duration = total_num_frames / video_fps if video_fps else 0

    # Per-request overrides (``None`` -> GLM defaults, identical to the HF
    # Glm46VVideoProcessor.sample_frames output).
    indices = glm_sample_frame_indices(
        total_num_frames,
        video_fps,
        duration,
        target_fps=video_config.get("fps"),
        max_frame_count=video_config.get("max_frames"),
    )

    # Decode ONLY the sampled frames on the SAME device as `vr` (created by
    # _load_single_item with use_gpu=use_image_processor_gpu). Keeping the decode
    # on the original device is critical: the HF GLM video processor's
    # resize/normalize/patch are torch ops that run on the frames' device. With
    # GPU frames the processor takes ~1ms; forcing frames to CPU makes the same
    # processor ~950ms. So we must NOT move decode to CPU here.
    #
    # num_decode_workers>1 enables K-way segmented parallel decode via independent
    # decoders (NVDEC has spare capacity: single-decoder GPU util is ~3%). Each
    # worker owns its decoder so the beta->ffmpeg NV12 rebuild stays thread-local.
    # Measured (80 frames 720p): K=1 362ms -> K=8 79ms (4.6x). K=1 falls back to
    # decoding on the original `vr`.
    if num_decode_workers and num_decode_workers > 1:
        frames = _decode_indices_parallel(
            vr._source, device=vr._device, indices=indices,
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
    """Decode ONLY the explicitly-given ``indices`` (already sampled).

    Used by the DP-sharded decode path: the caller (encoder) samples the full
    frame set once, assigns temporal units to TP ranks, and calls this per rank
    with just that rank's frame indices. Returns frames only (metadata is built
    globally by the caller from the full sampled indices).

    ``indices`` must already be the concrete per-rank frame indices. Empty input
    returns ``None`` (rank owns no temporal units).
    """
    indices = list(indices)
    if len(indices) == 0:
        return None
    video_config = video_config or {}
    if num_decode_workers and num_decode_workers > 1 and len(indices) > 1:
        frames = _decode_indices_parallel(
            vr._source, device=vr._device, indices=indices,
            num_workers=num_decode_workers,
        )
    else:
        frames = vr.get_frames_at(indices)

    max_tokens_per_frame = video_config.get("max_tokens_per_frame")
    if max_tokens_per_frame is not None:
        frames = _resize_frames_to_max_tokens(frames, max_tokens_per_frame)
    return frames


def preprocess_video_sync(vr):
    """Synchronous core of :func:`preprocess_video` (blocking frame decode)."""
    video_fps = vr.avg_fps
    total_num_frames = len(vr)
    duration = total_num_frames / video_fps if video_fps else 0

    indices = list(range(total_num_frames))
    frames = vr.get_frames_at(indices)

    metadata = _video_metadata(total_num_frames, video_fps, duration, indices)
    return frames, metadata

