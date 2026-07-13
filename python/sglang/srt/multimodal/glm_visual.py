# SPDX-License-Identifier: Apache-2.0
"""Shared GLM-V vision-encoder helpers.

The ViT feature-extraction path (optionally chunked to cap peak activation
memory) is identical across the GLM-4V and GLM5-next conditional-generation
models. This mixin factors it out so both reuse a single implementation.

Hosts must provide ``self.visual`` (a ``Glm4vVisionModel``) and
``self.use_data_parallel``.
"""

import logging
import math
from typing import List

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model

logger = logging.getLogger(__name__)


class GlmVisualEncoderMixin:
    """ViT feature extraction shared by GLM-4V and GLM5-next models."""

    def _run_visual_chunked(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Run the ViT, optionally splitting the input into chunks by patch /
        frame count to cap peak activation memory on long videos / many images.

        The ViT attention is block-diagonal per grid_thw row (frame/image), so
        splitting on row boundaries and concatenating the per-chunk outputs is
        bit-for-bit equivalent to a single forward. Controlled by the same env
        vars as the upstream Qwen3-VL impl (PR #14907):
          - SGLANG_VLM_MAX_PATCHES_PER_VIT: max patches (rows of pixel_values)
            per ViT call. 0 = unlimited.
          - SGLANG_VLM_MAX_IMAGES_PER_VIT: max grid_thw rows per ViT call.
            0 = unlimited.
        Both 0 (default) => single forward, identical to previous behavior.

        grid_thw: tensor [N, 3], each row [t, h, w]; splits land on row edges.
        """

        def _run(px: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual, px, g.tolist(), rope_type="rope_3d"
                )
            return self.visual(px, grid_thw=g)

        max_patches = envs.SGLANG_VLM_MAX_PATCHES_PER_VIT.get()
        max_images = envs.SGLANG_VLM_MAX_IMAGES_PER_VIT.get()

        if max_patches <= 0 and max_images <= 0:
            return _run(pixel_values, grid_thw)

        patches_per = [int(math.prod(g)) for g in grid_thw.tolist()]
        n = len(patches_per)
        cum = [0]
        for p in patches_per:
            cum.append(cum[-1] + p)
        assert pixel_values.size(0) == cum[-1], (pixel_values.size(0), cum[-1])

        outs = []
        s = 0
        while s < n:
            e, packed_patches, packed_images = s, 0, 0
            while e < n:
                nxt = patches_per[e]
                if max_patches > 0 and packed_patches + nxt > max_patches:
                    break
                if max_images > 0 and packed_images + 1 > max_images:
                    break
                packed_patches += nxt
                packed_images += 1
                e += 1
            # A single frame exceeding the patch limit still occupies its own
            # chunk (never split a frame); guarantees progress.
            if e == s:
                e = s + 1
            outs.append(_run(pixel_values[cum[s] : cum[e]], grid_thw[s:e]))
            s = e
        logger.debug(
            "[vit-chunk] max_patches=%d max_images=%d | %d rows / %d patches -> %d chunks",
            max_patches,
            max_images,
            n,
            cum[-1],
            len(outs),
        )
        return torch.cat(outs, dim=0)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in GLM-V, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        return self._run_visual_chunked(pixel_values, image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in GLM-V, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)

        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        temp_frames_hw = []
        for t, h, w in video_grid_thw:
            repeated_row = (
                torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)
            )
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = torch.cat(temp_frames_hw, dim=0)

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        return self._run_visual_chunked(pixel_values, flattened_video_grid_thw)
