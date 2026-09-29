"""
Multi-modality utils
"""

import copy
import hashlib
import mmap
import os
import pickle
from abc import abstractmethod
from collections import defaultdict
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.multimodal import gpu_tensor_hash
from sglang.srt.managers.schedule_batch import (
    CudaIpcTensorTransportProxy,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import flatten_nested_list, is_npu, print_warning_once
from sglang.utils import logger

_is_npu = is_npu()

# NOTE: Using the shared logger from sglang.utils instead of creating a module-specific logger
# to ensure consistent logging behavior across the codebase. This prevents issues with log
# propagation that can cause some log messages (like 'server is fired up') to not appear
# in the console when multimodal support is enabled.

# TODO(mick): nccl
# cuda_ipc: for intranode tensor sharing
TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


_GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None
_BUFFER_OFFSET = 0

_is_default_tensor_transport = None


def init_feature_buffer(device):
    global _GPU_FEATURE_BUFFER, _BUFFER_OFFSET
    if (
        device == "cpu"
        or envs.SGLANG_MM_BUFFER_SIZE_MB.get() == 0
        or _GPU_FEATURE_BUFFER is not None
    ):
        return
    try:
        size_mb = envs.SGLANG_MM_BUFFER_SIZE_MB.get()
        num_elements = int(size_mb * 1024 * 1024 / 4)
        _GPU_FEATURE_BUFFER = torch.empty(
            num_elements, dtype=torch.float32, device=device
        )
        logger.info(f"Preallocated {size_mb}MB GPU buffer")
    except RuntimeError as e:
        _GPU_FEATURE_BUFFER = None


def reset_buffer_offset():
    global _BUFFER_OFFSET
    _BUFFER_OFFSET = 0


def is_feature_buffer_initialized():
    global _GPU_FEATURE_BUFFER
    if _GPU_FEATURE_BUFFER is None:
        return False
    return True


def try_add_to_buffer(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    global _BUFFER_OFFSET

    if _GPU_FEATURE_BUFFER is None:
        return tensor

    tensor_size = tensor.numel()

    if _BUFFER_OFFSET + tensor_size <= _GPU_FEATURE_BUFFER.numel():
        buffer_view = _GPU_FEATURE_BUFFER[_BUFFER_OFFSET : _BUFFER_OFFSET + tensor_size]
        buffer_view.copy_(tensor.flatten(), non_blocking=True)
        result = buffer_view.view(tensor.shape)
        _BUFFER_OFFSET += tensor_size
        return result
    else:
        return tensor


class TransportProxyTensor(torch.Tensor):
    """
    A convenient torch.Tensor subclass that carries extra metadata and supports
    efficient inter-process communications
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        name: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        transport_mode: TensorTransportMode = "default",
        *args,
        **kwargs,
    ):

        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"
            )

        instance = data.as_subclass(cls)

        instance._metadata = {
            "name": name,
            "fields": fields if fields is not None else {},
            "transport_mode": transport_mode,
        }

        return instance

    def __getstate__(self):
        """
        Called during pickling. Implements the serialization logic.
        """
        # acquire all serialize metadata from _metadata
        state = {
            "metadata": self._metadata,
            "tensor_data": None,
            "ipc_extra": None,
        }
        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and self.is_cuda:
            try:
                storage = self.untyped_storage()
                handle = storage._share_cuda_()

                state["ipc_extra"] = {
                    "handle": handle,
                    "shape": self.shape,
                    "dtype": self.dtype,
                    "stride": self.stride(),
                    "device_index": self.device.index,
                    "storage_offset": self.storage_offset(),
                }
                state["tensor_data"] = None
            except Exception as e:
                # Failed to get CUDA IPC handle (possibly tp). Falling back to default transport.
                state["metadata"]["transport_mode"] = "default"
                state["tensor_data"] = self.as_subclass(torch.Tensor)
        else:
            state["metadata"]["transport_mode"] = "default"
            state["tensor_data"] = self.as_subclass(torch.Tensor)

        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Called during unpickling. Implements the deserialization logic.
        """
        self._metadata = state["metadata"]

        transport_mode = self._metadata.get("transport_mode", "default")

        if transport_mode == "cuda_ipc" and state["ipc_extra"] is not None:
            ipc_extra = state["ipc_extra"]
            handle, shape, dtype, stride, source_device_index, s_offset = (
                ipc_extra["handle"],
                ipc_extra["shape"],
                ipc_extra["dtype"],
                ipc_extra["stride"],
                ipc_extra["device_index"],
                ipc_extra["storage_offset"],
            )

            try:
                target_device = torch.device(f"cuda:{source_device_index}")
                with torch.cuda.device(target_device):
                    storage = torch.UntypedStorage._new_shared_cuda(*handle)
                    reconstructed_tensor = torch.empty(
                        0, dtype=dtype, device=target_device
                    ).set_(storage, storage_offset=s_offset, size=shape, stride=stride)
                    self.set_(reconstructed_tensor)
            except Exception as e:
                print(f"Error: Failed to deserialize from CUDA IPC handle ({e}).")
                raise e

        elif state["tensor_data"] is not None:
            self.set_(state["tensor_data"])
        else:
            raise pickle.UnpicklingError(
                "Invalid state for TransportProxyTensor: no tensor data found."
            )

    @property
    def name(self) -> Optional[str]:
        return self._metadata.get("name")

    @property
    def fields(self) -> Dict[str, Any]:
        return self._metadata.get("fields", {})

    @property
    def transport_mode(self) -> TensorTransportMode:
        return self._metadata.get("transport_mode", "default")


class MultiModalityDataPaddingPattern:
    """
    Data tokens (like image tokens) often need special handling during padding
    to maintain model compatibility. This class provides the interface for
    implementing different padding strategies for data tokens
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Pad the input ids sequence containing data tokens, and replace them with pad_values
        """
        pass


class MultiModalityDataPaddingPatternTokenPairs(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be enclosed by special token pairs (e.g. <image>...</image>, data_token_pairs)

    The padded value in a region enclosed by a token pair with be the same one, as the MultimodalDataItem's pad value

    This strategy should be applied when data content is marked by start/end token pairs in the input sequence.
    """

    def __init__(
        self,
        data_token_pairs: Optional[List[Tuple[int, int]]],
        data_start_token_ids: Optional[List[int]] = None,
    ) -> None:
        """

        Args:
            data_start_token_ids marks the start of a single multimodal data
            See Minicpmo's slice_start_id for example
        """
        self.data_token_id_pairs = data_token_pairs
        self.data_start_token_ids = data_start_token_ids or [
            s for s, _e in data_token_pairs
        ]

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        This function will replace the data-tokens in between with pad_values accordingly
        """
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        data_token_pairs = self.data_token_id_pairs
        mm_inputs.data_offsets = []
        if data_token_pairs is None:
            data_token_pairs = [mm_inputs.im_start_id, mm_inputs.im_end_id]
        if data_token_pairs is None:
            print_warning_once(
                "No data_token_pairs provided, RadixAttention might be influenced."
            )
            return input_ids
        start_token_ids = {s for s, _e in data_token_pairs}
        end_tokens_ids = {e for _s, e in data_token_pairs}

        padded_ids = []
        last_idx = 0
        data_idx = -1

        start_indices = [i for i, x in enumerate(input_ids) if x in start_token_ids]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_tokens_ids]

        if len(start_indices) != len(end_indices):
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            padded_ids.extend(input_ids[last_idx : start_idx + 1])

            if input_ids[start_idx] in self.data_start_token_ids:
                data_idx += 1
                mm_inputs.data_offsets += [start_idx]

            if data_idx >= len(pad_values):
                data_idx = len(pad_values) - 1

            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[data_idx]
            padded_ids.extend([pad_value] * num_tokens)

            last_idx = end_idx

        padded_ids.extend(input_ids[last_idx:])

        assert len(input_ids) == len(padded_ids), "Length validation fails"
        return padded_ids


class MultiModalityDataPaddingPatternMultimodalTokens(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be represented as repetitions of a single token
    e.g. <image><image>....<image>, or <audio><audio>...<audio>
    """

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Replaces multimodal tokens in input_ids with corresponding pad_values from mm_items.
        Each modality (image, audio, video) is handled separately based on its token_id.
        """
        if not input_ids or not mm_inputs.mm_items:
            return input_ids

        token_id_map = {
            Modality.IMAGE: mm_inputs.im_token_id,
            Modality.AUDIO: mm_inputs.audio_token_id,
            Modality.VIDEO: mm_inputs.video_token_id,
        }
        # NOTE: callers pass `req.origin_input_ids`, which is an `array("q")` in
        # this branch (upstream migrated it to a plain list). `array` has no
        # `.copy()` and rejects list slice assignment, so materialize a list
        # here. Callers already expect a list back (the previous implementation
        # returned `tensor.tolist()`).
        padded_input_ids = list(input_ids)

        # Updating list slices avoids two full list<->CPU-tensor conversions
        # for long prompts.
        for item in mm_inputs.mm_items:
            if token_id_map.get(item.modality) is None:
                continue
            for start, end in item.offsets:
                if start < 0 or end < start or end >= len(padded_input_ids):
                    raise ValueError(
                        f"Invalid multimodal offset ({start}, {end}) for "
                        f"input length {len(padded_input_ids)}"
                    )
                padded_input_ids[start : end + 1] = [item.pad_value] * (end - start + 1)

        return padded_input_ids


embedding_cache: Optional[MultiModalStaticCache] = None


def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)


def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, int, int]:
    """
    Extract a chunk of embeddings based on the specified prefix length, sequence length, and offset ranges.

    Args:
        embedding: The full embedding tensor to extract a chunk from
        extend_prefix_len: The starting position (prefix length) for extraction
        extend_seq_len: The number of tokens to extract
        items_offset: List of [start, end] offset ranges for multimodal items in the input sequence

    Returns:
        A tuple containing:
        - The extracted embedding chunk as a tensor
        - The start index used for extraction
        - The end index used for extraction

    Note:
        If there's no overlap between the requested range and the offset ranges,
        an empty tensor is returned with zeros for start and end indices.
    """
    start_index, end_index = 0, 0
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1
    # some models' embedding is 3-dim, reshape it to 2-dim
    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def _precomputed_chunk_by_item(
    items_per_req: List[MultimodalDataItem],
    extend_prefix_len: int,
    extend_seq_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Low-memory equivalent of slicing ``get_embedding_chunk`` out of
    ``torch.concat([item.precomputed_embeddings for item in items_per_req])``.

    The index walk in ``get_embedding_chunk`` accumulates independent per-span
    contributions, so applying the same arithmetic to each item's own spans
    (``item.offsets``) yields that item's row range inside the chunk window.
    Each item is therefore sliced in place -- ``reshape``/slice are views,
    zero-copy for SHM-backed tensors shared across TP ranks -- and only the
    window-sized slices are concatenated, never the full per-item concat (which
    the old path copied on every chunk, on every TP rank, then moved to GPU).

    When ``device`` is given, each window-sized slice (not the full item) is
    moved to it before the concat, so the concat and the scatter in
    ``embed_mm_inputs`` run on GPU and only window-sized bytes are transferred
    per chunk; the slice stays a CPU/SHM view until that H2D, so no CPU copy is
    added. With ``device=None`` the result stays on CPU (the caller's H2D).
    Byte-identical to the old concat-then-slice.

    Relies on ``item.offsets`` being in the same item order as the flattened
    ``items_offset`` the callers build via ``flatten_nested_list``.
    """
    window_start = extend_prefix_len
    window_end = extend_prefix_len + extend_seq_len - 1

    slices = []
    for item in items_per_req:
        start_index = 0
        end_index = 0
        for start, end in item.offsets:
            if window_start >= start and window_start <= end:
                start_index += window_start - start
            elif window_start > end:
                start_index += end - start + 1

            if window_end >= start and window_end <= end:
                end_index += window_end - start + 1
            elif window_end > end:
                end_index += end - start + 1

        if end_index > start_index:
            embedding = item.precomputed_embeddings
            # some models' embedding is 3-dim; reshape is a view on contiguous storage
            sl = embedding.reshape(-1, embedding.shape[-1])[start_index:end_index]
            slices.append(sl.to(device, non_blocking=True) if device is not None else sl)

    if not slices:
        embedding = items_per_req[0].precomputed_embeddings
        empty = embedding.reshape(-1, embedding.shape[-1])[:0]
        return empty.to(device) if device is not None else empty
    if len(slices) == 1:
        return slices[0]
    return torch.concat(slices)


def _get_precomputed_embedding(
    items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_embeddings, return their concatenation.
    If some but not all have precomputed_embeddings, raise NotImplementedError.
    If none have precomputed_embeddings, return None.

    Low-memory variant: items are sliced per-item over their own spans
    (zero-copy views -- keeps SHM-backed embeddings shared across TP ranks and
    off the GPU) and only the window-sized slices are moved to ``device`` and
    concatenated, instead of moving every full item to GPU and materializing the
    full concat per chunk.
    """
    precomputed_embeddings = []
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue

        items_per_req = items[items_size[i] : items_size[i + 1]]
        extend_len = extend_length[i] if i < len(extend_length) else 0

        if any(item.precomputed_embeddings is None for item in items_per_req):
            chunk = None
        else:
            chunk = _precomputed_chunk_by_item(
                items_per_req,
                extend_prefix_len=prefix_length[i],
                extend_seq_len=extend_len,
                device=device,
            )

        if chunk is None and len(items_per_req) > 1:
            return None
        precomputed_embeddings.append(chunk)

    if any(feature is not None for feature in precomputed_embeddings):
        if not all(feature is not None for feature in precomputed_embeddings):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )
        # Single request in the batch: keep the (possibly zero-copy) slice view
        # itself instead of forcing a concat copy.
        result = (
            precomputed_embeddings[0]
            if len(precomputed_embeddings) == 1
            else torch.concat(precomputed_embeddings)
        )
        # some models embedding is 3-dim, reshape it to 2-dim (similar to get_embedding_chunk)
        result = result.reshape(-1, result.shape[-1])
        return result
    return None


DataEmbeddingFunc = Callable[
    [List[MultimodalDataItem]], torch.Tensor | EVSEmbeddingResult
]


def _move_items_to_device(
    items: List[MultimodalDataItem], device: torch.device
) -> None:
    """Move item features to the target device (in-place, non-blocking)."""
    for item in items:
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item.feature = item.feature.to(device, non_blocking=True)


def _get_chunked_embedding_full(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Fallback: encode all items at once, cache combined result, extract chunk.
    Used for non-bundled items or EVS results.
    """
    item_hashes = [item.hash for item in embedding_items_per_req]
    embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = embedding_cache.get(item_hashes)

    if embedding_per_req is None:
        _move_items_to_device(embedding_items_per_req, device)
        embedding = data_embedding_func(embedding_items_per_req)
        embedding_per_req = (
            EmbeddingResult(embedding=embedding)
            if isinstance(embedding, torch.Tensor)
            else embedding
        )
        embedding_cache.set(embedding_items_hash, embedding_per_req)

    if isinstance(embedding_per_req, EVSEmbeddingResult):
        item = embedding_items_per_req[0]
        input_ids, items_offset = (
            embedding_per_req.redistribute_pruned_frames_placeholders(
                input_ids,
                items_offset,
                item=item,
                extend_prefix_len=extend_prefix_len,
                extend_seq_len=extend_seq_len,
            )
        )

    embedding_per_req_chunk, _, _ = get_embedding_chunk(
        embedding=embedding_per_req.embedding,
        extend_prefix_len=extend_prefix_len,
        extend_seq_len=extend_seq_len,
        items_offset=items_offset,
    )
    return embedding_per_req_chunk, input_ids


def _load_embedding_from_cache(
    embedding: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Move a cached embedding onto the target device for assembly.

    No-op when the cache already holds GPU tensors (default path); a single
    non-blocking H2D copy when the cache stored it on CPU.
    """
    if embedding.device != device:
        return embedding.to(device, non_blocking=True)
    return embedding


def _get_chunked_embedding_by_item(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Per-image chunk-aware encoding: only encode images overlapping with the
    current chunk, cache each image individually.
    Items must already be split per-image (each item has exactly one offset).
    """
    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len  # exclusive

    if extend_seq_len <= 0:
        return None

    # 1. Find items overlapping with current chunk
    # offsets are (start, end) inclusive on both ends
    overlapping = []
    for idx, (item, offset) in enumerate(zip(embedding_items_per_req, items_offset)):
        start, end = offset
        if end >= chunk_start and start < chunk_end:
            overlapping.append((idx, item, start, end))

    if not overlapping:
        return None

    # 2. Check per-image cache for each overlapping item
    cached_embeddings = {}  # idx -> tensor
    miss_items = []  # (idx, item, start, end)
    for idx, item, start, end in overlapping:
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embeddings[idx] = cached.embedding
        else:
            miss_items.append((idx, item, start, end))

    # 3. Batch encode all cache-miss items in one ViT call
    if miss_items:
        miss_item_list = [item for _, item, _, _ in miss_items]
        _move_items_to_device(miss_item_list, device)
        all_miss_embedding = data_embedding_func(miss_item_list)
        all_miss_embedding = all_miss_embedding.reshape(
            -1, all_miss_embedding.shape[-1]
        )

        # Split output by per-item token count
        token_counts = [end - start + 1 for _, _, start, end in miss_items]
        split_embeddings = torch.split(all_miss_embedding, token_counts, dim=0)

        for (idx, item, _, _), emb in zip(miss_items, split_embeddings):
            cached_embeddings[idx] = emb
            emb_result = EmbeddingResult(embedding=emb)
            embedding_cache.set(item.hash, emb_result)

    # 4. Assemble chunk: for each overlapping item, extract the overlap slice
    chunk_slices = []
    for idx, _, start, end in overlapping:
        emb = cached_embeddings[idx]  # shape: (end - start + 1, hidden)
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)  # inclusive
        local_start = overlap_start - start
        local_end = overlap_end - start + 1  # exclusive for slicing
        chunk_slices.append(emb[local_start:local_end])

    return torch.cat(chunk_slices, dim=0)


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Chunked prefill embedding: encode per-request items and extract the chunk.
    Items are already split per-image at processor stage.
    """
    embedding_list = []
    device = input_ids.device
    # FIXME(Xinyuan): temporary workaround for eagle3
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0

        # Skip if all items already prefilled
        if all(offset_end < prefix_length[i] for _, offset_end in items_offset):
            continue

        # Use per-image path when all items have exactly one offset (already
        # split per-image) — this avoids encoding images not in this chunk.
        # Fall back to combined path for non-split items or EVS.
        is_per_image = all(len(item.offsets) == 1 for item in embedding_items_per_req)

        if is_per_image:
            chunk_embedding = _get_chunked_embedding_by_item(
                data_embedding_func,
                embedding_items_per_req,
                items_offset,
                extend_prefix_len,
                extend_seq_len,
                device,
            )
            if chunk_embedding is not None:
                embedding_list.append(
                    _load_embedding_from_cache(chunk_embedding, device)
                )
        else:
            chunk_embedding, input_ids = _get_chunked_embedding_full(
                data_embedding_func,
                embedding_items_per_req,
                items_offset,
                extend_prefix_len,
                extend_seq_len,
                input_ids,
                device,
            )
            if chunk_embedding is not None:
                embedding_list.append(
                    _load_embedding_from_cache(chunk_embedding, device)
                )

    if len(embedding_list) == 0:
        return None, input_ids
    # Keep assembly on the model/input device. This also covers custom embedding
    # functions that return CPU tensors instead of following the item device.
    return torch.concat(embedding_list, dim=0), input_ids


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    logger,
) -> torch.Tensor:
    num_mm_tokens_in_embedding = embedding.shape[0]
    num_mm_tokens_in_input_ids = mask.sum().item()
    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text. "
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            f"tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            chunked_prefill_size = get_global_server_args().chunked_prefill_size
            if chunked_prefill_size != -1:
                logger.warning(
                    "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked prefill"
                )
            # extract from the end: this is a compromise
            if embedding.dim() == 2:
                embedding = embedding[-num_mm_tokens_in_input_ids:, :]
            else:
                num_multimodal = num_mm_tokens_in_input_ids // embedding.shape[0]
                embedding = embedding[-num_multimodal:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """
    Generate multimodal embeddings and create a mask for identifying their positions in the input sequence.

    Args:
        data_embedding_func: Function that generates embeddings for multimodal items
        embedding_items: List of multimodal items to embed
        placeholder_tensor: Tensor containing token IDs that serve as placeholders for multimodal content
        input_ids: The input token IDs tensor
        items_size: Cumulative sizes of multimodal items per request
        prefix_length: Prefix lengths for each request
        extend_length: Sequence lengths for each request
        items_offset_list: List of offset ranges for multimodal items in each request

    Returns:
        A tuple containing:
        - The generated embeddings tensor
        - A boolean mask tensor indicating where these embeddings should be placed
        - If EVS is used, the pruned input ids tensor; otherwise, the original input ids tensor
    """
    # 1. Get embedding
    embedding = _get_precomputed_embedding(
        embedding_items,
        items_size,
        prefix_length,
        extend_length,
        items_offset_list,
        device=input_ids.device,
    )
    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids
    # 2. Get mask
    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask, input_ids


def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
) -> Optional[torch.Tensor]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated
    """
    other_info = {}
    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    # deepstack_embeddings: per-modality
    modalities, embeddings, masks, deepstack_embeddings = [], [], [], []

    # 2. Get multimodal embedding separately
    # Try get mm embedding if any
    for modality in Modality.all():
        items = [
            item for item in item_flatten_list if item.is_modality(modality=modality)
        ]
        embedder = (
            None
            if data_embedding_func_mapping is None
            else data_embedding_func_mapping.get(modality, None)
        )
        if embedder is None:
            # "image", "video", etc
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)
        if len(items) != 0:
            assert embedder is not None, f"no embedding method found for {modality}"
            placeholder_tensor = torch.as_tensor(
                [item.pad_value for item in items],
                device=input_ids.device,
            )
            # calculate per request items length offset
            items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
            items_offsets = []
            for i, mm_inputs in enumerate(mm_inputs_list):
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size[i + 1] = len(mm_items)
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )
            items_size = torch.cumsum(items_size, dim=0).tolist()

            embedding, mask, input_ids = get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=placeholder_tensor,
                input_ids=input_ids,
                items_size=items_size,
                prefix_length=extend_prefix_lens,
                extend_length=extend_seq_lens,
                items_offset_list=items_offsets,
            )

            if use_deepstack.get(modality, None) and embedding is not None:
                embedding, deepstack_embedding = (
                    multimodal_model.separate_deepstack_embeds(embedding)
                )
                deepstack_embeddings += [deepstack_embedding]
            else:
                deepstack_embeddings += [None]
            modalities += [modality]
            embeddings += [embedding]
            masks += [mask]

    # 3. Get input embeddings
    vocab_size = input_embedding.num_embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_embeds = input_embedding(input_ids)

    # deepstack embedding
    if use_deepstack:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)

        deepstack_embedding_shape = input_embeds.shape[:-1] + (
            input_embeds.shape[-1] * num_deepstack_embeddings,
        )
        # a zero-filled embedding, with the same length of input_embeds, but different hidden_size
        input_deepstack_embeds = torch.zeros(
            deepstack_embedding_shape,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    # 4. scatter embeddings into input embedding
    for i, modality, embedding, mask in zip(
        range(len(embeddings)), modalities, embeddings, masks
    ):
        if embedding is None or mask is None:
            continue
        # in-place update
        indices = torch.where(mask.squeeze(dim=-1))[0]
        input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)
        if use_deepstack.get(modality, None):
            input_deepstack_embeds[indices] = deepstack_embeddings[i].to(
                input_embeds.device, input_embeds.dtype
            )

    return input_embeds, other_info


def _embed_mm_inputs_with_split(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
    use_deepstack: Dict[Modality, bool] = {},
):
    """Split batch into precomputed vs non-precomputed, embed each group, merge back."""
    precomputed_req_indices = []
    non_precomputed_req_indices = []
    for idx, mm_input in enumerate(mm_inputs_list):
        items = [item for item in mm_input.mm_items if item is not None]
        if items and all(
            getattr(item, "precomputed_embeddings", None) is not None for item in items
        ):
            precomputed_req_indices.append(idx)
        else:
            non_precomputed_req_indices.append(idx)

    embed_kwargs = dict(
        multimodal_model=multimodal_model,
        input_embedding=input_embedding,
        data_embedding_func_mapping=data_embedding_func_mapping,
        placeholder_tokens=placeholder_tokens,
        use_deepstack=use_deepstack,
    )

    if not precomputed_req_indices or not non_precomputed_req_indices:
        return embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            **embed_kwargs,
        )

    all_seq_lens = forward_batch.extend_seq_lens_cpu
    mm_batch_indices = [
        i for i, mm in enumerate(forward_batch.mm_inputs) if mm is not None
    ]
    token_starts = []
    cumulative = 0
    for sl in all_seq_lens:
        token_starts.append(cumulative)
        cumulative += sl

    vocab_size = input_embedding.num_embeddings
    input_embeds = input_embedding(input_ids.clamp(min=0, max=vocab_size - 1))
    other_info = {}

    input_deepstack_embeds = None
    if use_deepstack and multimodal_model is not None:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)
        input_deepstack_embeds = torch.zeros(
            input_ids.shape[0],
            input_embedding.embedding_dim * num_deepstack_embeddings,
            device=input_ids.device,
            dtype=input_embedding.weight.dtype,
        )
        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    for group_req_indices in [precomputed_req_indices, non_precomputed_req_indices]:
        sub_mm_inputs = [mm_inputs_list[i] for i in group_req_indices]
        sub_prefix_lens = [extend_prefix_lens[i] for i in group_req_indices]
        sub_seq_lens = [extend_seq_lens[i] for i in group_req_indices]
        group_batch_indices = [mm_batch_indices[i] for i in group_req_indices]
        sub_slices = [
            input_ids[token_starts[bi] : token_starts[bi] + all_seq_lens[bi]]
            for bi in group_batch_indices
        ]
        sub_input_ids = torch.cat(sub_slices)

        sub_embeds, sub_info = embed_mm_inputs(
            mm_inputs_list=sub_mm_inputs,
            extend_prefix_lens=sub_prefix_lens,
            extend_seq_lens=sub_seq_lens,
            input_ids=sub_input_ids,
            **embed_kwargs,
        )

        offset = 0
        for bi in group_batch_indices:
            req_len = all_seq_lens[bi]
            start = token_starts[bi]
            input_embeds[start : start + req_len] = sub_embeds[
                offset : offset + req_len
            ]
            if (
                input_deepstack_embeds is not None
                and "input_deepstack_embeds" in sub_info
            ):
                input_deepstack_embeds[start : start + req_len] = sub_info[
                    "input_deepstack_embeds"
                ][offset : offset + req_len]
            offset += req_len

    return input_embeds, other_info


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    data_embedding_funcs: Dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: Optional[dict[Modality, List[int]]] = None,
    use_deepstack: Dict[Modality, bool] = {},
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        data_embedding_funcs: A dictionary mapping from modality type to the corresponding embedding function.
        placeholder_tokens: Token IDs for multimodal placeholders
        use_deepstack: Whether to use deepstack embeddings for each modality, default False
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass
    """
    assert hasattr(language_model, "get_input_embeddings")
    embed_tokens = language_model.get_input_embeddings()
    if not hasattr(language_model, "pp_group") or language_model.pp_group.is_first_rank:
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            server_args = get_global_server_args()
            if server_args and server_args.enable_adaptive_dispatch_to_encoder:
                # Split by precomputed vs non-precomputed so get_embedding_and_mask only sees uniform batches
                input_embeds, other_info = _embed_mm_inputs_with_split(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )
            else:
                input_embeds, other_info = embed_mm_inputs(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    input_embedding=embed_tokens,
                    multimodal_model=multimodal_model,
                    data_embedding_func_mapping=data_embedding_funcs,
                    placeholder_tokens=placeholder_tokens,
                    use_deepstack=use_deepstack,
                )

            # add for qwen3_vl deepstack
            if use_deepstack:
                kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]
            # Offload GPU features to CPU instead of discarding them to balance memory
            # efficiency and data persistence.
            # In chunked-prefill, a request is processed across multiple batches, and
            # the original multimodal data must remain accessible until the entire
            # prefill phase is complete. Since the multimodal embedding cache is
            # best-effort, offloading to CPU ensures we have a reliable fallback
            # if a cache miss occurs in subsequent chunks, while still freeing up
            # critical GPU memory.
            if mm_inputs_list:
                for mm_input_obj in mm_inputs_list:
                    if mm_input_obj and hasattr(mm_input_obj, "mm_items"):
                        for mm_item in mm_input_obj.mm_items:
                            feature = getattr(mm_item, "feature", None)
                            if isinstance(feature, torch.Tensor) and feature.is_cuda:
                                mm_item.feature = feature.to("cpu", non_blocking=True)
                            if get_global_server_args().language_only:
                                precomputed_embeddings = getattr(
                                    mm_item, "precomputed_embeddings", None
                                )
                                if (
                                    isinstance(precomputed_embeddings, torch.Tensor)
                                    and precomputed_embeddings.is_cuda
                                ):
                                    mm_item.precomputed_embeddings = (
                                        precomputed_embeddings.to(
                                            "cpu", non_blocking=True
                                        )
                                    )
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = input_embeds
        else:
            input_embeds = embed_tokens(input_ids)
        # Copy to pre-allocated buffer if available (for CUDA graph address stability)
        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
    else:
        input_embeds = None

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        **kwargs,
    )
    return hidden_states


def get_multimodal_data_bounds(
    input_ids: torch.Tensor, pad_values: List[int], token_pairs: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Returns a tensor indicating the bounds of multimodal data (images, video, audio, etc.)

    Returns:
        [bounds_count, 2]
    """
    # All the multimodal data in the batch should share the same special bound token ids.
    start_tokens = {s for s, _e in token_pairs}
    end_tokens = {e for _s, e in token_pairs}

    assert all(isinstance(t, int) for t in start_tokens)
    assert all(isinstance(t, int) for t in end_tokens)

    start_cond = torch.isin(
        input_ids, torch.as_tensor(start_tokens, device=input_ids.device)
    )
    end_cond = torch.isin(
        input_ids, torch.as_tensor(end_tokens, device=input_ids.device)
    )

    (data_start_tokens,) = torch.where(start_cond)
    (data_end_tokens,) = torch.where(end_cond)

    data_start_tokens_cpu = data_start_tokens.cpu().tolist()
    data_end_tokens_cpu = data_end_tokens.cpu().tolist()

    # the im_start_id sometimes can be cached as prefix, but it is needed for the embedding of the multimodal data
    if len(data_start_tokens_cpu) != len(data_end_tokens_cpu):
        if (
            len(data_start_tokens_cpu) + 1 == len(data_end_tokens_cpu)
            and input_ids[0].item() in pad_values
            and data_end_tokens_cpu
            and data_start_tokens_cpu
            and data_end_tokens_cpu[0] < data_start_tokens_cpu[0]
        ):
            data_start_tokens_cpu.insert(0, 0)
    valid_mm_data_nums = min(len(data_start_tokens_cpu), len(data_end_tokens_cpu))

    if valid_mm_data_nums == 0:
        return torch.zeros((0, 2), device=input_ids.device)

    # Filter out pairs where start_token >= end_token
    valid_pairs = []
    for i in range(valid_mm_data_nums):
        start_token = data_start_tokens_cpu[i]
        end_token = data_end_tokens_cpu[i]
        if start_token < end_token:
            valid_pairs.append((start_token + 1, end_token - 1))

    if not valid_pairs:
        return torch.zeros((0, 2), device=input_ids.device)

    # Convert valid pairs to tensor
    valid_pairs_tensor = torch.as_tensor(valid_pairs, device=input_ids.device)
    return valid_pairs_tensor


def data_hash(data) -> int:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list) -> int:
    """
    hash a tensor or a tensor list
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensors = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        # GPU path: concat + triton hash (unchanged)
        if any(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
            tensor = torch.concat(tensors)
            return gpu_tensor_hash(tensor.cuda())
        # CPU path: hash each tensor incrementally without concat
        hasher = hashlib.sha256()
        for t in tensors:
            t = t.detach().contiguous()
            hasher.update(memoryview(t.view(torch.uint8).numpy()))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)

    # Single tensor
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()
    hasher = hashlib.sha256()
    hasher.update(memoryview(tensor.view(torch.uint8).numpy()))
    hash_bytes = hasher.digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def hash_feature(f):
    if isinstance(f, list):
        if isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        hasher = hashlib.sha256()
        hasher.update(memoryview(arr))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    elif isinstance(f, CudaIpcTensorTransportProxy):
        reconstruct_t = f.reconstruct_on_target_device(torch.cuda.current_device())
        return tensor_hash([reconstruct_t])
    return data_hash(f)


def extend_mrope_positions_for_retracted_request(
    mrope_positions: torch.Tensor, output_ids_len: int
) -> torch.Tensor:
    """
    Extend mrope_positions for retracted requests by appending positions for output_ids.

    When a request is retracted and has multimodal inputs with mrope_positions,
    we need to extend the positions to cover the output_ids that were already generated.
    For pure text tokens, all three dimensions use the same incremental sequence.

    Args:
        mrope_positions: The original mrope positions tensor, shape (3, origin_input_ids_len)
        output_ids_len: The number of output tokens to generate positions for

    Returns:
        Extended mrope_positions tensor with shape (3, origin_input_ids_len + output_ids_len)
    """
    if output_ids_len <= 0:
        return mrope_positions

    # Get the last position value corresponding to origin_input_ids
    # mrope_positions shape: (3, origin_input_ids_len)
    last_position = mrope_positions[:, -1]  # shape: (3,)

    # Generate pure text mrope positions for output_ids
    # All three dimensions for pure text are the same incremental sequence
    start_pos = last_position[0] + 1  # Start from last position + 1
    output_positions = (
        torch.arange(
            start_pos,
            start_pos + output_ids_len,
            dtype=torch.int64,
            device=mrope_positions.device,
        )
        .unsqueeze(0)
        .expand(3, -1)
    )  # shape: (3, output_ids_len)

    # Concatenate to the original mrope_positions
    return torch.cat([mrope_positions, output_positions], dim=1)


def _get_length(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, np.ndarray):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _slice_value(value, start, end):
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, np.ndarray):
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]
    try:
        return value[start:end]
    except Exception:
        return value


def _slice_model_data(
    data: dict,
    index: int,
    start: int,
    end: int,
    num_items: int,
    total_feature_len: Optional[int],
):
    sliced = {}
    for key, value in data.items():
        length = _get_length(value)
        if length == num_items:
            sliced[key] = _slice_value(value, index, index + 1)
        elif total_feature_len is not None and length == total_feature_len:
            sliced[key] = _slice_value(value, start, end)
        else:
            sliced[key] = value
    return sliced


def _try_simple_split(item, num_items, expanded_mm_items):
    """Try to split a bundled item by matching feature dim-0 to offset count.
    Returns True if split succeeded, False otherwise."""
    feature = item.feature if item.feature is not None else item.precomputed_embeddings
    if feature is None:
        return False

    if isinstance(feature, (torch.Tensor, np.ndarray)):
        feature_count = feature.shape[0]
    elif isinstance(feature, (list, tuple)):
        feature_count = len(feature)
    else:
        return False

    if feature_count != num_items:
        return False

    for i in range(num_items):
        new_item = copy.copy(item)
        if item.feature is not None:
            if isinstance(item.feature, (list, tuple)):
                new_item.feature = [item.feature[i]]
            else:
                new_item.feature = item.feature[i : i + 1]
        if item.precomputed_embeddings is not None:
            if isinstance(item.precomputed_embeddings, (list, tuple)):
                new_item.precomputed_embeddings = [item.precomputed_embeddings[i]]
            else:
                new_item.precomputed_embeddings = item.precomputed_embeddings[i : i + 1]
        new_item.offsets = [item.offsets[i]]
        new_data = {}
        for k, v in item.model_specific_data.items():
            if isinstance(v, (list, tuple)) and len(v) == num_items:
                new_data[k] = [v[i]]
            elif (
                isinstance(v, (torch.Tensor, np.ndarray))
                and len(v.shape) > 0
                and v.shape[0] == num_items
            ):
                new_data[k] = v[i : i + 1]
            else:
                new_data[k] = v
        new_item.model_specific_data = new_data
        new_item.hash = None
        expanded_mm_items.append(new_item)
    return True


def get_new_expanded_mm_items(original_mm_items):
    expanded_mm_items = []
    for item in original_mm_items:
        is_bundled = item.offsets is not None and len(item.offsets) > 1

        if is_bundled:
            num_items = len(item.offsets)

            if item.is_image():
                image_grid_thw = item.model_specific_data.get("image_grid_thw")
                grid_len = _get_length(image_grid_thw)
                if image_grid_thw is None or grid_len != num_items:
                    # No grid info — fall back to simple split by feature dim-0
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                patches_per_item = []
                for grid in image_grid_thw:
                    grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                    patches_per_item.append(int(torch.prod(grid_tensor).item()))

                cumulative = torch.cumsum(
                    torch.tensor(patches_per_item, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                for i in range(num_items):
                    start, end = slice_indices[i], slice_indices[i + 1]
                    new_item = copy.copy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    new_item.offsets = [item.offsets[i]]
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=i,
                        start=start,
                        end=end,
                        num_items=num_items,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)

            elif item.is_video():
                video_grid_thw = item.model_specific_data.get("video_grid_thw")
                if video_grid_thw is None:
                    if not _try_simple_split(item, num_items, expanded_mm_items):
                        expanded_mm_items.append(item)
                    continue

                # video_grid_thw shape: [num_videos, 3] where each row is [T, H, W]
                # When T > 1, item.offsets contains frames (num_items = total frames)
                # grid_len = num_videos, num_items = sum(T for each video) = total frames
                grid_len = _get_length(video_grid_thw)
                num_videos = grid_len

                # Calculate total frames and frames per video
                frames_per_video = []
                total_frames = 0
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        T = int(grid[0].item())  # T is the first element [T, H, W]
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        T = int(grid_tensor[0].item())
                    frames_per_video.append(T)
                    total_frames += T

                # num_items should equal total_frames when T > 1
                if num_items != total_frames:
                    expanded_mm_items.append(item)
                    continue

                # Calculate patches per video: T * H * W for each video
                patches_per_video = []
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        patches_per_video.append(int(torch.prod(grid).item()))
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        patches_per_video.append(int(torch.prod(grid_tensor).item()))

                # Calculate cumulative patches to get slice indices for each video
                cumulative = torch.cumsum(
                    torch.tensor(patches_per_video, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                # Group frames by video: calculate frame indices for each video
                frame_start_indices = [0]
                for i in range(num_videos):
                    frame_start_indices.append(
                        frame_start_indices[-1] + frames_per_video[i]
                    )

                # Expand each video into a separate item
                for video_idx in range(num_videos):
                    start, end = (
                        slice_indices[video_idx],
                        slice_indices[video_idx + 1],
                    )
                    frame_start, frame_end = (
                        frame_start_indices[video_idx],
                        frame_start_indices[video_idx + 1],
                    )

                    new_item = copy.copy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    # Group offsets for this video (all frames of this video)
                    new_item.offsets = item.offsets[frame_start:frame_end]
                    # For video_grid_thw, slice the corresponding row [T, H, W] for this video
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=video_idx,
                        start=start,
                        end=end,
                        num_items=num_videos,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)
            else:
                if not _try_simple_split(item, num_items, expanded_mm_items):
                    expanded_mm_items.append(item)

        else:
            expanded_mm_items.append(item)
    return expanded_mm_items


def _unregister_shm(shm_name: str) -> None:
    try:
        resource_tracker.unregister(f"/{shm_name}", "shared_memory")
    except Exception:
        pass


class _UntrackedPosixSharedMemory:
    """Open an existing POSIX SHM segment as an untracked read-only mapping."""

    def __init__(self, name: str):
        import _posixshmem

        self.name = name.lstrip("/")
        self._name = f"/{self.name}"
        fd = _posixshmem.shm_open(self._name, os.O_RDONLY, mode=0o600)
        try:
            self._mmap = mmap.mmap(
                fd,
                os.fstat(fd).st_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
            )
        finally:
            os.close(fd)
        self.buf = self._mmap

    def take_buffer(self):
        """Transfer mapping ownership to a consumer of the buffer protocol."""
        buf = self.buf
        self.buf = None
        self._mmap = None
        return buf

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        self.buf = None


def _open_shm_for_receiver(shm_name: str):
    if os.name == "posix":
        return _UntrackedPosixSharedMemory(shm_name)
    return shared_memory.SharedMemory(name=shm_name)


class ShmPointerMMData:
    """
    Wraps a tensor to be sent via a shared memory handle.
    This acts as a "pointer" to the tensor data across process boundaries.
    """

    def __init__(self, tensor: torch.Tensor):
        if not tensor.is_cpu:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        nbytes = tensor.numel() * tensor.element_size()
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        dst = None
        try:
            dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
            dst.copy_(tensor.view(torch.uint8).reshape(-1))
        except BaseException:
            dst = None
            try:
                shm.unlink()
            finally:
                shm.close()
            raise
        dst = None
        self.shm_name = shm.name
        self._shm_handle = shm
        self._shm_buffer = None
        self._shm_base_tensor = None
        self.tensor = None
        self._is_owner = True
        self._released = False

    def __getstate__(self):
        if self._released:
            raise RuntimeError("Cannot serialize a released shared-memory tensor")
        return {
            "shm_name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    def __setstate__(self, state):
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self._is_owner = False
        self._shm_handle = None
        self._shm_buffer = None
        self._shm_base_tensor = None
        self.tensor = None
        self._released = False

    @property
    def is_owner(self) -> bool:
        return self._is_owner

    def _ensure_open(self) -> None:
        """Open and map receiver SHM only when its tensor is first consumed."""
        if self._is_owner or self._released:
            raise RuntimeError("Shared-memory tensor is not available")
        if self.tensor is not None:
            return

        self._shm_handle = _open_shm_for_receiver(self.shm_name)
        try:
            if isinstance(self._shm_handle, _UntrackedPosixSharedMemory):
                self._shm_buffer = self._shm_handle.take_buffer()
                self._shm_handle = None
            else:
                self._shm_buffer = self._shm_handle.buf
            self._shm_base_tensor = torch.frombuffer(
                self._shm_buffer, dtype=self.dtype
            )
            self.tensor = self._shm_base_tensor.reshape(self.shape)
        except BaseException:
            self._release_receiver_references()
            raise

    def borrow(self) -> torch.Tensor:
        """Borrow a zero-copy tensor view backed by its storage-owned mapping."""
        if self._is_owner or self._released:
            raise RuntimeError("Shared-memory tensor is not available for borrowing")
        if os.name != "posix":
            raise RuntimeError("Zero-copy SHM borrowing is only supported on POSIX")
        self._ensure_open()
        tensor = self.tensor
        self._release_receiver_references()
        self._released = True
        return tensor

    def materialize(self) -> torch.Tensor:
        """Clone receiver data into owned memory and release its SHM mapping."""
        if self._is_owner or self._released:
            raise RuntimeError("Shared-memory tensor is not available to materialize")
        self._ensure_open()
        tensor = self.tensor.clone()
        self.release()
        return tensor

    def _release_receiver_references(self) -> None:
        shm_handle = self._shm_handle
        self._shm_handle = None
        self.tensor = None
        self._shm_base_tensor = None
        self._shm_buffer = None
        if shm_handle is not None:
            shm_handle.close()

    def release(self) -> None:
        """Release this process's owner or receiver handle exactly once."""
        if self._is_owner:
            if self._shm_handle is None:
                return
            shm_handle = self._shm_handle
            released = False
            try:
                shm_handle.unlink()
            except FileNotFoundError:
                _unregister_shm(self.shm_name)
                released = True
            else:
                released = True
            finally:
                shm_handle.close()
            if released:
                self._shm_handle = None
                self._released = True
        else:
            if self._released:
                return
            self._release_receiver_references()
            self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def _iter_shm_features(obj):
    if hasattr(obj, "batch"):
        for sub_obj in obj.batch:
            yield from _iter_shm_features(sub_obj)
        return
    if not hasattr(obj, "mm_inputs") or not obj.mm_inputs:
        return
    for item in obj.mm_inputs.mm_items:
        for attr in ("feature", "precomputed_embeddings"):
            value = getattr(item, attr, None)
            if isinstance(value, ShmPointerMMData):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (v for v in value if isinstance(v, ShmPointerMMData))


class ShmOwnerTable:
    """Keep tokenizer-owned SHM alive until every mapped request finishes."""

    def __init__(self):
        self._next_owner_handle = 0
        self._handle_to_names = {}
        self._owners = {}

    def register(self, obj) -> Optional[int]:
        owners = {
            pointer.shm_name: pointer
            for pointer in _iter_shm_features(obj)
            if pointer.is_owner
        }
        if not owners:
            return None
        for name, pointer in owners.items():
            current = self._owners.get(name)
            if current is not None and current[0] is not pointer:
                raise RuntimeError(f"Conflicting SHM owners for {name}")

        owner_handle = self._next_owner_handle
        self._next_owner_handle += 1
        self._handle_to_names[owner_handle] = tuple(owners)
        for name, pointer in owners.items():
            current = self._owners.get(name)
            self._owners[name] = (
                (pointer, 1) if current is None else (current[0], current[1] + 1)
            )
        return owner_handle

    def release(self, owner_handle: Optional[int]) -> None:
        if owner_handle is None:
            return
        names = self._handle_to_names.get(owner_handle)
        if names is None:
            return
        for name in names:
            owner, refcount = self._owners[name]
            if refcount == 1:
                owner.release()
        for name in names:
            owner, refcount = self._owners[name]
            if refcount > 1:
                self._owners[name] = (owner, refcount - 1)
            else:
                del self._owners[name]
        del self._handle_to_names[owner_handle]

    def close(self) -> None:
        for owner_handle in list(self._handle_to_names):
            self.release(owner_handle)

    def __len__(self) -> int:
        return len(self._handle_to_names)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _get_is_default_transport():
    global _is_default_tensor_transport
    if _is_default_tensor_transport is None:
        from sglang.srt.managers.tokenizer_manager import (
            _determine_tensor_transport_mode,
        )

        _is_default_tensor_transport = (
            _determine_tensor_transport_mode(get_global_server_args()) == "default"
        )
    return _is_default_tensor_transport


def wrap_shm_features(obj):
    """
    Scan the object for multimodal tensors and wrap them in SHM pointers.

    Encoder-disaggregated requests store their CPU tensor in
    ``precomputed_embeddings`` instead of ``feature``. Both fields must use
    the lightweight SHM transport; otherwise the full embedding is serialized
    through tokenizer ZMQ and again through the scheduler TP broadcast.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj

    if hasattr(obj, "batch"):
        wrapped_obj = copy.copy(obj)
        wrapped_obj.batch = [wrap_shm_features(sub_obj) for sub_obj in obj.batch]
        return wrapped_obj

    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        wrapped_obj = copy.copy(obj)
        wrapped_mm_inputs = copy.copy(obj.mm_inputs)
        wrapped_items = []
        for item in obj.mm_inputs.mm_items:
            wrapped_item = copy.copy(item)
            for attr in ("feature", "precomputed_embeddings"):
                value = getattr(item, attr, None)
                if isinstance(value, torch.Tensor) and value.is_cpu:
                    setattr(wrapped_item, attr, ShmPointerMMData(value))
                elif isinstance(value, (list, tuple)):
                    wrapped = [
                        (
                            ShmPointerMMData(t)
                            if isinstance(t, torch.Tensor) and t.is_cpu
                            else t
                        )
                        for t in value
                    ]
                    setattr(
                        wrapped_item,
                        attr,
                        type(value)(wrapped) if isinstance(value, tuple) else wrapped,
                    )
            wrapped_items.append(wrapped_item)
        wrapped_mm_inputs.mm_items = wrapped_items
        wrapped_obj.mm_inputs = wrapped_mm_inputs
        return wrapped_obj
    return obj


def unwrap_shm_features(obj):
    """
    Restore ShmPointerMMData wrappers back into standard torch.Tensors.
    Handles both single requests and batch requests.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj
    unwrap = (
        ShmPointerMMData.borrow
        if os.name == "posix" and envs.SGLANG_MM_SHM_ZERO_COPY.get()
        else ShmPointerMMData.materialize
    )
    # Handle batch requests
    if hasattr(obj, "batch"):
        for sub_obj in obj.batch:
            unwrap_shm_features(sub_obj)
        return obj
    # Handle single requests
    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        mm_items = obj.mm_inputs.mm_items
        for item in mm_items:
            for attr in ("feature", "precomputed_embeddings"):
                value = getattr(item, attr, None)
                if isinstance(value, ShmPointerMMData):
                    setattr(item, attr, unwrap(value))
                elif isinstance(value, (list, tuple)):
                    unwrapped = [
                        unwrap(t) if isinstance(t, ShmPointerMMData) else t
                        for t in value
                    ]
                    setattr(
                        item,
                        attr,
                        type(value)(unwrapped) if isinstance(value, tuple) else unwrapped,
                    )
    return obj
