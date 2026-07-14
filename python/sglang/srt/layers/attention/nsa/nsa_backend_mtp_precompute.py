"""Multi-step precompute utilities for Native Sparse Attention backend.

This module provides optimization utilities for multi-step speculative decoding
by precomputing shared metadata once and copying it to multiple backend instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.utils import compute_nsa_seqlens
from sglang.srt.layers.attention.utils import seqlens_expand_triton
from sglang.srt.speculative.spec_info import (
    get_spec_v2_full_overlap_max_kv_len,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


@dataclass
class PrecomputedMetadata:
    """Precomputed metadata shared across multiple backend instances.

    Used for multi-step speculative decoding where multiple backends
    need identical metadata. Precomputing once and copying N times
    is much faster than computing N times.

    """

    # Basic seqlens
    cache_seqlens: torch.Tensor  # int32, [bs]
    cu_seqlens_k: torch.Tensor  # int32, [bs+1]

    # Page table
    page_indices: torch.Tensor  # int32, [bs, max_len] or [expanded_bs, max_len]
    real_page_table: Optional[torch.Tensor]  # int32, transformed version

    # NSA seqlens
    seqlens_expanded: torch.Tensor  # int32, [expanded_size]
    nsa_cache_seqlens: torch.Tensor  # int32, [expanded_size]
    nsa_cu_seqlens_k: torch.Tensor  # int32, [expanded_size+1]
    seqlens_expanded_size: int

    # Dimensions
    max_len: int  # for decode/draft_extend
    max_seqlen_k: int  # for target_verify

    # FlashMLA (optional)
    flashmla_metadata: Optional[torch.Tensor] = None

    # Needed by the per-backend kpool write-plan rebuild on the fast
    # (precomputed) replay path. The plan is per-backend buffer state, so
    # we can't share it -- but we share the inputs. ``None`` is fine when
    # kpool layout is disabled (the rebuild is a no-op then).
    req_pool_indices: Optional[torch.Tensor] = None  # int64, [bs]


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths with padding."""
    assert seqlens.dtype == torch.int32
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


class NativeSparseAttnBackendMTPPrecomputeMixin:
    """Mixin class providing metadata precomputation for multi-step speculative decoding.

    This mixin provides the _precompute_replay_metadata method and its helpers,
    which are used to optimize CUDA graph replay in multi-step scenarios.
    """

    def _precompute_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
    ) -> PrecomputedMetadata:
        """Precompute all shared metadata for multi-step backends.

        seq_lens_cpu may be None in SpecV2 full-overlap replay. In that case
        spec_info.max_kv_len, or the captured page-table width as a fallback,
        provides a safe page-table upper bound.

        This function extracts and computes all operations that are
        identical across different backend instances in multi-step
        speculative decoding.

        Args:
            bs: Batch size
            req_pool_indices: Request pool indices [bs]
            seq_lens: Sequence lengths [bs]
            seq_lens_cpu: Sequence lengths on CPU [bs]
            forward_mode: Forward mode (decode/target_verify/draft_extend)
            spec_info: Speculative decoding info (for draft_extend mode)

        Returns:
            PrecomputedMetadata containing all shared intermediate results
        """
        # Slice inputs to batch size
        seq_lens = seq_lens[:bs]
        max_kv_len = get_spec_v2_full_overlap_max_kv_len(spec_info)
        if max_kv_len is not None:
            pt_width = self.decode_cuda_graph_metadata[bs].page_table_1.shape[1]
            max_kv_len = min(max_kv_len, pt_width)
        else:
            assert seq_lens_cpu is not None
            seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Dispatch to mode-specific precomputation
        if forward_mode.is_decode_or_idle():
            return self._precompute_decode_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu, max_kv_len
            )
        elif forward_mode.is_target_verify():
            return self._precompute_target_verify_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu, max_kv_len
            )
        elif forward_mode.is_draft_extend(include_v2=True):
            return self._precompute_draft_extend_mode(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_cpu,
                spec_info,
                max_kv_len,
                is_draft_extend_v2=forward_mode.is_draft_extend_v2(),
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")

    def _precompute_decode_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        max_kv_len: Optional[int] = None,
    ) -> PrecomputedMetadata:
        """Precompute metadata for normal decode mode."""
        max_len = (
            max_kv_len if max_kv_len is not None else int(seq_lens_cpu.max().item())
        )

        # Convert to int32 and compute cumsum
        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Get page indices from cache
        page_indices = self.req_to_token[req_pool_indices, :max_len].contiguous()

        # Compute NSA seqlens
        nsa_cache_seqlens = compute_nsa_seqlens(
            cache_seqlens,
            nsa_index_topk=self.nsa_index_topk,
            index_kpool=self.nsa_index_kpool,
        )
        seqlens_expanded = cache_seqlens
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # Compute NSA cumsum
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens)

        # Transform page table if needed
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None  # Will use page_indices directly

        # Compute FlashMLA metadata if needed
        flashmla_metadata = None
        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=nsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            nsa_cache_seqlens=nsa_cache_seqlens,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_len,
            max_seqlen_k=max_len,
            flashmla_metadata=flashmla_metadata,
            req_pool_indices=req_pool_indices,
        )

    def _precompute_target_verify_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        max_kv_len: Optional[int] = None,
    ) -> PrecomputedMetadata:
        """Precompute metadata for target verify mode."""
        max_seqlen_k = (
            max_kv_len
            if max_kv_len is not None
            else int(seq_lens_cpu.max().item() + self.speculative_num_draft_tokens)
        )

        # Cache seqlens with draft tokens
        cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Page indices (repeated for each draft token)
        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=self.speculative_num_draft_tokens, dim=0
        ).contiguous()

        # Generate expanded seqlens
        extend_seq_lens = torch.full(
            (bs,),
            self.speculative_num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        seqlens_expanded = seqlens_expand_triton(
            extend_seq_lens,
            cache_seqlens,
            self.speculative_num_draft_tokens * bs,
            self.speculative_num_draft_tokens,
        )

        # Compute NSA seqlens
        nsa_cache_seqlens = compute_nsa_seqlens(
            seqlens_expanded,
            self.nsa_index_topk,
            index_kpool=self.nsa_index_kpool,
        )
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # NSA cumsum
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens)

        # Transform page table
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None

        # FlashMLA metadata
        flashmla_metadata = None
        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=nsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            nsa_cache_seqlens=nsa_cache_seqlens,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=-1,  # Not used in this mode
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
            req_pool_indices=req_pool_indices,
        )

    def _precompute_draft_extend_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        spec_info: "SpecInput",
        max_kv_len: Optional[int] = None,
        is_draft_extend_v2: bool = False,
    ) -> PrecomputedMetadata:
        """Precompute metadata for draft extend mode."""
        max_seqlen_k = (
            max_kv_len if max_kv_len is not None else int(seq_lens_cpu.max().item())
        )

        # Cache seqlens
        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Extend seqlens from spec_info. V2 uses the explicit q_len tensor
        # prepared by draft-extend replay; V1 uses accepted-token lengths.
        if is_draft_extend_v2:
            extend_seq_lens = spec_info.extend_seq_lens_tensor[:bs]
            extend_seq_lens_cpu = getattr(spec_info, "extend_seq_lens_cpu", None)
            if extend_seq_lens_cpu is None:
                extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs
            else:
                extend_seq_lens_cpu = extend_seq_lens_cpu[:bs]
            extend_sum = sum(extend_seq_lens_cpu)
        else:
            extend_seq_lens = spec_info.accept_length[:bs]
            extend_seq_lens_cpu = extend_seq_lens.tolist()
            extend_sum = sum(extend_seq_lens_cpu)

        # Page indices (repeated per accept length)
        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=extend_seq_lens, dim=0
        ).contiguous()

        # Generate expanded seqlens. The V2 full-overlap path keeps seq_lens_cpu
        # deferred, so use the device-side expansion. V1 still requires the true
        # CPU mirror because its q_len is variable.
        if is_draft_extend_v2:
            seqlens_expanded = seqlens_expand_triton(
                extend_seq_lens,
                cache_seqlens,
                extend_sum,
                self.speculative_num_draft_tokens,
            )
        else:
            assert seq_lens_cpu is not None
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )

        # Compute NSA seqlens
        nsa_cache_seqlens = compute_nsa_seqlens(
            seqlens_expanded,
            self.nsa_index_topk,
            index_kpool=self.nsa_index_kpool,
        )
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # NSA cumsum
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens)

        # Transform page table
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None

        # FlashMLA metadata
        flashmla_metadata = None
        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=nsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            nsa_cache_seqlens=nsa_cache_seqlens,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_seqlen_k,
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
            req_pool_indices=req_pool_indices,
        )
