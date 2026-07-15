from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import get_nsa_index_kpool
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.kpool.kernels import (
    INDEX_HEAD_DIM,
    _torch_topk_pooled_history,
    all_gather_and_scatter_pool_slots,
    gather_index_k_scale_prefix_into,
    kpool_bf16_paged_mqa_logits,
    kpool_dequantize_fp8_paged_kv_cache,
    kpool_assemble_softmax_rotate_write_cache,
    kpool_write_tail_and_maybe_compress,
    scatter_kpool_tail_updates,
    topk_from_pooled_history_logits,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    DUAL_STREAM_TOKEN_THRESHOLD,
    BaseIndexerMetadata,
    Indexer,
    rotate_activation,
)
from sglang.srt.layers.attention.nsa.utils import (
    cp_all_gather_rerange_fused,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    is_nsa_prefill_cp_in_seq_split,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.attention.nsa_backend import TopkTransformMethod
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_cuda, is_hip, is_dcu, is_npu

if is_dcu():
    import deepgemm
    from lightop import gemmopt
    from lightop import op

if is_cuda():
    try:
        import deep_gemm
    except ImportError:
        deep_gemm = None


def _ensure_min_heads(x: torch.Tensor) -> torch.Tensor:
    """Broadcast dim=1 up to 8 heads for DeepGEMM.

    ``deep_gemm.fp8_paged_mqa_logits`` requires num_heads in {8, 16, 32, 64};
    small head counts (query and head-gate weights alike) are repeat-interleaved
    up to 8 to satisfy this constraint.
    """
    if (num_heads := x.size(1)) < 8:
        assert 8 % num_heads == 0
        x = x.repeat_interleave(8 // num_heads, dim=1)
    return x


class IndexerKPool(Indexer):
    """Pooled-history sparse indexer.

    Inherits ``Indexer`` and overrides only what differs for the kpool flow:
    head-gate projection (GLM head broadcast), key path (rotation deferred to
    the fused compress kernel), top-k (paged/ragged via the pooled FP8 cache),
    and ``forward_cuda`` end-to-end (compress-write + optional dual-stream
    gate precompute + kpool top-k).
    """

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        config: Optional[PretrainedConfig] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            index_topk=index_topk,
            q_lora_rank=q_lora_rank,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            layer_id=layer_id,
            scale_fmt=scale_fmt,
            block_size=block_size,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            prefix=prefix,
            quant_config=quant_config,
            alt_stream=alt_stream,
        )
        self.skip_rope = skip_rope

        assert config is not None, "IndexerKPool requires the model config"
        # Launch-time invariants (page_size, index_topk %, EAGLE topk)
        # are enforced in server_args; only the kernel-level DeepGEMM
        # divisibility check belongs here.
        self.index_kpool = get_nsa_index_kpool(config)
        assert self.index_kpool > 1, "IndexerKPool requires kpool enabled."
        assert (
            64 % self.index_kpool == 0
        ), f"index_kpool ({self.index_kpool}) must divide DeepGEMM page_size (64)"

        # Kpool-specific learned params: absolute positional embedding inside
        # each pool, and the gate projection for the per-token slot score.
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, self.hidden_size, dtype=torch.bfloat16)
        )

    @torch.compile(dynamic=True) if not is_hip() else lambda f: f
    def _project_and_scale_head_gates(self, x: torch.Tensor):
        # Reuse the parent's bf16 weights_proj path; only difference is the
        # GLM head broadcast below.
        weights = self._weights_proj_bf16_in_fp32_out(x)
        weights = _ensure_min_heads(weights)
        weights = weights * self.n_heads**-0.5
        return weights

    @torch.compile(dynamic=True) if not is_hip() else lambda f: f
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        # Override so the GLM head broadcast in _project_and_scale_head_gates
        # is picked up here too (parent inlines the projection instead of
        # delegating).
        weights = self._project_and_scale_head_gates(x)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    @torch.compile(dynamic=True) if not is_hip() else lambda f: f
    def _get_bf16_logits_head_gate(self, x: torch.Tensor):
        # BF16 logits path mirrors nsa_indexer.py: q is not FP8-quantized, so
        # there is no q_scale term. The MQA kernels consume dense fp32 weights.
        weights = self._project_and_scale_head_gates(x)
        return weights.unsqueeze(-1) * self.softmax_scale

    @staticmethod
    def _cp_gather_concat(
        tensors: List[torch.Tensor],
        cp_size: int,
        forward_batch: ForwardBatch,
    ) -> List[torch.Tensor]:
        """All-gather + rerange a list of same-dtype, same-N tensors with
        a single collective.

        Each tensor is flattened to ``(N, K_i)``, concatenated along the
        feature dim, gathered+rerange'd via ``cp_all_gather_rerange_output``,
        then split back and reshaped to ``(N_full, *trailing_i)``.
        Saves ``len(tensors)-1`` NCCL launches per layer.
        """
        if not tensors:
            return []
        n_local = tensors[0].shape[0]
        flats = []
        feature_sizes = []
        tails = []
        for t in tensors:
            assert t.shape[0] == n_local, "tensors must share first dim"
            assert t.dtype == tensors[0].dtype, "tensors must share dtype"
            tails.append(t.shape[1:])
            flat = t.reshape(n_local, -1).contiguous()
            feature_sizes.append(flat.shape[1])
            flats.append(flat)

        # Fused fast-path: gather + rerange + concat in one multimem collective,
        # reading each source in place. Only 1-2 sources are supported by the kernel.
        gathered = None
        if len(flats) <= 2:
            gathered = cp_all_gather_rerange_fused(flats, cp_size, forward_batch)
        if gathered is None:
            combined = flats[0] if len(flats) == 1 else torch.cat(flats, dim=1)
            gathered = cp_all_gather_rerange_output(
                combined,
                cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        n_full = gathered.shape[0]
        if len(flats) == 1:
            return [gathered.reshape(n_full, *tails[0])]
        out: List[torch.Tensor] = []
        offset = 0
        for size, tail in zip(feature_sizes, tails):
            out.append(gathered[:, offset : offset + size].reshape(n_full, *tail))
            offset += size
        return out

    def _compute_gate_score(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` through the kpool compress-gate."""
        return F.linear(x, self.index_kpool_compress_gate)

    def _compute_gate_score_if_missing(
        self, x, gate_score: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Materialize gate_score from x if not pre-computed.

        Pulled out of the per-mode compress functions so they don't each
        carry the same fallback branch.
        """
        if gate_score is not None:
            return gate_score
        return self._compute_gate_score(x)

    def _project_q(self, q_lora: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
        q_rope, _ = torch.split(
            query,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )
        return query, q_rope

    def _project_k(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )
        return key, k_rope

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        precompute_compress_gate: bool = False,
    ):
        """Override the parent helper.

        Differences vs ``Indexer._get_q_k_bf16``:
          * Skips ``rotate_activation(key)`` -- the kpool path rotates inside
            the fused compress kernel instead.
          * Optionally launches the compress-gate matmul on the alt stream
            when running under dual-stream decode/verify.
          * Under nsa_enable_prefill_cp this rank's K and gate_score are
            all-gathered (then rerange'd into natural order) into the full
            sequence before returning. The query path stays rank-local.
            Rotary embedding is applied locally before all-gather; this is
            safe because rope is a per-token op that commutes with gather.
        """

        use_cp = (
            nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp)
            and forward_batch.forward_mode.is_extend_without_speculative()
        )
        # precompute_compress_gate only fires in dual-stream decode/verify; CP
        # only fires in prefill. They are mutually exclusive by mode --
        # the assert guards against a future caller that breaks this.
        assert not (use_cp and precompute_compress_gate), (
            "precompute_compress_gate and CP are mutually exclusive (decode "
            "vs prefill)"
        )

        gate_score = None
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, q_rope = self._project_q(q_lora)
            with torch.cuda.stream(self.alt_stream):
                key, k_rope = self._project_k(x)
                if precompute_compress_gate:
                    gate_score = self._compute_gate_score(x)

            current_stream.wait_stream(self.alt_stream)
        else:
            query, q_rope = self._project_q(q_lora)
            key, k_rope = self._project_k(x)

        if not self.skip_rope and self.rope_head_dim > 0:
            # rotary_emb is in-place on the q_rope / k_rope views; no
            # slice-assign back into query / key is needed.
            self.rotary_emb(positions, q_rope, k_rope)

        query = rotate_activation(query)

        if use_cp:
            # Fused all-gather of K + gate_score (same N, both bf16). See
            # ``_cp_gather_concat`` for layout. gate_score is None here:
            # precompute_compress_gate is decode/verify-only, CP is prefill-only.
            gate_score = self._compute_gate_score(x)
            key, gate_score = self._cp_gather_concat(
                [key, gate_score], self.cp_size, forward_batch
            )

        return query, key, gate_score

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool = False,
    ):
        """Override: kpool keeps key un-rotated; rotation is fused into the
        compress kernel."""
        key, k_rope = self._project_k(x)
        if not self.skip_rope and self.rope_head_dim > 0:
            # rotary_emb is in-place on the k_rope view; no slice-assign
            # back into key is needed.
            self.rotary_emb(positions, k_rope, k_rope)
        return key

    def _compress_write_decode(
        self,
        key,
        gate_score,
        positions,
        forward_batch,
        layer_id,
        metadata,
    ):
        """Decode-step kpool update.

        Guard: this path reads ``out_cache_loc[:batch]`` which is
        rank-local under CP, but tail buffers are full-seq state. CP at
        decode would silently corrupt the cache, so reject early. In
        practice CP only kicks in at prefill (see ``can_cp_split`` /
        ``forward_mode.is_context_parallel_extend()``); the guard is a
        defense against future config drift.
        """
        if key.shape[0] == 0:
            return
        assert not (
            self.nsa_enable_prefill_cp and forward_batch.attn_cp_metadata is not None
        ), "kpool decode under nsa_enable_prefill_cp is not supported"
        batch = key.shape[0]
        pool = forward_batch.token_to_kv_pool
        tail_k_buf, tail_score_buf = pool.get_tail_buffers(layer_id)
        # Unified decode + verify entry point. Decode passes N=1 and the
        # plan's per-batch addressing (req / write_start / tail_logical_start
        # / write_loc) precomputed in init/update_kpool_write_plan; the
        # kernel writes 1 token per batch into the tail ring and, if that
        # write closed a pool, compresses it into the persistent FP8 cache.
        plan = metadata.attn_metadata.kpool_write_plan
        assert plan is not None, (
            "kpool_write_plan must be built before _compress_write_decode; "
            "see _build_kpool_metadata / init_kpool_write_plan_capture"
        )
        write_loc = plan.write_loc[:batch]
        kpool_write_tail_and_maybe_compress(
            pool=pool,
            buf=pool.get_kpool_index_k_with_scale_write_buffer(layer_id=layer_id),
            key=key,
            score=gate_score,
            tail_k=tail_k_buf,
            tail_score=tail_score_buf,
            ape=self.index_kpool_compress_ape,
            req_pool_indices=plan.req[:batch],
            write_start=plan.write_start[:batch],
            tail_logical_start=plan.tail_logical_start[:batch],
            write_loc=write_loc,
            out_cache_loc=forward_batch.out_cache_loc[:batch],
            num_draft_tokens=1,
            round_scale=self.scale_fmt is not None,
        )
        pool.commit_kpool_index_k_with_scale_write_buffer(layer_id, write_loc)

    def _compress_write_extend(
        self,
        key,
        gate_score,
        positions,
        forward_batch,
        layer_id,
        metadata,
        write_cache: bool = True,
    ):
        """Consume the precomputed ``kpool_extend_plan`` and issue up to
        three GPU launches: assemble slots, compress+write, scatter tail.

        All CPU planning (per-batch splice/bulk/tail decomposition) and
        the write_locs computation are done once in
        ``NativeSparseAttnBackend._init_kpool_extend_metadata`` and reused
        across every NSA layer.

        Under nsa_enable_prefill_cp, ``key`` and ``gate_score`` are the
        all-gathered full-sequence tensors. Each rank still computes the
        full compress, but only its owned pool slots are physically
        written to the FP8 cache (via write_mask); an all-gather then
        replicates those writes to every rank's buf. The tail buffer is
        computed on the full K, so all ranks arrive at the same
        post-extend tail state with zero extra communication.
        """
        plan = metadata.attn_metadata.kpool_extend_plan
        assert (
            plan is not None
        ), "kpool extend plan is required; check _init_kpool_extend_metadata"
        pool = forward_batch.token_to_kv_pool
        writes, tails, cp = plan.writes, plan.tails, plan.cp

        if writes.is_empty and tails.is_empty:
            return

        tail_k_buf, tail_score_buf = pool.get_tail_buffers(layer_id)

        # --- assemble + compress (owned pools under CP, all otherwise) ---
        if not writes.is_empty:
            buf = (
                pool.get_kpool_index_k_with_scale_write_buffer(layer_id=layer_id)
                if write_cache
                else pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            )
            # Fused gather + softmax + Hadamard + fp8 quant + cache write.
            kpool_assemble_softmax_rotate_write_cache(
                pool=pool,
                buf=buf,
                chunk_k=key,
                chunk_score=gate_score,
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                req_pool_idx=writes.req,
                n_from_tail=writes.n_from_tail,
                chunk_src_start=writes.chunk_src,
                tail_logical_base=writes.tail_logical_base,
                ape=self.index_kpool_compress_ape,
                loc=writes.write_loc,
                # CP: write only this rank's owned pools.
                write_mask=cp.local_write_mask if cp is not None else None,
                round_scale=self.scale_fmt is not None,
            )

            # CP: replicate owned-pool writes so every rank's buf matches.
            if cp is not None and write_cache:
                all_gather_and_scatter_pool_slots(
                    buf=buf,
                    local_locs=writes.write_loc,
                    owner_rank=cp.owner_rank,
                    cp_size=cp.size,
                    cp_rank=cp.rank,
                    slots_per_page=forward_batch.token_to_kv_pool.slots_per_page,
                )
            if write_cache:
                pool.commit_kpool_index_k_with_scale_write_buffer(
                    layer_id, writes.write_loc
                )

        # --- scatter tail updates --------------------------------------------
        if not tails.is_empty:
            scatter_kpool_tail_updates(
                pool=pool,
                chunk_k=key,
                chunk_score=gate_score,
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                req_pool_idx=tails.req,
                dst_logical_start=tails.dst_logical_start,
                chunk_src_start=tails.chunk_src,
                n_write=tails.n_write,
            )

    def _topk_from_kpool_logits(
        self,
        logits: torch.Tensor,
        pool_lens: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        topk_offsets: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        out_rows: Optional[int] = None,
        page_table_row_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run pooled-history topk; the fused kernel fills any
        ``out_rows`` past ``logits.shape[0]`` with -1 in-kernel so the
        caller doesn't pay a host-side ``torch.full`` + slice copy.

        ``out_rows`` is used when the caller has q padding past the
        plan's real-token count (mlp-sync TP/CP pad). The padded tail
        is sentinel-filled so downstream sparse-attn treats those rows
        as no-op. ``None`` (default) returns one row per logits row.

        ``page_table_row_index`` indirects the per-row page-table lookup
        (output row ``i`` reads page-table row ``page_table_row_index[i]``),
        letting the caller share a compact page table across q-tokens.
        """

        group_topk = self.index_topk // self.index_kpool
        supported_group_topk = (128, 160, 192, 224, 256, 512, 2048)
        if (
            get_global_server_args().enable_deterministic_inference
            or is_dcu()
            or group_topk not in supported_group_topk
        ):
            return _torch_topk_pooled_history(
                logits=logits,
                group_lengths=pool_lens,
                pool_size=self.index_kpool,
                topk=self.index_topk,
                page_table=page_table,
                topk_offsets=topk_offsets,
                seq_lens=seq_lens,
                row_starts=row_starts,
                out_rows=out_rows,
                page_table_row_index=page_table_row_index,
            )

        return topk_from_pooled_history_logits(
            logits=logits,
            group_lengths=pool_lens,
            pool_size=self.index_kpool,
            topk=self.index_topk,
            page_table=page_table,
            topk_offsets=topk_offsets,
            seq_lens=seq_lens,
            row_starts=row_starts,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )

    @staticmethod
    def _kpool_fused_topk_mapping(
        metadata: BaseIndexerMetadata,
        paged_page_table: Optional[torch.Tensor] = None,
        paged_page_table_row_index: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Resolve (page_table, topk_offsets, page_table_row_index) for the
        fused topk kernel.

        ``paged_page_table`` overrides the PAGED-method page table when the
        caller has a precomputed full-batch page table (ragged path uses
        ``plan.ragged_paged_page_table``); otherwise the decode default
        ``attn_metadata.page_table_1`` is used. ``paged_page_table_row_index``
        (ragged path) indirects the per-row page-table lookup so a compact
        page table (the full ``req_to_token``) can be shared across q-tokens
        instead of densely replicated per q-token.

        ``force_unfused_topk`` (hisparse decode) requires the topk kernel to
        emit *logical* positions so the downstream hisparse coordinator can
        do its own page-table swap; mirrors the community Indexer guard in
        ``NSAIndexerMetadata.topk_transform``.
        """

        if not envs.SGLANG_NSA_FUSE_TOPK.get() or getattr(
            metadata, "force_unfused_topk", False
        ):
            return None, None, None

        method = metadata.topk_transform_method
        if method == TopkTransformMethod.PAGED:
            page_table = (
                paged_page_table
                if paged_page_table is not None
                else metadata.attn_metadata.page_table_1
            )
            assert page_table is not None
            # Row index is only valid alongside the precomputed paged table.
            row_index = (
                paged_page_table_row_index if paged_page_table is not None else None
            )
            return page_table, None, row_index
        if method == TopkTransformMethod.RAGGED:
            return None, metadata.attn_metadata.topk_indices_offset, None
        return None, None, None

    def _full_topk_for_short_sequence(
        self, metadata: BaseIndexerMetadata, device: torch.device
    ) -> torch.Tensor:
        # Short sequences do not need a real topk: selecting arange covers the
        # whole sequence. Build the transformed indices directly so DCU/GLM5
        # topk=128 does not depend on fused RAGGED kernels specialized for 2048.
        seq_lens = metadata.get_seqlens_expanded().to(torch.int32)
        rows = seq_lens.shape[0]
        cols = torch.arange(self.index_topk, device=device, dtype=torch.int64)
        token_ids = cols.unsqueeze(0).expand(rows, -1)
        valid = cols.unsqueeze(0) < seq_lens.to(torch.int64).unsqueeze(1)

        if metadata.topk_transform_method == TopkTransformMethod.PAGED:
            page_table = metadata.attn_metadata.page_table_1
            if page_table.shape[0] != rows:
                token_to_batch_idx = metadata.get_token_to_batch_idx()
                if token_to_batch_idx is not None:
                    page_table = page_table.index_select(
                        0, token_to_batch_idx.to(page_table.device).to(torch.long)
                    )
                elif page_table.shape[0] == 1:
                    page_table = page_table.expand(rows, -1)
                else:
                    raise RuntimeError(
                        "Cannot align paged page_table rows with short-sequence "
                        f"topk rows: page_table={tuple(page_table.shape)}, rows={rows}"
                    )
            safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
            topk_full = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
        elif metadata.topk_transform_method == TopkTransformMethod.RAGGED:
            offsets = metadata.attn_metadata.topk_indices_offset
            if offsets is not None:
                if offsets.ndim == 2:
                    offsets = offsets.squeeze(1)
                topk_full = (token_ids + offsets.to(torch.int64).unsqueeze(1)).to(
                    torch.int32
                )
            else:
                topk_full = token_ids.to(torch.int32)
        else:
            topk_full = token_ids.to(torch.int32)

        topk_full = torch.where(valid, topk_full, torch.full_like(topk_full, -1))
        # Pad to kpool path width [qo_len, index_topk + kpool - 1] with -1 sentinels
        # so consumers like the MTP index-share buffer see a consistent shape.
        pad = torch.full(
            (topk_full.shape[0], self.index_kpool - 1),
            -1,
            dtype=topk_full.dtype,
            device=topk_full.device,
        )
        return torch.cat([topk_full, pad], dim=1)

    def _get_kpool_decode_metadata(
        self,
        metadata: BaseIndexerMetadata,
        block_tables: torch.Tensor,
        seqlens_32: torch.Tensor,
        block_kv: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_metadata = metadata.attn_metadata
        pool_seqlens = attn_metadata.pooled_cache_seqlens_int32
        pool_schedule_metadata = attn_metadata.pooled_paged_mqa_schedule_metadata

        if pool_seqlens is None or attn_metadata.pooled_index_kpool != self.index_kpool:
            pool_seqlens = torch.div(
                seqlens_32, self.index_kpool, rounding_mode="floor"
            ).to(torch.int32)
            pool_schedule_metadata = None
        else:
            pool_seqlens = pool_seqlens[: seqlens_32.shape[0]]

        # Dense kpool (pool_size | PAGE_SIZE): the pooled page table is
        # bitwise identical to the real page table -- consume it directly.
        pool_block_tables = block_tables

        if pool_schedule_metadata is None:
            if is_dcu():
                pool_schedule_metadata = None
            else:
                pool_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    pool_seqlens.unsqueeze(-1), block_kv, self.sm_count
                )

        return pool_seqlens, pool_block_tables, pool_schedule_metadata

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        """Override: pooled-history paged top-k via the kpool fused kernel.

        Decode consumes one row per request from ``cache_seqlens_int32``.
        KPool adds one constraint: cached pooled schedule metadata is reusable
        only for the row layout it was built for.
        """

        assert forward_batch.forward_mode.is_decode_or_idle()

        page_size = forward_batch.token_to_kv_pool.page_size

        block_tables = metadata.get_page_table_64()
        kv_cache_buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        seqlens_32 = metadata.get_seqlens_int32()
        assert len(q_fp8.shape) == 3
        num_q_padded = q_fp8.shape[0]
        n_real = seqlens_32.shape[0]
        if n_real < num_q_padded:
            q_fp8 = q_fp8[:n_real]
            weights = weights[:n_real]
        q_fp8 = q_fp8.unsqueeze(1)
        assert len(kv_cache_buf.shape) == 2
        # Anchor: index_kpool=1  -> block_kv=64, row=8448 B/page
        # Dense : index_kpool=16 -> block_kv=4,  row=528  B/page
        block_kv = page_size // self.index_kpool
        num_heads_kv = 1
        head_dim_with_sf = self.head_dim + 4  # +4 bytes for fp32 scale factor
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        pool_seqlens, pool_block_tables, pool_schedule_metadata = (
            self._get_kpool_decode_metadata(
                metadata, block_tables, seqlens_32, block_kv
            )
        )
        pool_max_seq_len = pool_block_tables.shape[1] * block_kv
        if is_dcu():
            from lightop import gemmopt
            kv_cache_fp8 = kv_cache_buf.view(
                kv_cache_buf.shape[0], block_kv, num_heads_kv, head_dim_with_sf
            )
            kv_cache_bf16 = kpool_dequantize_fp8_paged_kv_cache(kv_cache_fp8)
            logits = gemmopt.paged_mqa_logits(
                q_fp8,
                kv_cache_bf16,
                weights.float(),
                pool_seqlens,
                pool_block_tables,
                None,
                pool_max_seq_len,
            )
        else:
            kv_cache_fp8 = kv_cache_buf.view(
                kv_cache_buf.shape[0], block_kv, num_heads_kv, head_dim_with_sf
            )
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                pool_seqlens.unsqueeze(-1),
                pool_block_tables,
                pool_schedule_metadata,
                pool_max_seq_len,
                clean_logits=False,
            )

        page_table_1, topk_offsets, _ = self._kpool_fused_topk_mapping(metadata)
        return self._topk_from_kpool_logits(
            logits,
            pool_seqlens,
            seq_lens=seqlens_32,
            page_table=page_table_1,
            topk_offsets=topk_offsets,
            # ``out_rows`` makes the topk kernel pad its output back to the
            # padded q row count (caller upstream expects topk_indices.shape[0]
            # == hidden_states.shape[0]); padding rows are filled with -1.
            out_rows=num_q_padded if num_q_padded != n_real else None,
        )

    def _get_topk_ragged(
        self,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        """Kpool-aware ragged extend top-k.

        Single-launch path: one ``gather_index_k_scale_prefix_into`` over
        all batches' compressed K, one ``deep_gemm.fp8_mqa_logits`` on
        the concatenated K with per-q ``ks/ke`` from the plan, and one
        fused topk that honors per-row ``row_starts``. All CPU planning
        lives in ``KPoolExtendPlan`` (see planner.py).

        Signature matches ``Indexer._get_topk_ragged``; ``enable_dual_stream``
        is accepted but unused.
        """

        assert (
            forward_batch.forward_mode.is_extend_without_speculative()
            or forward_batch.forward_mode.is_draft_extend()
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        plan = metadata.attn_metadata.kpool_extend_plan
        assert (
            plan is not None
        ), "kpool extend plan is required; check _init_kpool_extend_metadata"

        device = q_fp8.device
        total_q = q_fp8.shape[0]
        # plan.seq_lens_expanded / pool_lens / ks/ke are sized to match
        # ``q_fp8`` (rank-local under CP round-robin-split, full-seq
        # otherwise) -- topk consumes the rank-local layout directly,
        # matching the community Indexer path.
        seq_lens_expanded = plan.seq_lens_expanded
        pool_lens = plan.pooled_seq_lens_expanded
        ks_per_q = plan.ragged_q_ks
        ke_per_q = plan.ragged_q_ke
        total_k_rows = plan.ragged_total_k_rows

        # The plan sizes ks/ke/seq_lens_expanded against real-token
        # count; q_fp8 may be longer (mlp-sync pad and/or CP gather
        # pad). fp8_mqa_logits requires q.size(0) == ks.size(0), so
        # slice in the logits branch; the topk kernel re-pads to
        # total_q with -1 via ``out_rows`` below.
        n_real = seq_lens_expanded.shape[0]
        assert (
            n_real <= total_q
        ), f"plan has more real rows ({n_real}) than q_fp8 ({total_q})"

        if total_k_rows > 0:
            # Layer-shared workspace allocated by the planner.
            k_u8 = plan.ragged_k_u8
            k_scale = plan.ragged_k_scale
            assert k_u8 is not None and k_scale is not None
            gather_index_k_scale_prefix_into(
                pool=forward_batch.token_to_kv_pool,
                buf=forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
                    layer_id=layer_id
                ),
                page_indices=plan.ragged_concat_page_table,
                seq_len=total_k_rows,
                k_out=k_u8,
                scale_out=k_scale,
            )
            k_fp8 = k_u8.view(torch.float8_e4m3fn)

            # Rows with ks==ke (pool_seq_len==0) get no writes; cleaned
            # to zero by ``clean_logits=True``.
            if is_dcu():
                kv_bf16 = (
                    k_fp8.to(torch.float32) * k_scale.to(torch.float32).unsqueeze(-1)
                ).to(torch.bfloat16).unsqueeze(1).contiguous()
                logits = op.mqa_logits(
                    q_fp8[:n_real].to(torch.bfloat16).contiguous(),
                    kv_bf16,
                    weights[:n_real].to(torch.float32).contiguous(),
                    ks_per_q,
                    ke_per_q,
                    q_fp8[:n_real].shape[0],
                    kv_bf16.shape[0],
                    q_fp8.shape[1],
                    q_fp8.shape[2],
                    None,
                    True,
                )
            else:
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8[:n_real].contiguous(),
                    (k_fp8.contiguous(), k_scale.contiguous()),
                    weights[:n_real].contiguous(),
                    ks_per_q,
                    ke_per_q,
                    clean_logits=True,
                )
        else:
            # No batch has any pool history yet -- topk falls through
            # to the tail-only path on a zero-width logits.
            logits = torch.empty((n_real, 0), dtype=torch.float32, device=device)

        # --- single fused topk over the whole batch -----------------
        page_table_all, topk_offsets_all, page_table_row_index = (
            self._kpool_fused_topk_mapping(
                metadata,
                paged_page_table=plan.ragged_paged_page_table,
                paged_page_table_row_index=plan.ragged_paged_page_table_row_index,
            )
        )

        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=seq_lens_expanded,
            page_table=page_table_all,
            topk_offsets=topk_offsets_all,
            row_starts=ks_per_q,
            out_rows=total_q,
            page_table_row_index=page_table_row_index,
        )

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: Optional[List[Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        """Mirror of ``Indexer._get_topk_ragged_with_cp`` for the kpool path.

        Used under ``--nsa-prefill-cp-mode in-seq-split``: the caller
        splits q_fp8/weights into prev/next halves and invokes this
        function once per half (each ``actual_seq_q``-long, attending to
        ``kv_len`` history tokens). Topk is taken in pool-level logits
        (``ke = pool_kv_len``) but expanded to token indices by
        ``_topk_from_kpool_logits`` so the per-q result has the same
        token-level layout as the non-CP path.

        Currently only the single-batch path (``cp_index is None``) is
        implemented; multi-batch CP has the same TODO as the community
        ``Indexer`` version (accuracy issues; see nsa_indexer.py).
        """
        assert (
            cp_index is None
        ), "kpool path does not support multi-batch CP yet (mirrors community TODO)"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        pool = forward_batch.token_to_kv_pool
        pool_size = self.index_kpool
        slots_per_page = pool.slots_per_page  # PAGE_SIZE // pool_size in dense mode
        device = q_fp8.device

        assert forward_batch.seq_lens_cpu is not None
        assert forward_batch.extend_seq_lens_cpu is not None
        # Mirror community arithmetic: prefix tokens already in cache, plus
        # this rank's half-share of the current chunk.
        kv_len_token = int(
            forward_batch.seq_lens_cpu[0].item()
            - forward_batch.extend_seq_lens_cpu[0]
            + kv_len
        )
        # Pool-level history this q half can attend to. Floor division
        # matches the planner's pooled_seq_lens math.
        pool_kv_len = kv_len_token // pool_size

        # Per-batch page table (this is a single-batch path; community
        # code also takes block_tables[0]).
        block_tables = metadata.get_page_table_64()
        bt_row = block_tables[0]

        # Token-level tail positions this rank's q rows attend to (right
        # edge inclusive); reused as seq_lens_expanded below.
        tail_tokens = torch.arange(
            kv_len_token - actual_seq_q + 1,
            kv_len_token + 1,
            dtype=torch.int32,
            device=device,
        )
        if pool_kv_len > 0:
            # ``gather_index_k_scale_prefix_into`` expects one packed-page
            # id per page (kernel does `token_id // slots_per_page` to
            # index into page_indices), not per pool entry. Take the
            # leading `n_pages` entries of this batch's page-table row.
            n_pages = (pool_kv_len + slots_per_page - 1) // slots_per_page
            packed_page_indices = bt_row[:n_pages].to(torch.int32).contiguous()
            k_u8 = torch.empty(
                (pool_kv_len, INDEX_HEAD_DIM), dtype=torch.uint8, device=device
            )
            k_scale = torch.empty((pool_kv_len,), dtype=torch.float32, device=device)
            buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            gather_index_k_scale_prefix_into(
                pool=pool,
                buf=buf,
                page_indices=packed_page_indices,
                seq_len=pool_kv_len,
                k_out=k_u8,
                scale_out=k_scale,
            )
            k_fp8 = k_u8.view(torch.float8_e4m3fn)

            # Pool-level ks/ke: ks all 0; ke = right-edge pool index
            # (exclusive), i.e. tail_token // pool_size.
            ks = torch.zeros((actual_seq_q,), dtype=torch.int32, device=device)
            ke = torch.div(tail_tokens, pool_size, rounding_mode="floor").to(
                torch.int32
            )

            if is_dcu():
                kv_bf16 = (
                    k_fp8.to(torch.float32) * k_scale.to(torch.float32).unsqueeze(-1)
                ).to(torch.bfloat16).unsqueeze(1).contiguous()
                logits = op.mqa_logits(
                    q_fp8.to(torch.bfloat16).contiguous(),
                    kv_bf16,
                    weights.to(torch.float32).contiguous(),
                    ks,
                    ke,
                    q_fp8.shape[0],
                    kv_bf16.shape[0],
                    q_fp8.shape[1],
                    q_fp8.shape[2],
                    None,
                    True,
                )
            else:
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8.contiguous(),
                    (k_fp8.contiguous(), k_scale.contiguous()),
                    weights.contiguous(),
                    ks,
                    ke,
                    clean_logits=True,
                )
            pool_lens = ke  # per-q pool-history length
        else:
            # No pool history yet; topk degenerates to tail-only.
            logits = torch.empty((actual_seq_q, 0), dtype=torch.float32, device=device)
            pool_lens = torch.zeros((actual_seq_q,), dtype=torch.int32, device=device)

        seq_lens_expanded = tail_tokens

        # Single-batch CP path. Resolve fuse-topk inputs for this batch:
        # PAGED -> replicate batch 0's page_table to actual_seq_q rows.
        # RAGGED -> topk_indices_offset is 0 for batch 0, so leave None
        # (full-seq metadata copy would shape-mismatch logits.shape[0]).
        page_table_all: Optional[torch.Tensor] = None
        if (
            envs.SGLANG_NSA_FUSE_TOPK.get()
            and metadata.topk_transform_method == TopkTransformMethod.PAGED
        ):
            page_table_all = bt_row.unsqueeze(0).expand(actual_seq_q, -1)

        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=seq_lens_expanded,
            page_table=page_table_all,
            topk_offsets=None,
            row_starts=None,  # ks is all-zero in single-batch; kernel default = 0
            out_rows=actual_seq_q,
        )

    def _forward_cuda_skip_logits(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert forward_batch.forward_mode.is_extend_without_speculative()

        use_cp = nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp)

        key = self._get_k_bf16(x, positions)
        gate_score = None
        if use_cp:
            gate_score = self._compute_gate_score(x)
            key, gate_score = self._cp_gather_concat(
                [key, gate_score], self.cp_size, forward_batch
            )

        # Skip-logits path is extend-only (caller asserts above).
        self._compress_write_extend(
            key=key,
            gate_score=self._compute_gate_score_if_missing(x, gate_score),
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
            metadata=metadata,
        )

        if not return_indices:
            return None

        topk_full = self._full_topk_for_short_sequence(metadata, x.device)
        # in-seq-split: topk_full spans the full batch; reorder to this rank's
        # q slice via zigzag. round-robin-split: metadata.seqlens_expanded is
        # already rank-local upstream (nsa_backend), so topk_full is rank-local
        # too -- no further split.
        if use_cp and is_nsa_prefill_cp_in_seq_split():
            return cp_split_and_rebuild_data(forward_batch, topk_full)
        return topk_full

    # ------------------------------------------------------------------
    # target_verify (chain-only, EAGLE topk=1)
    # ------------------------------------------------------------------

    def _forward_cuda_target_verify(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        metadata: BaseIndexerMetadata,
        enable_dual_stream: bool,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """Verify forward: draft K/score ring-written into the tail at logical
        positions ``[committed, committed+N)``, closed pools compressed
        directly into persistent FP8, paged top-k via the persistent
        page_table_64.

        Verify is essentially "extend assuming all drafts accepted": closed
        pools written here become correct iff the corresponding drafts
        accept; for any rejected draft past the accept frontier the next
        decode/extend will rewrite the pool from authoritative K. The tail
        ring layout means the post-sample commit is a no-op (just advance
        ``committed_seq_len`` upstream).

        CUDA-only: target_verify with kpool only runs on CUDA today.
        """
        #assert is_cuda(), "kpool ring-write path is CUDA-only"
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        plan = metadata.attn_metadata.kpool_write_plan
        assert plan is not None, (
            "kpool target_verify plan is required; see "
            "init_kpool_write_plan in nsa_backend.py"
        )

        # (1) Q/K bf16 projection (Hadamard already applied in _get_q_k_bf16).
        # Under cuda graph, Q projection stays on the current stream while K
        # projection and the compress-gate matmul run on the alt stream.
        query, key, gate_score_maybe = self._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream=enable_dual_stream,
            forward_batch=forward_batch,
            precompute_compress_gate=enable_dual_stream,
        )
        query = _ensure_min_heads(query)

        def _compress_write() -> None:
            # Fused: write N draft K/score into the tail ring, then compress
            # any closed pool from tail into the persistent FP8 cache. One
            # kernel launch (grid=(B,)) handles both steps per batch; the
            # compress half is gated in-kernel on pool-boundary crossing.
            pool = forward_batch.token_to_kv_pool
            tail_k_buf, tail_score_buf = pool.get_tail_buffers(layer_id)
            buf = pool.get_kpool_index_k_with_scale_write_buffer(layer_id=layer_id)
            kpool_write_tail_and_maybe_compress(
                pool=pool,
                buf=buf,
                key=key,
                score=self._compute_gate_score_if_missing(x, gate_score_maybe),
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                ape=self.index_kpool_compress_ape,
                req_pool_indices=plan.req,
                write_start=plan.write_start,
                tail_logical_start=plan.tail_logical_start,
                write_loc=plan.write_loc,
                out_cache_loc=forward_batch.out_cache_loc,
                num_draft_tokens=plan.num_draft_tokens,
                round_scale=self.scale_fmt is not None,
                # v2 only (None for target_verify): defer compress to the
                # round whose real advance crosses a pool boundary.
                effective_n_per_batch=plan.effective_n_per_batch,
            )
            pool.commit_kpool_index_k_with_scale_write_buffer(
                layer_id, plan.write_loc
            )

        # (2) Run tail/compress on the alt stream while the current stream
        # prepares the platform-specific query representation and head-gate
        # weights for paged top-k (BF16 on DCU, FP8 elsewhere).
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if return_indices:
                if is_dcu():
                    q_index = query
                    weights = self._get_bf16_logits_head_gate(x)
                else:
                    q_index, q_scale = act_quant(
                        query, self.block_size, self.scale_fmt
                    )
                    weights = self._get_logits_head_gate(x, q_scale)
            with torch.cuda.stream(self.alt_stream):
                _compress_write()
            current_stream.wait_stream(self.alt_stream)
        else:
            _compress_write()
            if return_indices:
                if is_dcu():
                    q_index = query
                    weights = self._get_bf16_logits_head_gate(x)
                else:
                    q_index, q_scale = act_quant(
                        query, self.block_size, self.scale_fmt
                    )
                    weights = self._get_logits_head_gate(x, q_scale)

        if not return_indices:
            return None

        # (3) Top-k via persistent paged page_table (no stitched reserve).
        return self._get_topk_paged_verify(
            forward_batch, layer_id, q_index, weights, plan, metadata
        )

    def _get_topk_paged_verify(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        plan,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        """Paged top-k over committed pool history + the just-written drafts.

        ``plan.paged_page_table`` is the persistent page_table_64 (no
        reserve stitching). Closed-pool FP8 was just written to those
        very pages above, so the paged logits kernel reads correct
        draft-assuming data.
        """
        page_size = forward_batch.token_to_kv_pool.page_size
        kv_cache_buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )
        block_kv = page_size // self.index_kpool
        head_dim_with_sf = self.head_dim + 4

        assert q_index.dim() == 3
        num_q_padded = q_index.shape[0]
        n_real = min(
            num_q_padded,
            plan.pool_seqlens_per_q.shape[0],
            plan.seqlens_per_q.shape[0],
            plan.paged_page_table.shape[0],
        )
        if n_real < num_q_padded:
            q_index = q_index[:n_real]
            weights = weights[:n_real]
        q_index = q_index.unsqueeze(1)
        assert weights.dim() == 3
        weights = weights.squeeze(2)
        pool_seqlens_per_q = plan.pool_seqlens_per_q[:n_real]
        seqlens_per_q = plan.seqlens_per_q[:n_real]
        paged_page_table = plan.paged_page_table[:n_real]

        pool_max_seq_len = paged_page_table.shape[1] * block_kv
        if is_dcu():
            from lightop import gemmopt
            kv_cache_fp8 = kv_cache_buf.view(
                kv_cache_buf.shape[0], block_kv, 1, head_dim_with_sf
            )
            kv_cache_bf16 = kpool_dequantize_fp8_paged_kv_cache(kv_cache_fp8)
            logits = gemmopt.paged_mqa_logits(
                q_index,
                kv_cache_bf16,
                weights.float(),
                pool_seqlens_per_q,
                paged_page_table,
                None,
                pool_max_seq_len,
            )
        else:
            kv_cache_fp8 = kv_cache_buf.view(
                kv_cache_buf.shape[0], block_kv, 1, head_dim_with_sf
            )
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_index,
                kv_cache_fp8,
                weights,
                pool_seqlens_per_q.unsqueeze(-1),
                paged_page_table,
                plan.pool_schedule_metadata,
                pool_max_seq_len,
                clean_logits=False,
            )

        # Fused top-k method dispatch. ``_kpool_fused_topk_mapping`` resolves
        # to page_table_1 (token-granularity, B*N rows after repeat_interleave),
        # which the topk transform expects. plan.paged_page_table is the
        # page_table_64 used by DeepGEMM above and must NOT leak to topk.
        page_table_for_topk, _, _ = self._kpool_fused_topk_mapping(metadata)

        return self._topk_from_kpool_logits(
            logits=logits,
            pool_lens=pool_seqlens_per_q,
            seq_lens=seqlens_per_q,
            page_table=page_table_for_topk,
            topk_offsets=None,
            out_rows=num_q_padded if num_q_padded != n_real else None,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if is_hip():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
        elif not is_npu():
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )
        if metadata is None:
            return None

        # Empty batch (e.g. cuda-graph max-pad slot): return all-invalid
        # before doing any projections / quant.
        assert forward_batch.seq_lens_cpu is not None
        if len(forward_batch.seq_lens_cpu) == 0:
            return torch.full(
                (x.shape[0], self.index_topk + self.index_kpool - 1),
                -1,
                dtype=torch.int,
                device=x.device,
            )

        # Cache mode predicates: forward_mode methods are non-trivial and
        # we'd otherwise call each one 2-4 times in this function.
        mode = forward_batch.forward_mode
        is_extend = mode.is_extend_without_speculative()
        is_decode = mode.is_decode_or_idle()
        is_target_verify = mode.is_target_verify()
        is_draft_extend_v2 = mode.is_draft_extend_v2()
        is_draft_extend_v1 = mode.is_draft_extend() and not is_draft_extend_v2

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # target_verify and draft_extend_v2 share the same KPoolWritePlan
        # path: fixed q_len = N per batch, identical tail-ring + close-pool
        # write semantics, identical paged topk shape ``[B*N, ...]``.
        # v1 draft_extend still has variable q_len = accept_length+1 and
        # falls through to the extend planner below.
        if is_target_verify or is_draft_extend_v2:
            return self._forward_cuda_target_verify(
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
                enable_dual_stream=enable_dual_stream,
                return_indices=return_indices,
            )

        # Skip-logits fast path: when every request fits inside the topk
        # window, the indexer just stores K and returns a dummy topk
        # without computing logits.
        if is_extend and forward_batch.seq_lens_cpu is not None:
            if forward_batch.seq_lens_cpu.max().item() <= self.index_topk:
                return self._forward_cuda_skip_logits(
                    x, positions, forward_batch, layer_id, metadata, return_indices
                )

        # Q/K projection (plus optional compress-gate matmul on the shared
        # alt stream for dual-stream decode).
        precompute_compress_gate = enable_dual_stream and is_decode
        query, key, gate_score = self._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream,
            forward_batch=forward_batch,
            precompute_compress_gate=precompute_compress_gate,
        )
        query = _ensure_min_heads(query)

        # Three scheduling paths:
        #   (a) dual-stream decode: compress runs on alt stream while q
        #       quant + logits-gate run on current stream; weights are
        #       ready before logits.
        #   (b) prefill with alt_stream available AND no CP: compress
        #       runs on alt stream in parallel with q quant + logits-gate
        #       on current stream. CP is excluded because compress_write
        #       contains an NCCL all-gather (all_gather_and_scatter_pool_slots)
        #       which must run on the current stream to keep collective
        #       ordering consistent across ranks.
        #   (c) fallback (no alt_stream, CP enabled, or unknown mode):
        #       sequential.
        use_cp = nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp)

        if is_decode:
            compress_fn = self._compress_write_decode
        elif is_extend or is_draft_extend_v1:
            # v1 draft_extend: variable q_len = accept_length+1; goes through
            # the extend planner like a short prefill.
            compress_fn = self._compress_write_extend
        else:
            raise NotImplementedError(
                "index_kpool_compress currently supports decode and extend only."
            )

        overlap_decode = enable_dual_stream and is_decode
        overlap_prefill = is_extend and self.alt_stream is not None and not use_cp
        if overlap_decode or overlap_prefill:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if is_dcu():
                q_index = query
                weights = self._get_bf16_logits_head_gate(x)
            else:
                q_index, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                weights = self._get_logits_head_gate(x, q_scale)
            with torch.cuda.stream(self.alt_stream):
                # Resolve gate_score on alt_stream so the fallback
                # F.linear overlaps with q quant on the current stream.
                compress_fn(
                    key=key,
                    gate_score=self._compute_gate_score_if_missing(x, gate_score),
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    metadata=metadata,
                )
            current_stream.wait_stream(self.alt_stream)
        else:
            compress_fn(
                key=key,
                gate_score=self._compute_gate_score_if_missing(x, gate_score),
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
            )
            if is_dcu():
                q_index = query
                weights = self._get_bf16_logits_head_gate(x)
            else:
                q_index, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                weights = self._get_logits_head_gate(x, q_scale)

        # K-only fast path (extend only): caller wants the cache
        # populated but not the topk indices.
        if (is_extend or is_draft_extend_v1) and not return_indices:
            return None

        # Topk dispatch:
        #   decode               -> paged kernel
        #   draft_extend v1      -> ragged extend-plan kernel; v1 has
        #                           variable accept_length+1 rows per request,
        #                           and the kpool planner already builds the
        #                           matching per-q pooled-K layout. v2 already
        #                           returned above via the verify path.
        #   prefill (in-seq-split CP) -> ragged_with_cp on prev/next halves
        #   prefill (otherwise)  -> ragged kernel; CP round-robin-split runs
        #                           rank-local because the planner already
        #                           produces rank-local ks/ke/page_table.
        if is_decode:
            return self._get_topk_paged(
                forward_batch, layer_id, q_index, weights, metadata
            )

        if is_draft_extend_v1:
            return self._get_topk_ragged(
                enable_dual_stream,
                forward_batch,
                layer_id,
                q_index,
                weights,
                metadata,
            )

        if (
            forward_batch.attn_cp_metadata is not None
            and is_nsa_prefill_cp_in_seq_split()
        ):
            cp_meta = forward_batch.attn_cp_metadata
            q_index_prev, q_index_next = torch.split(
                q_index, (q_index.shape[0] + 1) // 2, dim=0
            )
            weights_prev, weights_next = torch.split(
                weights, (weights.shape[0] + 1) // 2, dim=0
            )
            topk_prev = self._get_topk_ragged_with_cp(
                forward_batch,
                layer_id,
                q_index_prev,
                weights_prev,
                metadata,
                kv_len=cp_meta.kv_len_prev,
                actual_seq_q=cp_meta.actual_seq_q_prev,
            )
            topk_next = self._get_topk_ragged_with_cp(
                forward_batch,
                layer_id,
                q_index_next,
                weights_next,
                metadata,
                kv_len=cp_meta.kv_len_next,
                actual_seq_q=cp_meta.actual_seq_q_next,
            )
            return torch.cat([topk_prev, topk_next], dim=0)

        return self._get_topk_ragged(
            enable_dual_stream=enable_dual_stream,
            forward_batch=forward_batch,
            layer_id=layer_id,
            q_fp8=q_index,
            weights=weights,
            metadata=metadata,
        )
