from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.kpool.kernels import (
    all_gather_and_scatter_pool_slots,
    gather_index_k_scale_prefix_into,
    kpool_assemble_softmax_rotate_write_cache,
    kpool_decode_update_and_maybe_write_cache,
    scatter_kpool_tail_updates,
    topk_from_pooled_history_logits,
)
from sglang.srt.layers.attention.nsa.kpool.page_table import (
    PAGE_SIZE,
    build_pooled_page_table_64,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    DUAL_STREAM_TOKEN_THRESHOLD,
    BaseIndexerMetadata,
    Indexer,
    rotate_activation,
)
from sglang.srt.layers.attention.nsa.utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda, is_hip, is_npu

if is_cuda():
    try:
        import deep_gemm
    except ImportError:
        deep_gemm = None


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
            skip_rope=skip_rope,
        )

        assert config is not None, "IndexerKPool requires the model config"
        self.index_kpool = getattr(config, "index_kpool", 1)
        self.index_kpool_always_select_tail = getattr(
            config, "index_kpool_always_select_tail", False
        )
        self.index_kpool_compress = getattr(config, "index_kpool_compress", False)

        assert (
            self.index_kpool > 1
            and self.index_kpool_compress
            and self.index_kpool_always_select_tail
        ), (
            "IndexerKPool requires index_kpool > 1, index_kpool_compress=True, "
            "and index_kpool_always_select_tail=True."
        )

        assert self.index_topk % self.index_kpool == 0, (
            f"index_topk ({self.index_topk}) must be divisible by "
            f"index_kpool ({self.index_kpool})"
        )
        assert (
            64 % self.index_kpool == 0
        ), f"index_kpool ({self.index_kpool}) must divide page_size (64)"

        # Kpool-specific learned params: absolute positional embedding inside
        # each pool, and the gate projection for the per-token slot score.
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, self.hidden_size, dtype=torch.bfloat16)
        )

        # Extra stream so the gate matmul can overlap with q/k projection
        # when dual-stream is active during decode.
        self.compress_gate_stream = None
        if is_cuda() and self.alt_stream is not None:
            self.compress_gate_stream = torch.cuda.Stream()

    @torch.compile(dynamic=True) if not is_hip() else lambda f: f
    def _project_and_scale_head_gates(self, x: torch.Tensor):
        # Reuse the parent's bf16 weights_proj path; only difference is the
        # GLM head broadcast below.
        weights = self._weights_proj_bf16_in_fp32_out(x)
        # GLM models may use fewer than 32 heads; broadcast to 32 to match
        # downstream kernels that hard-code that count.
        if (num_heads := weights.size(1)) < 32:
            assert 32 % num_heads == 0
            weights = weights.repeat_interleave(32 // num_heads, dim=1)
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

    def _compute_gate_score_if_missing(
        self, x, gate_score: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Materialize gate_score from x if not pre-computed.

        Pulled out of the per-mode compress functions so they don't each
        carry the same fallback branch.
        """
        if gate_score is not None:
            return gate_score
        return F.linear(x, self.index_kpool_compress_gate)

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
          * Optionally launches the compress-gate matmul on a third stream
            when running under dual-stream decode.
          * Under nsa_enable_prefill_cp this rank's K and gate_score are
            all-gathered (then rerange'd into natural order) into the full
            sequence before returning. The query path stays rank-local.
            Rotary embedding is applied locally before all-gather; this is
            safe because rope is a per-token op that commutes with gather.
        """
        from einops import rearrange

        use_cp = (
            nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp)
            and forward_batch.forward_mode.is_extend_without_speculative()
        )
        # precompute_compress_gate only fires in dual-stream decode; CP
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
            if precompute_compress_gate:
                assert self.compress_gate_stream is not None
                self.compress_gate_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                key, _ = self.wk(x)
                key = self.k_norm(key)
                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            if precompute_compress_gate:
                with torch.cuda.stream(self.compress_gate_stream):
                    gate_score = F.linear(x, self.index_kpool_compress_gate)

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )

        if not self.skip_rope and self.rope_head_dim > 0:
            # rotary_emb is in-place on the q_rope / k_rope views; no
            # slice-assign back into query / key is needed.
            self.rotary_emb(positions, q_rope, k_rope)

        query = rotate_activation(query)

        if use_cp:
            # All-gather K and gate_score in one combined collective
            # (they share N and dtype=bf16). Rerange handles both
            # in-seq-split (zigzag) and round-robin layouts.
            if gate_score is None:
                gate_score = F.linear(x, self.index_kpool_compress_gate)
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
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )

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
            self.nsa_enable_prefill_cp and forward_batch.nsa_cp_metadata is not None
        ), "kpool decode under nsa_enable_prefill_cp is not supported"
        batch = key.shape[0]
        pool = forward_batch.token_to_kv_pool
        tail_k_buf, tail_score_buf = pool.get_compress_tail_buffers(layer_id)
        kpool_decode_update_and_maybe_write_cache(
            pool=pool,
            buf=pool.get_index_k_with_scale_buffer(layer_id=layer_id),
            tail_k=tail_k_buf,
            tail_score=tail_score_buf,
            key=key,
            slot_score=gate_score,
            ape=self.index_kpool_compress_ape,
            block_tables=metadata.get_page_table_64(),
            req_pool_indices=forward_batch.req_pool_indices[:batch],
            positions=positions[:batch],
            seq_lens=metadata.get_seqlens_int32()[:batch],
            out_cache_loc=forward_batch.out_cache_loc[:batch],
            round_scale=self.scale_fmt is not None,
        )

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

        tail_k_buf, tail_score_buf = pool.get_compress_tail_buffers(layer_id)

        # --- assemble + compress (owned pools under CP, all otherwise) ---
        if not writes.is_empty:
            # Fused gather + softmax + Hadamard + fp8 quant + cache write.
            # Skips the intermediate (n_pools, pool_size, head_dim) slot
            # tensors that the two-step (assemble_kpool_slots +
            # kpool_softmax_rotate_write_cache) variant materializes.
            kpool_assemble_softmax_rotate_write_cache(
                pool=pool,
                buf=pool.get_index_k_with_scale_buffer(layer_id=layer_id),
                chunk_k=key,
                chunk_score=gate_score,
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                req_pool_idx=writes.req,
                n_from_tail=writes.n_from_tail,
                chunk_src_start=writes.chunk_src,
                ape=self.index_kpool_compress_ape,
                loc=writes.write_loc,
                # CP: write only this rank's owned pools.
                write_mask=(cp.owner_rank == cp.rank) if cp is not None else None,
                round_scale=self.scale_fmt is not None,
            )

            # CP: replicate owned-pool writes so every rank's buf matches.
            if cp is not None and write_cache:
                all_gather_and_scatter_pool_slots(
                    buf=pool.get_index_k_with_scale_buffer(layer_id=layer_id),
                    local_locs=writes.write_loc,
                    owner_rank=cp.owner_rank,
                    cp_size=cp.size,
                )

        # --- scatter tail updates --------------------------------------------
        if not tails.is_empty:
            scatter_kpool_tail_updates(
                chunk_k=key,
                chunk_score=gate_score,
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                req_pool_idx=tails.req,
                dst_offset=tails.dst_offset,
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
    ) -> torch.Tensor:
        return topk_from_pooled_history_logits(
            logits=logits,
            group_lengths=pool_lens,
            pool_size=self.index_kpool,
            topk=self.index_topk,
            page_table=page_table,
            topk_offsets=topk_offsets,
            seq_lens=seq_lens,
            row_starts=row_starts,
        )

    @staticmethod
    def _kpool_fused_topk_mapping(
        metadata: BaseIndexerMetadata,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not envs.SGLANG_NSA_FUSE_TOPK.get():
            return None, None

        name = metadata.topk_transform_method.name
        if name == "PAGED":
            page_table_1 = metadata.attn_metadata.page_table_1
            assert page_table_1 is not None
            return page_table_1, None
        if name == "RAGGED":
            return None, metadata.attn_metadata.topk_indices_offset
        return None, None

    def _full_topk_for_short_sequence(
        self, metadata: BaseIndexerMetadata, device: torch.device
    ) -> torch.Tensor:
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=device,
        )
        return metadata.topk_transform(dummy_logits, self.index_topk)

    def _get_kpool_decode_metadata(
        self,
        metadata: BaseIndexerMetadata,
        block_tables: torch.Tensor,
        seqlens_32: torch.Tensor,
        block_kv: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_metadata = metadata.attn_metadata
        pool_seqlens = attn_metadata.pooled_cache_seqlens_int32
        pool_block_tables = attn_metadata.pooled_real_page_table
        pool_schedule_metadata = attn_metadata.pooled_paged_mqa_schedule_metadata

        if (
            pool_seqlens is None
            or pool_block_tables is None
            or attn_metadata.pooled_index_kpool != self.index_kpool
        ):
            pool_seqlens = torch.div(
                seqlens_32, self.index_kpool, rounding_mode="floor"
            ).to(torch.int32)
            pool_block_tables = build_pooled_page_table_64(
                block_tables, self.index_kpool
            ).contiguous()
            pool_schedule_metadata = None
        else:
            # Anchor: gather stride = index_kpool (16); dense: stride = 1.
            slots_per_pool_page = PAGE_SIZE // self.index_kpool
            gather_stride = max(
                1, self.index_kpool * slots_per_pool_page // PAGE_SIZE
            )
            pool_seqlens = pool_seqlens[: seqlens_32.shape[0]]
            pool_block_tables = pool_block_tables[
                : block_tables.shape[0],
                : (block_tables.shape[1] + gather_stride - 1) // gather_stride,
            ]

        if pool_schedule_metadata is None:
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

        Caller (``forward_cuda``) only dispatches here for decode (speculative
        modes raise NotImplementedError upstream), so we assume decode.
        """
        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        block_tables = metadata.get_page_table_64()
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        seqlens_32 = metadata.get_seqlens_int32()
        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)
        assert len(kv_cache_fp8.shape) == 2
        # Anchor: index_kpool=1  -> block_kv=64, row=8448 B/page
        # Dense : index_kpool=16 -> block_kv=4,  row=528  B/page
        block_kv = PAGE_SIZE // self.index_kpool
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        pool_seqlens, pool_block_tables, pool_schedule_metadata = (
            self._get_kpool_decode_metadata(
                metadata, block_tables, seqlens_32, block_kv
            )
        )
        pool_max_seq_len = pool_block_tables.shape[1] * block_kv
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

        page_table_1, topk_offsets = self._kpool_fused_topk_mapping(metadata)
        return self._topk_from_kpool_logits(
            logits,
            pool_seqlens,
            seq_lens=seqlens_32,
            page_table=page_table_1,
            topk_offsets=topk_offsets,
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
        assert forward_batch.forward_mode.is_extend_without_speculative()
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        attn_metadata = metadata.attn_metadata
        plan = attn_metadata.kpool_extend_plan
        assert (
            plan is not None
        ), "kpool extend plan is required; check _init_kpool_extend_metadata"

        device = q_fp8.device
        total_q = q_fp8.shape[0]
        seq_lens_expanded = metadata.get_seqlens_expanded()
        pool_lens = plan.pooled_seq_lens_expanded
        ks_per_q = plan.ragged_q_ks
        ke_per_q = ks_per_q + pool_lens
        total_k_rows = plan.ragged_total_k_rows

        # --- single fused gather of every batch's compressed K --------
        # Page-aligned per-batch starts mean the kernel's
        # ``page_indices[token_id // PAGE_SIZE]`` math is correct with a
        # flat concatenated page table.
        if total_k_rows > 0:
            # Reuse the per-forward workspace allocated by the planner;
            # all NSA layers share the same (total_k_rows, head_dim)
            # shape, so this saves ~2*64 allocations per forward.
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

            # --- single fp8_mqa_logits over the whole batch ----------
            # Per-q ks/ke index into the flat K buffer. Rows with
            # ks==ke (pool_seq_len==0) get no writes; cleaned to zero
            # by ``clean_logits=True`` below.
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8.contiguous(),
                (k_fp8.contiguous(), k_scale.contiguous()),
                weights.contiguous(),
                ks_per_q,
                ke_per_q,
                clean_logits=True,
            )
        else:
            # No batch has any pool history yet. Build a zero-width
            # logits and let topk fall through to the tail-only path.
            logits = torch.empty((total_q, 0), dtype=torch.float32, device=device)

        # --- single fused topk over the whole batch -----------------
        topk_method_name = metadata.topk_transform_method.name
        topk_offsets = attn_metadata.topk_indices_offset
        fuse_topk = envs.SGLANG_NSA_FUSE_TOPK.get()

        page_table_all = None
        topk_offsets_all = None
        if fuse_topk and topk_method_name == "PAGED":
            page_table_all = plan.ragged_paged_page_table
            assert (
                page_table_all is not None
            ), "kpool ragged topk under PAGED requires plan.ragged_paged_page_table"
        elif fuse_topk and topk_method_name == "RAGGED" and topk_offsets is not None:
            topk_offsets_all = topk_offsets

        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=seq_lens_expanded,
            page_table=page_table_all,
            topk_offsets=topk_offsets_all,
            row_starts=ks_per_q,
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
            gate_score = F.linear(x, self.index_kpool_compress_gate)
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
        if use_cp:
            return cp_split_and_rebuild_data(forward_batch, topk_full)
        return topk_full

    def _cp_topk_full(
        self,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        """CP topk for the kpool ragged path.

        Under nsa_enable_prefill_cp this rank's q tensors only cover its
        rank-local q slice, but the kpool topk path indexes into
        full-sequence metadata (pool counts, page tables) everywhere.
        The simplest correct fix is to all-gather q_fp8 / weights to the
        full sequence, run the standard topk over all global q tokens,
        then slice this rank's local tokens back out of the result.
        Wastes O(cp_size x) topk compute on each rank but reuses the
        non-CP indexing logic verbatim.

        Gather is fused: q_fp8 (1 byte/elem) and weights (2 bytes/elem)
        are dtype-aliased to uint8, concatenated, gathered in ONE NCCL
        call, then sliced + viewed back. The scatter-back at the end
        uses the same zigzag / round-robin layout the model applied to
        hidden_states at entry.
        """
        n_local = q_fp8.shape[0]
        assert weights.shape[0] == n_local, "q_fp8/weights N mismatch"
        q_uint8 = q_fp8.contiguous().view(torch.uint8).reshape(n_local, -1)
        w_uint8 = weights.contiguous().view(torch.uint8).reshape(n_local, -1)
        q_bytes = q_uint8.shape[1]
        w_bytes = w_uint8.shape[1]

        # Single NCCL collective for both tensors -- _cp_gather_concat
        # would also work but we already have a pre-concatenated buffer.
        (combined_full,) = self._cp_gather_concat(
            [torch.cat([q_uint8, w_uint8], dim=1)],
            self.cp_size,
            forward_batch,
        )
        n_full = combined_full.shape[0]
        q_fp8_full = (
            combined_full[:, :q_bytes]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .view(n_full, q_fp8.shape[1], q_fp8.shape[2])
        )
        weights_full = (
            combined_full[:, q_bytes : q_bytes + w_bytes]
            .contiguous()
            .view(weights.dtype)
            .view(n_full, *weights.shape[1:])
        )
        topk_full = self._get_topk_ragged(
            enable_dual_stream=enable_dual_stream,
            forward_batch=forward_batch,
            layer_id=layer_id,
            q_fp8=q_fp8_full,
            weights=weights_full,
            metadata=metadata,
        )
        return cp_split_and_rebuild_data(forward_batch, topk_full)

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
                (x.shape[0], self.index_topk),
                -1,
                dtype=torch.int,
                device="cuda",
            )

        # Cache mode predicates: forward_mode methods are non-trivial and
        # we'd otherwise call each one 2-4 times in this function.
        mode = forward_batch.forward_mode
        is_extend_prefill = mode.is_extend_without_speculative()
        is_decode = mode.is_decode_or_idle()
        is_speculative = mode.is_target_verify() or mode.is_draft_extend(
            include_v2=True
        )
        if is_speculative:
            raise NotImplementedError(
                "index_kpool > 1 with pooled FP8 index cache does not "
                "support target_verify/draft_extend yet."
            )

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # Skip-logits fast path: when every request fits inside the topk
        # window, the indexer just stores K and returns a dummy topk
        # without computing logits.
        if is_extend_prefill and forward_batch.seq_lens_cpu is not None:
            if forward_batch.seq_lens_cpu.max().item() <= self.index_topk:
                return self._forward_cuda_skip_logits(
                    x, positions, forward_batch, layer_id, metadata, return_indices
                )

        # Q/K projection (plus optional compress-gate matmul on a third
        # stream for dual-stream decode). index_kpool > 1 and
        # index_kpool_compress are class invariants (see __init__).
        precompute_compress_gate = (
            enable_dual_stream and is_decode and self.compress_gate_stream is not None
        )
        query, key, gate_score = self._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream,
            forward_batch=forward_batch,
            precompute_compress_gate=precompute_compress_gate,
        )
        if (num_heads := query.size(1)) < 32:
            assert 32 % num_heads == 0
            query = query.repeat_interleave(32 // num_heads, dim=1)

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
        weights = None
        if enable_dual_stream and is_decode:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if precompute_compress_gate:
                self.alt_stream.wait_stream(self.compress_gate_stream)
            with torch.cuda.stream(self.alt_stream):
                self._compress_write_decode(
                    key=key,
                    gate_score=self._compute_gate_score_if_missing(x, gate_score),
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    metadata=metadata,
                )
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            weights = self._get_logits_head_gate(x, q_scale)
            current_stream.wait_stream(self.alt_stream)
        elif is_extend_prefill and self.alt_stream is not None and not use_cp:
            # Compress (key + gate_score) runs on alt_stream concurrently
            # with q quant + gate matmul on the current stream. Both
            # depend only on prior projections, so the only sync is the
            # join before topk reads the freshly written cache.
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                self._compress_write_extend(
                    key=key,
                    gate_score=self._compute_gate_score_if_missing(x, gate_score),
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    metadata=metadata,
                    write_cache=True,
                )
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            weights = self._get_logits_head_gate(x, q_scale)
            current_stream.wait_stream(self.alt_stream)
            # K-only fast path: caller wants the cache populated but
            # not the topk indices.
            if not return_indices:
                return None
        else:
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            gate_score = self._compute_gate_score_if_missing(x, gate_score)
            if is_decode:
                self._compress_write_decode(
                    key=key,
                    gate_score=gate_score,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    metadata=metadata,
                )
            elif is_extend_prefill:
                self._compress_write_extend(
                    key=key,
                    gate_score=gate_score,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    metadata=metadata,
                    write_cache=True,
                )
                if not return_indices:
                    return None
            else:
                raise NotImplementedError(
                    "index_kpool_compress currently supports decode and extend only."
                )

        if weights is None:
            weights = self._get_logits_head_gate(x, q_scale)

        # Topk dispatch:
        #   decode    -> paged kernel
        #   prefill   -> ragged kernel (CP gathers q + slices result back)
        if is_decode:
            return self._get_topk_paged(
                forward_batch, layer_id, q_fp8, weights, metadata
            )

        if use_cp:
            return self._cp_topk_full(
                enable_dual_stream,
                forward_batch,
                layer_id,
                q_fp8,
                weights,
                metadata,
            )
        return self._get_topk_ragged(
            enable_dual_stream=enable_dual_stream,
            forward_batch=forward_batch,
            layer_id=layer_id,
            q_fp8=q_fp8,
            weights=weights,
            metadata=metadata,
        )
