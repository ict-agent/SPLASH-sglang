"""Context-parallel KDA prefill correctness tests.

Two tiers:

1. ``TestKDAContextParallel`` — exercises the parallel all-gather + merge
   *kernel* (``chunk_kda_cp`` + ``fla/cp`` + the plain-split metadata / context
   helpers) against a single-process reference (``chunk_kda`` over the full
   sequence). This isolates the kernel from the backend glue.

2. ``TestKDABackendForwardExtend`` — drives the *real*
   ``KDAAttnBackend.forward_extend`` CP path end-to-end (conv1d halo, prefix
   seed, all-gather + merge, reduce_scatter / all_reduce final-state writeback,
   and the extra_buffer radix-cache track incl. the cross-rank short
   recurrence) and compares this rank's output slice **and** the states written
   into the persistent mamba cache against a single-process reference that
   replays the identical op sequence (conv -> gate -> chunk_kda).

Both require >= 2 CUDA GPUs; one process per rank. Run directly:

    python -m pytest test/registered/attention/test_kda_cp.py -s
    # or a single scenario:
    python test/registered/attention/test_kda_cp.py

NOTE: authored on a CPU-only host; must be run on a multi-GPU box.
"""

import os
import types
import unittest
from itertools import accumulate

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _init_distributed(rank: int, world_size: int, port: str = "29555"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Tier 1: kernel-only test (chunk_kda_cp vs full-sequence chunk_kda)
# ---------------------------------------------------------------------------
def _worker(rank, world_size, lengths, H, D, dtype, ratio, ok_flag):
    from sglang.srt.layers.attention.fla.kda import chunk_kda, chunk_kda_cp
    from sglang.srt.layers.attention.linear.kda_cp_utils import (
        build_kda_fla_cp_context,
        kda_cp_continuation_segment_index,
        kda_cp_owner_of_global_token,
        prepare_kda_prefill_cp_metadata,
    )

    try:
        _init_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        T = sum(lengths)
        N = len(lengths)

        # ---- global inputs (generated identically on every rank) ----
        torch.manual_seed(1234)
        q = torch.randn(1, T, H, D, device=device, dtype=dtype)
        k = torch.randn(1, T, H, D, device=device, dtype=dtype)
        v = torch.randn(1, T, H, D, device=device, dtype=dtype)
        # Mild gate: a strong synthetic gate (logsigmoid(randn)) decays the SSM
        # state into the fp underflow regime over long sequences, making the
        # final-state comparison meaningless noise. Real KDA gates sit near 0
        # (decay ~1). Scale down so the accumulated state stays O(1).
        g = (
            F.logsigmoid(torch.randn(1, T, H, D, device=device, dtype=torch.float))
            .clamp_(min=-5.0)
            .mul_(0.001)
        )
        beta = torch.randn(1, T, H, device=device, dtype=dtype).sigmoid()

        cu = [0]
        for x in lengths:
            cu.append(cu[-1] + x)
        cu_seqlens_global = torch.tensor(cu, device=device, dtype=torch.long)

        # ---- single-process reference over the full sequence ----
        # NOTE: the KDA kernel writes its output IN PLACE over the `v` buffer
        # (chunk_gla_fwd_o_gk(o=v)), so the reference MUST run on clones or it
        # would clobber the q/k/v/g/beta that the CP call below reuses.
        ref_state = torch.zeros(N, H, D, D, device=device, dtype=torch.float32)
        o_ref, _ = chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            initial_state=ref_state,
            initial_state_indices=torch.arange(N, device=device, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens_global,
        )
        # ref_state now holds each sequence's final state (in-place).

        # ---- context-parallel run on this rank's token slice ----
        meta = prepare_kda_prefill_cp_metadata(
            total_tokens=T,
            extend_seq_lens_cpu=lengths,
            cp_rank=rank,
            cp_size=world_size,
            device=device,
        )
        ctx = build_kda_fla_cp_context(meta, dist.group.WORLD, conv1d_kernel_size=4)
        lo, hi = meta.local_start, meta.local_end
        n_local = len(meta.local_seq_lens_cpu)
        cont_idx = kda_cp_continuation_segment_index(meta)

        local_state = torch.zeros(
            max(n_local, 1), H, D, D, device=device, dtype=torch.float32
        )
        o_local, _ = chunk_kda_cp(
            q=q[:, lo:hi],
            k=k[:, lo:hi],
            v=v[:, lo:hi],
            g=g[:, lo:hi],
            beta=beta[:, lo:hi],
            initial_state=local_state,
            initial_state_indices=torch.arange(n_local, device=device, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=meta.local_query_start_loc,
            cp_context=ctx,
            continuation_h0=None,
            continuation_index=cont_idx,
        )

        passed = True

        # (1) output: compare this rank's slice against the reference slice.
        if hi > lo:
            diff = (o_local.float() - o_ref[:, lo:hi].float()).abs()
            denom = o_ref[:, lo:hi].float().abs().mean().clamp_min(1e-6)
            rel = diff.mean() / denom
            if not torch.isfinite(rel) or rel.item() > ratio:
                print(f"[rank {rank}] OUTPUT rel err {rel.item():.4e} > {ratio}")
                passed = False

        # (2) final state: the owner of each sequence's last token checks it.
        for seq_idx in range(N):
            seq_end = cu[seq_idx + 1]
            owner = kda_cp_owner_of_global_token(seq_end - 1, T, world_size)
            if owner != rank:
                continue
            local_seg = None
            for li, (rq, se) in enumerate(
                zip(meta.local_req_indices_cpu, meta.local_segment_global_ends_cpu)
            ):
                if rq == seq_idx and se == seq_end:
                    local_seg = li
                    break
            assert local_seg is not None, (rank, seq_idx)
            diff = (local_state[local_seg] - ref_state[seq_idx]).abs()
            denom = ref_state[seq_idx].abs().mean().clamp_min(1e-6)
            rel = diff.mean() / denom
            if not torch.isfinite(rel) or rel.item() > ratio:
                print(f"[rank {rank}] STATE seq {seq_idx} rel err {rel.item():.4e} > {ratio}")
                passed = False

        flag = torch.tensor([1 if passed else 0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok_flag.value = int(flag.item())
        dist.barrier()
    finally:
        _cleanup()


def _run(world_size, lengths, H=4, D=128, dtype=torch.bfloat16, ratio=2e-2):
    ok = mp.Manager().Value("i", 0)
    mp.start_processes(
        _worker,
        args=(world_size, lengths, H, D, dtype, ratio, ok),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    return ok.value == 1


# ---------------------------------------------------------------------------
# Tier 2: full KDAAttnBackend.forward_extend integration test
# ---------------------------------------------------------------------------
#
# Config knobs (per scenario):
#   lengths      : global per-request extend token counts
#   prefix_lens  : per-request cached-prefix length (0 => no prefix cache)
#   sharded      : persistent mamba cache stores 1/cp head+channel shard
#                  (reduce_scatter writeback) vs full (all_reduce writeback)
#   track        : exercise the extra_buffer radix-cache snapshot writeback
#
# The reference replays forward_extend's exact op order over the full sequence
# on every rank: conv1d (with prefix conv history) -> fused_kda_gate ->
# chunk_kda. It then checks, per rank: this rank's output slice, every
# request's final SSM + conv state, and (track) every tracked request's
# snapshot at bc = req_start + (lens // 64) * 64.

# Small but realistic KDA dims. head_k == head_v keeps every triton kernel on a
# well-supported shape; both are divisible by cp_size so the reduce_scatter
# fast path is exercised when sharded=True.
_H = 4          # num_q_heads
_DK = 128       # head_k_dim == head_q_dim
_DV = 128       # head_v_dim
_W = 4          # conv kernel width (conv state len = W - 1 = 3 < 64)
_CHUNK = 64     # FLA chunk size


def _make_layer(device, dtype, seed=7):
    gen = torch.Generator(device=device).manual_seed(seed)
    q_dim = k_dim = _H * _DK
    v_dim = _H * _DV
    conv_dim = q_dim + k_dim + v_dim
    layer = types.SimpleNamespace(
        layer_id=0,
        num_q_heads=_H,
        head_q_dim=_DK,
        head_k_dim=_DK,
        head_v_dim=_DV,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        # conv weight is [conv_dim, W]; bias [conv_dim].
        conv_weights=(
            0.1 * torch.randn(conv_dim, _W, device=device, dtype=dtype, generator=gen)
        ),
        bias=0.1 * torch.randn(conv_dim, device=device, dtype=dtype, generator=gen),
        A_log=torch.randn(_H, device=device, dtype=torch.float32, generator=gen),
        dt_bias=torch.randn(q_dim, device=device, dtype=torch.float32, generator=gen),
        beta_scale=1.0,
        safe_gate=False,
        safe_gate_lower_bound=-5.0,
    )
    return layer


def _apply_conv(layer, raw_mixed_TD, cu, seq_lens_cpu, conv_init, has_init, cache_idx):
    """Replicate forward_extend's per-request causal_conv1d over [D, T]."""
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_fn

    splits = [layer.q_dim, layer.k_dim, layer.v_dim]
    mixed = raw_mixed_TD.transpose(0, 1).contiguous()  # [D, T]
    q, k, v = mixed.split(splits, dim=0)
    qw, kw, vw = layer.conv_weights.split(splits, dim=0)
    qb, kb, vb = layer.bias.split(splits, dim=0)
    qs, ks, vs = conv_init.split(splits, dim=-2)  # [N, dim, W-1]

    def cv(x, w, b, s):
        return causal_conv1d_fn(
            x,
            w,
            b,
            activation="silu",
            # clone (don't mutate caller's init buffer) + match input dtype
            # (real forward_extend keeps conv_states in the activation dtype).
            conv_states=s.clone().to(x.dtype),
            has_initial_state=has_init,
            cache_indices=cache_idx,
            query_start_loc=cu,
            seq_lens_cpu=seq_lens_cpu,
        ).transpose(0, 1)

    q = cv(q, qw, qb, qs).unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
    k = cv(k, kw, kb, ks).unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
    v = cv(v, vw, vb, vs).unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
    return q, k, v  # each [1, T, H, d]


def _reference(layer, raw_mixed_TD, a_raw, b_raw, lengths, prefix_ssm, prefix_conv, device):
    """Full-sequence reference: conv -> gate -> chunk_kda. Returns
    (o [1,T,H,DV], post_q, post_k, post_v, ga, gb, ssm_final [N,H,DV,DK])."""
    from sglang.srt.layers.attention.fla.kda import chunk_kda, fused_kda_gate

    N = len(lengths)
    T = sum(lengths)
    cu = torch.tensor([0] + list(accumulate(lengths)), device=device, dtype=torch.int32)
    cache_idx = torch.arange(N, device=device, dtype=torch.int32)
    has_init = torch.tensor([p > 0 for p in prefix_conv[1]], device=device)

    post_q, post_k, post_v = _apply_conv(
        layer, raw_mixed_TD, cu, list(lengths), prefix_conv[0], has_init, cache_idx
    )

    ga, gb = fused_kda_gate(
        a_raw,
        layer.A_log,
        layer.head_k_dim,
        g_bias=layer.dt_bias,
        safe_gate=layer.safe_gate,
        lower_bound=layer.safe_gate_lower_bound,
        beta=b_raw,
        beta_scale=layer.beta_scale,
    )

    ssm_final = prefix_ssm.clone()
    o, _ = chunk_kda(
        q=post_q,
        k=post_k,
        v=post_v,
        g=ga,
        beta=gb,
        initial_state=ssm_final,
        initial_state_indices=cache_idx,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
    )
    return o, post_q, post_k, post_v, ga, gb, ssm_final


def _make_cp_group(world_size, rank):
    grp = types.SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        device_group=dist.group.WORLD,
    )

    def all_gather(t, dim=0):
        t = t.contiguous()
        out = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(out, t, group=dist.group.WORLD)
        return torch.cat(out, dim=dim)

    grp.all_gather = all_gather
    return grp


def _fe_worker(rank, world_size, cfg, ratio, ok_flag):
    import sglang.srt.layers.attention.linear.kda_backend as kb
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.layers.attention.fla.kda import chunk_kda
    from sglang.srt.layers.attention.linear.kda_cp_utils import (
        kda_cp_owner_of_global_token,
        prepare_kda_prefill_cp_metadata,
    )

    lengths = cfg["lengths"]
    prefix_lens = cfg.get("prefix_lens", [0] * len(lengths))
    sharded = cfg["sharded"]
    track = cfg.get("track", False)

    try:
        _init_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16
        N = len(lengths)
        T = sum(lengths)
        cp = world_size

        # Patch the two module-level hooks forward_extend consults.
        fake_group = _make_cp_group(cp, rank)
        kb.get_attention_cp_group = lambda: fake_group
        kb.is_kda_prefill_cp_plain_split = lambda: True

        layer = _make_layer(device, dtype)
        q_dim, k_dim, v_dim = layer.q_dim, layer.k_dim, layer.v_dim
        conv_dim = q_dim + k_dim + v_dim
        conv_state_len = _W - 1

        # ---- inputs: identical on every rank ----
        torch.manual_seed(4321)
        # Full sequence per request = prefix tokens ++ extend tokens. We compute
        # a "prefix" state by running the prefix tokens, then run the extend.
        pre_T = sum(prefix_lens)
        raw_full = torch.randn(T + pre_T, conv_dim, device=device, dtype=dtype)
        a_full = torch.randn(1, T + pre_T, q_dim, device=device, dtype=dtype)
        b_full = torch.randn(1, T + pre_T, _H, device=device, dtype=dtype)

        # Interleave: for each req, [prefix_i tokens | extend_i tokens].
        # Build index maps splitting the flat buffers into prefix / extend parts.
        pre_slices, ext_slices = [], []
        cur = 0
        for i in range(N):
            pre_slices.append((cur, cur + prefix_lens[i]))
            cur += prefix_lens[i]
            ext_slices.append((cur, cur + lengths[i]))
            cur += lengths[i]

        def gather_parts(buf, slices):
            return torch.cat([buf[..., s:e, :] if buf.dim() == 3 else buf[s:e]
                              for (s, e) in slices],
                             dim=-2 if buf.dim() == 3 else 0)

        raw_pre = gather_parts(raw_full, pre_slices)     # [pre_T, D]
        raw_ext = gather_parts(raw_full, ext_slices)     # [T, D]
        a_pre = gather_parts(a_full, pre_slices)
        a_ext = gather_parts(a_full, ext_slices)
        b_pre = gather_parts(b_full, pre_slices)
        b_ext = gather_parts(b_full, ext_slices)

        # ---- establish prefix state (conv + ssm) via full reference on prefix ----
        prefix_ssm = torch.zeros(N, _H, _DV, _DK, device=device, dtype=torch.float32)
        prefix_conv = torch.zeros(N, conv_dim, conv_state_len, device=device, dtype=torch.float32)
        if pre_T > 0:
            zero_conv = torch.zeros(N, conv_dim, conv_state_len, device=device, dtype=torch.float32)
            _, _, _, _, _, _, prefix_ssm = _reference(
                layer, raw_pre, a_pre, b_pre, prefix_lens,
                torch.zeros(N, _H, _DV, _DK, device=device, dtype=torch.float32),
                (zero_conv, [0] * N), device,
            )
            # prefix conv window = last W-1 raw prefix tokens per request.
            pcu = [0] + list(accumulate(prefix_lens))
            for i in range(N):
                s, e = pcu[i], pcu[i + 1]
                take = min(conv_state_len, e - s)
                if take > 0:
                    prefix_conv[i, :, conv_state_len - take:] = (
                        raw_pre[e - take:e].transpose(0, 1).float()
                    )

        # ---- reference over the EXTEND tokens, seeded by prefix state ----
        o_ref, post_q, post_k, post_v, ga, gb, ssm_final_ref = _reference(
            layer, raw_ext, a_ext, b_ext, lengths,
            prefix_ssm, (prefix_conv, prefix_lens), device,
        )
        # reference conv-final per request = last W-1 raw tokens of full seq.
        conv_final_ref = torch.zeros(N, conv_dim, conv_state_len, device=device, dtype=torch.float32)
        for i in range(N):
            ps, pe = pre_slices[i]
            es, ee = ext_slices[i]
            full_i = torch.cat([raw_full[ps:pe], raw_full[es:ee]], dim=0)  # [seq_i, D]
            take = min(conv_state_len, full_i.shape[0])
            conv_final_ref[i, :, conv_state_len - take:] = (
                full_i[-take:].transpose(0, 1).float()
            )

        # ---- persistent mamba cache (per-rank copy) ----
        num_slots = 4 * N + 4
        if sharded:
            conv_local = conv_dim // cp
            ssm_local = _H // cp
        else:
            conv_local = conv_dim
            ssm_local = _H
        persistent_conv = torch.zeros(num_slots, conv_local, conv_state_len, device=device, dtype=dtype)
        persistent_ssm = torch.zeros(num_slots, ssm_local, _DV, _DK, device=device, dtype=torch.float32)

        # Slots: request i final state -> slot i. Seed prefix state into them so
        # _gather_full_kda_state can reconstruct the (sharded) prefix.
        def shard(full, dim):
            if not sharded:
                return full
            loc = full.shape[dim] // cp
            return full.narrow(dim, rank * loc, loc).contiguous()

        for i in range(N):
            persistent_conv[i].copy_(shard(prefix_conv[i], 0).to(dtype))
            persistent_ssm[i].copy_(shard(prefix_ssm[i], 0))

        # ---- CP metadata + fake ForwardBatch / backend ----
        meta = prepare_kda_prefill_cp_metadata(
            total_tokens=T,
            extend_seq_lens_cpu=list(lengths),
            cp_rank=rank,
            cp_size=cp,
            device=device,
        )
        lo, hi = meta.local_start, meta.local_end

        cache_indices = torch.arange(N, device=device, dtype=torch.int32)
        fm = types.SimpleNamespace(
            query_start_loc=meta.local_query_start_loc,
            mamba_cache_indices=cache_indices,
            has_mamba_track_mask=False,
        )

        mamba_cache = types.SimpleNamespace(conv=[persistent_conv], temporal=persistent_ssm)

        track_mask = None
        track_indices = None
        track_seqlens = None
        if track:
            track_mask = torch.ones(N, dtype=torch.bool, device=device)
            # snapshot slots for request i -> slot 2*N + i (distinct from finals).
            track_indices = torch.arange(N, device=device, dtype=torch.int32) + 2 * N
            track_seqlens = torch.tensor(
                [prefix_lens[i] + lengths[i] for i in range(N)],
                device=device, dtype=torch.int32,
            )

        fb = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_target_verify=lambda: False,
                is_context_parallel_extend=lambda: True,
                is_extend=lambda: True,
            ),
            kda_cp_metadata=meta,
            extend_num_valid_tokens=hi - lo,
            extend_seq_lens_cpu=list(lengths),
            extend_prefix_lens=torch.tensor(prefix_lens, device=device),
            extend_prefix_lens_cpu=list(prefix_lens),
            mamba_track_mask=track_mask,
            mamba_track_indices=track_indices,
            mamba_track_seqlens=track_seqlens,
        )

        backend = KDAAttnBackend.__new__(KDAAttnBackend)
        backend.conv_states_shape = (num_slots, conv_local, conv_state_len)
        backend.forward_metadata = fm
        backend.req_to_token_pool = types.SimpleNamespace(
            mamba2_layer_cache=lambda lid: mamba_cache
        )
        backend.kernel_dispatcher = kb.KDAKernelDispatcher.__new__(kb.KDAKernelDispatcher)
        tri = TritonKDAKernel()
        backend.kernel_dispatcher.extend_kernel = tri
        backend.kernel_dispatcher.cp_kernel = tri
        backend.kernel_dispatcher.decode_kernel = tri
        backend.kernel_dispatcher.verify_kernel = tri

        # ---- rank-local inputs into forward_extend ----
        mixed_local = raw_ext[lo:hi].contiguous()           # [n_local, D]
        a_local = a_ext[:, lo:hi].contiguous()              # [1, n_local, q_dim]
        b_local = b_ext[:, lo:hi].contiguous()              # [1, n_local, H]

        o_local = backend.forward_extend(
            layer, fb, mixed_local, a_local, b_local
        )  # [1, n_local, H, DV]

        passed = True

        # (1) output slice
        if hi > lo:
            ol = o_local[:, : hi - lo].float()
            oref = o_ref[:, lo:hi].float()
            rel = (ol - oref).abs().mean() / oref.abs().mean().clamp_min(1e-6)
            if not torch.isfinite(rel) or rel.item() > ratio:
                print(f"[rank {rank}] FE OUTPUT rel {rel.item():.4e} > {ratio}")
                passed = False

        # (2) final SSM + conv state (owner writes; result is identical on all
        # ranks after the collective, so every rank can verify its shard).
        for i in range(N):
            ref_ssm = shard(ssm_final_ref[i], 0).float()
            got_ssm = persistent_ssm[i].float()
            rel = (got_ssm - ref_ssm).abs().mean() / ref_ssm.abs().mean().clamp_min(1e-6)
            if not torch.isfinite(rel) or rel.item() > ratio:
                print(f"[rank {rank}] FE FINAL-SSM req {i} rel {rel.item():.4e}")
                passed = False

            ref_conv = shard(conv_final_ref[i], 0).float()
            got_conv = persistent_conv[i].float()
            rel = (got_conv - ref_conv).abs().mean() / ref_conv.abs().mean().clamp_min(1e-6)
            if not torch.isfinite(rel) or rel.item() > ratio:
                print(f"[rank {rank}] FE FINAL-CONV req {i} rel {rel.item():.4e}")
                passed = False

        # (3) tracked snapshot at bc = req_start + (lens // 64) * 64
        if track:
            req_starts = [0] + list(accumulate(lengths))
            for i in range(N):
                lens = lengths[i]  # prefix already excluded (track_seqlens - prefix)
                cache_offset = lens if lens % _CHUNK == 0 else (lens // _CHUNK) * _CHUNK
                bc = req_starts[i] + cache_offset
                # reference snapshot state: chunk_kda over req i's first
                # cache_offset extend tokens, seeded by its prefix state.
                rs, re = req_starts[i], req_starts[i] + cache_offset
                snap_init = prefix_ssm[i : i + 1].clone()
                sub_cu = torch.tensor([0, cache_offset], device=device, dtype=torch.int32)
                chunk_kda(
                    q=post_q[:, rs:re],
                    k=post_k[:, rs:re],
                    v=post_v[:, rs:re],
                    g=ga[:, rs:re],
                    beta=gb[:, rs:re],
                    initial_state=snap_init,
                    initial_state_indices=torch.zeros(1, device=device, dtype=torch.int32),
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=sub_cu,
                )
                ref_snap = shard(snap_init[0], 0).float()
                slot = int(track_indices[i].item())
                got_snap = persistent_ssm[slot].float()
                rel = (got_snap - ref_snap).abs().mean() / ref_snap.abs().mean().clamp_min(1e-6)
                if not torch.isfinite(rel) or rel.item() > ratio:
                    print(f"[rank {rank}] FE TRACK-SSM req {i} bc {bc} rel {rel.item():.4e}")
                    passed = False

                # tracked conv window = last W-1 raw extend tokens before bc.
                take = min(conv_state_len, cache_offset)
                ref_win = torch.zeros(conv_dim, conv_state_len, device=device, dtype=torch.float32)
                if take > 0:
                    ref_win[:, conv_state_len - take:] = (
                        raw_ext[bc - take:bc].transpose(0, 1).float()
                    )
                ref_win = shard(ref_win, 0).float()
                got_win = persistent_conv[slot].float()
                rel = (got_win - ref_win).abs().mean() / ref_win.abs().mean().clamp_min(1e-6)
                if not torch.isfinite(rel) or rel.item() > ratio:
                    print(f"[rank {rank}] FE TRACK-CONV req {i} rel {rel.item():.4e}")
                    passed = False

        flag = torch.tensor([1 if passed else 0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok_flag.value = int(flag.item())
        dist.barrier()
    except Exception as e:  # surface the failing rank's traceback
        import traceback

        print(f"[rank {rank}] EXCEPTION: {e}")
        traceback.print_exc()
        ok_flag.value = 0
    finally:
        _cleanup()


def _run_fe(world_size, cfg, ratio=3e-2):
    ok = mp.Manager().Value("i", 0)
    mp.start_processes(
        _fe_worker,
        args=(world_size, cfg, ratio, ok),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    return ok.value == 1


@unittest.skipIf(torch.cuda.device_count() < 2, "needs >= 2 GPUs")
class TestKDAContextParallel(unittest.TestCase):
    """Kernel-only: chunk_kda_cp vs full-sequence chunk_kda."""

    def test_cp2_single_sequence(self):
        self.assertTrue(_run(2, [4096]))

    def test_cp2_boundary_aligned(self):
        self.assertTrue(_run(2, [2048, 2048]))

    def test_cp2_sequence_cut(self):
        self.assertTrue(_run(2, [1500, 2000, 596]))

    def test_cp4_single_long(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("needs >= 4 GPUs")
        self.assertTrue(_run(4, [8192]))

    def test_cp4_short_multihop(self):
        # Short single sequence at cp>=4 exercises MULTI-HOP merge (ranks >=2
        # from the request start). A long sequence dilutes the merge error below
        # tolerance, so this must be short. Pre-fix (pre_num_ranks==1) this was
        # ~0.24 rel err; post-fix ~0.006.
        if torch.cuda.device_count() < 4:
            self.skipTest("needs >= 4 GPUs")
        self.assertTrue(_run(4, [256], ratio=3e-2))

    def test_cp8_short_multihop(self):
        if torch.cuda.device_count() < 8:
            self.skipTest("needs >= 8 GPUs")
        self.assertTrue(_run(8, [256], ratio=3e-2))


@unittest.skipIf(torch.cuda.device_count() < 2, "needs >= 2 GPUs")
class TestKDABackendForwardExtend(unittest.TestCase):
    """Full KDAAttnBackend.forward_extend CP path (conv halo + merge + writeback)."""

    # cross-rank cut: req1 spans the rank boundary at T//2.
    _CUT = dict(lengths=[150, 200, 250], prefix_lens=[0, 0, 0])

    def test_cp2_no_prefix_full_state(self):
        self.assertTrue(_run_fe(2, {**self._CUT, "sharded": False}))

    def test_cp4_no_prefix_full_state(self):
        # cp>=4 multi-request through the full backend (conv halo + multi-hop
        # merge + writeback). Requests span multiple ranks -> exercises the
        # multi-hop path the cp2 tests miss.
        if torch.cuda.device_count() < 4:
            self.skipTest("needs >= 4 GPUs")
        self.assertTrue(_run_fe(4, {**self._CUT, "sharded": False}, ratio=4e-2))

    def test_cp8_no_prefix_full_state(self):
        if torch.cuda.device_count() < 8:
            self.skipTest("needs >= 8 GPUs")
        self.assertTrue(_run_fe(8, {**self._CUT, "sharded": False}, ratio=4e-2))

    def test_cp2_no_prefix_sharded(self):
        # reduce_scatter fast path (head/channel-sharded persistent cache).
        self.assertTrue(_run_fe(2, {**self._CUT, "sharded": True}))

    def test_cp2_prefix_full_state(self):
        # prefix cache: continuation_h0 + offset-0 seed + conv halo w/ history.
        cfg = dict(lengths=[150, 200, 250], prefix_lens=[128, 64, 256], sharded=False)
        self.assertTrue(_run_fe(2, cfg))

    def test_cp2_prefix_sharded(self):
        cfg = dict(lengths=[150, 200, 250], prefix_lens=[128, 64, 256], sharded=True)
        self.assertTrue(_run_fe(2, cfg))

    def test_cp2_notrack_short_reqs(self):
        # ISOLATION: same lengths as the track test but track=False. If OUTPUT
        # fails here too, the forward/merge is length-sensitive (track-independent).
        cfg = dict(lengths=[128, 200, 64], prefix_lens=[0, 0, 0], sharded=False, track=False)
        self.assertTrue(_run_fe(2, cfg))

    def test_cp2_track_aligned_lengths(self):
        # ISOLATION: track with lengths whose OUTPUT already passes (the
        # no_prefix config). Isolates the short-recurrence bug from any
        # length-sensitive forward-output bug.
        cfg = dict(lengths=[150, 200, 250], prefix_lens=[0, 0, 0], sharded=False, track=True)
        self.assertTrue(_run_fe(2, cfg))

    def test_cp2_track_extra_buffer(self):
        # extra_buffer radix-cache snapshot: req0/req2 chunk-aligned (snapshot ==
        # final), req1 crosses the rank boundary with a non-64-aligned start ->
        # exercises the short recurrence.
        cfg = dict(lengths=[128, 200, 64], prefix_lens=[0, 0, 0], sharded=False, track=True)
        self.assertTrue(_run_fe(2, cfg))

    def test_cp2_track_sharded(self):
        cfg = dict(lengths=[128, 200, 64], prefix_lens=[0, 0, 0], sharded=True, track=True)
        self.assertTrue(_run_fe(2, cfg))


if __name__ == "__main__":
    unittest.main()
