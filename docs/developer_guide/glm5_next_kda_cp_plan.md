# GLM5-Next KDA Prefill Context Parallel Plan

## Background

The current H100 prefill launch for GLM5-Next uses NSA/DSA context
parallelism and KDA tensor/head parallelism:

```bash
python -u -m sglang.launch_server \
  --model-path /cloud/workspace/sqq/models/dsa_300b_fp8_perblock_quant \
  --tokenizer-path /cloud/workspace/sqq/models/dsa_300b_fp8_perblock_quant \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_sparse \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --linear-attn-prefill-backend flash_kda
```

The target change is to let KDA also use context parallelism during prefill.
Unlike DSA, KDA is recurrent and needs natural sequence order for each local
token segment. Therefore KDA should not use DSA's round-robin token layout.

This plan adopts the following key decision:

**When KDA prefill CP is enabled, KDA attention weights are not sharded. Each CP
rank stores a replicated copy of the KDA attention weights and computes full
KDA attention activations for its current token slice.** This avoids activation
communication for current tokens, especially q/k/v/gate/beta and post-attn
hidden states.

Persistent KDA recurrent states in Prefix Cache still use the TP-sharded state
ABI. Runtime KDA-CP compute may temporarily materialize full-head states; cache
read/write converts between full runtime state and TP-sharded persistent state.

## Goals

1. Enable KDA prefill CP for GLM5-Next.
2. Keep DSA prefill CP on `round-robin-split`.
3. Add KDA `plain-split`: current packed chunk tokens are split into contiguous
   token ranges by CP rank.
4. Replicate KDA attention weights under KDA-CP to avoid per-token activation
   all-to-all/all-gather.
5. Store Prefix Cache KDA/memba states in TP-sharded form.
6. Preserve correctness for prefix-cache hit, branching prefix, chunked prefill,
   and P/D handoff state generation.

## Non-Goals

1. Do not rewrite DSA/NSA CP kernels.
2. Do not change MoE/MLP parallelism except where layer-boundary layout requires
   existing communication.
3. Do not define D-side execution behavior in this plan. The target business
   deployment uses P/D disaggregation, so this document only specifies the P-side
   prefill path and the state handoff ABI.
4. Do not change Prefix Cache node matching semantics except the KDA state
   storage/read/write path.
5. Do not support KDA-CP decode execution in the P worker or non-disaggregated
   server mode.

## Terminology

- **Global packed chunk**: the token tensor built for the current prefill
  forward, after scheduler packing.
- **DSA scattered layout**: DSA CP layout, currently `round-robin-split`, where
  rank `r` owns tokens whose global index is congruent to `r mod cp_size`.
- **KDA plain layout**: rank `r` owns one contiguous slice of the current packed
  chunk.
- **Runtime full KDA state**: temporary KDA recurrent state containing all KDA
  heads, used by replicated KDA attention compute on a CP rank.
- **Persistent TP-sharded KDA state**: Prefix Cache/mamba pool state sharded by
  TP/head dimension and stored in the current state pool ABI.

## High-Level Data Contract

During prefill:

1. Model-level hidden states stay in KDA plain layout across layers.
2. KDA layers consume and produce local plain token slices.
3. DSA/MLA layers convert plain -> DSA scattered before attention and convert
   DSA scattered -> plain after attention.
4. KDA attention weights are replicated on every CP rank.
5. KDA current-token activations are not communicated across CP ranks.
6. KDA recurrent state communication is allowed only at state boundaries:
   prefix-cache read, inter-rank recurrent state passing, and prefix-cache write.

At the P/D boundary:

1. The P side emits the same logical KV/cache and KDA/memba state artifacts that
   downstream workers expect today.
2. KDA/memba states are handed off in TP-sharded persistent form.
3. D-side scheduling, weight layout, and compute strategy are intentionally out
   of scope for this plan.

## Current H100 Implementation Status

The first KDA-CP implementation has been landed on the H100 workspace under
`/cloud/workspace/sqq/sglang`:

1. Server flags and `plain-split` metadata are added.
2. `ForwardBatch.kda_cp_metadata` is generated from the current packed prefill
   chunk when NSA/DSA CP is active.
3. KDA attention weights are replicated under KDA-CP (`tp_size=1` for KDA
   attention modules and full KDA attention tensor loading in `load_mgt.py`).
4. KDA projection paths no longer all-gather current-token hidden states when
   KDA-CP is active.
5. KDA output projection no longer reduce-scatters current-token activations
   when KDA-CP is active.
6. KDA backend consumes local `plain-split` varlen metadata:
   `local_query_start_loc`, `local_seq_lens_cpu`, and local segment cache
   indices.
7. Runtime KDA conv/SSM state is materialized as full-head temporary state by
   gathering the persistent TP-sharded state across the attention CP group.
8. Inter-rank KDA recurrent dependency is handled by a correctness-first
   neighbor state pass: the previous owner rank sends the full conv/SSM boundary
   state to the next owner rank. This replaces full current-token activation
   communication with state-boundary communication.
9. Segment-end Prefix Cache writeback broadcasts the owner rank's full runtime
   state and writes only the local TP shard into the persistent cache.
10. KDA-CP is guarded to P-side prefill mode (`--disaggregation-mode prefill`);
    decode and non-disaggregated server modes are intentionally unsupported.

The current writeback covers the common chunk-end/request-boundary tracking
case. Branching or forced intermediate tracking points inside a local segment
still need the same `h`-index extraction that the non-CP path uses in
`_init_track_ssm_indices`; this is the main remaining correctness item before
claiming full Prefix Cache parity for all branching cases.

## H100 Validation Results (2026-06-05)

Completed on `/cloud/workspace/sqq/sglang` with the H100 runtime:

1. `py_compile` passed for the edited KDA-CP files.
2. CLI help exposes `--enable-kda-prefill-context-parallel` and
   `--kda-prefill-cp-mode {plain-split}`.
3. `git diff --check` passed.
4. `test/registered/attention/test_kda_prefill_cp.py` passed: 8 tests.
5. `test/registered/attention/test_kda_kernels.py` passed on GPU 0.
6. `test/registered/attention/test_kda_flash_vs_triton.py` passed on GPU 0:
   8 tests.
7. 8-GPU dummy GLM5-Next KDA-CP launch passed with:
   `--enable-nsa-prefill-context-parallel`,
   `--nsa-prefill-cp-mode round-robin-split`,
   `--enable-kda-prefill-context-parallel`, and
   `--kda-prefill-cp-mode plain-split`.
8. P-side launch smoke passed: `/health` and `/model_info`.
9. 32K no-prefix prefill passed with a 32,700-token `input_ids` prompt:
   `prompt_tokens=32700` and `cached_tokens=0`.
10. Prefix Cache hit passed with text prompts after `/flush_cache`:
    seed request had `cached_tokens=0`, exact-prefix reuse had
    `cached_tokens=8192`, and branching-prefix reuse had `cached_tokens=8192`.
    An `input_ids` branching-prefix smoke also hit `cached_tokens=10752`.
11. 128K no-prefix prefill passed under a 1M-context launch:
    `prompt_tokens=131072` and `cached_tokens=0`.

## Launch Interface

Add server args:

```python
enable_kda_prefill_context_parallel: bool = False
kda_prefill_cp_mode: str = "plain-split"
```

Add choices:

```python
KDA_PREFILL_CP_SPLIT_CHOICES = ["plain-split"]
```

Expected launch:

```bash
python -u -m sglang.launch_server \
  --model-path /cloud/workspace/sqq/models/dsa_300b_fp8_perblock_quant \
  --tokenizer-path /cloud/workspace/sqq/models/dsa_300b_fp8_perblock_quant \
  --trust-remote-code \
  --tp-size 8 \
  --disaggregation-mode prefill \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_sparse \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --enable-kda-prefill-context-parallel \
  --kda-prefill-cp-mode plain-split \
  --linear-attn-prefill-backend flash_kda
```

Validation rules:

1. `--enable-kda-prefill-context-parallel` initially supports only GLM5-Next
   KDA layers.
2. `--kda-prefill-cp-mode` initially only accepts `plain-split`.
3. `--enable-kda-prefill-context-parallel` requires
   `--disaggregation-mode prefill`; decode and non-disaggregated server modes
   are out of scope.
4. If KDA-CP is enabled, KDA attention weight sharding must be disabled.
5. If Prefix Cache with KDA states is enabled, state storage mode must remain
   TP-sharded.
6. Warn that replicated KDA attention weights increase memory use.

## Phase 1: KDA CP Metadata

Add `python/sglang/srt/layers/attention/linear/kda_cp_utils.py`.

Suggested metadata:

```python
@dataclass
class KDAContextParallelMetadata:
    total_tokens: int
    cp_rank: int
    cp_size: int

    local_start: int
    local_end: int
    local_num_tokens: int
    local_padding: int

    # Varlen metadata for the local packed slice.
    local_seq_lens_cpu: list[int]
    local_query_start_loc: torch.Tensor
    local_req_indices_cpu: list[int]

    # For each local segment, offset inside the original request's extend chunk.
    local_req_extend_offsets_cpu: list[int]

    # For recurrent/cache bookkeeping.
    local_segment_global_starts_cpu: list[int]
    local_segment_global_ends_cpu: list[int]
```

Functions:

```python
def can_kda_cp_split(forward_batch, cp_size: int) -> bool
def prepare_kda_cp_metadata(forward_batch, total_tokens: int, cp_rank: int, cp_size: int)
def kda_cp_plain_split_tensor(x: torch.Tensor, meta: KDAContextParallelMetadata)
def kda_cp_build_local_query_start_loc(meta, device) -> torch.Tensor
def kda_cp_owner_of_global_token(global_token_idx: int, total_tokens: int, cp_size: int) -> int
```

Important details:

1. Split by the current packed chunk, not by full request sequence length.
2. A local rank may contain partial segments from multiple requests.
3. A local rank may contain a suffix of one request and a prefix of another.
4. `query_start_loc` passed to KDA kernels must describe only local real tokens,
   excluding any halo tokens.
5. Padding for equal communication shapes must not be visible to KDA recurrence.

## Phase 2: Model-Level Layout

Current `Glm5NextModel.forward` already has a plain hidden-state path for NSA CP.
Formalize it as the cross-layer contract whenever DSA CP or KDA CP is active.

At model entry:

1. Build `forward_batch.kda_cp_metadata` when KDA-CP is enabled.
2. Split `hidden_states` using KDA plain split.
3. Keep `hidden_states` in plain layout across the layer stack.
4. Do not globally rewrite `positions` into one fixed CP layout. DSA and KDA
   consume different layouts; positions should be converted only where DSA needs
   them.

At DSA/MLA layer boundary:

1. Before DSA attention: `plain -> round-robin-split`.
2. After DSA attention: `round-robin-split -> plain`.
3. Keep residual in the same layout as `hidden_states`.

At KDA layer boundary:

1. No hidden-state layout conversion.
2. KDA input is the local plain token slice.
3. KDA output is the local plain token slice.

## Phase 3: KDA Weight Replication

When KDA-CP is enabled, KDA attention weights must be replicated instead of
sharded.

Affected GLM5-Next KDA attention parameters:

1. `qkv_proj.weight` or fused `fused_qkvbfg_a_proj.weight`
2. `b_proj.weight` when non-fused
3. `f_a_proj.weight`
4. `f_b_proj.weight` or fused `fused_fg_b_proj.weight`
5. `g_a_proj.weight`
6. `g_b_proj.weight` or fused `fused_fg_b_proj.weight`
7. `qkv_conv1d.weight`
8. `A_log`
9. `dt_bias`
10. `o_norm.weight`
11. `o_proj.weight`

Implementation direction in `Glm5NextLinearAttention.__init__`:

```python
if enable_kda_prefill_context_parallel:
    kda_weight_tp_size = 1
    kda_weight_tp_rank = 0
else:
    # Existing behavior.
    if is_nsa_enable_prefill_cp():
        kda_weight_tp_size = get_attention_cp_size()
        kda_weight_tp_rank = get_attention_cp_rank()
    else:
        kda_weight_tp_size = get_attention_tp_size()
        kda_weight_tp_rank = get_attention_tp_rank()
```

With KDA-CP:

1. `self.local_num_heads == self.num_heads`.
2. All KDA projections instantiate with `tp_size=1`.
3. `o_proj` instantiates with `tp_size=1`.
4. KDA attention output is full hidden size for local tokens.
5. No KDA attention output all-reduce or reduce-scatter is needed.

Weight loading changes in `load_mgt.py`:

1. If KDA-CP is enabled, set `kda_shard_size = 1` and `kda_shard_rank = 0`.
2. Load full KDA attention tensors on every rank.
3. Do not shard `A_log` or `dt_bias`.
4. Do not shard `qkv_conv1d`.
5. Keep non-KDA DSA/MoE/MLP weight loading unchanged.

This is the main change from earlier head-sharded KDA designs. The extra weight
memory is intentional because it removes current-token activation communication
from the KDA prefill path.

## Phase 4: KDA Forward Path

Current KDA CP-like code gathers hidden states, computes local head shards, then
scatters/reduces output. That path should be bypassed for KDA-CP.

New KDA-CP forward behavior:

1. Input `hidden_states`: local plain token slice `[local_tokens, hidden]`.
2. KDA projections use replicated full weights and produce full-head local
   `q/k/v/beta/gate`.
3. Causal conv runs on local tokens plus required halo.
4. FlashKDA/Triton KDA runs on local real tokens.
5. `o_norm` and `o_proj` run locally with replicated full weights.
6. Return `[local_tokens, hidden]`.

Communication explicitly not used for current-token activations:

1. No q/k/v all-gather.
2. No head all-to-all.
3. No KDA output reduce-scatter.
4. No KDA output all-reduce.

Communication still allowed:

1. Small conv halo exchange.
2. Recurrent state passing/prefix scan.
3. Prefix Cache state read/write conversion between full runtime state and
   TP-sharded persistent state.

## Phase 5: Conv Halo

KDA has a short causal conv before recurrence. With plain token CP, rank `r`
needs the last `conv_kernel - 1` mixed-qkv rows preceding its local first token
when the local segment does not start at the beginning of the request's current
extend chunk.

Add helper:

```python
def kda_cp_exchange_conv_halo(mixed_qkv, meta, conv_kernel_size):
    ...
```

Requirements:

1. Halo rows are used only for causal conv state.
2. Halo rows are not included in `query_start_loc`.
3. Halo rows are not returned in KDA output.
4. If a local segment starts exactly at a prefix-cache boundary, use the conv
   state loaded from Prefix Cache instead of remote halo for tokens before the
   current chunk.
5. If a local segment starts inside the current chunk, get missing previous
   rows from the owning previous CP rank.

Initial implementation passes the final conv window together with the recurrent
SSM state between neighboring CP owners. This avoids even the small tail
all-gather and keeps the KDA boundary dependency in one state message.

## Phase 6: Recurrent State Passing

KDA recurrence creates an inter-rank dependency across plain token chunks:

```text
prefix state -> rank0 tokens -> rank1 tokens -> ... -> rankN tokens
```

Because each CP rank computes full heads with replicated weights, the recurrent
state passed between ranks is a full-head runtime state.

Implementation decision point:

1. Validate whether `flash_kda.fwd(..., enable_cp=True)` already performs the
   required CP state passing/prefix scan for plain contiguous token chunks.
2. If yes, wire KDA-CP metadata into FlashKDA and verify correctness.
3. If no, add an explicit state-passing protocol.

Fallback explicit protocol:

1. Gather TP-sharded prefix-cache state into full runtime initial state.
2. Each rank computes local summary/final state for its token slice.
3. Perform CP prefix state passing so rank `r` receives the state after ranks
   `< r`.
4. Re-run or continue KDA local computation with the correct initial state.
5. Produce local output and local final full state.

The exact fallback depends on whether the KDA kernel exposes composable block
summaries. If it does not, the correctness-first fallback is a two-pass local
recompute.

## Phase 7: Prefix Cache State ABI

Persistent Prefix Cache state remains TP-sharded even though KDA-CP runtime
uses full-head state.

State read path:

1. Prefix Cache match returns a TP-sharded mamba/memba state slot.
2. Each rank reads its local TP shard from `MambaPool`.
3. KDA-CP all-gathers TP shards into a temporary full runtime state.
4. FlashKDA/Triton KDA consumes the full runtime state.

State write path:

1. The CP rank that owns a track boundary produces the full runtime state for
   that boundary.
2. Split full state by TP/head dimension.
3. Send each shard to the corresponding TP rank.
4. Each TP rank writes only its local shard into the Prefix Cache state slot.

This keeps the Prefix Cache and P/D handoff ABI TP-sharded and avoids storing
duplicated full states in the cache.

Affected areas:

1. `Glm5NextConfig.mamba2_cache_params`: use TP size for persistent state shape
   when KDA-CP is enabled.
2. `MambaPool` state layout: keep `[num_layers, pool_size, tp_shard_dim, ...]`.
3. `MambaRadixCache` and `HiMambaRadixCache`: keep node `mamba_value` semantics,
   but route KDA-CP read/write through gather/scatter helpers.
4. PD transfer path: preserve `get_state_dim_per_tensor()` meaning, where the
   sliceable dimension is TP/head state dimension.

Add helper APIs:

```python
def kda_cp_gather_tp_sharded_state(layer_cache, cache_indices) -> FullKDAState
def kda_cp_scatter_full_state_to_tp_cache(full_state, layer_cache, dst_indices, owner_mask)
```

## Phase 8: Mamba Tracking Bookkeeping

Existing tracking uses global sequence lengths, `mamba_track_indices`, and
`mamba_track_seqlens`. Keep global sequence-length semantics.

Needed changes:

1. Track boundary ownership must be computed from the current chunk plain split.
2. Only the CP rank owning a boundary computes the full runtime state for that
   boundary.
3. All TP ranks participate in storing their TP shard after the owner rank
   scatters the state.
4. Branching prefix handling must use the same ownership logic.
5. Ping-pong extra buffer indices still identify persistent cache slots, not
   temporary full runtime state buffers.

Key cases:

1. Boundary lands inside a rank's local token slice.
2. Boundary lands exactly at a CP split boundary.
3. Boundary lands before the current chunk due to prefix-cache hit.
4. Boundary lands after the local rank's token slice.
5. Multiple requests packed into one prefill chunk.

## Phase 9: P/D Handoff Contract

This plan assumes the production path uses P/D disaggregation. The current scope
ends at the P-side handoff boundary.

P-side responsibilities:

1. Produce final hidden/logit outputs for prefill requests in the same external
   format expected by the scheduler.
2. Materialize KDA/memba recurrent states at tracked prefix-cache boundaries.
3. Convert full runtime KDA states back into TP-sharded persistent states before
   they enter Prefix Cache, host cache, storage, or PD transfer.
4. Preserve existing KV/cache metadata semantics for downstream workers.
5. Add validation/logging that KDA-CP state handoff used TP-sharded persistent
   state, not replicated full runtime state.

Out of scope:

1. D-side KDA compute strategy.
2. D-side KDA weight layout.
3. D-side state gather/scatter optimization.
4. D-side performance or CUDA graph behavior.
5. Decode execution in the P worker.

## Phase 10: Tests

Unit tests:

1. `plain-split` metadata for one request.
2. `plain-split` metadata for multiple packed requests.
3. Non-divisible token count.
4. Local segment crossing request boundary.
5. DSA layout roundtrip: plain -> round-robin -> plain.
6. Prefix-cache state gather/scatter roundtrip: TP-sharded -> full -> TP-sharded.
7. Boundary owner calculation for mamba tracking.

Kernel/runtime tests:

1. KDA no-prefix prefill: baseline TP vs KDA-CP logits.
2. KDA prefix-cache hit: baseline TP vs KDA-CP logits.
3. KDA branching prefix: radix split and reuse.
4. Chunked prefill with chunk sizes 32K and 64K.
5. Mixed DSA/KDA layer stack with GLM5-Next layer pattern.
6. FlashKDA backend and Triton fallback backend.

H100 end-to-end tests:

1. 32K prompt, no prefix cache hit.
2. 128K prompt, no prefix cache hit.
3. 1M prompt, no prefix cache hit.
4. Shared-prefix second request, device prefix-cache hit.
5. Branching shared prefix.
6. P/D handoff state metadata and TP-sharded state payload validation.

Performance counters:

1. Confirm no KDA current-token activation all-to-all/all-gather.
2. Measure KDA recurrent state communication volume.
3. Measure extra weight memory.
4. Compare TTFT against current DSA-CP/KDA-TP baseline.
5. Compare Prefix Cache hit latency.

## Implementation Order

### PR 1: Flags, Metadata, and Documentation

1. Add KDA-CP server args.
2. Add `KDAContextParallelMetadata`.
3. Add plain split helpers.
4. Add tests for metadata and layout.
5. No behavior change by default.

### PR 2: Layer Layout Contract

1. Keep GLM5-Next cross-layer hidden states in plain layout when KDA-CP is on.
2. Convert plain -> DSA scattered -> plain only around DSA attention.
3. Make positions conversion DSA-local.
4. Add layout roundtrip tests.

### PR 3: Replicated KDA Attention Weights

1. Change GLM5-Next KDA attention module init under KDA-CP to `tp_size=1`.
2. Change MGT weight loading under KDA-CP to load full KDA attention weights on
   every rank.
3. Ensure fused and non-fused KDA projection paths both work.
4. Add shape assertions.

### PR 4: KDA-CP Forward Without Prefix Cache

1. Run KDA prefill on local plain tokens with replicated weights.
2. Add conv halo.
3. Wire FlashKDA `enable_cp=True` if it already supports state passing.
4. Verify no-prefix prefill accuracy against baseline.

### PR 5: Prefix Cache TP-Sharded State

1. Keep persistent KDA state shape TP-sharded.
2. Add state gather on cache read.
3. Add full-state scatter on cache write.
4. Update mamba tracking ownership.
5. Verify prefix hit and branching prefix.

### PR 6: P/D Handoff and Integration

1. Validate that P-side state handoff remains TP-sharded.
2. Verify Prefix Cache, host cache, storage, and PD transfer state metadata.
3. Add H100 prefill launch script variant.
4. Add guardrails preventing full runtime KDA states from being persisted.

### PR 7: Performance and Cleanup

1. Replace all-gather halo with neighbor exchange if needed.
2. Optimize state gather/scatter.
3. Add logging for KDA-CP mode, weight replication, and state storage mode.
4. Add guardrails for unsupported configurations.

## Risks and Open Questions

1. **FlashKDA CP semantics**: must verify whether `enable_cp=True` implements
   the exact plain-split recurrent state passing needed here.
2. **Memory pressure**: KDA attention weights are replicated. This is expected,
   but H100 memory headroom must be measured with the target 300B checkpoint.
3. **P/D handoff contract**: the P side must not leak replicated full runtime
   KDA states into persistent cache or transfer paths.
4. **Prefix Cache state conversion**: TP-sharded persistent state plus full-head
   runtime state introduces new gather/scatter points. Bugs here may only show
   up on prefix-cache hits or branching prefixes.
5. **Multi-request chunk packing**: plain split by packed chunk is required, but
   local varlen metadata must still preserve request boundaries.
6. **MHC/layer communicator interaction**: residual and hidden-state layouts
   must stay aligned when DSA layers temporarily convert to scattered layout.

## Acceptance Criteria

1. KDA-CP P-side prefill launch starts with GLM5-Next on H100.
2. KDA attention weights are full-size and replicated when KDA-CP is enabled.
3. KDA prefill does not communicate q/k/v/gate/beta/current-token hidden
   activations across ranks.
4. Prefix Cache KDA states remain TP-sharded.
5. Baseline TP and KDA-CP logits match within expected numeric tolerance.
6. Prefix-cache hit and branching-prefix tests pass.
7. 1M context prefill runs without state/cache corruption.
8. P/D handoff artifacts keep TP-sharded KDA/memba state layout.
