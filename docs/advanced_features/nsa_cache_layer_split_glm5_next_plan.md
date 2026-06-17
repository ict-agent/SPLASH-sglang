# NSA Cache Layer Split for GLM5 Next

This document is the compact operator/developer note for
`--enable-nsa-cache-layer-split` on
`v0.5.10-glm5-next`.

The feature is a prefill-only optimization for NSA prefill context parallelism.
It shards NSA GPU KV/indexer cache by full-attention layer across prefill CP
ranks. It is not a decode cache-sharding feature.

## quick start

### Required Shape

Use layer split only on PD prefill workers:

- Model must be an NSA/DSA model, such as GLM5 Next hybrid NSA + KDA.
- Prefill must use NSA prefill CP:
  `--enable-nsa-prefill-context-parallel`.
- CP mode must be round-robin split:
  `--nsa-prefill-cp-mode round-robin-split`.
- Enable the feature only on P:
  `--enable-nsa-cache-layer-split`.
- Decode workers must not enable layer split.
- Non-PD all-in-one workers must not rely on this feature; current code
  automatically disables the flag outside `--disaggregation-mode prefill`.

### P Node Args

For the validated 70B topology, P ran with TP8/EP8:

```bash
python3 -W ignore -m sglang.launch_server \
  --model-path /cloud/workspace/sqq/models/hf_70b_fp8_perblock \
  --tokenizer-path /cloud/workspace/sqq/models/hf_70b_fp8_perblock \
  --trust-remote-code \
  --max-running-requests 32 \
  --disable-shared-experts-fusion \
  --disable-piecewise-cuda-graph \
  --disable-chunked-prefix-cache \
  --skip-server-warmup \
  --mamba-scheduler-strategy extra_buffer \
  --mem-fraction-static 0.80 \
  --context-length 1104096 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --page-size 64 \
  --host 0.0.0.0 \
  --port 48000 \
  --dist-init-addr 127.0.0.1:48100 \
  --watchdog-timeout 3600 \
  --dist-timeout 3600 \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_sparse \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --enable-nsa-cache-layer-split \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 48050 \
  --engine-info-bootstrap-port 48090 \
  --enable-request-time-stats-logging
```

Important P-side notes:

- Keep `--mamba-scheduler-strategy extra_buffer` when radix cache is enabled.
- `--disable-cuda-graph` was used for these validations.

### D Node Args

For the validated 70B topology, D ran with TP8/EP8:

```bash
python3 -W ignore -m sglang.launch_server \
  --model-path /cloud/workspace/sqq/models/hf_70b_fp8_perblock \
  --tokenizer-path /cloud/workspace/sqq/models/hf_70b_fp8_perblock \
  --trust-remote-code \
  --max-running-requests 32 \
  --disable-shared-experts-fusion \
  --disable-piecewise-cuda-graph \
  --disable-chunked-prefix-cache \
  --disable-radix-cache \
  --skip-server-warmup \
  --mamba-scheduler-strategy no_buffer \
  --mem-fraction-static 0.80 \
  --context-length 1104096 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --page-size 64 \
  --host 0.0.0.0 \
  --port 48001 \
  --dist-init-addr 127.0.0.1:48110 \
  --watchdog-timeout 3600 \
  --dist-timeout 3600 \
  --attention-backend nsa \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_sparse \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --disable-cuda-graph \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake
```

Important D-side notes:

- Do not pass `--enable-nsa-cache-layer-split`.
- Keep `--disable-radix-cache --mamba-scheduler-strategy no_buffer` for this
  PD validation shape.

### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.1.51.13:48000 48050 \
  --decode http://10.1.51.68:48001 \
  --host 0.0.0.0 \
  --port 48202 \
  --prometheus-port 48390
```

If the router exits with `FailedToCreateHTTPListener("Address already in use")`,
pick another `--prometheus-port`. The router HTTP port can stay unchanged if it
was not bound.

## test

Run GSM8K:

```bash
python3 -m sglang.test.few_shot_gsm8k \
  --num-questions 200 \
  --host 127.0.0.1 \
  --port 48202 \
  --parallel 8 \
  --temperature 0
```

## implement details

### Flag And Guard

Files:

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

Implementation:

- Added `--enable-nsa-cache-layer-split`.
- The flag is effective only when all are true:
  - model is NSA;
  - worker is PD prefill: `--disaggregation-mode prefill`;
  - `--enable-nsa-prefill-context-parallel` is enabled;
  - `--nsa-prefill-cp-mode round-robin-split`.
- Decode and non-PD workers auto-disable the flag to preserve ordinary local
  decode cache semantics.
- KV capacity estimation uses the number of owned full-attention layers plus
  one remote scratch layer.

### GPU KV And Indexer Layer Sharding

Files:

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `python/sglang/srt/layers/attention/nsa/kpool/indexer.py`

Implementation:

- Full-attention layers are assigned to CP ranks by contiguous ranges.
- On each P CP rank, only owned NSA KV/indexer layers allocate full storage.
- Non-owned layers allocate no local per-layer storage.
- A single remote scratch KV buffer is used to read one non-owned layer at a
  time through CP broadcast.
- NSA indexer cache has the same owned-layer allocation pattern and a remote
  scratch indexer buffer.
- Local writes are skipped for non-owned layers.
- Remote scratch state is invalidated when local writes could make it stale.
- Host-side cache metadata mirrors the device sharding so HiCache/host cache and
  PD transfer enumerate only owned layer buffers.

For GLM5 Next hybrid models, the split is scoped to the NSA full-attention
cache path under `HybridLinearKVPool.full_kv_pool`. KDA/mamba state keeps the
existing GLM5 Next behavior and is not layer-sharded by this feature.

### PD/Mooncake Compatibility

Files:

- `python/sglang/srt/disaggregation/common/conn.py`
- `python/sglang/srt/disaggregation/mooncake/conn.py`
- `python/sglang/srt/disaggregation/prefill.py`

Implementation:

- Layer split does not add a separate bootstrap metadata field. P-side transfer
  participation is derived from `enable_all_cp_ranks_for_transfer`; with GLM5
  hybrid CP this is already true, so every prefill CP rank participates because
  each CP rank owns different layers.
- Decode receives all CP-rank layer shards and keeps an ordinary unsplit local
  decode cache.
- Existing all-CP transfer behavior for hybrid MLA/KDA state is preserved.

### Broadcast Overlap

Files:

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/communicator_nsa_cp.py`
- `python/sglang/srt/layers/communicator_mhc_hybrid_cp.py`
- `python/sglang/srt/models/glm5_next.py`

Implementation:

- `Glm5NextModel` records the next full-attention layer id.
- After a DSA/full-attention layer finishes, the model calls the layer
  communicator hook `maybe_prefetch_next_full_attention_kv(...)`.
- NSA CP layer communicators perform the actual
  `prefetch_mla_kv_buffer(next_full_attention_layer_id)` call; the base
  communicator keeps the hook as a no-op.
- `MLATokenToKVPool.prefetch_kv_buffer()` broadcasts the next remote KV layer on
  a separate CUDA stream, `kv_broadcast_stream`.
- The async KV broadcast uses a dedicated layer-shard NCCL communicator created
  from the attention CP CPU group. This avoids sharing the attention CP
  communicator with forward-path all-gather/reduce-scatter collectives.
- The forward stream waits only when the remote layer is actually consumed or
  when a pending broadcast must be finalized.
- This overlaps remote KV broadcast with the intervening KDA layers.
- NSA indexer cache broadcast is still synchronous in this implementation; the
  overlap optimization targets the MLA/NSA KV buffer.

## results

### Latest Rebased 70B P8+D8 Sweep

GSM8K 200-question results:

| Client parallel | Accuracy | Invalid | Latency | Output throughput |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.820 | 0.000 | 275.918 s | 70.851 token/s |

### Cache Allocation Signals

Latest 70B P8+D8 runs:

- P with layer split:
  - `enable_nsa_cache_layer_split=True`
  - `attn_cp_size=8`
  - `#tokens=7632896`
  - ranks 0/1/2: `KV size: 18.26 GB`
  - ranks 3/4/5/6/7: `KV size: 9.13 GB`
- D without layer split:
  - `enable_nsa_cache_layer_split=False`
  - `attn_cp_size=1`
  - `#tokens=3996736`
  - each rank: `KV size: 52.57 GB`

Historical capacity sanity check with the same model/runtime args before the
prefill-only guard was tightened:

| Mode | Tokens | Per-rank KV allocation |
| --- | ---: | --- |
| No layer split | 531072 | 6.29 GB on every CP rank |
| Layer split | 1947328 | 4.19 GB on ranks 0/1/2; 2.10 GB on ranks 3-7 |

This showed about `3.67x` higher profiled token capacity and reduced summed
logged KV allocation from about `50.32 GB` to `23.07 GB`. The non-PD runtime
shape is no longer the supported way to use the flag; the numbers are kept only
as a memory-allocation sanity reference.

### Broadcast-Overlap Check

Dual-node P8+D8, GSM8K 200, `parallel=8`:

| Build | Accuracy | Invalid | Latency | Output throughput |
| --- | ---: | ---: | ---: | ---: |
| Layer split before broadcast overlap | 0.820 | 0.000 | 287.354 s | 69.983 token/s |
| Layer split with broadcast overlap | 0.830 | 0.000 | 270.888 s | 70.690 token/s |

The overlap change preserved correctness and gave a small throughput/latency
improvement in this GSM8K run. The bigger throughput gains in the latest sweep
come from raising client parallelism.

Long-prefix serving benchmark script:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 47430 \
  --model /cloud/workspace/sqq/models/dsa_300b_fp8_perblock_quant/ \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 55296 \
  --gsp-question-len 10240 \
  --gsp-output-len 1 \
  --num-prompts 8 \
  --request-rate inf \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --extra-request-body '{"bootstrap_host":"2.2.2.2","bootstrap_room":0}'
```

Long-prefix serving benchmark result:

| Metric | Before overlap | After overlap | Change |
| --- | ---: | ---: | ---: |
| Successful requests | 8 | 8 | - |
| Benchmark duration | 4.52 s | 3.92 s | -13.3% |
| Total input tokens | 545700 | 545461 | -0.04% |
| Request throughput | 1.77 req/s | 2.04 req/s | +15.3% |
| Input token throughput | 120830.32 tok/s | 139325.46 tok/s | +15.3% |
| Total token throughput | 120832.09 tok/s | 139327.50 tok/s | +15.3% |
| Mean E2E latency | 563.21 ms | 488.18 ms | -13.3% |
| Median E2E latency | 498.59 ms | 430.00 ms | -13.8% |
| Mean TTFT | 563.20 ms | 488.17 ms | -13.3% |
| Median TTFT | 498.58 ms | 429.99 ms | -13.8% |
| P99 TTFT | 992.68 ms | 889.04 ms | -10.4% |

This benchmark is output-length 1, so TTFT and E2E latency are effectively the
same signal. With comparable total input tokens, broadcast overlap improved
request/input-token throughput by about 15% and reduced mean TTFT/E2E latency by
about 13%.
