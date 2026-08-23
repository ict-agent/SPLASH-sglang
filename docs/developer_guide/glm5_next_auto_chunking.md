# GLM5-Next Auto Chunking

GLM5-Next auto chunking runs only on a PD-disaggregated prefill server. It
reduces the next chunk of an in-flight long prefill request and uses the
released, page-aligned token budget to finish short requests from the FCFS
waiting queue. This lowers head-of-line blocking without letting the long
request stop making progress.

The implementation is inspired by the adaptive chunking work on
`v0.5.10-prerelease`, but is intentionally limited to the following model
architectures in sglang-dcu:

- `Glm5NextForCausalLM`
- `Glm5NextForConditionalGeneration`

## Usage

Enable chunked prefill and short-request reservation together:

```bash
python3 -m sglang.launch_server \
  --model-path <glm5-next-model> \
  --disaggregation-mode prefill \
  --chunked-prefill-size 16384 \
  --prefill-short-req-reserve
```

The feature is off by default. Its tuning arguments are:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--prefill-short-req-threshold` | 16384 | Maximum existing-prefix-adjusted input length eligible as a short request. |
| `--prefill-short-req-max-reserve-ratio` | 0.5 | Maximum fraction of one chunk reserved for short requests. |
| `--prefill-short-req-scan-depth` | 8 | Number of requests inspected at the FCFS queue head. |
| `--prefill-long-req-starve-threshold` | 8 | Compressed rounds before the long request receives a full-budget round. |
| `--prefill-short-req-match-prefix` | off | Use a read-only radix-cache probe to estimate the reusable device prefix of scanned requests. |
| `--prefill-short-req-max-total-len` | 262144 | Requests longer than this are skipped; `0` disables the limit. |

## Admission and safety rules

By default, the planner uses only prefix information already stored on each
request. With `--prefill-short-req-match-prefix`, it additionally runs a
length-only radix-cache probe. Unlike the authoritative match performed during
admission, the probe does not split tree nodes, refresh LRU state, allocate a
Mamba slot, or mutate the request. Admission still performs a fresh match and
remains authoritative if the cache changes after the probe. The option adds at
most one CPU-side tree traversal per request in the configured scan window; it
does not launch model or GPU work.

Reservations are aligned to the configured KV page size. A short request is
promoted only when all of the following fit conservatively:

- its page-aligned prefill tokens;
- its clipped `max_new_tokens` reserve and one-page allocator overhead;
- its Mamba state slot and, when enabled, all ping-pong tracking-buffer slots;
- the request-pool and `--prefill-max-requests` limits.

At most one request may remain unfinished after a prefill batch. A reservation
is committed only after the existing chunked request is actually admitted.
After the configured number of compressed rounds, one full chunk is forced to
prevent starvation.

## Supported deployment mode

Auto chunking is limited to GLM5-Next PD prefill servers. NSA prefill context
parallelism is supported with `--nsa-prefill-cp-mode round-robin-split`, which
supports multi-request prefill batches.

Static incompatible combinations are rejected during server startup rather
than checked on every scheduling round:

- non-prefill or non-disaggregated serving;
- non-FCFS or priority scheduling;
- NSA `in-seq-split` CP or general prefill context parallelism;
- dynamic chunking;
- diffusion-LLM scheduling;
- prefill delaying;
- LoRA serving.

The scheduler logs both the reservation intent and actual short-request
admission on the statistics rank for operational verification.
