#!/usr/bin/env bash
#
# Thin wrapper around `python -m sglang.bench_serving` for online-serving
# throughput / latency benchmarks against an already-running SGLang server.
#
# Quick examples
# --------------
# 1) Random dataset, 1K in / 1K out, 32 concurrency, no profiling:
#      bash bench_serving.sh \
#          --base-url http://localhost:30000 \
#          --dataset random --input-len 1024 --output-len 1024 \
#          --num-prompts 200 --max-concurrency 32
#
# 2) Capture a torch-profiler trace (server must have been launched with
#    SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_traces):
#      bash bench_serving.sh \
#          --base-url http://localhost:30000 \
#          --dataset random --input-len 4096 --output-len 256 \
#          --num-prompts 32 --max-concurrency 8 \
#          --profile --profile-output-dir /tmp/sglang_traces/run1
#
# 3) ShareGPT, override output length to 256:
#      bash bench_serving.sh \
#          --base-url http://localhost:30000 \
#          --dataset sharegpt --dataset-path /data/ShareGPT_V3.json \
#          --output-len 256 --num-prompts 500 --request-rate 8
#
# 4) Generated-shared-prefix (long-prefix cache stress):
#      bash bench_serving.sh \
#          --base-url http://localhost:30000 \
#          --dataset generated-shared-prefix \
#          --gsp-num-groups 32 --gsp-prompts-per-group 16 \
#          --gsp-system-prompt-len 4096 --gsp-question-len 128 \
#          --gsp-output-len 256 --num-prompts 512
#
# Pass any flag this script does not know about straight through to
# bench_serving.py, e.g.  `--seed 42  --return-logprob  --tag run-A`.

set -euo pipefail

# ---- defaults -------------------------------------------------------------
BACKEND="sglang"
BASE_URL="http://localhost:30000"
HOST="0.0.0.0"
PORT="30000"
MODEL=""
TOKENIZER=""

DATASET="random"
DATASET_PATH=""
NUM_PROMPTS=1
REQUEST_RATE="inf"
MAX_CONCURRENCY=""
WARMUP_REQUESTS=1

# Common len knobs. Mapped onto the right --random-* / --sharegpt-* / --gsp-*
# flag based on $DATASET so callers don't have to remember every variant.
INPUT_LEN="131072"
OUTPUT_LEN="2"
RANGE_RATIO=1

# Profile knobs
PROFILE=0
PROFILE_OUTPUT_DIR="/tmp/sglang_traces"
PROFILE_PREFIX=""
PROFILE_STEPS="4"
PROFILE_START_STEP="0"
PROFILE_ACTIVITIES=""   # space-separated, e.g. "CPU GPU CUDA_PROFILER MEM"
PROFILE_BY_STAGE=0
PROFILE_STAGES="prefill"       # space-separated, e.g. "prefill decode"

OUTPUT_FILE=""
TAG=""
EXTRA_ARGS=()

usage() {
    sed -n '2,40p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;

        --backend)            BACKEND="$2"; shift 2 ;;
        --base-url)           BASE_URL="$2"; shift 2 ;;
        --host)               HOST="$2"; shift 2 ;;
        --port)               PORT="$2"; shift 2 ;;
        --model)              MODEL="$2"; shift 2 ;;
        --tokenizer)          TOKENIZER="$2"; shift 2 ;;

        --dataset|--dataset-name) DATASET="$2"; shift 2 ;;
        --dataset-path)       DATASET_PATH="$2"; shift 2 ;;
        --num-prompts)        NUM_PROMPTS="$2"; shift 2 ;;
        --request-rate)       REQUEST_RATE="$2"; shift 2 ;;
        --max-concurrency)    MAX_CONCURRENCY="$2"; shift 2 ;;
        --warmup-requests)    WARMUP_REQUESTS="$2"; shift 2 ;;

        # Length knobs (dataset-agnostic; mapped below)
        --input-len)          INPUT_LEN="$2"; shift 2 ;;
        --output-len)         OUTPUT_LEN="$2"; shift 2 ;;
        --range-ratio)        RANGE_RATIO="$2"; shift 2 ;;

        # Profile knobs
        --profile)            PROFILE=1; shift ;;
        --profile-output-dir) PROFILE_OUTPUT_DIR="$2"; shift 2 ;;
        --profile-prefix)     PROFILE_PREFIX="$2"; shift 2 ;;
        --profile-steps)      PROFILE_STEPS="$2"; shift 2 ;;
        --profile-start-step) PROFILE_START_STEP="$2"; shift 2 ;;
        --profile-activities) PROFILE_ACTIVITIES="$2"; shift 2 ;;
        --profile-by-stage)   PROFILE_BY_STAGE=1; shift ;;
        --profile-stages)     PROFILE_STAGES="$2"; shift 2 ;;
        --output-file)        OUTPUT_FILE="$2"; shift 2 ;;
        --tag)                TAG="$2"; shift 2 ;;

        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *)  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- validation -----------------------------------------------------------
if [[ -z "$BASE_URL" && -z "$PORT" ]]; then
    echo "error: pass --base-url http://host:port  (or --host/--port)" >&2
    exit 1
fi

# ---- assemble command -----------------------------------------------------
CMD=(python3 -m sglang.bench_serving
     --backend "$BACKEND"
     --dataset-name "$DATASET"
     --num-prompts "$NUM_PROMPTS"
     --request-rate "$REQUEST_RATE"
     --warmup-requests "$WARMUP_REQUESTS")

[[ -n "$BASE_URL" ]]        && CMD+=(--base-url "$BASE_URL")
[[ -n "$PORT" ]]            && CMD+=(--host "$HOST" --port "$PORT")
[[ -n "$MODEL" ]]           && CMD+=(--model "$MODEL")
[[ -n "$TOKENIZER" ]]       && CMD+=(--tokenizer "$TOKENIZER")
[[ -n "$DATASET_PATH" ]]    && CMD+=(--dataset-path "$DATASET_PATH")
[[ -n "$MAX_CONCURRENCY" ]] && CMD+=(--max-concurrency "$MAX_CONCURRENCY")
[[ -n "$OUTPUT_FILE" ]]     && CMD+=(--output-file "$OUTPUT_FILE")
[[ -n "$TAG" ]]             && CMD+=(--tag "$TAG")

# Map the generic --input-len / --output-len / --range-ratio onto whichever
# concrete flag the chosen dataset wants.
case "$DATASET" in
    random|random-ids|image)
        [[ -n "$INPUT_LEN" ]]   && CMD+=(--random-input-len "$INPUT_LEN")
        [[ -n "$OUTPUT_LEN" ]]  && CMD+=(--random-output-len "$OUTPUT_LEN")
        [[ -n "$RANGE_RATIO" ]] && CMD+=(--random-range-ratio "$RANGE_RATIO")
        ;;
    sharegpt)
        # ShareGPT has no input-len knob (it comes from the trace). We only
        # forward output-len / context-len overrides.
        [[ -n "$OUTPUT_LEN" ]]  && CMD+=(--sharegpt-output-len "$OUTPUT_LEN")
        [[ -n "$INPUT_LEN" ]]   && CMD+=(--sharegpt-context-len "$INPUT_LEN")
        ;;
    generated-shared-prefix)
        # For GSP, --input-len is interpreted as the per-question length and
        # --output-len as the per-request output length. Tune group/prefix
        # sizes via passthrough flags (--gsp-num-groups, --gsp-system-prompt-len, ...).
        [[ -n "$INPUT_LEN" ]]   && CMD+=(--gsp-question-len "$INPUT_LEN")
        [[ -n "$OUTPUT_LEN" ]]  && CMD+=(--gsp-output-len "$OUTPUT_LEN")
        [[ -n "$RANGE_RATIO" ]] && CMD+=(--gsp-range-ratio "$RANGE_RATIO")
        ;;
    *)
        # custom / openai / mmmu / mooncake / longbench_v2: pass len knobs
        # through verbatim if the caller set them.
        [[ -n "$INPUT_LEN" ]]   && EXTRA_ARGS+=(--random-input-len "$INPUT_LEN")
        [[ -n "$OUTPUT_LEN" ]]  && EXTRA_ARGS+=(--random-output-len "$OUTPUT_LEN")
        [[ -n "$RANGE_RATIO" ]] && EXTRA_ARGS+=(--random-range-ratio "$RANGE_RATIO")
        ;;
esac

if [[ "$PROFILE" -eq 1 ]]; then
    CMD+=(--profile)
    [[ -n "$PROFILE_OUTPUT_DIR" ]] && { mkdir -p "$PROFILE_OUTPUT_DIR"; CMD+=(--profile-output-dir "$PROFILE_OUTPUT_DIR"); }
    [[ -n "$PROFILE_PREFIX" ]]     && CMD+=(--profile-prefix "$PROFILE_PREFIX")
    [[ -n "$PROFILE_STEPS" ]]      && CMD+=(--profile-steps "$PROFILE_STEPS")
    [[ -n "$PROFILE_START_STEP" ]] && CMD+=(--profile-start-step "$PROFILE_START_STEP")
    [[ -n "$PROFILE_ACTIVITIES" ]] && CMD+=(--profile-activities $PROFILE_ACTIVITIES)
    [[ "$PROFILE_BY_STAGE" -eq 1 ]] && CMD+=(--profile-by-stage)
    [[ -n "$PROFILE_STAGES" ]] && CMD+=(--profile-stages $PROFILE_STAGES)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "+ ${CMD[*]}"
exec "${CMD[@]}"
