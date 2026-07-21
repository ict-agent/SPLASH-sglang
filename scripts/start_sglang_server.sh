#!/bin/bash
# =============================================================================
# 启动 sglang server (GLM-5-N-0707-INT8), 供 glm-evals-dcu 精度评测使用。
# 后台拉起 server 并阻塞等待 /health 就绪, 就绪后脚本返回 (server 仍在后台运行)。
# server pid 写入 PID_FILE, 调用方可据此在评测结束后 kill。
# CI test 阶段 docker cp 进容器执行 (见 .gitlab-ci.yml); 也可手动在容器内执行。
# 模型路径可用 MODEL_PATH 覆盖, 默认取容器内挂载点。
# =============================================================================
set -e
source /opt/dtk/env.sh

MODEL_PATH="${MODEL_PATH:-/models/GLM-5-N-0707-INT8/int8_ffn_mla_kda}"
PORT="${PORT:-30000}"
LOG_FILE="${LOG_FILE:-/workspace/sglang_server.log}"
PID_FILE="${PID_FILE:-/workspace/sglang_server.pid}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"  # 轮询次数, 每次 sleep 10s (默认 ~30min)

export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_CUSTOM_ALLREDUCE_INPUT_FENCE=thread0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_USE_TILELANG_MHC_POST=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=0

# ---- 后台拉起 server (nohup: 脚本返回后仍存活) ----
nohup sglang serve \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend dcu_mla \
  --linear-attn-backend triton \
  --json-model-override-args '{"disable_nsa": true}' \
  --quantization w8a8_int8 \
  --moe-runner-backend aiter \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 32 > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "sglang server 启动中 (pid ${SERVER_PID}), 等待 http://127.0.0.1:${PORT}/health_generate ..."

# ---- 阻塞等待 ready; 进程退出或超时则失败 ----
# /health_generate 会实际发送一个生成请求, 确保模型加载完毕且可推理
for _ in $(seq 1 "${READY_TIMEOUT}"); do
  if curl -sf --max-time 30 "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
    echo "sglang server ready (pid ${SERVER_PID})"
    exit 0
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "sglang server 启动失败, 见 ${LOG_FILE}" >&2
    exit 1
  fi
  sleep 10
done
echo "等待 sglang server ready 超时, 见 ${LOG_FILE}" >&2
exit 1

