# =============================================================================
# sglang shca17 运行镜像
#
# BASE_IMAGE: 已装完所有驱动 / 编译器 (DTK/UCX/OpenMPI/RCCL/shca17)，但未装任何
#             python whl 包的基础镜像。
# 本 Dockerfile 在构建时现装:
#   1) install_sglang_0.5.12_shca17.sh 中的依赖 whl 层 (ray/vllm/torch 栈等)
#   2) CI compile 阶段编译出的本仓库 whl (sgl-kernel / sglang / gateway)，force
#      覆盖依赖层里 pip 装的 sglang==0.5.12
#
# 参考: model-test-ci/docker/conf/install_sglang_0.5.12_shca17.sh (# ---layer--- 之后的 pip 段)
# =============================================================================
ARG BASE_IMAGE=42.228.13.241:5000/jenkins/base_env/shca17:ubuntu22.04-dtk2604-py3.10-dtk2604
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.10.0
# 本仓库编译出的 sglang 版本 (compile 阶段 SETUPTOOLS_SCM_PRETEND_VERSION)
ARG SGLANG_VERSION=0.5.12
# 内部 nightly 源 (dtk2604)；base 镜像通常已配 ~/.pip/pip.conf，这里再显式声明一次兜底
ARG PIP_INDEX_URL=http://42.228.13.241:666/nightly/dtk2604/+simple/
ARG PIP_TRUSTED_HOST=42.228.13.241
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# layer 1: ray / amdsmi / cupy + vllm 栈
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir ray[data,train,tune,serve] -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir amdsmi && \
    pip install --no-cache-dir cupy && \
    pip install --no-cache-dir apache-tvm-ffi==0.1.9 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir vllm==0.21.0 && \
    pip install --no-cache-dir vllm-hcu==0.21.0 && \
    pip cache purge

# ---------------------------------------------------------------------------
# layer 2: torch 栈 + hcu 生态 whl + sglang 的第三方依赖
# 注意: sglang / sgl-kernel / sglang-router 不在此从 pip 装 —— 由本仓库编译，
#       在 layer 3 用 wheelhouse 现装。
# 每条 pip 带上 torch==${TORCH_VERSION} 是为了锁定解析时的 torch 版本 (沿用原脚本写法)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir torch==${TORCH_VERSION} torchvision && \
    pip install --no-cache-dir torch==${TORCH_VERSION} flash-attn && \
    pip install --no-cache-dir torch==${TORCH_VERSION} lightop && \
    pip install --no-cache-dir torch==${TORCH_VERSION} lmslim && \
    pip install --no-cache-dir torch==${TORCH_VERSION} deepgemm && \
    pip install --no-cache-dir torch==${TORCH_VERSION} aiter && \
    pip install --no-cache-dir mooncake_transfer_engine_shca && \
    pip install --no-cache-dir torch==${TORCH_VERSION} deep_ep_shca && \
    pip install --no-cache-dir torch==${TORCH_VERSION} tilelang && \
    pip install --no-cache-dir torch==${TORCH_VERSION} vllm-hcu && \
    pip install --no-cache-dir torch==${TORCH_VERSION} causal_conv1d && \
    pip install --no-cache-dir torch==${TORCH_VERSION} flash_mla && \
    pip install --no-cache-dir torch==${TORCH_VERSION} fastsafetensors && \
    pip install --no-cache-dir torch==${TORCH_VERSION} sglang-router && \
    pip install --no-cache-dir numpy==1.25.0 && \
    pip uninstall -y starlette fastapi prometheus-fastapi-instrumentator && \
    pip install --no-cache-dir "fastapi==0.115.12" "starlette==0.46.2" "prometheus-fastapi-instrumentator==7.1.0" && \
    pip install --no-cache-dir nvidia-cutlass-dsl==4.4.2 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir sgl-deep-gemm==0.1.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir kernels==0.14 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip cache purge

# ---------------------------------------------------------------------------
# layer 3: 装入 CI 编译出的本仓库 whl (sgl-kernel / sglang)
# sglang-router 不编译，从 pip 源装 (其依赖 setproctitle/aiohttp/... 不含 sglang，
#   不会覆盖本地 sglang)。
# 再装 sglang[diffusion]==${SGLANG_VERSION}: 版本已被本地 whl 满足，pip 只补 diffusion
#   extra 依赖，不会重新从 pip 覆盖本地 sglang。
# 最后钉一次 numpy==1.25.0 (沿用原脚本，防依赖解析把 numpy 顶掉)。
# 构建上下文里需存在 wheelhouse/ (CI 从 compile 阶段产物 cp 过来)
# ---------------------------------------------------------------------------
COPY wheelhouse/ /tmp/wheelhouse/
RUN pip install --no-cache-dir /tmp/wheelhouse/*.whl && \
    pip install --no-cache-dir "sglang[diffusion]==${SGLANG_VERSION}" && \
    pip install --no-cache-dir numpy==1.25.0 && \
    rm -rf /tmp/wheelhouse && \
    pip cache purge && \
    source /opt/dtk/env.sh && \
    python -c "import sgl_kernel; import sglang; import sglang_router; print('sglang', sglang.__version__)"
