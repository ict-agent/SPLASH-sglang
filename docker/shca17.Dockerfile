# =============================================================================
# sglang shca17 运行镜像
# =============================================================================
ARG BASE_IMAGE=42.228.13.241:5000/jenkins/base_env/shca17:ubuntu22.04-dtk2604-py3.10-dtk2604
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.10.0
ARG SGLANG_VERSION=0.5.12
ARG PIP_INDEX_URL=http://42.228.13.241:666/nightly/dtk2604/+simple/
ARG PIP_TRUSTED_HOST=42.228.13.241

SHELL ["/bin/bash", "-c"]

# ---- 配置内部 pip 源 ----
RUN mkdir -p /root/.pip && \
    printf "[global]\nindex-url = ${PIP_INDEX_URL}\ntrusted-host = ${PIP_TRUSTED_HOST}\n" > /root/.pip/pip.conf

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
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir torch==${TORCH_VERSION} torchvision && \
    pip install --no-cache-dir torch==${TORCH_VERSION} flash-attn && \
    pip install --no-cache-dir torch==${TORCH_VERSION} lightop && \
    pip install --no-cache-dir torch==${torchversion} lmslim && \
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
    pip install --no-cache-dir triton==3.6.0 && \
    pip install --no-cache-dir numpy==1.25.0 && \
    pip uninstall -y starlette fastapi prometheus-fastapi-instrumentator && \
    pip install --no-cache-dir "fastapi==0.115.12" "starlette==0.46.2" "prometheus-fastapi-instrumentator==7.1.0" && \
    pip install --no-cache-dir torchaudio==2.11.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir nvidia-cutlass-dsl==4.4.2 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir sgl-deep-gemm==0.1.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir kernels==0.14 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip cache purge

# ---------------------------------------------------------------------------
# layer 2.5: 定制 rocblas / hipblaslt (覆盖镜像内的版本)
# ---------------------------------------------------------------------------
COPY rocblas-install/ /opt/rocblas-install/
COPY hipblaslt-install/ /opt/hipblaslt-install/
ENV LD_LIBRARY_PATH=/opt/rocblas-install/lib:/opt/hipblaslt-install/lib:${LD_LIBRARY_PATH}
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH' >> /root/.bashrc

# ---------------------------------------------------------------------------
# layer 3: 装入 CI 编译出的本仓库 whl (sgl-kernel / sglang)
# ---------------------------------------------------------------------------
COPY wheelhouse/ /tmp/wheelhouse/
RUN pip install --no-cache-dir /tmp/wheelhouse/*.whl && \
    pip install --no-cache-dir "sglang[diffusion]==${SGLANG_VERSION}" && \
    pip install --no-cache-dir numpy==1.25.0 && \
    rm -rf /tmp/wheelhouse && \
    pip install --no-cache-dir transformers==5.3.0 && \
    pip cache purge && \
    source /opt/dtk/env.sh && \
    python -c "import sgl_kernel; import sglang; import sglang_router; print('sglang', sglang.__version__)"

# ---- 删除内部 pip 源配置 ----
RUN rm -f /root/.pip/pip.conf
