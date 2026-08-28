# ==========================================
# Stage 1: Base image with CUDA and python dependencies
# ==========================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime system dependencies and python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Stage 2: Builder image to compile and install dependencies
# ==========================================
FROM base AS builder

# Install uv for fast Python package resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a virtual environment to isolate dependencies
RUN python3.10 -m venv /opt/venv

# flash-attn 編譯層刻意只 COPY patch 腳本、不碰 pyproject.toml ——
# 依賴清單的改動才不會觸發近一小時的 flash-attn 重編（pyproject.toml
# 在編譯層之後才 COPY 進來）。
COPY scripts/patch_flash_attention_arch.py ./scripts/patch_flash_attention_arch.py

# Target CUDA compute capabilities are configurable at build time so the same
# Dockerfile works on Ampere (8.6), Hopper (9.0) and Blackwell (12.0) hosts.
# Pass --build-arg TORCH_ARCH_LIST=12.0 to build for a single architecture only
# (faster, smaller image) when the target GPU is known ahead of time.
ARG TORCH_ARCH_LIST="8.6;9.0;12.0"
# MAX_JOBS caps parallel nvcc jobs so flash-attention compilation does not
# exhaust RAM. Empty (default) = auto-detect at build time: one job per 6GB
# of available RAM, capped at core count. Override: --build-arg MAX_JOBS=N.
ARG MAX_JOBS=""
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" \
    FLASH_ATTN_CUDA_ARCHS="${TORCH_ARCH_LIST}" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    NVCC_THREADS=1

RUN . /opt/venv/bin/activate && \
    jobs="${MAX_JOBS:-$(awk '/MemAvailable/{print int($2/1024/1024/6)}' /proc/meminfo)}" && \
    cores="$(nproc)" && \
    { [ -n "$jobs" ] && [ "$jobs" -ge 1 ]; } || jobs=1 && \
    [ "$jobs" -le "$cores" ] || jobs="$cores" && \
    export MAX_JOBS="$jobs" && \
    echo "flash-attn build: MAX_JOBS=$MAX_JOBS (cores=$cores)" && \
    uv pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --no-cache-dir wheel packaging psutil ninja && \
    python3 -c "import torch.utils.cpp_extension as c; p = c.__file__; content = open(p).read().replace('def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:', 'def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:\n    return'); open(p, 'w').write(content)" && \
    git clone --branch v2.6.3 --single-branch https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    python3 scripts/patch_flash_attention_arch.py /tmp/flash-attention/setup.py --architectures "${TORCH_ARCH_LIST}" && \
    sed -i 's/dprops->major == 9 \&\& dprops->minor == 0/(dprops->major == 9 \&\& dprops->minor == 0) || dprops->major >= 12/g' /tmp/flash-attention/csrc/flash_attn/flash_api.cpp && \
    cd /tmp/flash-attention && \
    uv pip install --no-build-isolation . && \
    cd /app && \
    rm -rf /tmp/flash-attention

# 應用層依賴（含測試依賴，讓容器內可直接跑 pytest）。位於編譯層之後，
# 改 pyproject.toml 只會重跑這一層（約一分鐘），不會重編 flash-attn。
COPY pyproject.toml uv.lock ./
RUN . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir -r pyproject.toml && \
    uv pip install --no-cache-dir nano-vllm-voxcpm && \
    uv pip install --no-cache-dir -r pyproject.toml --extra test

# ==========================================
# Stage 3: Runner image (Production/Runtime)
# ==========================================
FROM base AS runner

# Copy the pre-compiled virtual environment containing python packages from builder
COPY --from=builder /opt/venv /opt/venv

# Run bind-mounted workspace processes as the host user so generated files remain editable.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" voxcpm \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/bash voxcpm \
    && mkdir -p /home/voxcpm/.cache/huggingface /home/voxcpm/.cache/modelscope \
    && chown -R "${APP_UID}:${APP_GID}" /home/voxcpm

ENV HOME=/home/voxcpm

USER voxcpm

# Ensure the app starts by default on port 8000
EXPOSE 8000

# Default run command (will run from the mounted directory)
CMD ["python", "app.py", "--port", "8000", "--device", "cuda"]
