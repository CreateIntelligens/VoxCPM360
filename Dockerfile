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

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Set compile flags for compatibility with Blackwell GPU architecture using CUDA 12.8
# Only build for the architectures actually targeted (sm_120 for GB10/Blackwell).
# Each extra arch multiplies the number of compile units, RAM and build time.
# MAX_JOBS caps parallel nvcc jobs so flash-attention compilation does not exhaust RAM (OOM).
ENV TORCH_CUDA_ARCH_LIST="12.0" \
    FLASH_ATTN_CUDA_ARCHS="12.0" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    NVCC_THREADS=1 \
    MAX_JOBS=2

RUN . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --no-cache-dir wheel packaging psutil ninja && \
    python3 -c "import torch.utils.cpp_extension as c; p = c.__file__; content = open(p).read().replace('def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:', 'def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:\n    return'); open(p, 'w').write(content)" && \
    git clone --branch v2.6.3 --single-branch https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    sed -i 's/dprops->major == 9 \&\& dprops->minor == 0/(dprops->major == 9 \&\& dprops->minor == 0) || dprops->major >= 12/g' /tmp/flash-attention/csrc/flash_attn/flash_api.cpp && \
    python3 -c 'p = "/tmp/flash-attention/setup.py"; c = open(p).read().replace("arch=compute_80,code=sm_80", "arch=compute_120,code=sm_120").replace("arch=compute_90,code=sm_90", "arch=compute_120,code=sm_120"); open(p, "w").write(c)' && \
    cd /tmp/flash-attention && \
    uv pip install --no-build-isolation . && \
    cd /app && \
    rm -rf /tmp/flash-attention && \
    uv pip install --no-cache-dir -r pyproject.toml && \
    uv pip install --no-cache-dir nano-vllm-voxcpm

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
