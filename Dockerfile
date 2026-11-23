# ==========================================
# Stage 1: Base images with CUDA and python dependencies
# ==========================================
# CUDA 世代可切換，但 base image、torch variant、flash-attn 版本三者必須
# 同時改 —— 混搭會在執行期缺 libcudart。GB10/sm_121 需要 CUDA 13 的瘦長
# GEMM 調優，見 docs/plans/2026-08-31-gb10-cuda13-upgrade-design.md。
# builder 需要 devel 的 nvcc 編 flash-attn；runner 只用 runtime（省約 7GB）——
# torch 連的是 venv 自帶的 nvidia/cu13 與 cuDNN，devel 那套執行期用不到。
# 不可再縮成 564MB 的 `base` tag：flash-attn 連的是 image 的
# /usr/local/cuda/lib64/libcudart.so.13，不是 venv 那份。
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04

FROM ${CUDA_BASE_IMAGE} AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM ${CUDA_RUNTIME_IMAGE} AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 看似只屬於 build 的三樣，執行期都會用到：
# - gcc/g++：triton 在執行期編 kernel，缺了 torch.compile 會 InductorError
#   （VOXCPM_OPTIMIZE=true 失效）。Python.h 由 /opt/python 那份 Python 自帶。
# - git：castvoice 取 git hash 當 CastAgent 快取 key，缺了退化成 nogit。
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Stage 2: Builder image to compile and install dependencies
# ==========================================
FROM base AS builder

# Install uv for fast Python package resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Python 由 uv 安裝（Ubuntu 22.04 的 python3.11 套件停在 3.11.0），裝到固定路徑，
# runner 整份複製過去，venv 的 python 連結才找得到。--seed 帶 pip：flash-attn
# 從原始碼編時要用 pip wheel。
ENV UV_PYTHON_INSTALL_DIR=/opt/python
ARG PYTHON_VERSION=3.11
RUN uv python install "${PYTHON_VERSION}" && \
    uv venv --seed --python "${PYTHON_VERSION}" /opt/venv

# flash-attn 編譯層刻意只 COPY patch 腳本、不碰 pyproject.toml ——
# 依賴清單的改動才不會觸發近一小時的 flash-attn 重編（pyproject.toml
# 在編譯層之後才 COPY 進來）。
COPY scripts/patch_flash_attention_arch.py ./scripts/patch_flash_attention_arch.py

# 本機備妥對應平台的 wheel 即可跳過整段編譯。走 wheel 時 TORCH_ARCH_LIST
# 不生效 —— 架構已固化在 wheel 內，通用 wheel 應含三架構。
COPY wheels/ /tmp/wheels/

# Target CUDA compute capabilities are configurable at build time so the same
# Dockerfile works on Ampere (8.6), Hopper (9.0) and Blackwell (12.0) hosts.
# Pass --build-arg TORCH_ARCH_LIST=12.0 to build for a single architecture only
# (faster, smaller image) when the target GPU is known ahead of time.
ARG TORCH_ARCH_LIST="8.6;9.0;12.0"
# MAX_JOBS caps parallel nvcc jobs so flash-attention compilation does not
# exhaust RAM. Empty (default) = auto-detect at build time: one job per 6GB
# of available RAM, capped at core count. Override: --build-arg MAX_JOBS=N.
ARG MAX_JOBS=""
# 慢速線路（如 ARM 機到 pytorch CDN 僅 ~0.4MB/s）下載大 wheel 會觸發
# uv 預設逾時，可用 --build-arg UV_HTTP_TIMEOUT=900 放寬。
ARG UV_HTTP_TIMEOUT=300
ARG TORCH_CUDA_VARIANT=cu130
# 2.6.3 需手工 patch 才認得 Blackwell，2.7.4+ 原生支援。預編 wheel 的
# Release tag 依 CUDA variant 分流，因為兩者的 wheel 檔名相同但 ABI 不相容。
ARG FLASH_ATTN_VERSION=2.8.3
# torch 不釘版會隨 CDN 漂移（實測 2.11→2.13），與 Release 上按特定
# ABI 編的 flash-attn wheel 形成賭注式組合。釘在 A4000 已實測驗證的版本。
ARG TORCH_VERSION=2.13.0
# torchaudio 版號自 2.12 起與 torch 脫鉤（cu130 index 最新為 2.11.0）
ARG TORCHAUDIO_VERSION=2.11.0
ENV UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCH_LIST}" \
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
    uv pip install --no-cache-dir "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VARIANT}" --find-links /tmp/wheels && \
    uv pip install --no-cache-dir wheel packaging psutil ninja && \
    PYTAG="$(python3 -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')" && \
    WHEEL_NAME="flash_attn-${FLASH_ATTN_VERSION}-${PYTAG}-${PYTAG}-linux_$(uname -m).whl" && \
    if ! ls "/tmp/wheels/${WHEEL_NAME}" >/dev/null 2>&1; then \
        echo "flash-attn: fetching prebuilt ${WHEEL_NAME} from GH Release" && \
        { curl -fsSL --retry 3 -o "/tmp/wheels/${WHEEL_NAME}" \
            "https://github.com/CreateIntelligens/VoxCPM360/releases/download/flash-attn-wheels-${TORCH_CUDA_VARIANT}/${WHEEL_NAME}" \
            || { echo "flash-attn: wheel download failed, falling back to a ~1h source build"; \
                 rm -f "/tmp/wheels/${WHEEL_NAME}"; }; }; \
    fi && \
    if ls "/tmp/wheels/${WHEEL_NAME}" >/dev/null 2>&1; then \
        echo "flash-attn: installing prebuilt wheel ${WHEEL_NAME}, skipping compilation" && \
        uv pip install --no-cache-dir "/tmp/wheels/${WHEEL_NAME}"; \
    else \
        python3 -c "import torch.utils.cpp_extension as c; p = c.__file__; content = open(p).read().replace('def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:', 'def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:\n    return'); open(p, 'w').write(content)" && \
        git clone --branch "v${FLASH_ATTN_VERSION}" --single-branch https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
        if [ "${FLASH_ATTN_VERSION}" = "2.6.3" ]; then \
            # 2.6.3 不認得 Blackwell，需要手工 patch；2.7.4+ 官方支援、免 patch。
            python3 scripts/patch_flash_attention_arch.py /tmp/flash-attention/setup.py --architectures "${TORCH_ARCH_LIST}" && \
            sed -i 's/dprops->major == 9 \&\& dprops->minor == 0/(dprops->major == 9 \&\& dprops->minor == 0) || dprops->major >= 12/g' /tmp/flash-attention/csrc/flash_attn/flash_api.cpp; \
        fi && \
        cd /tmp/flash-attention && \
        python3 -m pip wheel --no-build-isolation --no-deps -w /tmp/wheels . && \
        uv pip install --no-cache-dir /tmp/wheels/flash_attn-*.whl && \
        cd /app && \
        rm -rf /tmp/flash-attention; \
    fi


# 應用層依賴（含 serve 與測試依賴，讓容器內可直接跑 pytest）：版本只由 uv.lock 決定。
# 位於編譯層之後，改 pyproject.toml／uv.lock 只會重跑這一層（約一分鐘），不會重編 flash-attn。
#
# 不用 `uv pip install -r pyproject.toml`（每次重新解析、完全不讀 lock，重建會偷升版），
# 也不用 `uv sync`／`uv pip sync`（會把 lock 以外的東西——編譯層的 torch、flash-attn——刪掉）。
# 改成 `uv export --locked` 匯出 requirements，再 `--no-deps` 只照單安裝：
# - --locked：lock 跟 pyproject.toml 不一致時 export 直接失敗，建置中止，不會靜默重解。
# - torch 這一串刻意不從 lock 裝：torch／torchaudio／flash-attn 由編譯層依 build args 裝
#   （PyTorch CUDA index 或預編 wheel），lock 解出的是 PyPI 的版本；triton、nvidia-*、
#   cuda-* 是 torch 的 CUDA 相依，必須跟編譯層裝的那一套同進退，否則會出現 13.0／13.4 混版。
#   --no-emit-package 只排除指定的套件本身，所以排除清單由 lock 的內容算出，不手寫。
# - 排除的版本在安裝前後比對，不同就讓建置失敗。
COPY pyproject.toml uv.lock ./
RUN . /opt/venv/bin/activate && \
    uv export --locked --no-emit-project --no-hashes --extra serve --extra test -o /tmp/all.txt && \
    excluded="$(sed -n -E 's/^((nvidia|cuda)-[a-z0-9-]+|torch|torchaudio|flash-attn|triton)==.*/\1/p' /tmp/all.txt)" && \
    echo "excluded from lock install: $(echo $excluded)" && \
    for n in torch torchaudio flash-attn triton nvidia-cublas; do \
        echo "$excluded" | grep -qx "$n" || { echo "exclude list missing $n, lock layout changed"; exit 1; }; \
    done && \
    uv export --locked --no-emit-project --extra serve --extra test \
        $(for n in $excluded; do printf -- '--no-emit-package %s ' "$n"; done) -o /tmp/req.txt && \
    guard='^(torch|torchaudio|flash[-_]attn|triton|nvidia-.*|cuda-.*)[ =@]' && \
    uv pip freeze | grep -iE "$guard" | sort > /tmp/torch-before.txt && \
    uv pip install --no-cache-dir --no-deps -r /tmp/req.txt && \
    uv pip freeze | grep -iE "$guard" | sort > /tmp/torch-after.txt && \
    diff /tmp/torch-before.txt /tmp/torch-after.txt && \
    rm -f /tmp/all.txt /tmp/req.txt /tmp/torch-before.txt /tmp/torch-after.txt

# 編譯後把 wheel 導出保存（每個 CPU 架構跑一次即可，之後所有機器免編譯）：
#   docker buildx build --target flash-wheel -o type=local,dest=wheels .
FROM scratch AS flash-wheel
COPY --from=builder /tmp/wheels/ /

# ==========================================
# Stage 3: Runner image (Production/Runtime)
# ==========================================
FROM runtime-base AS runner

# Copy the Python runtime and the pre-compiled virtual environment from builder
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv

COPY scripts/patch_nanovllm_timesteps.py /tmp/patch_nanovllm_timesteps.py
RUN python /tmp/patch_nanovllm_timesteps.py \
    && rm /tmp/patch_nanovllm_timesteps.py

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
