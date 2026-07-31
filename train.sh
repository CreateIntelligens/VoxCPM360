#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

config_path="${1:-conf/voxcpm_v2/trial_lora_20epochs.yaml}"

if [[ $# -gt 1 ]]; then
    echo "用法：$0 [config.yaml]" >&2
    echo "  TRAIN_RUNNER=docker|local 可強制指定執行方式（預設自動偵測）。" >&2
    echo "  NPROC_PER_NODE 可覆寫 GPU 數（僅 local 模式）。" >&2
    exit 2
fi

if [[ ! -f "$config_path" ]]; then
    echo "找不到訓練設定：$config_path" >&2
    exit 2
fi

# 執行方式：有 docker 就沿用容器編排；沒有（例如 Taipei-1 這類無 docker、
# 無 sudo 的 HPC 叢集，訓練由 Slurm + Enroot 拉起）則直接在當前環境啟動。
if [[ -n "${TRAIN_RUNNER:-}" ]]; then
    runner="$TRAIN_RUNNER"
elif docker compose version >/dev/null 2>&1; then
    runner=docker
else
    runner=local
fi

if [[ "$runner" == docker ]]; then
    docker compose config --quiet

    if [[ "$(docker compose --profile training ps --status running --services train)" == "train" ]]; then
        echo "已有 VoxCPM 訓練容器執行中，拒絕重複啟動。" >&2
        exit 1
    fi

    app_was_running=false
    if [[ "$(docker compose ps --status running --services app)" == "app" ]]; then
        app_was_running=true
        echo "停止推論服務，避免同時載入兩份模型。"
        docker compose stop -t 30 app
    fi

    restore_app() {
        status=$?
        trap - EXIT
        if [[ "$app_was_running" == true ]]; then
            echo "恢復原本執行中的推論服務。"
            docker compose start app || echo "警告：推論服務恢復失敗。" >&2
        fi
        exit "$status"
    }
    trap restore_app EXIT

    echo "訓練設定：$config_path"
    echo "容器統一記憶體上限：64 GiB（不允許額外 swap）"
    TRAIN_CONFIG_PATH="$config_path" docker compose --profile training up         --no-deps --no-build --abort-on-container-exit --exit-code-from train train
    exit $?
fi

export PYTORCH_JIT=0
export TOKENIZERS_PARALLELISM=false

# GPU 數決定啟動方式：torchrun 才會注入 WORLD_SIZE/RANK/LOCAL_RANK，
# 裸 python 會讓 Accelerator 取到預設 world_size=1 而只用單卡（見
# src/voxcpm/training/accelerator.py）。config 的有效 batch 是按多卡估算的，
# 誤走單卡會讓有效 batch 與 epoch 數同步縮小。
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    nproc_gpu="$NPROC_PER_NODE"
elif command -v nvidia-smi >/dev/null 2>&1; then
    nproc_gpu="$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 0)"
else
    nproc_gpu=0
fi

if [[ ! "$nproc_gpu" =~ ^[0-9]+$ ]] || [[ "$nproc_gpu" -lt 1 ]]; then
    echo "偵測不到 GPU，改以單進程執行。" >&2
    nproc_gpu=1
fi

echo "訓練設定：$config_path"
echo "GPU 數：$nproc_gpu"

if [[ "$nproc_gpu" -gt 1 ]]; then
    # 同一節點可能同時跑多個訓練作業（節點有 8 張 H100），而 torchrun 預設
    # 的 rendezvous port 29500 是固定的，第二個作業會 EADDRINUSE 直接崩。
    # 用 job id 推導 port，確保不同作業必然錯開。
    rdzv_port=$(( 29500 + ${SLURM_JOB_ID:-0} % 1000 ))
    exec torchrun --nproc_per_node="$nproc_gpu"         --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$rdzv_port"         scripts/train_voxcpm_finetune.py --config_path "$config_path"
else
    exec python -u scripts/train_voxcpm_finetune.py --config_path "$config_path"
fi
