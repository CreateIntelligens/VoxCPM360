#!/usr/bin/env bash
# 訓練完一鍵跑：收集 checkpoint → 生成試聽音檔 → Breeze-ASR 評分排名。
# 台語 val loss 不可信，排名結果用來縮小人耳要聽的範圍，最後仍以實聽為準。
#
# 用法: scripts/rank_lora.sh <輸出目錄> <訓練目錄或 checkpoint>...
#   例: scripts/rank_lora.sh scratch/vh387_rank checkpoints/vh387-hayley-orig checkpoints/vh387-hayley-mix
# 已生成、已辨識過的版本會略過，可以中斷後重跑。
set -euo pipefail

OUT="${1:?用法: $0 <輸出目錄> <訓練目錄或 checkpoint>...}"
shift
[ $# -gt 0 ] || { echo "至少給一個訓練目錄或 checkpoint" >&2; exit 1; }

# step_0000000 是訓練前的初始權重；最後一步常會連存兩次（如 399 與 400），
# 權重相同的只留一個，免得白生成一輪。
declare -A seen=()
ckpts=()
for arg in "$@"; do
    if [ -f "$arg/lora_weights.safetensors" ]; then
        candidates=("$arg")
    else
        candidates=("$arg"/step_*/)
    fi
    # 由後往前走，權重相同時留下較晚的那一步（例如 400 而不是 399）。
    for ((i = ${#candidates[@]} - 1; i >= 0; i--)); do
        ck="${candidates[i]%/}"
        [ -f "$ck/lora_weights.safetensors" ] || continue
        [ "$(basename "$ck")" = step_0000000 ] && continue
        digest=$(sha256sum "$ck/lora_weights.safetensors" | cut -d' ' -f1)
        [ -n "${seen[$digest]:-}" ] && { echo "略過 $ck（與 ${seen[$digest]} 相同）"; continue; }
        seen[$digest]="$ck"
        ckpts+=("$ck")
    done
done
echo "共 ${#ckpts[@]} 個 checkpoint"

mkdir -p "$OUT"
docker run --rm --device nvidia.com/gpu=all \
    -v "$PWD":/app -v /mnt/nas:/mnt/nas:ro \
    -w /app -u "$(id -u):$(id -g)" -e PYTHONPATH=/app/src \
    voxcpm360-app \
    python3 scripts/eval_vh387.py --ckpts "${ckpts[@]}" --out-dir "$OUT"

python3 scripts/score_lora_asr.py "$OUT"
