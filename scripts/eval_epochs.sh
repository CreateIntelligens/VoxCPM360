#!/usr/bin/env bash
# 逐一為訓練產出的 epoch checkpoint 生成驗收音檔。
# 台語的 loss/diff 不可靠，最終要靠耳朵判斷 —— 這支腳本讓每個 epoch
# 都有音檔可聽，而不是只能相信 val loss 挑出來的 best。
set -euo pipefail

CKPT_ROOT="${1:-checkpoints/lora-novel-e20}"
OUT_ROOT="${2:-scratch/epochs}"
VOICES="${VOICES:-cosy-young-female-01}"

for ck in "$CKPT_ROOT"/step_*/; do
    name=$(basename "$ck")
    [ -f "$ck/lora_weights.safetensors" ] || continue
    dest="$OUT_ROOT/$name"
    [ -d "$dest" ] && { echo "跳過 $name（已生成）"; continue; }

    echo "生成 $name ..."
    docker run --rm --device nvidia.com/gpu=all \
        -v "$PWD":/app -v /mnt/nas:/mnt/nas:ro \
        -w /app -u "$(id -u):$(id -g)" -e PYTHONPATH=/app/src \
        voxcpm360-app \
        python3 scripts/eval_lora_real.py \
            --lora-ckpt "$ck" --out-dir "$dest" --voice $VOICES \
        > "/tmp/eval_$name.log" 2>&1 || { echo "  失敗，見 /tmp/eval_$name.log"; continue; }
    echo "  -> $dest"
done
