#!/usr/bin/env bash
# GB10 是統一記憶體（CPU/GPU 共用 LPDDR5X），nvidia-smi 查不到 VRAM，
# 所以峰值要從 /proc/meminfo 的整體用量來看。
# 用法: ./scripts/watch_memory.sh [取樣秒數] [輸出檔]
set -euo pipefail

INTERVAL="${1:-5}"
OUT="${2:-/tmp/vox_mem_watch.log}"

peak_used=0
peak_swap=0

printf 'time\tused_GiB\tavail_GiB\tswap_GiB\n' | tee "$OUT"

while true; do
    read -r total avail <<<"$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print t, a}' /proc/meminfo)"
    read -r swt swf <<<"$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print t, f}' /proc/meminfo)"

    used=$(( (total - avail) / 1024 / 1024 ))
    availg=$(( avail / 1024 / 1024 ))
    swap=$(( (swt - swf) / 1024 / 1024 ))

    (( used > peak_used )) && peak_used=$used
    (( swap > peak_swap )) && peak_swap=$swap

    printf '%s\t%d\t%d\t%d\n' "$(date +%H:%M:%S)" "$used" "$availg" "$swap" | tee -a "$OUT"
    sleep "$INTERVAL"
done
