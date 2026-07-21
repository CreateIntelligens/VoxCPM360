#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

config_path="${1:-conf/voxcpm_v2/trial_lora_20epochs.yaml}"

if [[ $# -gt 1 ]]; then
    echo "用法：$0 [config.yaml]" >&2
    exit 2
fi

if [[ ! -f "$config_path" ]]; then
    echo "找不到訓練設定：$config_path" >&2
    exit 2
fi

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
TRAIN_CONFIG_PATH="$config_path" docker compose --profile training up \
    --no-deps --no-build --abort-on-container-exit --exit-code-from train train
