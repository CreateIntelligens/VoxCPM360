#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

docker compose --profile training config --format json | python3 -c '
import json
import sys

config = json.load(sys.stdin)
services = config["services"]
expected_limit = "68719476736"

for service_name in ("app", "train"):
    service = services[service_name]
    assert str(service["mem_limit"]) == expected_limit
    assert str(service["memswap_limit"]) == expected_limit
    assert service["user"] == "1000:1000"

train = services["train"]
assert train["image"] == "voxcpm360-app"
assert train["profiles"] == ["training"]
assert train["restart"] == "no"
assert train["command"][-1] == "conf/voxcpm_v2/trial_lora_20epochs.yaml"
assert train["volumes"][-1]["source"] == "/mnt/nas"
assert train["volumes"][-1]["target"] == "/mnt/nas"
assert train["volumes"][-1]["read_only"] is True
'

python3 - <<'PY'
import json
from pathlib import Path

for manifest in (
    Path("dataset/manifests/smoke_train.jsonl"),
    Path("dataset/manifests/smoke_val.jsonl"),
):
    for line_number, line in enumerate(manifest.open(encoding="utf-8"), 1):
        row = json.loads(line)
        dataset_id = row.get("dataset_id", 0)
        assert isinstance(dataset_id, int) and not isinstance(dataset_id, bool), (
            f"{manifest}:{line_number}: dataset_id must be an integer or omitted"
        )
PY

python3 - <<'PY'
from pathlib import Path

config_path = Path("conf/voxcpm_v2/trial_lora_20epochs.yaml")
values = {}
for line in config_path.read_text(encoding="utf-8").splitlines():
    if line.startswith(" ") or ":" not in line:
        continue
    key, value = line.split(":", 1)
    values[key] = value.strip()

for key in (
    "batch_size",
    "grad_accum_steps",
    "num_iters",
    "log_interval",
    "valid_interval",
    "save_interval",
    "warmup_steps",
    "max_steps",
):
    values[key] = int(values[key])

train_rows = sum(
    1 for _ in Path("dataset/manifests/smoke_train.jsonl").open(encoding="utf-8")
)
effective_batch = values["batch_size"] * values["grad_accum_steps"]
steps_per_epoch = (train_rows + effective_batch - 1) // effective_batch
assert values["num_iters"] == values["max_steps"] == steps_per_epoch * 20
assert values["valid_interval"] == steps_per_epoch
assert values["save_interval"] == steps_per_epoch * 5
assert values["log_interval"] == 10
assert values["warmup_steps"] == steps_per_epoch
assert values["save_path"] == "/app/checkpoints/trial_lora_20epochs"
PY

test -x ./train.sh
bash -n ./train.sh

fake_bin="$(mktemp -d)"
log_file="$(mktemp)"
trap 'rm -rf "$fake_bin" "$log_file"' EXIT

cat > "$fake_bin/docker" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$*" == "compose --profile training up "* ]]; then
    printf 'TRAIN_CONFIG_PATH=%s %s\n' "${TRAIN_CONFIG_PATH-}" "$*" \
        >> "$DOCKER_TEST_LOG"
else
    printf '%s\n' "$*" >> "$DOCKER_TEST_LOG"
fi

if [[ "$*" == "compose ps --status running --services app" ]]; then
    printf 'app\n'
fi
EOF
chmod +x "$fake_bin/docker"

DOCKER_TEST_LOG="$log_file" PATH="$fake_bin:$PATH" \
    ./train.sh conf/voxcpm_v2/smoke_lora.yaml

grep -Fxq "compose stop -t 30 app" "$log_file"
grep -Fxq \
    "TRAIN_CONFIG_PATH=conf/voxcpm_v2/smoke_lora.yaml compose --profile training up --no-deps --no-build --abort-on-container-exit --exit-code-from train train" \
    "$log_file"
grep -Fxq "compose start app" "$log_file"

: > "$log_file"
DOCKER_TEST_LOG="$log_file" PATH="$fake_bin:$PATH" ./train.sh
grep -Fxq \
    "TRAIN_CONFIG_PATH=conf/voxcpm_v2/trial_lora_20epochs.yaml compose --profile training up --no-deps --no-build --abort-on-container-exit --exit-code-from train train" \
    "$log_file"
