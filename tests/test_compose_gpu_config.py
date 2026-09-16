"""docker-compose.yml 的 GPU 宣告會依 .env 切換，Linux 預設必須維持 CDI。

Windows（Docker Desktop）讀不到 /var/run/cdi/nvidia.yaml，只能走 legacy
nvidia runtime；Linux 反過來必須用 CDI，否則 systemd daemon-reload 會洗掉
cgroup 規則讓 NVML 失效。兩邊共用同一份 compose 檔，靠變數插值分流 ——
所以「沒設變數時展開成 cdi」是一條必須被檢查的不變式，不是註解裡的約定。
"""

import shutil
import subprocess

import pytest
import yaml

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

# 需要 GPU 宣告的服務；web 是純 nginx，不在此列。
GPU_SERVICES = ("app", "train")


def _compose_config(env: dict[str, str]) -> dict:
    """跑 `docker compose config` 拿到插值後的結果。

    直接解析原始 YAML 看不出 ${VAR:-default} 實際會展開成什麼，那正是
    這裡要驗的東西，所以一定要讓 compose 自己算。
    """
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(REPO_ROOT),
            "-f",
            str(COMPOSE_FILE),
            "--profile",
            "training",
            "config",
        ],
        capture_output=True,
        text=True,
        # 不繼承呼叫端環境，免得開發機自己的 .env 汙染結果。
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", **env},
    )
    if result.returncode != 0:
        pytest.fail(f"docker compose config 失敗：{result.stderr}")
    return yaml.safe_load(result.stdout)


def _gpu_devices(config: dict, service: str) -> list[dict]:
    reservations = config["services"][service]["deploy"]["resources"]
    return reservations["reservations"]["devices"]


pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="需要 docker CLI 才能做 compose 插值",
)


@pytest.fixture(scope="module")
def default_config() -> dict:
    # COMPOSE_ENV_FILE 指向不存在的檔案，確保沒有任何 .env 被讀進來。
    return _compose_config({"COMPOSE_ENV_FILE": str(REPO_ROOT / ".env.absent")})


@pytest.fixture(scope="module")
def windows_config() -> dict:
    return _compose_config(
        {
            "COMPOSE_ENV_FILE": str(REPO_ROOT / ".env.absent"),
            "GPU_DRIVER": "nvidia",
            "GPU_DEVICE_ID": "0",
        }
    )


@pytest.mark.parametrize("service", GPU_SERVICES)
def test_linux_default_is_cdi(default_config: dict, service: str) -> None:
    devices = _gpu_devices(default_config, service)
    assert len(devices) == 1
    assert devices[0]["driver"] == "cdi"
    assert devices[0]["device_ids"] == ["nvidia.com/gpu=all"]
    assert devices[0]["capabilities"] == ["gpu"]


@pytest.mark.parametrize("service", GPU_SERVICES)
def test_env_override_switches_to_legacy_runtime(
    windows_config: dict, service: str
) -> None:
    devices = _gpu_devices(windows_config, service)
    assert len(devices) == 1
    assert devices[0]["driver"] == "nvidia"
    assert devices[0]["device_ids"] == ["0"]
    assert devices[0]["capabilities"] == ["gpu"]


def test_gpu_services_stay_in_sync(default_config: dict) -> None:
    """app 與 train 共用同一顆卡，宣告分歧過就是漏改了其中一個。"""
    declarations = [_gpu_devices(default_config, svc) for svc in GPU_SERVICES]
    assert all(d == declarations[0] for d in declarations)
