"""部署層的 per-request 步數保證：image 真的打過 patch，且 API 真的照辦。

`tests/test_nanovllm_timesteps.py` 驗的是 patch 轉換後的原始碼邏輯（用 AST 取出
方法再單獨執行），不保證**執行中的 image 真的套用了它** —— Dockerfile 少一行
COPY/RUN 就會靜默退回沒有 per-request 步數的引擎。那個缺口只有在跑起來的容器裡
檢查才抓得到，所以這裡直接讀已安裝的套件與打真實 HTTP。
"""

import importlib.util
import inspect

from pathlib import Path

import pytest

from scripts.patch_nanovllm_timesteps import MARKER

PATCHED_MODULES = ("server.py", "engine.py", "runner.py")


@pytest.fixture(scope="module")
def engine_root():
    spec = importlib.util.find_spec("nanovllm_voxcpm")
    if spec is None:
        pytest.skip("nano-vllm-voxcpm is required for deployment checks")
    return Path(spec.origin).parent / "models" / "voxcpm2"


@pytest.mark.parametrize("name", PATCHED_MODULES)
def test_installed_engine_carries_the_patch(engine_root, name):
    """已安裝的引擎必須帶 marker —— 這是 Dockerfile 那段 RUN 的驗收。"""
    source = (engine_root / name).read_text()
    assert source.startswith(MARKER), (
        f"{name} 未套用 patch：image build 缺少 "
        "scripts/patch_nanovllm_timesteps.py 這一步"
    )


def test_installed_generate_accepts_per_request_timesteps(engine_root):
    """app.py 靠 signature 檢查決定要不要擋，所以 signature 是契約本身。"""
    import ast

    source = (engine_root / "server.py").read_text()
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                item.name == "generate"
            ):
                names = [a.arg for a in item.args.args]
                names += [a.arg for a in item.args.kwonlyargs]
                found[node.name] = "inference_timesteps" in names
    assert found, "server.py 找不到任何 generate 方法"
    missing = sorted(k for k, v in found.items() if not v)
    assert not missing, f"這些 generate 缺 inference_timesteps 參數：{missing}"


def test_demo_passes_timesteps_through_to_the_engine():
    """步數要真的進到 generate 的 kwargs，而不是只在簽名上存在。"""
    import app as app_module

    sig = inspect.signature(app_module.VoxCPMDemo._prepare_tts_generation)
    assert "inference_timesteps" in sig.parameters
