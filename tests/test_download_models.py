from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_models.py"

spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
download_models = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(download_models)


def test_verify_flash_attn_passes_when_importable(capsys):
    download_models.verify_flash_attn()

    assert "ABI OK" in capsys.readouterr().out


def test_verify_flash_attn_explains_abi_mismatch(monkeypatch):
    """ABI 不合時 import 只會拋出 undefined symbol，看不出真正的原因。"""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("undefined symbol: _ZN3c105ErrorC1E")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="TORCH_VERSION"):
        download_models.verify_flash_attn()
