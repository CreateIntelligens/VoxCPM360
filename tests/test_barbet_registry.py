from __future__ import annotations

import json

from voxcpm.barbet_registry import (
    BarbetModelRegistry,
    discover_barbet_checkpoints,
)
from voxcpm.full_model_registry import discover_full_model_checkpoints


def _write_checkpoint(root, name, *, barbet=True, weight="model.safetensors"):
    path = root / name
    path.mkdir(parents=True)
    config = {"barbet_config": {}, "vox_lm_config": {}} if barbet else {}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "audiovae.pth").write_bytes(b"vae")
    (path / weight).write_bytes(b"weights")
    return path


def test_discovers_safetensors_and_pytorch_barbet_models(tmp_path):
    _write_checkpoint(tmp_path, "safetensors")
    _write_checkpoint(tmp_path, "pytorch", weight="pytorch_model.bin")

    checkpoints = discover_barbet_checkpoints([tmp_path])

    assert [checkpoint.name for checkpoint in checkpoints] == [
        "pytorch",
        "safetensors",
    ]
    assert all(checkpoint.valid for checkpoint in checkpoints)


def test_registry_uses_barbet_prefix_and_native_registry_ignores_it(tmp_path):
    path = _write_checkpoint(tmp_path, "tai8")

    registry = BarbetModelRegistry([tmp_path])

    assert registry.get("barbet::tai8").path == path
    assert discover_full_model_checkpoints([tmp_path]) == ()


def test_ignores_standalone_barbet_language_model(tmp_path):
    _write_checkpoint(tmp_path, "tts")
    language_model = tmp_path / "barbet-1b-base"
    language_model.mkdir()
    (language_model / "config.json").write_text(json.dumps({"model_type": "barbet"}), encoding="utf-8")
    (language_model / "model.safetensors").write_bytes(b"weights")

    checkpoints = discover_barbet_checkpoints([tmp_path])

    assert [checkpoint.name for checkpoint in checkpoints] == ["tts"]
