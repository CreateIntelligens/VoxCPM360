from __future__ import annotations

from voxcpm.full_model_registry import (
    FullModelRegistry,
    discover_full_model_checkpoints,
)


def _write_full(root, name, *, complete=True):
    path = root / name
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "config.json").write_text("{}", encoding="utf-8")
    if complete:
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (path / "audiovae.pth").write_bytes(b"vae")
    return path


def test_discovers_complete_and_incomplete_full_models(tmp_path):
    _write_full(tmp_path, "complete")
    _write_full(tmp_path, "incomplete", complete=False)
    lora = tmp_path / "lora" / "latest"
    lora.mkdir(parents=True)
    (lora / "lora_weights.safetensors").write_bytes(b"lora")

    checkpoints = discover_full_model_checkpoints([tmp_path])

    assert [item.name for item in checkpoints] == ["complete", "incomplete"]
    assert checkpoints[0].valid is True
    assert checkpoints[1].valid is False
    assert "tokenizer.json" in checkpoints[1].missing_files


def test_registry_uses_prefixed_ids_and_first_root_wins(tmp_path):
    preferred = tmp_path / "models"
    legacy = tmp_path / "checkpoints"
    preferred_path = _write_full(preferred, "tai8")
    _write_full(legacy, "tai8")

    registry = FullModelRegistry([preferred, legacy])

    checkpoint = registry.get("full::tai8")
    assert checkpoint.path == preferred_path
