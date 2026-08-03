from __future__ import annotations

import json

import pytest

from voxcpm.lora_registry import (
    BASE_MODEL_KEY,
    LoraRegistry,
    build_nano_lora_config,
    discover_lora_checkpoints,
)


def _write_checkpoint(root, run_name, config=None, *, latest=True):
    checkpoint_dir = root / run_name / "latest" if latest else root / run_name
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "lora_weights.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "lora_config.json").write_text(
        json.dumps(
            config
            or {
                "base_model": "openbmb/VoxCPM2",
                "lora_config": {
                    "enable_lm": True,
                    "enable_dit": True,
                    "enable_proj": False,
                    "r": 32,
                    "target_modules_lm": ["q_proj", "v_proj"],
                    "target_modules_dit": ["k_proj", "o_proj"],
                    "target_proj_modules": ["enc_to_lm_proj"],
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_dir


def test_discovers_only_complete_latest_checkpoints(tmp_path):
    second = _write_checkpoint(tmp_path, "zeta")
    first = _write_checkpoint(tmp_path, "alpha")
    step = tmp_path / "alpha" / "step_0000001"
    step.mkdir()
    (step / "lora_config.json").write_text("{}", encoding="utf-8")
    incomplete = tmp_path / "incomplete" / "latest"
    incomplete.mkdir(parents=True)
    (incomplete / "lora_config.json").write_text("{}", encoding="utf-8")

    checkpoints = discover_lora_checkpoints(tmp_path)

    assert [checkpoint.run_name for checkpoint in checkpoints] == ["alpha", "zeta"]
    assert [checkpoint.path for checkpoint in checkpoints] == [first, second]


def test_discovers_flat_checkpoint_directory(tmp_path):
    flat = _write_checkpoint(tmp_path, "flat", latest=False)

    checkpoints = discover_lora_checkpoints(tmp_path)

    assert [checkpoint.run_name for checkpoint in checkpoints] == ["flat"]
    assert checkpoints[0].path == flat


def test_registry_scans_multiple_roots_with_first_root_winning(tmp_path):
    preferred_root = tmp_path / "models"
    legacy_root = tmp_path / "checkpoints"
    preferred = _write_checkpoint(preferred_root, "shared")
    _write_checkpoint(legacy_root, "shared")
    legacy_only = _write_checkpoint(legacy_root, "legacy")

    registry = LoraRegistry(preferred_root, additional_roots=[legacy_root])

    assert [item.run_name for item in registry.checkpoints] == ["shared", "legacy"]
    assert registry.checkpoints[0].path == preferred
    assert registry.checkpoints[1].path == legacy_only


def test_ignores_malformed_checkpoint_config(tmp_path):
    latest = tmp_path / "broken" / "latest"
    latest.mkdir(parents=True)
    (latest / "lora_weights.safetensors").write_bytes(b"weights")
    (latest / "lora_config.json").write_text("{", encoding="utf-8")

    assert discover_lora_checkpoints(tmp_path) == ()


def test_builds_union_of_nano_lora_requirements(tmp_path):
    _write_checkpoint(tmp_path, "rank-16")
    _write_checkpoint(
        tmp_path,
        "rank-64",
        {
            "lora_config": {
                "enable_lm": True,
                "enable_dit": False,
                "enable_proj": True,
                "r": 64,
                "target_modules_lm": ["k_proj"],
                "target_modules_dit": [],
                "target_proj_modules": ["fusion_concat_proj"],
            }
        },
    )

    config = build_nano_lora_config(discover_lora_checkpoints(tmp_path))

    assert config["max_loras"] == 1
    assert config["max_lora_rank"] == 64
    assert config["enable_lm"] is True
    assert config["enable_dit"] is True
    assert config["enable_proj"] is True
    assert config["target_modules_lm"] == ["q_proj", "k_proj", "v_proj"]
    assert config["target_modules_dit"] == ["k_proj", "o_proj"]
    assert config["target_proj_modules"] == [
        "enc_to_lm_proj",
        "fusion_concat_proj",
    ]


class FakeServer:
    def __init__(self):
        self.calls = []

    def register_lora(self, name, path):
        self.calls.append((name, path))
        return {"name": name}


def test_registry_registers_selected_adapter_once(tmp_path):
    latest = _write_checkpoint(tmp_path, "trial")
    registry = LoraRegistry(tmp_path)
    server = FakeServer()

    assert registry.ensure_registered(server, BASE_MODEL_KEY) is None
    first_name = registry.ensure_registered(server, "trial")
    second_name = registry.ensure_registered(server, "trial")

    assert first_name == second_name
    assert server.calls == [(first_name, str(latest))]


def test_registry_rejects_a_stale_selection(tmp_path):
    registry = LoraRegistry(tmp_path)

    with pytest.raises(ValueError, match="重新掃描"):
        registry.ensure_registered(FakeServer(), "removed")
