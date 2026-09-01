from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_repo_modules(monkeypatch):
    for name, path in [
        ("voxcpm", SRC / "voxcpm"),
        ("voxcpm.model", SRC / "voxcpm" / "model"),
        ("voxcpm.modules", SRC / "voxcpm" / "modules"),
    ]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)

    hh = types.ModuleType("huggingface_hub")
    hh.snapshot_download = lambda *a, **k: "/tmp/fake"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hh)

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        @classmethod
        def model_rebuild(cls):
            return None

        @classmethod
        def model_validate_json(cls, s):
            return cls()

        def model_dump(self):
            return {}

    pydantic.BaseModel = BaseModel
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)

    torchaudio = types.ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)

    librosa = types.ModuleType("librosa")
    librosa.effects = types.SimpleNamespace(trim=lambda *a, **k: (None, (0, 0)))
    monkeypatch.setitem(sys.modules, "librosa", librosa)

    einops = types.ModuleType("einops")
    einops.rearrange = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "einops", einops)

    tqdm_pkg = types.ModuleType("tqdm")
    tqdm_pkg.__path__ = ["/nonexistent"]
    tqdm_pkg.tqdm = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_pkg)

    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto)

    transformers = types.ModuleType("transformers")

    class LlamaTokenizerFast:
        pass

    class PreTrainedTokenizer:
        pass

    transformers.LlamaTokenizerFast = LlamaTokenizerFast
    transformers.PreTrainedTokenizer = PreTrainedTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    internal_mods = {
        "voxcpm.modules.audiovae": ["AudioVAE", "AudioVAEConfig", "AudioVAEV2", "AudioVAEConfigV2"],
        "voxcpm.modules.layers": ["ScalarQuantizationLayer"],
        "voxcpm.modules.locdit": ["CfmConfig", "UnifiedCFM", "VoxCPMLocDiT", "VoxCPMLocDiTV2"],
        "voxcpm.modules.locenc": ["VoxCPMLocEnc"],
        "voxcpm.modules.minicpm4": ["MiniCPM4Config", "MiniCPMModel"],
        "voxcpm.modules.layers.lora": ["apply_lora_to_named_linear_modules", "LoRALinear"],
    }
    for modname, names in internal_mods.items():
        module = types.ModuleType(modname)
        for name in names:
            if name == "apply_lora_to_named_linear_modules":
                setattr(module, name, lambda *a, **k: None)
            else:
                setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, modname, module)

    _load_module("voxcpm.model.utils", SRC / "voxcpm" / "model" / "utils.py")
    voxcpm = _load_module("voxcpm.model.voxcpm", SRC / "voxcpm" / "model" / "voxcpm.py")
    voxcpm2 = _load_module("voxcpm.model.voxcpm2", SRC / "voxcpm" / "model" / "voxcpm2.py")
    return voxcpm.VoxCPMModel, voxcpm2.VoxCPM2Model


class DummyModel:
    device = "cpu"

    def named_parameters(self):
        return []


@pytest.mark.parametrize("module_name", ["v1", "v2"])
def test_load_lora_weights_accepts_tensor_only_legacy_checkpoints(monkeypatch, tmp_path, module_name):
    VoxCPMModel, VoxCPM2Model = bootstrap_repo_modules(monkeypatch)
    cls = VoxCPMModel if module_name == "v1" else VoxCPM2Model

    ckpt_path = tmp_path / "lora_weights.ckpt"
    torch.save({"state_dict": {"fake": torch.zeros(1)}}, ckpt_path)

    loaded, skipped = cls.load_lora_weights(DummyModel(), str(ckpt_path), device="cpu")

    assert loaded == []
    assert skipped == ["fake"]


@pytest.mark.parametrize("module_name", ["v1", "v2"])
def test_load_lora_weights_rejects_malicious_pickle_payloads(monkeypatch, tmp_path, module_name):
    VoxCPMModel, VoxCPM2Model = bootstrap_repo_modules(monkeypatch)
    cls = VoxCPMModel if module_name == "v1" else VoxCPM2Model

    ckpt_path = tmp_path / "lora_weights.ckpt"
    marker_path = tmp_path / f"{module_name}-marker.txt"

    class Exploit:
        def __reduce__(self):
            import pathlib

            return (pathlib.Path.write_text, (marker_path, f"{module_name} executed\n"))

    torch.save({"state_dict": {"fake": torch.zeros(1)}, "boom": Exploit()}, ckpt_path)

    with pytest.raises(Exception, match="Weights only load failed"):
        cls.load_lora_weights(DummyModel(), str(ckpt_path), device="cpu")

    assert not marker_path.exists()


def test_voxcpm_loads_lora_config_saved_with_checkpoint(monkeypatch, tmp_path):
    bootstrap_repo_modules(monkeypatch)
    core = _load_module("voxcpm.core", SRC / "voxcpm" / "core.py")
    captured = {}

    class FakeLoRAConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeTTSModel:
        sample_rate = 16_000

        @classmethod
        def from_local(cls, *args, **kwargs):
            captured["lora_config"] = kwargs["lora_config"]
            return cls()

        def load_lora_weights(self, path):
            captured["weights_path"] = path
            return [], []

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architecture": "voxcpm2"}',
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "lora"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "lora_config.json").write_text(
        '{"lora_config": {"r": 32, "alpha": 64}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(core, "LoRAConfig", FakeLoRAConfig)
    monkeypatch.setattr(core, "VoxCPM2Model", FakeTTSModel)

    core.VoxCPM(
        str(model_dir),
        enable_denoiser=False,
        optimize=False,
        lora_weights_path=str(checkpoint_dir),
    )

    assert captured["lora_config"].values == {"r": 32, "alpha": 64}
    assert captured["weights_path"] == str(checkpoint_dir)
