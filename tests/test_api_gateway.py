from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import api


class FakeRegistry:
    checkpoints = ()

    def refresh(self):
        return ()

    def describe(self, selection):
        return selection


class FakeDemo:
    def __init__(self, model_id="base", device="cpu"):
        self.lora_registry = FakeRegistry()
        self.calls = []
        self.model_id = model_id
        self.device = device
        self.loads = 0
        self.stops = 0

    def get_or_load_voxcpm(self):
        self.loads += 1
        return object()

    def stop_voxcpm(self):
        self.stops += 1

    def generate_tts_audio(self, **kwargs):
        self.calls.append(kwargs)
        return 16_000, np.zeros(320, dtype=np.float32)


class FakeBarbetCheckpoint:
    id = "barbet::barbet-tw-v1"
    label = "Barbet 台語 v1"
    description = "本機測試模型"
    valid = True


class FakeBarbetRegistry:
    def __init__(self, checkpoints=()):
        self.checkpoints = tuple(checkpoints)

    def refresh(self):
        return self.checkpoints

    def get(self, model_id):
        checkpoint = next(
            (item for item in self.checkpoints if item.id == model_id),
            None,
        )
        if checkpoint is None:
            raise ValueError(f"找不到 Barbet 模型：{model_id}")
        return checkpoint


class FakeBarbetRuntime:
    def __init__(self, checkpoints=()):
        self.registry = FakeBarbetRegistry(checkpoints)
        self.loaded_model_id = None
        self.calls = []

    def speakers(self, checkpoint):
        del checkpoint
        return [{"id": "voice-a", "name": "Voice A"}]

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        self.loaded_model_id = kwargs["model_id"]
        return 48_000, np.zeros(480, dtype=np.float32), 1.25

    def close(self):
        return None


def test_catalog_exposes_native_engine(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert [engine["id"] for engine in payload["engines"]] == ["voxcpm2"]
    assert payload["engines"][0]["models"][0]["id"] == "__base__"


def test_catalog_exposes_available_reference_presets(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(api, "_REFERENCE_AUDIO_DIR", tmp_path)
    for preset in api._REFERENCE_AUDIO_PRESETS:
        (tmp_path / preset["filename"]).write_bytes(b"audio")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert [preset["label"] for preset in payload["reference_presets"]] == [
        "中年女聲",
        "中年男聲",
        "Hayley 開心說開場白",
    ]
    assert payload["default_reference_preset_id"] == "middle-aged-female"


def test_native_synthesis_uses_selected_reference_preset(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.delenv("VOXCPM_DEFAULT_REFERENCE", raising=False)
    monkeypatch.setattr(api, "_REFERENCE_AUDIO_DIR", tmp_path)
    selected_reference = tmp_path / "tai8_female_drama1_002.wav"
    selected_reference.write_bytes(b"RIFF")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/synthesize",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": "台語測試",
                "reference_preset_id": "middle-aged-male",
            },
        )

    assert response.status_code == 200
    assert demo.calls[0]["reference_wav_path_input"] == str(selected_reference)


def test_uploaded_reference_overrides_selected_preset(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/synthesize",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": "台語測試",
                "reference_preset_id": "not-a-valid-preset",
            },
            files={"reference_audio": ("custom.wav", b"RIFF", "audio/wav")},
        )

    assert response.status_code == 200
    assert demo.calls[0]["reference_wav_path_input"].endswith(".wav")


def test_native_synthesis_returns_wav(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/synthesize",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": "台語測試",
                "cfg_value": "2.0",
                "inference_timesteps": "8",
                "normalize": "false",
                "denoise": "false",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert demo.calls[0]["inference_timesteps"] == 8


def test_full_native_checkpoint_switches_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    full = tmp_path / "tai8"
    full.mkdir()
    for filename in ("config.json", "tokenizer.json"):
        (full / filename).write_text("{}", encoding="utf-8")
    (full / "model.safetensors").write_bytes(b"weights")
    (full / "audiovae.pth").write_bytes(b"vae")
    monkeypatch.setenv("VOXCPM_FULL_MODEL_ROOTS", str(tmp_path))

    base_demo = FakeDemo()
    full_demo = FakeDemo(model_id=str(full))
    monkeypatch.setattr(api, "VoxCPMDemo", lambda **kwargs: full_demo)
    app = api.create_app(base_demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        catalog = client.get("/api/v1/catalog").json()
        response = client.post(
            "/api/v1/synthesize",
            data={
                "engine_id": "voxcpm2",
                "model_id": "full::tai8",
                "text": "全參模型測試",
            },
        )

        assert response.status_code == 200
        assert any(model["id"] == "full::tai8" for model in catalog["engines"][0]["models"])
        assert base_demo.stops == 1
        assert full_demo.loads == 1
        assert full_demo.calls[0]["model_selection"] == "__base__"


def test_local_barbet_catalog_and_synthesis(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(),),
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        catalog = client.get("/api/v1/catalog").json()
        response = client.post(
            "/api/v1/synthesize",
            data={
                "engine_id": "barbet",
                "model_id": "barbet::barbet-tw-v1",
                "text": "逐家好",
                "speaker_id": "voice-a",
                "cfg_value": "2.2",
                "inference_timesteps": "10",
            },
        )

    assert catalog["engines"][1]["models"][0]["speakers"][0]["id"] == "voice-a"
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-synthesis-time"] == "1.25s"
    assert runtime.calls[0]["speaker_id"] == "voice-a"
    assert runtime.calls[0]["model_id"] == "barbet::barbet-tw-v1"
