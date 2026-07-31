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


def test_catalog_exposes_native_engine(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), remote_models=(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert [engine["id"] for engine in payload["engines"]] == ["voxcpm2"]
    assert payload["engines"][0]["models"][0]["id"] == "__base__"


def test_native_synthesis_returns_wav(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, remote_models=(), mount_legacy=False)

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
    app = api.create_app(base_demo, remote_models=(), mount_legacy=False)

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
        assert any(
            model["id"] == "full::tai8"
            for model in catalog["engines"][0]["models"]
        )
        assert base_demo.stops == 1
        assert full_demo.loads == 1
        assert full_demo.calls[0]["model_selection"] == "__base__"


class FakeResponse:
    def __init__(
        self,
        *,
        payload=None,
        content=b"RIFFremote",
        headers=None,
        status_code=200,
    ):
        self._payload = payload or {}
        self.content = content
        self.headers = headers or {}
        self.text = ""
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_remote_model_catalog_and_synthesis(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    posts = []

    def fake_get(url, timeout):
        if url.endswith("/health"):
            return FakeResponse(payload={"status": "ok", "model_loaded": True, "gpu": "GB10"})
        if url.endswith("/api/models"):
            return FakeResponse(
                payload={
                    "models": [
                        {
                            "id": "barbet-tw-v1",
                            "label": "Barbet 台語 v1",
                            "online": True,
                            "loaded": True,
                        }
                    ]
                }
            )
        return FakeResponse(payload={"speakers": [{"id": "voice-a", "name": "Voice A"}]})

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return FakeResponse(headers={"X-Synthesis-Time": "1.25s"})

    monkeypatch.setattr(api.requests, "get", fake_get)
    monkeypatch.setattr(api.requests, "post", fake_post)
    remote = (
        api.RemoteModel(
            id="barbet-tw",
            label="Barbet 台語",
            base_url="http://bluemagpie:8000",
        ),
    )
    app = api.create_app(FakeDemo(), remote_models=remote, mount_legacy=False)

    with TestClient(app) as client:
        catalog = client.get("/api/v1/catalog").json()
        response = client.post(
            "/api/v1/synthesize",
            data={
                "engine_id": "bluemagpie",
                "model_id": "barbet-tw::barbet-tw-v1",
                "text": "逐家好",
                "speaker_id": "voice-a",
                "cfg_value": "2.2",
                "inference_timesteps": "10",
            },
        )

    assert catalog["engines"][1]["models"][0]["speakers"][0]["id"] == "voice-a"
    assert response.status_code == 200
    assert response.content == b"RIFFremote"
    assert response.headers["x-synthesis-time"] == "1.25s"
    assert posts[0][0].endswith("/api/synthesize")
    assert posts[0][1]["json"]["speaker_id"] == "voice-a"
    assert posts[0][1]["json"]["model_id"] == "barbet-tw-v1"
