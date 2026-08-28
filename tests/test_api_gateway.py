from __future__ import annotations

import asyncio
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

import api


@pytest.fixture(autouse=True)
def isolate_generation_history(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_HISTORY_DIR", tmp_path / "generation_history")
    monkeypatch.setattr(api, "_CASTVOICE_BATCH_DIR", tmp_path / "castvoice_batches")


class FakeRegistry:
    def __init__(self, checkpoints=()):
        self.checkpoints = tuple(checkpoints)

    def refresh(self):
        return self.checkpoints

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


class FakeStreamingServer:
    def __init__(self, sample_rate=16_000):
        self.sample_rate = sample_rate

    def get_model_info(self):
        return {"sample_rate": self.sample_rate}


class FakeStreamingDemo(FakeDemo):
    def __init__(self, chunks=None, *, error_after=None):
        super().__init__()
        self.server = FakeStreamingServer()
        self.chunks = chunks or [
            np.array([0.1, -0.1], dtype=np.float32),
            np.array([0.2, -0.2, 0.0], dtype=np.float32),
            np.array([0.3], dtype=np.float32),
        ]
        self.error_after = error_after
        self.stream_calls = []

    def get_or_load_voxcpm(self):
        self.loads += 1
        return self.server

    def generate_tts_audio_stream(self, request):
        self.stream_calls.append(request)
        for index, chunk in enumerate(self.chunks):
            if self.error_after == index:
                raise RuntimeError("stream failed")
            yield chunk
        if self.error_after == len(self.chunks):
            raise RuntimeError("stream failed")


class FakeBatchDemo(FakeDemo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_calls = []

    def generate_tts_audio_batch(self, requests):
        self.batch_calls.append(requests)
        return [
            (16_000, np.zeros(320, dtype=np.float32)) for _ in requests
        ]


class FakeOOMBatchDemo(FakeBatchDemo):
    def generate_tts_audio_batch(self, requests):
        self.batch_calls.append(requests)
        if len(requests) > 2:
            raise RuntimeError("CUDA out of memory")
        return [
            (16_000, np.zeros(320, dtype=np.float32)) for _ in requests
        ]


class FakeLoraCheckpoint:
    run_name = "lora-test"
    label = "LoRA test"


def wait_for_batch(client: TestClient, batch_id: str) -> dict:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/tts/synthesize/batch/{batch_id}").json()
        if status["done"]:
            return status
        time.sleep(0.05)
    raise AssertionError("batch did not finish in time")


def test_generation_history_partial_replace_is_rolled_back(monkeypatch):
    history_id = "a" * 32
    original_replace = api.os.replace
    replace_count = 0

    def fail_second_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("metadata replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(api.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="metadata replace failed"):
        api._save_generation_history({"id": history_id}, b"RIFF")

    assert list(api._HISTORY_DIR.glob("*")) == []


class FakeBarbetCheckpoint:
    id = "barbet::barbet-tw-v1"
    label = "Barbet 台語 v1"
    description = "本機測試模型"
    valid = True


class FakeBarbetCheckpointV2:
    id = "barbet::barbet-tw-v2"
    label = "Barbet 台語 v2"
    description = "本機測試模型 v2"
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
    def __init__(
        self,
        checkpoints=(),
        speakers=({"id": "voice-a", "name": "Voice A", "gender": "unknown"},),
        speakers_by_checkpoint=None,
    ):
        self.registry = FakeBarbetRegistry(checkpoints)
        self.loaded_model_id = None
        self.calls = []
        self._speakers = list(speakers)
        self._speakers_by_checkpoint = speakers_by_checkpoint or {}

    def speakers(self, checkpoint):
        if checkpoint.id in self._speakers_by_checkpoint:
            return self._speakers_by_checkpoint[checkpoint.id]
        return self._speakers

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
    assert payload["engines"][0]["models"][0]["id"] == "base::__base__"
    assert payload["engines"][0]["capabilities"]["streaming"] is True


def test_catalog_namespaces_lora_and_accepts_legacy_alias(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    demo.lora_registry = FakeRegistry((FakeLoraCheckpoint(),))
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    model_ids = [model["id"] for model in gateway.catalog()["engines"][0]["models"]]
    _, runtime_selection, canonical_id = gateway._switch_native_runtime("lora-test")

    assert "lora::lora-test" in model_ids
    assert runtime_selection == "lora-test"
    assert canonical_id == "lora::lora-test"


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
        preset["label"] for preset in api._REFERENCE_AUDIO_PRESETS
    ]
    assert [preset["language"] for preset in payload["reference_presets"]] == [
        preset["language"] for preset in api._REFERENCE_AUDIO_PRESETS
    ]
    assert payload["default_reference_preset_id"] == "cosy-young-female-01"


def test_model_registry_endpoint_returns_document(monkeypatch, tmp_path):
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(
        '{"_val_sets":{"tai8":"同一驗證集"},"models":[{"name":"tai8-best"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "_MODEL_REGISTRY_PATH", registry_path)
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/models/registry")

    assert response.status_code == 200
    assert response.json() == {
        "_val_sets": {"tai8": "同一驗證集"},
        "models": [{"name": "tai8-best"}],
    }


def test_model_registry_endpoint_returns_empty_document_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_MODEL_REGISTRY_PATH", tmp_path / "missing.json")
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/models/registry")

    assert response.status_code == 200
    assert response.json() == {"_val_sets": {}, "models": []}


def test_native_synthesis_uses_selected_reference_preset(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.delenv("VOXCPM_DEFAULT_REFERENCE", raising=False)
    monkeypatch.setattr(api, "_REFERENCE_AUDIO_DIR", tmp_path)
    selected_reference = tmp_path / "cosy-young-female-01.mp3"
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
                "reference_preset_id": "cosy-young-female-01",
            },
        )

    assert response.status_code == 200
    assert demo.calls[0]["reference_wav_path_input"] == str(selected_reference)
    assert demo.calls[0]["prompt_text"] == api._COSY_PROMPT_TEXT
    assert demo.calls[0]["control_instruction"] == ""


def test_native_prompt_cloning_drops_explicit_control_instruction(monkeypatch):
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
                "control_instruction": "用女聲說",
                "prompt_text": "參考音訊內容",
            },
            files={"reference_audio": ("custom.wav", b"RIFF", "audio/wav")},
        )
        history_item = client.get("/api/v1/history?limit=1").json()["items"][0]

    assert response.status_code == 200
    assert demo.calls[0]["control_instruction"] == ""
    assert demo.calls[0]["prompt_text"] == "參考音訊內容"
    assert history_item["control_instruction"] is None


def test_native_control_instruction_is_preserved_without_prompt_transcript(monkeypatch):
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
                "control_instruction": "溫暖沉穩",
            },
            files={"reference_audio": ("custom.wav", b"RIFF", "audio/wav")},
        )

    assert response.status_code == 200
    assert demo.calls[0]["control_instruction"] == "溫暖沉穩"
    assert demo.calls[0]["prompt_text"] == ""


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
    assert response.headers["x-request-id"]
    assert response.headers["x-queue-wait"].endswith("s")
    assert response.headers["x-gpu-job-time"].endswith("s")
    assert response.headers["x-total-time"].endswith("s")
    assert demo.calls[0]["inference_timesteps"] == 8


def test_native_streaming_synthesis_returns_unknown_length_wav_and_history(
    monkeypatch,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeStreamingDemo(
        chunks=[
            np.array([0.9, -0.9], dtype=np.float32),
            np.array([0.1, 0.0, -0.1], dtype=np.float32),
            np.array([0.2], dtype=np.float32),
        ]
    )
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/synthesize/stream",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": "台語串流測試",
            },
        )
        history = client.get("/api/v1/history?limit=1").json()["items"]

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert "content-length" not in response.headers
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-sample-rate"] == "16000"
    assert response.headers["x-request-id"]
    assert response.headers["x-history-id"] == history[0]["id"]
    assert response.headers["x-model-engine"] == "voxcpm2"
    assert response.headers["x-model-version"] == "base::__base__"
    assert response.headers["content-disposition"] == (
        'inline; filename="tts-output.wav"'
    )
    assert "x-queue-wait" not in response.headers
    assert "x-gpu-job-time" not in response.headers
    assert "x-synthesis-time" not in response.headers
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    assert int.from_bytes(response.content[4:8], "little") == 0xFFFFFFFF
    assert int.from_bytes(response.content[20:22], "little") == 1
    assert int.from_bytes(response.content[22:24], "little") == 1
    assert int.from_bytes(response.content[24:28], "little") == 16_000
    assert int.from_bytes(response.content[28:32], "little") == 32_000
    assert int.from_bytes(response.content[32:34], "little") == 2
    assert int.from_bytes(response.content[34:36], "little") == 16
    assert int.from_bytes(response.content[40:44], "little") == 0xFFFFFFFF
    assert len(response.content[44:]) == 6 * 2
    pcm = np.frombuffer(response.content[44:48], dtype="<i2")
    assert np.max(np.abs(pcm)) == pytest.approx(0.35 * 32767, abs=1)


def test_native_streaming_preserves_preset_cloning_rules(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(api, "_REFERENCE_AUDIO_DIR", tmp_path)
    reference = tmp_path / "cosy-young-female-01.mp3"
    reference.write_bytes(b"RIFF")
    demo = FakeStreamingDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/synthesize/stream",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": "台語串流測試",
                "reference_preset_id": "cosy-young-female-01",
                "control_instruction": "用女聲說",
            },
        )

    assert response.status_code == 200
    assert demo.stream_calls[0]["reference_wav_path_input"] == str(reference)
    assert demo.stream_calls[0]["prompt_text"] == api._COSY_PROMPT_TEXT
    assert demo.stream_calls[0]["control_instruction"] == ""


def test_native_streaming_removes_uploaded_reference_after_response(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    demo = FakeStreamingDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/synthesize/stream",
            data={
                "engine_id": "voxcpm2",
                "model_id": "__base__",
                "text": "台語串流測試",
                "prompt_text": "參考音訊內容",
            },
            files={"reference_audio": ("custom.wav", b"RIFF", "audio/wav")},
        )

    uploaded_path = Path(demo.stream_calls[0]["reference_wav_path_input"])
    assert response.status_code == 200
    assert not uploaded_path.exists()


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        (
            {"engine_id": "barbet"},
            "串流端點目前僅支援 voxcpm2 引擎",
        ),
        ({"speed": "1.5"}, "串流端點不支援語速調整"),
    ],
)
def test_native_streaming_rejects_unsupported_options(
    monkeypatch,
    overrides,
    detail,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(
        FakeStreamingDemo(),
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    payload = {
        "engine_id": "voxcpm2",
        "model_id": "__base__",
        "text": "台語串流測試",
        **overrides,
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/synthesize/stream", data=payload)

    assert response.status_code == 422
    assert response.json()["detail"] == detail


def test_native_streaming_queue_full_is_rejected_before_response_start(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "0")
    worker_blocked = threading.Event()
    release_worker = threading.Event()

    class BlockingStreamingDemo(FakeStreamingDemo):
        def generate_tts_audio_stream(self, request):
            yield np.array([0.1], dtype=np.float32)
            worker_blocked.set()
            assert release_worker.wait(timeout=2)
            yield np.array([0.2], dtype=np.float32)

    app = api.create_app(
        BlockingStreamingDemo(),
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/synthesize/stream"
    )

    async def run_scenario():
        active_stream = await app.state.tts_gateway.synthesize_native_stream(
            request_id="active-request",
            model_id="__base__",
            text="第一筆",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=True,
            denoise=False,
            inference_timesteps=30,
        )
        assert await asyncio.to_thread(worker_blocked.wait, 1)

        with pytest.raises(api.HTTPException) as exc_info:
            await endpoint(
                request=api.Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/api/v1/synthesize/stream",
                        "headers": [],
                    }
                ),
                engine_id="voxcpm2",
                model_id="__base__",
                text="第二筆",
                control_instruction="",
                prompt_text="",
                reference_preset_id="",
                speaker_id="",
                cfg_value=2.0,
                inference_timesteps=30,
                normalize=True,
                denoise=False,
                speed=1.0,
                seed=None,
                reference_audio=None,
            )
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers["X-Request-ID"]
        assert app.state.tts_gateway._inflight_jobs == 1

        release_worker.set()
        assert [chunk async for chunk in active_stream]
        await active_stream.aclose()
        assert app.state.tts_gateway._inflight_jobs == 0

    asyncio.run(run_scenario())


def test_native_streaming_queue_timeout_returns_503_and_recovers(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "1")
    monkeypatch.setenv("VOXCPM_QUEUE_TIMEOUT_SECONDS", "0.02")
    worker_blocked = threading.Event()
    release_worker = threading.Event()

    class BlockingStreamingDemo(FakeStreamingDemo):
        def generate_tts_audio_stream(self, request):
            yield np.array([0.1], dtype=np.float32)
            worker_blocked.set()
            assert release_worker.wait(timeout=2)
            yield np.array([0.2], dtype=np.float32)

    app = api.create_app(
        BlockingStreamingDemo(),
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    async def run_scenario():
        active_stream = await app.state.tts_gateway.synthesize_native_stream(
            request_id="active-request",
            model_id="__base__",
            text="第一筆",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=True,
            denoise=False,
            inference_timesteps=30,
        )
        assert await asyncio.to_thread(worker_blocked.wait, 1)

        with pytest.raises(api.HTTPException) as exc_info:
            await app.state.tts_gateway.synthesize_native_stream(
                request_id="timed-out-request",
                model_id="__base__",
                text="第二筆",
                control_instruction="",
                reference_path=None,
                prompt_text="",
                cfg_value=2.0,
                normalize=True,
                denoise=False,
                inference_timesteps=30,
            )
        assert exc_info.value.status_code == 503
        assert exc_info.value.headers["X-Request-ID"] == "timed-out-request"
        assert app.state.tts_gateway._inflight_jobs == 1

        release_worker.set()
        assert [chunk async for chunk in active_stream]
        await active_stream.aclose()
        assert app.state.tts_gateway._inflight_jobs == 0

        next_stream = await app.state.tts_gateway.synthesize_native_stream(
            request_id="next-request",
            model_id="__base__",
            text="第三筆",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=True,
            denoise=False,
            inference_timesteps=30,
        )
        assert [chunk async for chunk in next_stream]
        await next_stream.aclose()

    asyncio.run(run_scenario())


def test_native_streaming_final_send_failure_rolls_back_history(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    demo = FakeStreamingDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/synthesize/stream"
    )

    async def run_scenario():
        request = api.Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/synthesize/stream",
                "headers": [],
            }
        )
        response = await endpoint(
            request=request,
            engine_id="voxcpm2",
            model_id="__base__",
            text="台語串流測試",
            control_instruction="",
            prompt_text="參考音訊內容",
            reference_preset_id="",
            speaker_id="",
            cfg_value=2.0,
            inference_timesteps=30,
            normalize=True,
            denoise=False,
            speed=1.0,
            seed=None,
            reference_audio=api.UploadFile(
                file=io.BytesIO(b"RIFF"),
                filename="custom.wav",
            ),
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                raise OSError("client disconnected before final body")

        with pytest.raises(ClientDisconnect):
            await response(
                {
                    "type": "http",
                    "asgi": {"spec_version": "2.4"},
                },
                receive,
                send,
            )
        assert app.state.tts_gateway._inflight_jobs == 0

    asyncio.run(run_scenario())
    assert list(api._HISTORY_DIR.glob("*")) == []
    uploaded_path = Path(demo.stream_calls[0]["reference_wav_path_input"])
    assert not uploaded_path.exists()


def test_native_streaming_generation_failure_does_not_write_history(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    demo = FakeStreamingDemo(error_after=1)
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/synthesize/stream"
    )

    async def run_scenario():
        response = await endpoint(
            request=api.Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/api/v1/synthesize/stream",
                    "headers": [],
                }
            ),
            engine_id="voxcpm2",
            model_id="__base__",
            text="台語串流測試",
            control_instruction="",
            prompt_text="參考音訊內容",
            reference_preset_id="",
            speaker_id="",
            cfg_value=2.0,
            inference_timesteps=30,
            normalize=True,
            denoise=False,
            speed=1.0,
            seed=None,
            reference_audio=api.UploadFile(
                file=io.BytesIO(b"RIFF"),
                filename="custom.wav",
            ),
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            return None

        with pytest.raises(RuntimeError, match="stream failed"):
            await response(
                {
                    "type": "http",
                    "asgi": {"spec_version": "2.4"},
                },
                receive,
                send,
            )
        assert app.state.tts_gateway._inflight_jobs == 0

    asyncio.run(run_scenario())
    assert list(api._HISTORY_DIR.glob("*")) == []
    uploaded_path = Path(demo.stream_calls[0]["reference_wav_path_input"])
    assert not uploaded_path.exists()


def test_native_streaming_failure_before_first_chunk_keeps_response_unstarted(
    monkeypatch,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    app = api.create_app(
        FakeStreamingDemo(error_after=0),
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    async def run_scenario():
        with pytest.raises(RuntimeError, match="stream failed"):
            await app.state.tts_gateway.synthesize_native_stream(
                request_id="failed-request",
                model_id="__base__",
                text="台語串流測試",
                control_instruction="",
                reference_path=None,
                prompt_text="",
                cfg_value=2.0,
                normalize=True,
                denoise=False,
                inference_timesteps=30,
            )
        assert app.state.tts_gateway._inflight_jobs == 0
        assert not app.state.tts_gateway._gpu_lock.locked()

    asyncio.run(run_scenario())


def test_native_streaming_asgi23_disconnect_during_commit_rolls_back_history(
    monkeypatch,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    commit_started = threading.Event()
    release_commit = threading.Event()
    original_save = api._save_generation_history

    def blocking_save(record, wav):
        commit_started.set()
        assert release_commit.wait(timeout=2)
        original_save(record, wav)

    monkeypatch.setattr(api, "_save_generation_history", blocking_save)
    app = api.create_app(
        FakeStreamingDemo(),
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/synthesize/stream"
    )

    async def run_scenario():
        response = await endpoint(
            request=api.Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/api/v1/synthesize/stream",
                    "headers": [],
                }
            ),
            engine_id="voxcpm2",
            model_id="__base__",
            text="台語串流測試",
            control_instruction="",
            prompt_text="",
            reference_preset_id="",
            speaker_id="",
            cfg_value=2.0,
            inference_timesteps=30,
            normalize=True,
            denoise=False,
            speed=1.0,
            seed=None,
            reference_audio=None,
        )

        async def receive():
            await asyncio.to_thread(commit_started.wait, 2)
            return {"type": "http.disconnect"}

        async def send(message):
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                await asyncio.sleep(0)

        response_task = asyncio.create_task(
            response(
                {
                    "type": "http",
                    "asgi": {"spec_version": "2.3"},
                },
                receive,
                send,
            )
        )
        assert await asyncio.to_thread(commit_started.wait, 1)
        await asyncio.sleep(0.02)
        release_commit.set()
        await response_task
        assert app.state.tts_gateway._inflight_jobs == 0

    asyncio.run(run_scenario())
    assert list(api._HISTORY_DIR.glob("*")) == []


def test_native_streaming_asgi23_disconnect_cleans_worker_gate_and_upload(
    monkeypatch,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    worker_blocked = threading.Event()
    release_worker = threading.Event()
    worker_closed = threading.Event()

    class BlockingStreamingDemo(FakeStreamingDemo):
        def generate_tts_audio_stream(self, request):
            self.stream_calls.append(request)
            try:
                yield np.array([0.1], dtype=np.float32)
                worker_blocked.set()
                assert release_worker.wait(timeout=2)
                yield np.array([0.2], dtype=np.float32)
            finally:
                worker_closed.set()

    demo = BlockingStreamingDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/synthesize/stream"
    )

    async def run_scenario():
        response = await endpoint(
            request=api.Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/api/v1/synthesize/stream",
                    "headers": [],
                }
            ),
            engine_id="voxcpm2",
            model_id="__base__",
            text="台語串流測試",
            control_instruction="",
            prompt_text="參考音訊內容",
            reference_preset_id="",
            speaker_id="",
            cfg_value=2.0,
            inference_timesteps=30,
            normalize=True,
            denoise=False,
            speed=1.0,
            seed=None,
            reference_audio=api.UploadFile(
                file=io.BytesIO(b"RIFF"),
                filename="custom.wav",
            ),
        )
        first_pcm_sent = asyncio.Event()
        body_count = 0

        async def receive():
            await first_pcm_sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal body_count
            if message["type"] == "http.response.body":
                body_count += 1
                if body_count == 2:
                    first_pcm_sent.set()

        response_task = asyncio.create_task(
            response(
                {
                    "type": "http",
                    "asgi": {"spec_version": "2.3"},
                },
                receive,
                send,
            )
        )
        await asyncio.wait_for(first_pcm_sent.wait(), timeout=1)
        assert await asyncio.to_thread(worker_blocked.wait, 1)
        await asyncio.sleep(0.02)
        assert not response_task.done()
        assert app.state.tts_gateway._inflight_jobs == 1

        release_worker.set()
        await response_task
        assert worker_closed.is_set()
        assert app.state.tts_gateway._inflight_jobs == 0

        next_stream = await app.state.tts_gateway.synthesize_native_stream(
            request_id="next-request",
            model_id="__base__",
            text="第二筆",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=True,
            denoise=False,
            inference_timesteps=30,
        )
        release_worker.set()
        assert [chunk async for chunk in next_stream]
        await next_stream.aclose()

    asyncio.run(run_scenario())
    uploaded_path = Path(demo.stream_calls[0]["reference_wav_path_input"])
    assert not uploaded_path.exists()
    assert list(api._HISTORY_DIR.glob("*")) == []


def test_native_streaming_disconnect_waits_for_worker_before_releasing_gate(
    monkeypatch,
):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_DEFAULT_REFERENCE", "")
    worker_blocked = threading.Event()
    release_worker = threading.Event()
    worker_closed = threading.Event()

    class BlockingStreamingDemo(FakeStreamingDemo):
        def __init__(self):
            super().__init__()
            self.stream_count = 0

        def generate_tts_audio_stream(self, request):
            self.stream_count += 1
            try:
                yield np.array([0.1], dtype=np.float32)
                if self.stream_count == 1:
                    worker_blocked.set()
                    assert release_worker.wait(timeout=2)
                yield np.array([0.2], dtype=np.float32)
            finally:
                worker_closed.set()

    demo = BlockingStreamingDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/synthesize/stream"
    )

    async def run_scenario():
        request = api.Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/synthesize/stream",
                "headers": [],
            }
        )
        response = await endpoint(
            request=request,
            engine_id="voxcpm2",
            model_id="__base__",
            text="第一筆",
            control_instruction="",
            prompt_text="",
            reference_preset_id="",
            speaker_id="",
            cfg_value=2.0,
            inference_timesteps=30,
            normalize=True,
            denoise=False,
            speed=1.0,
            seed=None,
            reference_audio=None,
        )
        pcm_sent = asyncio.Event()
        body_count = 0

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal body_count
            if message["type"] != "http.response.body":
                return
            body_count += 1
            if body_count == 2:
                pcm_sent.set()
                raise OSError("client disconnected during stream")

        response_task = asyncio.create_task(
            response(
                {
                    "type": "http",
                    "asgi": {"spec_version": "2.4"},
                },
                receive,
                send,
            )
        )
        await asyncio.wait_for(pcm_sent.wait(), timeout=1)
        assert await asyncio.to_thread(worker_blocked.wait, 1)
        await asyncio.sleep(0.02)
        assert not response_task.done()
        assert app.state.tts_gateway._inflight_jobs == 1

        release_worker.set()
        with pytest.raises(ClientDisconnect):
            await response_task
        assert worker_closed.is_set()
        assert app.state.tts_gateway._inflight_jobs == 0

        next_stream = await app.state.tts_gateway.synthesize_native_stream(
            request_id="next-request",
            model_id="__base__",
            text="第二筆",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=True,
            denoise=False,
            inference_timesteps=30,
        )
        assert [chunk async for chunk in next_stream]
        await next_stream.aclose()
        assert app.state.tts_gateway._inflight_jobs == 0

    asyncio.run(run_scenario())


def test_demo_streaming_close_acloses_async_backend_generator():
    backend_closed = threading.Event()

    class AsyncPool:
        async def generate(self, **kwargs):
            try:
                yield np.array([0.1], dtype=np.float32)
                yield np.array([0.2], dtype=np.float32)
            finally:
                backend_closed.set()

    class Server:
        def __init__(self):
            self.server_pool = AsyncPool()
            self.loop = asyncio.new_event_loop()

    server = Server()
    demo = object.__new__(api.VoxCPMDemo)
    demo.get_or_load_voxcpm = lambda: server
    demo._prepare_tts_generation = lambda *args, **kwargs: {}

    stream = demo.generate_tts_audio_stream({})
    assert np.array_equal(next(stream), np.array([0.1], dtype=np.float32))
    stream.close()

    assert backend_closed.is_set()
    server.loop.close()


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
                # Bare directory names remain accepted as a compatibility alias.
                "model_id": "tai8",
                "text": "全參模型測試",
            },
        )

        assert response.status_code == 200
        assert any(model["id"] == "full::tai8" for model in catalog["engines"][0]["models"])
        assert base_demo.stops == 1
        assert full_demo.loads == 1
        assert full_demo.calls[0]["model_selection"] == "__base__"


def test_dynamic_batch_sizer_uses_cuda_and_cgroup_headroom(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cuda_headroom", staticmethod(lambda: 10 * gib)
    )
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cgroup_headroom", staticmethod(lambda: 20 * gib)
    )
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_RESERVE_GIB", "2")
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_PER_ITEM_GIB", "1.5")

    sizer = api.DynamicBatchSizer(max_concurrency=16)

    assert sizer.recommend(20) == 5
    assert sizer.recommend(3) == 3


def test_dynamic_batch_sizer_falls_back_to_one_without_memory_signal(monkeypatch):
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cuda_headroom", staticmethod(lambda: None)
    )
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cgroup_headroom", staticmethod(lambda: None)
    )

    assert api.DynamicBatchSizer(max_concurrency=16).recommend(20) == 1


def test_dynamic_batch_sizer_learns_headroom_and_shrinks_after_oom(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cuda_headroom", staticmethod(lambda: 10 * gib)
    )
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cgroup_headroom", staticmethod(lambda: 20 * gib)
    )
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_RESERVE_GIB", "2")
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_PER_ITEM_GIB", "1.5")
    sizer = api.DynamicBatchSizer(max_concurrency=16)

    assert sizer.recommend(20) == 5
    sizer.observe_success(size=5, elapsed=5.0, work_units=50)
    assert sizer.recommend(20) == 7

    sizer.observe_oom(size=7)
    assert sizer.recommend(20) == 3


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
                "prompt_text": "逐家好，食飽未",
                "speaker_id": "voice-a",
                "cfg_value": "2.2",
                "inference_timesteps": "10",
            },
        )
        history_response = client.get("/api/v1/history?limit=5")
        history_item = history_response.json()["items"][0]
        audio_response = client.get(history_item["audio_url"])

    barbet_engine = catalog["engines"][1]
    assert barbet_engine["capabilities"]["streaming"] is False
    assert barbet_engine["capabilities"]["prompt_transcript"] is True
    assert barbet_engine["models"][0]["speakers"][0]["id"] == "voice-a"
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-synthesis-time"] == "1.25s"
    assert response.headers["x-history-id"] == history_item["id"]
    actual_seed = int(response.headers["x-random-seed"])
    assert runtime.calls[0]["speaker_id"] == "voice-a"
    assert runtime.calls[0]["model_id"] == "barbet::barbet-tw-v1"
    assert runtime.calls[0]["prompt_text"] == "逐家好，食飽未"
    assert runtime.calls[0]["seed"] == actual_seed
    assert history_response.status_code == 200
    assert history_item["model_label"] == "Barbet 台語 v1"
    assert history_item["reference_label"].startswith("青年女聲 01")
    assert history_item["seed"] == actual_seed
    assert audio_response.status_code == 200
    assert audio_response.content.startswith(b"RIFF")


def test_native_and_barbet_share_bounded_gpu_queue(monkeypatch):
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "1")
    monkeypatch.setenv("VOXCPM_QUEUE_TIMEOUT_SECONDS", "1")

    native_started = threading.Event()
    release_native = threading.Event()

    class BlockingDemo(FakeDemo):
        def generate_tts_audio(self, **kwargs):
            self.calls.append(kwargs)
            native_started.set()
            assert release_native.wait(timeout=2)
            return 16_000, np.zeros(320, dtype=np.float32)

    demo = BlockingDemo()
    runtime = FakeBarbetRuntime(checkpoints=(FakeBarbetCheckpoint(),))
    gateway = api.TTSGateway(demo, barbet_runtime=runtime)

    async def run_scenario():
        native_task = asyncio.create_task(
            gateway.synthesize_native(
                request_id="native-active",
                model_id="__base__",
                text="第一筆",
                control_instruction="",
                reference_path=None,
                prompt_text="",
                cfg_value=2.0,
                normalize=False,
                denoise=False,
                inference_timesteps=10,
            )
        )
        assert await asyncio.to_thread(native_started.wait, 1)

        barbet_task = asyncio.create_task(
            gateway.synthesize_barbet(
                request_id="barbet-waiting",
                model_id="barbet::barbet-tw-v1",
                text="第二筆",
                reference_path=None,
                prompt_text="",
                speaker_id="voice-a",
                cfg_value=2.0,
                inference_timesteps=10,
                seed=1,
            )
        )
        for _ in range(20):
            if gateway._inflight_jobs == 2:
                break
            await asyncio.sleep(0.01)

        assert gateway._inflight_jobs == 2
        assert runtime.calls == []
        with pytest.raises(api.HTTPException) as exc_info:
            await gateway.synthesize_native(
                request_id="native-rejected",
                model_id="__base__",
                text="第三筆",
                control_instruction="",
                reference_path=None,
                prompt_text="",
                cfg_value=2.0,
                normalize=False,
                denoise=False,
                inference_timesteps=10,
            )
        assert exc_info.value.status_code == 429

        release_native.set()
        await asyncio.gather(native_task, barbet_task)

    asyncio.run(run_scenario())
    assert len(runtime.calls) == 1


def test_gpu_queue_wait_has_server_side_timeout(monkeypatch):
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "1")
    monkeypatch.setenv("VOXCPM_QUEUE_TIMEOUT_SECONDS", "0.02")

    native_started = threading.Event()
    release_native = threading.Event()

    class BlockingDemo(FakeDemo):
        def generate_tts_audio(self, **kwargs):
            self.calls.append(kwargs)
            native_started.set()
            assert release_native.wait(timeout=2)
            return 16_000, np.zeros(320, dtype=np.float32)

    gateway = api.TTSGateway(
        BlockingDemo(),
        barbet_runtime=FakeBarbetRuntime(checkpoints=(FakeBarbetCheckpoint(),)),
    )

    async def run_scenario():
        active_task = asyncio.create_task(
            gateway.synthesize_native(
                request_id="native-active",
                model_id="__base__",
                text="第一筆",
                control_instruction="",
                reference_path=None,
                prompt_text="",
                cfg_value=2.0,
                normalize=False,
                denoise=False,
                inference_timesteps=10,
            )
        )
        assert await asyncio.to_thread(native_started.wait, 1)
        with pytest.raises(api.HTTPException) as exc_info:
            await gateway.synthesize_barbet(
                request_id="barbet-timeout",
                model_id="barbet::barbet-tw-v1",
                text="第二筆",
                reference_path=None,
                prompt_text="",
                speaker_id="voice-a",
                cfg_value=2.0,
                inference_timesteps=10,
                seed=1,
            )
        assert exc_info.value.status_code == 503
        assert exc_info.value.headers["X-Request-ID"] == "barbet-timeout"
        release_native.set()
        await active_task

    asyncio.run(run_scenario())


def test_castvoice_voices_lists_available_voxcpm2_and_barbet_voices(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(),),
        speakers=({"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"},),
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/tts/voices")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"]
    voice_ids = {voice["voice_id"] for voice in body["voices"]}
    assert voice_ids == {
        definition["voice_id"] for definition in api._CASTVOICE_DEFINITIONS
    }
    for voice in body["voices"]:
        assert voice["gender"] in {"male", "female", "unknown"}


def test_castvoice_voices_omits_barbet_when_speaker_unavailable(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    # No checkpoints exposes "hung_yi_lee", so the barbet voice must drop out.
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        voices = client.get("/api/v1/tts/voices").json()["voices"]
        voice_ids = {voice["voice_id"] for voice in voices}

    assert "barbet-hung-yi-lee" not in voice_ids
    assert "voxcpm2-cosy-young-female-01" in voice_ids


def test_castvoice_synthesize_returns_mp3_for_voxcpm2_voice(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "台語測試", "voice_id": "voxcpm2-cosy-young-female-01"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0
    assert response.headers["x-request-id"]
    # Cloning mode: control_instruction must be cleared, prompt_text carried in.
    assert demo.calls[0]["control_instruction"] == ""
    assert demo.calls[0]["prompt_text"]


def test_castvoice_synthesize_returns_mp3_for_barbet_voice(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(),),
        speakers=({"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"},),
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "逐家好", "voice_id": "barbet-hung-yi-lee"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert runtime.calls[0]["speaker_id"] == "hung_yi_lee"
    assert runtime.calls[0]["model_id"] == "barbet::barbet-tw-v1"


def test_castvoice_synthesize_rejects_empty_text(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "  ", "voice_id": "voxcpm2-cosy-young-female-01"},
        )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_text", "message": "text 不可為空白"}


def test_castvoice_synthesize_rejects_unknown_voice(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "hi", "voice_id": "does-not-exist"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "voice_not_found"


def test_castvoice_synthesize_rejects_unsupported_format(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "hi",
                "voice_id": "voxcpm2-cosy-young-female-01",
                "format": "wav",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "unsupported_format"


def test_castvoice_synthesize_requires_bearer_token_when_configured(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(api, "_TTS_API_KEY", "secret-token")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        unauthorized = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "hi", "voice_id": "voxcpm2-cosy-young-female-01"},
        )
        authorized = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "hi", "voice_id": "voxcpm2-cosy-young-female-01"},
            headers={"Authorization": "Bearer secret-token"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_castvoice_synthesize_returns_429_with_retry_after_when_queue_full(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "0")

    native_started = threading.Event()
    release_native = threading.Event()

    class BlockingDemo(FakeDemo):
        def generate_tts_audio(self, **kwargs):
            self.calls.append(kwargs)
            native_started.set()
            assert release_native.wait(timeout=2)
            return 16_000, np.zeros(320, dtype=np.float32)

    app = api.create_app(BlockingDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                client.post,
                "/api/v1/tts/synthesize",
                json={"text": "第一筆", "voice_id": "voxcpm2-cosy-young-female-01"},
            )
            assert native_started.wait(timeout=2)
            second = client.post(
                "/api/v1/tts/synthesize",
                json={"text": "第二筆", "voice_id": "voxcpm2-cosy-young-female-01"},
            )
            release_native.set()
            first_response = first.result()

    assert second.status_code == 429
    assert second.json()["error"] == "rate_limited"
    assert second.headers["retry-after"] == "30"
    assert first_response.status_code == 200


def test_castvoice_synthesize_accepts_valid_voxcpm2_model_override(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "hi",
                "voice_id": "voxcpm2-cosy-young-female-01",
                "model_id": "__base__",
            },
        )

    assert response.status_code == 200
    assert demo.calls[0]["model_selection"] == "__base__"


def test_castvoice_synthesize_rejects_unknown_voxcpm2_model_override(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "hi",
                "voice_id": "voxcpm2-cosy-young-female-01",
                "model_id": "does-not-exist",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "model_not_found"


def test_castvoice_synthesize_accepts_valid_barbet_model_override(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(), FakeBarbetCheckpointV2()),
        speakers_by_checkpoint={
            "barbet::barbet-tw-v1": [
                {"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"}
            ],
            "barbet::barbet-tw-v2": [
                {"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"}
            ],
        },
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "逐家好",
                "voice_id": "barbet-hung-yi-lee",
                "model_id": "barbet::barbet-tw-v2",
            },
        )

    assert response.status_code == 200
    assert runtime.calls[0]["model_id"] == "barbet::barbet-tw-v2"
    assert runtime.calls[0]["speaker_id"] == "hung_yi_lee"


def test_castvoice_synthesize_rejects_barbet_model_without_requested_speaker(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(), FakeBarbetCheckpointV2()),
        speakers_by_checkpoint={
            "barbet::barbet-tw-v1": [
                {"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"}
            ],
            "barbet::barbet-tw-v2": [
                {"id": "someone_else", "name": "別人", "gender": "female"}
            ],
        },
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "逐家好",
                "voice_id": "barbet-hung-yi-lee",
                "model_id": "barbet::barbet-tw-v2",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "model_not_found"


def test_castvoice_synthesize_rejects_unknown_barbet_model_override(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(),),
        speakers_by_checkpoint={
            "barbet::barbet-tw-v1": [
                {"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"}
            ],
        },
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "逐家好",
                "voice_id": "barbet-hung-yi-lee",
                "model_id": "does-not-exist",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "model_not_found"


def test_castvoice_batch_item_supports_model_override(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(), FakeBarbetCheckpointV2()),
        speakers_by_checkpoint={
            "barbet::barbet-tw-v1": [
                {"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"}
            ],
            "barbet::barbet-tw-v2": [
                {"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"}
            ],
        },
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/tts/synthesize/batch",
            json={
                "items": [
                    {
                        "text": "逐家好",
                        "voice_id": "barbet-hung-yi-lee",
                        "model_id": "barbet::barbet-tw-v2",
                    }
                ]
            },
        )
        assert submit.status_code == 202
        status = wait_for_batch(client, submit.json()["batch_id"])

    assert status["items"][0]["status"] == "done"
    assert runtime.calls[0]["model_id"] == "barbet::barbet-tw-v2"


def test_castvoice_batch_submits_same_model_items_together(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(
        api.DynamicBatchSizer, "recommend", lambda self, pending: min(pending, 4)
    )
    demo = FakeBatchDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/tts/synthesize/batch",
            json={
                "items": [
                    {
                        "text": f"第 {index} 句",
                        "voice_id": "voxcpm2-cosy-young-female-01",
                        "model_id": "base::__base__",
                    }
                    for index in range(4)
                ]
            },
        )
        assert submit.status_code == 202
        status = wait_for_batch(client, submit.json()["batch_id"])

    assert [item["status"] for item in status["items"]] == ["done"] * 4
    assert len(demo.batch_calls) == 1
    assert len(demo.batch_calls[0]) == 4
    assert demo.calls == []


def test_castvoice_batch_retries_smaller_chunks_after_oom(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(
        api.DynamicBatchSizer, "recommend", lambda self, pending: min(pending, 4)
    )
    demo = FakeOOMBatchDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/tts/synthesize/batch",
            json={
                "items": [
                    {
                        "text": f"第 {index} 句",
                        "voice_id": "voxcpm2-cosy-young-female-01",
                    }
                    for index in range(4)
                ]
            },
        )
        status = wait_for_batch(client, submit.json()["batch_id"])

    assert [item["status"] for item in status["items"]] == ["done"] * 4
    assert [len(requests) for requests in demo.batch_calls] == [4, 2, 2]
