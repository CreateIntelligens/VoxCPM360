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
import gateway.castvoice
import gateway.history
import gateway.presets


@pytest.fixture(autouse=True)
def isolate_generation_history(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_HISTORY_DIR", tmp_path / "generation_history")
    monkeypatch.setattr(gateway.history, "_HISTORY_DIR", tmp_path / "generation_history")
    monkeypatch.setattr(api, "_CASTVOICE_BATCH_DIR", tmp_path / "castvoice_batches")
    monkeypatch.setattr(gateway.castvoice, "_CASTVOICE_BATCH_DIR", tmp_path / "castvoice_batches")
    monkeypatch.setenv("TTS_API_KEY", "")
    monkeypatch.setattr(api, "_TTS_API_KEY", "")
    monkeypatch.setattr(gateway.castvoice, "_TTS_API_KEY", "")


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
        self.warmups = []

    def get_or_load_voxcpm(self):
        self.loads += 1
        return object()

    def stop_voxcpm(self):
        self.stops += 1

    def warmup_voxcpm(self, **kwargs):
        self.warmups.append(kwargs)

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

    def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
        self.batch_calls.append(requests)
        return [(16_000, np.zeros(320, dtype=np.float32)) for _ in requests]


class FakeOOMBatchDemo(FakeBatchDemo):
    def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
        self.batch_calls.append(requests)
        if len(requests) > 2:
            raise RuntimeError("CUDA out of memory")
        return [(16_000, np.zeros(320, dtype=np.float32)) for _ in requests]


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


def synthesize_coalesced(
    gateway: api.TTSGateway,
    request_id: str,
    text: str,
    model_id: str = api.PUBLIC_BASE_MODEL_ID,
):
    return gateway.synthesize_native_coalesced(
        request_id=request_id,
        model_id=model_id,
        text=text,
        control_instruction="",
        reference_path=None,
        prompt_text="",
        cfg_value=2.0,
        normalize=False,
        denoise=False,
        inference_timesteps=10,
    )


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
    # voxcpm2 的 per-request inference_timesteps 不生效（引擎建構時固定），
    # catalog 必須誠實揭露讓前端隱藏該欄位。
    assert payload["engines"][0]["capabilities"]["inference_timesteps"] is False


def test_preload_warms_common_inference_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_PRELOAD", "true")
    monkeypatch.setattr(api, "_REFERENCE_AUDIO_DIR", tmp_path)
    preset = api._find_reference_preset(api._DEFAULT_REFERENCE_PRESET_ID)
    assert preset is not None
    preset_path = tmp_path / preset["filename"]
    preset_path.write_bytes(b"audio")
    demo = FakeDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    with TestClient(app) as client:
        assert client.get("/api/v1/catalog").status_code == 200

    assert demo.loads == 1
    assert demo.warmups == [
        {
            "reference_path": str(preset_path),
            "prompt_text": preset["prompt_text"],
        }
    ]


def test_warmup_runs_without_and_with_reference():
    demo = FakeDemo()

    api.VoxCPMDemo.warmup_voxcpm(
        demo,
        reference_path="/reference.wav",
        prompt_text="參考逐字稿",
    )

    assert demo.calls == [
        {
            "text_input": "暖機。",
            "do_normalize": False,
            "denoise": False,
            "seed": 0,
        },
        {
            "text_input": "暖機。",
            "reference_wav_path_input": "/reference.wav",
            "prompt_text": "參考逐字稿",
            "do_normalize": False,
            "denoise": False,
            "seed": 0,
        },
    ]


def test_warmup_failure_does_not_block_startup(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "true")
    demo = FakeDemo()

    def fail_warmup(**_kwargs):
        raise RuntimeError("warmup failed")

    demo.warmup_voxcpm = fail_warmup
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/catalog")

    assert response.status_code == 200
    assert demo.loads == 1


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
    monkeypatch.setattr(gateway.presets, "_REFERENCE_AUDIO_DIR", tmp_path)
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
    monkeypatch.setattr(gateway.presets, "_MODEL_REGISTRY_PATH", registry_path)
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
    monkeypatch.setattr(gateway.presets, "_MODEL_REGISTRY_PATH", tmp_path / "missing.json")
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
    monkeypatch.setattr(gateway.presets, "_REFERENCE_AUDIO_DIR", tmp_path)
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
    assert response.headers["x-batch-size"] == "1"
    assert demo.calls[0]["inference_timesteps"] == 8


def test_native_interactive_coalesces_requests_already_waiting(monkeypatch):
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "5")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "4")
    first_started = threading.Event()
    release_first = threading.Event()

    class BlockingBatchDemo(FakeBatchDemo):
        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            self.batch_calls.append(requests)
            if len(self.batch_calls) == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            return [(16_000, np.zeros(320, dtype=np.float32)) for _ in requests]

    gateway = api.TTSGateway(
        BlockingBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    async def run_scenario():
        first = asyncio.create_task(synthesize_coalesced(gateway, "request-0", "第零筆"))
        assert await asyncio.to_thread(first_started.wait, 1)
        waiting = [
            asyncio.create_task(synthesize_coalesced(gateway, f"request-{index}", f"第 {index} 筆"))
            for index in range(1, 6)
        ]
        for _ in range(100):
            if gateway._inflight_jobs == 6:
                break
            await asyncio.sleep(0.01)
        assert gateway._inflight_jobs == 6

        release_first.set()
        results = await asyncio.gather(first, *waiting)
        await gateway.close_coalescer()
        return results

    results = asyncio.run(run_scenario())

    assert sum(len(requests) for requests in gateway.demo.batch_calls) == 6
    assert [headers["X-Batch-Size"] for _, headers in results] == ["1", "4", "4", "4", "4", "1"]
    assert all(wav.startswith(b"RIFF") for wav, _ in results)
    assert gateway._inflight_jobs == 0


def test_native_interactive_batch_max_one_is_rollback(monkeypatch):
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "3")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "1")
    gateway = api.TTSGateway(
        FakeBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    async def run_scenario():
        await gateway._gpu_lock.acquire()
        tasks = [
            asyncio.create_task(synthesize_coalesced(gateway, f"request-{index}", f"第 {index} 筆"))
            for index in range(4)
        ]
        for _ in range(100):
            if gateway._inflight_jobs == 4:
                break
            await asyncio.sleep(0.01)
        gateway._gpu_lock.release()
        results = await asyncio.gather(*tasks)
        await gateway.close_coalescer()
        return results

    results = asyncio.run(run_scenario())

    assert [len(requests) for requests in gateway.demo.batch_calls] == [1, 1, 1, 1]
    assert all(headers["X-Batch-Size"] == "1" for _, headers in results)


def test_native_interactive_batch_max_is_clamped_to_engine_limit(monkeypatch):
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "99")

    gateway = api.TTSGateway(
        FakeBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    assert gateway._interactive_batch_max == 16


def test_native_interactive_preserves_fifo_across_models(monkeypatch):
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "3")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "4")

    class LoraCheckpoint:
        def __init__(self, run_name):
            self.run_name = run_name
            self.label = run_name

    demo = FakeBatchDemo()
    demo.lora_registry = FakeRegistry((LoraCheckpoint("model-a"), LoraCheckpoint("model-b")))
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        await gateway._gpu_lock.acquire()
        model_ids = ["lora::model-a", "lora::model-a", "lora::model-b", "lora::model-a"]
        tasks = [
            asyncio.create_task(
                synthesize_coalesced(
                    gateway,
                    f"request-{index}",
                    f"第 {index} 筆",
                    model_id,
                )
            )
            for index, model_id in enumerate(model_ids)
        ]
        for _ in range(100):
            if gateway._inflight_jobs == 4:
                break
            await asyncio.sleep(0.01)
        gateway._gpu_lock.release()
        await asyncio.gather(*tasks)
        await gateway.close_coalescer()

    asyncio.run(run_scenario())

    assert sum(len(requests) for requests in demo.batch_calls) == 4
    assert [requests[0]["model_selection"] for requests in demo.batch_calls] == [
        "model-a",
        "model-a",
        "model-b",
        "model-a",
    ]


def test_native_interactive_queue_timeout_removes_pending_item(monkeypatch):
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "1")
    monkeypatch.setenv("VOXCPM_QUEUE_TIMEOUT_SECONDS", "0.02")
    gateway = api.TTSGateway(
        FakeBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    async def run_scenario():
        await gateway._gpu_lock.acquire()
        with pytest.raises(api.HTTPException) as exc_info:
            await synthesize_coalesced(gateway, "timed-out", "逾時請求")
        assert exc_info.value.status_code == 503
        assert gateway._inflight_jobs == 0
        assert gateway.demo.batch_calls == []

        gateway._gpu_lock.release()
        for _ in range(100):
            drain_task = gateway._native_coalescer._drain_task
            if drain_task is None or drain_task.done():
                break
            await asyncio.sleep(0.01)
        result = await synthesize_coalesced(gateway, "recovered", "恢復請求")
        await gateway.close_coalescer()
        return result

    wav, headers = asyncio.run(run_scenario())

    assert wav.startswith(b"RIFF")
    assert headers["X-Batch-Size"] == "1"
    assert len(gateway.demo.batch_calls) == 1


def test_native_interactive_claimed_item_does_not_timeout_during_generation(
    monkeypatch,
):
    monkeypatch.setenv("VOXCPM_QUEUE_TIMEOUT_SECONDS", "0.02")
    generation_started = threading.Event()
    release_generation = threading.Event()

    class SlowBatchDemo(FakeBatchDemo):
        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            self.batch_calls.append(requests)
            generation_started.set()
            assert release_generation.wait(timeout=2)
            return [(16_000, np.zeros(320, dtype=np.float32)) for _ in requests]

    gateway = api.TTSGateway(
        SlowBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    async def run_scenario():
        task = asyncio.create_task(synthesize_coalesced(gateway, "claimed", "已開始生成"))
        assert await asyncio.to_thread(generation_started.wait, 1)
        await asyncio.sleep(0.04)
        assert not task.done()

        release_generation.set()
        result = await task
        await gateway.close_coalescer()
        return result

    wav, headers = asyncio.run(run_scenario())

    assert wav.startswith(b"RIFF")
    assert headers["X-Batch-Size"] == "1"
    assert gateway._inflight_jobs == 0


def test_native_interactive_rejects_submit_after_shutdown_without_admission(
    monkeypatch,
):
    gateway = api.TTSGateway(
        FakeBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    async def run_scenario():
        await gateway.close_coalescer()
        with pytest.raises(
            RuntimeError,
            match="Native interactive coalescer is closed",
        ):
            await synthesize_coalesced(gateway, "after-close", "關閉後送入")

    asyncio.run(run_scenario())

    assert gateway._inflight_jobs == 0
    assert gateway.demo.batch_calls == []


def test_native_interactive_completion_failure_resolves_request(monkeypatch):
    gateway = api.TTSGateway(
        FakeBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    def fail_timing_headers(queue_wait, execution_time):
        raise RuntimeError("header failure")

    monkeypatch.setattr(gateway, "_job_timing_headers", fail_timing_headers)

    async def run_scenario():
        with pytest.raises(RuntimeError, match="header failure"):
            await asyncio.wait_for(
                synthesize_coalesced(gateway, "completion-failure", "完成階段失敗"),
                timeout=1,
            )
        await gateway.close_coalescer()

    asyncio.run(run_scenario())

    assert gateway._inflight_jobs == 0
    assert len(gateway.demo.batch_calls) == 1


def test_native_interactive_queue_full_keeps_global_capacity(monkeypatch):
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "0")
    first_started = threading.Event()
    release_first = threading.Event()

    class BlockingBatchDemo(FakeBatchDemo):
        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            self.batch_calls.append(requests)
            first_started.set()
            assert release_first.wait(timeout=2)
            return [(16_000, np.zeros(320, dtype=np.float32)) for _ in requests]

    gateway = api.TTSGateway(
        BlockingBatchDemo(),
        barbet_runtime=FakeBarbetRuntime(),
    )

    async def run_scenario():
        active = asyncio.create_task(synthesize_coalesced(gateway, "active", "第一筆"))
        assert await asyncio.to_thread(first_started.wait, 1)
        with pytest.raises(api.HTTPException) as exc_info:
            await synthesize_coalesced(gateway, "rejected", "第二筆")
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers["Retry-After"] == "30"
        assert exc_info.value.headers["X-Request-ID"] == "rejected"
        release_first.set()
        await active
        await gateway.close_coalescer()

    asyncio.run(run_scenario())
    assert gateway._inflight_jobs == 0


def test_native_interactive_waits_for_streaming_gate_then_batches(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "1")
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "2")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "4")
    stream_blocked = threading.Event()
    release_stream = threading.Event()

    class StreamingBatchDemo(FakeStreamingDemo):
        def __init__(self):
            super().__init__()
            self.batch_calls = []

        def generate_tts_audio_stream(self, request):
            yield np.array([0.1], dtype=np.float32)
            stream_blocked.set()
            assert release_stream.wait(timeout=2)
            yield np.array([0.2], dtype=np.float32)

        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            self.batch_calls.append(requests)
            return [(16_000, np.zeros(320, dtype=np.float32)) for _ in requests]

    demo = StreamingBatchDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        stream = await gateway.synthesize_native_stream(
            request_id="streaming",
            model_id="__base__",
            text="串流請求",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=False,
            denoise=False,
            inference_timesteps=10,
        )
        assert await asyncio.to_thread(stream_blocked.wait, 1)
        tasks = [
            asyncio.create_task(synthesize_coalesced(gateway, f"queued-{index}", f"排隊 {index}")) for index in range(2)
        ]
        for _ in range(100):
            if gateway._inflight_jobs == 3:
                break
            await asyncio.sleep(0.01)
        assert gateway._inflight_jobs == 3
        assert demo.batch_calls == []

        release_stream.set()
        assert [chunk async for chunk in stream]
        await stream.aclose()
        results = await asyncio.gather(*tasks)
        await gateway.close_coalescer()
        return results

    results = asyncio.run(run_scenario())

    assert sum(len(requests) for requests in demo.batch_calls) == 2
    assert all(headers["X-Batch-Size"] == "2" for _, headers in results)
    assert gateway._inflight_jobs == 0
    assert not gateway._gpu_lock.locked()


def test_native_interactive_batch_isolates_http_errors_and_history(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "3")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "4")
    first_started = threading.Event()
    release_first = threading.Event()

    class IsolatedFailureDemo(FakeBatchDemo):
        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            self.batch_calls.append(requests)
            if len(self.batch_calls) == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            results = [
                RuntimeError("one item failed")
                if request["text_input"] == "失敗"
                else (16_000, np.zeros(320, dtype=np.float32))
                for request in requests
            ]
            if not return_exceptions:
                error = next(
                    (result for result in results if isinstance(result, Exception)),
                    None,
                )
                if error is not None:
                    raise error
            return results

    demo = IsolatedFailureDemo()
    app = api.create_app(
        demo,
        barbet_runtime=FakeBarbetRuntime(),
        mount_legacy=False,
    )
    payload = {
        "engine_id": "voxcpm2",
        "model_id": "__base__",
    }

    with TestClient(app, raise_server_exceptions=False) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            blocker = pool.submit(
                client.post,
                "/api/v1/synthesize",
                data={**payload, "text": "先佔住 GPU"},
            )
            assert first_started.wait(timeout=2)
            futures = [
                pool.submit(
                    client.post,
                    "/api/v1/synthesize",
                    data={**payload, "text": text},
                )
                for text in ("成功一", "失敗", "成功二")
            ]
            for _ in range(100):
                if app.state.tts_gateway._inflight_jobs == 4:
                    break
                time.sleep(0.01)
            release_first.set()
            blocker_response = blocker.result()
            responses = [future.result() for future in futures]
        history = client.get("/api/v1/history?limit=10").json()["items"]

    assert blocker_response.status_code == 200
    assert [response.status_code for response in responses] == [200, 500, 200]
    assert responses[0].headers["x-batch-size"] in {"1", "2", "3"}
    assert responses[2].headers["x-batch-size"] in {"1", "2", "3"}
    assert "x-history-id" not in responses[1].headers
    assert len(history) == 3
    assert {item["text"] for item in history} == {"先佔住 GPU", "成功一", "成功二"}


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
    monkeypatch.setattr(gateway.presets, "_REFERENCE_AUDIO_DIR", tmp_path)
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
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "1")
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

    monkeypatch.setattr(gateway.history, "_save_generation_history", blocking_save)
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
    if server.loop.is_running():
        server.loop.call_soon_threadsafe(server.loop.stop)
        loop_thread = getattr(demo, "_server_loop_thread", None)
        if loop_thread is not None:
            loop_thread.join(timeout=2.0)
    server.loop.close()


def test_demo_batch_can_return_backend_exceptions_per_item():
    completed = []

    class AsyncPool:
        async def generate(self, **request):
            try:
                if request["index"] == 1:
                    raise RuntimeError("item failed")
                yield np.array([request["index"]], dtype=np.float32)
            finally:
                completed.append(request["index"])

    class Server:
        def __init__(self):
            self.server_pool = AsyncPool()
            self.loop = asyncio.new_event_loop()

    server = Server()
    requests = [{"index": index} for index in range(3)]
    results = api.VoxCPMDemo._generate_tts_requests(
        server,
        requests,
        return_exceptions=True,
    )

    assert np.array_equal(results[0], np.array([0], dtype=np.float32))
    assert isinstance(results[1], RuntimeError)
    assert np.array_equal(results[2], np.array([2], dtype=np.float32))
    assert sorted(completed) == [0, 1, 2]

    with pytest.raises(RuntimeError, match="1 of 3 batched generations failed"):
        api.VoxCPMDemo._generate_tts_requests(server, requests)
    assert sorted(completed) == [0, 0, 1, 1, 2, 2]
    server.loop.close()


def test_demo_batch_isolates_request_preparation_errors():
    class Server:
        @staticmethod
        def get_model_info():
            return {"sample_rate": 16_000}

    demo = object.__new__(api.VoxCPMDemo)
    demo.get_or_load_voxcpm = lambda: Server()

    def prepare(server, temp_files, latent_cache, **request):
        if request["index"] == 1:
            raise ValueError("bad reference")
        return request

    demo._prepare_tts_generation = prepare
    demo._generate_tts_requests = lambda server, requests, return_exceptions=False: [
        np.array([request["index"]], dtype=np.float32) for request in requests
    ]
    results = demo.generate_tts_audio_batch(
        [{"index": index} for index in range(3)],
        return_exceptions=True,
    )

    assert results[0][0] == 16_000
    assert isinstance(results[1], ValueError)
    assert results[2][0] == 16_000


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
    # 拆分後 _switch_native_runtime 位於 gateway.gateway，patch 需同步指向
    import gateway.gateway as _gw_mod
    monkeypatch.setattr(_gw_mod, "VoxCPMDemo", lambda **kwargs: full_demo)
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


def test_demo_rejects_non_cuda_device(monkeypatch, tmp_path):
    """nano-vLLM 沒有 CPU 路徑，放行只會讓服務誤報 device 卻仍佔 GPU 0。"""
    import app as app_module

    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(app_module, "resolve_runtime_device", lambda *_: "cpu")
    monkeypatch.setattr(
        app_module, "_HISTORY_DIR", tmp_path / "history", raising=False
    )

    demo = app_module.VoxCPMDemo(device="cpu")
    with pytest.raises(ValueError, match="requires CUDA"):
        demo.get_or_load_voxcpm()


def test_dynamic_batch_sizer_sizes_from_cgroup_not_cuda(monkeypatch):
    """合批規模看 cgroup；CUDA free 是 KV cache 之外的餘量，不是 per-item 池。"""
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

    # (20 - 2) / 1.5 = 12，取 cgroup 上限而非較小的 CUDA 餘量。
    assert sizer.recommend(20) == 12
    assert sizer.recommend(3) == 3


def test_dynamic_batch_sizer_small_gpu_still_batches(monkeypatch):
    """16GB 卡的迴歸：模型佔滿後 CUDA 只剩 ~4.4GB，仍須能合批。"""
    gib = 1024**3
    monkeypatch.setattr(
        api.DynamicBatchSizer,
        "_cuda_headroom",
        staticmethod(lambda: int(4.4 * gib)),
    )
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cgroup_headroom", staticmethod(lambda: 60 * gib)
    )
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_RESERVE_GIB", "2")
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_PER_ITEM_GIB", "1.5")

    sizer = api.DynamicBatchSizer(max_concurrency=16)

    # 舊行為為 (4.4-2)/1.5 = 1（合不了批 → 學不到 → 永遠 1 的死結）。
    assert sizer.recommend(16) == 16


def test_dynamic_batch_sizer_cuda_guard_clamps_when_vram_critical(monkeypatch):
    """顯存低於 reserve 時咬住批量，避免與其他 GPU 工作並存時撐爆。"""
    gib = 1024**3
    monkeypatch.setattr(
        api.DynamicBatchSizer,
        "_cuda_headroom",
        staticmethod(lambda: int(0.5 * gib)),
    )
    monkeypatch.setattr(
        api.DynamicBatchSizer, "_cgroup_headroom", staticmethod(lambda: 60 * gib)
    )
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_RESERVE_GIB", "2")
    monkeypatch.setenv("VOXCPM_BATCH_MEMORY_PER_ITEM_GIB", "1.5")

    assert api.DynamicBatchSizer(max_concurrency=16).recommend(16) == 1


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

    assert sizer.recommend(20) == 12
    sizer.observe_success(size=12, elapsed=5.0, work_units=50)
    assert sizer.recommend(20) == 16

    sizer.observe_oom(size=16)
    # cap 砍半到 8，per-item 加倍到 2.25GiB：(20-2)/2.25 = 8。
    assert sizer.recommend(20) == 8


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
    monkeypatch.setattr(gateway.castvoice, "_TTS_API_KEY", "secret-token")
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


def test_castvoice_synthesize_passes_synthesis_params_to_native_engine(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "台語測試",
                "voice_id": "voxcpm2-cosy-young-female-01",
                "seed": 4242,
                "cfg_value": 3.5,
                "speed": 1.25,
                "normalize": False,
                "denoise": True,
            },
        )

    assert response.status_code == 200
    call = demo.calls[0]
    assert call["seed"] == 4242
    assert call["cfg_value_input"] == 3.5
    assert call["do_normalize"] is False
    assert call["denoise"] is True


def test_castvoice_synthesize_passes_synthesis_params_to_barbet_engine(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(),),
        speakers=({"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"},),
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "逐家好",
                "voice_id": "barbet-hung-yi-lee",
                "seed": 777,
                "cfg_value": 4.0,
            },
        )

    assert response.status_code == 200
    assert runtime.calls[0]["seed"] == 777
    assert runtime.calls[0]["cfg_value"] == 4.0


def test_castvoice_synthesize_without_params_keeps_existing_defaults(monkeypatch):
    """回歸：不帶新欄位時，送進引擎的值必須與加欄位之前完全一致。"""
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "台語測試", "voice_id": "voxcpm2-cosy-young-female-01"},
        )

    assert response.status_code == 200
    call = demo.calls[0]
    assert call["cfg_value_input"] == 2.0
    assert call["do_normalize"] is True
    assert call["denoise"] is False
    assert call["seed"] is None
    assert call["inference_timesteps"] == 30


def test_castvoice_synthesize_without_params_keeps_barbet_defaults(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    runtime = FakeBarbetRuntime(
        checkpoints=(FakeBarbetCheckpoint(),),
        speakers=({"id": "hung_yi_lee", "name": "李宏毅老師", "gender": "male"},),
    )
    app = api.create_app(FakeDemo(), barbet_runtime=runtime, mount_legacy=False)

    with TestClient(app) as client:
        responses = [
            client.post(
                "/api/v1/tts/synthesize",
                json={"text": "逐家好", "voice_id": "barbet-hung-yi-lee"},
            )
            for _ in range(2)
        ]

    assert [response.status_code for response in responses] == [200, 200]
    assert runtime.calls[0]["cfg_value"] == 2.0
    # barbet 端 seed=None 由 gateway 補隨機值：不指定時每次都該不一樣，
    # 這正是加上選填 seed 之前的行為。
    assert runtime.calls[0]["seed"] != runtime.calls[1]["seed"]


@pytest.mark.parametrize("cfg_value", [0.9, 5.1])
def test_castvoice_synthesize_rejects_out_of_range_cfg_value(monkeypatch, cfg_value):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "台語測試",
                "voice_id": "voxcpm2-cosy-young-female-01",
                "cfg_value": cfg_value,
            },
        )

    # CastAgent 風格的錯誤形狀，而不是 FastAPI 的 422。
    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_request",
        "message": "cfg_value 必須介於 1.0 與 5.0",
    }


@pytest.mark.parametrize("speed", [0.4, 2.1])
def test_castvoice_synthesize_rejects_out_of_range_speed(monkeypatch, speed):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "台語測試",
                "voice_id": "voxcpm2-cosy-young-female-01",
                "speed": speed,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_speed"


def test_castvoice_synthesize_rejects_inference_timesteps_field(monkeypatch):
    """voxcpm2 建構時定死步數，per-request 不生效，body 刻意不收這個欄位。"""
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    demo = FakeDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "台語測試",
                "voice_id": "voxcpm2-cosy-young-female-01",
                "inference_timesteps": 5,
            },
        )

    # pydantic 預設忽略未知欄位：請求仍成功，但步數維持既有的 30。
    assert response.status_code == 200
    assert demo.calls[0]["inference_timesteps"] == 30


def test_castvoice_batch_items_carry_independent_synthesis_params(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(
        api.DynamicBatchSizer, "recommend", lambda self, pending: min(pending, 4)
    )
    demo = FakeBatchDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/tts/synthesize/batch",
            json={
                "items": [
                    {
                        "text": "第一句",
                        "voice_id": "voxcpm2-cosy-young-female-01",
                        "seed": 111,
                        "cfg_value": 1.5,
                        "denoise": True,
                    },
                    {
                        "text": "第二句",
                        "voice_id": "voxcpm2-cosy-young-female-01",
                        "seed": 222,
                        "cfg_value": 4.5,
                        "normalize": False,
                    },
                ]
            },
        )
        assert submit.status_code == 202
        status = wait_for_batch(client, submit.json()["batch_id"])

    assert [item["status"] for item in status["items"]] == ["done", "done"]
    # 同一個 nano-vLLM chunk 內，每一筆仍帶著自己的參數。
    assert len(demo.batch_calls) == 1
    first, second = demo.batch_calls[0]
    assert (first["seed"], first["cfg_value_input"]) == (111, 1.5)
    assert first["denoise"] is True
    assert first["do_normalize"] is True
    assert (second["seed"], second["cfg_value_input"]) == (222, 4.5)
    assert second["do_normalize"] is False
    assert second["denoise"] is False


def test_castvoice_batch_without_params_keeps_existing_defaults(monkeypatch):
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(
        api.DynamicBatchSizer, "recommend", lambda self, pending: min(pending, 4)
    )
    demo = FakeBatchDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/tts/synthesize/batch",
            json={
                "items": [
                    {"text": f"第 {index} 句", "voice_id": "voxcpm2-cosy-young-female-01"}
                    for index in range(2)
                ]
            },
        )
        status = wait_for_batch(client, submit.json()["batch_id"])

    assert [item["status"] for item in status["items"]] == ["done", "done"]
    for request in demo.batch_calls[0]:
        assert request["cfg_value_input"] == 2.0
        assert request["do_normalize"] is True
        assert request["denoise"] is False
        assert request["seed"] is None


def test_castvoice_batch_marks_only_the_item_with_invalid_cfg_value_failed(monkeypatch):
    """單筆參數違規不能拖垮同批的其他句子。"""
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    monkeypatch.setattr(
        api.DynamicBatchSizer, "recommend", lambda self, pending: min(pending, 4)
    )
    demo = FakeBatchDemo()
    app = api.create_app(demo, barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/tts/synthesize/batch",
            json={
                "items": [
                    {
                        "text": "好的那句",
                        "voice_id": "voxcpm2-cosy-young-female-01",
                        "cfg_value": 2.5,
                    },
                    {
                        "text": "壞的那句",
                        "voice_id": "voxcpm2-cosy-young-female-01",
                        "cfg_value": 9.9,
                    },
                ]
            },
        )
        status = wait_for_batch(client, submit.json()["batch_id"])

    assert [item["status"] for item in status["items"]] == ["done", "failed"]
    assert status["items"][1]["error"] == "cfg_value 必須介於 1.0 與 5.0"


def test_concurrent_streaming_two_streams_ttfb_unblocked(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "2")
    stream1_first_sent = threading.Event()
    stream2_first_sent = threading.Event()

    class ConcurrentStreamingDemo(FakeDemo):
        def __init__(self):
            super().__init__()
            self.server = FakeStreamingServer()

        def get_or_load_voxcpm(self):
            return self.server

        def generate_tts_audio_stream(self, request):
            text = request["text_input"]
            if text == "stream-1":
                yield np.array([0.1], dtype=np.float32)
                stream1_first_sent.set()
                assert stream2_first_sent.wait(timeout=2.0)
                yield np.array([0.2], dtype=np.float32)
            else:
                yield np.array([0.5], dtype=np.float32)
                stream2_first_sent.set()
                assert stream1_first_sent.wait(timeout=2.0)
                yield np.array([0.6], dtype=np.float32)

    demo = ConcurrentStreamingDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        s1 = await gateway.synthesize_native_stream(
            request_id="req-1",
            model_id="__base__",
            text="stream-1",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=False,
            denoise=False,
            inference_timesteps=10,
        )
        s2 = await gateway.synthesize_native_stream(
            request_id="req-2",
            model_id="__base__",
            text="stream-2",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=False,
            denoise=False,
            inference_timesteps=10,
        )
        assert s1.session_concurrency == 1
        assert s2.session_concurrency == 2

        chunks1 = [c async for c in s1]
        chunks2 = [c async for c in s2]
        return chunks1, chunks2

    c1, c2 = asyncio.run(run_scenario())
    assert len(c1) == 2
    assert len(c2) == 2
    assert np.array_equal(c1[0], np.array([0.1], dtype=np.float32))
    assert np.array_equal(c2[0], np.array([0.5], dtype=np.float32))


def test_concurrent_streaming_and_batch_coexist(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "4")
    stream_running = threading.Event()
    batch_running = threading.Event()
    release_all = threading.Event()

    class MixedDemo(FakeDemo):
        def __init__(self):
            super().__init__()
            self.server = FakeStreamingServer()
            self.batch_calls = []

        def get_or_load_voxcpm(self):
            return self.server

        def generate_tts_audio_stream(self, request):
            stream_running.set()
            assert batch_running.wait(timeout=2.0)
            yield np.array([0.1], dtype=np.float32)
            assert release_all.wait(timeout=2.0)
            yield np.array([0.2], dtype=np.float32)

        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            self.batch_calls.append(requests)
            batch_running.set()
            assert stream_running.wait(timeout=2.0)
            assert release_all.wait(timeout=2.0)
            return [(16000, np.zeros(160, dtype=np.float32)) for _ in requests]

    demo = MixedDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        stream_task = asyncio.create_task(
            gateway.synthesize_native_stream(
                request_id="req-stream",
                model_id="__base__",
                text="streaming-request",
                control_instruction="",
                reference_path=None,
                prompt_text="",
                cfg_value=2.0,
                normalize=False,
                denoise=False,
                inference_timesteps=10,
            )
        )
        batch_task = asyncio.create_task(
            gateway.synthesize_native_batch(
                request_id="req-batch",
                model_id="__base__",
                requests=[{"text": "batch-1"}, {"text": "batch-2"}],
            )
        )
        assert await asyncio.to_thread(stream_running.wait, 2.0)
        assert await asyncio.to_thread(batch_running.wait, 2.0)
        release_all.set()

        stream, (batch_wavs, batch_headers) = await asyncio.gather(stream_task, batch_task)
        chunks = [c async for c in stream]
        return chunks, batch_wavs, batch_headers

    chunks, batch_wavs, batch_headers = asyncio.run(run_scenario())
    assert len(chunks) == 2
    assert len(batch_wavs) == 2
    assert gateway._session_gate.active_units == 0


def test_engine_concurrency_one_is_strictly_exclusive(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "1")
    stream1_started = threading.Event()
    release_stream1 = threading.Event()

    class ExclDemo(FakeDemo):
        def __init__(self):
            super().__init__()
            self.server = FakeStreamingServer()

        def get_or_load_voxcpm(self):
            return self.server

        def generate_tts_audio_stream(self, request):
            if request["text_input"] == "stream-1":
                stream1_started.set()
                yield np.array([0.1], dtype=np.float32)
                assert release_stream1.wait(timeout=2.0)
                yield np.array([0.2], dtype=np.float32)
            else:
                yield np.array([0.9], dtype=np.float32)

    demo = ExclDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        s1 = await gateway.synthesize_native_stream(
            request_id="req-1",
            model_id="__base__",
            text="stream-1",
            control_instruction="",
            reference_path=None,
            prompt_text="",
            cfg_value=2.0,
            normalize=False,
            denoise=False,
            inference_timesteps=10,
        )
        s2_task = asyncio.create_task(
            gateway.synthesize_native_stream(
                request_id="req-2",
                model_id="__base__",
                text="stream-2",
                control_instruction="",
                reference_path=None,
                prompt_text="",
                cfg_value=2.0,
                normalize=False,
                denoise=False,
                inference_timesteps=10,
            )
        )
        await asyncio.sleep(0.05)
        assert not s2_task.done()

        release_stream1.set()
        c1 = [c async for c in s1]
        await s1.aclose()

        s2 = await s2_task
        c2 = [c async for c in s2]
        return c1, c2

    c1, c2 = asyncio.run(run_scenario())
    assert len(c1) == 2
    assert len(c2) == 1
    assert gateway._session_gate.active_units == 0


def test_session_gate_capacity_full_queues_and_admit_rejects_429(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "2")
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "2")
    hold_gate = threading.Event()

    class GateDemo(FakeDemo):
        def __init__(self):
            super().__init__()
            self.server = FakeStreamingServer()

        def get_or_load_voxcpm(self):
            return self.server

        def generate_tts_audio_stream(self, request):
            yield np.array([0.1], dtype=np.float32)
            assert hold_gate.wait(timeout=2.0)
            yield np.array([0.2], dtype=np.float32)

    demo = GateDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        s1 = await gateway.synthesize_native_stream(
            request_id="req-1", model_id="__base__", text="1",
            control_instruction="", reference_path=None, prompt_text="",
            cfg_value=2.0, normalize=False, denoise=False, inference_timesteps=10,
        )
        s2 = await gateway.synthesize_native_stream(
            request_id="req-2", model_id="__base__", text="2",
            control_instruction="", reference_path=None, prompt_text="",
            cfg_value=2.0, normalize=False, denoise=False, inference_timesteps=10,
        )
        s3_task = asyncio.create_task(
            gateway.synthesize_native_stream(
                request_id="req-3", model_id="__base__", text="3",
                control_instruction="", reference_path=None, prompt_text="",
                cfg_value=2.0, normalize=False, denoise=False, inference_timesteps=10,
            )
        )
        await asyncio.sleep(0.05)
        assert not s3_task.done()

        with pytest.raises(api.HTTPException) as exc_info:
            await gateway.synthesize_native_stream(
                request_id="req-4", model_id="__base__", text="4",
                control_instruction="", reference_path=None, prompt_text="",
                cfg_value=2.0, normalize=False, denoise=False, inference_timesteps=10,
            )
        assert exc_info.value.status_code == 429

        hold_gate.set()
        await s1.aclose()
        await s2.aclose()
        s3 = await s3_task
        await s3.aclose()

    asyncio.run(run_scenario())
    assert gateway._inflight_jobs == 0
    assert gateway._session_gate.active_units == 0


def test_session_gate_model_switch_drains_and_preserves_fifo(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "2")
    gate = api._GPUSessionGate(concurrency=2)
    order = []

    async def run_scenario():
        c1 = await gate.acquire(engine_id="voxcpm2", model_id="model-A", units=1)
        c2 = await gate.acquire(engine_id="voxcpm2", model_id="model-A", units=1)
        assert c1 == 1 and c2 == 2
        assert gate.active_units == 2

        async def wait_b():
            await gate.acquire(engine_id="voxcpm2", model_id="model-B", units=1)
            order.append("B")
            gate.release(units=1)

        async def wait_a2():
            await gate.acquire(engine_id="voxcpm2", model_id="model-A", units=1)
            order.append("A2")
            gate.release(units=1)

        task_b = asyncio.create_task(wait_b())
        task_a2 = asyncio.create_task(wait_a2())
        await asyncio.sleep(0.02)

        gate.release(units=1)
        await asyncio.sleep(0.02)
        assert order == []

        gate.release(units=1)
        await asyncio.gather(task_b, task_a2)
        assert order == ["B", "A2"]

    asyncio.run(run_scenario())
    assert gate.active_units == 0


def test_concurrent_streaming_client_disconnect_isolation(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "2")
    stream1_closed = threading.Event()
    release_stream2 = threading.Event()

    class DisconnectDemo(FakeDemo):
        def __init__(self):
            super().__init__()
            self.server = FakeStreamingServer()

        def get_or_load_voxcpm(self):
            return self.server

        def generate_tts_audio_stream(self, request):
            if request["text_input"] == "stream-1":
                try:
                    yield np.array([0.1], dtype=np.float32)
                    for idx in range(100):
                        yield np.array([0.1 * (idx + 2)], dtype=np.float32)
                finally:
                    stream1_closed.set()
            else:
                yield np.array([0.8], dtype=np.float32)
                assert stream1_closed.wait(timeout=2.0)
                assert release_stream2.wait(timeout=2.0)
                yield np.array([0.9], dtype=np.float32)

    demo = DisconnectDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        s1 = await gateway.synthesize_native_stream(
            request_id="req-1", model_id="__base__", text="stream-1",
            control_instruction="", reference_path=None, prompt_text="",
            cfg_value=2.0, normalize=False, denoise=False, inference_timesteps=10,
        )
        s2 = await gateway.synthesize_native_stream(
            request_id="req-2", model_id="__base__", text="stream-2",
            control_instruction="", reference_path=None, prompt_text="",
            cfg_value=2.0, normalize=False, denoise=False, inference_timesteps=10,
        )
        assert gateway._session_gate.active_units == 2

        _ = await s1.__anext__()
        await s1.aclose()
        assert await asyncio.to_thread(stream1_closed.wait, 2.0)

        assert gateway._session_gate.active_units == 1

        release_stream2.set()
        c2 = [c async for c in s2]
        await s2.aclose()
        return c2

    c2 = asyncio.run(run_scenario())
    assert len(c2) == 2
    assert gateway._session_gate.active_units == 0


def test_barbet_and_voxcpm2_mutual_exclusion_with_drain(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "2")
    stream_running = threading.Event()
    release_stream = threading.Event()

    class StreamDemo(FakeDemo):
        def __init__(self):
            super().__init__()
            self.server = FakeStreamingServer()

        def get_or_load_voxcpm(self):
            return self.server

        def generate_tts_audio_stream(self, request):
            stream_running.set()
            yield np.array([0.1], dtype=np.float32)
            assert release_stream.wait(timeout=2.0)
            yield np.array([0.2], dtype=np.float32)

    demo = StreamDemo()
    barbet = FakeBarbetRuntime(checkpoints=(FakeBarbetCheckpoint(),))
    gateway = api.TTSGateway(demo, barbet_runtime=barbet)

    async def run_scenario():
        s1 = await gateway.synthesize_native_stream(
            request_id="voxcpm-1", model_id="__base__", text="voxcpm",
            control_instruction="", reference_path=None, prompt_text="",
            cfg_value=2.0, normalize=False, denoise=False, inference_timesteps=10,
        )
        assert await asyncio.to_thread(stream_running.wait, 2.0)

        barbet_task = asyncio.create_task(
            gateway.synthesize_barbet(
                request_id="barbet-1",
                model_id="barbet::barbet-tw-v1",
                text="barbet text",
                reference_path=None,
                prompt_text="",
                speaker_id="voice-a",
                cfg_value=2.0,
                inference_timesteps=10,
                seed=42,
            )
        )
        await asyncio.sleep(0.05)
        assert not barbet_task.done()

        release_stream.set()
        _ = [c async for c in s1]
        await s1.aclose()

        wav, headers = await barbet_task
        return wav, headers

    wav, headers = asyncio.run(run_scenario())
    assert wav.startswith(b"RIFF")
    assert headers["X-Engine-Concurrency"] == "1"
    assert gateway._session_gate.active_units == 0


def test_voxcpm_loop_thread_lifecycle_and_no_run_until_complete():
    class DummyServer:
        def __init__(self):
            self.loop = asyncio.new_event_loop()

    server = DummyServer()
    demo = api.VoxCPMDemo.__new__(api.VoxCPMDemo)
    demo._server_loop_thread = None
    demo._server_loop_lock = threading.Lock()
    demo.voxcpm_server = server

    demo._ensure_server_loop_running()
    assert demo._server_loop_thread is not None
    assert demo._server_loop_thread.is_alive()
    assert server.loop.is_running()

    thread = demo._server_loop_thread
    demo.stop_voxcpm()
    assert not server.loop.is_running()
    assert demo._server_loop_thread is None
    assert not thread.is_alive()


def test_stop_voxcpm_releases_cuda_cache(monkeypatch):
    """卸載要歸還顯存，與 BarbetRuntime._release_model 對稱。"""
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    demo = api.VoxCPMDemo.__new__(api.VoxCPMDemo)
    demo._server_loop_thread = None
    demo._server_loop_lock = threading.Lock()
    demo.voxcpm_server = object()

    demo.stop_voxcpm()

    assert calls == ["empty"]


def test_synthesize_without_model_id_uses_active_model(monkeypatch):
    """未指定 model_id 時沿用當前已載入模型，不觸發引擎切換。"""
    monkeypatch.setenv("VOXCPM_PRELOAD", "false")
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime(), mount_legacy=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/synthesize",
            data={"engine_id": "voxcpm2", "text": "測試句。"},
        )

    assert response.status_code == 200
    # 尚未載入任何模型時落到 base
    assert response.headers["X-Model-Version"] == "base::__base__"


def test_per_request_completion_short_sentences_return_early_without_waiting_long_sentence(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "4")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "4")

    class DelayDemo(FakeDemo):
        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            req = requests[0]
            text = req.get("text_input", "")
            if "long" in text:
                time.sleep(0.3)
            else:
                time.sleep(0.02)
            return [(16_000, np.zeros(320, dtype=np.float32))]

    demo = DelayDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        t0 = time.perf_counter()
        long_task = asyncio.create_task(
            synthesize_coalesced(gateway, "long-1", "long sentence text")
        )
        short_tasks = [
            asyncio.create_task(
                synthesize_coalesced(gateway, f"short-{i}", f"short text {i}")
            )
            for i in range(3)
        ]

        # 等待短句完成
        short_results = await asyncio.gather(*short_tasks)
        t_short = time.perf_counter() - t0

        # 短句應在 ~0.15s 內返回，此時長句尚未完成
        assert not long_task.done()
        assert t_short < 0.25

        # 等待長句完成
        long_result = await long_task
        t_long = time.perf_counter() - t0
        assert t_long >= 0.3

        await gateway.close_coalescer()
        return short_results, long_result

    short_results, long_result = asyncio.run(run_scenario())
    assert len(short_results) == 3
    assert long_result[0].startswith(b"RIFF")
    assert gateway._session_gate.active_units == 0


def test_per_request_completion_capacity_refill_immediate_dispatch(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "2")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "2")
    monkeypatch.setenv("VOXCPM_MAX_PENDING_SYNTHESIS", "2")

    task1_finish = threading.Event()
    task2_finish = threading.Event()
    task3_started = threading.Event()

    class RefillDemo(FakeDemo):
        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            req = requests[0]
            text = req.get("text_input", "")
            if text == "req-1":
                assert task1_finish.wait(timeout=2.0)
            elif text == "req-2":
                assert task2_finish.wait(timeout=2.0)
            elif text == "req-3":
                task3_started.set()
            return [(16_000, np.zeros(320, dtype=np.float32))]

    demo = RefillDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        t1 = asyncio.create_task(synthesize_coalesced(gateway, "1", "req-1"))
        t2 = asyncio.create_task(synthesize_coalesced(gateway, "2", "req-2"))
        await asyncio.sleep(0.02)
        assert gateway._session_gate.active_units == 2

        # 提交第 3 個請求（等待 gate 釋放）
        t3 = asyncio.create_task(synthesize_coalesced(gateway, "3", "req-3"))
        await asyncio.sleep(0.02)
        assert not task3_started.is_set()

        # 釋放第 1 個請求（release 1 單位）
        task1_finish.set()
        await t1

        # 第 3 個請求應立即遞補並開始執行（無需等第 2 個請求結束）
        assert await asyncio.to_thread(task3_started.wait, 1.0)
        assert not t2.done()

        task2_finish.set()
        await t2
        await t3
        await gateway.close_coalescer()

    asyncio.run(run_scenario())
    assert gateway._session_gate.active_units == 0


def test_per_request_completion_single_item_failure_isolation(monkeypatch):
    monkeypatch.setenv("VOXCPM_ENGINE_CONCURRENCY", "4")
    monkeypatch.setenv("VOXCPM_INTERACTIVE_BATCH_MAX", "4")

    class FailOneDemo(FakeDemo):
        def generate_tts_audio_batch(self, requests, *, return_exceptions=False):
            req = requests[0]
            text = req.get("text_input", "")
            if "fail" in text:
                raise ValueError("Intentional error in item")
            return [(16_000, np.zeros(320, dtype=np.float32))]

    demo = FailOneDemo()
    gateway = api.TTSGateway(demo, barbet_runtime=FakeBarbetRuntime())

    async def run_scenario():
        t_ok1 = asyncio.create_task(synthesize_coalesced(gateway, "ok1", "good text 1"))
        t_bad = asyncio.create_task(synthesize_coalesced(gateway, "bad", "fail text"))
        t_ok2 = asyncio.create_task(synthesize_coalesced(gateway, "ok2", "good text 2"))

        with pytest.raises(ValueError, match="Intentional error"):
            await t_bad

        res1 = await t_ok1
        res2 = await t_ok2
        await gateway.close_coalescer()
        return res1, res2

    res1, res2 = asyncio.run(run_scenario())
    assert res1[0].startswith(b"RIFF")
    assert res2[0].startswith(b"RIFF")
    assert gateway._session_gate.active_units == 0



def test_castvoice_model_version_reflects_runtime(monkeypatch):
    """model_version 必須含 git hash 與執行環境指紋，且可被環境變數覆寫。

    CastAgent 以此值作為快取 key 的一部分；同一份程式碼在不同 CUDA 棧
    （cu128／cu130）的節點上必須算出不同字串，否則跨節點的快取判斷失去依據。
    """
    import gateway.castvoice as gc

    version = gc._compute_castvoice_model_version()
    assert version.startswith("voxcpm360-castvoice-")
    parts = version.rsplit("-", 2)
    assert len(parts[-2]) == 8 and len(parts[-1]) == 8

    # 預設值改變 → 版本字串改變（相同輸入會產生不同輸出，快取必須失效）
    monkeypatch.setattr(gc, "_CASTVOICE_DEFAULT_CFG_VALUE", 3.0)
    assert gc._compute_castvoice_model_version() != version

    # 明確覆寫優先
    monkeypatch.setenv("VOXCPM_CASTVOICE_MODEL_VERSION", "pinned-1.0")
    assert gc._compute_castvoice_model_version() == "pinned-1.0"
