from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

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
    assert history_item["reference_label"].startswith("中年女聲")
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
