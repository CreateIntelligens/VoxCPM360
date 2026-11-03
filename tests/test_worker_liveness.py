"""推論子程序死掉（OOM killer）時：偵測、快速失敗、自動重載、health 回 503。"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

import api
import app as app_module
from app import EngineWorkerDied, VoxCPMDemo, engine_alive
from test_api_gateway import FakeBarbetRuntime, FakeDemo


class FakeProcess:
    def __init__(self, alive: bool):
        self.exitcode = None if alive else -9

    def is_alive(self):
        return self.exitcode is None


class FakeServerItem:
    def __init__(self, alive: bool):
        self.process = FakeProcess(alive)


class FakePool:
    def __init__(self, *alive: bool):
        self.servers = [FakeServerItem(a) for a in alive]

    async def hang_forever(self):
        await asyncio.sleep(3600)


class FakeSyncServer:
    """模仿 SyncVoxCPM2ServerPool：有 .loop 與 .server_pool。"""

    def __init__(self, *alive: bool):
        self.server_pool = FakePool(*alive)
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self.stopped = False

    def stop(self):
        self.stopped = True

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2)


def test_engine_alive_reports_process_state():
    assert engine_alive(FakeSyncServer(True)) is True
    assert engine_alive(FakeSyncServer(True, False)) is False
    assert engine_alive(FakePool(True)) is True  # 裸 async pool
    assert engine_alive(object()) is None  # 無法判斷


def _bare_demo(server):
    demo = VoxCPMDemo.__new__(VoxCPMDemo)
    demo.voxcpm_server = server
    demo._model_id = "ckpt-under-test"
    demo._server_loop_thread = None
    demo._server_loop_lock = threading.Lock()
    return demo


def test_call_engine_sync_fails_fast_when_worker_already_dead():
    server = FakeSyncServer(False)
    try:
        demo = _bare_demo(server)
        started = time.monotonic()
        with pytest.raises(EngineWorkerDied):
            demo._call_engine_sync(server, "hang_forever")
        assert time.monotonic() - started < 1.0
    finally:
        server.close()


def test_call_engine_sync_notices_worker_dying_mid_call():
    server = FakeSyncServer(True)
    try:
        demo = _bare_demo(server)

        def kill_later():
            time.sleep(0.3)
            server.server_pool.servers[0].process.exitcode = -9

        threading.Thread(target=kill_later, daemon=True).start()
        started = time.monotonic()
        with pytest.raises(EngineWorkerDied):
            demo._call_engine_sync(server, "hang_forever")
        # 舊版要等滿 300 秒；現在應在下一次每秒存活檢查就失敗
        assert time.monotonic() - started < 5.0
    finally:
        server.close()


def test_get_or_load_reloads_when_worker_is_dead(monkeypatch):
    dead = FakeSyncServer(False)
    fresh = object()
    try:
        demo = _bare_demo(dead)
        demo.device = "cuda:0"
        demo.optimize = False
        demo.gpu_memory_utilization = 0.35

        class Registry:
            checkpoints = ()

            def reset_registrations(self):
                pass

        demo.lora_registry = Registry()
        monkeypatch.setattr(app_module, "_shutdown_owned_engine_loop", lambda *a, **k: None)
        monkeypatch.setattr(app_module, "build_nano_lora_config", lambda _: {"max_lora_rank": 8})
        monkeypatch.setattr(app_module, "LoRAConfig", lambda **kwargs: kwargs)
        monkeypatch.setattr(app_module, "read_default_inference_timesteps", lambda: 10)
        monkeypatch.setattr(app_module.VoxCPM, "from_pretrained", lambda *a, **k: fresh)
        monkeypatch.setattr(demo, "_ensure_server_loop_running", lambda: None)
        monkeypatch.setattr(app_module, "read_mem_available_gib", lambda: 1000.0)

        assert demo.worker_alive() is False
        assert demo.get_or_load_voxcpm() is fresh
        assert dead.stopped is True
        assert demo.voxcpm_server is fresh
    finally:
        dead.close()


class FakeLoadedServer:
    # gateway 啟動時會檢查 generate 支援 inference_timesteps，假物件也要過
    def generate(self, *args, inference_timesteps=None, **kwargs):
        raise NotImplementedError


class DeadWorkerDemo(FakeDemo):
    def __init__(self):
        super().__init__()
        self.voxcpm_server = FakeLoadedServer()

    def worker_alive(self):
        return False


def test_health_returns_503_when_worker_is_dead():
    app = api.create_app(DeadWorkerDemo(), barbet_runtime=FakeBarbetRuntime())
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "worker_dead"
    assert payload["worker"]["alive"] is False
    assert payload["worker"]["loaded"] is True


def test_health_stays_ok_without_liveness_information():
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime())
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["worker"]["alive"] is None


def test_request_does_not_reload_dead_worker_when_memory_is_low(monkeypatch):
    """請求路徑也要守記憶體門檻，否則 watchdog 忍住了，第一筆請求還是會硬重載。"""
    import app as app_module

    class DeadServer:
        stopped = False

        def stop(self):
            self.stopped = True

    dead = DeadServer()
    demo = app_module.VoxCPMDemo.__new__(app_module.VoxCPMDemo)
    demo.voxcpm_server = dead
    monkeypatch.setattr(demo, "worker_alive", lambda: False)
    monkeypatch.setattr(app_module, "read_mem_available_gib", lambda: 5.0)
    monkeypatch.setenv("VOXCPM_WATCHDOG_MIN_AVAILABLE_GIB", "24")

    with pytest.raises(app_module.EngineWorkerDied, match="not reloading"):
        demo.get_or_load_voxcpm()

    assert dead.stopped is False
    assert demo.voxcpm_server is dead


def test_tts_health_reports_dead_worker():
    app = api.create_app(DeadWorkerDemo(), barbet_runtime=FakeBarbetRuntime())
    with TestClient(app) as client:
        response = client.get("/api/v1/tts/health")
    assert response.status_code == 503
    assert response.json() == {"status": "worker_dead"}


def test_tts_health_stays_ok_without_liveness_information():
    app = api.create_app(FakeDemo(), barbet_runtime=FakeBarbetRuntime())
    with TestClient(app) as client:
        response = client.get("/api/v1/tts/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_stays_503_until_watchdog_finishes_warmup():
    class AliveDemo(FakeDemo):
        def worker_alive(self):
            return True

    app = api.create_app(AliveDemo(), barbet_runtime=FakeBarbetRuntime())
    with TestClient(app) as client:
        watchdog = app.state.tts_gateway.worker_watchdog
        watchdog.reloading = True
        reloading = client.get("/api/v1/health")
        tts_reloading = client.get("/api/v1/tts/health")
        watchdog.reloading = False
        done = client.get("/api/v1/health")
    assert reloading.status_code == 503
    assert reloading.json()["status"] == "worker_dead"
    assert reloading.json()["worker"]["reloading"] is True
    assert tts_reloading.status_code == 503
    assert done.status_code == 200
    assert done.json()["worker"]["reloading"] is False


def test_model_switch_keeps_health_ok_and_holds_runtime_lock(monkeypatch, tmp_path):
    """切換模型不是 worker 死掉：/api/v1/health 要維持 200，否則呼叫端會在切換期間跳過這台。"""
    full = tmp_path / "other-full"
    full.mkdir()
    for filename in ("config.json", "tokenizer.json"):
        (full / filename).write_text("{}", encoding="utf-8")
    (full / "model.safetensors").write_bytes(b"weights")
    (full / "audiovae.pth").write_bytes(b"vae")
    monkeypatch.setenv("VOXCPM_FULL_MODEL_ROOTS", str(tmp_path))

    observed = {}
    app_holder = {}

    class SwitchingDemo(FakeDemo):
        def worker_alive(self):
            return None if self.model_id == "next" else True

        def get_or_load_voxcpm(self):
            gateway = app_holder["app"].state.tts_gateway
            observed["status"] = gateway.worker_status()
            result = []

            def probe():
                got = gateway._runtime_lock.acquire(blocking=False)
                result.append(not got)
                if got:
                    gateway._runtime_lock.release()

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join()
            observed["lock_held"] = result[0]
            return super().get_or_load_voxcpm()

    import gateway.gateway as gw_module

    monkeypatch.setattr(
        gw_module, "VoxCPMDemo", lambda **_kw: SwitchingDemo(model_id="next")
    )
    app = api.create_app(SwitchingDemo(), barbet_runtime=FakeBarbetRuntime())
    app_holder["app"] = app
    with TestClient(app):
        gateway = app.state.tts_gateway
        gateway._switch_native_runtime("full::other-full")

    assert observed["lock_held"] is True
    assert observed["status"]["alive"] is not False
    assert observed["status"]["reloading"] is False

