from __future__ import annotations

import asyncio
import threading
import time

from gateway.watchdog import WorkerWatchdog


class FakeDemo:
    def __init__(self, alive=True):
        self.alive = alive
        self.loads = 0
        self.load_error: Exception | None = None

    def worker_alive(self):
        return self.alive

    def get_or_load_voxcpm(self):
        self.loads += 1
        if self.load_error is not None:
            self.alive = None
            raise self.load_error
        self.alive = True


class Harness:
    def __init__(self, demo, *, available=100.0):
        self.demo = demo
        self.available = available
        self.now = 1000.0
        self.warmups = 0
        self.watchdog = WorkerWatchdog(
            lambda: demo,
            self._warmup,
            interval_seconds=15,
            min_available_gib=24,
            backoff_initial_seconds=60,
            backoff_max_seconds=900,
            stable_seconds=300,
            read_available=lambda: self.available,
            clock=lambda: self.now,
        )

    def _warmup(self, _demo):
        self.warmups += 1


def test_healthy_worker_is_left_alone():
    h = Harness(FakeDemo(alive=True))
    assert h.watchdog.tick() == "healthy"
    assert h.demo.loads == 0


def test_unloaded_or_unknown_worker_is_not_reloaded():
    h = Harness(FakeDemo(alive=None))
    assert h.watchdog.tick() == "idle"
    assert h.demo.loads == 0


def test_dead_worker_is_reloaded_and_warmed_up():
    h = Harness(FakeDemo(alive=False))
    assert h.watchdog.tick() == "reloaded"
    assert h.demo.loads == 1
    assert h.warmups == 1
    assert h.watchdog.reloading is False


def test_reloading_flag_is_set_until_warmup_finishes():
    h = Harness(FakeDemo(alive=False))
    seen = []
    h.watchdog._warmup = lambda _demo: seen.append(h.watchdog.reloading)
    h.watchdog.tick()
    assert seen == [True]
    assert h.watchdog.reloading is False


def test_warmup_failure_does_not_count_as_failed_reload():
    h = Harness(FakeDemo(alive=False))

    def broken(_demo):
        raise RuntimeError("warmup failed")

    h.watchdog._warmup = broken
    assert h.watchdog.tick() == "reloaded"
    assert h.demo.loads == 1


def test_low_memory_waits_then_reloads_once_memory_returns():
    h = Harness(FakeDemo(alive=False), available=23.9)
    assert h.watchdog.tick() == "low_memory"
    h.now += 15
    assert h.watchdog.tick() == "low_memory"
    assert h.demo.loads == 0

    h.available = 24.0
    h.now += 15
    assert h.watchdog.tick() == "reloaded"
    assert h.demo.loads == 1


def test_unreadable_memory_does_not_block_reload():
    h = Harness(FakeDemo(alive=False), available=None)
    assert h.watchdog.tick() == "reloaded"


def test_failed_reload_backs_off_exponentially_up_to_the_cap():
    h = Harness(FakeDemo(alive=False))
    h.demo.load_error = RuntimeError("boom")

    expected_delays = [60, 120, 240, 480, 900, 900]
    for attempt, delay in enumerate(expected_delays, start=1):
        assert h.watchdog.tick() == "failed"
        assert h.demo.loads == attempt
        # 退避期間不管檢查幾次都不能再載。
        h.now += delay - 1
        assert h.watchdog.tick() == "backoff"
        assert h.demo.loads == attempt
        h.now += 1


def test_recovers_after_failed_reload_even_though_server_is_gone():
    """重載失敗後 voxcpm_server 已被清掉、alive 變 None，不能因此當成閒置放棄。"""
    h = Harness(FakeDemo(alive=False))
    h.demo.load_error = RuntimeError("boom")
    assert h.watchdog.tick() == "failed"
    assert h.demo.alive is None

    h.demo.load_error = None
    h.now += 60
    assert h.watchdog.tick() == "reloaded"
    assert h.demo.alive is True


def test_dying_right_after_reload_backs_off_instead_of_looping():
    h = Harness(FakeDemo(alive=False))
    assert h.watchdog.tick() == "reloaded"

    h.demo.alive = False
    h.now += 30
    assert h.watchdog.tick() == "backoff"
    assert h.demo.loads == 1
    h.now += 60
    assert h.watchdog.tick() == "reloaded"
    assert h.demo.loads == 2

    # 再死一次，退避要加倍。
    h.demo.alive = False
    h.now += 30
    assert h.watchdog.tick() == "backoff"
    h.now += 60
    assert h.watchdog.tick() == "backoff"
    assert h.demo.loads == 2
    h.now += 60
    assert h.watchdog.tick() == "reloaded"


def test_death_long_after_reload_is_treated_as_new_incident():
    h = Harness(FakeDemo(alive=False))
    h.watchdog.tick()
    h.now += 400
    assert h.watchdog.tick() == "healthy"

    h.demo.alive = False
    h.now += 15
    assert h.watchdog.tick() == "reloaded"
    assert h.demo.loads == 2


def test_request_reload_clears_pending_state():
    h = Harness(FakeDemo(alive=False), available=1.0)
    assert h.watchdog.tick() == "low_memory"
    h.demo.alive = True
    assert h.watchdog.tick() == "healthy"
    assert h.demo.loads == 0
    h.demo.alive = None
    assert h.watchdog.tick() == "idle"


def _held_by_other_thread(lock):
    result = []

    def probe():
        got = lock.acquire(blocking=False)
        result.append(not got)
        if got:
            lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return result[0]


def test_reload_and_warmup_hold_the_runtime_guard():
    """切換模型會拿同一把鎖；重載途中切換，舊模型跟新模型會各佔一份引擎。"""
    guard = threading.RLock()
    h = Harness(FakeDemo(alive=False))
    h.watchdog._reload_guard = guard
    held = []
    original = h.demo.get_or_load_voxcpm

    def load():
        held.append(_held_by_other_thread(guard))
        original()

    h.demo.get_or_load_voxcpm = load
    h.watchdog._warmup = lambda _demo: held.append(_held_by_other_thread(guard))

    assert h.watchdog.tick() == "reloaded"
    assert held == [True, True]


def test_does_not_reload_old_runtime_after_switch_finished_while_waiting():
    """等鎖期間模型切換完成：要看切換後的 demo，不能把已卸載的舊模型載回來。"""
    old = FakeDemo(alive=False)
    new = FakeDemo(alive=True)
    current = {"demo": old}

    class SwitchWhileWaiting:
        def __enter__(self):
            current["demo"] = new

        def __exit__(self, *_exc):
            return False

    watchdog = WorkerWatchdog(
        lambda: current["demo"],
        lambda _demo: None,
        interval_seconds=15,
        min_available_gib=24,
        read_available=lambda: 100.0,
        clock=lambda: 1000.0,
        reload_guard=SwitchWhileWaiting(),
    )

    assert watchdog.tick() == "healthy"
    assert old.loads == 0
    assert new.loads == 0
    assert watchdog.reloading is False


def test_cancelling_run_waits_for_the_reload_in_progress():
    """關機時取消 asyncio 工作攔不住執行緒；要等重載做完才能關 gateway。"""
    started = threading.Event()
    finished = []

    class SlowDemo(FakeDemo):
        def get_or_load_voxcpm(self):
            started.set()
            time.sleep(0.2)
            finished.append(True)
            self.alive = True

    watchdog = WorkerWatchdog(
        lambda: demo,
        lambda _demo: None,
        interval_seconds=0,
        min_available_gib=24,
        read_available=lambda: 100.0,
    )
    demo = SlowDemo(alive=False)

    async def scenario():
        task = asyncio.create_task(watchdog.run())
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return list(finished)

    assert asyncio.run(scenario()) == [True]

