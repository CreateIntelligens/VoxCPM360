"""推論 worker 的背景看門狗：worker 被殺掉時主動重載。

呼叫端（openVman）看到 /api/v1/health 回 503 就跳過這台，所以沒有請求會進來；
重載若只靠請求觸發，worker 就永遠死著。2026-11-03 就這樣死了 41 分鐘。

worker 通常是被全機 OOM 殺的，所以不能一死就重載：可用記憶體不夠就先等，
重載失敗或重載完又馬上死就退避，不要連續重試把同機服務（Breeze 等）擠掉。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from typing import Any, ContextManager

from runtime_config import read_mem_available_gib, read_reload_min_available_gib

logger = logging.getLogger(__name__)


class WorkerWatchdog:
    def __init__(
        self,
        get_demo: Callable[[], Any],
        warmup: Callable[[Any], None],
        *,
        interval_seconds: float | None = None,
        min_available_gib: float | None = None,
        backoff_initial_seconds: float = 60.0,
        backoff_max_seconds: float = 900.0,
        stable_seconds: float = 300.0,
        read_available: Callable[[], float | None] = read_mem_available_gib,
        clock: Callable[[], float] = time.monotonic,
        reload_guard: ContextManager[Any] | None = None,
    ) -> None:
        self._get_demo = get_demo
        self._warmup = warmup
        # 查存活只是看子程序狀態，幾乎沒有成本；15 秒讓最壞情況也很快發現。
        self.interval_seconds = (
            float(os.environ.get("VOXCPM_WATCHDOG_INTERVAL_SECONDS", "15"))
            if interval_seconds is None
            else interval_seconds
        )
        self.min_available_gib = (
            read_reload_min_available_gib()
            if min_available_gib is None
            else min_available_gib
        )
        self._backoff_initial = backoff_initial_seconds
        self._backoff_max = backoff_max_seconds
        self._stable_seconds = stable_seconds
        self._read_available = read_available
        self._clock = clock
        self._reload_guard = reload_guard or contextlib.nullcontext()
        self._pending = False
        self._failures = 0
        self._next_attempt_at = 0.0
        self._last_reload_at: float | None = None
        self._waiting_for_memory = False
        self.reloading = False

    def _schedule_retry(self, now: float) -> None:
        delay = min(
            self._backoff_initial * 2 ** max(self._failures - 1, 0),
            self._backoff_max,
        )
        self._next_attempt_at = now + delay
        logger.warning(
            "worker watchdog: retry in %.0fs (failures=%d)", delay, self._failures
        )

    def tick(self) -> str:
        """檢查一次，回傳這次做了什麼（給測試與 log 用）。"""
        demo = self._get_demo()
        probe = getattr(demo, "worker_alive", None)
        alive = probe() if callable(probe) else None
        now = self._clock()

        if alive is True:
            self._pending = False
            self._waiting_for_memory = False
            if (
                self._failures
                and self._last_reload_at is not None
                and now - self._last_reload_at >= self._stable_seconds
            ):
                self._failures = 0
            return "healthy"
        if alive is None and not self._pending:
            # 尚未載入或無法判斷（含模型切換中），不是這裡要處理的狀況。
            return "idle"

        if not self._pending:
            self._pending = True
            logger.error("worker watchdog: nano-vllm worker is dead")
            if (
                self._last_reload_at is not None
                and now - self._last_reload_at < self._stable_seconds
            ):
                # 剛重載完又死，多半是記憶體還在被搶，直接重載只會再死一次。
                self._failures += 1
                self._schedule_retry(now)

        if now < self._next_attempt_at:
            return "backoff"

        available = self._read_available()
        if available is not None and available < self.min_available_gib:
            if not self._waiting_for_memory:
                self._waiting_for_memory = True
                logger.warning(
                    "worker watchdog: not reloading, %.1f GiB available "
                    "(need %.1f GiB)",
                    available,
                    self.min_available_gib,
                )
            return "low_memory"
        self._waiting_for_memory = False

        started_at = self._clock()
        self.reloading = True
        try:
            with self._reload_guard:
                # 等鎖期間可能已切換模型，或別的請求已經救回來了，要重新看一次。
                demo = self._get_demo()
                probe = getattr(demo, "worker_alive", None)
                if callable(probe) and probe() is True:
                    self._pending = False
                    return "healthy"
                demo.get_or_load_voxcpm()
                try:
                    # 跟啟動時一樣先熱機，/api/v1/health 恢復 200 時第一筆才不會慢。
                    self._warmup(demo)
                except Exception:
                    logger.exception("worker watchdog: warmup failed; continuing")
        except Exception:
            self._failures += 1
            logger.exception("worker watchdog: reload failed")
            self._schedule_retry(self._clock())
            return "failed"
        finally:
            self.reloading = False

        self._pending = False
        self._last_reload_at = self._clock()
        logger.warning(
            "worker watchdog: worker reloaded in %.1fs",
            self._last_reload_at - started_at,
        )
        return "reloaded"

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            check = asyncio.ensure_future(asyncio.to_thread(self.tick))
            try:
                await asyncio.shield(check)
            except asyncio.CancelledError:
                # 取消攔不住執行緒裡的重載；等它做完，關機才不會跟載入撞在一起。
                await asyncio.gather(check, return_exceptions=True)
                raise
            except Exception:
                logger.exception("worker watchdog: check failed")
