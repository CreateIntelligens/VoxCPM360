from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import secrets
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from anyio import CancelScope
import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from app import VoxCPMDemo, create_demo_interface
from voxcpm.barbet_registry import BarbetModelRegistry
from voxcpm.barbet_runtime import BarbetRuntime
from voxcpm.full_model_registry import FULL_MODEL_PREFIX, FullModelRegistry
from voxcpm.lora_registry import BASE_MODEL_KEY

logger = logging.getLogger(__name__)

from gateway.constants import BASE_MODEL_PREFIX, LORA_MODEL_PREFIX, PUBLIC_BASE_MODEL_ID
from gateway.streaming import _STREAMING_END, _StreamingReady, _PreparedSynthesisRequest, _NativeSynthesisStream, _ManagedStreamingResponse

@dataclass(eq=False)
class _SessionWaiter:
    engine_id: str
    model_id: str
    units: int
    future: asyncio.Future[int]


class _GPUSessionGate:
    """模型親和的 GPU 容量控制閘門。

    控制同時進入 GPU 的並發數（VOXCPM_ENGINE_CONCURRENCY）：
    - 同引擎、同模型：只要 refcount + units <= capacity 且前面無不同模型排隊者即可並發進入。
    - 不同模型/引擎（如 VoxCPM2 與 Barbet）：必須等待現有 session drain（refcount == 0）後才能切換。
    - Barbet：容量強制為 1 且與 VoxCPM2 互斥。
    - 嚴格維持跨模型/引擎的 FIFO 佇列順序，防止飢餓。
    """

    _EXCLUSIVE_ENGINES = frozenset({"barbet", "__exclusive__"})

    def __init__(self, concurrency: int = 4) -> None:
        self.concurrency = concurrency
        self._active_engine: str | None = None
        self._active_model_id: str | None = None
        self._active_units: int = 0
        self._waiters: list[_SessionWaiter] = []

    @property
    def active_units(self) -> int:
        return self._active_units

    @property
    def active_engine(self) -> str | None:
        return self._active_engine

    @property
    def active_model_id(self) -> str | None:
        return self._active_model_id

    def locked(self) -> bool:
        return self._active_units > 0

    def _effective_capacity(self, engine_id: str) -> int:
        if engine_id in self._EXCLUSIVE_ENGINES:
            return 1
        return self.concurrency

    def _can_join_active_session(
        self,
        engine_id: str,
        model_id: str,
        units: int,
    ) -> bool:
        return (
            self._active_units > 0
            and engine_id not in self._EXCLUSIVE_ENGINES
            and self._active_engine == engine_id
            and self._active_model_id == model_id
            and self._active_units + units <= self._effective_capacity(engine_id)
        )

    async def acquire(
        self,
        *,
        engine_id: str = "__exclusive__",
        model_id: str = "__exclusive__",
        units: int = 1,
        timeout: float | None = None,
    ) -> int:
        units = max(1, units)
        if not self._waiters:
            if self._active_units == 0:
                self._active_engine = engine_id
                self._active_model_id = model_id
                self._active_units = units
                return self._active_units
            if self._can_join_active_session(engine_id, model_id, units):
                self._active_units += units
                return self._active_units

        loop = asyncio.get_running_loop()
        future: asyncio.Future[int] = loop.create_future()
        waiter = _SessionWaiter(
            engine_id=engine_id,
            model_id=model_id,
            units=units,
            future=future,
        )
        self._waiters.append(waiter)

        try:
            if timeout is not None:
                refcount = await asyncio.wait_for(future, timeout=timeout)
            else:
                refcount = await future
            return refcount
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            elif future.done() and not future.cancelled() and not future.exception():
                self.release(units=units)
            self._wake_waiters()
            raise

    def release(self, units: int = 1) -> int:
        units = max(1, units)
        self._active_units = max(0, self._active_units - units)
        remaining = self._active_units
        if self._active_units == 0:
            self._active_engine = None
            self._active_model_id = None
        self._wake_waiters()
        return remaining

    def _wake_waiters(self) -> None:
        while self._waiters:
            waiter = self._waiters[0]
            if waiter.future.done():
                self._waiters.pop(0)
                continue

            if self._active_units == 0:
                self._active_engine = waiter.engine_id
                self._active_model_id = waiter.model_id
                self._active_units = waiter.units
            elif self._can_join_active_session(
                waiter.engine_id,
                waiter.model_id,
                waiter.units,
            ):
                self._active_units += waiter.units
            else:
                return

            self._waiters.pop(0)
            waiter.future.set_result(self._active_units)


@dataclass(eq=False)
class _NativeCoalescedItem:
    request_id: str
    model_id: str
    request: dict[str, Any]
    queued_at: float
    future: asyncio.Future[tuple[bytes, dict[str, str]]]


class _NativeCoalescer:
    """Collect queued interactive native requests without adding a time window."""

    def __init__(self, gateway: TTSGateway, *, batch_max: int) -> None:
        self._gateway = gateway
        self._batch_max = batch_max
        self._queue: list[_NativeCoalescedItem] = []
        self._queue_lock = asyncio.Lock()
        self._drain_task: asyncio.Task[None] | None = None
        self._completion_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    async def submit(
        self,
        *,
        request_id: str,
        model_id: str,
        request: dict[str, Any],
    ) -> tuple[bytes, dict[str, str]]:
        loop = asyncio.get_running_loop()
        async with self._queue_lock:
            if self._closed:
                raise RuntimeError("Native interactive coalescer is closed")
            queued_at = await self._gateway._admit_gpu_job(
                request_id=request_id,
                engine_id="voxcpm2",
                model_id=model_id,
            )
            item = _NativeCoalescedItem(
                request_id=request_id,
                model_id=model_id,
                request=request,
                queued_at=queued_at,
                future=loop.create_future(),
            )
            self._queue.append(item)
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain())

        try:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(item.future),
                    timeout=self._gateway._queue_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                if not await self._remove_if_pending(item):
                    return await asyncio.shield(item.future)
                item.future.cancel()
                await self._gateway._finish_gpu_jobs()
                queue_wait = time.perf_counter() - queued_at
                logger.warning(
                    "synthesis.request request_id=%s stage=rejected "
                    "reason=queue_timeout engine=voxcpm2 model=%s "
                    "queue_wait_seconds=%.3f",
                    request_id,
                    model_id,
                    queue_wait,
                )
                raise HTTPException(
                    status_code=503,
                    detail="等待語音生成資源逾時，請稍後再試",
                    headers={"Retry-After": "30", "X-Request-ID": request_id},
                ) from exc
        except asyncio.CancelledError:
            if await self._remove_if_pending(item):
                item.future.cancel()
                await self._gateway._finish_gpu_jobs()
            else:
                logger.warning(
                    "synthesis.request request_id=%s stage=client_cancelled "
                    "engine=voxcpm2 model=%s action=wait_for_batch",
                    request_id,
                    model_id,
                )
                try:
                    await asyncio.shield(item.future)
                except Exception as exc:  # noqa: BLE001 - client no longer receives this failure
                    logger.debug(
                        "synthesis.request request_id=%s stage=failed_after_cancel "
                        "engine=voxcpm2 model=%s error=%r",
                        request_id,
                        model_id,
                        exc,
                    )
            raise

    async def _remove_if_pending(self, item: _NativeCoalescedItem) -> bool:
        async with self._queue_lock:
            try:
                self._queue.remove(item)
            except ValueError:
                return False
            return True

    async def _take_batch(self) -> list[_NativeCoalescedItem]:
        async with self._queue_lock:
            if not self._queue:
                return []
            model_id = self._queue[0].model_id
            batch: list[_NativeCoalescedItem] = []
            while self._queue and len(batch) < self._batch_max and self._queue[0].model_id == model_id:
                batch.append(self._queue.pop(0))
            return batch

    async def _release_batch(self, canonical_id: str, batch_size: int) -> None:
        self._gateway._session_gate.release(batch_size)
        await self._gateway._finish_gpu_jobs(batch_size)
        logger.info(
            "synthesis.request stage=session_leave engine=voxcpm2 model=%s "
            "refcount=%d batch_size=%d",
            canonical_id,
            self._gateway._session_gate.active_units,
            batch_size,
        )

    async def _drain(self) -> None:
        while True:
            async with self._queue_lock:
                if not self._queue:
                    self._drain_task = None
                    return
                first_model = self._queue[0].model_id
                batch_count = 0
                for item in self._queue:
                    if item.model_id == first_model and batch_count < self._batch_max:
                        batch_count += 1
                    else:
                        break

            canonical_id = self._gateway.resolve_native_model_id(first_model)
            session_concurrency = await self._gateway._session_gate.acquire(
                engine_id="voxcpm2",
                model_id=canonical_id,
                units=batch_count,
                timeout=self._gateway._queue_timeout_seconds,
            )

            batch = await self._take_batch()
            if not batch:
                self._gateway._session_gate.release(units=batch_count)
                continue

            if len(batch) < batch_count:
                self._gateway._session_gate.release(units=batch_count - len(batch))

            logger.info(
                "synthesis.request stage=session_join engine=voxcpm2 model=%s "
                "refcount=%d capacity=%d batch_size=%d",
                canonical_id,
                self._gateway._session_gate.active_units,
                self._gateway._engine_concurrency,
                len(batch),
            )
            logger.info(
                "synthesis.request stage=coalesced batch_size=%d model=%s request_ids=%s",
                len(batch),
                first_model,
                ",".join(item.request_id for item in batch),
            )

            for item in batch:
                task = asyncio.create_task(
                    self._process_single_item(
                        item=item,
                        canonical_id=canonical_id,
                        submitted_batch_size=len(batch),
                    )
                )
                self._completion_tasks.add(task)
                task.add_done_callback(self._completion_tasks.discard)

    async def _process_single_item(
        self,
        item: _NativeCoalescedItem,
        canonical_id: str,
        submitted_batch_size: int,
    ) -> None:
        started_at = time.perf_counter()
        queue_wait = started_at - item.queued_at
        model_id = item.model_id
        request_id = item.request_id

        logger.info(
            "synthesis.request request_id=%s stage=started engine=voxcpm2 "
            "model=%s queue_wait_seconds=%.3f batch_size=%d",
            request_id,
            model_id,
            queue_wait,
            submitted_batch_size,
        )

        stage = "failed"
        try:
            sample_rate, audio = await self._gateway._generate_native_item_audio(
                model_id,
                item.request,
            )
            rendered_wav = await asyncio.to_thread(
                self._gateway._render_native_single_result,
                sample_rate,
                audio,
                item.request,
            )
            execution_time = time.perf_counter() - started_at
            try:
                timing_headers = self._gateway._job_timing_headers(
                    queue_wait,
                    execution_time,
                    concurrency=self._gateway._session_gate.active_units,
                )
            except TypeError:
                timing_headers = self._gateway._job_timing_headers(
                    queue_wait,
                    execution_time,
                )
            headers = {
                **timing_headers,
                "X-Batch-Size": str(submitted_batch_size),
            }
            if item.request.get("seed") is not None:
                # 與 Barbet 路徑一致：回報實際使用的種子（併發下為盡力重現）。
                headers["X-Random-Seed"] = str(item.request["seed"])
            if not item.future.done():
                item.future.set_result((rendered_wav, headers))
            stage = "completed"
        except Exception as exc:
            logger.exception(
                "synthesis.request request_id=%s stage=failed engine=voxcpm2 model=%s",
                request_id,
                model_id,
            )
            if not item.future.done():
                item.future.set_exception(exc)
        finally:
            self._gateway._session_gate.release(units=1)
            await self._gateway._finish_gpu_jobs(1)
            logger.info(
                "synthesis.request request_id=%s stage=%s engine=voxcpm2 "
                "model=%s queue_wait_seconds=%.3f execution_seconds=%.3f "
                "batch_size=%d active_units=%d",
                request_id,
                stage,
                model_id,
                queue_wait,
                time.perf_counter() - started_at,
                submitted_batch_size,
                self._gateway._session_gate.active_units,
            )

    async def close(self) -> None:
        self._closed = True
        async with self._queue_lock:
            pending = list(self._queue)
            self._queue.clear()
            drain_task = self._drain_task
        for item in pending:
            if not item.future.done():
                item.future.set_exception(RuntimeError("Native interactive coalescer is shutting down"))
            await self._gateway._finish_gpu_jobs()
        if drain_task is not None:
            await asyncio.shield(drain_task)
        if self._completion_tasks:
            await asyncio.gather(*self._completion_tasks, return_exceptions=True)


