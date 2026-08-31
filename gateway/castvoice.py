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
from gateway.concurrency import _SessionWaiter, _GPUSessionGate, _NativeCoalescedItem, _NativeCoalescer
from gateway.gateway import TTSGateway

class CastVoiceSynthesizeRequest(BaseModel):
    text: str
    voice_id: str
    format: str = "mp3"
    speed: float = 1.0
    # 選填：覆寫該 voice_id 預設綁定的模型/checkpoint，沿用同一個
    # 參考音／語者身份，只換底層權重。留空則用 voice_id 的預設模型。
    model_id: str | None = None


class CastVoiceBatchItemRequest(BaseModel):
    text: str
    voice_id: str
    speed: float = 1.0
    model_id: str | None = None


class CastVoiceBatchRequest(BaseModel):
    items: list[CastVoiceBatchItemRequest]


class _CastVoiceError(Exception):
    """單句合成失敗的統一錯誤形狀，帶著要回給呼叫端的 HTTP 狀態碼與訊息。"""

    def __init__(self, status_code: int, error: str, message: str, **headers: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.headers = headers


@dataclass
class _CastVoiceBatchItem:
    index: int
    text: str
    voice_id: str
    speed: float
    model_id: str | None = None
    status: str = "pending"  # pending | processing | done | failed
    error: str | None = None


@dataclass
class _CastVoiceBatch:
    items: list[_CastVoiceBatchItem]


class DynamicBatchSizer:
    """Choose a safe nano-vLLM chunk size from live GPU/container headroom.

    CUDA memory works for ordinary discrete GPUs and unified-memory devices
    whose nvidia-smi memory fields may be unavailable. Linux cgroup v1/v2
    headroom is a second ceiling; non-container hosts simply skip that check.
    """

    _GIB = 1024**3

    def __init__(self, max_concurrency: int, device: str = "cuda") -> None:
        self._max_concurrency = min(max_concurrency, 16)
        self._device = device if device.startswith("cuda") else None
        self._reserve_bytes = int(
            float(os.environ.get("VOXCPM_BATCH_MEMORY_RESERVE_GIB", "2.0"))
            * self._GIB
        )
        self._bytes_per_item = int(
            float(os.environ.get("VOXCPM_BATCH_MEMORY_PER_ITEM_GIB", "1.5"))
            * self._GIB
        )
        self._minimum_bytes_per_item = int(
            float(os.environ.get("VOXCPM_BATCH_MEMORY_MIN_PER_ITEM_GIB", "0.5"))
            * self._GIB
        )
        if self._bytes_per_item <= 0:
            raise ValueError("VOXCPM_BATCH_MEMORY_PER_ITEM_GIB must be > 0")
        if self._minimum_bytes_per_item <= 0:
            raise ValueError("VOXCPM_BATCH_MEMORY_MIN_PER_ITEM_GIB must be > 0")
        self._last_available_bytes: int | None = None
        self._best_size = 1
        self._best_work_rate = 0.0
        self._performance_cap = self._max_concurrency

    @staticmethod
    def _cgroup_headroom() -> int | None:
        paths = (
            (
                Path("/sys/fs/cgroup/memory.max"),
                Path("/sys/fs/cgroup/memory.current"),
            ),
            (
                Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            ),
        )
        for limit_path, current_path in paths:
            try:
                limit_raw = limit_path.read_text().strip()
                if limit_raw == "max":
                    continue
                limit = int(limit_raw)
                # cgroup v1 represents "unlimited" with a huge sentinel.
                if limit >= 1 << 60:
                    continue
                current = int(current_path.read_text().strip())
                return max(0, limit - current)
            except (OSError, ValueError):
                continue
        return None

    def _cuda_headroom(self) -> int | None:
        try:
            import torch

            if self._device is None or not torch.cuda.is_available():
                return None
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self._device)
            return int(free_bytes)
        except (ImportError, RuntimeError):
            return None

    def _available_bytes(self) -> int | None:
        headrooms = [
            value
            for value in (self._cuda_headroom(), self._cgroup_headroom())
            if value is not None
        ]
        return min(headrooms) if headrooms else None

    def recommend(self, pending_items: int) -> int:
        available = self._available_bytes()
        self._last_available_bytes = available
        if available is None:
            size = 1
            available_bytes = 0
        else:
            available_bytes = available
            usable_bytes = max(0, available_bytes - self._reserve_bytes)
            size = max(1, usable_bytes // self._bytes_per_item)
        selected = min(
            pending_items,
            self._max_concurrency,
            self._performance_cap,
            int(size),
        )
        logger.info(
            "castvoice.batch stage=sized pending=%d selected=%d "
            "available_gib=%.2f reserve_gib=%.2f estimated_item_gib=%.2f "
            "performance_cap=%d max=%d",
            pending_items,
            selected,
            available_bytes / self._GIB,
            self._reserve_bytes / self._GIB,
            self._bytes_per_item / self._GIB,
            self._performance_cap,
            self._max_concurrency,
        )
        return selected

    def observe_success(self, size: int, elapsed: float, work_units: int) -> None:
        """Learn marginal memory use and retain the best observed throughput."""
        available_after = self._available_bytes()
        if self._last_available_bytes is not None and available_after is not None:
            consumed = max(0, self._last_available_bytes - available_after)
            observed_per_item = consumed // max(1, size)
            self._bytes_per_item = max(
                self._minimum_bytes_per_item,
                int(self._bytes_per_item * 0.75 + observed_per_item * 0.25),
            )

        work_rate = work_units / elapsed if elapsed > 0 else 0.0
        if work_rate > self._best_work_rate * 1.05:
            self._best_work_rate = work_rate
            self._best_size = size
        elif (
            size > self._best_size
            and self._best_work_rate > 0
            and work_rate < self._best_work_rate * 0.85
        ):
            self._performance_cap = self._best_size
        logger.info(
            "castvoice.batch stage=learned size=%d work_rate=%.3f "
            "best_size=%d estimated_item_gib=%.2f performance_cap=%d",
            size,
            work_rate,
            self._best_size,
            self._bytes_per_item / self._GIB,
            self._performance_cap,
        )

    def observe_oom(self, size: int) -> None:
        """Shrink future chunks after an OOM; callers may retry smaller halves."""
        safe_size = max(1, size // 2)
        self._performance_cap = min(self._performance_cap, safe_size)
        self._bytes_per_item *= 2
        logger.warning(
            "castvoice.batch stage=oom size=%d new_cap=%d estimated_item_gib=%.2f",
            size,
            self._performance_cap,
            self._bytes_per_item / self._GIB,
        )


class CastVoiceBatchManager:
    """batch 合成的 in-process job queue。

    只有一個背景 worker，但同一 VoxCPM2 模型會以小批次送入 nano-vLLM
    做 continuous batching。這不會放寬全局 GPU gate：整個 chunk 仍只算一個
    GPU job，不同模型和 Barbet 也不會與它同時執行。
    取捨：process 重啟會遺失還沒跑完的 job（in-memory），但已完成的 mp3
    在完成當下就落地到磁碟，重啟後仍可透過 batch_id 查到已完成的部分
    （除非連 _batches 這個索引本身也重建 —— 目前沒做跨重啟的索引還原，
    這是刻意先求簡單，之後真的需要多機/不掉件再上 Redis）。
    """

    def __init__(
        self,
        *,
        synthesize_many: Callable[
            [list[_CastVoiceBatchItem]],
            Awaitable[list[tuple[bytes, str] | _CastVoiceError]],
        ],
        storage_dir: Path,
        chunk_size: Callable[[int], int],
    ) -> None:
        self._synthesize_many = synthesize_many
        self._storage_dir = storage_dir
        self._chunk_size = chunk_size
        self._batches: dict[str, _CastVoiceBatch] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def submit(self, requests: list[CastVoiceBatchItemRequest]) -> str:
        batch_id = uuid.uuid4().hex
        items = [
            _CastVoiceBatchItem(
                index=i,
                text=req.text,
                voice_id=req.voice_id,
                speed=req.speed,
                model_id=req.model_id,
            )
            for i, req in enumerate(requests)
        ]
        self._batches[batch_id] = _CastVoiceBatch(items=items)
        await asyncio.to_thread(
            (self._storage_dir / batch_id).mkdir, parents=True, exist_ok=True
        )
        self._queue.put_nowait(batch_id)
        return batch_id

    def get(self, batch_id: str) -> _CastVoiceBatch | None:
        return self._batches.get(batch_id)

    def audio_path(self, batch_id: str, index: int) -> Path:
        return self._storage_dir / batch_id / f"{index}.mp3"

    async def _run_worker(self) -> None:
        while True:
            batch_id = await self._queue.get()
            try:
                await self._process_batch(batch_id)
            except Exception:
                logger.exception(
                    "castvoice.batch stage=worker_failed batch_id=%s",
                    batch_id,
                )
            finally:
                self._queue.task_done()

    async def _process_batch(self, batch_id: str) -> None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return
        offset = 0
        while offset < len(batch.items):
            selected_size = self._chunk_size(len(batch.items) - offset)
            chunk = batch.items[offset : offset + selected_size]
            for item in chunk:
                item.status = "processing"
            started_at = time.perf_counter()
            results = await self._synthesize_many(chunk)
            if len(results) != len(chunk):
                raise RuntimeError("batch synthesizer returned an unexpected result count")
            for item, result in zip(chunk, results, strict=True):
                if isinstance(result, _CastVoiceError):
                    item.status = "failed"
                    item.error = result.message
                    continue
                mp3_bytes, _ = result
                await asyncio.to_thread(
                    self.audio_path(batch_id, item.index).write_bytes, mp3_bytes
                )
                item.status = "done"
            elapsed = time.perf_counter() - started_at
            logger.info(
                "castvoice.batch stage=chunk_completed batch_id=%s offset=%d "
                "size=%d elapsed_seconds=%.3f items_per_second=%.3f",
                batch_id,
                offset,
                len(chunk),
                elapsed,
                len(chunk) / elapsed if elapsed else 0.0,
            )
            offset += len(chunk)


