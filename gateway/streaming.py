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

@dataclass(frozen=True)
class _StreamingReady:
    sample_rate: int


@dataclass
class _PreparedSynthesisRequest:
    engine_id: str
    model_id: str
    text: str
    control_instruction: str
    prompt_text: str
    speaker_id: str
    cfg_value: float
    inference_timesteps: int
    normalize: bool
    denoise: bool
    speed: float
    seed: int | None
    active_reference: str | None
    reference_label: str
    temp_path: str | None


_STREAMING_END = object()


class _NativeSynthesisStream:
    """已通過 admission 並持有 GPU gate 的非同步音訊 iterator。"""

    def __init__(
        self,
        *,
        queue: asyncio.Queue[Any],
        stop_event: threading.Event,
        worker_task: asyncio.Task[None],
        cleanup_done: asyncio.Event,
        sample_rate: int,
        session_concurrency: int = 1,
    ) -> None:
        self._queue = queue
        self._stop_event = stop_event
        self._worker_task = worker_task
        self._cleanup_done = cleanup_done
        self.sample_rate = sample_rate
        self.session_concurrency = session_concurrency
        self._closed = False

    def __aiter__(self) -> _NativeSynthesisStream:
        return self

    async def __anext__(self) -> np.ndarray:
        item = await self._queue.get()
        if item is _STREAMING_END:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return np.asarray(item, dtype=np.float32)

    async def aclose(self) -> None:
        if self._closed:
            await self._cleanup_done.wait()
            return
        self._closed = True
        self._stop_event.set()
        try:
            await asyncio.shield(self._worker_task)
        except asyncio.CancelledError:
            # CUDA worker 無法被 asyncio 取消；即使 client task 正在取消，仍須
            # 等 thread 完整離開後才可讓下一筆取得 GPU gate。
            await self._worker_task
            raise
        finally:
            await self._cleanup_done.wait()


class _ManagedStreamingResponse(StreamingResponse):
    """先 commit 再送 ASGI final body；final send 失敗則 rollback。"""

    def __init__(
        self,
        *args: Any,
        on_complete: Callable[[], Awaitable[None]],
        on_rollback: Callable[[], Awaitable[None]],
        on_close: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_complete = on_complete
        self._on_rollback = on_rollback
        self._on_close = on_close

    async def stream_response(self, send: Callable[..., Awaitable[None]]) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        async for chunk in self.body_iterator:
            if not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(self.charset)
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                }
            )
        try:
            # ASGI 2.3 以 cancel scope 回報 client disconnect。history 寫入在
            # thread 中不可取消，故必須讓 commit 完整落地後再 rollback；否則
            # rollback 可能先找不到檔案，thread 隨後才把失敗串流寫進 history。
            with CancelScope(shield=True):
                await self._on_complete()
            await send(
                {"type": "http.response.body", "body": b"", "more_body": False}
            )
        except BaseException:
            with CancelScope(shield=True):
                await self._on_rollback()
            raise

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[dict[str, Any]]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette 的 ASGI 2.3 disconnect listener 會取消 streaming task；
            # worker close 與 GPU gate 釋放仍須完成，不能繼承該 cancel scope。
            with CancelScope(shield=True):
                await self._on_close()


