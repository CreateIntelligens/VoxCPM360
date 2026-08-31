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

class TTSGateway:
    def __init__(
        self,
        demo: VoxCPMDemo,
        barbet_runtime: BarbetRuntime | None = None,
    ) -> None:
        self.demo = demo
        self._base_demo = demo
        self._native_demo = demo
        self._native_runtime_id = BASE_MODEL_KEY
        self._active_native_selection = PUBLIC_BASE_MODEL_ID
        full_roots_setting = os.environ.get(
            "VOXCPM_FULL_MODEL_ROOTS",
            "/app/models/native:/app/checkpoints",
        )
        self.full_model_registry = FullModelRegistry(
            Path(value) for value in full_roots_setting.split(os.pathsep) if value.strip()
        )
        if barbet_runtime is None:
            barbet_roots_setting = os.environ.get(
                "VOXCPM_BARBET_MODEL_ROOTS",
                "/app/models/barbet",
            )
            barbet_registry = BarbetModelRegistry(
                Path(value) for value in barbet_roots_setting.split(os.pathsep) if value.strip()
            )
            barbet_runtime = BarbetRuntime(
                barbet_registry,
                device=os.environ.get("VOXCPM_DEVICE", "auto"),
            )
        self.barbet_runtime = barbet_runtime
        # VoxCPM2 and Barbet share the same physical GPU.  Separate per-engine
        # locks allow both runtimes to enter CUDA concurrently, so all model
        # loading and inference must pass through one process-wide gate.
        self._engine_concurrency = self._read_int_setting(
            "VOXCPM_ENGINE_CONCURRENCY",
            default=4,
            minimum=1,
        )
        self._session_gate = _GPUSessionGate(
            concurrency=self._engine_concurrency,
        )
        self._gpu_lock = self._session_gate
        self._admission_lock = asyncio.Lock()
        self._inflight_jobs = 0
        self._max_pending_jobs = self._read_int_setting(
            "VOXCPM_MAX_PENDING_SYNTHESIS",
            default=2,
            minimum=0,
        )
        self._queue_timeout_seconds = self._read_float_setting(
            "VOXCPM_QUEUE_TIMEOUT_SECONDS",
            default=120.0,
            minimum=0.001,
        )
        self._interactive_batch_max = min(
            self._read_int_setting(
                "VOXCPM_INTERACTIVE_BATCH_MAX",
                default=4,
                minimum=1,
            ),
            16,
        )
        self._native_coalescer = _NativeCoalescer(
            self,
            batch_max=self._interactive_batch_max,
        )

    @staticmethod
    def _read_int_setting(name: str, *, default: int, minimum: int) -> int:
        raw_value = os.environ.get(name, str(default))
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    @staticmethod
    def _read_float_setting(name: str, *, default: float, minimum: float) -> float:
        raw_value = os.environ.get(name, str(default))
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    async def _admit_gpu_job(
        self,
        *,
        request_id: str,
        engine_id: str,
        model_id: str,
    ) -> float:
        queued_at = time.perf_counter()
        async with self._admission_lock:
            capacity = self._max_pending_jobs + 1
            if self._inflight_jobs >= capacity:
                logger.warning(
                    "synthesis.request request_id=%s stage=rejected reason=queue_full "
                    "engine=%s model=%s inflight=%d capacity=%d",
                    request_id,
                    engine_id,
                    model_id,
                    self._inflight_jobs,
                    capacity,
                )
                raise HTTPException(
                    status_code=429,
                    detail="語音生成佇列已滿，請稍後再試",
                    headers={"Retry-After": "30", "X-Request-ID": request_id},
                )
            self._inflight_jobs += 1
            queue_position = self._inflight_jobs - 1

        logger.info(
            "synthesis.request request_id=%s stage=queued engine=%s model=%s queue_position=%d inflight=%d",
            request_id,
            engine_id,
            model_id,
            queue_position,
            self._inflight_jobs,
        )
        return queued_at

    async def _finish_gpu_jobs(self, count: int = 1) -> None:
        async with self._admission_lock:
            self._inflight_jobs -= count

    async def _run_gpu_job(
        self,
        *,
        request_id: str,
        engine_id: str,
        model_id: str,
        units: int = 1,
        work: Callable[[], Any],
    ) -> tuple[Any, float, float, int]:
        """Run one non-cancellable CUDA job behind the shared bounded gate.

        A thread already executing CUDA cannot be safely stopped by cancelling
        its asyncio waiter.  If the HTTP task is cancelled, keep the gate held
        until the worker thread exits so another model is never started on top
        of the abandoned job.
        """
        queued_at = await self._admit_gpu_job(
            request_id=request_id,
            engine_id=engine_id,
            model_id=model_id,
        )

        canonical_id = (
            self.resolve_native_model_id(model_id)
            if engine_id == "voxcpm2"
            else model_id
        )

        acquired = False
        started_at: float | None = None
        session_concurrency = 1
        try:
            try:
                session_concurrency = await self._session_gate.acquire(
                    engine_id=engine_id,
                    model_id=canonical_id,
                    units=units,
                    timeout=self._queue_timeout_seconds,
                )
                acquired = True
            except asyncio.TimeoutError as exc:
                queue_wait = time.perf_counter() - queued_at
                logger.warning(
                    "synthesis.request request_id=%s stage=rejected reason=queue_timeout "
                    "engine=%s model=%s queue_wait_seconds=%.3f",
                    request_id,
                    engine_id,
                    model_id,
                    queue_wait,
                )
                raise HTTPException(
                    status_code=503,
                    detail="等待語音生成資源逾時，請稍後再試",
                    headers={"Retry-After": "30", "X-Request-ID": request_id},
                ) from exc

            started_at = time.perf_counter()
            queue_wait = started_at - queued_at
            logger.info(
                "synthesis.request request_id=%s stage=session_join engine=%s model=%s "
                "refcount=%d capacity=%d",
                request_id,
                engine_id,
                canonical_id,
                session_concurrency,
                self._session_gate._effective_capacity(engine_id),
            )
            logger.info(
                "synthesis.request request_id=%s stage=started engine=%s model=%s "
                "queue_wait_seconds=%.3f",
                request_id,
                engine_id,
                model_id,
                queue_wait,
            )

            worker_task = asyncio.create_task(asyncio.to_thread(work))
            try:
                result = await asyncio.shield(worker_task)
            except asyncio.CancelledError:
                logger.warning(
                    "synthesis.request request_id=%s stage=client_cancelled "
                    "engine=%s model=%s action=wait_for_worker",
                    request_id,
                    engine_id,
                    model_id,
                )
                try:
                    await worker_task
                except Exception:
                    logger.exception(
                        "synthesis.request request_id=%s stage=failed_after_cancel "
                        "engine=%s model=%s",
                        request_id,
                        engine_id,
                        model_id,
                    )
                raise

            execution_time = time.perf_counter() - started_at
            logger.info(
                "synthesis.request request_id=%s stage=completed engine=%s model=%s "
                "queue_wait_seconds=%.3f execution_seconds=%.3f",
                request_id,
                engine_id,
                model_id,
                queue_wait,
                execution_time,
            )
            return result, queue_wait, execution_time, session_concurrency
        except (HTTPException, asyncio.CancelledError):
            raise
        except Exception:
            execution_time = (
                time.perf_counter() - started_at if started_at is not None else 0.0
            )
            logger.exception(
                "synthesis.request request_id=%s stage=failed engine=%s model=%s "
                "execution_seconds=%.3f",
                request_id,
                engine_id,
                model_id,
                execution_time,
            )
            raise
        finally:
            if acquired:
                self._session_gate.release(units=units)
                logger.info(
                    "synthesis.request request_id=%s stage=session_leave engine=%s model=%s "
                    "refcount=%d",
                    request_id,
                    engine_id,
                    canonical_id,
                    self._session_gate.active_units,
                )
            await self._finish_gpu_jobs(units)

    async def _run_gpu_job_streaming(
        self,
        *,
        request_id: str,
        model_id: str,
        work: Callable[
            [asyncio.Queue[Any], threading.Event, asyncio.AbstractEventLoop],
            None,
        ],
    ) -> _NativeSynthesisStream:
        """建立已取得 GPU gate 的 streaming session。

        Admission、排隊逾時與 worker 初始化都在回傳前完成，因此 429、503
        與首段生成前的錯誤仍能成為正式 HTTP status，而非已送出 200 後才截斷。
        """
        engine_id = "voxcpm2"
        queued_at = await self._admit_gpu_job(
            request_id=request_id,
            engine_id=engine_id,
            model_id=model_id,
        )

        canonical_id = self.resolve_native_model_id(model_id)
        acquired = False
        cleanup_owns_resources = False
        session_concurrency = 1
        try:
            try:
                session_concurrency = await self._session_gate.acquire(
                    engine_id=engine_id,
                    model_id=canonical_id,
                    units=1,
                    timeout=self._queue_timeout_seconds,
                )
                acquired = True
            except asyncio.TimeoutError as exc:
                queue_wait = time.perf_counter() - queued_at
                logger.warning(
                    "synthesis.request request_id=%s stage=rejected "
                    "reason=queue_timeout engine=%s model=%s "
                    "queue_wait_seconds=%.3f",
                    request_id,
                    engine_id,
                    model_id,
                    queue_wait,
                )
                raise HTTPException(
                    status_code=503,
                    detail="等待語音生成資源逾時，請稍後再試",
                    headers={"Retry-After": "30", "X-Request-ID": request_id},
                ) from exc

            started_at = time.perf_counter()
            queue_wait = started_at - queued_at
            logger.info(
                "synthesis.request request_id=%s stage=session_join engine=voxcpm2 model=%s "
                "refcount=%d capacity=%d",
                request_id,
                canonical_id,
                session_concurrency,
                self._session_gate.concurrency,
            )
            logger.info(
                "synthesis.request request_id=%s stage=started engine=%s model=%s "
                "queue_wait_seconds=%.3f",
                request_id,
                engine_id,
                model_id,
                queue_wait,
            )

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=8)
            stop_event = threading.Event()
            cleanup_done = asyncio.Event()
            worker_task = asyncio.create_task(
                asyncio.to_thread(work, queue, stop_event, loop)
            )
            cleanup_owns_resources = True

            async def cleanup() -> None:
                execution_time = time.perf_counter() - started_at
                if acquired:
                    self._session_gate.release(units=1)
                    logger.info(
                        "synthesis.request request_id=%s stage=session_leave engine=voxcpm2 model=%s "
                        "refcount=%d",
                        request_id,
                        canonical_id,
                        self._session_gate.active_units,
                    )
                await self._finish_gpu_jobs()
                logger.info(
                    "synthesis.request request_id=%s stage=worker_finished "
                    "engine=%s model=%s queue_wait_seconds=%.3f "
                    "execution_seconds=%.3f",
                    request_id,
                    engine_id,
                    model_id,
                    queue_wait,
                    execution_time,
                )
                cleanup_done.set()

            def schedule_cleanup(_: asyncio.Task[None]) -> None:
                loop.create_task(cleanup())

            worker_task.add_done_callback(schedule_cleanup)

            try:
                ready = await queue.get()
            except asyncio.CancelledError:
                stop_event.set()
                await asyncio.shield(worker_task)
                await cleanup_done.wait()
                raise
            if isinstance(ready, BaseException):
                stop_event.set()
                await worker_task
                await cleanup_done.wait()
                raise ready
            if not isinstance(ready, _StreamingReady):
                stop_event.set()
                await worker_task
                await cleanup_done.wait()
                raise RuntimeError("串流 worker 未回報取樣率")
            return _NativeSynthesisStream(
                queue=queue,
                stop_event=stop_event,
                worker_task=worker_task,
                cleanup_done=cleanup_done,
                sample_rate=ready.sample_rate,
                session_concurrency=session_concurrency,
            )
        except BaseException:
            if not cleanup_owns_resources and acquired:
                self._session_gate.release(units=1)
                logger.info(
                    "synthesis.request request_id=%s stage=session_leave engine=voxcpm2 model=%s "
                    "refcount=%d",
                    request_id,
                    canonical_id,
                    self._session_gate.active_units,
                )
            if not cleanup_owns_resources:
                await self._finish_gpu_jobs()
            raise

    def _native_models(self) -> list[dict[str, Any]]:
        self.demo.lora_registry.refresh()
        self.full_model_registry.refresh()
        models: list[dict[str, Any]] = [
            {
                "id": PUBLIC_BASE_MODEL_ID,
                "label": "VoxCPM2 基礎模型",
                "kind": "base",
                "description": "原生 MiniCPM4 Text-Semantic LM",
                "loaded": self._active_native_selection == PUBLIC_BASE_MODEL_ID,
            }
        ]
        for checkpoint in self.full_model_registry.checkpoints:
            models.append(
                {
                    "id": checkpoint.id,
                    "label": checkpoint.label,
                    "kind": "full",
                    "description": checkpoint.description,
                    "online": checkpoint.valid,
                    "loaded": self._active_native_selection == checkpoint.id,
                }
            )
        for checkpoint in self.demo.lora_registry.checkpoints:
            models.append(
                {
                    "id": f"{LORA_MODEL_PREFIX}{checkpoint.run_name}",
                    "label": checkpoint.label,
                    "kind": "lora",
                    "description": self.demo.lora_registry.describe(checkpoint.run_name),
                    "loaded": self._active_native_selection
                    == f"{LORA_MODEL_PREFIX}{checkpoint.run_name}",
                }
            )
        return models

    def resolve_native_model_id(self, model_id: str) -> str:
        """Return the canonical public id for a native VoxCPM2 model.

        Catalog ids are always namespaced (``base::``, ``full::``, ``lora::``).
        Legacy base/LoRA ids and bare full-checkpoint directory names remain
        accepted as unambiguous input aliases.
        """
        requested_id = model_id.strip()
        # 未指定模型 → 沿用當前已載入的模型，絕不觸發引擎切換（full↔base
        # 切換實測 ~80 秒，期間所有請求陪等）。尚未載入任何模型時落到
        # base。實際使用的模型由回應的 X-Model-Version 回報。
        if not requested_id:
            return self._active_native_selection or PUBLIC_BASE_MODEL_ID
        self.demo.lora_registry.refresh()
        self.full_model_registry.refresh()

        if requested_id in {BASE_MODEL_KEY, PUBLIC_BASE_MODEL_ID}:
            return PUBLIC_BASE_MODEL_ID

        full_checkpoints = {
            checkpoint.id: checkpoint
            for checkpoint in self.full_model_registry.checkpoints
            if checkpoint.valid
        }
        lora_names = {
            checkpoint.run_name for checkpoint in self.demo.lora_registry.checkpoints
        }

        if requested_id.startswith(FULL_MODEL_PREFIX):
            if requested_id in full_checkpoints:
                return requested_id
            raise ValueError(f"找不到模型 {requested_id}")

        if requested_id.startswith(LORA_MODEL_PREFIX):
            lora_name = requested_id.removeprefix(LORA_MODEL_PREFIX)
            if lora_name in lora_names:
                return requested_id
            raise ValueError(f"找不到模型 {requested_id}")

        full_id = f"{FULL_MODEL_PREFIX}{requested_id}"
        matches_full = full_id in full_checkpoints
        matches_lora = requested_id in lora_names
        if matches_full and matches_lora:
            raise ValueError(
                f"模型名稱 {requested_id} 同時存在 full 與 LoRA，請使用完整前綴"
            )
        if matches_full:
            return full_id
        if matches_lora:
            return f"{LORA_MODEL_PREFIX}{requested_id}"
        raise ValueError(f"找不到模型 {requested_id}")

    def _switch_native_runtime(self, model_id: str) -> tuple[VoxCPMDemo, str, str]:
        canonical_id = self.resolve_native_model_id(model_id)
        is_full_model = canonical_id.startswith(FULL_MODEL_PREFIX)
        is_lora = canonical_id.startswith(LORA_MODEL_PREFIX)
        desired_runtime_id = canonical_id if is_full_model else BASE_MODEL_KEY
        runtime_selection = (
            canonical_id.removeprefix(LORA_MODEL_PREFIX)
            if is_lora
            else BASE_MODEL_KEY
        )
        if desired_runtime_id == self._native_runtime_id:
            return self._native_demo, runtime_selection, canonical_id

        checkpoint = (
            self.full_model_registry.get(canonical_id) if is_full_model else None
        )
        previous_demo = self._native_demo
        previous_runtime_id = self._native_runtime_id
        previous_demo.stop_voxcpm()

        next_demo = (
            VoxCPMDemo(model_id=str(checkpoint.path), device=self._base_demo.device)
            if checkpoint is not None
            else self._base_demo
        )
        try:
            next_demo.get_or_load_voxcpm()
        except Exception:
            logger.exception("Failed to switch native runtime to %s", canonical_id)
            self._native_demo = previous_demo
            self._native_runtime_id = previous_runtime_id
            raise

        self._native_demo = next_demo
        self._native_runtime_id = desired_runtime_id
        logger.info("Native runtime switched to %s", desired_runtime_id)
        return next_demo, runtime_selection, canonical_id

    def close(self) -> None:
        stop = getattr(self._native_demo, "stop_voxcpm", None)
        if callable(stop):
            stop()
        self.barbet_runtime.close()

    async def close_coalescer(self) -> None:
        await self._native_coalescer.close()

    def _barbet_models(self) -> list[dict[str, Any]]:
        checkpoints = self.barbet_runtime.registry.refresh()
        return [
            {
                "id": checkpoint.id,
                "label": checkpoint.label,
                "kind": "checkpoint",
                "description": checkpoint.description,
                "online": checkpoint.valid,
                "loaded": checkpoint.id == self.barbet_runtime.loaded_model_id,
                "speakers": self.barbet_runtime.speakers(checkpoint),
            }
            for checkpoint in checkpoints
        ]

    def catalog(self) -> dict[str, Any]:
        barbet_models = self._barbet_models()

        engines = [
            {
                "id": "voxcpm2",
                "label": "VoxCPM2",
                "family": "minicpm",
                "description": "原生 VoxCPM2 與訓練版本",
                "online": True,
                "capabilities": {
                    "control_instruction": True,
                    "prompt_transcript": True,
                    "reference_audio": True,
                    "speaker_selection": False,
                    "seed": False,
                    "streaming": True,
                    # nano-vLLM 引擎在建構時固定 diffusion 步數
                    # （VOXCPM_INFERENCE_TIMESTEPS），per-request 的
                    # inference_timesteps 參數不生效 —— 前端據此隱藏該欄位。
                    "inference_timesteps": False,
                },
                "models": self._native_models(),
            }
        ]
        if barbet_models:
            engines.append(
                {
                    "id": "barbet",
                    "label": "Barbet",
                    "family": "barbet",
                    "description": "Barbet TSLM + VoxCPM2 聲學模型",
                    "online": any(model["online"] for model in barbet_models),
                    "capabilities": {
                        "control_instruction": False,
                        "prompt_transcript": True,
                        "reference_audio": True,
                        "speaker_selection": True,
                        "seed": True,
                        "streaming": False,
                        # Barbet 路徑每次請求真的會套用 inference_timesteps。
                        "inference_timesteps": True,
                    },
                    "models": barbet_models,
                }
            )
        return {"engines": engines}

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """後製變速。模型本身沒有語速控制 —— VoxCPM / Barbet 都是 AR + flow
        matching，沒有 duration predictor，語速隱含繼承自參考音。這裡用相位
        聲碼器做不變調的 time-stretch，是唯一不重訓就能調的手段。
        """
        if abs(speed - 1.0) < 1e-3:
            return audio
        return librosa.effects.time_stretch(
            np.asarray(audio, dtype=np.float32), rate=speed
        )

    @staticmethod
    def _peak_normalize(audio: np.ndarray, target_peak: float = 0.35) -> np.ndarray:
        """把輸出峰值壓到 target_peak，避免音量爆掉。

        Barbet 實測 peak 0.672（參考音僅 0.259），聽感是「音量過大像吼叫」。
        注意 API 的 `normalize` 參數是**文字**正規化（TextNormalizer，處理數字
        符號），與音量無關 —— 兩者別搞混。

        只縮不放：本來就小聲的輸出不動它，以免把底噪一起放大。
        """
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > target_peak:
            audio = audio * (target_peak / peak)
        return audio

    @classmethod
    def _wav_response(cls, sample_rate: int, audio: np.ndarray, speed: float = 1.0) -> bytes:
        audio = cls._apply_speed(np.asarray(audio, dtype=np.float32), speed)
        audio = cls._peak_normalize(audio)
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    @staticmethod
    def _streaming_wav_header(sample_rate: int) -> bytes:
        """建立 unknown-length PCM16 mono WAV header。"""
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            0xFFFFFFFF,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            0xFFFFFFFF,
        )

    @staticmethod
    def _job_timing_headers(
        queue_wait: float,
        execution_time: float,
        concurrency: int | None = None,
    ) -> dict[str, str]:
        headers = {
            "X-Queue-Wait": f"{queue_wait:.3f}s",
            "X-GPU-Job-Time": f"{execution_time:.3f}s",
        }
        if concurrency is not None:
            headers["X-Engine-Concurrency"] = str(concurrency)
        return headers

    def _render_native_single_result(
        self,
        sample_rate: int,
        audio: np.ndarray,
        request: dict[str, Any],
    ) -> bytes:
        return self._wav_response(
            sample_rate,
            audio,
            speed=float(request.get("speed", 1.0)),
        )

    async def _generate_native_item_audio(
        self,
        model_id: str,
        request: dict[str, Any],
    ) -> tuple[int, np.ndarray]:
        """單一請求走 demo 公開批次 API（單元素批）。

        prep（reference encode／denoise／暫存檔生命週期）由 demo 層自理；
        多個 item 併發呼叫時，引擎以 continuous batching 自然合流。
        不得直呼 demo 私有方法——測試替身只保證公開介面。
        """
        selected_demo, runtime_selection, canonical_id = self._switch_native_runtime(model_id)

        def run_single() -> tuple[int, np.ndarray]:
            if hasattr(selected_demo, "generate_tts_audio_batch"):
                results = selected_demo.generate_tts_audio_batch(
                    [
                        {
                            "text_input": request.get("text", ""),
                            "control_instruction": request.get("control_instruction", ""),
                            "reference_wav_path_input": request.get("reference_path"),
                            "prompt_text": request.get("prompt_text", ""),
                            "cfg_value_input": float(request.get("cfg_value", 2.0)),
                            "do_normalize": bool(request.get("normalize", True)),
                            "denoise": bool(request.get("denoise", False)),
                            "inference_timesteps": int(request.get("inference_timesteps", 10)),
                            "model_selection": runtime_selection,
                        }
                    ]
                )
                sample_rate, audio = results[0]
                return int(sample_rate), audio

            if hasattr(selected_demo, "generate_tts_audio"):
                sample_rate, audio = selected_demo.generate_tts_audio(
                    text_input=request.get("text", ""),
                    control_instruction=request.get("control_instruction", ""),
                    reference_wav_path_input=request.get("reference_path"),
                    prompt_text=request.get("prompt_text", ""),
                    cfg_value_input=float(request.get("cfg_value", 2.0)),
                    inference_timesteps=int(request.get("inference_timesteps", 10)),
                    model_selection=runtime_selection,
                    speed=float(request.get("speed", 1.0)),
                )
                return int(sample_rate), audio

            raise NotImplementedError("Demo has no generate_tts_audio_batch or generate_tts_audio")

        result = await asyncio.to_thread(run_single)
        self._active_native_selection = canonical_id
        return result

    async def synthesize_native(
        self,
        *,
        request_id: str,
        model_id: str,
        text: str,
        control_instruction: str,
        reference_path: str | None,
        prompt_text: str,
        cfg_value: float,
        normalize: bool,
        denoise: bool,
        inference_timesteps: int,
        speed: float = 1.0,
    ) -> tuple[bytes, dict[str, str]]:
        wavs, headers = await self.synthesize_native_batch(
            request_id=request_id,
            model_id=model_id,
            requests=[
                {
                    "text": text,
                    "control_instruction": control_instruction,
                    "reference_path": reference_path,
                    "prompt_text": prompt_text,
                    "cfg_value": cfg_value,
                    "normalize": normalize,
                    "denoise": denoise,
                    "inference_timesteps": inference_timesteps,
                    "speed": speed,
                }
            ],
        )
        return wavs[0], headers

    async def synthesize_native_coalesced(
        self,
        *,
        request_id: str,
        model_id: str,
        text: str,
        control_instruction: str,
        reference_path: str | None,
        prompt_text: str,
        cfg_value: float,
        normalize: bool,
        denoise: bool,
        inference_timesteps: int,
        speed: float = 1.0,
    ) -> tuple[bytes, dict[str, str]]:
        return await self._native_coalescer.submit(
            request_id=request_id,
            model_id=model_id,
            request={
                "text": text,
                "control_instruction": control_instruction,
                "reference_path": reference_path,
                "prompt_text": prompt_text,
                "cfg_value": cfg_value,
                "normalize": normalize,
                "denoise": denoise,
                "inference_timesteps": inference_timesteps,
                "speed": speed,
            },
        )

    async def synthesize_native_stream(
        self,
        *,
        request_id: str,
        model_id: str,
        text: str,
        control_instruction: str,
        reference_path: str | None,
        prompt_text: str,
        cfg_value: float,
        normalize: bool,
        denoise: bool,
        inference_timesteps: int,
    ) -> _NativeSynthesisStream:
        generation_request = {
            "text_input": text,
            "control_instruction": control_instruction,
            "reference_wav_path_input": reference_path,
            "prompt_text": prompt_text,
            "cfg_value_input": cfg_value,
            "do_normalize": normalize,
            "denoise": denoise,
            "inference_timesteps": inference_timesteps,
        }

        def work(
            queue: asyncio.Queue[Any],
            stop_event: threading.Event,
            loop: asyncio.AbstractEventLoop,
        ) -> None:
            generator: Any = None

            def put(item: Any) -> bool:
                future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                while True:
                    try:
                        future.result(timeout=0.1)
                        return True
                    except FutureTimeoutError:
                        if stop_event.is_set():
                            future.cancel()
                            return False

            try:
                selected_demo, runtime_selection, canonical_id = self._switch_native_runtime(model_id)
                server = selected_demo.get_or_load_voxcpm()
                if hasattr(selected_demo, "_call_engine_sync"):
                    info = selected_demo._call_engine_sync(server, "get_model_info")
                else:
                    info = server.get_model_info()
                sample_rate = int(info["sample_rate"])
                generator = selected_demo.generate_tts_audio_stream(
                    {**generation_request, "model_selection": runtime_selection}
                )
                # Python generator 的準備工作要到第一次 next() 才執行。先取首段，
                # 可讓 reference encode／denoise／空串流等前置失敗在 200 headers
                # 送出前被回報；TTFB 仍等同首段音訊實際可用的時間。
                first_chunk = next(generator)
                if not put(_StreamingReady(sample_rate=sample_rate)):
                    return
                if not put(first_chunk):
                    return
                for chunk in generator:
                    if stop_event.is_set() or not put(chunk):
                        break
                else:
                    self._active_native_selection = canonical_id
                    put(_STREAMING_END)
            except BaseException as exc:
                if not stop_event.is_set():
                    put(exc)
            finally:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()

        return await self._run_gpu_job_streaming(
            request_id=request_id,
            model_id=model_id,
            work=work,
        )

    async def synthesize_native_batch(
        self,
        *,
        request_id: str,
        model_id: str,
        requests: list[dict[str, Any]],
    ) -> tuple[list[bytes], dict[str, str]]:
        """Generate a same-model chunk within one GPU admission slot.

        ``VoxCPMDemo`` submits all requests to nano-vLLM's async server pool,
        which can continuously batch their text/audio tokens. Model switching
        and the shared VoxCPM2/Barbet GPU exclusion still happen exactly once
        around the whole chunk.
        """
        if not requests:
            return [], self._job_timing_headers(0.0, 0.0)

        def generate() -> list[tuple[int, np.ndarray]]:
            results = self._generate_native_batch_results(
                model_id,
                requests,
                isolate_errors=False,
            )
            for result in results:
                if isinstance(result, Exception):
                    raise result
            return [result for result in results if not isinstance(result, Exception)]

        results, queue_wait, execution_time, session_concurrency = await self._run_gpu_job(
            request_id=request_id,
            engine_id="voxcpm2",
            model_id=model_id,
            units=len(requests),
            work=generate,
        )
        wavs = [
            self._wav_response(
                sample_rate,
                audio,
                float(request.get("speed", 1.0)),
            )
            for (sample_rate, audio), request in zip(results, requests, strict=True)
        ]
        return wavs, self._job_timing_headers(
            queue_wait,
            execution_time,
            concurrency=session_concurrency,
        )

    def _generate_native_batch_results(
        self,
        model_id: str,
        requests: list[dict[str, Any]],
        *,
        isolate_errors: bool,
    ) -> list[tuple[int, np.ndarray] | Exception]:
        selected_demo, runtime_selection, canonical_id = self._switch_native_runtime(model_id)
        generation_requests = [
            {
                "text_input": request["text"],
                "control_instruction": request.get("control_instruction", ""),
                "reference_wav_path_input": request.get("reference_path"),
                "prompt_text": request.get("prompt_text", ""),
                "cfg_value_input": request.get("cfg_value", 2.0),
                "do_normalize": request.get("normalize", True),
                "denoise": request.get("denoise", True),
                "inference_timesteps": request.get("inference_timesteps", 10),
                "model_selection": runtime_selection,
            }
            for request in requests
        ]
        batch_generate = getattr(selected_demo, "generate_tts_audio_batch", None)
        if callable(batch_generate):
            if isolate_errors:
                results = batch_generate(
                    generation_requests,
                    return_exceptions=True,
                )
            else:
                results = batch_generate(generation_requests)
        elif isolate_errors:
            results = []
            for generation_request in generation_requests:
                try:
                    results.append(selected_demo.generate_tts_audio(**generation_request))
                except Exception as exc:  # noqa: BLE001 - isolate this request from its batch
                    results.append(exc)
        else:
            results = [
                selected_demo.generate_tts_audio(**generation_request) for generation_request in generation_requests
            ]
        self._active_native_selection = canonical_id
        return results

    def _render_native_batch_results(
        self,
        results: list[tuple[int, np.ndarray] | Exception],
        requests: list[dict[str, Any]],
    ) -> list[bytes | Exception]:
        rendered: list[bytes | Exception] = []
        for result, request in zip(results, requests, strict=True):
            if isinstance(result, Exception):
                rendered.append(result)
                continue
            try:
                sample_rate, audio = result
                rendered.append(
                    self._wav_response(
                        sample_rate,
                        audio,
                        float(request.get("speed", 1.0)),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate WAV post-processing failures
                rendered.append(exc)
        return rendered

    async def synthesize_barbet(
        self,
        *,
        request_id: str,
        model_id: str,
        text: str,
        reference_path: str | None,
        prompt_text: str,
        speaker_id: str,
        cfg_value: float,
        inference_timesteps: int,
        seed: int | None,
        speed: float = 1.0,
    ) -> tuple[bytes, dict[str, str]]:
        try:
            self.barbet_runtime.registry.get(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        actual_seed = seed if seed is not None else secrets.randbelow(2**31)

        def generate() -> tuple[int, np.ndarray, float]:
            return self.barbet_runtime.synthesize(
                model_id=model_id,
                text=text,
                reference_path=reference_path,
                prompt_text=prompt_text,
                speaker_id=speaker_id,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                seed=actual_seed,
            )

        result, queue_wait, execution_time, session_concurrency = await self._run_gpu_job(
            request_id=request_id,
            engine_id="barbet",
            model_id=model_id,
            units=1,
            work=generate,
        )
        sample_rate, audio, elapsed = result
        headers = {
            "X-Synthesis-Time": f"{elapsed:.2f}s",
            "X-Random-Seed": str(actual_seed),
            **self._job_timing_headers(
                queue_wait,
                execution_time,
                concurrency=session_concurrency,
            ),
        }
        return self._wav_response(sample_rate, audio, speed), headers

    def find_barbet_model_for_speaker(self, speaker_id: str) -> str | None:
        """回傳第一個含有此 speaker_id centroid、且可用的 checkpoint id。"""
        for checkpoint in self.barbet_runtime.registry.refresh():
            if not checkpoint.valid:
                continue
            if any(
                speaker["id"] == speaker_id
                for speaker in self.barbet_runtime.speakers(checkpoint)
            ):
                return checkpoint.id
        return None


