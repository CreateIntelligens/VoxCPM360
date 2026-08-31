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

# ── 自 gateway 套件 re-export（拆分相容層，2026-08-31）──
from gateway.presets import _COSY_PROMPT_TEXT, _DEFAULT_CONTROL_INSTRUCTION, _DEFAULT_REFERENCE_PRESET_ID, _HISTORY_DIR, _LANG_NAN_TW, _LANG_ZH_TW, _MODEL_REGISTRY_PATH, _REFERENCE_AUDIO_DIR, _REFERENCE_AUDIO_PRESETS, _VOXCPM2_FIXED_TIMESTEPS, _by_id, _find_reference_preset
from gateway.history import _delete_generation_history, _load_generation_history, _save_generation_history, _wav_to_mp3
from gateway.castvoice import _CASTVOICE_DEFAULT_CFG_VALUE, _CASTVOICE_DEFAULT_DENOISE, _CASTVOICE_DEFAULT_NORMALIZE, _CASTVOICE_BATCH_DIR, _CASTVOICE_BATCH_MAX_ITEMS, _CASTVOICE_DEFINITIONS, _CASTVOICE_DEFINITIONS_BY_ID, _CASTVOICE_MODEL_VERSION, _TTS_API_KEY
from gateway.constants import BASE_MODEL_PREFIX, LORA_MODEL_PREFIX, PUBLIC_BASE_MODEL_ID
from gateway.streaming import _STREAMING_END, _StreamingReady, _PreparedSynthesisRequest, _NativeSynthesisStream, _ManagedStreamingResponse
from gateway.concurrency import _SessionWaiter, _GPUSessionGate, _NativeCoalescedItem, _NativeCoalescer
from gateway.gateway import TTSGateway
from gateway.castvoice import CastVoiceSynthesizeRequest, CastVoiceBatchItemRequest, CastVoiceBatchRequest, _CastVoiceError, _CastVoiceBatchItem, _CastVoiceBatch, DynamicBatchSizer, CastVoiceBatchManager



from gateway import castvoice as gw_castvoice, history as gw_history, presets as gw_presets

def register_castvoice_routes(app, gateway, demo, helpers):
    """CastAgent /api/v1/tts/* 路由群（外部合約，phase 2c 純搬遷）。"""
    _available_reference_presets = helpers["_available_reference_presets"]
    _resolve_reference_audio = helpers["_resolve_reference_audio"]
    def _require_castvoice_auth(request: Request) -> None:
        """驗證 Authorization: Bearer <TTS_API_KEY>。未設定 TTS_API_KEY 時
        （本機開發）略過驗證。"""
        if not gw_castvoice._TTS_API_KEY:
            return
        auth_header = request.headers.get("authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or token != gw_castvoice._TTS_API_KEY:
            raise HTTPException(status_code=401, detail="缺少或無效的 API token")

    def _castvoice_error(status_code: int, error: str, message: str, **headers: str) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": error, "message": message},
            headers=headers or None,
        )

    def _castvoice_error_from_http(exc: HTTPException) -> _CastVoiceError:
        error_code = {429: "rate_limited", 503: "service_unavailable"}.get(
            exc.status_code, "request_failed"
        )
        return _CastVoiceError(
            exc.status_code,
            error_code,
            str(exc.detail),
            **dict(exc.headers or {}),
        )

    async def _resolve_native_model_override(model_id: str) -> str:
        """驗證使用者指定的 model_id 是否為合法的 voxcpm2 原生模型
        （base / full checkpoint / lora）。合法就回傳正式 namespaced id；
        舊 ID 與裸 checkpoint 名稱仍可作為別名，否則拋 model_not_found。"""
        try:
            return await asyncio.to_thread(gateway.resolve_native_model_id, model_id)
        except ValueError as exc:
            raise _CastVoiceError(
                400, "model_not_found", f"找不到模型 {model_id}"
            ) from exc

    async def _resolve_barbet_model_override(model_id: str, speaker_id: str) -> str:
        """驗證使用者指定的 model_id 是可用的 Barbet checkpoint，且該
        checkpoint 內含 speaker_id 的 centroid（否則克隆對象根本不存在）。"""
        checkpoints = await asyncio.to_thread(gateway.barbet_runtime.registry.refresh)
        checkpoint = next((c for c in checkpoints if c.id == model_id), None)
        if checkpoint is None or not checkpoint.valid:
            raise _CastVoiceError(400, "model_not_found", f"找不到模型 {model_id}")
        speakers = await asyncio.to_thread(gateway.barbet_runtime.speakers, checkpoint)
        if not any(speaker["id"] == speaker_id for speaker in speakers):
            raise _CastVoiceError(
                400, "model_not_found", f"模型 {model_id} 沒有語者 {speaker_id}"
            )
        return model_id


    def _castvoice_synthesis_params(
        *,
        speed: float,
        cfg_value: float | None,
        normalize: bool | None,
        denoise: bool | None,
    ) -> dict[str, Any]:
        """把選填的合成參數收斂成實際要傳給引擎的值，並套用範圍檢查。

        範圍界線與互動端點 /api/v1/synthesize 相同，但違規回 CastAgent 風格
        的 _CastVoiceError（400 invalid_request），不是 FastAPI 的 422。
        """
        if not 0.5 <= speed <= 2.0:
            raise _CastVoiceError(400, "invalid_speed", "speed 必須介於 0.5 與 2.0")
        if cfg_value is not None and not 1.0 <= cfg_value <= 5.0:
            raise _CastVoiceError(
                400, "invalid_request", "cfg_value 必須介於 1.0 與 5.0"
            )
        return {
            "speed": speed,
            "cfg_value": (
                _CASTVOICE_DEFAULT_CFG_VALUE if cfg_value is None else cfg_value
            ),
            "normalize": (
                _CASTVOICE_DEFAULT_NORMALIZE if normalize is None else normalize
            ),
            "denoise": _CASTVOICE_DEFAULT_DENOISE if denoise is None else denoise,
        }

    async def _synthesize_castvoice(
        text: str,
        voice_id: str,
        speed: float,
        model_id: str | None = None,
        *,
        seed: int | None = None,
        cfg_value: float | None = None,
        normalize: bool | None = None,
        denoise: bool | None = None,
    ) -> tuple[bytes, str]:
        """單句 CastAgent 語音合成的共用邏輯。單筆 /synthesize 與 batch worker
        都走這裡，失敗一律拋 _CastVoiceError，呼叫端自行決定怎麼回應
        （單筆轉成 HTTP response；batch 則記到該筆 item 的 error 欄位）。

        model_id 是選填的模型覆寫：留空沿用 voice_id 綁定的預設模型；有給
        則驗證該模型存在（且對 barbet 而言含有這個語者），沿用同一個參考音
        ／語者身份、只換底層權重。

        seed / cfg_value / normalize / denoise 同為選填，None 代表沿用
        castvoice 既有預設值。
        """
        text = text.strip()
        if not text:
            raise _CastVoiceError(400, "invalid_text", "text 不可為空白")
        params = _castvoice_synthesis_params(
            speed=speed,
            cfg_value=cfg_value,
            normalize=normalize,
            denoise=denoise,
        )

        definition = _CASTVOICE_DEFINITIONS_BY_ID.get(voice_id)
        if definition is None:
            raise _CastVoiceError(400, "voice_not_found", f"找不到語音 {voice_id}")

        request_id = uuid.uuid4().hex
        try:
            if definition["engine_id"] == "voxcpm2":
                preset = _find_reference_preset(definition["reference_preset_id"])
                if preset is None:
                    raise _CastVoiceError(
                        503, "voice_unavailable", f"語音 {voice_id} 目前無法使用"
                    )
                resolved_model_id = (
                    await _resolve_native_model_override(model_id)
                    if model_id
                    else PUBLIC_BASE_MODEL_ID
                )
                reference_path = _resolve_reference_audio(definition["reference_preset_id"])
                wav, _ = await gateway.synthesize_native(
                    request_id=request_id,
                    model_id=resolved_model_id,
                    text=text,
                    control_instruction="",
                    reference_path=reference_path,
                    prompt_text=preset.get("prompt_text", ""),
                    cfg_value=params["cfg_value"],
                    normalize=params["normalize"],
                    denoise=params["denoise"],
                    inference_timesteps=30,
                    speed=params["speed"],
                    seed=seed,
                )
            else:
                speaker_id = definition["speaker_id"]
                if model_id:
                    resolved_model_id = await _resolve_barbet_model_override(
                        model_id, speaker_id
                    )
                else:
                    resolved_model_id = await asyncio.to_thread(
                        gateway.find_barbet_model_for_speaker, speaker_id
                    )
                    if resolved_model_id is None:
                        raise _CastVoiceError(
                            503, "voice_unavailable", f"語音 {voice_id} 目前無法使用"
                        )
                wav, _ = await gateway.synthesize_barbet(
                    request_id=request_id,
                    model_id=resolved_model_id,
                    text=text,
                    reference_path=None,
                    prompt_text="",
                    speaker_id=speaker_id,
                    cfg_value=params["cfg_value"],
                    inference_timesteps=30,
                    seed=seed,
                    speed=params["speed"],
                )
        except HTTPException as exc:
            raise _castvoice_error_from_http(exc) from exc
        except _CastVoiceError:
            raise
        except Exception as exc:
            logger.exception(
                "castvoice.synthesize request_id=%s stage=failed voice_id=%s",
                request_id,
                voice_id,
            )
            raise _CastVoiceError(500, "internal_error", "語音合成失敗") from exc

        try:
            mp3_bytes = await asyncio.to_thread(_wav_to_mp3, wav)
        except RuntimeError as exc:
            logger.exception(
                "castvoice.synthesize request_id=%s stage=mp3_encode_failed", request_id
            )
            raise _CastVoiceError(500, "internal_error", "音檔轉檔失敗") from exc

        return mp3_bytes, request_id

    batch_sizer = DynamicBatchSizer(
        gateway._read_int_setting(
            "VOXCPM_BATCH_MAX_CONCURRENCY",
            default=16,
            minimum=1,
        ),
        device=demo.device,
    )

    def _is_out_of_memory(error: BaseException) -> bool:
        current: BaseException | None = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            message = str(current).lower()
            if "out of memory" in message or "cuda oom" in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    async def _synthesize_castvoice_many(
        items: list[_CastVoiceBatchItem],
    ) -> list[tuple[bytes, str] | _CastVoiceError]:
        """Use true nano-vLLM batching when a chunk targets one native model.

        Mixed engines/models and invalid items retain the single-item behavior,
        including independent errors, by falling back to ordered serial calls.
        """

        async def synthesize_serially() -> list[tuple[bytes, str] | _CastVoiceError]:
            serial_results: list[tuple[bytes, str] | _CastVoiceError] = []
            for item in items:
                try:
                    serial_results.append(
                        await _synthesize_castvoice(
                            item.text,
                            item.voice_id,
                            item.speed,
                            item.model_id,
                            seed=item.seed,
                            cfg_value=item.cfg_value,
                            normalize=item.normalize,
                            denoise=item.denoise,
                        )
                    )
                except _CastVoiceError as exc:
                    serial_results.append(exc)
            return serial_results

        if len(items) < 2:
            return await synthesize_serially()
        definitions = [
            _CASTVOICE_DEFINITIONS_BY_ID.get(item.voice_id) for item in items
        ]
        if any(
            definition is None or definition["engine_id"] != "voxcpm2"
            for definition in definitions
        ):
            return await synthesize_serially()

        try:
            resolved_models: list[str] = []
            requests: list[dict[str, Any]] = []
            for item, definition in zip(items, definitions, strict=True):
                text = item.text.strip()
                if not text:
                    raise _CastVoiceError(400, "invalid_text", "text 不可為空白")
                params = _castvoice_synthesis_params(
                    speed=item.speed,
                    cfg_value=item.cfg_value,
                    normalize=item.normalize,
                    denoise=item.denoise,
                )
                assert definition is not None
                preset = _find_reference_preset(definition["reference_preset_id"])
                if preset is None:
                    raise _CastVoiceError(
                        503,
                        "voice_unavailable",
                        f"語音 {item.voice_id} 目前無法使用",
                    )
                resolved_model_id = (
                    await _resolve_native_model_override(item.model_id)
                    if item.model_id
                    else PUBLIC_BASE_MODEL_ID
                )
                resolved_models.append(resolved_model_id)
                requests.append(
                    {
                        "text": text,
                        "control_instruction": "",
                        "reference_path": _resolve_reference_audio(
                            definition["reference_preset_id"]
                        ),
                        "prompt_text": preset.get("prompt_text", ""),
                        "cfg_value": params["cfg_value"],
                        "normalize": params["normalize"],
                        "denoise": params["denoise"],
                        "inference_timesteps": 30,
                        "speed": params["speed"],
                        "seed": item.seed,
                    }
                )
        except (_CastVoiceError, HTTPException):
            return await synthesize_serially()

        if len(set(resolved_models)) != 1:
            return await synthesize_serially()

        request_ids = [uuid.uuid4().hex for _ in items]
        started_at = time.perf_counter()
        try:
            wavs, _ = await gateway.synthesize_native_batch(
                request_id=f"batch-{request_ids[0]}",
                model_id=resolved_models[0],
                requests=requests,
            )
            mp3s = await asyncio.gather(
                *(asyncio.to_thread(_wav_to_mp3, wav) for wav in wavs)
            )
            batch_sizer.observe_success(
                len(items),
                time.perf_counter() - started_at,
                sum(max(1, len(item.text.strip())) for item in items),
            )
        except HTTPException as exc:
            error = _castvoice_error_from_http(exc)
            return [error for _ in items]
        except Exception as exc:
            if _is_out_of_memory(exc) and len(items) > 1:
                batch_sizer.observe_oom(len(items))
                midpoint = len(items) // 2
                logger.warning(
                    "castvoice.batch stage=retry_smaller model=%s old_size=%d "
                    "new_sizes=%d,%d",
                    resolved_models[0],
                    len(items),
                    midpoint,
                    len(items) - midpoint,
                )
                first = await _synthesize_castvoice_many(items[:midpoint])
                second = await _synthesize_castvoice_many(items[midpoint:])
                return [*first, *second]
            logger.exception(
                "castvoice.batch stage=native_chunk_failed model=%s size=%d",
                resolved_models[0],
                len(items),
            )
            error = _CastVoiceError(500, "internal_error", "語音合成失敗")
            return [error for _ in items]

        return list(zip(mp3s, request_ids, strict=True))

    batch_manager = CastVoiceBatchManager(
        synthesize_many=_synthesize_castvoice_many,
        storage_dir=gw_castvoice._CASTVOICE_BATCH_DIR,
        chunk_size=batch_sizer.recommend,
    )
    logger.info("CastVoice dynamic batching enabled (runtime maximum: 16)")
    app.state.castvoice_batch_manager = batch_manager

    @app.get("/api/v1/tts/health")
    async def castvoice_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/tts/voices")
    async def castvoice_voices() -> dict[str, Any]:
        available_preset_ids = {
            preset["id"] for preset in _available_reference_presets()
        }
        voices: list[dict[str, str]] = []
        for definition in _CASTVOICE_DEFINITIONS:
            if definition["engine_id"] == "voxcpm2":
                if definition["reference_preset_id"] not in available_preset_ids:
                    continue
            elif definition["engine_id"] == "barbet":
                model_id = await asyncio.to_thread(
                    gateway.find_barbet_model_for_speaker, definition["speaker_id"]
                )
                if model_id is None:
                    continue
            voice: dict[str, str] = {
                "voice_id": definition["voice_id"],
                "label": definition["label"],
                "gender": definition["gender"],
                "language": definition["language"],
                "desc": definition["desc"],
            }
            voices.append(voice)
        return {"model_version": _CASTVOICE_MODEL_VERSION, "voices": voices}

    @app.post("/api/v1/tts/synthesize")
    async def castvoice_synthesize(request: Request, body: CastVoiceSynthesizeRequest) -> Response:
        _require_castvoice_auth(request)

        if body.format != "mp3":
            return _castvoice_error(
                400, "unsupported_format", f"不支援的格式：{body.format}，目前僅支援 mp3"
            )

        try:
            mp3_bytes, request_id = await _synthesize_castvoice(
                body.text,
                body.voice_id,
                body.speed,
                body.model_id,
                seed=body.seed,
                cfg_value=body.cfg_value,
                normalize=body.normalize,
                denoise=body.denoise,
            )
        except _CastVoiceError as exc:
            return _castvoice_error(exc.status_code, exc.error, exc.message, **exc.headers)

        return Response(
            content=mp3_bytes,
            media_type="audio/mpeg",
            headers={"X-Request-ID": request_id},
        )

    @app.post("/api/v1/tts/synthesize/batch")
    async def castvoice_synthesize_batch(
        request: Request, body: CastVoiceBatchRequest
    ) -> Response:
        _require_castvoice_auth(request)

        if not body.items:
            return _castvoice_error(400, "invalid_batch", "items 不可為空")
        if len(body.items) > _CASTVOICE_BATCH_MAX_ITEMS:
            return _castvoice_error(
                400,
                "batch_too_large",
                f"單批次最多 {_CASTVOICE_BATCH_MAX_ITEMS} 筆，收到 {len(body.items)} 筆",
            )

        batch_id = await batch_manager.submit(body.items)
        return JSONResponse(
            status_code=202,
            content={
                "batch_id": batch_id,
                "total": len(body.items),
                "status_url": f"/api/v1/tts/synthesize/batch/{batch_id}",
            },
        )

    @app.get("/api/v1/tts/synthesize/batch/{batch_id}")
    async def castvoice_batch_status(request: Request, batch_id: str) -> Response:
        _require_castvoice_auth(request)
        batch = batch_manager.get(batch_id)
        if batch is None:
            return _castvoice_error(404, "batch_not_found", f"找不到 batch {batch_id}")

        counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
        items_payload: list[dict[str, Any]] = []
        for item in batch.items:
            counts[item.status] += 1
            entry: dict[str, Any] = {"index": item.index, "status": item.status}
            if item.status == "done":
                entry["audio_url"] = (
                    f"/api/v1/tts/synthesize/batch/{batch_id}/{item.index}/audio"
                )
            elif item.status == "failed":
                entry["error"] = item.error
            items_payload.append(entry)

        return JSONResponse(
            status_code=200,
            content={
                "batch_id": batch_id,
                "total": len(batch.items),
                "counts": counts,
                "done": counts["done"] + counts["failed"] == len(batch.items),
                "items": items_payload,
            },
        )

    @app.get("/api/v1/tts/synthesize/batch/{batch_id}/{index}/audio")
    async def castvoice_batch_item_audio(
        request: Request, batch_id: str, index: int
    ) -> Response:
        _require_castvoice_auth(request)
        batch = batch_manager.get(batch_id)
        if batch is None or not 0 <= index < len(batch.items):
            return _castvoice_error(404, "item_not_found", "找不到該筆項目")
        item = batch.items[index]
        if item.status != "done":
            return _castvoice_error(404, "item_not_ready", "該筆尚未完成或已失敗")
        audio_path = batch_manager.audio_path(batch_id, index)
        if not audio_path.is_file():
            return _castvoice_error(404, "item_not_found", "音檔不存在")
        return FileResponse(audio_path, media_type="audio/mpeg")


    return batch_manager
