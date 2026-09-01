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

# 相容層：保留舊的 import 路徑，避免既有呼叫端隨模組拆分而改寫。
from gateway.presets import _COSY_PROMPT_TEXT, _DEFAULT_CONTROL_INSTRUCTION, _DEFAULT_REFERENCE_PRESET_ID, _HISTORY_DIR, _LANG_NAN_TW, _LANG_ZH_TW, _MODEL_REGISTRY_PATH, _REFERENCE_AUDIO_DIR, _REFERENCE_AUDIO_PRESETS, _VOXCPM2_FIXED_TIMESTEPS, _by_id, _find_reference_preset
from gateway.history import _delete_generation_history, _load_generation_history, _save_generation_history, _wav_to_mp3
from gateway.castvoice import _CASTVOICE_BATCH_DIR, _CASTVOICE_BATCH_MAX_ITEMS, _CASTVOICE_DEFINITIONS, _CASTVOICE_DEFINITIONS_BY_ID, _CASTVOICE_MODEL_VERSION, _TTS_API_KEY
from gateway.constants import BASE_MODEL_PREFIX, LORA_MODEL_PREFIX, PUBLIC_BASE_MODEL_ID
from gateway.streaming import _STREAMING_END, _StreamingReady, _PreparedSynthesisRequest, _NativeSynthesisStream, _ManagedStreamingResponse
from gateway.concurrency import _SessionWaiter, _GPUSessionGate, _NativeCoalescedItem, _NativeCoalescer
from gateway.gateway import TTSGateway
from gateway.castvoice import CastVoiceSynthesizeRequest, CastVoiceBatchItemRequest, CastVoiceBatchRequest, _CastVoiceError, _CastVoiceBatchItem, _CastVoiceBatch, DynamicBatchSizer, CastVoiceBatchManager



from gateway import castvoice as gw_castvoice, history as gw_history, presets as gw_presets

def register_interactive_routes(app, gateway, history_lock):
    """互動合成路由群（自 api.create_app 搬遷，2026-08-31 phase 2b）。"""
    def _available_reference_presets() -> list[dict[str, str]]:
        return [
            {
                "id": preset["id"],
                "label": preset["label"],
                "description": preset["description"],
                "language": preset["language"],
                # 前端顯示用 —— 讓使用者看得到內建參考音的逐字稿，
                # 也能理解「為什麼不用自己填」（未上傳時後端會自動帶入）。
                "prompt_text": preset.get("prompt_text", ""),
            }
            for preset in _REFERENCE_AUDIO_PRESETS
            if (gw_presets._REFERENCE_AUDIO_DIR / preset["filename"]).is_file()
        ]

    async def _catalog_payload() -> dict[str, Any]:
        payload = await asyncio.to_thread(gateway.catalog)
        presets = _available_reference_presets()
        default_id = (
            _DEFAULT_REFERENCE_PRESET_ID
            if any(preset["id"] == _DEFAULT_REFERENCE_PRESET_ID for preset in presets)
            else (presets[0]["id"] if presets else "")
        )
        return {
            **payload,
            "reference_presets": presets,
            "default_reference_preset_id": default_id,
        }

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        catalog = await _catalog_payload()
        return {"status": "ok", **catalog}

    @app.get("/api/v1/catalog")
    async def catalog() -> dict[str, Any]:
        return await _catalog_payload()

    @app.get("/api/v1/models/registry")
    async def model_registry() -> dict[str, Any]:
        empty_registry: dict[str, Any] = {"_val_sets": {}, "models": []}
        try:
            payload = await asyncio.to_thread(
                lambda: json.loads(gw_presets._MODEL_REGISTRY_PATH.read_text(encoding="utf-8")),
            )
        except FileNotFoundError:
            logger.warning("Model registry not found at %s", gw_presets._MODEL_REGISTRY_PATH)
            return empty_registry
        except (OSError, json.JSONDecodeError):
            logger.exception("Unable to read model registry from %s", gw_presets._MODEL_REGISTRY_PATH)
            return empty_registry

        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            logger.error("Invalid model registry document at %s", gw_presets._MODEL_REGISTRY_PATH)
            return empty_registry
        return {
            "_val_sets": payload.get("_val_sets", {}),
            "models": payload["models"],
        }

    @app.get("/api/v1/history")
    async def generation_history(limit: int = 100) -> dict[str, Any]:
        safe_limit = min(max(limit, 1), 100)
        async with history_lock:
            items = await asyncio.to_thread(_load_generation_history, safe_limit)
        return {"items": items}

    @app.get("/api/v1/history/{history_id}/audio")
    async def generation_history_audio(history_id: str) -> FileResponse:
        try:
            if uuid.UUID(hex=history_id).hex != history_id:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="找不到生成紀錄") from exc
        audio_path = gw_history._HISTORY_DIR / f"{history_id}.wav"
        if not audio_path.is_file():
            raise HTTPException(status_code=404, detail="找不到生成音檔")
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename=f"voxcpm360-{history_id}.wav",
        )

    def _resolve_reference_audio(
        reference_preset_id: str,
    ) -> str | None:
        """解析白名單內建音檔，未指定時沿用環境覆寫或預設音檔。"""
        requested_id = reference_preset_id.strip()
        if requested_id:
            preset = _find_reference_preset(requested_id)
            if preset is None:
                raise HTTPException(
                    status_code=422,
                    detail="選取的內建參考聲音不存在",
                )
            path = gw_presets._REFERENCE_AUDIO_DIR / preset["filename"]
            if not path.is_file():
                raise HTTPException(
                    status_code=422,
                    detail=f"內建參考聲音目前不可用：{preset['label']}",
                )
            return str(path)

        override = os.environ.get("VOXCPM_DEFAULT_REFERENCE")
        if override is not None:
            if not override.strip():
                return None
            path = Path(override)
            return str(path) if path.is_file() else None

        default_preset = _find_reference_preset("")
        if default_preset is not None:
            path = gw_presets._REFERENCE_AUDIO_DIR / default_preset["filename"]
            if path.is_file():
                return str(path)
        return None

    def _remove_temp_file(temp_path: str | None) -> None:
        if not temp_path:
            return
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    async def _prepare_synthesis_request(
        *,
        engine_id: str,
        model_id: str,
        text: str,
        control_instruction: str,
        prompt_text: str,
        reference_preset_id: str,
        speaker_id: str,
        cfg_value: float,
        inference_timesteps: int,
        normalize: bool,
        denoise: bool,
        speed: float,
        seed: int | None,
        reference_audio: UploadFile | None,
        streaming: bool = False,
    ) -> _PreparedSynthesisRequest:
        text = text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="請輸入要合成的文字")
        if not 1.0 <= cfg_value <= 5.0:
            raise HTTPException(status_code=422, detail="CFG 必須介於 1.0 與 5.0")
        if not 1 <= inference_timesteps <= 50:
            raise HTTPException(status_code=422, detail="取樣步數必須介於 1 與 50")
        if not 0.5 <= speed <= 2.0:
            raise HTTPException(status_code=422, detail="語速必須介於 0.5 與 2.0")
        if streaming and engine_id != "voxcpm2":
            raise HTTPException(
                status_code=422,
                detail="串流端點目前僅支援 voxcpm2 引擎",
            )
        if streaming and abs(speed - 1.0) >= 1e-3:
            raise HTTPException(
                status_code=422,
                detail="串流端點不支援語速調整",
            )

        temp_path: str | None = None
        active_reference: str | None = None
        reference_label = "未指定參考音"
        try:
            if reference_audio is not None:
                uploaded_name = reference_audio.filename or "reference.wav"
                suffix = Path(uploaded_name).suffix or ".wav"
                payload = await reference_audio.read()
                reference_label = (
                    f"自訂：{uploaded_name}（{len(payload) / 1024 / 1024:.2f} MB）"
                )
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:
                    temp_file.write(payload)
                    temp_path = temp_file.name
                active_reference = temp_path
            else:
                active_reference = _resolve_reference_audio(reference_preset_id)
                selected_preset = _find_reference_preset(reference_preset_id)
                if selected_preset is not None:
                    # preset 的逐字稿屬於 cloning 條件，不是可省略的顯示文案。
                    if not prompt_text.strip():
                        prompt_text = selected_preset.get("prompt_text", "")
                    reference_label = (
                        f"{selected_preset['label']} · "
                        f"{selected_preset['description']}"
                    )

            if engine_id == "voxcpm2":
                try:
                    model_id = await asyncio.to_thread(
                        gateway.resolve_native_model_id,
                        model_id,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                # cloning 會把 reference + transcript 當成語音前綴；文字控制
                # 會被直接接到正文，兩條路徑不可並用。
                if active_reference and prompt_text.strip():
                    control_instruction = ""
                elif not control_instruction.strip():
                    control_instruction = _DEFAULT_CONTROL_INSTRUCTION
            elif engine_id != "barbet":
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到推論引擎：{engine_id}",
                )

            return _PreparedSynthesisRequest(
                engine_id=engine_id,
                model_id=model_id,
                text=text,
                control_instruction=control_instruction,
                prompt_text=prompt_text,
                speaker_id=speaker_id,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
                speed=speed,
                seed=seed,
                active_reference=active_reference,
                reference_label=reference_label,
                temp_path=temp_path,
            )
        except BaseException:
            _remove_temp_file(temp_path)
            raise

    async def _build_history_record(
        prepared: _PreparedSynthesisRequest,
        *,
        history_id: str,
        extra_headers: dict[str, str],
    ) -> dict[str, Any]:
        # catalog() 會 refresh registries 並掃 checkpoint 目錄，不可阻塞 event loop。
        catalog_payload = await asyncio.to_thread(gateway.catalog)
        selected_engine = _by_id(
            catalog_payload["engines"],
            prepared.engine_id,
        )
        selected_model = _by_id(
            selected_engine.get("models", []),
            prepared.model_id,
        )
        selected_speaker = _by_id(
            selected_model.get("speakers", []),
            prepared.speaker_id,
        )
        return {
            "id": history_id,
            "text": prepared.text,
            "engine_id": prepared.engine_id,
            "engine_label": selected_engine.get("label", prepared.engine_id),
            "model_id": prepared.model_id,
            "model_label": selected_model.get("label", prepared.model_id),
            "reference_label": prepared.reference_label,
            "speaker_label": selected_speaker.get("name"),
            "seed": int(extra_headers["X-Random-Seed"])
            if "X-Random-Seed" in extra_headers
            else None,
            "cfg_value": prepared.cfg_value,
            "inference_timesteps": prepared.inference_timesteps,
            "speed": prepared.speed,
            "normalize": prepared.normalize,
            "denoise": prepared.denoise,
            "prompt_text": prepared.prompt_text.strip() or None,
            "control_instruction": prepared.control_instruction.strip() or None,
            "duration_label": extra_headers.get("X-Synthesis-Time"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/api/v1/synthesize")
    async def synthesize(
        request: Request,
        engine_id: str = Form(...),
        model_id: str = Form(""),  # 留空 = 沿用當前已載入模型（不觸發切換）
        text: str = Form(...),
        control_instruction: str = Form(""),
        prompt_text: str = Form(""),
        reference_preset_id: str = Form(""),
        speaker_id: str = Form(""),
        cfg_value: float = Form(2.0),
        # 10 太低，diffusion 沒收斂完就輸出 —— 實聽是「亂叫、聽不懂」。
        # 25~30 明顯較穩；代價是生成時間約 3 倍。
        inference_timesteps: int = Form(30),
        # 預設開啟音量正規化 —— 關閉時實測輸出 peak 逼近 1.0（參考音僅 0.259），
        # 聽感是「音量爆掉像大聲吼叫」。
        normalize: bool = Form(True),
        denoise: bool = Form(False),
        # 模型無語速控制，這是合成後的 time-stretch。0.8~1.25 聽感乾淨，
        # 超出後相位聲碼器的金屬感開始明顯，故上下限收在 0.5/2.0。
        speed: float = Form(1.0),
        seed: int | None = Form(None),
        reference_audio: UploadFile | None = File(None),
    ) -> Response:
        request_id = uuid.uuid4().hex
        request_started_at = time.perf_counter()
        logger.info(
            "synthesis.request request_id=%s stage=received engine=%s model=%s "
            "content_type=%s text_chars=%d",
            request_id,
            engine_id,
            model_id,
            request.headers.get("content-type", ""),
            len(text),
        )
        prepared = await _prepare_synthesis_request(
            engine_id=engine_id,
            model_id=model_id,
            text=text,
            control_instruction=control_instruction,
            prompt_text=prompt_text,
            reference_preset_id=reference_preset_id,
            speaker_id=speaker_id,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            speed=speed,
            seed=seed,
            reference_audio=reference_audio,
        )
        try:
            if prepared.engine_id == "voxcpm2":
                wav, extra_headers = await gateway.synthesize_native_coalesced(
                    request_id=request_id,
                    model_id=prepared.model_id,
                    text=prepared.text,
                    control_instruction=prepared.control_instruction,
                    reference_path=prepared.active_reference,
                    prompt_text=prepared.prompt_text,
                    cfg_value=prepared.cfg_value,
                    normalize=prepared.normalize,
                    denoise=prepared.denoise,
                    inference_timesteps=prepared.inference_timesteps,
                    speed=prepared.speed,
                    seed=prepared.seed,
                )
            else:
                wav, extra_headers = await gateway.synthesize_barbet(
                    request_id=request_id,
                    model_id=prepared.model_id,
                    text=prepared.text,
                    reference_path=prepared.active_reference,
                    prompt_text=prepared.prompt_text,
                    speaker_id=prepared.speaker_id,
                    cfg_value=prepared.cfg_value,
                    inference_timesteps=prepared.inference_timesteps,
                    seed=prepared.seed,
                    speed=prepared.speed,
                )

            headers = {
                "Content-Disposition": 'inline; filename="tts-output.wav"',
                "X-Model-Engine": prepared.engine_id,
                "X-Model-Version": prepared.model_id,
                "X-Request-ID": request_id,
                **extra_headers,
            }
            # voxcpm2 引擎的 diffusion 步數在建構時固定，請求參數不生效；
            # 誠實回報實際生效值，客戶端不必猜（見 catalog capabilities）。
            if prepared.engine_id == "voxcpm2":
                headers["X-Inference-Timesteps-Effective"] = str(
                    _VOXCPM2_FIXED_TIMESTEPS
                )
            history_id = uuid.uuid4().hex
            record = await _build_history_record(
                prepared,
                history_id=history_id,
                extra_headers=extra_headers,
            )
            async with history_lock:
                await asyncio.to_thread(gw_history._save_generation_history, record, wav)
            headers["X-History-ID"] = history_id
            headers["X-Total-Time"] = f"{time.perf_counter() - request_started_at:.3f}s"
            return Response(content=wav, media_type="audio/wav", headers=headers)
        finally:
            _remove_temp_file(prepared.temp_path)

    @app.post("/api/v1/synthesize/stream")
    async def synthesize_stream(
        request: Request,
        engine_id: str = Form(...),
        model_id: str = Form(""),  # 留空 = 沿用當前已載入模型（不觸發切換）
        text: str = Form(...),
        control_instruction: str = Form(""),
        prompt_text: str = Form(""),
        reference_preset_id: str = Form(""),
        speaker_id: str = Form(""),
        cfg_value: float = Form(2.0),
        inference_timesteps: int = Form(30),
        normalize: bool = Form(True),
        denoise: bool = Form(False),
        speed: float = Form(1.0),
        seed: int | None = Form(None),
        reference_audio: UploadFile | None = File(None),
    ) -> Response:
        request_id = uuid.uuid4().hex
        logger.info(
            "synthesis.request request_id=%s stage=received engine=%s model=%s "
            "content_type=%s text_chars=%d streaming=true",
            request_id,
            engine_id,
            model_id,
            request.headers.get("content-type", ""),
            len(text),
        )
        prepared = await _prepare_synthesis_request(
            engine_id=engine_id,
            model_id=model_id,
            text=text,
            control_instruction=control_instruction,
            prompt_text=prompt_text,
            reference_preset_id=reference_preset_id,
            speaker_id=speaker_id,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            speed=speed,
            seed=seed,
            reference_audio=reference_audio,
            streaming=True,
        )
        try:
            stream = await gateway.synthesize_native_stream(
                request_id=request_id,
                model_id=prepared.model_id,
                text=prepared.text,
                control_instruction=prepared.control_instruction,
                reference_path=prepared.active_reference,
                prompt_text=prepared.prompt_text,
                cfg_value=prepared.cfg_value,
                normalize=prepared.normalize,
                denoise=prepared.denoise,
                inference_timesteps=prepared.inference_timesteps,
            )
        except BaseException:
            _remove_temp_file(prepared.temp_path)
            raise

        history_id = uuid.uuid4().hex
        audio_chunks: list[np.ndarray] = []

        async def body() -> AsyncIterator[bytes]:
            yield gateway._streaming_wav_header(stream.sample_rate)
            try:
                async for chunk in stream:
                    normalized = gateway._peak_normalize(
                        np.asarray(chunk, dtype=np.float32)
                    )
                    audio_chunks.append(normalized)
                    yield (
                        np.clip(normalized, -1.0, 1.0) * 32767
                    ).astype("<i2").tobytes()
            except asyncio.CancelledError:
                logger.warning(
                    "synthesis.request request_id=%s stage=client_cancelled "
                    "engine=voxcpm2 model=%s streaming=true",
                    request_id,
                    prepared.model_id,
                )
                raise
            except Exception:
                logger.exception(
                    "synthesis.request request_id=%s stage=failed "
                    "engine=voxcpm2 model=%s streaming=true",
                    request_id,
                    prepared.model_id,
                )
                # 已送出 200 時只能截斷 body；re-raise 可避免 ASGI 送出正常結尾。
                raise

        async def commit_history() -> None:
            audio = np.concatenate(audio_chunks, axis=0)
            wav = gateway._wav_response(stream.sample_rate, audio)
            record = await _build_history_record(
                prepared,
                history_id=history_id,
                extra_headers={},
            )
            async with history_lock:
                await asyncio.to_thread(gw_history._save_generation_history, record, wav)

        async def close_stream() -> None:
            try:
                await stream.aclose()
            finally:
                _remove_temp_file(prepared.temp_path)

        async def rollback_history() -> None:
            async with history_lock:
                await asyncio.to_thread(
                    _delete_generation_history,
                    history_id,
                )

        headers = {
            "Content-Disposition": 'inline; filename="tts-output.wav"',
            "X-Accel-Buffering": "no",
            "X-History-ID": history_id,
            "X-Model-Engine": "voxcpm2",
            "X-Model-Version": prepared.model_id,
            "X-Request-ID": request_id,
            "X-Sample-Rate": str(stream.sample_rate),
            "X-Engine-Concurrency": str(stream.session_concurrency),
        }
        return _ManagedStreamingResponse(
            body(),
            media_type="audio/wav",
            headers=headers,
            on_complete=commit_history,
            on_rollback=rollback_history,
            on_close=close_stream,
        )


    return {
        "_available_reference_presets": _available_reference_presets,
        "_resolve_reference_audio": _resolve_reference_audio,
    }
