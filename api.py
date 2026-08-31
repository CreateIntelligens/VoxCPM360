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
from gateway.castvoice import _CASTVOICE_BATCH_DIR, _CASTVOICE_BATCH_MAX_ITEMS, _CASTVOICE_DEFINITIONS, _CASTVOICE_DEFINITIONS_BY_ID, _CASTVOICE_MODEL_VERSION, _TTS_API_KEY
from gateway.constants import BASE_MODEL_PREFIX, LORA_MODEL_PREFIX, PUBLIC_BASE_MODEL_ID
from gateway.streaming import _STREAMING_END, _StreamingReady, _PreparedSynthesisRequest, _NativeSynthesisStream, _ManagedStreamingResponse
from gateway.concurrency import _SessionWaiter, _GPUSessionGate, _NativeCoalescedItem, _NativeCoalescer
from gateway.gateway import TTSGateway
from gateway.castvoice import CastVoiceSynthesizeRequest, CastVoiceBatchItemRequest, CastVoiceBatchRequest, _CastVoiceError, _CastVoiceBatchItem, _CastVoiceBatch, DynamicBatchSizer, CastVoiceBatchManager



def create_app(
    demo: VoxCPMDemo | None = None,
    *,
    barbet_runtime: BarbetRuntime | None = None,
    mount_legacy: bool = True,
) -> FastAPI:
    demo = demo or VoxCPMDemo(
        model_id=os.environ.get("MODEL_ID", "openbmb/VoxCPM2"),
        device=os.environ.get("VOXCPM_DEVICE", "auto"),
    )
    gateway = TTSGateway(demo, barbet_runtime=barbet_runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if os.environ.get("VOXCPM_PRELOAD", "true").lower() == "true":
            logger.info("Preloading VoxCPM2 runtime")
            await asyncio.to_thread(demo.get_or_load_voxcpm)
        app.state.castvoice_batch_manager.start()
        try:
            yield
        finally:
            await app.state.castvoice_batch_manager.stop()
            await gateway.close_coalescer()
            await asyncio.to_thread(gateway.close)

    app = FastAPI(
        title="VoxCPM 360 Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.tts_gateway = gateway
    history_lock = asyncio.Lock()

    # 缺檔的 preset 會被 _available_reference_presets 靜默濾掉，
    # 啟動時先記 warning，避免資產遺失只表現為「選單少一項」。
    for preset in _REFERENCE_AUDIO_PRESETS:
        preset_path = _REFERENCE_AUDIO_DIR / preset["filename"]
        if not preset_path.is_file():
            logger.warning(
                "reference preset audio missing; hidden from catalog: id=%s path=%s",
                preset["id"],
                preset_path,
            )

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
            if (_REFERENCE_AUDIO_DIR / preset["filename"]).is_file()
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
                lambda: json.loads(_MODEL_REGISTRY_PATH.read_text(encoding="utf-8")),
            )
        except FileNotFoundError:
            logger.warning("Model registry not found at %s", _MODEL_REGISTRY_PATH)
            return empty_registry
        except (OSError, json.JSONDecodeError):
            logger.exception("Unable to read model registry from %s", _MODEL_REGISTRY_PATH)
            return empty_registry

        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            logger.error("Invalid model registry document at %s", _MODEL_REGISTRY_PATH)
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
        audio_path = _HISTORY_DIR / f"{history_id}.wav"
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
            path = _REFERENCE_AUDIO_DIR / preset["filename"]
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
            path = _REFERENCE_AUDIO_DIR / default_preset["filename"]
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
                await asyncio.to_thread(_save_generation_history, record, wav)
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
                await asyncio.to_thread(_save_generation_history, record, wav)

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

    def _require_castvoice_auth(request: Request) -> None:
        """驗證 Authorization: Bearer <TTS_API_KEY>。未設定 TTS_API_KEY 時
        （本機開發）略過驗證。"""
        if not _TTS_API_KEY:
            return
        auth_header = request.headers.get("authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or token != _TTS_API_KEY:
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

    async def _synthesize_castvoice(
        text: str, voice_id: str, speed: float, model_id: str | None = None
    ) -> tuple[bytes, str]:
        """單句 CastAgent 語音合成的共用邏輯。單筆 /synthesize 與 batch worker
        都走這裡，失敗一律拋 _CastVoiceError，呼叫端自行決定怎麼回應
        （單筆轉成 HTTP response；batch 則記到該筆 item 的 error 欄位）。

        model_id 是選填的模型覆寫：留空沿用 voice_id 綁定的預設模型；有給
        則驗證該模型存在（且對 barbet 而言含有這個語者），沿用同一個參考音
        ／語者身份、只換底層權重。
        """
        text = text.strip()
        if not text:
            raise _CastVoiceError(400, "invalid_text", "text 不可為空白")
        if not 0.5 <= speed <= 2.0:
            raise _CastVoiceError(400, "invalid_speed", "speed 必須介於 0.5 與 2.0")

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
                    cfg_value=2.0,
                    normalize=True,
                    denoise=False,
                    inference_timesteps=30,
                    speed=speed,
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
                    cfg_value=2.0,
                    inference_timesteps=30,
                    seed=None,
                    speed=speed,
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
                            item.text, item.voice_id, item.speed, item.model_id
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
                if not 0.5 <= item.speed <= 2.0:
                    raise _CastVoiceError(
                        400, "invalid_speed", "speed 必須介於 0.5 與 2.0"
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
                        "cfg_value": 2.0,
                        "normalize": True,
                        "denoise": False,
                        "inference_timesteps": 30,
                        "speed": item.speed,
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
        storage_dir=_CASTVOICE_BATCH_DIR,
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
                body.text, body.voice_id, body.speed, body.model_id
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

    if mount_legacy:
        import gradio as gr

        legacy = create_demo_interface(demo)
        legacy.queue(max_size=10, default_concurrency_limit=1)
        app = gr.mount_gradio_app(
            app,
            legacy,
            path="/legacy",
            show_error=True,
            allowed_paths=[str(Path.cwd() / "assets")],
        )
        app.state.tts_gateway = gateway
        app.state.castvoice_batch_manager = batch_manager

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-legacy", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(
        create_app(mount_legacy=not args.no_legacy),
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
