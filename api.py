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


def _default_warmup_reference() -> tuple[str | None, str]:
    preset = _find_reference_preset(_DEFAULT_REFERENCE_PRESET_ID)
    if preset is None:
        return None, ""

    preset_path = _REFERENCE_AUDIO_DIR / preset["filename"]
    if not preset_path.is_file():
        return None, ""

    return str(preset_path), preset.get("prompt_text", "")



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
            reference_path, prompt_text = _default_warmup_reference()
            try:
                await asyncio.to_thread(
                    demo.warmup_voxcpm,
                    reference_path=reference_path,
                    prompt_text=prompt_text,
                )
            except Exception:
                logger.exception(
                    "VoxCPM2 inference warmup failed; continuing startup"
                )
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

    from gateway.routes.interactive import register_interactive_routes
    from gateway.routes.castvoice import register_castvoice_routes

    helpers = register_interactive_routes(app, gateway, history_lock)
    batch_manager = register_castvoice_routes(app, gateway, demo, helpers)

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
