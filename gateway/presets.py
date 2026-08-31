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

_REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_reference"
_MODEL_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "docs" / "model_registry.json"
_HISTORY_DIR = Path(
    os.environ.get(
        "VOXCPM_HISTORY_DIR",
        Path(__file__).resolve().parent.parent / "data" / "generation_history",
    )
)
# 語言標籤（BCP 47）：nan-TW 臺灣台語、zh-TW 臺灣華語。
_LANG_NAN_TW = "nan-TW"
_LANG_ZH_TW = "zh-TW"
# cosy 系列音檔是同一句話由不同聲音錄製，逐字稿因此共用同一字串。
# prompt_text 是參考音檔的逐字稿，必須與音檔實際發音一字不差，
# 不是可替換的歡迎語（詳見 assets/default_reference/README.md）。
_COSY_PROMPT_TEXT = (
    "你好，歡迎使用創造智能台語生成服務，很高興為你服務，請輸入想要生成的文本內容"
)
_REFERENCE_AUDIO_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "cosy-child-female-01",
        "label": "孩童女聲 01（臺灣台語）",
        "filename": "cosy-child-female-01.mp3",
        "description": "基頻約 340 Hz，音色高亢",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "female",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-teen-female-01",
        "label": "少女聲 01（臺灣台語）",
        "filename": "cosy-teen-female-01.mp3",
        "description": "基頻約 320 Hz，音色清亮",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "female",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-teen-female-02",
        "label": "少女聲 02（臺灣台語）",
        "filename": "cosy-teen-female-02.mp3",
        "description": "基頻約 320 Hz",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "female",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-teen-female-03",
        "label": "少女聲 03（臺灣台語）",
        "filename": "cosy-teen-female-03.mp3",
        "description": "基頻約 314 Hz，音色清亮",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "female",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-young-female-01",
        "label": "青年女聲 01（臺灣台語）",
        "filename": "cosy-young-female-01.mp3",
        "description": "基頻約 256 Hz",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "female",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-senior-female-01",
        "label": "年長女聲 01（臺灣台語）",
        "filename": "cosy-senior-female-01.mp3",
        "description": "基頻約 216 Hz，音色偏低沉",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "female",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-child-male-01",
        "label": "孩童男聲 01（臺灣台語）",
        "filename": "cosy-child-male-01.mp3",
        "description": "基頻約 276 Hz，音色高亢",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-child-male-02",
        "label": "孩童男聲 02（臺灣台語）",
        "filename": "cosy-child-male-02.mp3",
        "description": "基頻約 278 Hz，音色高亢",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-teen-male-01",
        "label": "少年聲 01（臺灣台語）",
        "filename": "cosy-teen-male-01.mp3",
        "description": "基頻約 246 Hz",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-teen-male-02",
        "label": "少年聲 02（臺灣台語）",
        "filename": "cosy-teen-male-02.mp3",
        "description": "基頻約 262 Hz",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-young-male-01",
        "label": "青年男聲 01（臺灣台語）",
        "filename": "cosy-young-male-01.mp3",
        "description": "基頻約 160 Hz，音域約 107–282 Hz",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-young-male-02",
        "label": "青年男聲 02（臺灣台語）",
        "filename": "cosy-young-male-02.mp3",
        "description": "基頻約 162 Hz，音域較集中、語調平穩",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-young-male-03",
        "label": "青年男聲 03（臺灣台語）",
        "filename": "cosy-young-male-03.mp3",
        "description": "基頻約 113 Hz，音色低沉",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
    {
        "id": "cosy-senior-male-01",
        "label": "年長男聲 01（臺灣台語）",
        "filename": "cosy-senior-male-01.mp3",
        "description": "基頻約 118 Hz，音域約 91–158 Hz",
        "prompt_text": _COSY_PROMPT_TEXT,
        "gender": "male",
        "language": _LANG_NAN_TW,
    },
)
_DEFAULT_REFERENCE_PRESET_ID = "cosy-young-female-01"
# voxcpm2 引擎實際生效的 diffusion 步數（nano-vLLM 建構時固定，
# 與 app.py 讀同一個環境變數）；per-request 的 inference_timesteps
# 對 voxcpm2 不生效，catalog capabilities 與回應 header 據此誠實揭露。
_VOXCPM2_FIXED_TIMESTEPS = int(os.environ.get("VOXCPM_INFERENCE_TIMESTEPS", "10"))

_DEFAULT_CONTROL_INSTRUCTION = os.environ.get(
    "VOXCPM_DEFAULT_CONTROL_INSTRUCTION", "用台語說"
)


def _by_id(items: Any, item_id: str) -> dict[str, Any]:
    """在 catalog 的 engines/models/speakers 清單裡依 id 取項目，找不到回空 dict
    讓呼叫端可以直接 .get() 不必再判 None。"""
    return next((item for item in items if item.get("id") == item_id), {})


def _find_reference_preset(preset_id: str) -> dict[str, str] | None:
    """依 id 取內建參考音設定，空字串代表採用預設。"""
    return next(
        (
            item
            for item in _REFERENCE_AUDIO_PRESETS
            if item["id"] == (preset_id.strip() or _DEFAULT_REFERENCE_PRESET_ID)
        ),
        None,
    )


