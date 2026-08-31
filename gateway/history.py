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

from gateway.presets import _COSY_PROMPT_TEXT, _DEFAULT_CONTROL_INSTRUCTION, _DEFAULT_REFERENCE_PRESET_ID, _HISTORY_DIR, _LANG_NAN_TW, _LANG_ZH_TW, _MODEL_REGISTRY_PATH, _REFERENCE_AUDIO_DIR, _REFERENCE_AUDIO_PRESETS, _VOXCPM2_FIXED_TIMESTEPS, _by_id, _find_reference_preset

def _save_generation_history(record: dict[str, Any], wav: bytes) -> None:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_id = record["id"]
    wav_path = _HISTORY_DIR / f"{history_id}.wav"
    metadata_path = _HISTORY_DIR / f"{history_id}.json"
    wav_temp = _HISTORY_DIR / f".{history_id}.wav.tmp"
    metadata_temp = _HISTORY_DIR / f".{history_id}.json.tmp"
    try:
        wav_temp.write_bytes(wav)
        metadata_temp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(wav_temp, wav_path)
        os.replace(metadata_temp, metadata_path)
    except BaseException:
        # WAV 與 metadata 是同一筆紀錄；只成功 replace 其中一個時也要回滾。
        for final_path in (wav_path, metadata_path):
            try:
                final_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        # 部分寫入失敗時不可留下會永遠累積的隱藏暫存檔。
        for temp_path in (wav_temp, metadata_temp):
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _delete_generation_history(history_id: str) -> None:
    for suffix in (".wav", ".json"):
        try:
            (_HISTORY_DIR / f"{history_id}{suffix}").unlink()
        except FileNotFoundError:
            pass


def _wav_to_mp3(wav_bytes: bytes) -> bytes:
    """呼叫系統 ffmpeg 把 wav 轉成 mp3。CastAgent 的 pipeline 全程走 mp3
    （快取路徑、混音、拼接皆是），所以在 API 端轉一次比 CastAgent 每句都轉划算。
    """
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0 or not process.stdout:
        raise RuntimeError(
            f"ffmpeg wav->mp3 轉檔失敗（exit={process.returncode}）："
            f"{process.stderr.decode('utf-8', errors='replace')[-500:]}"
        )
    return process.stdout


def _load_generation_history(limit: int) -> list[dict[str, Any]]:
    if not _HISTORY_DIR.is_dir():
        return []
    # 紀錄寫入後不再修改，故 mtime 等同 created_at —— 先用 stat 排序再讀取，
    # 只解析需要的那幾筆。前端每 15 秒輪詢一次，全量讀取會隨紀錄數無上限成長。
    try:
        candidates = sorted(
            _HISTORY_DIR.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for metadata_path in candidates:
        if len(records) >= limit:
            break
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping invalid generation history file: %s", metadata_path)
            continue
        history_id = record.get("id")
        if not isinstance(history_id, str):
            continue
        if not (_HISTORY_DIR / f"{history_id}.wav").is_file():
            continue
        record["audio_url"] = f"/api/v1/history/{history_id}/audio"
        records.append(record)
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return records


