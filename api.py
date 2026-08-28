from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import secrets
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from app import VoxCPMDemo, create_demo_interface
from voxcpm.barbet_registry import BarbetModelRegistry
from voxcpm.barbet_runtime import BarbetRuntime
from voxcpm.full_model_registry import FULL_MODEL_PREFIX, FullModelRegistry
from voxcpm.lora_registry import BASE_MODEL_KEY

logger = logging.getLogger(__name__)
BASE_MODEL_PREFIX = "base::"
LORA_MODEL_PREFIX = "lora::"
PUBLIC_BASE_MODEL_ID = f"{BASE_MODEL_PREFIX}{BASE_MODEL_KEY}"
_REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent / "assets" / "default_reference"
_MODEL_REGISTRY_PATH = Path(__file__).resolve().parent / "docs" / "model_registry.json"
_HISTORY_DIR = Path(
    os.environ.get(
        "VOXCPM_HISTORY_DIR",
        Path(__file__).resolve().parent / "data" / "generation_history",
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

# CastAgent 自建 TTS API 規格採用固定的「演員聲音」清單，而不是把
# engine/model/speaker 的排列組合整個攤平給外部服務。這裡手動選一小批
# 品質確認過的聲音出來，voice_id 是外部合約的一部分、不可隨便改名。
_CASTVOICE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "voice_id": "barbet-hung-yi-lee",
        "label": "李宏毅老師（華語・男）",
        "gender": "male",
        "language": _LANG_ZH_TW,
        "desc": "Barbet TSLM 固定語者音色",
        "engine_id": "barbet",
        "speaker_id": "hung_yi_lee",
    },
    {
        "voice_id": "voxcpm2-cosy-child-female-01",
        "label": "孩童女聲 01（臺灣台語）",
        "gender": "female",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 340 Hz，音色高亢，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-child-female-01",
    },
    {
        "voice_id": "voxcpm2-cosy-teen-female-01",
        "label": "少女聲 01（臺灣台語）",
        "gender": "female",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 320 Hz，音色清亮，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-teen-female-01",
    },
    {
        "voice_id": "voxcpm2-cosy-teen-female-02",
        "label": "少女聲 02（臺灣台語）",
        "gender": "female",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 320 Hz，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-teen-female-02",
    },
    {
        "voice_id": "voxcpm2-cosy-teen-female-03",
        "label": "少女聲 03（臺灣台語）",
        "gender": "female",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 314 Hz，音色清亮，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-teen-female-03",
    },
    {
        "voice_id": "voxcpm2-cosy-young-female-01",
        "label": "青年女聲 01（臺灣台語）",
        "gender": "female",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 256 Hz，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-young-female-01",
    },
    {
        "voice_id": "voxcpm2-cosy-senior-female-01",
        "label": "年長女聲 01（臺灣台語）",
        "gender": "female",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 216 Hz，音色偏低沉，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-senior-female-01",
    },
    {
        "voice_id": "voxcpm2-cosy-child-male-01",
        "label": "孩童男聲 01（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 276 Hz，音色高亢，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-child-male-01",
    },
    {
        "voice_id": "voxcpm2-cosy-child-male-02",
        "label": "孩童男聲 02（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 278 Hz，音色高亢，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-child-male-02",
    },
    {
        "voice_id": "voxcpm2-cosy-teen-male-01",
        "label": "少年聲 01（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 246 Hz，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-teen-male-01",
    },
    {
        "voice_id": "voxcpm2-cosy-teen-male-02",
        "label": "少年聲 02（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 262 Hz，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-teen-male-02",
    },
    {
        "voice_id": "voxcpm2-cosy-young-male-01",
        "label": "青年男聲 01（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 160 Hz，音域約 107–282 Hz，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-young-male-01",
    },
    {
        "voice_id": "voxcpm2-cosy-young-male-02",
        "label": "青年男聲 02（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 162 Hz，音域較集中、語調平穩，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-young-male-02",
    },
    {
        "voice_id": "voxcpm2-cosy-young-male-03",
        "label": "青年男聲 03（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 113 Hz，音色低沉，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-young-male-03",
    },
    {
        "voice_id": "voxcpm2-cosy-senior-male-01",
        "label": "年長男聲 01（臺灣台語）",
        "gender": "male",
        "language": _LANG_NAN_TW,
        "desc": "基頻約 118 Hz，音域約 91–158 Hz，VoxCPM2 zero-shot 聲音克隆",
        "engine_id": "voxcpm2",
        "reference_preset_id": "cosy-senior-male-01",
    },
)
_CASTVOICE_DEFINITIONS_BY_ID = {
    definition["voice_id"]: definition for definition in _CASTVOICE_DEFINITIONS
}
_CASTVOICE_MODEL_VERSION = os.environ.get(
    "VOXCPM_CASTVOICE_MODEL_VERSION", "voxcpm360-castvoice-1.0"
)
_TTS_API_KEY = os.environ.get("TTS_API_KEY", "").strip()
_CASTVOICE_BATCH_DIR = Path(
    os.environ.get(
        "VOXCPM_CASTVOICE_BATCH_DIR",
        Path(__file__).resolve().parent / "data" / "castvoice_batches",
    )
)
_CASTVOICE_BATCH_MAX_ITEMS = int(os.environ.get("VOXCPM_CASTVOICE_BATCH_MAX_ITEMS", "500"))

# 訓練資料的文字全是華語漢字（例：「我跟他的感情真的很好」而非台語漢字
# 「我佮伊的感情真正好」），模型學到的是「華語漢字 -> 台語發音」。但底模
# VoxCPM2 在大量華語上預訓練，同一批漢字有很強的「唸成華語」先驗，微調
# 兩個 epoch 蓋不過去 —— 實聽就是「全都講中文」。
# 這裡於使用者未填 control_instruction 時自動帶入語言指令，把台語這個
# 條件明確給模型。空字串可停用。
# ⚠️ 僅 voxcpm2 有效：barbet 的 capabilities.control_instruction 為 False，
# api.py 的 barbet 呼叫端根本不傳這個參數。
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


def _save_generation_history(record: dict[str, Any], wav: bytes) -> None:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_id = record["id"]
    wav_path = _HISTORY_DIR / f"{history_id}.wav"
    metadata_path = _HISTORY_DIR / f"{history_id}.json"
    wav_temp = _HISTORY_DIR / f".{history_id}.wav.tmp"
    metadata_temp = _HISTORY_DIR / f".{history_id}.json.tmp"
    wav_temp.write_bytes(wav)
    metadata_temp.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(wav_temp, wav_path)
    os.replace(metadata_temp, metadata_path)


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
        self._gpu_lock = asyncio.Lock()
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

    async def _run_gpu_job(
        self,
        *,
        request_id: str,
        engine_id: str,
        model_id: str,
        work: Callable[[], Any],
    ) -> tuple[Any, float, float]:
        """Run one non-cancellable CUDA job behind the shared bounded gate.

        A thread already executing CUDA cannot be safely stopped by cancelling
        its asyncio waiter.  If the HTTP task is cancelled, keep the gate held
        until the worker thread exits so another model is never started on top
        of the abandoned job.
        """
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
            "synthesis.request request_id=%s stage=queued engine=%s model=%s "
            "queue_position=%d inflight=%d",
            request_id,
            engine_id,
            model_id,
            queue_position,
            self._inflight_jobs,
        )

        acquired = False
        started_at: float | None = None
        try:
            try:
                await asyncio.wait_for(
                    self._gpu_lock.acquire(),
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
            return result, queue_wait, execution_time
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
                self._gpu_lock.release()
            async with self._admission_lock:
                self._inflight_jobs -= 1

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
    def _job_timing_headers(queue_wait: float, execution_time: float) -> dict[str, str]:
        return {
            "X-Queue-Wait": f"{queue_wait:.3f}s",
            "X-GPU-Job-Time": f"{execution_time:.3f}s",
        }

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
            selected_demo, runtime_selection, canonical_id = (
                self._switch_native_runtime(model_id)
            )
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
                results = batch_generate(generation_requests)
            else:
                results = [
                    selected_demo.generate_tts_audio(**generation_request)
                    for generation_request in generation_requests
                ]
            self._active_native_selection = canonical_id
            return results

        results, queue_wait, execution_time = await self._run_gpu_job(
            request_id=request_id,
            engine_id="voxcpm2",
            model_id=model_id,
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
        return wavs, self._job_timing_headers(queue_wait, execution_time)

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

        result, queue_wait, execution_time = await self._run_gpu_job(
            request_id=request_id,
            engine_id="barbet",
            model_id=model_id,
            work=generate,
        )
        sample_rate, audio, elapsed = result
        headers = {
            "X-Synthesis-Time": f"{elapsed:.2f}s",
            "X-Random-Seed": str(actual_seed),
            **self._job_timing_headers(queue_wait, execution_time),
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
            await asyncio.to_thread(gateway.close)

    app = FastAPI(
        title="VoxCPM 360 Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.tts_gateway = gateway
    history_lock = asyncio.Lock()

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

    @app.post("/api/v1/synthesize")
    async def synthesize(
        request: Request,
        engine_id: str = Form(...),
        model_id: str = Form(...),
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
        text = text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="請輸入要合成的文字")
        if not 1.0 <= cfg_value <= 5.0:
            raise HTTPException(status_code=422, detail="CFG 必須介於 1.0 與 5.0")
        if not 1 <= inference_timesteps <= 50:
            raise HTTPException(status_code=422, detail="取樣步數必須介於 1 與 50")
        if not 0.5 <= speed <= 2.0:
            raise HTTPException(status_code=422, detail="語速必須介於 0.5 與 2.0")

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
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(payload)
                    temp_path = temp_file.name
                active_reference = temp_path
            else:
                # 未上傳則採用預設台語參考音檔。不可寫進 temp_path ——
                # 那個變數在 finally 會被 os.unlink，會刪掉預設檔本身。
                active_reference = _resolve_reference_audio(reference_preset_id)
                selected_preset = _find_reference_preset(reference_preset_id)
                if selected_preset is not None:
                    # 內建參考音的逐字稿我們自己知道，使用者沒填就自動帶入。
                    # 不帶的話 barbet_runtime.py 會因 prompt_text 為空而把
                    # prompt_wav_path 設成 ""，參考音整個被丟掉、克隆完全失效
                    # （2026-08-05 實聽發現，見 AGENTS.md 7.6.1）。
                    # 使用者上傳的音檔不在此列 —— 我們不知道它的逐字稿。
                    if not prompt_text.strip():
                        prompt_text = selected_preset.get("prompt_text", "")
                    reference_label = (
                        f"{selected_preset['label']} · {selected_preset['description']}"
                    )

            if engine_id == "voxcpm2":
                try:
                    model_id = await asyncio.to_thread(
                        gateway.resolve_native_model_id, model_id
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                # prompt cloning 與文字控制是兩條互斥路徑。VoxCPM2 並沒有獨立的
                # instruction channel；app.py 會把 control 包成「(指令)正文」送進
                # TSLM。若同時帶 prompt 音訊／逐字稿，部分微調模型會把指令當正文
                # 朗讀出來。舊版 Gradio 也明確在 cloning 模式清空 control。
                if active_reference and prompt_text.strip():
                    control_instruction = ""
                # 非 cloning 模式下，使用者沒指定才帶入預設語言指令。
                elif not control_instruction.strip():
                    control_instruction = _DEFAULT_CONTROL_INSTRUCTION
                wav, extra_headers = await gateway.synthesize_native(
                    request_id=request_id,
                    model_id=model_id,
                    text=text,
                    control_instruction=control_instruction,
                    reference_path=active_reference,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    normalize=normalize,
                    denoise=denoise,
                    inference_timesteps=inference_timesteps,
                    speed=speed,
                )
            elif engine_id == "barbet":
                wav, extra_headers = await gateway.synthesize_barbet(
                    request_id=request_id,
                    model_id=model_id,
                    text=text,
                    reference_path=active_reference,
                    prompt_text=prompt_text,
                    speaker_id=speaker_id,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    seed=seed,
                    speed=speed,
                )
            else:
                raise HTTPException(status_code=404, detail=f"找不到推論引擎：{engine_id}")

            headers = {
                "Content-Disposition": 'inline; filename="tts-output.wav"',
                "X-Model-Engine": engine_id,
                "X-Model-Version": model_id,
                "X-Request-ID": request_id,
                **extra_headers,
            }
            history_id = uuid.uuid4().hex
            # catalog() 會 refresh 三個 registry 並掃過多 GB 的 checkpoint 目錄，
            # 不能在 event loop 上跑 —— 與 /api/v1/catalog 端點的做法一致。
            catalog = await asyncio.to_thread(gateway.catalog)
            selected_engine = _by_id(catalog["engines"], engine_id)
            selected_model = _by_id(selected_engine.get("models", []), model_id)
            selected_speaker = _by_id(selected_model.get("speakers", []), speaker_id)
            record = {
                "id": history_id,
                "text": text,
                "engine_id": engine_id,
                "engine_label": selected_engine.get("label", engine_id),
                "model_id": model_id,
                "model_label": selected_model.get("label", model_id),
                "reference_label": reference_label,
                "speaker_label": selected_speaker.get("name"),
                "seed": int(extra_headers["X-Random-Seed"])
                if "X-Random-Seed" in extra_headers
                else None,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
                "speed": speed,
                "normalize": normalize,
                "denoise": denoise,
                "prompt_text": prompt_text.strip() or None,
                "control_instruction": control_instruction.strip() or None,
                "duration_label": extra_headers.get("X-Synthesis-Time"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            async with history_lock:
                await asyncio.to_thread(_save_generation_history, record, wav)
            headers["X-History-ID"] = history_id
            headers["X-Total-Time"] = (
                f"{time.perf_counter() - request_started_at:.3f}s"
            )
            return Response(content=wav, media_type="audio/wav", headers=headers)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

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
