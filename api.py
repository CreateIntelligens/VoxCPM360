from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import secrets
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app import VoxCPMDemo, create_demo_interface
from voxcpm.barbet_registry import BarbetModelRegistry
from voxcpm.barbet_runtime import BarbetRuntime
from voxcpm.full_model_registry import FULL_MODEL_PREFIX, FullModelRegistry
from voxcpm.lora_registry import BASE_MODEL_KEY

logger = logging.getLogger(__name__)
_REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent / "assets" / "default_reference"
_MODEL_REGISTRY_PATH = Path(__file__).resolve().parent / "docs" / "model_registry.json"
_HISTORY_DIR = Path(
    os.environ.get(
        "VOXCPM_HISTORY_DIR",
        Path(__file__).resolve().parent / "data" / "generation_history",
    )
)
_REFERENCE_AUDIO_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "middle-aged-female",
        "label": "中年女聲",
        "filename": "tai8_drama1_005.wav",
        "description": "tai8 台語劇集片段，約 5.2 秒",
        "prompt_text": "碧玉拿給我看他先生的相片好像不是長這樣",
    },
    {
        "id": "middle-aged-male",
        "label": "中年男聲",
        "filename": "tai8_female_drama1_002.wav",
        "description": "tai8 台語劇集片段，約 2.7 秒",
        "prompt_text": "只要一套比基尼這樣就夠了",
    },
    {
        "id": "hayley-happy-opening",
        "label": "Hayley 開心說開場白",
        "filename": "hayley_happy_opening.mp3",
        "description": "Hayley 開心語氣開場白，約 12.9 秒",
        "prompt_text": "Hi，我是創造智能的 AI 代言人愛卡，想知道你的 MBTI 是哪一型嗎？還是對我們的 AI 服務好奇，我都可以告訴你，快來跟我聊聊吧！",
    },
)
_DEFAULT_REFERENCE_PRESET_ID = "middle-aged-female"

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
        self._active_native_selection = BASE_MODEL_KEY
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
        self._native_lock = asyncio.Lock()
        self._barbet_lock = asyncio.Lock()

    def _native_models(self) -> list[dict[str, Any]]:
        self.demo.lora_registry.refresh()
        self.full_model_registry.refresh()
        models: list[dict[str, Any]] = [
            {
                "id": BASE_MODEL_KEY,
                "label": "VoxCPM2 基礎模型",
                "kind": "base",
                "description": "原生 MiniCPM4 Text-Semantic LM",
                "loaded": self._active_native_selection == BASE_MODEL_KEY,
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
                    "id": checkpoint.run_name,
                    "label": checkpoint.label,
                    "kind": "lora",
                    "description": self.demo.lora_registry.describe(checkpoint.run_name),
                    "loaded": self._active_native_selection == checkpoint.run_name,
                }
            )
        return models

    def _switch_native_runtime(self, model_id: str) -> tuple[VoxCPMDemo, str]:
        is_full_model = model_id.startswith(FULL_MODEL_PREFIX)
        desired_runtime_id = model_id if is_full_model else BASE_MODEL_KEY
        runtime_selection = BASE_MODEL_KEY if is_full_model else model_id
        if desired_runtime_id == self._native_runtime_id:
            return self._native_demo, runtime_selection

        checkpoint = self.full_model_registry.get(model_id) if is_full_model else None
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
            logger.exception("Failed to switch native runtime to %s", model_id)
            self._native_demo = previous_demo
            self._native_runtime_id = previous_runtime_id
            raise

        self._native_demo = next_demo
        self._native_runtime_id = desired_runtime_id
        logger.info("Native runtime switched to %s", desired_runtime_id)
        return next_demo, runtime_selection

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

    async def synthesize_native(
        self,
        *,
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
        async with self._native_lock:
            selected_demo, runtime_selection = await asyncio.to_thread(
                self._switch_native_runtime,
                model_id,
            )
            sample_rate, audio = await asyncio.to_thread(
                selected_demo.generate_tts_audio,
                text_input=text,
                control_instruction=control_instruction,
                reference_wav_path_input=reference_path,
                prompt_text=prompt_text,
                cfg_value_input=cfg_value,
                do_normalize=normalize,
                denoise=denoise,
                inference_timesteps=inference_timesteps,
                model_selection=runtime_selection,
            )
            self._active_native_selection = model_id
        return self._wav_response(sample_rate, audio, speed), {}

    async def synthesize_barbet(
        self,
        *,
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
        async with self._barbet_lock:
            sample_rate, audio, elapsed = await asyncio.to_thread(
                self.barbet_runtime.synthesize,
                model_id=model_id,
                text=text,
                reference_path=reference_path,
                prompt_text=prompt_text,
                speaker_id=speaker_id,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                seed=actual_seed,
            )
        return self._wav_response(sample_rate, audio, speed), {
            "X-Synthesis-Time": f"{elapsed:.2f}s",
            "X-Random-Seed": str(actual_seed),
        }


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
        try:
            yield
        finally:
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
                # 使用者沒指定就帶入預設語言指令（見 _DEFAULT_CONTROL_INSTRUCTION）
                if not control_instruction.strip():
                    control_instruction = _DEFAULT_CONTROL_INSTRUCTION
                wav, extra_headers = await gateway.synthesize_native(
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
            return Response(content=wav, media_type="audio/wav", headers=headers)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

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
