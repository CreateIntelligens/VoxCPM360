from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app import VoxCPMDemo, create_demo_interface
from voxcpm.barbet_registry import BarbetModelRegistry
from voxcpm.barbet_runtime import BarbetRuntime
from voxcpm.full_model_registry import FULL_MODEL_PREFIX, FullModelRegistry
from voxcpm.lora_registry import BASE_MODEL_KEY

logger = logging.getLogger(__name__)
_REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent / "assets" / "default_reference"
_REFERENCE_AUDIO_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "middle-aged-female",
        "label": "中年女聲",
        "filename": "tai8_drama1_005.wav",
        "description": "tai8 台語劇集片段，約 5.2 秒",
    },
    {
        "id": "middle-aged-male",
        "label": "中年男聲",
        "filename": "tai8_female_drama1_002.wav",
        "description": "tai8 台語劇集片段，約 2.7 秒",
    },
    {
        "id": "hayley-happy-opening",
        "label": "Hayley 開心說開場白",
        "filename": "hayley_happy_opening.mp3",
        "description": "Hayley 開心語氣開場白，約 12.9 秒",
    },
)
_DEFAULT_REFERENCE_PRESET_ID = "middle-aged-female"


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
                "description": "原生 VoxCPM2 與本機 LoRA",
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
                    "description": "本機 Barbet TSLM + VoxCPM2 聲學模型",
                    "online": any(model["online"] for model in barbet_models),
                    "capabilities": {
                        "control_instruction": False,
                        "prompt_transcript": False,
                        "reference_audio": True,
                        "speaker_selection": True,
                        "seed": True,
                    },
                    "models": barbet_models,
                }
            )
        return {"engines": engines}

    @staticmethod
    def _wav_response(sample_rate: int, audio: np.ndarray) -> bytes:
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
        return self._wav_response(sample_rate, audio), {}

    async def synthesize_barbet(
        self,
        *,
        model_id: str,
        text: str,
        reference_path: str | None,
        speaker_id: str,
        cfg_value: float,
        inference_timesteps: int,
        seed: int | None,
    ) -> tuple[bytes, dict[str, str]]:
        try:
            self.barbet_runtime.registry.get(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async with self._barbet_lock:
            sample_rate, audio, elapsed = await asyncio.to_thread(
                self.barbet_runtime.synthesize,
                model_id=model_id,
                text=text,
                reference_path=reference_path,
                speaker_id=speaker_id,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                seed=seed,
            )
        return self._wav_response(sample_rate, audio), {
            "X-Synthesis-Time": f"{elapsed:.2f}s",
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

    def _available_reference_presets() -> list[dict[str, str]]:
        return [
            {
                "id": preset["id"],
                "label": preset["label"],
                "description": preset["description"],
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

    def _resolve_reference_audio(
        reference_preset_id: str,
    ) -> str | None:
        """解析白名單內建音檔，未指定時沿用環境覆寫或預設音檔。"""
        requested_id = reference_preset_id.strip()
        if requested_id:
            preset = next(
                (item for item in _REFERENCE_AUDIO_PRESETS if item["id"] == requested_id),
                None,
            )
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

        default_preset = next(
            (item for item in _REFERENCE_AUDIO_PRESETS if item["id"] == _DEFAULT_REFERENCE_PRESET_ID),
            None,
        )
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
        inference_timesteps: int = Form(10),
        normalize: bool = Form(False),
        denoise: bool = Form(False),
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

        temp_path: str | None = None
        active_reference: str | None = None
        try:
            if reference_audio is not None:
                uploaded_name = reference_audio.filename or "reference.wav"
                suffix = Path(uploaded_name).suffix or ".wav"
                payload = await reference_audio.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(payload)
                    temp_path = temp_file.name
                active_reference = temp_path
            else:
                # 未上傳則採用預設台語參考音檔。不可寫進 temp_path ——
                # 那個變數在 finally 會被 os.unlink，會刪掉預設檔本身。
                active_reference = _resolve_reference_audio(reference_preset_id)

            if engine_id == "voxcpm2":
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
                )
            elif engine_id == "barbet":
                wav, extra_headers = await gateway.synthesize_barbet(
                    model_id=model_id,
                    text=text,
                    reference_path=active_reference,
                    speaker_id=speaker_id,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    seed=seed,
                )
            else:
                raise HTTPException(status_code=404, detail=f"找不到推論引擎：{engine_id}")

            headers = {
                "Content-Disposition": 'inline; filename="tts-output.wav"',
                "X-Model-Engine": engine_id,
                "X-Model-Version": model_id,
                **extra_headers,
            }
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
