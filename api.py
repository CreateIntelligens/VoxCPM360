from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app import VoxCPMDemo, create_demo_interface
from voxcpm.full_model_registry import FULL_MODEL_PREFIX, FullModelRegistry
from voxcpm.lora_registry import BASE_MODEL_KEY

logger = logging.getLogger(__name__)
REMOTE_MODEL_SEPARATOR = "::"


@dataclass(frozen=True)
class RemoteModel:
    id: str
    label: str
    base_url: str
    description: str = ""


def load_remote_models() -> tuple[RemoteModel, ...]:
    raw = os.environ.get("BLUEMAGPIE_MODELS_JSON", "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"BLUEMAGPIE_MODELS_JSON is invalid JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError("BLUEMAGPIE_MODELS_JSON must be a JSON array")

        models: list[RemoteModel] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("Each BlueMagpie model entry must be an object")
            models.append(
                RemoteModel(
                    id=str(item["id"]),
                    label=str(item.get("label") or item["id"]),
                    base_url=str(item["base_url"]).rstrip("/"),
                    description=str(item.get("description") or ""),
                )
            )
        return tuple(models)

    base_url = os.environ.get("BLUEMAGPIE_URL", "").strip()
    if not base_url:
        return ()
    return (
        RemoteModel(
            id="bluemagpie-base",
            label="BlueMagpie（Barbet）",
            base_url=base_url.rstrip("/"),
            description="Barbet TSLM + VoxCPM2 acoustic stack",
        ),
    )


class TTSGateway:
    def __init__(
        self,
        demo: VoxCPMDemo,
        remote_models: tuple[RemoteModel, ...] | None = None,
        request_timeout: float = 900.0,
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
            Path(value)
            for value in full_roots_setting.split(os.pathsep)
            if value.strip()
        )
        self.remote_models = remote_models if remote_models is not None else load_remote_models()
        self.request_timeout = request_timeout
        self._native_lock = asyncio.Lock()
        self._remote_locks = {model.id: asyncio.Lock() for model in self.remote_models}

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

    def _remote_health(self, model: RemoteModel) -> tuple[bool, dict[str, Any]]:
        try:
            response = requests.get(
                f"{model.base_url}/health",
                timeout=min(self.request_timeout, 5.0),
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("status") == "ok", payload
        except (requests.RequestException, ValueError):
            return False, {}

    def _remote_speakers(self, model: RemoteModel) -> list[dict[str, Any]]:
        try:
            response = requests.get(
                f"{model.base_url}/api/speakers",
                timeout=min(self.request_timeout, 5.0),
            )
            response.raise_for_status()
            payload = response.json()
            speakers = payload.get("speakers", [])
            return speakers if isinstance(speakers, list) else []
        except (requests.RequestException, ValueError):
            return []

    def _remote_versions(
        self,
        backend: RemoteModel,
        *,
        online: bool,
        health: dict[str, Any],
        speakers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        if online:
            try:
                response = requests.get(
                    f"{backend.base_url}/api/models",
                    timeout=min(self.request_timeout, 5.0),
                )
                if response.status_code != 404:
                    response.raise_for_status()
                    payload = response.json()
                    discovered = payload.get("models", [])
                    if isinstance(discovered, list):
                        versions = [
                            item for item in discovered if isinstance(item, dict)
                        ]
            except (requests.RequestException, ValueError):
                versions = []

        if not versions:
            return [
                {
                    "id": backend.id,
                    "label": backend.label,
                    "kind": "checkpoint",
                    "description": (
                        backend.description if online else "BlueMagpie 後端目前離線"
                    ),
                    "online": online,
                    "gpu": health.get("gpu"),
                    "speakers": speakers if online else [],
                }
            ]

        entries = []
        for version in versions:
            if not version.get("id"):
                continue
            remote_id = str(version["id"])
            entries.append(
                {
                    "id": f"{backend.id}{REMOTE_MODEL_SEPARATOR}{remote_id}",
                    "label": str(version.get("label") or remote_id),
                    "kind": "checkpoint",
                    "description": str(
                        version.get("description") or backend.description
                    ),
                    "online": bool(version.get("online", True)),
                    "loaded": bool(version.get("loaded", False)),
                    "gpu": health.get("gpu"),
                    "speakers": speakers,
                }
            )
        return entries

    def catalog(self) -> dict[str, Any]:
        remote_entries = []
        for backend in self.remote_models:
            online, health = self._remote_health(backend)
            speakers = self._remote_speakers(backend) if online else []
            remote_entries.extend(
                self._remote_versions(
                    backend,
                    online=online,
                    health=health,
                    speakers=speakers,
                )
            )

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
        if remote_entries:
            engines.append(
                {
                    "id": "bluemagpie",
                    "label": "BlueMagpie",
                    "family": "barbet",
                    "description": "Barbet TSLM + VoxCPM2 聲學模型",
                    "online": any(entry["online"] for entry in remote_entries),
                    "capabilities": {
                        "control_instruction": False,
                        "prompt_transcript": False,
                        "reference_audio": True,
                        "speaker_selection": True,
                        "seed": True,
                    },
                    "models": remote_entries,
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

    def _find_remote(self, model_id: str) -> tuple[RemoteModel, str | None]:
        for model in self.remote_models:
            if model.id == model_id:
                return model, None
            prefix = f"{model.id}{REMOTE_MODEL_SEPARATOR}"
            if model_id.startswith(prefix):
                return model, model_id.removeprefix(prefix)
        raise HTTPException(status_code=404, detail=f"找不到 BlueMagpie 模型版本：{model_id}")

    def _request_remote(
        self,
        model: RemoteModel,
        *,
        remote_model_id: str | None,
        text: str,
        reference_path: str | None,
        reference_name: str,
        speaker_id: str,
        cfg_value: float,
        inference_timesteps: int,
        seed: int | None,
    ) -> requests.Response:
        try:
            if reference_path:
                data: dict[str, str] = {
                    "target_text": text,
                    "cfg_value": str(cfg_value),
                    "inference_timesteps": str(inference_timesteps),
                }
                if remote_model_id:
                    data["model_id"] = remote_model_id
                if seed is not None:
                    data["seed"] = str(seed)
                with open(reference_path, "rb") as audio_file:
                    response = requests.post(
                        f"{model.base_url}/api/synthesize/clone",
                        data=data,
                        files={"reference_audio": (reference_name, audio_file, "audio/wav")},
                        timeout=self.request_timeout,
                    )
            else:
                payload: dict[str, Any] = {
                    "model_id": remote_model_id,
                    "target_text": text,
                    "cfg_value": cfg_value,
                    "inference_timesteps": inference_timesteps,
                    "retry_badcase": True,
                    "seed": seed,
                }
                if speaker_id:
                    payload["speaker_id"] = speaker_id
                response = requests.post(
                    f"{model.base_url}/api/synthesize",
                    json=payload,
                    timeout=self.request_timeout,
                )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            detail = str(exc)
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text or detail
            raise HTTPException(status_code=502, detail=f"BlueMagpie 推論失敗：{detail}") from exc

    async def synthesize_remote(
        self,
        *,
        model_id: str,
        text: str,
        reference_path: str | None,
        reference_name: str,
        speaker_id: str,
        cfg_value: float,
        inference_timesteps: int,
        seed: int | None,
    ) -> tuple[bytes, dict[str, str]]:
        model, remote_model_id = self._find_remote(model_id)
        async with self._remote_locks[model.id]:
            response = await asyncio.to_thread(
                self._request_remote,
                model,
                remote_model_id=remote_model_id,
                text=text,
                reference_path=reference_path,
                reference_name=reference_name,
                speaker_id=speaker_id,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                seed=seed,
            )
        headers = {}
        synthesis_time = response.headers.get("X-Synthesis-Time")
        if synthesis_time:
            headers["X-Synthesis-Time"] = synthesis_time
        return response.content, headers


def create_app(
    demo: VoxCPMDemo | None = None,
    *,
    remote_models: tuple[RemoteModel, ...] | None = None,
    mount_legacy: bool = True,
) -> FastAPI:
    demo = demo or VoxCPMDemo(
        model_id=os.environ.get("MODEL_ID", "openbmb/VoxCPM2"),
        device=os.environ.get("VOXCPM_DEVICE", "auto"),
    )
    gateway = TTSGateway(demo, remote_models=remote_models)

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

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        catalog = await asyncio.to_thread(gateway.catalog)
        return {"status": "ok", **catalog}

    @app.get("/api/v1/catalog")
    async def catalog() -> dict[str, Any]:
        return await asyncio.to_thread(gateway.catalog)

    @app.post("/api/v1/synthesize")
    async def synthesize(
        engine_id: str = Form(...),
        model_id: str = Form(...),
        text: str = Form(...),
        control_instruction: str = Form(""),
        prompt_text: str = Form(""),
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
        reference_name = "reference.wav"
        try:
            if reference_audio is not None:
                reference_name = reference_audio.filename or reference_name
                suffix = Path(reference_name).suffix or ".wav"
                payload = await reference_audio.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(payload)
                    temp_path = temp_file.name

            if engine_id == "voxcpm2":
                wav, extra_headers = await gateway.synthesize_native(
                    model_id=model_id,
                    text=text,
                    control_instruction=control_instruction,
                    reference_path=temp_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    normalize=normalize,
                    denoise=denoise,
                    inference_timesteps=inference_timesteps,
                )
            elif engine_id == "bluemagpie":
                wav, extra_headers = await gateway.synthesize_remote(
                    model_id=model_id,
                    text=text,
                    reference_path=temp_path,
                    reference_name=reference_name,
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
