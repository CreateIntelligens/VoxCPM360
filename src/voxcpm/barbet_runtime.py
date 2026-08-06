from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from voxcpm.barbet_registry import BarbetCheckpoint, BarbetModelRegistry

logger = logging.getLogger(__name__)

_SPEAKER_LABELS = {"hung_yi_lee": "李宏毅老師"}


class BarbetRuntime:
    """Own and dynamically switch local Barbet TTS checkpoints."""

    def __init__(self, registry: BarbetModelRegistry, device: str = "auto") -> None:
        self.registry = registry
        self.device = self._resolve_device(device)
        self.loaded_model_id: str | None = None
        self._loaded_checkpoint: BarbetCheckpoint | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._speaker_encoder: Any = None
        self._lock = threading.RLock()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def speakers(self, checkpoint: BarbetCheckpoint) -> list[dict[str, Any]]:
        speakers: dict[str, dict[str, Any]] = {}
        for directory in (checkpoint.path, checkpoint.path / "checkpoints"):
            if not directory.is_dir():
                continue
            for path in directory.glob("*_speaker_centroids.pt"):
                speaker_id = path.name.removesuffix("_speaker_centroids.pt")
                speakers[speaker_id] = {
                    "id": speaker_id,
                    "name": _SPEAKER_LABELS.get(speaker_id, speaker_id),
                    "desc": "模型內建語者",
                    "is_custom": False,
                }
        return sorted(speakers.values(), key=lambda item: item["name"].casefold())

    def _release_model(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_checkpoint = None
        self.loaded_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_id: str) -> None:
        if self.loaded_model_id == model_id and self._model is not None:
            return

        checkpoint = self.registry.get(model_id)
        if self._model is not None:
            logger.info("Unloading Barbet model %s", self.loaded_model_id)
            self._release_model()

        from bluemagpie import BlueMagpieModel
        from transformers import PreTrainedTokenizerFast

        logger.info("Loading local Barbet model %s from %s", model_id, checkpoint.path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(checkpoint.path / "tokenizer.json"))
        model = BlueMagpieModel.from_local(
            str(checkpoint.path),
            tokenizer=tokenizer,
            training=False,
            device=self.device,
        )
        self._tokenizer = tokenizer
        self._model = model
        self._loaded_checkpoint = checkpoint
        self.loaded_model_id = model_id

    def _extract_speaker_centroid(self, reference_path: str) -> torch.Tensor:
        from bluemagpie.centroid import _load_encoder, extract_speaker_centroid

        if self._speaker_encoder is None:
            self._speaker_encoder = _load_encoder(
                "speechbrain/spkrec-ecapa-voxceleb",
                "cpu",
            )
        return extract_speaker_centroid(
            reference_path,
            device="cpu",
            encoder=self._speaker_encoder,
        )

    def _load_speaker_centroid(self, speaker_id: str) -> torch.Tensor | None:
        if not speaker_id or self._loaded_checkpoint is None:
            return None
        for directory in (
            self._loaded_checkpoint.path,
            self._loaded_checkpoint.path / "checkpoints",
        ):
            path = directory / f"{speaker_id}_speaker_centroids.pt"
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            speaker_ids = payload.get("speaker_ids", [])
            if speaker_id in speaker_ids:
                return payload["centroids"][speaker_ids.index(speaker_id)]
        return None

    def synthesize(
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
    ) -> tuple[int, np.ndarray, float]:
        with self._lock:
            self._load_model(model_id)
            centroid = (
                self._extract_speaker_centroid(reference_path)
                if reference_path
                else self._load_speaker_centroid(speaker_id)
            )
            prompt_text = prompt_text.strip()
            if prompt_text and not reference_path:
                raise ValueError("Barbet prompt 逐字稿必須搭配參考音訊")
            started_at = time.perf_counter()
            audio = self._model.generate(
                target_text=text,
                prompt_text=prompt_text,
                prompt_wav_path=reference_path if prompt_text else "",
                speaker_centroid=centroid,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                retry_badcase=True,
                seed=seed,
            )
            elapsed = time.perf_counter() - started_at
            sample_rate = int(self._model.sample_rate)
        return sample_rate, audio.squeeze().float().cpu().numpy(), elapsed

    def close(self) -> None:
        with self._lock:
            self._release_model()
            self._speaker_encoder = None
