from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

BARBET_MODEL_PREFIX = "barbet::"
_REQUIRED_FILES = ("config.json", "tokenizer.json", "audiovae.pth")
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class BarbetCheckpoint:
    name: str
    path: Path
    label: str
    description: str
    missing_files: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{BARBET_MODEL_PREFIX}{self.name}"

    @property
    def valid(self) -> bool:
        return not self.missing_files


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_barbet_checkpoint(path: Path) -> bool:
    config = _read_json(path / "config.json")
    return "barbet_config" in config and "vox_lm_config" in config


def _inspect_checkpoint(path: Path) -> BarbetCheckpoint | None:
    if not (path / "config.json").is_file() or not is_barbet_checkpoint(path):
        return None

    missing = [filename for filename in _REQUIRED_FILES if not (path / filename).is_file()]
    if not any((path / filename).is_file() for filename in _WEIGHT_FILES):
        missing.append("model.safetensors 或 pytorch_model.bin")

    metadata = _read_json(path / "model.json")
    description = str(metadata.get("description") or "")
    if missing:
        description = f"缺少檔案：{', '.join(missing)}"
    elif not description:
        description = "Barbet TSLM + VoxCPM2 聲學模型"

    return BarbetCheckpoint(
        name=path.name,
        path=path,
        label=str(metadata.get("label") or path.name),
        description=description,
        missing_files=tuple(missing),
    )


def discover_barbet_checkpoints(
    roots: Iterable[Path],
) -> tuple[BarbetCheckpoint, ...]:
    """Discover local Barbet TTS checkpoints; the first root wins duplicates."""
    checkpoints: dict[str, BarbetCheckpoint] = {}
    for root in (Path(value) for value in roots):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_dir():
                continue
            checkpoint = _inspect_checkpoint(path)
            if checkpoint is None:
                continue
            if checkpoint.name in checkpoints:
                logger.warning(
                    "Ignoring duplicate Barbet model %s from %s",
                    checkpoint.name,
                    path,
                )
                continue
            checkpoints[checkpoint.name] = checkpoint
    return tuple(checkpoints.values())


class BarbetModelRegistry:
    def __init__(self, roots: Iterable[Path]) -> None:
        self.roots = tuple(Path(root) for root in roots)
        self._checkpoints: dict[str, BarbetCheckpoint] = {}
        self.refresh()

    @property
    def checkpoints(self) -> tuple[BarbetCheckpoint, ...]:
        return tuple(self._checkpoints.values())

    def refresh(self) -> tuple[BarbetCheckpoint, ...]:
        found = discover_barbet_checkpoints(self.roots)
        self._checkpoints = {checkpoint.id: checkpoint for checkpoint in found}
        return found

    def get(self, model_id: str) -> BarbetCheckpoint:
        checkpoint = self._checkpoints.get(model_id)
        if checkpoint is None:
            self.refresh()
            checkpoint = self._checkpoints.get(model_id)
        if checkpoint is None:
            raise ValueError(f"找不到 Barbet 模型：{model_id}")
        if not checkpoint.valid:
            raise ValueError(checkpoint.description)
        return checkpoint
