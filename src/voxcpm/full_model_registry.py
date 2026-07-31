from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

FULL_MODEL_PREFIX = "full::"
_REQUIRED_FILES = ("config.json", "tokenizer.json", "audiovae.pth")
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class FullModelCheckpoint:
    name: str
    path: Path
    label: str
    description: str
    missing_files: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{FULL_MODEL_PREFIX}{self.name}"

    @property
    def valid(self) -> bool:
        return not self.missing_files


def _read_metadata(path: Path) -> dict[str, str]:
    metadata_path = path / "model.json"
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _inspect_checkpoint(path: Path) -> FullModelCheckpoint | None:
    # Ignore LoRA/training directories that contain no full-model weight file.
    if not any((path / filename).is_file() for filename in _WEIGHT_FILES):
        return None

    missing = [filename for filename in _REQUIRED_FILES if not (path / filename).is_file()]

    metadata = _read_metadata(path)
    description = str(metadata.get("description") or "")
    if missing:
        description = f"缺少檔案：{', '.join(missing)}"
    elif not description:
        description = "完整 VoxCPM2 全參 checkpoint"

    return FullModelCheckpoint(
        name=path.name,
        path=path,
        label=str(metadata.get("label") or path.name),
        description=description,
        missing_files=tuple(missing),
    )


def discover_full_model_checkpoints(
    roots: Iterable[Path],
) -> tuple[FullModelCheckpoint, ...]:
    """Discover full VoxCPM checkpoints; the first root wins duplicate names."""
    checkpoints: dict[str, FullModelCheckpoint] = {}
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
                logger.warning("Ignoring duplicate full model %s from %s", checkpoint.name, path)
                continue
            checkpoints[checkpoint.name] = checkpoint
    return tuple(checkpoints.values())


class FullModelRegistry:
    def __init__(self, roots: Iterable[Path]) -> None:
        self.roots = tuple(Path(root) for root in roots)
        self._checkpoints: dict[str, FullModelCheckpoint] = {}
        self.refresh()

    @property
    def checkpoints(self) -> tuple[FullModelCheckpoint, ...]:
        return tuple(self._checkpoints.values())

    def refresh(self) -> tuple[FullModelCheckpoint, ...]:
        found = discover_full_model_checkpoints(self.roots)
        self._checkpoints = {checkpoint.id: checkpoint for checkpoint in found}
        return found

    def get(self, model_id: str) -> FullModelCheckpoint:
        checkpoint = self._checkpoints.get(model_id)
        if checkpoint is None:
            self.refresh()
            checkpoint = self._checkpoints.get(model_id)
        if checkpoint is None:
            raise ValueError(f"找不到完整 VoxCPM 模型：{model_id}")
        if not checkpoint.valid:
            raise ValueError(checkpoint.description)
        return checkpoint
