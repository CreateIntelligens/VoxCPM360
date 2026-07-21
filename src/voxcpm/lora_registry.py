from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

BASE_MODEL_KEY = "__base__"
BASE_MODEL_LABEL = "基礎模型"
_DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
_DEFAULT_PROJ_MODULES = (
    "enc_to_lm_proj",
    "lm_to_dit_proj",
    "res_to_dit_proj",
)


def _ordered_targets(
    names: Iterable[str],
    preferred_order: tuple[str, ...],
) -> list[str]:
    unique_names = set(names)
    preferred_names = [name for name in preferred_order if name in unique_names]
    return preferred_names + sorted(unique_names.difference(preferred_order))


@dataclass(frozen=True)
class LoraCheckpoint:
    run_name: str
    path: Path
    config: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.run_name} / latest"


def discover_lora_checkpoints(root: Path) -> tuple[LoraCheckpoint, ...]:
    """Scan root directory and discover all valid LoRA checkpoints containing latest/lora_config.json."""
    root = Path(root)
    if not root.is_dir():
        return ()

    checkpoints = []
    for run_dir in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        latest = run_dir / "latest"
        config_path = latest / "lora_config.json"
        weights_path = latest / "lora_weights.safetensors"
        if not run_dir.is_dir() or not config_path.is_file() or not weights_path.is_file():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not isinstance(
                config.get("lora_config"), dict
            ):
                raise ValueError("lora_config must be an object")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Ignoring invalid LoRA checkpoint %s: %s", latest, exc)
            continue

        checkpoints.append(
            LoraCheckpoint(
                run_name=run_dir.name,
                path=latest,
                config=config,
            )
        )

    return tuple(checkpoints)


def build_nano_lora_config(
    checkpoints: Iterable[LoraCheckpoint],
) -> dict[str, Any]:
    """Construct union LoRA config required by nano-vllm runtime server."""
    checkpoint_list = tuple(checkpoints)
    if not checkpoint_list:
        return {
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": False,
            "max_loras": 1,
            "max_lora_rank": 32,
            "target_modules_lm": list(_DEFAULT_TARGET_MODULES),
            "target_modules_dit": list(_DEFAULT_TARGET_MODULES),
            "target_proj_modules": list(_DEFAULT_PROJ_MODULES),
        }

    configs = tuple(
        checkpoint.config["lora_config"] for checkpoint in checkpoint_list
    )

    def collect_targets(key: str, preferred_order: tuple[str, ...]) -> list[str]:
        names = (
            str(name)
            for config in configs
            for name in config.get(key, ())
        )
        return _ordered_targets(names, preferred_order)

    return {
        "enable_lm": any(bool(config.get("enable_lm", True)) for config in configs),
        "enable_dit": any(
            bool(config.get("enable_dit", True)) for config in configs
        ),
        "enable_proj": any(
            bool(config.get("enable_proj", False)) for config in configs
        ),
        "max_loras": 1,
        "max_lora_rank": max(int(config.get("r", 32)) for config in configs),
        "target_modules_lm": collect_targets(
            "target_modules_lm", _DEFAULT_TARGET_MODULES
        ),
        "target_modules_dit": collect_targets(
            "target_modules_dit", _DEFAULT_TARGET_MODULES
        ),
        "target_proj_modules": collect_targets(
            "target_proj_modules", _DEFAULT_PROJ_MODULES
        ),
    }


class LoraRegistry:
    """Manages discovered LoRA checkpoints and adapter registration with VoxCPM server."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self._checkpoints: dict[str, LoraCheckpoint] = {}
        self._registered: dict[str, str] = {}
        self.refresh()

    @property
    def checkpoints(self) -> tuple[LoraCheckpoint, ...]:
        return tuple(self._checkpoints.values())

    def refresh(self) -> tuple[LoraCheckpoint, ...]:
        checkpoints = discover_lora_checkpoints(self.root)
        self._checkpoints = {
            checkpoint.run_name: checkpoint for checkpoint in checkpoints
        }
        self._registered = {
            key: value
            for key, value in self._registered.items()
            if key in self._checkpoints
        }
        return checkpoints

    def choices(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = [(BASE_MODEL_LABEL, BASE_MODEL_KEY)]
        for checkpoint in self.checkpoints:
            items.append((checkpoint.label, checkpoint.run_name))
        return items

    def describe(self, selection: str) -> str:
        if selection == BASE_MODEL_KEY:
            return "目前使用基礎模型。"
        checkpoint = self._checkpoints.get(selection)
        if checkpoint is None:
            return "找不到這個訓練結果，請重新掃描模型。"
        return f"下一次生成將使用 {checkpoint.label}。"

    def ensure_registered(self, server: Any, selection: str) -> str | None:
        if selection == BASE_MODEL_KEY:
            return None

        checkpoint = self._checkpoints.get(selection)
        if checkpoint is None:
            raise ValueError("找不到選取的訓練結果，請重新掃描模型。")

        with self._lock:
            registered_name = self._registered.get(selection)
            if registered_name is not None:
                return registered_name

            adapter_name = self._adapter_name(checkpoint)
            response = server.register_lora(adapter_name, str(checkpoint.path))
            registered_name = str(response["name"])
            self._registered[selection] = registered_name
            logger.info(
                "Registered LoRA adapter %s from %s",
                registered_name,
                checkpoint.path,
            )
            return registered_name

    @staticmethod
    def _adapter_name(checkpoint: LoraCheckpoint) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", checkpoint.run_name).strip("-")
        slug = slug or "checkpoint"
        digest = hashlib.sha256(str(checkpoint.path).encode("utf-8")).hexdigest()[:8]
        return f"ui-{slug}-{digest}"
