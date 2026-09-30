"""Validate native runtime requirements before exposing synthesis endpoints."""

import inspect
import os
from pathlib import Path
from typing import Any

from scripts.patch_nanovllm_timesteps import MARKER


def read_default_inference_timesteps() -> int:
    setting = "VOXCPM_INFERENCE_TIMESTEPS"
    raw = os.environ.get(setting, "10")
    try:
        steps = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{setting} must be an integer from 1 to 50; got {raw!r}"
        ) from exc
    if not 1 <= steps <= 50:
        raise ValueError(
            f"{setting} must be an integer from 1 to 50; got {raw!r}"
        )
    return steps


def require_per_request_timesteps(server: Any = None) -> None:
    from nanovllm_voxcpm.models.voxcpm2 import server as native_server

    engine_root = Path(native_server.__file__).parent
    for name in ("server.py", "engine.py", "runner.py"):
        if not (engine_root / name).read_text().startswith(MARKER):
            raise RuntimeError(
                f"nano-vLLM {name} lacks the per-request timesteps patch; "
                "rebuild the app image with scripts/patch_nanovllm_timesteps.py"
            )

    # The signature check also catches an already loaded or replaced server.
    engines = [
        native_server.SyncVoxCPM2ServerPool,
        native_server.AsyncVoxCPM2ServerPool,
        native_server.AsyncVoxCPM2Server,
    ]
    if server is not None:
        engines.append(server)
    for engine in engines:
        generate = getattr(engine, "generate", None)
        try:
            supported = generate is not None and (
                "inference_timesteps" in inspect.signature(generate).parameters
            )
        except (TypeError, ValueError):
            supported = False
        if not supported:
            raise RuntimeError(
                "nano-vLLM lacks per-request timesteps; rebuild the app image "
                "with scripts/patch_nanovllm_timesteps.py"
            )
