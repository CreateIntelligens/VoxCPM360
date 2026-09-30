"""Configuration and deployment checks that protect catalog claims."""

from pathlib import Path

import pytest

import runtime_config
from gateway.gateway import TTSGateway


@pytest.mark.parametrize("value", ["0", "51", "-1", "abc", "1.5", ""])
def test_rejects_invalid_default_timesteps(monkeypatch, value):
    monkeypatch.setenv("VOXCPM_INFERENCE_TIMESTEPS", value)
    with pytest.raises(
        ValueError,
        match="VOXCPM_INFERENCE_TIMESTEPS must be an integer from 1 to 50",
    ):
        runtime_config.read_default_inference_timesteps()


@pytest.mark.parametrize("value", ["1", "10", "50"])
def test_accepts_valid_default_timesteps(monkeypatch, value):
    monkeypatch.setenv("VOXCPM_INFERENCE_TIMESTEPS", value)
    assert runtime_config.read_default_inference_timesteps() == int(value)


def test_gateway_refuses_unpatched_engine_before_advertising_capability(monkeypatch):
    original = Path.read_text

    def unpatched_runner(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        if path.name == "runner.py":
            return source.removeprefix(runtime_config.MARKER)
        return source

    monkeypatch.setattr(Path, "read_text", unpatched_runner)
    with pytest.raises(
        RuntimeError, match="runner.py lacks the per-request timesteps patch"
    ):
        TTSGateway(object())


def test_gateway_refuses_loaded_engine_without_timesteps():
    class UnpatchedEngine:
        def generate(self, target_text):
            return iter(())

    class Demo:
        voxcpm_server = UnpatchedEngine()

    with pytest.raises(RuntimeError, match="lacks per-request timesteps"):
        TTSGateway(Demo())
