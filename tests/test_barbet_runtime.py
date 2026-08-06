from types import SimpleNamespace

import torch

from voxcpm.barbet_runtime import BarbetRuntime


class _FakeModel:
    sample_rate = 48_000

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1, 480)


def _runtime(monkeypatch):
    runtime = BarbetRuntime(SimpleNamespace(), device="cpu")
    runtime._model = _FakeModel()
    runtime._loaded_checkpoint = SimpleNamespace(path=None)
    runtime.loaded_model_id = "barbet::test"
    monkeypatch.setattr(runtime, "_load_model", lambda model_id: None)
    monkeypatch.setattr(runtime, "_extract_speaker_centroid", lambda path: torch.ones(192))
    return runtime


def test_prompt_transcript_uses_reference_as_prompt_audio(monkeypatch):
    runtime = _runtime(monkeypatch)

    sample_rate, audio, _ = runtime.synthesize(
        model_id="barbet::test",
        text="欲來去食飯",
        reference_path="/tmp/reference.wav",
        prompt_text="你今天有食飽未",
        speaker_id="",
        cfg_value=2.0,
        inference_timesteps=10,
        seed=42,
    )

    call = runtime._model.calls[0]
    assert sample_rate == 48_000
    assert audio.shape == (480,)
    assert call["prompt_text"] == "你今天有食飽未"
    assert call["prompt_wav_path"] == "/tmp/reference.wav"
    assert call["speaker_centroid"].shape == (192,)


def test_empty_prompt_keeps_centroid_only_mode(monkeypatch):
    runtime = _runtime(monkeypatch)

    runtime.synthesize(
        model_id="barbet::test",
        text="欲來去食飯",
        reference_path="/tmp/reference.wav",
        prompt_text="  ",
        speaker_id="",
        cfg_value=2.0,
        inference_timesteps=10,
        seed=None,
    )

    call = runtime._model.calls[0]
    assert call["prompt_text"] == ""
    assert call["prompt_wav_path"] == ""
