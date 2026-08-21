from __future__ import annotations

import numpy as np
import pytest

from app import VoxCPMDemo


class FakeRegistry:
    def __init__(self):
        self.selections = []

    def ensure_registered(self, server, selection):
        self.selections.append(selection)
        return "trial-adapter"


class FakeServer:
    def __init__(self):
        self.lora_names = []
        self.target_texts = []
        self.max_generate_lengths = []

    def generate(
        self,
        target_text,
        prompt_latents=None,
        prompt_text="",
        max_generate_length=2000,
        temperature=1.0,
        cfg_value=2.0,
        ref_audio_latents=None,
        lora_name=None,
    ):
        self.lora_names.append(lora_name)
        self.target_texts.append(target_text)
        self.max_generate_lengths.append(max_generate_length)
        yield np.zeros(16, dtype=np.float32)

    def get_model_info(self):
        return {"sample_rate": 16000}


def test_generate_passes_selected_lora_to_nano_server():
    server = FakeServer()
    registry = FakeRegistry()
    demo = VoxCPMDemo.__new__(VoxCPMDemo)
    demo.voxcpm_server = server
    demo.lora_registry = registry
    demo.text_normalizer = None
    demo.denoiser = None

    sample_rate, audio = demo.generate_tts_audio(
        text_input="測試",
        do_normalize=False,
        denoise=False,
        model_selection="trial_lora_20epochs",
    )

    assert sample_rate == 16000
    assert audio.shape == (16,)
    assert registry.selections == ["trial_lora_20epochs"]
    assert server.lora_names == ["trial-adapter"]
    assert server.max_generate_lengths == [22]


def test_generate_length_guard_is_configurable():
    server = FakeServer()
    demo = VoxCPMDemo.__new__(VoxCPMDemo)
    demo.voxcpm_server = server
    demo.lora_registry = FakeRegistry()
    demo.text_normalizer = None
    demo.denoiser = None
    demo.max_generate_length = 15
    demo.max_audio_text_ratio = 3.0

    demo.generate_tts_audio(
        text_input="這是一段較長的測試文字",
        do_normalize=False,
        denoise=False,
    )

    assert server.max_generate_lengths == [15]


def test_prompt_cloning_does_not_prefix_control_instruction(tmp_path):
    server = FakeServer()
    demo = VoxCPMDemo.__new__(VoxCPMDemo)
    demo.voxcpm_server = server
    demo.lora_registry = FakeRegistry()
    demo.text_normalizer = None
    demo.denoiser = None
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"RIFF")
    server.encode_latents = lambda *_args: np.zeros((1, 1), dtype=np.float32)

    demo.generate_tts_audio(
        text_input="真正要說的內容",
        control_instruction="用女聲說",
        reference_wav_path_input=str(reference),
        prompt_text="參考音訊內容",
        do_normalize=False,
        denoise=False,
    )

    assert server.target_texts == ["真正要說的內容"]


def test_defaults_to_guarded_gpu_memory_utilization(monkeypatch, tmp_path):
    monkeypatch.delenv("VOXCPM_GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.setenv("VOXCPM_LORA_ROOT", str(tmp_path))

    demo = VoxCPMDemo(device="cpu")

    assert demo.gpu_memory_utilization == 0.35


def test_rejects_invalid_gpu_memory_utilization(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_GPU_MEMORY_UTILIZATION", "1.1")
    monkeypatch.setenv("VOXCPM_LORA_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="must be in"):
        VoxCPMDemo(device="cpu")
