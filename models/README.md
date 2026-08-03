# Local model store

Press **重新掃描** in the React studio after adding or removing a model.
No `.env` change or web image rebuild is needed.

## Native VoxCPM2 LoRA

Put each version in `models/native/<version-name>/`.
Both layouts below are supported:

```text
models/native/my-taiwanese-v1/
├── lora_config.json
└── lora_weights.safetensors
```

```text
models/native/my-taiwanese-v1/latest/
├── lora_config.json
└── lora_weights.safetensors
```

Training outputs already present in `/checkpoints` remain visible for backward
compatibility.

## Native VoxCPM2 full checkpoint

Put a full fine-tuned version in `models/native/<version-name>/`:

```text
models/native/tai8-full-step3000/
├── config.json
├── tokenizer.json
├── model.safetensors
└── audiovae.pth
```

Full checkpoints appear as `FULL`. Selecting one stops the active nano-vLLM
worker and loads that complete model; switching back to Base or a LoRA reloads
the original VoxCPM2 runtime. The first request after a switch is therefore
slower than an ordinary LoRA switch.

## Barbet TTS

Put each complete TTS checkpoint in `models/barbet/<version-name>/`:

```text
models/barbet/barbet-taiwanese-v1/
├── config.json
├── tokenizer.json
├── model.safetensors (or pytorch_model.bin)
└── audiovae.pth
```

Optional `model.json`:

```json
{
  "label": "Barbet 台語 v1",
  "description": "2026-07 台語訓練版"
}
```

A standalone `barbet-1b-base` language-model directory is not a complete TTS
checkpoint: it does not contain the trained Barbet-to-VoxCPM bridge, diffusion
decoder weights, and AudioVAE required by the integrated TTS runtime.

### tai8 + Barbet merge

`models/barbet/tai8-barbet-merge-v0/` is an assembled checkpoint, not a new
training run. It keeps the Barbet language model and trained BlueMagpie bridge
from the official BlueMagpie checkpoint, then replaces only the compatible
VoxCPM2 acoustic modules and AudioVAE with
`checkpoints/full_ft_tai8_step3000/`.

The merge directory is retained only as an earlier experiment. Use a trained
`bm-*` checkpoint for actual Barbet TTS; replacing only compatible acoustic
tensors does not transfer the Taiwanese-language knowledge learned by the
text-semantic model.
