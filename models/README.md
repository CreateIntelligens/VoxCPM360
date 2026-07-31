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

## Barbet / BlueMagpie TTS

Put each complete TTS checkpoint in `models/barbet/<version-name>/`:

```text
models/barbet/barbet-taiwanese-v1/
├── config.json
├── tokenizer.json
├── pytorch_model.bin
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
decoder weights, and AudioVAE required by BlueMagpie.

### tai8 + Barbet merge

`models/barbet/tai8-barbet-merge-v0/` is an assembled checkpoint, not a new
training run. It keeps the Barbet language model and trained BlueMagpie bridge
from the official BlueMagpie checkpoint, then replaces only the compatible
VoxCPM2 acoustic modules and AudioVAE with
`checkpoints/full_ft_tai8_step3000/`.

The reusable merge tool is
`../BlueMagpie-TTS/scripts/merge_voxcpm_acoustic.py`. It performs a dry-run
name/shape check, refuses to overwrite an existing output, writes to a sibling
temporary directory, reloads the completed checkpoint, and only then renames
it into the model store. Source checkpoints are read-only inputs.
