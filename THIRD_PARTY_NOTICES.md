# Third-party notices

VoxCPM360 includes source code from the following Apache-2.0 projects so its
Barbet runtime can be built and run without a second source checkout or API
service:

- **BlueMagpie-TTS**, Copyright 2026 OpenFormosa. The integrated sources live
  in `src/bluemagpie/`; its license is retained at
  `src/bluemagpie/LICENSE`.
- **Barbet**, Copyright 2026 OpenFormosa. The integrated sources live in
  `src/barbet/`; its license is retained at `src/barbet/LICENSE`.
- **VoxCPM**, Copyright 2026 OpenBMB. BlueMagpie-TTS's vendored acoustic
  implementation and provenance are retained under
  `src/bluemagpie/_vendor/voxcpm/`.

`src/bluemagpie/model.py` is modified locally to support safetensors
checkpoints produced by VoxCPM360 training.
