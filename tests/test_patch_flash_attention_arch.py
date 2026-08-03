from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "patch_flash_attention_arch.py"
_SPEC = importlib.util.spec_from_file_location("patch_flash_attention_arch", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_cuda_gencode_flags_supports_blackwell():
    assert _MODULE.cuda_gencode_flags("12.0") == [
        "-gencode",
        "arch=compute_120,code=sm_120",
    ]


def test_cuda_gencode_flags_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid CUDA architecture"):
        _MODULE.cuda_gencode_flags("blackwell")
