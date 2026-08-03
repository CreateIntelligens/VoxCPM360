#!/usr/bin/env python3
"""Make FlashAttention 2.6.3 honor the image's target CUDA architectures."""

from __future__ import annotations

import argparse
from pathlib import Path

_HARDCODED_FLAGS = """    # cc_flag.append("-gencode")
    # cc_flag.append("arch=compute_75,code=sm_75")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")"""


def cuda_gencode_flags(architectures: str) -> list[str]:
    flags: list[str] = []
    for raw_arch in architectures.replace(",", ";").split(";"):
        arch = raw_arch.strip().removesuffix("+PTX")
        if not arch:
            continue
        numeric = arch.replace(".", "")
        if not numeric.isdigit():
            raise ValueError(f"Invalid CUDA architecture: {raw_arch!r}")
        flags.extend(["-gencode", f"arch=compute_{numeric},code=sm_{numeric}"])
    if not flags:
        raise ValueError("At least one CUDA architecture is required")
    return flags


def patch_setup_text(source: str, architectures: str) -> str:
    if _HARDCODED_FLAGS not in source:
        raise RuntimeError("FlashAttention setup.py layout is not the expected v2.6.3 form")
    flags = cuda_gencode_flags(architectures)
    replacement = f"    cc_flag = {flags!r}"
    return source.replace(_HARDCODED_FLAGS, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("setup_py", type=Path)
    parser.add_argument("--architectures", required=True)
    args = parser.parse_args()

    source = args.setup_py.read_text(encoding="utf-8")
    patched = patch_setup_text(source, args.architectures)
    args.setup_py.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
