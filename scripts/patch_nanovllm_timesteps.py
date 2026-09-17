"""Add per-request VoxCPM2 diffusion steps to nano-vllm-voxcpm 2.0.4.

Apply at image build time. Alternate step counts use eager execution, because
existing CUDA graphs contain the startup step count. Requests are grouped inside
one worker step, preserving sequence order and the default graph fast path.
"""

import argparse
import ast
import importlib.util
from pathlib import Path

MARKER = "# VoxCPM360 per-request diffusion steps v1\n"


def replace(source, old, new, count):
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(
            f"Unsupported nano-vLLM source: expected {count} matches, "
            f"got {actual} for {old!r}"
        )
    return source.replace(old, new)


def transform(name, source):
    if source.startswith(MARKER):
        return source
    if name == "server.py":
        source = replace(
            source, "        seed: int | None = None,\n",
            "        seed: int | None = None,\n"
            "        inference_timesteps: int | None = None,\n", 4,
        )
        source = replace(
            source, "                seed=seed,\n",
            "                seed=seed,\n"
            "                inference_timesteps=inference_timesteps,\n", 1,
        )
        source = replace(
            source, "            seed=seed,\n        )",
            "            seed=seed,\n"
            "            inference_timesteps=inference_timesteps,\n        )", 1,
        )
        for indent in (16, 12):
            old = "\n" + " " * indent + "seed,\n"
            source = replace(
                source, old,
                old + " " * indent + "inference_timesteps,\n",
                2 if indent == 16 else 1,
            )
    elif name == "engine.py":
        source = replace(
            source, "    seed_step: int = 0\n",
            "    seed_step: int = 0\n"
            "    inference_timesteps: int | None = None\n", 1,
        )
        source = replace(
            source, "        seed: int | None = None,\n",
            "        seed: int | None = None,\n"
            "        inference_timesteps: int | None = None,\n", 1,
        )
        source = replace(
            source, "        if max_generate_length < 1:",
            "        if inference_timesteps is not None and (\n"
            "            isinstance(inference_timesteps, bool)\n"
            "            or not isinstance(inference_timesteps, int)\n"
            "            or not 1 <= inference_timesteps <= 50\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                \"inference_timesteps must be an integer from 1 to 50\"\n"
            "            )\n"
            "        if max_generate_length < 1:", 1,
        )
        source = replace(
            source, "                seed_step=0,\n",
            "                seed_step=0,\n"
            "                inference_timesteps=inference_timesteps,\n", 1,
        )
        for indent in (20, 16):
            old = "\n" + " " * indent + "seed_step=seq.custom_payload.seed_step,\n"
            source = replace(
                source, old, old + " " * indent
                + "inference_timesteps=seq.custom_payload.inference_timesteps,\n",
                1,
            )
    elif name == "runner.py":
        source = replace(
            source, "    seed_step: int = 0\n",
            "    seed_step: int = 0\n"
            "    inference_timesteps: int | None = None\n", 1,
        )
        signature = (
            "    def run(self, seqs: list[RunnerTask[VoxCPM2Payload]], "
            "is_prefill: bool):\n"
        )
        wrapper = '''    def run(self, seqs: list[RunnerTask[VoxCPM2Payload]], is_prefill: bool):
        groups = {}
        for index, seq in enumerate(seqs):
            steps = seq.custom_payload.inference_timesteps
            if steps is None:
                steps = self.inference_timesteps
            groups.setdefault(steps, []).append((index, seq))
        results = [None] * len(seqs)
        decoder = self.model.feat_decoder
        previous_steps = decoder.inference_timesteps
        previous_eager = self.enforce_eager
        try:
            for steps, indexed_seqs in groups.items():
                decoder.inference_timesteps = steps
                # Captured graphs retain the startup diffusion loop length.
                self.enforce_eager = (
                    previous_eager or steps != self.inference_timesteps
                )
                outputs = self._run_fixed_timesteps(
                    [seq for _, seq in indexed_seqs], is_prefill
                )
                for (index, _), output in zip(indexed_seqs, outputs):
                    results[index] = output
        finally:
            decoder.inference_timesteps = previous_steps
            self.enforce_eager = previous_eager
        return results

'''
        source = replace(
            source, signature,
            wrapper + signature.replace("def run(", "def _run_fixed_timesteps("),
            1,
        )
    else:
        raise ValueError(name)
    ast.parse(source)
    return MARKER + source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    root = args.package_root
    if root is None:
        spec = importlib.util.find_spec("nanovllm_voxcpm")
        if spec is None or spec.origin is None:
            raise RuntimeError("nano-vllm-voxcpm is not installed")
        root = Path(spec.origin).parent
    model_dir = root / "models" / "voxcpm2"
    # Validate all inputs before touching an installed package.
    updates = {
        model_dir / name: transform(name, (model_dir / name).read_text())
        for name in ("server.py", "engine.py", "runner.py")
    }
    for path, source in updates.items():
        if path.read_text() != source:
            path.write_text(source)
        print(f"per-request timesteps ready: {path}")


if __name__ == "__main__":
    main()
