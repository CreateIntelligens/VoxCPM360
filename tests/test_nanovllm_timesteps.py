"""Verify the installed engine path, not just API response headers."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.patch_nanovllm_timesteps import MARKER, transform


@pytest.fixture(scope="module")
def engine_root():
    spec = importlib.util.find_spec("nanovllm_voxcpm")
    if spec is None:
        pytest.skip("nano-vllm-voxcpm is required for runtime patch checks")
    return Path(spec.origin).parent / "models" / "voxcpm2"


def load_method(source, class_name, method_name, scope):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, (ast.FunctionDef,
                  ast.AsyncFunctionDef)) and n.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, "<installed-runtime>", "exec", flags=0x1000000), scope)
    return scope[method_name]


def make_runner(engine_root):
    source = transform("runner.py", (engine_root / "runner.py").read_text())
    run = load_method(source, "VoxCPM2Runner", "run", {})
    runner = SimpleNamespace(
        inference_timesteps=10,
        enforce_eager=False,
        model=SimpleNamespace(feat_decoder=SimpleNamespace(
            inference_timesteps=10)),
    )
    calls = []

    def execute(seqs, is_prefill):
        steps = runner.model.feat_decoder.inference_timesteps
        calls.append((steps, runner.enforce_eager, is_prefill))
        return [(seq.identity, steps) for seq in seqs]

    runner._run_fixed_timesteps = execute
    return run, runner, calls


def tasks(steps):
    return [SimpleNamespace(identity=i, custom_payload=SimpleNamespace(
        inference_timesteps=step)) for i, step in enumerate(steps)]


@pytest.mark.parametrize("is_prefill", [True, False])
def test_mixed_requests_keep_order_and_graph_isolation(engine_root, is_prefill):
    run, runner, calls = make_runner(engine_root)
    result = run(runner, tasks([30, 10, 20, 30, None]), is_prefill)
    assert result == [(0, 30), (1, 10), (2, 20), (3, 30), (4, 10)]
    assert calls == [(30, True, is_prefill), (10, False, is_prefill),
                     (20, True, is_prefill)]
    assert runner.model.feat_decoder.inference_timesteps == 10
    assert runner.enforce_eager is False


def test_failed_request_restores_decoder_and_graph_mode(engine_root):
    run, runner, _ = make_runner(engine_root)

    def fail(*args):
        raise RuntimeError("inference failed")

    runner._run_fixed_timesteps = fail
    with pytest.raises(RuntimeError, match="inference failed"):
        run(runner, tasks([20]), False)
    assert runner.model.feat_decoder.inference_timesteps == 10
    assert runner.enforce_eager is False


@pytest.mark.parametrize("steps", [10, 20, 30])
def test_real_cfm_executes_requested_diffusion_schedule(engine_root, steps):
    spec = importlib.util.spec_from_file_location(
        "timesteps_solver", engine_root / "model_utils_cfm.py")
    solver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solver
    spec.loader.exec_module(solver)
    scope = {"torch": torch, "build_cfm_t_span": solver.build_cfm_t_span}
    forward = load_method((engine_root / "model.py").read_text(),
                          "UnifiedCFM", "forward", scope)
    schedules = []
    estimator_calls = []

    def integrate(x, t_span, mu, cond, cfg_value):
        schedules.append(t_span)
        ops = SimpleNamespace(
            estimator=lambda x, *args: (
                estimator_calls.append(1) or torch.ones_like(x)),
            optimized_scale=solver.compute_optimized_scale,
        )
        return solver.solve_euler(
            solver.EulerSolverInputs(x, t_span, mu, cond, cfg_value),
            solver.EulerSolverConfig(2, False), ops)

    decoder = SimpleNamespace(inference_timesteps=steps, in_channels=2,
                              patch_size=2, solve_euler=integrate)
    forward(decoder, torch.ones(1, 2), torch.zeros(1, 2, 2),
            torch.ones(1), torch.full((1,), 2.0), torch.ones(1, 2, 2))
    assert len(schedules[0]) == steps + 1
    assert len(estimator_calls) == (
        steps - solver.compute_zero_init_steps(steps + 1))


@pytest.mark.parametrize("name", ["server.py", "engine.py", "runner.py"])
def test_patch_is_valid_and_idempotent(engine_root, name):
    result = transform(name, (engine_root / name).read_text())
    assert result.startswith(MARKER)
    ast.parse(result)
    assert transform(name, result) == result


def test_incompatible_dependency_fails_explicitly():
    with pytest.raises(RuntimeError, match="Unsupported nano-vLLM"):
        transform("server.py", "# incompatible version")
