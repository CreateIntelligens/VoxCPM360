"""Ensure checkpoint loading never enables unrestricted pickle execution."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_PATHS = [
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "app.py",
    REPO_ROOT / "lora_ft_webui.py",
]


def _python_files():
    for entry in SCANNED_PATHS:
        if entry.is_file() and entry.suffix == ".py":
            yield entry
        elif entry.is_dir():
            yield from entry.rglob("*.py")


def _is_torch_load(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "load"
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _uses_safe_weight_loading(node: ast.Call) -> bool:
    return any(
        keyword.arg == "weights_only"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in node.keywords
    )


def test_every_torch_load_uses_weights_only():
    offenders = []
    checked = 0

    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_torch_load(node):
                continue
            checked += 1
            if not _uses_safe_weight_loading(node):
                relative_path = path.relative_to(REPO_ROOT)
                offenders.append(f"{relative_path}:{node.lineno}")

    assert checked > 0
    assert not offenders, (
        "torch.load without weights_only=True:\n  " + "\n  ".join(offenders)
    )


def test_weights_only_rejects_malicious_pickle(tmp_path):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "executed.txt"

    class Exploit:
        def __reduce__(self):
            import pathlib

            return (pathlib.Path.write_text, (marker, "executed\n"))

    checkpoint = tmp_path / "optimizer.pth"
    torch.save(
        {"state_dict": {"weight": torch.zeros(1)}, "payload": Exploit()},
        checkpoint,
    )

    with pytest.raises(Exception):
        torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert not marker.exists()
