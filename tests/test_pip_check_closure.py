"""Dependency-check ownership, marker traversal and command failures."""

import builtins
import importlib.metadata
import subprocess
import sys

import pytest


def test_classify_uses_declaring_distribution():
    from scripts.pip_check_closure import classify

    external = "jax 0.7.2 has requirement numpy>=2.0, but you have numpy 1.26.4."
    internal = "torch 2.15.0 requires sympy, which is not installed."
    missing = "ipython 7.34.0 requires jedi, which is not installed."
    normalized = "My_Package 1.0 requires missing, which is not installed."
    assert classify(
        [external, internal, missing, normalized], {"numpy", "torch", "my-package"}
    ) == ([internal, normalized], [external, missing])
    assert classify([external], {"jax"}) == ([external], [])


def test_closure_installed_base_dependencies():
    from scripts.pip_check_closure import closure

    reached = closure("demand-forecast", [])
    assert {"demand-forecast", "numpy", "torch", "pandas"} <= reached
    assert "fev" not in reached


def test_closure_installed_bench_dependencies():
    pytest.importorskip("fev")
    from scripts.pip_check_closure import closure

    assert {"numpy", "torch", "pandas", "fev"} <= closure("demand-forecast", ["bench"])


def test_closure_follows_transitive_extras_markers_cycles_and_missing_packages(monkeypatch):
    from scripts.pip_check_closure import closure

    metadata = {
        "root-package": [
            "shared",
            "branch[fast-mode]; extra == 'bench'",
            "absent",
            "wrong-python; python_version < '1'",
        ],
        "branch": ["shared[feature]", "fast-dep; extra == 'fast-mode'"],
        "shared": ["root-package[bench]", "leaf; extra == 'feature'", "base"],
        "leaf": None,
        "base": None,
        "fast-dep": None,
    }

    def requires(name):
        if name not in metadata:
            raise importlib.metadata.PackageNotFoundError(name)
        return metadata[name]

    monkeypatch.setattr(importlib.metadata, "requires", requires)
    assert closure("Root_Package", ["bench"]) == set(metadata)
    assert closure("absent", []) == set()


def test_missing_packaging_has_actionable_error(monkeypatch):
    from scripts.pip_check_closure import closure

    real_import = builtins.__import__

    def without_packaging(name, *args, **kwargs):
        if name.startswith("packaging"):
            raise ImportError("missing packaging")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_packaging)
    with pytest.raises(ImportError, match="install packaging"):
        closure("demand-forecast", [])


@pytest.mark.parametrize(
    "returncode,stdout,stderr,expected",
    [
        (0, "No broken requirements found.\n", "", 0),
        (1, "jax 0.7.2 has requirement numpy>=2.0, but you have numpy 1.26.4.\n", "", 0),
        (1, "torch 2.15.0 requires sympy, which is not installed.\n", "", 1),
        (1, "", "No module named pip\n", 1),
        (1, "unexpected pip output\n", "", 1),
        (2, "", "pip failed\n", 1),
    ],
)
def test_main_reports_raw_output_and_exit_status(
    monkeypatch, tmp_path, capsys, returncode, stdout, stderr, expected
):
    from scripts import pip_check_closure as script

    def run(command, **kwargs):
        assert command == [sys.executable, "-m", "pip", "check"]
        assert kwargs["capture_output"] and kwargs["text"]
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    def closure(root, extras):
        assert root == "demand-forecast" and extras == ["bench"]
        return {"demand-forecast", "torch", "numpy"}

    monkeypatch.setattr(script.subprocess, "run", run)
    monkeypatch.setattr(script, "closure", closure)
    report = tmp_path / "nested" / "pip-check.txt"
    assert script.main(["--extras", "bench", "--report", str(report)]) == expected
    assert report.read_text() == stdout + stderr
    output = capsys.readouterr()
    if stdout.startswith("jax"):
        assert "Recorded" in output.out and stdout.strip() in output.out
    if stdout.startswith("torch"):
        assert "Failures" in output.out and stdout.strip() in output.out
