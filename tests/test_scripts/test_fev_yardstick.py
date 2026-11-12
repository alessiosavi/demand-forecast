"""CLI help must work without benchmark/model libraries, even outside the tree."""

import os
import subprocess
import sys
from pathlib import Path


def test_help_uses_only_standard_library(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/fev_yardstick.py"
    # -S removes site-packages, including fev, torch and both model packages.
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--cross-series" in result.stdout and "--revision" in result.stdout


def test_main_imports_outside_checkout(tmp_path):
    import pytest

    pytest.importorskip("fev")
    script = Path(__file__).resolve().parents[2] / "scripts/fev_yardstick.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            "chronos2",
            "--weights",
            "unused",
            "--tasks",
            str(tmp_path / "missing.yaml"),
            "--out-dir",
            str(tmp_path / "out"),
            "--model-name",
            "reference",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2 and "Yardstick failed:" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
