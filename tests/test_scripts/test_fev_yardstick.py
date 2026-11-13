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
    assert "simons" in result.stdout and "--checkpoint" in result.stdout
    assert "--processing" in result.stdout


def test_simons_checkpoint_alias_and_required_processing(tmp_path, monkeypatch, capsys):
    import pytest

    pytest.importorskip("fev")
    import pandas as pd

    from demand_forecast.bench import yardsticks
    from scripts.fev_yardstick import main

    captured = []

    def run(tasks, options, out_dir, **kwargs):
        captured.append(options.resolved())
        return pd.DataFrame([{"task": "synthetic", "model_name": "simons-raw"}])

    monkeypatch.setattr(
        yardsticks, "load_tasks", lambda *args: [type("Task", (), {"task_name": "synthetic"})()]
    )
    monkeypatch.setattr(yardsticks, "run_yardstick", run)
    args = [
        "--model",
        "simons",
        "--checkpoint",
        str(tmp_path / "export"),
        "--tasks",
        str(tmp_path / "tasks.yaml"),
        "--model-name",
        "simons-raw",
        "--out-dir",
        str(tmp_path / "out"),
    ]
    assert main([*args, "--processing", "raw"]) == 0
    assert captured[0].weights == str(tmp_path / "export")
    assert captured[0].processing == "raw"
    assert main(args) == 2
    assert "processing" in capsys.readouterr().err


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
