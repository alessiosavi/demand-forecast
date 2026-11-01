"""Package exports stay importable after the legacy pipeline is retired."""

import importlib.util
import subprocess
import sys
import textwrap


def test_public_lazy_exports_resolve_and_legacy_modules_are_gone():
    import demand_forecast.core as core
    import demand_forecast.data as data
    import demand_forecast.inference as inference
    import demand_forecast.utils as utils

    for module in (core, data, utils, inference):
        for name in module.__all__:
            assert getattr(module, name) is not None

    for module_name in (
        "core.pipeline",
        "core.forecast_run",
        "core.benchmark",
        "core.trainer",
        "core.evaluator",
        "data.loader",
        "data.preprocessor",
        "data.dataset",
        "data.dataloader",
        "utils.time_features",
        "utils.metrics",
        "inference.confidence",
    ):
        assert importlib.util.find_spec(f"demand_forecast.{module_name}") is None


def test_cli_help_lists_only_live_commands_without_bench(tmp_path):
    # A fresh process catches eager optional imports even after bench tests ran.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import sys

                for name in ("fev", "datasets", "statsforecast"):
                    sys.modules[name] = None

                from typer.testing import CliRunner
                from typer.main import get_command
                from demand_forecast.cli import app
                from demand_forecast.core.run import train_from_settings, forecast_from_bundle
                from demand_forecast.inference import Predictor
                from demand_forecast.bench.specs import load_task_specs

                assert len(load_task_specs("weekly")) == 7
                runner = CliRunner()
                result = runner.invoke(app, ["--help"])
                assert result.exit_code == 0, (result.output, result.exception)
                assert all(command in result.output for command in (
                    "train", "forecast", "fev", "generate-data", "generate-sales", "version"
                )), result.output
                assert set(get_command(app).commands) == {
                    "train", "forecast", "fev", "generate-data", "generate-sales", "version"
                }
                result = runner.invoke(app, [
                    "fev", "baseline", "--tasks", "weekly", "--out-dir", "baseline"
                ])
                assert result.exit_code != 0, result.output
                assert "install demand-forecast[bench]" in result.output, result.output
            """),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_version_is_2_0_0():
    from typer.testing import CliRunner

    from demand_forecast.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == "demand-forecast version 2.0.0"


def test_pretrain_imports_without_bench(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import importlib
                import sys

                for name in ("pyarrow", "datasets", "huggingface_hub", "fev", "statsforecast"):
                    sys.modules[name] = None

                importlib.import_module("demand_forecast.pretrain")
                importlib.import_module("demand_forecast.pretrain.corpus")
            """),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
