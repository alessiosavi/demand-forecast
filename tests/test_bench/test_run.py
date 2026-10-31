import subprocess
import sys

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import leaderboard, run_baseline  # noqa: E402
from demand_forecast.cli import app  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402


def test_run_baseline_on_fixture_is_resumable(tmp_path, monkeypatch):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    first = run_baseline([spec], tmp_path / "out")

    def fail_if_recomputed(*args):
        pytest.fail("completed task was recomputed")

    monkeypatch.setattr(type(spec), "to_task", fail_if_recomputed)
    second = run_baseline([spec], tmp_path / "out")
    assert len(first) == len(second) == 1
    assert second["SQL"].iloc[0] == first["SQL"].iloc[0]
    assert second["num_forecasts"].iloc[0] == 12
    board = leaderboard(tmp_path / "out/summaries.csv")
    assert board.loc["seasonal_naive", "skill_score"] == pytest.approx(0.0)
    assert {"skill_score_lower", "skill_score_upper"} <= set(board.columns)


def test_run_persists_each_completed_task_and_preserves_other_models(tmp_path, monkeypatch, capsys):
    first_spec = fixture_spec(write_fev_fixture(tmp_path / "first"), name="first")
    second_spec = fixture_spec(write_fev_fixture(tmp_path / "second"), name="second")
    real_to_task = type(first_spec).to_task

    def interrupted(spec):
        if spec.name == "second":
            raise RuntimeError("interrupted")
        return real_to_task(spec)

    monkeypatch.setattr(type(first_spec), "to_task", interrupted)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_baseline([first_spec, second_spec], out)
    assert "Skipping completed task" not in capsys.readouterr().out
    saved = pd.read_csv(out / "summaries.csv", float_precision="round_trip")
    assert saved["task"].tolist() == ["first"]
    other_model = saved.assign(task="second", model_name="other")
    pd.concat([saved, other_model]).to_csv(out / "summaries.csv", index=False)
    monkeypatch.setattr(type(first_spec), "to_task", real_to_task)
    resumed = run_baseline([first_spec, second_spec, second_spec], out)
    assert capsys.readouterr().out.splitlines() == [
        "Skipping completed task first",
        "Skipping completed task second",
    ]
    assert list(zip(resumed["task"], resumed["model_name"], strict=True)) == [
        ("first", "seasonal_naive"),
        ("second", "other"),
        ("second", "seasonal_naive"),
    ]
    assert resumed["SQL"].iloc[0] == saved["SQL"].iloc[0]


def test_leaderboard_uses_requested_baseline_and_bootstrap(tmp_path, monkeypatch):
    csv = tmp_path / "summaries.csv"
    pd.DataFrame([{"model_name": "reference", "SQL": 1.25}]).to_csv(csv, index=False)
    expected = pd.DataFrame({"skill_score": [0.0]}, index=["reference"])

    def checked_leaderboard(summaries, *, baseline_model, n_resamples):
        assert summaries["SQL"].tolist() == [1.25]
        assert baseline_model == "reference" and n_resamples == 1000
        return expected

    monkeypatch.setattr(fev, "leaderboard", checked_leaderboard)
    assert leaderboard(csv, baseline="reference") is expected


def test_cli_runs_local_baseline_then_leaderboard(tmp_path):
    parquet = write_fev_fixture(tmp_path)
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "name": "fixture",
                        "dataset_path": str(parquet),
                        "horizon": 4,
                        "num_windows": 2,
                        "eval_metric": "SQL",
                        "quantile_levels": [0.1, 0.5, 0.9],
                        "known_dynamic_columns": ["promo"],
                        "past_dynamic_columns": ["oil"],
                        "static_columns": ["cat"],
                    }
                ]
            }
        )
    )
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text("fixture:\n  target_support: nonnegative\n  categorical_statics: [cat]\n")
    runner = CliRunner()
    baseline = runner.invoke(
        app,
        [
            "fev",
            "baseline",
            "--tasks",
            str(tasks),
            "--profiles",
            str(profiles),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert baseline.exit_code == 0, baseline.output
    board = runner.invoke(app, ["fev", "leaderboard", str(tmp_path / "out/summaries.csv")])
    assert board.exit_code == 0, board.output
    assert "seasonal_naive" in board.output and "skill_score" in board.output


def test_cli_and_specs_import_without_bench_and_operations_explain_missing_extra():
    code = """
import sys
for dependency in ('fev', 'datasets', 'statsforecast'):
    sys.modules[dependency] = None
from typer.testing import CliRunner
from demand_forecast.cli import app
from demand_forecast.bench.specs import load_task_specs
from demand_forecast.bench.run import run_baseline, leaderboard
specs = load_task_specs('weekly')
assert len(specs) == 7
for args in (['--help'], ['fev', '--help']):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    if args[0] == 'fev':
        assert 'baseline' in result.output and 'leaderboard' in result.output
for operation in (specs[0].to_task, lambda: run_baseline(specs, None), lambda: leaderboard(None)):
    try:
        operation()
    except ImportError as exc:
        assert str(exc) == 'install demand-forecast[bench]', str(exc)
    else:
        raise AssertionError('missing dependency did not raise ImportError')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
