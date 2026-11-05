"""Offline survey acceptance through real fev 0.10.0 parquet paths."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("fev")

from demand_forecast import __version__  # noqa: E402
from demand_forecast.bench.run import (  # noqa: E402
    assemble_predictions,
    check_panel,
    run_fev,
    settings_for_task,
)
from demand_forecast.bench.specs import (  # noqa: E402
    FEV_DATASETS_REVISION,
    fev_bench_tasks,
    load_task_specs,
)
from demand_forecast.cli import app  # noqa: E402
from demand_forecast.config import Settings  # noqa: E402
from demand_forecast.core.run import run_identity  # noqa: E402
from demand_forecast.core.training import InterruptedRun, NonFiniteTraining  # noqa: E402
from demand_forecast.data.loader_fev import panel_from_window  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402
from tests.helpers.settings import tiny_settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "demand_forecast/bench/tasks"


def multivariate_spec(tmp_path):
    import datasets

    path = write_fev_fixture(tmp_path, freq="h", n_series=3, length=80)
    data = datasets.Dataset.from_parquet(str(path))
    data = data.add_column("temperature", [[-float(i + 1) for i in range(80)]] * 3)
    data.to_parquet(path)
    return fixture_spec(
        path,
        name="fixture/different-targets",
        target=["temperature", "target"],
        target_support={"temperature": "real", "target": "nonnegative"},
    )


def test_selectors_and_frozen_identity():
    specs = load_task_specs("fev100")
    assert len(specs) == 100
    assert [s.name for s in specs] == [s.name for s in fev_bench_tasks()]
    assert [s.name for s in load_task_specs("m5_1W,ETT_1H,uk_covid_utla_1W")] == [
        "m5_1W",
        "ETT_1H",
        "uk_covid_utla_1W",
    ]
    with pytest.raises(ValueError, match="unknown task: absent"):
        load_task_specs("absent")
    frozen = json.loads((Path(__file__).parent / "fixtures/fev_frozen_identities.json").read_text())
    settings = Settings.from_yaml(ROOT / "configs/fev-v2-standard.yaml")
    for spec in load_task_specs("weekly") + load_task_specs("heldout"):
        assert asdict(spec) == frozen[spec.name]["spec"]
        assert spec.task_definition() == frozen[spec.name]["definition"]
        task_settings = settings_for_task(spec, settings, "W-MON")
        identity = run_identity(
            task_settings,
            settings.seed,
            "frozen-test-fingerprint",
            __version__,
            pd.Timestamp("2024-01-01"),
            spec.task_definition(),
        )
        assert identity == frozen[spec.name]["identity"]
        snapshot_name = (
            "uk_covid_utla_1W/cumulative" if spec.name == "uk_covid_utla_1W" else spec.name
        )
        other = next(s for s in specs if s.name == snapshot_name)
        assert asdict(replace(other, name=spec.name)) == asdict(spec)


def test_profiles_match_pinned_schema():
    schema = yaml.safe_load((TASKS / "schema_fev100.yaml").read_text())
    assert schema["revision"] == FEV_DATASETS_REVISION
    # Explicit frozen-protocol exceptions: these numeric identifiers were
    # categorical before the survey. Do not silently rewrite their identities.
    legacy = {
        "rohlik_sales_1W": {"product_unique_id"},
        "rossmann_1W": {"Store", "Promo2"},
        "walmart": {"Store", "Dept"},
        "favorita_stores_1W": {"store_nbr", "cluster"},
        "favorita_transactions_1W": {"store_nbr", "cluster"},
    }
    for spec in load_task_specs("fev100"):
        info = schema["configs"][spec.dataset_config]
        features = info["features"]
        assert (
            set(spec.targets + spec.static_columns + spec.known_dynamic + spec.past_dynamic)
            <= features.keys()
        )
        categorical = {
            c
            for c in spec.static_columns
            if features[c].get("dtype") in ("string", "large_string")
            or "class_label" in features[c]
        }
        assert set(spec.categorical_statics) == categorical | legacy.get(spec.name, set())
        assert info["frequency"]


@pytest.mark.parametrize("freq,horizon,length", [("5min", 288, 700), ("MS", 5, 60), ("YS", 5, 60)])
def test_frequency_horizon_model_and_panel_check(tmp_path, capsys, freq, horizon, length):
    path = write_fev_fixture(tmp_path, freq=freq, length=length, n_series=3)
    spec = fixture_spec(path, horizon=horizon, num_windows=1)
    check_panel(spec)
    output = capsys.readouterr().out
    assert f"PANEL_OK fixture series=3 max_len={length - horizon} horizon={horizon}" in output
    settings = tiny_settings(base_steps=1, d_model=8)
    settings.forecast.max_context = 16
    summary = run_fev([spec], settings, tmp_path / "out")
    assert len(summary) == 1
    assert np.isfinite(summary.SQL.iloc[0])


def test_multitarget_model_roundtrip_and_support(tmp_path, capsys):
    spec = multivariate_spec(tmp_path)
    task = spec.to_task()
    past, future = task.get_window(0).get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)
    assert len(panel.series) == 6
    assert "__fev_target__" in panel.roles.static_categorical
    assert panel.roles.known_dynamic == ("promo",)
    assert "temperature" not in panel.roles.past_dynamic
    # Reverse model output order to exercise explicit identity alignment.
    ids = [s.series_id for s in panel.series]
    arrays = np.arange(6 * 4 * 9).reshape(6, 4, 9)
    predictions = assemble_predictions(spec, past, ids[::-1], arrays[::-1])
    cleaned = task.clean_and_validate_predictions(predictions)
    assert set(cleaned) == {"temperature", "target"}
    assert cleaned["temperature"][0]["predictions"] == arrays[0, :, 4].tolist()
    check_panel(spec)
    assert "targets=2" in capsys.readouterr().out
    settings = tiny_settings(base_steps=2, d_model=8)
    settings.training.refit = "full"
    result = run_fev([spec], settings, tmp_path / "out")
    assert result.num_forecasts.iloc[0] == 12
    assert np.isfinite(result.SQL.iloc[0])
    with np.load(tmp_path / "out" / spec.name / "window_0.npz") as saved:
        assert (saved["arrays"][:3] < 0).any()
        assert (saved["arrays"][3:] >= 0).all()
    rerun = run_fev([spec], settings, tmp_path / "out", resume=True)
    pd.testing.assert_frame_equal(result, rerun)


def cli_files(tmp_path, specs, settings):
    tasks = []
    profiles = {}
    for spec in specs:
        definition = asdict(spec)
        profiles[spec.name] = {
            "target_support": definition.pop("target_support"),
            "categorical_statics": definition.pop("categorical_statics"),
        }
        tasks.append(definition)
    task_file, profile_file, config = [
        tmp_path / name for name in ("tasks.yaml", "profiles.yaml", "config.yaml")
    ]
    task_file.write_text(yaml.safe_dump({"tasks": tasks}))
    profile_file.write_text(yaml.safe_dump(profiles))
    config.write_text(yaml.safe_dump(settings.model_dump(mode="json")))
    return [
        "fev",
        "run",
        "--tasks",
        str(task_file),
        "--profiles",
        str(profile_file),
        "--out-dir",
        str(tmp_path / "out"),
    ], config


def test_keep_going_cli_retry_and_completed_skip(tmp_path, monkeypatch):
    from demand_forecast.core import run

    path = write_fev_fixture(tmp_path)
    specs = [fixture_spec(path, name=n) for n in ("bad", "later")]
    args, config = cli_files(tmp_path, specs, tiny_settings(base_steps=1, d_model=8))
    args += ["--config", str(config), "--keep-going"]
    original = run.train_panel
    calls = []

    def fail_one(*a, **kw):
        name = kw["task_definition"]["name"]
        calls.append(name)
        if name == "bad":
            raise NonFiniteTraining("forced failure")
        return original(*a, **kw)

    monkeypatch.setattr(run, "train_panel", fail_one)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 3, result.output
    assert calls == ["bad", "later"]
    failure_path = tmp_path / "out/failures.json"
    failure = json.loads(failure_path.read_text())["bad"]
    assert failure["exception_type"] == "NonFiniteTraining"
    assert failure["message"] == "forced failure"
    assert failure["utc_time"].endswith("+00:00")
    assert "forced failure" in failure["traceback"][-1]
    assert pd.read_csv(tmp_path / "out/summaries.csv").task.tolist() == ["later"]
    monkeypatch.setattr(run, "train_panel", original)
    result = CliRunner().invoke(app, [*args, "--resume"])
    assert result.exit_code == 0, result.output
    assert "Skipping completed task later" in result.output
    assert json.loads(failure_path.read_text()) == {}
    assert set(pd.read_csv(tmp_path / "out/summaries.csv").task) == {"bad", "later"}


def test_panel_check_cli_pass_and_fail_without_config(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    bad = replace(spec, name="missing", dataset_path=str(tmp_path / "absent.parquet"))
    args, _ = cli_files(tmp_path, [bad, spec], tiny_settings())
    result = CliRunner().invoke(app, [*args, "--panel-check", "--keep-going"])
    assert result.exit_code == 3, result.output
    assert "PANEL_FAIL missing" in result.output
    failure_line = next(
        line for line in result.stdout.splitlines() if line.startswith("PANEL_FAIL")
    )
    assert "datasets.load_dataset" in failure_line and "arguments" in failure_line
    assert str(tmp_path / "absent.parquet") in failure_line
    assert "PANEL_OK fixture" in result.output
    assert not (tmp_path / "out/summaries.csv").exists()
    assert not list((tmp_path / "out").rglob("*.pt"))
    result = CliRunner().invoke(app, [*args, "--panel-check"])
    assert result.exit_code != 0
    assert "PANEL_OK" not in result.output


@pytest.mark.parametrize("error", [KeyboardInterrupt(), InterruptedRun("stop")])
def test_keep_going_does_not_swallow_interrupt(tmp_path, monkeypatch, error):
    from demand_forecast.bench import run

    def interrupt(spec):
        raise error

    monkeypatch.setattr(run, "check_panel", interrupt)
    with pytest.raises(type(error)):
        run_fev(
            [fixture_spec(tmp_path / "unused")],
            tiny_settings(),
            tmp_path / "out",
            keep_going=True,
            panel_check=True,
        )
    assert not (tmp_path / "out/panel_failures.json").exists()


def test_survey_recipe_unchanged():
    assert yaml.safe_load((ROOT / "configs/fev-v2-survey.yaml").read_text()) == yaml.safe_load(
        (ROOT / "configs/ablations/2026-11-04/refit-weekly.yaml").read_text()
    )


def test_terminal_nonfinite_task_is_restarted_on_keep_going_resume(tmp_path, monkeypatch):
    from demand_forecast.core import training

    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=20, d_model=8)
    settings.training.patience = 100
    original = training.pinball_loss
    monkeypatch.setattr(
        training, "pinball_loss", lambda pred, *args: (pred.sum() * float("nan"), 1)
    )
    result = run_fev([spec], settings, tmp_path / "out", keep_going=True)
    assert result.attrs["failed_tasks"] == [spec.name]
    assert (tmp_path / "out/fixture/checkpoints/latest.pt").exists()
    monkeypatch.setattr(training, "pinball_loss", original)
    result = run_fev([spec], settings, tmp_path / "out", keep_going=True, resume=True)
    assert result.attrs["failed_tasks"] == []
    assert len(result) == 1
    assert json.loads((tmp_path / "out/failures.json").read_text()) == {}


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("CUDA out of memory"),
        NotImplementedError("unsupported property"),
        ValueError("invalid data"),
    ],
)
def test_keep_going_general_errors(tmp_path, monkeypatch, exception):
    from demand_forecast.bench import run

    seen = []

    def check(spec):
        seen.append(spec.name)
        if spec.name == "bad":
            raise exception

    monkeypatch.setattr(run, "check_panel", check)
    specs = [fixture_spec(tmp_path / "unused", name=name) for name in ["bad", "good"]]
    result = run_fev(specs, tiny_settings(), tmp_path / "out", keep_going=True, panel_check=True)
    assert seen == ["bad", "good"]
    assert result.attrs["failed_tasks"] == ["bad"]


def test_all_snapshot_frequencies_and_horizons_build_bounded_batches():
    from demand_forecast.data.encoders import StaticEncoder
    from demand_forecast.data.frequency import calendar_features, grid_range, seasonal_period
    from demand_forecast.data.panel import Panel, Roles, Series
    from demand_forecast.data.sampler import make_batch

    metadata = yaml.safe_load((TASKS / "schema_fev100.yaml").read_text())["configs"]
    for spec in load_task_specs("fev100"):
        freq = metadata[spec.dataset_config]["frequency"]
        grid = grid_range(pd.Timestamp("2000-01-01"), 25, freq)
        assert np.isfinite(calendar_features(grid, freq)).all()
        series = Series("a", grid[0], 20, np.arange(20, dtype=float), {}, {})
        panel = Panel(
            [series], Roles("target", (), (), (), ()), freq, seasonal_period(freq), "real"
        )
        encoder = StaticEncoder().fit(panel, ["a"])
        batch = make_batch(panel, [(0, 20)], spec.horizon, 7, encoder)
        assert batch.target.shape == (1, 7)
        assert batch.dec_cov.shape[1] == spec.horizon


def test_transfer_to_hourly_multitarget_and_long_horizon(tmp_path, pretrained_univariate_bundle):
    import torch

    from demand_forecast.config.settings import ForecastConfig
    from demand_forecast.core.transfer import init_from_pretrained
    from demand_forecast.data.encoders import StaticEncoder
    from demand_forecast.data.sampler import make_batch
    from demand_forecast.models.base import ModelSpec
    from tests.helpers.models import build_v2

    task_spec = multivariate_spec(tmp_path)
    task = task_spec.to_task()
    past, future = task.get_window(0).get_input_data()
    panel = panel_from_window(past, future, task_spec, task.freq)
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    model_spec = ModelSpec.from_panel(panel, encoder, ForecastConfig(horizon=288), 16, 4)
    source = pretrained_univariate_bundle
    model = build_v2(model_spec, **source.backbone_kwargs)
    report = init_from_pretrained(model, source)
    assert all(
        torch.equal(value, source.model.backbone.state_dict()[key])
        for key, value in model.backbone.state_dict().items()
    )
    assert report.fresh and all(
        name.startswith(("statics.", "static_gate.")) for name in report.fresh
    )
    batch = make_batch(panel, [(0, 20), (3, 20)], 288, 7, encoder)
    with torch.no_grad():
        output = model.eval().predict(batch)
    assert output.shape == (2, 288, 9)
    assert torch.isfinite(output).all()
    assert (output[1] >= 0).all()


def test_resume_validates_completed_prediction_payload(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path), num_windows=1)
    settings = tiny_settings(base_steps=1, d_model=8)
    out = tmp_path / "out"
    run_fev([spec], settings, out)
    path = out / "fixture/window_0.npz"
    with np.load(path) as saved:
        corrupted = dict(saved)
    corrupted["arrays"][0, 0, 0] = np.nan
    np.savez(path, **corrupted)
    with pytest.raises(ValueError, match="invalid value|NaN|finite"):
        run_fev([spec], settings, out, resume=True)


@pytest.mark.parametrize(
    "support", [{"target": "real"}, {"temperature": "unknown", "target": "real"}]
)
def test_support_mapping_requires_exact_valid_targets(tmp_path, support):
    with pytest.raises(ValueError, match="target_support"):
        fixture_spec(tmp_path / "unused", target=["target", "temperature"], target_support=support)


def test_multitarget_identity_includes_target_selection(tmp_path):
    base = fixture_spec(tmp_path / "unused", target=["a", "b"], target_support="real")
    changed = replace(base, target=["a", "c"])
    settings = tiny_settings()
    identities = [
        run_identity(
            settings_for_task(spec, settings, "h"),
            0,
            "same-dataset",
            __version__,
            pd.Timestamp("2024-01-01"),
            spec.task_definition(),
        )
        for spec in [base, changed]
    ]
    assert identities[0] != identities[1]
