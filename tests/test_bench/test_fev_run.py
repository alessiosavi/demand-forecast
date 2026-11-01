import json
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from typer.testing import CliRunner

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast import __version__  # noqa: E402
from demand_forecast.bench.run import (  # noqa: E402
    prediction_diagnostics,
    predictions_for_fev,
    resolved_origin,
    run_baseline,
    run_fev,
    settings_for_task,
    window_origins,
)
from demand_forecast.bench.specs import dataset_fingerprint  # noqa: E402
from demand_forecast.cli import app  # noqa: E402
from demand_forecast.config import Settings  # noqa: E402
from demand_forecast.core.run import run_identity  # noqa: E402
from demand_forecast.core.training import InterruptedRun, Trainer  # noqa: E402
from demand_forecast.core.validation import plan_inner_validation  # noqa: E402
from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.data.loader_fev import item_ids, panel_from_window  # noqa: E402
from demand_forecast.data.panel import Panel  # noqa: E402
from demand_forecast.inference.predictor import Predictor  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402
from tests.helpers.reference_sampler import assert_batch_matches_reference  # noqa: E402
from tests.helpers.settings import tiny_settings  # noqa: E402

Q9 = [i / 10 for i in range(1, 10)]
ARR = np.arange(2 * 3 * 9, dtype=float).reshape(2, 3, 9)
RAGGED = (40, 38, 40, 28, 38, 40)


def file_states(directory: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_predictions_for_fev_reorders_and_fails_on_missing():
    out = predictions_for_fev(["b", "a"], ARR, ["a", "b"], Q9)
    assert out[0]["predictions"] == ARR[1, :, 4].tolist()
    assert set(out[0]) == {"predictions", *map(str, Q9)}
    for index, q in enumerate(Q9):
        assert out[0][str(q)] == ARR[1, :, index].tolist()
        assert out[1][str(q)] == ARR[0, :, index].tolist()
    with pytest.raises(ValueError, match="missing forecast"):
        predictions_for_fev(["b", "a"], ARR, ["a", "b", "c"], Q9)


def test_offline_fev_task_end_to_end_with_resume_and_replacement(tmp_path, monkeypatch):
    from demand_forecast.core import run
    from demand_forecast.inference.predictor import Predictor

    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=20, d_model=16)
    out = tmp_path / "out"
    seen = []
    forecast = Predictor.forecast_with_diagnostics

    def capture(predictor, panel, origin, batch_size=512):
        seen.append((origin, [series.history_length for series in panel.series]))
        return forecast(predictor, panel, origin, batch_size)

    monkeypatch.setattr(Predictor, "forecast_with_diagnostics", capture)
    first = run_fev([spec], settings, out)
    assert np.isfinite(first.SQL.iloc[0])
    assert first.num_forecasts.iloc[0] == 12
    assert bool(first.trained_on_this_dataset.iloc[0]) is False
    assert first.model_name.iloc[0] == "demand-forecast-v2"
    assert first.training_time_s.iloc[0] > 0 and first.inference_time_s.iloc[0] > 0
    assert seen[0][0] < seen[1][0]
    assert seen[0][1] == [32] * 6 and seen[1][1] == [36] * 6
    original_second = seen[1]
    task_dir = out / spec.name
    manifest = json.loads((task_dir / "bundle/manifest.json").read_text())
    with np.load(task_dir / "window_1.npz", allow_pickle=False) as saved:
        expected = saved["arrays"].copy()
        assert saved["identity"].item() == manifest["run_identity"]
        assert saved["ids"].tolist() == [str(i) for i in range(6)]
    (task_dir / "window_1.npz").unlink()
    unchanged = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in task_dir.rglob("*")
        if path.is_file()
    }

    def no_training(*args, **kwargs):
        pytest.fail("resume with a complete bundle must not build or train a model")

    monkeypatch.setattr(run, "build_model", no_training)
    seen.clear()
    second = run_fev([spec], settings, out, resume=True)
    assert second.SQL.iloc[0] == pytest.approx(first.SQL.iloc[0])
    assert len(pd.read_csv(out / "summaries.csv")) == 1
    assert seen == [original_second]
    with np.load(task_dir / "window_1.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["arrays"], expected)
    assert all(
        (path.read_bytes(), path.stat().st_mtime_ns) == state for path, state in unchanged.items()
    )
    seen.clear()
    run_fev([spec], settings, out, resume=True)
    assert seen == []


# Historical 67d4bd7 maps to bdb8c119f87750066026ddce48121d61aa0fdb8c in rewritten history.
@pytest.mark.parametrize(
    "config,expected",
    [
        ("tiny", "0109546d94a7e19ca25a3d8e427e46af981b5f04ee5a5ca772cc610d0397e03a"),
        ("frozen", "576741a436a27e53d0643128fc79169f9ccf0a1906c5141eed60c71f7c0a0250"),
    ],
)
def test_aligned_fixture_identity_pinned_at_67d4bd7(tmp_path, config, expected):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = (
        tiny_settings(base_steps=20, d_model=16)
        if config == "tiny"
        else Settings.from_yaml(
            Path(__file__).resolve().parents[2] / "configs/fev-v2-standard.yaml"
        )
    )
    task = spec.to_task()
    fingerprint = dataset_fingerprint(task)
    _, future = task.get_window(0).get_input_data()
    origin = resolved_origin(window_origins(future, spec.id_column, spec.timestamp_column))
    assert (
        run_identity(
            settings_for_task(spec, settings, task.freq),
            settings.seed,
            fingerprint,
            __version__,
            origin,
            spec.task_definition(),
        )
        == expected
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        ("tiny", "0109546d94a7e19ca25a3d8e427e46af981b5f04ee5a5ca772cc610d0397e03a"),
        ("frozen", "576741a436a27e53d0643128fc79169f9ccf0a1906c5141eed60c71f7c0a0250"),
    ],
)
def test_identity_unchanged_with_new_fields_unset(tmp_path, config, expected):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = (
        tiny_settings(base_steps=20, d_model=16)
        if config == "tiny"
        else Settings.from_yaml(
            Path(__file__).resolve().parents[2] / "configs/fev-v2-standard.yaml"
        )
    )
    settings.training = type(settings.training).model_validate(
        {
            **settings.training.model_dump(),
            "init_from": None,
            "zero_shot": False,
            "eval_at_start": False,
        }
    )
    task = spec.to_task()
    _, future = task.get_window(0).get_input_data()
    origin = resolved_origin(window_origins(future, spec.id_column, spec.timestamp_column))
    assert (
        run_identity(
            settings_for_task(spec, settings, task.freq),
            settings.seed,
            dataset_fingerprint(task),
            __version__,
            origin,
            spec.task_definition(),
        )
        == expected
    )


@pytest.mark.parametrize("quick", [False, True])
def test_interrupted_training_before_first_bundle_resumes_and_changed_config_is_rejected(
    tmp_path,
    monkeypatch,
    capsys,
    quick,
):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    out = tmp_path / "out"
    settings = tiny_settings(base_steps=40, d_model=16)
    if quick:
        settings.training.max_steps_cap = 21
        settings.training.val_items_max = 2
    seen = []
    fit = Trainer.fit

    def capture(trainer, panel, plan, *args, **kwargs):
        seen.append(plan.val_items.tolist())
        return fit(trainer, panel, plan, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", capture)
    with pytest.raises(InterruptedRun):
        run_fev([spec], settings, out, stop_after=15)
    assert not (out / spec.name / "bundle").exists()
    assert torch.load(out / spec.name / "checkpoints/latest.pt", weights_only=False)["step"] == 15
    interrupted_progress = json.loads((out / spec.name / "checkpoints/progress.json").read_text())
    if quick:
        assert interrupted_progress["max_steps"] == 21
        assert interrupted_progress["n_val_items"] == 2
    full = run_fev([spec], settings, tmp_path / "ref")
    resumed = run_fev([spec], settings, out, resume=True)
    assert resumed.SQL.iloc[0] == pytest.approx(full.SQL.iloc[0])
    manifest = json.loads((out / spec.name / "bundle/manifest.json").read_text())
    assert manifest["resumed_from_step"] == 15
    assert len(seen) == 3 and seen[0] == seen[1] == seen[2]
    report = json.loads((out / spec.name / "run.json").read_text())
    reference = json.loads((tmp_path / "ref" / spec.name / "run.json").read_text())
    assert report["plan"]["val_items"] == reference["plan"]["val_items"] == seen[0]
    assert report["fit_result"] == reference["fit_result"]
    if quick:
        assert report["val_items_max"] == report["n_val_items"] == 2
        assert report["n_val_items_eligible"] == 6
        assert report["resolved_budget"] == {"max_steps": 21, "eval_every": 4}
        assert report["fit_result"]["steps"] <= 21
        assert seen[0] == sorted(
            np.random.default_rng(settings.seed).choice(6, size=2, replace=False).tolist()
        )
    else:
        assert report["val_items_max"] is None
        assert report["n_val_items"] == report["n_val_items_eligible"] == 6
    for index in range(spec.num_windows):
        with (
            np.load(tmp_path / "ref" / spec.name / f"window_{index}.npz") as expected,
            np.load(out / spec.name / f"window_{index}.npz") as actual,
        ):
            np.testing.assert_array_equal(actual["arrays"], expected["arrays"])
    before = file_states(out)
    run_fev([spec], settings, out, resume=True)
    assert f"Skipping completed task {spec.name}" in capsys.readouterr().out
    assert file_states(out) == before
    assert len(seen) == 3
    for field in ("max_steps_cap", "val_items_max"):
        changed = settings.model_copy(deep=True)
        setattr(changed.training, field, (getattr(settings.training, field) or 1) + 1)
        with pytest.raises(ValueError, match="run identity mismatch"):
            run_fev([spec], changed, out, resume=True)
    with pytest.raises(ValueError, match="identity"):
        run_fev([spec], tiny_settings(base_steps=40, d_model=8), out, resume=True)
    with pytest.raises(ValueError, match="identity"):
        run_fev([replace(spec, num_windows=1)], settings, out, resume=True)


def test_per_window_is_explicitly_unimplemented(tmp_path):
    with pytest.raises(NotImplementedError, match="per_window arrives with the phase-4 plan"):
        run_fev(
            [fixture_spec(write_fev_fixture(tmp_path))],
            tiny_settings(),
            tmp_path / "o",
            policy="per_window",
        )


def test_settings_for_task_copies_base_and_all_roles(tmp_path):
    base = tiny_settings()
    before = base.model_dump()
    spec = fixture_spec(
        tmp_path / "unused.parquet",
        id_column="item",
        timestamp_column="date",
        target="sales",
        target_support="real",
        static_columns=["size", "cat"],
        horizon=3,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    settings = settings_for_task(spec, base, "D")
    assert settings.data.model_dump() == {
        "source": "fev",
        "path": None,
        "id_columns": ["item"],
        "timestamp_column": "date",
        "target": "sales",
        "target_support": "real",
        "known_dynamic": ["promo"],
        "past_dynamic": ["oil"],
        "static_path": None,
        "static_categorical": ["cat"],
        "static_numeric": ["size"],
        "freq": "D",
        "missing_target": "nan",
        "duplicates": "error",
        "resample": None,
    }
    assert settings.forecast.horizon == 3
    assert settings.forecast.quantiles == [0.1, 0.5, 0.9]
    assert settings.forecast.max_context == base.forecast.max_context
    assert settings.model_dump(exclude={"data", "forecast"}) == base.model_dump(
        exclude={"data", "forecast"}
    )
    settings.model.d_model = 8
    settings.forecast.quantiles.append(0.95)
    assert base.model_dump() == before
    assert spec.quantile_levels == [0.1, 0.5, 0.9]


def test_window_origins_are_per_item_and_resolved_origin_is_latest():
    rows = [
        {"item_id": "a", "timestamp": ["2023-01-01", "2023-01-08"]},
        {"item_id": "b", "timestamp": ["2022-12-25", "2023-01-01"]},
    ]
    origins = window_origins(rows, "item_id")
    assert origins == {"a": pd.Timestamp("2023-01-01"), "b": pd.Timestamp("2022-12-25")}
    assert resolved_origin(origins) == pd.Timestamp("2023-01-01")
    assert window_origins([{"id": 1, "date": ["2023-01-01"]}], "id", "date") == {
        "1": pd.Timestamp("2023-01-01")
    }
    with pytest.raises(ValueError, match="^empty future rows for item a$"):
        window_origins([{"item_id": "a", "timestamp": []}], "item_id")
    with pytest.raises(ValueError, match="^window has no items$"):
        resolved_origin({})
    with pytest.raises(ValueError, match="^duplicate future series id$"):
        window_origins([rows[0], rows[0]], "item_id")


def test_ragged_ends_forecast_each_item_at_its_own_origin(tmp_path, capsys, caplog):
    spec = fixture_spec(write_fev_fixture(tmp_path, lengths=RAGGED), num_windows=1)
    settings = tiny_settings(base_steps=20, d_model=16)
    out = tmp_path / "run"
    with caplog.at_level("INFO", logger="demand_forecast.bench.run"):
        summaries = run_fev([spec], settings, out)
    task = spec.to_task()
    past, future = task.get_window(0).get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)
    origins = window_origins(future, spec.id_column, spec.timestamp_column)
    latest, earliest = max(origins.values()), min(origins.values())
    assert len(set(origins.values())) == 3
    assert (
        f"origins: task=fixture window=0 distinct=3 earliest={earliest} latest={latest}"
        in caplog.text
    )
    plan = plan_inner_validation(panel, latest, spec.horizon, settings.training.min_train_history)
    assert sorted(panel.series[i].series_id for i in plan.val_items) == ["0", "1", "2", "4", "5"]
    assert earliest < plan.train_upto
    report = json.loads((out / spec.name / "run.json").read_text())
    assert pd.Timestamp(report["origin"]) == latest
    assert pd.Timestamp(report["earliest_origin"]) == earliest
    assert report["n_val_items"] == 5 < len(panel)
    assert report["fit_result"]["stop_reason"] == "max_steps"
    with np.load(out / spec.name / "window_0.npz", allow_pickle=False) as saved:
        ids, arrays = saved["ids"].tolist(), saved["arrays"]
    assert ids == item_ids(panel)
    predictor = Predictor.from_bundle(out / spec.name / "bundle")
    for origin in sorted(set(origins.values())):
        members = [key for key in ids if origins[key] == origin]
        group = Panel(
            [panel.series[panel.index_of(key)] for key in members],
            panel.roles,
            panel.freq,
            panel.seasonal_period,
            panel.target_support,
        )
        gids, expected, _ = predictor.forecast_with_diagnostics(group, origin)
        assert gids == members
        np.testing.assert_array_equal(arrays[[ids.index(key) for key in members]], expected)
    single = replace(panel, series=[panel.series[panel.index_of("3")]])
    _, alone, _ = predictor.forecast_with_diagnostics(single, origins["3"])
    np.testing.assert_allclose(arrays[ids.index("3")], alone[0], rtol=0, atol=1e-5)
    assert summaries.loc[0, "num_forecasts"] == len(RAGGED)
    assert summaries.loc[0, "max_distinct_origins"] == 3
    before = file_states(out)
    run_fev([spec], settings, out, resume=True)
    assert f"Skipping completed task {spec.name}" in capsys.readouterr().out
    assert file_states(out) == before


def test_ragged_run_passes_fev_report(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path, lengths=RAGGED), num_windows=1)
    out = tmp_path / "run"
    run_baseline([spec], out)
    run_fev([spec], tiny_settings(base_steps=20, d_model=16), out)
    definition = asdict(spec)
    profiles = {
        spec.name: {key: definition.pop(key) for key in ("target_support", "categorical_statics")}
    }
    tasks_path, profiles_path = tmp_path / "tasks.yaml", tmp_path / "profiles.yaml"
    tasks_path.write_text(yaml.safe_dump({"tasks": [definition]}))
    profiles_path.write_text(yaml.safe_dump(profiles))
    report_path = tmp_path / "report.md"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts/fev_report.py"),
            "--run-dir",
            str(out),
            "--tasks",
            str(tasks_path),
            "--profiles",
            str(profiles_path),
            "--baseline",
            "seasonal_naive",
            "--out",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Complete" in report_path.read_text()


def test_resume_accepts_legacy_run_directory_without_new_keys(tmp_path, capsys):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=20, d_model=16)
    out = tmp_path / "run"
    run_fev([spec], settings, out)
    report_path = out / spec.name / "run.json"
    report = json.loads(report_path.read_text())
    del report["earliest_origin"]
    report_path.write_text(json.dumps(report))
    manifest_path = out / spec.name / "bundle/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["run"]["earliest_origin"]
    manifest_path.write_text(json.dumps(manifest))
    summary_path = out / "summaries.csv"
    summaries = pd.read_csv(summary_path).drop(columns="max_distinct_origins")
    if "extra_info" in summaries:

        def strip_origins(value):
            extra_info = json.loads(value)
            extra_info.pop("max_distinct_origins", None)
            return json.dumps(extra_info)

        summaries["extra_info"] = summaries["extra_info"].map(strip_origins)
    summaries.to_csv(summary_path, index=False)
    before = file_states(out)
    legacy_row = pd.read_csv(summary_path, float_precision="round_trip")
    run_fev([spec], settings, out, resume=True)
    assert f"Skipping completed task {spec.name}" in capsys.readouterr().out
    assert file_states(out) == before

    new_spec = fixture_spec(write_fev_fixture(tmp_path / "new"), name="new")
    mixed = run_fev([spec, new_spec], settings, out, resume=True)
    assert f"Skipping completed task {spec.name}" in capsys.readouterr().out
    assert file_states(out / spec.name) == {
        path: state for path, state in before.items() if path.is_relative_to(out / spec.name)
    }
    retained = mixed.loc[mixed.task == spec.name].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        retained.drop(columns="max_distinct_origins"), legacy_row, check_exact=True
    )
    assert retained.max_distinct_origins.isna().all()
    assert mixed.loc[mixed.task == new_spec.name, "max_distinct_origins"].tolist() == [1]


def test_fingerprint_loads_dataset_and_task_definition_is_complete(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    task = spec.to_task()
    assert task._dataset_fingerprint is None
    fingerprint = dataset_fingerprint(task)
    assert fingerprint and fingerprint == task._dataset_fingerprint
    assert dataset_fingerprint(task) == fingerprint
    assert spec.task_definition() == {
        "name": spec.name,
        "horizon": 4,
        "num_windows": 2,
        "initial_cutoff": None,
        "window_step_size": None,
        "min_context_length": 1,
        "seasonality": 1,
        "quantile_levels": Q9,
    }


@pytest.mark.parametrize(
    "ids,arrays,expected_ids,message",
    [
        (["a", "a"], ARR, ["a", "b"], "duplicate"),
        (["a", "b"], ARR, ["a", "a"], "duplicate"),
        (["a", "b"], ARR, ["a"], "unexpected forecast"),
        (["a"], ARR, ["a"], "shape"),
        (["a", "b"], ARR[:, :, :3], ["a", "b"], "shape"),
    ],
)
def test_prediction_adapter_rejects_ambiguous_coverage(ids, arrays, expected_ids, message):
    with pytest.raises(ValueError, match=message):
        predictions_for_fev(ids, arrays, expected_ids, Q9)


@pytest.mark.parametrize(
    "artifact", ["checkpoint", "bundle", "window", "window_missing_identity", "orphan_window"]
)
def test_resume_rejects_each_stale_artifact_before_recomputation(tmp_path, monkeypatch, artifact):
    from demand_forecast.core import run
    from demand_forecast.inference.predictor import Predictor

    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=2, d_model=8)
    out = tmp_path / "out"
    run_fev([spec], settings, out)
    task_dir = out / spec.name
    (task_dir / "window_0.npz").unlink()
    if artifact == "checkpoint":
        path = task_dir / "checkpoints/latest.pt"
        state = torch.load(path, weights_only=False)
        state["run_identity"] = "wrong"
        torch.save(state, path)
    elif artifact == "bundle":
        path = task_dir / "bundle/manifest.json"
        state = json.loads(path.read_text())
        state["run_identity"] = "wrong"
        path.write_text(json.dumps(state))
    else:
        path = task_dir / "window_1.npz"
        with np.load(path) as saved:
            state = {key: saved[key] for key in saved.files}
        if artifact == "window_missing_identity":
            del state["identity"]
        else:
            state["identity"] = "wrong"
        np.savez(path, **state)
        if artifact == "orphan_window":
            shutil.rmtree(task_dir / "bundle")
            shutil.rmtree(task_dir / "checkpoints")
            (task_dir / "run.json").unlink()

    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("identity checks must happen before training or inference")

    monkeypatch.setattr(run, "build_model", forbidden)
    monkeypatch.setattr(Predictor, "forecast_with_diagnostics", forbidden)
    with pytest.raises(ValueError, match="^run identity mismatch$"):
        run_fev([spec], settings, out, resume=True)
    assert before == {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}


def test_summaries_replace_only_matching_task_and_model_and_survive_later_failure(
    tmp_path, monkeypatch
):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=2, d_model=8)
    other = replace(spec, name="other")
    out = tmp_path / "out"
    to_task = type(spec).to_task

    def fail_second(current):
        if current.name == "other":
            raise RuntimeError("second task failed")
        return to_task(current)

    monkeypatch.setattr(type(spec), "to_task", fail_second)
    with pytest.raises(RuntimeError, match="second task failed"):
        run_fev([spec, other], settings, out, model_name="custom")
    saved = pd.read_csv(out / "summaries.csv")
    assert saved.task.tolist() == [spec.name]
    expected_sql = saved.SQL.iloc[0]
    extra = pd.concat([saved.assign(task="other"), saved.assign(model_name="other")])
    pd.concat([saved.assign(SQL=-99), saved.assign(SQL=-98), extra]).to_csv(
        out / "summaries.csv", index=False
    )
    result = run_fev([spec], settings, out, resume=True, model_name="custom")
    assert len(result) == 3
    row = result.loc[(result.task == spec.name) & (result.model_name == "custom")]
    assert len(row) == 1 and row.SQL.iloc[0] == pytest.approx(expected_sql)
    pd.testing.assert_frame_equal(
        result.iloc[:2].reset_index(drop=True), extra.reset_index(drop=True)
    )


def test_later_window_unseen_id_uses_unchanged_encoder_with_custom_columns(tmp_path, monkeypatch):
    from datasets import Dataset

    from demand_forecast.inference.predictor import Predictor

    parquet = write_fev_fixture(tmp_path)
    dataset = Dataset.from_parquet(str(parquet))

    def shorten_last(row):
        if row["id"] == "5":
            for name in ("timestamp", "target", "promo", "oil"):
                row[name] = row[name][-8:]
        return row

    dataset = dataset.map(shorten_last).rename_columns(
        {"id": "item", "timestamp": "date", "target": "sales"}
    )
    custom = tmp_path / "custom.parquet"
    dataset.to_parquet(custom)
    spec = fixture_spec(custom, id_column="item", timestamp_column="date", target="sales")
    forecast = Predictor.forecast_with_diagnostics
    encoders, codes = [], []

    def capture(predictor, panel, origin, batch_size=512):
        encoder = predictor.bundle.encoder
        encoders.append(encoder.to_json())
        codes.append({s.series_id: encoder.transform(s)[0]["__series__"] for s in panel.series})
        ids, arrays, diagnostics = forecast(predictor, panel, origin, batch_size)
        # Exercise the runner's alignment, with forecasts deliberately out of order.
        return ids[::-1], arrays[::-1], diagnostics

    monkeypatch.setattr(Predictor, "forecast_with_diagnostics", capture)
    result = run_fev([spec], tiny_settings(base_steps=2, d_model=8), tmp_path / "out")
    assert np.isfinite(result.SQL.iloc[0]) and result.num_forecasts.iloc[0] == 11
    assert encoders[0] == encoders[1]
    assert "5" not in codes[0] and codes[1]["5"] == 0
    assert all(codes[0][key] == codes[1][key] for key in codes[0])


def test_cli_run_accepts_custom_tasks_and_profiles(tmp_path):
    from dataclasses import asdict

    from demand_forecast.core.bundle import load_bundle

    spec = fixture_spec(write_fev_fixture(tmp_path))
    definition = asdict(spec)
    del definition["target_support"], definition["categorical_statics"]
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(yaml.safe_dump({"tasks": [definition]}))
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text("fixture:\n  target_support: real\n  categorical_statics: [cat]\n")
    config = tmp_path / "cfg.yaml"
    tiny_settings(base_steps=2, d_model=8).to_yaml(config)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "fev",
            "run",
            "--tasks",
            str(tasks),
            "--profiles",
            str(profiles),
            "--config",
            str(config),
            "--out-dir",
            str(out),
            "--stop-after",
            "2",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    summary = pd.read_csv(out / "summaries.csv")
    assert summary.task.tolist() == ["fixture"] and np.isfinite(summary.SQL.iloc[0])
    bundle = load_bundle(out / "fixture/bundle")
    assert bundle.dataset_cfg.target_support == "real"
    assert bundle.dataset_cfg.static_categorical == ["cat"]
    assert bundle.manifest["run"]["fit_result"]["stop_reason"] == "max_steps"


@pytest.mark.parametrize("quick", [False, True])
def test_cli_run_interrupt_resume_policy_and_missing_bench(tmp_path, monkeypatch, quick):
    from dataclasses import asdict

    spec = fixture_spec(
        write_fev_fixture(tmp_path, statics=()),
        name="hierarchical_sales_1W",
        static_columns=[],
        categorical_statics=[],
    )
    definition = asdict(spec)
    del definition["target_support"], definition["categorical_statics"]
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(yaml.safe_dump({"tasks": [definition]}))
    config = tmp_path / "cfg.yaml"
    settings = tiny_settings(base_steps=10, d_model=8)
    if quick:
        settings.training.max_steps_cap = 7
        settings.training.val_items_max = 2
    settings.to_yaml(config)
    out = tmp_path / "out"
    args = [
        "fev",
        "run",
        "--tasks",
        str(tasks),
        "--config",
        str(config),
        "--policy",
        "first_window",
        "--out-dir",
        str(out),
        "--model-name",
        "cli-model",
    ]
    runner = CliRunner()
    stopped = runner.invoke(app, [*args, "--stop-after", "3"])
    assert stopped.exit_code == 1 and "interrupted at step 3" in stopped.output
    assert not (out / spec.name / "bundle").exists()
    if quick:
        progress = json.loads((out / spec.name / "checkpoints/progress.json").read_text())
        assert progress["max_steps"] == 7 and progress["n_val_items"] == 2
    resumed = runner.invoke(app, [*args, "--resume"])
    assert resumed.exit_code == 0, (resumed.output, resumed.exception)
    summary = pd.read_csv(out / "summaries.csv")
    assert summary.model_name.tolist() == ["cli-model"] and np.isfinite(summary.SQL.iloc[0])
    if quick:
        report = json.loads((out / spec.name / "run.json").read_text())
        assert report["resolved_budget"]["max_steps"] == 7
        assert report["n_val_items"] == report["val_items_max"] == 2
        assert report["n_val_items_eligible"] == 6
    before = file_states(out)
    skipped = runner.invoke(app, [*args, "--resume"])
    assert skipped.exit_code == 0 and f"Skipping completed task {spec.name}" in skipped.output
    assert file_states(out) == before
    unimplemented = runner.invoke(app, [*args, "--policy", "per_window"])
    assert unimplemented.exit_code == 2
    assert "per_window arrives with the phase-4 plan" in unimplemented.output
    monkeypatch.setitem(sys.modules, "fev", None)
    unavailable = runner.invoke(app, args)
    assert unavailable.exit_code == 2 and "install demand-forecast[bench]" in unavailable.output


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
def test_cli_fev_run_exits_1_on_nonfinite_training_and_writes_no_summary_row(
    tmp_path, monkeypatch, entrypoint
):
    from demand_forecast.core import training

    spec = fixture_spec(
        write_fev_fixture(tmp_path, statics=()),
        name="hierarchical_sales_1W",
        static_columns=[],
        categorical_statics=[],
    )
    settings = tiny_settings(base_steps=20, d_model=8)
    settings.training.patience = 100
    out = tmp_path / "out"
    task_dir = out / spec.name
    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    definition = asdict(spec)
    del definition["target_support"], definition["categorical_statics"]
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(yaml.safe_dump({"tasks": [definition]}))
    config = tmp_path / "cfg.yaml"
    settings.to_yaml(config)
    args = [
        "fev",
        "run",
        "--tasks",
        str(tasks),
        "--config",
        str(config),
        "--out-dir",
        str(out),
        "--stop-after",
        "20",
    ]
    expected = (
        "Training stopped after 20 consecutive non-finite losses at step 20; "
        f"checkpoint {task_dir / 'checkpoints/latest.pt'}"
    )

    def invoke(resume):
        if entrypoint == "api":
            with pytest.raises(training.NonFiniteTraining) as caught:
                run_fev([spec], settings, out, resume=resume, stop_after=20)
            assert str(caught.value) == expected
        else:
            result = CliRunner().invoke(app, [*args, *(["--resume"] if resume else [])])
            assert result.exit_code == 1 and expected in result.stderr
            assert (
                "Summaries saved" not in result.stdout and "Skipping completed" not in result.stdout
            )

    with pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"):
        invoke(resume=False)
    assert (task_dir / "checkpoints/latest.pt").exists()

    def forbidden(*args, **kwargs):
        pytest.fail("resume must preserve the failed task without retraining")

    monkeypatch.setattr(training, "pinball_loss", forbidden)
    invoke(resume=True)
    assert not (out / "summaries.csv").exists()
    assert not (task_dir / "bundle").exists() and not (task_dir / "run.json").exists()
    assert not list(task_dir.glob("window_*.npz"))
    progress = json.loads((task_dir / "checkpoints/progress.json").read_text())
    assert progress["step"] == 20 and progress["stop_reason"] == "nonfinite_losses"


def test_fev_resume_rejects_nonfinite_checkpoint_even_with_completed_artifacts(
    tmp_path, monkeypatch, capsys
):
    from demand_forecast.core import training

    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=20, d_model=8)
    settings.training.patience = 100
    out = tmp_path / "out"
    run_fev([spec], settings, out)
    summary = (out / "summaries.csv").read_bytes()
    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    with (
        pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"),
        pytest.raises(training.NonFiniteTraining),
    ):
        run_fev([spec], settings, out)
    capsys.readouterr()

    def forbidden(*args, **kwargs):
        pytest.fail("terminal checkpoint resumed training or inference")

    monkeypatch.setattr(training, "pinball_loss", forbidden)
    monkeypatch.setattr(Predictor, "forecast_with_diagnostics", forbidden)
    with pytest.raises(training.NonFiniteTraining, match="20 consecutive non-finite losses"):
        run_fev([spec], settings, out, resume=True)
    assert "Skipping completed" not in capsys.readouterr().out
    assert (out / "summaries.csv").read_bytes() == summary


@pytest.fixture
def completed_task(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=4, d_model=8)
    out = tmp_path / "out"
    run_fev([spec], settings, out)
    return spec, settings, out


def test_resume_of_completed_task_skips_without_fit(completed_task, monkeypatch, capsys):
    from hashlib import sha256

    from demand_forecast.core.training import Trainer

    spec, settings, out = completed_task
    before = sha256((out / "summaries.csv").read_bytes()).hexdigest()
    monkeypatch.setattr(Trainer, "fit", lambda *a, **k: pytest.fail("unexpected fit"))
    run_fev([spec], settings, out, resume=True)
    assert f"Skipping completed task {spec.name}" in capsys.readouterr().out
    assert sha256((out / "summaries.csv").read_bytes()).hexdigest() == before


def _assert_rows_rebuilt_from_persisted_timings(completed_task, mode, capsys):
    spec, settings, out = completed_task
    path = out / "summaries.csv"
    original = pd.read_csv(path)
    row = original.assign(run_identity="stale", SQL=-99)
    if mode == "duplicate":
        row = pd.concat([original, original.assign(SQL=-98)])
    elif mode == "deleted":
        row = original.iloc[:0]
    row.to_csv(path, index=False)
    rebuilt = run_fev([spec], settings, out, resume=True)
    assert "Skipping completed task" not in capsys.readouterr().out
    assert len(rebuilt) == 1 and rebuilt.SQL.iloc[0] == pytest.approx(original.SQL.iloc[0])
    report = json.loads((out / spec.name / "run.json").read_text())
    assert rebuilt.run_identity.iloc[0] == report["run_identity"]
    timing = report["timings"]["cumulative"]
    assert rebuilt.training_time_s.iloc[0] == sum(
        timing[k] for k in ("train_s", "eval_s", "checkpoint_s")
    )
    durations = []
    for i in range(spec.num_windows):
        with np.load(out / spec.name / f"window_{i}.npz", allow_pickle=False) as data:
            durations.append(float(data["inference_time_s"]))
    assert rebuilt.inference_time_s.iloc[0] == sum(durations)


def test_completed_task_with_stale_identity_row_is_recomputed(completed_task, capsys):
    _assert_rows_rebuilt_from_persisted_timings(completed_task, "stale", capsys)


def test_duplicate_rows_are_replaced_not_skipped(completed_task, capsys):
    _assert_rows_rebuilt_from_persisted_timings(completed_task, "duplicate", capsys)


def test_deleted_row_is_rebuilt_from_persisted_timings(completed_task, capsys):
    _assert_rows_rebuilt_from_persisted_timings(completed_task, "deleted", capsys)


def test_legacy_npz_without_diagnostics_reports_unknown(completed_task, capsys):
    spec, settings, out = completed_task
    path = out / spec.name / "window_1.npz"
    with np.load(path, allow_pickle=False) as data:
        legacy = {key: data[key] for key in ("ids", "arrays", "identity")}
    np.savez(path, **legacy)
    before = path.read_bytes()
    rebuilt = run_fev([spec], settings, out, resume=True)
    assert "Skipping completed task" not in capsys.readouterr().out
    for key in ("inference_time_s", "fallback_no_context", "fallback_undefined_scale"):
        assert pd.isna(rebuilt[key].iloc[0])
    assert path.read_bytes() == before


def test_npz_carries_fallback_counts(tmp_path):
    from datasets import Dataset

    parquet = write_fev_fixture(tmp_path)
    data = Dataset.from_parquet(str(parquet))

    def fallback_rows(row):
        if row["id"] in ("0", "1"):
            row["target"] = [np.nan if row["id"] == "0" else 6.0] * len(row["target"])
        return row

    data.map(fallback_rows).to_parquet(parquet)
    spec = fixture_spec(parquet)
    result = run_fev([spec], tiny_settings(base_steps=2, d_model=8), tmp_path / "out")
    for i in range(spec.num_windows):
        with np.load(tmp_path / "out" / spec.name / f"window_{i}.npz", allow_pickle=False) as data:
            assert data["fallback_no_context"].item() == 1
            assert data["fallback_undefined_scale"].item() >= 1
            for key in (
                "inference_time_s",
                "fallback_no_context",
                "fallback_undefined_scale",
                "device",
            ):
                assert data[key].shape == () and data[key].dtype.kind in "fiuU"
            assert data["inference_time_s"].dtype == np.float64
            assert data["fallback_no_context"].dtype == np.int64
    assert result.fallback_no_context.iloc[0] == 2


def test_extra_info_columns_present_and_baseline_rows_nan(completed_task):
    from demand_forecast.bench.run import run_baseline

    spec, settings, out = completed_task
    result = run_baseline([spec], out)
    keys = [
        "run_identity",
        "fallback_no_context",
        "fallback_undefined_scale",
        "inference_device",
        "resolved_max_steps",
        "n_fit_ids",
        "n_val_items",
        "nonfinite_losses",
        "sessions",
        "cold_covariate_columns",
        "clipped_covariate_values",
        "floored_covariate_columns",
    ]
    model = result.loc[result.model_name == "demand-forecast-v2"].iloc[0]
    baseline = result.loc[result.model_name == "seasonal_naive"].iloc[0]
    assert model[keys].notna().all()
    assert baseline[keys].isna().all()
    assert all(f"SQL[{q}]" in result for q in Q9)


def test_run_fev_uses_settings_device(tmp_path, monkeypatch):
    from demand_forecast.core import run
    from demand_forecast.inference.predictor import Predictor

    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=2, d_model=8)
    seen = []
    load = Predictor.from_bundle
    monkeypatch.setattr(run, "resolve_device", lambda settings: torch.device("cpu:0"))

    def capture(path, device="cpu"):
        seen.append(device)
        return load(path, device="cpu")

    monkeypatch.setattr(Predictor, "from_bundle", capture)
    run_fev([spec], settings, tmp_path / "out")
    assert seen and all(device == torch.device("cpu:0") for device in seen)


@pytest.mark.parametrize("artifact", ["bundle/manifest.json", "window_1.npz"])
def test_completed_task_with_mismatched_artifact_identity_raises(completed_task, artifact):
    spec, settings, out = completed_task
    path = out / spec.name / artifact
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        data["run_identity"] = "foreign"
        path.write_text(json.dumps(data))
    else:
        with np.load(path, allow_pickle=False) as saved:
            data = dict(saved)
        np.savez(path, **{**data, "identity": "foreign"})
    with pytest.raises(ValueError, match="^run identity mismatch$"):
        run_fev([spec], settings, out, resume=True)


def test_npz_publication_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    from demand_forecast.bench.run import _save_predictions

    def fail(stream, **kwargs):
        stream.write(b"partial archive")
        raise OSError("injected failure")

    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="injected failure"):
        _save_predictions(
            tmp_path / "window_0.npz",
            ["a"],
            np.zeros((1, 4, 9)),
            "identity",
            {
                "inference_time_s": 0.1,
                "fallback_no_context": 0,
                "fallback_undefined_scale": 0,
                "device": "cpu",
                "n_items": 1,
                "cold_covariate_columns": 0,
                "clipped_covariate_values": 0,
                "floored_covariate_columns": 0,
            },
        )
    assert list(tmp_path.iterdir()) == []


# Historical 2863356 maps to 93edad09a4334245e11013949a77165830dda58a in rewritten history.
def test_aligned_fixture_batch_is_byte_identical_to_2863356(tmp_path, monkeypatch):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    task = spec.to_task()
    past, future = task.get_window(0).get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)
    panel.roles = replace(panel.roles, known_dynamic=(*panel.roles.known_dynamic, "constant"))
    for series in panel.series:
        series.dynamic["constant"] = np.r_[np.full(32, 100.0), np.full(4, 110.0)]
    encoder = StaticEncoder().fit(panel, item_ids(panel))
    assert_batch_matches_reference(monkeypatch, panel, [(0, 28), (1, 32)], 4, 16, encoder)


def test_item_with_cold_known_covariate_at_its_origin_gets_finite_forecasts(tmp_path):
    from datasets import Dataset, Sequence, Value

    parquet = write_fev_fixture(
        tmp_path, lengths=(44, 42, 44, 40, 42, 44), length=44, known=("promo", "price")
    )
    data = Dataset.from_parquet(str(parquet))
    features = data.features.copy()
    features["promo"] = Sequence(Value("float64"))
    features["price"] = Sequence(Value("float64"))

    def late_column(row):
        row["promo"] = [np.nan] * 36 + [1e5] * (len(row["promo"]) - 36)
        # fev casts dynamics to float32; keep the tiny variation representable.
        row["price"] = [100.0 + 1e-4] + [100.0] * 35 + [110.0] * (len(row["promo"]) - 36)
        return row

    data.map(late_column, features=features).to_parquet(parquet)
    spec = fixture_spec(parquet, known_dynamic=["promo", "price"])
    out = tmp_path / "out"
    result = run_fev([spec], tiny_settings(base_steps=2, d_model=8), out)
    counts = []
    floored_counts = []
    for i in range(spec.num_windows):
        with np.load(out / spec.name / f"window_{i}.npz", allow_pickle=False) as saved:
            assert saved["ids"].tolist() == [str(i) for i in range(6)]
            assert saved["arrays"].shape == (6, 4, 9)
            assert np.isfinite(saved["arrays"]).all()
            assert saved["cold_covariate_columns"].dtype == np.int64
            assert saved["clipped_covariate_values"].dtype == np.int64
            counts.append(saved["cold_covariate_columns"].item())
            assert saved["floored_covariate_columns"].dtype == np.int64
            floored_counts.append(saved["floored_covariate_columns"].item())
    # Origins [36,34,36,32,34,36], then [40,38,40,36,38,40].
    assert counts == [5, 1]
    assert floored_counts == [6, 1]
    assert result.floored_covariate_columns.iloc[0] == 7
    assert result.cold_covariate_columns.iloc[0] == 6
    assert result.clipped_covariate_values.iloc[0] == 0
    assert result.num_forecasts.iloc[0] == 12 and np.isfinite(result.SQL.iloc[0])


@pytest.mark.parametrize(
    "kept_counter", [None, "cold_covariate_columns", "floored_covariate_columns"]
)
def test_legacy_npz_keeps_original_diagnostics_and_sums_present_counters(
    completed_task, kept_counter, monkeypatch
):
    spec, settings, out = completed_task
    original = pd.read_csv(out / "summaries.csv")
    for index in range(spec.num_windows):
        path = out / spec.name / f"window_{index}.npz"
        with np.load(path, allow_pickle=False) as saved:
            legacy = {
                key: saved[key]
                for key in saved.files
                if key
                not in (
                    "cold_covariate_columns",
                    "clipped_covariate_values",
                    "floored_covariate_columns",
                )
            }
        if kept_counter is not None and index == 1:
            legacy[kept_counter] = np.asarray(7, dtype=np.int64)
        np.savez(path, **legacy)
        with np.load(path, allow_pickle=False) as saved:
            diagnostics = prediction_diagnostics(saved)
            assert diagnostics is not None
            for key in (
                "inference_time_s",
                "fallback_no_context",
                "fallback_undefined_scale",
                "device",
            ):
                assert diagnostics[key] == saved[key].item()
            assert diagnostics["clipped_covariate_values"] is None
            for key in ("cold_covariate_columns", "floored_covariate_columns"):
                assert diagnostics[key] == (7 if kept_counter == key and index == 1 else None)
    # Rebuild a missing summary using saved windows, with no inference rerun.
    original.iloc[:0].to_csv(out / "summaries.csv", index=False)
    monkeypatch.setattr(
        Predictor, "forecast_with_diagnostics", lambda *a, **k: pytest.fail("unexpected inference")
    )
    before = file_states(out / spec.name)
    rebuilt = run_fev([spec], settings, out, resume=True)
    for key in ("inference_time_s", "fallback_no_context", "fallback_undefined_scale"):
        assert rebuilt[key].iloc[0] == pytest.approx(original[key].iloc[0])
    assert rebuilt.inference_device.iloc[0] == original.inference_device.iloc[0] == "cpu"
    assert pd.isna(rebuilt.clipped_covariate_values.iloc[0])
    for key in ("cold_covariate_columns", "floored_covariate_columns"):
        if kept_counter == key:
            assert rebuilt[key].iloc[0] == 7
        else:
            assert pd.isna(rebuilt[key].iloc[0])
    assert file_states(out / spec.name) == before


def test_cli_fev_run_zero_shot_tiny_end_to_end(tmp_path, tiny_pretrained_bundle_dir, monkeypatch):
    pytest.importorskip("pyarrow")
    from demand_forecast.core.bundle import load_bundle

    spec = fixture_spec(write_fev_fixture(tmp_path / "data"))
    definition = asdict(spec)
    roles = {
        spec.name: {key: definition.pop(key) for key in ("categorical_statics", "target_support")}
    }
    tasks, profiles = tmp_path / "tasks.yaml", tmp_path / "roles.yaml"
    tasks.write_text(yaml.safe_dump({"tasks": [definition]}))
    profiles.write_text(yaml.safe_dump(roles))
    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((root / "configs/fev-v2-pretrained-zero-shot.yaml").read_text())
    raw["model"] = tiny_settings().model.model_dump(mode="json")
    raw["device"] = "cpu"
    raw["forecast"]["max_context"] = 64
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(raw))
    out = tmp_path / "out"
    monkeypatch.setattr(Trainer, "fit", lambda *a, **k: pytest.fail("zero-shot trained"))
    args = [
        "fev",
        "run",
        "--tasks",
        str(tasks),
        "--profiles",
        str(profiles),
        "--config",
        str(config),
        "--init-from",
        str(tiny_pretrained_bundle_dir),
        "--out-dir",
        str(out),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    frame = pd.read_csv(out / "summaries.csv")
    assert len(frame) == 1 and frame.training_time_s.iloc[0] == 0.0
    assert frame.num_forecasts.iloc[0] == 12 and np.isfinite(frame.SQL.iloc[0])
    bundle = load_bundle(out / spec.name / "bundle")
    report = json.loads((out / spec.name / "run.json").read_text())
    assert bundle.manifest["zero_shot"] is True
    assert report["fit_result"]["stop_reason"] == "zero_shot"
    assert (
        report["transfer"]["pretrain_identity"]
        == load_bundle(tiny_pretrained_bundle_dir).manifest["pretrain_identity"]
    )
    before = file_states(out)
    result = CliRunner().invoke(app, [*args, "--resume"])
    assert result.exit_code == 0, result.output
    assert file_states(out) == before
    # Baseline fixtures add the evidence required by the strict report auditor.
    run_baseline([spec], out)
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/fev_report.py"),
            "--run-dir",
            str(out),
            "--tasks",
            str(tasks),
            "--profiles",
            str(profiles),
            "--out",
            str(out / "report.md"),
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    text = (out / "report.md").read_text()
    assert "zero_shot" in text and report["transfer"]["pretrain_identity"][:8] in text
