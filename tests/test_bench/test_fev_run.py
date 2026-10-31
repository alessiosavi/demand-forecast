import json
import shutil
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from typer.testing import CliRunner

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import (  # noqa: E402
    predictions_for_fev,
    run_fev,
    settings_for_task,
    window_origin,
)
from demand_forecast.bench.specs import dataset_fingerprint  # noqa: E402
from demand_forecast.cli import app  # noqa: E402
from demand_forecast.core.training import InterruptedRun  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402
from tests.helpers.settings import tiny_settings  # noqa: E402

Q9 = [i / 10 for i in range(1, 10)]
ARR = np.arange(2 * 3 * 9, dtype=float).reshape(2, 3, 9)


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

    def capture(predictor, panel, origin):
        seen.append((origin, [series.history_length for series in panel.series]))
        return forecast(predictor, panel, origin)

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


def test_interrupted_training_before_first_bundle_resumes_and_changed_config_is_rejected(
    tmp_path,
):
    spec = fixture_spec(write_fev_fixture(tmp_path))
    out = tmp_path / "out"
    settings = tiny_settings(base_steps=40, d_model=16)
    with pytest.raises(InterruptedRun):
        run_fev([spec], settings, out, stop_after=15)
    assert not (out / spec.name / "bundle").exists()
    assert torch.load(out / spec.name / "checkpoints/latest.pt", weights_only=False)["step"] == 15
    full = run_fev([spec], settings, tmp_path / "ref")
    resumed = run_fev([spec], settings, out, resume=True)
    assert resumed.SQL.iloc[0] == pytest.approx(full.SQL.iloc[0])
    manifest = json.loads((out / spec.name / "bundle/manifest.json").read_text())
    assert manifest["resumed_from_step"] == 15
    for index in range(spec.num_windows):
        with (
            np.load(tmp_path / "ref" / spec.name / f"window_{index}.npz") as expected,
            np.load(out / spec.name / f"window_{index}.npz") as actual,
        ):
            np.testing.assert_array_equal(actual["arrays"], expected["arrays"])
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


def test_window_origin_requires_one_shared_origin_and_accepts_custom_column():
    rows = [{"timestamp": ["2023-01-01"]}, {"timestamp": ["2023-01-01"]}]
    assert window_origin(rows) == pd.Timestamp("2023-01-01")
    assert window_origin([{"date": ["2023-01-01"]}], "date") == pd.Timestamp("2023-01-01")
    rows[1]["timestamp"] = ["2023-01-08"]
    with pytest.raises(ValueError, match="windows with item-specific origins are not supported"):
        window_origin(rows)


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

    def capture(predictor, panel, origin):
        encoder = predictor.bundle.encoder
        encoders.append(encoder.to_json())
        codes.append({s.series_id: encoder.transform(s)[0]["__series__"] for s in panel.series})
        ids, arrays, diagnostics = forecast(predictor, panel, origin)
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


def test_cli_run_interrupt_resume_policy_and_missing_bench(tmp_path, monkeypatch):
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
    tiny_settings(base_steps=10, d_model=8).to_yaml(config)
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
    resumed = runner.invoke(app, [*args, "--resume"])
    assert resumed.exit_code == 0, (resumed.output, resumed.exception)
    summary = pd.read_csv(out / "summaries.csv")
    assert summary.model_name.tolist() == ["cli-model"] and np.isfinite(summary.SQL.iloc[0])
    unimplemented = runner.invoke(app, [*args, "--policy", "per_window"])
    assert unimplemented.exit_code == 2
    assert "per_window arrives with the phase-4 plan" in unimplemented.output
    monkeypatch.setitem(sys.modules, "fev", None)
    unavailable = runner.invoke(app, args)
    assert unavailable.exit_code == 2 and "install demand-forecast[bench]" in unavailable.output


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
            },
        )
    assert list(tmp_path.iterdir()) == []
