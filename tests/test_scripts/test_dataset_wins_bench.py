"""Offline fev integration checks for shifted tasks and saved ensembles."""

import zlib
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import pytest
import yaml

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import assemble_predictions  # noqa: E402
from demand_forecast.bench.specs import load_task_specs  # noqa: E402
from scripts import fev_ensemble_runs, fev_shift_tasks  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


@pytest.mark.parametrize("count,step", [(1, None), (3, 2), (6, 1), (20, 4)])
@pytest.mark.parametrize("cutoff", [-100, None])
def test_shifted_integer_fev_cutoffs_match_official_boundary(tmp_path, count, step, cutoff):
    definition = {
        "dataset_path": "unused/dataset",
        "dataset_config": "m5_1W",
        "static_columns": ["cat_id", "dept_id", "item_id", "state_id", "store_id"],
        "horizon": 4,
        "num_windows": count,
        "window_step_size": step,
    }
    if cutoff is not None:
        definition["initial_cutoff"] = cutoff
    tasks, out = tmp_path / "tasks.yaml", tmp_path / "shifted.yaml"
    tasks.write_text(yaml.safe_dump({"tasks": [definition]}))
    official = load_task_specs(tasks)[0].to_task()
    assert fev_shift_tasks.main(["--tasks", str(tasks), "--names", "m5_1W", "--out", str(out)]) == 0
    shifted = load_task_specs(out)[0].to_task()
    assert shifted.cutoffs[-1] + shifted.horizon == official.cutoffs[0]


@pytest.mark.parametrize(
    "name,cutoff,days,count",
    [
        ("rohlik_sales_1W", "2023-05-05", 56, None),
        ("rohlik_sales_1W", "2023-08-25", 56, 2),
        ("rohlik_sales_1D", "2023-10-20", 14, None),
    ],
)
def test_shifted_timestamp_yaml_constructs_fev_task(tmp_path, name, cutoff, days, count):
    out = tmp_path / "timestamp.yaml"
    args = [
        "--names",
        name,
        "--out",
        str(out),
        "--cutoff",
        f"{name}={cutoff}",
        "--step",
        f"{name}={days}D",
    ]
    if count is not None:
        args.extend(["--windows", f"{name}={count}"])
    assert fev_shift_tasks.main(args) == 0
    task = load_task_specs(out)[0].to_task()
    assert task.cutoffs == [
        (pd.Timestamp(cutoff) + pd.Timedelta(days=days * index)).isoformat()
        for index in range(count or 4)
    ]
    assert pd.Timestamp(task.cutoffs[-1]) + pd.Timedelta(days=days) == pd.Timestamp("2023-12-15")


@pytest.fixture
def local_runs(tmp_path):
    spec = fixture_spec(write_fev_fixture(tmp_path / "data", n_series=2, length=32))
    task = spec.to_task()
    runs, forecasts = [], []
    for seed in range(2):
        directory = tmp_path / f"seed{seed}"
        (directory / spec.name).mkdir(parents=True)
        predictions, arrays_per_window = [], []
        for index, window in enumerate(task.iter_windows()):
            past, _ = window.get_input_data()
            ids = [row["id"] for row in past]
            arrays = np.broadcast_to(
                np.arange(9, dtype=np.float32) + 8 + seed * 2 + index,
                (len(ids), 4, 9),
            ).copy()
            np.savez(
                directory / spec.name / f"window_{index}.npz",
                ids=np.asarray(ids),
                arrays=arrays,
                identity=f"seed{seed}",
            )
            predictions.append(assemble_predictions(spec, past, ids, arrays))
            arrays_per_window.append(arrays)
        # Independent oracle: call fev itself, as the runner did before extraction.
        summary = task.evaluation_summary(
            predictions,
            model_name=f"seed{seed}",
            trained_on_this_dataset=False,
            per_quantile_scores=True,
        )
        pd.DataFrame([{"task": spec.name, **summary}]).to_csv(
            directory / "summaries.csv", index=False
        )
        runs.append(directory)
        forecasts.append(arrays_per_window)
    return spec, runs, forecasts


def test_one_run_ensemble_is_exact(local_runs, tmp_path):
    spec, runs, forecasts = local_runs
    out = tmp_path / "one"
    assert (
        fev_ensemble_runs.main(["--runs", str(runs[0]), "--out", str(out), "--model-name", "one"])
        == 0
    )
    original = pd.read_csv(runs[0] / "summaries.csv", float_precision="round_trip")
    result = pd.read_csv(out / "summaries.csv", float_precision="round_trip")
    assert result.test_error.iloc[0] == original.test_error.iloc[0]
    for index, expected in enumerate(forecasts[0]):
        with np.load(out / spec.name / f"window_{index}.npz") as saved:
            np.testing.assert_array_equal(saved["arrays"], expected)


def test_two_run_mean_and_monotonicity(local_runs, tmp_path):
    spec, runs, forecasts = local_runs
    # Cross two quantiles in one input so re-sorting is actually exercised.
    path = runs[1] / spec.name / "window_0.npz"
    with np.load(path) as saved:
        values = dict(saved)
    values["arrays"] = values["arrays"][..., ::-1].copy()
    values["arrays"][..., 0] += 4
    np.savez(path, **values)
    forecasts[1][0] = values["arrays"]
    out = tmp_path / "two"
    result = fev_ensemble_runs.ensemble_runs(runs, out, "mean")
    expected_predictions = []
    for index, window in enumerate(spec.to_task().iter_windows()):
        expected = np.sort(np.mean([forecasts[0][index], forecasts[1][index]], axis=0), axis=-1)
        with np.load(out / spec.name / f"window_{index}.npz") as saved:
            np.testing.assert_array_equal(saved["arrays"], expected)
            assert (np.diff(saved["arrays"], axis=-1) >= 0).all()
            past, _ = window.get_input_data()
            expected_predictions.append(
                assemble_predictions(spec, past, saved["ids"].tolist(), expected)
            )
    oracle = spec.to_task().evaluation_summary(expected_predictions, model_name="oracle")
    assert result.test_error.iloc[0] == oracle["test_error"]


@pytest.mark.parametrize(
    "mismatch", ["ids", "shape", "quantiles", "saved_quantiles", "windows", "tasks", "fingerprint"]
)
def test_ensemble_mismatches_fail(local_runs, tmp_path, capsys, mismatch):
    spec, runs, _ = local_runs
    path = runs[1] / spec.name / "window_0.npz"
    if mismatch in {"ids", "shape", "saved_quantiles"}:
        with np.load(path) as saved:
            values = dict(saved)
        if mismatch == "ids":
            values["ids"] = np.asarray(["unexpected", "1"])
        elif mismatch == "shape":
            values["arrays"] = values["arrays"][:, :-1]
        else:
            values["quantile_levels"] = np.arange(9) / 9
        np.savez(path, **values)
    elif mismatch == "windows":
        path.unlink()
    else:
        csv = runs[1] / "summaries.csv"
        rows = pd.read_csv(csv)
        if mismatch == "quantiles":
            rows["quantile_levels"] = "[0.1, 0.5, 0.9]"
        elif mismatch == "fingerprint":
            rows["dataset_fingerprint"] = "different"
        else:
            rows["task_name"] = "other"
            rows["task"] = "other"
        rows.to_csv(csv, index=False)
    out = tmp_path / "bad"
    assert (
        fev_ensemble_runs.main(
            ["--runs", *map(str, runs), "--out", str(out), "--model-name", "bad"]
        )
        == 2
    )
    assert "Ensemble refused" in capsys.readouterr().err
    assert not (out / "summaries.csv").exists()


def test_ensemble_reconstructs_mixed_csv_cutoff_types(local_runs):
    _, runs, _ = local_runs
    row = pd.read_csv(runs[0] / "summaries.csv").iloc[0].to_dict()
    row["initial_cutoff"], row["window_step_size"] = "-8", "4"
    task = fev_ensemble_runs.task_from_row(row)
    assert task.cutoffs == [-8, -4]


@pytest.mark.parametrize("corruption,error", [("truncated", BadZipFile), ("deflate", zlib.error)])
def test_ensemble_corrupt_npz_is_refused(local_runs, tmp_path, capsys, corruption, error):
    spec, runs, _ = local_runs
    path = runs[1] / spec.name / "window_0.npz"
    if corruption == "truncated":
        # Remove the ZIP end-of-directory record, preserving the NPZ signature.
        path.write_bytes(path.read_bytes()[:-22])
    else:
        with np.load(path) as saved:
            values = dict(saved)
        np.savez_compressed(path, **values)
        with ZipFile(path) as archive:
            header = archive.getinfo("arrays.npy").header_offset
        data = bytearray(path.read_bytes())
        name_length = int.from_bytes(data[header + 26 : header + 28], "little")
        extra_length = int.from_bytes(data[header + 28 : header + 30], "little")
        # Reserved DEFLATE block type, at the start of arrays.npy's compressed data.
        data[header + 30 + name_length + extra_length] = 0x06
        path.write_bytes(data)
    with pytest.raises(error), np.load(path, allow_pickle=False) as saved:
        saved["arrays"]
    out = tmp_path / "corrupt"
    assert (
        fev_ensemble_runs.main(
            ["--runs", *map(str, runs), "--out", str(out), "--model-name", "bad"]
        )
        == 2
    )
    stderr = capsys.readouterr().err
    assert "Ensemble refused: " in stderr
    assert "Traceback" not in stderr
    assert not (out / "summaries.csv").exists()
