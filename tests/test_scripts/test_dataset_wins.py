"""Offline protocol checks using synthetic fev rows and tiny local forecasts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from demand_forecast.bench.run import assemble_predictions
from demand_forecast.bench.specs import load_task_specs
from demand_forecast.config import Settings
from scripts import fev_ensemble_runs, fev_shift_tasks, fev_wins
from tests.helpers.fev_fixture import write_fev_fixture
from tests.helpers.fev_spec import fixture_spec


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


def task_yaml(tmp_path, **updates):
    definition = {
        "dataset_path": "unused/dataset",
        "dataset_config": "m5_1W",
        "static_columns": ["cat_id", "dept_id", "item_id", "state_id", "store_id"],
        "horizon": 4,
        "num_windows": 1,
        "initial_cutoff": -100,
    }
    definition.update(updates)
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump({"tasks": [definition]}))
    return path, definition


@pytest.mark.parametrize("count,step", [(1, None), (3, 2), (6, 1), (20, 4)])
@pytest.mark.parametrize("explicit_cutoff", [True, False])
def test_shift_formula_and_output_loads(tmp_path, count, step, explicit_cutoff):
    path, original = task_yaml(tmp_path, num_windows=count, window_step_size=step)
    if not explicit_cutoff:
        del original["initial_cutoff"]
        path.write_text(yaml.safe_dump({"tasks": [original]}))
    out = tmp_path / "shifted.yaml"
    assert fev_shift_tasks.main(["--tasks", str(path), "--names", "m5_1W", "--out", str(out)]) == 0
    spec = load_task_specs(out)[0]
    expected_count = min(max(count, 4), 8)
    effective_step = step if step is not None else 4
    cutoff = -100 if explicit_cutoff else -4 - (count - 1) * effective_step
    assert spec.name == "m5_1W"
    assert spec.num_windows == expected_count
    assert spec.initial_cutoff == cutoff - ((expected_count - 1) * effective_step + 4)
    # Last backtest horizon ends EXACTLY where the first official horizon starts.
    shifted = spec.to_task()
    assert shifted.cutoffs[-1] + spec.horizon == cutoff


def test_official_is_unmodified_and_names_file_and_windows_override(tmp_path):
    path, original = task_yaml(tmp_path)
    names = tmp_path / "names.txt"
    names.write_text("m5_1W\n")
    out = tmp_path / "official.yaml"
    args = ["--tasks", str(path), "--names", f"@{names}", "--out", str(out)]
    assert fev_shift_tasks.main([*args, "--mode", "official"]) == 0
    assert yaml.safe_load(out.read_text())["tasks"] == [original]
    assert load_task_specs(out) == load_task_specs(path)
    assert fev_shift_tasks.main([*args, "--windows", "m5_1W=2"]) == 0
    assert load_task_specs(out)[0].num_windows == 2
    assert fev_shift_tasks.main([*args, "--mode", "official", "--windows", "m5_1W=2"]) == 2


@pytest.mark.parametrize("updates", [{"initial_cutoff": "2023-12-15"}, {"window_step_size": "1D"}])
def test_time_units_fail_with_task_name(tmp_path, capsys, updates):
    path, _ = task_yaml(tmp_path, **updates)
    out = tmp_path / "out.yaml"
    assert fev_shift_tasks.main(["--tasks", str(path), "--names", "m5_1W", "--out", str(out)]) == 2
    assert "m5_1W" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("count,cutoff", [(None, "2023-05-05"), (2, "2023-08-25")])
def test_timestamp_override_round_trips_and_constructs_fev_task(tmp_path, count, cutoff):
    out = tmp_path / "timestamp.yaml"
    args = [
        "--names",
        "rohlik_sales_1W",
        "--out",
        str(out),
        "--cutoff",
        f"rohlik_sales_1W={cutoff}",
        "--step",
        "rohlik_sales_1W=56D",
    ]
    if count is not None:
        args.extend(["--windows", f"rohlik_sales_1W={count}"])
    assert fev_shift_tasks.main(args) == 0
    spec = load_task_specs(out)[0]
    assert (spec.initial_cutoff, spec.window_step_size, spec.num_windows) == (
        cutoff,
        "56D",
        count or 4,
    )
    task = spec.to_task()
    assert task.cutoffs == [
        (pd.Timestamp(cutoff) + pd.Timedelta(days=56 * index)).isoformat()
        for index in range(count or 4)
    ]
    assert pd.Timestamp(task.cutoffs[-1]) + pd.Timedelta(days=56) == pd.Timestamp("2023-12-15")
    original = next(s for s in load_task_specs("fev100") if s.name == spec.name).to_dict()
    original.update(initial_cutoff=cutoff, window_step_size="56D", num_windows=count or 4)
    assert spec.to_dict() == original


def test_timestamp_overrides_repeat_for_rohlik_tasks(tmp_path):
    out = tmp_path / "rohlik.yaml"
    assert (
        fev_shift_tasks.main(
            [
                "--names",
                "rohlik_sales_1W,rohlik_sales_1D",
                "--out",
                str(out),
                "--cutoff",
                "rohlik_sales_1W=2023-05-05",
                "--step",
                "rohlik_sales_1W=56D",
                "--cutoff",
                "rohlik_sales_1D=2023-10-20",
                "--step",
                "rohlik_sales_1D=14D",
            ]
        )
        == 0
    )
    specs = load_task_specs(out)
    assert [s.name for s in specs] == ["rohlik_sales_1W", "rohlik_sales_1D"]
    for spec, cutoff, days in zip(specs, ["2023-05-05", "2023-10-20"], [56, 14]):
        assert (spec.initial_cutoff, spec.window_step_size, spec.num_windows) == (
            cutoff,
            f"{days}D",
            4,
        )
        task = spec.to_task()
        assert len(task.cutoffs) == 4
        assert pd.Timestamp(task.cutoffs[-1]) + pd.Timedelta(days=days) == pd.Timestamp(
            "2023-12-15"
        )


@pytest.mark.parametrize(
    "override",
    [
        ["--cutoff", "rohlik_sales_1W=2023-05-05"],
        ["--step", "rohlik_sales_1W=56D"],
    ],
)
def test_timestamp_requires_both_overrides(tmp_path, capsys, override):
    out = tmp_path / "out.yaml"
    assert (
        fev_shift_tasks.main(
            [
                "--names",
                "rohlik_sales_1W",
                "--out",
                str(out),
                *override,
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "rohlik_sales_1W" in error
    assert "both --cutoff and --step" in error
    assert not out.exists()


@pytest.mark.parametrize(
    "cutoff,count,end",
    [
        ("2023-05-06", 4, "2023-12-16"),
        ("2023-05-05", 3, "2023-10-20"),
        ("2023-05-05T00:00:01", 4, "2023-12-15T00:00:01"),
    ],
)
def test_timestamp_boundary_mismatch_reports_both_dates(tmp_path, capsys, cutoff, count, end):
    out = tmp_path / "out.yaml"
    assert (
        fev_shift_tasks.main(
            [
                "--names",
                "rohlik_sales_1W",
                "--out",
                str(out),
                "--cutoff",
                f"rohlik_sales_1W={cutoff}",
                "--step",
                "rohlik_sales_1W=56D",
                "--windows",
                f"rohlik_sales_1W={count}",
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "rohlik_sales_1W" in error
    assert f"ends at {pd.Timestamp(end).isoformat()}" in error
    assert "official cutoff is 2023-12-15T00:00:00" in error
    assert not out.exists()


@pytest.mark.parametrize("flag,value", [("--cutoff", "2023-05-05"), ("--step", "56D")])
def test_timestamp_override_rejects_integer_task(tmp_path, capsys, flag, value):
    path, _ = task_yaml(tmp_path)
    out = tmp_path / "out.yaml"
    assert (
        fev_shift_tasks.main(
            [
                "--tasks",
                str(path),
                "--names",
                "m5_1W",
                "--out",
                str(out),
                flag,
                f"m5_1W={value}",
            ]
        )
        == 2
    )
    assert f"m5_1W: {flag} requires an official timestamp cutoff" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("flag,value", [("--cutoff", "2023-05-05"), ("--step", "56D")])
def test_timestamp_override_rejects_official_mode(tmp_path, capsys, flag, value):
    out = tmp_path / "out.yaml"
    assert (
        fev_shift_tasks.main(
            [
                "--names",
                "rohlik_sales_1W",
                "--out",
                str(out),
                "--mode",
                "official",
                flag,
                f"rohlik_sales_1W={value}",
            ]
        )
        == 2
    )
    assert "cannot modify official task definitions" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "cutoff,step",
    [
        ("not-a-date", "56D"),
        ("NaT", "56D"),
        ("2023-05-05", "not-an-offset"),
        ("2023-12-15", "0D"),
        ("2024-07-26", "-56D"),
    ],
)
def test_timestamp_override_rejects_invalid_dates_and_steps(tmp_path, capsys, cutoff, step):
    out = tmp_path / "out.yaml"
    assert (
        fev_shift_tasks.main(
            [
                "--names",
                "rohlik_sales_1W",
                "--out",
                str(out),
                "--cutoff",
                f"rohlik_sales_1W={cutoff}",
                "--step",
                f"rohlik_sales_1W={step}",
            ]
        )
        == 2
    )
    assert "rohlik_sales_1W: invalid timestamp override" in capsys.readouterr().err
    assert not out.exists()


def test_unknown_name_fails_without_output(tmp_path, capsys):
    out = tmp_path / "out.yaml"
    assert fev_shift_tasks.main(["--names", "typo", "--out", str(out)]) == 2
    assert "unknown task: typo" in capsys.readouterr().err
    assert not out.exists()


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


def test_positive_cutoff_cannot_wrap_to_negative_index(tmp_path, capsys):
    path, _ = task_yaml(tmp_path, initial_cutoff=10)
    out = tmp_path / "out.yaml"
    assert fev_shift_tasks.main(["--tasks", str(path), "--names", "m5_1W", "--out", str(out)]) == 2
    assert "m5_1W: shifted cutoff has insufficient history" in capsys.readouterr().err
    assert not out.exists()


def write_rows(path, cells, model="model"):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "task_name": name,
                "test_error": score,
                "model_name": model,
                "trained_on_this_dataset": leak,
            }
            for name, score, leak in cells
        ],
        columns=["task_name", "test_error", "model_name", "trained_on_this_dataset"],
    ).to_csv(path, index=False)
    return path


def test_selection_rule_missing_arm_and_reference(tmp_path):
    a0 = write_rows(
        tmp_path / "a0.csv", [("gain", 1.0, False), ("small", 1.0, False), ("only", 2.0, False)]
    )
    a1 = write_rows(
        tmp_path / "a1.csv",
        [("gain", 0.99, False), ("small", 0.995, False), ("no_ref", 0.1, False)],
    )
    report = fev_wins.select_arms({"A0": a0, "A1": a1}, "A0", 0.01)
    choices = report["tasks"]
    assert choices["gain"]["chosen"] == "A1"  # Inclusive 1% boundary.
    assert choices["small"]["chosen"] == "A0"
    assert choices["only"]["chosen"] == "A0"
    assert choices["only"]["scores"]["A1"] is None
    assert choices["no_ref"]["chosen"] is None
    assert "reference" in choices["no_ref"]["reason"]
    out = tmp_path / "choices.json"
    assert (
        fev_wins.main(
            [
                "select",
                "--arm",
                f"A0={a0}",
                "--arm",
                f"A1={a1}",
                "--out",
                str(out),
                "--names",
                "gain,no_ref,absent",
            ]
        )
        == 0
    )
    assert set(json.loads(out.read_text())["tasks"]) == {"gain", "no_ref", "absent"}
    assert len(pd.read_csv(out.with_suffix(".csv"))) == 3


def test_standing_leaks_ties_missing_cells_and_artifacts(tmp_path):
    published = tmp_path / "published"
    write_rows(published / "leaky.csv", [("win", 0.1, True)], "Leaky")
    write_rows(published / "best.csv", [("win", 0.8, False), ("tie", 0.8, False)], "Best")
    write_rows(published / "close.csv", [("win", 0.84, False)], "Close")  # Missing tie.
    write_rows(
        published / "naive.csv", [("win", 1.2, False), ("tie", 1.2, False)], "Seasonal Naive"
    )
    entry = write_rows(
        tmp_path / "entry" / "summaries.csv", [("win", 0.7, False), ("tie", 0.8, False)], "Ours"
    )
    report = fev_wins.standing([entry.parent], published, ["win", "tie", "missing"])
    win, tie, missing = report["tasks"]
    assert win["best_published_model"] == "Best"
    assert win["best_published_sql"] == 0.8
    assert win["seasonal_naive_sql"] == 1.2
    assert win["gap_percent"] == pytest.approx(-12.5)
    assert win["rank"] == 1 and win["first"]
    assert win["published_within_5pct"] == 2
    assert tie["rank"] == 1 and not tie["first"]
    assert tie["eligible_published_models"] == 2
    assert missing["rank"] is None and not missing["first"]
    assert report["totals"] == {"tasks": 3, "scored": 2, "first": 1, "top_3": 2, "top_5": 2}
    leaky = fev_wins.standing([entry], published, include_leaking=True)["tasks"][1]
    assert leaky["best_published_model"] == "Leaky"
    assert leaky["rank"] == 2 and not leaky["first"]
    out = tmp_path / "standing"
    assert (
        fev_wins.main(
            [
                "standing",
                "--entry",
                str(entry.parent),
                "--published",
                str(published),
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    assert {p.name for p in out.iterdir()} == {"standing.csv", "standing.json", "standing.md"}
    assert json.loads((out / "standing.json").read_text())["totals"]["first"] == 1
    assert "win" in (out / "standing.md").read_text()


@pytest.mark.parametrize("command", ["select", "standing"])
def test_wins_input_errors_exit_two(tmp_path, command):
    out = str(tmp_path / "out.json")
    args = (
        ["select", "--arm", "invalid", "--out", out]
        if command == "select"
        else ["standing", "--entry", "missing.csv", "--published", "missing", "--out-dir", out]
    )
    assert fev_wins.main(args) == 2


def test_standing_rank_totals_missing_entry_and_zero_best(tmp_path):
    published = tmp_path / "published"
    for i in range(1, 7):
        write_rows(
            published / f"model{i}.csv",
            [
                ("top3", i, False),
                ("top5", i, False),
                ("outside", i, False),
                ("absent", i, False),
                ("zero", 0, False),
            ],
            f"M{i}",
        )
    entry = write_rows(
        tmp_path / "entry.csv",
        [("top3", 2.5, False), ("top5", 4.5, False), ("outside", 7, False), ("zero", 1, False)],
    )
    report = fev_wins.standing([entry], published)
    rows = {row["task_name"]: row for row in report["tasks"]}
    assert rows["top3"]["rank"] == 3
    assert rows["top5"]["rank"] == 5
    assert rows["outside"]["rank"] == 7
    assert rows["absent"]["our_sql"] is None
    assert rows["zero"]["gap_percent"] is None
    assert report["totals"] == {"tasks": 5, "scored": 4, "first": 0, "top_3": 1, "top_5": 2}


def test_standing_rejects_duplicate_entries_and_unknown_leak_flags(tmp_path):
    entry = write_rows(tmp_path / "entry.csv", [("task", 1, False)])
    published = tmp_path / "published"
    csv = write_rows(published / "model.csv", [("task", 2, False)])
    with pytest.raises(ValueError, match="more than one row"):
        fev_wins.standing([entry, entry], published)
    rows = pd.read_csv(csv)
    rows["trained_on_this_dataset"] = "unknown"
    rows.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="invalid leak flag"):
        fev_wins.standing([entry], published)


def test_hygiene_config_matches_tuned_plus_four_controls():
    root = Path(__file__).resolve().parents[2]
    tuned = Settings.from_yaml(root / "configs/fev-v2-pretrained-finetune-tuned.yaml")
    hygiene = Settings.from_yaml(root / "configs/fev-v2-pretrained-finetune-hygiene.yaml")
    tuned.training.warmup_frac = 0.1
    tuned.training.select_min_gain = 0.05
    tuned.training.exception_policy = "pretrained"
    tuned.training.weight_average = "tail"
    assert hygiene.model_dump() == tuned.model_dump()
