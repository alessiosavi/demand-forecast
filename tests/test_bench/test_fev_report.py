"""Offline evidence reports: alignment, exclusions and strict completeness."""

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import run_baseline, run_fev  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402
from tests.helpers.settings import tiny_settings  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fev_report.py"


@pytest.fixture(scope="module")
def report_template(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path_factory.mktemp("report-source")
    first = fixture_spec(write_fev_fixture(directory / "first"), name="first")
    parquet = write_fev_fixture(directory / "second")
    rows = pq.read_table(parquet).to_pylist()
    rows[0]["target"] = [6.0] * 40
    rows[1]["target"][33] = None
    for key in ("timestamp", "target", "promo", "oil"):
        rows[5][key] = rows[5][key][-8:]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    second = fixture_spec(parquet, name="second")
    specs = [first, second]
    out = directory / "out"
    run_baseline(specs, out)
    run_fev(specs, tiny_settings(base_steps=2, d_model=8), out)
    definitions, profiles = [], {}
    for spec in specs:
        definition = asdict(spec)
        profiles[spec.name] = {
            key: definition.pop(key) for key in ("target_support", "categorical_statics")
        }
        definitions.append(definition)
    tasks = directory / "tasks.yaml"
    tasks.write_text(yaml.safe_dump({"tasks": definitions}))
    roles = directory / "roles.yaml"
    roles.write_text(yaml.safe_dump(profiles))
    return specs, out, tasks, roles


@pytest.fixture
def report_run(tmp_path, report_template):
    specs, source, tasks, roles = report_template
    out = tmp_path / "out"
    shutil.copytree(source, out)
    return specs, out, tasks, roles


def report_module():
    spec = importlib.util.spec_from_file_location("fev_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke_report(run, *extra):
    _, out, tasks, roles = run
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(out),
            "--tasks",
            str(tasks),
            "--profiles",
            str(roles),
            "--out",
            str(out / "report.md"),
            *extra,
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    return process, (out / "report.md").read_text() if (out / "report.md").exists() else ""


def table(report, title):
    section = report.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    keys = [v.strip() for v in lines[0].strip("|").split("|")]
    return [
        dict(zip(keys, (v.strip() for v in line.strip("|").split("|")), strict=True))
        for line in lines[2:]
    ]


def test_report_two_tasks_and_exclusions(report_run):
    process, report = invoke_report(report_run)
    assert process.returncode == 0, process.stdout + process.stderr
    for term in (
        "first",
        "second",
        "v2/baseline SQL ratio",
        "coverage_80",
        "width_80",
        "doubled, unscaled pinball",
        "SQL[0.1]",
        "checkpoints_bytes",
        "Complete",
    ):
        assert term in report
    diagnostics = table(report, "Diagnostics")
    second = next(row for row in diagnostics if row["task"] == "second")
    assert int(second["excluded_item_window_pairs"]) == 2
    assert int(second["excluded_horizon_cells"]) == 1
    assert int(second["excluded_SQL_cells_knots"]) == (8 + 1) * 9
    metrics = table(report, "Scores")
    summary = pd.read_csv(report_run[1] / "summaries.csv")
    for row in metrics:
        name = row["task"]
        rows = summary.loc[summary.task == name].set_index("model_name")
        expected = rows.loc["demand-forecast-v2", "SQL"] / rows.loc["seasonal_naive", "SQL"]
        assert float(row["v2/baseline SQL ratio"]) == pytest.approx(expected)
        assert int(row["num_forecasts"]) == (12 if name == "first" else 11)
    assert "quantile" in report and "item-window" in report
    # Independent interval and pinball arithmetic, with unequal window sizes.
    for spec in report_run[0]:
        coverages, widths, losses = [], [], []
        for index, window in enumerate(spec.to_task().iter_windows()):
            truth = window.get_ground_truth()
            labels = np.asarray([row[spec.target] for row in truth], dtype=float)
            with np.load(report_run[1] / spec.name / f"window_{index}.npz") as saved:
                by_id = dict(zip(saved["ids"].tolist(), saved["arrays"], strict=True))
                arrays = np.asarray([by_id[str(row[spec.id_column])] for row in truth], dtype=float)
            valid = ~np.isnan(labels)
            coverages.append(
                ((labels >= arrays[..., 0]) & (labels <= arrays[..., -1]))[valid].mean()
            )
            widths.append((arrays[..., -1] - arrays[..., 0])[valid].mean())
            difference = labels[..., None] - arrays
            q = np.asarray(spec.quantile_levels)
            losses.append(
                np.nanmean(2 * np.maximum(q * difference, (q - 1) * difference), axis=(0, 1))
            )
        row = next(row for row in diagnostics if row["task"] == spec.name)
        assert float(row["coverage_80"]) == pytest.approx(np.mean(coverages))
        assert float(row["width_80"]) == pytest.approx(np.mean(widths))
        knot_row = next(
            row for row in table(report, "Doubled, unscaled pinball") if row["task"] == spec.name
        )
        for q, loss in zip(spec.quantile_levels, np.mean(losses, axis=0), strict=True):
            assert float(knot_row[f"pinball_{q}"]) == pytest.approx(loss)


@pytest.mark.parametrize("legacy", ["none", "first", "all"])
def test_report_optional_validation_columns_and_legacy_runs(report_run, legacy):
    specs, out, _, _ = report_run
    for spec in specs:
        path = out / spec.name / "run.json"
        report = json.loads(path.read_text())
        if legacy in ("all", spec.name):
            report.pop("val_items_max", None)
            report.pop("n_val_items_eligible", None)
        else:
            report["val_items_max"] = 2
            report["n_val_items_eligible"] = 6
        path.write_text(json.dumps(report))
    rendered, issues = report_module().build_report(out, specs)
    assert not issues
    for row in table(rendered, "Training"):
        if legacy == "all":
            assert "val_items_max" not in row and "n_val_items_eligible" not in row
        elif row["task"] == legacy:
            assert row["val_items_max"] == row["n_val_items_eligible"] == "unknown"
        else:
            assert row["val_items_max"] == "2" and row["n_val_items_eligible"] == "6"


@pytest.mark.parametrize(
    "fit_count,cumulative_count,expected_count", [(7, 0, 7), (None, 9, 9), (0, 9, 0)]
)
def test_report_nonfinite_losses_gate_and_fit_result_precedence(
    report_run, fit_count, cumulative_count, expected_count
):
    specs, out, _, _ = report_run
    path = out / "first/run.json"
    report = json.loads(path.read_text())
    if fit_count is None:
        report["fit_result"].pop("nonfinite_losses", None)
    else:
        report["fit_result"]["nonfinite_losses"] = fit_count
    report["timings"]["cumulative"]["nonfinite_losses"] = cumulative_count
    path.write_text(json.dumps(report))

    rendered, issues = report_module().build_report(out, specs)
    expected_issues = (
        [f"first: {expected_count} non-finite training losses"] if expected_count else []
    )
    assert issues == expected_issues
    assert table(rendered, "Training")[0]["nonfinite_losses"] == str(expected_count)
    process, rendered = invoke_report(report_run)
    assert process.returncode == int(expected_count > 0), process.stdout + process.stderr
    if expected_count:
        assert expected_issues[0] in rendered
        process, rendered = invoke_report(report_run, "--allow-partial")
        assert process.returncode == 0, process.stdout + process.stderr
        assert "Incomplete" in rendered and expected_issues[0] in rendered


@pytest.mark.parametrize("fit_count_present", [False, True])
@pytest.mark.parametrize("legacy", ["none", "first", "all"])
def test_report_optional_covariate_counters_and_legacy_shapes(
    report_run, fit_count_present, legacy
):
    specs, out, _, _ = report_run
    counters = {
        "cold_covariate_columns": 3,
        "floored_covariate_columns": 5,
        "clipped_covariate_values": 11,
    }
    for spec in specs:
        path = out / spec.name / "run.json"
        report = json.loads(path.read_text())
        if fit_count_present:
            report["fit_result"]["nonfinite_losses"] = 0
        else:
            report["fit_result"].pop("nonfinite_losses", None)
        cumulative = report["timings"]["cumulative"]
        for key, value in counters.items():
            if legacy in ("all", spec.name):
                cumulative.pop(key, None)
            else:
                cumulative[key] = value
        path.write_text(json.dumps(report))

    rendered, issues = report_module().build_report(out, specs)
    assert not issues
    for row in table(rendered, "Training"):
        assert row["nonfinite_losses"] == "0"
        for key, value in counters.items():
            assert row[key] == ("unknown" if legacy in ("all", row["task"]) else str(value))


@pytest.mark.parametrize("fit_count_present", [False, True])
def test_report_cumulative_nonfinite_count_remains_required(report_run, fit_count_present):
    specs, out, _, _ = report_run
    path = out / "first/run.json"
    report = json.loads(path.read_text())
    if fit_count_present:
        report["fit_result"]["nonfinite_losses"] = 0
    else:
        report["fit_result"].pop("nonfinite_losses", None)
    report["timings"]["cumulative"].pop("nonfinite_losses")
    path.write_text(json.dumps(report))

    _, issues = report_module().build_report(out, specs)
    assert issues == ["first: missing/non-finite timings.cumulative.nonfinite_losses"]


def test_report_quick_model_name_and_validation_counts(report_run):
    specs, out, _, _ = report_run
    settings = tiny_settings(base_steps=10, d_model=8)
    settings.training.max_steps_cap = 7
    settings.training.val_items_max = 2
    run_fev(specs, settings, out, model_name="demand-forecast-v2-quick")
    process, rendered = invoke_report(report_run, "--model-name", "demand-forecast-v2-quick")
    assert process.returncode == 0, process.stdout + process.stderr
    for row in table(rendered, "Training"):
        report = json.loads((out / row["task"] / "run.json").read_text())
        assert row["val_items_max"] == row["n_val_items"] == "2"
        assert int(row["n_val_items_eligible"]) == report["n_val_items_eligible"] > 2
        assert row["max_steps"] == "7"


def test_report_reordered_npz_ids_still_align(report_run):
    specs, out, _, _ = report_run
    module = report_module()
    before, issues = module.build_report(out, specs)
    assert not issues
    path = out / "first/window_0.npz"
    with np.load(path, allow_pickle=False) as saved:
        data = dict(saved)
    data["ids"], data["arrays"] = data["ids"][::-1], data["arrays"][::-1]
    np.savez(path, **data)
    after, issues = module.build_report(out, specs)
    assert not issues
    assert table(before, "Diagnostics") == table(after, "Diagnostics")
    assert table(before, "Doubled, unscaled pinball") == table(after, "Doubled, unscaled pinball")


def test_report_missing_task_fails_and_allow_partial_succeeds(report_run):
    _, out, _, _ = report_run
    shutil.rmtree(out / "second")
    frame = pd.read_csv(out / "summaries.csv")
    frame.loc[frame.task != "second"].to_csv(out / "summaries.csv", index=False)
    process, report = invoke_report(report_run)
    assert process.returncode == 1, process.stdout + process.stderr
    assert "second: expected exactly one model row" in report
    process, report = invoke_report(report_run, "--allow-partial")
    assert process.returncode == 0
    assert "Incomplete" in report


@pytest.mark.parametrize(
    "corruption", ["fingerprint", "count", "nonfinite", "duplicate", "identity", "legacy"]
)
def test_report_detects_incomplete_or_noncomparable_evidence(report_run, corruption):
    specs, out, _, _ = report_run
    path = out / "summaries.csv"
    frame = pd.read_csv(path)
    model = (frame.task == "first") & (frame.model_name == "demand-forecast-v2")
    if corruption == "fingerprint":
        frame.loc[model, "dataset_fingerprint"] = "foreign"
    elif corruption == "count":
        frame.loc[model, "num_forecasts"] = 99
    elif corruption == "nonfinite":
        frame.loc[model, "SQL"] = np.inf
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.loc[model]])
    elif corruption == "identity":
        frame.loc[model, "run_identity"] = "foreign"
    else:
        saved_path = out / "first/window_0.npz"
        with np.load(saved_path, allow_pickle=False) as data:
            legacy = {key: data[key] for key in ("arrays", "ids", "identity")}
        np.savez(saved_path, **legacy)
    frame.to_csv(path, index=False)
    report, issues = report_module().build_report(out, specs)
    assert issues
    assert "Incomplete" in report
    if corruption in ("fingerprint", "count", "nonfinite", "duplicate", "identity"):
        assert table(report, "Scores")[0]["v2/baseline SQL ratio"] == "not comparable"
    if corruption == "legacy":
        assert "unknown" in report and "diagnostics" in " ".join(issues)


@pytest.mark.parametrize("failure_at", ["task", "dataset", "iteration_start", "iteration_next"])
def test_report_task_runtime_failure_preserves_other_task_and_writes_report(
    report_run, monkeypatch, failure_at
):
    specs, out, tasks, roles = report_run
    module = report_module()
    to_task = type(specs[0]).to_task

    def fail(*args, **kwargs):
        raise RuntimeError("Failed to load the dataset")

    def faulty_task(spec):
        if spec.name != "first":
            return to_task(spec)
        if failure_at == "task":
            fail()
        task = to_task(spec)
        if failure_at == "dataset":
            monkeypatch.setattr(task, "load_full_dataset", fail)
        elif failure_at == "iteration_start":
            monkeypatch.setattr(task, "iter_windows", fail)
        else:
            iter_windows = task.iter_windows

            def interrupted_windows():
                yield next(iter(iter_windows()))
                fail()

            monkeypatch.setattr(task, "iter_windows", interrupted_windows)
        return task

    monkeypatch.setattr(type(specs[0]), "to_task", faulty_task)

    def assert_report(report):
        completeness = report.split("## Completeness\n", 1)[1]
        assert "first: task could not be audited: Failed to load the dataset" in completeness
        assert "Incomplete" in completeness
        assert "second:" not in completeness
        for title, column in (
            ("Training", "steps"),
            ("Scores", "SQL"),
            ("Diagnostics", "coverage_80"),
            ("Doubled, unscaled pinball", "pinball_0.1"),
        ):
            row = next(row for row in table(report, title) if row["task"] == "second")
            assert np.isfinite(float(row[column]))

    report, issues = module.build_report(out, specs)
    assert issues
    assert_report(report)
    output = out / "failed-task-report.md"
    args = [
        "--run-dir",
        str(out),
        "--tasks",
        str(tasks),
        "--profiles",
        str(roles),
        "--out",
        str(output),
    ]
    assert module.main(args) == 1
    assert_report(output.read_text())
    assert module.main([*args, "--allow-partial"]) == 0
    assert_report(output.read_text())


def test_report_window_runtime_failure_keeps_later_windows_and_tasks(report_run, monkeypatch):
    specs, out, _, _ = report_run
    module = report_module()
    load = np.load

    def fail_first_window(path, **kwargs):
        if Path(path) == out / "first/window_0.npz":
            raise RuntimeError("Failed to read window evidence")
        return load(path, **kwargs)

    monkeypatch.setattr(module.np, "load", fail_first_window)
    report, issues = module.build_report(out, specs)
    assert any(
        "first: window_0.npz" in issue and "Failed to read window evidence" in issue
        for issue in issues
    )
    rows = table(report, "Window diagnostics")
    assert len(rows) == 4
    failed = next(row for row in rows if row["task"] == "first" and row["window"] == "0")
    assert failed["inference_time_s"] == "unknown"
    later = next(row for row in rows if row["task"] == "first" and row["window"] == "1")
    assert float(later["inference_time_s"]) > 0
    assert not any(issue.startswith("second:") for issue in issues)


@pytest.mark.parametrize("skips", [0, 3, None, "absent"])
def test_report_shows_scaler_skips(report_run, skips):
    specs, out, _, _ = report_run
    frame = pd.read_csv(out / "summaries.csv", float_precision="round_trip")
    assert (frame.loc[frame.model_name == "demand-forecast-v2", "scaler_skips"] == 0).all()
    for spec in specs:
        path = out / spec.name / "run.json"
        report = json.loads(path.read_text())
        cumulative = report["timings"]["cumulative"]
        cumulative.pop("scaler_skips", None)
        if skips != "absent":
            cumulative["scaler_skips"] = skips
        path.write_text(json.dumps(report))
    rendered, issues = report_module().build_report(out, specs)
    assert not issues  # Legacy fallback evidence is valid, with unknown skip totals.
    assert {row["scaler_skips"] for row in table(rendered, "Timings")} == {
        "unknown" if skips in (None, "absent") else str(skips)
    }
