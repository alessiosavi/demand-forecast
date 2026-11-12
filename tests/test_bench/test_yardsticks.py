"""Offline contracts for external-model evaluation; no optional inference packages."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.yardsticks import (  # noqa: E402
    Options,
    load_tasks,
    run_yardstick,
)
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import datasets.config
    import huggingface_hub.constants

    monkeypatch.setattr(datasets.config, "HF_DATASETS_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def predict(self, windows, horizon, levels):
        self.calls.append(windows)
        return [
            [
                np.repeat(it.target[:, -1:, None], horizon, axis=1)
                + np.asarray(levels)[None, None, :]
                for it in w.items
            ]
            for w in windows
        ]


@pytest.fixture
def scenario(tmp_path):
    from datasets import Dataset, Sequence, Value

    path = write_fev_fixture(tmp_path / "data", n_series=3, lengths=(40, 38, 36))
    data = Dataset.from_parquet(str(path))
    features = data.features.copy()
    features["other"] = Sequence(Value("float32"))
    data.map(lambda row: {"other": [2 * v for v in row["target"]]}, features=features).to_parquet(
        path
    )
    definition = {
        "task_name": "synthetic/new",
        "dataset_path": str(path),
        "target": ["target", "other"],
        "known_dynamic_columns": ["promo"],
        "past_dynamic_columns": ["oil"],
        "static_columns": ["cat"],
        "horizon": 4,
        "num_windows": 2,
        "eval_metric": "SQL",
        "extra_metrics": ["MASE", "WQL", "WAPE"],
        "quantile_levels": [i / 10 for i in range(1, 10)],
    }
    task_file = tmp_path / "tasks.yaml"
    task_file.write_text(yaml.safe_dump({"tasks": [definition]}))
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    options = Options(model="chronos2", weights=str(weights), model_name="reference", device="cpu")
    return load_tasks(task_file), options, tmp_path / "out"


def test_runner_same_fev_inputs_multitarget_audit_and_resume(scenario):
    tasks, options, out = scenario
    fake = FakeAdapter()
    frame = run_yardstick(tasks, options, out, adapter_factory=lambda _: fake)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.task == row.task_name == "synthetic/new"
    assert row.training_time_s == 0 and not row.trained_on_this_dataset
    assert row.inference_time_s > 0 and row.num_forecasts == 12
    assert {
        "SQL",
        "MASE",
        "WQL",
        "dataset_fingerprint",
        "fev_version",
        "weights_revision",
        "cross_series",
        "max_context",
        "covariates",
        "batch_size",
        "device",
    } <= set(frame)
    for w, actual in zip(tasks[0].iter_windows(), fake.calls[0], strict=True):
        past, future = w.get_input_data()
        for i, it in enumerate(actual.items):
            np.testing.assert_equal(
                it.target, np.stack([past[i][t] for t in tasks[0].target_columns])
            )
            np.testing.assert_equal(it.past["promo"], past[i]["promo"])
            np.testing.assert_equal(it.past["oil"], past[i]["oil"])
            np.testing.assert_equal(it.future["promo"], future[i]["promo"])
        assert actual.ids == list(past["id"])
    evidence = out / "synthetic/new"
    with np.load(evidence / "window_0.npz", allow_pickle=False) as data:
        assert data["arrays"].shape == (3, 2, 4, 9)
        assert data["targets"].tolist() == tasks[0].target_columns
        assert data["identity"].item() == row.run_identity
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    run_yardstick(
        tasks,
        options,
        out,
        resume=True,
        adapter_factory=lambda _: pytest.fail("resume loaded model"),
    )
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    (evidence / "window_1.npz").unlink()
    run_yardstick(tasks, options, out, resume=True, adapter_factory=lambda _: fake)
    assert len(fake.calls) == 2


@pytest.mark.parametrize("quantiles", [[i / 10 for i in range(1, 10)], [0.1, 0.9]])
def test_single_target_scored_and_audited_with_fev_validation(scenario, quantiles):
    from demand_forecast.bench.yardsticks.runner import fev_predictions
    from scripts.fev_report import build_report

    tasks, options, out = scenario
    task = fev.Task(**{**tasks[0].to_dict(), "target": "target", "quantile_levels": quantiles})
    frame = run_yardstick([task], options, out, adapter_factory=lambda _: FakeAdapter())
    row = frame.iloc[0]
    assert row.num_forecasts == 6
    assert np.isfinite(row[["SQL", "MASE", "WQL", "WAPE"]].astype(float)).all()
    assert {key for key in frame if key.startswith("SQL[")} == {f"SQL[{q}]" for q in quantiles}
    with np.load(out / task.task_name / "window_0.npz", allow_pickle=False) as saved:
        levels = saved["quantiles"].tolist()
        assert levels == sorted(set(quantiles) | {0.5})
        assert saved["arrays"].shape == (3, 1, 4, len(levels))
        predictions = fev_predictions(saved["arrays"], levels, ["target"])
        assert isinstance(predictions, dict) and set(predictions) == {"target"}
        cleaned = task.clean_and_validate_predictions(predictions)
        past, _ = task.get_window(0).get_input_data()
        for i, forecast in enumerate(cleaned["target"]):
            np.testing.assert_allclose(forecast["predictions"], past[i]["target"][-1] + 0.5)
    report, issues = build_report(out, [task], model_name=options.model_name)
    assert issues == []
    assert "Complete:" in report


@pytest.mark.parametrize(
    "actual,expected,matches",
    [
        ("-13", -13, True),
        ("-13.0", -13, True),
        ("4e0", 4, True),
        ("0.25", 0.25, True),
        ("not-a-number", 4, False),
        ("2022-06-26T00:00:00", 4, False),
        ("4W-SUN", 4, False),
        ("5", 4, False),
        ("NaN", 4, False),
        (np.nan, 4, False),
        (None, 4, False),
        (np.nan, np.nan, False),
        ("TRUE", True, True),
        ("false", False, True),
        ("1", True, False),
        (np.nan, None, True),
        (None, None, True),
        ("None", None, False),
    ],
)
def test_summary_numeric_conversion_preserves_bool_and_none(actual, expected, matches):
    from demand_forecast.bench.yardsticks.runner import _summary_matches

    assert bool(_summary_matches({"value": actual}, {"value": expected})) is matches


def test_resume_mixed_timestamp_and_integer_cutoffs_preserves_evidence(
    scenario, tmp_path, monkeypatch, capsys
):
    from demand_forecast.bench.yardsticks import runner
    from scripts.fev_yardstick import main

    tasks, options, out = scenario
    definition = tasks[0].to_dict()
    task_file = tmp_path / "mixed.yaml"
    task_file.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        **definition,
                        "task_name": "timestamp",
                        "initial_cutoff": "2022-06-26",
                        "window_step_size": "4W-SUN",
                    },
                    {**definition, "task_name": "integer"},
                ]
            }
        )
    )
    tasks = load_tasks(task_file)
    frame = run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    assert frame.loc[frame.task == "integer", "initial_cutoff"].iloc[0] == "-8"
    assert frame.loc[frame.task == "integer", "window_step_size"].iloc[0] == "4"
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    capsys.readouterr()
    monkeypatch.setattr(runner, "_adapter", lambda *args: pytest.fail("resume loaded model"))
    assert (
        main(
            [
                "--model",
                "chronos2",
                "--weights",
                options.weights,
                "--tasks",
                str(task_file),
                "--out-dir",
                str(out),
                "--model-name",
                options.model_name,
                "--device",
                "cpu",
                "--resume",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Skipping completed task timestamp" in output
    assert "Skipping completed task integer" in output
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize("model", ["chronos2", "timestral"])
def test_context_and_covariates_applied_before_adapter(scenario, model):
    tasks, options, out = scenario
    options = replace(options, model=model, max_context=5, covariates="none")
    fake = FakeAdapter()
    run_yardstick(tasks, options, out, adapter_factory=lambda _: fake)
    for w in fake.calls[0]:
        for it in w.items:
            assert it.target.shape == (2, 5)
            assert it.past == it.future == {}
    assert options.resolved().batch_size == (100 if model == "chronos2" else 1024)


def test_cross_series_off_splits_targets_but_retains_covariates(scenario):
    tasks, options, out = scenario
    fake = FakeAdapter()
    frame = run_yardstick(
        tasks, replace(options, cross_series="off"), out, adapter_factory=lambda _: fake
    )
    for w in fake.calls[0]:
        assert len(w.items) == 6
        assert all(it.target.shape[0] == 1 for it in w.items)
        assert all(set(it.past) == {"promo", "oil"} for it in w.items)
    assert frame.num_forecasts.iloc[0] == 12
    with pytest.raises(ValueError, match="Chronos"):
        replace(options, model="timestral", cross_series="off").resolved()


@pytest.mark.parametrize(
    "change",
    [
        dict(max_context=5),
        dict(covariates="none"),
        dict(cross_series="off"),
        dict(batch_size=3),
        dict(model_name="different"),
        dict(revision="different"),
    ],
)
def test_resume_refuses_changed_identity(scenario, change):
    tasks, options, out = scenario
    run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    with pytest.raises(ValueError, match="identity"):
        run_yardstick(
            tasks,
            replace(options, **change),
            out,
            resume=True,
            adapter_factory=lambda _: pytest.fail("loaded"),
        )


def test_local_weights_content_is_part_of_identity(scenario):
    tasks, options, out = scenario
    run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    (Path(options.weights) / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity"):
        run_yardstick(tasks, options, out, resume=True)


def test_task_change_does_not_resume(scenario):
    tasks, options, out = scenario
    run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    tasks[0].num_windows = 1
    with pytest.raises(ValueError, match="identity"):
        run_yardstick(tasks, options, out, resume=True)


def test_failure_journal_and_retry(scenario):
    tasks, options, out = scenario

    class Broken(FakeAdapter):
        def predict(self, *args):
            raise RuntimeError("synthetic failure")

    frame = run_yardstick(tasks, options, out, keep_going=True, adapter_factory=lambda _: Broken())
    assert frame.empty
    assert (
        json.loads((out / "failures.json").read_text())["synthetic/new"]["exception_type"]
        == "RuntimeError"
    )
    run_yardstick(tasks, options, out, resume=True, adapter_factory=lambda _: FakeAdapter())
    assert json.loads((out / "failures.json").read_text()) == {}


def test_schema_report_standing_and_merge(scenario, tmp_path):
    from demand_forecast.bench.specs import FevTaskSpec
    from scripts.fev_report import build_report
    from scripts.fev_wins import standing

    tasks, options, out = scenario
    frame = run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    published = tmp_path / "published"
    published.mkdir()
    frame.assign(model_name="published").to_csv(published / "published.csv", index=False)
    assert standing([out], published)["totals"]["scored"] == 1
    values = tasks[0].to_dict()
    values["name"] = values.pop("task_name")
    values.pop("max_context_length", None)
    values.pop("generate_univariate_targets_from", None)
    values["known_dynamic"] = values.pop("known_dynamic_columns")
    values["past_dynamic"] = values.pop("past_dynamic_columns")
    spec = FevTaskSpec(**values, target_support="real")
    report, issues = build_report(out, [spec], model_name="reference")
    assert not issues, issues
    assert "Complete" in report and "reference" in report
    frame.loc[0, "SQL"] += 1
    frame.to_csv(out / "summaries.csv", index=False)
    _, issues = build_report(out, [spec], model_name="reference")
    assert any("SQL" in issue for issue in issues)


def test_merge_nested_yardstick_evidence(scenario, tmp_path):
    import shutil

    from scripts.fev_merge_runs import merge_runs

    tasks, options, out = scenario
    frame = run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    name = tasks[0].task_name
    split = tmp_path / "split"
    isolated = split / name
    shutil.copytree(out / name, isolated / name)
    frame.to_csv(isolated / "summaries.csv", index=False)
    merged = merge_runs(split, [name])
    assert (merged / name / "run.json").read_bytes() == (out / name / "run.json").read_bytes()


@pytest.mark.parametrize("name", ["../escape", "/escape", "a/../b", "a//b", "merged/x"])
def test_merge_rejects_unsafe_nested_names(tmp_path, name):
    from scripts.fev_merge_runs import merge_runs

    with pytest.raises(ValueError, match="unsafe"):
        merge_runs(tmp_path, [name])
    assert not (tmp_path / "merged").exists()


def test_resume_repairs_stale_switch_column_without_model(scenario):
    import pandas as pd

    tasks, options, out = scenario
    expected = run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    changed = expected.copy()
    changed.loc[0, "weights_revision"] = "wrong"
    changed.to_csv(out / "summaries.csv", index=False)
    actual = run_yardstick(
        tasks, options, out, resume=True, adapter_factory=lambda _: pytest.fail("loaded model")
    )
    assert actual.weights_revision.iloc[0] == expected.weights_revision.iloc[0]
    # A deleted summary is reconstructed from checksummed prediction files too.
    (out / "summaries.csv").unlink()
    actual = run_yardstick(
        tasks, options, out, resume=True, adapter_factory=lambda _: pytest.fail("loaded model")
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_fail_fast_also_records_failure(scenario):
    tasks, options, out = scenario

    class Nonfinite(FakeAdapter):
        def predict(self, windows, horizon, levels):
            result = super().predict(windows, horizon, levels)
            result[0][0][:] = np.nan
            return result

    with pytest.raises(ValueError, match="non-finite"):
        run_yardstick(tasks, options, out, adapter_factory=lambda _: Nonfinite())
    assert (
        json.loads((out / "failures.json").read_text())["synthetic/new"]["exception_type"]
        == "ValueError"
    )
    assert not (out / "synthetic/new/run.json").exists()


def test_names_selection_and_relative_dataset_path(scenario, tmp_path):
    tasks, _, _ = scenario
    values = tasks[0].to_dict()
    values["dataset_path"] = "data/data.parquet"
    path = tmp_path / "relative.yaml"
    path.write_text(yaml.safe_dump({"tasks": [values]}))
    assert load_tasks(path, ["synthetic/new"])[0].dataset_path == tasks[0].dataset_path
    with pytest.raises(ValueError, match="names"):
        load_tasks(path, ["missing"])


def test_native_yaml_reporting_and_merge_clis(scenario, tmp_path):
    import shutil

    from scripts.fev_merge_runs import main as merge_main
    from scripts.fev_report import main as report_main

    tasks, options, out = scenario
    frame = run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    args = [
        "--run-dir",
        str(out),
        "--tasks",
        str(tmp_path / "tasks.yaml"),
        "--model-name",
        "reference",
        "--out",
        str(tmp_path / "report.md"),
    ]
    assert report_main(args) == 0
    name = tasks[0].task_name
    split = tmp_path / "split"
    shutil.copytree(out / name, split / name / name)
    frame.to_csv(split / name / "summaries.csv", index=False)
    assert merge_main(["--run-root", str(split), "--tasks", str(tmp_path / "tasks.yaml")]) == 0


def test_cli_keep_going_reports_refused_identity_as_failure(scenario, monkeypatch, tmp_path):
    from demand_forecast.bench.yardsticks import runner
    from scripts.fev_yardstick import main

    tasks, options, out = scenario
    run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    values = yaml.safe_load((tmp_path / "tasks.yaml").read_text())
    values["tasks"][0]["num_windows"] = 1
    (tmp_path / "tasks.yaml").write_text(yaml.safe_dump(values))
    monkeypatch.setattr(runner, "_adapter", lambda *a: pytest.fail("model load"))
    args = [
        "--model",
        "chronos2",
        "--weights",
        options.weights,
        "--tasks",
        str(tmp_path / "tasks.yaml"),
        "--out-dir",
        str(out),
        "--model-name",
        options.model_name,
        "--device",
        "cpu",
        "--resume",
        "--keep-going",
    ]
    assert main(args) == 1


@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("https://example.com/data.parquet", "https://example.com/data.parquet"),
        ("./sub/data.arrow", "LOCAL"),
    ],
)
def test_native_dataset_urls_and_arrow_paths(tmp_path, dataset, expected):
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump({"tasks": [{"task_name": "local", "dataset_path": dataset}]}))
    task = load_tasks(path)[0]
    assert task.dataset_path == (
        str(tmp_path / "sub/data.arrow") if expected == "LOCAL" else expected
    )


def test_report_all_failed_native_tasks_is_incomplete_not_exception(scenario, tmp_path):
    from scripts.fev_report import main

    tasks, options, out = scenario

    class Broken(FakeAdapter):
        def predict(self, *args):
            raise RuntimeError("synthetic failure")

    run_yardstick(tasks, options, out, keep_going=True, adapter_factory=lambda _: Broken())
    report = tmp_path / "failed.md"
    args = [
        "--run-dir",
        str(out),
        "--tasks",
        str(tmp_path / "tasks.yaml"),
        "--model-name",
        "reference",
        "--out",
        str(report),
    ]
    assert main(args) == 1
    assert "Incomplete" in report.read_text()
    (out / "summaries.csv").unlink()
    assert main(args) == 1
    assert "Incomplete" in report.read_text()


@pytest.mark.parametrize(
    "missing", ["configuration_identity", "run_identity", "dataset_fingerprint", "model_name"]
)
def test_report_damaged_summary_is_incomplete(scenario, tmp_path, missing):
    from scripts.fev_report import build_report, main

    tasks, options, out = scenario
    frame = run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    frame.drop(columns=missing).to_csv(out / "summaries.csv", index=False)
    report, issues = build_report(out, tasks, model_name=options.model_name)
    assert "Incomplete" in report
    assert any(missing in issue for issue in issues)
    destination = tmp_path / "damaged.md"
    assert (
        main(
            [
                "--run-dir",
                str(out),
                "--tasks",
                str(tmp_path / "tasks.yaml"),
                "--model-name",
                options.model_name,
                "--out",
                str(destination),
            ]
        )
        == 1
    )
    assert "Incomplete" in destination.read_text() and missing in destination.read_text()


@pytest.mark.parametrize("selection", ["csv", "file"])
def test_report_names_audits_only_selected_yaml_tasks(scenario, tmp_path, selection):
    from scripts.fev_report import main

    tasks, options, out = scenario
    definitions = [
        tasks[0].to_dict(),
        {**tasks[0].to_dict(), "task_name": "second"},
        {**tasks[0].to_dict(), "task_name": "unselected"},
    ]
    task_file = tmp_path / "tasks.yaml"
    task_file.write_text(yaml.safe_dump({"tasks": definitions}))
    selected = load_tasks(task_file, ["synthetic/new", "second"])
    run_yardstick(selected, options, out, adapter_factory=lambda _: FakeAdapter())
    # A merged run can be detected from its summary without a root manifest.
    (out / "yardstick.json").unlink()
    destination = tmp_path / "subset.md"
    args = [
        "--run-dir",
        str(out),
        "--tasks",
        str(task_file),
        "--model-name",
        options.model_name,
        "--out",
        str(destination),
    ]
    assert main(args) == 1
    assert "unselected: expected exactly one model row" in destination.read_text()
    names = "synthetic/new,second"
    if selection == "file":
        names_file = tmp_path / "names.txt"
        names_file.write_text("synthetic/new\nsecond\n")
        names = f"@{names_file}"
    assert main([*args, "--names", names]) == 0
    report = destination.read_text()
    assert "Complete:" in report and "unselected" not in report
    assert "- synthetic/new:" in report and "- second:" in report


@pytest.mark.parametrize(
    "selector", ["missing.yaml", "first,second", "unreadable.yaml", "malformed.yaml"]
)
def test_report_unreadable_yardstick_tasks_exits_cleanly(scenario, tmp_path, capsys, selector):
    from scripts.fev_report import main

    tasks, options, out = scenario
    run_yardstick(tasks, options, out, adapter_factory=lambda _: FakeAdapter())
    task_file = tmp_path / selector
    if selector == "unreadable.yaml":
        task_file.mkdir()
    elif selector == "malformed.yaml":
        task_file.write_text("tasks: [")
    destination = tmp_path / "invalid.md"
    assert (
        main(
            [
                "--run-dir",
                str(out),
                "--tasks",
                str(task_file),
                "--model-name",
                options.model_name,
                "--out",
                str(destination),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "--tasks" in error and "YAML" in error and str(task_file) in error
    assert "Traceback" not in error and not destination.exists()
