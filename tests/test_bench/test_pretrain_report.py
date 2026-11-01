"""Offline comparison, pinned CSV schema, and real saved-window diagnostics."""

import importlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import settings_for_task  # noqa: E402
from demand_forecast.bench.specs import load_task_specs  # noqa: E402
from demand_forecast.config import Settings  # noqa: E402
from scripts import fev_report  # noqa: E402
from tests.test_bench.test_fev_report import report_template, table  # noqa: E402, F401

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/pretrain_report"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/pretrain_report.py"


def report_module():
    assert SCRIPT.is_file(), "pretrain_report.py is not implemented"
    return importlib.import_module("scripts.pretrain_report")


@pytest.fixture
def comparisons(tmp_path):
    published = tmp_path / "published"
    shutil.copytree(FIXTURES, published)
    reference = pd.read_csv(published / "seasonal_naive.csv")
    baseline = reference.assign(model_name="seasonal_naive")
    candidate = reference.assign(model_name="candidate", task=["m5_1W", "covid"])
    candidate["test_error"] = candidate["SQL"] = baseline.test_error / 2
    # Keep the three model excerpts unmodified, including Timestral's distinct
    # fingerprints; the remaining forty models are explicitly synthetic.
    for i in range(40):
        synthetic = reference.assign(model_name=f"synthetic-{i:02d}")
        synthetic["test_error"] = synthetic["SQL"] = baseline.test_error * (1 + i / 100)
        synthetic.to_csv(published / f"synthetic-{i:02d}.csv", index=False)
    return candidate, baseline, published


def test_rank_uses_leaderboard_order_not_skill():
    module = report_module()
    board = pd.DataFrame(
        {"win_rate": [0.9, 0.7, 0.7, 0.2], "skill_score": [0.2, 0.6, 0.8, 0.1]},
        index=["win-winner", "candidate", "skill-winner", "last"],
    )
    assert module.rank_labels(board, "candidate") == ("2= of 4", "2 of 4")
    assert module.rank_labels(board, "skill-winner") == ("2= of 4", "1 of 4")
    assert module.rank_labels(board, "win-winner") == ("1 of 4", "3 of 4")


def test_published_baseline_dropped_and_model_count_asserted(comparisons):
    module = report_module()
    candidate, baseline, directory = comparisons
    baseline = baseline.assign(test_error=baseline.test_error * 2, SQL=baseline.SQL * 2)
    published = module.load_published(directory)
    board = module.comparison_board(candidate, baseline, published)
    assert len(board) == 45
    assert "Seasonal Naive" not in board.index
    assert "Timestral-5M" in board.index
    assert board.loc["seasonal_naive", "skill_score"] == 0
    assert board.loc["candidate", "skill_score"] == pytest.approx(0.75)
    assert board.loc["candidate", "skill_score_lower"] == pytest.approx(0.75)
    assert board.loc["candidate", "skill_score_upper"] == pytest.approx(0.75)
    (directory / "synthetic-00.csv").unlink()
    with pytest.raises(ValueError, match="43"):
        module.load_published(directory)


@pytest.mark.parametrize(
    "change,match",
    [
        ("baseline_fingerprint", "dataset_fingerprint"),
        ("baseline_missing_fingerprint", "dataset_fingerprint"),
        ("candidate_missing_fingerprint", "dataset_fingerprint"),
        ("candidate_missing_num_forecasts", "num_forecasts"),
        ("baseline_absent_num_forecasts", "num_forecasts"),
        ("num_forecasts", "num_forecasts"),
        ("partially_reported_num_forecasts", "num_forecasts"),
        ("horizon", "pivot|coverage"),
        ("missing_task", "pivot|coverage"),
        ("duplicate", "duplicate"),
        ("missing_score", "missing|finite"),
        ("second_candidate", "one.*model_name"),
    ],
)
def test_fingerprint_or_num_forecasts_mismatch_fails(comparisons, change, match):
    module = report_module()
    candidate, baseline, directory = comparisons
    published = module.load_published(directory)
    if change == "baseline_fingerprint":
        baseline.loc[0, "dataset_fingerprint"] = "different"
    elif change == "baseline_missing_fingerprint":
        baseline.loc[0, "dataset_fingerprint"] = None
    elif change == "candidate_missing_fingerprint":
        candidate.loc[0, "dataset_fingerprint"] = None
    elif change == "candidate_missing_num_forecasts":
        candidate.loc[0, "num_forecasts"] = np.nan
    elif change == "baseline_absent_num_forecasts":
        baseline = baseline.drop(columns="num_forecasts")
    elif change in ("num_forecasts", "horizon"):
        published.loc[published.index[0], change] = 99
    elif change == "partially_reported_num_forecasts":
        published.loc[published.index[:2], "num_forecasts"] = [None, 99]
    elif change == "missing_task":
        published = published.iloc[1:]
    elif change == "duplicate":
        published = pd.concat([published, published.iloc[:1]], ignore_index=True)
    elif change == "missing_score":
        published.loc[published.index[0], "test_error"] = np.nan
    else:
        candidate = pd.concat([candidate, candidate.assign(model_name="another")])
    with pytest.raises(ValueError, match=match):
        module.comparison_board(candidate, baseline, published)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "fingerprint",
        "missing_fingerprint",
        "absent_fingerprint",
        "missing_count",
        "absent_count",
        "absent_count_column",
    ],
)
def test_published_comparability(comparisons, change):
    module = report_module()
    candidate, baseline, directory = comparisons
    if change == "absent_count":
        path = directory / "synthetic-00.csv"
        pd.read_csv(path).drop(columns="num_forecasts").to_csv(path, index=False)
    published = module.load_published(directory)
    first = published.index[published.model_name.eq("Chronos-Bolt")][0]
    if change in ("fingerprint", "missing_fingerprint"):
        published.loc[first, "dataset_fingerprint"] = (
            "different" if change == "fingerprint" else None
        )
    elif change == "absent_fingerprint":
        published = published.drop(columns="dataset_fingerprint")
    elif change == "missing_count":
        index = published.index[published.model_name.eq("synthetic-00")][0]
        published.loc[index, "num_forecasts"] = None
    elif change == "absent_count_column":
        published = published.drop(columns="num_forecasts")
    board = module.comparison_board(candidate, baseline, published)
    assert len(board) == 45
    rows = module.comparability_rows(candidate, published)
    assert len(rows) == 43
    assert [row["model_name"] for row in rows] == sorted(published.model_name.unique())
    by_model = {row["model_name"]: row for row in rows}
    assert by_model["Timestral-5M"]["fingerprint matches"] == "0/2"
    assert by_model["Timestral-5M"]["fev_version"] == "0.11.0.dev0"
    for model, row in by_model.items():
        expected = "2/2"
        if model == "Timestral-5M" or change == "absent_fingerprint":
            expected = "0/2"
        elif model == "Chronos-Bolt" and change in ("fingerprint", "missing_fingerprint"):
            expected = "1/2"
        assert row["fingerprint matches"] == expected
        unreported = change == "absent_count_column" or (
            model == "synthetic-00" and change in ("missing_count", "absent_count")
        )
        assert row["num_forecasts"] == ("unreported" if unreported else "equal")


def test_covid_task_filtered_by_definition_not_name(comparisons):
    module = report_module()
    candidate, baseline, directory = comparisons
    candidate = candidate.iloc[1:].copy()
    candidate["task_name"] = "uk_covid_utla_1W"  # Local alias differs from upstream.
    published = module.load_published(directory)
    # Same alias and dataset, wrong target: this row must never enter the board.
    distractor = published.loc[published.dataset_config == "uk_covid_utla_1W"].copy()
    distractor["target"] = "new_cases"
    distractor["task_name"] = "uk_covid_utla_1W"
    distractor["dataset_fingerprint"] = "wrong-definition"
    published = pd.concat([published, distractor], ignore_index=True)
    board = module.comparison_board(candidate, baseline, published)
    assert len(board) == 45
    assert board.loc["Chronos-Bolt", "skill_score"] == pytest.approx(
        1 - 17.4887255416 / 24.6697283717
    )
    assert board.loc["candidate", "skill_score"] == pytest.approx(0.5)


CONFIGS = SCRIPT.parents[1] / "configs"
RECIPE_FILES = {
    "frozen": "fev-v2-standard.yaml",
    "zero-shot": "fev-v2-pretrained-zero-shot.yaml",
    "finetune": "fev-v2-pretrained-finetune.yaml",
    "wide-random": "fev-v2-wide-random.yaml",
    "heldout-frozen": "fev-v2-standard.yaml",
}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Real task/recipe definitions with synthetic summaries and no large models."""
    module = report_module()
    weekly, heldout = load_task_specs("weekly"), load_task_specs("heldout")
    records = []
    for spec in [*weekly, *heldout]:
        records.append(
            {
                **asdict(spec.to_task()),
                "task": spec.name,
                "model_name": "seasonal_naive",
                "test_error": 1.0,
                "SQL": 1.0,
                "training_time_s": 0.0,
                "inference_time_s": 1.0,
                "trained_on_this_dataset": False,
                "fev_version": "0.10.0",
                "dataset_fingerprint": f"synthetic-{spec.name}",
                "num_forecasts": 3 * spec.num_windows,
            }
        )
    reference = pd.DataFrame(records)
    baseline, published = tmp_path / "baseline", tmp_path / "published"
    baseline.mkdir()
    published.mkdir()
    reference.to_csv(baseline / "summaries.csv", index=False)
    for i in range(43):
        frame = reference.assign(model_name=f"synthetic-{i:02d}")
        if i in (0, 2):
            frame.loc[0, "dataset_fingerprint"] = "different" if i == 0 else None
        elif i == 1:
            frame = frame.drop(columns="num_forecasts").assign(fev_version="0.6.1")
        frame.to_csv(published / f"synthetic-{i:02d}.csv", index=False)
    reference.assign(model_name="Seasonal Naive").to_csv(
        published / "seasonal_naive.csv", index=False
    )
    pretrain = tmp_path / "pretrain-bundle"
    pretrain.mkdir()
    (pretrain / "manifest.json").write_text(
        json.dumps(
            {
                "pretrain_identity": "pretraining-identity",
                "sha256": {"model.pt": "a" * 64},
                "best_step": 17,
                "git_sha": "pretrain-sha",
                "external_diagnostic": {"name": "nn5", "ext_sql": 0.7, "ext_sql_naive": 1.2},
            }
        )
    )
    runs = {}
    for name, filename in RECIPE_FILES.items():
        specs = (
            [*weekly, *heldout]
            if name == "zero-shot"
            else (heldout if name == "heldout-frozen" else weekly)
        )
        run_dir = runs[name] = tmp_path / name
        run_dir.mkdir()
        selected = reference.loc[reference.task.isin([spec.name for spec in specs])].copy()
        selected["model_name"] = name
        selected["test_error"] = selected["SQL"] = 0.5
        selected["run_identity"] = [f"{name}:{task}" for task in selected.task]
        selected.to_csv(run_dir / "summaries.csv", index=False)
        for spec in specs:
            settings = settings_for_task(spec, Settings.from_yaml(CONFIGS / filename), "W-SUN")
            record = {"run_identity": f"{name}:{spec.name}", "fit_result": {"best_step": 2}}
            if name in ("zero-shot", "finetune"):
                settings.training.init_from = str(pretrain)
                record["transfer"] = {
                    "pretrain_identity": "pretraining-identity",
                    "model_sha256": "a" * 64,
                }
            record["settings"] = settings.model_dump(mode="json")
            task_dir = run_dir / spec.name
            task_dir.mkdir()
            (task_dir / "run.json").write_text(json.dumps(record))

    # Only the dataset-reading boundary is replaced. Task/recipe checks, source
    # manifests, report rendering, and fev's actual ranking all run normally.
    def metrics(run_dir, task):
        return {
            "coverage_80": [0.75] * task.num_windows,
            "MASE": [2.0] * task.num_windows,
            "dataset_fingerprint": f"synthetic-{task.task_name}",
            "num_forecasts": 3 * task.num_windows,
            "run_identity": f"{Path(run_dir).name}:{task.task_name}",
        }

    monkeypatch.setattr(module.fev_report, "window_metrics", metrics)
    return {
        "runs": runs,
        "baseline": baseline,
        "published": published,
        "fev_commit": "193e4fc",
        "spend": {"runs": {name: {"spend_usd": 1.25, "wall_time_s": 73} for name in runs}},
        "leaked": ["solar_1W"],
    }


def test_preregistered_report(experiment):
    # Display names are not part of the preregistered pivot definition.
    for directory in experiment["runs"].values():
        path = directory / "summaries.csv"
        pd.read_csv(path).assign(task_name="display alias").to_csv(path, index=False)
    report = report_module().build_report(**experiment)
    assert "post_hoc: no" in report and "preregistered 7/5 task sets" in report
    assert "one source bundle" in report
    rows = table(report, "Comparison")
    assert len(rows) == 6
    for row in rows:
        assert len(row["per-task SQL"].split(";")) == (7 if row["block"] == "frozen tasks" else 5)
        assert row["mean MASE"] == "2" and row["mean coverage_80"] == "0.75"
        assert row["spend USD"] == "1.25" and row["wall time s"] == "73"
    comparability = table(report, "Comparability")
    assert len(comparability) == 43
    assert [row["model_name"] for row in comparability] == [f"synthetic-{i:02d}" for i in range(43)]
    for i, row in enumerate(comparability):
        assert row["fingerprint matches"] == ("11/12" if i in (0, 2) else "12/12")
        assert row["num_forecasts"] == ("unreported" if i == 1 else "equal")
        assert row["fev_version"] == ("0.6.1" if i == 1 else "0.10.0")
    assert (
        report.index("## Comparison")
        < report.index("## Comparability")
        < report.index("## Leakage")
    )
    assert "different revision" in report and "where reported" in report and "solar_1W" in report
    transferred = [
        row for row in table(report, "Provenance") if row["run"] in ("zero-shot", "finetune")
    ]
    assert len(transferred) == 19
    assert all(row["pretrain best step"] == "17" for row in transferred)
    assert all(
        row["external SQL"] == "0.7" and row["external naive SQL"] == "1.2" for row in transferred
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing", "frozen tasks.*m5_1W"),
        ("extra", "frozen tasks.*unexpected-task"),
        ("swapped", "frozen tasks.*horizon"),
        ("heldout_frozen", "held-out tasks.*m5_1W"),
    ],
)
def test_preregistered_task_sets_required(experiment, change, match):
    label = "heldout-frozen" if change == "heldout_frozen" else "frozen"
    path = experiment["runs"][label] / "summaries.csv"
    frame = pd.read_csv(path)
    if change == "missing":
        frame = frame.iloc[1:]
        # Maintain internal agreement to demonstrate that a curated six-task
        # experiment is rejected even when every other run covers those six.
        for name in ("zero-shot", "finetune", "wide-random"):
            other_path = experiment["runs"][name] / "summaries.csv"
            other = pd.read_csv(other_path)
            other.loc[other.task != "m5_1W"].to_csv(other_path, index=False)
    elif change == "extra":
        extra = frame.iloc[:1].assign(task="unexpected-task", dataset_config="unexpected-task")
        frame = pd.concat([frame, extra], ignore_index=True)
    elif change == "swapped":
        frame.loc[0, "horizon"] += 1
    else:
        frozen = pd.read_csv(experiment["runs"]["frozen"] / "summaries.csv")
        frame.iloc[0] = frozen.iloc[0]
        frame["model_name"] = label
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        report_module().build_report(**experiment)


@pytest.mark.parametrize(
    "label,section,key,value",
    [
        ("frozen", "model", "d_model", 256),
        ("frozen", "training", "max_steps_cap", 10),
        ("zero-shot", "training", "init_from", None),
        ("zero-shot", "training", "zero_shot", False),
        ("wide-random", "training", "init_from", "unexpected-bundle"),
        ("zero-shot", "transfer", "model_sha256", "b" * 64),
        ("finetune", "transfer", "pretrain_identity", "another-source"),
    ],
)
def test_declared_recipes_and_one_source_required(experiment, label, section, key, value):
    frame = pd.read_csv(experiment["runs"][label] / "summaries.csv")
    # Last zero-shot task is held out: the source must agree across both blocks.
    task = frame.task.iloc[-1 if label == "zero-shot" else 0]
    path = experiment["runs"][label] / task / "run.json"
    record = json.loads(path.read_text())
    destination = record["transfer"] if section == "transfer" else record["settings"][section]
    destination[key] = value
    if label == "frozen" and section == "model":
        destination.update(
            Settings.from_yaml(CONFIGS / RECIPE_FILES["zero-shot"]).model.model_dump()
        )
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError) as error:
        report_module().build_report(**experiment)
    assert label in str(error.value) and task in str(error.value) and key in str(error.value)


def test_configs_dir_and_normalized_defaults(experiment, tmp_path):
    configs = tmp_path / "declared-configs"
    configs.mkdir()
    for filename in set(RECIPE_FILES.values()):
        settings = Settings.from_yaml(CONFIGS / filename)
        settings.log_level = "DEBUG"
        settings.to_yaml(configs / filename)
    for directory in experiment["runs"].values():
        for path in directory.glob("*/run.json"):
            record = json.loads(path.read_text())
            record["settings"]["log_level"] = "DEBUG"
            # Omitted defaults must compare equal to explicit YAML defaults.
            record["settings"]["training"].pop("val_items_max")
            path.write_text(json.dumps(record))
    spend = tmp_path / "spend.json"
    spend.write_text(json.dumps(experiment["spend"]))
    out = tmp_path / "report.md"
    args = [
        part
        for name, directory in experiment["runs"].items()
        for part in (f"--{name}", str(directory))
    ]
    args += [
        "--baseline",
        str(experiment["baseline"]),
        "--published",
        str(experiment["published"]),
        "--fev-commit",
        "193e4fc",
        "--spend",
        str(spend),
        "--out",
        str(out),
        "--configs-dir",
        str(configs),
    ]
    assert report_module().main(args) == 0
    assert "post_hoc: no" in out.read_text()


@pytest.mark.parametrize("source_hash", ["a" * 64, "b" * 64, None])
def test_provenance_binds_source_checkpoint(experiment, source_hash):
    directory = experiment["runs"]["zero-shot"]
    row = pd.read_csv(directory / "summaries.csv").iloc[0]
    record = json.loads((directory / row.task / "run.json").read_text())
    path = Path(record["settings"]["training"]["init_from"]) / "manifest.json"
    source = json.loads(path.read_text())
    source["sha256"]["model.pt"] = source_hash
    path.write_text(json.dumps(source))
    if source_hash == "a" * 64:
        result = report_module()._provenance(directory, row)
        assert result["pretrain identity"] == "pretraining-identity"
        assert result["pretrain best step"] == 17
        assert result["external SQL"] == 0.7
    else:
        with pytest.raises(ValueError, match="model_sha256"):
            report_module()._provenance(directory, row)


def test_coverage_from_window_metrics(tmp_path, report_template):  # noqa: F811 -- pytest fixture
    specs, source, _, _ = report_template
    for spec in specs:
        expected_coverage, expected_mase = [], []
        for index, window in enumerate(spec.to_task().iter_windows()):
            labels = np.asarray([r[spec.target] for r in window.get_ground_truth()], dtype=float)
            past, _ = window.get_input_data()
            histories = {
                str(r[spec.id_column]): np.asarray(r[spec.target], dtype=float) for r in past
            }
            ids = [str(r[spec.id_column]) for r in window.get_ground_truth()]
            with np.load(source / spec.name / f"window_{index}.npz") as saved:
                by_id = dict(zip(saved["ids"].tolist(), saved["arrays"], strict=True))
                arrays = np.asarray([by_id[key] for key in ids])
            valid = ~np.isnan(labels)
            expected_coverage.append(
                ((labels >= arrays[..., 0]) & (labels <= arrays[..., -1]))[valid].mean()
            )
            scales = np.asarray([np.nanmean(np.abs(np.diff(histories[key]))) for key in ids])
            with np.errstate(divide="ignore", invalid="ignore"):
                errors = np.abs(labels - arrays[..., 4]) / scales[:, None]
            expected_mase.append(errors[np.isfinite(errors)].mean())
        metrics = fev_report.window_metrics(source, spec.to_task())
        assert metrics["coverage_80"] == pytest.approx(expected_coverage)
        assert metrics["MASE"] == pytest.approx(expected_mase)

    # A finite average must not conceal a corrupted prediction cell.
    damaged = tmp_path / "damaged"
    shutil.copytree(source, damaged)
    path = damaged / specs[0].name / "window_0.npz"
    with np.load(path, allow_pickle=False) as saved:
        payload = dict(saved)
    payload["arrays"][0, 0, 0] = np.nan
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="non-finite predictions"):
        fev_report.window_metrics(damaged, specs[0].to_task())
