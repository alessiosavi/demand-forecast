"""Offline comparison, pinned CSV schema, and real saved-window diagnostics."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fev")
pytestmark = pytest.mark.bench

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


def test_coverage_from_window_metrics(tmp_path, report_template):  # noqa: F811 -- pytest fixture
    report_module()
    specs, source, _, _ = report_template
    summary = pd.read_csv(source / "summaries.csv")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    summary.loc[summary.model_name == "seasonal_naive"].to_csv(
        baseline / "summaries.csv", index=False
    )
    candidates = summary.loc[summary.model_name != "seasonal_naive"].copy()
    candidates["coverage_80"] = 999  # Neither metric may be read from this column.
    candidates["MASE"] = 999
    published = tmp_path / "published"
    published.mkdir()
    for i in range(43):
        frame = candidates.assign(model_name=f"synthetic-{i:02d}")
        if i in (0, 2):
            frame.loc[frame.index[0], "dataset_fingerprint"] = "different" if i == 0 else None
        elif i == 1:
            frame = frame.drop(columns="num_forecasts").assign(fev_version="0.6.1")
        frame.to_csv(published / f"synthetic-{i:02d}.csv", index=False)
    candidates.assign(model_name="Seasonal Naive").to_csv(
        published / "seasonal_naive.csv", index=False
    )
    pretrain = tmp_path / "pretrain-bundle"
    pretrain.mkdir()
    (pretrain / "manifest.json").write_text(
        json.dumps(
            {
                "pretrain_identity": "pretraining-identity",
                "best_step": 17,
                "git_sha": "pretrain-sha",
                "external_diagnostic": {"name": "nn5", "ext_sql": 0.7, "ext_sql_naive": 1.2},
            }
        )
    )
    args = []
    for name in ("frozen", "zero-shot", "finetune", "wide-random", "heldout-frozen"):
        run_dir = tmp_path / name
        shutil.copytree(source, run_dir)
        selected = (
            candidates
            if name == "zero-shot"
            else candidates.loc[
                candidates.task == ("second" if name == "heldout-frozen" else "first")
            ]
        )
        selected.assign(model_name=name).to_csv(run_dir / "summaries.csv", index=False)
        if name in ("zero-shot", "finetune"):
            for spec in specs:
                path = run_dir / spec.name / "run.json"
                record = json.loads(path.read_text())
                record["settings"]["training"]["init_from"] = str(pretrain)
                record["transfer"] = {"pretrain_identity": "pretraining-identity"}
                path.write_text(json.dumps(record))
        args.extend([f"--{name}", str(run_dir)])
    spend = tmp_path / "spend.json"
    spend.write_text(
        json.dumps(
            {
                "runs": {
                    name: {"spend_usd": 1.25, "wall_time_s": 73}
                    for name in ("frozen", "zero-shot", "finetune", "wide-random", "heldout-frozen")
                }
            }
        )
    )
    output = tmp_path / "REPORT.md"
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *args,
            "--baseline",
            str(baseline),
            "--published",
            str(published),
            "--fev-commit",
            "193e4fc",
            "--spend",
            str(spend),
            "--out",
            str(output),
            "--leaked",
            "solar_1W",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = output.read_text()
    rows = table(report, "Comparison")
    assert len(rows) == 6
    assert {row["block"] for row in rows} == {"frozen tasks", "held-out tasks"}
    comparability = table(report, "Comparability")
    assert len(comparability) == 43
    assert [row["model_name"] for row in comparability] == [f"synthetic-{i:02d}" for i in range(43)]
    for i, row in enumerate(comparability):
        assert row["fingerprint matches"] == ("1/2" if i in (0, 2) else "2/2")
        assert row["num_forecasts"] == ("unreported" if i == 1 else "equal")
        assert row["fev_version"] == ("0.6.1" if i == 1 else "0.10.0")
    assert (
        report.index("## Comparison")
        < report.index("## Comparability")
        < report.index("## Leakage")
    )
    assert "different revision" in report and "where reported" in report
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
        for row in rows:
            if spec.name in row["per-task SQL"]:
                assert float(row["mean coverage_80"]) == pytest.approx(np.mean(expected_coverage))
                assert float(row["mean MASE"]) == pytest.approx(np.mean(expected_mase))
                assert row["spend USD"] == "1.25" and row["wall time s"] == "73"
    for term in (
        "post_hoc: no",
        "solar_1W",
        "193e4fc",
        "pretraining-identity",
        "17",
        "params",
        "0.7",
        "1.2",
        "training time s",
        "inference time s",
        "pretrain-sha",
    ):
        assert term in report
    provenance = table(report, "Provenance")
    assert all(int(row["params"]) > 0 for row in provenance)
    transferred = [row for row in provenance if row["run"] in ("zero-shot", "finetune")]
    assert all(row["pretrain best step"] == "17" for row in transferred)
    assert all(row["external SQL"] == "0.7" for row in transferred)
    assert all(row["external naive SQL"] == "1.2" for row in transferred)

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
