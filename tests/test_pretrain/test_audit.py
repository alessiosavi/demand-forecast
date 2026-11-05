"""Offline run-11 controls and complete audit orchestration acceptance checks."""

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("pyarrow")
pytest.importorskip("datasets")

from demand_forecast.bench.specs import (  # noqa: E402
    FEV_DATASETS_REVISION,
    FevDatasetTask,
    fev_bench_tasks,
    fev_bench_tasks_path,
)
from demand_forecast.cli import app  # noqa: E402
from demand_forecast.pretrain import audit  # noqa: E402
from demand_forecast.pretrain import audit_match as match
from demand_forecast.pretrain.audit_controls import control_cases, run_control  # noqa: E402
from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest  # noqa: E402
from tests.helpers.audit_mask_reference import verify as mask_verify  # noqa: E402

CASES, RULES = control_cases()


@pytest.mark.parametrize("name", CASES)
def test_run11_controls(name):
    result = run_control(name, CASES[name])
    assert result["ok"], result
    if name == "indexed_1100_times_shift+10_fallback":
        assert result["counts"]["masked_rows"] == 1
        assert result["counts"]["initial_truncated_rows"] == 1
        assert result["counts"]["postpass_rows"] == 1
        assert result["counts"]["unresolved_rows"] == 0
        assert result["counts"]["copies"] == 1100


@pytest.mark.parametrize("name", RULES)
def test_run11_rule_self_tests(name):
    got, expected = RULES[name]
    assert got == expected


def test_control_inventory():
    assert sum(spec[2] for spec in CASES.values()) == 26
    assert sum(not spec[2] for spec in CASES.values()) == 14
    assert len(RULES) == 10


def test_merge_spans_matches_mask_slices():
    rng = np.random.default_rng(2718)
    for _ in range(500):
        length = int(rng.integers(10, 501))
        spans = []
        for _ in range(int(rng.integers(0, 80))):
            start = int(rng.integers(-2 * length, length + 51))
            spans.append((start, start + int(rng.integers(1, 51))))
        # Explicit duplicates, touching, overlap, overrun and out-of-range starts.
        spans.extend([(2, 6), (6, 9), (2, 6), (3, 8), (length - 3, length + 20)])
        spans.extend([(length, length + 10), (length + 5, length + 30)])
        rng.shuffle(spans)
        mask = np.zeros(length, bool)
        for start, end in spans:
            mask[start:end] = True
        assert match.merge_spans(iter(spans), length) == match.intervals_of(mask)
    assert match.merge_spans([], 100) == []
    assert match.merge_spans([(100, 120), (4, 4), (8, 3)], 10) == []
    assert match.merge_spans([(0, 20)], 0) == []


def test_verify_matches_mask_reference_with_multiple_offsets_and_groups():
    rng = np.random.default_rng(1234)
    target = rng.normal(size=128)
    first = np.concatenate((2 * target[:64] + 5, -3 * target[64:] + 9))
    second = np.concatenate((4 * target[:64] - 7, -2 * target[64:] + 3))
    source = np.concatenate((first, second))
    positions = [80, 0, 96, 16, 112, 32, 64, 48, 16, 8, 72]
    hits = [("affine", pos, pos - offset, 16) for offset in (0, -128) for pos in positions]
    scale = max(1.0, float(np.abs(source).max()))
    for offset in (0, -128):
        assert (
            len(
                {
                    match.affine_key(target, source, tp, sp, span, fam, scale)
                    for fam, tp, sp, span in hits
                    if tp - sp == offset
                }
            )
            == 2
        )
    expected = mask_verify(target, source, hits)
    assert expected[0]["is_copy"]
    assert {rec["offset"] for rec in expected[1]} == {0, -128}
    assert {rec["fit_mode"] for rec in expected[1]} == {"grouped"}
    assert match.verify(target, source, hits) == expected


def test_verify_matches_mask_reference_with_clipped_spans():
    target = np.random.default_rng(321).normal(size=128)
    source = target.copy()
    hits = [("exact", pos, pos, span) for pos, span in [(100, 50), (128, 8), (140, 8)]]
    expected = mask_verify(target, source, hits)
    assert expected[0]["t_start"] == 100
    assert expected[0]["t_end"] == 128
    assert match.verify(target, source, hits) == expected


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("15min", "15T"),
        ("15T", "15T"),
        ("30T", "30T"),
        ("60min", "1H"),
        ("H", "1H"),
        ("D", "1D"),
        ("B", "1D"),
        ("W-WED", "1W"),
        ("7D", "1W"),
        ("MS", "1M"),
        ("M", "1M"),
        ("ME", "1M"),
        ("QS-JAN", "1Q"),
        ("Q-DEC", "1Q"),
        ("Y", "1Y"),
        ("YS", "1Y"),
        ("5min", None),
        ("2W", None),
        ("bad", None),
    ],
)
def test_frequency_classes(alias, expected):
    assert audit.frequency_class(alias) == expected


def test_pinned_yaml_identity_and_repeated_configs():
    path = fev_bench_tasks_path()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == "c7160f61a5e1ded66a3954ef1c514d55d13be18534b34fca817356312a6520a9"
    assert sha in Path(str(path)).with_suffix(".PROVENANCE.md").read_text()
    tasks = fev_bench_tasks()
    assert len(tasks) == len({task.name for task in tasks}) == 100
    assert len({(task.dataset_path, task.dataset_config) for task in tasks}) == 96
    assert len(fev_bench_tasks("weekly")) == 7
    assert len(fev_bench_tasks("heldout")) == 5
    task = next(t for t in tasks if t.name == "uk_covid_utla_1W/cumulative")
    assert task.targets == ["cumulative_cases"]
    assert next(t for t in tasks if t.name == "ETT_15T").targets == [
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT",
    ]


def test_remote_loader_revision(monkeypatch, tmp_path):
    loader = Mock()
    monkeypatch.setattr("datasets.load_dataset", loader)
    task = FevDatasetTask("test", "autogluon/fev_datasets", "test")
    audit.load_task_dataset(task, tmp_path)
    loader.assert_called_once_with(
        "autogluon/fev_datasets",
        "test",
        split="train",
        revision=FEV_DATASETS_REVISION,
        cache_dir=str(tmp_path),
    )


def _parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def local_audit(tmp_path):
    """Two shards, two kinds/classes, target/covariate copies and a policy negative."""
    rng = np.random.default_rng(100)
    x = rng.normal(50, 15, 280)
    y = rng.normal(100, 25, 280)
    daily = rng.normal(20, 3, 280)
    negative = np.array([1.0, 2.0, 1.0] + [0.0] * 23)
    root = tmp_path / "corpus"

    def corpus_row(name, values, freq):
        return {
            "id": name,
            "target": values.tolist(),
            "start": pd.Timestamp("2010-01-01"),
            "freq": freq,
            "source": "fixture",
            "folder": name + "-folder",
        }

    _parquet(
        root / "training_corpus/chronos_real/train-00000-of-00001.parquet",
        [
            corpus_row("literal", x[10:90], "D"),
            corpus_row("affine", 3 * y[20:100] - 50, "D"),
            corpus_row("negative", 2 * negative + 100, "D"),
            corpus_row("skip-5min", x, "5min"),
        ],
    )
    weekly = match.weekly_sums(daily, "2020-01-01", 3)
    _parquet(
        root / "training_corpus/gifteval_pretrain/train-00000-of-00001.parquet",
        [
            corpus_row("weekly", weekly, "W-FRI"),
        ],
    )
    write_manifest(corpus_manifest(root), root / "manifest.json")
    (root / "corpus.json").write_text(
        json.dumps({"totals": {"series_kept": 5, "points_kept": 280 + 80 + 80 + 26 + len(weekly)}})
    )
    stamps = pd.date_range("2020-01-01", periods=len(x), freq="D").tolist()
    _parquet(
        tmp_path / "copies.parquet",
        [
            {
                "id": "bench-0",
                "timestamp": stamps,
                "target": x.tolist(),
                "other_target": y.tolist(),
            },
            {
                "id": "bench-duplicate",
                "timestamp": stamps,
                "target": x.tolist(),
                "other_target": y.tolist(),
            },
        ],
    )
    _parquet(
        tmp_path / "covariate.parquet",
        [
            {
                "id": "cov-0",
                "timestamp": stamps,
                "target": rng.normal(size=len(daily)).tolist(),
                "known": daily.tolist(),
            },
        ],
    )
    _parquet(
        tmp_path / "negative.parquet",
        [
            {"id": "negative-0", "timestamp": stamps[: len(negative)], "target": negative.tolist()},
        ],
    )
    definitions = [
        {
            "task_name": "copies",
            "dataset_path": "fixture/fev",
            "dataset_config": "copies",
            "local_path": "copies.parquet",
            "target": ["target", "other_target"],
        },
        {
            "task_name": "covariate-only",
            "dataset_path": "fixture/fev",
            "dataset_config": "cov",
            "local_path": "covariate.parquet",
            "known_dynamic_columns": ["known"],
        },
        {
            "task_name": "negative",
            "dataset_path": "fixture/fev",
            "dataset_config": "negative",
            "local_path": "negative.parquet",
        },
    ]
    task_yaml = tmp_path / "tasks.yaml"
    task_yaml.write_text(yaml.safe_dump({"tasks": definitions}))
    return root, task_yaml


def test_end_to_end_parallel_resume_and_declaration(local_audit, tmp_path, monkeypatch):
    root, tasks = local_audit
    one, two = tmp_path / "one", tmp_path / "two"
    serial_progress, parallel_progress = [], []
    first = audit.run_audit(root, one, tasks=tasks, workers=1, progress=serial_progress.append)
    second = audit.run_audit(root, two, tasks=tasks, workers=2, progress=parallel_progress.append)
    for report, messages in ((first, serial_progress), (second, parallel_progress)):
        completed = [line for line in messages if line.startswith("Scanned ")]
        assert len(completed) == report["summary"]["column_scans"]
        expected_labels = sorted(
            f"Scanned {scan['task']} / {scan['column']} / {scan['variant']} / {scan['kind']}"
            for scan in report["scans"]
        )
        assert sorted(line.rsplit(" in ", 1)[0] for line in completed) == expected_labels
        for line in completed:
            assert re.search(r" in \d+\.\d+s$", line), line
        for path in ((one if report is first else two) / "scans").glob("*.json"):
            assert "elapsed_seconds" not in json.loads(path.read_text())
    for report in (first, second):
        assert report["trained_on_datasets"] == ["copies", "covariate-only"]
        assert report["summary"]["unresolved_rows"] == 0
        assert report["summary"]["skipped_corpus_series"] == 1
        assert report["summary"]["column_scans"] == 40  # five columns × (daily + seven weekly)
        copies = report["copies"]
        exact = [p for p in copies if p["corpus_id"] == "literal" and p["column"] == "target"]
        assert any(
            p["t_start"] == 10
            and p["t_end"] == 90
            and p["index_start"] == 0
            and p["index_end"] == 80
            for p in exact
        )
        assert exact[0]["coef"] == pytest.approx([1, 0], abs=1e-10)
        assert exact[0]["duplicate_rows"] == 2
        assert exact[0]["benchmark_ids"] == ["bench-0", "bench-duplicate"]
        affine = [p for p in copies if p["corpus_id"] == "affine" and p["column"] == "other_target"]
        assert any(p["t_start"] == 20 and p["t_end"] == 100 for p in affine)
        assert affine[0]["coef"] == pytest.approx([3, -50], abs=1e-10)
        assert any(
            p["corpus_id"] == "weekly"
            and p["variant"] == "weekly@3"
            and p["role"] == "covariate"
            and p["segment"] >= 24
            for p in copies
        )
        negative = [r for r in report["policy_excluded_targets"] if r["task"] == "negative"]
        assert negative and all("off_modal_lt4" in r["degenerate_reason"] for r in negative)
        native = next(
            s
            for s in report["scans"]
            if s["task"] == "copies" and s["column"] == "target" and s["variant"] == "native"
        )
        assert native["counts"]["n_series"] == 2
        assert native["counts"]["unique_series"] == 1
        assert all(native["counts"]["windows"][family] > 0 for family in match.FAMILIES)
    assert {k: v for k, v in first.items() if k not in ("generated_at", "elapsed_seconds")} == {
        k: v for k, v in second.items() if k not in ("generated_at", "elapsed_seconds")
    }
    assert json.loads((one / "declaration.json").read_text())["trained_on_datasets"] == [
        "copies",
        "covariate-only",
    ]
    assert "Known limits" in (one / "audit.md").read_text()
    mtimes = {p: p.stat().st_mtime_ns for p in (one / "scans").glob("*.json")}
    monkeypatch.setattr(
        audit, "_scan_job", Mock(side_effect=AssertionError("resume re-scanned a completed column"))
    )
    monkeypatch.setattr(
        audit, "load_task_dataset", Mock(side_effect=AssertionError("resume reloaded a dataset"))
    )
    resumed = audit.run_audit(
        root, one, tasks=tasks, workers=1, resume=True, progress=lambda _: None
    )
    assert resumed["copies"] == first["copies"]
    assert mtimes == {p: p.stat().st_mtime_ns for p in mtimes}


def test_uncapped_postpass_and_oversize_guard(monkeypatch):
    x = np.random.default_rng(1).normal(size=32)
    index = match.Index([x, x, x], ["a", "b", "c"])
    monkeypatch.setattr(match, "CAP", {f: 0 for f in match.FAMILIES})
    monkeypatch.setattr(match, "FALLBACK_TOPK", 1)
    pairs = []
    c = match.scan("test", "target", [x], index, {}, pairs, [])
    assert c["initial_truncated_rows"] == c["postpass_rows"] == 1
    assert c["copies"] == 3 and c["unresolved_rows"] == 0
    assert {p["index_id"] for p in pairs} == {"a", "b", "c"}
    monkeypatch.setattr(match, "FALLBACK_MAX_EXPANSIONS", 1)
    monkeypatch.setattr(
        match, "_expand_arrays", Mock(side_effect=AssertionError("allocated beyond guard"))
    )
    c = match.scan("test", "target", [x], index, {}, [], [])
    assert c["unresolved_rows"] == c["fallback_oversize_rows"] == 1
    assert c["unresolved_ids"] == [0]


def test_dedup_preserves_missingness_and_calendar():
    assert match._row_key([np.inf]) != match._row_key([np.nan])
    assert match._row_key([-1.5e308]) != match._row_key([np.nan])
    x = np.random.default_rng(21).normal(size=30)
    index = match.Index([x], ["x"])
    pairs = []
    c = match.scan(
        "t", "c", [x, x, x], index, {}, pairs, [], dims=[np.full(30, n) for n in (30, 30, 31)]
    )
    assert c["unique_series"] == 2
    assert c["copies"] == 3
    assert sorted(p["duplicate_rows"] for p in pairs) == [1, 2]


def test_resume_rejects_changed_shard(local_audit, tmp_path):
    root, tasks = local_audit
    path = next((root / "training_corpus/chronos_real").glob("*.parquet"))
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="differs from manifest"):
        audit.run_audit(root, tmp_path / "out", tasks=tasks, workers=1, resume=True)


def test_resume_fingerprint_changes_and_incomplete_scan(tmp_path):
    path = tmp_path / "scan.json"
    job = {"fingerprint": "new", "result_path": str(path)}
    path.write_text(json.dumps({"fingerprint": "old", "counts": {"unresolved_rows": 0}}))
    assert audit._resume_result(job) is None
    path.write_text(json.dumps({"fingerprint": "new", "counts": {"unresolved_rows": 1}}))
    assert audit._resume_result(job) is None


def test_cli_exit_codes(monkeypatch, tmp_path):
    mocked = Mock(return_value={"summary": {"copies": 2, "unresolved_rows": 0}})
    monkeypatch.setattr(audit, "run_audit", mocked)
    runner = CliRunner()
    args = ["pretrain", "corpus", "audit", "--root", str(tmp_path), "--out", str(tmp_path / "out")]
    assert runner.invoke(app, args).exit_code == 0
    mocked.return_value["summary"]["unresolved_rows"] = 1
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "1 unresolved" in result.output
    assert runner.invoke(app, ["pretrain", "corpus", "audit"]).exit_code == 2


@pytest.mark.parametrize(
    "manifest,error",
    [
        ({"path": "shard.parquet"}, "string indices must be integers"),
        ([{}], "'path'"),
    ],
    ids=["object_root", "missing_key"],
)
def test_cli_reports_malformed_manifest(tmp_path, manifest, error):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = CliRunner().invoke(
        app,
        ["pretrain", "corpus", "audit", "--root", str(tmp_path), "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert error in result.stderr
    assert len(result.stderr.splitlines()) == 1
    assert result.stdout == ""
    assert "Traceback" not in result.output


def test_rejection_records_are_scan_local_and_target_records_uncapped(monkeypatch):
    x = np.array([1.0, 2.0, 1.0] + [0.0] * 23)
    index = match.Index([2 * x + 100], ["policy"])
    monkeypatch.setattr(match, "MAX_RECORDS", 0)
    records = []
    c = match.scan("task", "target", [x], index, {}, [], records, all_rejections=True)
    assert records and c["rejection_records_omitted"] == 0
    records = []
    c = match.scan("task", "covariate", [x], index, {}, [], records)
    assert records == [] and c["rejection_records_omitted"] > 0


def test_monthly_variants_preserve_calendar_and_original_row_ids(tmp_path):
    root = tmp_path / "corpus"
    rng = np.random.default_rng(0)
    daily = np.abs(rng.standard_normal(1400)).cumsum() + rng.standard_normal(1400) * 3
    start = pd.Timestamp("2011-03-02")
    sums, means, _ = match.monthly_aggs(daily, start)
    _parquet(
        root / "training_corpus/chronos_real/train-00000-of-00001.parquet",
        [
            {
                "id": name,
                "target": values.tolist(),
                "start": pd.Timestamp("2011-03-01"),
                "freq": "MS",
                "source": "fixture",
                "folder": "months",
            }
            for name, values in [("sum", sums + 5), ("mean", means * 2.5 + 1)]
        ],
    )
    write_manifest(corpus_manifest(root), root / "manifest.json")
    stamps = pd.date_range(start, periods=len(daily), freq="D").tolist()
    _parquet(
        tmp_path / "daily.parquet",
        [
            {"id": "too-short", "timestamp": stamps[:60], "target": daily[:60].tolist()},
            {"id": "long", "timestamp": stamps, "target": daily.tolist()},
        ],
    )
    _parquet(
        tmp_path / "monthly.parquet",
        [
            {
                "id": "native",
                "timestamp": pd.date_range("2011-03-01", periods=len(sums), freq="MS").tolist(),
                "target": sums.tolist(),
            },
        ],
    )
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "task_name": "daily",
                        "dataset_path": "fixture/fev",
                        "local_path": "daily.parquet",
                    },
                    {
                        "task_name": "monthly",
                        "dataset_path": "fixture/fev",
                        "local_path": "monthly.parquet",
                    },
                ]
            }
        )
    )
    report = audit.run_audit(
        root,
        tmp_path / "out",
        tasks=tasks,
        kinds=["chronos_real"],
        workers=1,
        progress=lambda _: None,
    )
    assert report["trained_on_datasets"] == ["daily", "monthly"]
    for variant in ("monthly@sum", "monthly@mean"):
        scan = next(s for s in report["scans"] if s["variant"] == variant)
        assert scan["counts"]["aggregation_ineligible_rows"] == 1
        assert scan["copies"]
        assert all(
            p["benchmark_ids"] == ["long"] and p["series_indices"] == [1] for p in scan["copies"]
        )
    assert any(p["task"] == "monthly" and p["variant"] == "native" for p in report["copies"])


def test_near_segments_are_retained_alongside_copied_prefixes():
    x = np.random.default_rng(0).normal(100, 100, 120)
    y = x.copy()
    y[60] += 4
    y[80] -= 4
    pairs = []
    c = match.scan("near", "target", [x], match.Index([y], ["y"]), {}, pairs, [])
    assert c["near_pairs"] >= 1
    near = [p for p in pairs if not p["is_copy"]]
    assert near and all(
        p["r2"] >= 0.999 and p["max_resid"] > max(0.01, 0.005 * p["index_range"]) for p in near
    )


def test_real_cli_reports_unresolved_after_postpass(local_audit, tmp_path, monkeypatch):
    root, tasks = local_audit
    monkeypatch.setattr(match, "CAP", {family: 0 for family in match.FAMILIES})
    monkeypatch.setattr(match, "FALLBACK_MAX_EXPANSIONS", 1)
    out = tmp_path / "unresolved"
    result = CliRunner().invoke(
        app,
        [
            "pretrain",
            "corpus",
            "audit",
            "--root",
            str(root),
            "--out",
            str(out),
            "--tasks",
            str(tasks),
            "--workers",
            "1",
        ],
    )
    assert result.exit_code == 2, result.output
    report = json.loads((out / "audit.json").read_text())
    assert report["summary"]["unresolved_rows"] > 0
    assert report["summary"]["postpass_rows"] > 0
    assert any(s["unresolved"] for s in report["scans"])
    assert json.loads((out / "declaration.json").read_text())["complete"] is False


def test_nonfinite_rejection_residual_is_valid_json(tmp_path):
    record = match.fit_segment(np.ones(24), np.ones(24), 0, 24, 0)
    assert record["degenerate_copy_rejected"]
    audit._write_json(tmp_path / "record.json", record)
    assert json.loads((tmp_path / "record.json").read_text())["max_resid"] is None
