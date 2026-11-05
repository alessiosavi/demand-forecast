"""Offline run-11 controls and complete audit orchestration acceptance checks."""

import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
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
    np.testing.assert_equal(match.verify(target, source, hits), expected)


def test_verify_matches_mask_reference_with_clipped_spans():
    target = np.random.default_rng(321).normal(size=128)
    source = target.copy()
    hits = [("exact", pos, pos, span) for pos, span in [(100, 50), (128, 8), (140, 8)]]
    expected = mask_verify(target, source, hits)
    assert expected[0]["t_start"] == 100
    assert expected[0]["t_end"] == 128
    np.testing.assert_equal(match.verify(target, source, hits), expected)


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


def test_worker_dataset_caches_multiple_tasks(monkeypatch, tmp_path):
    loader = Mock(side_effect=lambda *_: object())
    monkeypatch.setattr(audit, "load_task_dataset", loader)
    tasks = [json.dumps({"name": name, "dataset_path": name}) for name in ("first", "second")]
    audit._worker_dataset.cache_clear()
    try:
        first, second = [audit._worker_dataset(task, str(tmp_path)) for task in tasks]
        assert audit._worker_dataset(tasks[0], str(tmp_path)) is first
        assert audit._worker_dataset(tasks[1], str(tmp_path)) is second
        assert loader.call_count == 2
        assert audit._worker_dataset(tasks[0], str(tmp_path / "other")) is not first
        assert loader.call_count == 3
    finally:
        audit._worker_dataset.cache_clear()


def _init_worker_with_hf_imported(*args):
    """Exercise spawn callers that import HF before the pool initializer runs."""
    from datasets import config
    from huggingface_hub import constants

    assert not config.HF_DATASETS_OFFLINE
    assert not constants.HF_HUB_OFFLINE
    audit._init_worker(*args)


def _worker_offline_state():
    from datasets import config
    from huggingface_hub import constants

    return (
        os.getpid(),
        os.environ.get("HF_HUB_OFFLINE"),
        os.environ.get("HF_DATASETS_OFFLINE"),
        config.HF_HUB_OFFLINE,
        config.HF_DATASETS_OFFLINE,
        constants.HF_HUB_OFFLINE,
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


@pytest.mark.parametrize("reuse_out", [False, True], ids=["new_out", "same_out"])
def test_serial_audit_reloads_changed_dataset(local_audit, tmp_path, monkeypatch, reuse_out):
    import pyarrow.parquet as pq

    root, tasks = local_audit
    out = tmp_path / "out"
    loader = Mock(wraps=audit.load_task_dataset)
    monkeypatch.setattr(audit, "load_task_dataset", loader)
    audit._worker_dataset.cache_clear()
    try:
        first = audit.run_audit(root, out, tasks=tasks, workers=1, progress=lambda _: None)
        first_cache_size = audit._worker_dataset.cache_info().currsize
        definitions = fev_bench_tasks(tasks)
        # Each task is loaded for metadata and once more for all serial scans.
        assert loader.call_count == 2 * len(definitions)

        path = tmp_path / "copies.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["target"] = (np.asarray(row["target"]) + 100).tolist()
        previous_mtime_ns = path.stat().st_mtime_ns
        _parquet(path, rows)
        # Invalidate datasets' local-file cache even on coarse-timestamp filesystems.
        os.utime(path, ns=(path.stat().st_atime_ns, previous_mtime_ns + 2_000_000_000))
        loader.reset_mock()
        second_out = out if reuse_out else tmp_path / "second"
        second = audit.run_audit(root, second_out, tasks=tasks, workers=1, progress=lambda _: None)

        for report, coef in ((first, [1, 0]), (second, [1, -100])):
            literal = [
                copy
                for copy in report["copies"]
                if copy["corpus_id"] == "literal" and copy["column"] == "target"
            ]
            assert literal
            assert all(copy["coef"] == pytest.approx(coef, abs=1e-10) for copy in literal)
        assert second["local_dataset_sha256"]["copies"] != first["local_dataset_sha256"]["copies"]
        assert loader.call_count == 2 * len(definitions)
        assert {Path(call.args[1]) for call in loader.call_args_list} == {
            second_out / ".cache" / "datasets" / FEV_DATASETS_REVISION
        }
        assert first_cache_size == 0
        assert audit._worker_dataset.cache_info().currsize == 0
    finally:
        audit._worker_dataset.cache_clear()


def test_serial_audit_clears_datasets_on_entry_and_error(local_audit, tmp_path):
    root, tasks = local_audit
    task = fev_bench_tasks(tasks)[0]
    audit._worker_dataset.cache_clear()
    try:
        audit._worker_dataset(json.dumps(asdict(task)), str(tmp_path / "prewarmed"))
        assert audit._worker_dataset.cache_info().currsize == 1

        def interrupt_scan(message):
            if message.startswith("Reading task schema"):
                assert audit._worker_dataset.cache_info().currsize == 0
            if message.startswith("Scanned "):
                assert audit._worker_dataset.cache_info().currsize > 0
                raise RuntimeError("interrupted after loading dataset")

        with pytest.raises(RuntimeError, match="interrupted after loading dataset"):
            audit.run_audit(root, tmp_path / "out", tasks=tasks, workers=1, progress=interrupt_scan)
        assert audit._worker_dataset.cache_info().currsize == 0
    finally:
        audit._worker_dataset.cache_clear()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("offline_env", [None, "0"])
def test_audit_workers_offline_without_changing_parent(
    local_audit, tmp_path, monkeypatch, workers, offline_env
):
    root, tasks = local_audit
    # Warm lazy loader imports (threadpoolctl sets an unrelated OpenMP env var).
    audit.load_task_dataset(fev_bench_tasks(tasks)[0], audit._dataset_cache_dir(tmp_path / "out"))
    for flag in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if offline_env is None:
            monkeypatch.delenv(flag, raising=False)
        else:
            monkeypatch.setenv(flag, offline_env)
    parent_env = dict(os.environ)
    parent_state = _worker_offline_state()
    states = []
    parallel_scan_jobs = audit._parallel_scan_jobs

    def record_offline_state(pool, *args, **kwargs):
        states.append(pool.submit(_worker_offline_state).result(timeout=30))
        yield from parallel_scan_jobs(pool, *args, **kwargs)

    def pool_with_hf_imported(**kwargs):
        assert kwargs["initializer"] is audit._init_worker
        return ProcessPoolExecutor(**{**kwargs, "initializer": _init_worker_with_hf_imported})

    monkeypatch.setattr(audit, "ProcessPoolExecutor", pool_with_hf_imported)
    monkeypatch.setattr(audit, "_parallel_scan_jobs", record_offline_state)
    report = audit.run_audit(root, tmp_path / "out", tasks=tasks, workers=workers)

    assert report["trained_on_datasets"] == ["copies", "covariate-only"]
    assert dict(os.environ) == parent_env
    assert _worker_offline_state() == parent_state
    assert len(states) == (2 if workers == 2 else 0)  # One pool per kind/frequency class.
    for pid, *flags in states:
        assert pid != os.getpid()
        assert flags == ["1", "1", True, True, True]


def test_parallel_resume_caches_datasets_before_starting_workers(
    local_audit, tmp_path, monkeypatch
):
    root, tasks = local_audit
    out = tmp_path / "out"
    first = audit.run_audit(root, out, tasks=tasks, workers=1, progress=lambda _: None)
    cache_dir = audit._dataset_cache_dir(out)
    shutil.rmtree(cache_dir)
    shutil.rmtree(out / "scans")
    loader = Mock(wraps=audit.load_task_dataset)
    monkeypatch.setattr(audit, "load_task_dataset", loader)
    monkeypatch.setattr(
        audit, "_task_metadata", Mock(side_effect=AssertionError("metadata should be reused"))
    )
    definitions = fev_bench_tasks(tasks)

    def pool_after_cache(**kwargs):
        assert loader.call_count == len(definitions)
        assert [call.args for call in loader.call_args_list] == [
            (task, cache_dir) for task in definitions
        ]
        assert len(list(cache_dir.rglob("*.arrow"))) == len(definitions)
        return ProcessPoolExecutor(**kwargs)

    monkeypatch.setattr(audit, "ProcessPoolExecutor", pool_after_cache)
    messages = []
    resumed = audit.run_audit(
        root, out, tasks=tasks, workers=2, resume=True, progress=messages.append
    )
    assert resumed["copies"] == first["copies"]
    assert messages.count("Task datasets cached: 3") == 1


def test_end_to_end_parallel_resume_and_declaration(local_audit, tmp_path, monkeypatch):
    root, tasks = local_audit
    one, two = tmp_path / "one", tmp_path / "two"
    serial_progress, parallel_progress = [], []

    def track_release(out, messages):
        released = []

        def progress(line):
            if line.startswith("Indexing "):
                assert all(not path.exists() for path in released)
            if line.startswith("Released "):
                _, kind, cls, _ = line.split()
                work = next(out.glob(".audit-work-*"))
                path = work / kind / cls
                assert not path.exists()
                if out == one:
                    assert audit._WORKER_INDEX is None
                released.append(path)
            messages.append(line)

        return progress

    first = audit.run_audit(
        root, one, tasks=tasks, workers=1, progress=track_release(one, serial_progress)
    )
    monkeypatch.setattr(audit, "SCAN_CHUNK_ROWS", 1)
    monkeypatch.setattr(audit, "SCAN_CHUNK_MIN_GROUPS", 1)
    second = audit.run_audit(
        root, two, tasks=tasks, workers=2, progress=track_release(two, parallel_progress)
    )
    assert {p.name: p.read_bytes() for p in (one / "scans").glob("*.json")} == {
        p.name: p.read_bytes() for p in (two / "scans").glob("*.json")
    }
    for report, messages in ((first, serial_progress), (second, parallel_progress)):
        assert "Released chronos_real 1D index" in messages
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
    resume_progress = []
    resumed = audit.run_audit(
        root,
        one,
        tasks=tasks,
        workers=1,
        resume=True,
        progress=track_release(one, resume_progress),
    )
    assert "Released chronos_real 1D index" in resume_progress
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


def test_parallel_multiple_chunks_and_whole_jobs_match_serial(local_audit, tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    root, tasks = local_audit
    definitions = yaml.safe_load(tasks.read_text())
    definitions["tasks"] = [t for t in definitions["tasks"] if t["task_name"] == "copies"]
    tasks.write_text(yaml.safe_dump(definitions))
    path = tmp_path / "copies.parquet"
    rows = pq.read_table(path).to_pylist()
    for shift in (10, 20, 30):
        rows.append(
            {
                **rows[0],
                "id": f"shift-{shift}",
                "target": (np.asarray(rows[0]["target"]) + shift).tolist(),
                "other_target": (np.asarray(rows[0]["other_target"]) + shift).tolist(),
            }
        )
    _parquet(path, rows)
    expected = None
    submitted = []
    executor = audit.ProcessPoolExecutor

    def observed_pool(**kwargs):
        assert kwargs["max_workers"] == 2
        pool = executor(**kwargs)
        submit = pool.submit

        def observed_submit(fn, payload):
            submitted.append((fn.__name__, len(payload.get("groups", []))))
            return submit(fn, payload)

        pool.submit = observed_submit
        return pool

    monkeypatch.setattr(audit, "ProcessPoolExecutor", observed_pool)
    for label, workers, threshold, size in (
        ("serial", 1, 1, 1),
        ("whole", 2, 256, 32),
        ("chunk1", 2, 1, 1),
        ("chunk2", 2, 1, 2),
        ("chunk_all", 2, 1, 32),
    ):
        monkeypatch.setattr(audit, "SCAN_CHUNK_ROWS", threshold)
        monkeypatch.setattr(audit, "SCAN_CHUNK_MIN_GROUPS", size)
        submitted.clear()
        out = tmp_path / label
        audit.run_audit(
            root, out, tasks=tasks, kinds=["chronos_real"], workers=workers, progress=lambda _: None
        )
        scans = {p.name: p.read_bytes() for p in (out / "scans").glob("*.json")}
        if expected is None:
            expected = scans
        assert scans == expected
        if label.startswith("chunk"):
            assert ("_chunk_job", min(size, 4)) in submitted
        elif label == "whole":
            assert all(name == "_timed_scan_job" for name, _ in submitted)
        else:
            assert not submitted


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


def test_monthly_variants_preserve_calendar_and_original_row_ids(tmp_path, monkeypatch):
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
    monkeypatch.setattr(audit, "SCAN_CHUNK_ROWS", 1)
    monkeypatch.setattr(audit, "SCAN_CHUNK_MIN_GROUPS", 1)
    audit.run_audit(
        root,
        tmp_path / "chunked",
        tasks=tasks,
        kinds=["chronos_real"],
        workers=2,
        progress=lambda _: None,
    )
    assert {p.name: p.read_bytes() for p in (tmp_path / "out/scans").glob("*.json")} == {
        p.name: p.read_bytes() for p in (tmp_path / "chunked/scans").glob("*.json")
    }


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
