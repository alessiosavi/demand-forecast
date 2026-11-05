"""Row-group chunking preserves scan order, accounting and record limits."""

import pickle
from dataclasses import asdict
from itertools import chain

import numpy as np
import pytest

from demand_forecast.bench.specs import FevDatasetTask
from demand_forecast.pretrain import audit
from demand_forecast.pretrain import audit_match as match


@pytest.mark.parametrize("all_rejections", [False, True])
@pytest.mark.parametrize("caps", [(2, 1), (2, 100), (100, 1)])
def test_chunk_record_caps_match_serial(monkeypatch, all_rejections, caps):
    x = np.array([1.0, 2.0, 1.0] + [0.0] * 23)
    rows = [x + shift for shift in range(5)] + [x.copy()]
    index = match.Index([2 * x + 100], ["policy"])
    monkeypatch.setattr(match, "MAX_RECORDS", caps[0])
    monkeypatch.setattr(match, "MAX_RECORDS_PER_REASON", caps[1])
    pairs, records = [], []
    expected = match.scan("t", "c", rows, index, {}, pairs, records, all_rejections=all_rejections)
    assert expected["degenerate_rejections"] >= 5
    assert bool(expected["rejection_records_omitted"]) is not all_rejections
    groups = match.group_rows(rows)
    assert groups == [(0, [0, 5]), (1, [1]), (2, [2]), (3, [3]), (4, [4])]
    for size in (1, 2, len(groups)):
        chunks = [
            [match.scan_group("t", "c", i, idxs, rows, index) for i, idxs in groups[j : j + size]]
            for j in range(0, len(groups), size)
        ]
        # Process workers serialize results before the parent merges them.
        chunks = pickle.loads(pickle.dumps(chunks))
        counts = match.scan("t", "c", [], index, {}, [], [])
        counts["unique_series"] = len(groups)
        actual_pairs, actual_records = [], []
        actual = match.finish_scan(
            counts,
            chain.from_iterable(chunks),
            actual_pairs,
            actual_records,
            all_rejections=all_rejections,
        )
        assert actual == expected
        assert audit._json_bytes((actual_pairs, actual_records)) == audit._json_bytes(
            (pairs, records)
        )


def test_order_pending_uses_eligible_work_and_job_ties():
    metadata = [
        {
            "task": {"name": "a"},
            "rows": [
                {"frequency_class": "1D", "length": 60, "daily_regular": True},
                {"frequency_class": "1D", "length": 400, "daily_regular": False},
                {"frequency_class": "1H", "length": 1000, "daily_regular": False},
            ],
        },
        {
            "task": {"name": "b"},
            "rows": [
                {"frequency_class": "1D", "length": 60, "daily_regular": True},
            ],
        },
    ]
    jobs = [
        {
            "task": {"name": name},
            "kind": "chronos_real",
            "column": column,
            "frequency_class": cls,
            "variant": variant,
        }
        for name, column, cls, variant in [
            ("b", "target", "1W", "weekly@0"),
            ("a", "target", "1W", "weekly@1"),
            ("a", "target", "1D", "native"),
            ("a", "target", "1H", "native"),
            ("a", "other", "1W", "weekly@0"),
        ]
    ]
    assert audit._job_work(jobs[0], metadata[1]["rows"]) == 60
    assert audit._job_work(jobs[1], metadata[0]["rows"]) == 60
    assert audit._order_pending(jobs, metadata) == [jobs[i] for i in (3, 2, 4, 1, 0)]


def test_grouping_needs_no_index_and_chunks_read_only_representatives(tmp_path, monkeypatch):
    daily = np.random.default_rng(8).normal(size=280)
    metadata = [
        {
            "row": i,
            "frequency_class": "1D",
            "daily_regular": True,
            "start": "2020-01-01",
            "length": length,
        }
        for i, length in enumerate((20, 280, 20, 280))
    ]
    reads = []

    class Dataset:
        def select_columns(self, columns):
            return self

        def __getitem__(self, i):
            reads.append(i)
            return {"target": daily[: metadata[i]["length"]] + i}

    path = tmp_path / "metadata.json"
    audit._write_json(path, {"rows": metadata})
    job = {
        "task": asdict(FevDatasetTask("t", "local")),
        "column": "target",
        "frequency_class": "1W",
        "variant": "weekly@0",
        "metadata_path": str(path),
        "cache_dir": str(tmp_path),
    }
    monkeypatch.setattr(audit, "_worker_dataset", lambda *args: Dataset())
    monkeypatch.setattr(audit, "_WORKER_INDEX", None)
    grouped = audit._group_job(job)
    assert grouped["kept"] == [1, 3]
    assert grouped["groups"] == [(0, [0]), (1, [1])]
    reads.clear()
    monkeypatch.setattr(audit, "_WORKER_INDEX", match.Index([], []))
    results = audit._chunk_job(
        {"job": job, "kept": grouped["kept"], "groups": grouped["groups"][1:]}
    )
    assert [r["i"] for r in results] == [1]
    assert reads == [3]


@pytest.mark.parametrize("resolve,guard", [(False, 50_000_000), (True, 50_000_000), (True, 1)])
def test_chunk_postpass_accounting_and_unresolved_ids(monkeypatch, resolve, guard):
    x = np.random.default_rng(9).normal(size=32)
    rows = [x, x + 10, x.copy()]
    index = match.Index([x, x.copy()], ["a", "b"])
    monkeypatch.setattr(match, "CAP", dict.fromkeys(match.FAMILIES, 0))
    monkeypatch.setattr(match, "FALLBACK_TOPK", 1)
    monkeypatch.setattr(match, "FALLBACK_MAX_EXPANSIONS", guard)
    pairs, records = [], []
    expected = match.scan("t", "c", rows, index, {}, pairs, records, resolve=resolve)
    groups = match.group_rows(rows)
    counts = match.scan("t", "c", [], index, {}, [], [])
    counts["unique_series"] = len(groups)
    actual_pairs, actual_records = [], []
    actual = match.finish_scan(
        counts,
        (match.scan_group("t", "c", i, idxs, rows, index, resolve=resolve) for i, idxs in groups),
        actual_pairs,
        actual_records,
    )
    assert actual == expected
    assert (actual_pairs, actual_records) == (pairs, records)
    assert expected["initial_unresolved_rows"] == len(rows)
    assert expected["postpass_rows"] == (len(rows) if resolve else 0)
    assert expected["unresolved_rows"] == (0 if resolve and guard > 1 else len(rows))
