"""Packed lookup keeps exact 128-bit ranges, cap diagnostics and proposal order."""

import sqlite3
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from scripts.corpus_v4 import audit_sketch as s  # noqa: E402


def _reference(rows, keys):
    """Literal posting walk, independent of either binary search implementation."""
    pairs, summaries, heavy = {}, [], []
    for token, pos, label in keys:
        hi, lo = int(token[:16], 16), int(token[16:], 16)
        hits = [r for r in rows if int(r["hi"]) == hi and int(r["lo"]) == lo]
        if len(hits) > s.POSTING_CAP:
            heavy.append(
                dict(
                    key=token,
                    family=label,
                    postings=len(hits),
                    retained=s.POSTING_CAP,
                    reason="posting_cap",
                )
            )
        for row in hits[: s.POSTING_CAP]:
            owner, off = int(row["owner"]), pos - int(row["pos"])
            if label == "summary":
                summaries.append(owner)
            else:
                pairs.setdefault((owner, off), s.Candidate(off, label))
    return sorted(pairs.items()), sorted(set(summaries)), heavy


def test_lookup_exact_word_boundaries_caps_duplicates_and_missing_keys(tmp_path, monkeypatch):
    maximum = (1 << 64) - 1
    records = [(0, 0, 1, 0), (maximum, maximum, 2, 90)]
    for lo, count in ((0, 64), (1, 65), (2, 63), (maximum, 131)):
        records.extend((7, lo, owner % 17 + 1, owner % 11) for owner in range(count))
    rows = np.asarray(records, dtype=s.POSTING)
    rows.sort(order=["hi", "lo", "owner", "pos"])
    rows.tofile(tmp_path / "postings.bin")
    np.zeros((17, s.FEATURES), dtype="<f4").tofile(tmp_path / "features.bin")
    spool = tmp_path / "benchmark.sqlite"
    sqlite3.connect(spool).close()
    spool.with_suffix(".values").touch()
    spool.with_suffix(".stamps").touch()
    keys = tuple(
        (f"{hi:016x}{lo:016x}", pos, label)
        for hi, lo, pos, label in (
            (0, 0, 0, "raw"),
            (7, 0, 0, "prefix/24"),
            (7, 1, 40, "suffix/24"),
            (7, 1, 40, "suffix/64"),
            (7, 2, 0, "summary"),
            (7, maximum, 0, "summary"),
            (7, 3, 0, "absent-low-word"),
            (8, 0, 0, "absent-high-word"),
            (maximum, maximum, 100, "suffix/256"),
        )
    )
    query = replace(s.sketch(np.arange(400.0)), keys=keys)
    expected = _reference(rows, keys)
    index = s.SketchIndex(tmp_path, spool)
    searchsorted = np.searchsorted

    def observed(array, *args, **kwargs):
        # A padded structured input makes NumPy pack the WHOLE index for each
        # search. Scalar-word views must stay views of the mapped posting file.
        assert array.dtype.names is None
        assert np.shares_memory(array, index.rows)
        return searchsorted(array, *args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", observed)
    try:
        assert index.lookup(query) == expected
        assert index.lookup(replace(query, keys=())) == ([], [], [])
    finally:
        index.close()


def test_profile_comparison_detects_compressed_byte_or_report_changes(tmp_path):
    from scripts.corpus_v4.profile_sketch import _compare

    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    for directory in (before, after):
        (directory / "journal.jsonl.gz").write_bytes(b"compressed-journal")
        (directory / "result.json").write_text('{"count": 1}\n')
    assert _compare(before, after)["identical"]
    (after / "result.json").write_text('{"count": 2}\n')
    assert _compare(before, after)["differences"] == ["result.json"]
    (after / "journal.jsonl.gz").write_bytes(b"changed-compressed-journal")
    assert not _compare(before, after)["identical"]


def test_series_reader_matches_mappings_and_bounds_the_owner_cache(tmp_path, monkeypatch):
    import pandas as pd

    from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries

    rng = np.random.default_rng(11)
    values = rng.normal(size=400)
    values[::17] = np.nan
    stamps = pd.date_range("2000", periods=len(values), freq="ME")
    spool, identity = s.prepare_benchmarks(
        tmp_path / "state",
        [BenchmarkSeries("monthly", str(i), "x", "ME", values + i, stamps) for i in range(4)],
    )
    directory = tmp_path / "index"
    s.build_index(spool, directory, identity, memory_budget_bytes=2 * 1024**3)
    monkeypatch.setattr(s, "SERIES_CACHE_ENTRIES", 2)
    index = s.SketchIndex(directory, spool)
    try:
        for owner in (1, 2, 1, 3):
            expected = s._series(index.db, spool, owner)
            actual = index.series(owner)
            assert actual[0] == expected[0]
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_array_equal(actual[2], expected[2])
            assert not actual[1].flags.writeable
        assert list(index.series_cache) == [1, 3]
        assert index.series(1) is index.series(1)
        assert index.series_cache_bytes <= s.SERIES_CACHE_BYTES
        # Enforce the byte bound independently of the entry-count bound.
        monkeypatch.setattr(s, "SERIES_CACHE_BYTES", index.series_cache_bytes // 2)
        index.series(4)
        assert index.series_cache_bytes <= s.SERIES_CACHE_BYTES
        # Oversized owners retain the original disposable mapping path.
        monkeypatch.setattr(s, "SERIES_READ_BYTES", 8)
        oversized = index.series(2)
        assert isinstance(oversized[1], np.memmap)
        assert 2 not in index.series_cache
        np.testing.assert_array_equal(oversized[1], values + 1)
    finally:
        index.close()
    assert not index.series_cache
    assert index.values_file.closed and index.stamps_file.closed
