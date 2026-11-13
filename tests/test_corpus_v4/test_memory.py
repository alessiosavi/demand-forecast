"""Offline large-page/long-row regressions; RSS is measured in a fresh process."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.corpus_v4.native import iter_rows, measure_folder  # noqa: E402


def large_shard(path, kind, rows, *, steps=2_100_001):
    """Same release nesting/types; fewer rows, deliberately longer than the average."""
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 1 if kind == "univariate" else 3
    dtype = np.float32 if kind == "univariate" else np.float64
    leaf = pa.array(np.zeros(rows * steps * channels, dtype=dtype))
    series = pa.ListArray.from_arrays(
        pa.array(np.arange(rows * channels + 1, dtype=np.int64) * steps, type=pa.int32()), leaf
    )
    target = (
        pa.ListArray.from_arrays(pa.array(np.arange(rows + 1, dtype=np.int32)), series)
        if kind == "univariate"
        else pa.ListArray.from_arrays(
            pa.array(np.arange(rows + 1, dtype=np.int64) * steps, type=pa.int32()),
            leaf.slice(0, rows * steps),
        )
    )
    columns = dict(
        item_id=[str(i) for i in range(rows)],
        source=["fixture"] * rows,
        freq=["h"] * rows,
        target=target,
    )
    if kind == "univariate":
        columns.update(domain=["x"] * rows, subdomain=["y"] * rows)
    else:
        cov_series = series.slice(rows, rows * 2)
        columns["covariates"] = pa.ListArray.from_arrays(
            pa.array(np.arange(rows + 1, dtype=np.int32) * 2), cov_series
        )
        columns["covariate_names"] = [["a", "b"]] * rows
    pq.write_table(
        pa.table(columns),
        path,
        row_group_size=rows,
        use_dictionary=False,
        compression="snappy",
        data_page_size=64 * 1024,
    )
    return steps


@pytest.mark.parametrize("kind,rows", [("univariate", 68), ("covariates", 12)])
def test_large_group_long_rows_measured_and_scanned_with_bounded_rss(tmp_path, kind, rows):
    steps = large_shard(tmp_path / "train-000.parquet", kind, rows)
    assert (
        pq.ParquetFile(tmp_path / "train-000.parquet").metadata.row_group(0).total_byte_size
        > 512 * 1024**2
    )
    script = """
import json,sys
from scripts.corpus_v4.native import measure_folder, iter_rows, peak_rss_bytes
from scripts.corpus_v4.matching import chunks, find_index_matches
from demand_forecast.pretrain.audit_match import Index
stats = measure_folder(sys.argv[1], sys.argv[2], memory_budget_bytes=512*1024**2)
index = Index([__import__('numpy').zeros(96)], ['fixture'])
scanned = 0
for row in iter_rows(sys.argv[1], sys.argv[2], memory_budget_bytes=512*1024**2):
    for values in (*row['target'], *row['covariates']):
        for offset, part in chunks(values):
            assert len(part) <= 65536
            assert not find_index_matches(part, index)[0]
            scanned += 1
print(json.dumps(dict(stats=stats, scanned=scanned, scan_peak_rss=peak_rss_bytes())))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), kind],
        capture_output=True,
        text=True,
        timeout=90,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    stats = payload["stats"]
    assert stats["rows"] == rows and stats["longest_row"] == steps
    assert stats["values"] == rows * steps * (1 if kind == "univariate" else 3)
    assert stats["largest_row_group_bytes"] > 512 * 1024**2
    assert 0 < stats["measurement"]["peak_rss_bytes"] < 512 * 1024**2
    assert stats["measurement"]["wall_seconds"] > 0
    assert payload["scanned"] > rows
    assert payload["scan_peak_rss"] < 512 * 1024**2


def test_memory_budget_refuses_with_measured_numbers(tmp_path):
    large_shard(tmp_path / "train-000.parquet", "univariate", 1, steps=70_000_001)
    with pytest.raises(
        ValueError,
        match=r"memory budget.*peak_rss_bytes=.*row_bytes=280000004.*budget_bytes=268435456",
    ):
        measure_folder(tmp_path, "univariate", memory_budget_bytes=256 * 1024**2)


def test_nullable_and_integer_channels_remain_numeric(tmp_path):
    row = dict(item_id="a", source="x", freq="h", target=[[None, 1, 2], [3, 4, None]])
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "train-000.parquet")
    stats = measure_folder(tmp_path, "univariate")
    assert stats["observed_target_values"] == 4
    rec = next(iter_rows(tmp_path, "univariate"))
    np.testing.assert_allclose(rec["target"], [[np.nan, 1, 2], [3, 4, np.nan]], equal_nan=True)


def test_large_parent_checks_and_aggregation_match_existing_semantics():
    from scripts.corpus_v4.audit_fev_overlap import _identifying_parent, _variants
    from scripts.corpus_v4.matching import aggregate_fixed, chunks, identifying_parent

    rng = np.random.default_rng(43)
    for values in (
        np.arange(140000, dtype=float),
        np.tile([2.0, 7.0, 3.0, 4.0], 35000),
        rng.normal(size=140000),
        np.r_[np.zeros(130000), rng.normal(size=10000)],
    ):
        values[::117] = np.nan
        assert _identifying_parent(values) == identifying_parent(values)
    values = rng.normal(size=400000).astype(np.float32)
    variants = list(_variants(values, ("fixed", 1), ("fixed", 3)))
    for aggregated, metadata in variants:
        expected = aggregate_fixed(values.astype(float), 3, metadata["phase"], metadata["method"])
        actual = np.empty(len(expected))
        for start, part in chunks(aggregated):
            actual[start : start + len(part)] = part
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_all_missing_null_type_is_valid(tmp_path):
    row = dict(item_id="a", source="x", freq="h", target=[[None] * 64])
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "train-000.parquet")
    stats = measure_folder(tmp_path, "univariate")
    assert stats["target_values"] == 64 and stats["observed_target_values"] == 0


def test_many_row_groups_keep_global_row_indices(tmp_path):
    rows = [
        dict(item_id=str(i), source="x", freq="h", target=[[float(i)] * 32]) for i in range(199)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train-000.parquet", row_group_size=1)
    assert [rec["row_index"] for rec in iter_rows(tmp_path, "univariate")] == list(range(199))


def test_benchmark_spool_long_history_and_int64_offsets(tmp_path):
    from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries, BenchmarkStore

    values = np.zeros(2_100_001)
    store = BenchmarkStore(
        tmp_path / "benchmark.sqlite", [BenchmarkSeries("b", "a", "target", "h", values)]
    )
    try:
        item = next(iter(store))
        assert len(item.values) == len(values) and isinstance(item.values, np.memmap)
        np.testing.assert_array_equal(item.values, values)
        # Sparse fixture proves spool offsets do not wrap at signed 32-bit limits.
        offset = 2**31 + 4096
        with store.values_path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(np.arange(96, dtype="<f8").tobytes())
        store.db.execute("UPDATE series SET start=?, length=96", (offset,))
        np.testing.assert_array_equal(next(iter(store)).values, np.arange(96))
        assert store.db.execute("PRAGMA cache_size").fetchone()[0] == -8192
    finally:
        store.close()


def test_worker_watchdog_records_actual_rss_when_an_allocation_exceeds_budget():
    script = """
import time
from scripts.corpus_v4.native import start_memory_watchdog, peak_rss_bytes
start_memory_watchdog(peak_rss_bytes() + 32*1024**2)
payload = bytearray(96*1024**2)
time.sleep(2)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 97, result.stderr
    assert "memory budget exceeded" in result.stderr
    assert "peak_rss_bytes=" in result.stderr and "budget_bytes=" in result.stderr


def test_benchmark_event_batches_are_bounded_without_losing_histories(tmp_path):
    from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries, BenchmarkStore, _index_batches

    store = BenchmarkStore(
        tmp_path / "benchmark.sqlite",
        (BenchmarkSeries("b", str(i), "target", "h", np.zeros(96)) for i in range(257)),
    )
    try:
        batches = list(_index_batches(store, "h", "D", "benchmark_to_corpus"))
        assert [len(events) for _, _, events in batches] == [128, 128, 1]
    finally:
        store.close()
