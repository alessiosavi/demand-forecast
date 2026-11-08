"""Tiny real-source shards exercise v3 grids without changing the legacy reader."""

import logging

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench

from demand_forecast.pretrain.corpus import (  # noqa: E402
    MAX_SERIES_POINTS,
    PRETRAIN_FREQ,
    PRETRAIN_START,
    read_shard,
    read_shard_v3,
    shard_panels,
)
from demand_forecast.pretrain.periods import pretrain_offset  # noqa: E402


def _shard(tmp_path, rows):
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("target", pa.list_(pa.float64())),
            ("start", pa.timestamp("ns")),
            ("freq", pa.string()),
            ("source", pa.string()),
            ("folder", pa.string()),
        ]
    )
    path = tmp_path / "tiny.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _row(i, freq="D", start="2020-01-01", target=None):
    return dict(
        id=f"real-{i}",
        target=[-2.0, np.nan, 3.0, np.nan] if target is None else target,
        start=pd.Timestamp(start) if start else None,
        freq=freq,
        source="recorded-source",
        folder="recorded-folder",
    )


@pytest.mark.parametrize("kind", ["chronos_real", "gifteval_pretrain"])
def test_two_frequency_shard_has_homogeneous_panels_and_row_map(tmp_path, kind):
    rows = [_row(0), _row(1, "H"), _row(2), _row(3, "60min")]
    path = _shard(tmp_path, rows)
    loaded = read_shard_v3(path, kind=kind)
    groups = shard_panels(loaded.records)
    assert loaded.grid_fallback == 0
    assert len(groups) == 2
    assert [g.row_indices for g in groups] == [(0, 2), (1, 3)]
    assert [g.panel.seasonal_period for g in groups] == [7, 24]
    assert [g.panel.freq for g in groups] == [pretrain_offset("D"), pretrain_offset("h")]
    for group in groups:
        assert group.calendar_available
        assert group.panel.target_support == "real"
        for local_index, row_index in enumerate(group.row_indices):
            record = loaded.records[row_index]
            item = group.panel.series[local_index]
            assert record.series is item
            assert record.source == "recorded-source" and record.folder == "recorded-folder"
            assert record.row_index == row_index and record.crop_offset == 0
            assert item.start == rows[row_index]["start"]
            assert item.target.dtype == np.float32
            assert item.dynamic == item.statics == {}
            np.testing.assert_array_equal(item.target, rows[row_index]["target"])
    assert all(
        g.panel.seasonal_period == 1
        for g in shard_panels(loaded.records, period_policy="legacy_one")
    )
    # The existing reader still deliberately discards these real grids.
    assert all(item.start == PRETRAIN_START for item in read_shard(path))


@pytest.mark.parametrize(
    "freq,start",
    [
        ("5T", "2020-01-01"),
        ("D", "2020-01-01"),
        ("W-MON", "2000-01-03"),
        ("MS", "2000-01-01"),
        ("ME", "2000-01-31"),
        ("QS-APR", "2000-04-01"),
        ("QE-MAR", "2000-03-31"),
    ],
)
@pytest.mark.parametrize("discarded", [0, 1, 3])
def test_crop_advances_start_on_source_grid(tmp_path, freq, start, discarded):
    # Only discarded grid positions need materializing; never build the full source grid.
    values = np.arange(MAX_SERIES_POINTS + discarded, dtype=np.float64)
    values[-1] = np.nan
    path = _shard(tmp_path, [_row(0, freq, start, values)])
    record = read_shard_v3(path, kind="chronos_real", dtype=np.float64).records[0]
    expected_start = pd.date_range(start, periods=discarded + 1, freq=pretrain_offset(freq))[-1]
    assert record.calendar_available
    assert record.crop_offset == discarded
    assert record.series.start == expected_start
    assert record.series.length == MAX_SERIES_POINTS
    assert record.series.target.dtype == np.float64
    np.testing.assert_array_equal(record.series.target, values[-MAX_SERIES_POINTS:])


def test_daily_crop_preserves_local_time_across_dst(tmp_path):
    import pyarrow.parquet as pq

    start = pd.Timestamp("2020-03-28", tz="Europe/Rome")
    path = tmp_path / "dst.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["local-day", "elapsed-day"],
                "start": [start, start],
                "freq": ["D", "24h"],
                "target": [np.arange(MAX_SERIES_POINTS + 3)] * 2,
            }
        ),
        path,
    )
    loaded = read_shard_v3(path, kind="gifteval_pretrain")
    groups = shard_panels(loaded.records)
    assert loaded.grid_fallback == 0
    assert len(groups) == 2
    assert [g.row_indices for g in groups] == [(0,), (1,)]
    for record, freq, hour in zip(loaded.records, ("D", "24h"), (0, 1), strict=True):
        assert record.series.start == pd.date_range(start, periods=4, freq=freq)[-1]
        assert record.series.start.hour == hour


def test_missing_invalid_grids_are_counted_and_separate_from_real_weekly(tmp_path, caplog):
    rows = [
        _row(0, "W-SUN", "2020-01-05"),
        _row(1, start=None),
        _row(2, freq=None),
        _row(3, "unknown"),
        _row(4, "W-MON", "2020-01-01"),
        _row(5, "0D"),
    ]
    with caplog.at_level(logging.INFO):
        loaded = read_shard_v3(_shard(tmp_path, rows), kind="chronos_real")
    groups = shard_panels(loaded.records)
    assert loaded.grid_fallback == 5
    assert "grid_fallback=5" in caplog.text
    assert len(groups) == 2
    assert [g.row_indices for g in groups] == [(0,), (1, 2, 3, 4, 5)]
    assert [g.calendar_available for g in groups] == [True, False]
    assert all(g.panel.freq == pretrain_offset(PRETRAIN_FREQ) for g in groups)
    assert all(record.series.start == PRETRAIN_START for record in loaded.records[1:])


def test_missing_grid_columns_fall_back_per_real_row(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"id": ["a", "b"], "target": [[1.0, 2.0], [3.0, 4.0]]}), path)
    loaded = read_shard_v3(path, kind="chronos_real")
    assert loaded.grid_fallback == 2
    assert shard_panels(loaded.records)[0].row_indices == (0, 1)


@pytest.mark.parametrize(
    "kind,policy",
    [
        ("tsmixup", "source"),
        ("kernel_synth", "source"),
        ("chronos_real", "legacy_weekly"),
    ],
)
def test_synthetic_and_legacy_policy_keep_the_legacy_grid(tmp_path, kind, policy):
    path = _shard(tmp_path, [_row(0, target=np.arange(MAX_SERIES_POINTS + 2))])
    loaded = read_shard_v3(path, kind=kind, grid_policy=policy)
    (group,) = shard_panels(loaded.records)
    (legacy,) = read_shard(path)
    assert loaded.grid_fallback == 0
    assert not group.calendar_available
    assert group.panel.seasonal_period == 1
    assert group.panel.freq == pretrain_offset(PRETRAIN_FREQ)
    assert group.panel.series[0].start == legacy.start == PRETRAIN_START
    np.testing.assert_array_equal(group.panel.series[0].target, legacy.target)


def test_empty_shard_has_no_groups(tmp_path):
    loaded = read_shard_v3(_shard(tmp_path, []), kind="chronos_real")
    assert loaded.grid_fallback == 0
    assert shard_panels(loaded.records) == []


def test_calendar_offsets_keep_separate_anchored_panels(tmp_path):
    grids = [
        ("W-MON", "2020-01-06", 1),
        ("W-SUN", "2020-01-05", 1),
        ("MS", "2020-01-01", 12),
        ("M", "2020-01-31", 12),
        ("QS-APR", "2020-04-01", 4),
        ("Q-MAR", "2020-03-31", 4),
        ("YS-JUL", "2020-07-01", 1),
        ("A-JUN", "2020-06-30", 1),
        ("2h", "2020-01-01", 1),
    ]
    path = _shard(tmp_path, [_row(i, freq, start) for i, (freq, start, _) in enumerate(grids)])
    loaded = read_shard_v3(path, kind="gifteval_pretrain")
    assert loaded.grid_fallback == 0
    groups = shard_panels(loaded.records)
    assert len(groups) == len(grids)
    for i, (group, (freq, start, period)) in enumerate(zip(groups, grids, strict=True)):
        assert group.calendar_available
        assert group.row_indices == (i,)
        assert group.panel.freq == pretrain_offset(freq)
        assert group.panel.seasonal_period == period
        pd.testing.assert_index_equal(
            group.panel.timestamps(0).as_unit("ns"),
            pd.date_range(start, periods=4, freq=pretrain_offset(freq)).as_unit("ns"),
        )
