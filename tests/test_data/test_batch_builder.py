"""Exact 2863356 batch parity across grids, exceptional inputs and cache reuse."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data import sampler
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import seasonal_period
from tests.helpers.panels import weekly_panel
from tests.helpers.reference_batch import (
    assert_batches_equal,
    reference_align_to_origin,
    reference_batches,
    reference_make_batch,
)


def batch_case(name):
    panel = weekly_panel([31, 11, 47])
    horizon, context = 7, 8
    calls = [([(0, 6), (1, 8), (2, 19), (0, 0), (2, 46)], None)]
    if name in ("mixed_starts", "late_start"):
        panel.series[1].start += pd.Timedelta(weeks=20 if name == "late_start" else 2)
        calls[0][0][1] = (1, 0 if name == "late_start" else 8)
    elif name == "tz_collision":
        panel.freq = "h"
        start = pd.Timestamp("2023-03-25 23:00", tz="UTC")
        panel.series[0].start = start
        panel.series[1].start = start.tz_convert("Europe/Rome")
        panel.series[2].start = start.tz_convert("America/New_York")
        assert panel.series[0].start == panel.series[1].start
        calls = [([(0, 8), (1, 8), (2, 8)], start + pd.Timedelta(hours=11))]
    elif name == "cutoff_growth":
        cutoff = panel.timestamps(2)[35]
        # Same start and cutoff: the short request must not poison the long one.
        calls = [([(1, 2)], cutoff), ([(2, 33), (0, 30)], cutoff)]
    elif name == "phase_mismatch":
        panel.freq = "2D"
        panel.series[1].start += pd.Timedelta(days=1)
        calls = [(calls[0][0], pd.Timestamp("2023-01-16"))]
    elif name == "hourly_dst":
        panel.freq = "h"
        for series, start in zip(
            panel.series, ["2023-03-25 22:00", "2023-10-28 22:00", "2023-03-26 00:00"]
        ):
            series.start = pd.Timestamp(start, tz="Europe/Rome")
    elif name == "monthly":
        panel.freq = "ME"
        for series in panel.series:
            series.start = pd.Timestamp("2020-01-31")
    elif name == "ragged_aligned":
        panel.series[1].dynamic["promo"] = np.arange(30, dtype=float)
        items = [(1, 25), (0, 31), (2, 46)]
        original = deepcopy(panel)
        aligned = sampler.align_to_origin(panel, items)
        reference = reference_align_to_origin(panel, items)
        for actual, expected, before in zip(aligned.series, reference.series, original.series):
            np.testing.assert_equal(actual.target, expected.target)
            np.testing.assert_equal(actual.dynamic, expected.dynamic)
            np.testing.assert_equal(
                panel.series[panel.index_of(before.series_id)].target, before.target
            )
        panel = aligned
        assert np.isnan(panel.series[1].dynamic["oil"][11:]).all()
        assert panel.series[1].dynamic["promo"][25] == 25
        calls = [(items, None)]
    elif name == "indicator_column":
        panel.series[0].dynamic["promo"][:6] = [0, 1, np.nan, 0, 1, 0]
        panel.series[0].dynamic["promo"][6:9] = [2, 7, np.nan]
    elif name == "all_nan_past":
        panel.series[0].target[:6] = np.nan
        panel.series[0].dynamic["promo"][:6] = np.nan
        panel.series[0].dynamic["oil"][:] = np.nan
        panel.series[1].dynamic = {}
    elif name == "label_cutoff":
        calls = [(calls[0][0], panel.timestamps(0)[8] + pd.Timedelta(days=2))]
    elif name == "week53":
        for series in panel.series:
            series.start = pd.Timestamp("2020-12-06")
    elif name == "daily":
        panel.freq = "D"
        for series in panel.series:
            series.start = pd.Timestamp("2024-02-25", tz="Europe/Rome")
    elif name == "origin_zero":
        calls = [([(0, 0), (1, 0)], None)]
    elif name == "long_context":
        context = 512
    panel.seasonal_period = seasonal_period(panel.freq)
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    if name == "unseen_statics":
        panel.series[0].series_id = "unseen"
        panel.series[0].statics = {"cat": "unseen", "size": np.nan}
        panel.series[1].statics = {}
    return panel, horizon, context, encoder, calls


@pytest.mark.parametrize(
    "case",
    [
        "shared_start",
        "mixed_starts",
        "tz_collision",
        "cutoff_growth",
        "phase_mismatch",
        "hourly_dst",
        "monthly",
        "late_start",
        "ragged_aligned",
        "indicator_column",
        "all_nan_past",
        "label_cutoff",
        "week53",
        "daily",
        "origin_zero",
        "long_context",
        "unseen_statics",
    ],
)
def test_build_matches_reference(case):
    panel, horizon, context, encoder, calls = batch_case(case)
    builder = sampler.BatchBuilder(panel, horizon, context, encoder)
    for items, cutoff in calls:
        expected = reference_make_batch(panel, items, horizon, context, encoder, cutoff)
        assert_batches_equal(builder.build(items, cutoff), expected)
        assert_batches_equal(
            sampler.make_batch(panel, items, horizon, context, encoder, cutoff), expected
        )


@pytest.mark.parametrize("case", ["overflow_scale", "empty", "bad_items"])
def test_build_errors_match_reference(case):
    panel, horizon, context, encoder, _ = batch_case("shared_start")
    items_list = [[(0, 6)]]
    if case == "overflow_scale":
        # The first target tests loc overflow; the second tests scale overflow.
        targets = [np.full(31, 1e40), np.tile([-1e40, 1e40, 0.0], 11)[:31]]
    else:
        targets = [panel.series[0].target]
    if case == "empty":
        panel.series = []
        items_list = [[(0, 0)], []]
    elif case == "bad_items":
        items_list = [[(-1, 2)], [(3, 1)], [(0, -1)], [(0, 32)], [(0, 1.5)], [1], [[1, 2, 3]]]
    for target in targets:
        if panel.series:
            panel.series[0].target = target
        builder = sampler.BatchBuilder(panel, horizon, context, encoder)
        for items in items_list:
            try:
                expected = reference_make_batch(panel, items, horizon, context, encoder)
            except Exception as error:
                for build in (
                    builder.build,
                    lambda rows: sampler.make_batch(panel, rows, horizon, context, encoder),
                ):
                    with pytest.raises(type(error)) as actual:
                        build(items)
                    assert str(actual.value) == str(error)
            else:
                assert_batches_equal(builder.build(items), expected)


def test_build_matches_reference_random_origins():
    panel = weekly_panel([80, 17, 3, 29, 65])
    rng = np.random.default_rng(0)
    for i, series in enumerate(panel.series):
        series.start += pd.Timedelta(weeks=i)
        series.target[rng.random(series.length) < 0.2] = np.nan
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    cutoff = panel.timestamps(0)[50]
    origins = sampler.build_origins(panel, 13, 1, upto=cutoff)
    builder = sampler.BatchBuilder(panel, 13, 32, encoder)
    for _ in range(20):
        items = origins[rng.integers(len(origins), size=128)]
        assert_batches_equal(
            builder.build(items, cutoff),
            reference_make_batch(panel, items, 13, 32, encoder, cutoff),
        )


def test_label_upto_varies_across_calls():
    panel, horizon, context, encoder, calls = batch_case("shared_start")
    items = calls[0][0]
    builder = sampler.BatchBuilder(panel, horizon, context, encoder)
    for cutoff in [
        panel.timestamps(0)[8],
        None,
        panel.timestamps(0)[1],
        pd.NaT,
        panel.timestamps(0)[20],
        panel.timestamps(0)[8],
    ]:
        assert_batches_equal(
            builder.build(items, cutoff),
            reference_make_batch(panel, items, horizon, context, encoder, cutoff),
        )


def test_earlier_batch_unchanged():
    panel, horizon, context, encoder, calls = batch_case("shared_start")
    builder = sampler.BatchBuilder(panel, horizon, context, encoder)
    first = builder.build(calls[0][0])
    snapshot = first.clone()
    second = builder.build([(2, 40), (1, 2)], panel.timestamps(0)[8])
    second.target.fill_(123)
    second.cat["cat"].fill_(0)
    third = builder.build(calls[0][0])
    assert_batches_equal(first, snapshot)
    assert_batches_equal(third, snapshot)


def test_sampler_batches_match_reference():
    panel, horizon, context, encoder, _ = batch_case("shared_start")
    cutoff = panel.timestamps(0)[20]
    origins = sampler.build_origins(panel, horizon, 1, upto=cutoff)
    optimized = sampler.WindowSampler(
        panel, origins, horizon, context, encoder, seed=0, label_upto=cutoff
    )
    reference = sampler.WindowSampler(
        panel, origins, horizon, context, encoder, seed=0, label_upto=cutoff
    )
    for epoch in range(2):
        optimized.set_epoch(epoch)
        reference.set_epoch(epoch)
        actual, expected = list(optimized.batches(2)), list(reference_batches(reference, 2))
        assert len(actual) == len(expected) == 2
        for a, b in zip(actual, expected):
            assert_batches_equal(a, b)
        assert optimized.state() == reference.state()


@pytest.mark.parametrize("aware", [False, True])
def test_label_cutoff_timezone_errors_match_reference(aware):
    panel, horizon, context, encoder, _ = batch_case("tz_collision" if aware else "shared_start")
    cutoff = pd.Timestamp("2023-03-26", tz=None if aware else "UTC")
    with pytest.raises(TypeError) as expected:
        reference_make_batch(panel, [(0, 6)], horizon, context, encoder, cutoff)
    with pytest.raises(TypeError) as actual:
        sampler.BatchBuilder(panel, horizon, context, encoder).build([(0, 6)], cutoff)
    assert str(actual.value) == str(expected.value)


def test_repeated_build_reuses_grid_calendar_and_statics(monkeypatch):
    panel, horizon, context, encoder, calls = batch_case("cutoff_growth")
    counts = {"grid": 0, "calendar": 0, "statics": 0}

    def count(name, function):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(sampler, "grid_range", count("grid", sampler.grid_range))
    monkeypatch.setattr(sampler, "calendar_features", count("calendar", sampler.calendar_features))
    monkeypatch.setattr(encoder, "transform", count("statics", encoder.transform))
    builder = sampler.BatchBuilder(panel, horizon, context, encoder)
    assert counts == {"grid": 0, "calendar": 0, "statics": 0}
    for _ in range(2):
        for items, cutoff in calls:
            builder.build(items, cutoff)
    assert counts == {"grid": 1, "calendar": 1, "statics": 3}
