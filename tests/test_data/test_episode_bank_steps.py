"""Causal per-step bank values and fixed calendar slots for v5 episodes."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.episode.bank import step_features
from demand_forecast.data.episode.calendar import calendar_arrays
from demand_forecast.data.episode.scale import fit_scale, to_scaled
from demand_forecast.data.episode.schema import (
    BANK_WINDOW,
    CALENDAR_FIELDS,
    CALENDAR_SIZES,
    STEP_FAMILIES,
    STEP_FEATURES,
)
from demand_forecast.data.frequency import (
    CANONICAL_CALENDAR_SCHEMA,
    calendar_feature_mask,
    calendar_features,
)
from tests.helpers.episodes import toy_panel


def test_step_features_are_causal():
    rng = np.random.default_rng(1)
    raw = rng.poisson(2.0, 600).astype(float)
    raw[rng.random(600) < 0.1] = np.nan
    s = to_scaled(raw, fit_scale(raw, 300, 7))
    a, fa = step_features(s, raw, season=7, long_lag=365, left=300, right=500)
    for k in (300, 371, 499):
        raw2, s2 = raw.copy(), s.copy()
        raw2[k:] = 99.0
        s2[k:] = 99.0
        b, fb = step_features(s2, raw2, season=7, long_lag=365, left=300, right=500)
        assert np.array_equal(a[: k - 300 + 1], b[: k - 300 + 1]) and np.array_equal(
            fa[: k - 300 + 1], fb[: k - 300 + 1]
        )


def test_lags_reach_left_of_the_window():
    raw = np.arange(1000, dtype=float)
    s = raw.copy()
    v, f = step_features(s, raw, season=7, long_lag=365, left=900, right=901)
    i = STEP_FEATURES.index("lag_long")
    assert f[0, i] and np.isclose(v[0, i], np.arcsinh(900 - 365))


def test_zero_features_use_raw_zeros():
    raw = np.array([0.0, 0.0, 2.0, 0.0, 5.0])
    s = raw - 1.0
    v, f = step_features(s, raw, season=1, long_lag=None, left=4, right=5)
    z = STEP_FAMILIES["zeros"]
    assert np.allclose(v[0, z], [0.75, np.log1p(1), np.log1p(1)])
    assert not f[0, STEP_FEATURES.index("lag_season")]
    assert not f[0, STEP_FEATURES.index("lag_long")]


def test_gaps_and_empty_windows():
    raw = np.full(600, np.nan)
    v, f = step_features(raw, raw, season=7, long_lag=365, left=0, right=600)
    assert v.shape == f.shape == (600, 15)
    assert v.dtype == np.float32 and f.dtype == np.bool_
    assert np.isfinite(v).all() and (v == 0.0).all() and not f.any()


def test_calendar_arrays():
    panel = toy_panel()
    phases, ids = calendar_arrays(panel, 3, 40)
    assert phases.shape == (40, 16) and ids.shape == (40, 6)
    assert phases.dtype == np.float32 and ids.dtype == np.int16
    assert (ids[:, CALENDAR_FIELDS.index("hour")] == -1).all()
    assert ((ids[:, CALENDAR_FIELDS.index("dow")] >= 0) & (ids[:, 1] <= 6)).all()
    np.testing.assert_array_equal(
        phases, calendar_features(panel.timestamps(3, 40), panel.freq, schema="canonical")
    )
    assert calendar_arrays(toy_panel(dated=False), 3, 40) is None


@pytest.mark.parametrize("aligned", [False, True])
def test_undated_panels_have_no_calendar_even_when_unaligned(aligned):
    panel = replace(toy_panel(dated=False), aligned=aligned)
    panel.validate()
    assert calendar_arrays(panel, -16, 32) is None


@pytest.mark.parametrize(
    ("freq", "epoch"),
    [
        ("15min", "2020-12-31 22:00"),
        ("h", "2020-12-31 22:00"),
        ("D", "2020-12-31"),
        ("W-MON", "2020-12-28"),
        ("ME", "2020-12-31"),
        ("MS", "2020-12-01"),
        ("QS-JAN", "2020-10-01"),
        ("YS", "2020-01-01"),
        ("BME", "2020-12-31"),
        ("B", "2020-12-31"),
    ],
)
def test_calendar_ids_follow_timestamps_and_the_canonical_mask(freq, epoch):
    panel = replace(toy_panel(), freq=freq, epoch=pd.Timestamp(epoch))
    index = panel.timestamps(-2, 20)
    phases, ids = calendar_arrays(panel, -2, 20)
    expected = np.column_stack(
        [
            index.hour,
            index.day_of_week,
            index.day - 1,
            index.month - 1,
            index.isocalendar().week.to_numpy(dtype=np.int16) - 1,
            index.quarter - 1,
        ]
    )
    mask = calendar_feature_mask(freq, "canonical")
    for j, (field, size) in enumerate(zip(CALENDAR_FIELDS, CALENDAR_SIZES)):
        if mask[CANONICAL_CALENDAR_SCHEMA.index(f"{field}_sin")]:
            np.testing.assert_array_equal(ids[:, j], expected[:, j])
            assert ((ids[:, j] >= 0) & (ids[:, j] < size)).all()
        else:
            assert (ids[:, j] == -1).all()
    np.testing.assert_array_equal(phases, calendar_features(index, freq, schema="canonical"))


def test_calendar_iso_week_leap_day_and_dst():
    panel = replace(toy_panel(), epoch=pd.Timestamp("2020-12-31"))
    _, ids = calendar_arrays(panel, 0, 5)
    np.testing.assert_array_equal(ids[:, 4], [52, 52, 52, 52, 0])
    leap = replace(panel, epoch=pd.Timestamp("2024-02-28"))
    _, ids = calendar_arrays(leap, 0, 3)
    np.testing.assert_array_equal(ids[:, 2], [27, 28, 0])
    dst = replace(panel, freq="h", epoch=pd.Timestamp("2024-03-31", tz="Europe/Rome"))
    _, ids = calendar_arrays(dst, 0, 4)
    np.testing.assert_array_equal(ids[:, 0], [0, 1, 3, 4])


def test_empty_calendar_and_step_interval():
    phases, ids = calendar_arrays(toy_panel(), 0, 0)
    assert phases.shape == (0, 16) and ids.shape == (0, 6)
    assert phases.dtype == np.float32 and ids.dtype == np.int16
    for raw, left in ((np.array([], dtype=float), 0), (np.arange(10.0), 5)):
        v, f = step_features(raw, raw, season=7, long_lag=None, left=left, right=left)
        assert v.shape == f.shape == (0, 15)
        assert v.dtype == np.float32 and f.dtype == np.bool_


def test_rolling_and_differences_use_observed_scaled_cells():
    raw = np.array([1.0, 3.0, np.nan, 7.0, 9.0, 1000.0])
    v, f = step_features(raw, raw, season=3, long_lag=4, left=5, right=6)
    expected = [19 / 3, np.sqrt(56 / 9), 3, 9, 5, np.sqrt(10), 1, 9, 2, 6, 0, 3]
    np.testing.assert_allclose(v[0, :12], np.arcsinh(expected), rtol=1e-6)
    np.testing.assert_array_equal(f[0, :12], [True] * 10 + [False, True])
    assert v[0, 10] == 0.0


def test_short_and_long_windows_are_distinct():
    raw = np.arange(40.0)
    v, f = step_features(raw, raw, season=7, long_lag=None, left=30, right=31)
    expected = [26, 2, 23, 29, 15.5, np.sqrt(65.25), 2, 29]
    np.testing.assert_allclose(v[0, :8], np.arcsinh(expected), rtol=1e-6)
    assert f[0, :8].all()


def test_partial_support_and_missing_lags_have_false_flags():
    raw = np.array([4.0, np.nan, 6.0])
    v, f = step_features(raw, raw, season=2, long_lag=3, left=0, right=3)
    assert not f[0].any() and (v[0] == 0).all()
    np.testing.assert_array_equal(f[1, :8], [True, False, True, True] * 2)
    np.testing.assert_array_equal(v[1, :8], np.arcsinh([4, 0, 4, 4] * 2).astype(np.float32))
    np.testing.assert_array_equal(f[2, 8:12], [False, False, True, False])
    np.testing.assert_array_equal(v[2, 8:12], np.arcsinh([0, 0, 4, 0]).astype(np.float32))


def test_zero_runs_skip_missing_cells_but_elapsed_time_counts_grid_steps():
    raw = np.array([2.0, 0.0, np.nan, 0.0, np.nan, 3.0, 0.0])
    v, f = step_features(raw - 10, raw, season=1, long_lag=None, left=2, right=7)
    np.testing.assert_allclose(
        v[:, 12:],
        [
            [0.5, np.log1p(1), np.log1p(1)],
            [0.5, np.log1p(1), np.log1p(2)],
            [2 / 3, np.log1p(2), np.log1p(3)],
            [2 / 3, np.log1p(2), np.log1p(4)],
            [0.5, 0, 0],
        ],
        rtol=1e-6,
    )
    assert f[:, 12:].all()
    assert not f[:, 9:12].any()


def test_since_nonzero_is_unavailable_until_a_nonzero_has_been_observed():
    raw = np.array([0.0, np.nan, 0.0, 9.0])
    v, f = step_features(raw, raw, season=1, long_lag=None, left=3, right=4)
    np.testing.assert_allclose(v[0, 12:], [1, np.log1p(2), 0])
    np.testing.assert_array_equal(f[0, 12:], [True, True, False])


def test_bank_window_bounds_zeros_and_lags():
    raw = np.zeros(BANK_WINDOW + 2)
    raw[0] = 2
    v, f = step_features(
        raw,
        raw,
        season=BANK_WINDOW,
        long_lag=BANK_WINDOW,
        left=BANK_WINDOW,
        right=BANK_WINDOW + 2,
    )
    np.testing.assert_allclose(v[:, 12], [(BANK_WINDOW - 1) / BANK_WINDOW, 1])
    np.testing.assert_allclose(v[:, 13], np.log1p([BANK_WINDOW - 1, BANK_WINDOW]))
    np.testing.assert_allclose(v[:, 14], [np.log1p(BANK_WINDOW - 1), 0])
    np.testing.assert_array_equal(f[:, 14], [True, False])
    assert f[:, 10:12].all() and not f[:, 9].any()
    np.testing.assert_allclose(v[:, 10:12], np.arcsinh([[2, 2], [0, 0]]))
    _, too_far = step_features(
        raw,
        raw,
        season=BANK_WINDOW + 1,
        long_lag=BANK_WINDOW + 1,
        left=BANK_WINDOW + 1,
        right=BANK_WINDOW + 2,
    )
    assert not too_far[:, 9:12].any()


def test_history_before_the_bank_window_is_inert():
    raw = np.arange(BANK_WINDOW + 20, dtype=float)
    changed = raw.copy()
    changed[:10] = -1e9
    kwargs = dict(season=BANK_WINDOW + 10, long_lag=None, left=BANK_WINDOW + 10, right=len(raw))
    a, fa = step_features(raw, raw, **kwargs)
    b, fb = step_features(changed, changed, **kwargs)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fa, fb)


def test_subinterval_calls_agree_and_do_not_mutate_inputs():
    raw = np.random.default_rng(10).normal(size=200)
    raw[[0, 8, 75]] = np.nan
    scaled = to_scaled(raw, fit_scale(raw, 50, 7))
    before = raw.copy(), scaled.copy()
    a, fa = step_features(scaled, raw, season=7, long_lag=30, left=0, right=200)
    b, fb = step_features(scaled, raw, season=7, long_lag=30, left=100, right=200)
    np.testing.assert_array_equal(a[100:], b)
    np.testing.assert_array_equal(fa[100:], fb)
    np.testing.assert_array_equal(raw, before[0])
    np.testing.assert_array_equal(scaled, before[1])


@pytest.mark.parametrize(
    ("scaled", "raw", "left", "right"),
    [
        (np.ones((2, 2)), np.ones(4), 0, 2),
        (np.ones(4), np.ones((2, 2)), 0, 2),
        (np.ones(4), np.ones(3), 0, 2),
        (np.ones(4), np.ones(4), -1, 2),
        (np.ones(4), np.ones(4), 2, 1),
        (np.ones(4), np.ones(4), 0, 5),
    ],
)
def test_step_features_reject_invalid_shapes_or_intervals(scaled, raw, left, right):
    with pytest.raises(ValueError):
        step_features(scaled, raw, season=7, long_lag=None, left=left, right=right)
