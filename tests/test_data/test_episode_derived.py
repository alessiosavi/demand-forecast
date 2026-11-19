"""Causal increment spread, delay scans and onset profiles."""

from importlib import import_module

import numpy as np
import pytest

from demand_forecast.data.episode.kinds import is_missing, label_of
from demand_forecast.data.episode.value_stats import category_step_stats
from tests.helpers.lag_scan_reference import CASES, ar, case
from tests.helpers.lag_scan_reference import scan_delay as reference_scan


@pytest.fixture
def spread():
    return import_module("demand_forecast.data.episode.derived").increment_spread


def _loop(labels, target, *, left, right, frozen_at):
    result = np.zeros(right - left, dtype=np.float32)
    for i, t in enumerate(range(left, right)):
        if not 0 <= t < len(labels) or is_missing(labels[t]):
            continue
        end = t if frozen_at is None else min(t, frozen_at)
        label = label_of(labels[t])
        total, n = 0.0, 0
        for p in range(1, min(end, len(target))):
            if (
                not is_missing(labels[p - 1])
                and not is_missing(labels[p])
                and label_of(labels[p - 1]) == label == label_of(labels[p])
                and np.isfinite(target[p - 1 : p + 1]).all()
            ):
                d = target[p] - target[p - 1]
                total += d * d
                n += 1
        if n:
            result[i] = np.sqrt(total / n / 2)
    return result


@pytest.mark.parametrize("frozen_at", [None, 0, 8, 14])
def test_increment_spread_equals_a_loop_reference(spread, frozen_at):
    labels = np.array(
        [1, "1", "b", "b", None, "b", "b", "1", 1, "b", "b", "b", "1", "solo", "1"],
        dtype=object,
    )
    target = np.array([1, 4, 5, 8, 9, np.nan, 12, 3, 5, 1, np.inf, 9, 3])
    bounds = dict(left=-2, right=18, frozen_at=frozen_at)
    actual = spread(labels, target, **bounds)
    expected = _loop(labels, target, **bounds)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_increment_spread_recovers_a_planted_ratio_on_a_trending_series(spread):
    labels = np.resize(np.repeat(["low", "high"], 16), 4098)
    labels[-2:] = ["low", "high"]
    target = 0.05 * np.arange(len(labels)) + np.random.default_rng(37).normal(
        scale=np.where(labels == "high", 3.0, 1.0)
    )
    bounds = dict(left=4096, right=4098, frozen_at=4096)
    increments = spread(labels, target, **bounds)
    levels, _ = category_step_stats(labels, target, **bounds)
    assert increments[1] / increments[0] == pytest.approx(3, rel=0.25)
    assert levels[1, 1] / levels[0, 1] < 2.25


@pytest.mark.parametrize("frozen_at", [None, 12])
def test_increment_spread_reads_nothing_at_or_after_its_bound(spread, frozen_at):
    labels = np.resize(np.repeat(["a", "b", "c"], 3), 24)
    target = np.arange(24, dtype=np.float64) ** 2
    changed = target.copy()
    changed[12:] = -1e6 * target[12:]
    right = 13 if frozen_at is None else 24
    bounds = dict(left=0, right=right, frozen_at=frozen_at)
    expected = spread(labels, target, **bounds)
    assert expected[-1] > 0
    np.testing.assert_array_equal(spread(labels, changed, **bounds), expected)


@pytest.fixture
def onset():
    return import_module("demand_forecast.data.episode.derived").onset_profile


def _onset_reference(labels, target, *, left, right, frozen_at, season):
    def onset_at(t):
        if not 0 <= t < len(labels) or is_missing(labels[t]):
            return None
        label = label_of(labels[t])
        for start in range(t, max(0, t - 31) - 1, -1):
            if start == 0 or is_missing(labels[start - 1]):
                return None
            if label_of(labels[start - 1]) != label:
                return start, label
        return None

    values = np.zeros(right - left)
    observed = np.zeros(right - left, dtype=bool)
    for i, t in enumerate(range(left, right)):
        active = onset_at(t)
        if active is None:
            continue
        start, label = active
        age = t - start
        bound = t if frozen_at is None else min(t, frozen_at)
        total, count = 0.0, 0
        for p in range(age + 1, min(bound, len(target))):
            previous = p - age
            if onset_at(p) != (previous, label) or not np.isfinite(target[p]):
                continue
            before = target[max(0, previous - min(season, 8)) : previous]
            finite = before[np.isfinite(before)]
            if len(finite):
                base = finite.mean()
                if np.isfinite(base):
                    total += target[p] - base
                    count += 1
        if count:
            values[i] = (count / (count + 2)) * (total / count)
            observed[i] = True
    return values, observed


@pytest.mark.parametrize("frozen_at", [None, 0, 31, 197])
@pytest.mark.parametrize("season", [1, 7, 24])
def test_onset_profile_equals_a_loop_reference(onset, frozen_at, season):
    rng = np.random.default_rng(189)
    names = np.array([1, "1", "b", "c", None, "", [], ("b",)], dtype=object)
    labels = np.repeat(rng.choice(names, size=51), rng.integers(1, 9, size=51))
    target = rng.normal(size=len(labels) - 11)
    target[::11], target[5::29] = np.nan, np.inf
    bounds = dict(left=-3, right=len(labels) + 4, frozen_at=frozen_at, season=season)
    expected, flags = _onset_reference(labels, target, **bounds)
    actual, observed = onset(labels, target, **bounds)
    assert actual.shape == observed.shape == (bounds["right"] - bounds["left"],)
    assert observed.dtype == bool
    np.testing.assert_array_equal(observed, flags)
    np.testing.assert_array_equal(actual, expected)


def test_profile_shows_a_planted_pulse_at_its_age(onset):
    labels = np.tile(np.repeat(["a", "b", "c"], 16), 5)
    pulse = np.isin(np.arange(len(labels)) % 48, [2, 3, 20, 21, 38, 39])
    target = 5.0 * pulse
    bounds = dict(left=192, right=240, frozen_at=192, season=7)
    values, observed = onset(labels, target, **bounds)
    pulse = pulse[192:]
    assert observed.all()
    np.testing.assert_allclose(values[pulse], 5.0, atol=2.0, rtol=0)
    assert (np.abs(values[~pulse]) < 1.0).all()


def test_alternating_labels_do_not_count_a_pulse_twice(onset):
    labels = np.array(["b", "a", "b", "a", "b", "a", "b"])
    target = np.array([0, 6, 0, 6, 0, 6, 0], dtype=float)
    values, observed = onset(labels, target, left=5, right=7, frozen_at=5, season=1)
    np.testing.assert_array_equal(observed, [True, True])
    np.testing.assert_array_equal(values, [3.0, -3.0])


def test_first_cell_gap_and_single_occurrence_give_no_value(onset):
    labels = np.array(["a", "a", "b", "b", None, "b", "b", "c", "c", "a", "a"])
    target = np.arange(len(labels), dtype=float)
    values, observed = onset(labels, target, left=0, right=len(labels), frozen_at=None, season=2)
    assert not observed.any() and not values.any()


def test_ages_31_and_32(onset):
    labels = np.array(["c"] + ["a"] * 34 + ["b"] + ["a"] * 34)
    target = np.zeros(len(labels))
    target[32:34] = 9
    values, observed = onset(labels, target, left=66, right=70, frozen_at=36, season=1)
    np.testing.assert_array_equal(observed, [True, True, False, False])
    np.testing.assert_array_equal(values, [0.0, 3.0, 0.0, 0.0])


def _scan(x, y, *, season=1, no_evidence=0.0):
    return import_module("demand_forecast.data.episode.derived").scan_delay(
        x, y, season=season, no_evidence=no_evidence
    )


@pytest.mark.parametrize("name", CASES)
def test_scan_equals_the_reference_on_its_cases(name):
    for length in (96, 192, 512):
        for season in (1, 7, 24):
            for seed in range(20):
                rng = np.random.default_rng([seed, length, season, CASES.index(name)])
                x, y, _ = case(name, rng, length, season)
                delay, *scores = _scan(x, y, season=season)
                expected_delay, *expected_scores = reference_scan(x, y, season)
                assert delay == expected_delay, (length, season, seed)
                assert scores == pytest.approx(expected_scores, abs=1e-9, rel=0)


@pytest.mark.parametrize("delay", [3, 11])
def test_scan_finds_a_planted_delay(delay):
    rng = np.random.default_rng(501)
    # A weekly cycle aliases delay 11 with 4. The noise component identifies the lag.
    x = 4 * np.sin(2 * np.pi * np.arange(512) / 7) + rng.normal(size=512)
    y = np.roll(x, delay) + rng.normal(scale=0.1, size=512)
    found, score, contemporaneous = _scan(x, y, season=7)
    assert found == delay and score > contemporaneous
    assert _scan(x, y, season=7, no_evidence=score) == (0, score, contemporaneous)


@pytest.mark.parametrize("length,expected", [(46, 0), (47, 3)])
def test_scan_needs_at_least_32_common_cells(length, expected):
    rng = np.random.default_rng(827)
    x = rng.normal(size=length)
    assert _scan(x, np.roll(x, 3))[0] == expected


def test_no_row_for_a_contemporaneous_autocorrelated_driver():
    x = ar(np.random.default_rng(821), 512, (0.92,))
    delay, score, lag_zero = _scan(x, x)
    assert delay == 0
    assert score < lag_zero == pytest.approx(1.0)


@pytest.mark.parametrize("name", CASES[:5])
def test_unrelated_series_rarely_build_a_row(name):
    season, length = 7, 512
    admitted = 0
    for seed in range(400):
        rng = np.random.default_rng([seed, length, season, CASES.index(name)])
        x, y, _ = case(name, rng, length, season)
        admitted += _scan(x, y, season=season)[0] > 0
    assert admitted <= 12


@pytest.mark.parametrize("name", CASES[:5])
def test_unrelated_series_with_gaps_rarely_build_a_row(name):
    season, length = 7, 512
    admitted = 0
    for seed in range(400):
        rng = np.random.default_rng([seed, length, season, CASES.index(name)])
        x, y, _ = case(name, rng, length, season)
        # Repeat the review's independent 20% missingness in each series.
        x = np.where(rng.random(length) < 0.2, np.nan, x)
        y = np.where(rng.random(length) < 0.2, np.nan, y)
        admitted += _scan(x, y, season=season)[0] > 0
    assert admitted <= 12, f"{name}: {admitted}/400 admitted with gaps"


@pytest.mark.parametrize("name", CASES[:5])
def test_unrelated_series_with_a_fifth_removed_rarely_build_a_row(name):
    """Every scan runs here: exactly 102 of 512 cells are removed from each series."""
    season, length = 7, 512
    admitted = 0
    for seed in range(400):
        rng = np.random.default_rng([seed, length, season, CASES.index(name), 102])
        x, y, _ = case(name, rng, length, season)
        x, y = x.copy(), y.copy()
        x[rng.choice(length, 102, replace=False)] = np.nan
        y[rng.choice(length, 102, replace=False)] = np.nan
        admitted += _scan(x, y, season=season)[0] > 0
    assert admitted <= 12, f"{name}: {admitted}/400 admitted with a fifth of the cells removed"


def test_no_row_for_a_driver_whose_cycle_is_the_season():
    x = np.tile([-1.0, 2, -1, 0, 0, 0, 0], 70)
    y = np.roll(x, 3) + np.random.default_rng(50).normal(scale=0.2, size=len(x))
    assert _scan(x, y, season=7)[0] == 0


def test_missing_cells_are_left_out_and_a_fifth_missing_stops_the_scan():
    rng = np.random.default_rng(827)
    x = rng.normal(size=500)
    y = np.roll(x, 5) + rng.normal(scale=0.1, size=len(x))
    x[:100] = np.nan
    y[::21] = np.inf

    # Independently fit an intercept and a slope to only the present observations.
    def adjusted(values):
        observed = np.isfinite(values)
        design = np.column_stack((np.ones(len(values)), np.arange(len(values))))
        coef = np.linalg.lstsq(design[observed], values[observed], rcond=None)[0]
        return values - design @ coef

    dx, dy = adjusted(x), adjusted(y)
    expected = []
    for lag in (5, 0):
        a, b = dx[64 - lag : len(x) - lag], dy[64:]
        pairs = np.isfinite(a) & np.isfinite(b)
        expected.append(abs(np.corrcoef(a[pairs], b[pairs])[0, 1]))
    delay, *scores = _scan(x, y)
    assert delay == 5
    assert scores == pytest.approx(expected, abs=1e-9, rel=0)
    x[100] = np.nan
    assert _scan(x, y) == (0, 0.0, 0.0)
    assert _scan(y, x) == (0, 0.0, 0.0)


def _naive_reference(scaled, *, season, long_lag, left, right, origin):
    width = season * ((8 + season - 1) // season)
    drift_width = season * ((64 + season - 1) // season)
    periods = {"last": 1}
    if season > 1:
        periods["season"] = season
    if long_lag is not None and long_lag > season:
        periods["season2"] = long_lag
    result = {name: (np.zeros(right - left), np.zeros(right - left, bool)) for name in periods}
    result["drift"] = (np.zeros(right - left), np.zeros(right - left, bool))
    for index, cell in enumerate(range(left, right)):
        bound = cell if origin is None else min(cell, origin)
        for name, period in periods.items():
            limit = width if name == "last" else period
            for j in range(min(bound, len(scaled)) - 1, max(0, bound - 4 * limit) - 1, -1):
                if (cell - j) % period == 0 and np.isfinite(scaled[j]):
                    result[name][0][index] = scaled[j]
                    result[name][1][index] = True
                    if name == ("season" if season > 1 else "last"):
                        differences = [
                            (scaled[i] - scaled[i - season]) / season
                            for i in range(
                                max(season, bound - drift_width), min(bound, len(scaled))
                            )
                            if np.isfinite(scaled[i]) and np.isfinite(scaled[i - season])
                        ]
                        if 2 * len(differences) >= width:
                            result["drift"][0][index] = scaled[j] + np.mean(differences) * (
                                cell - j
                            )
                            result["drift"][1][index] = True
                    break
    return result


@pytest.mark.parametrize("season", [1, 7, 24, 400])
@pytest.mark.parametrize("long_lag", [None, 168, 364])
def test_naive_series_equals_a_loop_reference(season, long_lag):
    naive_series = import_module("demand_forecast.data.episode.derived").naive_series
    rng = np.random.default_rng(719)
    for trial in range(25):
        size = 0 if trial == 0 else int(rng.integers(1, 1000))
        scaled = rng.normal(size=size)
        scaled[rng.random(size) < 0.23] = np.nan
        scaled[rng.random(size) < 0.03] = np.inf
        left = int(rng.integers(-40, size + 50))
        right = left + int(rng.integers(0, 78))
        origin = None if trial % 3 == 0 else int(rng.integers(left - 8, right + 8))
        bounds = dict(season=season, long_lag=long_lag, left=left, right=right, origin=origin)
        actual = naive_series(scaled, **bounds)
        expected = _naive_reference(scaled, **bounds)
        assert actual.keys() == expected.keys()
        for name, (values, observed) in actual.items():
            expected_values, expected_observed = expected[name]
            assert observed.dtype == np.bool_
            np.testing.assert_array_equal(observed, expected_observed)
            np.testing.assert_allclose(values, expected_values, atol=1e-12, rtol=1e-12)


def test_drift_slope_uses_the_long_window():
    naive_series = import_module("demand_forecast.data.episode.derived").naive_series
    season = 7
    origin = 77
    scaled = np.arange(origin, dtype=np.float64)
    # The 56 older seasonal differences have slope 1 and the last 14 have slope 3.
    scaled[-14:] += 2 * season
    scaled[-7:] += 2 * season
    values, observed = naive_series(
        scaled, season=season, long_lag=None, left=origin, right=origin + season + 1, origin=origin
    )["drift"]

    # Endpoints a season apart use the same base, so their change measures the fitted slope.
    assert observed.all()
    assert (values[-1] - values[0]) / season == pytest.approx((56 + 14 * 3) / 70)


@pytest.mark.parametrize(("gap", "origin", "expected"), [(1, 40, 38), (31, 40, 8), (32, 40, None)])
def test_gap_before_the_origin(gap, origin, expected):
    naive_series = import_module("demand_forecast.data.episode.derived").naive_series
    scaled = np.arange(60, dtype=np.float64)
    scaled[origin - gap : origin] = np.nan
    values, observed = naive_series(
        scaled, season=1, long_lag=None, left=origin, right=origin + 3, origin=origin
    )["last"]
    np.testing.assert_array_equal(observed, np.full(3, expected is not None))
    np.testing.assert_allclose(values, np.full(3, expected or 0))
