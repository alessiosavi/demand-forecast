"""Causal increment spread and delay scans."""

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
