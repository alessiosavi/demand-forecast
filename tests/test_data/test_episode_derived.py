"""Causal increment spread for dynamic categories."""

from importlib import import_module

import numpy as np
import pytest

from demand_forecast.data.episode.kinds import is_missing, label_of
from demand_forecast.data.episode.value_stats import category_step_stats


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
