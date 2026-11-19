"""Permutation calibration for static category relevance."""

import os
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from demand_forecast.data.episode.kinds import (
    _simple_object_types,
    is_missing,
    label_of,
    labels_of,
    observed_mask,
)
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.screening import (
    ScreeningConfig,
    relevance_category,
    relevance_static_category,
    relevance_static_number,
    screen,
)
from tests.helpers.fair_reference import fair_rank, fair_rank_masked
from tests.helpers.v5_sampler import paired_static_panel


# Reference scorers and scoring primitives from 1f12c3c.
def _base_cell_labels(value: object) -> tuple[str, ...]:
    if is_missing(value):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(sorted({label_of(v) for v in value if not is_missing(v)}))
    return (label_of(value),)


def _base_label_sets(values: np.ndarray) -> np.ndarray:
    kind = values.dtype.kind
    if values.ndim == 0 or (
        kind not in "biufUS" and (kind != "O" or _simple_object_types(values) is None)
    ):
        return np.frompyfunc(_base_cell_labels, 1, 1)(values)
    observed = observed_mask(values)
    names = labels_of(values[observed])
    singletons = np.fromiter(zip(names), dtype=object, count=len(names))
    cells = np.empty(values.shape, dtype=object)
    cells.fill(())
    cells[observed] = singletons
    return cells


def _base_center(values: np.ndarray) -> np.ndarray:
    # The shares and correlations are invariant to this rescaling; it avoids overflow.
    values = np.asarray(values, dtype=np.float64)
    values = values / max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    return values - values.mean()


def _base_adjusted(
    between: np.ndarray | float, total: np.ndarray | float, n: int, k: int
) -> np.ndarray:
    between, total = np.broadcast_arrays(
        np.asarray(between, dtype=np.float64), np.asarray(total, dtype=np.float64)
    )
    if k < 2 or n <= k:
        return np.zeros_like(total, dtype=np.float64)
    eta = np.divide(between, total, out=np.zeros_like(total, dtype=np.float64), where=total > 0)
    return np.where(total > 0, np.clip(1 - (1 - eta) * (n - 1) / (n - k), 0, 1), 0.0)


def _base_share(values: np.ndarray, codes: np.ndarray, k: int) -> float:
    counts = np.bincount(codes, minlength=k)
    sums = np.bincount(codes, weights=values, minlength=k)
    between = np.sum(sums**2 / np.maximum(counts, 1)) - values.sum() ** 2 / len(values)
    total = np.sum((values - values.mean()) ** 2)
    return float(_base_adjusted(between, total, len(values), k))


def _base_lagged(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        raise ValueError("lags must be nonnegative")
    stop = min(len(y), len(x) + lag)
    return x[: max(0, stop - lag)], y[min(lag, stop) : stop]


def _base_correlation(x: np.ndarray, y: np.ndarray, minimum: int) -> float:
    observed = np.isfinite(x) & np.isfinite(y)
    if observed.sum() < max(2, minimum):
        return 0.0
    x, y = _base_center(x[observed]), _base_center(y[observed])
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(abs(x @ y) / denominator, 0, 1)) if denominator > 0 else 0.0


def _base_multilabel_score(labels: np.ndarray, values: np.ndarray) -> float:
    """Score each membership against absence without an items-by-labels matrix."""
    lengths = np.frompyfunc(len, 1, 1)(labels).astype(np.int64)
    rows = np.repeat(np.arange(len(values)), lengths)
    codes, unique = pd.factorize(np.concatenate(labels), sort=False)
    n, k = len(values), len(unique)
    if n <= 2 or not k:
        return 0.0
    present = np.bincount(codes, minlength=k)
    counts = np.column_stack([present, n - present])
    sums = np.bincount(codes, weights=values[rows], minlength=k)
    sums = np.column_stack([sums, values.sum() - sums])
    means = sums / np.maximum(counts, 1)
    squared_means = np.sum(sums**2 / np.maximum(counts, 1), axis=1)
    mean_share = _base_adjusted(squared_means - values.sum() ** 2 / n, np.sum(values**2), n, 2)

    # All-item absolute deviations at every absent-group mean, via sorted prefix sums.
    ordered = np.sort(values)
    prefix = np.r_[0.0, ordered.cumsum()]
    positions = np.searchsorted(ordered, means[:, 1])
    all_absent = means[:, 1] * (2 * positions - n) + prefix[-1] - 2 * prefix[positions]
    present_abs = np.bincount(codes, weights=np.abs(values[rows] - means[codes, 0]), minlength=k)
    absent_abs = all_absent - np.bincount(
        codes, weights=np.abs(values[rows] - means[codes, 1]), minlength=k
    )
    deviations = np.column_stack([present_abs, absent_abs])
    correction = deviations.sum(axis=1) ** 2 / n
    between = np.sum(deviations**2 / np.maximum(counts, 1), axis=1) - correction
    total = np.sum(values**2) - squared_means - correction
    spread_share = _base_adjusted(between, total, n, 2)
    scores = np.where((counts > 0).all(axis=1), np.maximum(mean_share, spread_share), 0.0)
    return float(scores.max(initial=0.0))


def _base_category_shares(labels, values):
    codes, unique = pd.factorize(np.frompyfunc(lambda cell: cell[0], 1, 1)(labels), sort=False)
    means = np.bincount(codes, weights=values) / np.bincount(codes)
    return (
        _base_share(values, codes, len(unique)),
        _base_share(np.abs(values - means[codes]), codes, len(unique)),
    )


def _base_category_score(labels, y, minimum):
    observed = np.frompyfunc(bool, 1, 1)(labels).astype(bool) & np.isfinite(y)
    if observed.sum() < max(2, minimum):
        return 0.0
    labels, values = labels[observed], _base_center(y[observed])
    if (np.frompyfunc(len, 1, 1)(labels).astype(np.int64) > 1).any():
        return _base_multilabel_score(labels, values)
    return max(_base_category_shares(labels, values))


def _base_dynamic_category(labels, y, lags, min_steps):
    labels, y = _base_label_sets(np.asarray(labels, dtype=object)), np.asarray(y, dtype=np.float64)
    return max(
        (_base_category_score(*_base_lagged(labels, y, lag), min_steps) for lag in lags),
        default=0.0,
    )


def _base_static_category(labels, summaries, flags, min_items):
    cells = np.empty(len(labels), dtype=object)
    cells[:] = labels
    cells = _base_label_sets(cells)
    return max(
        (
            _base_category_score(cells, np.where(flags[:, j], summaries[:, j], np.nan), min_items)
            for j in range(summaries.shape[1])
        ),
        default=0.0,
    )


def _base_static_number(
    values: np.ndarray, summaries: np.ndarray, flags: np.ndarray, min_items: int
) -> float:
    """Largest absolute Spearman correlation on a summary entry's common items."""
    values = np.asarray(values, dtype=np.float64)
    score = 0.0
    for j in range(summaries.shape[1]):
        observed = flags[:, j] & np.isfinite(summaries[:, j]) & np.isfinite(values)
        if observed.sum() >= max(2, min_items):
            score = max(
                score,
                _base_correlation(
                    rankdata(values[observed]), rankdata(summaries[observed, j]), min_items
                ),
            )
    return score


def _static_score(labels, summaries, min_items=8, *, flags=None):
    if flags is None:
        flags = np.ones_like(summaries, dtype=bool)
    return relevance_static_category(labels, summaries, flags, min_items, fair_static_spread=True)


@pytest.mark.parametrize("multilabel", [False, True])
def test_default_static_score_is_byte_equal_to_the_base_and_uses_no_generator(
    multilabel, monkeypatch
):
    rng = np.random.default_rng(31)

    def unexpected(*args, **kwargs):
        pytest.fail("legacy screening tried to create a random generator")

    monkeypatch.setattr(np.random, "default_rng", unexpected)
    for group_size in (2, 3, 4, 8):
        labels = [(f"group{i // group_size}",) for i in range(48)]
        if multilabel:
            labels = [(*cell, f"family{i % 3}") for i, cell in enumerate(labels)]
        summaries = rng.lognormal(size=(len(labels), 3))
        flags = rng.random(summaries.shape) > 0.05
        args = labels, summaries, flags, 8
        expected = _base_static_category(*args)
        assert expected > 0
        assert struct.pack("d", relevance_static_category(*args)) == struct.pack("d", expected)


@pytest.mark.parametrize("enabled_first", [False, True])
def test_static_candidates_cache_keeps_the_two_modes_separate(enabled_first):
    panel = paired_static_panel()
    stats = PanelStats(panel, "y0", 48)
    scores = {}
    for enabled in (enabled_first, not enabled_first, enabled_first):
        config = ScreeningConfig(fair_static_spread=enabled)
        score = next(
            c.score for c in screen(panel, stats, 0, ("y0",), 48, config) if c.name == "pair"
        )
        assert score == scores.setdefault(enabled, score)
    assert scores[False] == pytest.approx(1.0)
    assert scores[True] < scores[False]


def _group_codes(group_size):
    counts = [2, 3] * 96 if group_size == "mixed" else [group_size] * (480 // group_size)
    return np.repeat(np.arange(len(counts)), counts)


def _null_scores(group_size, distribution):
    rng = np.random.default_rng(20261119)
    if group_size == "multi":
        labels = [(f"pair{i // 2}", "shared") if i < 200 else ("shared",) for i in range(480)]
    else:
        labels = [(f"group{i}",) for i in _group_codes(group_size)]
    draw = getattr(rng, distribution)
    return np.array([_static_score(labels, draw(size=(28, 480)).T) for _ in range(60)])


@pytest.mark.parametrize("group_size", [2, 3, 4, 8, 16, 48, "mixed", "multi"])
@pytest.mark.parametrize("distribution", ["normal", "lognormal"])
def test_shuffled_static_labels_score_near_zero(group_size, distribution):
    scores = _null_scores(group_size, distribution)
    positive = np.count_nonzero(scores > 0)
    assert positive <= 6 and scores.max() < 0.25, (positive, scores.max())


def _block_mask_null_scores(block_size):
    rng = np.random.default_rng(77)
    labels = [(f"group{i // 2}",) for i in range(480)]
    flags = np.zeros((480, 28), dtype=bool)
    for j in range(flags.shape[1]):
        start = j * block_size
        flags[start : start + block_size, j] = True
    return np.array(
        [_static_score(labels, rng.lognormal(size=(28, 480)).T, flags=flags) for _ in range(30)]
    )


@pytest.mark.parametrize("block_size", [8, 10, 12])
def test_label_aligned_block_mask_null_scores_near_zero(block_size):
    scores = _block_mask_null_scores(block_size)
    assert np.count_nonzero(scores > 0) <= 4, scores


def _mixed_label_mask_null_scores():
    rng = np.random.default_rng(4242)
    scores = []
    for panel in range(60):
        multi_items = np.arange(0, 2 * (1 + panel % 3), 2)
        labels = [(f"group{i // 2}",) for i in range(480)]
        for i in multi_items:
            labels[i] = (*labels[i], "shared")
        others = np.setdiff1d(np.arange(480), multi_items)
        summaries = rng.lognormal(size=(28, 480)).T
        flags = np.zeros(summaries.shape, dtype=bool)
        for j in range(flags.shape[1]):
            included = multi_items if j % 2 == 0 else multi_items[:0]
            flags[included, j] = True
            flags[rng.choice(others, size=24 - len(included), replace=False), j] = True
        scores.append(_static_score(labels, summaries, flags=flags))
    return np.array(scores)


def test_mixed_label_sparse_mask_null_scores_near_zero():
    scores = _mixed_label_mask_null_scores()
    assert np.count_nonzero(scores > 0) <= 6, scores


def _signal_scores(group_size, effect):
    rng = np.random.default_rng(20261119)
    codes = _group_codes(group_size)
    labels = [(f"group{i}",) for i in codes]
    groups = len(np.unique(codes))
    scores = []
    for _ in range(30):
        summaries = rng.normal(size=(28, 480)).T
        if effect == "level":
            summaries[:, 0] += rng.normal(0, 8, size=groups)[codes]
        else:
            # Match each deviation with its negative so the group means are equal.
            noise = rng.uniform(0.5, 1.5, size=(groups, group_size // 2))
            noise[1::2] *= 4
            summaries[:, 0] = np.column_stack([noise, -noise]).ravel()
        scores.append(_static_score(labels, summaries))
    return np.array(scores)


@pytest.mark.parametrize("group_size", [2, 4, 16, "mixed"])
def test_rank_calibration_finds_level_signal(group_size):
    scores = _signal_scores(group_size, "level")
    assert np.all(scores > 0) and np.median(scores) > 0.4


def test_rank_calibration_finds_spread_signal_with_sixteen_items_per_label():
    assert np.median(_signal_scores(16, "spread")) > 0.2


def test_rank_calibration_rarely_selects_paired_groups_on_spread_alone():
    assert np.count_nonzero(_signal_scores(2, "spread") > 0) < 30 / 4


def _reference_rank_score(components):
    top, second = np.sort(components, axis=0)[-1:-3:-1]
    ref = np.where(components == top, second, top)
    remaining = 1 - ref
    calibrated = np.maximum(0, components - ref) / np.where(remaining > 1e-9, remaining, np.inf)
    scores = calibrated.max(axis=1)
    q = scores[1:].max()
    return max(0.0, (scores[0] - q) / (1 - q)) if 1 - q > 1e-9 else 0.0


def _reference_cases():
    rng = np.random.default_rng(368)
    for case in range(40):
        n, k, width = rng.integers([24, 2, 3], [100, 12, 10])
        codes = rng.permutation(np.arange(n) % k)
        summaries = rng.normal(size=(n, width))
        if case % 3 != 0:
            summaries[:, 0] += rng.normal(0, 5, size=k)[codes]
        yield summaries, codes, k


def _scalar_reference(summaries, cells, flags, *, multilabel):
    memberships = sorted({label for cell in cells for label in cell})
    rng = np.random.default_rng([0xB03, len(cells), len(memberships)])
    permutations = [np.arange(len(cells))] + [rng.permutation(len(cells)) for _ in range(32)]
    labelled = np.fromiter(map(bool, cells), dtype=bool)
    components = []
    for column, valid in zip(summaries.T, flags.T, strict=True):
        items = np.flatnonzero(valid & np.isfinite(column) & labelled)
        if len(items) < 8:
            continue
        values = _base_center(column[items])
        rows = []
        for permutation in permutations:
            labels = cells[items[np.argsort(np.argsort(permutation[items]))]]
            if multilabel:
                row = []
                for label in memberships:
                    binary = _object_cells(
                        [("in",) if label in cell else ("out",) for cell in labels]
                    )
                    row.extend(_base_category_shares(binary, values))
                rows.append(row)
            else:
                rows.append(_base_category_shares(labels, values))
        components.append(rows)
    return _reference_rank_score(np.concatenate(components, axis=1))


def _object_cells(labels):
    cells = np.empty(len(labels), dtype=object)
    cells[:] = labels
    return cells


@pytest.mark.parametrize("masked", [False, True])
def test_rank_calibration_matches_single_label_reference_on_forty_cases(masked):
    rng = np.random.default_rng(95)
    positive = 0
    for summaries, codes, k in _reference_cases():
        labels = [(f"group{i}",) for i in codes]
        if masked:
            flags = rng.random(summaries.shape) > rng.uniform(0.1, 0.65, size=summaries.shape[1])
            flags[:, flags.sum(axis=0) < 8] = False
            order = np.lexsort(np.where(flags, summaries, np.inf).T[::-1])
            expected = fair_rank_masked(summaries[order], flags[order], codes[order], seed=0xB03)
        else:
            flags = np.ones_like(summaries, dtype=bool)
            order = np.lexsort(summaries.T[::-1])
            expected = fair_rank(summaries[order], codes[order], k, seed=0xB03)
        assert _static_score(labels, summaries, flags=flags) == pytest.approx(
            expected, abs=1e-9, rel=0
        )
        positive += expected > 0
    assert positive >= 20


@pytest.mark.parametrize("multilabel", [False, True])
def test_rank_calibration_reuses_label_cell_permutations_with_varying_masks(multilabel):
    rng = np.random.default_rng(352)
    n = 90
    summaries = rng.normal(size=(n, 6))
    flags = rng.random(summaries.shape) > 0.2
    flags[:, 5] = False
    summaries[::13, 3] = np.nan
    codes = np.arange(n) % 3
    summaries[:, 1] += 5 * codes
    multi_items = {1, 38, 82} if multilabel else set()
    # Keep the signal entry all-singleton and keep the multi-label cells in another entry.
    flags[sorted(multi_items), 1] = False
    flags[sorted(multi_items), 0] = True
    labels = [
        () if i % 7 == 0 else (f"group{code}", "shared") if i in multi_items else (f"group{code}",)
        for i, code in enumerate(codes)
    ]
    order = np.lexsort(np.where(flags & np.isfinite(summaries), summaries, np.inf).T[::-1])
    cells = _object_cells(labels)[order]
    expected = _scalar_reference(summaries[order], cells, flags[order], multilabel=multilabel)
    assert expected > 0
    assert _static_score(labels, summaries, flags=flags) == pytest.approx(expected, abs=1e-9, rel=0)


def _deterministic_static_scores():
    rng = np.random.default_rng(20261119)
    codes = np.repeat(np.arange(6), 16)
    summaries = rng.normal(size=(len(codes), 5))
    summaries[:, 0] += np.where(codes % 2 == 0, 8, -8)
    summaries = summaries.round(1)
    summaries[1] = summaries[0]
    labels = [(f"group{i % 2}",) for i in codes]
    labels[1] = ("group1",)
    multi = [(label, "shared") for (label,) in labels]
    return labels, multi, summaries


def _deterministic_score_hex():
    labels, multi, summaries = _deterministic_static_scores()
    return [_static_score(labels, summaries).hex(), _static_score(multi, summaries).hex()]


def test_static_score_is_deterministic_across_calls_and_processes():
    expected = _deterministic_score_hex()
    assert _deterministic_score_hex() == expected
    assert all(float.fromhex(value) > 0.0 for value in expected)
    child_seed = "104" if os.environ.get("PYTHONHASHSEED") == "103" else "103"
    command = (
        "from tests.test_data.test_static_category_spread import _deterministic_score_hex; "
        "print(*_deterministic_score_hex())"
    )
    actual = subprocess.check_output(
        [sys.executable, "-c", command],
        env={**os.environ, "PYTHONHASHSEED": child_seed},
        text=True,
    )
    assert actual.split() == expected


@pytest.mark.parametrize("multilabel", [False, True])
def test_static_spread_is_independent_of_item_order(multilabel):
    single, multi, summaries = _deterministic_static_scores()
    labels = multi if multilabel else single
    order = np.random.default_rng(25).permutation(len(labels))
    expected = _static_score(labels, summaries)
    assert expected > 0.0
    assert _static_score([labels[i] for i in order], summaries[order]).hex() == expected.hex()


def test_dynamic_category_and_static_number_scores_are_byte_equal():
    rng = np.random.default_rng(22)
    labels = np.array(["a", "a", "b", "b"] * 40, dtype=object)
    values = np.r_[np.nan, rng.normal(size=159) * np.where(labels[:-1] == "a", 1.0, 4.0)]
    multi = np.empty(len(labels), dtype=object)
    multi[:] = [("a", "shared") if label == "a" else ("shared",) for label in labels]
    labels[::13] = None
    multi[::13] = None
    for cells in (labels, multi):
        args = cells, values, (0, 1, 7), 16
        expected = _base_dynamic_category(*args)
        assert expected > 0.0
        assert struct.pack("d", relevance_category(*args)) == struct.pack("d", expected)

    summaries = np.column_stack([values, rng.normal(size=len(values)), values**2])
    flags = np.isfinite(summaries)
    flags[::7] = False
    args = values + rng.normal(size=len(values)), summaries, flags, 8
    expected = _base_static_number(*args)
    assert 0.0 < expected < 1.0
    assert struct.pack("d", relevance_static_number(*args)) == struct.pack("d", expected)


def test_multilabel_two_item_membership_spread_is_degenerate():
    # Every 2/2 split has unequal within-group ranges, hence a raw spread of one.
    labels = [("shared", "member")] * 2 + [("shared",)] * 2
    values = np.array([0.0, 11.0, 2.0, 8.0])
    assert _static_score(labels, values[:, None], min_items=4) == 0.0
