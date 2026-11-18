"""Fixed screening and cutoff-only relevance, before learned feature gates.

``no_evidence`` only zeroes scores below it; it removes nothing. Scores feed
the learned gates and the order inside each budget.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .kinds import flag_codes, infer_kind, is_missing, label_of
from .panel_stats import PanelStats
from .scale import fit_scale, to_scaled
from .schema import (
    KIND_CATEGORY,
    KIND_FLAG,
    KIND_NUMBER,
    MAX_ATTRIBUTE_TOKENS,
    MAX_FEATURE_ROWS,
    MAX_FORECAST_ROWS,
    MAX_STATIC_NUMBERS,
    Panel,
)


@dataclass(frozen=True)
class ScreeningConfig:
    """Minimum common support and the diagnostic no-evidence threshold."""

    min_steps: int = 16
    min_items: int = 8
    no_evidence: float = 0.0


@dataclass(frozen=True)
class Candidate:
    """One row or static attribute, including the reason for a fixed-rule removal."""

    name: str
    source: str
    kind: int | str
    score: float
    removed: str | None
    size: int = 1


def _cell_labels(value: object) -> tuple[str, ...]:
    if is_missing(value):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(sorted({label_of(v) for v in value if not is_missing(v)}))
    return (label_of(value),)


def _label_sets(values: np.ndarray) -> np.ndarray:
    return np.frompyfunc(_cell_labels, 1, 1)(values)


def _center(values: np.ndarray) -> np.ndarray:
    # The shares and correlations are invariant to this rescaling; it avoids overflow.
    values = np.asarray(values, dtype=np.float64)
    values = values / max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    return values - values.mean()


def _adjusted(between: np.ndarray | float, total: np.ndarray | float, n: int, k: int) -> np.ndarray:
    between, total = np.broadcast_arrays(
        np.asarray(between, dtype=np.float64), np.asarray(total, dtype=np.float64)
    )
    if k < 2 or n <= k:
        return np.zeros_like(total, dtype=np.float64)
    eta = np.divide(between, total, out=np.zeros_like(total, dtype=np.float64), where=total > 0)
    return np.where(total > 0, np.clip(1 - (1 - eta) * (n - 1) / (n - k), 0, 1), 0.0)


def _share(values: np.ndarray, codes: np.ndarray, k: int) -> float:
    counts = np.bincount(codes, minlength=k)
    sums = np.bincount(codes, weights=values, minlength=k)
    between = np.sum(sums**2 / np.maximum(counts, 1)) - values.sum() ** 2 / len(values)
    total = np.sum((values - values.mean()) ** 2)
    return float(_adjusted(between, total, len(values), k))


def adjusted_share(values: np.ndarray, groups: np.ndarray) -> float:
    """Adjusted between-label variance share, excluding unobserved pairs."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    observed = np.isfinite(values) & ~np.frompyfunc(is_missing, 1, 1)(groups).astype(bool)
    if not observed.any():
        return 0.0
    codes, labels = pd.factorize(np.frompyfunc(label_of, 1, 1)(groups[observed]), sort=False)
    return _share(_center(values[observed]), codes, len(labels))


def _lagged(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        raise ValueError("lags must be nonnegative")
    stop = min(len(y), len(x) + lag)
    return x[: max(0, stop - lag)], y[min(lag, stop) : stop]


def _correlation(x: np.ndarray, y: np.ndarray, minimum: int) -> float:
    observed = np.isfinite(x) & np.isfinite(y)
    if observed.sum() < max(2, minimum):
        return 0.0
    x, y = _center(x[observed]), _center(y[observed])
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(abs(x @ y) / denominator, 0, 1)) if denominator > 0 else 0.0


def relevance_number(x: np.ndarray, y: np.ndarray, lags: tuple[int, ...], min_steps: int) -> float:
    """Largest absolute Pearson correlation of x[t-lag] with y[t]."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return max((_correlation(*_lagged(x, y, lag), min_steps) for lag in lags), default=0.0)


def _multilabel_score(labels: np.ndarray, values: np.ndarray) -> float:
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
    mean_share = _adjusted(squared_means - values.sum() ** 2 / n, np.sum(values**2), n, 2)

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
    spread_share = _adjusted(between, total, n, 2)
    scores = np.where((counts > 0).all(axis=1), np.maximum(mean_share, spread_share), 0.0)
    return float(scores.max(initial=0.0))


def _category_score(labels: np.ndarray, y: np.ndarray, minimum: int) -> float:
    observed = np.frompyfunc(bool, 1, 1)(labels).astype(bool) & np.isfinite(y)
    if observed.sum() < max(2, minimum):
        return 0.0
    labels, values = labels[observed], _center(y[observed])
    if (np.frompyfunc(len, 1, 1)(labels).astype(np.int64) > 1).any():
        return _multilabel_score(labels, values)
    codes, unique = pd.factorize(np.frompyfunc(lambda cell: cell[0], 1, 1)(labels), sort=False)
    means = np.bincount(codes, weights=values) / np.bincount(codes)
    return max(
        _share(values, codes, len(unique)),
        _share(np.abs(values - means[codes]), codes, len(unique)),
    )


def relevance_category(
    labels: np.ndarray, y: np.ndarray, lags: tuple[int, ...], min_steps: int
) -> float:
    """Largest adjusted mean or spread share, with per-label multilabel scores."""
    labels, y = _label_sets(np.asarray(labels, dtype=object)), np.asarray(y, dtype=np.float64)
    return max((_category_score(*_lagged(labels, y, lag), min_steps) for lag in lags), default=0.0)


def relevance_static_category(
    labels: list[tuple[str, ...]], summaries: np.ndarray, flags: np.ndarray, min_items: int
) -> float:
    """Largest mean or spread share across valid per-item summary entries."""
    cells = np.empty(len(labels), dtype=object)
    cells[:] = labels
    cells = _label_sets(cells)
    return max(
        (
            _category_score(cells, np.where(flags[:, j], summaries[:, j], np.nan), min_items)
            for j in range(summaries.shape[1])
        ),
        default=0.0,
    )


def relevance_static_number(
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
                _correlation(
                    rankdata(values[observed]), rankdata(summaries[observed, j]), min_items
                ),
            )
    return score


def _other_stats(stats: PanelStats, target: str) -> PanelStats:
    key = "target", target
    if key not in stats.screen_cache:
        stats.screen_cache[key] = PanelStats(stats.panel, target, stats.cutoff)
    return stats.screen_cache[key]


def _static_candidates(stats: PanelStats, min_items: int) -> tuple[Candidate, ...]:
    cache_key = "static", min_items
    if cache_key in stats.screen_cache:
        return stats.screen_cache[cache_key]
    candidates = []
    seen = set()
    for name, kind in sorted(stats.static_kinds.items()):
        labels = [stats.labels(name, i) for i in range(len(stats.panel.items))]
        removed = None
        if kind == "number":
            values = np.array(
                [
                    np.nan if is_missing(item.static.get(name)) else float(item.static[name])
                    for item in stats.panel.items
                ]
            )
            signature = tuple(np.where(np.isfinite(values), values, None))
            score = relevance_static_number(values, stats.summaries, stats.summary_flags, min_items)
        else:
            signature = tuple(labels)
            flattened = [label for cell in labels for label in cell]
            _, counts = np.unique(flattened, return_counts=True)
            if all(len(cell) == 1 for cell in labels) and len(counts) == len(labels):
                removed = "entity"
            elif not (counts >= 2).any():
                removed = "unshared"
            score = relevance_static_category(
                labels, stats.summaries, stats.summary_flags, min_items
            )
        key = kind, signature
        if key in seen:
            removed = "duplicate"
        seen.add(key)
        candidates.append(
            Candidate(name, "static", kind, score if removed is None else 0.0, removed)
        )
    stats.screen_cache[cache_key] = tuple(candidates)
    return stats.screen_cache[cache_key]


def screen(
    panel: Panel,
    stats: PanelStats,
    item_index: int,
    targets: tuple[str, ...],
    cutoff: int,
    config: ScreeningConfig = ScreeningConfig(),
) -> list[Candidate]:
    """Return every candidate in (source, name) order, using only permitted history."""
    if cutoff != stats.cutoff:
        raise ValueError("cutoff must match stats.cutoff")
    if not 0 <= item_index < len(panel.items):
        raise IndexError("item index outside the panel")
    item = panel.items[item_index]
    end = max(0, cutoff - item.start)
    histories = []
    for target in sorted(targets):
        raw = item.targets[target][:end]
        scale = (
            stats.scales[item_index]
            if target == stats.target
            else fit_scale(raw, len(raw), panel.season)
        )
        histories.append(to_scaled(raw, scale))
    lags = tuple(dict.fromkeys((0, 1, panel.season)))
    candidates = []
    seen = set()
    for name, raw in sorted(item.dynamic.items()):
        raw = raw[:end]
        source, kind = panel.roles[name], infer_kind(raw)
        cells = _label_sets(raw)
        signature = tuple(cells)
        observed = {cell for cell in signature if cell}
        removed = None
        if source == "past":
            if not observed:
                removed = "empty"
            elif len(observed) == 1:
                removed = "constant"
            elif (kind, signature) in seen:
                removed = "duplicate"
            seen.add((kind, signature))
        values = flag_codes(raw, len(raw)) if kind == KIND_FLAG else raw
        relevance = relevance_category if kind == KIND_CATEGORY else relevance_number
        score = (
            max((relevance(values, y, lags, config.min_steps) for y in histories), default=0.0)
            if removed is None
            else 0.0
        )
        candidates.append(Candidate(name, source, kind, score, removed))
    for name in sorted(set(item.targets) - set(targets)):
        score = max(
            (
                relevance_number(item.targets[name][:end], y, (0,), config.min_steps)
                for y in histories
            ),
            default=0.0,
        )
        candidates.append(Candidate(name, "sibling", KIND_NUMBER, score, None))
    static = {}
    if stats.static_kinds:
        for target in sorted(targets):
            current = stats if target == stats.target else _other_stats(stats, target)
            for candidate in _static_candidates(current, config.min_items):
                previous = static.get(candidate.name)
                if previous is None or candidate.score > previous.score:
                    size = (
                        1
                        if candidate.kind == "number"
                        else len(stats.labels(candidate.name, item_index))
                    )
                    if size == 0:
                        continue
                    static[candidate.name] = replace(candidate, size=size)
    candidates.extend(static.values())
    return sorted(
        (replace(c, score=0.0) if c.score < config.no_evidence else c for c in candidates),
        key=lambda c: (c.source, c.name),
    )


def select(
    candidates: list[Candidate],
    *,
    feature_budget: int = MAX_FEATURE_ROWS,
    sibling_budget: int = MAX_FORECAST_ROWS - 1,
    attribute_budget: int = MAX_ATTRIBUTE_TOKENS,
) -> list[Candidate]:
    """Allocate independent row budgets, admitting complete static expansions only."""
    order = {"known": 0, "past": 1, "sibling": 2, "static": 3}
    selected = []
    numbers = 0
    for candidate in sorted(candidates, key=lambda c: (order[c.source], -c.score, c.name)):
        if candidate.removed is not None:
            continue
        if candidate.source in ("known", "past"):
            if feature_budget <= 0:
                continue
            feature_budget -= 1
        elif candidate.source == "sibling":
            if sibling_budget <= 0:
                continue
            sibling_budget -= 1
        else:
            number = candidate.kind == "number"
            if candidate.size > attribute_budget or (number and numbers >= MAX_STATIC_NUMBERS):
                continue
            attribute_budget -= candidate.size
            numbers += int(number)
        selected.append(candidate)
    return selected
