"""Causal response statistics for the active value of a categorical column."""

import numpy as np
import pandas as pd

from .derived import increment_spread
from .kinds import is_missing, label_of
from .schema import VALUE_STATS


def category_step_stats(
    labels: np.ndarray,
    scaled_target: np.ndarray,
    *,
    left: int,
    right: int,
    frozen_at: int | None = None,
    use_increment_spread: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode completed earlier responses, freezing history over a known horizon.

    Every response offset has its own completion bound and skips missing targets.
    All four means shrink by n/(n+4), using offset zero's observed support n.
    Cells outside the label span and labels without base support are unavailable.
    """
    labels = np.asarray(labels, dtype=object)
    target = np.asarray(scaled_target, dtype=np.float64)
    if labels.ndim != 1 or target.ndim != 1:
        raise ValueError("labels and scaled_target must be one-dimensional")
    if right < left:
        raise ValueError("right must be at least left")
    values = np.zeros((right - left, VALUE_STATS), dtype=np.float32)
    flags = np.zeros(right - left, dtype=bool)
    steps = np.arange(max(0, left), min(right, len(labels)))
    if not len(steps):
        return values, flags

    missing = np.frompyfunc(is_missing, 1, 1)(labels).astype(bool)
    names = np.frompyfunc(label_of, 1, 1)(labels)
    names[missing] = None
    codes, unique = pd.factorize(names, sort=False)
    # Stable grouping retains chronological accumulation within every label.
    order = np.argsort(codes, kind="stable")
    edges = np.searchsorted(codes[order], np.arange(len(unique) + 1))
    for code in range(len(unique)):
        occurrences = order[edges[code] : edges[code + 1]]
        start, stop = np.searchsorted(occurrences, [steps[0], steps[-1] + 1])
        selected = occurrences[start:stop]
        if not len(selected):
            continue
        rows = selected - left
        ends = selected if frozen_at is None else np.minimum(selected, frozen_at)
        means = np.zeros((len(rows), 4), dtype=np.float64)
        spread = np.zeros(len(rows), dtype=np.float64)
        support = np.zeros(len(rows), dtype=np.int64)
        for offset in range(4):
            completed = occurrences + offset
            completed = completed[(completed < len(target)) & (completed < ends.max())]
            responses = target[completed]
            finite = np.isfinite(responses)
            completed, responses = completed[finite], responses[finite]
            n = np.searchsorted(completed, ends, side="left")
            # Shifting before accumulation preserves variation around large levels.
            anchor = responses[0] if len(responses) else 0.0
            with np.errstate(over="ignore", invalid="ignore"):
                shifted = responses - anchor
                total = np.r_[0.0, shifted.cumsum()][n]
                mean = total / np.maximum(n, 1)
                means[:, offset] = np.where(n > 0, anchor + mean, 0.0)
                if offset == 0:
                    support = n
                    if not use_increment_spread:
                        squares = np.r_[0.0, (shifted**2).cumsum()][n]
                        spread = np.sqrt(np.maximum(0.0, squares / np.maximum(n, 1) - mean**2))
        valid = support > 0
        shrink = support / (support + 4)
        with np.errstate(over="ignore", invalid="ignore"):
            values[rows, 0] = means[:, 0] * shrink
            values[rows, 1] = spread
            values[rows, 2:5] = means[:, 1:] * shrink[:, None]
        values[rows, 5] = np.log1p(support)
        values[rows[~valid]] = 0
        flags[rows] = valid
    if use_increment_spread:
        values[:, 1] = increment_spread(labels, target, left=left, right=right, frozen_at=frozen_at)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return values, flags
