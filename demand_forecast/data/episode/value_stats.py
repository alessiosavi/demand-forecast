"""Causal response statistics for the active value of a categorical column."""

import numpy as np
import pandas as pd

from .kinds import is_missing, label_of
from .schema import VALUE_STATS


def category_step_stats(
    labels: np.ndarray,
    scaled_target: np.ndarray,
    *,
    left: int,
    right: int,
    frozen_at: int | None = None,
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
    bounds = steps if frozen_at is None else np.minimum(steps, frozen_at)
    for code in range(len(unique)):
        selected = codes[steps] == code
        if not selected.any():
            continue
        rows = steps[selected] - left
        ends = bounds[selected]
        occurrences = np.flatnonzero(codes == code)
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
            total = np.r_[0.0, responses.cumsum()][n]
            means[:, offset] = total / np.maximum(n, 1)
            if offset == 0:
                support = n
                squares = np.r_[0.0, (responses**2).cumsum()][n]
                spread = np.sqrt(np.maximum(0.0, squares / np.maximum(n, 1) - means[:, 0] ** 2))
        valid = support > 0
        shrink = support / (support + 4)
        values[rows, 0] = means[:, 0] * shrink
        values[rows, 1] = spread
        values[rows, 2:5] = means[:, 1:] * shrink[:, None]
        values[rows, 5] = np.log1p(support)
        values[rows[~valid]] = 0
        flags[rows] = valid
    return values, flags
