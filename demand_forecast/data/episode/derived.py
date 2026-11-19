"""Optional causal presentations of episode inputs."""

import numpy as np
import pandas as pd

from .kinds import is_missing, label_of


def increment_spread(
    labels: np.ndarray,
    target: np.ndarray,
    *,
    left: int,
    right: int,
    frozen_at: int | None,
) -> np.ndarray:
    """Return the RMS increment divided by sqrt(2) for each active label."""
    labels = np.asarray(labels, dtype=object)
    target = np.asarray(target)
    if labels.ndim != 1 or target.ndim != 1:
        raise ValueError("labels and target must be one-dimensional")
    if right < left:
        raise ValueError("right must be at least left")
    values = np.zeros(right - left, dtype=np.float32)
    start, stop = max(0, left), min(right, len(labels))
    bound = stop - 1 if frozen_at is None else min(stop - 1, frozen_at)
    end = min(bound, len(target))
    if start >= stop or end <= 1:
        return values

    labels = labels[:stop]
    names = np.frompyfunc(label_of, 1, 1)(labels)
    names[np.frompyfunc(is_missing, 1, 1)(labels).astype(bool)] = None
    codes, unique = pd.factorize(names, sort=False)
    # Group by label while keeping each group's original time order.
    order = np.argsort(codes, kind="stable")
    edges = np.searchsorted(codes[order], np.arange(len(unique) + 1))
    target = target[:end].astype(np.float64, copy=False)
    for code in range(len(unique)):
        occurrences = order[edges[code] : edges[code + 1]]
        selected = occurrences[np.searchsorted(occurrences, start) :]
        if not len(selected):
            continue
        completed = occurrences[1:][np.diff(occurrences) == 1]
        completed = completed[completed < end]
        finite = np.isfinite(target[completed]) & np.isfinite(target[completed - 1])
        completed = completed[finite]
        ends = selected if frozen_at is None else np.minimum(selected, frozen_at)
        count = np.searchsorted(completed, ends, side="left")
        with np.errstate(over="ignore", invalid="ignore"):
            difference = target[completed] - target[completed - 1]
            squares = np.r_[0.0, (difference**2).cumsum()][count]
            values[selected - left] = np.sqrt(squares / np.maximum(count, 1) / 2)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
