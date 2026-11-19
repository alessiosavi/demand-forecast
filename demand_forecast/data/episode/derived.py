"""Optional causal presentations of episode inputs."""

import numpy as np
import pandas as pd

from .kinds import labels_of, observed_mask

LAG_SCAN_WINDOW = 512


def _adjust(values: np.ndarray, observed: np.ndarray, phases: int) -> np.ndarray:
    """Remove a linear trend, then phase means, fitting only observed cells."""
    time = np.arange(len(values)) - (len(values) - 1) / 2
    time -= time[observed].mean()
    centered = values - values[observed].mean()
    fitted_time = time[observed]
    residual = centered - time * (fitted_time @ centered[observed]) / (fitted_time @ fitted_time)
    if phases:
        phase = np.arange(len(values)) % phases
        count = np.bincount(phase[observed], minlength=phases)
        total = np.bincount(phase[observed], residual[observed], minlength=phases)
        residual -= (total / np.maximum(count, 1))[phase]
    return residual


def _acf(values: np.ndarray, observed: np.ndarray, lags: int) -> np.ndarray:
    centered = values - values[observed].mean()
    if observed.all():
        # Preserve the frozen calibration on complete series.
        energy = centered @ centered
        if energy == 0:
            return np.zeros(lags)
        return np.array([(centered[k:] @ centered[:-k]) / energy for k in range(1, lags + 1)])

    acf = np.zeros(lags)
    for lag in range(1, lags + 1):
        paired = observed[lag:] & observed[:-lag]
        left, right = centered[:-lag][paired], centered[lag:][paired]
        norm = np.linalg.norm(left) * np.linalg.norm(right)
        if norm:
            acf[lag - 1] = (right @ left) / norm
    return acf


def _pearson(x: np.ndarray, y: np.ndarray, has_missing: bool) -> tuple[float, int]:
    """Absolute correlation and the number of complete pairs it rests on."""
    if has_missing:
        paired = np.isfinite(x) & np.isfinite(y)
        if paired.sum() < 2:
            return 0.0, int(paired.sum())
        x, y = x[paired], y[paired]
    score = float(abs(np.corrcoef(x, y)[0, 1]))
    return (0.0 if has_missing and np.isnan(score) else score), len(x)


def scan_delay(
    x: np.ndarray, y: np.ndarray, *, season: int, no_evidence: float
) -> tuple[int, float, float]:
    """Return the admitted delay (or zero), best positive-lag score and lag-zero score."""
    x = np.asarray(x[-LAG_SCAN_WINDOW:], dtype=np.float64)
    y = np.asarray(y[-LAG_SCAN_WINDOW:], dtype=np.float64)
    width = len(x)
    max_delay = min(64, width // 3)
    common = width - max_delay
    phases = season if season > 1 and width >= 8 * season else 0
    dof = common - 2 - phases
    if max_delay < 8 or common < 32 or dof < 24:
        return 0, 0.0, 0.0

    x_observed, y_observed = np.isfinite(x), np.isfinite(y)
    observed = min(x_observed.sum(), y_observed.sum())
    if 5 * observed < 4 * width:
        return 0, 0.0, 0.0
    has_missing = observed < width
    with np.errstate(invalid="ignore", divide="ignore"):
        x = _adjust(x, x_observed, phases)
        y = _adjust(y, y_observed, phases)
        scores, pairs = zip(
            *(
                _pearson(x[max_delay - delay : width - delay], y[max_delay:], has_missing)
                for delay in range(max_delay + 1)
            ),
            strict=True,
        )
        acf_lags = min(max_delay, common // 4)
        inflation = max(
            1.0, 1 + 2 * float(_acf(x, x_observed, acf_lags) @ _acf(y, y_observed, acf_lags))
        )

    # argmax returns the first of any exact ties.
    delay = int(np.argmax(scores[1:])) + 1
    score, contemporaneous = scores[delay], scores[0]
    # Missing cells shorten the evidence: the threshold counts the chosen delay's complete pairs.
    dof = pairs[delay] - 2 - phases
    if pairs[delay] < 32 or dof < 24:
        return 0, score, contemporaneous
    threshold = max(0.2, 3.5 * np.sqrt(inflation / dof))
    admitted = score >= threshold and score > contemporaneous and score > no_evidence
    return (delay if admitted else 0), score, contemporaneous


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
    names = labels_of(labels)
    names[~observed_mask(labels)] = None
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
