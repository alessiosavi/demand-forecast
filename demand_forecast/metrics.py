# Metric definitions adapted from fev v0.10.0, src/fev/metrics.py:
# https://github.com/autogluon/fev/blob/v0.10.0/src/fev/metrics.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: univariate numpy API, ragged histories, quiet empty reductions,
# interval diagnostics, and median-based summary.
# See THIRD_PARTY_NOTICES.md for the upstream notice and full license.
"""Float64 forecast metrics matching fev, independent of the bench extra.

Labels and point predictions have shape (items, horizon); quantile predictions
have shape (items, horizon, knots). Histories contain the whole available past.
Missing labels are NaN. Reductions with no contributing observations return NaN.
"""

from collections.abc import Sequence

import numpy as np


def _nanmean(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Like numpy.nanmean, returning NaN quietly for empty reductions."""
    observed = ~np.isnan(values)
    count = observed.sum(axis=axis)
    with np.errstate(invalid="ignore", over="ignore"):
        total = np.where(observed, values, 0.0).sum(axis=axis, dtype=np.float64)
        return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def quantile_loss(y_true: np.ndarray, q_pred: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Doubled pinball loss at each cell and knot, in original target units."""
    labels = np.asarray(y_true, dtype=np.float64)[..., None]
    predictions = np.asarray(q_pred, dtype=np.float64)
    levels = np.asarray(quantiles, dtype=np.float64)
    if levels.size == 0:
        raise ValueError("quantile loss requires at least one quantile")
    with np.errstate(invalid="ignore", over="ignore"):
        return 2 * np.abs((labels - predictions) * ((labels <= predictions) - levels))


def seasonal_error_per_item(y_past: list[np.ndarray], m: int) -> np.ndarray:
    """Mean absolute m-lag differences, ignoring NaN differences as fev does.

    The lag m must be positive. Histories with no observed lag pair have NaN
    scale; constant histories have zero scale. No scale floor is applied.
    """
    errors = np.full(len(y_past), np.nan, dtype=np.float64)
    for index, history in enumerate(y_past):
        values = np.asarray(history, dtype=np.float64)
        if len(values) > m:
            with np.errstate(invalid="ignore", over="ignore"):
                errors[index] = _nanmean(np.abs(values[m:] - values[:-m]))
    return errors


def _scaled_mean(errors: np.ndarray, y_past: list[np.ndarray], m: int) -> np.ndarray:
    scales = seasonal_error_per_item(y_past, m).reshape((-1,) + (1,) * (errors.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = errors / scales
    # fev filters nonfinite quotients, rather than prefiltering scales: zero
    # and NaN scales contribute nothing; finite error / infinite scale is zero.
    return _nanmean(np.where(np.isfinite(scaled), scaled, np.nan), axis=(0, 1))


def sql(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    y_past: list[np.ndarray],
    m: int,
    quantiles: Sequence[float],
) -> float:
    """Scaled quantile loss, pooled over observed cells then averaged over knots.

    A knot without any finite scaled error makes the overall score NaN, matching
    fev's ordinary mean over per-knot scores.
    """
    return float(np.mean(_scaled_mean(quantile_loss(y_true, q_pred, quantiles), y_past, m)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_past: list[np.ndarray], m: int) -> float:
    """Mean absolute error divided by each item's full-past seasonal error."""
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
    return float(_scaled_mean(errors, y_past, m))


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Pooled mean absolute error / max(pooled mean absolute label, eps)."""
    labels = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return float(
            _nanmean(np.abs(labels - predictions)) / np.maximum(_nanmean(np.abs(labels)), eps)
        )


def wql(
    y_true: np.ndarray, q_pred: np.ndarray, quantiles: Sequence[float], eps: float = 1.0
) -> float:
    """Mean over knots of pooled doubled pinball / floored mean absolute label."""
    losses = _nanmean(quantile_loss(y_true, q_pred, quantiles), axis=(0, 1))
    denominator = np.maximum(_nanmean(np.abs(np.asarray(y_true, dtype=np.float64))), eps)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return float(np.mean(losses / denominator))


def interval_coverage(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantiles: Sequence[float],
    lo: float = 0.1,
    hi: float = 0.9,
) -> tuple[float, float]:
    """Inclusive coverage and width on labelled cells with observed endpoints.

    Use the nearest available knot for each endpoint (first knot breaks ties).
    Both diagnostics are NaN if no labelled interval is available.
    """
    labels = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(q_pred, dtype=np.float64)
    levels = np.asarray(quantiles, dtype=np.float64)
    lower = predictions[..., np.argmin(np.abs(levels - lo))]
    upper = predictions[..., np.argmin(np.abs(levels - hi))]
    observed = ~np.isnan(labels) & ~np.isnan(lower) & ~np.isnan(upper)
    coverage = np.where(observed, (lower <= labels) & (labels <= upper), np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        widths = np.where(observed, upper - lower, np.nan)
    return float(_nanmean(coverage)), float(_nanmean(widths))


def summarize(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    y_past: list[np.ndarray],
    m: int,
    quantiles: Sequence[float],
) -> dict[str, float]:
    """Report fev metrics, 80% interval diagnostics, and per-knot doubled pinball.

    Point metrics use the 0.5 knot, which must be present in quantiles.
    """
    predictions = np.asarray(q_pred, dtype=np.float64)
    median = predictions[..., list(quantiles).index(0.5)]
    coverage, width = interval_coverage(y_true, predictions, quantiles)
    result = {
        "SQL": sql(y_true, predictions, y_past, m, quantiles),
        "MASE": mase(y_true, median, y_past, m),
        "WAPE": wape(y_true, median),
        "WQL": wql(y_true, predictions, quantiles),
        "coverage_80": coverage,
        "width_80": width,
    }
    per_knot = _nanmean(quantile_loss(y_true, predictions, quantiles), axis=(0, 1))
    result.update(
        {f"pinball_{q}": float(value) for q, value in zip(quantiles, per_knot, strict=True)}
    )
    return result
