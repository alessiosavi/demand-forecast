"""Competition metrics for the Kaggle bench.

Standalone float64 NumPy implementations, independent of the fev-based
metrics in ``demand_forecast.metrics``. Reductions skip pairs with non-finite
labels or predictions and return NaN when no pair contributes.
"""

from collections.abc import Callable

import numpy as np

__all__ = ["METRIC_FUNCTIONS", "rmse", "rmsle"]


def _sqrt_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, *, log: bool) -> float:
    """Root mean squared error over finite pairs, optionally in log1p space."""
    labels = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        if log:
            labels = np.log1p(labels)
            predictions = np.log1p(predictions)
        observed = np.isfinite(labels) & np.isfinite(predictions)
        squared = np.where(observed, (labels - predictions) ** 2, 0.0)
    count = observed.sum()
    if count == 0:
        return float("nan")
    total = squared.sum(dtype=np.float64)
    return float(np.sqrt(total / count))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared logarithmic error over finite pairs."""
    return _sqrt_mean_squared_error(y_true, y_pred, log=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error over finite pairs."""
    return _sqrt_mean_squared_error(y_true, y_pred, log=False)


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmsle": rmsle,
    "rmse": rmse,
}
