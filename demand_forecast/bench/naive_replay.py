"""Model-free replay of the benchmark's Seasonal Naive forecast and its v5 variants."""

import numpy as np
from scipy.special import ndtri


def _v5_point_lookup(history: np.ndarray, lag: int, phases: np.ndarray) -> np.ndarray:
    if lag == 1:
        recent = history[-32:]
        finite = recent[np.isfinite(recent)]
        return np.full(phases.size, finite[-1] if finite.size else np.nan, dtype=history.dtype)

    positions = history.size - lag + phases[:, None] - lag * np.arange(4)
    in_history = positions >= 0
    candidates = np.full(positions.shape, np.nan, dtype=history.dtype)
    candidates[in_history] = history[positions[in_history]]
    finite = np.isfinite(candidates)
    first = candidates[np.arange(phases.size), finite.argmax(axis=1)]
    return np.where(finite.any(axis=1), first, np.nan)


def _lag_sigma(
    values: np.ndarray, lag: int, count: int, *, where: bool | np.ndarray = True
) -> float:
    if count <= 0:
        return 0.0
    # Keep the leading NaNs in the reduction, as StatsForecast does for fitted residuals.
    residuals = np.full_like(values, np.nan)
    np.subtract(values[lag:], values[:-lag], out=residuals[lag:], where=where)
    return np.sqrt(np.nansum(residuals**2) / count)


def seasonal_naive_replay(
    history: np.ndarray,
    lag: int,
    horizon: int,
    quantile_levels: list[float],
    *,
    v5_point: bool = False,
    v5_pairs: bool = False,
    v5_patch: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return the point, quantiles, missing point cells and sigma fallbacks for one series."""
    if history.dtype not in (np.float32, np.float64):
        history = history.astype(np.float32)
    filled = history
    missing = np.isnan(history)
    if missing.any():
        sources = np.maximum.accumulate(np.where(missing, 0, np.arange(1, history.size + 1)))
        filled = np.r_[np.zeros(1, dtype=history.dtype), history][sources]
    filled = filled[-1000:]

    steps = np.arange(horizon)
    phases = steps % lag
    tail = filled[-lag:]
    cycle = np.full(lag, np.nan, dtype=history.dtype)
    cycle[: tail.size] = tail
    point = cycle[phases]
    point_lookup_misses = 0
    if v5_point:
        looked_up = _v5_point_lookup(history, lag, phases)
        seen = np.isfinite(looked_up)
        point = np.where(seen, looked_up, point)
        point_lookup_misses = horizon - np.count_nonzero(seen)

    with np.errstate(invalid="ignore", over="ignore"):
        sigma = _lag_sigma(filled, lag, filled.size - lag)
        sigma_fallbacks = 0
        if v5_pairs:
            raw = history[-1000:]
            paired = np.isfinite(raw[lag:]) & np.isfinite(raw[:-lag])
            count = np.count_nonzero(paired)
            if count >= max(8, lag):
                sigma = _lag_sigma(raw, lag, count, where=paired)
            else:
                sigma_fallbacks = 1

        growth = np.sqrt(1 + steps // lag)
        if v5_patch:
            for start in range(0, horizon, 16):
                patch = growth[start : start + 16]
                patch[:] = patch.mean()

        levels = np.array([round(abs(q - 0.5) * 200, 10) for q in quantile_levels])
        z = ndtri(0.5 + levels / 200) * np.where(np.asarray(quantile_levels) < 0.5, -1, 1)
        width = sigma * growth
        quantiles = (point[:, None] + width[:, None] * z).astype(history.dtype)

    point[np.isnan(point)] = 0.0
    quantiles[np.isnan(quantiles)] = 0.0
    return point, quantiles, point_lookup_misses, sigma_fallbacks
