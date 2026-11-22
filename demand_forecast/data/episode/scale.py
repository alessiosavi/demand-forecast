"""Prefix-only affine scaling, separate from the model input's soft cap."""

from dataclasses import dataclass

import numpy as np

from demand_forecast.data.scaling import seasonal_error

from .schema import MAX_CONTEXT

_FLOAT64_MAX = float(np.finfo(np.float64).max)


@dataclass(frozen=True)
class Scale:
    """Location and positive scale in the row's original units."""

    loc: float
    scale: float


def finite_scale_values(values: np.ndarray, cutoff: int, *, kind: str | None = None) -> np.ndarray:
    """Return the finite prefix cells used to fit this kind of scale."""
    end = min(len(values), max(0, cutoff))
    if kind == "spread" and end > MAX_CONTEXT:
        window = values[end - MAX_CONTEXT : end]
        finite = window[np.isfinite(window)]
        if finite.size:
            return finite
    prefix = values[:end]
    return prefix[np.isfinite(prefix)]


def _spread_frame(finite: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return the finite cells and median in units of their largest magnitude."""
    magnitude = float(np.max(np.abs(finite)))
    with np.errstate(under="ignore"):
        unit = finite / magnitude if magnitude else finite
    return unit, float(np.median(unit)), magnitude


def spread_location(finite: np.ndarray) -> float:
    """Take the spread median without overflowing on large even-sized samples."""
    _, median, magnitude = _spread_frame(finite)
    return median * magnitude


def _rescale(value: float, magnitude: float) -> float:
    # Equality can overflow too when the division rounds the threshold up.
    if value > 1.0 and magnitude >= _FLOAT64_MAX / value:
        return _FLOAT64_MAX
    return value * magnitude


def fit_scale(
    values: np.ndarray,
    cutoff: int,
    season: int,
    *,
    floor: float | None = None,
    kind: str | None = "seasonal",
) -> Scale:
    """Fit a prefix-only scale with an optional floor relative to the mean absolute level."""
    if kind not in (None, "seasonal", "spread"):
        raise ValueError("scale_kind must be 'seasonal' or 'spread'")
    prefix = np.asarray(values[: max(0, cutoff)], dtype=np.float64)
    finite = finite_scale_values(prefix, len(prefix), kind=kind)
    if not len(finite):
        return Scale(0.0, 1.0)
    if kind == "spread":
        unit, median, magnitude = _spread_frame(finite)
        with np.errstate(under="ignore"):
            rms = float(np.sqrt(np.mean(np.square(unit - median))))
        scale = _rescale(rms, magnitude) or 1.0
        if floor is not None:
            mean_abs = _rescale(float(np.mean(np.abs(unit))), magnitude)
            scale = max(scale, _rescale(floor, mean_abs))
        return Scale(median * magnitude, scale)
    loc = float(np.median(finite))
    scale = seasonal_error(prefix, season)
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.mean(np.abs(finite - loc)))
    scale = scale if scale > 0 else 1.0
    if floor is not None:
        scale = max(scale, floor * float(np.mean(np.abs(finite))))
    return Scale(loc, scale)


def to_scaled(values: np.ndarray, s: Scale) -> np.ndarray:
    """Apply affine scaling without capping labels or changing missing cells."""
    return (np.asarray(values, dtype=np.float64) - s.loc) / s.scale


def compress(x: np.ndarray) -> np.ndarray:
    """Soft-cap model inputs with the inverse hyperbolic sine."""
    return np.arcsinh(x)
