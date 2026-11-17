"""Prefix-only affine scaling, separate from the model input's soft cap."""

from dataclasses import dataclass

import numpy as np

from demand_forecast.data.scaling import seasonal_error


@dataclass(frozen=True)
class Scale:
    """Location and positive scale in the row's original units."""

    loc: float
    scale: float


def fit_scale(values: np.ndarray, cutoff: int, season: int) -> Scale:
    """Fit the median and seasonal error using only values before the cutoff."""
    prefix = np.asarray(values[:cutoff], dtype=np.float64)
    finite = prefix[np.isfinite(prefix)]
    if not len(finite):
        return Scale(0.0, 1.0)
    loc = float(np.median(finite))
    scale = seasonal_error(prefix, season)
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.mean(np.abs(finite - loc)))
    return Scale(loc, scale if scale > 0 else 1.0)


def to_scaled(values: np.ndarray, s: Scale) -> np.ndarray:
    """Apply affine scaling without capping labels or changing missing cells."""
    return (np.asarray(values, dtype=np.float64) - s.loc) / s.scale


def compress(x: np.ndarray) -> np.ndarray:
    """Soft-cap model inputs with the inverse hyperbolic sine."""
    return np.arcsinh(x)
