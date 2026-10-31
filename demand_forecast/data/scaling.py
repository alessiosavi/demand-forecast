"""Per-sample target location, full-past seasonal scale and eligibility."""

from dataclasses import dataclass

import numpy as np


def seasonal_error(values: np.ndarray, m: int) -> float:
    """Mean absolute m-lag error of a 1D history, for a positive seasonal lag.

    Only pairs with two non-NaN endpoints contribute; missing positions are
    never removed before pairing. No observed pair gives an undefined scale.
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= m:
        return float("nan")
    current, previous = values[m:], values[:-m]
    observed = ~np.isnan(current) & ~np.isnan(previous)
    if not observed.any():
        return float("nan")
    with np.errstate(invalid="ignore", over="ignore"):
        return float(np.abs(current[observed] - previous[observed]).mean())


@dataclass(frozen=True)
class Frame:
    """Affine normalisation parameters and independent scale/context flags."""

    loc: float
    scale: float
    scale_raw: float
    eligible: bool
    has_context: bool


def sample_frame(context: np.ndarray, past: np.ndarray, m: int, min_scale: float = 1e-6) -> Frame:
    """Use the context median and full-past seasonal error, flooring eligible scales only."""
    context = np.asarray(context, dtype=float)
    has_context = bool((~np.isnan(context)).any())
    loc = float(np.nanmedian(context)) if has_context else 0.0
    scale_raw = seasonal_error(past, m)
    eligible = bool(np.isfinite(scale_raw) and scale_raw > 0)
    scale = max(scale_raw, min_scale) if eligible else 1.0
    return Frame(
        loc=loc, scale=scale, scale_raw=scale_raw, eligible=eligible, has_context=has_context
    )
