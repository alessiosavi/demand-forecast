"""Vectorised causal features from each row's bounded, earlier history."""

import numpy as np
import pandas as pd

from .scale import compress
from .schema import BANK_WINDOW, STEP_FEATURES


def step_features(
    scaled: np.ndarray,
    raw: np.ndarray,
    *,
    season: int,
    long_lag: int | None,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 15 features and availability flags for local indices [left, right).

    Every feature reads strictly earlier grid cells, at most BANK_WINDOW back.
    Rolling deviations use population standard deviation with two-cell support.
    Missing cells neither extend nor end a zero run; elapsed time uses grid steps.
    A zero run is counted inside the bank window only, so it is capped at BANK_WINDOW.
    """
    scaled = np.asarray(scaled, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    if scaled.ndim != 1 or raw.ndim != 1 or scaled.shape != raw.shape:
        raise ValueError("scaled and raw must be one-dimensional arrays of equal length")
    if not 0 <= left <= right <= len(raw):
        raise ValueError("expected 0 <= left <= right <= the row length")
    if season < 1 or (long_lag is not None and long_lag < 1):
        raise ValueError("season and long_lag must be positive")
    values = np.zeros((right - left, len(STEP_FEATURES)), dtype=np.float64)
    flags = np.zeros(values.shape, dtype=bool)
    if left == right:
        return values.astype(np.float32), flags

    history_left = max(0, left - BANK_WINDOW)
    steps = np.arange(left, right) - history_left
    history = scaled[history_left:right]
    series = pd.Series(np.where(np.isfinite(history), history, np.nan))
    for column, window in ((0, max(4, season)), (4, max(16, 4 * season))):
        rolling = series.rolling(min(window, BANK_WINDOW), min_periods=1)
        statistics = np.column_stack(
            [
                stat.shift(1).to_numpy()[steps]
                for stat in (rolling.mean(), rolling.std(ddof=0), rolling.min(), rolling.max())
            ]
        )
        valid = np.isfinite(statistics)
        valid[:, 1] &= rolling.count().shift(1).to_numpy()[steps] >= 2
        values[:, column : column + 4] = np.where(valid, statistics, 0.0)
        flags[:, column : column + 4] = valid

    def lagged(lag: int) -> tuple[np.ndarray, np.ndarray]:
        positions = steps - lag
        valid = (positions >= 0) & (lag <= BANK_WINDOW)
        cells = history[np.maximum(positions, 0)]
        return cells, valid & np.isfinite(cells)

    previous, previous_valid = lagged(1)
    for column, lag, difference in (
        (8, 2, True),
        (9, season + 1, True),
        (10, season, False),
        (11, long_lag, False),
    ):
        if lag is None or (column in (9, 10) and season == 1):
            continue
        cells, valid = lagged(lag)
        if difference:
            cells = previous - cells
            valid &= previous_valid & np.isfinite(cells)
        values[:, column] = np.where(valid, cells, 0.0)
        flags[:, column] = valid
    values[:, :12] = compress(values[:, :12])

    raw_history = raw[history_left:right]
    observed = np.isfinite(raw_history)
    zeros = observed & (raw_history == 0)
    observed_prefix = np.r_[0, np.cumsum(observed)]
    zero_prefix = np.r_[0, np.cumsum(zeros)]
    window_left = np.maximum(0, steps - BANK_WINDOW)
    count = observed_prefix[steps] - observed_prefix[window_left]
    zero_count = zero_prefix[steps] - zero_prefix[window_left]
    has_history = count > 0
    np.divide(zero_count, count, out=values[:, 12], where=has_history)
    flags[:, 12] = has_history

    positions = np.arange(len(raw_history))
    last_nonzero = np.maximum.accumulate(np.where(observed & ~zeros, positions, -1))
    last = np.r_[-1, last_nonzero][steps]
    run_left = np.maximum(window_left, last + 1)
    values[:, 13] = np.log1p(zero_prefix[steps] - zero_prefix[run_left])
    flags[:, 13] = has_history
    flags[:, 14] = last >= window_left
    values[:, 14] = np.where(flags[:, 14], np.log1p(steps - 1 - last), 0.0)
    return values.astype(np.float32), flags
