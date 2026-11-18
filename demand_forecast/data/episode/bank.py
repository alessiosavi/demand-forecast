"""Vectorised causal features from each row's bounded, earlier history."""

from typing import Any

import numpy as np
import pandas as pd

from .scale import compress
from .schema import BANK_WINDOW, STEP_FEATURES, SUMMARY_FEATURES

# Reference arguments for the parity fixture (season=7, long_lag=30).
# summary_vectors always uses its caller's lags, without mutating this table.
TSFRESH_PARITY: dict[str, tuple[str, dict[str, Any]]] = {
    "skewness": ("skewness", {}),
    "kurtosis": ("kurtosis", {}),
    "trend_slope": ("linear_trend", {"param": [{"attr": "slope"}]}),
    "trend_strength": ("linear_trend", {"param": [{"attr": "rvalue"}]}),
    "acf_1": ("autocorrelation", {"lag": 1}),
    "acf_season": ("autocorrelation", {"lag": 7}),
    "acf_long": ("autocorrelation", {"lag": 30}),
    "mean_abs_change": ("mean_abs_change", {}),
    "cid_ce": ("cid_ce", {"normalize": False}),
    "beyond_2_sigma": ("ratio_beyond_r_sigma", {"r": 2}),
    "longest_above": ("longest_strike_above_mean", {}),
    "longest_below": ("longest_strike_below_mean", {}),
}


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


def _sample_shape_statistics(
    centered: np.ndarray, sum_squares: np.ndarray, count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjusted skewness and excess kurtosis from zero-padded deviations."""
    n = count.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # These agree with pandas 3.1.1's reducers up to floating-point rounding
        # when the common offset is not many orders of magnitude above the spread.
        # The affine scale's median location removes the offset when the fitting
        # history represents the summary window. For a row constant up to rounding
        # noise around a large offset, both reducers work on rounding noise and
        # their results can differ. This calculation is deterministic across NumPy
        # versions and corrects the central sums for the residual mean.
        residual = centered.sum(axis=1)
        shift = residual / n
        shift_squared = shift * shift
        correction = residual * shift
        squared = centered * centered
        cubes = (squared * centered).sum(axis=1)
        fourth_powers = (squared * squared).sum(axis=1)
        m2 = np.maximum(sum_squares - correction, 0.0)
        m3 = (cubes - 3 * sum_squares * shift) + (shift + shift) * correction
        m4 = ((fourth_powers - 4 * cubes * shift) + 6 * sum_squares * shift_squared) - (
            3 * correction * shift_squared
        )
        m4 = np.maximum(m4, 0.0)

        skewness = (m3 / (m2 * np.sqrt(m2))) * (n * np.sqrt(n - 1) / (n - 2))
        kurtosis = ((m4 / (m2 * m2)) * ((n + 1) * n) - 3 * (n - 1)) * (
            (n - 1) / ((n - 2) * (n - 3))
        )
    return np.where((count >= 3) & (m2 > 0), skewness, np.nan), np.where(
        (count >= 4) & (m2 > 0), kurtosis, np.nan
    )


def summary_vectors(
    scaled: np.ndarray,
    raw: np.ndarray,
    *,
    season: int,
    long_lag: int | None,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarise caller-supplied histories strictly before the statistics cutoff.

    Leading missing cells are padding; later gaps and trailing missing cells
    belong to the span n. Nonseasonal calculators, runs and intermittency use
    observed sequences, while seasonal pairs and profiles retain grid positions.
    Trend slope, complexity and runs are normalised by n, including its gaps.
    Scaled inputs are affine-scaled values, before the per-step asinh cap.
    cv2_nonzero is invalid for fewer than 2 non-zero values, or their mean is 0.
    """
    scaled = np.asarray(scaled, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    if scaled.ndim != 2 or raw.ndim != 2 or scaled.shape != raw.shape:
        raise ValueError("scaled and raw must be two-dimensional arrays of equal shape")
    if season < 1 or (long_lag is not None and long_lag < 1):
        raise ValueError("season and long_lag must be positive")
    rows, length = scaled.shape
    values = np.zeros((rows, len(SUMMARY_FEATURES)), dtype=np.float64)
    flags = np.zeros(values.shape, dtype=bool)
    if rows == 0 or length == 0:
        return values.astype(np.float32), flags

    def store(name: str, data: np.ndarray, valid: np.ndarray) -> None:
        column = SUMMARY_FEATURES.index(name)
        valid = valid & np.isfinite(data)
        values[:, column] = np.where(valid, np.clip(data, -10, 10), 0.0)
        flags[:, column] = valid

    observed = np.isfinite(scaled) & np.isfinite(raw)
    count = observed.sum(axis=1)
    has_values = count > 0
    first = observed.argmax(axis=1)
    span = np.where(has_values, length - first, 0)
    safe_count = np.maximum(count, 1)
    safe_span = np.maximum(span, 1)
    positions = np.arange(length)
    packed_valid = positions[None, :] < count[:, None]

    # Scatter all observed cells into contiguous sequences, without a row loop.
    row_index, grid_index = np.nonzero(observed)
    packed_index = observed.cumsum(axis=1)[row_index, grid_index] - 1
    packed = np.zeros_like(scaled)
    packed_raw = np.zeros_like(raw)
    packed[row_index, packed_index] = scaled[row_index, grid_index]
    packed_raw[row_index, packed_index] = raw[row_index, grid_index]

    # Undefined arithmetic is filtered by store, so invalid entries stay finite zero.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean = packed.sum(axis=1) / safe_count
        centered = np.where(packed_valid, packed - mean[:, None], 0.0)
        sum_squares = (centered**2).sum(axis=1)
        variance = sum_squares / safe_count
        nonconstant = ((packed != packed[:, :1]) & packed_valid).any(axis=1)
        skewness, kurtosis = _sample_shape_statistics(centered, sum_squares, count)
        shape_valid = (count >= 4) & nonconstant
        store("skewness", skewness, shape_valid)
        store("kurtosis", kurtosis, shape_valid)

        raw_mean = packed_raw.sum(axis=1) / safe_count
        raw_variance = np.where(packed_valid, packed_raw - raw_mean[:, None], 0.0)
        raw_std = np.sqrt((raw_variance**2).sum(axis=1) / safe_count)
        abs_level = np.abs(packed_raw).sum(axis=1) / safe_count
        store("spread_over_level", raw_std / abs_level, has_values & (abs_level > 0))
        store("zero_share", ((packed_raw == 0) & packed_valid).sum(axis=1) / safe_count, has_values)

        nonzero = packed_valid & (packed_raw != 0)
        nonzero_count = nonzero.sum(axis=1)
        first_nonzero = np.where(nonzero, positions, length).min(axis=1)
        last_nonzero = np.where(nonzero, positions, -1).max(axis=1)
        mean_gap = (last_nonzero - first_nonzero) / np.maximum(nonzero_count - 1, 1)
        store("adi", np.log(mean_gap), nonzero_count >= 2)
        nonzero_mean = np.where(nonzero, packed_raw, 0.0).sum(axis=1) / np.maximum(nonzero_count, 1)
        nonzero_deviation = np.where(nonzero, packed_raw - nonzero_mean[:, None], 0.0)
        nonzero_variance = (nonzero_deviation**2).sum(axis=1) / np.maximum(nonzero_count, 1)
        store(
            "cv2_nonzero",
            nonzero_variance / nonzero_mean**2,
            (nonzero_count >= 2) & (nonzero_mean != 0),
        )

        time_centered = positions[None, :] - (count[:, None] - 1) / 2
        time_squares = count * (count.astype(np.float64) ** 2 - 1) / 12
        covariance = (time_centered * centered).sum(axis=1)
        trend_valid = (count >= 3) & nonconstant
        store("trend_slope", covariance / time_squares * span, trend_valid)
        store(
            "trend_strength",
            np.clip(covariance**2 / (time_squares * sum_squares), 0, 1),
            trend_valid,
        )

        grid_centered = np.where(observed, scaled - mean[:, None], 0.0)
        for name, lag in (("acf_1", 1), ("acf_season", season), ("acf_long", long_lag)):
            if lag is None or lag >= length or (name == "acf_season" and season == 1):
                continue
            cells = centered if name == "acf_1" else grid_centered
            valid_cells = packed_valid if name == "acf_1" else observed
            pairs = valid_cells[:, :-lag] & valid_cells[:, lag:]
            pair_count = pairs.sum(axis=1)
            products = np.where(pairs, cells[:, :-lag] * cells[:, lag:], 0.0).sum(axis=1)
            # tsfresh treats variances close to zero as undefined autocorrelations.
            store(
                name,
                products / (pair_count * variance),
                nonconstant & (pair_count > 0) & ~np.isclose(variance, 0),
            )

        spectrum_valid = (count >= 16) & nonconstant
        spectrum_period = np.zeros(rows)
        spectrum_strength = np.zeros(rows)
        # rFFT has a scalar length: batch equal spans to preserve ragged-row spectra.
        for n in np.unique(span[spectrum_valid]):
            selected = np.flatnonzero(spectrum_valid & (span == n))
            cells = np.take_along_axis(
                grid_centered[selected], first[selected, None] + np.arange(n), axis=1
            )
            power = np.abs(np.fft.rfft(cells, axis=1)) ** 2
            k = 1 + power[:, 1:].argmax(axis=1)
            spectrum_period[selected] = np.log(n / k) / np.log(n)
            spectrum_strength[selected] = power[np.arange(len(selected)), k] / power[:, 1:].sum(
                axis=1
            )
        store("spec_period", spectrum_period, spectrum_valid & np.isfinite(spectrum_strength))
        store("spec_strength", spectrum_strength, spectrum_valid)

        differences = np.where(packed_valid[:, 1:], np.diff(packed, axis=1), 0.0)
        store(
            "mean_abs_change",
            np.abs(differences).sum(axis=1) / np.maximum(count - 1, 1),
            count >= 2,
        )
        store("cid_ce", np.sqrt((differences**2).sum(axis=1) / safe_span), count >= 2)
        beyond = (
            (np.abs(centered) > 2 * np.sqrt(variance)[:, None])
            & packed_valid
            & nonconstant[:, None]
        )
        store("beyond_2_sigma", beyond.sum(axis=1) / safe_count, has_values)
        for name, condition in (("longest_above", centered > 0), ("longest_below", centered < 0)):
            condition &= packed_valid & nonconstant[:, None]
            last_break = np.maximum.accumulate(np.where(~condition, positions, -1), axis=1)
            longest = np.where(condition, positions - last_break, 0).max(axis=1)
            store(name, longest / safe_span, has_values)

        phase_bin = ((end - length + positions) % season) * min(8, season) // season
        for b in range(min(8, season) if season > 1 else 0):
            cells = observed & (phase_bin == b)
            support = cells.sum(axis=1)
            profile = np.where(cells, scaled, 0.0).sum(axis=1) / np.maximum(support, 1)
            store(f"profile_{b}", profile, support > 0)
        store("log_length", np.log(safe_span) / np.log(BANK_WINDOW), has_values)
        store("missing_share", 1 - count / safe_span, has_values)

    return values.astype(np.float32), flags
