"""Ragged series on a shared frequency grid, with explicit column roles."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_forecast.data.frequency import grid_range, steps_between, validate_on_grid


@dataclass(frozen=True)
class Roles:
    """Column names grouped by their availability and encoding."""

    target: str
    known_dynamic: tuple[str, ...]
    past_dynamic: tuple[str, ...]
    static_categorical: tuple[str, ...]
    static_numeric: tuple[str, ...]


@dataclass
class Series:
    """One history, with known covariates optionally extending into the future."""

    series_id: str
    start: pd.Timestamp
    history_length: int
    target: np.ndarray
    dynamic: dict[str, np.ndarray]
    statics: dict[str, object]

    @property
    def length(self) -> int:
        """Number of historical grid positions, including missing observations."""
        return self.history_length


@dataclass
class Panel:
    """Series sharing roles and a frequency, with explicit target support."""

    series: list[Series]
    roles: Roles
    freq: str | pd.offsets.BaseOffset
    seasonal_period: int
    target_support: str

    def __post_init__(self) -> None:
        if self.target_support not in ("nonnegative", "real"):
            raise ValueError("target_support must be 'nonnegative' or 'real'")

    def index_of(self, series_id: str) -> int:
        """Return the series' position, raising KeyError for an unknown id."""
        for i, series in enumerate(self.series):
            if series.series_id == series_id:
                return i
        raise KeyError(series_id)

    def timestamps(self, i: int, length: int | None = None) -> pd.DatetimeIndex:
        """Return the series' historical grid, or a requested number of steps."""
        series = self.series[i]
        return grid_range(series.start, length or series.history_length, self.freq)

    def __len__(self) -> int:
        return len(self.series)


def build_series(
    series_id: str,
    frame: pd.DataFrame,
    *,
    freq: str | pd.offsets.BaseOffset,
    roles: Roles,
    missing_target: str,
    statics: dict[str, object],
    history_end: pd.Timestamp | None = None,
) -> Series:
    """Place numeric observations on a grid starting at the earliest timestamp.

    An explicit, inclusive ``history_end`` fixes the history even if its target
    is missing or its row is absent. Otherwise only observed target or past
    covariate values extend history; with none, history has zero length.
    ``missing_target`` fills absent historical rows only, preserving explicit
    NaNs. Known covariates cover at least the history and all supplied future
    rows. Empty frames and duplicate timestamps must be resolved by the loader.
    """
    if missing_target not in ("zero", "nan"):
        raise ValueError("missing_target must be 'zero' or 'nan'")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame must have a DatetimeIndex")
    if len(frame) == 0:
        raise ValueError(f"Series {series_id!r}: cannot build from an empty frame")
    validate_on_grid(frame.index, freq, series_id)
    if not frame.index.is_unique:
        raise ValueError(f"Series {series_id!r}: duplicate timestamps")

    frame = frame.sort_index()
    start = frame.index[0]
    positions = np.array([steps_between(start, timestamp, freq) for timestamp in frame.index])
    if history_end is not None:
        history_end = pd.Timestamp(history_end)
        validate_on_grid(pd.DatetimeIndex([start, history_end]), freq, series_id)
        if history_end < start:
            raise ValueError(f"Series {series_id!r}: history_end precedes the first timestamp")
        history_length = steps_between(start, history_end, freq) + 1
    else:
        observed = frame[[roles.target, *roles.past_dynamic]].notna().any(axis=1).to_numpy()
        history_length = int(positions[observed][-1]) + 1 if observed.any() else 0

    historical_rows = positions < history_length
    historical_positions = positions[historical_rows]
    target = np.full(history_length, 0.0 if missing_target == "zero" else np.nan)
    target[historical_positions] = frame.loc[historical_rows, roles.target].to_numpy(
        dtype=float, na_value=np.nan
    )

    dynamic = {}
    known_length = max(history_length, int(positions[-1]) + 1)
    for name in roles.known_dynamic:
        values = np.full(known_length, np.nan)
        values[positions] = frame[name].to_numpy(dtype=float, na_value=np.nan)
        dynamic[name] = values
    for name in roles.past_dynamic:
        values = np.full(history_length, np.nan)
        values[historical_positions] = frame.loc[historical_rows, name].to_numpy(
            dtype=float, na_value=np.nan
        )
        dynamic[name] = values

    return Series(series_id, start, history_length, target, dynamic, dict(statics))
