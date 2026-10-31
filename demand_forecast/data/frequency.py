"""Frequency conventions shared by the data core and benchmarks."""

import re

import numpy as np
import pandas as pd


class GridError(ValueError):
    """An unresolved frequency or an observation outside its series' grid."""

    def __init__(
        self,
        message: str,
        *,
        series_id: str | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        super().__init__(message)
        self.series_id = series_id
        self.timestamp = timestamp


def to_offset(freq: str | pd.offsets.BaseOffset) -> pd.offsets.BaseOffset:
    """Parse an offset, accepting the legacy H/M aliases on pandas 2 and 3."""
    legacy = None
    if isinstance(freq, str):
        # H and M warn in pandas 2.2 and are removed in pandas 3.
        legacy = re.fullmatch(r"(\d*)(H|M)", freq.strip())
        if legacy:
            count, alias = legacy.groups()
            freq = count + {"H": "h", "M": "ME"}[alias]
    try:
        return pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        if legacy and legacy.group(2) == "M":
            # pandas 2.0/2.1 predate ME; preserve the original MonthEnd behavior.
            return pd.offsets.MonthEnd(int(legacy.group(1) or 1))
        raise


def seasonal_period(freq: str | pd.offsets.BaseOffset) -> int:
    """Return 7 for daily, 24 for hourly, 12 for monthly, otherwise 1.

    Equivalent fixed-duration offsets share a period. Other step sizes,
    including weekly offsets with any anchor, use the nonseasonal period 1.
    """
    offset = to_offset(freq)
    if isinstance(offset, (pd.offsets.Tick, pd.offsets.Day)):
        if offset.nanos == pd.Timedelta(days=1).value:
            return 7
        if offset.nanos == pd.Timedelta(hours=1).value:
            return 24
    elif isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)) and offset.n == 1:
        return 12
    return 1


def resolve_freq(indexes: list[pd.DatetimeIndex], pinned: str | None) -> pd.offsets.BaseOffset:
    """Use the pinned frequency or require agreement among inferable series."""
    if pinned is not None:
        return to_offset(pinned)
    resolved = None
    for index in indexes:
        inferred = pd.infer_freq(index) if len(index) >= 3 else None
        if inferred is None:
            continue
        offset = to_offset(inferred)
        if resolved is not None and offset != resolved:
            raise GridError(
                f"Inferred frequencies disagree: {resolved.freqstr} and {offset.freqstr}; "
                "pin data.freq and resample explicitly if needed"
            )
        resolved = offset
    if resolved is None:
        raise GridError("No series has an inferable frequency; pin data.freq")
    return resolved


def _requires_midnight(offset: pd.offsets.BaseOffset) -> bool:
    if isinstance(offset, pd.offsets.Tick):
        return offset.nanos >= pd.Timedelta(days=1).value
    return not isinstance(offset, (pd.offsets.BusinessHour, pd.offsets.CustomBusinessHour))


def validate_on_grid(
    index: pd.DatetimeIndex, freq: str | pd.offsets.BaseOffset, series_id: str
) -> None:
    """Reject the first timestamp violating the anchor, phase or midnight rule."""
    offset = to_offset(freq)
    if offset.n <= 0:
        raise ValueError("Grid frequency must have a positive step")
    midnight = _requires_midnight(offset)
    for timestamp in index:
        try:
            if pd.isna(timestamp) or (midnight and timestamp != timestamp.normalize()):
                raise GridError("Timestamp must be present and at midnight")
            steps_between(index[0], timestamp, offset)
        except GridError as error:
            raise GridError(
                f"Series {series_id!r}: timestamp {timestamp} is off the {offset.freqstr} grid; "
                "use data.resample to resample explicitly",
                series_id=series_id,
                timestamp=timestamp,
            ) from error


def grid_range(
    start: pd.Timestamp, periods: int, freq: str | pd.offsets.BaseOffset
) -> pd.DatetimeIndex:
    """Generate grid timestamps using pandas' calendar and time-zone semantics."""
    offset = to_offset(freq)
    if offset.n <= 0:
        raise ValueError("Grid frequency must have a positive step")
    return pd.date_range(start=start, periods=periods, freq=offset)


def steps_between(a: pd.Timestamp, b: pd.Timestamp, freq: str | pd.offsets.BaseOffset) -> int:
    """Return the signed number of whole offset steps, raising for a partial step.

    Days and weeks use local calendar time; sub-daily ticks use elapsed time.
    Common offsets are counted without allocating a grid. Other pandas offsets
    (such as business calendars) fall back to an exact date-range membership check.
    """
    offset = to_offset(freq)
    if offset.n <= 0:
        raise ValueError("Grid frequency must have a positive step")
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if pd.isna(a) or pd.isna(b) or not offset.is_on_offset(a) or not offset.is_on_offset(b):
        raise GridError(f"Endpoints {a} and {b} must lie on {offset.freqstr} offsets", timestamp=b)

    if isinstance(offset, (pd.offsets.Day, pd.offsets.Week)):
        # Day is a Tick in pandas 2, but date_range still preserves local midnight
        # across DST. Handle it before Tick for consistent pandas 2/3 behavior.
        days = offset.n * (7 if isinstance(offset, pd.offsets.Week) else 1)
        steps, remainder = divmod(
            b.tz_localize(None) - a.tz_localize(None), pd.Timedelta(days=days)
        )
    elif isinstance(offset, pd.offsets.Tick):
        steps, remainder = divmod(b - a, pd.Timedelta(offset.nanos, unit="ns"))
    elif isinstance(
        offset,
        (
            pd.offsets.MonthBegin,
            pd.offsets.MonthEnd,
            pd.offsets.QuarterBegin,
            pd.offsets.QuarterEnd,
            pd.offsets.YearBegin,
            pd.offsets.YearEnd,
        ),
    ):
        months = 1
        if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
            months = 3
        elif isinstance(offset, (pd.offsets.YearBegin, pd.offsets.YearEnd)):
            months = 12
        steps, remainder = divmod((b.year - a.year) * 12 + b.month - a.month, months * offset.n)
        if remainder == 0 and a + steps * offset != b:
            remainder = 1
    else:
        start, end = (a, b) if a <= b else (b, a)
        grid = pd.date_range(start=start, end=end, freq=offset)
        steps = (len(grid) - 1) * (1 if a <= b else -1)
        remainder = 0 if len(grid) and grid[-1] == end else 1
    if remainder:
        raise GridError(
            f"Timestamp {b} is not a whole number of {offset.freqstr} steps from {a}", timestamp=b
        )
    return int(steps)


def _calendar_components(offset: pd.offsets.BaseOffset) -> list[tuple[str, int]]:
    daily = [("dow", 7), ("dom", 31), ("doy", 366), ("woy", 53), ("month", 12)]
    if isinstance(offset, pd.offsets.Week):
        return [("woy", 53), ("month", 12)]
    if isinstance(offset, pd.offsets.Day):
        return daily
    if isinstance(offset, pd.offsets.Tick):
        return [("hour", 24), *daily] if offset.nanos < pd.Timedelta(days=1).value else daily
    if isinstance(
        offset,
        (
            pd.offsets.MonthBegin,
            pd.offsets.MonthEnd,
            pd.offsets.BusinessMonthBegin,
            pd.offsets.BusinessMonthEnd,
            pd.offsets.CustomBusinessMonthBegin,
            pd.offsets.CustomBusinessMonthEnd,
        ),
    ):
        return [("month", 12), ("quarter", 4)]
    return [("month", 12)]


def calendar_feature_names(freq: str | pd.offsets.BaseOffset) -> list[str]:
    """Names in output order, with sine immediately followed by cosine."""
    return [
        f"{name}_{function}"
        for name, _ in _calendar_components(to_offset(freq))
        for function in ("sin", "cos")
    ]


def calendar_features(index: pd.DatetimeIndex, freq: str | pd.offsets.BaseOffset) -> np.ndarray:
    """Encode calendar fields as float32 sine/cosine pairs in named column order.

    Use pandas' native numbering (hour/day-of-week start at zero, other fields
    at one), ISO weeks, and fixed cycle lengths 24/7/31/366/53/12/4. Fixed maxima
    keep the same date's encoding independent of its series and leap-year status.
    """
    components = _calendar_components(to_offset(freq))
    fields = {
        "hour": index.hour,
        "dow": index.dayofweek,
        "dom": index.day,
        "doy": index.dayofyear,
        "woy": index.isocalendar().week,
        "month": index.month,
        "quarter": index.quarter,
    }
    features = np.empty((len(index), 2 * len(components)), dtype=np.float32)
    for column, (name, period) in enumerate(components):
        angle = fields[name].to_numpy(dtype=np.float64, na_value=np.nan) * (2 * np.pi / period)
        features[:, 2 * column] = np.sin(angle)
        features[:, 2 * column + 1] = np.cos(angle)
    return features
