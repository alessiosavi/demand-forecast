"""Recipe-v3 offset parsing and scaler periods, isolated from bench conventions."""

import calendar
import re

import pandas as pd


def pretrain_offset(freq: str | pd.offsets.BaseOffset) -> pd.offsets.BaseOffset:
    """Parse legacy/modern aliases without losing begin/end or month/week anchors."""
    if isinstance(freq, str):
        freq = freq.strip()
        match = re.fullmatch(r"(\d*)(MS|ME|M|QS|QE|Q|YS|YE|Y|AS|A)(?:-([A-Z]{3}))?", freq)
        if match:
            count, alias, anchor = match.groups()
            n = int(count or 1)
            months = {name.upper(): i for i, name in enumerate(calendar.month_abbr) if i}
            if alias in ("MS", "ME", "M") and anchor is None:
                cls = pd.offsets.MonthBegin if alias == "MS" else pd.offsets.MonthEnd
                return cls(n)
            if anchor is not None and anchor not in months:
                raise ValueError(f"Unknown calendar anchor: {anchor}")
            if alias in ("QS", "QE", "Q"):
                begin = alias == "QS"
                cls = pd.offsets.QuarterBegin if begin else pd.offsets.QuarterEnd
                # pandas 2/3 use startingMonth, not month. Bare aliases use
                # January/December, unlike the constructors' March default.
                return cls(n, startingMonth=months[anchor] if anchor else (1 if begin else 12))
            if alias in ("YS", "YE", "Y", "AS", "A"):
                begin = alias in ("YS", "AS")
                cls = pd.offsets.YearBegin if begin else pd.offsets.YearEnd
                return cls(n, month=months[anchor] if anchor else (1 if begin else 12))
        match = re.fullmatch(r"(\d*)(H|T)", freq)
        if match:
            count, alias = match.groups()
            freq = count + {"H": "h", "T": "min"}[alias]
    offset = pd.tseries.frequencies.to_offset(freq)
    if isinstance(offset, pd.offsets.Tick) and not isinstance(offset, pd.offsets.Day):
        # Equivalent tick classes need explicit normalization on pandas 3.
        # Keep 24h as elapsed hours: converting it to Day changes DST semantics.
        for cls, nanos in (
            (pd.offsets.Hour, 3_600_000_000_000),
            (pd.offsets.Minute, 60_000_000_000),
            (pd.offsets.Second, 1_000_000_000),
            (pd.offsets.Milli, 1_000_000),
            (pd.offsets.Micro, 1_000),
            (pd.offsets.Nano, 1),
        ):
            if offset.nanos % nanos == 0:
                return cls(offset.nanos // nanos)
    return offset


def pretrain_period(freq: str | pd.offsets.BaseOffset | None) -> int:
    """Return the v3 frequency convention; unspecified/unsupported offsets use one."""
    if freq is None:
        return 1
    try:
        offset = pretrain_offset(freq)
    except (TypeError, ValueError):
        return 1
    if isinstance(offset, (pd.offsets.Tick, pd.offsets.Day)):
        return {
            pd.Timedelta(5, "min").value: 288,
            pd.Timedelta(10, "min").value: 144,
            pd.Timedelta(15, "min").value: 96,
            pd.Timedelta(30, "min").value: 48,
            pd.Timedelta(1, "h").value: 24,
            pd.Timedelta(1, "D").value: 7,
        }.get(offset.nanos, 1)
    if offset.n == 1:
        if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
            return 12
        if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
            return 4
    return 1


def pretrain_long_lag(
    freq: str | pd.offsets.BaseOffset | None, seasonal_period: int | None = None
) -> int:
    """Long-cycle lag by frequency; unsupported grids use the sample's period.

    Without a task-specific period, fall back to the recipe-v3 convention.
    """
    fallback = pretrain_period(freq) if seasonal_period is None else seasonal_period
    if freq is None:
        return fallback
    try:
        offset = pretrain_offset(freq)
    except (TypeError, ValueError):
        return fallback
    if isinstance(offset, (pd.offsets.Tick, pd.offsets.Day)):
        return {
            pd.Timedelta(5, "min").value: 2016,
            pd.Timedelta(10, "min").value: 1008,
            pd.Timedelta(15, "min").value: 672,
            pd.Timedelta(30, "min").value: 336,
            pd.Timedelta(1, "h").value: 168,
            pd.Timedelta(1, "D").value: 364,
        }.get(offset.nanos, fallback)
    if offset.n == 1:
        if isinstance(offset, pd.offsets.Week):
            return 52
        if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
            return 24
        if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
            return 8
    return fallback
