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
            pd.Timedelta(minutes=5).value: 288,
            pd.Timedelta(minutes=10).value: 144,
            pd.Timedelta(minutes=15).value: 96,
            pd.Timedelta(minutes=30).value: 48,
            pd.Timedelta(hours=1).value: 24,
            pd.Timedelta(days=1).value: 7,
        }.get(offset.nanos, 1)
    if offset.n == 1:
        if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
            return 12
        if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
            return 4
    return 1
