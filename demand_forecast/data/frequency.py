"""Frequency conventions shared by the data core and benchmarks."""

import re

import pandas as pd


def seasonal_period(freq: str | pd.offsets.BaseOffset) -> int:
    """Return 7 for daily, 24 for hourly, 12 for monthly, otherwise 1.

    Equivalent fixed-duration offsets share a period. Other step sizes,
    including weekly offsets with any anchor, use the nonseasonal period 1.
    """
    if isinstance(freq, str):
        # H and M warn in pandas 2.2 and are removed in pandas 3. Construct
        # their offsets directly, including aliases with a numeric multiplier.
        legacy = re.fullmatch(r"(\d*)(H|M)", freq.strip())
        if legacy:
            count, alias = legacy.groups()
            offset_type = pd.offsets.Hour if alias == "H" else pd.offsets.MonthEnd
            freq = offset_type(int(count or 1))
    offset = pd.tseries.frequencies.to_offset(freq)
    if isinstance(offset, (pd.offsets.Tick, pd.offsets.Day)):
        if offset.nanos == pd.Timedelta(days=1).value:
            return 7
        if offset.nanos == pd.Timedelta(hours=1).value:
            return 24
    elif isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)) and offset.n == 1:
        return 12
    return 1
