"""Recipe-v3 scaler conventions leave shared benchmark frequency logic alone."""

import pandas as pd
import pytest

from demand_forecast.pretrain.periods import pretrain_offset, pretrain_period


@pytest.mark.parametrize(
    "freq,expected",
    [
        ("5T", 288),
        ("5min", 288),
        ("300s", 288),
        ("10T", 144),
        ("15T", 96),
        ("30T", 48),
        ("H", 24),
        ("h", 24),
        ("60min", 24),
        ("D", 7),
        ("24h", 7),
        ("W-SUN", 1),
        ("W-MON", 1),
        ("MS", 12),
        ("M", 12),
        ("ME", 12),
        ("QS", 4),
        ("Q", 4),
        ("QE", 4),
        ("QS-APR", 4),
        ("Q-MAR", 4),
        ("YS", 1),
        ("Y", 1),
        ("YE", 1),
        ("A", 1),
        ("A-JUN", 1),
        (None, 1),
        ("", 1),
        ("unknown", 1),
        ("min", 1),
        ("2h", 1),
        ("2D", 1),
        ("2W-MON", 1),
        ("2M", 1),
        ("2QS", 1),
        ("B", 1),
        ("BM", 1),
        ("BME", 1),
        ("-5min", 1),
        ("0D", 1),
        (pd.offsets.DateOffset(months=1), 1),
    ],
)
def test_pretrain_period_convention(freq, expected):
    assert pretrain_period(freq) == expected


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("W-MON", pd.offsets.Week(weekday=0)),
        ("W-SUN", pd.offsets.Week(weekday=6)),
        ("MS", pd.offsets.MonthBegin()),
        ("M", pd.offsets.MonthEnd()),
        ("ME", pd.offsets.MonthEnd()),
        ("QS", pd.offsets.QuarterBegin(startingMonth=1)),
        ("QS-APR", pd.offsets.QuarterBegin(startingMonth=4)),
        ("Q", pd.offsets.QuarterEnd(startingMonth=12)),
        ("Q-MAR", pd.offsets.QuarterEnd(startingMonth=3)),
        ("QE-MAR", pd.offsets.QuarterEnd(startingMonth=3)),
        ("YS-JUL", pd.offsets.YearBegin(month=7)),
        ("A-JUN", pd.offsets.YearEnd(month=6)),
        ("YE-JUN", pd.offsets.YearEnd(month=6)),
    ],
)
def test_pretrain_offset_preserves_calendar_anchors(alias, expected):
    assert pretrain_offset(alias) == expected


def test_equivalent_ticks_normalize_without_losing_daily_calendar_semantics():
    assert pretrain_offset("60min") == pretrain_offset("H")
    assert pretrain_offset("300s") == pretrain_offset("5T")
    assert isinstance(pretrain_offset("24h"), pd.offsets.Hour)
    assert isinstance(pretrain_offset("D"), pd.offsets.Day)
