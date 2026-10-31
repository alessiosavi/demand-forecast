import warnings

import pandas as pd
import pytest

from demand_forecast.data.frequency import seasonal_period


@pytest.mark.parametrize(
    "freq,expected",
    [
        ("W-SUN", 1),
        ("W-WED", 1),
        ("D", 7),
        ("H", 24),
        ("h", 24),
        ("1H", 24),
        ("MS", 12),
        ("M", 12),
        ("1M", 12),
        ("15min", 1),
        ("60min", 24),
        ("24h", 7),
        ("2D", 1),
        ("2h", 1),
        ("2MS", 1),
        ("B", 1),
        ("s", 1),
        (pd.offsets.Week(weekday=2), 1),
        (pd.offsets.Day(), 7),
        (pd.offsets.Hour(), 24),
        (pd.offsets.MonthBegin(), 12),
        (pd.offsets.MonthEnd(), 12),
        (pd.offsets.Minute(15), 1),
    ],
)
def test_seasonal_period(freq, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert seasonal_period(freq) == expected


def test_seasonal_period_rejects_invalid_frequency():
    with pytest.raises(ValueError):
        seasonal_period("not-a-frequency")
