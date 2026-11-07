from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.panel import Panel, Roles, Series
from demand_forecast.kaggle.baselines import seasonal_naive_points


def _panel(values, freq="D", seasonal=7, support="nonnegative"):
    """One-series Panel on a grid starting 2017-01-01.

    "Observed" means a finite target value; note that the loader's zero-fill
    policy counts filled zeros as observed.
    """
    values = np.asarray(values, dtype=np.float64)
    series = Series(
        series_id="s",
        start=pd.Timestamp("2017-01-01"),
        history_length=len(values),
        target=values,
        dynamic={},
        statics={},
    )
    return Panel(
        series=[series],
        roles=Roles("target", (), (), (), ()),
        freq=freq,
        seasonal_period=seasonal,
        target_support=support,
    )


def test_point_equals_value_one_season_back():
    panel = _panel(np.arange(1.0, 22.0))
    origin = pd.Timestamp("2017-01-15")  # first forecast step
    frame = seasonal_naive_points(panel, origin, horizon=3)
    first = frame[(frame.series_id == "s") & (frame.timestamp == origin)]
    assert first.point.iloc[0] == pytest.approx(8.0)  # value at 2017-01-08 = one week back


def test_missing_lag_uses_last_observed():
    values = np.array([np.nan] * 15 + [5.0, np.nan, np.nan])
    panel = _panel(values)
    frame = seasonal_naive_points(panel, pd.Timestamp("2017-01-18"), horizon=1)
    assert frame.point.iloc[0] == pytest.approx(5.0)


def test_single_observation_series():
    values = np.array([3.0] + [np.nan] * 20)
    panel = _panel(values)
    frame = seasonal_naive_points(panel, pd.Timestamp("2017-01-20"), horizon=2)
    # Series == approx(scalar) miscompares under pytest 9 / pandas 3; compare as plain floats.
    assert frame.point.tolist() == pytest.approx([3.0, 3.0])


def test_no_history_is_zero_for_nonnegative_support():
    panel = _panel(np.full(21, np.nan))
    frame = seasonal_naive_points(panel, pd.Timestamp("2017-01-20"), horizon=1)
    assert frame.point.iloc[0] == 0.0


def test_no_history_is_nan_for_real_support():
    panel = _panel(np.full(21, np.nan), support="real")
    frame = seasonal_naive_points(panel, pd.Timestamp("2017-01-20"), horizon=1)
    assert np.isnan(frame.point.iloc[0])


def test_observation_inside_the_lag_window_is_ignored():
    values = np.arange(1.0, 22.0)
    values[10] = 99.0  # 2017-01-11: after the lag point, before the origin
    panel = _panel(values)
    frame = seasonal_naive_points(panel, pd.Timestamp("2017-01-15"), horizon=1)
    assert frame.point.iloc[0] == pytest.approx(8.0)  # 2017-01-08, not 2017-01-11


def test_frame_covers_series_times_horizon_on_grid():
    panel = _panel(np.arange(1.0, 22.0))
    panel = replace(panel, series=[*panel.series, replace(panel.series[0], series_id="t")])
    origin = pd.Timestamp("2017-01-15")
    frame = seasonal_naive_points(panel, origin, horizon=3)
    assert list(frame.columns) == ["series_id", "timestamp", "point"]
    frame = frame.sort_values(["series_id", "timestamp"], ignore_index=True)
    assert frame["timestamp"].tolist() == list(pd.date_range(origin, periods=3, freq="D")) * 2
    assert frame["point"].tolist() == pytest.approx([8.0, 9.0, 10.0, 8.0, 9.0, 10.0])
