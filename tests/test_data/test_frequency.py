import warnings

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.frequency import (
    GridError,
    calendar_feature_names,
    calendar_features,
    grid_range,
    resolve_freq,
    seasonal_period,
    steps_between,
    to_offset,
    validate_on_grid,
)


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


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("H", pd.offsets.Hour()),
        ("2H", pd.offsets.Hour(2)),
        ("M", pd.offsets.MonthEnd()),
        ("2M", pd.offsets.MonthEnd(2)),
    ],
)
def test_to_offset_normalizes_legacy_aliases_without_warnings(alias, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert to_offset(alias) == expected
        assert to_offset(expected) == expected


def test_resolve_freq_infers_sunday_weeks_and_short_series_inherit():
    long = pd.date_range("2023-01-01", periods=10, freq="W-SUN")
    short = pd.DatetimeIndex(["2023-03-05", "2023-04-02"])
    assert resolve_freq([long, short], None).freqstr == "W-SUN"


@pytest.mark.parametrize(
    "indexes",
    [
        [],
        [pd.DatetimeIndex([])],
        [pd.DatetimeIndex(["2023-01-01", "2023-02-15"])],
        [pd.DatetimeIndex(["2023-01-01", "2023-01-08", "2023-01-29"])],
    ],
)
def test_resolve_freq_without_inferable_series_asks_to_pin(indexes):
    with pytest.raises(GridError, match="pin data.freq") as exc:
        resolve_freq(indexes, None)
    assert exc.value.series_id is None and exc.value.timestamp is None


def test_resolve_freq_gapped_series_inherits_inferred_frequency():
    gapped = pd.DatetimeIndex(["2023-01-01", "2023-01-08", "2023-01-29"])
    regular = pd.date_range("2023-02-05", periods=4, freq="W-SUN")
    assert resolve_freq([gapped, regular], None) == to_offset("W-SUN")


@pytest.mark.parametrize("other", ["W-MON", "2W-SUN", "D"])
def test_resolve_freq_requires_all_inferred_frequencies_to_agree(other):
    indexes = [pd.date_range("2023-01-01", periods=5, freq=f) for f in ["W-SUN", other]]
    with pytest.raises(GridError, match="disagree"):
        resolve_freq(indexes, None)


@pytest.mark.parametrize("pinned", ["H", "M", "W-MON"])
def test_resolve_freq_pinned_wins_over_conflicting_inferences(pinned):
    indexes = [pd.date_range("2023-01-01", periods=5, freq=f) for f in ["W-SUN", "D"]]
    assert resolve_freq(indexes, pinned) == to_offset(pinned)
    assert resolve_freq([], pinned) == to_offset(pinned)


def test_pinned_monday_grid_rejects_sunday_data():
    idx = pd.date_range("2023-01-01", periods=5, freq="W-SUN")
    with pytest.raises(GridError) as exc:
        validate_on_grid(idx, resolve_freq([idx], "W-MON"), "s1")
    assert (exc.value.series_id, exc.value.timestamp) == ("s1", idx[0])
    assert "s1" in str(exc.value) and "2023-01-01" in str(exc.value)
    assert "resample" in str(exc.value)


def test_time_of_day_is_off_grid_for_one_row_and_for_a_whole_series():
    f = to_offset("W-SUN")
    idx = pd.DatetimeIndex([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-08 13:00")])
    with pytest.raises(GridError) as exc:
        validate_on_grid(idx, f, "s1")
    assert (exc.value.series_id, exc.value.timestamp) == ("s1", idx[1])
    idx = pd.date_range("2023-01-01 13:00", periods=4, freq=f)
    with pytest.raises(GridError) as exc:
        validate_on_grid(idx, f, "s2")
    assert (exc.value.series_id, exc.value.timestamp) == ("s2", idx[0])
    validate_on_grid(pd.date_range("2023-01-01 13:00", periods=4, freq=to_offset("H")), "H", "s3")


@pytest.mark.parametrize("freq", ["D", "2D", "W-WED", "MS", "M", "QS", "YS", "24H", "48h"])
def test_daily_and_coarser_grids_reject_nonmidnight_singletons(freq):
    timestamp = pd.date_range("2023-01-01", periods=1, freq=to_offset(freq))[0]
    validate_on_grid(pd.DatetimeIndex([timestamp]), freq, "s1")
    bad = timestamp + pd.Timedelta(nanoseconds=1)
    with pytest.raises(GridError) as exc:
        validate_on_grid(pd.DatetimeIndex([bad]), freq, "s1")
    assert exc.value.timestamp == bad


@pytest.mark.parametrize(
    "freq,start,off_grid",
    [
        ("2W-SUN", "2023-01-01", "2023-01-08"),
        ("2D", "2023-01-01", "2023-01-02"),
        ("2MS", "2023-01-01", "2023-02-01"),
        ("2M", "2023-01-31", "2023-02-28"),
        ("2QS", "2023-01-01", "2023-04-01"),
        ("2YS", "2023-01-01", "2024-01-01"),
        ("2H", "2023-01-01 13:05", "2023-01-01 14:05"),
        ("15min", "2023-01-01 13:05", "2023-01-01 13:12"),
    ],
)
def test_grid_accepts_gaps_but_rejects_fractional_steps(freq, start, off_grid):
    regular = pd.date_range(start, periods=5, freq=to_offset(freq))
    validate_on_grid(regular[[0, 2, 4]], freq, "s1")
    idx = pd.DatetimeIndex([regular[0], pd.Timestamp(off_grid), regular[-1]])
    with pytest.raises(GridError) as exc:
        validate_on_grid(idx, freq, "s1")
    assert (exc.value.series_id, exc.value.timestamp) == ("s1", idx[1])
    with pytest.raises(GridError):
        steps_between(regular[0], idx[1], freq)


def test_grid_reports_first_offender_in_input_order_and_rejects_nat():
    idx = pd.DatetimeIndex(["2023-01-01", "2023-01-15", "2023-01-08", "2023-01-02"])
    with pytest.raises(GridError) as exc:
        validate_on_grid(idx, "2W-SUN", "s1")
    assert exc.value.timestamp == idx[2]
    with pytest.raises(GridError) as exc:
        validate_on_grid(pd.DatetimeIndex([pd.NaT]), "D", "s2")
    assert exc.value.series_id == "s2" and pd.isna(exc.value.timestamp)
    validate_on_grid(pd.DatetimeIndex([]), "W-SUN", "empty")


def test_steps_between_and_grid_range():
    f = to_offset("W-SUN")
    assert steps_between(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-22"), f) == 3
    assert grid_range(pd.Timestamp("2023-01-01"), 3, f)[-1] == pd.Timestamp("2023-01-15")


@pytest.mark.parametrize(
    "freq,start,end,steps",
    [
        ("MS", "2023-01-01", "2023-04-01", 3),
        ("M", "2024-01-31", "2024-03-31", 2),
        ("2W-WED", "2023-01-04", "2023-02-15", 3),
        ("2QS", "2023-01-01", "2024-07-01", 3),
        ("2YS", "2020-01-01", "2026-01-01", 3),
        ("B", "2023-01-06", "2023-01-10", 2),
        ("15min", "2023-01-01 13:05", "2023-01-01 13:50", 3),
    ],
)
def test_step_counts_are_exact_signed_and_match_generated_grid(freq, start, end, steps):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    assert steps_between(a, b, freq) == steps
    assert steps_between(b, a, freq) == -steps
    assert steps_between(a, a, freq) == 0
    assert grid_range(a, steps + 1, freq)[-1] == b
    assert len(grid_range(a, 0, freq)) == 0


@pytest.mark.parametrize(
    "freq,start",
    [
        ("D", "2023-03-25"),
        ("2D", "2023-03-25"),
        ("W-SUN", "2023-03-19"),
        ("MS", "2023-03-01"),
        ("H", "2023-03-26 00:30"),
    ],
)
def test_grids_and_steps_agree_across_daylight_saving_time(freq, start):
    idx = grid_range(pd.Timestamp(start, tz="Europe/Rome"), 5, freq)
    validate_on_grid(idx, freq, "dst")
    assert steps_between(idx[0], idx[-1], freq) == 4


@pytest.mark.parametrize(
    "freq,fields",
    [
        ("W-SUN", ["woy", "month"]),
        ("W-WED", ["woy", "month"]),
        ("D", ["dow", "dom", "doy", "woy", "month"]),
        ("24H", ["dow", "dom", "doy", "woy", "month"]),
        ("H", ["hour", "dow", "dom", "doy", "woy", "month"]),
        ("15min", ["hour", "dow", "dom", "doy", "woy", "month"]),
        ("MS", ["month", "quarter"]),
        ("M", ["month", "quarter"]),
        ("2MS", ["month", "quarter"]),
        ("QS", ["month"]),
        ("B", ["month"]),
    ],
)
def test_calendar_feature_names_width_dtype_and_empty_input(freq, fields):
    idx = pd.date_range("2023-01-01", periods=3, freq=to_offset(freq))
    names = [f"{field}_{function}" for field in fields for function in ["sin", "cos"]]
    assert calendar_feature_names(freq) == names
    features = calendar_features(idx, freq)
    assert features.shape == (3, len(names)) and features.dtype == np.float32
    assert np.isfinite(features).all()
    empty = calendar_features(pd.DatetimeIndex([]), freq)
    assert empty.shape == (0, len(names)) and empty.dtype == np.float32
    np.testing.assert_array_equal(features, calendar_features(idx, to_offset(freq)))


def test_calendar_features_use_iso_week_and_float_sine_cosine_values():
    # 2021-01-01 is a Friday in ISO week 53 of 2020, not week 1 of 2021.
    idx = pd.DatetimeIndex(["2021-01-01 06:00"])
    angles = 2 * np.pi * np.array([6 / 24, 4 / 7, 1 / 31, 1 / 366, 53 / 53, 1 / 12])
    expected = np.column_stack([np.sin(angles), np.cos(angles)]).reshape(1, -1)
    np.testing.assert_allclose(calendar_features(idx, "H"), expected, atol=1e-7)
    np.testing.assert_allclose(calendar_features(idx, "D"), expected[:, 2:], atol=1e-7)
    np.testing.assert_allclose(calendar_features(idx, "W-SUN"), expected[:, -4:], atol=1e-7)


def test_calendar_features_include_leap_day_and_quarter():
    idx = pd.DatetimeIndex(["2020-12-31"])
    daily = calendar_features(idx, "D")
    np.testing.assert_allclose(daily[0, 4:6], [0, 1], atol=1e-7)  # day 366
    np.testing.assert_allclose(calendar_features(idx, "M"), [[0, 1, 0, 1]], atol=1e-7)
    np.testing.assert_allclose(calendar_features(idx, "QS"), [[0, 1]], atol=1e-7)
