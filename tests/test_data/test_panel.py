from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.frequency import GridError, seasonal_period
from demand_forecast.data.panel import Panel, Roles, Series, build_series

W_SUN = pd.tseries.frequencies.to_offset("W-SUN")
ROLES = Roles(
    target="y",
    known_dynamic=("promo",),
    past_dynamic=(),
    static_categorical=(),
    static_numeric=(),
)


def test_absent_rows_follow_missing_target_and_covariates_are_nan():
    frame = pd.DataFrame(
        {"y": [1.0, 3.0], "promo": [0.0, 1.0]},
        index=pd.DatetimeIndex(["2023-01-08", "2023-01-22"]),
    )
    s = build_series("a", frame, freq=W_SUN, roles=ROLES, missing_target="zero", statics={})
    assert s.target.tolist() == [1.0, 0.0, 3.0]
    assert np.isnan(s.dynamic["promo"][1]) and s.length == 3
    assert np.isnan(
        build_series("a", frame, freq=W_SUN, roles=ROLES, missing_target="nan", statics={}).target[
            1
        ]
    )


def test_history_end_preserves_trailing_missing_and_future_known_rows():
    idx = pd.date_range("2023-01-01", periods=6, freq=W_SUN)
    frame = pd.DataFrame(
        {"y": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan], "promo": [0, 1, 1, 0, 1, 1]},
        index=idx,
    )
    s = build_series(
        "a", frame, freq=W_SUN, roles=ROLES, missing_target="zero", statics={}, history_end=idx[3]
    )
    assert s.length == 4 and np.isnan(s.target[2:]).all()
    assert s.dynamic["promo"].tolist() == [0, 1, 1, 0, 1, 1]
    assert (
        build_series("a", frame, freq=W_SUN, roles=ROLES, missing_target="zero", statics={}).length
        == 2
    )


@pytest.mark.parametrize("missing_target,absent_value", [("zero", 0.0), ("nan", np.nan)])
def test_past_covariates_extend_history_but_known_covariates_do_not(missing_target, absent_value):
    roles = replace(ROLES, past_dynamic=("oil", "price"))
    idx = pd.date_range("2023-01-01", periods=7, freq=W_SUN)
    frame = pd.DataFrame(
        {
            "y": [np.nan, 2.0, np.nan, np.nan, np.nan],
            "oil": [np.nan, 3.0, 4.0, np.nan, np.nan],
            "price": [np.nan, np.nan, np.nan, 0.0, np.nan],
            "promo": [0, 1, 0, 1, 1],
        },
        index=idx[[0, 2, 3, 4, 6]],
    )
    s = build_series("a", frame, freq=W_SUN, roles=roles, missing_target=missing_target, statics={})
    assert s.start == idx[0] and s.length == s.history_length == 5
    np.testing.assert_equal(s.target, [np.nan, absent_value, 2.0, np.nan, np.nan])
    np.testing.assert_equal(s.dynamic["oil"], [np.nan, np.nan, 3.0, 4.0, np.nan])
    np.testing.assert_equal(s.dynamic["price"], [np.nan, np.nan, np.nan, np.nan, 0.0])
    np.testing.assert_equal(s.dynamic["promo"], [0, np.nan, 1, 0, 1, np.nan, 1])


def test_explicit_history_end_excludes_future_targets_and_past_covariates():
    roles = replace(ROLES, past_dynamic=("oil",))
    idx = pd.date_range("2023-01-01", periods=6, freq=W_SUN)
    frame = pd.DataFrame(
        {"y": [1.0, np.nan, 999.0], "oil": [2.0, 3.0, 999.0], "promo": [0, 1, 1]},
        index=idx[[0, 2, 5]],
    )
    s = build_series(
        "a", frame, freq=W_SUN, roles=roles, missing_target="zero", statics={}, history_end=idx[3]
    )
    assert s.length == 4
    np.testing.assert_equal(s.target, [1.0, 0.0, np.nan, 0.0])
    np.testing.assert_equal(s.dynamic["oil"], [2.0, np.nan, 3.0, np.nan])
    np.testing.assert_equal(s.dynamic["promo"], [0, np.nan, 1, np.nan, np.nan, 1])


@pytest.mark.parametrize("missing_target,absent_value", [("zero", 0.0), ("nan", np.nan)])
def test_history_end_beyond_frame_pads_all_arrays(missing_target, absent_value):
    idx = pd.date_range("2023-01-01", periods=4, freq=W_SUN)
    roles = replace(ROLES, past_dynamic=("oil",))
    frame = pd.DataFrame({"y": [np.nan], "promo": [1], "oil": [2]}, index=idx[:1])
    s = build_series(
        "a",
        frame,
        freq=W_SUN,
        roles=roles,
        missing_target=missing_target,
        statics={},
        history_end=idx[-1],
    )
    assert s.length == 4
    np.testing.assert_equal(s.target, [np.nan, absent_value, absent_value, absent_value])
    np.testing.assert_equal(s.dynamic["promo"], [1, np.nan, np.nan, np.nan])
    np.testing.assert_equal(s.dynamic["oil"], [2, np.nan, np.nan, np.nan])


def test_all_missing_history_keeps_start_and_known_future_grid():
    idx = pd.date_range("2023-01-01", periods=3, freq=W_SUN)
    roles = replace(ROLES, past_dynamic=("oil",))
    frame = pd.DataFrame(
        {"y": [np.nan, np.nan], "oil": [np.nan, np.nan], "promo": [1, 0]}, index=idx[[0, 2]]
    )
    s = build_series("a", frame, freq=W_SUN, roles=roles, missing_target="zero", statics={})
    assert s.start == idx[0] and s.history_length == 0
    assert s.target.size == s.dynamic["oil"].size == 0
    np.testing.assert_equal(s.dynamic["promo"], [1, np.nan, 0])


@pytest.mark.parametrize(
    "freq,start,tz",
    [
        (pd.offsets.MonthEnd(), "2024-01-31", None),
        (pd.offsets.Week(2, weekday=2), "2023-01-04", None),
        (pd.offsets.Day(), "2023-03-25", "Europe/Rome"),
        (pd.offsets.Hour(), "2023-10-29 00:30", "Europe/Rome"),
    ],
)
def test_placement_and_panel_timestamps_follow_calendar_grid(freq, start, tz):
    idx = pd.date_range(start, periods=5, freq=freq, tz=tz)
    frame = pd.DataFrame({"y": [1, 3, np.nan], "promo": [0, 1, 2]}, index=idx[[0, 2, 4]])
    s = build_series("a", frame, freq=freq, roles=ROLES, missing_target="zero", statics={})
    assert s.length == 3 and s.start == idx[0]
    np.testing.assert_equal(s.target, [1, 0, 3])
    np.testing.assert_equal(s.dynamic["promo"], [0, np.nan, 1, np.nan, 2])
    panel = Panel([s], ROLES, freq, seasonal_period(freq), "real")
    pd.testing.assert_index_equal(panel.timestamps(0), idx[:3])
    pd.testing.assert_index_equal(panel.timestamps(0, length=5), idx)


@pytest.mark.parametrize(
    "freq,timestamps,bad_index",
    [
        (W_SUN, ["2023-01-02"], 0),
        (W_SUN, ["2023-01-01", "2023-01-08 13:00"], 1),
        (pd.offsets.Week(2, weekday=6), ["2023-01-01", "2023-01-08"], 1),
        (W_SUN, ["2023-01-01", None], 1),
    ],
)
def test_off_grid_frame_rows_propagate_grid_error_even_in_future(freq, timestamps, bad_index):
    idx = pd.DatetimeIndex([pd.Timestamp(value) for value in timestamps])
    frame = pd.DataFrame({"y": [np.nan] * len(idx), "promo": [1] * len(idx)}, index=idx)
    with pytest.raises(GridError) as exc:
        build_series("a", frame, freq=freq, roles=ROLES, missing_target="zero", statics={})
    assert exc.value.series_id == "a"
    if pd.isna(idx[bad_index]):
        assert pd.isna(exc.value.timestamp)
    else:
        assert exc.value.timestamp == idx[bad_index]


@pytest.mark.parametrize("history_end", ["2023-01-02", "2023-01-08 13:00"])
def test_history_end_must_be_on_grid(history_end):
    idx = pd.DatetimeIndex(["2023-01-01"])
    frame = pd.DataFrame({"y": [1.0], "promo": [0]}, index=idx)
    with pytest.raises(GridError) as exc:
        build_series(
            "a",
            frame,
            freq=W_SUN,
            roles=ROLES,
            missing_target="zero",
            statics={},
            history_end=pd.Timestamp(history_end),
        )
    assert exc.value.series_id == "a" and exc.value.timestamp == pd.Timestamp(history_end)


def test_nullable_numeric_columns_preserve_explicit_missing_values():
    idx = pd.date_range("2023-01-01", periods=3, freq=W_SUN)
    frame = pd.DataFrame(
        {
            "y": pd.array([1, None, 3], dtype="Float64"),
            "promo": pd.array([0, None, 1], dtype="Int64"),
        },
        index=idx,
    )
    s = build_series("a", frame, freq=W_SUN, roles=ROLES, missing_target="zero", statics={})
    np.testing.assert_equal(s.target, [1, np.nan, 3])
    np.testing.assert_equal(s.dynamic["promo"], [0, np.nan, 1])


def test_series_without_covariates_and_inputs_are_not_mutated():
    roles = Roles("sales", (), (), ("cat",), ("size",))
    idx = pd.DatetimeIndex(["2023-01-15", "2023-01-01"])
    frame = pd.DataFrame({"sales": [3.0, -1.0], "cat": ["B", "A"]}, index=idx)
    original = frame.copy(deep=True)
    statics = {"cat": "A", "size": 2.0}
    s = build_series("a", frame, freq=W_SUN, roles=roles, missing_target="zero", statics=statics)
    assert isinstance(s, Series) and s.series_id == "a" and s.start == idx[1]
    assert s.target.tolist() == [-1, 0, 3] and s.dynamic == {} and s.statics == statics
    s.target[0] = 10
    s.statics["cat"] = "Z"
    pd.testing.assert_frame_equal(frame, original)
    assert statics == {"cat": "A", "size": 2.0}


def test_panel_lookup_lengths_and_explicit_metadata():
    idx = pd.date_range("2023-01-01", periods=3, freq=W_SUN)
    frame = pd.DataFrame({"y": [1, 2, 3], "promo": [0, 1, 0]}, index=idx)
    a = build_series("a", frame.iloc[:1], freq=W_SUN, roles=ROLES, missing_target="nan", statics={})
    b = build_series("b", frame, freq=W_SUN, roles=ROLES, missing_target="nan", statics={})
    panel = Panel([b, a], ROLES, W_SUN, 52, "nonnegative")
    assert len(panel) == 2 and panel.index_of("b") == 0 and panel.index_of("a") == 1
    assert panel.roles == ROLES and panel.freq == W_SUN
    assert panel.seasonal_period == 52 and panel.target_support == "nonnegative"
    assert a.length == a.history_length == 1 and b.length == b.history_length == 3
    pd.testing.assert_index_equal(panel.timestamps(1), idx[:1])
    pd.testing.assert_index_equal(panel.timestamps(1, length=0), idx[:1])
    pd.testing.assert_index_equal(panel.timestamps(0, length=1), idx[:1])
    with pytest.raises(KeyError, match="absent"):
        panel.index_of("absent")
    assert len(Panel([], ROLES, W_SUN, 1, "real")) == 0


def test_roles_are_frozen_and_target_support_is_required():
    with pytest.raises(FrozenInstanceError):
        ROLES.target = "other"
    with pytest.raises(TypeError, match="target_support"):
        Panel([], ROLES, W_SUN, 1)


@pytest.mark.parametrize("target_support", ["", "positive", None])
def test_panel_rejects_unknown_target_support(target_support):
    with pytest.raises(ValueError, match="target_support"):
        Panel([], ROLES, W_SUN, 1, target_support)


def test_build_series_rejects_empty_duplicate_or_invalid_span():
    idx = pd.DatetimeIndex(["2023-01-01"])
    frame = pd.DataFrame({"y": [1.0], "promo": [0]}, index=idx)
    kwargs = {"freq": W_SUN, "roles": ROLES, "missing_target": "zero", "statics": {}}
    with pytest.raises(ValueError, match="empty"):
        build_series("a", frame.iloc[:0], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        build_series("a", pd.concat([frame, frame]), **kwargs)
    with pytest.raises(ValueError, match="history_end"):
        build_series("a", frame, history_end=pd.Timestamp("2022-12-25"), **kwargs)
    with pytest.raises(ValueError, match="missing_target"):
        build_series("a", frame, **{**kwargs, "missing_target": "typo"})
