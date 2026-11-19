"""The conversion works on in-memory runner inputs without fev."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.bench.yardsticks.runner import SeriesInput, WindowInput
from demand_forecast.data.episode.schema import MAX_HORIZON, PATCH
from demand_forecast.data.frequency import grid_range, steps_between, to_offset, validate_on_grid


def series(index, first, *, length=3, freq="D", past=None, future=None, static=None):
    from demand_forecast.bench.v5_panels import time_axis

    dates = pd.date_range(first, periods=length + 2, freq=freq).to_numpy()
    return SeriesInput(
        index,
        (0,),
        np.arange(length, dtype=np.float64)[None, :],
        past or {},
        future or {},
        axis=time_axis(dates[:length], freq, future=dates[length:], horizon=2),
        static=static,
    )


def convert(*items, freq="D", names=("demand",), known=(), past=(), horizon=2, **kwargs):
    from demand_forecast.bench.v5_panels import window_panels

    return window_panels(
        WindowInput([f"sku-{i}" for i in range(len(items))], list(names), list(items), freq),
        name="example",
        horizon=horizon,
        known=known,
        past=past,
        **kwargs,
    )


def test_dated_ragged_placement_and_history_tails():
    first = series(
        1,
        "2024-01-03",
        past={"plan": np.array([10, 20, 30]), "sales": np.array([4, 5, 6])},
        future={"plan": np.array([None, 50], dtype=object)},
    )
    second = series(
        0,
        "2024-01-01",
        length=4,
        past={"plan": np.array([8, 9, 10, 11]), "sales": np.array([0, 1, 2, 3])},
        future={"plan": np.array([12, 13])},
    )
    result = convert(first, second, known=["plan"], past=["sales"])
    (part,) = result.parts
    assert not part.alone
    assert result.reasons == {}
    assert part.positions == (0, 1)
    assert part.origins == (5, 4)
    assert part.panel.epoch == pd.Timestamp("2024-01-01")
    assert part.panel.aligned
    assert part.panel.freq == "D"
    assert part.panel.roles == {"plan": "known", "sales": "past"}
    assert [item.item_id for item in part.panel.items] == ["sku-1", "sku-0"]
    assert [item.start for item in part.panel.items] == [2, 0]
    item = part.panel.items[0]
    np.testing.assert_array_equal(item.targets["demand"], [0, 1, 2, np.nan, np.nan])
    np.testing.assert_array_equal(item.dynamic["plan"], [10, 20, 30, np.nan, 50])
    np.testing.assert_array_equal(item.dynamic["sales"], [4, 5, 6, np.nan, np.nan])
    part.panel.validate()


def test_normalizes_declared_and_observed_cells():
    from demand_forecast.bench.v5_panels import window_panels

    item = series(
        0,
        "2024-01-01",
        past={
            "null_number": np.array([None, pd.NA, np.nan], dtype=object),
            "number": np.array([np.int64(3), None, np.float32(2.5)], dtype=object),
            "text": np.array([np.str_("a"), np.nan, "b"], dtype=object),
            "flag": np.array([np.bool_(True), None, np.bool_(False)], dtype=object),
            "inferred_flag": np.array([np.bool_(False), None, np.bool_(True)], dtype=object),
            "weather": np.array([0, np.nan, 1], dtype=float),
        },
        future={"null_number": np.array([8, None], dtype=object)},
        static={
            "groups": np.array([np.str_("a"), None, np.int64(2)], dtype=object),
            "class": np.int64(1),
            "classes": np.array([0, -1, 1]),
            "flag": np.bool_(True),
            "number": np.int64(9),
        },
    )
    window = WindowInput(
        ["sku"],
        ["demand"],
        [item],
        "D",
        declared={
            "null_number": "number",
            "text": "string",
            "flag": "boolean",
            "weather": "class_label",
            "class": "class_label",
            "classes": "class_label",
        },
        class_labels={"weather": ("sun", "rain"), "class": ("x", "y"), "classes": ("x", "y")},
    )
    (part,) = window_panels(
        window,
        name="mixed",
        horizon=2,
        known=["null_number"],
        past=["number", "text", "flag", "inferred_flag", "weather"],
    ).parts
    (converted,) = part.panel.items
    for name, expected in {
        "null_number": [np.nan, np.nan, np.nan, 8, np.nan],
        "number": [3, np.nan, 2.5, np.nan, np.nan],
        "flag": [1, np.nan, 0, np.nan, np.nan],
        "inferred_flag": [0, np.nan, 1, np.nan, np.nan],
    }.items():
        assert converted.dynamic[name].dtype == np.float64
        np.testing.assert_array_equal(converted.dynamic[name], expected)
    for name, expected in {
        "text": ["a", None, "b", None, None],
        "weather": ["sun", None, "rain", None, None],
    }.items():
        assert converted.dynamic[name].dtype == object
        assert converted.dynamic[name].tolist() == expected
    assert type(converted.dynamic["text"][0]) is str
    assert converted.static == {
        "groups": ("a", None, 2),
        "class": "y",
        "classes": ("x", None, "y"),
        "flag": True,
        "number": 9,
    }
    assert type(converted.static["groups"][0]) is str
    assert type(converted.static["groups"][2]) is int
    assert type(converted.static["flag"]) is bool
    assert type(converted.static["number"]) is int


@pytest.mark.parametrize("values", [np.array([[1], [2], [3]]), np.array([1, [2], 3], dtype=object)])
def test_nested_dynamic_cell_is_refused(values):
    item = series(0, "2024-01-01", past={"nested": values})
    with pytest.raises(ValueError, match="nested"):
        convert(item, past=["nested"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.int64(1), "rain"),
        (np.float64(1.0), "rain"),
        (np.float32(-1.0), None),
        (np.str_("rain"), "rain"),
        (None, None),
    ],
)
def test_class_label_accepts_codes_names_and_null(value, expected):
    from demand_forecast.bench.v5_panels import _cell

    assert _cell(value, "class_label", ("sun", "rain"), "weather") == expected


@pytest.mark.parametrize("value", ["snow", "", 2, -2, 0.5, True, 10**40])
def test_class_label_error_identifies_column_and_value(value):
    from demand_forecast.bench.v5_panels import _cell

    with pytest.raises(ValueError) as error:
        _cell(value, "class_label", ("sun", "rain"), "weather")
    assert "weather" in str(error.value)
    assert repr(value) in str(error.value)


@pytest.mark.parametrize(
    ("history_days", "freq", "future_days", "horizon", "reason"),
    [
        ([1, 3], "D", None, None, "history_gap"),
        ([1, 1, 3], "D", None, None, "history_gap"),
        ([1, 2], "D", [4, 5], 2, "future_off_grid"),
        ([1, 2], "D", [3], 2, "future_length"),
        ([1], "D", [2, 3], 2, None),
        ([1], "D", [3, 4], 2, "future_off_grid"),
        ([1, 2], "D", [3, 4], None, None),
        ([1, 2], "D", None, 2, None),
        ([1, 2], "zebra", None, None, "invalid_frequency"),
        ([1, 2], None, None, None, "missing_frequency"),
        ([1, 2], "0D", None, None, "non_positive_frequency"),
        ([1, 2], "-1D", None, None, "non_positive_frequency"),
        ([1, 2], "W-SUN", None, None, "history_off_grid"),
        ([], "D", None, None, "empty_history"),
    ],
)
def test_time_axis(history_days, freq, future_days, horizon, reason):
    from demand_forecast.bench.v5_panels import time_axis

    def dates(days):
        return np.array([f"2024-01-{day:02}" for day in days], dtype="datetime64[ns]")

    axis = time_axis(
        dates(history_days),
        freq,
        future=None if future_days is None else dates(future_days),
        horizon=horizon,
    )
    assert axis.first == (pd.Timestamp("2024-01-01") if history_days else None)
    assert axis.on_grid is (reason is None)
    assert axis.reason == reason


def test_time_axis_rejects_two_dimensional_history():
    from demand_forecast.bench.v5_panels import TimeAxis, time_axis

    history = pd.date_range("2024-01-01", periods=4).to_numpy().reshape(2, 2)
    assert time_axis(history, "D") == TimeAxis(
        None, False, "invalid_history_timestamps", missing_timestamps=True
    )


def test_time_axis_rejects_two_dimensional_future():
    from demand_forecast.bench.v5_panels import TimeAxis, time_axis

    dates = pd.date_range("2024-01-01", periods=6).to_numpy()
    assert time_axis(dates[:2], "D", future=dates[2:].reshape(2, 2), horizon=2) == TimeAxis(
        pd.Timestamp("2024-01-01"), False, "invalid_future_timestamps", missing_timestamps=True
    )


def _old_time_axis(history, freq, *, future=None, horizon=None):
    """First-round per-timestamp proof, retained as a reference."""
    try:
        dates = pd.DatetimeIndex(history)
    except (TypeError, ValueError, OverflowError):
        return None, False, "invalid_history_timestamps"
    if not len(dates):
        return None, False, "empty_history"
    first = None if pd.isna(dates[0]) else dates[0]
    if freq is None:
        return first, False, "missing_frequency"
    try:
        offset = to_offset(freq)
    except (TypeError, ValueError):
        return first, False, "invalid_frequency"
    if offset.n <= 0:
        return first, False, "non_positive_frequency"
    try:
        validate_on_grid(dates, offset, "history")
        span = steps_between(dates[0], dates[-1], offset)
        expected = grid_range(dates[0], len(dates), offset)
    except (ValueError, OverflowError):
        return first, False, "history_off_grid"
    if span != len(dates) - 1 or not (dates == expected).all():
        return first, False, "history_gap"

    if future is not None:
        try:
            future_dates = pd.DatetimeIndex(future)
        except (TypeError, ValueError, OverflowError):
            return first, False, "invalid_future_timestamps"
        if horizon is not None and len(future_dates) != horizon:
            return first, False, "future_length"
        try:
            expected = grid_range(dates[-1], len(future_dates) + 1, offset)[1:]
        except (ValueError, OverflowError):
            return first, False, "future_off_grid"
        if not (future_dates == expected).all():
            return first, False, "future_off_grid"
    return first, True, None


@pytest.mark.parametrize("tz", [None, "America/New_York"])
@pytest.mark.parametrize("length", [1, 2, 17, 400])
@pytest.mark.parametrize("freq", ["D", "h", "15min", "W-MON", "ME", "MS", "QE", "2MS", "3ME", "B"])
def test_vectorized_time_axis_matches_per_timestamp_proof(freq, length, tz):
    from demand_forecast.bench import v5_panels

    dates = pd.date_range("2000-03-31", periods=length + 3, freq=freq, tz=tz)
    history, future = dates[:length], dates[length : length + 2]
    phase = pd.Timedelta(seconds=13)
    cases = {
        "consecutive": (history, future, 2),
        "no future": (history, None, 2),
        "unspecified horizon": (history, future, None),
        "empty history": (history[:0], future, 2),
        "unparseable history": (["not a date"], future, 2),
        "NaT first": (history.where(np.arange(length) != 0), future, 2),
        "NaT last": (history.where(np.arange(length) != length - 1), future, 2),
        "wrong local time or phase": (history[:-1].append(history[-1:] + phase), future, 2),
        "wrong anchor": (history[:-1].append(history[-1:] + pd.Timedelta(days=1)), future, 2),
        "unparseable future": (history, ["not a date"], 2),
        "short future": (history, future[:1], 2),
        "late future": (history, dates[-2:], 2),
        "duplicate future": (history, future[:1].repeat(2), 2),
        "NaT future": (history, future.where([True, False]), 2),
    }
    if length > 1:
        cases.update(
            gap=(history[:-1].append(dates[length : length + 1]), future, 2),
            duplicate=(history[:-1].append(history[-2:-1]), future, 2),
            reversed=(history[::-1], future, 2),
        )
        off_step = {
            "2MS": pd.offsets.MonthBegin(),
            "3ME": pd.offsets.MonthEnd(),
            "B": pd.offsets.Week(weekday=5),
        }.get(freq)
        if off_step is not None:
            cases["wrong calendar step"] = (history[:-1].append(history[-1:] + off_step), future, 2)
    parsed = (to_offset(freq), None)
    for case, (past, next_dates, horizon) in cases.items():
        expected = _old_time_axis(past, freq, future=next_dates, horizon=horizon)
        axis = v5_panels._prove_time_axis(past, parsed, future=next_dates, horizon=horizon)
        assert (axis.first, axis.on_grid, axis.reason) == expected, case


@pytest.mark.parametrize(("freq", "tz"), [("24h", "America/New_York"), ("25h", None)])
def test_history_at_midnight_with_ticks_of_at_least_one_day(freq, tz):
    from demand_forecast.bench.v5_panels import time_axis

    dates = pd.date_range("2024-03-09", periods=6, freq=freq, tz=tz)
    expected = _old_time_axis(dates[:4], freq, future=dates[4:], horizon=2)
    assert expected[2] == "history_off_grid"
    axis = time_axis(dates[:4], freq, future=dates[4:], horizon=2)
    assert (axis.first, axis.on_grid, axis.reason) == expected


def test_hourly_phase_is_checked_on_the_common_grid():
    half_hour = series(0, "2024-01-01 00:30", freq="h", static={"site": "b"})
    whole_hour = series(1, "2024-01-01 00:00", freq="h", static={"site": "a"})
    assert whole_hour.axis.on_grid and half_hour.axis.on_grid
    result = convert(half_hour, whole_hour, freq="h")
    dated, alone = result.parts
    assert dated.positions == (1,)
    assert dated.panel.items[0].static == {"site": "a"}
    assert alone.positions == (0,)
    assert alone.panel.freq == "h"
    assert alone.panel.items[0].static == {}
    assert result.reasons == {0: "off_common_grid"}


def test_one_off_grid_item_leaves_nine_dated_peers():
    from demand_forecast.bench.v5_panels import TimeAxis, describe

    items = [series(i, "2024-01-01", length=PATCH, static={"site": "same"}) for i in range(10)]
    items[4] = replace(items[4], axis=TimeAxis(pd.Timestamp("2024-01-01"), False, "history_gap"))
    result = convert(*items)
    dated, alone = result.parts
    assert dated.positions == (0, 1, 2, 3, 5, 6, 7, 8, 9)
    assert [item.static for item in dated.panel.items] == [{"site": "same"}] * 9
    assert dated.origins == (PATCH,) * 9
    assert alone.positions == (4,)
    assert alone.origins == (PATCH,)
    assert alone.alone and not alone.panel.aligned
    assert alone.panel.epoch is None
    assert alone.panel.items[0].start == 0
    assert alone.panel.items[0].static == {}
    assert result.reasons == {4: "history_gap"}
    assert describe([result]) == {
        "v5_items_alone": 1,
        "v5_items_off_grid": 1,
        "v5_items_without_time_axis": 0,
        "v5_short_items": 0,
        "v5_max_distinct_origins": 1,
        "v5_windows_over_origin_budget": 0,
        "v5_mixed_kind_columns": 0,
    }


def test_origin_budget_counts_the_dated_origins_before_fallback():
    from demand_forecast.bench.v5_panels import describe

    items = [series(i, "2024-01-01", length=i + 1, static={"site": "same"}) for i in range(17)]
    within_budget = convert(*items[:-1])
    assert within_budget.parts[0].origins == tuple(range(1, 17))
    assert len(within_budget.parts) == 1

    over_budget = convert(*items)
    assert [part.positions for part in over_budget.parts] == [(i,) for i in range(17)]
    assert [part.origins for part in over_budget.parts] == [(i + 1,) for i in range(17)]
    assert all(part.alone and part.panel.items[0].static == {} for part in over_budget.parts)
    assert over_budget.reasons == dict.fromkeys(range(17), "over_origin_budget")
    assert describe([within_budget, over_budget]) == {
        "v5_items_alone": 17,
        "v5_items_off_grid": 0,
        "v5_items_without_time_axis": 0,
        "v5_short_items": 30,
        "v5_max_distinct_origins": 17,
        "v5_windows_over_origin_budget": 1,
        "v5_mixed_kind_columns": 0,
    }


@pytest.mark.parametrize("freq", [None, "zebra", "0D"])
def test_unusable_frequency_keeps_one_step_unpadded(freq):
    result = convert(series(0, "2024-01-01", length=1), freq=freq)
    (part,) = result.parts
    assert part.alone
    assert part.panel.freq is None
    assert part.panel.epoch is None
    assert part.origins == (1,)
    np.testing.assert_array_equal(part.panel.items[0].targets["demand"], [0, np.nan, np.nan])


def test_describe_distinguishes_missing_axes_from_grid_failures():
    from demand_forecast.bench.v5_panels import describe, time_axis

    item = series(0, "2024-01-01", length=2)
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    no_timestamps = replace(item, axis=None)
    nat = replace(item, axis=time_axis(dates[:2].where([True, False]), "D"))
    nat_future = replace(
        item, axis=time_axis(dates[:2], "D", future=dates[2:].where([True, False]), horizon=2)
    )
    gap = replace(item, axis=time_axis(dates[[0, 2]], "D"))
    counts = describe(
        [
            convert(no_timestamps, nat, nat_future, gap),
            convert(item, freq=None),
            convert(item, freq="unparseable"),
            convert(item, freq="0D"),
            convert(item, max_origins=0),
        ]
    )
    assert counts["v5_items_alone"] == 8
    assert counts["v5_items_off_grid"] == 7
    assert counts["v5_items_without_time_axis"] == 5


@pytest.mark.parametrize(
    ("removal", "dynamic", "static"),
    [
        ("features", set(), {"repeated", "price", "tags"}),
        ("known", {"category", "number"}, {"repeated", "price", "tags"}),
        ("past", {"plan"}, {"repeated", "price", "tags"}),
        ("dynamic_categories", {"plan", "number"}, {"repeated", "price", "tags"}),
        ("statics", {"plan", "category", "number"}, set()),
        ("static_categories", {"plan", "category", "number"}, {"price"}),
        ("static_numbers", {"plan", "category", "number"}, {"repeated", "tags"}),
    ],
)
def test_remove_exactly_the_requested_inputs(removal, dynamic, static):
    from demand_forecast.bench.v5_panels import REMOVABLE, describe, window_panels

    assert removal in REMOVABLE
    items = [
        series(
            i,
            "2024-01-01",
            past={
                "plan": np.array([1, 2, 3]),
                "category": np.array(["red", "green", "blue"]) if i == 0 else np.arange(3),
                "number": np.array([3.5, 4.5, 1.5]),
            },
            future={"plan": np.array([4, 5])},
            static={"repeated": np.int64(4), "price": i + 0.5, "tags": ["a", "b"]},
        )
        for i in range(3)
    ]
    result = window_panels(
        WindowInput(["a", "b", "c"], ["demand"], items, "D", {"repeated": "number"}),
        name="example",
        horizon=2,
        known=["plan"],
        past=["category", "number"],
        without=frozenset({removal}),
    )
    (part,) = result.parts
    assert set(part.panel.roles) == dynamic
    for item in part.panel.items:
        assert set(item.dynamic) == dynamic
        assert set(item.static) == static
    assert describe([result])["v5_mixed_kind_columns"] == 1
    part.panel.validate()


def test_target_order_follows_the_item_indices():
    item = replace(
        series(0, "2024-01-01"),
        target_indices=(2, 0),
        target=np.array([[10, 11, 12], [4, 5, 6]], dtype=np.float64),
    )
    (part,) = convert(item, names=("first", "unused", "third")).parts
    targets = part.panel.items[0].targets
    assert list(targets) == ["third", "first"]
    np.testing.assert_array_equal(targets["third"], [10, 11, 12, np.nan, np.nan])
    np.testing.assert_array_equal(targets["first"], [4, 5, 6, np.nan, np.nan])


def test_refuses_invalid_panel_inputs():
    item = series(0, "2024-01-01", past={"undeclared_role": np.arange(3)})
    with pytest.raises(ValueError, match="horizon"):
        convert(item, horizon=MAX_HORIZON + 1)
    with pytest.raises(ValueError, match="duplicate target"):
        convert(item, names=("duplicate", "duplicate"))
    with pytest.raises(ValueError, match="no targets"):
        convert(replace(item, target_indices=(), target=np.empty((0, 3))))
    with pytest.raises(ValueError, match="undeclared_role"):
        convert(item)


def test_unknown_removals_are_named():
    item = series(0, "2024-01-01")
    with pytest.raises(ValueError, match=r"'dynamic_category'.*'imaginary'"):
        convert(item, without=frozenset({"known", "imaginary", "dynamic_category"}))
