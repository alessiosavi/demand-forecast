"""Inner-validation eligibility and isolation of the permitted training split."""

import numpy as np
import pandas as pd
import pytest

from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.frequency import grid_range
from tests.helpers.panels import synthetic_panel, weekly_panel


def test_rossmann_like_first_window_triggers_exception(panel29, origin29):
    plan = plan_inner_validation(panel29, origin29, 13)
    assert plan.exception and plan.train_upto == origin29 - panel29.freq and len(plan.fit_ids) == 3
    assert plan.val_origin == origin29 - 13 * panel29.freq
    assert plan.val_items.shape == (0,) and plan.val_items.dtype == np.int64
    assert plan.fit_ids == ["a", "b", "c"]


def test_rohlik_like_eight_weeks_h8_triggers_exception(panel8, origin8):
    assert plan_inner_validation(panel8, origin8, 8).exception


def test_constant_histories_are_not_validation_items(panel_constant_long, origin_cl):
    assert plan_inner_validation(panel_constant_long, origin_cl, 4).exception


def test_mixed_panel_keeps_only_eligible_items(panel_mixed, origin_mixed):
    plan = plan_inner_validation(panel_mixed, origin_mixed, 13)
    assert (
        not plan.exception
        and plan.val_items.tolist() == [0]
        and plan.train_upto == origin_mixed - 14 * panel_mixed.freq
    )
    assert plan.val_items.dtype == np.int64
    assert plan.val_origin == origin_mixed - 13 * panel_mixed.freq
    # Short series can fit preprocessing/training even when they cannot validate.
    assert plan.fit_ids == ["a", "b"]


def test_minimum_history_counts_observations_and_preserves_panel_order():
    panel = weekly_panel([30] * 3)
    for series, count, name in zip(panel.series, [12, 11, 13], ["z", "a", "m"]):
        series.series_id = name
        series.target[count:26] = np.nan
    origin = panel.timestamps(0, 31)[30]

    plan = plan_inner_validation(panel, origin, 4)

    assert not plan.exception
    assert plan.val_items.tolist() == [0, 2]
    assert plan.val_items.dtype == np.int64
    assert plan.fit_ids == ["z", "a", "m"]


def test_minimum_history_override_uses_exact_threshold(panel29, origin29):
    plan = plan_inner_validation(panel29, origin29, 13, min_train_history=16)
    assert not plan.exception and plan.val_items.tolist() == [0, 1, 2]
    assert plan_inner_validation(panel29, origin29, 13, min_train_history=17).exception


def test_seasonal_error_retains_missing_grid_positions_and_honors_override():
    panel = weekly_panel([30])
    panel.series[0].target[1:26:2] = np.nan
    origin = panel.timestamps(0, 31)[30]

    # Thirteen past observations, but no observed one-step seasonal pairs.
    assert plan_inner_validation(panel, origin, 4).exception
    plan = plan_inner_validation(panel, origin, 4, seasonal_period=2)
    assert not plan.exception and plan.val_items.tolist() == [0]
    assert panel.seasonal_period == 1

    panel.seasonal_period = 2
    assert not plan_inner_validation(panel, origin, 4).exception
    assert plan_inner_validation(panel, origin, 4, seasonal_period=1).exception


@pytest.mark.parametrize("past_kind", ["constant", "infinite", "undefined"])
def test_validation_values_cannot_make_past_seasonal_error_eligible(past_kind):
    panel = weekly_panel([30])
    if past_kind == "constant":
        panel.series[0].target[:26] = 6
    elif past_kind == "infinite":
        panel.series[0].target[0] = np.inf
    else:
        panel.seasonal_period = 26
    origin = panel.timestamps(0, 31)[30]

    plan = plan_inner_validation(panel, origin, 4)

    assert plan.exception and plan.val_items.size == 0
    assert plan.fit_ids == ["a"]


@pytest.mark.parametrize(
    "label_position,eligible", [(25, False), (26, True), (29, True), (30, False)]
)
def test_validation_block_includes_its_start_and_excludes_forecast_origin(label_position, eligible):
    panel = weekly_panel([31])
    panel.series[0].target[26:] = np.nan
    panel.series[0].target[label_position] = 0
    origin = panel.timestamps(0)[30]

    plan = plan_inner_validation(panel, origin, 4)

    assert plan.exception is not eligible
    assert plan.val_items.tolist() == ([0] if eligible else [])


@pytest.mark.parametrize("history_length", [0, 1, 26])
def test_targets_outside_historical_span_cannot_validate_or_fit(history_length):
    panel = weekly_panel([30])
    origin = panel.timestamps(0, 31)[30]
    panel.series[0].history_length = history_length
    panel.series[0].target[0] = np.nan

    plan = plan_inner_validation(panel, origin, 4)

    assert plan.exception
    assert plan.fit_ids == (["a"] if history_length == 26 else [])


def test_fit_ids_follow_inclusive_training_cutoff_in_both_modes():
    panel = weekly_panel([31] * 5)
    for series, name in zip(panel.series, ["z", "a", "label_only", "forecast_only", "missing"]):
        series.series_id = name
    for series in panel.series[1:]:
        series.target[:] = np.nan
    panel.series[1].target[25] = 0  # Exactly at the normal training cutoff.
    panel.series[2].target[26] = 0  # Exactly at the validation origin.
    panel.series[3].target[30] = 0  # Exactly at the forecast origin.
    origin = panel.timestamps(0)[30]

    plan = plan_inner_validation(panel, origin, 4)
    assert not plan.exception
    assert plan.train_upto == panel.timestamps(0)[25]
    assert plan.fit_ids == ["z", "a"]

    panel.series[0].target[:26] = 6  # Remove the only eligible validation item.
    plan = plan_inner_validation(panel, origin, 4)
    assert plan.exception
    assert plan.train_upto == panel.timestamps(0)[29]
    assert plan.fit_ids == ["z", "a", "label_only"]


def test_series_start_dates_define_their_own_cutoff_positions():
    panel = weekly_panel([30, 20, 20])
    panel.series[1].start = panel.timestamps(0)[20]
    panel.series[2].start = panel.timestamps(0, 31)[30]
    origin = panel.timestamps(0, 31)[30]

    plan = plan_inner_validation(panel, origin, 4)

    assert not plan.exception and plan.val_items.tolist() == [0]
    assert plan.fit_ids == ["a", "b"]


def test_fit_cutoff_is_inclusive_even_for_series_on_different_grid_phases():
    panel = synthetic_panel(3, 30, "2W-MON", seed=0)
    grid = panel.timestamps(0, 31)
    for series in panel.series[1:]:
        series.start += pd.offsets.Week()
        series.target[:] = np.nan
    panel.series[1].target[25] = 1  # Between train_upto and val_origin.
    panel.series[2].target[29] = 1  # Between origin - step and origin.

    plan = plan_inner_validation(panel, grid[30], 4)
    assert not plan.exception and plan.train_upto == grid[25]
    assert plan.fit_ids == ["0"]

    panel.series[0].target[:] = 6
    plan = plan_inner_validation(panel, grid[30], 4)
    assert plan.exception and plan.train_upto == grid[29]
    assert plan.fit_ids == ["0", "1"]


@pytest.mark.parametrize("lengths", [[], [0], [1], [30, 40]])
def test_empty_or_unobserved_panels_have_no_fit_ids(lengths):
    panel = weekly_panel(lengths)
    for series in panel.series:
        series.target[:] = np.nan
    origin = pd.Timestamp("2024-01-07")

    plan = plan_inner_validation(panel, origin, 4)

    assert plan.exception and plan.fit_ids == []
    assert plan.val_items.shape == (0,) and plan.val_items.dtype == np.int64
    assert plan.train_upto == pd.Timestamp("2023-12-31")


def test_singleton_can_fit_under_short_history_exception():
    panel = weekly_panel([1])
    origin = panel.timestamps(0, 2)[1]

    plan = plan_inner_validation(panel, origin, 4)

    assert plan.exception and plan.fit_ids == ["a"]
    assert plan.train_upto == panel.series[0].start


@pytest.mark.parametrize("freq", ["D", "h", "ME", "2W-MON"])
def test_cutoffs_follow_panel_frequency(freq):
    panel = synthetic_panel(1, 30, freq, seed=0)
    grid = panel.timestamps(0, 31)

    plan = plan_inner_validation(panel, grid[30], 4)

    assert not plan.exception and plan.val_items.tolist() == [0]
    assert plan.val_origin == grid[26]
    assert plan.train_upto == grid[25]


@pytest.mark.parametrize("start", ["2024-03-10", "2024-10-06"])
def test_daily_cutoffs_follow_local_grid_across_dst(start):
    panel = synthetic_panel(1, 24, "D", seed=0)
    panel.series[0].start = pd.Timestamp(start, tz="Europe/Rome")
    grid = grid_range(panel.series[0].start, 25, panel.freq)

    plan = plan_inner_validation(panel, grid[24], 4)

    assert not plan.exception
    assert plan.val_origin == grid[20]
    assert plan.train_upto == grid[19]
    panel.series[0].target[:] = 6
    assert plan_inner_validation(panel, grid[24], 4).train_upto == grid[23]
