from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from demand_forecast.data.scaling import Frame, sample_frame, seasonal_error

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def test_constant_history_is_ineligible_not_floored():
    frame = sample_frame(np.array([5.0, 5.0]), np.array([5.0, 5.0]), 1)
    assert frame == Frame(loc=5.0, scale=1.0, scale_raw=0.0, eligible=False, has_context=True)


def test_scale_uses_full_past_not_cropped_context():
    past = np.array([0.0, 10.0, 11.0, 12.0])
    frame = sample_frame(past[-2:], past, 1)
    assert frame.scale_raw == pytest.approx(4.0)
    assert frame.scale == pytest.approx(4.0)
    assert frame.loc == 11.5
    assert frame.eligible and frame.has_context


def test_missing_values_are_skipped_and_empty_context_flagged():
    assert seasonal_error(np.array([1.0, np.nan, 3.0, 4.0]), 1) == pytest.approx(1.0)
    assert np.isnan(seasonal_error(np.array([1.0]), 1))
    frame = sample_frame(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), 1)
    assert not frame.has_context
    assert frame.loc == 0.0 and frame.scale == 1.0
    assert np.isnan(frame.scale_raw) and not frame.eligible


@pytest.mark.parametrize(
    ("values", "m", "expected"),
    [
        ([1.0, np.nan, 3.0, 4.0], 2, 2.0),
        ([1.0, 10.0, 3.0, 14.0, 8.0], 2, 11.0 / 3.0),
        ([-5.0, 2.0, -9.0, 1.0], 1, 28.0 / 3.0),
    ],
)
def test_seasonal_error_keeps_lag_positions_and_uses_absolute_differences(values, m, expected):
    assert seasonal_error(np.array(values), m) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "m"),
    [([], 1), ([1.0], 1), ([1.0, 2.0], 2), ([1.0, 2.0], 3), ([1.0, np.nan, 3.0], 1)],
)
def test_no_observed_lag_pairs_give_nan_and_an_ineligible_frame(values, m):
    past = np.array(values)
    assert np.isnan(seasonal_error(past, m))
    frame = sample_frame(np.array([7.0]), past, m, min_scale=10.0)
    assert np.isnan(frame.scale_raw) and not frame.eligible
    assert frame.scale == 1.0 and frame.loc == 7.0 and frame.has_context


@pytest.mark.parametrize("context", [[], [np.nan, np.nan]])
def test_no_context_is_independent_of_positive_full_past_scale(context):
    frame = sample_frame(np.array(context), np.array([1.0, 3.0]), 1)
    assert frame == Frame(loc=0.0, scale=2.0, scale_raw=2.0, eligible=True, has_context=False)


def test_location_is_the_observed_context_median():
    context = np.array([np.nan, -8.0, 2.0, 4.0, 100.0])
    frame = sample_frame(context, context, 1)
    assert frame.loc == 3.0 and frame.has_context


@pytest.mark.parametrize("min_scale", [1e-6, 0.25])
def test_positive_scale_is_floored_without_changing_raw_scale_or_eligibility(min_scale):
    past = np.array([0.0, 1e-8])
    frame = sample_frame(past, past, 1, min_scale=min_scale)
    assert frame.scale_raw == pytest.approx(1e-8, rel=1e-12, abs=0.0)
    assert frame.scale == min_scale and frame.eligible


def test_default_floor_is_applied_to_a_positive_scale():
    past = np.array([0.0, 1e-8])
    assert sample_frame(past, past, 1).scale == 1e-6


def test_large_floor_does_not_change_ineligible_scale():
    past = np.array([5.0, 5.0])
    frame = sample_frame(past, past, 1, min_scale=10.0)
    assert frame.scale == 1.0 and frame.scale_raw == 0.0 and not frame.eligible


@pytest.mark.parametrize("past", [[1.0, np.inf, 4.0], [1.0, -np.inf, 4.0]])
def test_infinite_seasonal_error_is_ineligible(past):
    frame = sample_frame(np.array([4.0]), np.array(past), 1)
    assert np.isposinf(frame.scale_raw)
    assert frame.scale == 1.0 and not frame.eligible


def test_undefined_error_from_observed_infinities_is_not_discarded():
    past = np.array([np.inf, np.inf, 2.0, 4.0])
    frame = sample_frame(past[-2:], past, 1)
    assert np.isnan(frame.scale_raw)
    assert frame.scale == 1.0 and not frame.eligible


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint8])
def test_numeric_dtypes_use_absolute_differences_without_mutating_inputs(dtype):
    past = np.array([5, 2, 9, 1], dtype=dtype)
    before = past.copy()
    frame = sample_frame(past[-2:], past, 1)
    assert frame.scale_raw == pytest.approx(6.0)
    assert frame.loc == 5.0
    np.testing.assert_array_equal(past, before)


def test_frame_is_frozen():
    frame = sample_frame(np.array([1.0, 3.0]), np.array([1.0, 3.0]), 1)
    with pytest.raises(FrozenInstanceError):
        frame.scale = 2.0
