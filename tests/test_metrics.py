import numpy as np
import pytest

from demand_forecast.metrics import (
    interval_coverage,
    mase,
    quantile_loss,
    seasonal_error_per_item,
    sql,
    summarize,
    wape,
    wql,
)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def test_sql_hand_value_excludes_zero_scale_items_and_missing_labels():
    y_past = [np.array([0.0, 10.0]), np.array([5.0, 5.0]), np.array([0.0, 2.0])]
    y_true = np.array([[12.0, np.nan], [6.0, 6.0], [np.nan, np.nan]])
    q = np.array([[[11.0], [11.0]], [[5.0], [5.0]], [[1.0], [1.0]]])
    assert sql(y_true, q, y_past, 1, [0.5]) == pytest.approx(2 * 0.5 * 1.0 / 10.0)
    assert mase(y_true, q[..., 0], y_past, 1) == pytest.approx(0.1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint8])
def test_quantile_loss_is_doubled_float64_pinball_without_mutation(dtype):
    labels = np.array([[4, 4]], dtype=dtype)
    predictions = np.array([[[2, 4, 6], [6, 2, 4]]], dtype=dtype)
    before = predictions.copy()
    losses = quantile_loss(labels, predictions, [0.1, 0.5, 0.9])
    assert losses.dtype == np.float64
    np.testing.assert_allclose(losses, [[[0.4, 0.0, 0.4], [3.6, 2.0, 0.0]]])
    np.testing.assert_array_equal(predictions, before)


def test_quantile_loss_preserves_missing_labels_at_every_knot():
    losses = quantile_loss(np.array([[np.nan, 2.0]]), np.zeros((1, 2, 2)), [0.1, 0.9])
    np.testing.assert_allclose(losses, [[[np.nan, np.nan], [0.4, 3.6]]])


def test_seasonal_errors_use_whole_ragged_histories_and_keep_missing_positions():
    histories = [
        np.array([0.0, 10.0, 11.0, 12.0]),
        np.array([1.0, np.nan, 3.0, 4.0]),
        np.array([5.0, 5.0]),
        np.array([1.0]),
        np.array([]),
        np.array([np.nan, np.nan]),
    ]
    errors = seasonal_error_per_item(histories, 1)
    assert errors.dtype == np.float64
    np.testing.assert_allclose(errors, [4.0, 1.0, 0.0, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(
        seasonal_error_per_item(histories, 2), [6.5, 2.0, np.nan, np.nan, np.nan, np.nan]
    )


def test_scaled_metrics_pool_observed_cells_instead_of_averaging_item_means():
    labels = np.array([[2.0, np.nan], [10.0, 10.0]])
    predictions = np.zeros_like(labels)
    histories = [np.array([0.0, 2.0]), np.array([0.0, 5.0])]
    assert mase(labels, predictions, histories, 1) == pytest.approx(5 / 3)
    assert sql(labels, predictions[..., None], histories, 1, [0.5]) == pytest.approx(5 / 3)


def test_scaled_metrics_exclude_undefined_histories_without_flooring_small_scales():
    histories = [np.array([1.0]), np.array([np.nan, np.nan]), np.array([0.0, 1e-9])]
    labels = np.array([[100.0], [100.0], [1e-6]])
    predictions = np.zeros_like(labels)
    assert mase(labels, predictions, histories, 1) == pytest.approx(1000.0)
    assert sql(labels, predictions[..., None], histories, 1, [0.5]) == pytest.approx(1000.0)


def test_sql_averages_per_knot_and_discards_nonfinite_scaled_errors_like_fev():
    labels = np.array([[2.0, 4.0]])
    predictions = np.array([[[np.inf, 0.0], [0.0, 0.0]]])
    histories = [np.array([0.0, 2.0])]
    # q=.25 has one finite scaled error (1); q=.75 has two (1.5, 3).
    assert sql(labels, predictions, histories, 1, [0.25, 0.75]) == pytest.approx(1.625)
    predictions[..., 0] = np.inf
    assert np.isnan(sql(labels, predictions, histories, 1, [0.25, 0.75]))


def test_infinite_history_follows_fev_scale_arithmetic():
    histories = [np.array([np.inf, np.inf, 1.0]), np.array([0.0, 2.0])]
    np.testing.assert_array_equal(seasonal_error_per_item(histories, 1), [np.inf, 2.0])
    labels = np.array([[8.0], [4.0]])
    predictions = np.zeros_like(labels)
    # fev divides first: a finite error / infinite scale contributes zero.
    assert mase(labels, predictions, histories, 1) == 1.0
    assert sql(labels, predictions[..., None], histories, 1, [0.5]) == 1.0


@pytest.mark.parametrize("eps,expected", [(1.0, 16 / 28), (10.0, 0.4)])
def test_weighted_errors_are_pooled_ratios_with_absolute_targets(eps, expected):
    labels = np.array([[-2.0, np.nan, np.nan], [6.0, 10.0, -10.0]])
    predictions = np.array([[0.0, 999.0, 999.0], [4.0, 4.0, -4.0]])
    assert wape(labels, predictions, eps=eps) == pytest.approx(expected)
    assert wql(labels, predictions[..., None], [0.5], eps=eps) == pytest.approx(expected)


def test_weighted_errors_default_epsilon_dominates_tiny_targets():
    labels = np.array([[1e-6, 3e-6]])
    assert wape(labels, np.zeros_like(labels)) == pytest.approx(2e-6, abs=0)
    assert wql(labels, np.zeros((1, 2, 2)), [0.1, 0.5]) == pytest.approx(1.2e-6, abs=0)


@pytest.mark.parametrize("knots", [[0.1, 0.5, 0.9], [0.2, 0.5, 0.8]])
def test_interval_coverage_is_inclusive_uses_nearest_knots_and_masks_missing_labels(knots):
    labels = np.array([[1.0, 4.0, 8.0, np.nan]])
    predictions = np.array([[[1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 50, 100]]])
    coverage, width = interval_coverage(labels, predictions, knots)
    assert coverage == pytest.approx(2 / 3)
    assert width == 2.0
    assert interval_coverage(labels, predictions, knots, lo=0.5, hi=0.5) == (0.0, 0.0)


def test_summary_uses_median_knot_and_reports_doubled_pinball_per_knot():
    result = summarize(
        np.array([[4.0]]),
        np.array([[[1.0, 2.0, 4.0, 5.0]]]),
        [np.array([0.0, 2.0])],
        1,
        [0.1, 0.5, 0.8, 0.9],
    )
    assert result == pytest.approx(
        {
            "SQL": 0.35,
            "MASE": 1.0,
            "WAPE": 0.5,
            "WQL": 0.175,
            "coverage_80": 1.0,
            "width_80": 4.0,
            "pinball_0.1": 0.6,
            "pinball_0.5": 2.0,
            "pinball_0.8": 0.0,
            "pinball_0.9": 0.2,
        }
    )
    assert all(isinstance(value, float) for value in result.values())


@pytest.mark.parametrize("histories", [[np.array([5.0, 5.0])], [np.array([1.0])]])
def test_no_eligible_scale_only_makes_scaled_metrics_nan(histories):
    result = summarize(np.array([[6.0]]), np.array([[[5.0]]]), histories, 1, [0.5])
    assert np.isnan(result.pop("SQL")) and np.isnan(result.pop("MASE"))
    assert all(np.isfinite(value) for value in result.values())


@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_no_labels_gives_nan_for_every_summary_metric(shape):
    result = summarize(
        np.full(shape, np.nan),
        np.ones((*shape, 3)),
        [np.array([0.0, 2.0])] * shape[0],
        1,
        [0.1, 0.5, 0.9],
    )
    assert all(np.isnan(value) for value in result.values())
