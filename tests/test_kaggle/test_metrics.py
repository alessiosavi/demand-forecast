import numpy as np
import pytest

from demand_forecast.kaggle.metrics import METRIC_FUNCTIONS, rmse, rmsle


def test_rmsle_matches_hand_computation():
    y_true = np.array([1.0, 10.0])
    y_pred = np.array([2.0, 0.0])
    expected = np.sqrt(
        (np.log1p(2.0) - np.log1p(1.0)) ** 2 / 2 + (np.log1p(0.0) - np.log1p(10.0)) ** 2 / 2
    )
    assert rmsle(y_true, y_pred) == pytest.approx(expected)


def test_rmsle_zero_prediction_is_finite():
    assert rmsle(np.array([0.0]), np.array([0.0])) == 0.0


def test_rmsle_skips_nan_labels():
    y_true = np.array([1.0, np.nan])
    y_pred = np.array([2.0, 5.0])
    assert rmsle(y_true, y_pred) == pytest.approx(abs(np.log1p(2.0) - np.log1p(1.0)))


def test_rmse_matches_hand_computation():
    assert rmse(np.array([1.0, 3.0]), np.array([2.0, 1.0])) == pytest.approx(
        np.sqrt((1.0 + 4.0) / 2)
    )


def test_registry_exposes_competition_metrics():
    assert set(METRIC_FUNCTIONS) == {"rmsle", "rmse"}
