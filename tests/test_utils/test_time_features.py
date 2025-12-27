"""Tests for time feature utilities."""

import numpy as np
import pandas as pd

from demand_forecast.utils.time_features import calculate_time_features


class TestCalculateTimeFeatures:
    """Tests for calculate_time_features function."""

    def test_basic_output_shape(self):
        """Test that output has correct shape."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        features = calculate_time_features(dates)

        # Should have 4 features: sin_week, cos_week, sin_year, cos_year
        assert features.shape == (10, 4)

    def test_values_in_range(self):
        """Test that sin/cos values are in [-1, 1]."""
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        features = calculate_time_features(dates)

        assert np.all(features >= -1)
        assert np.all(features <= 1)

    def test_periodicity(self):
        """Test that features are periodic."""
        # Create two years of data
        dates = pd.date_range("2023-01-01", periods=730, freq="D")
        features = calculate_time_features(dates)

        # Values should be similar for same day of year
        # Week features may differ due to ISO week numbering across year boundaries
        # Month features (columns 2,3) should be very close for same calendar day
        day_0_year_1 = features[0]
        day_0_year_2 = features[365]

        # Month features (sin/cos of month) should be nearly identical
        np.testing.assert_array_almost_equal(day_0_year_1[2:], day_0_year_2[2:], decimal=5)

        # Week features may vary slightly due to ISO week year boundaries
        # Just check they're in valid range
        assert np.all(np.abs(day_0_year_1[:2]) <= 1.0)
        assert np.all(np.abs(day_0_year_2[:2]) <= 1.0)

    def test_empty_input(self):
        """Test with empty date index."""
        dates = pd.DatetimeIndex([])
        features = calculate_time_features(dates)

        assert features.shape == (0, 4)

    def test_single_date(self):
        """Test with single date."""
        dates = pd.DatetimeIndex(["2023-06-15"])
        features = calculate_time_features(dates)

        assert features.shape == (1, 4)
