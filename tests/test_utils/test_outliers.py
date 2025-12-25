"""Tests for outlier detection utilities."""

import pandas as pd

from demand_forecast.utils.outliers import remove_outliers


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_no_outliers(self):
        """Test with data that has no outliers."""
        data = pd.Series([1, 2, 3, 4, 5])
        clipped, has_outliers = remove_outliers(data, n=3)

        assert not has_outliers
        assert clipped.equals(data)

    def test_with_outliers(self):
        """Test with data containing outliers."""
        data = pd.Series([1, 2, 3, 100, 4, 5])
        clipped, has_outliers = remove_outliers(data, n=2)

        assert has_outliers
        assert clipped.max() < 100

    def test_empty_series(self):
        """Test with empty series."""
        data = pd.Series([], dtype=float)
        clipped, has_outliers = remove_outliers(data)

        assert not has_outliers
        assert len(clipped) == 0

    def test_single_value(self):
        """Test with single value."""
        data = pd.Series([5.0])
        clipped, has_outliers = remove_outliers(data)

        assert not has_outliers
        assert clipped.iloc[0] == 5.0

    def test_negative_values(self):
        """Test with negative outliers."""
        data = pd.Series([1, 2, 3, -100, 4, 5])
        clipped, has_outliers = remove_outliers(data, n=2)

        assert has_outliers
        assert clipped.min() > -100

    def test_custom_n(self):
        """Test with different n values."""
        data = pd.Series([1, 2, 3, 10, 4, 5])

        # With n=1, 10 should be an outlier
        _, has_outliers_strict = remove_outliers(data, n=1)

        # With n=5, 10 should not be an outlier
        _, has_outliers_loose = remove_outliers(data, n=5)

        assert has_outliers_strict
        assert not has_outliers_loose
