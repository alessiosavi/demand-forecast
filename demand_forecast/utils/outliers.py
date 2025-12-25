"""Outlier detection and removal utilities."""

import pandas as pd


def remove_outliers(data: pd.Series, n: int = 3) -> tuple[pd.Series, bool]:
    """Remove outliers using Z-score method.

    Values with |Z-score| > n are clipped to the boundary values.

    Args:
        data: Series of numeric values.
        n: Z-score threshold for outlier detection. Default is 3.

    Returns:
        Tuple of (clipped_data, has_outliers) where:
            - clipped_data: Series with outliers clipped to boundary values
            - has_outliers: Boolean indicating if any outliers were found

    Example:
        >>> data = pd.Series([1, 2, 3, 100, 4, 5])
        >>> clipped, had_outliers = remove_outliers(data, n=2)
        >>> had_outliers
        True
    """
    mean = data.mean()
    std = data.std()

    if std == 0:
        return data, False

    data_zscore = (data - mean) / std

    # Identify outliers (|Z| > n)
    outliers_z = data[abs(data_zscore) > n]
    has_outliers = len(outliers_z) > 0

    clip_data = data
    if has_outliers:
        lower_bound = mean - n * std
        upper_bound = mean + n * std
        clip_data = data.clip(lower=lower_bound, upper=upper_bound)

    return clip_data, has_outliers
