"""Time feature calculations for demand forecasting."""

from datetime import timedelta
from typing import Literal, overload

import numpy as np
import pandas as pd


@overload
def calculate_time_features(
    series: pd.DatetimeIndex,
    label: Literal["present", "future"] = "present",
    window: int = 1,
) -> np.ndarray:
    ...


@overload
def calculate_time_features(
    series: pd.DataFrame,
    label: Literal["present", "future"] = "present",
    window: int = 1,
) -> None:
    ...


def calculate_time_features(
    series: pd.DataFrame | pd.DatetimeIndex,
    label: Literal["present", "future"] = "present",
    window: int = 1,
) -> np.ndarray | None:
    """Calculate or add cyclical time features.

    Decomposes time into sin/cos coordinates for week and month of year.
    This ensures that the first week of the year is close to the last week
    rather than the farthest (as would be the case with linear encoding).

    Args:
        series: Either a DatetimeIndex (returns features as array) or
            DataFrame with DatetimeIndex (modifies in-place).
        label: Whether features are for present ("p_") or future ("f_") time.
        window: Number of weeks to offset for future features.

    Returns:
        If series is DatetimeIndex: numpy array of shape (n_dates, 4) with
            columns [week_sin, week_cos, month_sin, month_cos].
        If series is DataFrame: None (modifies in-place by adding columns).

    Note:
        When modifying DataFrame in-place, adds columns:
        - {prefix}_t_sin, {prefix}_t_cos: Week of year (1-53) cyclical encoding
        - {prefix}_m_sin, {prefix}_m_cos: Month of year (1-12) cyclical encoding
    """
    # Determine if we're working with a DatetimeIndex or DataFrame
    if isinstance(series, pd.DatetimeIndex):
        # Return features as numpy array
        if label == "present":
            calendar = series
        else:
            calendar = series + timedelta(days=window * 7)

        if len(calendar) == 0:
            return np.empty((0, 4))

        week_of_year = calendar.isocalendar().week.values
        month_of_year = calendar.month.values

        week_max = 53
        month_max = 12

        features = np.column_stack(
            [
                np.sin(week_of_year * (2 * np.pi / week_max)),
                np.cos(week_of_year * (2 * np.pi / week_max)),
                np.sin(month_of_year * (2 * np.pi / month_max)),
                np.cos(month_of_year * (2 * np.pi / month_max)),
            ]
        )
        return features

    # DataFrame case: modify in-place
    if label == "present":
        prefix = "p"
        calendar = series.index
    else:
        prefix = "f"
        calendar = series.index + timedelta(days=window * 7)

    week_of_year = calendar.isocalendar().week.values
    month_of_year = calendar.month.values

    week_max = 53
    month_max = 12

    # Week features (cyclical encoding)
    series[f"{prefix}_t_sin"] = np.sin(week_of_year * (2 * np.pi / week_max))
    series[f"{prefix}_t_cos"] = np.cos(week_of_year * (2 * np.pi / week_max))

    # Month features (cyclical encoding)
    series[f"{prefix}_m_sin"] = np.sin(month_of_year * (2 * np.pi / month_max))
    series[f"{prefix}_m_cos"] = np.cos(month_of_year * (2 * np.pi / month_max))

    return None
