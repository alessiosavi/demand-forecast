"""Time feature calculations for demand forecasting."""

from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd


def calculate_time_features(
    series: pd.DataFrame,
    label: Literal["present", "future"] = "present",
    window: int = 1,
) -> None:
    """Add cyclical time features to a DataFrame.

    Decomposes time into sin/cos coordinates for week and month of year.
    This ensures that the first week of the year is close to the last week
    rather than the farthest (as would be the case with linear encoding).

    Args:
        series: DataFrame with DatetimeIndex to add features to.
        label: Whether features are for present ("p_") or future ("f_") time.
        window: Number of weeks to offset for future features.

    Note:
        Modifies the DataFrame in-place by adding columns:
        - {prefix}_t_sin, {prefix}_t_cos: Week of year (1-53) cyclical encoding
        - {prefix}_m_sin, {prefix}_m_cos: Month of year (1-12) cyclical encoding
    """
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
