"""Seasonal-naive baseline for the Kaggle bench.

Standalone grid-aware point forecasts, used as the reference row of the
mirror scoring. Points are the last observed target one seasonal period
back, with the origin positioned by the panel's own grid.
"""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

from demand_forecast.data.frequency import grid_range, steps_between
from demand_forecast.data.panel import Panel

__all__ = ["seasonal_naive_points"]


def seasonal_naive_points(panel: Panel, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Point forecasts with columns ``series_id``, ``timestamp`` and ``point``.

    Timestamps are the ``horizon`` grid steps starting at ``origin`` (the first
    forecast step), once per series. At grid position ``p`` relative to the
    series start, the point is the last observed target at or before position
    ``p - seasonal_period``; if there is none, the last observed target at or
    before ``p``. "Observed" means a finite target value, so the loader's
    zero-fill policy counts filled zeros as observed. A step with no candidate
    at all gets 0.0 under ``target_support == "nonnegative"``, else NaN.
    """
    timestamps = grid_range(origin, horizon, panel.freq)
    empty_level = 0.0 if panel.target_support == "nonnegative" else float("nan")

    series_ids: list[str] = []
    points: list[float] = []
    for series in panel.series:
        target = np.asarray(series.target, dtype=np.float64)
        observed = np.flatnonzero(np.isfinite(target))
        for timestamp in timestamps:
            position = steps_between(series.start, timestamp, panel.freq)
            point = empty_level
            if observed.size:
                index = np.searchsorted(observed, position - panel.seasonal_period, side="right")
                if index == 0:
                    index = np.searchsorted(observed, position, side="right")
                if index > 0:
                    point = float(target[observed[index - 1]])
            series_ids.append(series.series_id)
            points.append(point)
    return pd.DataFrame(
        {
            "series_id": pd.Series(series_ids, dtype="object"),
            "timestamp": pd.DatetimeIndex(np.tile(timestamps.to_numpy(), len(panel.series))),
            "point": pd.Series(points, dtype="float64"),
        }
    )
