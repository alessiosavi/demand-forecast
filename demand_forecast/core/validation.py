"""Select an inner-validation split, or use all past for short histories."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from demand_forecast.data.frequency import grid_range, to_offset
from demand_forecast.data.panel import Panel
from demand_forecast.data.scaling import seasonal_error


@dataclass
class ValidationPlan:
    """Inclusive training cutoff and eligible items at a shared validation origin."""

    train_upto: pd.Timestamp
    val_origin: pd.Timestamp
    val_items: np.ndarray
    exception: bool
    fit_ids: list[str]


def subsample_validation(plan: ValidationPlan, max_items: int | None, seed: int) -> ValidationPlan:
    """Limit validation items deterministically without changing the training split."""
    if max_items is None or len(plan.val_items) <= max_items:
        return plan
    items = np.random.default_rng(seed).choice(plan.val_items, size=max_items, replace=False)
    return replace(plan, val_items=np.sort(items))


def plan_inner_validation(
    panel: Panel,
    origin: pd.Timestamp,
    horizon: int,
    min_train_history: int | None = None,
    seasonal_period: int | None = None,
) -> ValidationPlan:
    """Reserve the last horizon grid steps before an on-grid forecast origin.

    Validation needs enough observed past, a positive finite seasonal error,
    and at least one historical label. If no series qualifies, the training
    split includes all history before ``origin``. Fit ids include any series
    observed in the chosen split, independently of validation eligibility.
    """
    min_train_history = horizon + 8 if min_train_history is None else min_train_history
    seasonal_period = panel.seasonal_period if seasonal_period is None else seasonal_period
    # Walk the grid so daily steps retain local midnight across DST on pandas 2/3.
    boundaries = pd.date_range(end=origin, periods=horizon + 2, freq=to_offset(panel.freq))
    val_origin = boundaries[1]
    val_items = []
    fit_before_validation = []
    fit_before_origin = []
    cutoff_positions = {}

    for i, series in enumerate(panel.series):
        key = (series.start, series.history_length)
        if key not in cutoff_positions:
            grid = grid_range(series.start, series.history_length, panel.freq)
            cutoff_positions[key] = (
                int(grid.searchsorted(val_origin, side="left")),
                int(grid.searchsorted(origin, side="left")),
                int(grid.searchsorted(boundaries[0], side="right")),
                int(grid.searchsorted(boundaries[-2], side="right")),
            )
        val_pos, origin_pos, train_end, full_train_end = cutoff_positions[key]
        target = series.target[: series.history_length]
        past = target[:val_pos]
        observed = ~np.isnan(target)
        past_count = np.count_nonzero(observed[:val_pos])
        has_label = bool(observed[val_pos:origin_pos].any())
        if observed[:train_end].any():
            fit_before_validation.append(series.series_id)
        if observed[:full_train_end].any():
            fit_before_origin.append(series.series_id)
        if past_count >= min_train_history and has_label:
            error = seasonal_error(past, seasonal_period)
            if np.isfinite(error) and error > 0:
                val_items.append(i)

    exception = len(val_items) == 0
    return ValidationPlan(
        train_upto=boundaries[-2] if exception else boundaries[0],
        val_origin=val_origin,
        val_items=np.asarray(val_items, dtype=np.int64),
        exception=exception,
        fit_ids=fit_before_origin if exception else fit_before_validation,
    )
