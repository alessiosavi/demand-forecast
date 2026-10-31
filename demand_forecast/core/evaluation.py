"""Evaluate quantile forecasts in original units against complete past histories."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import steps_between
from demand_forecast.data.panel import Panel
from demand_forecast.data.sampler import make_batch
from demand_forecast.metrics import summarize
from demand_forecast.models.base import ForecastModel


def evaluate(
    model: ForecastModel,
    panel: Panel,
    items: np.ndarray,
    origin: pd.Timestamp,
    horizon: int,
    max_context: int,
    encoder: StaticEncoder,
    quantiles: Sequence[float],
    batch_size: int = 512,
) -> dict[str, float]:
    """Predict at each item's grid position, then resume training mode.

    Context may be cropped for prediction, but metric scaling always uses the
    full past. Missing targets, including a partial horizon, remain NaN.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    positions = [(int(i), steps_between(panel.series[i].start, origin, panel.freq)) for i in items]
    y_true = np.full((len(positions), horizon), np.nan, dtype=np.float64)
    q_pred = np.empty((len(positions), horizon, len(quantiles)), dtype=np.float64)
    y_past = []
    batch_panel = panel
    for row, (i, pos) in enumerate(positions):
        series = panel.series[i]
        target = series.target[: series.history_length]
        labels = target[pos : pos + horizon]
        y_true[row, : len(labels)] = labels
        y_past.append(target[:pos])
        if pos > series.history_length:
            # make_batch requires an origin within history. Represent a ragged
            # item's gap to the common origin as missing past, without changing
            # the caller's panel or inventing observations for metric scaling.
            if batch_panel is panel:
                batch_panel = replace(panel, series=list(panel.series))
            extended = np.full(pos, np.nan, dtype=np.float64)
            extended[: len(target)] = target
            batch_panel.series[i] = replace(series, target=extended, history_length=pos)

    device = model.fallback_level.device
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(positions), batch_size):
                batch = make_batch(
                    batch_panel,
                    positions[start : start + batch_size],
                    horizon,
                    max_context,
                    encoder,
                ).to(device)
                # predict already denormalises and applies support/fallbacks.
                q_pred[start : start + batch.size] = model.predict(batch).cpu().numpy()
            return summarize(y_true, q_pred, y_past, panel.seasonal_period, quantiles)
    finally:
        model.train()
