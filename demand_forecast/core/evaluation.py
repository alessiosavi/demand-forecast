"""Evaluate quantile forecasts in original units against complete past histories."""

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import torch

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import steps_between
from demand_forecast.data.panel import Panel
from demand_forecast.data.sampler import Batch, align_to_origin, make_batch
from demand_forecast.metrics import summarize
from demand_forecast.models.base import ForecastModel


@dataclass(frozen=True)
class PreparedValidation:
    """CPU inputs retained for one fit session, in validation item order."""

    batches: tuple[Batch, ...]
    y_true: np.ndarray
    y_past: tuple[np.ndarray, ...]
    seasonal_period: int

    @property
    def nbytes(self) -> int:
        """Payload bytes of the retained arrays and tensors, excluding metadata."""
        total = self.y_true.nbytes + sum(values.nbytes for values in self.y_past)
        for batch in self.batches:
            for field in fields(Batch):
                value = getattr(batch, field.name)
                tensors = value.values() if isinstance(value, dict) else [value]
                total += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return total


def prepare_validation(
    panel: Panel,
    items: np.ndarray,
    origin: pd.Timestamp,
    horizon: int,
    max_context: int,
    encoder: StaticEncoder,
    batch_size: int = 512,
) -> PreparedValidation:
    """Build ordered CPU batches and metric inputs once, without model work.

    Context may be cropped for prediction, but metric scaling always uses the
    full past. Missing targets, including a partial horizon, remain NaN.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    positions = [(int(i), steps_between(panel.series[i].start, origin, panel.freq)) for i in items]
    y_true = np.full((len(positions), horizon), np.nan, dtype=np.float64)
    y_past = []
    batch_panel = align_to_origin(panel, positions)
    for row, (i, pos) in enumerate(positions):
        series = panel.series[i]
        target = series.target[: series.history_length]
        labels = target[pos : pos + horizon]
        y_true[row, : len(labels)] = labels
        y_past.append(target[:pos].copy())

    batches = tuple(
        make_batch(
            batch_panel, positions[start : start + batch_size], horizon, max_context, encoder
        )
        for start in range(0, len(positions), batch_size)
    )
    return PreparedValidation(batches, y_true, tuple(y_past), panel.seasonal_period)


def evaluate_prepared(
    model: ForecastModel, prepared: PreparedValidation, quantiles: Sequence[float]
) -> dict[str, float]:
    """Score reusable CPU batches in original units, then resume training mode."""
    q_pred = np.empty((*prepared.y_true.shape, len(quantiles)), dtype=np.float64)

    device = model.fallback_level.device
    model.eval()
    try:
        with torch.no_grad():
            start = 0
            for cpu_batch in prepared.batches:
                batch = cpu_batch.to(device)
                # predict already denormalises and applies support/fallbacks.
                q_pred[start : start + batch.size] = model.predict(batch).cpu().numpy()
                start += batch.size
            return summarize(
                prepared.y_true, q_pred, prepared.y_past, prepared.seasonal_period, quantiles
            )
    finally:
        model.train()


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
    """Prepare and score validation inputs for callers that do not retain them."""
    return evaluate_prepared(
        model,
        prepare_validation(panel, items, origin, horizon, max_context, encoder, batch_size),
        quantiles,
    )
