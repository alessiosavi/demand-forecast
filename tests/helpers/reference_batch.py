# Frozen reference: verbatim functions from commit 4e33a3e7d811b5fa9dcc1101a48f6c1fc454dfc7 (rewritten history: ba7fc31e569609d25cee013604f6dd299d0baa6e), renamed reference_*.
# Keep these bodies independent of the live alignment, batching and evaluation paths.
"""Frozen batching/evaluation oracle and exact tensor comparison for parity tests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import fields, replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import (
    calendar_feature_names,
    calendar_features,
    grid_range,
    steps_between,
)
from demand_forecast.data.panel import Panel
from demand_forecast.data.sampler import Batch
from demand_forecast.data.scaling import sample_frame
from demand_forecast.metrics import summarize
from demand_forecast.models.base import ForecastModel

COVARIATE_BOUND: float = 1e3
REL_STD_FLOOR: float = 1e-3


# fmt: off
class DynamicDiagnostics(NamedTuple):
    """Cold/floored sample-column flags and the number of clipped valid values."""

    cold: bool
    clipped: int
    floored: bool


def reference_positive(value: int, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def reference_items_array(panel: Panel, items: Iterable[tuple[int, int]]) -> np.ndarray:
    items = np.asarray(items if isinstance(items, np.ndarray) else list(items))
    if items.shape == (0,):
        return np.empty((0, 2), dtype=np.int64)
    if items.ndim != 2 or items.shape[1] != 2 or items.dtype.kind not in "iu":
        raise ValueError("items must be integer (series_index, origin_pos) pairs")
    if np.any(items < 0) or np.any(items[:, 0] >= len(panel)):
        raise ValueError("items contain an invalid series index or origin position")
    return items


def reference_add_row_diagnostics(batch: Any, row: int, diag: DynamicDiagnostics) -> None:
    """Accumulate one sample-column's diagnostics in its batch row.

    Any object exposing per-row-indexable cold_covariates, clipped_covariates, floored_covariates may be passed.
    """
    batch.cold_covariates[row] += diag.cold
    batch.clipped_covariates[row] += diag.clipped
    batch.floored_covariates[row] += diag.floored


def reference_align_to_origin(panel: Panel, items: Iterable[tuple[int, int]]) -> Panel:
    """Extend ragged histories to nonnegative origin positions without mutation.

    Only affected series are copied. Gaps contain NaN targets and past-only
    covariates, while known covariates retain all available values. This anchors
    both context and future features at the requested origin in ``make_batch``.
    Callers clamp positions before a series' start to zero for empty context.
    """
    aligned = panel
    for i, pos in reference_items_array(panel, items):
        i, pos = int(i), int(pos)
        series = aligned.series[i]
        length = series.history_length
        if pos <= length:
            continue
        if aligned is panel:
            aligned = replace(panel, series=list(panel.series))
        target = np.full(pos, np.nan, dtype=np.float64)
        target[:length] = series.target[:length]
        dynamic = dict(series.dynamic)
        for name in panel.roles.past_dynamic:
            values = np.full(pos, np.nan, dtype=np.float64)
            source = series.dynamic.get(name)
            if source is not None:
                end = min(length, len(source))
                values[:end] = source[:end]
            dynamic[name] = values
        aligned.series[i] = replace(series, target=target, dynamic=dynamic, history_length=pos)
    return aligned


def reference_dynamic_values(
    source: np.ndarray | None, start: int, origin: int, stop: int
) -> tuple[np.ndarray, np.ndarray, DynamicDiagnostics]:
    """Standardise a dynamic column using only its finite, whole past.

    Classify indicators from the past so future values cannot change the context
    transform. A column with no observed past but finite values in this slice is
    cold: mask off the entire slice. Floor positive standard deviations relative
    to the largest absolute past value; a zero std still uses scale 1. Bound valid
    values before conversion to float32; non-finite source values are missing
    throughout.
    """
    values = np.full(stop - start, np.nan, dtype=np.float64)
    if source is None:
        return (
            np.zeros(stop - start, dtype=np.float32),
            np.zeros(stop - start, dtype=bool),
            DynamicDiagnostics(cold=False, clipped=0, floored=False),
        )
    end = min(stop, len(source))
    if end > start:
        values[: end - start] = source[start:end]
    past = np.asarray(source[:origin], dtype=np.float64)
    past = past[np.isfinite(past)]
    mask = np.isfinite(values)
    if not past.size and mask.any():
        return (
            np.zeros(stop - start, dtype=np.float32),
            np.zeros(stop - start, dtype=bool),
            DynamicDiagnostics(cold=True, clipped=0, floored=False),
        )
    values[~mask] = np.nan
    floored = False
    if past.size and not np.all((past == 0) | (past == 1)):
        std = past.std()
        scale = std or 1.0
        if std > 0:
            scale = max(std, REL_STD_FLOOR * np.max(np.abs(past)))
            floored = bool(scale > std)
        values = (values - past.mean()) / scale
    clipped = int(np.count_nonzero(mask & (np.abs(values) > COVARIATE_BOUND)))
    values = np.clip(values, -COVARIATE_BOUND, COVARIATE_BOUND)
    return (
        np.where(mask, values, 0).astype(np.float32),
        mask,
        DynamicDiagnostics(cold=False, clipped=clipped, floored=floored),
    )


def reference_make_batch(
    panel: Panel,
    items: Iterable[tuple[int, int]],
    horizon: int,
    max_context: int,
    encoder: StaticEncoder,
    label_upto: pd.Timestamp | None = None,
) -> Batch:
    """Build left-padded CPU tensors without fitting any statistics on the future.

    Target location uses the cropped context; target scale and dynamic feature
    statistics use the whole past. Calendar sin/cos features keep their native
    scale. Masked cells (including withheld labels) contain zero.
    """
    reference_positive(horizon, "horizon")
    reference_positive(max_context, "max_context")
    items = reference_items_array(panel, items)
    for i, origin in items:
        if origin > panel.series[i].history_length:
            raise ValueError("origin position exceeds series history_length")
    size = len(items)
    length = min(max_context, int(items[:, 1].max())) if size else 0
    n_calendar = len(calendar_feature_names(panel.freq))
    known = panel.roles.known_dynamic
    dynamic = (*known, *panel.roles.past_dynamic)
    n_enc, n_dec = n_calendar + len(dynamic), n_calendar + len(known)
    batch = Batch(
        target=torch.zeros((size, length), dtype=torch.float32),
        observed=torch.zeros((size, length), dtype=torch.bool),
        padding=torch.zeros((size, length), dtype=torch.bool),
        enc_cov=torch.zeros((size, length, n_enc), dtype=torch.float32),
        enc_cov_mask=torch.zeros((size, length, n_enc), dtype=torch.bool),
        dec_cov=torch.zeros((size, horizon, n_dec), dtype=torch.float32),
        dec_cov_mask=torch.zeros((size, horizon, n_dec), dtype=torch.bool),
        cat={name: torch.zeros(size, dtype=torch.long) for name in encoder.vocab_sizes},
        num=torch.zeros((size, encoder.n_numeric), dtype=torch.float32),
        num_mask=torch.zeros((size, encoder.n_numeric), dtype=torch.bool),
        labels=torch.zeros((size, horizon), dtype=torch.float32),
        label_mask=torch.zeros((size, horizon), dtype=torch.bool),
        loc=torch.zeros(size, dtype=torch.float32),
        scale=torch.ones(size, dtype=torch.float32),
        eligible=torch.zeros(size, dtype=torch.bool),
        has_context=torch.zeros(size, dtype=torch.bool),
        loss_eligible=torch.zeros(size, dtype=torch.bool),
        series_index=torch.tensor(items[:, 0], dtype=torch.long),
        origin_pos=torch.tensor(items[:, 1], dtype=torch.long),
        cold_covariates=torch.zeros(size, dtype=torch.int64),
        clipped_covariates=torch.zeros(size, dtype=torch.int64),
        floored_covariates=torch.zeros(size, dtype=torch.int64),
    )
    for row, (i, origin) in enumerate(items):
        i, origin = int(i), int(origin)
        series = panel.series[i]
        start = max(0, origin - max_context)
        context_length = origin - start
        dest = slice(length - context_length, length)
        context = np.asarray(series.target[start:origin], dtype=np.float64)
        frame = sample_frame(context, series.target[:origin], panel.seasonal_period)
        observed = ~np.isnan(context)
        batch.target[row, dest] = torch.from_numpy(
            np.where(observed, (context - frame.loc) / frame.scale, 0).astype(np.float32)
        )
        batch.observed[row, dest] = torch.from_numpy(observed)
        batch.padding[row, dest] = True
        batch.loc[row], batch.scale[row] = frame.loc, frame.scale
        batch.eligible[row], batch.has_context[row] = frame.eligible, frame.has_context

        grid = grid_range(series.start, origin + horizon, panel.freq)
        calendar = torch.from_numpy(calendar_features(grid[start:], panel.freq))
        batch.enc_cov[row, dest, :n_calendar] = calendar[:context_length]
        batch.dec_cov[row, :, :n_calendar] = calendar[context_length:]
        batch.enc_cov_mask[row, dest, :n_calendar] = True
        batch.dec_cov_mask[row, :, :n_calendar] = True
        for column, name in enumerate(dynamic, start=n_calendar):
            is_known = column < n_dec
            stop = origin + horizon if is_known else origin
            values, mask, diagnostics = reference_dynamic_values(
                series.dynamic.get(name), start, origin, stop
            )
            reference_add_row_diagnostics(batch, row, diagnostics)
            batch.enc_cov[row, dest, column] = torch.from_numpy(values[:context_length])
            batch.enc_cov_mask[row, dest, column] = torch.from_numpy(mask[:context_length])
            if is_known:
                batch.dec_cov[row, :, column] = torch.from_numpy(values[context_length:])
                batch.dec_cov_mask[row, :, column] = torch.from_numpy(mask[context_length:])

        label_end = min(origin + horizon, series.history_length)
        labels = np.asarray(series.target[origin:label_end], dtype=np.float64)
        label_mask = ~np.isnan(labels)
        if label_upto is not None:
            label_mask &= grid[origin:label_end] <= label_upto
        batch.labels[row, : len(labels)] = torch.from_numpy(
            np.where(label_mask, (labels - frame.loc) / frame.scale, 0).astype(np.float32)
        )
        batch.label_mask[row, : len(labels)] = torch.from_numpy(label_mask)
        codes, num, num_mask = encoder.transform(series)
        for name, code in codes.items():
            batch.cat[name][row] = code
        batch.num[row] = torch.from_numpy(num)
        batch.num_mask[row] = torch.from_numpy(num_mask)
    batch.loss_eligible = batch.eligible & batch.label_mask.any(dim=1)
    return batch


def reference_batches(self, batch_size: int) -> Iterator[Batch]:
    """Batch consecutive remaining draws, retaining a final partial batch."""
    reference_positive(batch_size, "batch_size")
    items = []
    for item in self:
        items.append(item)
        if len(items) == batch_size:
            yield reference_make_batch(
                self.panel, items, self.horizon, self.max_context, self.encoder, self.label_upto
            )
            items = []
    if items:
        yield reference_make_batch(
            self.panel, items, self.horizon, self.max_context, self.encoder, self.label_upto
        )


def reference_evaluate(
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
    batch_panel = reference_align_to_origin(panel, positions)
    for row, (i, pos) in enumerate(positions):
        series = panel.series[i]
        target = series.target[: series.history_length]
        labels = target[pos : pos + horizon]
        y_true[row, : len(labels)] = labels
        y_past.append(target[:pos])

    device = model.fallback_level.device
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(positions), batch_size):
                batch = reference_make_batch(
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


# fmt: on
def assert_batches_equal(actual: Batch, expected: Batch) -> None:
    """Require every field's keys, dtype, shape and values to match exactly."""
    for field in fields(Batch):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, dict):
            assert left.keys() == right.keys(), field.name
            pairs = [(f"{field.name}.{key}", left[key], right[key]) for key in right]
        else:
            pairs = [(field.name, left, right)]
        for name, a, b in pairs:
            assert a.dtype == b.dtype, name
            assert a.shape == b.shape, name
            assert torch.equal(a, b), name
