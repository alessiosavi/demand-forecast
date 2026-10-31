"""Index-based training origins, masked batches and resumable series sampling."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import torch

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import calendar_feature_names, calendar_features, grid_range
from demand_forecast.data.panel import Panel
from demand_forecast.data.scaling import sample_frame


@dataclass
class Batch:
    """Normalised targets and covariates; boolean masks are True for valid cells."""

    target: torch.Tensor
    observed: torch.Tensor
    padding: torch.Tensor
    enc_cov: torch.Tensor
    enc_cov_mask: torch.Tensor
    dec_cov: torch.Tensor
    dec_cov_mask: torch.Tensor
    cat: dict[str, torch.Tensor]
    num: torch.Tensor
    num_mask: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    eligible: torch.Tensor
    has_context: torch.Tensor
    loss_eligible: torch.Tensor
    series_index: torch.Tensor
    origin_pos: torch.Tensor

    @property
    def size(self) -> int:
        return self.target.shape[0]

    def to(self, device: torch.device | str) -> Batch:
        """Return a batch with every tensor, including categorical codes, on device."""
        return self._map_tensors(lambda tensor: tensor.to(device))

    def clone(self) -> Batch:
        """Copy every tensor without sharing storage with this batch."""
        return self._map_tensors(torch.clone)

    def _map_tensors(self, function) -> Batch:
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                {name: function(tensor) for name, tensor in value.items()}
                if isinstance(value, dict)
                else function(value)
            )
        return Batch(**values)


def _positive(value: int, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def build_origins(
    panel: Panel,
    horizon: int,
    min_context: int,
    upto: pd.Timestamp | None = None,
    require_label: bool = True,
) -> np.ndarray:
    """Index every eligible origin using cumulative observed counts per series.

    Counts include all historical positions before the origin, independently of
    the context crop. ``upto`` is inclusive and may fall between grid positions.
    The final history boundary is available when labels are not required.
    """
    _positive(horizon, "horizon")
    _positive(min_context, "min_context")
    if len(panel) > np.iinfo(np.int32).max:
        raise ValueError("series indices exceed int32 capacity")
    chunks = []
    cutoff_positions = {}
    for i, series in enumerate(panel.series):
        length = series.history_length
        if length > np.iinfo(np.int32).max:
            raise ValueError("origin positions exceed int32 capacity")
        limit = length
        if upto is not None:
            key = (series.start, length)
            if key not in cutoff_positions:
                grid = grid_range(series.start, length + 1, panel.freq)
                cutoff_positions[key] = int(grid.searchsorted(upto, side="right")) - 1
            limit = cutoff_positions[key]
        if limit < min_context:
            continue
        counts = np.empty(length + 1, dtype=np.int64)
        counts[0] = 0
        np.cumsum(~np.isnan(series.target[:length]), out=counts[1:])
        positions = np.arange(limit + 1, dtype=np.int64)
        valid = counts[positions] >= min_context
        if require_label:
            ends = np.minimum(positions + horizon, min(length, limit + 1))
            valid &= counts[ends] > counts[positions]
        positions = positions[valid]
        if positions.size:
            chunk = np.empty((positions.size, 2), dtype=np.int32)
            chunk[:, 0] = i
            chunk[:, 1] = positions
            chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.int32)


def _items_array(panel: Panel, items: Iterable[tuple[int, int]]) -> np.ndarray:
    items = np.asarray(items if isinstance(items, np.ndarray) else list(items))
    if items.shape == (0,):
        return np.empty((0, 2), dtype=np.int64)
    if items.ndim != 2 or items.shape[1] != 2 or items.dtype.kind not in "iu":
        raise ValueError("items must be integer (series_index, origin_pos) pairs")
    if np.any(items < 0) or np.any(items[:, 0] >= len(panel)):
        raise ValueError("items contain an invalid series index or origin position")
    return items


def _dynamic_values(
    source: np.ndarray | None, start: int, origin: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice one dynamic column and standardise using only its observed past."""
    values = np.full(stop - start, np.nan, dtype=np.float64)
    if source is None:
        return np.zeros(stop - start, dtype=np.float32), np.zeros(stop - start, dtype=bool)
    end = min(stop, len(source))
    if end > start:
        values[: end - start] = source[start:end]
    past = np.asarray(source[:origin], dtype=np.float64)
    past = past[~np.isnan(past)]
    # An unobserved past uses the identity transform. Classify indicators from
    # the past as well, so future values cannot change the context transform.
    if past.size and not np.all((past == 0) | (past == 1)):
        values = (values - past.mean()) / (past.std() or 1.0)
    mask = ~np.isnan(values)
    return np.where(mask, values, 0).astype(np.float32), mask


def make_batch(
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
    _positive(horizon, "horizon")
    _positive(max_context, "max_context")
    items = _items_array(panel, items)
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
            values, mask = _dynamic_values(series.dynamic.get(name), start, origin, stop)
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


class WindowSampler:
    """One uniform origin per eligible series, in a seeded permutation per epoch.

    Iteration consumes the current epoch; call ``set_epoch`` to start another.
    The draw list is reproducible from seed/epoch, so only epoch and position
    need checkpointing. Restore with the same seed, origins and panel.
    """

    def __init__(
        self,
        panel: Panel,
        origins: np.ndarray,
        horizon: int,
        max_context: int,
        encoder: StaticEncoder,
        seed: int = 0,
        label_upto: pd.Timestamp | None = None,
    ) -> None:
        _positive(horizon, "horizon")
        _positive(max_context, "max_context")
        _positive(seed + 1, "seed + 1")
        origins = _items_array(panel, origins)
        lengths = np.array([series.history_length for series in panel.series], dtype=np.int64)
        if np.any(origins[:, 1] > lengths[origins[:, 0]]):
            raise ValueError("origin position exceeds series history_length")
        # Avoid sorting the large, already-grouped output of build_origins.
        if np.any(origins[1:, 0] < origins[:-1, 0]):
            origins = origins[np.argsort(origins[:, 0], kind="stable")]
        self.origins = origins
        _, self._starts, self._counts = np.unique(
            origins[:, 0], return_index=True, return_counts=True
        )
        self.panel, self.encoder = panel, encoder
        self.horizon, self.max_context = horizon, max_context
        self.seed, self.label_upto = int(seed), label_upto
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self._starts)

    def set_epoch(self, epoch: int) -> None:
        """Reset this epoch to its first draw, independent of global RNG state."""
        _positive(epoch + 1, "epoch + 1")
        self.epoch, self.position = int(epoch), 0
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(len(self))
        starts = self._starts[order]
        selected = rng.integers(starts, starts + self._counts[order])
        self._draws = self.origins[selected]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        while self.position < len(self):
            i, origin = self._draws[self.position]
            # Advance before yielding so a checkpoint describes the next draw.
            self.position += 1
            yield int(i), int(origin)

    def batches(self, batch_size: int) -> Iterator[Batch]:
        """Batch consecutive remaining draws, retaining a final partial batch."""
        _positive(batch_size, "batch_size")
        items = []
        for item in self:
            items.append(item)
            if len(items) == batch_size:
                yield make_batch(
                    self.panel, items, self.horizon, self.max_context, self.encoder, self.label_upto
                )
                items = []
        if items:
            yield make_batch(
                self.panel, items, self.horizon, self.max_context, self.encoder, self.label_upto
            )

    def state(self) -> dict:
        return {"epoch": self.epoch, "position": self.position}

    def load_state(self, d: dict) -> None:
        """Reconstruct the epoch's draw list and resume at its next position."""
        position = d["position"]
        if not isinstance(position, (int, np.integer)) or not 0 <= position <= len(self):
            raise ValueError("sampler position must be between zero and the epoch length")
        self.set_epoch(d["epoch"])
        self.position = int(position)
