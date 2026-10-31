"""Index-based training origins, masked batches and resumable series sampling."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields, replace

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


def align_to_origin(panel: Panel, items: Iterable[tuple[int, int]]) -> Panel:
    """Extend ragged histories to nonnegative origin positions without mutation.

    Only affected series are copied. Gaps contain NaN targets and past-only
    covariates, while known covariates retain all available values. This anchors
    both context and future features at the requested origin in ``make_batch``.
    Callers clamp positions before a series' start to zero for empty context.
    """
    aligned = panel
    for i, pos in _items_array(panel, items):
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


class BatchBuilder:
    """Reuse grid/calendar and static features for a fixed panel and fitted encoder.

    A builder belongs to one sampling/fit session. The panel's starts, history
    lengths, statics and encoder must stay fixed during that session. Target and
    dynamic reductions retain the reference's per-row slices and operation order.
    Each build owns fresh buffers, so later batches cannot overwrite earlier ones.
    """

    def __init__(
        self, panel: Panel, horizon: int, max_context: int, encoder: StaticEncoder
    ) -> None:
        _positive(horizon, "horizon")
        _positive(max_context, "max_context")
        self.panel, self.encoder = panel, encoder
        self.horizon, self.max_context = horizon, max_context
        self._lengths = {}
        for series in panel.series:
            # Timestamp equality compares instants, while calendar features use
            # local time. Equal instants in different zones need separate grids.
            key = (series.start, str(series.start.tz))
            self._lengths[key] = max(self._lengths.get(key, 0), series.history_length)
        self._grids = {}
        self._statics = {}
        self._cutoffs = {}

    def _grid(self, key):
        if key not in self._grids:
            # Build at full length on first use: a short same-start series must
            # not bound the label cut-off cached for a later, longer request.
            grid = grid_range(key[0], self._lengths[key] + self.horizon, self.panel.freq)
            self._grids[key] = grid, calendar_features(grid, self.panel.freq)
        return self._grids[key]

    def _label_limit(self, key, grid, label_upto):
        cache_key = (key, label_upto)
        if cache_key not in self._cutoffs:
            # Preserve pandas comparison errors (including timezone mismatches).
            # searchsorted alone accepts some values the reference rejects, and
            # puts NaT at the end even though every <= NaT comparison is false.
            grid[:0] <= label_upto
            self._cutoffs[cache_key] = (
                0 if pd.isna(label_upto) else int(grid.searchsorted(label_upto, side="right"))
            )
        return self._cutoffs[cache_key]

    def build(
        self, items: Iterable[tuple[int, int]], label_upto: pd.Timestamp | None = None
    ) -> Batch:
        """Build left-padded CPU tensors with the frozen per-row numeric semantics."""
        panel, encoder = self.panel, self.encoder
        horizon, max_context = self.horizon, self.max_context
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
        arrays = dict(
            target=np.zeros((size, length), dtype=np.float32),
            observed=np.zeros((size, length), dtype=bool),
            padding=np.zeros((size, length), dtype=bool),
            enc_cov=np.zeros((size, length, n_enc), dtype=np.float32),
            enc_cov_mask=np.zeros((size, length, n_enc), dtype=bool),
            dec_cov=np.zeros((size, horizon, n_dec), dtype=np.float32),
            dec_cov_mask=np.zeros((size, horizon, n_dec), dtype=bool),
            num=np.zeros((size, encoder.n_numeric), dtype=np.float32),
            num_mask=np.zeros((size, encoder.n_numeric), dtype=bool),
            labels=np.zeros((size, horizon), dtype=np.float32),
            label_mask=np.zeros((size, horizon), dtype=bool),
            eligible=np.zeros(size, dtype=bool),
            has_context=np.zeros(size, dtype=bool),
            series_index=np.array(items[:, 0], dtype=np.int64, copy=True),
            origin_pos=np.array(items[:, 1], dtype=np.int64, copy=True),
        )
        cat = {name: np.zeros(size, dtype=np.int64) for name in encoder.vocab_sizes}
        # Keep torch's per-scalar float32 overflow checks, not NumPy's inf cast.
        loc = torch.zeros(size, dtype=torch.float32)
        scale = torch.ones(size, dtype=torch.float32)
        for row, (i, origin) in enumerate(items):
            i, origin = int(i), int(origin)
            series = panel.series[i]
            start = max(0, origin - max_context)
            context_length = origin - start
            dest = slice(length - context_length, length)
            context = np.asarray(series.target[start:origin], dtype=np.float64)
            frame = sample_frame(context, series.target[:origin], panel.seasonal_period)
            observed = ~np.isnan(context)
            arrays["target"][row, dest] = np.where(
                observed, (context - frame.loc) / frame.scale, 0
            ).astype(np.float32)
            arrays["observed"][row, dest] = observed
            arrays["padding"][row, dest] = True
            loc[row], scale[row] = frame.loc, frame.scale
            arrays["eligible"][row], arrays["has_context"][row] = frame.eligible, frame.has_context

            key = (series.start, str(series.start.tz))
            grid, full_calendar = self._grid(key)
            calendar = full_calendar[start : origin + horizon]
            arrays["enc_cov"][row, dest, :n_calendar] = calendar[:context_length]
            arrays["dec_cov"][row, :, :n_calendar] = calendar[context_length:]
            arrays["enc_cov_mask"][row, dest, :n_calendar] = True
            arrays["dec_cov_mask"][row, :, :n_calendar] = True
            for column, name in enumerate(dynamic, start=n_calendar):
                is_known = column < n_dec
                stop = origin + horizon if is_known else origin
                values, mask = _dynamic_values(series.dynamic.get(name), start, origin, stop)
                arrays["enc_cov"][row, dest, column] = values[:context_length]
                arrays["enc_cov_mask"][row, dest, column] = mask[:context_length]
                if is_known:
                    arrays["dec_cov"][row, :, column] = values[context_length:]
                    arrays["dec_cov_mask"][row, :, column] = mask[context_length:]

            label_end = min(origin + horizon, series.history_length)
            labels = np.asarray(series.target[origin:label_end], dtype=np.float64)
            label_mask = ~np.isnan(labels)
            if label_upto is not None:
                cutoff = self._label_limit(key, grid, label_upto)
                label_mask &= np.arange(origin, label_end) < cutoff
            arrays["labels"][row, : len(labels)] = np.where(
                label_mask, (labels - frame.loc) / frame.scale, 0
            ).astype(np.float32)
            arrays["label_mask"][row, : len(labels)] = label_mask
            if i not in self._statics:
                self._statics[i] = encoder.transform(series)
            codes, num, num_mask = self._statics[i]
            for name, code in codes.items():
                cat[name][row] = code
            arrays["num"][row] = num
            arrays["num_mask"][row] = num_mask
        arrays["loss_eligible"] = arrays["eligible"] & arrays["label_mask"].any(axis=1)
        return Batch(
            **{name: torch.from_numpy(values) for name, values in arrays.items()},
            cat={name: torch.from_numpy(values) for name, values in cat.items()},
            loc=loc,
            scale=scale,
        )


def make_batch(
    panel: Panel,
    items: Iterable[tuple[int, int]],
    horizon: int,
    max_context: int,
    encoder: StaticEncoder,
    label_upto: pd.Timestamp | None = None,
) -> Batch:
    """Build one batch; retain a BatchBuilder to reuse features across calls."""
    return BatchBuilder(panel, horizon, max_context, encoder).build(items, label_upto)


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
        self.builder = BatchBuilder(panel, horizon, max_context, encoder)
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
                yield self.builder.build(items, self.label_upto)
                items = []
        if items:
            yield self.builder.build(items, self.label_upto)

    def state(self) -> dict:
        return {"epoch": self.epoch, "position": self.position}

    def load_state(self, d: dict) -> None:
        """Reconstruct the epoch's draw list and resume at its next position."""
        position = d["position"]
        if not isinstance(position, (int, np.integer)) or not 0 <= position <= len(self):
            raise ValueError("sampler position must be between zero and the epoch length")
        self.set_epoch(d["epoch"])
        self.position = int(position)
