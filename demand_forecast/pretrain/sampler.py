"""Deterministic, resumable corpus streams with bounded shard prefetch."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, fields
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import Any
from weakref import finalize

import numpy as np
import torch

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.panel import Series
from demand_forecast.data.sampler import Batch, BatchBuilder, _positive
from demand_forecast.pretrain import corpus
from demand_forecast.pretrain.config import FEV_HORIZONS
from demand_forecast.pretrain.corpus import ShardPanelGroup, ShardRecords, ShardRef

Kind = str
KIND_CODE = {"tsmixup": 0, "kernel_synth": 1, "chronos_real": 2, "gifteval_pretrain": 3}


@dataclass
class StreamState:
    """Position of the next batch within the current pass and shuffled shard.

    ``batch_cursor`` counts batches already yielded from that shard. The RNG
    state is captured after drawing its origins and series permutation.
    """

    pass_index: int
    shard_cursor: int
    batch_cursor: int
    rng_state: dict[str, Any]


@dataclass
class V3StreamState(StreamState):
    """V3 next pair slot, resolved visit budget and empty-pass detection."""

    visit_cursor: int
    visit_quota: int
    pass_has_batches: bool


@dataclass
class CorpusSamplerState:
    """Next mixing step and independent positions of the two corpus streams."""

    step: int
    tsmixup: StreamState
    kernel_synth: StreamState

    def stream(self, kind: str) -> StreamState:
        return getattr(self, kind)


@dataclass
class WeightedSamplerState:
    """Independent stream positions and exact integer scheduling credits."""

    step: int
    streams: dict[str, StreamState]
    credits: dict[str, int]
    weights: dict[str, int]
    last_kind: str

    def stream(self, kind: str) -> StreamState:
        return self.streams[kind]


def sampler_state_from_dict(value: dict) -> CorpusSamplerState | WeightedSamplerState:
    """Read legacy checkpoints as well as weighted checkpoints."""

    def stream(v):
        return (V3StreamState if "visit_cursor" in v else StreamState)(**v)

    if "streams" in value:
        return WeightedSamplerState(
            **(value | {"streams": {k: stream(v) for k, v in value["streams"].items()}})
        )
    return CorpusSamplerState(
        value["step"], stream(value["tsmixup"]), stream(value["kernel_synth"])
    )


class ShardStream(Iterator[Batch]):
    """Visit each eligible series once per pass, in independently seeded shards.

    Eligibility here means length > min_context, not loss eligibility. Origins
    are drawn in file order, then permuted and divided into batches, retaining
    the tail. Series indices continue to address the original shard rows.

    Only the current panel and one prefetched shard are retained. Prefetch does
    no sampling and cannot advance checkpoint state. ``shard_loads`` counts
    successful reads, including speculative reads and reads needed on resume;
    it is an execution diagnostic, not checkpointed sampling state. Call
    ``close()`` when done; finalization also shuts down an abandoned executor.
    """

    def __init__(
        self,
        refs: Sequence[ShardRef],
        kind: Kind,
        *,
        horizon: int,
        max_context: int,
        min_context: int,
        batch_size: int,
        seed: int,
        encoder: StaticEncoder,
        prefetch: int = 1,
    ) -> None:
        for name, value in (
            ("horizon", horizon),
            ("max_context", max_context),
            ("min_context", min_context),
            ("batch_size", batch_size),
            ("seed + 1", seed + 1),
        ):
            _positive(value, name)
        if kind not in KIND_CODE or any(ref.kind != kind for ref in refs):
            raise ValueError("shard refs must match the stream kind")
        if len({ref.index for ref in refs}) != len(refs):
            raise ValueError("overlapping shard refs within a stream")
        for ref in refs:
            _positive(ref.index + 1, "shard index + 1")
        if prefetch not in (0, 1):
            raise ValueError("prefetch must be 0 or 1")
        self.refs, self.kind = tuple(refs), kind
        self.horizon, self.max_context = horizon, max_context
        self.min_context, self.batch_size = min_context, batch_size
        self.seed, self.encoder = int(seed), encoder
        self.pass_index = self.shard_cursor = self.batch_cursor = 0
        self.shard_loads = 0
        self._order = self._shard_order(0)
        self._rng = np.random.default_rng((self.seed, KIND_CODE[kind], 0))
        self._builder: BatchBuilder | None = None
        self._draws = np.empty((0, 2), dtype=np.int64)
        self._pass_has_batches = False
        self._closed = False
        self._future: Future[list[Series]] | None = None
        self._executor = (
            ThreadPoolExecutor(1, thread_name_prefix=f"corpus-{kind}") if prefetch else None
        )
        self._finalizer = (
            finalize(self, self._executor.shutdown, wait=False, cancel_futures=True)
            if self._executor is not None
            else None
        )

    def _shard_order(self, pass_index: int) -> np.ndarray:
        return np.random.default_rng((self.seed, KIND_CODE[self.kind], pass_index)).permutation(
            len(self.refs)
        )

    def _read(self, path: Path) -> list[Series]:
        series = corpus.read_shard(path)
        self.shard_loads += 1
        return series

    def _prefetch_next(self) -> None:
        if self._executor is None:
            return
        cursor = self.shard_cursor + 1
        order = self._order
        if cursor == len(order):
            # An entirely empty pass terminates, so do not read another pass.
            if not len(self._draws) and not self._pass_has_batches:
                return
            cursor, order = 0, self._shard_order(self.pass_index + 1)
        ref = self.refs[int(order[cursor])]
        self._future = self._executor.submit(self._read, ref.path)

    def _load_shard(self) -> None:
        ref = self.refs[int(self._order[self.shard_cursor])]
        if self._future is None:
            series = self._read(ref.path)
        else:
            future, self._future = self._future, None
            series = future.result()
        panel = corpus.shard_panel(series)
        self._builder = BatchBuilder(
            panel, self.horizon, self.max_context, self.encoder, calendar=False
        )
        lengths = np.array([item.length for item in series], dtype=np.int64)
        indices = np.flatnonzero(lengths > self.min_context)
        self._rng = np.random.default_rng(
            (self.seed, KIND_CODE[self.kind], self.pass_index, ref.index)
        )
        origins = self._rng.integers(self.min_context, lengths[indices])
        draws = np.column_stack((indices, origins))
        self._draws = draws[self._rng.permutation(len(draws))]
        self._prefetch_next()

    def __iter__(self) -> ShardStream:
        return self

    def __next__(self) -> Batch:
        if self._closed:
            raise StopIteration
        try:
            if not self.refs:
                raise ValueError("corpus has no eligible series")
            while True:
                if self._builder is None:
                    self._load_shard()
                start = self.batch_cursor * self.batch_size
                if start < len(self._draws):
                    batch = self._builder.build(self._draws[start : start + self.batch_size])
                    # Advance before returning: a checkpoint points to the next batch.
                    self.batch_cursor += 1
                    self._pass_has_batches = True
                    return batch
                self._builder = None
                self._draws = np.empty((0, 2), dtype=np.int64)
                self.batch_cursor = 0
                self.shard_cursor += 1
                if self.shard_cursor == len(self.refs):
                    if not self._pass_has_batches:
                        raise ValueError("corpus has no eligible series")
                    self.pass_index += 1
                    self.shard_cursor = 0
                    self._order = self._shard_order(self.pass_index)
                    self._pass_has_batches = False
        except Exception:
            self.close()
            raise

    def state(self) -> StreamState:
        """Return a detached snapshot; no background read can mutate it."""
        return StreamState(
            self.pass_index,
            self.shard_cursor,
            self.batch_cursor,
            deepcopy(self._rng.bit_generator.state),
        )

    def _discard_prefetch(self) -> None:
        future, self._future = self._future, None
        if future is not None and not future.cancel():
            try:
                future.result()
            except Exception:
                # This read belongs to the discarded timeline. Its error must
                # surface only if the restored stream actually reaches it.
                pass

    def load_state(self, state: StreamState) -> None:
        """Restore with the same refs, seed and batching parameters.

        Recreate only the active shard's draws, without replaying prior shards.
        Previously prefetched data belongs to the old timeline and is discarded.
        """
        if self._closed:
            raise RuntimeError("cannot restore a closed shard stream")
        for name in ("pass_index", "shard_cursor", "batch_cursor"):
            _positive(getattr(state, name) + 1, name + " + 1")
        if state.shard_cursor >= max(1, len(self.refs)):
            raise ValueError("shard_cursor is outside the corpus")
        rng = np.random.default_rng((self.seed, KIND_CODE[self.kind], state.pass_index))
        rng.bit_generator.state = deepcopy(state.rng_state)
        self._discard_prefetch()
        self._builder = None
        self._draws = np.empty((0, 2), dtype=np.int64)
        self.pass_index, self.shard_cursor = state.pass_index, state.shard_cursor
        self.batch_cursor = state.batch_cursor
        self._order = self._shard_order(self.pass_index)
        self._pass_has_batches = self.batch_cursor > 0
        try:
            if self.batch_cursor:
                if not self.refs:
                    raise ValueError("batch_cursor is outside the shard")
                self._load_shard()
                if self.batch_cursor > (len(self._draws) + self.batch_size - 1) // self.batch_size:
                    raise ValueError("batch_cursor is outside the shard")
            self._rng = rng
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Finish any running read, release its data and stop the owned worker."""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._finalizer.detach()
        self._future = None
        self._builder = None
        self._draws = np.empty((0, 2), dtype=np.int64)


class V3BatchBuilder:
    """Build homogeneous grids, then pad and scatter back to requested row order.

    Items address original shard rows, even when a group has noncontiguous rows.
    All groups share the canonical input layout; no Batch layout change is needed.
    """

    def __init__(
        self,
        groups: Sequence[ShardPanelGroup],
        horizon,
        max_context,
        encoder,
        *,
        scale_floor_frac=0.0,
    ):
        self.builders = [
            (
                group,
                BatchBuilder(
                    group.panel,
                    horizon,
                    max_context,
                    encoder,
                    calendar=group.calendar_available,
                    calendar_schema="canonical",
                    target_lags="cycle",
                    scale_floor_frac=scale_floor_frac,
                ),
            )
            for group in groups
        ]
        self._routes = {}
        self._row_indices = [torch.tensor(group.row_indices) for group, _ in self.builders]
        for group_index, (group, _) in enumerate(self.builders):
            for local_index, row_index in enumerate(group.row_indices):
                if row_index in self._routes:
                    raise ValueError("v3 rows must address exactly one panel group")
                self._routes[row_index] = (group_index, local_index)

    def build(self, items, *, horizon: int | None = None) -> Batch:
        items = np.asarray(items, dtype=np.int64).reshape(-1, 2)
        chunks, positions = [], []
        selected_groups = {}
        for row, (index, origin) in enumerate(items):
            if int(index) not in self._routes:
                raise ValueError("v3 items must address exactly one panel group")
            group_index, local_index = self._routes[int(index)]
            selected, pairs = selected_groups.setdefault(group_index, ([], []))
            selected.append(row)
            pairs.append((local_index, int(origin)))
        for group_index, (selected, pairs) in selected_groups.items():
            group, builder = self.builders[group_index]
            chunk = builder.build(pairs)
            if horizon is not None:
                if not 1 <= horizon <= builder.horizon:
                    raise ValueError("sampled horizon must fit the builder horizon")
                # The decoder takes its length from these tensors, not the config.
                # Keep one grid/builder per group, rather than one per horizon.
                for name in ("dec_cov", "dec_cov_mask", "labels", "label_mask"):
                    setattr(chunk, name, getattr(chunk, name)[:, :horizon])
                chunk.loss_eligible = chunk.eligible & chunk.label_mask.any(dim=1)
            chunk.series_index = self._row_indices[group_index][chunk.series_index]
            chunks.append(chunk)
            positions.extend(selected)
        if not chunks:
            raise ValueError("v3 batch requires at least one item")
        length = max(chunk.target.shape[1] for chunk in chunks)
        order = torch.tensor(np.argsort(positions))
        context_fields = {"target", "observed", "padding", "enc_cov", "enc_cov_mask"}

        def combine(name, values):
            if name in context_fields:
                padded = []
                for value in values:
                    buffer = value.new_zeros((value.shape[0], length, *value.shape[2:]))
                    if value.shape[1]:
                        buffer[:, -value.shape[1] :] = value
                    padded.append(buffer)
                values = padded
            return torch.cat(values)[order]

        return Batch(
            **{
                field.name: (
                    {
                        name: combine(name, [chunk.cat[name] for chunk in chunks])
                        for name in chunks[0].cat
                    }
                    if field.name == "cat"
                    else combine(field.name, [getattr(chunk, field.name) for chunk in chunks])
                )
                for field in fields(Batch)
            }
        )


class V3ShardStream(ShardStream):
    """Sample full windows within a group, with a resumable pair-slot quota.

    Groups are selected proportional to their usable-series counts. Conditional
    on that group, horizons are uniform over its feasible set. Each pair draws
    a feasible series uniformly, then an eligible origin uniformly within that
    series, with replacement. Default visits draw as many pairs as there are
    usable series, retaining a partial batch.
    """

    def __init__(
        self,
        *args,
        period_policy,
        grid_policy,
        scale_floor_frac,
        horizon_policy="fixed",
        shard_visit_quota=None,
        **kwargs,
    ):
        self.period_policy = period_policy
        self.grid_policy = grid_policy
        self.scale_floor_frac = scale_floor_frac
        if horizon_policy not in ("fixed", "fev_set"):
            raise ValueError("unknown horizon_policy")
        if shard_visit_quota is not None:
            _positive(shard_visit_quota, "shard_visit_quota")
        self.horizon_policy = horizon_policy
        self.configured_quota = shard_visit_quota
        self.visit_cursor = self._visit_quota = 0
        self._groups = []
        self.last_group = ""
        super().__init__(*args, **kwargs)
        if horizon_policy == "fev_set" and self.horizon != max(FEV_HORIZONS):
            self.close()
            raise ValueError("fev_set requires horizon = 288")

    def _read(self, path: Path) -> ShardRecords:
        records = corpus.read_shard_v3(path, kind=self.kind, grid_policy=self.grid_policy)
        self.shard_loads += 1
        return records

    def _load_shard(self) -> None:
        ref = self.refs[int(self._order[self.shard_cursor])]
        if self._future is None:
            loaded = self._read(ref.path)
        else:
            future, self._future = self._future, None
            loaded = future.result()
        groups = corpus.shard_panels(loaded.records, period_policy=self.period_policy)
        self._builder = V3BatchBuilder(
            groups,
            self.horizon,
            self.max_context,
            self.encoder,
            scale_floor_frac=self.scale_floor_frac,
        )
        self._groups = []
        horizons = FEV_HORIZONS if self.horizon_policy == "fev_set" else (self.horizon,)
        for group in groups:
            minimum = max(self.min_context, group.panel.seasonal_period + 1)
            lengths = np.array([s.history_length for s in group.panel.series], dtype=np.int64)
            feasible = {}
            for horizon in horizons:
                counts = np.maximum(0, lengths - horizon - minimum + 1)
                indices = np.flatnonzero(counts)
                if len(indices):
                    feasible[horizon] = (indices, counts[indices])
            if feasible:
                usable = len(next(iter(feasible.values()))[0])
                key = (
                    f"{group.panel.freq.freqstr}|m={group.panel.seasonal_period}"
                    f"|calendar={int(group.calendar_available)}"
                )
                self._groups.append((group, minimum, feasible, usable, key))
        usable = sum(g[3] for g in self._groups)
        self._visit_quota = (self.configured_quota or usable) if usable else 0
        self._rng = np.random.default_rng(
            (self.seed, KIND_CODE[self.kind], self.pass_index, ref.index)
        )
        self._prefetch_next()

    def _prefetch_next(self) -> None:
        if self._executor is None:
            return
        cursor, order = self.shard_cursor + 1, self._order
        if cursor == len(order):
            if not self._visit_quota and not self._pass_has_batches:
                return
            cursor, order = 0, self._shard_order(self.pass_index + 1)
        self._future = self._executor.submit(self._read, self.refs[int(order[cursor])].path)

    def __next__(self) -> Batch:
        if self._closed:
            raise StopIteration
        try:
            if not self.refs:
                raise ValueError(f"{self.kind} source has no usable windows")
            while True:
                if self._builder is None:
                    self._load_shard()
                if self.visit_cursor < self._visit_quota:
                    weights = np.array([g[3] for g in self._groups], dtype=np.float64)
                    group, minimum, feasible, _, key = self._groups[
                        int(self._rng.choice(len(weights), p=weights / weights.sum()))
                    ]
                    horizon = int(self._rng.choice(tuple(feasible)))
                    indices, counts = feasible[horizon]
                    size = min(self.batch_size, self._visit_quota - self.visit_cursor)
                    rows = self._rng.integers(len(indices), size=size)
                    origins = minimum + self._rng.integers(counts[rows])
                    original_rows = np.asarray(group.row_indices)[indices[rows]]
                    batch = self._builder.build(
                        np.column_stack((original_rows, origins)), horizon=horizon
                    )
                    self.last_group = key
                    self.visit_cursor += size
                    self.batch_cursor += 1
                    self._pass_has_batches = True
                    return batch
                self._builder = None
                self._groups = []
                self.visit_cursor = self._visit_quota = self.batch_cursor = 0
                self.shard_cursor += 1
                if self.shard_cursor == len(self.refs):
                    if not self._pass_has_batches:
                        raise ValueError(f"{self.kind} source has no usable windows")
                    self.pass_index += 1
                    self.shard_cursor = 0
                    self._order = self._shard_order(self.pass_index)
                    self._pass_has_batches = False
        except Exception:
            self.close()
            raise

    def state(self) -> V3StreamState:
        return V3StreamState(
            self.pass_index,
            self.shard_cursor,
            self.batch_cursor,
            deepcopy(self._rng.bit_generator.state),
            self.visit_cursor,
            self._visit_quota,
            self._pass_has_batches,
        )

    def load_state(self, state: StreamState) -> None:
        if self._closed:
            raise RuntimeError("cannot restore a closed shard stream")
        if not isinstance(state, V3StreamState):
            raise ValueError("v3 sampler requires v3 window-policy state")
        for name in ("pass_index", "shard_cursor", "batch_cursor", "visit_cursor", "visit_quota"):
            _positive(getattr(state, name) + 1, name + " + 1")
        if state.shard_cursor >= max(1, len(self.refs)):
            raise ValueError("shard_cursor is outside the corpus")
        self._discard_prefetch()
        self._builder = None
        self._groups = []
        self.pass_index, self.shard_cursor = state.pass_index, state.shard_cursor
        self.batch_cursor, self.visit_cursor = state.batch_cursor, state.visit_cursor
        self._order = self._shard_order(self.pass_index)
        self._pass_has_batches = state.pass_has_batches
        self._visit_quota = 0
        try:
            if self.batch_cursor:
                self._load_shard()
                if (
                    self._visit_quota != state.visit_quota
                    or not 0 < self.visit_cursor <= self._visit_quota
                    or self.batch_cursor
                    > (self._visit_quota + self.batch_size - 1) // self.batch_size
                    or not self._pass_has_batches
                    or self.visit_cursor
                    != min(self.batch_cursor * self.batch_size, self._visit_quota)
                ):
                    raise ValueError("v3 visit cursor/quota is outside the shard")
            elif state.visit_cursor or state.visit_quota:
                raise ValueError("unstarted v3 visit must have zero cursor/quota")
            self._rng.bit_generator.state = deepcopy(state.rng_state)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        super().close()
        self._groups = []


class CorpusSampler:
    """Mix independent shard streams on an exact, zero-based batch schedule.

    ``kernel_synth_every=0`` disables the auxiliary stream; 1 selects it every
    time. Public ``tsmixup`` and ``kernel_synth`` streams expose their load
    counters. Call ``close()`` when training finishes or is interrupted.

    With ``sources`` and ``weights``, use smooth weighted round-robin: add
    each weight to its credit, select the largest credit (ties by KIND_CODE),
    and subtract the sum of weights from that kind. Decimal weights are
    converted to exact integers; checkpoints store credits and every stream.
    The legacy constructor retains its original schedule and RNG seeds.
    """

    def __init__(
        self,
        tsmixup: Sequence[ShardRef] = (),
        kernel_synth: Sequence[ShardRef] = (),
        *,
        horizon: int,
        max_context: int,
        min_context: int,
        batch_size: int,
        seed: int,
        kernel_synth_every: int = 10,
        holdout: Sequence[ShardRef] = (),
        sources: dict[str, Sequence[ShardRef]] | None = None,
        weights: dict[str, float] | None = None,
        recipe: str = "v2",
        period_policy: str = "legacy_one",
        grid_policy: str = "legacy_weekly",
        horizon_policy: str = "fixed",
        shard_visit_quota: dict[str, int] | None = None,
        scale_floor_frac: float = 0.0,
    ) -> None:
        _positive(kernel_synth_every + 1, "kernel_synth_every + 1")
        primary = {(ref.kind, ref.index) for ref in tsmixup}
        auxiliary = {(ref.kind, ref.index) for ref in kernel_synth}
        held_out = {(ref.kind, ref.index) for ref in holdout}
        if primary & auxiliary or (primary | auxiliary) & held_out:
            raise ValueError("overlapping corpus refs between streams or holdout")
        self.weights = None
        if sources is not None:
            if tsmixup or kernel_synth:
                raise ValueError("use sources or legacy positional streams, not both")
            if not sources or weights is None or sources.keys() != weights.keys():
                raise ValueError("sources and weights must name the same nonempty set of kinds")
            if set(sources) - KIND_CODE.keys() or any(not refs for refs in sources.values()):
                raise ValueError("weighted sources require nonempty registered kinds")
            if any((ref.kind, ref.index) in held_out for refs in sources.values() for ref in refs):
                raise ValueError("overlapping corpus refs between streams or holdout")
            fractions = {k: Fraction(str(v)) for k, v in weights.items()}
            if any(v <= 0 for v in fractions.values()):
                raise ValueError("weights must be finite and positive")
            denominator = lcm(*(v.denominator for v in fractions.values()))
            self.weights = {
                k: int(fractions[k] * denominator) for k in sorted(sources, key=KIND_CODE.get)
            }
            self.credits = dict.fromkeys(self.weights, 0)
        elif weights is not None:
            raise ValueError("weights require sources")
        # Corpus roles contain no statics; fitting an empty panel avoids an
        # eager shard read and leaves every series in the unknown-id bucket.
        self.encoder = StaticEncoder().fit(corpus.shard_panel([]), [])
        kwargs = dict(
            horizon=horizon,
            max_context=max_context,
            min_context=min_context,
            batch_size=batch_size,
            seed=seed,
            encoder=self.encoder,
        )
        refs_by_kind = (
            sources if sources is not None else {"tsmixup": tsmixup, "kernel_synth": kernel_synth}
        )
        self.streams = {}
        stream_type = ShardStream
        if recipe == "v3":
            stream_type = V3ShardStream
            kwargs.update(
                period_policy=period_policy,
                grid_policy=grid_policy,
                scale_floor_frac=scale_floor_frac,
                horizon_policy=horizon_policy,
            )
        elif recipe != "v2":
            raise ValueError("recipe must be v2 or v3")
        try:
            for kind, refs in refs_by_kind.items():
                extra = (
                    dict(shard_visit_quota=(shard_visit_quota or {}).get(kind))
                    if recipe == "v3"
                    else {}
                )
                self.streams[kind] = stream_type(refs, kind, **kwargs, **extra)
        except BaseException:
            for stream in self.streams.values():
                stream.close()
            raise
        self.tsmixup = self.streams.get("tsmixup")
        self.kernel_synth = self.streams.get("kernel_synth")
        self.kernel_synth_every = kernel_synth_every
        self.step = 0
        self.last_kind = ""
        self._closed = False

    def _kind(self, step: int) -> Kind:
        every = self.kernel_synth_every
        return "kernel_synth" if every and step % every == every - 1 else "tsmixup"

    def batches(self) -> Iterator[Batch]:
        """Yield batches, advancing the mixing step before exposing each one."""
        while not self._closed:
            if self.weights is None:
                kind = self._kind(self.step)
            else:
                for kind, weight in self.weights.items():
                    self.credits[kind] += weight
                kind = max(self.weights, key=self.credits.get)
                self.credits[kind] -= sum(self.weights.values())
            try:
                batch = next(self.streams[kind])
            except Exception:
                self.close()
                raise
            self.last_kind = kind
            self.step += 1
            yield batch

    def state(self) -> CorpusSamplerState | WeightedSamplerState:
        """Snapshot both streams, including an auxiliary stream not yet consumed."""
        if self.weights is not None:
            return WeightedSamplerState(
                self.step,
                {k: v.state() for k, v in self.streams.items()},
                self.credits.copy(),
                self.weights.copy(),
                self.last_kind,
            )
        return CorpusSamplerState(self.step, self.tsmixup.state(), self.kernel_synth.state())

    def load_state(self, state: CorpusSamplerState | WeightedSamplerState) -> None:
        """Restore both streams and the schedule from a detached checkpoint."""
        _positive(state.step + 1, "step + 1")
        if self.weights is not None:
            if (
                not isinstance(state, WeightedSamplerState)
                or state.weights != self.weights
                or state.streams.keys() != self.streams.keys()
                or state.credits.keys() != self.weights.keys()
            ):
                raise ValueError("weighted checkpoint does not match the sampler")
            for kind, stream in self.streams.items():
                stream.load_state(state.streams[kind])
            self.step, self.last_kind = state.step, state.last_kind
            self.credits = state.credits.copy()
            return
        if not isinstance(state, CorpusSamplerState):
            raise ValueError("legacy sampler requires a legacy checkpoint")
        self.tsmixup.load_state(state.tsmixup)
        self.kernel_synth.load_state(state.kernel_synth)
        self.step = state.step
        self.last_kind = self._kind(self.step - 1) if self.step else ""

    def close(self) -> None:
        """Stop both streams and release their current and prefetched shards."""
        self._closed = True
        for stream in self.streams.values():
            stream.close()
