"""Deterministic, resumable corpus streams with bounded shard prefetch."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from weakref import finalize

import numpy as np

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.panel import Series
from demand_forecast.data.sampler import Batch, BatchBuilder, _positive
from demand_forecast.pretrain import corpus
from demand_forecast.pretrain.corpus import ShardRef

Kind = Literal["tsmixup", "kernel_synth"]
KIND_CODE = {"tsmixup": 0, "kernel_synth": 1}


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
class CorpusSamplerState:
    """Next mixing step and independent positions of the two corpus streams."""

    step: int
    tsmixup: StreamState
    kernel_synth: StreamState


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


class CorpusSampler:
    """Mix independent shard streams on an exact, zero-based batch schedule.

    ``kernel_synth_every=0`` disables the auxiliary stream; 1 selects it every
    time. Public ``tsmixup`` and ``kernel_synth`` streams expose their load
    counters. Call ``close()`` when training finishes or is interrupted.
    """

    def __init__(
        self,
        tsmixup: Sequence[ShardRef],
        kernel_synth: Sequence[ShardRef],
        *,
        horizon: int,
        max_context: int,
        min_context: int,
        batch_size: int,
        seed: int,
        kernel_synth_every: int = 10,
        holdout: Sequence[ShardRef] = (),
    ) -> None:
        _positive(kernel_synth_every + 1, "kernel_synth_every + 1")
        primary = {(ref.kind, ref.index) for ref in tsmixup}
        auxiliary = {(ref.kind, ref.index) for ref in kernel_synth}
        held_out = {(ref.kind, ref.index) for ref in holdout}
        if primary & auxiliary or (primary | auxiliary) & held_out:
            raise ValueError("overlapping corpus refs between streams or holdout")
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
        self.tsmixup = ShardStream(tsmixup, "tsmixup", **kwargs)
        self.kernel_synth = ShardStream(kernel_synth, "kernel_synth", **kwargs)
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
            kind = self._kind(self.step)
            try:
                batch = next(self.kernel_synth if kind == "kernel_synth" else self.tsmixup)
            except Exception:
                self.close()
                raise
            self.last_kind = kind
            self.step += 1
            yield batch

    def state(self) -> CorpusSamplerState:
        """Snapshot both streams, including an auxiliary stream not yet consumed."""
        return CorpusSamplerState(self.step, self.tsmixup.state(), self.kernel_synth.state())

    def load_state(self, state: CorpusSamplerState) -> None:
        """Restore both streams and the schedule from a detached checkpoint."""
        _positive(state.step + 1, "step + 1")
        self.tsmixup.load_state(state.tsmixup)
        self.kernel_synth.load_state(state.kernel_synth)
        self.step = state.step
        self.last_kind = self._kind(self.step - 1) if self.step else ""

    def close(self) -> None:
        """Stop both streams and release their current and prefetched shards."""
        self._closed = True
        self.tsmixup.close()
        self.kernel_synth.close()
