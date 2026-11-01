"""Corpus mixing, exact replay, and finite batches from degenerate shards."""

from __future__ import annotations

from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import fields, replace
from itertools import islice
from pathlib import Path
from threading import current_thread, main_thread

import pytest
import torch

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench

from demand_forecast.data.sampler import Batch  # noqa: E402
from demand_forecast.pretrain import corpus  # noqa: E402
from demand_forecast.pretrain.corpus import ShardRef, corpus_manifest  # noqa: E402
from demand_forecast.pretrain.sampler import CorpusSampler, ShardStream  # noqa: E402
from tests.helpers.corpus_fixture import write_corpus_fixture  # noqa: E402
from tests.helpers.reference_batch import assert_batches_equal  # noqa: E402

SAMPLER_KWARGS = dict(horizon=4, max_context=16, min_context=8, batch_size=4, seed=0)


@pytest.fixture
def two_kind_fixture(tmp_path: Path) -> dict[str, list[ShardRef]]:
    """Two distinguishable shards per kind, in manifest order."""
    for kind in ("tsmixup", "kernel_synth"):
        for index in (0, 1):
            write_corpus_fixture(tmp_path, kind=kind, index=index, seed=index)
    refs = corpus_manifest(tmp_path)
    return {kind: [ref for ref in refs if ref.kind == kind] for kind in ("tsmixup", "kernel_synth")}


@pytest.fixture
def constant_shard_fixture(tmp_path: Path) -> dict[str, list[ShardRef]]:
    """A constant primary stream and a normal auxiliary stream."""
    write_corpus_fixture(tmp_path, constant=True)
    write_corpus_fixture(tmp_path, kind="kernel_synth")
    refs = corpus_manifest(tmp_path)
    return {kind: [ref for ref in refs if ref.kind == kind] for kind in ("tsmixup", "kernel_synth")}


@pytest.mark.parametrize("every,expected", [(10, [9, 19, 29]), (0, []), (1, list(range(30)))])
def test_schedule_is_exactly_one_in_ten(two_kind_fixture, every, expected):
    with closing(
        CorpusSampler(**two_kind_fixture, **SAMPLER_KWARGS, kernel_synth_every=every)
    ) as sampler:
        batches = sampler.batches()
        kinds = []
        for _ in range(30):
            next(batches)
            kinds.append(sampler.last_kind)
        assert [i for i, kind in enumerate(kinds) if kind == "kernel_synth"] == expected
        assert sampler.state().step == 30


@pytest.mark.parametrize("saved_after", [0, 1, 3, 6, 12])
def test_deterministic_and_resumable_through_a_second_kernel_synth_batch(
    two_kind_fixture, saved_after, monkeypatch
):
    workers = set()
    read_shard = corpus.read_shard

    def observed_read(path):
        worker = current_thread()
        if worker is not main_thread():
            workers.add(worker)
        return read_shard(path)

    monkeypatch.setattr(corpus, "read_shard", observed_read)
    with ExitStack() as stack:
        reference, original, resumed = [
            stack.enter_context(closing(CorpusSampler(**two_kind_fixture, **SAMPLER_KWARGS)))
            for _ in range(3)
        ]
        expected = list(islice(reference.batches(), 20))
        iterator = original.batches()
        first = [next(iterator) for _ in range(saved_after)]
        snapshot = original.state()
        frozen = deepcopy(snapshot)
        # Advancing the source must not mutate a saved checkpoint.
        next(iterator)
        assert snapshot == frozen
        # Loading over an already active iterator must discard its stale prefetch.
        resumed_iterator = resumed.batches()
        next(resumed_iterator)
        resumed.load_state(snapshot)
        rest = list(islice(resumed_iterator, 20 - saved_after))
        for actual, wanted in zip(first + rest, expected, strict=True):
            assert_batches_equal(actual, wanted)
        assert resumed.last_kind == "kernel_synth"
        assert resumed.state() == reference.state()
        assert snapshot == frozen

        # Prefetch cannot affect draws, including partial batches and pass rollover.
        streams = [
            stack.enter_context(
                closing(
                    ShardStream(
                        two_kind_fixture["tsmixup"],
                        "tsmixup",
                        **(SAMPLER_KWARGS | {"batch_size": 5}),
                        encoder=original.encoder,
                        prefetch=prefetch,
                    )
                )
            )
            for prefetch in (0, 1)
        ]
        for _ in range(4):
            indices = []
            for size in (5, 5, 2):
                synchronous, prefetched = [next(stream) for stream in streams]
                assert_batches_equal(prefetched, synchronous)
                assert synchronous.size == size
                indices.extend(synchronous.series_index.tolist())
                assert (synchronous.origin_pos >= 8).all()
                assert (synchronous.origin_pos < 90).all()
            assert sorted(indices) == list(range(12))
        assert streams[0].shard_loads == 4
        assert 4 <= streams[1].shard_loads <= 5
    assert workers  # Reads actually ran in the background.
    assert all(not worker.is_alive() for worker in workers)


def test_tolerates_ineligible_shard(constant_shard_fixture, tmp_path):
    # All-NaN targets also have sufficient length; neither kind of ineligibility
    # may suppress batches or turn an infinite stream into an empty-pass error.
    for all_nan in (False, True):
        if all_nan:
            write_corpus_fixture(tmp_path, all_nan=True)
            refs = corpus_manifest(tmp_path)
            constant_shard_fixture = {
                kind: [ref for ref in refs if ref.kind == kind]
                for kind in ("tsmixup", "kernel_synth")
            }
        with closing(CorpusSampler(**constant_shard_fixture, **SAMPLER_KWARGS)) as sampler:
            iterator = sampler.batches()
            for i in range(20):
                batch = next(iterator)
                if i in (9, 19):
                    assert sampler.last_kind == "kernel_synth"
                    assert batch.loss_eligible.any()
                else:
                    assert not batch.loss_eligible.any()
                for field in fields(Batch):
                    value = getattr(batch, field.name)
                    tensors = value.values() if isinstance(value, dict) else (value,)
                    assert all(torch.isfinite(tensor).all() for tensor in tensors), field.name


def test_rejects_overlapping_refs_and_empty_corpus(two_kind_fixture, tmp_path):
    primary = two_kind_fixture["tsmixup"]
    auxiliary = two_kind_fixture["kernel_synth"]
    moved = replace(primary[0], path=tmp_path / "relocated.parquet")
    for kwargs in (
        dict(tsmixup=primary, kernel_synth=[moved]),
        two_kind_fixture | {"holdout": [moved]},
        two_kind_fixture | {"holdout": [auxiliary[0]]},
    ):
        with pytest.raises(ValueError, match="overlap"):
            CorpusSampler(**kwargs, **SAMPLER_KWARGS)

    with pytest.raises(ValueError, match="corpus has no eligible series"):
        with closing(CorpusSampler([], [], **SAMPLER_KWARGS)) as sampler:
            next(sampler.batches())

    short_root = tmp_path / "short"
    write_corpus_fixture(short_root, lengths=(8, 8), index=0)
    write_corpus_fixture(short_root, n_series=0, index=1)
    short_refs = corpus_manifest(short_root)
    with closing(CorpusSampler(short_refs, [], **SAMPLER_KWARGS, kernel_synth_every=0)) as sampler:
        with pytest.raises(ValueError, match="corpus has no eligible series"):
            next(sampler.batches())

    # Skip short shards, retain the tail batch, and allow an empty disabled kind.
    write_corpus_fixture(short_root, lengths=(9, 9), n_series=3, index=2)
    with closing(
        CorpusSampler(corpus_manifest(short_root), [], **SAMPLER_KWARGS, kernel_synth_every=0)
    ) as sampler:
        for batch in islice(sampler.batches(), 4):
            assert batch.size == 3
            assert batch.origin_pos.tolist() == [8, 8, 8]
            assert batch.label_mask.sum(dim=1).tolist() == [1, 1, 1]


def test_calendar_off_and_series_code_zero(two_kind_fixture):
    with closing(CorpusSampler(**two_kind_fixture, **SAMPLER_KWARGS)) as sampler:
        assert sampler.encoder.vocab_sizes == {"__series__": 2}
        assert sampler.encoder.vocabularies == {"__series__": {}}
        for batch in islice(sampler.batches(), 10):
            assert not batch.enc_cov.any()
            assert not batch.dec_cov.any()
            assert not batch.enc_cov_mask.any()
            assert not batch.dec_cov_mask.any()
            assert batch.cat.keys() == {"__series__"}
            assert not batch.cat["__series__"].any()
