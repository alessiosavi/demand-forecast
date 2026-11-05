"""Long corpus histories stay on the synthetic weekly grid without overflow."""

import logging
from contextlib import closing
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
import torch

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench

from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.data.sampler import Batch, BatchBuilder  # noqa: E402
from demand_forecast.pretrain import corpus  # noqa: E402
from demand_forecast.pretrain.sampler import CorpusSampler  # noqa: E402
from demand_forecast.pretrain.validate import holdout_pairs  # noqa: E402
from tests.helpers.corpus_fixture import write_corpus_fixture  # noqa: E402


@pytest.fixture
def long_series_shard(tmp_path):
    import pyarrow.parquet as pq

    targets = [np.arange(700_000, dtype=np.float64), np.arange(100, dtype=np.float64)]
    path = tmp_path / "training_corpus/gifteval_pretrain/train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pyarrow.table(
            {
                "id": ["long", "short"],
                "target": targets,
                "start": [pd.Timestamp("2020-01-01")] * 2,
                "freq": ["min"] * 2,
                "source": ["gifteval_pretrain"] * 2,
                "folder": ["fixture"] * 2,
            }
        ),
        path,
    )
    return path, targets


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_shard_keeps_recent_points_and_logs_truncation(long_series_shard, caplog, dtype):
    path, targets = long_series_shard
    with caplog.at_level(logging.INFO, logger=corpus.__name__):
        series = corpus.read_shard(path, dtype=dtype)

    assert [item.series_id for item in series] == ["long", "short"]
    assert [item.history_length for item in series] == [8192, 100]
    for item, expected in zip(series, (targets[0][-8192:], targets[1]), strict=True):
        assert item.target.dtype == dtype
        assert item.start == corpus.PRETRAIN_START
        assert item.dynamic == item.statics == {}
        np.testing.assert_array_equal(item.target, expected.astype(dtype))
    records = [record for record in caplog.records if record.name == corpus.__name__]
    assert len(records) == 1
    assert "Truncated 1 series" in records[0].getMessage()
    assert str(path) in records[0].getMessage()


def test_long_series_builds_training_batches(long_series_shard, tmp_path):
    path, _ = long_series_shard
    panel = corpus.shard_panel(corpus.read_shard(path))
    refs = corpus.corpus_manifest(tmp_path)
    with closing(
        CorpusSampler(
            sources={"gifteval_pretrain": refs},
            weights={"gifteval_pretrain": 1},
            horizon=52,
            max_context=512,
            min_context=8,
            batch_size=2,
            seed=0,
        )
    ) as sampler:
        batch = next(sampler.batches())

    assert batch.size == 2
    assert sorted(batch.series_index.tolist()) == [0, 1]
    assert batch.target.shape == (2, 512)
    assert batch.labels.shape == (2, 52)
    assert batch.loss_eligible.all()
    assert not batch.enc_cov_mask.any()
    assert not batch.dec_cov_mask.any()
    for index, origin in zip(batch.series_index.tolist(), batch.origin_pos.tolist(), strict=True):
        assert 8 <= origin < panel.series[index].history_length
    for field in fields(Batch):
        value = getattr(batch, field.name)
        tensors = value.values() if isinstance(value, dict) else (value,)
        assert all(torch.isfinite(tensor).all() for tensor in tensors), field.name
    for index, length in enumerate((8192, 100)):
        assert len(panel.timestamps(index).as_unit("ns")) == length


def test_holdout_uses_capped_history_and_builds_final_window(long_series_shard, tmp_path):
    _, targets = long_series_shard
    panel, origins = holdout_pairs(
        corpus.corpus_manifest(tmp_path)[0], horizon=52, n_pairs=2, seed=0
    )
    by_id = {
        item.series_id: (item, origin) for item, origin in zip(panel.series, origins, strict=True)
    }
    assert by_id["long"][1] == 8192 - 52
    assert by_id["short"][1] == 100 - 52
    np.testing.assert_array_equal(by_id["long"][0].target, targets[0][-8192:])
    np.testing.assert_array_equal(by_id["short"][0].target, targets[1])
    encoder = StaticEncoder().fit(corpus.shard_panel([]), [])
    batch = BatchBuilder(panel, 52, 512, encoder, calendar=False).build(enumerate(origins))
    assert batch.size == 2
    assert batch.label_mask.all()
    assert batch.loss_eligible.all()


@pytest.mark.parametrize("length,truncated", [(8192, 0), (8193, 1)])
def test_read_shard_cap_boundary_preserves_missing_positions(tmp_path, caplog, length, truncated):
    import pyarrow.parquet as pq

    values = np.arange(length, dtype=np.float32)
    values[-8192] = values[-10] = values[-1] = np.nan
    path = tmp_path / "boundary.parquet"
    pq.write_table(pyarrow.table({"id": ["boundary"], "target": [values]}), path)
    with caplog.at_level(logging.INFO, logger=corpus.__name__):
        item = corpus.read_shard(path)[0]

    assert item.history_length == 8192
    np.testing.assert_array_equal(item.target, values[-8192:])
    assert f"Truncated {truncated} series" in caplog.text


def test_legacy_golden_fixture_histories_stay_below_cap(tmp_path):
    import pyarrow.parquet as pq

    # Reconstruct every shard used by test_legacy_fifty_batches_golden without
    # changing those fixtures or the frozen identity/batch digests.
    for kind in ("tsmixup", "kernel_synth"):
        for index in (0, 1):
            path = write_corpus_fixture(tmp_path, kind=kind, index=index, seed=index)
            targets = pq.read_table(path, columns=["target"])["target"].to_pylist()
            assert all(len(target) < corpus.MAX_SERIES_POINTS for target in targets)
            for item, target in zip(corpus.read_shard(path), targets, strict=True):
                assert item.history_length == len(target)
                assert item.target.tobytes() == np.asarray(target, dtype=np.float32).tobytes()
