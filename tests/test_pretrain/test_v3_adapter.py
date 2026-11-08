"""Real-grid training, validation and bundle telemetry on tiny offline shards."""

from contextlib import closing
from dataclasses import fields

import numpy as np
import pytest
import torch

pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench

from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.data.frequency import (  # noqa: E402
    CANONICAL_CALENDAR_SCHEMA,
    calendar_feature_mask,
    calendar_features,
    grid_range,
)
from demand_forecast.data.sampler import TARGET_LAG_SCHEMA  # noqa: E402
from demand_forecast.pretrain.config import PretrainConfig, pretrain_identity  # noqa: E402
from demand_forecast.pretrain.corpus import (  # noqa: E402
    ShardRef,
    corpus_manifest,
    read_shard_v3,
    shard_panel,
    shard_panels,
    write_manifest,
)
from demand_forecast.pretrain.sampler import V3BatchBuilder, V3ShardStream  # noqa: E402
from tests.test_pretrain.test_recipe_grids import _row, _shard  # noqa: E402


def _mixed_shard(tmp_path):
    values = (np.arange(420, dtype=float) ** 2).tolist()
    return _shard(
        tmp_path,
        [
            _row(0, target=values),
            _row(1, "h", target=values),
            _row(2, start=None, target=values),
            _row(3, target=values),
        ],
    )


def _builder(path, kind="chronos_real"):
    groups = shard_panels(read_shard_v3(path, kind=kind).records)
    encoder = StaticEncoder().fit(shard_panel([]), [])
    return V3BatchBuilder(groups, 4, 80, encoder), groups, encoder


def test_v3_adapter_real_calendars_lags_padding_and_requested_row_order(tmp_path):
    builder, groups, _ = _builder(_mixed_shard(tmp_path))
    items = [(1, 190), (3, 15), (2, 90), (0, 380), (1, 20)]
    batch = builder.build(items)
    assert batch.series_index.tolist() == [i for i, _ in items]
    assert batch.origin_pos.tolist() == [o for _, o in items]
    assert batch.target.shape == (5, 80)
    assert batch.enc_cov.shape == (5, 80, 18)
    assert batch.dec_cov.shape == (5, 4, 18)
    assert not batch.padding[1, :-15].any()
    assert not batch.enc_cov_mask[1, :-15].any()
    for row, (index, origin) in enumerate(items):
        group = next(g for g in groups if index in g.row_indices)
        series = group.panel.series[group.row_indices.index(index)]
        start, length = max(0, origin - 80), min(origin, 80)
        grid = grid_range(series.start, origin + 4, group.panel.freq)
        if group.calendar_available:
            expected = calendar_features(grid, group.panel.freq, "canonical")
            np.testing.assert_array_equal(batch.enc_cov[row, -length:, :16], expected[start:origin])
            np.testing.assert_array_equal(batch.dec_cov[row, :, :16], expected[origin:])
            expected_mask = np.broadcast_to(
                calendar_feature_mask(group.panel.freq, "canonical"), (length, 16)
            )
            np.testing.assert_array_equal(batch.enc_cov_mask[row, -length:, :16], expected_mask)
        else:
            assert not batch.enc_cov_mask[row, :, :16].any()
            assert not batch.dec_cov_mask[row, :, :16].any()
            assert not batch.enc_cov[row, :, :16].any()
            assert not batch.dec_cov[row, :, :16].any()
            assert not batch.enc_cov_mask[row, :, 16].any()  # Period one is masked.
            assert not batch.dec_cov_mask[row, :, 16].any()
            assert batch.dec_cov_mask[row, :, 17].all()  # Weekly long lag is 52.
        period, long = next(b.lags for g, b in builder.builders if g is group)
        for slot, lag in enumerate((period, long), start=16):
            for h in range(4):
                pos = origin + h - lag
                valid = 0 <= pos < origin and not (slot == 16 and lag == 1)
                assert batch.dec_cov_mask[row, h, slot].item() == valid
                if valid:
                    expected = (series.target[pos] - batch.loc[row].item()) / batch.scale[
                        row
                    ].item()
                    assert batch.dec_cov[row, h, slot].item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("prefetch", [0, 1])
def test_v3_stream_reproducible_row_mapping_and_resume(tmp_path, prefetch):
    path = _mixed_shard(tmp_path)
    encoder = StaticEncoder().fit(shard_panel([]), [])
    ref = ShardRef(path, "chronos_real", 0, "unused", path.stat().st_size)
    with closing(
        V3ShardStream(
            [ref],
            "chronos_real",
            horizon=4,
            max_context=80,
            min_context=8,
            batch_size=3,
            seed=7,
            encoder=encoder,
            prefetch=prefetch,
            period_policy="frequency",
            grid_policy="source",
            scale_floor_frac=0.0,
        )
    ) as stream:
        first = next(stream)
        reference, _, _ = _builder(path)
        expected = reference.build(
            list(zip(first.series_index.tolist(), first.origin_pos.tolist()))
        )
        assert torch.equal(first.enc_cov, expected.enc_cov)
        state = stream.state()
        expected = [next(stream) for _ in range(4)]
        stream.load_state(state)
        for wanted in expected:
            actual = next(stream)
            for field in fields(actual):
                a, b = getattr(actual, field.name), getattr(wanted, field.name)
                if isinstance(a, dict):
                    assert all(torch.equal(a[k], b[k]) for k in a)
                else:
                    assert torch.equal(a, b), field.name


def test_v3_validation_preserves_grids_uses_adapter_and_cell_period(
    tmp_path, tiny_pretrain_cfg, monkeypatch
):
    from demand_forecast import metrics
    from demand_forecast.pretrain import validate

    path = _mixed_shard(tmp_path)
    cfg = PretrainConfig.model_validate(
        tiny_pretrain_cfg.model_dump()
        | {
            "recipe": "v3",
            "period_policy": "frequency",
            "grid_policy": "source",
        }
    )
    ref = ShardRef(path, "chronos_real", 1, "unused", 0)
    pairs = validate.holdout_pairs_v3([ref], cfg)
    assert sorted(g.panel.seasonal_period for g in pairs.groups) == [1, 7, 24]
    assert sum(len(g.panel) for g in pairs.groups) == 4
    _, _, encoder = _builder(path)
    seen = []
    build = V3BatchBuilder.build

    def observe(self, items):
        batch = build(self, items)
        seen.append(batch.clone())
        return batch

    monkeypatch.setattr(V3BatchBuilder, "build", observe)

    class LastValue(torch.nn.Module):
        def predict(self, batch):
            level = batch.target[:, -1] * batch.scale + batch.loc
            return level[:, None, None].expand(-1, 4, 3)

    model = LastValue().train()
    scores = validate.evaluate_pairs_v3(
        model,
        pairs,
        horizon=4,
        max_context=80,
        encoder=encoder,
        quantiles=[0.1, 0.5, 0.9],
        batch_size=3,
        device="cpu",
    )
    assert model.training and len(seen) == 2
    expected = []
    for group in pairs.groups:
        for series in group.panel.series:
            past, future = series.target[:-4], series.target[-4:]
            scale = metrics.seasonal_error_per_item([past], group.panel.seasonal_period)[0]
            expected.append(np.mean(np.abs(future - past[-1])) / scale)
    assert scores["SQL"] == pytest.approx(np.mean(expected), rel=1e-6)
    assert scores["naive_sql"] == pytest.approx(scores["SQL"], rel=1e-6)
    assert scores["n_excluded"] == 0 and scores["n_items"] == 4


def _v3_cfg(tiny_pretrain_cfg, tmp_path):
    real = tiny_pretrain_cfg.corpus.root / "training_corpus/chronos_real"
    real.mkdir()
    _mixed_shard(tmp_path).rename(real / "train-00000-of-00002.parquet")
    # A separate, disjoint real holdout has the same tiny grids.
    _mixed_shard(tmp_path).rename(real / "train-00001-of-00002.parquet")
    write_manifest(
        corpus_manifest(tiny_pretrain_cfg.corpus.root),
        tiny_pretrain_cfg.corpus.root / "manifest.json",
    )
    raw = tiny_pretrain_cfg.model_dump() | {
        "recipe": "v3",
        "period_policy": "frequency",
        "grid_policy": "source",
    }
    raw["corpus"] = dict(
        root=tiny_pretrain_cfg.corpus.root,
        external_validation=None,
        val_pairs_per_kind=4,
        sources={
            kind: dict(shards=[0], holdout=[1], weight=1)
            for kind in ("tsmixup", "kernel_synth", "chronos_real")
        },
    )
    raw["training"].update(max_steps=3, eval_every=3, checkpoint_every=2)
    return PretrainConfig.model_validate(raw)


def test_v3_zero_initialization_exposures_bundle_and_resume(tmp_path, tiny_pretrain_cfg):
    from demand_forecast.core.bundle import load_bundle
    from demand_forecast.core.training import InterruptedRun
    from demand_forecast.pretrain.train import _PretrainLoop, pretrain

    cfg = _v3_cfg(tiny_pretrain_cfg, tmp_path)
    loop = _PretrainLoop(cfg)
    try:
        assert len(loop.panel) == 0  # Spec does not read the first synthetic shard.
        assert loop.model.spec.calendar_schema == CANONICAL_CALENDAR_SCHEMA
        assert loop.model.spec.lag_schema == TARGET_LAG_SCHEMA
        assert not loop.model.enc_proj.weight[:, 2:].any()
        assert not loop.model.dec_proj.weight.any()
        batch, batch_s, _ = loop.next_batch()
        loop.train_step(batch, batch_s)
        for side, mask in (("encoder", batch.enc_cov_mask), ("decoder", batch.dec_cov_mask)):
            counts = loop.timings["feature_exposure"][side]
            assert list(counts) == list(CANONICAL_CALENDAR_SCHEMA + TARGET_LAG_SCHEMA)
            assert list(counts.values()) == mask.sum((0, 1)).tolist()
    finally:
        loop.sampler.close()
    full = pretrain(cfg, tmp_path / "full")
    with pytest.raises(InterruptedRun):
        pretrain(cfg, tmp_path / "resumed", stop_after=2)
    resumed = pretrain(cfg, tmp_path / "resumed", resume=True)
    a, b = load_bundle(tmp_path / "full/bundle"), load_bundle(tmp_path / "resumed/bundle")
    assert full.history == resumed.history
    for name, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[name]), name
    exposure = a.manifest["feature_exposure"]
    assert exposure == full.timings["feature_exposure"] == resumed.timings["feature_exposure"]
    assert exposure["encoder"]["dow_sin"] > 0
    assert exposure["encoder"]["quarter_sin"] == 0
    names = CANONICAL_CALENDAR_SCHEMA + TARGET_LAG_SCHEMA
    for name in ("dow_sin", "lag_period", "lag_long"):
        column = names.index(name)
        assert a.model.enc_proj.weight[:, 2 + column].any()
        assert a.model.dec_proj.weight[:, column].any()
    for column in (14, 15):
        for offset in (0, len(names)):
            assert not a.model.enc_proj.weight[:, 2 + offset + column].any()
            assert not a.model.dec_proj.weight[:, offset + column].any()
    assert a.model.spec.calendar_schema == CANONICAL_CALENDAR_SCHEMA
    assert a.model.spec.lag_schema == TARGET_LAG_SCHEMA


def test_v3_identity_hashes_schema_names(tiny_pretrain_cfg, monkeypatch):
    from demand_forecast.data import frequency

    cfg = PretrainConfig.model_validate(tiny_pretrain_cfg.model_dump() | {"recipe": "v3"})
    before = pretrain_identity(cfg, "2.0.0", {})
    monkeypatch.setattr(frequency, "CANONICAL_CALENDAR_SCHEMA", ("different",))
    assert pretrain_identity(cfg, "2.0.0", {}) != before


def test_synthetic_source_keeps_long_lag_without_inventing_calendar(tmp_path):
    builder, groups, _ = _builder(_mixed_shard(tmp_path), kind="tsmixup")
    assert len(groups) == 1 and not groups[0].calendar_available
    assert groups[0].panel.seasonal_period == 1
    batch = builder.build([(2, 100), (0, 70)])
    assert not batch.enc_cov[..., :17].any()
    assert not batch.enc_cov_mask[..., :17].any()
    assert not batch.dec_cov[..., :17].any()
    assert not batch.dec_cov_mask[..., :17].any()
    assert batch.dec_cov_mask[..., 17].all()


def test_v3_pretrain_probe_measures_named_inputs(tmp_path, tiny_pretrain_cfg):
    from demand_forecast.pretrain.probe import pretrain_probe

    cfg = _v3_cfg(tiny_pretrain_cfg, tmp_path)
    report = pretrain_probe(cfg, steps=2, out=tmp_path / "probe-v3.json")
    assert np.isfinite(report["val_sql"])
    assert report["batch_build_s_per_step"] > 0
    assert report["shard_load_s"] > 0


def test_shipped_v3_config_samples_fev_horizons_and_selects_real():
    from pathlib import Path

    configs = Path(__file__).resolve().parents[2] / "configs"
    v2 = PretrainConfig.from_yaml(configs / "pretrain-v2-corpus2.yaml")
    v3 = PretrainConfig.from_yaml(configs / "pretrain-v3-corpus2.yaml")
    assert (v3.recipe, v3.period_policy, v3.grid_policy) == ("v3", "frequency", "source")
    assert (v3.forecast.calendar_schema, v3.forecast.target_lags) == ("canonical", "cycle")
    assert v3.forecast.horizon == 288 and v3.horizon_policy == "fev_set"
    assert v3.training.batch_size == v3.training.val_batch_size == 96
    assert v3.corpus.model_dump(
        exclude={"selection_shards", "external_validation"}
    ) == v2.corpus.model_dump(exclude={"external_validation"})
    assert v3.corpus.external_validation is None
    assert v3.corpus.selection_shards == {"chronos_real": [9], "gifteval_pretrain": [14]}
    assert v3.training.ema_decay == 0.999 and v3.training.keep_eval_snapshots
    assert v3.training.model_dump(
        exclude={"batch_size", "val_batch_size", "ema_decay", "keep_eval_snapshots"}
    ) == v2.training.model_dump(exclude={"batch_size", "val_batch_size"})
