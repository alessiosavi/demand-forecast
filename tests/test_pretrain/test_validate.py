"""Local hold-outs and original-unit validation without fev or downloads."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench

from demand_forecast.config.settings import ForecastConfig, ModelConfig  # noqa: E402
from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.data.panel import Series  # noqa: E402
from demand_forecast.data.sampler import BatchBuilder  # noqa: E402
from demand_forecast.metrics import sql, summarize  # noqa: E402
from demand_forecast.models.base import ForecastModel, ModelSpec  # noqa: E402
from demand_forecast.pretrain.config import (  # noqa: E402
    CorpusConfig,
    PretrainConfig,
    PretrainTrainingConfig,
)
from demand_forecast.pretrain.corpus import corpus_manifest, read_shard, shard_panel  # noqa: E402
from demand_forecast.pretrain.validate import (  # noqa: E402
    ValidationSet,
    evaluate_pairs,
    external_pairs,
    holdout_pairs,
)
from tests.helpers.corpus_fixture import write_corpus_fixture  # noqa: E402
from tests.helpers.toy_backbone import ToyBackbone  # noqa: E402

QUANTILES = (0.1, 0.5, 0.9)


def _toy_model(panel, *, dropout=0.0):
    encoder = StaticEncoder().fit(shard_panel([]), [])
    spec = ModelSpec.from_panel(
        panel, encoder, ForecastConfig(horizon=3, quantiles=list(QUANTILES)), 8, 2
    )
    backbone = ToyBackbone(8)
    backbone.projection = nn.Sequential(backbone.projection, nn.Dropout(dropout))
    return ForecastModel(spec, backbone), encoder


def _panel():
    return shard_panel(
        [
            Series(
                f"series-{i}",
                pd.Timestamp("2000-01-02"),
                16 + i,
                (100 + 10 * i + np.arange(16 + i) ** 2).astype(np.float32),
                {},
                {},
            )
            for i in range(3)
        ]
    )


@pytest.mark.parametrize("support", ["real", "nonnegative"])
def test_evaluate_pairs_matches_metrics_summarize_on_toy_model(support):
    panel = replace(_panel(), target_support=support)
    model, encoder = _toy_model(panel)
    # Force negative forecasts so the nonnegative support path is exercised.
    with torch.no_grad():
        model.head.projection.bias[1] = -100
    pairs = [(2, 12), (0, 10), (1, 13)]
    builder = BatchBuilder(panel, 3, 4, encoder, calendar=False)
    model.eval()
    with torch.no_grad():
        q_pred = np.concatenate([model.predict(builder.build([pair])).numpy() for pair in pairs])
    assert np.all(q_pred == 0) if support == "nonnegative" else np.all(q_pred < 0)
    y_true = np.stack([panel.series[i].target[origin : origin + 3] for i, origin in pairs])
    y_past = [panel.series[i].target[:origin] for i, origin in pairs]
    expected = summarize(y_true, q_pred, y_past, m=1, quantiles=QUANTILES)
    naive = np.broadcast_to(np.array([past[-1] for past in y_past])[:, None, None], (3, 3, 3))
    expected.update(
        n_items=3,
        n_excluded=0,
        naive_sql=sql(y_true, naive, y_past, m=1, quantiles=QUANTILES),
    )

    actual = evaluate_pairs(
        model,
        panel,
        pairs,
        horizon=3,
        max_context=4,
        encoder=encoder,
        quantiles=QUANTILES,
        batch_size=2,
        device="cpu",
    )
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert model.training is False


def test_restores_train_mode_and_is_deterministic_with_dropout():
    panel = _panel()
    model, encoder = _toy_model(panel, dropout=0.5)
    model.train()
    kwargs = dict(
        horizon=3, max_context=4, encoder=encoder, quantiles=QUANTILES, batch_size=2, device="cpu"
    )
    pairs = [(i, series.length - 3) for i, series in enumerate(panel.series)]
    first = evaluate_pairs(model, panel, pairs, **kwargs)
    assert model.training is True
    second = evaluate_pairs(model, panel, pairs, **kwargs)
    assert first == second
    assert model.training is True
    assert model.backbone.projection[1].training is True


def test_all_constant_returns_nan_sql_with_exclusions(tmp_path):
    write_corpus_fixture(tmp_path, n_series=3, lengths=(16, 16), constant=True)
    panel, origins = holdout_pairs(corpus_manifest(tmp_path)[0], horizon=3, n_pairs=8, seed=9)
    model, encoder = _toy_model(panel)
    actual = evaluate_pairs(
        model,
        panel,
        list(enumerate(origins)),
        horizon=3,
        max_context=4,
        encoder=encoder,
        quantiles=QUANTILES,
        device="cpu",
    )
    assert np.isnan(actual["SQL"])
    assert np.isnan(actual["naive_sql"])
    assert actual["n_excluded"] == actual["n_items"] == 3


def test_holdout_pairs_skips_short_series_and_is_seeded(tmp_path):
    path = write_corpus_fixture(tmp_path, n_series=30, lengths=(7, 16), seed=2)
    ref = corpus_manifest(tmp_path)[0]
    source = read_shard(path)
    eligible = [series for series in source if series.length >= 11]
    assert 5 < len(eligible) < len(source)
    assert any(series.length == 11 for series in eligible)
    first, origins = holdout_pairs(ref, horizon=3, n_pairs=5, seed=7)
    same, same_origins = holdout_pairs(ref, horizon=3, n_pairs=5, seed=7)
    other, _ = holdout_pairs(ref, horizon=3, n_pairs=5, seed=8)

    def ids(panel):
        return [series.series_id for series in panel.series]

    assert ids(first) == ids(same)
    assert ids(first) != ids(other)
    assert len(first) == len(set(ids(first))) == 5
    assert origins.dtype.kind in "iu"
    np.testing.assert_array_equal(origins, same_origins)
    np.testing.assert_array_equal(origins, [series.length - 3 for series in first.series])
    assert all(series.length >= 11 for series in first.series)
    all_eligible, _ = holdout_pairs(ref, horizon=3, n_pairs=100, seed=7)
    assert set(ids(all_eligible)) == {series.series_id for series in eligible}


@pytest.mark.parametrize("training", [True, False])
def test_evaluate_pairs_restores_mode_on_prediction_error(monkeypatch, training):
    panel = _panel()
    model, encoder = _toy_model(panel)
    model.train(training)

    def fail(batch):
        assert not torch.is_grad_enabled()
        assert not model.training
        assert not batch.enc_cov_mask.any()
        assert not batch.dec_cov_mask.any()
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(RuntimeError, match="prediction failed"):
        evaluate_pairs(
            model,
            panel,
            [(0, 10)],
            horizon=3,
            max_context=4,
            encoder=encoder,
            quantiles=QUANTILES,
            device="cpu",
        )
    assert model.training is training


def test_evaluate_pairs_missing_labels_and_last_observed_naive():
    panel = _panel()
    panel.series[0].target = np.array([1, 3, 6, np.nan, 8, np.nan, 10], dtype=np.float32)
    panel.series[1].target = np.ones(7, dtype=np.float32)
    panel.series[2].target = np.full(7, np.nan, dtype=np.float32)
    for series in panel.series:
        series.history_length = 7
    model, encoder = _toy_model(panel)
    actual = evaluate_pairs(
        model,
        panel,
        [(0, 4), (1, 4), (2, 4)],
        horizon=3,
        max_context=2,
        encoder=encoder,
        quantiles=QUANTILES,
        device="cpu",
    )
    assert actual["n_excluded"] == 2
    assert actual["n_items"] == 3
    # Only row 0 contributes: lag-one scale = (2 + 3) / 2 = 2.5;
    # last observed = 6; mean absolute future error = (2 + 4) / 2 = 3.
    assert actual["naive_sql"] == pytest.approx(1.2)
    assert np.isfinite(actual["SQL"])


def test_empty_holdout_and_evaluation(tmp_path):
    write_corpus_fixture(tmp_path, n_series=2, lengths=(5, 5))
    panel, origins = holdout_pairs(corpus_manifest(tmp_path)[0], horizon=3, n_pairs=4, seed=0)
    assert len(panel) == origins.size == 0
    assert origins.dtype.kind in "iu"
    model, encoder = _toy_model(panel)
    result = evaluate_pairs(
        model,
        panel,
        [],
        horizon=3,
        max_context=4,
        encoder=encoder,
        quantiles=QUANTILES,
        device="cpu",
    )
    assert result["n_items"] == result["n_excluded"] == 0
    assert np.isnan(result["SQL"])
    assert np.isnan(result["naive_sql"])


def _external_fixture(root, *, freq="W-SUN"):
    datasets = pytest.importorskip("datasets")
    timestamps = pd.date_range("2017-04-03", periods=12, freq=freq)
    values = [-5.0, 1.0, 4.0, 2.0, 6.0, np.nan, 7.0, 9.0, 10.0, 12.0, 13.0, 14.0]
    datasets.Dataset.from_dict(
        {"id": ["external"], "timestamp": [list(timestamps.to_pydatetime())], "target": [values]}
    ).save_to_disk(str(root / "monash_nn5_weekly"))
    return timestamps, values


@pytest.mark.parametrize("freq", ["W-SUN", "W-WED"])
def test_external_pairs_preserves_real_weekly_timestamps(tmp_path, freq):
    timestamps, values = _external_fixture(tmp_path, freq=freq)
    panel, origins = external_pairs("monash_nn5_weekly", tmp_path, horizon=3)
    assert panel.freq == freq
    assert panel.target_support == "real"
    assert panel.seasonal_period == 1
    assert panel.series[0].series_id == "external"
    assert panel.series[0].start == timestamps[0]
    pd.testing.assert_index_equal(panel.timestamps(0), timestamps)
    np.testing.assert_array_equal(panel.series[0].target, values)
    np.testing.assert_array_equal(origins, [9])
    assert origins.dtype.kind in "iu"


def _config(root):
    return PretrainConfig(
        corpus=CorpusConfig(
            root=root, tsmixup_shards=[0], kernel_synth_shards=[0], val_pairs_per_kind=2
        ),
        forecast=ForecastConfig(horizon=3, max_context=4, quantiles=list(QUANTILES)),
        model=ModelConfig(d_model=8, nhead=2),
        training=PretrainTrainingConfig(max_steps=1, val_batch_size=1),
        seed=7,
    )


def test_validation_set_selects_holdouts_and_keeps_external_diagnostic_separate(tmp_path):
    for kind, index in [
        ("tsmixup", 0),
        ("tsmixup", 199),
        ("kernel_synth", 0),
        ("kernel_synth", 19),
    ]:
        write_corpus_fixture(tmp_path, kind=kind, index=index, n_series=3, lengths=(14, 16))
    _external_fixture(tmp_path)
    cfg = _config(tmp_path)
    model, encoder = _toy_model(_panel())
    validation = ValidationSet.build(cfg, corpus_manifest(tmp_path), encoder)
    expected = {}
    for kind, index in [("tsmixup", 199), ("kernel_synth", 19), ("external", None)]:
        panel, origins = getattr(validation, kind)
        if index is not None:
            assert len(panel) == 2
            assert all(series.series_id.startswith(f"{kind}-{index}-") for series in panel.series)
        expected[kind] = evaluate_pairs(
            model,
            panel,
            list(enumerate(origins)),
            horizon=3,
            max_context=4,
            encoder=encoder,
            quantiles=QUANTILES,
            batch_size=1,
            device="cpu",
        )
    actual = validation.evaluate(model, device="cpu")
    assert actual == {
        "val_sql_tsmixup": expected["tsmixup"]["SQL"],
        "val_sql_kernel_synth": expected["kernel_synth"]["SQL"],
        "val_sql": (expected["tsmixup"]["SQL"] + expected["kernel_synth"]["SQL"]) / 2,
        "ext_sql": expected["external"]["SQL"],
        "ext_sql_naive": expected["external"]["naive_sql"],
        "n_excluded_tsmixup": 0,
        "n_excluded_kernel_synth": 0,
        "n_excluded_external": 0,
    }
    validation.external = None
    without_external = validation.evaluate(model, device="cpu")
    assert without_external["val_sql"] == actual["val_sql"]
    assert without_external["ext_sql"] is None
    assert without_external["ext_sql_naive"] is None
    assert without_external["n_excluded_external"] is None


@pytest.mark.parametrize("constant", [False, True])
def test_validation_set_missing_sources_and_nan_selection_score(tmp_path, constant):
    cfg = _config(tmp_path)
    model, encoder = _toy_model(_panel())
    missing = ValidationSet.build(cfg, [], encoder)
    assert missing.tsmixup is missing.kernel_synth is missing.external is None
    assert missing.evaluate(model, device="cpu") == dict.fromkeys(
        [
            "val_sql_tsmixup",
            "val_sql_kernel_synth",
            "val_sql",
            "ext_sql",
            "ext_sql_naive",
            "n_excluded_tsmixup",
            "n_excluded_kernel_synth",
            "n_excluded_external",
        ]
    )
    write_corpus_fixture(tmp_path, index=199, n_series=2, lengths=(14, 14), constant=constant)
    validation = ValidationSet.build(cfg, corpus_manifest(tmp_path), encoder)
    scores = validation.evaluate(model, device="cpu")
    assert scores["val_sql_kernel_synth"] is scores["n_excluded_kernel_synth"] is None
    if constant:
        assert np.isnan(scores["val_sql"])
        assert np.isnan(scores["val_sql_tsmixup"])
        assert scores["n_excluded_tsmixup"] == 2
    else:
        assert np.isfinite(scores["val_sql"])
        assert scores["val_sql"] == scores["val_sql_tsmixup"]
