"""Portable inference artifacts, original-unit quantiles and complete panel coverage."""

import hashlib
import json
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from demand_forecast import __version__
from demand_forecast.config.settings import DatasetConfig, ForecastConfig
from demand_forecast.core.bundle import load_bundle, save_bundle
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import grid_range, to_offset
from demand_forecast.data.sampler import make_batch
from demand_forecast.inference import Predictor
from demand_forecast.models.base import ModelSpec
from tests.helpers.models import build_v2
from tests.helpers.panels import weekly_panel

SCRIPT_RELOAD = Path(__file__).resolve().parents[1] / "helpers" / "reload_forecast.py"


def test_fresh_process_reload_parity(
    tmp_path, bundle_dir, panel29, origin29, tiny_v2_trained, forecast29, encoder29
):
    ids, first = Predictor.from_bundle(bundle_dir).forecast_arrays(panel29, origin29)
    assert ids == [s.series_id for s in panel29.series] and first.shape == (3, 13, 9)
    tiny_v2_trained.eval()
    batch = make_batch(panel29, [(i, 29) for i in range(3)], 13, 64, encoder29)
    np.testing.assert_array_equal(first, tiny_v2_trained.predict(batch).numpy())
    np.save(tmp_path / "first.npy", first)
    with (tmp_path / "panel29.pkl").open("wb") as stream:
        pickle.dump(panel29, stream)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_RELOAD),
            str(bundle_dir),
            str(tmp_path / "panel29.pkl"),
            str(origin29),
            str(tmp_path / "second.npy"),
        ],
        cwd=tmp_path,
        check=True,
        timeout=30,
    )
    assert np.array_equal(np.load(tmp_path / "second.npy"), first)


def test_single_observation_and_unseen_id_get_finite_fallbacks(bundle_dir, origin29):
    panel = weekly_panel([1, 8, 8])
    panel.series[0].series_id = "single"
    panel.series[0].target[:] = 7.0
    panel.series[1].series_id = "unseen"
    panel.series[1].statics["cat"] = "new category"
    panel.series[2].series_id = "missing"
    panel.series[2].target[:] = np.nan
    predictor = Predictor.from_bundle(bundle_dir)
    assert predictor.bundle.encoder.transform(panel.series[1])[0]["__series__"] == 0
    df = predictor.forecast(panel, origin29)
    assert len(df) == len(panel) * 13
    assert np.isfinite(df[["q0.1", "q0.5", "q0.9", "point"]].to_numpy()).all()
    assert (df.loc[df.series_id == "single", "q0.5"] >= 0).all()
    assert (df.loc[df.series_id == "single", "q0.5"] == 7).all()
    assert (df.loc[df.series_id == "missing", "q0.5"] == 0).all()


def test_origin_before_earliest_is_refused(bundle_dir, panel29, origin29):
    predictor = Predictor.from_bundle(bundle_dir)
    for method in (predictor.forecast, predictor.forecast_arrays):
        with pytest.raises(ValueError, match="^origin before earliest permissible origin$"):
            method(panel29, origin29 - panel29.freq)


def test_bundle_metadata_configs_state_and_weights_only_load(
    tmp_path, tiny_v2_trained, forecast29, encoder29, panel29, cutoff29, origin29, monkeypatch
):
    directory = tmp_path / "b"
    dataset = DatasetConfig(
        path=tmp_path / "absent.csv",
        static_path=tmp_path / "absent-statics.csv",
        id_columns=["id"],
        target="y",
        target_support="nonnegative",
        freq="W-SUN",
    )
    caller_manifest = {"seed": 3, "run_identity": "test-run", "input_hashes": {"data": "abc"}}
    save_bundle(
        directory,
        model=tiny_v2_trained,
        family="v2",
        backbone_kwargs=asdict(tiny_v2_trained.backbone.cfg),
        forecast_cfg=forecast29,
        dataset_cfg=dataset,
        encoder=encoder29,
        panel=panel29,
        training_cutoff=cutoff29,
        earliest_origin=origin29,
        manifest=caller_manifest,
    )
    assert {p.name for p in directory.iterdir()} == {
        "model.pt",
        "config.json",
        "encoder.json",
        "panel.json",
        "manifest.json",
    }
    config = json.loads((directory / "config.json").read_text())
    assert config["family"] == "v2"
    assert config["backbone_kwargs"] == asdict(tiny_v2_trained.backbone.cfg)
    assert config["forecast"] == forecast29.model_dump(mode="json")
    assert config["dataset"] == dataset.model_dump(mode="json")
    assert config["model_spec"] == json.loads(json.dumps(asdict(tiny_v2_trained.spec)))
    assert json.loads((directory / "encoder.json").read_text()) == encoder29.to_json()
    metadata = json.loads((directory / "panel.json").read_text())
    assert metadata == {
        "freq": "W-SUN",
        "seasonal_period": 1,
        "target_support": "nonnegative",
        "roles": json.loads(json.dumps(asdict(panel29.roles))),
        "training_cutoff": cutoff29.isoformat(),
        "earliest_origin": origin29.isoformat(),
        "fallback_level": tiny_v2_trained.fallback_level.item(),
    }
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["package_version"] == __version__ and manifest["bundle_version"] == 1
    assert manifest["selection_used_labels_until"] == cutoff29.isoformat()
    assert pd.Timestamp(manifest["created_at"]).tzinfo is not None
    assert "git_sha" in manifest
    for key, value in caller_manifest.items():
        assert manifest[key] == value
    assert set(manifest["sha256"]) == {"model.pt", "config.json", "encoder.json", "panel.json"}
    for name, digest in manifest["sha256"].items():
        assert digest == hashlib.sha256((directory / name).read_bytes()).hexdigest()
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["weights_only"] is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    loaded = load_bundle(directory, device="cpu")
    assert len(calls) == 1 and not loaded.model.training
    assert loaded.forecast_cfg == forecast29 and loaded.dataset_cfg == dataset
    assert loaded.encoder.to_json() == encoder29.to_json()
    assert loaded.model.spec == tiny_v2_trained.spec
    assert loaded.panel_metadata == metadata and loaded.manifest == manifest
    state = loaded.model.state_dict()
    for name, expected in tiny_v2_trained.state_dict().items():
        assert torch.equal(state[name], expected)
    assert all(t.device.type == "cpu" for t in state.values())


@pytest.mark.parametrize("failure", [None, "manifest", "publish"])
def test_bundle_replacement_publishes_only_complete_artifacts(
    bundle_dir,
    tiny_v2_trained,
    forecast29,
    encoder29,
    panel29,
    cutoff29,
    origin29,
    monkeypatch,
    failure,
):
    from demand_forecast.core import bundle

    before = {p.name: p.read_bytes() for p in bundle_dir.iterdir()}
    original = load_bundle(bundle_dir)
    tiny_v2_trained.fallback_level.add_(7.0)
    write_json = bundle._write_json
    replace_directory = os.replace
    publish_failed = False

    def fail_manifest(path, value):
        if path.name == "manifest.json":
            raise OSError("manifest write failed")
        write_json(path, value)

    def fail_publish(source, target):
        nonlocal publish_failed
        if target == bundle_dir and not publish_failed:
            publish_failed = True
            raise OSError("publish failed")
        replace_directory(source, target)

    if failure == "manifest":
        monkeypatch.setattr(bundle, "_write_json", fail_manifest)
    elif failure == "publish":
        monkeypatch.setattr(os, "replace", fail_publish)

    def save():
        save_bundle(
            bundle_dir,
            model=tiny_v2_trained,
            family="v2",
            backbone_kwargs=asdict(tiny_v2_trained.backbone.cfg),
            forecast_cfg=forecast29,
            dataset_cfg=None,
            encoder=encoder29,
            panel=panel29,
            training_cutoff=cutoff29,
            earliest_origin=origin29,
            manifest={"run_identity": "replacement"},
        )

    if failure:
        with pytest.raises(OSError, match=failure):
            save()
        assert {p.name: p.read_bytes() for p in bundle_dir.iterdir()} == before
    else:
        save()

    loaded = load_bundle(bundle_dir)
    expected_model = original.model if failure else tiny_v2_trained
    for name, value in expected_model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], value)
    if failure:
        assert loaded.manifest == original.manifest
    else:
        assert loaded.manifest["run_identity"] == "replacement"
    for name, digest in loaded.manifest["sha256"].items():
        assert hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest() == digest
    assert not list(bundle_dir.parent.glob("*.tmp-*"))


def test_forecast_dataframe_batches_and_aligned_origins(bundle_dir, origin29, monkeypatch):
    panel = weekly_panel([45, 3, 7])
    panel.series[2].start = origin29 + pd.offsets.Week(weekday=6)
    predictor = Predictor.from_bundle(bundle_dir)
    calls = []
    original_predict = predictor.bundle.model.predict

    def predict(batch):
        assert not predictor.bundle.model.training and not torch.is_grad_enabled()
        calls.extend(zip(batch.series_index.tolist(), batch.origin_pos.tolist()))
        return original_predict(batch)

    monkeypatch.setattr(predictor.bundle.model, "predict", predict)
    df = predictor.forecast(panel, origin29, batch_size=2)
    assert calls == [(0, 29), (1, 29), (2, 0)]
    quantiles = [f"q{q}" for q in predictor.bundle.forecast_cfg.quantiles]
    assert list(df) == ["series_id", "origin", "timestamp", *quantiles, "point"]
    assert df.series_id.tolist() == [s.series_id for s in panel.series for _ in range(13)]
    assert (df.origin == origin29).all()
    for _, rows in df.groupby("series_id", sort=False):
        assert rows.timestamp.tolist() == list(grid_range(origin29, 13, panel.freq))
    assert np.isfinite(df[quantiles].to_numpy()).all()
    assert (np.diff(df[quantiles].to_numpy(), axis=-1) >= 0).all()
    np.testing.assert_array_equal(df.point, df["q0.5"])
    ids, values = predictor.forecast_arrays(panel, origin29)
    assert ids == [s.series_id for s in panel.series]
    np.testing.assert_allclose(df[quantiles].to_numpy(), values.reshape(-1, 9), rtol=1e-5)
    assert (df.loc[df.series_id == panel.series[2].series_id, quantiles] == 0).all().all()
    assert [s.history_length for s in panel.series] == [45, 3, 7]


def test_ragged_history_forecasts_at_common_origin_without_mutating_panel(
    bundle_dir, panel29, origin29, monkeypatch
):
    import demand_forecast.core.evaluation as evaluation

    panel = deepcopy(panel29)
    series = panel.series[0]
    series.history_length = 26  # Three missing grid positions before origin29.
    series.target = series.target[:26].copy()
    series.dynamic["promo"] = (np.arange(42) % 3 == 0).astype(np.float32)
    # Past-only values outside declared history must not leak into the gap,
    # even if a supplied array happens to contain them.
    series.dynamic["oil"][26:] = 1e6
    original = deepcopy(panel)
    extended = deepcopy(panel)
    extended.series[0].history_length = 29
    extended.series[0].target = np.r_[series.target, [np.nan] * 3]
    extended.series[0].dynamic["oil"] = np.r_[series.dynamic["oil"][:26], [np.nan] * 3]
    predictor = Predictor.from_bundle(bundle_dir)
    expected = predictor.forecast(extended, origin29)
    original_predict = predictor.bundle.model.predict

    def check_batch(batch):
        assert batch.origin_pos.tolist() == [29, 29, 29]
        assert batch.padding[0, -3:].all() and not batch.observed[0, -3:].any()
        assert not batch.enc_cov_mask[0, -3:, -1].any()  # Missing past-only oil.
        assert batch.dec_cov_mask[0, :, -1].all()  # Known promo remains available.
        np.testing.assert_array_equal(batch.dec_cov[0, :, -1], series.dynamic["promo"][29:42])
        return original_predict(batch)

    monkeypatch.setattr(predictor.bundle.model, "predict", check_batch)
    actual = predictor.forecast(panel, origin29)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    rows = actual.loc[actual.series_id == series.series_id]
    assert rows.timestamp.tolist() == list(grid_range(origin29, 13, panel.freq))
    assert (rows.origin == origin29).all()

    def check_evaluation(y_true, q_pred, y_past, m, quantiles):
        columns = [f"q{q}" for q in quantiles]
        np.testing.assert_array_equal(q_pred, expected[columns].to_numpy().reshape(3, 13, 9))
        np.testing.assert_array_equal(y_past[0], original.series[0].target)
        assert np.isnan(y_true[0]).all()
        return {"SQL": 0.0}

    monkeypatch.setattr(evaluation, "summarize", check_evaluation)
    evaluation.evaluate(
        predictor.bundle.model,
        panel,
        np.arange(len(panel)),
        origin29,
        13,
        64,
        predictor.bundle.encoder,
        predictor.bundle.forecast_cfg.quantiles,
    )
    for actual_series, original_series in zip(panel.series, original.series):
        assert actual_series.series_id == original_series.series_id
        assert actual_series.start == original_series.start
        assert actual_series.history_length == original_series.history_length
        assert actual_series.statics == original_series.statics
        np.testing.assert_array_equal(actual_series.target, original_series.target)
        assert actual_series.dynamic.keys() == original_series.dynamic.keys()
        for name in actual_series.dynamic:
            np.testing.assert_array_equal(
                actual_series.dynamic[name], original_series.dynamic[name]
            )


def test_empty_panel_and_invalid_batch_size(bundle_dir, panel29, origin29):
    predictor = Predictor.from_bundle(bundle_dir)
    panel = replace(panel29, series=[])
    ids, values = predictor.forecast_arrays(panel, origin29)
    assert ids == [] and values.shape == (0, 13, 9)
    assert predictor.forecast(panel, origin29).empty
    for size in (0, -1, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            predictor.forecast(panel29, origin29, batch_size=size)


@pytest.mark.parametrize("name", ["model.pt", "config.json", "encoder.json", "panel.json"])
def test_bundle_rejects_corrupt_artifacts(bundle_dir, name):
    with (bundle_dir / name).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(bundle_dir)


def test_custom_quantiles_real_fallbacks_and_calendar_timestamps(tmp_path, monkeypatch):
    # Cover a nonzero persisted fallback, negative observations and a calendar
    # offset across DST. Each series is its own batch, including an empty context.
    panel = replace(weekly_panel([1, 2, 2]), freq=to_offset("ME"), target_support="real")
    for series in panel.series:
        series.start = pd.Timestamp("2024-01-31", tz="Europe/Rome")
    panel.series[0].target[:] = -8.0
    panel.series[1].target[:] = np.nan
    origin = panel.series[0].start + 2 * panel.freq
    panel.series[2].start = origin + panel.freq
    encoder = StaticEncoder().fit(panel, [panel.series[0].series_id])
    cfg = ForecastConfig(horizon=3, quantiles=[0.25, 0.5, 0.75], max_context=64)
    model = build_v2(ModelSpec.from_panel(panel, encoder, cfg, 16, 4))
    model.fallback_level.fill_(-4.25)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(subprocess, "run", no_git)
    save_bundle(
        tmp_path,
        model=model,
        family="v2",
        backbone_kwargs=asdict(model.backbone.cfg),
        forecast_cfg=cfg,
        dataset_cfg=None,
        encoder=encoder,
        panel=panel,
        training_cutoff=origin - panel.freq,
        earliest_origin=origin,
        manifest={},
    )
    predictor = Predictor.from_bundle(tmp_path)
    assert predictor.bundle.dataset_cfg is None and predictor.bundle.manifest["git_sha"] is None
    df = predictor.forecast(panel, origin, batch_size=1)
    assert list(df) == ["series_id", "origin", "timestamp", "q0.25", "q0.5", "q0.75", "point"]
    np.testing.assert_array_equal(df.point, np.repeat([-8.0, -4.25, -4.25], 3))
    for _, rows in df.groupby("series_id", sort=False):
        assert rows.timestamp.tolist() == list(grid_range(origin, 3, panel.freq))
        assert (rows.origin == origin).all()


@pytest.mark.parametrize("field", ["freq", "roles", "target_support", "seasonal_period"])
def test_panel_schema_mismatch_is_refused(bundle_dir, panel29, origin29, field):
    overrides = {
        "freq": "W-MON",
        "roles": replace(panel29.roles, known_dynamic=("another_column",)),
        "target_support": "real",
        "seasonal_period": 7,
    }
    panel = replace(panel29, **{field: overrides[field]})
    with pytest.raises(ValueError, match="match bundle"):
        Predictor.from_bundle(bundle_dir).forecast(panel, origin29)


def test_bundle_version_and_off_grid_origin_are_refused(bundle_dir, panel29, origin29):
    predictor = Predictor.from_bundle(bundle_dir)
    with pytest.raises(ValueError, match="grid"):
        predictor.forecast(panel29, origin29 + pd.Timedelta(hours=1))
    path = bundle_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["bundle_version"] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported bundle version"):
        load_bundle(bundle_dir)


def test_predictor_diagnostics_match_arrays(bundle_dir, panel29, origin29):
    predictor = Predictor.from_bundle(bundle_dir)
    ids, arrays = predictor.forecast_arrays(panel29, origin29)
    actual_ids, actual, diagnostics = predictor.forecast_with_diagnostics(
        panel29, origin29, batch_size=2
    )
    assert actual_ids == ids
    np.testing.assert_allclose(actual, arrays, rtol=1e-5, atol=1e-5)
    assert diagnostics["n_items"] == len(ids)
    assert diagnostics["fallback_no_context"] == diagnostics["fallback_undefined_scale"] == 0
    assert diagnostics["device"] == "cpu"
    assert diagnostics["inference_time_s"] > 0


@pytest.fixture
def bundle_and_panel(bundle_dir, panel29, origin29):
    return Predictor.from_bundle(bundle_dir), panel29, origin29


def test_forecast_per_item_origins_matches_single_origin_when_shared(bundle_and_panel):
    predictor, panel, origin = bundle_and_panel
    ids, arrays, diag = predictor.forecast_with_diagnostics(panel, origin)
    ids2, arrays2, diag2 = predictor.forecast_per_item_origins(panel, {key: origin for key in ids})
    assert ids2 == ids
    np.testing.assert_array_equal(arrays2, arrays)
    for key in (
        "n_items",
        "fallback_no_context",
        "fallback_undefined_scale",
        "device",
        "cold_covariate_columns",
        "clipped_covariate_values",
        "floored_covariate_columns",
    ):
        assert diag2[key] == diag[key]
    assert diag2["n_origins"] == 1 and diag2["inference_time_s"] > 0


def test_forecast_per_item_origins_rejects_missing_origin(bundle_and_panel):
    predictor, panel, origin = bundle_and_panel
    with pytest.raises(ValueError, match="^missing origin for item a$"):
        predictor.forecast_per_item_origins(panel, {})
    with pytest.raises(ValueError, match="^missing origin for item c$"):
        predictor.forecast_per_item_origins(panel, {"a": origin, "b": origin})


def test_forecast_per_item_origins_preserves_order_and_sums_fallbacks(
    bundle_and_panel, monkeypatch
):
    predictor, panel, origin = bundle_and_panel
    panel.series[0].target[:] = np.nan
    panel.series[1].target[:] = 6.0
    origins = {"a": origin + panel.freq, "b": origin, "c": origin + panel.freq}
    panel.series[0].dynamic["promo"] = np.r_[np.full(30, np.nan), np.full(13, 1e5)]
    panel.series[1].dynamic["promo"] = np.r_[3.0 + 1e-6, np.full(28, 3.0), np.full(13, 4.0)]
    panel.series[2].dynamic["promo"] = np.r_[np.zeros(30), np.full(13, 1e5)]
    forecast = predictor.forecast_with_diagnostics
    expected = {}
    totals = {
        "n_items": 0,
        "fallback_no_context": 0,
        "fallback_undefined_scale": 0,
        "cold_covariate_columns": 0,
        "clipped_covariate_values": 0,
        "floored_covariate_columns": 0,
    }
    for group_origin in (origin + panel.freq, origin):
        group = replace(
            panel, series=[s for s in panel.series if origins[s.series_id] == group_origin]
        )
        ids, arrays, diag = forecast(group, group_origin, batch_size=1)
        expected.update(zip(ids, arrays))
        for key in totals:
            totals[key] += diag[key]
    seen = []

    def capture(group, group_origin, batch_size=512):
        seen.append((group_origin, [s.series_id for s in group.series], batch_size))
        ids, arrays, diag = forecast(group, group_origin, batch_size)
        return ids[::-1], arrays[::-1], diag

    monkeypatch.setattr(predictor, "forecast_with_diagnostics", capture)
    ids, arrays, diag = predictor.forecast_per_item_origins(panel, origins, batch_size=1)
    assert ids == ["a", "b", "c"]
    np.testing.assert_array_equal(arrays, np.stack([expected[key] for key in ids]))
    assert seen == [(origin + panel.freq, ["a", "c"], 1), (origin, ["b"], 1)]
    assert {key: diag[key] for key in totals} == totals
    assert totals == {
        "n_items": 3,
        "fallback_no_context": 1,
        "fallback_undefined_scale": 1,
        "cold_covariate_columns": 1,
        "clipped_covariate_values": 13,
        "floored_covariate_columns": 1,
    }
    assert diag["n_origins"] == 2 and diag["device"] == "cpu"
    assert diag["inference_time_s"] > 0


@pytest.mark.parametrize("guarded", [False, True])
def test_predictor_reuses_builder_across_chunks_matches_reference(
    bundle_and_panel, monkeypatch, guarded
):
    from demand_forecast.data.sampler import BatchBuilder
    from tests.helpers.reference_batch import reference_align_to_origin, reference_make_batch

    predictor, panel, origin = bundle_and_panel
    if guarded:
        panel.series[0].dynamic["promo"] = np.r_[np.full(29, np.nan), np.full(13, 1e5)]
        panel.series[1].dynamic["promo"] = np.r_[
            100.0 + 1e-6, np.full(28, 100.0), np.full(13, 110.0)
        ]
        panel.series[2].dynamic["promo"] = np.r_[1e-9, np.zeros(28), np.full(13, 0.5)]
    # 513 items cross the public default chunk boundary, including unseen ids.
    panel = replace(
        panel, series=[replace(panel.series[i % 3], series_id=str(i)) for i in range(513)]
    )
    calls = []
    original = BatchBuilder.__init__

    def capture(self, *args, **kwargs):
        calls.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(BatchBuilder, "__init__", capture)
    ids, actual, diagnostics = predictor.forecast_with_diagnostics(panel, origin)
    assert len(calls) == 1
    items = [(i, 29) for i in range(513)]
    aligned = reference_align_to_origin(panel, items)
    cfg = predictor.bundle.forecast_cfg
    expected = []
    counters = dict.fromkeys(("cold_covariates", "clipped_covariates", "floored_covariates"), 0)
    with torch.no_grad():
        for start in range(0, len(items), 512):
            batch = reference_make_batch(
                aligned,
                items[start : start + 512],
                cfg.horizon,
                cfg.max_context,
                predictor.bundle.encoder,
            )
            expected.append(predictor.bundle.model.predict(batch).numpy())
            for field in counters:
                counters[field] += int(getattr(batch, field).sum())
    assert ids == [str(i) for i in range(513)]
    assert diagnostics["n_items"] == 513
    for field, key in zip(
        counters,
        ("cold_covariate_columns", "clipped_covariate_values", "floored_covariate_columns"),
    ):
        assert diagnostics[key] == counters[field]
        if guarded:
            assert counters[field] > 0
    np.testing.assert_array_equal(actual, np.concatenate(expected))
