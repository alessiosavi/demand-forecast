import json
import shutil

import numpy as np
import pandas as pd
import pytest

from demand_forecast.core.bundle import load_bundle
from demand_forecast.core.run import _identity_settings, run_identity, train_panel
from demand_forecast.core.training import InterruptedRun, resolve_budget
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.frequency import to_offset
from tests.helpers.panels import weekly_panel
from tests.helpers.settings import tiny_settings


@pytest.fixture
def tiny_panel_and_settings(panel29, origin29):
    return panel29, origin29, tiny_settings(base_steps=2, d_model=8)


@pytest.fixture
def quick_panel_and_settings():
    panel = weekly_panel([40] * 6)
    settings = tiny_settings(base_steps=20, d_model=8)
    settings.seed = 23
    return panel, panel.timestamps(0, 41)[40], settings


@pytest.mark.parametrize("recipe,expected", [(None, 1), ("v2", 1), ("v3", 7)])
def test_pretraining_frame_resolution_keeps_inner_sql_task_period(
    tmp_path, tiny_pretrained_bundle_dir, monkeypatch, recipe, expected
):
    from dataclasses import replace

    import torch

    from demand_forecast.core import training
    from demand_forecast.core.evaluation import evaluate_prepared
    from demand_forecast.data.scaling import seasonal_error
    from demand_forecast.metrics import sql

    source = shutil.copytree(tiny_pretrained_bundle_dir, tmp_path / "source")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if recipe is None:
        manifest.pop("config", None)
    else:
        manifest["config"]["recipe"] = recipe
        manifest["config"]["period_policy"] = "frequency" if recipe == "v3" else "legacy_one"
    manifest_path.write_text(json.dumps(manifest))
    panel = replace(weekly_panel([40] * 2), freq=to_offset("D"), seasonal_period=14)
    for series in panel.series:
        series.target = np.arange(40, dtype=float)
    origin = panel.timestamps(0, 41)[40]
    settings = tiny_settings(base_steps=2)
    settings.training.init_from = str(source)
    settings.forecast.scaler_period = "pretraining"
    prepared = []
    prepare = training.prepare_validation

    def capture(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(training, "prepare_validation", capture)
    path = train_panel(panel, origin, settings, tmp_path / "run", dataset_fingerprint="fp")
    bundle = load_bundle(path)
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["scaler_period_resolved"]["period"] == expected
    assert report["scaler_period_resolved"]["source"] == "pretraining"
    assert bundle.scaler_period == expected
    assert prepared and prepared[0].seasonal_period == 14
    batch = prepared[0].batches[0]
    assert batch.scale.tolist() == [float(expected)] * 2
    assert seasonal_error(prepared[0].y_past[0], 14) == 14
    with torch.no_grad():
        predictions = bundle.model.predict(batch).numpy()
    measured = evaluate_prepared(bundle.model, prepared[0], settings.forecast.quantiles)["SQL"]
    reference = sql(
        prepared[0].y_true, predictions, prepared[0].y_past, 14, settings.forecast.quantiles
    )
    assert measured == pytest.approx(reference)
    assert measured != pytest.approx(
        sql(
            prepared[0].y_true,
            predictions,
            prepared[0].y_past,
            expected,
            settings.forecast.quantiles,
        )
    )


def test_pretraining_period_without_source_reports_task_fallback(tmp_path, tiny_panel_and_settings):
    panel, origin, settings = tiny_panel_and_settings
    panel.seasonal_period = 7
    settings.forecast.scaler_period = "pretraining"
    path = train_panel(panel, origin, settings, tmp_path, dataset_fingerprint="fp")
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["scaler_period_resolved"] == {
        "requested": "pretraining",
        "source": "task",
        "period": 7,
        "reason": "no_init_from",
    }
    assert load_bundle(path).scaler_period == "task"


def test_scaler_defaults_are_omitted_from_identity():
    settings = tiny_settings()
    assert settings.forecast.scaler_period == "task"
    assert settings.training.loss_weight == "none"
    assert "scaler_period" not in _identity_settings(settings)["forecast"]
    assert "loss_weight" not in _identity_settings(settings)["training"]


def test_loss_weight_counters_are_cumulative_across_resume(tmp_path, monkeypatch):
    from demand_forecast.core import training

    panel = weekly_panel([40] * 3)
    panel.seasonal_period = 7
    panel.series[0].target = 1000 + np.arange(40) * 1e-4
    panel.series[1].target = np.resize([0, 10, 0, 10, 0, 10, 0], 40).astype(float)
    panel.series[2].target[:] = 5
    origin = panel.timestamps(0, 41)[40]
    settings = tiny_settings(base_steps=4, d_model=8)
    settings.forecast.scale_floor_frac = 0.1
    settings.training.loss_weight = "metric_scale"
    settings.training.min_train_history = 1000
    monkeypatch.setattr(
        training, "build_origins", lambda *a, **k: np.array([(i, 28) for i in range(3)])
    )
    with pytest.raises(InterruptedRun):
        train_panel(panel, origin, settings, tmp_path, dataset_fingerprint="fp", stop_after=2)
    path = train_panel(panel, origin, settings, tmp_path, dataset_fingerprint="fp", resume=True)
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["fit_result"]["steps"] == 4
    assert report["loss_weight_clipped"] == 4
    assert report["loss_weight_excluded"] == 8
    assert load_bundle(path).manifest["run"] == report


@pytest.mark.parametrize(
    "section,field", [("forecast", "scaler_period"), ("training", "loss_weight")]
)
def test_scaler_and_loss_weight_reject_invalid_values(section, field):
    from demand_forecast.config import Settings

    data = tiny_settings().model_dump()
    data[section][field] = "invalid"
    with pytest.raises(ValueError, match=field):
        Settings.model_validate(data)


def test_quick_fields_subsample_validation_and_are_reported(
    tmp_path, quick_panel_and_settings, caplog
):
    panel, origin, settings = quick_panel_and_settings
    eligible = plan_inner_validation(panel, origin, settings.forecast.horizon)
    assert len(eligible.val_items) >= 4
    assert resolve_budget(settings.training.base_steps, len(eligible.fit_ids), 5)[0] > 7
    quick = settings.model_copy(deep=True)
    quick.training.val_items_max = 2
    quick.training.max_steps_cap = 7
    with caplog.at_level("INFO", logger="demand_forecast.core.run"):
        bundle = train_panel(panel, origin, quick, tmp_path / "q", dataset_fingerprint="fp")
    report = json.loads((tmp_path / "q/run.json").read_text())
    assert report["val_items_max"] == 2 and report["n_val_items"] == 2
    assert report["n_val_items_eligible"] == len(eligible.val_items)
    assert report["resolved_budget"] == {"max_steps": 7, "eval_every": 1}
    assert report["fit_result"]["steps"] <= 7
    assert report["plan"]["train_upto"] == eligible.train_upto.isoformat()
    assert report["plan"]["val_origin"] == eligible.val_origin.isoformat()
    assert report["plan"]["fit_ids"] == eligible.fit_ids
    assert report["plan"]["exception"] == eligible.exception
    assert report["plan"]["val_items"] == sorted(
        np.random.default_rng(quick.seed).choice(eligible.val_items, size=2, replace=False).tolist()
    )
    assert "val_items=2/6" in caplog.text
    assert load_bundle(bundle).manifest["run"] == report


@pytest.mark.parametrize("field", ["max_steps_cap", "val_items_max"])
def test_identity_ignores_unset_quick_fields_and_changes_when_set(field):
    settings = tiny_settings()
    dumped = settings.model_dump(mode="json")
    base = run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None)
    assert dumped["training"]["max_steps_cap"] is None
    assert dumped["training"]["val_items_max"] is None
    expected = settings.model_dump(mode="json")
    del expected["training"]["max_steps_cap"], expected["training"]["val_items_max"]
    for key in (
        "init_from",
        "zero_shot",
        "eval_at_start",
        "trainable",
        "series_embedding",
        "warmup_frac",
        "min_steps",
        "selection",
        "weight_average",
        "ema_decay",
        "average_last_k",
        "select_min_gain",
        "exception_policy",
        "loss_weight",
    ):
        del expected["training"][key]
    del expected["forecast"]["scale_floor_frac"]
    del expected["forecast"]["scaler_period"]
    assert _identity_settings(settings) == expected
    assert settings.model_dump(mode="json") == dumped
    quick = settings.model_copy(deep=True)
    setattr(quick.training, field, 300)
    assert _identity_settings(quick)["training"][field] == 300
    assert run_identity(quick, 0, "fp", "2.0.0", "2023-01-01", None) != base


@pytest.fixture
def pretrained_bundle_manifest(tmp_path):
    directory = tmp_path / "pretrained"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"pretrain_identity": "a" * 64, "sha256": {"model.pt": "b" * 64}}),
        encoding="utf-8",
    )
    return directory


def test_identity_uses_bundle_ids_not_path(tmp_path, pretrained_bundle_manifest):
    settings = tiny_settings()
    base = run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None)
    copied = shutil.copytree(pretrained_bundle_manifest, tmp_path / "copied")
    identities = []
    for directory in (pretrained_bundle_manifest, copied):
        settings.training.init_from = str(directory)
        dumped = settings.model_dump(mode="json")
        assert _identity_settings(settings)["training"]["init_from"] == {
            "pretrain_identity": "a" * 64,
            "model_sha256": "b" * 64,
        }
        identities.append(run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None))
        assert settings.model_dump(mode="json") == dumped
    assert identities[0] == identities[1] != base


@pytest.mark.parametrize("field", ["pretrain_identity", "model.pt"])
def test_identity_changes_with_bundle_provenance(pretrained_bundle_manifest, field):
    settings = tiny_settings()
    settings.training.init_from = str(pretrained_bundle_manifest)
    base = run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None)
    path = pretrained_bundle_manifest / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if field == "pretrain_identity":
        manifest[field] = "c" * 64
    else:
        manifest["sha256"][field] = "c" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None) != base


def test_identity_rejects_bundle_without_pretrain_identity(pretrained_bundle_manifest):
    path = pretrained_bundle_manifest / "manifest.json"
    path.write_text(json.dumps({"sha256": {"model.pt": "b" * 64}}), encoding="utf-8")
    settings = tiny_settings()
    settings.training.init_from = str(pretrained_bundle_manifest)
    with pytest.raises(ValueError, match="pretrain_identity") as caught:
        run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None)
    assert str(pretrained_bundle_manifest) in str(caught.value)


def test_identity_changes_with_zero_shot_and_eval_at_start():
    settings = tiny_settings()
    identities = set()
    for zero_shot, eval_at_start in ((False, False), (True, False), (False, True), (True, True)):
        # CLI overrides supply init_from later; config validation must accept null.
        settings.training = type(settings.training).model_validate(
            {
                **settings.training.model_dump(),
                "init_from": None,
                "zero_shot": zero_shot,
                "eval_at_start": eval_at_start,
            }
        )
        payload = _identity_settings(settings)["training"]
        for field, enabled in (("zero_shot", zero_shot), ("eval_at_start", eval_at_start)):
            if enabled:
                assert payload[field] is True
            else:
                assert field not in payload
        identities.add(run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None))
    assert len(identities) == 4


def test_canonical_identity_payload_omits_unset_new_fields():
    settings = tiny_settings()
    dumped = settings.model_dump(mode="json")
    assert dumped["training"]["init_from"] is None
    assert dumped["training"]["zero_shot"] is False
    assert dumped["training"]["eval_at_start"] is False
    payload = _identity_settings(settings)
    assert {"init_from", "zero_shot", "eval_at_start", "trainable", "series_embedding"}.isdisjoint(
        payload["training"]
    )
    assert "scale_floor_frac" not in payload["forecast"]
    assert (
        payload["training"]["refit"] == "none"
    )  # This existing field remains in legacy identities.
    assert settings.model_dump(mode="json") == dumped


def test_train_panel_records_earliest_origin_and_checks_it_on_reuse(
    tmp_path, tiny_panel_and_settings
):
    panel, origin, settings = tiny_panel_and_settings
    step = to_offset(panel.freq)
    bundle = train_panel(panel, origin, settings, tmp_path / "a", dataset_fingerprint="fp")
    assert load_bundle(bundle).panel_metadata["earliest_origin"] == origin.isoformat()
    assert (
        json.loads((tmp_path / "a/run.json").read_text())["earliest_origin"] == origin.isoformat()
    )
    earlier = origin - 3 * step
    bundle = train_panel(
        panel, origin, settings, tmp_path / "b", dataset_fingerprint="fp", earliest_origin=earlier
    )
    loaded = load_bundle(bundle)
    assert loaded.panel_metadata["earliest_origin"] == earlier.isoformat()
    assert loaded.manifest["run"]["earliest_origin"] == earlier.isoformat()
    assert (
        json.loads((tmp_path / "b/run.json").read_text())["earliest_origin"] == earlier.isoformat()
    )
    with pytest.raises(ValueError, match="^earliest_origin after origin$"):
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "c",
            dataset_fingerprint="fp",
            earliest_origin=origin + step,
        )
    before = {path: path.read_bytes() for path in (tmp_path / "b").rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="^bundle earliest_origin mismatch$"):
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "b",
            dataset_fingerprint="fp",
            resume=True,
            earliest_origin=origin,
        )
    assert (
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "b",
            dataset_fingerprint="fp",
            resume=True,
            earliest_origin=earlier,
        )
        == bundle
    )
    assert (
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "b",
            dataset_fingerprint="fp",
            resume=True,
        )
        == bundle
    )
    assert before == {
        path: path.read_bytes() for path in (tmp_path / "b").rglob("*") if path.is_file()
    }


def test_train_panel_rejects_off_grid_earliest_origin(tmp_path, tiny_panel_and_settings):
    panel, origin, settings = tiny_panel_and_settings
    with pytest.raises(ValueError, match="off the .* grid"):
        train_panel(
            panel,
            origin,
            settings,
            tmp_path,
            dataset_fingerprint="fp",
            earliest_origin=origin - pd.Timedelta(days=1),
        )


def test_covariate_counters_reach_run_json_after_stop_and_resume(tmp_path, tiny_panel_and_settings):
    panel, origin, settings = tiny_panel_and_settings
    with pytest.raises(InterruptedRun):
        train_panel(panel, origin, settings, tmp_path, dataset_fingerprint="fp", stop_after=1)
    progress = json.loads((tmp_path / "checkpoints/progress.json").read_text())
    for key in ("cold_covariate_columns", "clipped_covariate_values"):
        assert progress["timings"][key] == 0
    # The seeded first epoch includes series b at origin 2: oil's std is ~0.05,
    # below its relative floor (~0.0511). Later draws need no additional floor.
    assert progress["timings"]["floored_covariate_columns"] == 1
    bundle = train_panel(panel, origin, settings, tmp_path, dataset_fingerprint="fp", resume=True)
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["resumed_from_step"] == 1
    assert report["timings"]["cumulative"]["sessions"] == 2
    for key in ("cold_covariate_columns", "clipped_covariate_values"):
        assert report["timings"]["cumulative"][key] == 0
    assert report["timings"]["cumulative"]["floored_covariate_columns"] == 1
    assert load_bundle(bundle).manifest["run"] == report


def test_train_panel_raises_nonfinite_training_without_bundle_or_report(
    tmp_path, quick_panel_and_settings, monkeypatch
):
    import torch

    from demand_forecast.core import run, training

    panel, origin, settings = quick_panel_and_settings
    settings.training.patience = 100
    out = tmp_path / "failed"
    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))

    def forbidden(*args, **kwargs):
        pytest.fail("failed training must not publish a bundle")

    monkeypatch.setattr(run, "save_bundle", forbidden)
    with (
        pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"),
        pytest.raises(training.NonFiniteTraining) as caught,
    ):
        train_panel(panel, origin, settings, out, dataset_fingerprint="fp")
    expected = (
        "Training stopped after 20 consecutive non-finite losses at step 20; "
        f"checkpoint {out / 'checkpoints/latest.pt'}"
    )
    assert str(caught.value) == expected
    assert isinstance(caught.value, RuntimeError) and not isinstance(caught.value, InterruptedRun)
    assert not (out / "bundle").exists() and not (out / "run.json").exists()
    saved = torch.load(out / "checkpoints/latest.pt", weights_only=False)
    assert saved["terminal_reason"] == "nonfinite_losses" and saved["step"] == 20
    monkeypatch.setattr(training, "pinball_loss", forbidden)
    with pytest.raises(training.NonFiniteTraining) as resumed:
        train_panel(panel, origin, settings, out, dataset_fingerprint="fp", resume=True)
    assert str(resumed.value) == expected
    assert not (out / "bundle").exists() and not (out / "run.json").exists()
    assert json.loads((out / "checkpoints/progress.json").read_text())["stop_reason"] == (
        "nonfinite_losses"
    )


@pytest.mark.parametrize("marker", ["both", "reason", "counter"])
def test_terminal_checkpoint_precedes_reuse_of_an_older_bundle(
    tmp_path, quick_panel_and_settings, monkeypatch, marker
):
    import torch

    from demand_forecast.core import training

    panel, origin, settings = quick_panel_and_settings
    settings.training.patience = 100
    args = (panel, origin, settings, tmp_path)
    train_panel(*args, dataset_fingerprint="fp")
    artifacts = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and "checkpoints" not in path.parts
    }
    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    with (
        pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"),
        pytest.raises(training.NonFiniteTraining),
    ):
        train_panel(*args, dataset_fingerprint="fp")
    path = tmp_path / "checkpoints/latest.pt"
    state = torch.load(path, weights_only=False)
    assert state["terminal_reason"] == "nonfinite_losses" and state["consecutive_nonfinite"] == 20
    if marker == "reason":
        state["consecutive_nonfinite"] = 0
    elif marker == "counter":
        state["terminal_reason"] = None
    torch.save(state, path)

    def forbidden(*args, **kwargs):
        pytest.fail("terminal checkpoint resumed training")

    monkeypatch.setattr(training, "pinball_loss", forbidden)
    with pytest.raises(
        training.NonFiniteTraining, match="20 consecutive non-finite losses at step 20"
    ):
        train_panel(*args, dataset_fingerprint="fp", resume=True)
    assert all(path.read_bytes() == content for path, content in artifacts.items())
    assert json.loads((tmp_path / "checkpoints/progress.json").read_text())["stop_reason"] == (
        "nonfinite_losses"
    )


def test_train_panel_zero_shot_requires_init_from(tmp_path, tiny_panel_and_settings):
    panel, origin, settings = tiny_panel_and_settings
    settings.training.zero_shot = True
    with pytest.raises(ValueError, match="^zero_shot requires init_from$"):
        train_panel(panel, origin, settings, tmp_path, dataset_fingerprint="fp")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.bench
@pytest.mark.parametrize("short_history", [False, True])
def test_train_panel_zero_shot_writes_bundle_progress_and_no_training(
    tmp_path, panel29, origin29, tiny_pretrained_bundle_dir, monkeypatch, short_history
):
    pytest.importorskip("pyarrow")
    from demand_forecast.core import run

    def forbidden(*a, **k):
        pytest.fail("zero-shot must not construct or fit a Trainer")

    monkeypatch.setattr(run.Trainer, "fit", forbidden)
    monkeypatch.setattr(run.Trainer, "__init__", forbidden)
    settings = tiny_settings()
    settings.training.init_from = str(tiny_pretrained_bundle_dir)
    settings.training.zero_shot = True
    settings.training.eval_at_start = True
    if short_history:
        panel29 = weekly_panel([8, 8])
        origin29 = panel29.timestamps(0, 9)[8]
    plan = plan_inner_validation(panel29, origin29, settings.forecast.horizon)
    assert plan.exception is short_history
    bundle_dir = train_panel(panel29, origin29, settings, tmp_path, dataset_fingerprint="fp")
    bundle = load_bundle(bundle_dir)
    report = json.loads((tmp_path / "run.json").read_text())
    progress = json.loads((tmp_path / "checkpoints/progress.json").read_text())
    assert bundle.manifest["zero_shot"] is True and bundle.manifest["run"] == report
    assert report["fit_result"] == {
        "steps": 0,
        "best_step": 0,
        "best_score": None,
        "stop_reason": "zero_shot",
        "nonfinite_losses": 0,
    }
    assert report["resolved_budget"] == {"max_steps": 0, "eval_every": 0}
    assert report["n_val_items"] == 0 and report["amp_dtype"] is None
    assert report["timings"]["training_time_s"] == 0.0 and report["telemetry"] == []
    cumulative = report["timings"]["cumulative"]
    assert cumulative["sessions"] == 1
    assert all(value == 0 for key, value in cumulative.items() if key != "sessions")
    assert progress == {
        "run_identity": report["run_identity"],
        "step": 0,
        "stop_reason": "zero_shot",
        "max_steps": 0,
        "eval_every": 0,
        "n_fit_ids": len(report["plan"]["fit_ids"]),
        "n_val_items": 0,
        "exception": plan.exception,
        "device": "cpu",
        "amp_dtype": None,
        "timings": cumulative,
        "telemetry_tail": [],
    }
    assert not (tmp_path / "checkpoints/latest.pt").exists()
    source = load_bundle(tiny_pretrained_bundle_dir)
    transfer = report["transfer"]
    assert set(transfer) == {"pretrain_identity", "model_sha256", "copied", "zeroed", "fresh"}
    assert transfer["pretrain_identity"] == source.manifest["pretrain_identity"]
    assert transfer["model_sha256"] == source.manifest["sha256"]["model.pt"]
    assert set(transfer["copied"] + transfer["zeroed"] + transfer["fresh"]) == set(
        dict(bundle.model.named_parameters())
    )
    plan = plan_inner_validation(panel29, origin29, settings.forecast.horizon)
    observed = [
        s.target[panel29.timestamps(i) <= plan.train_upto]
        for i, s in enumerate(panel29.series)
        if s.series_id in plan.fit_ids
    ]
    assert bundle.model.fallback_level.item() == pytest.approx(
        np.nanmedian(np.concatenate(observed))
    )
    assert bundle.encoder.vocab_sizes["cat"] > 0
    assert bundle.panel_metadata["training_cutoff"] == plan.train_upto.isoformat()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert (
        train_panel(panel29, origin29, settings, tmp_path, dataset_fingerprint="fp", resume=True)
        == bundle_dir
    )
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.bench
def test_zero_shot_predictor_close_to_source_on_targets_only(
    tmp_path, panel29, origin29, tiny_pretrained_bundle_dir
):
    pytest.importorskip("pyarrow")
    from dataclasses import replace

    import torch

    from demand_forecast.data.encoders import StaticEncoder
    from demand_forecast.data.panel import Roles
    from demand_forecast.data.sampler import make_batch
    from demand_forecast.inference.predictor import Predictor

    settings = tiny_settings()
    settings.training.init_from = str(tiny_pretrained_bundle_dir)
    settings.training.zero_shot = True
    directory = train_panel(panel29, origin29, settings, tmp_path, dataset_fingerprint="fp")
    ids, actual = Predictor.from_bundle(directory).forecast_arrays(panel29, origin29)
    source = load_bundle(tiny_pretrained_bundle_dir)
    uni = replace(
        panel29,
        roles=Roles(panel29.roles.target, (), (), (), ()),
        series=[replace(s, dynamic={}, statics={}) for s in panel29.series],
    )
    batch = make_batch(
        uni,
        [(i, 29) for i in range(len(uni.series))],
        4,
        settings.forecast.max_context,
        StaticEncoder().fit(uni, []),
        calendar=False,
    )
    with torch.no_grad():
        expected = (
            (source.model.eval()(batch) * batch.scale[:, None, None] + batch.loc[:, None, None])
            .clamp_min(0)
            .numpy()
        )
    assert ids == [s.series_id for s in panel29.series]
    assert np.allclose(actual, expected, rtol=0, atol=1e-4 * batch.scale.numpy()[:, None, None])


@pytest.mark.bench
def test_train_panel_init_from_transfers_then_trains(
    tmp_path, panel29, origin29, tiny_pretrained_bundle_dir, monkeypatch
):
    pytest.importorskip("pyarrow")
    import torch

    from demand_forecast.core.run import Trainer

    source = load_bundle(tiny_pretrained_bundle_dir)
    settings = tiny_settings(base_steps=4)
    settings.training.init_from = str(tiny_pretrained_bundle_dir)
    settings.training.eval_at_start = True
    fit = Trainer.fit

    def verify_transferred(trainer, *a, **k):
        for name, tensor in source.model.state_dict().items():
            if name.startswith(("backbone.", "head.")):
                assert torch.equal(trainer.model.state_dict()[name], tensor), name
        assert torch.count_nonzero(trainer.model.enc_proj.weight[:, 2:]) == 0
        assert torch.count_nonzero(trainer.model.dec_proj.weight) == 0
        return fit(trainer, *a, **k)

    monkeypatch.setattr(Trainer, "fit", verify_transferred)
    directory = train_panel(panel29, origin29, settings, tmp_path, dataset_fingerprint="fp")
    report = json.loads((tmp_path / "run.json").read_text())
    saved = torch.load(tmp_path / "checkpoints/latest.pt", weights_only=False)
    assert report["fit_result"]["steps"] == saved["step"] == 4
    assert saved["initial_eval_done"] is True and saved["history"][0]["step"] == 0
    assert saved["optimizer"]["state"]
    assert any(
        not torch.equal(value, saved["model"][key])
        for key, value in source.model.state_dict().items()
        if key.startswith("backbone.")
    )
    assert report["transfer"]["pretrain_identity"] == source.manifest["pretrain_identity"]
    assert "zero_shot" not in load_bundle(directory).manifest


@pytest.mark.parametrize("stop_after", [None, 2])
def test_exception_pretrained_ships_initial_weights_and_skips_refit(
    tmp_path, panel29, origin29, tiny_pretrained_bundle_dir, monkeypatch, stop_after
):
    from copy import deepcopy

    import torch

    from demand_forecast.core.training import Trainer

    settings = tiny_settings(base_steps=4)
    settings.training.init_from = str(tiny_pretrained_bundle_dir)
    settings.training.min_train_history = 1000
    settings.training.exception_policy = "pretrained"
    settings.training.refit = "full"
    fit, initials = Trainer.fit, []

    def capture(self, panel, plan, *args, **kwargs):
        assert plan.exception and kwargs["keep_initial"]
        assert "step_limit" not in kwargs  # no refit invocation
        initials.append(deepcopy(self.model.state_dict()))
        return fit(self, panel, plan, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", capture)
    if stop_after:
        with pytest.raises(InterruptedRun):
            train_panel(
                panel29,
                origin29,
                settings,
                tmp_path,
                dataset_fingerprint="fp",
                stop_after=stop_after,
            )
    bundle_path = train_panel(
        panel29,
        origin29,
        settings,
        tmp_path,
        dataset_fingerprint="fp",
        resume=stop_after is not None,
    )
    bundle = load_bundle(bundle_path)
    report = json.loads((tmp_path / "run.json").read_text())
    checkpoint = torch.load(tmp_path / "checkpoints/latest.pt", weights_only=False)
    assert report["exception_policy_applied"] == "pretrained"
    assert report["fit_result"]["best_step"] == 0
    assert report["fit_result"]["steps"] == 4
    assert report["refit"]["performed"] is False
    assert report["refit"]["reason"] == "no_inner_validation"
    assert report["n_evaluations"] == 0 and report["early_stop_reachable"] is False
    for key, value in bundle.model.state_dict().items():
        assert torch.equal(value, checkpoint["best_state"][key])
        if key != "fallback_level":
            assert torch.equal(value, initials[0][key]), key
    assert not (tmp_path / "checkpoints/refit").exists()


def test_exception_pretrained_without_source_uses_final(tmp_path, tiny_panel_and_settings):
    import torch

    panel, origin, settings = tiny_panel_and_settings
    settings.training.min_train_history = 1000
    settings.training.exception_policy = "pretrained"
    path = train_panel(panel, origin, settings, tmp_path, dataset_fingerprint="fp")
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["exception_policy_applied"] == "final"
    assert report["fit_result"]["best_step"] == report["fit_result"]["steps"] == 2
    saved = torch.load(tmp_path / "checkpoints/latest.pt", weights_only=False)
    assert all(
        torch.equal(value, saved["model"][key])
        for key, value in load_bundle(path).model.state_dict().items()
    )
