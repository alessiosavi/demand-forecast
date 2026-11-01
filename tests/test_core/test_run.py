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
    for key in ("init_from", "zero_shot", "eval_at_start"):
        del expected["training"][key]
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
    assert {"init_from", "zero_shot", "eval_at_start"}.isdisjoint(payload["training"])
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
def test_train_panel_zero_shot_writes_bundle_progress_and_no_training(
    tmp_path, panel29, origin29, tiny_pretrained_bundle_dir, monkeypatch
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
        "exception": False,
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
