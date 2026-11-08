"""Offline acceptance checks for optional fine-tune protocols."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from demand_forecast.config import ForecastConfig, TrainingConfig
from demand_forecast.core import training
from demand_forecast.core.bundle import load_bundle
from demand_forecast.core.run import _identity_settings, run_identity, train_panel
from demand_forecast.core.training import InterruptedRun, Trainer
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.panel import Roles
from demand_forecast.data.sampler import BatchBuilder
from demand_forecast.data.scaling import sample_frame
from demand_forecast.inference.predictor import Predictor
from tests.helpers.panels import weekly_panel
from tests.helpers.settings import tiny_settings


@pytest.fixture
def protocol_case():
    panel = weekly_panel([40] * 3)
    return panel, panel.timestamps(0, 41)[40], tiny_settings(base_steps=6, d_model=8)


def _run(case, directory, **kwargs):
    panel, origin, settings = case
    bundle = train_panel(panel, origin, settings, directory, dataset_fingerprint="fp", **kwargs)
    return load_bundle(bundle), json.loads((directory / "run.json").read_text())


def _select_step(monkeypatch, step):
    original = Trainer._select_best

    def select(self, score):
        original(self, float(abs(self.step - step) + 1))

    monkeypatch.setattr(Trainer, "_select_best", select)


def _assert_weights_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_refit_full_uses_same_initial_weights_full_cutoff_and_selection_schedule(
    tmp_path, protocol_case, monkeypatch
):
    panel, origin, settings = protocol_case
    settings.training.refit = "full"
    _select_step(monkeypatch, 3)
    calls = []
    fit = Trainer.fit

    def capture(self, panel, plan, *args, **kwargs):
        initial = deepcopy(self.model.state_dict())
        result = fit(self, panel, plan, *args, **kwargs)
        calls.append((self, plan, initial, result))
        return result

    monkeypatch.setattr(Trainer, "fit", capture)
    bundle, report = _run(protocol_case, tmp_path)
    selection, refit = calls
    assert report["fit_result"]["best_step"] == 3
    assert report["fit_result"]["steps"] == 6
    assert report["refit"] == {
        "mode": "full",
        "performed": True,
        "steps": 3,
        "train_upto": panel.timestamps(0)[-1].isoformat(),
        "reason": None,
    }
    assert bundle.panel_metadata["training_cutoff"] == report["refit"]["train_upto"]
    assert refit[1].exception and not len(refit[1].val_items)
    assert refit[1].train_upto < origin
    _assert_weights_equal(selection[2], refit[2])
    _assert_weights_equal(bundle.model.state_dict(), refit[0].model.state_dict())
    assert refit[0].step == refit[0].step_limit == 3
    assert refit[0].max_steps == selection[0].max_steps == 6
    assert refit[0].warmup_steps == selection[0].warmup_steps == 2
    assert refit[0].optimizer.param_groups[0]["lr"] == pytest.approx(
        settings.training.learning_rate * training.lr_factor(3, max_steps=6, warmup_steps=2)
    )
    assert all("SQL" not in record for record in refit[3].history)
    assert report["validation_history"] == [r for r in selection[3].history if "SQL" in r]
    for key in ("train_s", "eval_s", "checkpoint_s", "floored_target_samples"):
        assert report["timings"]["cumulative"][key] == sum(c[3].timings[key] for c in calls)


@pytest.mark.parametrize("reason", ["best_step_zero", "no_inner_validation"])
def test_refit_skips_zero_step_and_exception_with_reason(
    tmp_path, protocol_case, monkeypatch, reason
):
    settings = protocol_case[2]
    settings.training.refit = "full"
    if reason == "best_step_zero":
        settings.training.eval_at_start = True
        _select_step(monkeypatch, 0)
    else:
        settings.training.min_train_history = 1000
    _, report = _run(protocol_case, tmp_path)
    assert report["refit"] == {
        "mode": "full",
        "performed": False,
        "steps": None,
        "train_upto": None,
        "reason": reason,
    }
    assert not (tmp_path / "checkpoints/refit").exists()
    if reason == "best_step_zero":
        checkpoint = torch.load(tmp_path / "checkpoints/latest.pt", weights_only=False)
        _assert_weights_equal(
            load_bundle(tmp_path / "bundle").model.state_dict(), checkpoint["best_state"]
        )


@pytest.mark.parametrize("stop_after", [6, 7])
def test_refit_interruption_resumes_refit_without_selection(
    tmp_path, protocol_case, monkeypatch, stop_after
):
    settings = protocol_case[2]
    settings.training.refit = "full"
    _select_step(monkeypatch, 3)
    expected, expected_report = _run(protocol_case, tmp_path / "whole")
    interrupted = tmp_path / "interrupted"
    with pytest.raises(InterruptedRun):
        _run(protocol_case, interrupted, stop_after=stop_after)
    assert (interrupted / "checkpoints/selection.json").exists()
    if stop_after == 7:
        state = torch.load(interrupted / "checkpoints/refit/latest.pt", weights_only=False)
        assert state["step"] == 1 and state["max_steps"] == 6 and state["step_limit"] == 3
    selection_bytes = (interrupted / "checkpoints/latest.pt").read_bytes()
    fit = Trainer.fit

    def only_refit(self, panel, plan, *args, **kwargs):
        assert plan.exception and kwargs["step_limit"] == 3
        return fit(self, panel, plan, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", only_refit)
    resumed, report = _run(protocol_case, interrupted, resume=True)
    _assert_weights_equal(expected.model.state_dict(), resumed.model.state_dict())
    assert (interrupted / "checkpoints/latest.pt").read_bytes() == selection_bytes
    assert report["fit_result"] == expected_report["fit_result"]
    assert report["validation_history"] == expected_report["validation_history"]
    assert report["refit"] == expected_report["refit"]
    with pytest.raises(ValueError, match="identity mismatch"):
        protocol_case[2].training.learning_rate *= 2
        _run(protocol_case, interrupted, resume=True)


def test_zero_shot_refit_full_rejected_at_validation_and_runtime(tmp_path, protocol_case):
    with pytest.raises(ValueError, match="zero_shot.*refit"):
        TrainingConfig(zero_shot=True, refit="full")
    settings = protocol_case[2]
    settings.training.zero_shot, settings.training.refit = True, "full"
    settings.training.init_from = "unused"
    with pytest.raises(ValueError, match="zero_shot.*refit"):
        _run(protocol_case, tmp_path)


def test_heads_only_freezes_backbone_updates_heads_and_records_counts(
    tmp_path, protocol_case, monkeypatch
):
    protocol_case[2].training.trainable = "heads_only"
    fit = Trainer.fit
    seen = []

    def capture(self, *args, **kwargs):
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        result = fit(self, *args, **kwargs)
        changed = []
        for name, parameter in self.model.named_parameters():
            if name.startswith("backbone."):
                assert not parameter.requires_grad
                assert torch.equal(before[name], parameter), name
            elif not torch.equal(before[name], parameter):
                changed.append(name)
        assert changed
        assert any(name.startswith("head.") for name in changed)
        assert all(
            p.requires_grad for group in self.optimizer.param_groups for p in group["params"]
        )
        seen.append(self.model)
        return result

    monkeypatch.setattr(Trainer, "fit", capture)
    _, report = _run(protocol_case, tmp_path)
    model = seen[0]
    assert report["trainable"] == {
        "mode": "heads_only",
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params_total": sum(p.numel() for p in model.parameters()),
    }
    assert 0 < report["trainable"]["params_trainable"] < report["trainable"]["params_total"]


def test_heads_only_checkpoint_resume_is_exact(tmp_path, protocol_case):
    protocol_case[2].training.trainable = "heads_only"
    expected, _ = _run(protocol_case, tmp_path / "whole")
    with pytest.raises(InterruptedRun):
        _run(protocol_case, tmp_path / "parts", stop_after=2)
    resumed, report = _run(protocol_case, tmp_path / "parts", resume=True)
    _assert_weights_equal(expected.model.state_dict(), resumed.model.state_dict())
    assert report["resumed_from_step"] == 2


@pytest.mark.parametrize("with_statics", [True, False])
def test_series_embedding_false_roundtrips_and_predictor_loads(
    tmp_path, protocol_case, with_statics
):
    panel, origin, settings = protocol_case
    settings.training.series_embedding = False
    if not with_statics:
        panel = replace(panel, roles=Roles("y", ("promo",), ("oil",), (), ()))
    bundle, _ = _run((panel, origin, settings), tmp_path)
    encoded = bundle.encoder.to_json()
    assert "__series__" not in encoded["vocabularies"]
    assert StaticEncoder.from_json(encoded).to_json() == encoded
    panel.series[0].series_id = "later-unseen-id"
    ids, values = Predictor(bundle).forecast_arrays(panel, origin)
    assert ids[0] == "later-unseen-id" and np.isfinite(values).all()


def test_pretrained_transfer_with_heads_only_and_no_series_embedding(
    tmp_path, tiny_pretrained_bundle_dir
):
    panel = weekly_panel([40] * 2)
    settings = tiny_settings(base_steps=4)
    settings.training.init_from = str(tiny_pretrained_bundle_dir)
    settings.training.trainable = "heads_only"
    settings.training.series_embedding = False
    settings.training.refit = "full"
    bundle, report = _run((panel, panel.timestamps(0, 41)[40], settings), tmp_path)
    source = load_bundle(tiny_pretrained_bundle_dir)
    _assert_weights_equal(bundle.model.backbone.state_dict(), source.model.backbone.state_dict())
    assert report["transfer"]["pretrain_identity"] == source.manifest["pretrain_identity"]
    assert "__series__" not in bundle.encoder.vocabularies


def test_scale_floor_zero_preserves_frame_and_positive_floor_uses_cropped_context():
    past = np.array([1.0, 1000.0, 1000.001, 1000.002])
    context = past[-2:]
    legacy = sample_frame(context, past, 1)
    assert sample_frame(context, past, 1, scale_floor_frac=0.0) == legacy
    # The cropped context controls the relative floor, while raw scale uses full past.
    frame = sample_frame(context, context, 1, scale_floor_frac=0.1)
    assert frame.scale == pytest.approx(100.00015)
    assert frame.scale_floored and frame.eligible
    constant = sample_frame(context[:1], context[:1], 1, scale_floor_frac=0.1)
    assert not constant.eligible and constant.scale == 1.0 and not constant.scale_floored


def test_scale_floor_counts_training_and_prediction_samples(tmp_path, protocol_case):
    panel, origin, settings = protocol_case
    for series in panel.series:
        series.target = 1000.0 + np.arange(40) * 1e-3
    settings.forecast.scale_floor_frac = 0.1
    settings.forecast.max_context = 8
    bundle, report = _run(protocol_case, tmp_path)
    assert (
        report["timings"]["cumulative"]["floored_target_samples"]
        == 3 * report["fit_result"]["steps"]
    )
    _, values, diagnostics = Predictor(bundle).forecast_with_diagnostics(panel, origin)
    assert diagnostics["floored_target_samples"] == 3 and np.isfinite(values).all()
    builder = BatchBuilder(panel, 4, 8, bundle.encoder, scale_floor_frac=0.1)
    batch = builder.build([(0, 40)])
    assert builder.floored_target_samples == 1
    assert batch.scale.item() == pytest.approx(0.1 * panel.series[0].target[-8:].mean())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "trainable", "heads_only"),
        ("training", "series_embedding", False),
        ("training", "refit", "full"),
        ("training", "warmup_frac", 0.1),
        ("training", "min_steps", 1000),
        ("training", "selection", "smoothed"),
        ("training", "weight_average", "ema"),
        ("training", "ema_decay", 0.9),
        ("training", "average_last_k", 2),
        ("training", "select_min_gain", 0.05),
        ("training", "exception_policy", "pretrained"),
        ("forecast", "scale_floor_frac", 0.1),
    ],
)
def test_protocol_nondefault_changes_identity(section, field, value):
    settings = tiny_settings()
    args = (0, "fp", "2.0.0", pd.Timestamp("2023-01-01"), None)
    before = run_identity(settings, *args)
    setattr(getattr(settings, section), field, value)
    assert _identity_settings(settings)[section][field] == value
    assert run_identity(settings, *args) != before


@pytest.mark.parametrize("value", [-0.1, 1.0, float("nan"), float("inf")])
def test_scale_floor_rejects_invalid_fractions(value):
    with pytest.raises(ValueError, match="scale_floor_frac"):
        ForecastConfig(horizon=4, scale_floor_frac=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_frac", None),
        ("min_steps", None),
        ("selection", "best"),
        ("weight_average", "none"),
        ("ema_decay", 0.99),
        ("average_last_k", 3),
        ("select_min_gain", 0.0),
        ("exception_policy", "final"),
    ],
)
def test_hygiene_defaults_are_omitted_from_identity(field, value):
    settings = tiny_settings()
    before = _identity_settings(settings)
    setattr(settings.training, field, value)
    assert field not in _identity_settings(settings)["training"]
    assert _identity_settings(settings) == before


@pytest.mark.parametrize(
    "values",
    [
        {"warmup_frac": -0.1},
        {"warmup_frac": 1.1},
        {"warmup_frac": float("nan")},
        {"min_steps": 0},
        {"selection": "median"},
        {"weight_average": "swa"},
        {"ema_decay": 1.0},
        {"ema_decay": -0.1},
        {"average_last_k": 0},
        {"select_min_gain": -0.1},
        {"select_min_gain": 1.0},
        {"exception_policy": "best"},
    ],
)
def test_hygiene_config_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


@pytest.mark.parametrize("average", ["none", "ema", "tail"])
def test_fractional_warmup_is_reused_by_refit(tmp_path, protocol_case, monkeypatch, average):
    settings = protocol_case[2]
    settings.training.refit = "full"
    settings.training.warmup_frac = 0.5
    settings.training.weight_average = average
    _select_step(monkeypatch, 3)
    calls = []
    fit = Trainer.fit

    def capture(self, *args, **kwargs):
        result = fit(self, *args, **kwargs)
        calls.append((self, result))
        return result

    monkeypatch.setattr(Trainer, "fit", capture)
    bundle, report = _run(protocol_case, tmp_path)
    assert [c[0].warmup_steps for c in calls] == [3, 3]
    assert report["resolved_warmup_steps"] == 3
    assert report["n_evaluations"] == 6
    assert report["early_stop_reachable"] is True
    _, result = calls[-1]
    _assert_weights_equal(bundle.model.state_dict(), result.best_state)


@pytest.mark.parametrize("stop_after", [2, 6, 7])
def test_refit_unfinished_rerun_takes_precedence_over_old_bundle(
    tmp_path, protocol_case, monkeypatch, stop_after
):
    protocol_case[2].training.refit = "full"
    _select_step(monkeypatch, 3)
    expected, _ = _run(protocol_case, tmp_path)
    with pytest.raises(InterruptedRun):
        _run(protocol_case, tmp_path, stop_after=stop_after)
    fit = Trainer.fit
    calls = []

    def capture(self, *args, **kwargs):
        calls.append(kwargs.get("step_limit"))
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", capture)
    resumed, _ = _run(protocol_case, tmp_path, resume=True)
    assert 3 in calls
    _assert_weights_equal(expected.model.state_dict(), resumed.model.state_dict())


def test_refit_full_plan_keeps_initial_vocabulary_for_newly_observed_series(
    tmp_path, protocol_case, monkeypatch
):
    panel, origin, settings = protocol_case
    panel.series[0].target[:-4] = np.nan
    settings.training.refit = "full"
    _select_step(monkeypatch, 3)
    fit = Trainer.fit
    refit_ids = []

    def capture(self, panel, plan, *args, **kwargs):
        if kwargs.get("step_limit"):
            refit_ids.extend(plan.fit_ids)
        return fit(self, panel, plan, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", capture)
    bundle, report = _run(protocol_case, tmp_path)
    assert panel.series[0].series_id not in report["plan"]["fit_ids"]
    assert panel.series[0].series_id in refit_ids
    assert bundle.encoder.transform(panel.series[0])[0]["__series__"] == 0


def test_per_window_policy_changes_identity_only_when_enabled():
    settings = tiny_settings()
    args = (settings, 0, "fp", "2.0.0", "2023-01-01", None)
    default = run_identity(*args)
    assert run_identity(*args, policy="first_window") == default
    assert run_identity(*args, policy="per_window") != default


def test_scale_floor_validation_uses_same_frame_as_training(protocol_case):
    from demand_forecast.core.evaluation import prepare_validation
    from demand_forecast.core.validation import plan_inner_validation
    from demand_forecast.data.frequency import steps_between

    panel, origin, settings = protocol_case
    for series in panel.series:
        series.target = 1000.0 + np.arange(40) * 1e-3
    plan = plan_inner_validation(panel, origin, 4)
    encoder = StaticEncoder().fit(panel, plan.fit_ids)
    prepared = prepare_validation(
        panel, plan.val_items, plan.val_origin, 4, 8, encoder, scale_floor_frac=0.1
    )
    items = [
        (int(i), steps_between(panel.series[i].start, plan.val_origin, panel.freq))
        for i in plan.val_items
    ]
    training_batch = BatchBuilder(panel, 4, 8, encoder, scale_floor_frac=0.1).build(items)
    assert torch.equal(prepared.batches[0].scale, training_batch.scale)
    assert torch.equal(prepared.batches[0].target, training_batch.target)
    assert (training_batch.scale > 100).all()


@pytest.mark.parametrize("mode", ["ema", "tail"])
@pytest.mark.parametrize("refit", ["none", "full"])
def test_averaged_model_resume_exact(tmp_path, protocol_case, monkeypatch, mode, refit):
    settings = protocol_case[2]
    settings.training.weight_average = mode
    settings.training.selection = "smoothed"
    settings.training.ema_decay = 0.5
    settings.training.average_last_k = 2
    settings.training.eval_at_start = True
    settings.training.refit = refit
    # Force a positive selected step for refit, retaining the actual smoothed code.
    original = Trainer._select_best

    def select(self, score):
        return original(self, float(10 - self.step))

    monkeypatch.setattr(Trainer, "_select_best", select)
    expected, before = _run(protocol_case, tmp_path / "whole")
    # Pause in selection, or in the refit after selection completed.
    with pytest.raises(InterruptedRun):
        _run(protocol_case, tmp_path / "parts", stop_after=2 if refit == "none" else 8)
    resumed, report = _run(protocol_case, tmp_path / "parts", resume=True)
    _assert_weights_equal(expected.model.state_dict(), resumed.model.state_dict())
    assert report["fit_result"] == before["fit_result"]
    assert report["validation_history"] == before["validation_history"]


@pytest.mark.parametrize("mode", ["none", "ema", "tail"])
def test_completed_smoothed_early_stop_refit_reuses_bundle(
    tmp_path, protocol_case, monkeypatch, mode
):
    settings = protocol_case[2]
    settings.training.selection = "smoothed"
    settings.training.patience = 1
    settings.training.refit = "full"
    settings.training.weight_average = mode
    original = Trainer._select_best

    def select(self, score):
        original(self, {1: 1.2, 2: 0.8, 3: 1.0}[self.step])

    monkeypatch.setattr(Trainer, "_select_best", select)
    _, report = _run(protocol_case, tmp_path)
    assert report["fit_result"]["steps"] == report["fit_result"]["best_step"] == 3
    assert report["fit_result"]["stop_reason"] == "early_stop"
    assert report["refit"]["steps"] == 3
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(Trainer, "fit", lambda *a, **k: pytest.fail("completed run retrained"))
    _run(protocol_case, tmp_path, resume=True)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
