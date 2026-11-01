"""Tiny CPU training, validation and exact continuation of interrupted runs."""

import json
import random
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core.bundle import save_bundle
from demand_forecast.core.evaluation import evaluate
from demand_forecast.core.training import InterruptedRun, TrainConfig, Trainer, resolve_budget
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.sampler import make_batch
from demand_forecast.inference.predictor import Predictor
from demand_forecast.metrics import summarize
from demand_forecast.models.base import ModelSpec
from tests.helpers.models import build_v2
from tests.helpers.panels import weekly_panel

SEED = 3
RUN_ID = "test-run"
cpu = torch.device("cpu")


@pytest.fixture
def fresh_for():
    def build(panel, horizon):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        ids = [s.series_id for s in panel.series if np.isfinite(s.target[: s.length]).any()]
        encoder = StaticEncoder().fit(panel, ids)
        spec = ModelSpec.from_panel(panel, encoder, ForecastConfig(horizon=horizon), 16, 4)
        return build_v2(spec)

    return build


@pytest.fixture
def fresh(fresh_for, panel_long):
    return lambda: fresh_for(panel_long, 4)


@pytest.fixture
def fresh29(fresh_for, panel29):
    return lambda: fresh_for(panel29, 13)


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[k], v) for k, v in expected.items())


def test_fit_prepares_validation_once(
    tmp_path, monkeypatch, fresh, panel_long, plan_long, encoder_long, caplog
):
    from demand_forecast.core import training

    prepare = training.prepare_validation
    calls = []

    def capture(panel, items, *args, **kwargs):
        calls.append(items.copy())
        return prepare(panel, items, *args, **kwargs)

    monkeypatch.setattr(training, "prepare_validation", capture)
    plan = replace(plan_long, val_items=plan_long.val_items[[1, 5, 7]])
    cfg = TrainConfig(base_steps=4, batch_size=8, amp=False)
    forecast = ForecastConfig(horizon=4)
    with caplog.at_level("INFO"):
        Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
            panel_long, plan, forecast, encoder_long, checkpoint_dir=tmp_path, stop_after=2
        )
    assert len(calls) == 1
    assert "validation prepared: items=3 batches=1 bytes=" in caplog.text
    Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long, plan, forecast, encoder_long, checkpoint_dir=tmp_path, resume=True
    )
    assert len(calls) == 2
    for items in calls:
        np.testing.assert_array_equal(items, plan.val_items)
    for skipped in (
        replace(plan, exception=True),
        replace(plan, val_items=np.array([], dtype=int)),
    ):
        Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(panel_long, skipped, forecast, encoder_long)
    assert len(calls) == 2


@pytest.fixture(params=["ordinary", "guarded"])
def reference_path_case(request, monkeypatch, panel_long):
    if request.param == "guarded":
        from demand_forecast.core import training

        for i, series in enumerate(panel_long.series):
            series.dynamic["promo"] = (
                np.r_[np.full(30, np.nan), np.full(50, 1e5)],
                np.r_[1e-9, np.zeros(27), np.full(52, 0.5)],
                np.r_[100.0 + 1e-6, np.full(27, 100.0), np.full(52, 110.0)],
            )[i % 3]
            series.dynamic["oil"][1:3] = [np.inf, -np.inf]
        monkeypatch.setattr(
            training, "build_origins", lambda *a, **k: np.array([(i, 28) for i in range(20)])
        )
    return request.param


def test_fit_matches_reference_path(
    monkeypatch, fresh, panel_long, plan_long, encoder_long, reference_path_case
):
    from demand_forecast.data.sampler import WindowSampler
    from tests.helpers.reference_batch import reference_batches, reference_evaluate

    cfg = TrainConfig(base_steps=6, batch_size=8, warmup_steps=2, patience=10, amp=False)
    forecast = ForecastConfig(horizon=4)
    optimized = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long, plan_long, forecast, encoder_long
    )
    with monkeypatch.context() as patch:
        patch.setattr(WindowSampler, "batches", reference_batches)
        reference = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
            panel_long, plan_long, forecast, encoder_long, evaluate_fn=reference_evaluate
        )
    np.testing.assert_equal(optimized.history, reference.history)
    assert_state_equal(optimized.best_state, reference.best_state)
    assert (optimized.best_step, optimized.stop_reason) == (
        reference.best_step,
        reference.stop_reason,
    )
    for key in ("cold_covariate_columns", "clipped_covariate_values", "floored_covariate_columns"):
        assert optimized.timings[key] == reference.timings[key]
        if reference_path_case == "guarded":
            assert optimized.timings[key] > 0


def test_reference_checkpoint_resumes_on_optimized_path(
    tmp_path, monkeypatch, fresh, panel_long, plan_long, encoder_long, reference_path_case
):
    from shutil import copyfile

    from demand_forecast.data.sampler import WindowSampler
    from tests.helpers.reference_batch import reference_batches, reference_evaluate

    cfg = TrainConfig(base_steps=6, batch_size=8, warmup_steps=2, patience=10, amp=False)
    forecast = ForecastConfig(horizon=4)
    reference_dir, optimized_dir = tmp_path / "reference", tmp_path / "optimized"
    optimized_dir.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(WindowSampler, "batches", reference_batches)
        interrupted = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
            panel_long,
            plan_long,
            forecast,
            encoder_long,
            checkpoint_dir=reference_dir,
            stop_after=2,
            evaluate_fn=reference_evaluate,
        )
        assert interrupted.stop_reason == "interrupted"
        rewrite_scaler_timings(reference_dir / "latest.pt")
        copyfile(reference_dir / "latest.pt", optimized_dir / "latest.pt")
        reference_trainer = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
        reference = reference_trainer.fit(
            panel_long,
            plan_long,
            forecast,
            encoder_long,
            checkpoint_dir=reference_dir,
            resume=True,
            evaluate_fn=reference_evaluate,
        )
    optimized_trainer = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    optimized = optimized_trainer.fit(
        panel_long,
        plan_long,
        forecast,
        encoder_long,
        checkpoint_dir=optimized_dir,
        resume=True,
    )

    def equal(actual, expected):
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                equal(a, b)
        else:
            np.testing.assert_equal(actual, expected)

    for state in ("model", "optimizer", "scheduler", "scaler"):
        equal(
            getattr(optimized_trainer, state).state_dict(),
            getattr(reference_trainer, state).state_dict(),
        )
    assert optimized_trainer.sampler.state() == reference_trainer.sampler.state()
    assert optimized.timings["scaler_skips"] is reference.timings["scaler_skips"] is None
    for key in ("cold_covariate_columns", "clipped_covariate_values", "floored_covariate_columns"):
        assert optimized.timings[key] == reference.timings[key]
        if reference_path_case == "guarded":
            assert optimized.timings[key] > 0
    for field in ("history", "best_state", "best_step", "best_score", "steps", "stop_reason"):
        equal(getattr(optimized, field), getattr(reference, field))
    # The saved model is the live optimizer pair; the returned model is best-state.
    actual = torch.load(optimized_dir / "latest.pt", weights_only=False)
    expected = torch.load(reference_dir / "latest.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "scaler", "sampler", "rng", "patience_counter"):
        equal(actual[key], expected[key])


def test_fit_custom_evaluate_fn_keeps_old_call(
    monkeypatch, fresh, panel_long, plan_long, encoder_long
):
    from demand_forecast.core import training

    def forbidden(*args, **kwargs):
        pytest.fail("custom evaluator must not prepare validation")

    monkeypatch.setattr(training, "prepare_validation", forbidden)
    model, forecast, calls = fresh(), ForecastConfig(horizon=4), []

    def custom(*args):
        calls.append(args)
        return {"SQL": 1.0}

    result = Trainer(model, TrainConfig(base_steps=2, amp=False), cpu, RUN_ID, SEED).fit(
        panel_long, plan_long, forecast, encoder_long, evaluate_fn=custom
    )
    assert len(calls) == 2 and result.best_score == 1.0
    for args in calls:
        assert len(args) == 8
        assert args[0] is model and args[1] is panel_long and args[2] is plan_long.val_items
        assert args[3:6] == (plan_long.val_origin, 4, forecast.max_context)
        assert args[6] is encoder_long and args[7] is forecast.quantiles


def test_prepare_time_charged_once(
    tmp_path, monkeypatch, fresh, panel_long, plan_long, encoder_long
):
    from demand_forecast.core import training

    now = [0.0]
    prepare, score = training.prepare_validation, training.evaluate_prepared
    monkeypatch.setattr(training.time, "perf_counter", lambda: now[0])

    def timed_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        now[0] += 7.0
        return result

    def timed_score(*args, **kwargs):
        result = score(*args, **kwargs)
        now[0] += 3.0
        return result

    monkeypatch.setattr(training, "prepare_validation", timed_prepare)
    monkeypatch.setattr(training, "evaluate_prepared", timed_score)
    cfg, forecast = TrainConfig(base_steps=4, amp=False), ForecastConfig(horizon=4)
    first = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long, plan_long, forecast, encoder_long, checkpoint_dir=tmp_path, stop_after=2
    )
    assert first.timings["eval_s"] == 13.0
    resumed = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long, plan_long, forecast, encoder_long, checkpoint_dir=tmp_path, resume=True
    )
    assert resumed.timings["eval_s"] == 26.0
    assert [row["eval_s"] for row in resumed.telemetry] == [3.0] * 4


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((7, 5), torch.float16),
        ((8, 0), torch.bfloat16),
        ((8, 6), torch.bfloat16),
        ((9, 0), torch.bfloat16),
    ],
)
def test_amp_dtype_for_uses_native_bf16_rule(monkeypatch, capability, expected):
    from demand_forecast.core.training import amp_dtype_for

    devices = []

    def get_capability(device):
        devices.append(device)
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    device = torch.device("cuda", 1)
    assert amp_dtype_for(device) is expected
    assert devices == [device]
    assert amp_dtype_for(cpu) is None
    assert devices == [device]


def test_resolve_amp_wiring(monkeypatch):
    from demand_forecast.core.training import resolve_amp

    def unexpected_query(*args):
        pytest.fail("disabled AMP queried CUDA capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_query)
    assert resolve_amp(True, cpu) == (False, torch.float16, False)
    assert resolve_amp(False, torch.device("cuda")) == (False, torch.float16, False)
    devices = []

    def get_capability(device):
        devices.append(device)
        return (7, 5) if device.index == 1 else (8, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    device = torch.device("cuda", 1)
    assert resolve_amp(True, device) == (True, torch.float16, True)
    assert devices == [device] and devices[0].index == 1
    assert resolve_amp(True, torch.device("cuda")) == (True, torch.bfloat16, False)


@pytest.mark.parametrize(
    "base_steps,n_series,eval_every,expected",
    [(3000, 118, 200, (500, 100)), (3000, 30490, 200, (12000, 200)), (60, 3, 200, (60, 12))],
)
def test_resolve_budget_examples(base_steps, n_series, eval_every, expected):
    assert resolve_budget(base_steps, n_series, eval_every) == expected


@pytest.mark.parametrize(
    "args,expected",
    [
        ((3000, 118, 200, None), (500, 100)),
        ((3000, 30490, 200, None), (12000, 200)),
        ((60, 3, 200, None), (60, 12)),
        ((3000, 30490, 200, 300), (300, 60)),
        ((3000, 30490, 50, 300), (300, 50)),
        ((3000, 118, 200, 300), (300, 60)),
        ((3000, 118, 200, 10000), (500, 100)),
        ((60, 3, 200, 1), (1, 1)),
    ],
)
def test_resolve_budget_cap_applies_after_clip_and_before_eval_rule(args, expected):
    assert resolve_budget(*args) == expected


@pytest.mark.parametrize("cap", [0, -1])
def test_resolve_budget_rejects_nonpositive_cap(cap):
    with pytest.raises(ValueError, match="max_steps_cap"):
        resolve_budget(3000, 118, 200, cap)


@pytest.mark.parametrize("scores,best_step", [([1.0, 0.9], 40), ([1.0, 1.1], 20)])
@pytest.mark.parametrize("stop_after", [None, 40])
def test_early_stop_and_absolute_best_restoration(
    scores, best_step, stop_after, tmp_path, fresh, panel_long, plan_long, encoder_long
):
    cfg = TrainConfig(
        base_steps=120, eval_every=20, patience=1, rel_tol=1.0, batch_size=16, amp=False
    )
    stub = iter(scores)
    model = fresh()
    states = {}

    def fake_eval(m, *a, **k):
        score = next(stub)
        states[score] = deepcopy(m.state_dict())
        return {"SQL": score}

    r = Trainer(model, cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        stop_after=stop_after,
        evaluate_fn=fake_eval,
    )
    assert r.stop_reason == "early_stop" and r.steps == 40 and r.best_step == best_step
    assert r.best_score == min(scores)
    assert_state_equal(model.state_dict(), states[min(scores)])
    assert_state_equal(r.best_state, states[min(scores)])
    assert len(r.history) == 40
    assert [h["step"] for h in r.history if "SQL" in h] == [20, 40]
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["step"] == 40


@pytest.mark.parametrize("exception", [False, True])
@pytest.mark.parametrize("resume_at", [0, 2])
@pytest.mark.parametrize("cap", [None, 6])
def test_stop_after_at_remaining_budget_completes(
    tmp_path, fresh, panel_long, plan_long, encoder_long, exception, resume_at, cap
):
    cfg = TrainConfig(
        base_steps=6 if cap is None else 20,
        max_steps_cap=cap,
        batch_size=8,
        patience=10,
        amp=False,
    )
    plan = replace(plan_long, exception=exception)
    forecast = ForecastConfig(horizon=4)

    def fake_eval(*args):
        assert not exception
        return {"SQL": 1.0}

    if resume_at:
        stopped = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
            panel_long,
            plan,
            forecast,
            encoder_long,
            checkpoint_dir=tmp_path,
            stop_after=resume_at,
            evaluate_fn=fake_eval,
        )
        assert stopped.stop_reason == "interrupted" and stopped.steps == resume_at

    model = fresh()
    result = Trainer(model, cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan,
        forecast,
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=bool(resume_at),
        stop_after=6 - resume_at,
        evaluate_fn=fake_eval,
    )
    assert result.stop_reason == ("exception_budget" if exception else "max_steps")
    assert result.steps == 6 and result.best_step == (6 if exception else 1)
    assert_state_equal(model.state_dict(), result.best_state)
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["step"] == 6
    assert_state_equal(saved["best_state"], result.best_state)


def test_exception_plan_trains_fixed_budget_without_validation(
    tmp_path, fresh29, panel29, plan29, encoder29
):
    model = fresh29()

    def forbidden_eval(*a, **k):
        pytest.fail("exception plans must not evaluate")

    r = Trainer(model, TrainConfig(base_steps=60, batch_size=8, amp=False), cpu, RUN_ID, SEED).fit(
        panel29,
        plan29,
        ForecastConfig(horizon=13),
        encoder29,
        checkpoint_dir=tmp_path,
        evaluate_fn=forbidden_eval,
    )
    assert r.stop_reason == "exception_budget" and r.steps == 60
    assert not any("SQL" in h for h in r.history)
    assert r.best_score is None and r.best_step == 60
    assert_state_equal(r.best_state, model.state_dict())
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["step"] == saved["best_step"] == 60 and saved["best_score"] is None
    assert_state_equal(saved["best_state"], model.state_dict())
    expected = np.median(np.concatenate([s.target for s in panel29.series]))
    assert model.fallback_level.item() == expected


@pytest.mark.parametrize(
    "panel,plan,enc,horizon",
    [("panel8", "plan8", "encoder8", 8), ("panel_constant_long", "plan_cl", "encoder_cl", 4)],
)
def test_short_or_constant_panels_fit_and_forecast(panel, plan, enc, horizon, request, fresh_for):
    p, pl, e = (request.getfixturevalue(n) for n in (panel, plan, enc))
    m = fresh_for(p, horizon)
    r = Trainer(m, TrainConfig(base_steps=30, batch_size=4, amp=False), cpu, RUN_ID, SEED).fit(
        p, pl, ForecastConfig(horizon=horizon), e
    )
    assert r.steps > 0
    assert torch.isfinite(m.predict(make_batch(p, [(0, p.series[0].length)], horizon, 64, e))).all()


def test_singleton_only_panel_returns_no_training_result_with_observed_median(
    fresh_for, panel_singletons, plan_singletons, encoder_singletons
):
    m = fresh_for(panel_singletons, 4)
    initial = deepcopy(m.state_dict())
    r = Trainer(m, TrainConfig(base_steps=30, batch_size=4, amp=False), cpu, RUN_ID, SEED).fit(
        panel_singletons, plan_singletons, ForecastConfig(horizon=4), encoder_singletons
    )
    assert r.stop_reason == "no_training_data" and r.steps == 0
    assert m.fallback_level.item() == 6.0
    assert r.best_score is None and r.best_step == 0 and r.history == []
    initial["fallback_level"].fill_(6.0)
    assert_state_equal(m.state_dict(), initial)
    assert_state_equal(r.best_state, initial)
    assert torch.isfinite(
        m.predict(make_batch(panel_singletons, [(0, 1)], 4, 64, encoder_singletons))
    ).all()


@pytest.mark.parametrize("support", ["nonnegative", "real"])
def test_all_missing_panel_trains_nothing_and_serves_fallbacks_after_reload(
    tmp_path,
    support,
    fresh_for,
    panel_all_missing,
    plan_all_missing,
    encoder_all_missing,
    cutoff_am,
    origin_am,
):
    p = replace(panel_all_missing, target_support=support)
    m = fresh_for(p, 4)
    r = Trainer(m, TrainConfig(base_steps=30, batch_size=4, amp=False), cpu, RUN_ID, SEED).fit(
        p, plan_all_missing, ForecastConfig(horizon=4), encoder_all_missing
    )
    assert r.stop_reason == "no_training_data" and r.steps == 0
    assert m.fallback_level.item() == 0.0 and plan_all_missing.fit_ids == []
    save_bundle(
        tmp_path / "b",
        model=m,
        family="v2",
        backbone_kwargs=asdict(m.backbone.cfg),
        forecast_cfg=ForecastConfig(horizon=4),
        dataset_cfg=None,
        encoder=encoder_all_missing,
        panel=p,
        training_cutoff=cutoff_am,
        earliest_origin=origin_am,
        manifest={},
    )
    df = Predictor.from_bundle(tmp_path / "b").forecast(p, origin_am)
    assert np.isfinite(df[["q0.1", "q0.5", "q0.9", "point"]].to_numpy()).all()
    assert (df["q0.5"] == 0).all()


@pytest.mark.parametrize("support", ["nonnegative", "real"])
def test_all_missing_panel_returns_zero_fallback_without_training(
    fresh_for, panel_all_missing, plan_all_missing, encoder_all_missing, support
):
    p = replace(panel_all_missing, target_support=support)
    model = fresh_for(p, 4)
    model.fallback_level.fill_(99.0)
    result = Trainer(
        model, TrainConfig(base_steps=30, batch_size=4, amp=False), cpu, RUN_ID, SEED
    ).fit(p, plan_all_missing, ForecastConfig(horizon=4), encoder_all_missing)
    assert result.stop_reason == "no_training_data" and result.steps == 0
    assert plan_all_missing.fit_ids == [] and model.fallback_level.item() == 0.0
    prediction = model.predict(make_batch(p, [(0, 8)], 4, 64, encoder_all_missing))
    assert torch.equal(prediction, torch.zeros_like(prediction))


def test_resume_after_interruption_matches_uninterrupted_run(
    tmp_path, fresh, panel_long, plan_long, encoder_long
):
    cfg = TrainConfig(
        base_steps=60, eval_every=20, warmup_steps=5, checkpoint_every=10, batch_size=8, amp=False
    )
    full_model = fresh()
    full = Trainer(full_model, cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path / "a",
    )
    stopped = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path / "b",
        stop_after=25,
    )
    assert stopped.stop_reason == "interrupted"
    saved = torch.load(tmp_path / "b/latest.pt", weights_only=False)
    assert saved["step"] == 25
    assert saved["max_steps"] == 60 and saved["eval_every"] == 12
    resumed_model = fresh()
    resumed = Trainer(resumed_model, cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path / "b",
        resume=True,
    )
    assert [h["loss"] for h in resumed.history] == [h["loss"] for h in full.history]
    assert resumed.best_score == full.best_score and resumed.history[:25] == stopped.history
    assert resumed.history == full.history and resumed.best_step == full.best_step
    assert_state_equal(resumed_model.state_dict(), full_model.state_dict())
    with pytest.raises(ValueError, match="identity"):
        Trainer(fresh(), cfg, cpu, "other-identity", SEED).fit(
            panel_long,
            plan_long,
            ForecastConfig(horizon=4),
            encoder_long,
            checkpoint_dir=tmp_path / "b",
            resume=True,
        )


def test_final_evaluation_and_relative_patience_reset(fresh, panel_long, plan_long, encoder_long):
    scores = iter([10.0, 8.0, 7.99, 7.98, 7.0, 6.0])
    cfg = TrainConfig(base_steps=13, eval_every=2, patience=2, rel_tol=0.01, amp=False)
    trainer = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    result = trainer.fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        evaluate_fn=lambda *a: {"SQL": next(scores), "MASE": 3.0},
    )
    assert result.stop_reason == "early_stop" and result.steps == 8
    assert result.best_step == 8 and result.best_score == 7.98

    calls = []
    cfg.patience = 10

    def fake_eval(*args):
        calls.append(args)
        return {"SQL": 1.0, "MASE": 2.0}

    result = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long, plan_long, ForecastConfig(horizon=4), encoder_long, evaluate_fn=fake_eval
    )
    assert result.stop_reason == "max_steps" and result.steps == 13
    assert [h["step"] for h in result.history if "SQL" in h] == [2, 4, 6, 8, 10, 12, 13]
    assert result.history[-1]["MASE"] == 2.0
    assert len(calls) == 7 and calls[0][1] is panel_long
    np.testing.assert_array_equal(calls[0][2], plan_long.val_items)
    assert calls[0][3:6] == (plan_long.val_origin, 4, 512)
    assert calls[0][6] is encoder_long


def test_fit_respects_fit_ids_cutoff_and_normalised_labels(
    fresh, panel_long, plan_long, encoder_long, monkeypatch
):
    import demand_forecast.core.training as training

    plan = replace(plan_long, fit_ids=[panel_long.series[0].series_id], exception=True)
    model = fresh()
    expected_median = np.median(panel_long.series[0].target[:76])
    # Make excluded targets so large that leakage into the fallback is unmistakable.
    panel_long.series[0].target[76:] = 1e6
    for s in panel_long.series[1:]:
        s.target[:] = 1e6
    batches = []
    original_forward = model.forward

    def forward(batch):
        batches.append(batch)
        return original_forward(batch)

    monkeypatch.setattr(model, "forward", forward)
    original_loss = training.pinball_loss

    def loss(pred, labels, label_mask, loss_eligible, quantiles):
        batch = batches[-1]
        assert labels is batch.labels and label_mask is batch.label_mask
        assert loss_eligible is batch.loss_eligible
        return original_loss(pred, labels, label_mask, loss_eligible, quantiles)

    monkeypatch.setattr(training, "pinball_loss", loss)
    result = Trainer(model, TrainConfig(base_steps=4, amp=False), cpu, RUN_ID, SEED).fit(
        panel_long, plan, ForecastConfig(horizon=4, min_context=75), encoder_long
    )
    assert result.steps == 4 and model.fallback_level.item() == expected_median
    for batch in batches:
        assert batch.series_index.tolist() == [0] and batch.origin_pos.tolist() == [75]
        assert batch.label_mask.tolist() == [[True, False, False, False]]


def test_fit_does_not_reseed_and_cpu_amp_is_disabled(
    fresh, panel_long, plan_long, encoder_long, monkeypatch
):
    model = fresh()

    def forbidden_seed(*a, **k):
        pytest.fail("Trainer must not reseed")

    monkeypatch.setattr(torch, "manual_seed", forbidden_seed)
    monkeypatch.setattr(np.random, "seed", forbidden_seed)
    monkeypatch.setattr(random, "seed", forbidden_seed)

    def check_forward(_model, _batch):
        assert not torch.is_autocast_enabled("cpu")

    model.register_forward_pre_hook(check_forward)
    trainer = Trainer(model, TrainConfig(base_steps=1, amp=True), cpu, RUN_ID, SEED)
    trainer.fit(panel_long, plan_long, ForecastConfig(horizon=4), encoder_long)
    assert not trainer.scaler.is_enabled()
    assert issubclass(InterruptedRun, RuntimeError)


def test_evaluate_uses_full_past_original_units_missing_labels_and_item_origins(
    fresh_for, monkeypatch
):
    import demand_forecast.core.evaluation as evaluation

    panel = weekly_panel([8, 7, 9])
    panel.series[1].start = panel.timestamps(0)[1]
    panel.series[2].target[7] = np.nan
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    model = fresh_for(panel, 4)
    quantiles = model.spec.quantiles
    items = np.array([2, 1, 0])
    origin = panel.timestamps(0)[6]
    calls = []

    def predict(batch):
        assert not model.training and not torch.is_grad_enabled()
        calls.extend(zip(batch.series_index.tolist(), batch.origin_pos.tolist()))
        assert batch.target.shape[1] == 2
        return (10.0 + batch.series_index[:, None, None]).expand(-1, 4, len(quantiles))

    monkeypatch.setattr(model, "predict", predict)
    true = np.array(
        [
            [panel.series[2].target[6], np.nan, panel.series[2].target[8], np.nan],
            [*panel.series[1].target[5:7], np.nan, np.nan],
            [*panel.series[0].target[6:8], np.nan, np.nan],
        ]
    )
    past = [panel.series[2].target[:6], panel.series[1].target[:5], panel.series[0].target[:6]]
    predictions = np.broadcast_to(np.array([12, 11, 10])[:, None, None], (3, 4, len(quantiles)))

    def summarize_spy(y_true, q_pred, y_past, m, q):
        np.testing.assert_equal(y_true, true)
        np.testing.assert_equal(q_pred, predictions)
        for actual, expected in zip(y_past, past):
            np.testing.assert_equal(actual, expected)
        assert m == panel.seasonal_period and tuple(q) == quantiles
        return summarize(y_true, q_pred, y_past, m, q)

    monkeypatch.setattr(evaluation, "summarize", summarize_spy)
    result = evaluate(model, panel, items, origin, 4, 2, encoder, quantiles, batch_size=2)
    assert result == summarize(true, predictions, past, panel.seasonal_period, quantiles)
    assert model.training and calls == [(2, 6), (1, 5), (0, 6)]


def test_evaluate_restores_training_after_failure(
    fresh, panel_long, plan_long, encoder_long, monkeypatch
):
    model = fresh()

    def fail(batch):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(RuntimeError, match="prediction failed"):
        evaluate(
            model,
            panel_long,
            plan_long.val_items,
            plan_long.val_origin,
            4,
            64,
            encoder_long,
            model.spec.quantiles,
        )
    assert model.training


def test_repeated_resume_restores_all_rng_streams_and_the_original_budget(
    tmp_path, fresh, panel_long, plan_long, encoder_long
):
    cfg = TrainConfig(
        base_steps=12,
        batch_size=8,
        eval_every=2,
        warmup_steps=2,
        patience=20,
        checkpoint_every=2,
        amp=False,
    )
    forecast = ForecastConfig(horizon=4)

    def draws():
        return {
            "torch": torch.rand(()).item(),
            "numpy": np.random.rand(),
            "python": random.random(),
        }

    def random_eval(*a):
        metrics = draws()
        return {"SQL": sum(metrics.values()), **metrics}

    full_model = fresh()
    full = Trainer(full_model, cfg, cpu, RUN_ID, SEED).fit(
        panel_long, plan_long, forecast, encoder_long, evaluate_fn=random_eval
    )
    expected_rng = draws()
    first_trainer = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    # A missing latest.pt with resume=True must begin a fresh run.
    first = first_trainer.fit(
        panel_long,
        plan_long,
        forecast,
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
        stop_after=3,
        evaluate_fn=random_eval,
    )
    assert first.steps == 3 and first.stop_reason == "interrupted"
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert {
        "run_identity",
        "step",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "sampler",
        "history",
        "best_state",
        "best_score",
        "best_step",
        "patience_counter",
        "max_steps",
        "eval_every",
    } <= saved.keys()
    assert saved["run_identity"] == RUN_ID
    assert saved["sampler"] == {"epoch": 0, "position": 20}
    assert (saved["rng"]["cuda"] is None) == (not torch.cuda.is_available())
    # Perturb RNGs through fresh() before each resume, including the sampler seed.
    # The checkpoint budget wins even if a caller supplies a different budget.
    other_cfg = replace(cfg, base_steps=24, eval_every=1, warmup_steps=4)
    second = Trainer(fresh(), other_cfg, cpu, RUN_ID, 99).fit(
        panel_long,
        plan_long,
        forecast,
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
        stop_after=2,
        evaluate_fn=random_eval,
    )
    assert second.steps == 5 and second.stop_reason == "interrupted"
    assert second.history[:3] == first.history
    resumed_model = fresh()
    resumed = Trainer(resumed_model, other_cfg, cpu, RUN_ID, 99).fit(
        panel_long,
        plan_long,
        forecast,
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
        evaluate_fn=random_eval,
    )
    assert resumed.steps == 12 and resumed.history == full.history
    assert draws() == expected_rng
    assert_state_equal(resumed_model.state_dict(), full_model.state_dict())
    # A completed checkpoint selects the same weights without training again.
    completed_model = fresh()
    completed = Trainer(completed_model, cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        forecast,
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
        evaluate_fn=random_eval,
    )
    assert completed.history == full.history and completed.stop_reason == full.stop_reason
    assert_state_equal(completed_model.state_dict(), full_model.state_dict())


def test_checkpoint_cadence_warmup_cosine_and_gradient_clipping(
    tmp_path, fresh, panel_long, plan_long, encoder_long, monkeypatch
):
    cfg = TrainConfig(
        base_steps=6,
        batch_size=8,
        learning_rate=0.03,
        weight_decay=0.04,
        warmup_steps=2,
        checkpoint_every=2,
        grad_clip=0.2,
        amp=False,
        patience=10,
    )
    trainer = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    rates, clips, checkpoints = [], [], []
    original_step = torch.optim.AdamW.step
    original_clip = torch.nn.utils.clip_grad_norm_
    original_checkpoint = trainer.checkpoint

    def step(optimizer, *args, **kwargs):
        rates.append(optimizer.param_groups[0]["lr"])
        assert optimizer.param_groups[0]["weight_decay"] == cfg.weight_decay
        return original_step(optimizer, *args, **kwargs)

    def clip(parameters, max_norm, *args, **kwargs):
        clips.append(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    def checkpoint(path):
        original_checkpoint(path)
        saved = torch.load(path, weights_only=False)
        checkpoints.append(saved["step"])
        assert len(saved["history"]) == saved["step"]

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(trainer, "checkpoint", checkpoint)
    trainer.fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        evaluate_fn=lambda *a: {"SQL": 1.0},
    )
    expected = np.array([0.0, 0.5, 1.0, (1 + 2**-0.5) / 2, 0.5, (1 - 2**-0.5) / 2])
    np.testing.assert_allclose(rates, cfg.learning_rate * expected, atol=1e-12)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0
    assert clips == [cfg.grad_clip] * 6 and checkpoints == [2, 4, 6]


def test_resume_preserves_patience_and_completed_early_stop(
    tmp_path, fresh, panel_long, plan_long, encoder_long
):
    cfg = TrainConfig(base_steps=20, batch_size=8, eval_every=2, patience=2, amp=False)
    first = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        stop_after=4,
        evaluate_fn=lambda *a: {"SQL": 1.0},
    )
    assert first.stop_reason == "interrupted" and first.best_step == 2
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["patience_counter"] == 1
    model = fresh()
    resumed = Trainer(model, cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
        evaluate_fn=lambda *a: {"SQL": 1.0},
    )
    assert resumed.stop_reason == "early_stop" and resumed.steps == 6
    assert_state_equal(model.state_dict(), first.best_state)

    def forbidden_eval(*a):
        pytest.fail("a completed early stop must not resume optimization")

    again = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
        evaluate_fn=forbidden_eval,
    )
    assert (
        again.steps == 6 and again.history == resumed.history and again.stop_reason == "early_stop"
    )


def test_evaluate_handles_an_item_ending_before_origin_without_mutating_panel(
    fresh_for, monkeypatch
):
    import demand_forecast.core.evaluation as evaluation

    panel = weekly_panel([8, 3])
    original = deepcopy(panel.series[1])
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    model = fresh_for(panel, 4)
    original_summarize = evaluation.summarize

    def check(y_true, q_pred, y_past, m, quantiles):
        assert np.isnan(y_true[1]).all() and np.isfinite(q_pred).all()
        np.testing.assert_equal(y_past[1], original.target)
        return original_summarize(y_true, q_pred, y_past, m, quantiles)

    monkeypatch.setattr(evaluation, "summarize", check)
    result = evaluate(
        model, panel, np.array([0, 1]), panel.timestamps(0)[6], 4, 4, encoder, model.spec.quantiles
    )
    assert np.isfinite(result["SQL"])
    assert panel.series[1].length == original.length
    np.testing.assert_equal(panel.series[1].target, original.target)


def test_timing_categories_sum_with_controlled_clock(
    tmp_path, monkeypatch, fresh, panel_long, plan_long, encoder_long
):
    from demand_forecast.core import training

    now = [0.0]
    monkeypatch.setattr(training.time, "perf_counter", lambda: now[0])
    batches = training.WindowSampler.batches
    save = torch.save

    def timed_batches(self, size):
        for batch in batches(self, size):
            now[0] += 2.0
            yield batch

    def timed_save(*args, **kwargs):
        now[0] += 5.0
        return save(*args, **kwargs)

    def timed_eval(*args):
        now[0] += 3.0
        return {"SQL": 1.0}

    monkeypatch.setattr(training.WindowSampler, "batches", timed_batches)
    monkeypatch.setattr(torch, "save", timed_save)
    cfg = TrainConfig(base_steps=4, batch_size=8, amp=False, checkpoint_every=2)
    trainer = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    result = trainer.fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        stop_after=2,
        evaluate_fn=timed_eval,
    )
    assert result.timings == {
        "train_s": 4.0,
        "eval_s": 6.0,
        "checkpoint_s": 5.0,
        "sessions": 1,
        "peak_cuda_mem_bytes": 0,
        "nonfinite_losses": 0,
        "scaler_skips": 0,
        "cold_covariate_columns": 0,
        "clipped_covariate_values": 0,
        "floored_covariate_columns": 0,
    }
    resumed = Trainer(fresh(), cfg, cpu, RUN_ID, SEED).fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
        evaluate_fn=timed_eval,
    )
    assert resumed.timings["train_s"] == 8.0
    assert resumed.timings["eval_s"] == 12.0
    assert resumed.timings["checkpoint_s"] == 10.0
    assert resumed.timings["sessions"] == 2
    assert [r["train_s_per_step"] for r in resumed.telemetry] == [2.0] * 4
    assert [r["elapsed_s"] for r in resumed.telemetry] == [5.0, 10.0, 20.0, 25.0]
    assert all(
        set(r) == {"step", "elapsed_s", "train_s_per_step", "eval_s", "peak_cuda_mem_bytes"}
        for r in resumed.telemetry
    )


def test_nonfinite_losses_counter_is_cumulative(
    tmp_path, monkeypatch, fresh29, panel29, plan29, encoder29
):
    from demand_forecast.core import training

    loss_fn = training.pinball_loss

    def nonfinite_loss(*args):
        loss, count = loss_fn(*args)
        return loss + loss.new_tensor(float("nan")), count

    monkeypatch.setattr(training, "pinball_loss", nonfinite_loss)
    cfg = TrainConfig(base_steps=4, batch_size=8, amp=False)
    first = Trainer(fresh29(), cfg, cpu, RUN_ID, SEED).fit(
        panel29,
        plan29,
        ForecastConfig(horizon=13),
        encoder29,
        checkpoint_dir=tmp_path,
        stop_after=2,
    )
    assert first.timings["nonfinite_losses"] == 2
    second = Trainer(fresh29(), cfg, cpu, RUN_ID, SEED).fit(
        panel29,
        plan29,
        ForecastConfig(horizon=13),
        encoder29,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert second.timings["nonfinite_losses"] == 4


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_fit_stops_after_consecutive_nonfinite_losses_with_checkpoint_and_progress(
    tmp_path, monkeypatch, fresh29, panel29, plan29, encoder29, invalid
):
    from demand_forecast.core import training

    trainer = Trainer(
        fresh29(), TrainConfig(base_steps=60, batch_size=8, amp=False), cpu, RUN_ID, SEED
    )
    initial = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    diagnostics = dict.fromkeys(
        ("cold_covariate_columns", "clipped_covariate_values", "floored_covariate_columns"), 0
    )
    batches = training.WindowSampler.batches

    def counted_batches(sampler, size):
        for batch in batches(sampler, size):
            for field, key in zip(
                ("cold_covariates", "clipped_covariates", "floored_covariates"), diagnostics
            ):
                # Nonzero diagnostics must survive even when the step is skipped.
                getattr(batch, field).fill_(1)
                diagnostics[key] += int(getattr(batch, field).sum())
            yield batch

    def forbidden(*args, **kwargs):
        pytest.fail("invalid loss reached backward, gradient clipping, optimizer or scaler")

    def invalid_loss(pred, *args):
        loss = pred.sum() * pred.new_tensor(invalid)
        loss.register_hook(forbidden)
        return loss, 1

    monkeypatch.setattr(training, "pinball_loss", invalid_loss)
    monkeypatch.setattr(training.WindowSampler, "batches", counted_batches)
    for name in ("scale", "unscale_", "step", "update", "get_scale"):
        monkeypatch.setattr(trainer.scaler, name, forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", forbidden)
    scaler_state = deepcopy(trainer.scaler.state_dict())
    with pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"):
        result = trainer.fit(
            panel29, plan29, ForecastConfig(horizon=13), encoder29, checkpoint_dir=tmp_path
        )
    assert training.NONFINITE_STOP_AFTER == 20
    assert result.stop_reason == trainer.terminal_reason == "nonfinite_losses"
    assert result.steps == result.timings["nonfinite_losses"] == trainer.consecutive_nonfinite == 20
    assert result.timings["scaler_skips"] == 0
    assert len(result.history) == 20 and all(not np.isfinite(r["loss"]) for r in result.history)
    assert result.timings["train_s"] > 0 and result.timings["checkpoint_s"] > 0
    for key, count in diagnostics.items():
        assert count > 0 and result.timings[key] == count
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["step"] == saved["consecutive_nonfinite"] == 20
    assert saved["terminal_reason"] == "nonfinite_losses" and len(saved["history"]) == 20
    assert saved["scheduler"]["last_epoch"] == 20
    assert saved["optimizer"]["state"] == {}
    assert saved["scaler"] == scaler_state
    for name, parameter in trainer.model.named_parameters():
        assert torch.isfinite(parameter).all() and parameter.grad is None
        assert torch.equal(parameter, initial[name])
        assert torch.equal(saved["model"][name], initial[name])
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stop_reason"] == "nonfinite_losses" and progress["step"] == 20
    assert progress["timings"] == result.timings


def test_finite_loss_resets_the_consecutive_counter_and_parameters_are_not_poisoned(
    tmp_path, monkeypatch, fresh29, panel29, plan29, encoder29
):
    from demand_forecast.core import training

    trainer = Trainer(
        fresh29(),
        TrainConfig(base_steps=39, batch_size=8, warmup_steps=2, amp=False),
        cpu,
        RUN_ID,
        SEED,
    )
    snapshots, optimizer_steps = [], []
    step = torch.optim.AdamW.step

    def counted_step(optimizer, *args, **kwargs):
        optimizer_steps.append(trainer.step + 1)
        return step(optimizer, *args, **kwargs)

    def intermittent_loss(pred, *args):
        snapshots.append({name: p.detach().clone() for name, p in trainer.model.named_parameters()})
        loss = pred.square().mean()
        if len(snapshots) != 20:
            loss = loss * loss.new_tensor(float("nan"))
        return loss, 1

    monkeypatch.setattr(training, "pinball_loss", intermittent_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    with pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"):
        result = trainer.fit(
            panel29, plan29, ForecastConfig(horizon=13), encoder29, checkpoint_dir=tmp_path
        )
    assert result.stop_reason == "exception_budget" and result.steps == 39
    assert result.timings["nonfinite_losses"] == 38 and trainer.consecutive_nonfinite == 19
    assert trainer.terminal_reason is None and optimizer_steps == [20]
    assert trainer.scheduler.last_epoch == 39
    assert any(not torch.equal(snapshots[0][k], snapshots[20][k]) for k in snapshots[0])
    for state in snapshots[:20]:
        assert_state_equal(state, snapshots[0])
    for state in snapshots[20:]:
        assert_state_equal(state, snapshots[20])
    for name, p in trainer.model.named_parameters():
        assert torch.isfinite(p).all() and p.grad is None
        assert torch.equal(p, snapshots[20][name])
    assert json.loads((tmp_path / "progress.json").read_text())["stop_reason"] == "exception_budget"


@pytest.mark.parametrize("marker", ["both", "reason", "counter", "no_origins"])
def test_resume_after_nonfinite_stop_returns_the_terminal_result_without_training(
    tmp_path, monkeypatch, fresh29, panel29, plan29, encoder29, marker
):
    from demand_forecast.core import training

    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    cfg = TrainConfig(base_steps=60, batch_size=8, amp=False)
    args = (panel29, plan29, ForecastConfig(horizon=13), encoder29)
    with pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"):
        Trainer(fresh29(), cfg, cpu, RUN_ID, SEED).fit(*args, checkpoint_dir=tmp_path)
    path = tmp_path / "latest.pt"
    saved = torch.load(path, weights_only=False)
    if marker == "reason":
        saved["consecutive_nonfinite"] = 0
    elif marker == "counter":
        saved["terminal_reason"] = None
    elif marker == "no_origins":
        # Keep the synthetic empty sampler valid so this exercises the entry
        # guard's priority over no_training_data, not position validation.
        saved["sampler"]["position"] = 0
    torch.save(saved, path)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("terminal checkpoint resumed training or evaluation")

    monkeypatch.setattr(training, "pinball_loss", forbidden)
    monkeypatch.setattr(training.WindowSampler, "batches", forbidden)
    if marker == "no_origins":
        monkeypatch.setattr(training, "build_origins", lambda *a, **k: np.empty((0, 2), dtype=int))
    trainer = Trainer(fresh29(), cfg, cpu, RUN_ID, SEED)
    result = trainer.fit(*args, checkpoint_dir=tmp_path, resume=True, evaluate_fn=forbidden)
    assert result.stop_reason == "nonfinite_losses" and result.steps == 20
    assert len(result.history) == 20 and result.timings["nonfinite_losses"] == 20
    assert result.timings["sessions"] == 2
    assert trainer.consecutive_nonfinite == saved["consecutive_nonfinite"]
    assert trainer.scheduler.state_dict() == saved["scheduler"]
    assert_state_equal(trainer.model.state_dict(), saved["model"])
    assert path.read_bytes() == before
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stop_reason"] == "nonfinite_losses" and progress["step"] == 20


@pytest.mark.parametrize("boundary", ["max_steps", "cap", "stop_after", "both"])
@pytest.mark.parametrize("exception", [False, True])
def test_failure_wins_over_stop_after_and_completion_boundaries(
    tmp_path, monkeypatch, fresh, panel_long, plan_long, encoder_long, boundary, exception
):
    from demand_forecast.core import training

    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    cfg = TrainConfig(
        base_steps=20 if boundary in ("max_steps", "both") else 60,
        max_steps_cap=20 if boundary == "cap" else None,
        batch_size=8,
        patience=100,
        amp=False,
    )
    trainer = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    evaluations = []

    def evaluate(*args):
        evaluations.append(trainer.step)
        return {"SQL": 1.0}

    with pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"):
        result = trainer.fit(
            panel_long,
            replace(plan_long, exception=exception),
            ForecastConfig(horizon=4),
            encoder_long,
            checkpoint_dir=tmp_path,
            stop_after=20 if boundary in ("stop_after", "both") else None,
            evaluate_fn=evaluate,
        )
    assert result.steps == 20 and result.stop_reason == "nonfinite_losses"
    assert 20 not in evaluations and "SQL" not in result.history[-1]
    assert json.loads((tmp_path / "progress.json").read_text())["stop_reason"] == "nonfinite_losses"


def test_nonfinite_stop_without_checkpoint_dir_creates_fallback_and_progress(
    tmp_path, monkeypatch, caplog, fresh29, panel29, plan29, encoder29
):
    from demand_forecast.core import training

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    trainer = Trainer(fresh29(), TrainConfig(base_steps=60, amp=False), cpu, RUN_ID, SEED)
    with (
        caplog.at_level("INFO", logger="demand_forecast.core.training"),
        pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"),
    ):
        result = trainer.fit(panel29, plan29, ForecastConfig(horizon=13), encoder29)
    assert result.steps == 20 and result.stop_reason == "nonfinite_losses"
    assert (tmp_path / "checkpoints/latest.pt").exists()
    progress = json.loads((tmp_path / "checkpoints/progress.json").read_text())
    assert progress["stop_reason"] == "nonfinite_losses" and progress["step"] == 20
    assert "checkpoint_dir" in caplog.text and "checkpoints" in caplog.text


@pytest.mark.parametrize("legacy", [False, True])
def test_nonfinite_counter_resumes_and_legacy_checkpoint_defaults(
    tmp_path, monkeypatch, fresh29, panel29, plan29, encoder29, legacy
):
    from demand_forecast.core import training

    monkeypatch.setattr(training, "pinball_loss", lambda pred, *a: (pred.sum() * float("nan"), 1))
    cfg = TrainConfig(base_steps=60, batch_size=8, amp=False)
    args = (panel29, plan29, ForecastConfig(horizon=13), encoder29)
    with pytest.warns(UserWarning, match=r"lr_scheduler.step\(\).*before.*optimizer.step"):
        first = Trainer(fresh29(), cfg, cpu, RUN_ID, SEED).fit(
            *args, checkpoint_dir=tmp_path, stop_after=19
        )
    assert first.stop_reason == "interrupted"
    assert json.loads((tmp_path / "progress.json").read_text())["stop_reason"] == "interrupted"
    path = tmp_path / "latest.pt"
    saved = torch.load(path, weights_only=False)
    assert saved["consecutive_nonfinite"] == 19 and saved["terminal_reason"] is None
    if legacy:
        del saved["consecutive_nonfinite"], saved["terminal_reason"]
        torch.save(saved, path)
    trainer = Trainer(fresh29(), cfg, cpu, RUN_ID, SEED)
    resume = trainer.resume

    def checked_resume(path):
        resume(path)
        assert trainer.consecutive_nonfinite == (0 if legacy else 19)
        assert trainer.terminal_reason is None

    monkeypatch.setattr(trainer, "resume", checked_resume)
    result = trainer.fit(*args, checkpoint_dir=tmp_path, resume=True)
    assert result.steps == (39 if legacy else 20)
    assert result.stop_reason == "nonfinite_losses" and trainer.consecutive_nonfinite == 20
    assert len(result.history) == result.timings["nonfinite_losses"] == result.steps


def test_telemetry_without_validation(tmp_path, fresh29, panel29, plan29, encoder29):
    import json

    result = Trainer(fresh29(), TrainConfig(base_steps=2, amp=False), cpu, RUN_ID, SEED).fit(
        panel29,
        plan29,
        ForecastConfig(horizon=13),
        encoder29,
        checkpoint_dir=tmp_path,
    )
    assert result.telemetry == []
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["exception"] is True
    assert progress["step"] == 2
    assert progress["telemetry_tail"] == []


def test_fit_logs_budget_and_evaluations(
    caplog, tmp_path, fresh, panel_long, plan_long, encoder_long
):
    import logging

    with caplog.at_level(logging.INFO):
        Trainer(fresh(), TrainConfig(base_steps=2, amp=False), cpu, RUN_ID, SEED).fit(
            panel_long,
            plan_long,
            ForecastConfig(horizon=4),
            encoder_long,
            checkpoint_dir=tmp_path,
        )
    for message in (
        "fit: max_steps=2 eval_every=1",
        "eval: step=1/2 inner_SQL=",
        "checkpoint: step=2 path=",
        "fit done: stop_reason=max_steps steps=2",
    ):
        assert message in caplog.text


def test_cuda_peak_sampling_is_monotone_without_cuda(monkeypatch, fresh):
    trainer = Trainer(fresh(), TrainConfig(amp=False), cpu, RUN_ID, SEED)
    trainer.device = torch.device("cuda:1")
    peaks = iter([8, 4, 12])
    seen = []

    def peak(device):
        seen.append(device)
        return next(peaks)

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak)
    trainer._sample_peak()
    trainer._sample_peak()
    assert trainer.timings["peak_cuda_mem_bytes"] == 8
    trainer._sample_peak()
    assert trainer.timings["peak_cuda_mem_bytes"] == 12
    assert seen == [torch.device("cuda:1")] * 3


def test_cuda_session_reset_and_sampling_points_without_gpu(
    tmp_path, monkeypatch, fresh, panel_long, plan_long, encoder_long
):
    from contextlib import nullcontext

    from demand_forecast.core import training
    from demand_forecast.data.sampler import Batch

    trainer = Trainer(fresh(), TrainConfig(base_steps=2, amp=False), cpu, RUN_ID, SEED)
    trainer.device = torch.device("cuda:1")
    tensor, move = torch.tensor, Batch.to
    # Exercise CUDA telemetry while all actual tensor work stays on the CPU.
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: tensor(*a, **{**kw, "device": cpu}))
    monkeypatch.setattr(Batch, "to", lambda self, device: move(self, cpu))
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    calls = []
    copy = training.deepcopy

    def copy_state(value):
        if isinstance(value, dict) and any(isinstance(v, torch.Tensor) for v in value.values()):
            calls.append(("copy", trainer.device))
        return copy(value)

    monkeypatch.setattr(training, "deepcopy", copy_state)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: calls.append(("reset", d)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: calls.append(("sync", d)))
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda d: calls.append(("peak", d)) or 123
    )
    result = trainer.fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        evaluate_fn=lambda *a: {"SQL": 1.0},
    )
    assert calls == [
        (name, trainer.device)
        for name in (
            "copy",
            "reset",
            "copy",
            "sync",
            "peak",
            "sync",
            "peak",
            "peak",
            "copy",
            "peak",
        )
    ]
    assert result.timings["peak_cuda_mem_bytes"] == 123


class ScriptedScaler:
    """Exercise overflow skips on CPU without changing AMP/device resolution."""

    scales = (8, 4, 4, 8, 4, 4, 8)

    def __init__(self):
        self.index = 0
        self.optimizer_steps = 0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scales[self.index]

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        if self.scales[self.index + 1] >= self.get_scale():
            optimizer.step()
            self.optimizer_steps += 1

    def update(self):
        self.index += 1

    def state_dict(self):
        return {"index": self.index, "optimizer_steps": self.optimizer_steps}

    def load_state_dict(self, state):
        self.index = state["index"]
        self.optimizer_steps = state["optimizer_steps"]


def rewrite_scaler_timings(path, *, payload="absent", footer="absent"):
    """Rewrite both timing sources, preserving other fields and serialization."""
    import json
    from zipfile import ZipFile

    with ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/timings.json"))
        footer_timings = json.loads(archive.read(member))
    state = torch.load(path, weights_only=False)
    for timing, value in ((state["timings"], payload), (footer_timings, footer)):
        timing.pop("scaler_skips", None)
        if value != "absent":
            timing["scaler_skips"] = value
    torch.save(state, path)
    with ZipFile(path, "a") as archive:
        root = archive.namelist()[0].split("/")[0]
        archive.writestr(f"{root}/timings.json", json.dumps(footer_timings))


def test_scaler_skips_counted(fresh, panel_long, plan_long, encoder_long):
    trainer = Trainer(fresh(), TrainConfig(base_steps=6, amp=False), cpu, RUN_ID, SEED)
    trainer.scaler = ScriptedScaler()
    with pytest.warns(UserWarning, match="lr_scheduler.step"):
        result = trainer.fit(panel_long, plan_long, ForecastConfig(horizon=4), encoder_long)
    assert type(result.timings["scaler_skips"]) is int
    assert result.timings["scaler_skips"] == 2
    assert trainer.scaler.optimizer_steps == 4
    assert result.steps == trainer.scheduler.last_epoch == 6


def test_scaler_skips_cumulative_across_resume(
    tmp_path, fresh, panel_long, plan_long, encoder_long
):
    cfg = TrainConfig(base_steps=6, amp=False)
    first = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    first.scaler = ScriptedScaler()
    with pytest.warns(UserWarning, match="lr_scheduler.step"):
        result = first.fit(
            panel_long,
            plan_long,
            ForecastConfig(horizon=4),
            encoder_long,
            checkpoint_dir=tmp_path,
            stop_after=3,
        )
    assert result.timings["scaler_skips"] == 1
    resumed = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    resumed.scaler = ScriptedScaler()
    result = resumed.fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert result.timings["scaler_skips"] == 2
    assert resumed.scaler.optimizer_steps == 4


def test_scaler_skips_zero_without_scaler(fresh, panel_long, plan_long, encoder_long):
    trainer = Trainer(fresh(), TrainConfig(base_steps=2, amp=False), cpu, RUN_ID, SEED)
    assert not trainer.scaler.is_enabled()
    result = trainer.fit(panel_long, plan_long, ForecastConfig(horizon=4), encoder_long)
    assert type(result.timings["scaler_skips"]) is int
    assert result.timings["scaler_skips"] == 0


@pytest.mark.parametrize(
    "payload,footer,expected",
    [("absent", "absent", None), (7, "absent", 8), ("absent", 9, 10), (7, 9, 10), (7, None, None)],
)
def test_scaler_skips_unknown_after_legacy_resume(
    tmp_path, fresh, panel_long, plan_long, encoder_long, payload, footer, expected
):
    cfg = TrainConfig(base_steps=6, amp=False)
    first = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    first.scaler = ScriptedScaler()
    with pytest.warns(UserWarning, match="lr_scheduler.step"):
        first.fit(
            panel_long,
            plan_long,
            ForecastConfig(horizon=4),
            encoder_long,
            checkpoint_dir=tmp_path,
            stop_after=3,
        )
    rewrite_scaler_timings(tmp_path / "latest.pt", payload=payload, footer=footer)
    resumed = Trainer(fresh(), cfg, cpu, RUN_ID, SEED)
    resumed.scaler = ScriptedScaler()
    result = resumed.fit(
        panel_long,
        plan_long,
        ForecastConfig(horizon=4),
        encoder_long,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert result.timings["scaler_skips"] == expected
    assert resumed.scaler.optimizer_steps == 4
    import json
    from zipfile import ZipFile

    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["timings"]["scaler_skips"] == expected
    with ZipFile(tmp_path / "latest.pt") as archive:
        member = next(n for n in archive.namelist() if n.endswith("/timings.json"))
        assert json.loads(archive.read(member))["scaler_skips"] == expected


def test_covariate_counters_accumulate_and_resume(tmp_path, monkeypatch, fresh_for):
    import json

    from demand_forecast.core import training
    from demand_forecast.core.validation import plan_inner_validation

    panel = weekly_panel([40] * 3)
    panel.roles = replace(panel.roles, known_dynamic=("promo", "tiny", "price"))
    for series in panel.series:
        series.dynamic["promo"] = np.r_[np.full(30, np.nan), np.full(10, 1e5)]
        series.dynamic["tiny"] = np.r_[1e-9, np.zeros(27), np.full(12, 0.5)]
        series.dynamic["price"] = np.r_[100.0 + 1e-6, np.full(27, 100.0), np.full(12, 110.0)]
    plan = plan_inner_validation(panel, panel.timestamps(0, 41)[40], 4)
    encoder = StaticEncoder().fit(panel, plan.fit_ids)
    # Keep sampling real batches at the pathological origin on both sides of resume.
    monkeypatch.setattr(
        training, "build_origins", lambda *a, **k: np.array([(i, 28) for i in range(3)])
    )
    cfg = TrainConfig(base_steps=4, batch_size=8, amp=False)
    forecast = ForecastConfig(horizon=4, max_context=32)
    first = Trainer(fresh_for(panel, 4), cfg, cpu, RUN_ID, SEED).fit(
        panel,
        plan,
        forecast,
        encoder,
        checkpoint_dir=tmp_path,
        stop_after=2,
    )
    assert first.stop_reason == "interrupted"
    expected = {
        "cold_covariate_columns": 6,
        "clipped_covariate_values": 24,
        "floored_covariate_columns": 6,
    }
    progress = json.loads((tmp_path / "progress.json").read_text())
    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    for key, value in expected.items():
        assert first.timings[key] == progress["timings"][key] == state["timings"][key] == value
    resumed = Trainer(fresh_for(panel, 4), cfg, cpu, RUN_ID, SEED).fit(
        panel,
        plan,
        forecast,
        encoder,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert resumed.steps == 4 and resumed.timings["nonfinite_losses"] == 0
    progress = json.loads((tmp_path / "progress.json").read_text())
    for key, value in expected.items():
        assert resumed.timings[key] == progress["timings"][key] == value * 2
