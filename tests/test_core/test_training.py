"""Tiny CPU training, validation and exact continuation of interrupted runs."""

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
def test_stop_after_at_remaining_budget_completes(
    tmp_path, fresh, panel_long, plan_long, encoder_long, exception, resume_at
):
    cfg = TrainConfig(base_steps=6, batch_size=8, patience=10, amp=False)
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
