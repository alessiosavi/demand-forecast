"""CPU-only OOM injection and effective-batch parity for fine-tuning/refits."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core.inference_batching import task_batch_limits
from demand_forecast.core.training import TrainConfig, Trainer
from demand_forecast.data.sampler import WindowSampler
from demand_forecast.models.losses import pinball_loss


class TinyModel(torch.nn.Module):
    """A deterministic quantile model with sample-dependent gradients."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64))
        self.register_buffer("fallback_level", torch.tensor(0.0))

    def forward(self, batch):
        return (batch.target[:, -1, None, None] * self.weight).expand(-1, batch.labels.shape[1], -1)

    def predict(self, batch):
        """Return predictions in original units, as the real model does."""
        return self(batch) * batch.scale[:, None, None] + batch.loc[:, None, None]


@pytest.fixture
def case(monkeypatch, panel_long, plan_long, encoder_long):
    batches = WindowSampler.batches

    def masked_batches(sampler, size):
        for batch in batches(sampler, size):
            # Unequal cell counts, excluded rows and zero-cell chunks must all
            # retain the full batch's observed-cell mean after splitting.
            batch.label_mask &= torch.arange(4)[None, :] < (batch.series_index % 5)[:, None]
            batch.loss_eligible &= batch.series_index % 4 != 0
            yield batch

    monkeypatch.setattr(WindowSampler, "batches", masked_batches)
    cfg = TrainConfig(base_steps=5, batch_size=7, warmup_steps=0, amp=False, patience=100)
    args = (
        panel_long,
        replace(plan_long, exception=True),
        ForecastConfig(horizon=4, max_context=32, quantiles=[0.1, 0.5, 0.9]),
        encoder_long,
    )
    return cfg, args


def trainer_for(cfg):
    return Trainer(TinyModel(), cfg, "cpu", "oom-test", 3)


def assert_same_run(actual, expected):
    actual_trainer, actual_result = actual
    expected_trainer, expected_result = expected
    assert actual_result.steps == expected_result.steps
    assert actual_trainer.scheduler.state_dict() == expected_trainer.scheduler.state_dict()
    assert actual_trainer.sampler.state() == expected_trainer.sampler.state()
    np.testing.assert_allclose(
        [row["loss"] for row in actual_result.history],
        [row["loss"] for row in expected_result.history],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(actual_trainer.model.weight, expected_trainer.model.weight)
    for key, value in actual_trainer.optimizer.state_dict()["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(
                tensor, expected_trainer.optimizer.state_dict()["state"][key][name]
            )
    for key in ("nonfinite_losses", "cold_covariate_columns", "clipped_covariate_values"):
        assert actual_result.timings[key] == expected_result.timings[key]


def test_no_oom_matches_original_full_batch_loop(case, monkeypatch):
    cfg, args = case
    legacy = trainer_for(cfg)

    def original_backward(self, cpu_batch, quantiles):
        batch = cpu_batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            loss, _ = pinball_loss(
                self.model(batch), batch.labels, batch.label_mask, batch.loss_eligible, quantiles
            )
        value = float(loss.detach().item())
        if np.isfinite(value):
            self.scaler.scale(loss).backward()
        return value

    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "_backward_batch", original_backward, raising=False)
        expected = legacy.fit(*args)
    actual = trainer_for(cfg)
    result = actual.fit(*args)
    assert_same_run((actual, result), (legacy, expected))
    assert result.history == expected.history
    assert torch.equal(actual.model.weight, legacy.model.weight)


@pytest.mark.parametrize("failure", ["forward", "backward", "transfer"])
def test_training_oom_retries_same_step_and_preserves_effective_batch(
    case, monkeypatch, caplog, failure
):
    cfg, args = case
    baseline = trainer_for(cfg)
    expected = baseline.fit(*args)
    actual = trainer_for(cfg)
    calls, optimizer_steps, cache_calls = [], [], []
    failed = False
    forward = actual.model.forward

    def maybe_fail(batch):
        nonlocal failed
        calls.append((actual.step, batch.size))
        predictions = forward(batch)
        if actual.step == 2 and not failed:
            failed = True
            if failure == "backward":

                def oom(grad):
                    raise torch.cuda.OutOfMemoryError("simulated backward OOM")

                predictions.register_hook(oom)
            elif failure == "forward":
                raise torch.cuda.OutOfMemoryError("simulated forward OOM")
        return predictions

    monkeypatch.setattr(actual.model, "forward", maybe_fail)
    if failure == "transfer":
        from demand_forecast.data.sampler import Batch

        transfer = Batch.to

        def fail_transfer(batch, device):
            nonlocal failed
            if actual.step == 2 and not failed:
                failed = True
                calls.append((actual.step, batch.size))
                raise torch.cuda.OutOfMemoryError("simulated transfer OOM")
            return transfer(batch, device)

        monkeypatch.setattr(Batch, "to", fail_transfer)
    step = torch.optim.AdamW.step

    def counted_step(optimizer, *a, **kw):
        optimizer_steps.append(actual.step)
        return step(optimizer, *a, **kw)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_calls.append(True))
    result = actual.fit(*args)
    assert_same_run((actual, result), (baseline, expected))
    assert optimizer_steps == list(range(cfg.base_steps))
    assert calls == [
        (0, 7),
        (1, 7),
        (2, 6),
        (2, 3),
        (2, 3),
        (3, 3),
        (3, 3),
        (3, 1),
        (4, 3),
        (4, 3),
        (4, 1),
    ]
    assert len(cache_calls) == 1
    assert "Training OOM" in caplog.text and "to 3" in caplog.text


def test_second_microbatch_oom_discards_partial_gradients(case, monkeypatch):
    cfg, args = case
    baseline = trainer_for(cfg)
    expected = baseline.fit(*args)
    actual = trainer_for(cfg)
    forward, calls = actual.model.forward, []
    failures = 0

    def fail_twice(batch):
        nonlocal failures
        calls.append((actual.step, batch.size))
        predictions = forward(batch)
        if actual.step == 1 and failures == 0:
            failures += 1
            raise torch.cuda.OutOfMemoryError("split to three")
        if actual.step == 1 and batch.size == 3 and calls.count((1, 3)) == 2:
            failures += 1

            def oom(grad):
                raise torch.cuda.OutOfMemoryError("after partial backward")

            predictions.register_hook(oom)
        return predictions

    monkeypatch.setattr(actual.model, "forward", fail_twice)
    result = actual.fit(*args)
    assert failures == 2
    assert_same_run((actual, result), (baseline, expected))
    assert [size for step, size in calls if step == 1] == [7, 3, 3, *([1] * 7)]


def test_training_oom_raises_only_at_one_and_does_not_advance(case, monkeypatch, caplog):
    cfg, args = case
    actual = trainer_for(cfg)
    calls, cleared = [], []

    def always_oom(batch):
        calls.append(batch.size)
        raise torch.cuda.OutOfMemoryError("floor exhausted")

    monkeypatch.setattr(actual.model, "forward", always_oom)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    with pytest.raises(torch.cuda.OutOfMemoryError, match="floor exhausted"):
        actual.fit(*args)
    assert calls == [7, 3, 1]
    assert len(cleared) >= 2
    assert actual.step == actual.scheduler.last_epoch == 0
    assert actual.history == [] and actual.optimizer.state_dict()["state"] == {}
    assert all(parameter.grad is None for parameter in actual.model.parameters())
    assert caplog.text.count("Training OOM") == 2


def test_non_oom_is_not_retried(case, monkeypatch):
    cfg, args = case
    actual = trainer_for(cfg)
    calls = []

    def fail(batch):
        calls.append(batch.size)
        raise RuntimeError("a different defect")

    monkeypatch.setattr(actual.model, "forward", fail)
    with pytest.raises(RuntimeError, match="a different defect"):
        actual.fit(*args)
    assert calls == [7]


@pytest.mark.parametrize("eval_at_start", [False, True])
def test_fit_validation_backs_off_below_eight(
    case, plan_long, tmp_path, monkeypatch, caplog, eval_at_start
):
    cfg, args = case
    args = (args[0], plan_long, *args[2:])
    actual = trainer_for(replace(cfg, base_steps=2))
    predict, calls = actual.model.predict, []

    def fail_large_batches(batch):
        calls.append((actual.step, batch.size))
        if batch.size > 1:
            raise torch.cuda.OutOfMemoryError("validation requires one row")
        return predict(batch)

    monkeypatch.setattr(actual.model, "predict", fail_large_batches)
    result = actual.fit(*args, checkpoint_dir=tmp_path, eval_at_start=eval_at_start)
    assert result.steps == 2 and actual.model.training
    assert [size for step, size in calls if step == (0 if eval_at_start else 1)][:5] == [
        20,
        10,
        8,
        4,
        2,
    ]
    assert [size for step, size in calls if step == 2] == [1] * 20
    assert "Validation OOM" in caplog.text and "to 1" in caplog.text


def test_oom_reduction_survives_checkpoint_resume(case, tmp_path, monkeypatch):
    cfg, args = case
    full = trainer_for(cfg)
    forward = full.model.forward

    def oom_once(batch):
        if full.step == 1 and batch.size > 3:
            raise torch.cuda.OutOfMemoryError("reduce before checkpoint")
        return forward(batch)

    monkeypatch.setattr(full.model, "forward", oom_once)
    expected = full.fit(*args)
    interrupted = trainer_for(cfg)
    forward_first = interrupted.model.forward

    def oom_before_checkpoint(batch):
        if interrupted.step == 1 and batch.size > 3:
            raise torch.cuda.OutOfMemoryError("reduce before checkpoint")
        return forward_first(batch)

    monkeypatch.setattr(interrupted.model, "forward", oom_before_checkpoint)
    interrupted.fit(*args, checkpoint_dir=tmp_path, stop_after=2)
    resumed = trainer_for(cfg)
    sizes, forward_resumed = [], resumed.model.forward

    def record(batch):
        sizes.append(batch.size)
        return forward_resumed(batch)

    monkeypatch.setattr(resumed.model, "forward", record)
    result = resumed.fit(*args, checkpoint_dir=tmp_path, resume=True)
    assert max(sizes) == 3
    assert_same_run((resumed, result), (full, expected))
    assert result.history == expected.history


def test_microbatch_amp_scales_and_clips_once_per_effective_batch(case, monkeypatch):
    cfg, args = case
    baseline = trainer_for(cfg)
    baseline.scaler = torch.amp.GradScaler("cpu")
    expected = baseline.fit(*args)
    actual = trainer_for(cfg)
    actual.scaler = torch.amp.GradScaler("cpu")
    clip, clips, forward = torch.nn.utils.clip_grad_norm_, [], actual.model.forward

    def oom(batch):
        if batch.size > 3:
            raise torch.cuda.OutOfMemoryError("use accumulated scaled gradients")
        return forward(batch)

    def count_clip(*a, **kw):
        clips.append(actual.step)
        return clip(*a, **kw)

    monkeypatch.setattr(actual.model, "forward", oom)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", count_clip)
    result = actual.fit(*args)
    assert_same_run((actual, result), (baseline, expected))
    assert actual.scaler.state_dict() == baseline.scaler.state_dict()
    assert clips == list(range(cfg.base_steps))


def test_oom_with_no_eligible_cells_keeps_zero_loss(case, monkeypatch):
    cfg, args = case
    batches = WindowSampler.batches

    def no_cells(sampler, size):
        for batch in batches(sampler, size):
            batch.loss_eligible.fill_(False)
            yield batch

    monkeypatch.setattr(WindowSampler, "batches", no_cells)
    baseline = trainer_for(cfg)
    expected = baseline.fit(*args)
    actual = trainer_for(cfg)
    forward = actual.model.forward

    def oom(batch):
        if batch.size > 3:
            raise torch.cuda.OutOfMemoryError("zero-cell batch")
        return forward(batch)

    monkeypatch.setattr(actual.model, "forward", oom)
    result = actual.fit(*args)
    assert [row["loss"] for row in result.history] == [0.0] * cfg.base_steps
    assert_same_run((actual, result), (baseline, expected))


def test_nonfinite_microbatch_discards_earlier_gradients(case, monkeypatch):
    cfg, args = case
    actual = trainer_for(replace(cfg, base_steps=1))
    forward, attempts = actual.model.forward, []
    initial = actual.model.weight.detach().clone()

    def oom_then_nan(batch):
        attempts.append(batch.size)
        if batch.size > 3:
            raise torch.cuda.OutOfMemoryError("split the batch")
        predictions = forward(batch)
        return predictions * (float("nan") if len(attempts) == 3 else 1.0)

    monkeypatch.setattr(actual.model, "forward", oom_then_nan)
    with pytest.warns(UserWarning, match="lr_scheduler.step"):
        result = actual.fit(*args)
    assert attempts == [7, 3, 3]
    assert result.timings["nonfinite_losses"] == 1 and result.steps == 1
    assert np.isnan(result.history[0]["loss"])
    assert actual.model.weight.grad is None
    assert torch.equal(actual.model.weight, initial)
    assert actual.optimizer.state_dict()["state"] == {}


def test_retry_restores_rng_and_releases_failed_activations(case, monkeypatch):
    import weakref

    cfg, args = case
    actual = trainer_for(replace(cfg, base_steps=1))
    forward, draws, references = actual.model.forward, [], []

    def oom_once(batch):
        draws.append(torch.rand(()).item())
        predictions = forward(batch)
        if len(draws) == 1:
            references.append(weakref.ref(predictions))
            raise torch.cuda.OutOfMemoryError("failed graph must be released")
        return predictions

    def check_reclaimed():
        assert references[0]() is None

    monkeypatch.setattr(actual.model, "forward", oom_once)
    monkeypatch.setattr(torch.cuda, "empty_cache", check_reclaimed)
    actual.fit(*args)
    assert draws[0] == draws[1]


def test_task_reuses_training_limit_for_full_history_refit(case, monkeypatch):
    cfg, args = case
    with task_batch_limits():
        selection = trainer_for(cfg)
        forward = selection.model.forward

        def oom(batch):
            if batch.size > 3:
                raise torch.cuda.OutOfMemoryError("selection reduction")
            return forward(batch)

        monkeypatch.setattr(selection.model, "forward", oom)
        selection.fit(*args)
        refit = trainer_for(cfg)
        forward_refit, sizes = refit.model.forward, []

        def record(batch):
            sizes.append(batch.size)
            return forward_refit(batch)

        monkeypatch.setattr(refit.model, "forward", record)
        result = refit.fit(*args, step_limit=3, schedule_max_steps=9)
        assert max(sizes) == 3
        assert result.steps == refit.scheduler.last_epoch == 3
        assert refit.max_steps == 9
    with task_batch_limits():
        assert trainer_for(cfg)._micro_batch_limit.size is None


def test_validation_raises_only_when_one_row_fails(case, plan_long, monkeypatch):
    cfg, args = case
    args = (args[0], plan_long, *args[2:])
    actual = trainer_for(cfg)
    sizes = []

    def oom(batch):
        sizes.append(batch.size)
        raise torch.cuda.OutOfMemoryError("validation floor exhausted")

    monkeypatch.setattr(actual.model, "predict", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="validation floor exhausted"):
        actual.fit(*args)
    assert sizes == [20, 10, 8, 4, 2, 1]
    assert actual.model.training


def test_validation_oom_metrics_match_no_oom(case, plan_long, monkeypatch):
    cfg, args = case
    args = (args[0], plan_long, *args[2:])
    baseline = trainer_for(replace(cfg, base_steps=2))
    expected = baseline.fit(*args)
    actual = trainer_for(replace(cfg, base_steps=2))
    predict = actual.model.predict

    def oom(batch):
        if batch.size > 3:
            raise torch.cuda.OutOfMemoryError("validation reduction")
        return predict(batch)

    monkeypatch.setattr(actual.model, "predict", oom)
    result = actual.fit(*args)
    assert result.history == expected.history
    assert_same_run((actual, result), (baseline, expected))
