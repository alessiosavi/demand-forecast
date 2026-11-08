"""Quantile training with split-aware sampling, SQL selection and exact resume."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from zipfile import ZipFile

import numpy as np
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core.evaluation import (
    PreparedValidation,
    evaluate,
    evaluate_prepared,
    prepare_validation,
)
from demand_forecast.core.inference_batching import batch_limit, is_oom_error
from demand_forecast.core.validation import ValidationPlan
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import grid_range
from demand_forecast.data.panel import Panel
from demand_forecast.data.sampler import Batch, WindowSampler, build_origins
from demand_forecast.models.base import ForecastModel
from demand_forecast.models.losses import pinball_loss

logger = logging.getLogger(__name__)

NONFINITE_STOP_AFTER = 20


def _empty_timings() -> dict:
    return dict(
        train_s=0.0,
        eval_s=0.0,
        checkpoint_s=0.0,
        sessions=0,
        peak_cuda_mem_bytes=0,
        nonfinite_losses=0,
        scaler_skips=0,
        cold_covariate_columns=0,
        clipped_covariate_values=0,
        floored_covariate_columns=0,
        floored_target_samples=0,
    )


def _set_fallback_level(model: ForecastModel, panel: Panel, fit_indices, train_upto) -> None:
    """Use only observed fit targets through the permitted training boundary."""
    observed = []
    cutoff_positions = {}
    for i in fit_indices:
        series = panel.series[i]
        key = (series.start, series.history_length)
        if key not in cutoff_positions:
            grid = grid_range(series.start, series.history_length, panel.freq)
            cutoff_positions[key] = int(grid.searchsorted(train_upto, side="right"))
        values = series.target[: cutoff_positions[key]]
        observed.append(values[~np.isnan(values)])
    values = np.concatenate(observed) if observed else np.empty(0)
    model.fallback_level.fill_(float(np.median(values)) if values.size else 0.0)


class InterruptedRun(RuntimeError):  # noqa: N818 - public interruption signal from the run protocol
    """A checkpointed training session ended before a bundle could be written."""


class NonFiniteTraining(RuntimeError):  # noqa: N818 - public training failure signal
    """Consecutive non-finite losses stopped training with a terminal checkpoint."""


# Settings controls consumed by the run protocol rather than the trainer.
TRAIN_CONFIG_EXCLUDE = frozenset(
    {
        "refit",
        "min_train_history",
        "val_items_max",
        "init_from",
        "zero_shot",
        "eval_at_start",
        "trainable",
        "series_embedding",
        "exception_policy",
    }
)


@dataclass
class TrainConfig:
    base_steps: int = 3000
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_every: int = 200
    patience: int = 5
    rel_tol: float = 0.002
    amp: bool = True
    checkpoint_every: int = 200
    max_steps_cap: int | None = None
    warmup_frac: float | None = None
    min_steps: int | None = None
    selection: Literal["best", "smoothed"] = "best"
    weight_average: Literal["none", "ema", "tail"] = "none"
    ema_decay: float = 0.99
    average_last_k: int = 3
    select_min_gain: float = 0.0


@dataclass
class FitResult:
    """Training outcome; ``nonfinite_losses`` marks a checkpointed terminal failure."""

    best_state: dict
    best_score: float | None
    best_step: int
    steps: int
    history: list[dict]
    stop_reason: str
    timings: dict
    telemetry: list[dict]


def resolve_budget(
    base_steps: int,
    n_series: int,
    eval_every: int,
    max_steps_cap: int | None = None,
    min_steps: int | None = None,
) -> tuple[int, int]:
    """Scale and clip steps, optionally cap them, then resolve evaluation frequency."""
    if base_steps < 1 or n_series < 0 or eval_every < 1:
        raise ValueError("base_steps and eval_every must be positive; n_series must be nonnegative")
    if max_steps_cap is not None and max_steps_cap < 1:
        raise ValueError("max_steps_cap must be positive")
    if min_steps is not None and min_steps < 1:
        raise ValueError("min_steps must be positive")
    floor = min(500 if min_steps is None else min_steps, base_steps)
    max_steps = int(np.clip(base_steps * n_series / 2000, floor, 4 * base_steps))
    if max_steps_cap is not None:
        max_steps = min(max_steps, max_steps_cap)
    return max_steps, min(eval_every, max(max_steps // 5, 1))


def resolve_warmup(max_steps: int, warmup_steps: int, warmup_frac: float | None) -> int:
    """Retain absolute warmup unless a fraction of the schedule is requested."""
    return warmup_steps if warmup_frac is None else round(warmup_frac * max_steps)


def evaluation_count(max_steps: int, eval_every: int, eval_at_start: bool = False) -> int:
    """Count planned evaluations, including the final partial interval."""
    return math.ceil(max_steps / eval_every) + int(eval_at_start) if max_steps else 0


def _cpu_state(state: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


class _WeightAverage:
    """A device-local EMA or CPU tail ring, separate from the live optimizer."""

    def __init__(self, cfg: TrainConfig, state: dict):
        self.mode, self.decay, self.k = cfg.weight_average, cfg.ema_decay, cfg.average_last_k
        self.device = next(iter(state.values())).device
        self.shadow = (
            {
                key: value.detach().clone()
                for key, value in state.items()
                if value.is_floating_point()
            }
            if self.mode == "ema"
            else None
        )
        self.tail: list[dict] = []
        self.last_step: int | None = None

    def update(self, state: dict) -> None:
        if self.mode == "ema":
            for key, value in state.items():
                if value.is_floating_point():
                    self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)

    def sample(self, state: dict, step: int) -> None:
        if self.mode == "tail" and self.last_step != step:
            self.tail.append(_cpu_state(state))
            self.tail = self.tail[-self.k :]
            self.last_step = step

    def weights(self, state: dict) -> dict:
        if self.mode == "none":
            return state
        averaged = {}
        for key, value in state.items():
            if not value.is_floating_point():
                averaged[key] = value.detach().cpu().clone()
            elif self.mode == "ema":
                averaged[key] = self.shadow[key]
            elif self.tail:
                total = self.tail[0][key].clone()
                for snapshot in self.tail[1:]:
                    total.add_(snapshot[key])
                averaged[key] = total.div_(len(self.tail))
            else:
                averaged[key] = value.detach().cpu().clone()
        return averaged

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "tail": self.tail, "last_step": self.last_step}

    def load_state_dict(self, state: dict) -> None:
        self.shadow, self.tail, self.last_step = state["shadow"], state["tail"], state["last_step"]
        if self.shadow is not None:
            self.shadow = {key: value.to(self.device) for key, value in self.shadow.items()}


def lr_factor(step: int, *, max_steps: int, warmup_steps: int) -> float:
    """Linear warm-up followed by cosine decay, zero at and beyond the budget."""
    if step >= max_steps:
        return 0.0
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def amp_dtype_for(device: torch.device) -> torch.dtype | None:
    """Use native bf16 on compute capability 8.0+ (Ampere and newer), else fp16.

    ``is_bf16_supported()`` counts emulation, so it is not used. Non-CUDA
    devices return None without querying CUDA.
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def resolve_amp(amp: bool, device: torch.device) -> tuple[bool, torch.dtype, bool]:
    """Resolve autocast and scaling without querying CUDA when AMP is disabled."""
    enabled = amp and device.type == "cuda"
    if not enabled:
        return False, torch.float16, False
    dtype = amp_dtype_for(device)
    assert dtype is not None  # device is CUDA when AMP is enabled
    return True, dtype, dtype is torch.float16


def _evaluate_with_backoff(model, prepared: PreparedValidation, quantiles) -> dict[str, float]:
    """Extend the existing validation back-off down to one row during fit."""
    while True:
        try:
            return evaluate_prepared(model, prepared, quantiles)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            largest = max(batch.size for batch in prepared.batches)
            size = min(prepared.batch_limit.size or largest, largest)
            if size == 1:
                raise
        # Release the failed forward's traceback before reclaiming CUDA memory.
        gc.collect()
        torch.cuda.empty_cache()
        prepared.batch_limit.size = max(1, size // 2)
        logger.warning(
            "Validation OOM: reducing batch size from %d to %d", size, prepared.batch_limit.size
        )


class Trainer:
    """Train a caller-seeded model; the sampler has its own explicit seed."""

    def __init__(
        self,
        model: ForecastModel,
        cfg: TrainConfig,
        device: torch.device | str,
        run_identity: str,
        seed: int,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.cfg, self.run_identity, self.seed = cfg, run_identity, seed
        self._micro_batch_limit = batch_limit("training")
        self.timings = _empty_timings()
        self.telemetry: list[dict] = []
        self.amp, self.amp_dtype, use_scaler = resolve_amp(cfg.amp, self.device)
        # Retain compatibility with installations exposing only the CUDA scaler API.
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def _lr_factor(self, step: int) -> float:
        return lr_factor(step, max_steps=self.max_steps, warmup_steps=self.warmup_steps)

    def _backward_microbatch(self, cpu_batch: Batch, quantiles, total_cells: int | None) -> float:
        # A separate frame releases device inputs and activations both on success
        # (before transferring the next chunk) and when an OOM unwinds the call.
        batch = cpu_batch.to(self.device)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            loss, cells = pinball_loss(
                self.model(batch), batch.labels, batch.label_mask, batch.loss_eligible, quantiles
            )
            if total_cells is not None:
                loss = loss * (cells / max(total_cells, 1))
        value = float(loss.detach().item())
        if math.isfinite(value):
            self.scaler.scale(loss).backward()
        return value

    def _backward_batch(self, cpu_batch: Batch, quantiles) -> float:
        """Retry the same effective batch, discarding all failed-attempt gradients."""
        rng = torch.get_rng_state()
        rng_device = self.device
        cuda_rng = (
            torch.cuda.get_rng_state(rng_device)
            if rng_device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()
            else None
        )
        while True:
            size = min(self._micro_batch_limit.size or cpu_batch.size, cpu_batch.size)
            self.optimizer.zero_grad(set_to_none=True)
            try:
                if size == cpu_batch.size:
                    # Preserve the original loss and gradient arithmetic without OOM.
                    return self._backward_microbatch(cpu_batch, quantiles, None)
                total_cells = int(
                    (cpu_batch.label_mask & cpu_batch.loss_eligible[:, None]).sum().item()
                )
                value = 0.0
                for start in range(0, cpu_batch.size, size):
                    chunk = cpu_batch._map_tensors(lambda tensor: tensor[start : start + size])
                    value += self._backward_microbatch(chunk, quantiles, total_cells)
                    if not math.isfinite(value):
                        self.optimizer.zero_grad(set_to_none=True)
                        break
                return value
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                if size == 1:
                    raise
            # The exception and traceback must be gone before empty_cache().
            gc.collect()
            torch.cuda.empty_cache()
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, rng_device)
            self._micro_batch_limit.size = max(1, size // 2)
            logger.warning(
                "Training OOM: reducing micro-batch size from %d to %d at step %d",
                size,
                self._micro_batch_limit.size,
                self.step + 1,
            )

    def fit(
        self,
        panel: Panel,
        plan: ValidationPlan,
        forecast_cfg: ForecastConfig,
        encoder: StaticEncoder,
        checkpoint_dir: Path | None = None,
        resume: bool = False,
        stop_after: int | None = None,
        evaluate_fn=evaluate,
        eval_at_start: bool = False,
        *,
        step_limit: int | None = None,
        schedule_max_steps: int | None = None,
        keep_initial: bool = False,
    ) -> FitResult:
        """Fit the permitted split, or continue its checkpoint without reseeding.

        ``stop_after`` counts steps in this call, not lifetime steps. If it is
        supplied without a directory, checkpoints default to ``checkpoints/``.
        Checkpoints retain the live optimizer/model pair; best weights are
        restored only when training completes, never on interruption.
        ``eval_at_start`` selects and checkpoints the untrained model exactly
        once before any optimizer step, also defaulting to ``checkpoints/``.
        ``step_limit`` stops a full-history refit at the selected step while
        ``schedule_max_steps`` retains the selection run's LR schedule length.
        Non-finite losses skip backward, optimizer and all scaler operations,
        but advance the scheduler and step count. Skipping the first optimizer
        update therefore produces PyTorch's expected scheduler-order warning.
        After 20 consecutive non-finite losses, checkpoint (defaulting to
        ``checkpoints/``) and return ``nonfinite_losses``; resume preserves
        that terminal result without further training.
        Transfer/forward/backward OOMs retry the same sampled batch with smaller
        micro-batches down to one row. Losses and gradients are weighted by
        observed eligible cells; clipping and optimizer/scheduler updates still
        run once per effective batch. Reductions survive checkpoint/resume and
        are shared with refits inside a task_batch_limits context.
        Optimizer updates are not replayed: AdamW can partially mutate its state
        and parameters before throwing, so retrying requires separate rollback.
        """
        if stop_after is not None and stop_after < 1:
            raise ValueError("stop_after must be positive")
        if self.cfg.batch_size < 1 or self.cfg.checkpoint_every < 1 or self.cfg.patience < 1:
            raise ValueError("batch_size, checkpoint_every and patience must be positive")
        if self.cfg.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if self.cfg.warmup_frac is not None and not 0 <= self.cfg.warmup_frac <= 1:
            raise ValueError("warmup_frac must be between 0 and 1")
        if self.cfg.selection not in {"best", "smoothed"}:
            raise ValueError("unknown selection")
        if self.cfg.weight_average not in {"none", "ema", "tail"}:
            raise ValueError("unknown weight_average")
        if not 0 <= self.cfg.ema_decay < 1 or self.cfg.average_last_k < 1:
            raise ValueError("ema_decay must be in [0, 1); average_last_k must be positive")
        if not 0 <= self.cfg.select_min_gain < 1:
            raise ValueError("select_min_gain must be in [0, 1)")
        if checkpoint_dir is None and (stop_after is not None or resume or eval_at_start):
            checkpoint_dir = Path("checkpoints")
            logger.info("checkpoint_dir not supplied; using %s", checkpoint_dir)
        checkpoint_path = Path(checkpoint_dir) / "latest.pt" if checkpoint_dir is not None else None
        self.progress_path = checkpoint_path.with_name("progress.json") if checkpoint_path else None
        self.plan = plan
        horizon, max_context = forecast_cfg.horizon, forecast_cfg.max_context
        self.max_steps, self.eval_every = resolve_budget(
            self.cfg.base_steps,
            len(plan.fit_ids),
            self.cfg.eval_every,
            self.cfg.max_steps_cap,
            self.cfg.min_steps,
        )
        if schedule_max_steps is not None:
            self.max_steps = schedule_max_steps
            if self.cfg.weight_average == "tail":
                self.eval_every = min(self.cfg.eval_every, max(self.max_steps // 5, 1))
        self.step_limit = self.max_steps if step_limit is None else step_limit
        if not 1 <= self.step_limit <= self.max_steps:
            raise ValueError("step_limit must be between 1 and max_steps")
        self.warmup_steps = resolve_warmup(
            self.max_steps, self.cfg.warmup_steps, self.cfg.warmup_frac
        )
        fit_ids = set(plan.fit_ids)
        fit_indices = [i for i, series in enumerate(panel.series) if series.series_id in fit_ids]
        origins = build_origins(panel, horizon, forecast_cfg.min_context, upto=plan.train_upto)
        origins = origins[np.isin(origins[:, 0], fit_indices)]
        _set_fallback_level(self.model, panel, fit_indices, plan.train_upto)

        self.sampler = WindowSampler(
            panel,
            origins,
            horizon,
            max_context,
            encoder,
            seed=self.seed,
            label_upto=plan.train_upto,
            scale_floor_frac=forecast_cfg.scale_floor_frac,
            calendar_schema=forecast_cfg.calendar_schema,
            target_lags=forecast_cfg.target_lags,
        )
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            parameters, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_factor)
        self.step, self.best_step, self.patience_counter = 0, 0, 0
        self.consecutive_nonfinite = 0
        self.initial_eval_done = False
        self.terminal_reason: str | None = None
        self.history: list[dict] = []
        self.best_score: float | None = None
        self.best_state = deepcopy(self.model.state_dict())
        self.average = _WeightAverage(self.cfg, self.model.state_dict())
        self.selection_states: list[dict] = []
        self.selection_left_score: float | None = None
        self.initial_score: float | None = None
        self.timings, self.telemetry = _empty_timings(), []
        self._last_eval_step, self._last_eval_train_s = 0, 0.0
        if resume and checkpoint_path is not None and checkpoint_path.exists():
            self.resume(checkpoint_path)
        self.timings["sessions"] += 1
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        logger.info(
            "fit: max_steps=%d eval_every=%d fit_ids=%d val_items=%d origins=%d device=%s amp=%s",
            self.max_steps,
            self.eval_every,
            len(plan.fit_ids),
            len(plan.val_items),
            len(origins),
            self.device,
            self.amp_dtype if self.amp else None,
        )
        if (
            self.terminal_reason == "nonfinite_losses"
            or self.consecutive_nonfinite >= NONFINITE_STOP_AFTER
        ):
            return self._result("nonfinite_losses")
        if self.cfg.selection == "smoothed" and self.terminal_reason in {"early_stop", "max_steps"}:
            self.model.load_state_dict(self.best_state)
            return self._result(self.terminal_reason)
        if not len(origins):
            return self._result("no_training_data")

        prepared = None
        if evaluate_fn is evaluate and not plan.exception and len(plan.val_items):
            prepare_started = time.perf_counter()
            prepared = prepare_validation(
                panel,
                plan.val_items,
                plan.val_origin,
                horizon,
                max_context,
                encoder,
                scale_floor_frac=forecast_cfg.scale_floor_frac,
                calendar_schema=forecast_cfg.calendar_schema,
                target_lags=forecast_cfg.target_lags,
            )
            build_s = time.perf_counter() - prepare_started
            self.timings["eval_s"] += build_s
            logger.info(
                "validation prepared: items=%d batches=%d bytes=%d build_s=%.1f",
                len(plan.val_items),
                len(prepared.batches),
                prepared.nbytes,
                build_s,
            )

        if (
            eval_at_start
            and not plan.exception
            and origins.size > 0
            and self.step == 0
            and not self.initial_eval_done
        ):
            eval_started = time.perf_counter()
            with self._evaluation_weights():
                if prepared is not None:
                    metrics = _evaluate_with_backoff(self.model, prepared, forecast_cfg.quantiles)
                else:
                    metrics = evaluate_fn(
                        self.model,
                        panel,
                        plan.val_items,
                        plan.val_origin,
                        horizon,
                        max_context,
                        encoder,
                        forecast_cfg.quantiles,
                    )
                self.history.append({"step": 0, **metrics})
                self._select_best(float(metrics["SQL"]))
            self.initial_eval_done = True
            self.model.train()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            eval_s = time.perf_counter() - eval_started
            self.timings["eval_s"] += eval_s
            self._sample_peak()
            self.telemetry.append(
                {
                    "step": 0,
                    "elapsed_s": sum(
                        self.timings[key] for key in ("train_s", "eval_s", "checkpoint_s")
                    ),
                    "train_s_per_step": None,
                    "eval_s": eval_s,
                    "peak_cuda_mem_bytes": self.timings["peak_cuda_mem_bytes"],
                }
            )
            self.checkpoint(checkpoint_path)

        quantiles = torch.tensor(forecast_cfg.quantiles, device=self.device)
        self.model.train()
        session_steps = 0
        floored_before = self.sampler.builder.floored_target_samples
        while (
            self.step < self.step_limit
            and self.terminal_reason is None
            and (plan.exception or self.patience_counter < self.cfg.patience)
        ):
            if self.sampler.position == len(self.sampler):
                self.sampler.set_epoch(self.sampler.epoch + 1)
            train_started = time.perf_counter()
            for batch in self.sampler.batches(self.cfg.batch_size):
                loss_value = self._backward_batch(batch, quantiles)
                finite_loss = math.isfinite(loss_value)
                if finite_loss:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else None
                    self.scaler.update()
                    skipped_update = (
                        scale_before is not None and self.scaler.get_scale() < scale_before
                    )
                    if skipped_update and self.timings["scaler_skips"] is not None:
                        self.timings["scaler_skips"] += 1
                    if not skipped_update and self.cfg.weight_average == "ema":
                        self.average.update(self.model.state_dict())
                self.scheduler.step()
                self.step += 1
                session_steps += 1
                record = {"step": self.step, "loss": loss_value}
                self.timings["train_s"] += time.perf_counter() - train_started
                self.timings["nonfinite_losses"] += int(not finite_loss)
                self.consecutive_nonfinite = 0 if finite_loss else self.consecutive_nonfinite + 1
                self.timings["cold_covariate_columns"] += int(batch.cold_covariates.sum().item())
                self.timings["clipped_covariate_values"] += int(
                    batch.clipped_covariates.sum().item()
                )
                self.timings["floored_covariate_columns"] += int(
                    batch.floored_covariates.sum().item()
                )
                floored_now = self.sampler.builder.floored_target_samples
                self.timings["floored_target_samples"] += floored_now - floored_before
                floored_before = floored_now
                self.history.append(record)
                if self.consecutive_nonfinite >= NONFINITE_STOP_AFTER:
                    self.terminal_reason = "nonfinite_losses"
                    if checkpoint_path is None:
                        checkpoint_dir = Path("checkpoints")
                        checkpoint_path = checkpoint_dir / "latest.pt"
                        self.progress_path = checkpoint_dir / "progress.json"
                        logger.info("checkpoint_dir not supplied; using %s", checkpoint_dir)
                    self.checkpoint(checkpoint_path)
                    return self._result("nonfinite_losses")
                if plan.exception and (
                    self.step % self.eval_every == 0 or self.step == self.step_limit
                ):
                    # Refit has no validation, but uses the same tail sampling cadence.
                    self.average.sample(self.model.state_dict(), self.step)
                if not plan.exception and (
                    self.step % self.eval_every == 0 or self.step == self.max_steps
                ):
                    eval_started = time.perf_counter()
                    with self._evaluation_weights():
                        if prepared is not None:
                            metrics = _evaluate_with_backoff(
                                self.model, prepared, forecast_cfg.quantiles
                            )
                        else:
                            metrics = evaluate_fn(
                                self.model,
                                panel,
                                plan.val_items,
                                plan.val_origin,
                                horizon,
                                max_context,
                                encoder,
                                forecast_cfg.quantiles,
                            )
                        record.update(metrics)
                        self._select_best(float(metrics["SQL"]))
                    self.model.train()
                    if self.device.type == "cuda":
                        # Include any asynchronous best-state copies in evaluation
                        # time instead of charging them to the next optimizer step.
                        torch.cuda.synchronize(self.device)
                    eval_s = time.perf_counter() - eval_started
                    self.timings["eval_s"] += eval_s
                    self._sample_peak()
                    train_s_per_step = (self.timings["train_s"] - self._last_eval_train_s) / (
                        self.step - self._last_eval_step
                    )
                    elapsed_s = sum(
                        self.timings[key] for key in ("train_s", "eval_s", "checkpoint_s")
                    )
                    self.telemetry.append(
                        {
                            "step": self.step,
                            "elapsed_s": elapsed_s,
                            "train_s_per_step": train_s_per_step,
                            "eval_s": eval_s,
                            "peak_cuda_mem_bytes": self.timings["peak_cuda_mem_bytes"],
                        }
                    )
                    self._last_eval_step, self._last_eval_train_s = (
                        self.step,
                        self.timings["train_s"],
                    )
                    logger.info(
                        "eval: step=%d/%d inner_SQL=%.4f best=%.4f@%d patience=%d "
                        "train_s_per_step=%.4f eval_s=%.1f elapsed_s=%.0f peak_cuda_mem_MiB=%d",
                        self.step,
                        self.max_steps,
                        metrics["SQL"],
                        self.best_score if self.best_score is not None else float("nan"),
                        self.best_step,
                        self.patience_counter,
                        train_s_per_step,
                        eval_s,
                        elapsed_s,
                        self.timings["peak_cuda_mem_bytes"] // 2**20,
                    )
                finished = self.step == self.step_limit or (
                    not plan.exception and self.patience_counter >= self.cfg.patience
                )
                interrupted = stop_after is not None and session_steps >= stop_after
                if finished and not plan.exception and self.cfg.selection == "smoothed":
                    # Fix the stopping boundary before the final edge can reset patience.
                    self.terminal_reason = (
                        "early_stop" if self.patience_counter >= self.cfg.patience else "max_steps"
                    )
                    self._finish_selection()
                if plan.exception and (finished or interrupted) and not keep_initial:
                    self.best_state = deepcopy(self.average.weights(self.model.state_dict()))
                    self.best_step = self.step
                if checkpoint_path is not None and (
                    self.step % self.cfg.checkpoint_every == 0 or finished or interrupted
                ):
                    self.checkpoint(checkpoint_path)
                if finished:
                    break
                if interrupted:
                    return self._result("interrupted")
                train_started = time.perf_counter()

        reason = (
            "exception_budget"
            if plan.exception
            else ("early_stop" if self.patience_counter >= self.cfg.patience else "max_steps")
        )
        if not plan.exception and self.cfg.selection == "smoothed":
            reason = self.terminal_reason or reason
        if not plan.exception or keep_initial or self.cfg.weight_average != "none":
            self.model.load_state_dict(self.best_state)
        return self._result(reason)

    def _select_best(self, score: float) -> None:
        if self.step == 0:
            self.initial_score = score
        if self.cfg.selection == "smoothed":
            if len(self.selection_states) == 2:
                self.selection_left_score = self.selection_states.pop(0)["score"]
            self.selection_states.append(
                {"step": self.step, "score": score, "state": _cpu_state(self.model.state_dict())}
            )
            if len(self.selection_states) == 2:
                values = [entry["score"] for entry in self.selection_states]
                if self.selection_left_score is not None:
                    values.insert(0, self.selection_left_score)
                self._select_smoothed(self.selection_states[0], sum(values) / len(values))
        else:
            self._consider_state(score, self.step, self.model.state_dict())

    def _finish_selection(self) -> None:
        if self.selection_states:
            values = [entry["score"] for entry in self.selection_states]
            self._select_smoothed(self.selection_states[-1], sum(values) / len(values))

    def _select_smoothed(self, entry: dict, score: float) -> None:
        self._consider_state(score, entry["step"], entry["state"])
        # Raw SQL remains available; expose the actual selection criterion too.
        record = next(
            record for record in reversed(self.history) if record["step"] == entry["step"]
        )
        record["selection_score"] = score

    def _consider_state(self, score: float, step: int, state: dict) -> None:
        """Keep every absolute improvement, but reset patience only past tolerance."""
        eligible = not (
            step > 0
            and self.cfg.select_min_gain > 0
            and self.initial_score is not None
            and not score < self.initial_score * (1 - self.cfg.select_min_gain)
        )
        previous = self.best_score
        improved = (
            eligible
            and math.isfinite(score)
            and (previous is None or score < previous * (1 - self.cfg.rel_tol))
        )
        self.patience_counter = 0 if improved else self.patience_counter + 1
        if eligible and math.isfinite(score) and (previous is None or score < previous):
            self.best_score, self.best_step = score, step
            self.best_state = deepcopy(state)

    @contextmanager
    def _evaluation_weights(self):
        if self.cfg.weight_average == "none":
            yield
            return
        live = _cpu_state(self.model.state_dict())
        self.average.sample(live, self.step)
        self.model.load_state_dict(self.average.weights(live))
        try:
            yield
        finally:
            self.model.load_state_dict(live)

    def _result(self, reason: str) -> FitResult:
        best_state = deepcopy(self.best_state)
        self._sample_peak()
        self._write_progress(reason)
        logger.info("fit done: stop_reason=%s steps=%d", reason, self.step)
        return FitResult(
            best_state=best_state,
            best_score=self.best_score,
            best_step=self.best_step,
            steps=self.step,
            history=deepcopy(self.history),
            stop_reason=reason,
            timings=deepcopy(self.timings),
            telemetry=deepcopy(self.telemetry),
        )

    def _sample_peak(self) -> None:
        if self.device.type == "cuda":
            self.timings["peak_cuda_mem_bytes"] = max(
                self.timings["peak_cuda_mem_bytes"], torch.cuda.max_memory_allocated(self.device)
            )

    def _write_progress(self, stop_reason: str | None = None) -> None:
        from demand_forecast.core.run import _write_run

        if self.progress_path is not None:
            _write_run(
                self.progress_path,
                {
                    "run_identity": self.run_identity,
                    "step": self.step,
                    "stop_reason": stop_reason,
                    "max_steps": self.max_steps,
                    "eval_every": self.eval_every,
                    "n_fit_ids": len(self.plan.fit_ids),
                    "n_val_items": len(self.plan.val_items),
                    "exception": self.plan.exception,
                    "device": str(self.device),
                    "amp_dtype": str(self.amp_dtype).removeprefix("torch.") if self.amp else None,
                    "timings": self.timings,
                    "telemetry_tail": self.telemetry[-3:],
                },
            )

    def checkpoint(self, path: Path) -> None:
        """Atomically save the live training state, including all RNG streams."""
        started = time.perf_counter()
        self._sample_peak()
        state = {
            "run_identity": self.run_identity,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "micro_batch_size": self._micro_batch_limit.size,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "sampler": self.sampler.state(),
            "sampler_seed": self.sampler.seed,
            "history": self.history,
            "initial_eval_done": self.initial_eval_done,
            "best_state": self.best_state,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "patience_counter": self.patience_counter,
            "consecutive_nonfinite": self.consecutive_nonfinite,
            "terminal_reason": self.terminal_reason,
            "max_steps": self.max_steps,
            "step_limit": self.step_limit,
            "eval_every": self.eval_every,
            "warmup_steps": self.warmup_steps,
            "timings": self.timings,
            "telemetry": self.telemetry,
            "last_eval_step": self._last_eval_step,
            "last_eval_train_s": self._last_eval_train_s,
        }
        if self.cfg.weight_average != "none":
            state["weight_average"] = self.average.state_dict()
        if self.cfg.selection == "smoothed":
            state["selection_states"] = self.selection_states
            state["selection_left_score"] = self.selection_left_score
        if self.initial_score is not None:
            state["initial_score"] = self.initial_score
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as file:
            temporary = Path(file.name)
        try:
            torch.save(state, temporary)
            # The footer includes this write's measured serialization duration.
            # Footer/rename/progress publication are excluded: a file cannot
            # contain the duration of its own final publication. torch.load
            # ignores this member; resume reads it as the authoritative timings.
            self.timings["checkpoint_s"] += time.perf_counter() - started
            with ZipFile(temporary, "a") as archive:
                root = archive.namelist()[0].split("/")[0]
                archive.writestr(f"{root}/timings.json", json.dumps(self.timings, allow_nan=False))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._write_progress(self.terminal_reason)
        logger.info("checkpoint: step=%d path=%s", self.step, path)

    def resume(self, path: Path) -> None:
        """Restore a checkpoint after fit has constructed optimizer and sampler."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("run_identity") != self.run_identity:
            raise ValueError("run identity mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        saved_limit = state.get("micro_batch_size")
        if saved_limit is not None:
            self._micro_batch_limit.size = min(
                self._micro_batch_limit.size or saved_limit, saved_limit
            )
        self.sampler.seed = state["sampler_seed"]
        self.sampler.load_state(state["sampler"])
        self.step_limit = state.get("step_limit", state["max_steps"])
        for name in (
            "step",
            "history",
            "best_state",
            "best_score",
            "best_step",
            "patience_counter",
            "max_steps",
            "eval_every",
            "warmup_steps",
        ):
            setattr(self, name, state[name])
        # Legacy checkpoints cannot establish a cumulative skip total. The
        # payload may supply it; the measured ZIP footer takes precedence.
        self.timings["scaler_skips"] = None
        self.consecutive_nonfinite = state.get("consecutive_nonfinite", 0)
        self.initial_eval_done = state.get("initial_eval_done", False)
        self.terminal_reason = state.get("terminal_reason")
        if "weight_average" in state:
            self.average.load_state_dict(state["weight_average"])
        self.selection_states = state.get("selection_states", [])
        self.selection_left_score = state.get("selection_left_score")
        self.initial_score = state.get("initial_score")
        if self.initial_score is None and self.initial_eval_done:
            self.initial_score = next(
                (entry["SQL"] for entry in self.history if entry["step"] == 0 and "SQL" in entry),
                None,
            )
        self.timings.update(state.get("timings", {}))
        with ZipFile(path) as archive:
            root = archive.namelist()[0].split("/")[0]
            member = f"{root}/timings.json"
            if member in archive.namelist():
                self.timings.update(json.loads(archive.read(member)))
        self.telemetry = state.get("telemetry", [])
        self._last_eval_step = state.get("last_eval_step", self.step)
        self._last_eval_train_s = state.get("last_eval_train_s", self.timings["train_s"])
        # Restore last: model/optimizer/sampler construction must not consume
        # any part of the checkpoint's continuation stream.
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
