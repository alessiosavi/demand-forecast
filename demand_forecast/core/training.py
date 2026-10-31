"""Quantile training with split-aware sampling, SQL selection and exact resume."""

from __future__ import annotations

import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core.evaluation import evaluate
from demand_forecast.core.validation import ValidationPlan
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import grid_range
from demand_forecast.data.panel import Panel
from demand_forecast.data.sampler import WindowSampler, build_origins
from demand_forecast.models.base import ForecastModel
from demand_forecast.models.losses import pinball_loss


class InterruptedRun(RuntimeError):  # noqa: N818 - public interruption signal from the run protocol
    """A checkpointed training session ended before a bundle could be written."""


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


@dataclass
class FitResult:
    best_state: dict
    best_score: float | None
    best_step: int
    steps: int
    history: list[dict]
    stop_reason: str


def resolve_budget(base_steps: int, n_series: int, eval_every: int) -> tuple[int, int]:
    """Scale steps by panel size, retaining small budgets used by CPU tests."""
    if base_steps < 1 or n_series < 0 or eval_every < 1:
        raise ValueError("base_steps and eval_every must be positive; n_series must be nonnegative")
    max_steps = int(np.clip(base_steps * n_series / 2000, min(500, base_steps), 4 * base_steps))
    return max_steps, min(eval_every, max(max_steps // 5, 1))


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
        self.amp = cfg.amp and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if self.amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        use_scaler = self.amp and self.amp_dtype == torch.float16
        # The unified scaler API arrived after torch 2.0, our minimum version.
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def _lr_factor(self, step: int) -> float:
        if step >= self.max_steps:
            return 0.0
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

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
    ) -> FitResult:
        """Fit the permitted split, or continue its checkpoint without reseeding.

        ``stop_after`` counts steps in this call, not lifetime steps. If it is
        supplied without a directory, checkpoints default to ``checkpoints/``.
        Checkpoints retain the live optimizer/model pair; best weights are
        restored only when training completes, never on interruption.
        """
        if stop_after is not None and stop_after < 1:
            raise ValueError("stop_after must be positive")
        if self.cfg.batch_size < 1 or self.cfg.checkpoint_every < 1 or self.cfg.patience < 1:
            raise ValueError("batch_size, checkpoint_every and patience must be positive")
        if self.cfg.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if checkpoint_dir is None and (stop_after is not None or resume):
            checkpoint_dir = Path("checkpoints")
        checkpoint_path = Path(checkpoint_dir) / "latest.pt" if checkpoint_dir is not None else None
        horizon, max_context = forecast_cfg.horizon, forecast_cfg.max_context
        self.max_steps, self.eval_every = resolve_budget(
            self.cfg.base_steps, len(plan.fit_ids), self.cfg.eval_every
        )
        self.warmup_steps = self.cfg.warmup_steps
        fit_ids = set(plan.fit_ids)
        fit_indices = [i for i, series in enumerate(panel.series) if series.series_id in fit_ids]
        origins = build_origins(panel, horizon, forecast_cfg.min_context, upto=plan.train_upto)
        origins = origins[np.isin(origins[:, 0], fit_indices)]
        observed = []
        cutoff_positions = {}
        for i in fit_indices:
            series = panel.series[i]
            key = (series.start, series.history_length)
            if key not in cutoff_positions:
                grid = grid_range(series.start, series.history_length, panel.freq)
                cutoff_positions[key] = int(grid.searchsorted(plan.train_upto, side="right"))
            values = series.target[: cutoff_positions[key]]
            observed.append(values[~np.isnan(values)])
        values = np.concatenate(observed) if observed else np.empty(0)
        self.model.fallback_level.fill_(float(np.median(values)) if values.size else 0.0)

        self.sampler = WindowSampler(
            panel,
            origins,
            horizon,
            max_context,
            encoder,
            seed=self.seed,
            label_upto=plan.train_upto,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_factor)
        self.step, self.best_step, self.patience_counter = 0, 0, 0
        self.history: list[dict] = []
        self.best_score: float | None = None
        self.best_state = deepcopy(self.model.state_dict())
        if resume and checkpoint_path is not None and checkpoint_path.exists():
            self.resume(checkpoint_path)
        if not len(origins):
            return self._result("no_training_data")

        quantiles = torch.tensor(forecast_cfg.quantiles, device=self.device)
        self.model.train()
        session_steps = 0
        while self.step < self.max_steps and (
            plan.exception or self.patience_counter < self.cfg.patience
        ):
            if self.sampler.position == len(self.sampler):
                self.sampler.set_epoch(self.sampler.epoch + 1)
            for batch in self.sampler.batches(self.cfg.batch_size):
                batch = batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                    loss, _ = pinball_loss(
                        self.model(batch),
                        batch.labels,
                        batch.label_mask,
                        batch.loss_eligible,
                        quantiles,
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.step += 1
                session_steps += 1
                record = {"step": self.step, "loss": float(loss.detach().item())}
                self.history.append(record)
                if not plan.exception and (
                    self.step % self.eval_every == 0 or self.step == self.max_steps
                ):
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
                finished = self.step == self.max_steps or (
                    not plan.exception and self.patience_counter >= self.cfg.patience
                )
                interrupted = stop_after is not None and session_steps >= stop_after
                if plan.exception and (finished or interrupted):
                    self.best_state = deepcopy(self.model.state_dict())
                    self.best_step = self.step
                if checkpoint_path is not None and (
                    self.step % self.cfg.checkpoint_every == 0 or finished or interrupted
                ):
                    self.checkpoint(checkpoint_path)
                if interrupted:
                    return self._result("interrupted")
                if finished:
                    break

        reason = (
            "exception_budget"
            if plan.exception
            else ("early_stop" if self.patience_counter >= self.cfg.patience else "max_steps")
        )
        if not plan.exception:
            self.model.load_state_dict(self.best_state)
        return self._result(reason)

    def _select_best(self, score: float) -> None:
        """Keep every absolute improvement, but reset patience only past tolerance."""
        previous = self.best_score
        improved = math.isfinite(score) and (
            previous is None or score < previous * (1 - self.cfg.rel_tol)
        )
        self.patience_counter = 0 if improved else self.patience_counter + 1
        if math.isfinite(score) and (previous is None or score < previous):
            self.best_score, self.best_step = score, self.step
            self.best_state = deepcopy(self.model.state_dict())

    def _result(self, reason: str) -> FitResult:
        return FitResult(
            best_state=deepcopy(self.best_state),
            best_score=self.best_score,
            best_step=self.best_step,
            steps=self.step,
            history=deepcopy(self.history),
            stop_reason=reason,
        )

    def checkpoint(self, path: Path) -> None:
        """Atomically save the live training state, including all RNG streams."""
        state = {
            "run_identity": self.run_identity,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "sampler": self.sampler.state(),
            "sampler_seed": self.sampler.seed,
            "history": self.history,
            "best_state": self.best_state,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "patience_counter": self.patience_counter,
            "max_steps": self.max_steps,
            "eval_every": self.eval_every,
            "warmup_steps": self.warmup_steps,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as file:
            temporary = Path(file.name)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def resume(self, path: Path) -> None:
        """Restore a checkpoint after fit has constructed optimizer and sampler."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("run_identity") != self.run_identity:
            raise ValueError("run identity mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.sampler.seed = state["sampler_seed"]
        self.sampler.load_state(state["sampler"])
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
        # Restore last: model/optimizer/sampler construction must not consume
        # any part of the checkpoint's continuation stream.
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
