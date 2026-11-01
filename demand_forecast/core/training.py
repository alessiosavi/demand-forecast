"""Quantile training with split-aware sampling, SQL selection and exact resume."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

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

logger = logging.getLogger(__name__)


def _empty_timings() -> dict:
    return dict(
        train_s=0.0,
        eval_s=0.0,
        checkpoint_s=0.0,
        sessions=0,
        peak_cuda_mem_bytes=0,
        nonfinite_losses=0,
        cold_covariate_columns=0,
        clipped_covariate_values=0,
        floored_covariate_columns=0,
    )


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
    max_steps_cap: int | None = None


@dataclass
class FitResult:
    best_state: dict
    best_score: float | None
    best_step: int
    steps: int
    history: list[dict]
    stop_reason: str
    timings: dict
    telemetry: list[dict]


def resolve_budget(
    base_steps: int, n_series: int, eval_every: int, max_steps_cap: int | None = None
) -> tuple[int, int]:
    """Scale and clip steps, optionally cap them, then resolve evaluation frequency."""
    if base_steps < 1 or n_series < 0 or eval_every < 1:
        raise ValueError("base_steps and eval_every must be positive; n_series must be nonnegative")
    if max_steps_cap is not None and max_steps_cap < 1:
        raise ValueError("max_steps_cap must be positive")
    max_steps = int(np.clip(base_steps * n_series / 2000, min(500, base_steps), 4 * base_steps))
    if max_steps_cap is not None:
        max_steps = min(max_steps, max_steps_cap)
    return max_steps, min(eval_every, max(max_steps // 5, 1))


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
        self.timings = _empty_timings()
        self.telemetry: list[dict] = []
        self.amp, self.amp_dtype, use_scaler = resolve_amp(cfg.amp, self.device)
        # Retain compatibility with installations exposing only the CUDA scaler API.
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
        self.progress_path = checkpoint_path.with_name("progress.json") if checkpoint_path else None
        self.plan = plan
        horizon, max_context = forecast_cfg.horizon, forecast_cfg.max_context
        self.max_steps, self.eval_every = resolve_budget(
            self.cfg.base_steps, len(plan.fit_ids), self.cfg.eval_every, self.cfg.max_steps_cap
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
            train_started = time.perf_counter()
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
                self.timings["train_s"] += time.perf_counter() - train_started
                self.timings["nonfinite_losses"] += int(not math.isfinite(record["loss"]))
                self.timings["cold_covariate_columns"] += int(batch.cold_covariates.sum().item())
                self.timings["clipped_covariate_values"] += int(
                    batch.clipped_covariates.sum().item()
                )
                self.timings["floored_covariate_columns"] += int(
                    batch.floored_covariates.sum().item()
                )
                self.history.append(record)
                if not plan.exception and (
                    self.step % self.eval_every == 0 or self.step == self.max_steps
                ):
                    eval_started = time.perf_counter()
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
        best_state = deepcopy(self.best_state)
        self._sample_peak()
        self._write_progress()
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

    def _write_progress(self) -> None:
        from demand_forecast.core.run import _write_run

        if self.progress_path is not None:
            _write_run(
                self.progress_path,
                {
                    "run_identity": self.run_identity,
                    "step": self.step,
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
            "timings": self.timings,
            "telemetry": self.telemetry,
            "last_eval_step": self._last_eval_step,
            "last_eval_train_s": self._last_eval_train_s,
        }
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
        self._write_progress()
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
