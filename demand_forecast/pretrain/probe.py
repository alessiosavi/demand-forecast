"""Local timing probes using the actual corpus and benchmark tensor shapes."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from demand_forecast.bench.specs import FevTaskSpec
from demand_forecast.config import Settings
from demand_forecast.core.run import _write_run, resolve_device
from demand_forecast.core.training import Trainer
from demand_forecast.pretrain.config import PretrainConfig
from demand_forecast.pretrain.train import _peak_rss, _PretrainLoop, _seed, _sync


def _reset_peak(device: torch.device) -> None:
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak(device: torch.device) -> int | None:
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None


class _ProbeTrainer(Trainer):
    """Wait for each update before Trainer records its per-step wall time."""

    def _lr_factor(self, step: int) -> float:
        # Trainer advances its scheduler after the optimizer/scaler and before
        # closing the step timer. Synchronizing here also covers skipped updates.
        _sync(self.device)
        return super()._lr_factor(step)


def _shard_transition(loop: _PretrainLoop) -> float:
    """Measure one uncached-in-memory transition without training through a shard.

    Disable prefetch only in this independent stream, so the measurement includes
    an actual read, panel/draw setup and its first batch. Public resume state
    skips the remaining batches of the first shard without optimizer work.
    """
    from demand_forecast.pretrain.sampler import ShardStream

    cfg = loop.cfg
    stream = ShardStream(
        loop.sampler.tsmixup.refs,
        "tsmixup",
        horizon=cfg.forecast.horizon,
        max_context=cfg.forecast.max_context,
        min_context=cfg.forecast.min_context,
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
        encoder=loop.encoder,
        prefetch=0,
    )
    try:
        next(stream)
        n_eligible = sum(series.length > cfg.forecast.min_context for series in loop.panel.series)
        stream.load_state(
            replace(stream.state(), batch_cursor=math.ceil(n_eligible / cfg.training.batch_size))
        )
        started = time.perf_counter()
        next(stream)
        return time.perf_counter() - started
    finally:
        stream.close()


def pretrain_probe(cfg: PretrainConfig, *, steps: int = 50, out: Path) -> dict:
    """Warm up for ten steps, then measure training and independent overheads."""
    if steps < 1:
        raise ValueError("steps must be positive")
    started = time.perf_counter()
    loop = _PretrainLoop(cfg)
    try:
        _sync(loop.device)
        init_s = time.perf_counter() - started
        for _ in range(10):
            batch, batch_s, _ = loop.next_batch()
            loop.train_step(batch, batch_s)
        _reset_peak(loop.device)
        train_s = loop.timings["train_s"]
        batch_times = []
        for _ in range(steps):
            batch, batch_s, load_s = loop.next_batch()
            loop.train_step(batch, batch_s)
            if not load_s:
                batch_times.append(batch_s)
        nonfinite_losses = loop.timings["nonfinite_losses"]
        if nonfinite_losses != 0:
            raise RuntimeError(f"pretraining probe observed nonfinite_losses={nonfinite_losses}")
        s_per_step = (loop.timings["train_s"] - train_s) / steps
        # A tiny shard can transition on every step. Measure a resident batch
        # explicitly in that case, without adding any optimizer updates.
        if not batch_times:
            from demand_forecast.data.sampler import BatchBuilder

            builder = BatchBuilder(
                loop.panel,
                cfg.forecast.horizon,
                cfg.forecast.max_context,
                loop.encoder,
                calendar=False,
            )
            items = [
                (i, cfg.forecast.min_context)
                for i, series in enumerate(loop.panel.series)
                if series.length > cfg.forecast.min_context
            ][: cfg.training.batch_size]
            started = time.perf_counter()
            builder.build(items)
            batch_times.append(time.perf_counter() - started)
        train_peak = _cuda_peak(loop.device)
        shard_load_s = _shard_transition(loop)
        with TemporaryDirectory(prefix="pretrain-probe-") as directory:
            started = time.perf_counter()
            loop.checkpoint(Path(directory) / "latest.pt")
            checkpoint_s = time.perf_counter() - started
        _reset_peak(loop.device)
        started = time.perf_counter()
        scores = loop.validation.evaluate(loop.model, device=loop.device)
        _sync(loop.device)
        eval_s = time.perf_counter() - started
        val_sql = scores["val_sql"]
        if val_sql is None or not math.isfinite(val_sql):
            raise RuntimeError(
                f"pretraining probe requires a finite val_sql; observed val_sql={val_sql}"
            )
        eval_peak = _cuda_peak(loop.device)
        report = dict(
            nonfinite_losses=nonfinite_losses,
            val_sql=val_sql,
            s_per_step=s_per_step,
            batch_build_s_per_step=float(np.mean(batch_times)),
            init_s=init_s,
            shard_load_s=shard_load_s,
            checkpoint_s=checkpoint_s,
            eval_s=eval_s,
            eval_peak_cuda_mem_bytes=eval_peak,
            peak_cuda_mem_bytes=max(train_peak, eval_peak) if train_peak is not None else None,
            peak_rss_bytes=_peak_rss(),
            params=sum(p.numel() for p in loop.model.parameters()),
        )
        _write_run(Path(out), report)
        return report
    finally:
        loop.sampler.close()


def _window_panel(task, index: int, spec: FevTaskSpec, max_series: int | None):
    from demand_forecast.bench.run import window_origins
    from demand_forecast.data.loader_fev import panel_from_window

    past, future = task.get_window(index).get_input_data()
    if max_series is not None:
        past = past.select(range(min(max_series, len(past))))
        ids = {str(row[spec.id_column]) for row in past}
        future = future.select(
            [i for i, row in enumerate(future) if str(row[spec.id_column]) in ids]
        )
    panel = panel_from_window(past, future, spec, task.freq)
    return panel, window_origins(future, spec.id_column, spec.timestamp_column)


def task_probe(
    specs: list[FevTaskSpec],
    settings: Settings,
    *,
    steps: int = 30,
    out: Path,
    max_series: int | None = None,
) -> dict:
    """Measure first-window training/validation and last-window inference per task."""
    from demand_forecast.bench.run import resolved_origin, settings_for_task
    from demand_forecast.core.bundle import Bundle
    from demand_forecast.core.evaluation import evaluate_prepared, prepare_validation
    from demand_forecast.core.training import (
        NonFiniteTraining,
        TrainConfig,
        resolve_budget,
    )
    from demand_forecast.core.validation import plan_inner_validation, subsample_validation
    from demand_forecast.data.encoders import StaticEncoder
    from demand_forecast.inference.predictor import Predictor
    from demand_forecast.models.base import ModelSpec
    from demand_forecast.models.registry import build_model

    if steps < 1 or (max_series is not None and max_series < 1):
        raise ValueError("steps and max_series must be positive")
    device = resolve_device(settings)
    out = Path(out)
    reports = {}
    for spec in specs:
        _seed(settings.seed)
        _reset_peak(device)
        started = time.perf_counter()
        task = spec.to_task()
        panel, origins = _window_panel(task, 0, spec, max_series)
        task_settings = settings_for_task(spec, settings, task.freq)
        forecast = task_settings.forecast
        plan = plan_inner_validation(
            panel, resolved_origin(origins), forecast.horizon, settings.training.min_train_history
        )
        plan = subsample_validation(plan, settings.training.val_items_max, settings.seed)
        encoder = StaticEncoder().fit(panel, plan.fit_ids)
        model_spec = ModelSpec.from_panel(
            panel, encoder, forecast, settings.model.d_model, settings.model.cat_emb_dim
        )
        model = build_model(settings.model.family, model_spec, settings.model.backbone_kwargs()).to(
            device
        )
        _sync(device)
        init_s = time.perf_counter() - started
        resolved_steps, resolved_eval = resolve_budget(
            settings.training.base_steps,
            len(plan.fit_ids),
            settings.training.eval_every,
            settings.training.max_steps_cap,
        )

        # Keep Trainer's actual split and sampler, but suppress model selection
        # during the bounded timing run. Validation is measured exactly once below.
        train_cfg = TrainConfig(
            **settings.training.model_dump(
                exclude={
                    "refit",
                    "min_train_history",
                    "val_items_max",
                    "init_from",
                    "zero_shot",
                    "eval_at_start",
                }
            )
        )
        train_cfg.batch_size = 128
        train_cfg.base_steps = train_cfg.max_steps_cap = steps
        trainer = _ProbeTrainer(model, train_cfg, device, f"probe-{spec.name}", settings.seed)
        _sync(device)
        started = time.perf_counter()
        result = trainer.fit(panel, replace(plan, exception=True), forecast, encoder)
        _sync(device)
        fit_wall_s = time.perf_counter() - started
        # Trainer's train_s excludes origin enumeration, fallback and optimizer
        # setup, and the initial state copy. Charge that overhead to init once.
        fit_setup_s = fit_wall_s - result.timings["train_s"]
        init_s += fit_setup_s
        if result.stop_reason == "nonfinite_losses":
            raise NonFiniteTraining(f"Task probe stopped on non-finite losses: {spec.name}")
        if not result.steps:
            raise ValueError(f"task has no training origins: {spec.name}")
        s_per_step = result.timings["train_s"] / result.steps
        del result
        with TemporaryDirectory(prefix="task-probe-") as directory:
            _sync(device)
            started = time.perf_counter()
            trainer.checkpoint(Path(directory) / "latest.pt")
            _sync(device)
            ckpt_s = time.perf_counter() - started

        started = time.perf_counter()
        prepared = prepare_validation(
            panel,
            plan.val_items,
            plan.val_origin,
            forecast.horizon,
            forecast.max_context,
            encoder,
            batch_size=512,
        )
        prep_s = time.perf_counter() - started
        _reset_peak(device)
        started = time.perf_counter()
        evaluate_prepared(model, prepared, forecast.quantiles)
        _sync(device)
        eval_s = time.perf_counter() - started
        val_peak = _cuda_peak(device)
        del prepared
        # Inference uses only the model, as it would after loading a bundle.
        # Trainer's bound scheduler callback forms a cycle; collect it before
        # measuring inference so optimizer/best-state copies do not inflate VRAM.
        del trainer
        gc.collect()
        model.zero_grad(set_to_none=True)

        started = time.perf_counter()
        last_panel, last_origins = _window_panel(task, spec.num_windows - 1, spec, max_series)
        predictor = Predictor(
            Bundle(
                model,
                settings.model.family,
                settings.model.backbone_kwargs(),
                forecast,
                task_settings.data,
                encoder,
                {
                    "freq": panel.freq,
                    "seasonal_period": panel.seasonal_period,
                    "target_support": panel.target_support,
                    "roles": {
                        "target": panel.roles.target,
                        "known_dynamic": panel.roles.known_dynamic,
                        "past_dynamic": panel.roles.past_dynamic,
                        "static_categorical": panel.roles.static_categorical,
                        "static_numeric": panel.roles.static_numeric,
                    },
                    "earliest_origin": min(origins.values()).isoformat(),
                },
                {},
            )
        )
        _sync(device)
        init_last_s = time.perf_counter() - started
        _reset_peak(device)
        started = time.perf_counter()
        predictor.forecast_per_item_origins(last_panel, last_origins, batch_size=512)
        _sync(device)
        infer_s = time.perf_counter() - started
        report = dict(
            init_s=init_s,
            fit_setup_s=fit_setup_s,  # Included in init_s; not an additional cost.
            prep_s=prep_s,
            eval_s=eval_s,
            ckpt_s=ckpt_s,
            s_per_step=s_per_step,
            n_fit_ids=len(plan.fit_ids),
            n_val_items=len(plan.val_items),
            resolved_max_steps=resolved_steps,
            resolved_eval_every=resolved_eval,
            windows=spec.num_windows,
            init_last_s=init_last_s,
            infer_s=infer_s,
            infer_peak_cuda_mem_bytes=_cuda_peak(device),
            val_peak_cuda_mem_bytes=val_peak,
            peak_rss_bytes=_peak_rss(),
        )
        _write_run(out / f"{spec.name}.json", report)
        reports[spec.name] = report
        # Release this model before constructing the next task's model.
        del predictor, model, panel, last_panel, task
    return reports
