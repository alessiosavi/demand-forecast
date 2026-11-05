"""Reproducible training runs and long-panel forecasts from portable bundles."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch

from demand_forecast import __version__
from demand_forecast.config import Settings
from demand_forecast.core.bundle import load_bundle, load_bundle_manifest, save_bundle
from demand_forecast.core.training import (
    NONFINITE_STOP_AFTER,
    InterruptedRun,
    NonFiniteTraining,
    TrainConfig,
    Trainer,
    _empty_timings,
    _set_fallback_level,
    resolve_budget,
)
from demand_forecast.core.transfer import init_from_pretrained
from demand_forecast.core.validation import plan_inner_validation, subsample_validation
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import grid_range, validate_on_grid
from demand_forecast.data.loader_long import load_long
from demand_forecast.data.panel import Panel
from demand_forecast.inference.predictor import Predictor
from demand_forecast.models.base import ModelSpec
from demand_forecast.models.registry import build_model

logger = logging.getLogger(__name__)


def _identity_settings(settings: Settings) -> dict:
    """Preserve legacy defaults and identify initialization by bundle provenance."""
    payload = settings.model_dump(mode="json")
    training = payload["training"]
    for key in ("max_steps_cap", "val_items_max", "init_from"):
        if training[key] is None:
            training.pop(key)
    for key in ("zero_shot", "eval_at_start"):
        if training[key] is False:
            training.pop(key)
    for key, default in (("trainable", "all"), ("series_embedding", True)):
        if training[key] == default:
            training.pop(key)
    if payload["forecast"]["scale_floor_frac"] == 0.0:
        payload["forecast"].pop("scale_floor_frac")
    if settings.training.init_from is not None:
        path = Path(settings.training.init_from)
        manifest = load_bundle_manifest(path)
        if "pretrain_identity" not in manifest:
            raise ValueError(f"Bundle {path} is missing pretrain_identity")
        training["init_from"] = {
            "pretrain_identity": manifest["pretrain_identity"],
            "model_sha256": manifest["sha256"]["model.pt"],
        }
    return payload


def run_identity(
    settings: Settings,
    seed: int,
    dataset_fingerprint: str,
    package_version: str,
    origin: pd.Timestamp | str,
    task_definition: dict | None,
    *,
    policy: str = "first_window",
) -> str:
    """Hash canonical configuration, data, software and resolved training boundary."""
    payload = {
        "settings": _identity_settings(settings),
        "seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "package_version": package_version,
        "origin": pd.Timestamp(origin).isoformat(),
        "task_definition": task_definition,
    }
    if policy not in {"first_window", "per_window"}:
        raise ValueError(f"unknown fev policy: {policy}")
    if policy == "per_window":
        payload["policy"] = policy
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_device(settings: Settings) -> torch.device:
    """Use the requested device, otherwise CUDA when available or CPU."""
    return torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _check_refit(settings: Settings) -> None:
    if settings.training.zero_shot and settings.training.refit == "full":
        raise ValueError("zero_shot is incompatible with refit=full")


def _initial_model(settings: Settings, spec: ModelSpec):
    """Reproduce the task initialization, including transfer and frozen parameters."""
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    model = build_model(settings.model.family, spec, settings.model.backbone_kwargs())
    transfer = None
    if settings.training.init_from:
        source = load_bundle(settings.training.init_from, device="cpu")
        transferred = init_from_pretrained(model, source)
        transfer = {
            "pretrain_identity": source.manifest["pretrain_identity"],
            "model_sha256": source.manifest["sha256"]["model.pt"],
            "copied": transferred.copied,
            "zeroed": transferred.zeroed,
            "fresh": transferred.fresh,
        }
    if settings.training.trainable == "heads_only":
        model.backbone.requires_grad_(False)
    return model, transfer


def _check_fit_result(result, checkpoint: Path) -> None:
    if result.stop_reason == "nonfinite_losses":
        raise NonFiniteTraining(
            f"Training stopped after {NONFINITE_STOP_AFTER} consecutive non-finite losses "
            f"at step {result.steps}; checkpoint {checkpoint}"
        )
    if result.stop_reason == "interrupted":
        raise InterruptedRun(
            f"Training interrupted at step {result.steps}; resume from {checkpoint}"
        )


def _combine_timings(selection: dict, refit: dict) -> dict:
    """Sum measured phase counters and durations, retaining the peak memory maximum."""
    return {
        key: (
            None
            if value is None or refit[key] is None
            else max(value, refit[key])
            if key == "peak_cuda_mem_bytes"
            else value + refit[key]
        )
        for key, value in selection.items()
    }


def _finite_json(value):
    """Keep strict JSON while representing unavailable diagnostics as null."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_run(path: Path, report: dict) -> None:
    """Publish a complete report, retaining the old one until replacement succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".run-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(_finite_json(report), indent=2, allow_nan=False) + "\n")
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def train_panel(
    panel: Panel,
    origin: pd.Timestamp,
    settings: Settings,
    out_dir: Path,
    *,
    dataset_fingerprint: str,
    task_definition: dict | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    earliest_origin: pd.Timestamp | None = None,
    policy: str = "first_window",
) -> Path:
    """Fit the permitted history, checkpoint interruptions and save the selected model.

    Resume checks every existing run/checkpoint/bundle identity, even when a
    completed bundle is reusable. The report is also retained in the manifest
    so an interruption between bundle and report writes needs no retraining.
    ``earliest_origin`` defaults to ``origin`` and sets the bundle's global
    permissible forecast floor, not a guarantee about any individual item's past.
    """
    if settings.training.zero_shot and not settings.training.init_from:
        raise ValueError("zero_shot requires init_from")
    _check_refit(settings)
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    started = time.perf_counter()
    origin, out_dir = pd.Timestamp(origin), Path(out_dir)
    validate_on_grid(pd.DatetimeIndex([origin]), panel.freq, "training origin")
    explicit_earliest_origin = earliest_origin is not None
    earliest_origin = origin if earliest_origin is None else pd.Timestamp(earliest_origin)
    validate_on_grid(pd.DatetimeIndex([origin, earliest_origin]), panel.freq, "earliest_origin")
    if earliest_origin > origin:
        raise ValueError("earliest_origin after origin")
    for series in panel.series:
        validate_on_grid(pd.DatetimeIndex([series.start, origin]), panel.freq, series.series_id)
    identity = run_identity(
        settings,
        settings.seed,
        dataset_fingerprint,
        __version__,
        origin,
        task_definition,
        policy=policy,
    )
    bundle_dir = out_dir / "bundle"
    checkpoint = out_dir / "checkpoints/latest.pt"
    report_path = out_dir / "run.json"
    selection_path = out_dir / "checkpoints/selection.json"
    refit_checkpoint = out_dir / "checkpoints/refit/latest.pt"
    selection = None
    resumed_from_step = 0
    if not resume and settings.training.refit == "full":
        selection_path.unlink(missing_ok=True)
        refit_checkpoint.unlink(missing_ok=True)
    if resume:
        # A completed bundle must not hide a mismatched checkpoint or report.
        failed_checkpoint = False
        unfinished_phase = False
        for existing_checkpoint in (checkpoint, refit_checkpoint):
            if not existing_checkpoint.exists():
                continue
            state = torch.load(existing_checkpoint, map_location="cpu", weights_only=False)
            if state.get("run_identity") != identity:
                raise ValueError("run identity mismatch")
            if existing_checkpoint == checkpoint:
                resumed_from_step = state["step"]
            failed_checkpoint |= (
                state.get("terminal_reason") == "nonfinite_losses"
                or state.get("consecutive_nonfinite", 0) >= NONFINITE_STOP_AFTER
            )
            if settings.training.refit == "full":
                limit = state.get("step_limit", state["max_steps"])
                unfinished_phase |= state["step"] < limit and (
                    existing_checkpoint == refit_checkpoint
                    or state["patience_counter"] < settings.training.patience
                )
            del state
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection.get("run_identity") != identity:
                raise ValueError("run identity mismatch")
            unfinished_phase |= not refit_checkpoint.exists()
        for path in (report_path, bundle_dir / "manifest.json"):
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")).get("run_identity") != identity:
                    raise ValueError("run identity mismatch")
        # A failed rerun's checkpoint takes precedence over an older bundle;
        # let the trainer restore and publish its terminal result below.
        if (
            not failed_checkpoint
            and not unfinished_phase
            and (bundle_dir / "manifest.json").exists()
        ):
            bundle = load_bundle(bundle_dir)
            if explicit_earliest_origin and earliest_origin != pd.Timestamp(
                bundle.panel_metadata["earliest_origin"]
            ):
                raise ValueError("bundle earliest_origin mismatch")
            if not report_path.exists():
                _write_run(report_path, bundle.manifest["run"])
            return bundle_dir

    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    plan = plan_inner_validation(
        panel, origin, settings.forecast.horizon, settings.training.min_train_history
    )
    steps, _ = resolve_budget(
        settings.training.base_steps,
        len(plan.fit_ids),
        settings.training.eval_every,
        settings.training.max_steps_cap,
    )
    logger.info(
        "training budget: task=%s items=%d max_steps=%d",
        (task_definition or {}).get("name", "long"),
        len(plan.fit_ids),
        0 if settings.training.zero_shot else steps,
    )
    eligible = plan
    plan = subsample_validation(plan, settings.training.val_items_max, settings.seed)
    logger.info(
        "plan: train_upto=%s val_origin=%s exception=%s val_items=%d/%d",
        plan.train_upto,
        plan.val_origin,
        plan.exception,
        len(plan.val_items),
        len(eligible.val_items),
    )
    encoder = StaticEncoder().fit(
        panel, plan.fit_ids, series_embedding=settings.training.series_embedding
    )
    spec = ModelSpec.from_panel(
        panel, encoder, settings.forecast, settings.model.d_model, settings.model.cat_emb_dim
    )
    backbone_kwargs = settings.model.backbone_kwargs()
    model, transfer = _initial_model(settings, spec)
    device = resolve_device(settings)
    refit = {
        "mode": settings.training.refit,
        "performed": False,
        "steps": None,
        "train_upto": None,
        "reason": None,
    }
    training_cutoff = plan.train_upto
    validation_history = []
    if settings.training.zero_shot:
        fit_ids = set(plan.fit_ids)
        fit_indices = [i for i, series in enumerate(panel.series) if series.series_id in fit_ids]
        _set_fallback_level(model, panel, fit_indices, plan.train_upto)
        model.to(device)
        fit_started = time.perf_counter()
        training_time_s = 0.0
        fit_result = {
            "steps": 0,
            "best_step": 0,
            "best_score": None,
            "stop_reason": "zero_shot",
            "nonfinite_losses": 0,
        }
        resolved_budget = {"max_steps": 0, "eval_every": 0}
        n_val_items, amp_dtype, telemetry = 0, None, []
        timings = _empty_timings() | {"sessions": 1}
        _write_run(
            out_dir / "checkpoints/progress.json",
            {
                "run_identity": identity,
                "step": 0,
                "stop_reason": "zero_shot",
                **resolved_budget,
                "n_fit_ids": len(plan.fit_ids),
                "n_val_items": 0,
                "exception": plan.exception,
                "device": str(device),
                "amp_dtype": None,
                "timings": timings,
                "telemetry_tail": [],
            },
        )
    else:
        cfg = TrainConfig(
            **settings.training.model_dump(
                exclude={
                    "refit",
                    "min_train_history",
                    "val_items_max",
                    "init_from",
                    "zero_shot",
                    "eval_at_start",
                    "trainable",
                    "series_embedding",
                }
            )
        )
        fit_started = time.perf_counter()
        selection_steps_this_call = 0
        if selection is None:
            trainer = Trainer(model, cfg, device, identity, settings.seed)
            result = trainer.fit(
                panel,
                plan,
                settings.forecast,
                encoder,
                checkpoint_dir=out_dir / "checkpoints",
                resume=resume,
                stop_after=stop_after,
                eval_at_start=settings.training.eval_at_start,
            )
            _check_fit_result(result, checkpoint)
            selection_steps_this_call = result.steps - resumed_from_step
            selection = {
                "run_identity": identity,
                "fit_result": {
                    "steps": result.steps,
                    "best_step": result.best_step,
                    "best_score": result.best_score,
                    "stop_reason": result.stop_reason,
                    "nonfinite_losses": result.timings["nonfinite_losses"],
                },
                "resolved_budget": {
                    "max_steps": trainer.max_steps,
                    "eval_every": trainer.eval_every,
                },
                "amp_dtype": str(trainer.amp_dtype).removeprefix("torch.") if trainer.amp else None,
                "timings": result.timings,
                "telemetry": result.telemetry,
                "validation_history": [entry for entry in result.history if "SQL" in entry],
            }
            # A completed selection is durable before the first refit step.
            if settings.training.refit == "full" and not plan.exception and result.best_step > 0:
                _write_run(selection_path, selection)
            del trainer, result
        fit_result = selection["fit_result"]
        resolved_budget = selection["resolved_budget"]
        n_val_items = len(plan.val_items)
        amp_dtype = selection["amp_dtype"]
        timings, telemetry = selection["timings"], selection["telemetry"]
        validation_history = selection["validation_history"]
        if settings.training.refit == "full":
            if plan.exception:
                refit["reason"] = "no_inner_validation"
            elif fit_result["best_step"] == 0:
                refit["reason"] = "best_step_zero"
            else:
                remaining = None if stop_after is None else stop_after - selection_steps_this_call
                if remaining is not None and remaining <= 0:
                    raise InterruptedRun(f"Selection complete; resume refit from {selection_path}")
                full_plan = plan_inner_validation(
                    panel,
                    origin,
                    settings.forecast.horizon,
                    settings.training.min_train_history,
                    full_history=True,
                )
                # Keep the selection encoder/spec so initial weights and vocabularies
                # are identical. Newly observed fit ids use its unknown codes.
                del model
                model, _ = _initial_model(settings, spec)
                trainer = Trainer(model, cfg, device, identity, settings.seed)
                refit_result = trainer.fit(
                    panel,
                    full_plan,
                    settings.forecast,
                    encoder,
                    checkpoint_dir=refit_checkpoint.parent,
                    resume=resume,
                    stop_after=remaining,
                    step_limit=fit_result["best_step"],
                    schedule_max_steps=resolved_budget["max_steps"],
                )
                _check_fit_result(refit_result, refit_checkpoint)
                timings = _combine_timings(timings, refit_result.timings)
                training_cutoff = full_plan.train_upto
                refit.update(
                    performed=True,
                    steps=refit_result.steps,
                    train_upto=training_cutoff.isoformat(),
                )
        training_time_s = time.perf_counter() - fit_started

    report = {
        "run_identity": identity,
        "settings": settings.model_dump(mode="json"),
        "policy": policy,
        "seed": settings.seed,
        "dataset_fingerprint": dataset_fingerprint,
        "package_version": __version__,
        "origin": origin.isoformat(),
        "earliest_origin": earliest_origin.isoformat(),
        "task_definition": task_definition,
        "resumed_from_step": resumed_from_step,
        "device": str(device),
        "resolved_budget": resolved_budget,
        "n_fit_ids": len(plan.fit_ids),
        "n_val_items": n_val_items,
        "val_items_max": settings.training.val_items_max,
        "n_val_items_eligible": len(eligible.val_items),
        "amp_dtype": amp_dtype,
        "telemetry": telemetry,
        "validation_history": validation_history,
        "refit": refit,
        "trainable": {
            "mode": settings.training.trainable,
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "params_total": sum(p.numel() for p in model.parameters()),
        },
        "plan": {
            "train_upto": plan.train_upto.isoformat(),
            "val_origin": plan.val_origin.isoformat(),
            "val_items": plan.val_items.tolist(),
            "exception": plan.exception,
            "fit_ids": plan.fit_ids,
        },
        "fit_result": fit_result,
        # Session timings exclude serialization and earlier interrupted sessions.
        "timings": {
            "setup_time_s": fit_started - started,
            "training_time_s": training_time_s,
            "cumulative": timings,
        },
    }
    if transfer is not None:
        report["transfer"] = transfer
    report = _finite_json(report)
    save_bundle(
        bundle_dir,
        model=model,
        family=settings.model.family,
        backbone_kwargs=backbone_kwargs,
        forecast_cfg=settings.forecast,
        dataset_cfg=settings.data,
        encoder=encoder,
        panel=panel,
        training_cutoff=training_cutoff,
        earliest_origin=earliest_origin,
        manifest={
            "run_identity": identity,
            "seed": settings.seed,
            "dataset_fingerprint": dataset_fingerprint,
            "task_definition": task_definition,
            "resumed_from_step": resumed_from_step,
            "training_time_s": training_time_s,
            "run": report,
            **({"zero_shot": True} if settings.training.zero_shot else {}),
        },
    )
    _write_run(report_path, report)
    return bundle_dir


def _input_fingerprint(settings: Settings) -> str:
    """SHA256 of long then static file bytes, with lengths framing each file."""
    digest = hashlib.sha256()
    for path in (settings.data.path, settings.data.static_path):
        if path is not None:
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _require_long(settings: Settings) -> None:
    if settings.data.source != "long":
        raise ValueError("This command requires data.source=long")


def train_from_settings(settings: Settings, out_dir: Path, resume: bool = False) -> Path:
    """Train immediately after the latest history, excluding future-only rows.

    An entirely unobserved panel has no last observation: use the latest series
    start as its origin, yielding an empty fit split and a zero fallback bundle.
    """
    _check_refit(settings)
    _require_long(settings)
    panel = load_long(settings.data)
    if not panel.series:
        raise ValueError("Cannot train an empty panel")
    ends = [
        grid_range(series.start, series.history_length, panel.freq)[-1]
        for series in panel.series
        if series.history_length
    ]
    origin = (
        grid_range(max(ends), 2, panel.freq)[1]
        if ends
        else max(series.start for series in panel.series)
    )
    return train_panel(
        panel,
        origin,
        settings,
        out_dir,
        dataset_fingerprint=_input_fingerprint(settings),
        resume=resume,
    )


def forecast_from_bundle(
    bundle_dir: Path, settings: Settings, origin: pd.Timestamp, out: Path
) -> pd.DataFrame:
    """Load the declared long panel and write bundle forecasts in original units."""
    _require_long(settings)
    panel = load_long(settings.data)
    predictor = Predictor.from_bundle(bundle_dir, device=resolve_device(settings))
    forecasts = predictor.forecast(panel, pd.Timestamp(origin))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out, index=False)
    return forecasts
