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
from demand_forecast.core.bundle import load_bundle, save_bundle
from demand_forecast.core.training import InterruptedRun, TrainConfig, Trainer
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import grid_range, validate_on_grid
from demand_forecast.data.loader_long import load_long
from demand_forecast.data.panel import Panel
from demand_forecast.inference.predictor import Predictor
from demand_forecast.models.base import ModelSpec
from demand_forecast.models.registry import build_model

logger = logging.getLogger(__name__)


def run_identity(
    settings: Settings,
    seed: int,
    dataset_fingerprint: str,
    package_version: str,
    origin: pd.Timestamp | str,
    task_definition: dict | None,
) -> str:
    """Hash canonical configuration, data, software and resolved training boundary."""
    payload = {
        "settings": settings.model_dump(mode="json"),
        "seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "package_version": package_version,
        "origin": pd.Timestamp(origin).isoformat(),
        "task_definition": task_definition,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_device(settings: Settings) -> torch.device:
    """Use the requested device, otherwise CUDA when available or CPU."""
    return torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _check_refit(settings: Settings) -> None:
    if settings.training.refit == "full":
        raise NotImplementedError("refit=full arrives with the phase-4 plan")


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
) -> Path:
    """Fit the permitted history, checkpoint interruptions and save the selected model.

    Resume checks every existing run/checkpoint/bundle identity, even when a
    completed bundle is reusable. The report is also retained in the manifest
    so an interruption between bundle and report writes needs no retraining.
    """
    _check_refit(settings)
    started = time.perf_counter()
    origin, out_dir = pd.Timestamp(origin), Path(out_dir)
    validate_on_grid(pd.DatetimeIndex([origin]), panel.freq, "training origin")
    for series in panel.series:
        validate_on_grid(pd.DatetimeIndex([series.start, origin]), panel.freq, series.series_id)
    identity = run_identity(
        settings, settings.seed, dataset_fingerprint, __version__, origin, task_definition
    )
    bundle_dir = out_dir / "bundle"
    checkpoint = out_dir / "checkpoints/latest.pt"
    report_path = out_dir / "run.json"
    resumed_from_step = 0
    if resume:
        # A completed bundle must not hide a mismatched checkpoint or report.
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state.get("run_identity") != identity:
                raise ValueError("run identity mismatch")
            resumed_from_step = state["step"]
            del state
        for path in (report_path, bundle_dir / "manifest.json"):
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")).get("run_identity") != identity:
                    raise ValueError("run identity mismatch")
        if (bundle_dir / "manifest.json").exists():
            bundle = load_bundle(bundle_dir)
            if not report_path.exists():
                _write_run(report_path, bundle.manifest["run"])
            return bundle_dir

    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    plan = plan_inner_validation(
        panel, origin, settings.forecast.horizon, settings.training.min_train_history
    )
    logger.info(
        "plan: train_upto=%s val_origin=%s exception=%s",
        plan.train_upto,
        plan.val_origin,
        plan.exception,
    )
    encoder = StaticEncoder().fit(panel, plan.fit_ids)
    spec = ModelSpec.from_panel(
        panel, encoder, settings.forecast, settings.model.d_model, settings.model.cat_emb_dim
    )
    backbone_kwargs = settings.model.backbone_kwargs()
    model = build_model(settings.model.family, spec, backbone_kwargs)
    cfg = TrainConfig(**settings.training.model_dump(exclude={"refit", "min_train_history"}))
    device = resolve_device(settings)
    trainer = Trainer(model, cfg, device, identity, settings.seed)
    fit_started = time.perf_counter()
    result = trainer.fit(
        panel,
        plan,
        settings.forecast,
        encoder,
        checkpoint_dir=out_dir / "checkpoints",
        resume=resume,
        stop_after=stop_after,
    )
    training_time_s = time.perf_counter() - fit_started
    if result.stop_reason == "interrupted":
        raise InterruptedRun(
            f"Training interrupted at step {result.steps}; resume from {checkpoint}"
        )

    report = {
        "run_identity": identity,
        "settings": settings.model_dump(mode="json"),
        "seed": settings.seed,
        "dataset_fingerprint": dataset_fingerprint,
        "package_version": __version__,
        "origin": origin.isoformat(),
        "task_definition": task_definition,
        "resumed_from_step": resumed_from_step,
        "device": str(device),
        "resolved_budget": {"max_steps": trainer.max_steps, "eval_every": trainer.eval_every},
        "n_fit_ids": len(plan.fit_ids),
        "n_val_items": len(plan.val_items),
        "amp_dtype": str(trainer.amp_dtype).removeprefix("torch.") if trainer.amp else None,
        "telemetry": result.telemetry,
        "plan": {
            "train_upto": plan.train_upto.isoformat(),
            "val_origin": plan.val_origin.isoformat(),
            "val_items": plan.val_items.tolist(),
            "exception": plan.exception,
            "fit_ids": plan.fit_ids,
        },
        "fit_result": {
            "steps": result.steps,
            "best_step": result.best_step,
            "best_score": result.best_score,
            "stop_reason": result.stop_reason,
            "nonfinite_losses": result.timings["nonfinite_losses"],
        },
        # Session timings exclude serialization and earlier interrupted sessions.
        "timings": {
            "setup_time_s": fit_started - started,
            "training_time_s": training_time_s,
            "cumulative": result.timings,
        },
    }
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
        training_cutoff=plan.train_upto,
        earliest_origin=origin,
        manifest={
            "run_identity": identity,
            "seed": settings.seed,
            "dataset_fingerprint": dataset_fingerprint,
            "task_definition": task_definition,
            "resumed_from_step": resumed_from_step,
            "training_time_s": training_time_s,
            "run": report,
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
