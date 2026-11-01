"""Resumable model and reference forecasts with official fev evaluation."""

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import pandas as pd

from demand_forecast.bench.specs import FevTaskSpec, dataset_fingerprint
from demand_forecast.config import DatasetConfig, ForecastConfig, Settings

logger = logging.getLogger(__name__)
DIAGNOSTIC_DTYPES = {
    "inference_time_s": np.float64,
    "fallback_no_context": np.int64,
    "fallback_undefined_scale": np.int64,
    "device": str,
    "cold_covariate_columns": np.int64,
    "clipped_covariate_values": np.int64,
    "floored_covariate_columns": np.int64,
}
OPTIONAL_DIAGNOSTICS = {
    "cold_covariate_columns",
    "clipped_covariate_values",
    "floored_covariate_columns",
}


def settings_for_task(spec: FevTaskSpec, base: Settings, freq: str) -> Settings:
    """Copy the base controls, replacing data roles and the forecast task."""
    settings = base.model_copy(deep=True)
    settings.data = DatasetConfig(
        source="fev",
        id_columns=[spec.id_column],
        timestamp_column=spec.timestamp_column,
        target=spec.target,
        target_support=spec.target_support,
        known_dynamic=spec.known_dynamic,
        past_dynamic=spec.past_dynamic,
        static_categorical=spec.categorical_statics,
        static_numeric=[s for s in spec.static_columns if s not in spec.categorical_statics],
        freq=freq,
        missing_target="nan",
    )
    settings.forecast = ForecastConfig(
        **{
            **base.forecast.model_dump(),
            "horizon": spec.horizon,
            "quantiles": spec.quantile_levels,
        }
    )
    return settings


def window_origins(
    future_data, id_column: str, timestamp_column: str = "timestamp"
) -> dict[str, pd.Timestamp]:
    """Read each item's first future timestamp, rejecting empty or duplicate rows."""
    origins = {}
    for row in future_data:
        key = str(row[id_column])
        if key in origins:
            raise ValueError("duplicate future series id")
        if len(row[timestamp_column]) == 0:
            raise ValueError(f"empty future rows for item {key}")
        origins[key] = pd.Timestamp(row[timestamp_column][0])
    return origins


def resolved_origin(origins: Mapping[str, pd.Timestamp]) -> pd.Timestamp:
    """Use the latest item origin for window identity, validation and training."""
    if not origins:
        raise ValueError("window has no items")
    return max(origins.values())


def predictions_for_fev(
    ids: list[str], arrays: np.ndarray, item_ids: list[str], quantiles: list[float]
) -> list[dict]:
    """Align original-unit quantiles to fev's positional input order by item id."""
    arrays = np.asarray(arrays)
    if arrays.ndim != 3 or arrays.shape[0] != len(ids) or arrays.shape[2] != len(quantiles):
        raise ValueError("forecast arrays must have shape (items, horizon, quantiles)")
    if len(set(ids)) != len(ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate forecast item ids")
    positions = {key: index for index, key in enumerate(ids)}
    if any(key not in positions for key in item_ids):
        raise ValueError("missing forecast for item")
    if len(ids) != len(item_ids):
        raise ValueError("unexpected forecast item")
    median_index = list(quantiles).index(0.5)
    return [
        {
            "predictions": arrays[positions[key], :, median_index].tolist(),
            **{str(q): arrays[positions[key], :, i].tolist() for i, q in enumerate(quantiles)},
        }
        for key in item_ids
    ]


def _check_prediction_identity(saved, identity: str) -> None:
    if "identity" not in saved or not np.array_equal(saved["identity"], np.asarray(identity)):
        raise ValueError("run identity mismatch")


def prediction_diagnostics(saved) -> dict | None:
    """Read scalar diagnostics; legacy windows remain unknown, never zero."""
    if not all(key in saved for key in DIAGNOSTIC_DTYPES.keys() - OPTIONAL_DIAGNOSTICS):
        return None
    return {key: saved[key].item() if key in saved else None for key in DIAGNOSTIC_DTYPES}


def _save_predictions(
    path: Path, ids: list[str], arrays: np.ndarray, identity: str, diagnostics: dict
) -> None:
    """Publish a complete window so an interrupted write leaves it missing."""
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            np.savez(
                stream,
                ids=np.asarray(ids, dtype=str),
                arrays=arrays,
                identity=identity,
                **{
                    key: np.asarray(diagnostics[key], dtype=dtype)
                    for key, dtype in DIAGNOSTIC_DTYPES.items()
                },
            )
            stream.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def run_fev(
    specs: Iterable[FevTaskSpec],
    settings: Settings,
    out_dir: Path,
    policy: Literal["first_window", "per_window"] = "first_window",
    resume: bool = False,
    model_name: str | None = None,
    stop_after: int | None = None,
) -> pd.DataFrame:
    """Train once per task, extend context per window, and replace summary rows.

    Resume validates saved identities and computes only missing artifacts.
    Timings come from persisted trainer and prediction measurements, including
    earlier sessions. Fully verified completed tasks retain their summary row.

    Early-ending items are forecast for calendar periods that other items'
    available past (and selection labels) already cover. No item's own future
    is used: fev truncates each item at its own cutoff and the loader reads only
    known covariates from future rows. This equals the information set fev gives
    every model and is not an out-of-time test for those items.
    """
    from demand_forecast import __version__
    from demand_forecast.core.run import resolve_device, run_identity, train_panel
    from demand_forecast.data.loader_fev import item_ids, panel_from_window
    from demand_forecast.inference.predictor import Predictor

    if policy == "per_window":
        raise NotImplementedError("per_window arrives with the phase-4 plan")
    if policy != "first_window":
        raise ValueError(f"unknown fev policy: {policy}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries.csv"
    summaries = (
        pd.read_csv(path, float_precision="round_trip")
        if path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    model_name = model_name or f"demand-forecast-{settings.model.family}"
    for spec in specs:
        task = spec.to_task()
        fp = dataset_fingerprint(task)
        task_settings = settings_for_task(spec, settings, task.freq)
        definition = spec.task_definition()
        task_dir = out_dir / spec.name
        predictions = []
        diagnostics_per_window = []
        max_distinct_origins = 0
        skipped = False
        for index, window in enumerate(task.iter_windows()):
            past, future = window.get_input_data()
            started = time.perf_counter()
            panel = panel_from_window(past, future, spec, task.freq)
            logger.info(
                "panel: task=%s window=%d series=%d build_s=%.1f",
                spec.name,
                index,
                len(panel.series),
                time.perf_counter() - started,
            )
            origins = window_origins(future, spec.id_column, spec.timestamp_column)
            origin = resolved_origin(origins)
            earliest = min(origins.values())
            n_origins = len(set(origins.values()))
            max_distinct_origins = max(max_distinct_origins, n_origins)
            logger.info(
                "origins: task=%s window=%d distinct=%d earliest=%s latest=%s",
                spec.name,
                index,
                n_origins,
                earliest,
                origin,
            )
            if index == 0:
                identity = run_identity(
                    task_settings, task_settings.seed, fp, __version__, origin, definition
                )
                logger.info("identity: task=%s %s", spec.name, identity)
                if resume:
                    # Check even later windows before doing any training or inference.
                    for prediction_path in task_dir.glob("window_*.npz"):
                        with np.load(prediction_path, allow_pickle=False) as saved:
                            _check_prediction_identity(saved, identity)
                reused = resume and (task_dir / "bundle/manifest.json").exists()
                train_panel(
                    panel,
                    origin,
                    task_settings,
                    task_dir,
                    dataset_fingerprint=fp,
                    task_definition=definition,
                    resume=resume,
                    stop_after=stop_after,
                    earliest_origin=earliest,
                )
                report = json.loads((task_dir / "run.json").read_text(encoding="utf-8"))
                cumulative = report["timings"].get("cumulative", {})
                training_time_s = (
                    sum(cumulative[key] for key in ("train_s", "eval_s", "checkpoint_s"))
                    if all(
                        cumulative.get(key) is not None
                        for key in ("train_s", "eval_s", "checkpoint_s")
                    )
                    else None
                )
                rows = summaries.loc[
                    (summaries.task == spec.name) & (summaries.model_name == model_name)
                ]
                if reused and len(rows) == 1 and rows.iloc[0].get("run_identity") == identity:
                    complete = True
                    for i in range(spec.num_windows):
                        saved_path = task_dir / f"window_{i}.npz"
                        if not saved_path.exists():
                            complete = False
                            break
                        with np.load(saved_path, allow_pickle=False) as saved:
                            _check_prediction_identity(saved, identity)
                            if prediction_diagnostics(saved) is None:
                                complete = False
                    if complete:
                        print(f"Skipping completed task {spec.name}", flush=True)
                        skipped = True
                        break
            prediction_path = task_dir / f"window_{index}.npz"
            if resume and prediction_path.exists():
                with np.load(prediction_path, allow_pickle=False) as saved:
                    _check_prediction_identity(saved, identity)
                    ids, arrays = saved["ids"].tolist(), saved["arrays"]
                    diagnostics = prediction_diagnostics(saved)
            else:
                predictor = Predictor.from_bundle(
                    task_dir / "bundle", device=resolve_device(settings)
                )
                ids, arrays, diagnostics = predictor.forecast_per_item_origins(panel, origins)
                _save_predictions(prediction_path, ids, arrays, identity, diagnostics)
            diagnostics_per_window.append(diagnostics)
            predictions.append(
                predictions_for_fev(ids, arrays, item_ids(panel), task.quantile_levels)
            )
        if skipped:
            continue

        def diagnostic_sum(key):
            if key in OPTIONAL_DIAGNOSTICS:
                values = [
                    d[key] for d in diagnostics_per_window if d is not None and d[key] is not None
                ]
                return sum(values) if values else None
            return (
                sum(d[key] for d in diagnostics_per_window)
                if all(d is not None for d in diagnostics_per_window)
                else None
            )

        devices = {d["device"] for d in diagnostics_per_window if d is not None}
        summary = {
            "task": spec.name,
            **task.evaluation_summary(
                predictions,
                model_name=model_name,
                trained_on_this_dataset=False,
                training_time_s=training_time_s,
                inference_time_s=diagnostic_sum("inference_time_s"),
                extra_info={
                    "run_identity": identity,
                    "max_distinct_origins": max_distinct_origins,
                    "fallback_no_context": diagnostic_sum("fallback_no_context"),
                    "fallback_undefined_scale": diagnostic_sum("fallback_undefined_scale"),
                    "cold_covariate_columns": diagnostic_sum("cold_covariate_columns"),
                    "clipped_covariate_values": diagnostic_sum("clipped_covariate_values"),
                    "floored_covariate_columns": diagnostic_sum("floored_covariate_columns"),
                    "inference_device": ",".join(sorted(devices))
                    if all(d is not None for d in diagnostics_per_window)
                    else "unknown",
                    "resolved_max_steps": report.get("resolved_budget", {}).get("max_steps"),
                    "n_fit_ids": report.get("n_fit_ids"),
                    "n_val_items": report.get("n_val_items"),
                    "nonfinite_losses": cumulative.get("nonfinite_losses"),
                    "sessions": cumulative.get("sessions"),
                },
                per_quantile_scores=True,
            ),
        }
        summaries = summaries.loc[
            ~((summaries["task"] == spec.name) & (summaries["model_name"] == model_name))
        ]
        row = pd.DataFrame([summary])
        summaries = row if summaries.empty else pd.concat([summaries, row], ignore_index=True)
        _save_summaries(summaries, path)
    if not path.exists():
        _save_summaries(summaries, path)
    return pd.read_csv(path, float_precision="round_trip")


def run_baseline(specs: Iterable[FevTaskSpec], out_dir: Path) -> pd.DataFrame:
    """Save each completed task; skip existing seasonal-naive task rows.

    Resume is keyed by task name and model. Use a new output directory when
    changing a task definition or its dataset. The returned frame uses the
    CSV representation, including string-valued lists and metric definitions.
    """
    try:
        from demand_forecast.bench.baselines import SeasonalNaiveBaseline, baseline_summary
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries.csv"
    summaries = (
        pd.read_csv(path, float_precision="round_trip")
        if path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    completed = set(
        summaries.loc[summaries["model_name"] == SeasonalNaiveBaseline.model_name, "task"]
    )
    for spec in specs:
        if spec.name in completed:
            print(f"Skipping completed task {spec.name}", flush=True)
            continue
        summary = {"task": spec.name, **baseline_summary(spec.to_task())}
        row = pd.DataFrame([summary])
        summaries = row if summaries.empty else pd.concat([summaries, row], ignore_index=True)
        _save_summaries(summaries, path)
        completed.add(spec.name)
    if not path.exists():
        _save_summaries(summaries, path)
    return pd.read_csv(path, float_precision="round_trip")


def _save_summaries(summaries: pd.DataFrame, path: Path) -> None:
    """Replace the CSV only once its complete contents have been written."""
    with NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        summaries.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def leaderboard(summaries_csv: Path, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Return fev's model-indexed skill scores with 1,000 bootstrap resamples."""
    try:
        import fev
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc

    summaries = pd.read_csv(summaries_csv, float_precision="round_trip")
    return fev.leaderboard(summaries, baseline_model=baseline, n_resamples=1000)
