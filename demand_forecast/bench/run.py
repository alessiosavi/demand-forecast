"""Resumable model and reference forecasts with official fev evaluation."""

import hashlib
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
    "floored_target_samples": np.int64,
}
OPTIONAL_DIAGNOSTICS = {
    "cold_covariate_columns",
    "clipped_covariate_values",
    "floored_covariate_columns",
    "floored_target_samples",
}


def settings_for_task(spec: FevTaskSpec, base: Settings, freq: str) -> Settings:
    """Copy the base controls, replacing data roles and the forecast task."""
    settings = base.model_copy(deep=True)
    settings.data = DatasetConfig(
        source="fev",
        id_columns=[spec.id_column],
        timestamp_column=spec.timestamp_column,
        target=spec.target if isinstance(spec.target, str) else "__fev_value__",
        target_support=spec.target_support,
        known_dynamic=spec.known_dynamic,
        past_dynamic=spec.past_dynamic,
        static_categorical=spec.categorical_statics
        + (["__fev_target__"] if isinstance(spec.target, list) else []),
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


def expanded_window_origins(future_data, spec: FevTaskSpec) -> dict[str, pd.Timestamp]:
    """Use the same item/target identities as the fev panel loader."""
    from demand_forecast.data.loader_fev import target_series_id

    origins = window_origins(future_data, spec.id_column, spec.timestamp_column)
    if isinstance(spec.target, str):
        return origins
    return {
        target_series_id(key, target): origin
        for target in spec.targets
        for key, origin in origins.items()
    }


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
                    if key in diagnostics or key not in OPTIONAL_DIAGNOSTICS
                },
            )
            stream.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _completed_training(directory: Path, identity: str, earliest, settings: Settings) -> bool:
    """Verify reusable artifacts without needing any observation arrays.

    Keep the terminal/refit guards from train_panel: a prior successful bundle
    cannot hide a failed or unfinished rerun. Such cases take the normal path.
    """
    import torch

    from demand_forecast.core.run import load_bundle
    from demand_forecast.core.training import NONFINITE_STOP_AFTER

    checkpoint = directory / "checkpoints/latest.pt"
    refit_checkpoint = directory / "checkpoints/refit/latest.pt"
    selection_path = directory / "checkpoints/selection.json"
    reusable = True
    for path in (checkpoint, refit_checkpoint):
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("run_identity") != identity:
            raise ValueError("run identity mismatch")
        reusable &= not (
            state.get("terminal_reason") == "nonfinite_losses"
            or state.get("consecutive_nonfinite", 0) >= NONFINITE_STOP_AFTER
        )
        if settings.training.refit == "full":
            limit = state.get("step_limit", state["max_steps"])
            reusable &= not (
                state["step"] < limit
                and (
                    path == refit_checkpoint
                    or (
                        state.get("terminal_reason") not in {"early_stop", "max_steps"}
                        and state["patience_counter"] < settings.training.patience
                    )
                )
            )
        del state
    for path in (selection_path, directory / "run.json", directory / "bundle/manifest.json"):
        if (
            path.exists()
            and json.loads(path.read_text(encoding="utf-8")).get("run_identity") != identity
        ):
            raise ValueError("run identity mismatch")
    if selection_path.exists() and not refit_checkpoint.exists():
        reusable = False
    if not reusable:
        return False
    bundle = load_bundle(directory / "bundle")
    if earliest != pd.Timestamp(bundle.panel_metadata["earliest_origin"]):
        raise ValueError("bundle earliest_origin mismatch")
    return True


def _completed_task(task, spec, settings, directory, row, fingerprint, policy, identities) -> bool:
    """Skip only after validating every completed window and its training artifacts."""
    from demand_forecast import __version__
    from demand_forecast.core.run import run_identity

    training_dirs = (
        [directory]
        if policy == "first_window"
        else [directory / f"window_{index}" for index in range(spec.num_windows)]
    )
    if not all((directory / f"window_{i}.npz").exists() for i in range(spec.num_windows)):
        return False
    if not all(
        (path / "bundle/manifest.json").exists() and (path / "run.json").exists()
        for path in training_dirs
    ):
        return False
    window_identities = []
    training_checks = []
    earliest_origins = []
    for index, window in enumerate(task.iter_windows()):
        past, future = window.get_input_data()
        origins = expanded_window_origins(future, spec)
        earliest_origins.append(min(origins.values()))
        if index == 0 or policy == "per_window":
            identity = run_identity(
                settings,
                settings.seed,
                fingerprint,
                __version__,
                resolved_origin(origins),
                spec.task_definition(task.freq),
                policy=policy,
            )
            if index == 0 and identities is not None:
                identities[spec.name] = identity
            window_identities.append(identity)
            if policy == "first_window":
                if row.get("run_identity") != identity:
                    return False
                # Include orphan windows, as the normal resume path does.
                for path in directory.glob("window_*.npz"):
                    with np.load(path, allow_pickle=False) as saved:
                        _check_prediction_identity(saved, identity)
            training_dir = directory if policy == "first_window" else directory / f"window_{index}"
            training_checks.append((training_dir, identity))
        with np.load(directory / f"window_{index}.npz", allow_pickle=False) as saved:
            _check_prediction_identity(saved, identity)
            task.clean_and_validate_predictions(
                assemble_predictions(spec, past, saved["ids"].tolist(), saved["arrays"])
            )
            if prediction_diagnostics(saved) is None:
                return False
    # Both policies save the task-wide floor, including items first seen later.
    earliest = min(earliest_origins)
    if not all(
        _completed_training(path, training_identity, earliest, settings)
        for path, training_identity in training_checks
    ):
        return False
    if policy == "per_window":
        identity = hashlib.sha256(
            json.dumps(window_identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return row.get("run_identity") == identity


def _run_fev_tasks(
    specs: Iterable[FevTaskSpec],
    settings: Settings,
    out_dir: Path,
    policy: Literal["first_window", "per_window"] = "first_window",
    resume: bool = False,
    model_name: str | None = None,
    stop_after: int | None = None,
    run_identities: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Train once per task or once per window, and replace summary rows.

    For per_window, the summary identity is SHA256 of the ordered window
    identities encoded as a compact JSON list (UTF-8, separators=(',', ':')).

    Resume validates saved identities and computes only missing artifacts.
    Timings come from persisted trainer and prediction measurements, including
    earlier sessions. Fully verified completed tasks retain their summary row.
    ``NonFiniteTraining`` propagates before a failed task can write a summary
    row; its terminal checkpoint remains the marker for subsequent resumes.

    Early-ending items are forecast for calendar periods that other items'
    available past (and selection labels) already cover. No item's own future
    is used: fev truncates each item at its own cutoff and the loader reads only
    known covariates from future rows. This equals the information set fev gives
    every model and is not an out-of-time test for those items.
    """
    from demand_forecast import __version__
    from demand_forecast.bench.panels import TaskPanelCache
    from demand_forecast.core.run import _write_run, resolve_device, run_identity, train_panel
    from demand_forecast.inference.predictor import Predictor

    if policy not in {"first_window", "per_window"}:
        raise ValueError(f"unknown fev policy: {policy}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries.csv"
    summaries = (
        pd.read_csv(path, float_precision="round_trip")
        if path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    if "policy" not in summaries:
        summaries["policy"] = "first_window"
    model_name = model_name or f"demand-forecast-{settings.model.family}"
    for spec in specs:
        task = spec.to_task()
        fp = dataset_fingerprint(task)
        task_settings = settings_for_task(spec, settings, task.freq)
        definition = spec.task_definition(task.freq)
        period_source = "task_spec" if spec.seasonality is not None else "frequency"
        # Later windows can introduce items whose own cutoffs precede window 0.
        # Read only origins here; build each heavy panel once in the forecast loop.
        origins_per_window = [
            expanded_window_origins(window.get_input_data()[1], spec)
            for window in task.iter_windows()
        ]
        window_0_fit_origin = resolved_origin(origins_per_window[0])
        earliest = min(origin for origins in origins_per_window for origin in origins.values())
        origin_diagnostics = {
            "window_0_earliest_origin": min(origins_per_window[0].values()).isoformat(),
            "all_windows_earliest_origin": earliest.isoformat(),
            "window_0_fit_origin": window_0_fit_origin.isoformat(),
            "items_before_window_0_fit_origin": sum(
                origin < window_0_fit_origin
                for origins in origins_per_window
                for origin in origins.values()
            ),
        }
        task_dir = out_dir / spec.name
        rows = summaries.loc[(summaries.task == spec.name) & (summaries.model_name == model_name)]
        if resume and len(rows) == 1 and (stop_after is None or stop_after >= 1):
            if _completed_task(
                task, spec, task_settings, task_dir, rows.iloc[0], fp, policy, run_identities
            ):
                print(f"Skipping completed task {spec.name}", flush=True)
                continue
        started = time.perf_counter()
        panel_cache = TaskPanelCache(task.load_full_dataset(), spec, task.freq)
        logger.info(
            "panel history: task=%s series=%d build_s=%.1f",
            spec.name,
            len(panel_cache.panel.series),
            time.perf_counter() - started,
        )
        predictions = []
        diagnostics_per_window = []
        max_distinct_origins = 0
        skipped = False
        window_identities = []
        training_times = []
        reports = []
        all_reused = resume
        for index, window in enumerate(task.iter_windows()):
            past, future = window.get_input_data()
            started = time.perf_counter()
            panel = panel_cache.for_window(window, past)
            logger.info(
                "panel: task=%s window=%d series=%d build_s=%.1f "
                "seasonal_period=%d seasonal_period_source=%s",
                spec.name,
                index,
                len(panel.series),
                time.perf_counter() - started,
                panel.seasonal_period,
                period_source,
            )
            origins = origins_per_window[index]
            origin = resolved_origin(origins)
            n_origins = len(set(origins.values()))
            max_distinct_origins = max(max_distinct_origins, n_origins)
            logger.info(
                "origins: task=%s window=%d distinct=%d earliest=%s latest=%s",
                spec.name,
                index,
                n_origins,
                min(origins.values()),
                origin,
            )
            if index == 0 or policy == "per_window":
                training_dir = (
                    task_dir if policy == "first_window" else task_dir / f"window_{index}"
                )
                identity = run_identity(
                    task_settings,
                    task_settings.seed,
                    fp,
                    __version__,
                    origin,
                    definition,
                    policy=policy,
                )
                logger.info("identity: task=%s %s", spec.name, identity)
                if index == 0 and run_identities is not None:
                    # Reporting only: never changes artifact validation or resume.
                    run_identities[spec.name] = identity
                if resume:
                    # Check even later windows before doing any training or inference.
                    paths = (
                        task_dir.glob("window_*.npz")
                        if policy == "first_window"
                        else [task_dir / f"window_{index}.npz"]
                    )
                    for prediction_path in paths:
                        if not prediction_path.exists():
                            continue
                        with np.load(prediction_path, allow_pickle=False) as saved:
                            _check_prediction_identity(saved, identity)
                reused = resume and (training_dir / "bundle/manifest.json").exists()
                all_reused &= reused and (task_dir / f"window_{index}.npz").exists()
                train_panel(
                    panel,
                    origin,
                    task_settings,
                    training_dir,
                    dataset_fingerprint=fp,
                    task_definition=definition,
                    resume=resume,
                    stop_after=stop_after,
                    earliest_origin=earliest,
                    policy=policy,
                )
                report = json.loads((training_dir / "run.json").read_text(encoding="utf-8"))
                run_diagnostics = {
                    **origin_diagnostics,
                    "seasonal_period": panel.seasonal_period,
                    "seasonal_period_source": period_source,
                }
                if any(report.get(key) != value for key, value in run_diagnostics.items()):
                    report.update(run_diagnostics)
                    _write_run(training_dir / "run.json", report)
                cumulative = report["timings"].get("cumulative", {})
                training_time_s = (
                    sum(cumulative[key] for key in ("train_s", "eval_s", "checkpoint_s"))
                    if all(
                        cumulative.get(key) is not None
                        for key in ("train_s", "eval_s", "checkpoint_s")
                    )
                    else None
                )
                training_times.append(training_time_s)
                reports.append(report)
                window_identities.append(identity)
                rows = summaries.loc[
                    (summaries.task == spec.name) & (summaries.model_name == model_name)
                ]
                if (
                    policy == "first_window"
                    and reused
                    and len(rows) == 1
                    and rows.iloc[0].get("run_identity") == identity
                ):
                    complete = True
                    for i in range(spec.num_windows):
                        saved_path = task_dir / f"window_{i}.npz"
                        if not saved_path.exists():
                            complete = False
                            break
                        with np.load(saved_path, allow_pickle=False) as saved:
                            _check_prediction_identity(saved, identity)
                            saved_past, _ = task.get_window(i).get_input_data()
                            task.clean_and_validate_predictions(
                                assemble_predictions(
                                    spec, saved_past, saved["ids"].tolist(), saved["arrays"]
                                )
                            )
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
                    task.clean_and_validate_predictions(
                        assemble_predictions(spec, past, ids, arrays)
                    )
            else:
                predictor = Predictor.from_bundle(
                    training_dir / "bundle", device=resolve_device(settings)
                )
                ids, arrays, diagnostics = predictor.forecast_per_item_origins(panel, origins)
                _save_predictions(prediction_path, ids, arrays, identity, diagnostics)
                del predictor
            diagnostics_per_window.append(diagnostics)
            predictions.append(assemble_predictions(spec, past, ids, arrays))
        if skipped:
            continue
        if policy == "per_window":
            identity = hashlib.sha256(
                json.dumps(window_identities, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                all_reused
                and all(d is not None for d in diagnostics_per_window)
                and len(rows) == 1
                and rows.iloc[0].get("run_identity") == identity
            ):
                print(f"Skipping completed task {spec.name}", flush=True)
                continue
            training_time_s = (
                sum(training_times) if all(t is not None for t in training_times) else None
            )
            cumulative = {
                key: (
                    sum(r["timings"]["cumulative"][key] for r in reports)
                    if all(r["timings"].get("cumulative", {}).get(key) is not None for r in reports)
                    else None
                )
                for key in ("nonfinite_losses", "scaler_skips", "sessions")
            }

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
                    "policy": policy,
                    "max_distinct_origins": max_distinct_origins,
                    "fallback_no_context": diagnostic_sum("fallback_no_context"),
                    "fallback_undefined_scale": diagnostic_sum("fallback_undefined_scale"),
                    "cold_covariate_columns": diagnostic_sum("cold_covariate_columns"),
                    "clipped_covariate_values": diagnostic_sum("clipped_covariate_values"),
                    "floored_covariate_columns": diagnostic_sum("floored_covariate_columns"),
                    "floored_target_samples": diagnostic_sum("floored_target_samples"),
                    "inference_device": ",".join(sorted(devices))
                    if all(d is not None for d in diagnostics_per_window)
                    else "unknown",
                    "resolved_max_steps": report.get("resolved_budget", {}).get("max_steps"),
                    "n_fit_ids": report.get("n_fit_ids"),
                    "n_val_items": report.get("n_val_items"),
                    "nonfinite_losses": cumulative.get("nonfinite_losses"),
                    "scaler_skips": cumulative.get("scaler_skips"),
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
    if not path.exists() or "policy" not in pd.read_csv(path, nrows=0).columns:
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


def assemble_predictions(spec: FevTaskSpec, past, ids: list[str], arrays: np.ndarray):
    """Reassemble independent target series in fev's original positional item order."""
    from demand_forecast.data.loader_fev import target_series_id

    id_rows = past.select_columns([spec.id_column]) if hasattr(past, "select_columns") else past
    items = [str(row[spec.id_column]) for row in id_rows]
    if isinstance(spec.target, str):
        return predictions_for_fev(ids, arrays, items, spec.quantile_levels)
    expected = [target_series_id(item, target) for target in spec.targets for item in items]
    rows = predictions_for_fev(ids, arrays, expected, spec.quantile_levels)
    return {
        target: rows[index * len(items) : (index + 1) * len(items)]
        for index, target in enumerate(spec.targets)
    }


def check_panel(spec: FevTaskSpec) -> None:
    """Validate the first window with seasonal-naive predictions, without training."""
    from fev.metrics import get_metric

    from demand_forecast.data.loader_fev import item_ids, panel_from_window

    task = spec.to_task()
    window = task.get_window(0)
    past, future = window.get_input_data()
    panel = panel_from_window(past, future, spec, task.freq)
    arrays = np.empty((len(panel.series), spec.horizon, len(spec.quantile_levels)))
    period = max(task.seasonality, panel.seasonal_period)
    for index, series in enumerate(panel.series):
        values = pd.Series(series.target).ffill().fillna(0).to_numpy()
        tail = values[-period:]
        forecast = np.resize(tail, spec.horizon)
        arrays[index] = forecast[:, None]
    finite = sum(np.isfinite(series.target).sum() for series in panel.series)
    negative = sum(
        ((series.target < 0) & np.isfinite(series.target)).sum() for series in panel.series
    )
    negative_share = negative / finite if finite else float("nan")
    predictions = assemble_predictions(spec, past, item_ids(panel), arrays)
    cleaned = task.clean_and_validate_predictions(predictions)
    window.compute_metrics(
        cleaned,
        metrics=[get_metric(metric) for metric in [spec.eval_metric, *spec.extra_metrics]],
        seasonality=task.seasonality,
        quantile_levels=spec.quantile_levels,
    )
    print(
        f"PANEL_OK {spec.name} series={len(panel.series)} "
        f"max_len={max(s.history_length for s in panel.series)} "
        f"horizon={spec.horizon} freq={task.freq} targets={len(spec.targets)} "
        f"negative_share={negative_share:.8g}",
        flush=True,
    )


def _save_failures(path: Path, failures: dict) -> None:
    with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(failures, stream, indent=2)
            stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run_fev(
    specs: Iterable[FevTaskSpec],
    settings: Settings,
    out_dir: Path,
    policy: Literal["first_window", "per_window"] = "first_window",
    resume: bool = False,
    model_name: str | None = None,
    stop_after: int | None = None,
    *,
    keep_going: bool = False,
    panel_check: bool = False,
) -> pd.DataFrame:
    """Record task failures without changing normal artifact reuse or resume.

    Journals are reporting only. Restarting from scratch requires the operator
    to remove the task directory; resume retains even terminal checkpoints.
    """
    import gc
    import traceback
    from datetime import datetime, timezone

    import torch

    from demand_forecast.core.inference_batching import task_batch_limits
    from demand_forecast.core.training import InterruptedRun

    if policy not in {"first_window", "per_window"}:
        raise ValueError(f"unknown fev policy: {policy}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A preflight must not clear training failures, or prevent their retries.
    failure_path = out_dir / ("panel_failures.json" if panel_check else "failures.json")
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    failed = []
    succeeded = 0
    for spec in specs:
        run_identities: dict[str, str] = {}

        try:
            if panel_check:
                check_panel(spec)
            else:
                with task_batch_limits():
                    _run_fev_tasks(
                        [spec],
                        settings,
                        out_dir,
                        policy=policy,
                        resume=resume,
                        model_name=model_name,
                        stop_after=stop_after,
                        run_identities=run_identities,
                    )
        except InterruptedRun:
            raise
        except Exception as exc:
            if panel_check:
                message = " ".join(str(exc).splitlines())
                print(f"PANEL_FAIL {spec.name} {type(exc).__name__}: {message}", flush=True)
            if not keep_going:
                if panel_check:
                    print(f"PANEL_SUMMARY ok={succeeded} failed=1", flush=True)
                raise
            logger.exception("Task %s failed", spec.name)
            failures[spec.name] = {
                "task": spec.name,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-20:],
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "run_identity": run_identities.get(spec.name),
            }
            _save_failures(failure_path, failures)
            failed.append(spec.name)
        else:
            succeeded += 1
            if spec.name in failures:
                del failures[spec.name]
                _save_failures(failure_path, failures)
        # Execute after the exception scope: its traceback can retain the model.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if panel_check:
        print(f"PANEL_SUMMARY ok={succeeded} failed={len(failed)}", flush=True)
    path = out_dir / "summaries.csv"
    result = (
        pd.read_csv(path, float_precision="round_trip")
        if path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    result.attrs["failed_tasks"] = failed
    return result
