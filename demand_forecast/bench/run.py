"""Resumable model and reference forecasts with official fev evaluation."""

import time
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import pandas as pd

from demand_forecast.bench.specs import FevTaskSpec, dataset_fingerprint
from demand_forecast.config import DatasetConfig, ForecastConfig, Settings


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


def window_origin(future_data, timestamp_column: str = "timestamp") -> pd.Timestamp:
    """Require a shared first future timestamp across all items in a window."""
    origins = {pd.Timestamp(row[timestamp_column][0]) for row in future_data}
    if len(origins) != 1:
        raise ValueError("windows with item-specific origins are not supported")
    return origins.pop()


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


def _save_predictions(path: Path, ids: list[str], arrays: np.ndarray, identity: str) -> None:
    """Publish a complete window so an interrupted write leaves it missing."""
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            np.savez(stream, ids=np.asarray(ids, dtype=str), arrays=arrays, identity=identity)
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
    Timings cover this invocation: the call to train_panel (including bundle
    reuse on resume) and forecasts actually computed, excluding bundle loading.
    """
    from demand_forecast import __version__
    from demand_forecast.core.run import run_identity, train_panel
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
        training_time_s = inference_time_s = 0.0
        for index, window in enumerate(task.iter_windows()):
            past, future = window.get_input_data()
            panel = panel_from_window(past, future, spec, task.freq)
            origin = window_origin(future, spec.timestamp_column)
            if index == 0:
                identity = run_identity(
                    task_settings, task_settings.seed, fp, __version__, origin, definition
                )
                if resume:
                    # Check even later windows before doing any training or inference.
                    for prediction_path in task_dir.glob("window_*.npz"):
                        with np.load(prediction_path, allow_pickle=False) as saved:
                            _check_prediction_identity(saved, identity)
                started = time.perf_counter()
                train_panel(
                    panel,
                    origin,
                    task_settings,
                    task_dir,
                    dataset_fingerprint=fp,
                    task_definition=definition,
                    resume=resume,
                    stop_after=stop_after,
                )
                training_time_s = time.perf_counter() - started
            prediction_path = task_dir / f"window_{index}.npz"
            if resume and prediction_path.exists():
                with np.load(prediction_path, allow_pickle=False) as saved:
                    _check_prediction_identity(saved, identity)
                    ids, arrays = saved["ids"].tolist(), saved["arrays"]
            else:
                predictor = Predictor.from_bundle(task_dir / "bundle")
                started = time.perf_counter()
                ids, arrays = predictor.forecast_arrays(panel, origin)
                inference_time_s += time.perf_counter() - started
                _save_predictions(prediction_path, ids, arrays, identity)
            predictions.append(
                predictions_for_fev(ids, arrays, item_ids(panel), task.quantile_levels)
            )
        summary = {
            "task": spec.name,
            **task.evaluation_summary(
                predictions,
                model_name=model_name,
                trained_on_this_dataset=False,
                training_time_s=training_time_s,
                inference_time_s=inference_time_s,
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
