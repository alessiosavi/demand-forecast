"""Selection, refit and store export orchestration for the forecast command."""

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import davies_bouldin_score  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from demand_forecast.core.pipeline import ForecastPipeline  # noqa: E402
from demand_forecast.core.tuning import HyperparameterTuner, SearchSpace, TuningConfig  # noqa: E402
from demand_forecast.data.feature_engineering import extract_metafeatures  # noqa: E402
from demand_forecast.utils.device import describe_device  # noqa: E402
from demand_forecast.utils.metrics import wape  # noqa: E402
from demand_forecast.utils.visualization import (  # noqa: E402
    plot_clustering,
    plot_forecast_horizon,
    plot_prediction_quality,
    plot_training_history,
    save_prediction_report,
)

logger = logging.getLogger(__name__)
EXPORT_COLUMNS = ["product_id", "origin", "week_start", "qty"]


def validate_forecast_contract(frame, product_ids, origin, horizon=16):
    """Validate the exact Cartesian product, dates and units before any CSV write."""
    if list(frame.columns) != EXPORT_COLUMNS:
        raise ValueError(f"Forecast columns must be exactly {EXPORT_COLUMNS}")
    origin = pd.Timestamp(origin)
    if origin.weekday() != 0 or origin != origin.normalize():
        raise ValueError("Expected origin must be a Monday")
    weeks = pd.date_range(origin, periods=horizon, freq="W-MON")
    origins = pd.to_datetime(frame.origin, errors="raise")
    dates = pd.to_datetime(frame.week_start, errors="raise")
    if not origins.eq(origin).all():
        raise ValueError("Wrong forecast origin")
    keys = list(zip(frame.product_id, dates))
    expected = {(sku, week) for sku in product_ids for week in weeks}
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate forecast keys")
    if set(keys) != expected:
        raise ValueError("Forecast keys differ from catalogue IDs x expected weeks")
    values = frame.qty.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecast quantities must be finite and non-negative")


def package_commit():
    """Installed Git HEAD with a dirty suffix, or the environment identity outside Git."""
    return package_source_identity()[0]


def package_source_identity():
    """Return source identity and whether local Git changes were detected."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        )
    except FileNotFoundError:
        result = None
    if result is None or result.returncode != 0:
        return os.environ.get("DEMAND_FORECAST_COMMIT") or "unknown", False
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    if status.returncode != 0:
        raise RuntimeError("Cannot determine package source dirtiness with git status")
    dirty = bool(status.stdout.strip())
    return result.stdout.strip() + ("-dirty" if dirty else ""), dirty


def score_forecast(pipeline, frame, origin):
    """Align untouched weekly actuals and score model/last-eight-week WAPE in percent."""
    cfg = pipeline.settings.data
    weekly = pipeline._weekly.reset_index().rename(
        columns={
            cfg.sku_column: "product_id",
            cfg.date_column: "week_start",
            cfg.quantity_column: "actual",
        }
    )
    scored = frame.assign(week_start=pd.to_datetime(frame.week_start)).merge(
        weekly[["product_id", "week_start", "actual"]],
        on=["product_id", "week_start"],
        validate="one_to_one",
    )
    if len(scored) != len(frame):
        raise ValueError("Missing weekly actuals for forecast score")
    pre = pipeline._weekly.loc[
        (pipeline._weekly.index < origin)
        & (pipeline._weekly.index >= origin - pd.Timedelta(weeks=8))
    ]
    naive = pre.groupby(cfg.sku_column)[cfg.quantity_column].mean()
    score = wape(scored.actual, scored.qty)
    baseline = wape(scored.actual, scored.product_id.map(naive))
    return (
        scored,
        None if score is None else 100.0 * score,
        None if baseline is None else 100.0 * baseline,
    )


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_clustering(pipeline, holdout, out_dir):
    cfg = pipeline.settings.data
    history = pipeline._weekly.loc[pipeline._weekly.index < holdout].copy()
    ids = sorted(history[cfg.sku_column].unique())
    history["sku_code"] = history[cfg.sku_column].map({sku: i for i, sku in enumerate(ids)})
    features = extract_metafeatures(
        history,
        store_column=pipeline.store_column,
        quantity_column=cfg.quantity_column,
        date_column=cfg.date_column,
        cache_path=out_dir / "artifacts/metafeatures.csv",
    )
    matrix = MinMaxScaler().fit_transform(features.iloc[:, 2:].fillna(0))
    inertia, dbi, ks = [], [], []
    for k in range(2, min(10, len(matrix) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=pipeline.settings.seed).fit(matrix)
        ks.append(k)
        inertia.append(model.inertia_)
        dbi.append(
            davies_bouldin_score(matrix, model.labels_)
            if 1 < len(set(model.labels_)) < len(matrix)
            else np.nan
        )
    elbow = ks[int(np.argmax(np.diff(inertia, n=2))) + 1] if len(ks) >= 3 else (ks[0] if ks else 1)
    plot_clustering(
        ks, inertia, dbi, elbow, save_path=out_dir / "plots/clustering_elbow.png", show=False
    )


def pooled_loader(loaders, batch_size, shuffle, seed):
    datasets = [dl.dataset for dl in loaders.values()]
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=datasets[0].collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def forecast_examples(pipeline, holdout, future, out_dir):
    cfg = pipeline.settings.data
    weekly = pipeline._weekly
    origin = pd.Timestamp(holdout.origin.iloc[0])
    volumes = (
        weekly.loc[weekly.index < origin]
        .groupby(cfg.sku_column)[cfg.quantity_column]
        .sum()
        .sort_values(kind="stable")
    )
    ids = volumes.index[np.linspace(0, len(volumes) - 1, min(6, len(volumes)), dtype=int)]
    fig, axes = plt.subplots(len(ids), 1, figsize=(13, 3 * len(ids)), squeeze=False)
    for ax, sku in zip(axes[:, 0], ids):
        history = weekly[weekly[cfg.sku_column] == sku].sort_index()
        h, f = holdout[holdout.product_id == sku], future[future.product_id == sku]
        ax.plot(history.index, history[cfg.quantity_column], label="Weekly actual")
        ax.plot(pd.to_datetime(h.week_start), h.qty, label="Holdout prediction")
        ax.plot(pd.to_datetime(f.week_start), f.qty, label="Future forecast")
        ax.axvline(origin, color="grey", linestyle="--")
        ax.set_title(sku, fontsize=9)
        ax.set_ylabel("Units/week")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plots/forecast_examples.png", dpi=150)
    plt.close(fig)


def tuning_plots(tuner, plot_dir, notes):
    completed = tuner.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if completed:
        ax = optuna.visualization.matplotlib.plot_optimization_history(tuner.study)
        ax.figure.savefig(plot_dir / "optuna_history.png", bbox_inches="tight", dpi=150)
        plt.close(ax.figure)
    else:
        notes.append("optuna_history.png skipped: no completed trials")
    if len(completed) >= 2:
        try:
            ax = optuna.visualization.matplotlib.plot_param_importances(tuner.study)
            ax.figure.savefig(plot_dir / "optuna_importance.png", bbox_inches="tight", dpi=150)
            plt.close(ax.figure)
        except (ValueError, RuntimeError) as exc:
            notes.append(f"optuna_importance.png skipped: {exc}")
            plt.close("all")
    else:
        notes.append("optuna_importance.png skipped: fewer than 2 completed trials")


def run_forecast(
    settings,
    out_dir: Path,
    *,
    max_products=None,
    budget_seconds=None,
    expect_origins=None,
    preprocess_only=False,
    started=None,
):
    started = time.monotonic() if started is None else started
    started_at = datetime.now(timezone.utc).isoformat()
    deadline = started + (settings.budget_seconds if budget_seconds is None else budget_seconds)
    if settings.model.model_type != "standard":
        raise ValueError(
            "forecast supports model.model_type='standard' only; V2/lightweight are not wired to static codes"
        )
    if settings.timeseries.n_out != 16:
        raise ValueError("Store forecasts require n_out=16 (the contract horizon)")
    if not settings.data.products_path:
        raise ValueError("forecast requires a product catalogue")
    if settings.data.filter_sparse_skus:
        raise ValueError("Catalogue-complete forecast requires filter_sparse_skus=false")
    if settings.data.resample_period != "W-MON":
        raise ValueError("Catalogue-complete forecast requires resample_period=W-MON")
    if settings.training.flatten_loss:
        raise ValueError("forecast requires flatten_loss=false (weekly loss)")
    out_dir = Path(out_dir)
    for relative in ("plots", "artifacts"):
        directory = out_dir / relative
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(
                f"Use an empty output directory; {directory} already contains artifacts"
            )
        directory.mkdir(parents=True, exist_ok=True)
    pipeline = ForecastPipeline(settings)
    if pipeline.device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    prep_started = time.monotonic()
    pipeline.load_data(max_products=max_products)
    val_from, holdout_from, origin_future = pipeline.cutoffs()
    if expect_origins is not None:
        expected = tuple(pd.Timestamp(origin) for origin in expect_origins)
        if (holdout_from, origin_future) != expected:
            raise ValueError(
                f"Forecast origins differ: derived {holdout_from.date()},{origin_future.date()}; "
                f"expected {','.join(str(origin.date()) for origin in expected)}"
            )
    diagnostic_clustering(pipeline, holdout_from, out_dir)
    pipeline.fit_preprocessing(val_from)
    pipeline.create_datasets(validation_origin=val_from)
    preprocessing_s = time.monotonic() - prep_started
    logger.info(
        "Selection preprocessing (including load and diagnostic) seconds=%.3f", preprocessing_s
    )
    if preprocess_only:
        train_count = sum(len(train.y) for train, _ in pipeline._raw_datasets.values())
        pipeline.fit_preprocessing(holdout_from)
        pipeline.create_datasets(refit=True)
        result = {
            "n_products": len(pipeline.sku_to_index),
            "inner_train_windows": train_count,
            "full_train_windows": sum(len(train.y) for train, _ in pipeline._raw_datasets.values()),
            "preprocessing_s": time.monotonic() - prep_started,
            "elapsed_s": time.monotonic() - started,
        }
        (out_dir / "preprocessing.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    train_loaders, val_loaders, _, _ = pipeline.dataloaders()
    cfg = settings.tuning
    tuner = HyperparameterTuner(
        "standard",
        pooled_loader(train_loaders, settings.training.batch_size, True, settings.seed),
        pooled_loader(val_loaders, settings.training.batch_size, False, settings.seed),
        pipeline.model_kwargs(),
        config=TuningConfig(
            **{k: v for k, v in cfg.model_dump().items() if k in TuningConfig.__dataclass_fields__},
            seed=settings.seed,
        ),
        search_space=SearchSpace.hackathon(),
        device=pipeline.device,
        num_epochs_per_trial=cfg.epochs_per_trial,
        early_stop_patience=cfg.early_stop_patience,
        deadline=deadline,
        base_training_params={
            k: getattr(settings.training, k)
            for k in ("learning_rate", "weight_decay", "batch_size")
        },
    )
    resolved = tuner.tune(n_trials=cfg.n_trials if cfg.enabled else 0, show_progress_bar=False)
    for key, value in resolved.items():
        if key in type(settings.model).model_fields:
            setattr(settings.model, key, value)
        if key in ("learning_rate", "weight_decay", "batch_size"):
            setattr(settings.training, key, value)
    pipeline._set_seeds(settings.seed)
    pipeline.build_model()
    selection = pipeline.train(plot=True, plot_dir=out_dir / "plots", deadline=deadline)
    plot_training_history(
        selection.history["train_loss"],
        selection.history["val_loss"],
        save_path=out_dir / "plots/training_history.png",
        show=False,
    )
    best_epoch = selection.best_epoch
    ids = pipeline.products[settings.data.product_key_column].tolist()
    validation = pipeline.forecast(val_from)
    validate_forecast_contract(validation, ids, val_from)
    validation.to_csv(
        out_dir / "artifacts/validation_predictions.csv", index=False, float_format="%.2f"
    )
    _, validation_score, validation_baseline = score_forecast(
        pipeline, pd.read_csv(out_dir / "artifacts/validation_predictions.csv"), val_from
    )
    logger.info(
        "Validation WAPE (percent): wape_validation=%s wape_validation_naive_last8=%s",
        validation_score,
        validation_baseline,
    )
    # Fresh scalers, encoders, embeddings and weights. Never fit on holdout quantities.
    pipeline.fit_preprocessing(holdout_from)
    pipeline.create_datasets(refit=True)
    pipeline._set_seeds(settings.seed)
    pipeline.build_model()
    settings.training.num_epochs = best_epoch
    refit = pipeline.train(deadline=deadline, scheduler_epochs=selection.num_epochs)
    holdout = pipeline.forecast(holdout_from)
    future = pipeline.forecast(origin_future)
    for frame, origin, filename in (
        (holdout, holdout_from, "holdout_predictions.csv"),
        (future, origin_future, "forecasts.csv"),
    ):
        validate_forecast_contract(frame, ids, origin)
        frame.to_csv(out_dir / filename, index=False, float_format="%.2f")
    # Score the rounded file itself once, aligned by product and week.
    holdout = pd.read_csv(out_dir / "holdout_predictions.csv")
    cfg_data = settings.data
    scored, score, baseline = score_forecast(pipeline, holdout, holdout_from)
    actual = scored.actual.to_numpy().reshape(len(ids), 16)
    prediction = scored.qty.to_numpy().reshape(len(ids), 16)
    plot_prediction_quality(
        actual.ravel(),
        prediction.ravel(),
        title="Holdout weekly demand (units)",
        save_path=out_dir / "plots/holdout_prediction_quality.png",
        show=False,
    )
    plot_forecast_horizon(
        actual, prediction, 16, save_path=out_dir / "plots/holdout_forecast_horizon.png", show=False
    )
    save_prediction_report(
        holdout.rename(columns={"product_id": "sku", "qty": "prediction"}),
        out_dir / "plots",
        prefix="holdout",
    )
    forecast_examples(pipeline, holdout, future, out_dir)
    notes = [tuner.fallback_reason] if tuner.fallback_reason else []
    tuning_plots(tuner, out_dir / "plots", notes)
    tuner.get_study_dataframe().to_csv(out_dir / "artifacts/optuna_trials.csv", index=False)
    pipeline.save(out_dir / "artifacts/model.pt")
    commit, source_dirty = package_source_identity()
    result = {
        "model": "AdvancedDemandForecastModel " + json.dumps(resolved, sort_keys=True),
        "wape_validation": validation_score,
        "wape_validation_naive_last8": validation_baseline,
        "wape_holdout": score,
        "wape_naive_last8": baseline,
        "device": describe_device(pipeline.device),
        "optuna_trials": len(tuner.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))),
        "best_params": resolved,
        "best_epoch": best_epoch,
        "refit_epochs_completed": len(refit.history["train_loss"]),
        "plots": sorted(
            str(path.relative_to(out_dir))
            for path in (out_dir / "plots").iterdir()
            if path.is_file() and path.suffix == ".png"
        ),
        "holdout_from": holdout_from.date().isoformat(),
        "origin_future": origin_future.date().isoformat(),
        "n_products": len(ids),
        "k": cfg_data.k,
        "scaling_group": cfg_data.scaling_group,
        "package_commit": commit,
        "source_dirty": source_dirty,
        "inputs_sha256": {
            "products": sha256(cfg_data.products_path),
            "sales": sha256(cfg_data.input_path),
        },
        "started_at": started_at,
        "notes": notes,
        "preprocessing_s": preprocessing_s,
        "stage_preprocessing_s": pipeline.preprocessing_seconds,
        "selection_seconds_per_epoch": selection.epoch_seconds,
        "refit_seconds_per_epoch": refit.epoch_seconds,
        "tuning_fallback": tuner.fallback_reason,
    }
    result["budget_exceeded"] = time.monotonic() >= deadline
    if result["budget_exceeded"]:
        logger.warning(
            "Command budget exceeded; exporting completed work (refit epochs=%d)",
            len(refit.history["train_loss"]),
        )
    result["elapsed_s"] = time.monotonic() - started
    (out_dir / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    logger.info(
        "Forecast complete: products=%d WAPE=%s naive=%s elapsed_s=%.3f",
        len(ids),
        score,
        baseline,
        result["elapsed_s"],
    )
    return result
