"""Sequential, equal-cap model comparison with CSV-derived scores and provenance checks."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from demand_forecast.core.forecast_run import (  # noqa: E402
    forecast_provenance,
    run_forecast,
    validate_forecast_contract,
)
from demand_forecast.core.pipeline import ForecastPipeline  # noqa: E402
from demand_forecast.core.tuning import SearchSpace  # noqa: E402
from demand_forecast.utils.metrics import wape  # noqa: E402

logger = logging.getLogger(__name__)
MODEL_TYPES = ("standard", "advanced", "lightweight_tcn", "lightweight_mixer")
PLOT_NAMES = (
    "wape_by_model",
    "wape_by_horizon",
    "example_products",
    "accuracy_vs_cost",
    "learning_curves",
    "wape_by_category",
    "total_demand",
    "optuna_best_so_far",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model_settings(settings, name):
    """Apply a family to a pristine deep copy, never inheriting another run's tuning."""
    if name not in MODEL_TYPES:
        raise ValueError(f"Unknown benchmark model {name!r}; choose from {','.join(MODEL_TYPES)}")
    result = settings.model_copy(deep=True)
    if name.startswith("lightweight_"):
        result.model.model_type = "lightweight"
        result.model.lightweight_variant = name.removeprefix("lightweight_")
    else:
        result.model.model_type = name
    return result


def comparison_history(pipeline):
    cfg = pipeline.settings.data
    weekly = pipeline._weekly.reset_index().rename(
        columns={
            cfg.sku_column: "product_id",
            cfg.date_column: "week_start",
            cfg.quantity_column: "actual",
        }
    )[["product_id", "week_start", "actual"]]
    products = pipeline.products.set_index(cfg.product_key_column)
    categories = (
        products["category"].fillna("(missing)").astype(str)
        if "category" in products
        else pd.Series("(all)", index=products.index)
    )
    weekly["category"] = weekly.product_id.map(categories)
    return weekly


def example_products(weekly, holdout):
    """One pre-holdout volume leader per largest category, with deterministic ties."""
    volumes = (
        weekly[weekly.week_start < pd.Timestamp(holdout)]
        .groupby(["category", "product_id"], as_index=False)
        .actual.sum()
    )
    categories = (
        volumes.groupby("category", as_index=False)
        .actual.sum()
        .sort_values(["actual", "category"], ascending=[False, True])
        .head(6)
        .category
    )
    return [
        volumes[volumes.category == category]
        .sort_values(["actual", "product_id"], ascending=[False, True])
        .iloc[0]
        .product_id
        for category in categories
    ]


def baseline_forecast(weekly, ids, origin, name, horizon=16):
    origin = pd.Timestamp(origin)
    frame = pd.MultiIndex.from_product(
        [ids, pd.date_range(origin, periods=horizon, freq="W-MON")],
        names=["product_id", "week_start"],
    ).to_frame(index=False)
    if name == "naive_last8":
        pre = weekly[
            (weekly.week_start < origin) & (weekly.week_start >= origin - pd.Timedelta(weeks=8))
        ]
        if not pre.groupby("product_id").size().reindex(ids, fill_value=0).eq(8).all():
            raise ValueError("Last-eight baseline needs eight exact weeks per product")
        frame["qty"] = frame.product_id.map(pre.groupby("product_id").actual.mean())
    elif name == "seasonal_naive":
        previous = weekly[["product_id", "week_start", "actual"]].copy()
        previous["week_start"] += pd.Timedelta(weeks=52)
        frame = frame.merge(
            previous, on=["product_id", "week_start"], how="left", validate="one_to_one"
        )
        frame = frame.rename(columns={"actual": "qty"})
    else:
        raise ValueError(f"Unknown baseline: {name}")
    frame["origin"] = origin.date().isoformat()
    frame = frame[["product_id", "origin", "week_start", "qty"]]
    validate_forecast_contract(frame, ids, origin, horizon)
    return frame


def aligned_scores(frame, weekly):
    frame = frame.assign(week_start=pd.to_datetime(frame.week_start))
    scored = frame.merge(weekly, on=["product_id", "week_start"], how="left", validate="one_to_one")
    if scored.actual.isna().any() or scored.category.isna().any():
        raise ValueError("Missing actuals or categories for prediction keys")
    scored["horizon"] = (scored.week_start - pd.to_datetime(scored.origin)).dt.days // 7 + 1
    return scored


def percent_wape(frame):
    value = wape(frame.actual, frame.qty)
    return None if value is None else 100.0 * value


def performance(validation, holdout):
    total = float(holdout.actual.sum())
    return {
        "wape_validation": percent_wape(validation),
        "wape_holdout": percent_wape(holdout),
        "holdout_bias_percent": 100 * float((holdout.qty - holdout.actual).sum()) / total
        if total
        else None,
        "wape_by_horizon": {
            str(key): percent_wape(group) for key, group in holdout.groupby("horizon")
        },
        "wape_by_category": {
            str(key): percent_wape(group) for key, group in holdout.groupby("category")
        },
    }


def _read_prediction(path, ids, origin):
    frame = pd.read_csv(path, dtype={"product_id": str})
    validate_forecast_contract(frame, ids, origin)
    return frame.sort_values(["product_id", "week_start"]).reset_index(drop=True)


def compare(settings, out_dir, models=MODEL_TYPES, *, max_products=None, expect_origins=None):
    """Rebuild reports without training; reject incompatible or stale input artifacts."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "benchmark_request.json").read_text())
    pipeline = ForecastPipeline(settings.model_copy(deep=True))
    pipeline.load_data(max_products=max_products)
    provenance = forecast_provenance(pipeline, settings.budget_seconds, max_products)
    if manifest["provenance"] != provenance:
        raise ValueError("Benchmark inputs or protocol differ from the frozen request")
    _check_origins(provenance, expect_origins)
    weekly = comparison_history(pipeline)
    examples = example_products(weekly, provenance["holdout_from"])
    if manifest["example_products"] != examples:
        raise ValueError("Example products differ from the frozen pre-holdout selection")
    ids = provenance["product_ids"]
    results, predictions = {}, {}
    for name in models:
        expected_settings = model_settings(settings, name).model_dump(mode="json")
        directory = out_dir / "models" / name
        metadata = json.loads((directory / "run.json").read_text())
        for key, value in provenance.items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(f"{name}: incompatible provenance field {key}")
        if metadata.get("model_type") != name:
            raise ValueError(f"{name}: model_type does not match directory")
        # Paths may be relocated for comparison; hashes still have to match.
        actual_settings = metadata.get("requested_settings", {})
        expected_settings = _without_paths(expected_settings)
        if _without_paths(actual_settings) != expected_settings:
            raise ValueError(f"{name}: requested settings differ from frozen configuration")
        fields = (
            "model_type",
            "status",
            "error",
            "n_parameters",
            "train_seconds",
            "inference_seconds",
            "forecast_pipeline_seconds",
            "forecast_pipeline_timing",
            "network_latency_ms",
            "network_latency_protocol",
            "best_epoch",
            "refit_epochs_completed",
            "trial_counts",
            "stop_reasons",
            "trial_stop_reasons",
            "device",
            "history",
            "trial_values",
            "notes",
            "requested_settings",
            "resolved_settings",
            "search_space",
            "package_commit",
            "source_dirty",
        )
        result = {key: metadata[key] for key in fields if key in metadata}
        result["kind"] = "neural"
        result["rank"] = None
        if metadata.get("status") != "error":
            counts = metadata.get("trial_counts", {})
            complete = (
                counts.get("completed", 0) >= 1
                and metadata.get("best_epoch", 0) >= 1
                and metadata.get("refit_epochs_completed") == metadata.get("best_epoch")
            )
            result["status"] = "ok" if complete else "incomplete"
            frames = [
                aligned_scores(
                    _read_prediction(directory / filename, ids, provenance[origin]), weekly
                )
                for filename, origin in (
                    ("validation_predictions.csv", "validation_from"),
                    ("holdout_predictions.csv", "holdout_from"),
                )
            ]
            _read_prediction(directory / "forecasts.csv", ids, provenance["origin_future"])
            result.update(performance(*frames))
            predictions[name] = frames[1]
        results[name] = result

    for name in ("naive_last8", "seasonal_naive"):
        frames = [
            aligned_scores(baseline_forecast(weekly, ids, provenance[origin], name), weekly)
            for origin in ("validation_from", "holdout_from")
        ]
        results[name] = {
            "status": "ok",
            "kind": "baseline",
            "n_parameters": 0,
            "train_seconds": 0,
            "device": {"device": "cpu"},
            **performance(*frames),
        }
        predictions[name] = frames[1]
    ranking = sorted(
        [
            name
            for name, result in results.items()
            if result["status"] == "ok" and result.get("wape_validation") is not None
        ],
        key=lambda name: (results[name]["wape_validation"], name),
    )
    for rank, name in enumerate(ranking, 1):
        results[name]["rank"] = rank
    report = {
        **provenance,
        "requested_settings": manifest["settings"],
        "search_spaces": manifest["search_spaces"],
        "example_products": examples,
        "ranking_metric": "wape_validation",
        "ranking": ranking,
        "best_neural_model": next(
            (name for name in ranking if results[name]["kind"] == "neural"), None
        ),
        "results": results,
        "notes": [
            "Equal caps, not equal completed work. Incomplete/error runs are excluded from ranking.",
            "All scores are recomputed in original units from aligned CSVs; zero-demand groups are null.",
            "Seasonal naive uses the exact W-MON week 52 weeks earlier.",
            "V2 auxiliary quantile/decomposition output heads are untrained under point MSE.",
            "Mixer does not use future_time inputs.",
            "Configuration frozen before training; previously observed standard holdout scores are not used for selection.",
        ],
        "plots": [f"plots/{name}.png" for name in PLOT_NAMES],
    }
    write_json(out_dir / "benchmark.json", report)
    write_table(out_dir / "benchmark.md", report)
    comparison_plots(out_dir / "plots", report, predictions, weekly)
    return report


def _without_paths(settings):
    result = {**settings, "data": dict(settings.get("data", {}))}
    for key in ("input_path", "products_path"):
        result["data"].pop(key, None)
    return result


def _check_origins(provenance, expected):
    if expected is not None and tuple(pd.Timestamp(x) for x in expected) != tuple(
        pd.Timestamp(provenance[key]) for key in ("holdout_from", "origin_future")
    ):
        raise ValueError("Forecast origins differ from --expect-origins")


def run_benchmark(
    settings,
    out_dir,
    models=MODEL_TYPES,
    *,
    max_products=None,
    budget_seconds=None,
    expect_origins=None,
    compare_only=False,
):
    """Run each family with its own full cap; isolate failures and compare saved results."""
    settings = settings.model_copy(deep=True)
    if budget_seconds is not None:
        settings.budget_seconds = budget_seconds
    models = tuple(models)
    if not models or len(set(models)) != len(models):
        raise ValueError("Choose a non-empty list of distinct model families")
    for name in models:
        model_settings(settings, name)
    if settings.tuning.storage is not None or settings.tuning.n_jobs != 1:
        raise ValueError("Benchmark requires fresh in-memory studies and tuning.n_jobs=1")
    if settings.tuning.metric != "mse" or settings.tuning.direction != "minimize":
        raise ValueError("Benchmark requires tuning.metric=mse and direction=minimize")
    out_dir = Path(out_dir)
    if not compare_only:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError("Use an empty benchmark output directory (or --compare-only)")
        pipeline = ForecastPipeline(settings.model_copy(deep=True))
        pipeline.load_data(max_products=max_products)
        provenance = forecast_provenance(pipeline, settings.budget_seconds, max_products)
        _check_origins(provenance, expect_origins)
        weekly = comparison_history(pipeline)
        # Validate baseline coverage before spending any training budget.
        for origin in ("validation_from", "holdout_from"):
            baseline_forecast(
                weekly, provenance["product_ids"], provenance[origin], "seasonal_naive"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / "benchmark_request.json",
            {
                "settings": settings.model_dump(mode="json"),
                "models": models,
                "provenance": provenance,
                "example_products": example_products(weekly, provenance["holdout_from"]),
                "search_spaces": {name: asdict(SearchSpace.hackathon(name)) for name in models},
            },
        )
        for name in models:
            current = model_settings(settings, name)
            requested = current.model_dump(mode="json")
            directory = out_dir / "models" / name
            try:
                run_forecast(
                    current,
                    directory,
                    max_products=max_products,
                    budget_seconds=settings.budget_seconds,
                    expect_origins=expect_origins,
                    measure_latency=True,
                )
            except Exception as exc:
                logger.exception("Benchmark model %s failed", name)
                directory.mkdir(parents=True, exist_ok=True)
                write_json(
                    directory / "run.json",
                    {
                        **provenance,
                        "model_type": name,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "requested_settings": requested,
                    },
                )
    return compare(
        settings, out_dir, models, max_products=max_products, expect_origins=expect_origins
    )


def write_table(path, report):
    def number(value):
        return "—" if value is None else f"{value:.2f}"

    lines = [
        "# Model benchmark",
        "",
        "Ranked by validation WAPE only; all WAPE and bias values are percentages.",
        "Equal caps; actual completed work may differ. Incomplete/error runs are unranked.",
        "",
        "| Model | Status | Rank | Validation WAPE | Holdout WAPE | Holdout bias | Parameters | Train s | Forecast pipeline s | Network ms / 512 | Best epoch | Trials C/P/F | Device |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    ordered = report["ranking"] + [
        name for name in report["results"] if name not in report["ranking"]
    ]
    for name in ordered:
        r = report["results"][name]
        counts = r.get("trial_counts", {})
        values = [
            name,
            r["status"],
            str(r.get("rank") or "—"),
            *[
                number(r.get(key))
                for key in ("wape_validation", "wape_holdout", "holdout_bias_percent")
            ],
            str(r.get("n_parameters", "—")),
            *[
                number(r.get(key))
                for key in ("train_seconds", "forecast_pipeline_seconds", "network_latency_ms")
            ],
            str(r.get("best_epoch", "—")),
            "/".join(str(counts.get(key, "—")) for key in ("completed", "pruned", "failed")),
            r.get("device", {}).get("device", "—"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "Forecast pipeline time includes pandas preparation, batching, transfer and inverse scaling.",
        "Network latency: 5 warmups, mean of 20 forward passes, synchronized on CUDA; fixed 512 inputs (cycled for small catalogues).",
        "",
    ]
    for name, result in report["results"].items():
        if result.get("error"):
            lines.append(f"- {name}: {result['error']}")
        if result.get("stop_reasons"):
            lines.append(
                f"- {name} stop reasons: {json.dumps(result['stop_reasons'], sort_keys=True)}"
            )
    lines += ["", *report["notes"], "", *[f"![{Path(p).stem}]({p})" for p in report["plots"]]]
    Path(path).write_text("\n".join(lines) + "\n")


def comparison_plots(directory, report, predictions, weekly):
    directory.mkdir(parents=True, exist_ok=True)
    results = report["results"]
    names = list(predictions)
    labels = {
        name: name + (" (incomplete)" if results[name]["status"] == "incomplete" else "")
        for name in names
    }
    holdout = pd.Timestamp(report["holdout_from"])
    history = weekly[weekly.week_start >= holdout - pd.Timedelta(weeks=52)]

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=150)
        plt.close(fig)

    def demand_lines(ax, sku=None):
        actual = history if sku is None else history[history.product_id == sku]
        totals = actual.groupby("week_start").actual.sum()
        ax.plot(totals.index, totals.values, label="Actual", color="black", linewidth=2)
        for name, frame in predictions.items():
            subset = frame if sku is None else frame[frame.product_id == sku]
            total = subset.groupby("week_start").qty.sum()
            ax.plot(total.index, total.values, label=labels[name], alpha=0.8)
        ax.axvline(holdout, color="grey", linestyle="--")
        ax.set_ylabel("Units/week")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    for offset, field, label in (
        (-0.2, "wape_validation", "Validation"),
        (0.2, "wape_holdout", "Holdout"),
    ):
        ax.bar(
            x + offset,
            [np.nan if results[n][field] is None else results[n][field] for n in names],
            width=0.4,
            label=label,
        )
    ax.set_xticks(x, [labels[n] for n in names], rotation=20, ha="right")
    ax.set_ylabel("WAPE (%)")
    ax.legend()
    finish(fig, "wape_by_model")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in names:
        values = results[name]["wape_by_horizon"]
        ax.plot([int(k) for k in values], list(values.values()), marker=".", label=labels[name])
    ax.set(xlabel="Holdout horizon week", ylabel="WAPE (%)", xticks=range(1, 17))
    ax.legend(fontsize=8)
    finish(fig, "wape_by_horizon")

    examples = report["example_products"]
    fig, axes = plt.subplots(
        max(1, len(examples)), 1, figsize=(13, 3 * max(1, len(examples))), squeeze=False
    )
    for ax, sku in zip(axes[:, 0], examples):
        demand_lines(ax, sku)
        ax.set_title(f"{sku} — selected by pre-holdout category volume")
    axes[0, 0].legend(fontsize=7, ncol=3)
    finish(fig, "example_products")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in names:
        r = results[name]
        if r.get("wape_holdout") is not None and r.get("train_seconds") is not None:
            ax.scatter(
                r["train_seconds"],
                r["wape_holdout"],
                s=max(
                    25,
                    500
                    * r.get("n_parameters", 0)
                    / max(1, max(results[n].get("n_parameters", 0) for n in names)),
                ),
            )
            ax.annotate(labels[name], (r["train_seconds"], r["wape_holdout"]), fontsize=8)
    ax.set(
        xlabel="Training seconds (tuning + selection + refit)",
        ylabel="Holdout WAPE (%)",
        title="Marker area proportional to parameters (minimum area for baselines)",
    )
    finish(fig, "accuracy_vs_cost")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in names:
        values = results[name].get("history", {}).get("val_loss", [])
        if values:
            ax.plot(range(1, len(values) + 1), values, marker=".", label=labels[name])
    ax.set(xlabel="Selection epoch", ylabel="Validation MSE (standardized targets)")
    if ax.lines:
        ax.legend(fontsize=8)
    finish(fig, "learning_curves")

    categories = sorted({key for n in names for key in results[n]["wape_by_category"]})
    matrix = np.array(
        [[results[n]["wape_by_category"].get(c) for c in categories] for n in names], dtype=float
    )
    fig, ax = plt.subplots(figsize=(max(8, len(categories)), 5))
    heatmap = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto")
    ax.set_xticks(range(len(categories)), categories, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), [labels[n] for n in names])
    fig.colorbar(heatmap, ax=ax, label="Holdout WAPE (%) — missing cells: zero demand")
    finish(fig, "wape_by_category")

    fig, ax = plt.subplots(figsize=(13, 5))
    demand_lines(ax)
    ax.set_title("Total demand: 52 history weeks and untouched holdout")
    ax.legend(fontsize=8)
    finish(fig, "total_demand")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in names:
        values = results[name].get("trial_values", [])
        if values:
            ax.plot(
                range(1, len(values) + 1),
                np.minimum.accumulate(values),
                marker=".",
                label=labels[name],
            )
    ax.set(xlabel="Completed trial (in order)", ylabel="Best validation MSE so far")
    if ax.lines:
        ax.legend(fontsize=8)
    finish(fig, "optuna_best_so_far")
