"""Model-free, identity-audited return/price evaluation for the world-signals probe."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.bench.baselines import SeasonalNaiveBaseline
from demand_forecast.bench.run import predictions_for_fev
from demand_forecast.bench.specs import dataset_fingerprint, load_task_specs
from demand_forecast.metrics import (
    _nanmean,
    mase,
    quantile_loss,
    seasonal_error_per_item,
    sql,
    wape,
    wql,
)

from .colab_bundle import verify
from .fetch_prices import PANEL_SYMBOLS

RUN_NAMES = ("scratch-on", "scratch-off", "scratch-off-s1", "l3-on", "l3-off", "l3-off-s1")
L3_PRETRAIN_IDENTITY = "83d6acb1f7f2c1b57ba33c0ca0a870775369d52bf8fb739a9e3f7e28858826c2"
MODEL_GEOMETRY = {
    "l3": {
        "d_model": 320,
        "nhead": 5,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 1280,
    },
    "scratch": {
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 256,
    },
}
FOLDS = (2022, 2023, 2024, 2025)
METRICS = ("SQL", "MASE", "WQL", "WAPE")
NOISE_METRICS = (*METRICS, "directional_accuracy", "macroWAPE", "price_WAPE", "price_MASE")
LIMITATIONS = """- Train once per fold; context extends at subsequent evaluation windows.
- Checkpoint selection uses a single 13-day inner validation window.
- Full runs have a 500-step budget; smoke has 150 steps.
- GDELT was dropped after the historical coverage preflight.
- GPR is revised history, published weekly, not vintage-exact data.
- Yahoo GOLD/SILVER are continuous futures proxies; roll and settlement semantics are unverified.
- Session masks use realized sessions from the sidecar, not a prospective exchange calendar.
- CryptoPanic news coverage ends 2025-12-03; per-asset feature coverage varies (see bundle manifest).
- Pretraining corpus overlap with this panel has not been audited.
- Seasonal naive forward-fills its own inputs; the producer and model inputs remain unfilled.
- Random-walk return quantiles are degenerate at zero; price paths are derived point forecasts, not price quantiles.
- Yahoo-derived artifacts remain private; source licenses and attribution are in the bundle manifest.
"""


def price_point_path(median_returns, last_close, session_mask):
    """Compound (N,H) median returns along axis 1, carrying non-session prices.

    This is a derived point path, not a median or any other quantile of price.
    """
    returns = np.asarray(median_returns, dtype=float)
    anchors = np.asarray(last_close, dtype=float)
    sessions = np.asarray(session_mask, dtype=bool)
    if returns.ndim != 2 or anchors.shape != (returns.shape[0],) or sessions.shape != returns.shape:
        raise ValueError("Expected shapes returns=(N,H), last_close=(N,), session_mask=(N,H)")
    if (
        not np.isfinite(anchors).all()
        or (anchors <= 0).any()
        or not np.isfinite(returns[sessions]).all()
    ):
        raise ValueError("Nonfinite returns/anchors or nonpositive price anchor")
    with np.errstate(over="ignore", invalid="ignore"):
        result = anchors[:, None] * np.exp(np.cumsum(np.where(sessions, returns, 0.0), axis=1))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite compounded price path")
    return result


def sql_contributions(actual, arrays, histories, seasonality, quantiles):
    """Per (id,step) doubled pinball / past seasonal scale, averaged over knots.

    Undefined scaled losses are excluded exactly as in fev SQL. Each surviving
    cell carries equal weight in the bootstrap, including across windows.
    """
    scales = seasonal_error_per_item(histories, seasonality)
    with np.errstate(divide="ignore", invalid="ignore"):
        losses = quantile_loss(actual, arrays, quantiles) / scales[:, None, None]
    losses = np.where(np.isfinite(losses), losses, np.nan)
    return _nanmean(losses, axis=2)


def return_window_metrics(actual, arrays, histories, seasonality, quantiles):
    arrays, actual = np.asarray(arrays, dtype=float), np.asarray(actual, dtype=float)
    if (
        arrays.shape != (*actual.shape, len(quantiles))
        or actual.ndim != 2
        or not np.isfinite(arrays).all()
    ):
        raise ValueError("Finite predictions with shape (N,H,Q) required")
    median = arrays[..., list(quantiles).index(0.5)]
    # A zero forecast has no direction; zero/NaN actuals and forecasts are
    # excluded. Record the denominator so RW's undefined accuracy is explicit.
    directional = np.isfinite(actual) & (actual != 0) & (median != 0)
    correct = int(((np.sign(median) == np.sign(actual)) & directional).sum())
    count = int(directional.sum())
    return {
        "SQL": sql(actual, arrays, histories, seasonality, quantiles),
        "MASE": mase(actual, median, histories, seasonality),
        # fev 0.10.0 defaults epsilon to 0; repo helpers default to 1.
        "WQL": wql(actual, arrays, quantiles, eps=0.0),
        "WAPE": wape(actual, median, eps=0.0),
        "directional_accuracy": correct / count if count else np.nan,
        "directional_correct": correct,
        "directional_count": count,
    }


def _interval(delta, values, blocks, n, seed):
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "delta": float(delta),
        "ci_low": float(low),
        "ci_high": float(high),
        "blocks": blocks,
        "resamples": n,
        "seed": seed,
    }


def block_bootstrap(loss_on, loss_off, *, n=2000, seed=0):
    """Paired bootstrap; axis 0 is whole windows (all assets and horizons).

    Resample windows uniformly with replacement. Recompute the mean over all
    finite cells in the sampled windows; do not average unequal window means.
    """
    on, off = np.asarray(loss_on, dtype=float), np.asarray(loss_off, dtype=float)
    if on.shape != off.shape or on.ndim < 2 or len(on) == 0:
        raise ValueError("Expected paired arrays (windows, ...cells)")
    if not np.array_equal(np.isfinite(on), np.isfinite(off)):
        raise ValueError("Expected identical paired finite-cell masks")
    diff = (on - off).reshape(len(on), -1)
    counts = np.isfinite(diff).sum(axis=1)
    if not counts.all() or n <= 0:
        raise ValueError("Every block needs finite cells and resamples must be positive")
    sums = np.nansum(diff, axis=1)
    draws = np.random.default_rng(seed).integers(0, len(on), size=(n, len(on)))
    sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return _interval(sums.sum() / counts.sum(), sampled, len(on), n, seed)


def price_bootstrap(on_errors, off_errors, denominators, *, n=2000, seed=0):
    """Recompute macro per-asset WAPE ratios from sampled whole windows."""
    on, off, denom = (np.asarray(a, dtype=float) for a in (on_errors, off_errors, denominators))
    if on.shape != off.shape or on.shape != denom.shape or on.ndim != 2 or (denom <= 0).any():
        raise ValueError("Expected paired (windows,assets) price sums with positive denominators")
    draws = np.random.default_rng(seed).integers(0, len(on), size=(n, len(on)))
    delta = np.mean((on.sum(axis=0) - off.sum(axis=0)) / denom.sum(axis=0))
    sampled = np.mean(
        (on[draws].sum(axis=1) - off[draws].sum(axis=1)) / denom[draws].sum(axis=1), axis=1
    )
    return _interval(delta, sampled, len(on), n, seed)


def verdict(fold_table, pooled_sql_ci):
    """Apply DESIGN's operational definitions; inputs must be L3 seed-0 folds.

    Columns: fold, delta_SQL, delta_macroWAPE, best_step_on, noise_SQL,
    noise_macroWAPE. Scratch and baseline comparisons never gate the verdict.
    """
    table = pd.DataFrame(fold_table).copy()
    if len(table) != 4 or set(table.fold) != set(FOLDS):
        raise ValueError("Verdict requires the same four unique folds 2022–2025")
    columns = ["delta_SQL", "delta_macroWAPE", "best_step_on", "noise_SQL", "noise_macroWAPE"]
    if not np.isfinite(table[columns].to_numpy(dtype=float)).all():
        raise ValueError("Verdict requires finite metrics, noise floors and best steps")
    if (table[["best_step_on", "noise_SQL", "noise_macroWAPE"]] < 0).any().any():
        raise ValueError("Negative best step or noise floor")
    ci_low, ci_high = map(float, pooled_sql_ci)
    if not np.isfinite([ci_low, ci_high]).all() or ci_low > ci_high:
        raise ValueError("Invalid pooled SQL interval")
    rows = []
    for row in table.sort_values("fold").to_dict("records"):
        win = row["delta_SQL"] < 0 and row["delta_macroWAPE"] <= 0 and row["best_step_on"] > 0
        rows.append(
            {
                "fold": int(row["fold"]),
                "label": "no-learning" if row["best_step_on"] == 0 else ("win" if win else "loss"),
                "delta_SQL": row["delta_SQL"],
                "delta_macroWAPE": row["delta_macroWAPE"],
                "best_step_on": int(row["best_step_on"]),
                "noise_SQL": row["noise_SQL"],
                "noise_macroWAPE": row["noise_macroWAPE"],
                "within_noise_SQL": abs(row["delta_SQL"]) <= row["noise_SQL"],
                "within_noise_macroWAPE": abs(row["delta_macroWAPE"]) <= row["noise_macroWAPE"],
            }
        )
    wins = sum(row["label"] == "win" for row in rows)
    wins_above_noise = sum(row["label"] == "win" and not row["within_noise_SQL"] for row in rows)
    losses_sql = sum(row["delta_SQL"] > 0 for row in rows)
    label = "inconclusive"
    if wins >= 3:
        label = "solid" if ci_high < 0 and wins_above_noise >= 3 else "suggestive"
    elif losses_sql >= 3 and ci_low > 0:
        label = "negative"
    return {
        "verdict": label,
        "decision_arm": "l3",
        "seed": 0,
        "fold_wins": wins,
        "wins_above_SQL_noise": wins_above_noise,
        "pooled_delta_SQL_ci": [ci_low, ci_high],
        "folds": rows,
    }


@dataclass
class WindowData:
    index: int
    cutoff: pd.Timestamp
    ids: list[str]
    dates: np.ndarray
    actual: np.ndarray
    histories: list
    anchors: np.ndarray
    sessions: np.ndarray
    actual_price: np.ndarray
    price_scale: np.ndarray


def window_data(task, prices):
    """Re-derive fev windows; join all histories and sidecar dates by ID/date."""
    price_by_id = {
        str(key): group.set_index("date").sort_index() for key, group in prices.groupby("id")
    }
    output = []
    for index, window in enumerate(task.iter_windows(num_proc=1)):
        past, future = window.get_input_data()
        # datasets' numpy formatter downcasts float64 lists to float32; fev
        # evaluates Arrow float64 directly. Preserve that precision here too.
        truth = {
            str(row[task.id_column]): row for row in window.get_ground_truth().with_format(None)
        }
        past_by_id = {
            str(row[task.id_column]): row
            for row in past.with_format(None).select_columns(
                [task.id_column, task.timestamp_column, task.target]
            )
        }
        future_by_id = {
            str(row[task.id_column]): row
            for row in future.select_columns([task.id_column, task.timestamp_column])
        }
        ids = sorted(truth)
        if set(ids) != set(past_by_id) or set(ids) != set(future_by_id):
            raise ValueError("Past/future/truth ID mismatch")
        cutoff = pd.Timestamp(window.cutoff)
        expected_dates = pd.date_range(
            cutoff + pd.Timedelta(days=1), periods=task.horizon
        ).to_numpy()
        dates, actual, histories, anchors, sessions, closes, scales = [], [], [], [], [], [], []
        for key in ids:
            date = np.asarray(truth[key][task.timestamp_column], dtype="datetime64[ns]")
            if not np.array_equal(date, expected_dates) or not np.array_equal(
                date, np.asarray(future_by_id[key][task.timestamp_column], dtype="datetime64[ns]")
            ):
                raise ValueError(f"{task.task_name}/{index}/{key}: future grid mismatch")
            if pd.Timestamp(past_by_id[key][task.timestamp_column][-1]) != cutoff:
                raise ValueError(f"{key}: history does not end at cutoff")
            sidecar = price_by_id[key]
            valid = sidecar.loc[sidecar.session & np.isfinite(sidecar.close)]
            anchor = valid.loc[:cutoff, "close"]
            if anchor.empty:
                raise ValueError(f"{key}: no last close at or before cutoff")
            initial = valid.loc[: pd.Timestamp(task.initial_cutoff), "close"].to_numpy()
            if len(initial) < 2:
                raise ValueError(f"{key}: no session-to-session price scale")
            scale = np.mean(np.abs(np.diff(initial)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{key}: undefined price MASE scale")
            if not pd.DatetimeIndex(date).isin(sidecar.index).all():
                raise ValueError(f"{key}: sidecar missing future dates")
            future_prices = sidecar.loc[date]
            # Crypto forecasts compound on every calendar day, even if a quote is absent.
            session = (
                np.ones(task.horizon, dtype=bool)
                if key not in {"GOLD", "SILVER"}
                else future_prices.session.to_numpy(dtype=bool)
            )
            actual_close = future_prices.close.to_numpy(dtype=float)
            actual_close = np.where(
                future_prices.session.to_numpy(dtype=bool) & np.isfinite(actual_close),
                actual_close,
                np.nan,
            )
            dates.append(date)
            actual.append(truth[key][task.target])
            histories.append(past_by_id[key][task.target])
            anchors.append(anchor.iloc[-1])
            sessions.append(session)
            closes.append(actual_close)
            scales.append(scale)
        output.append(
            WindowData(
                index,
                cutoff,
                ids,
                np.array(dates),
                np.array(actual, dtype=float),
                histories,
                np.array(anchors),
                np.array(sessions),
                np.array(closes),
                np.array(scales),
            )
        )
    return output


def load_arrays(path, identity, ids, horizon, quantiles):
    with np.load(path, allow_pickle=False) as saved:
        if not identity or saved["identity"].item() != identity:
            raise ValueError(f"{path}: run identity mismatch")
        rows = predictions_for_fev(saved["ids"].tolist(), saved["arrays"], ids, quantiles)
    arrays = np.asarray([[row[str(q)] for q in quantiles] for row in rows], dtype=float).transpose(
        0, 2, 1
    )
    if arrays.shape != (len(ids), horizon, len(quantiles)) or not np.isfinite(arrays).all():
        raise ValueError(f"{path}: invalid shape or nonfinite predictions")
    return arrays


def _summary_rows(directory):
    path = directory / "summaries.csv"
    if not path.exists():
        raise ValueError(f"{path}: missing")
    rows = pd.read_csv(path, float_precision="round_trip")
    if not {"task", "model_name", "SQL"}.issubset(rows):
        raise ValueError(f"{path}: required columns missing")
    if rows.duplicated(["task", "model_name"]).any():
        raise ValueError(f"{path}: duplicate task/model rows")
    return rows


def _check_summary(row, task, metrics):
    if row.get("dataset_fingerprint") != dataset_fingerprint(task):
        raise ValueError(f"{task.task_name}: summary/bundle dataset fingerprint mismatch")
    for key in METRICS:
        value = row.get(key)
        if (
            value is None
            or not np.isfinite(value)
            or not np.isclose(metrics[key], value, rtol=1e-6, atol=0)
        ):
            raise ValueError(
                f"{task.task_name}: {key} cross-check failed: recomputed={metrics[key]}, summaries.csv={value}"
            )
    if not np.isclose(metrics["SQL"], row.get("test_error", np.nan), rtol=1e-6, atol=0):
        raise ValueError(f"{task.task_name}: SQL/test_error mismatch")


def _training(run, spec, summary, report, directory):
    history_path = directory / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    step0 = (
        []
        if history is None
        else [
            row["SQL"]
            for row in history["history"]
            if row.get("step") == 0 and row.get("SQL") is not None
        ]
    )
    if len(step0) > 1:
        raise ValueError(f"{directory}: duplicate step-0 history")
    fit = report.get("fit_result", {})
    if history:
        for key in ("best_step", "best_score"):
            if history.get(key) != fit.get(key):
                raise ValueError(f"{directory}: history/run {key} mismatch")
        if history.get("max_steps") != report.get("resolved_budget", {}).get("max_steps"):
            raise ValueError(f"{directory}: history/run max_steps mismatch")
    result = {
        "run": run,
        "task": spec.name,
        "fold": int(str(spec.initial_cutoff)[:4]),
        **{key: fit.get(key) for key in ("best_step", "steps", "best_score", "stop_reason")},
        "max_steps": report.get("resolved_budget", {}).get("max_steps"),
        "n_val_items": report.get("n_val_items"),
        "history_present": history is not None,
        "step0_SQL": step0[0] if step0 else None,
        "step0_pair_equal": None,
        "run_identity": report.get("run_identity"),
    }
    for key in (
        "nonfinite_losses",
        "cold_covariate_columns",
        "floored_covariate_columns",
        "clipped_covariate_values",
        "fallback_no_context",
        "fallback_undefined_scale",
    ):
        result[key] = summary.get(key)
    return result


def _validate_arm(run, report, directory):
    """Check registered controls; return settings eligible for paired comparison."""
    smoke = run in {"smoke", "probe_smoke"}
    if run not in RUN_NAMES and not smoke:
        raise ValueError(f"{directory}: unregistered run directory {run!r}")
    arm = "smoke" if smoke else run.split("-")[0]
    settings = report.get("settings")
    if not isinstance(settings, dict) or not isinstance(settings.get("training"), dict):
        raise ValueError(f"{directory}: missing effective settings/training")
    seed = 1 if run.endswith("-s1") else 0
    if report.get("seed") != seed or settings.get("seed") != seed:
        raise ValueError(f"{directory}: {run} requires seed {seed} in run.json and settings")
    model = settings.get("model", {})
    for key, expected in MODEL_GEOMETRY["scratch" if smoke else arm].items():
        if model.get(key) != expected:
            raise ValueError(f"{directory}: {run} requires model.{key} == {expected}")
    budget = 150 if smoke else 500
    if report.get("resolved_budget", {}).get("max_steps") != budget:
        raise ValueError(f"{directory}: {run} requires resolved_budget.max_steps == {budget}")
    if arm == "l3":
        transfer = report.get("transfer")
        if (
            not isinstance(transfer, dict)
            or transfer.get("pretrain_identity") != L3_PRETRAIN_IDENTITY
        ):
            raise ValueError(
                f"{directory}: L3 transfer.pretrain_identity must be {L3_PRETRAIN_IDENTITY}"
            )
    elif "transfer" in report:
        raise ValueError(f"{directory}: {run} must have no transfer metadata")
    return arm, {key: value for key, value in settings.items() if key not in {"data", "seed"}}


def _run_metadata(directories):
    """Validate every supplied task's controls before any predictions are scored."""
    reports, arm_settings = {}, {}
    for run, directory in directories:
        paths = sorted(directory.glob("*/run.json"))
        if not paths:
            raise ValueError(f"{directory}: no task run.json files")
        for path in paths:
            report = json.loads(path.read_text())
            arm, settings = _validate_arm(run, report, path.parent)
            if arm in arm_settings and settings != arm_settings[arm][1]:
                raise ValueError(
                    f"{path}: effective settings differ within {arm} arm from {arm_settings[arm][0]} "
                    "after removing data roles and seed"
                )
            arm_settings.setdefault(arm, (path, settings))
            reports[path.parent] = report
    return reports


def _step0_checks(diagnostics, *, partial):
    issues = []
    for fold in FOLDS:
        pairs = [
            next((row for row in diagnostics if row["run"] == run and row["fold"] == fold), None)
            for run in ("l3-on", "l3-off")
        ]
        if all(row is not None and row["history_present"] for row in pairs):
            a, b = [row["step0_SQL"] for row in pairs]
            finite = a is not None and b is not None and np.isfinite([a, b]).all()
            equal = bool(finite and np.isclose(a, b, rtol=1e-4, atol=0))
            for row in pairs:
                row["step0_pair_equal"] = equal
            if not equal:
                issues.append(f"L3 {fold}: finite step-0 SQL equality check failed ({a}, {b})")
        elif not partial:
            issues.append(f"L3 {fold}: history.json with finite step-0 SQL required for ON and OFF")
    if issues and not partial:
        raise ValueError("Full report requires all four L3 step-0 checks: " + "; ".join(issues))
    return issues


def score_task(run, spec, contexts, forecasts, cell_frames, path_frames):
    """Aggregate task scores from windows and price ratios from all fold cells."""
    quantiles = spec.quantile_levels
    rows, contributions, price_errors, price_denominators, price_counts = [], [], [], [], []
    asset_returns = {key: [] for key in contexts[0].ids}
    for data, arrays in zip(contexts, forecasts, strict=True):
        metrics = return_window_metrics(
            data.actual, arrays, data.histories, spec.seasonality, quantiles
        )
        rows.append(metrics)
        cells = sql_contributions(data.actual, arrays, data.histories, spec.seasonality, quantiles)
        contributions.append(cells)
        median = arrays[..., list(quantiles).index(0.5)]
        path = price_point_path(median, data.anchors, data.sessions)
        errors = np.abs(path - data.actual_price)
        price_errors.append(np.nansum(errors, axis=1))
        price_denominators.append(np.nansum(np.abs(data.actual_price), axis=1))
        price_counts.append(np.isfinite(data.actual_price).sum(axis=1))
        for i, key in enumerate(data.ids):
            asset_returns[key].append(
                return_window_metrics(
                    data.actual[i : i + 1],
                    arrays[i : i + 1],
                    [data.histories[i]],
                    spec.seasonality,
                    quantiles,
                )
            )
        keys = {
            "run": run,
            "task": spec.name,
            "fold": int(str(spec.initial_cutoff)[:4]),
            "window": data.index,
            "cutoff": data.cutoff,
            "id": np.repeat(data.ids, spec.horizon),
            "step": np.tile(np.arange(1, spec.horizon + 1), len(data.ids)),
            "date": data.dates.ravel(),
        }
        cell_frames.append(
            pd.DataFrame(
                {
                    **keys,
                    "actual_return": data.actual.ravel(),
                    "median_return": median.ravel(),
                    "SQL_contribution": cells.ravel(),
                }
            )
        )
        path_frames.append(
            pd.DataFrame(
                {
                    **keys,
                    "last_close": np.repeat(data.anchors, spec.horizon),
                    "session": data.sessions.ravel(),
                    "actual_close": data.actual_price.ravel(),
                    "point_path": path.ravel(),
                    "absolute_error": errors.ravel(),
                }
            )
        )
    err, denom, counts = map(np.asarray, (price_errors, price_denominators, price_counts))
    if not (denom.sum(axis=0) > 0).all() or not counts.sum(axis=0).all():
        raise ValueError(f"{run}/{spec.name}: missing asset price observations")
    asset_wape = err.sum(axis=0) / denom.sum(axis=0)
    asset_mase = err.sum(axis=0) / counts.sum(axis=0) / contexts[0].price_scale
    count = sum(row["directional_count"] for row in rows)
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    metrics.update(
        {
            "directional_accuracy": sum(row["directional_correct"] for row in rows) / count
            if count
            else np.nan,
            "directional_count": count,
            "macroWAPE": float(asset_wape.mean()),
            "price_WAPE": float(err.sum() / denom.sum()),
            "price_MASE": float(asset_mase.mean()),
        }
    )
    assets = []
    for i, key in enumerate(contexts[0].ids):
        values = asset_returns[key]
        n = sum(row["directional_count"] for row in values)
        assets.append(
            {
                "run": run,
                "task": spec.name,
                "fold": int(str(spec.initial_cutoff)[:4]),
                "id": key,
                **{metric: float(np.mean([row[metric] for row in values])) for metric in METRICS},
                "directional_accuracy": sum(row["directional_correct"] for row in values) / n
                if n
                else np.nan,
                "directional_count": n,
                "price_WAPE": asset_wape[i],
                "price_MASE": asset_mase[i],
                "price_scale_initial_cutoff": contexts[0].price_scale[i],
                "price_cells": int(counts[:, i].sum()),
            }
        )
    return {
        "metrics": metrics,
        "assets": assets,
        "sql_cells": np.array(contributions),
        "price_errors": err,
        "price_denominators": denom,
        "contexts": contexts,
    }


def _paired(on, off):
    if len(on["contexts"]) != len(off["contexts"]):
        raise ValueError("ON/OFF window counts differ")
    for a, b in zip(on["contexts"], off["contexts"], strict=True):
        if (
            a.ids != b.ids
            or a.cutoff != b.cutoff
            or not np.array_equal(a.dates, b.dates)
            or not np.array_equal(a.actual, b.actual, equal_nan=True)
        ):
            raise ValueError("ON/OFF window alignment or actuals differ")
    if not np.array_equal(on["price_denominators"], off["price_denominators"]):
        raise ValueError("ON/OFF price actuals differ")


def comparison_tables(results, diagnostics):
    folds, bootstraps, noise = [], [], []
    diagnostic_index = {(row["run"], row["fold"]): row for row in diagnostics}
    for arm in ("scratch", "l3"):
        paired = []
        for fold in FOLDS:
            on = results.get((f"{arm}-on", fold))
            off = results.get((f"{arm}-off", fold))
            seed1 = results.get((f"{arm}-off-s1", fold))
            if off is not None and seed1 is not None:
                _paired(off, seed1)
                for metric in NOISE_METRICS:
                    noise.append(
                        {
                            "arm": arm,
                            "fold": fold,
                            "metric": metric,
                            "off_seed0": off["metrics"][metric],
                            "off_seed1": seed1["metrics"][metric],
                            "noise_floor": abs(off["metrics"][metric] - seed1["metrics"][metric]),
                        }
                    )
            if on is None or off is None:
                continue
            _paired(on, off)
            paired.append((on, off))
            row = {
                "arm": arm,
                "fold": fold,
                "best_step_on": diagnostic_index[(f"{arm}-on", fold)]["best_step"],
            }
            for metric in NOISE_METRICS:
                delta = on["metrics"][metric] - off["metrics"][metric]
                floor = (
                    abs(off["metrics"][metric] - seed1["metrics"][metric])
                    if seed1 is not None
                    else np.nan
                )
                row.update(
                    {
                        f"on_{metric}": on["metrics"][metric],
                        f"off_{metric}": off["metrics"][metric],
                        f"delta_{metric}": delta,
                        f"noise_{metric}": floor,
                        f"within_noise_{metric}": bool(abs(delta) <= floor)
                        if np.isfinite(floor) and np.isfinite(delta)
                        else None,
                    }
                )
            rw = results[("random_walk", fold)]
            row["random_walk_macroWAPE"] = rw["metrics"]["macroWAPE"]
            row["price_skill_vs_random_walk"] = (
                1 - on["metrics"]["macroWAPE"] / rw["metrics"]["macroWAPE"]
            )
            row["label"] = (
                "no-learning"
                if row["best_step_on"] == 0
                else (
                    "win"
                    if row["best_step_on"] is not None
                    and row["best_step_on"] > 0
                    and row["delta_SQL"] < 0
                    and row["delta_macroWAPE"] <= 0
                    else "loss"
                )
            )
            folds.append(row)
            bootstraps.extend(_bootstrap_rows(arm, fold, [on], [off]))
        if paired:
            bootstraps.extend(
                _bootstrap_rows(arm, "pooled", [p[0] for p in paired], [p[1] for p in paired])
            )
    return folds, bootstraps, noise


def _bootstrap_rows(arm, fold, on, off):
    sql_result = block_bootstrap(
        np.concatenate([p["sql_cells"] for p in on]), np.concatenate([p["sql_cells"] for p in off])
    )
    price_result = price_bootstrap(
        np.concatenate([p["price_errors"] for p in on]),
        np.concatenate([p["price_errors"] for p in off]),
        np.concatenate([p["price_denominators"] for p in on]),
    )
    return [
        {
            "arm": arm,
            "fold": fold,
            "metric": "SQL",
            "estimator": "mean per-cell contribution",
            **sql_result,
        },
        {
            "arm": arm,
            "fold": fold,
            "metric": "macroWAPE",
            "estimator": "recomputed per-asset ratios",
            **price_result,
        },
    ]


def _markdown(frame):
    if frame.empty:
        return "No evidence available.\n"

    def cell(value):
        if pd.isna(value):
            return "unknown"
        if isinstance(value, float):
            return f"{value:.9g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines) + "\n"


def build_report(bundle, runs, out, *, partial=False):
    bundle, runs, out = map(Path, (bundle, runs, out))
    # An unsuccessful rerun must not leave an earlier verdict looking current.
    (out / "verdict.json").unlink(missing_ok=True)
    verify(bundle)
    specs = {
        spec.name: spec
        for variant in ("on", "off", "smoke")
        for spec in load_task_specs(bundle / f"probe_{variant}.yaml", bundle / "profiles.yaml")
    }
    prices = pd.read_parquet(bundle / "prices.parquet")
    prices["date"] = pd.to_datetime(prices.date)
    if prices.duplicated(["id", "date"]).any():
        raise ValueError("Duplicate sidecar id/date")
    if (runs / "summaries.csv").is_file():
        present = [(runs.name, runs)]
    else:
        present = [
            (path.name, path)
            for path in sorted(runs.iterdir())
            if path.is_dir()
            and path.name != "baseline"
            and (
                path.name in {*RUN_NAMES, "smoke", "probe_smoke"}
                or (path / "summaries.csv").is_file()
                or any(path.glob("*/run.json"))
            )
        ]
    if not partial:
        missing = [name for name in RUN_NAMES if not (runs / name).is_dir()]
        if missing:
            raise ValueError(
                f"Missing required run directories: {missing}; use --partial for smoke"
            )
        directories = [(name, runs / name) for name in RUN_NAMES]
    else:
        directories = present
    if not directories:
        raise ValueError("No runs with summaries.csv found")
    reports = _run_metadata(present)
    cache, used, results = {}, set(), {}
    tasks, assets, diagnostics, baseline_rows, cells, paths, issues = [], [], [], [], [], [], []
    synthetic = False

    def context(spec):
        if spec.name not in cache:
            task = spec.to_task()
            cache[spec.name] = (task, window_data(task, prices))
        return cache[spec.name]

    for run, directory in directories:
        summaries = _summary_rows(directory)
        if summaries.task.duplicated().any():
            raise ValueError(f"{directory}: multiple models for one task")
        if not partial:
            variant = "on" if run.endswith("-on") else "off"
            expected = {f"probe_{fold}_{variant}" for fold in FOLDS}
            if set(summaries.task) != expected:
                raise ValueError(f"{run}: requires all four expected tasks {sorted(expected)}")
        for summary in summaries.to_dict("records"):
            name = summary["task"]
            if name not in specs:
                raise ValueError(f"Unknown bundle task: {name}")
            spec = specs[name]
            if run in RUN_NAMES and not name.endswith("_on" if run.endswith("-on") else "_off"):
                raise ValueError(f"{run}/{name}: incorrect ON/OFF task")
            task, contexts = context(spec)
            if not partial and (
                spec.num_windows != 18
                or spec.horizon != 13
                or any(data.ids != sorted(PANEL_SYMBOLS) for data in contexts)
            ):
                raise ValueError(
                    "Full report requires 18 windows × 13 steps × the ten registered assets"
                )
            task_dir = directory / name
            if task_dir not in reports:
                raise ValueError(f"{task_dir}: run.json missing")
            report = reports[task_dir]
            synthetic |= bool(report.get("synthetic", False))
            identity = report.get("run_identity")
            if not identity or summary.get("run_identity") != identity:
                raise ValueError(f"{task_dir}: summary/run identity mismatch")
            if report.get("task_definition") != spec.task_definition():
                raise ValueError(f"{task_dir}: run/bundle task definition mismatch")
            if report.get("dataset_fingerprint") != dataset_fingerprint(task):
                raise ValueError(f"{task_dir}: run/bundle dataset fingerprint mismatch")
            present = {p.name for p in task_dir.glob("window_*.npz")}
            if present != {f"window_{i}.npz" for i in range(spec.num_windows)}:
                raise ValueError(f"{task_dir}: incomplete/extra prediction windows")
            forecasts = [
                load_arrays(
                    task_dir / f"window_{data.index}.npz",
                    identity,
                    data.ids,
                    spec.horizon,
                    spec.quantile_levels,
                )
                for data in contexts
            ]
            scored = score_task(run, spec, contexts, forecasts, cells, paths)
            _check_summary(summary, task, scored["metrics"])
            diag = _training(run, spec, summary, report, task_dir)
            diagnostics.append(diag)
            fold = int(str(spec.initial_cutoff)[:4])
            if (run, fold) in results:
                raise ValueError(f"Duplicate fold in run {run}: {fold}")
            results[(run, fold)] = scored
            tasks.append(
                {
                    "run": run,
                    "task": name,
                    "fold": fold,
                    **scored["metrics"],
                    "summary_SQL": summary["SQL"],
                    "SQL_difference": scored["metrics"]["SQL"] - summary["SQL"],
                    "SQL_crosscheck": "passed",
                    "run_identity": identity,
                }
            )
            assets.extend(scored["assets"])
            used.add(name)

    baseline_dir = runs / "baseline"
    saved_baselines = (
        _summary_rows(baseline_dir)
        if (baseline_dir / "summaries.csv").is_file()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    for name in sorted(used):
        spec = specs[name]
        task, contexts = context(spec)
        naive = SeasonalNaiveBaseline().fit_predict(task)
        naive_arrays = []
        # fev baseline predictions are positional by contract; attach their
        # source window IDs first, then perform the same explicit ID alignment.
        for raw, window, data in zip(naive, task.iter_windows(num_proc=1), contexts, strict=True):
            cleaned = task.clean_and_validate_predictions(raw)[task.target].with_format(None)
            source_ids = [str(row[task.id_column]) for row in window.get_ground_truth()]
            values = np.asarray(
                [[row[str(q)] for q in spec.quantile_levels] for row in cleaned], dtype=float
            ).transpose(0, 2, 1)
            aligned = predictions_for_fev(source_ids, values, data.ids, spec.quantile_levels)
            naive_arrays.append(
                np.asarray(
                    [[row[str(q)] for q in spec.quantile_levels] for row in aligned]
                ).transpose(0, 2, 1)
            )
        for model in ("random_walk", "seasonal_naive"):
            forecasts = (
                [
                    np.zeros((len(data.ids), spec.horizon, len(spec.quantile_levels)))
                    for data in contexts
                ]
                if model == "random_walk"
                else naive_arrays
            )
            rows = saved_baselines.loc[
                (saved_baselines.task == name) & (saved_baselines.model_name == model)
            ]
            if model == "random_walk" and not rows.empty:
                for data in contexts:
                    saved = load_arrays(
                        baseline_dir / name / f"window_{data.index}.npz",
                        rows.iloc[0].get("run_identity"),
                        data.ids,
                        spec.horizon,
                        spec.quantile_levels,
                    )
                    if np.any(saved != 0):
                        raise ValueError(f"{name}: random-walk saved forecasts are not zero")
            scored = score_task(model, spec, contexts, forecasts, cells, paths)
            if not rows.empty:
                _check_summary(rows.iloc[0].to_dict(), task, scored["metrics"])
            fold = int(str(spec.initial_cutoff)[:4])
            if (model, fold) in results:
                other = results[(model, fold)]
                _paired(other, scored)
                if not np.allclose(
                    other["sql_cells"], scored["sql_cells"], rtol=1e-10, atol=0, equal_nan=True
                ):
                    raise ValueError(f"{model}/{fold}: baseline differs across ON/OFF tasks")
            results[(model, fold)] = scored
            baseline_rows.append(
                {
                    "model": model,
                    "task": name,
                    "fold": fold,
                    **scored["metrics"],
                    "SQL_crosscheck": "passed" if not rows.empty else "no saved summary",
                    "quantiles_degenerate": model == "random_walk",
                }
            )
            assets.extend(scored["assets"])

    issues = _step0_checks(diagnostics, partial=partial)
    folds, bootstrap, noise = comparison_tables(results, diagnostics)
    decision = None
    if not partial:
        pooled = next(
            row
            for row in bootstrap
            if row["arm"] == "l3" and row["fold"] == "pooled" and row["metric"] == "SQL"
        )
        decision = verdict(
            [row for row in folds if row["arm"] == "l3"], (pooled["ci_low"], pooled["ci_high"])
        )
        decision.update(
            {
                "synthetic": synthetic,
                "audit_passed": not issues,
                "issues": issues,
                "SQL_crosscheck": "passed",
                "price_claim": [
                    {
                        "fold": row["fold"],
                        "on_macroWAPE": row["on_macroWAPE"],
                        "random_walk_macroWAPE": row["random_walk_macroWAPE"],
                        "price_skill_vs_random_walk": row["price_skill_vs_random_walk"],
                    }
                    for row in folds
                    if row["arm"] == "l3"
                ],
            }
        )
    table_data = {
        "per_task_metrics": tasks,
        "per_asset_price": assets,
        "fold_deltas": folds,
        "bootstrap": bootstrap,
        "noise_floor": noise,
        "training_diagnostics": diagnostics,
        "baselines": baseline_rows,
    }
    empty_columns = {
        "fold_deltas": ["arm", "fold", "delta_SQL", "delta_macroWAPE"],
        "bootstrap": ["arm", "fold", "metric", "delta", "ci_low", "ci_high", "blocks"],
        "noise_floor": ["arm", "fold", "metric", "noise_floor"],
    }
    tables = {
        name: pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_columns.get(name, []))
        for name, rows in table_data.items()
    }
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / "tables" / f"{name}.csv", index=False)
    pd.concat(cells, ignore_index=True).to_parquet(out / "return_cells.parquet", index=False)
    pd.concat(paths, ignore_index=True).to_parquet(out / "price_paths.parquet", index=False)
    if decision is not None:
        (out / "verdict.json").write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    else:
        (out / "verdict.json").unlink(missing_ok=True)  # Do not leave a stale full verdict.
    heading = "# World-signals probe report\n\n"
    if synthetic:
        heading += "**FAKE / SYNTHETIC predictions and training diagnostics. Pipeline dry run only; no experimental result.**\n\n"
    heading += f"Verdict: {decision['verdict'] if decision else 'SKIPPED (--partial)'}. SQL cross-check: passed for every supplied summary.\n\n"
    heading += "Decision arm: L3 seed 0. Scratch is context only. Baselines do not gate the verdict. Positive price skill means lower macroWAPE than random walk.\n\n"
    heading += "SQL/MASE/WQL/WAPE average fev's pooled window scores. SQL bootstrap estimates the mean per-cell ON−OFF contribution; windows with more observed cells carry more weight. One block is one whole evaluation window; 2,000 paired resamples, seed 0, percentile 95% intervals. Price bootstrap recomputes per-asset WAPE ratios after resampling windows, then averages assets (including for pooled folds). Noise flags use |delta| ≤ |OFF s0−OFF s1|. Direction excludes zero/NaN actuals and zero forecasts; denominators are reported.\n\n"
    heading += "Price paths and return-cell audit records: [price_paths.parquet](price_paths.parquet), [return_cells.parquet](return_cells.parquet). Price MASE uses session-to-session close changes only through the fold's initial cutoff. Paths compound the median return over sessions and are point forecasts, not price quantiles.\n\n"
    heading += (
        "## Audit notes\n\n"
        + (
            "\n".join(f"- {issue}" for issue in issues)
            if issues
            else "No failed checks. Missing history remains unknown; it is never imputed."
        )
        + "\n\n"
    )
    for name, frame in tables.items():
        heading += f"## {name}\n\n[CSV](tables/{name}.csv)\n\n" + _markdown(frame) + "\n"
    heading += (
        "## Interpretation\n\nTo complete after artifact review: describe fold spread, uncertainty, seed noise, learning diagnostics and price skill versus random walk.\n\n## Limitations\n\n"
        + LIMITATIONS
    )
    (out / "REPORT.md").write_text(heading)
    return {
        "verdict": decision,
        "issues": issues,
        "tasks": len(tasks),
        "baseline_rows": len(baseline_rows),
        "SQL_crosscheck": "passed",
        "synthetic": synthetic,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_report(args.bundle, args.runs, args.out, partial=args.partial)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Report failed: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["issues"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
