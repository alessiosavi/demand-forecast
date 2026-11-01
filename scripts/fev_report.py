"""Audit persisted fev evidence and report scores, diagnostics and completeness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from demand_forecast.bench.run import prediction_diagnostics, predictions_for_fev
from demand_forecast.bench.specs import FevTaskSpec, dataset_fingerprint, load_task_specs
from demand_forecast.metrics import quantile_loss, seasonal_error_per_item, summarize

METRICS = ("SQL", "MASE", "WAPE", "WQL")


def _finite(value) -> bool:
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))


def _cell(value) -> str:
    if value is None or (isinstance(value, (float, np.number)) and not _finite(value)):
        return "unknown"
    if isinstance(value, (float, np.floating)):
        return f"{value:.9g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict]) -> str:
    if not rows:
        return "No evidence available.\n"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(_cell(row.get(key)) for key in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _read_json(path: Path, issues: list[str], task: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        issues.append(f"{task}: missing/unreadable {path.name}: {exc}")
        return {}


def _one_row(frame: pd.DataFrame, task: str, model: str, label: str, issues: list[str]) -> dict:
    rows = frame.loc[(frame.task == task) & (frame.model_name == model)]
    if len(rows) != 1:
        issues.append(f"{task}: expected exactly one {label} row, found {len(rows)}")
        return {}
    return rows.iloc[0].to_dict()


def _required(values: dict, keys, task: str, label: str, issues: list[str]) -> None:
    for key in keys:
        if not _finite(values.get(key)):
            issues.append(f"{task}: missing/non-finite {label}.{key}")


def _bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _window_evidence(task, task_dir, identity, issues):
    windows, diagnostics, metrics = [], [], []
    forecast_count = excluded_pairs = excluded_cells = excluded_sql = 0
    for index, window in enumerate(task.iter_windows()):
        past, _ = window.get_input_data()
        truth = window.get_ground_truth()
        item_ids = [str(row[task.id_column]) for row in truth]
        forecast_count += len(item_ids)
        row = {
            "task": task.task_name,
            "window": index,
            "fallback_no_context": None,
            "fallback_undefined_scale": None,
            "inference_time_s": None,
            "device": None,
        }
        windows.append(row)
        path = task_dir / f"window_{index}.npz"
        try:
            with np.load(path, allow_pickle=False) as saved:
                if identity is None or saved["identity"].item() != identity:
                    raise ValueError("run identity mismatch")
                # Reuse the runner's strict missing/duplicate/extra-id checks.
                aligned = predictions_for_fev(
                    saved["ids"].tolist(), saved["arrays"], item_ids, task.quantile_levels
                )
                arrays = np.asarray(
                    [[item[str(q)] for q in task.quantile_levels] for item in aligned],
                    dtype=np.float64,
                ).transpose(0, 2, 1)
                if arrays.shape != (len(item_ids), task.horizon, len(task.quantile_levels)):
                    raise ValueError("forecast horizon does not match task")
                diagnostic = prediction_diagnostics(saved)
            if diagnostic is None:
                issues.append(f"{task.task_name}: window_{index}.npz diagnostics unknown (legacy)")
            else:
                row.update(diagnostic)
                _required(
                    diagnostic,
                    ("fallback_no_context", "fallback_undefined_scale", "inference_time_s"),
                    task.task_name,
                    f"window_{index}",
                    issues,
                )
                if not diagnostic["device"]:
                    issues.append(f"{task.task_name}: window_{index} device unknown")
            diagnostics.append(diagnostic)
            finite_predictions = np.isfinite(arrays).all()
            if not finite_predictions:
                issues.append(f"{task.task_name}: window_{index} non-finite predictions")
            labels = np.asarray([row[task.target] for row in truth], dtype=np.float64)
            past_by_id = {
                str(row[task.id_column]): np.asarray(row[task.target], dtype=np.float64)
                for row in past
            }
            histories = [past_by_id[key] for key in item_ids]
            metrics.append(
                summarize(labels, arrays, histories, task.seasonality, task.quantile_levels)
            )
            if finite_predictions:
                row.update({key: metrics[-1][key] for key in ("coverage_80", "MASE")})
            scales = seasonal_error_per_item(histories, task.seasonality)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                scaled = quantile_loss(labels, arrays, task.quantile_levels) / scales[:, None, None]
            nonfinite = ~np.isfinite(scaled)
            observed = ~np.isnan(labels)
            # Missing labels are counted separately, without inflating the pair count.
            excluded_pairs += int((nonfinite & observed[..., None]).any(axis=(1, 2)).sum())
            excluded_cells += int((~observed).sum())
            excluded_sql += int(nonfinite.sum())
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            issues.append(f"{task.task_name}: window_{index}.npz missing/invalid: {exc}")
            # Keep one entry per expected window, including partial runs.
            if len(diagnostics) <= index:
                diagnostics.append(None)
    complete = len(metrics) == task.num_windows
    averaged = (
        {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
        if complete and metrics
        else {}
    )
    _required(
        averaged,
        [*METRICS, "coverage_80", "width_80", *[f"pinball_{q}" for q in task.quantile_levels]],
        task.task_name,
        "recomputed",
        issues,
    )
    total = {
        "task": task.task_name,
        "coverage_80": averaged.get("coverage_80"),
        "width_80": averaged.get("width_80"),
        "excluded_item_window_pairs": excluded_pairs if complete else None,
        "excluded_horizon_cells": excluded_cells if complete else None,
        "excluded_SQL_cells_knots": excluded_sql if complete else None,
    }
    for key in ("fallback_no_context", "fallback_undefined_scale", "inference_time_s"):
        total[key] = (
            sum(d[key] for d in diagnostics) if all(d is not None for d in diagnostics) else None
        )
    total["inference_device"] = (
        ",".join(sorted({d["device"] for d in diagnostics}))
        if all(d is not None for d in diagnostics)
        else None
    )
    return windows, total, averaged, forecast_count


def window_metrics(run_dir: Path, task) -> dict:
    """Recompute per-window coverage_80 and MASE from saved predictions for a fev.Task.

    Also return the loaded dataset fingerprint, forecast count and run identity
    so a caller can check that the metrics belong to its evaluation summaries.
    Legacy timing diagnostics are not needed for these two metrics.
    """
    if not isinstance(task.target, str):
        raise ValueError("window_metrics requires a univariate task")
    task_dir = Path(run_dir) / task.task_name
    issues: list[str] = []
    report = _read_json(task_dir / "run.json", issues, task.task_name)
    fingerprint = dataset_fingerprint(task)
    rows, _, _, count = _window_evidence(task, task_dir, report.get("run_identity"), issues)
    result = {key: [row.get(key) for row in rows] for key in ("coverage_80", "MASE")}
    if len(rows) != task.num_windows or any(
        not _finite(value) for values in result.values() for value in values
    ):
        raise ValueError(f"{task.task_name}: incomplete window metrics: {'; '.join(issues)}")
    return {
        **result,
        "dataset_fingerprint": fingerprint,
        "num_forecasts": count,
        "run_identity": report.get("run_identity"),
    }


def build_report(
    run_dir: Path,
    specs: list[FevTaskSpec],
    baseline: str = "seasonal_naive",
    model_name: str = "demand-forecast-v2",
) -> tuple[str, list[str]]:
    """Return Markdown and explicit completeness failures, without changing evidence."""
    run_dir = Path(run_dir)
    issues: list[str] = []
    training, timing, scores, diagnostics = [], [], [], []
    knots, windows, sizes, fingerprints = [], [], [], []
    summary_path = run_dir / "summaries.csv"
    if summary_path.exists():
        frame = pd.read_csv(summary_path, float_precision="round_trip")
        if not {"task", "model_name"} <= set(frame.columns):
            issues.append("summaries.csv: missing task/model_name columns")
            frame = pd.DataFrame(columns=["task", "model_name"])
    else:
        issues.append("summaries.csv: missing")
        frame = pd.DataFrame(columns=["task", "model_name"])
    for spec in specs:
        task_dir = run_dir / spec.name
        sizes.append(
            {
                "task": spec.name,
                "checkpoints_bytes": _bytes(task_dir / "checkpoints"),
                "bundle_bytes": _bytes(task_dir / "bundle"),
                "NPZ_bytes": sum(path.stat().st_size for path in task_dir.glob("window_*.npz")),
            }
        )
        model = _one_row(frame, spec.name, model_name, "model", issues)
        reference = _one_row(frame, spec.name, baseline, "baseline", issues)
        report = _read_json(task_dir / "run.json", issues, spec.name)
        manifest = _read_json(task_dir / "bundle/manifest.json", issues, spec.name)
        progress = _read_json(task_dir / "checkpoints/progress.json", issues, spec.name)
        identity = report.get("run_identity")
        identity_ok = bool(identity) and all(
            value.get("run_identity") == identity for value in (model, manifest, progress)
        )
        if not identity_ok:
            issues.append(
                f"{spec.name}: run identity mismatch or unknown in model/run/bundle/progress"
            )
        fit, budget = report.get("fit_result", {}), report.get("resolved_budget", {})
        cumulative = report.get("timings", {}).get("cumulative", {})
        nonfinite_losses = fit.get("nonfinite_losses", cumulative.get("nonfinite_losses"))
        if _finite(nonfinite_losses) and nonfinite_losses > 0:
            issues.append(f"{spec.name}: {nonfinite_losses} non-finite training losses")
        training.append(
            {
                "task": spec.name,
                "transfer": report["transfer"]["pretrain_identity"][:8]
                if report.get("transfer")
                else "-",
                **{key: budget.get(key) for key in ("max_steps", "eval_every")},
                "n_fit_ids": report.get("n_fit_ids"),
                "n_val_items": report.get("n_val_items"),
                **{
                    key: report[key]
                    for key in ("val_items_max", "n_val_items_eligible")
                    if key in report
                },
                "exception": report.get("plan", {}).get("exception"),
                **{
                    key: fit.get(key) for key in ("steps", "best_step", "best_score", "stop_reason")
                },
                "nonfinite_losses": nonfinite_losses,
                **{
                    key: cumulative.get(key)
                    for key in (
                        "cold_covariate_columns",
                        "floored_covariate_columns",
                        "clipped_covariate_values",
                    )
                },
                "resumed_from_step": report.get("resumed_from_step"),
            }
        )
        timing.append(
            {
                "task": spec.name,
                **{
                    key: cumulative.get(key)
                    for key in (
                        "sessions",
                        "train_s",
                        "eval_s",
                        "checkpoint_s",
                        "nonfinite_losses",
                        "scaler_skips",
                    )
                },
                "peak_cuda_mem_MiB": cumulative["peak_cuda_mem_bytes"] / 2**20
                if _finite(cumulative.get("peak_cuda_mem_bytes"))
                else None,
                "amp_dtype": report.get("amp_dtype")
                or ("disabled" if "amp_dtype" in report else None),
            }
        )
        _required(budget, ("max_steps", "eval_every"), spec.name, "resolved_budget", issues)
        _required(report, ("n_fit_ids", "n_val_items"), spec.name, "run", issues)
        _required(
            cumulative,
            (
                "sessions",
                "train_s",
                "eval_s",
                "checkpoint_s",
                "peak_cuda_mem_bytes",
                "nonfinite_losses",
            ),
            spec.name,
            "timings.cumulative",
            issues,
        )
        _required(
            model,
            [
                *METRICS,
                "num_forecasts",
                "training_time_s",
                "inference_time_s",
                *[f"SQL[{q}]" for q in spec.quantile_levels],
            ],
            spec.name,
            "model",
            issues,
        )
        _required(reference, [*METRICS, "num_forecasts"], spec.name, "baseline", issues)
        actual_fp, forecast_count, averaged = None, None, {}
        try:
            task = spec.to_task()
            if not isinstance(spec.target, str):
                raise ValueError("report requires a univariate task")
            actual_fp = dataset_fingerprint(task)
            # Guard the whole iteration, including failures creating or advancing
            # task.iter_windows() outside the helper's per-window handler.
            window_rows, total, averaged, forecast_count = _window_evidence(
                task, task_dir, identity, issues
            )
            windows.extend(window_rows)
            diagnostics.append(total)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            issues.append(f"{spec.name}: task could not be audited: {exc}")
        fp_ok = actual_fp is not None and all(
            row.get("dataset_fingerprint") == actual_fp
            for row in (model, reference, report, manifest)
        )
        if not fp_ok:
            issues.append(f"{spec.name}: dataset fingerprint mismatch or unknown")
        count_ok = (
            forecast_count is not None
            and model.get("num_forecasts") == reference.get("num_forecasts") == forecast_count
        )
        if not count_ok:
            issues.append(f"{spec.name}: num_forecasts mismatch or unknown")
        comparable = (
            fp_ok
            and count_ok
            and identity_ok
            and _finite(model.get("SQL"))
            and _finite(reference.get("SQL"))
            and reference["SQL"] != 0
        )
        scores.append(
            {
                "task": spec.name,
                **{key: model.get(key) for key in (*METRICS, "num_forecasts")},
                "v2/baseline SQL ratio": model["SQL"] / reference["SQL"]
                if comparable
                else "not comparable",
                **{f"SQL[{q}]": model.get(f"SQL[{q}]") for q in spec.quantile_levels},
            }
        )
        knots.append(
            {
                "task": spec.name,
                **{f"pinball_{q}": averaged.get(f"pinball_{q}") for q in spec.quantile_levels},
            }
        )
        fingerprints.append(
            {
                "task": spec.name,
                "loaded_fingerprint": actual_fp,
                "model_fingerprint": model.get("dataset_fingerprint"),
                "baseline_fingerprint": reference.get("dataset_fingerprint"),
                "run_identity": identity,
            }
        )
    sizes.append(
        {
            "task": "TOTAL",
            **{
                key: sum(row[key] for row in sizes)
                for key in ("checkpoints_bytes", "bundle_bytes", "NPZ_bytes")
            },
        }
    )
    sections = [
        "# fev execution evidence\n",
        "num_forecasts counts item-window forecasts, not horizon cells. Scores and recomputed "
        "diagnostics use fev's equal-window averaging. Per-knot pinball is doubled, unscaled pinball "
        "in original units. best_score is inner SQL.\n",
        "Exclusions count item-window pairs with any non-finite scaled contribution on an observed label, "
        "including zero/undefined seasonal error. Missing labels are counted as excluded horizon cells. "
        "excluded_SQL_cells_knots counts all excluded (item, horizon, quantile) contributions; "
        "SQL averages finite cells per quantile, then averages quantiles. Forecasts are never dropped.\n",
        "Trainer elapsed_s is cumulative active train/eval/checkpoint time. checkpoint_s measures "
        "state serialization; its timing footer, atomic rename and progress publication are excluded. "
        "CUDA memory is the maximum allocated tensor memory across sessions; host RAM and sampled "
        "device-wide VRAM come from the external run logs (sampled maxima). Scaler skips count "
        "attempted fp16 optimizer steps skipped after overflow; legacy totals remain unknown.\n",
    ]
    for title, rows in (
        ("Training", training),
        ("Timings", timing),
        ("Scores", scores),
        ("Diagnostics", diagnostics),
        ("Doubled, unscaled pinball", knots),
        ("Window diagnostics", windows),
        ("Artifact sizes", sizes),
        ("Identities", fingerprints),
    ):
        sections.extend([f"## {title}\n", _table(rows)])
    sections.append("## Completeness\n")
    sections.append(
        "Incomplete\n\n" + "\n".join(f"- {issue}" for issue in issues)
        if issues
        else "Complete: every selected task has matching evidence and finite required metrics."
    )
    return "\n".join(sections) + "\n", issues


def _write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(report)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="weekly",
        help="weekly, heldout, comma-separated weekly names, or task YAML",
    )
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--baseline", default="seasonal_naive")
    parser.add_argument("--model-name", default="demand-forecast-v2")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # Even an empty selection requires the benchmark extra; no optional imports
    # at module scope, so the core package remains usable without fev.
    try:
        import fev  # noqa: F401
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc
    report, issues = build_report(
        args.run_dir, load_task_specs(args.tasks, args.profiles), args.baseline, args.model_name
    )
    _write_report(args.out, report)
    print(f"{'Incomplete' if issues else 'Complete'} report saved to {args.out}", flush=True)
    return int(bool(issues) and not args.allow_partial)


if __name__ == "__main__":
    raise SystemExit(main())
