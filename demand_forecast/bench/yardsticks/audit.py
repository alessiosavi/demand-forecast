"""Recompute fev scores from yardstick artifacts without requiring training files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.bench.run import score_predictions
from demand_forecast.bench.specs import dataset_fingerprint

from .runner import (
    Options,
    digest,
    fev_predictions,
    file_digest,
    prepare_windows,
    read_window,
)


def audit_report(run_dir: Path, specs, model_name: str) -> tuple[str, list[str]]:
    """Verify configurations, item/target coverage, checksums, timing and scores."""
    issues = []
    path = run_dir / "summaries.csv"
    frame = pd.read_csv(path, float_precision="round_trip") if path.exists() else pd.DataFrame()
    if not {"task", "model_name"} <= set(frame.columns):
        issues.append("summaries.csv: missing or missing task/model_name columns")
        frame = pd.DataFrame(columns=["task", "model_name"])
    lines = ["# fev yardstick evidence", "", f"Model: {model_name}", ""]
    for spec in specs:
        name = spec.name if hasattr(spec, "name") else spec.task_name
        try:
            rows = frame.loc[(frame.task == name) & (frame.model_name == model_name)]
            if len(rows) != 1:
                raise ValueError("expected exactly one model row")
            row = rows.iloc[0]
            directory = run_dir / name
            record = json.loads((directory / "run.json").read_text())
            # Keep configuration beside task evidence too, so merge_runs links
            # remain self-contained when the root manifest is not linked.
            config = record["configuration"]
            config_id = config["configuration_identity"]
            if (
                digest({k: v for k, v in config.items() if k != "configuration_identity"})
                != config_id
                or row.configuration_identity != config_id
            ):
                raise ValueError("configuration identity mismatch")
            task = spec.to_task() if hasattr(spec, "to_task") else spec
            payload = {
                "configuration_identity": config_id,
                "task": task.to_dict(),
                "dataset_fingerprint": dataset_fingerprint(task),
            }
            identity = digest(payload)
            if (
                row.run_identity != identity
                or record["run_identity"] != identity
                or record["task"] != payload["task"]
                or row.dataset_fingerprint != payload["dataset_fingerprint"]
            ):
                raise ValueError("task/dataset/run identity mismatch")
            options = Options(**config["options"])
            for key, value in config["options"].items():
                actual = row[key]
                if (value is None and not pd.isna(actual)) or (
                    value is not None and actual != value
                ):
                    raise ValueError(f"{key}: configuration column mismatch")
            windows = prepare_windows(task, options)
            predictions, elapsed = [], 0.0
            for i, window in enumerate(windows):
                path = directory / f"window_{i}.npz"
                if file_digest(path) != record["window_sha256"][path.name]:
                    raise ValueError(f"{path.name}: checksum mismatch")
                arrays, duration = read_window(
                    path, window, record["levels"], identity, task.horizon
                )
                elapsed += duration
                predictions.append(fev_predictions(arrays, record["levels"], window.targets))
            recomputed = score_predictions(
                task,
                predictions,
                model_name=model_name,
                training_time_s=0.0,
                inference_time_s=elapsed,
                trained_on_this_dataset=False,
            )
            for key, value in recomputed.items():
                if isinstance(value, (int, float, bool)):
                    if key not in row or not np.isclose(
                        float(row[key]), float(value), rtol=1e-9, atol=1e-12, equal_nan=False
                    ):
                        issues.append(f"{name}: {key} mismatch or non-finite")
                elif key in ("fev_version", "dataset_fingerprint") and row[key] != value:
                    issues.append(f"{name}: {key} mismatch")
            lines.append(
                f"- {name}: SQL={recomputed.get('SQL')}; {recomputed['num_forecasts']} item-target-window forecasts; weights={options.weights_revision}"
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"{name}: {exc}")
    lines += [
        "",
        "## Completeness",
        "",
        "Incomplete"
        if issues
        else "Complete: every selected task has matching evidence and recomputed scores.",
    ]
    lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines) + "\n", issues
