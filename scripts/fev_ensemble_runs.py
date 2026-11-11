"""Average saved quantile forecasts and evaluate them with the fev runner scorer."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zlib
from dataclasses import fields
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from demand_forecast.bench.run import _save_summaries, assemble_predictions, score_predictions
from demand_forecast.bench.specs import FevTaskSpec


def task_from_row(row: dict):
    """Recover fev's serialized evaluation definition, including list-valued fields."""
    import fev

    values = {}
    for field in fields(fev.Task):
        if field.name not in row:
            raise ValueError(f"summary missing task definition field: {field.name}")
        value = row[field.name]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = None
        elif isinstance(value, str) and value.startswith(("[", "{")):
            value = ast.literal_eval(value)
        elif field.name in {"initial_cutoff", "window_step_size"} and isinstance(value, str):
            # Mixed timestamp/integer columns are read from CSV as strings.
            try:
                value = int(value)
            except ValueError:
                pass
        values[field.name] = value
    return fev.Task(**values)


def _read_run(directory: Path) -> dict[str, dict]:
    frame = pd.read_csv(directory / "summaries.csv", float_precision="round_trip")
    if not {"task_name", "model_name", "test_error"} <= set(frame) or frame.empty:
        raise ValueError(f"{directory}: empty summary or missing fev columns")
    if frame.task_name.isna().any() or frame.task_name.duplicated().any():
        raise ValueError(f"{directory}: expected exactly one row per task")
    rows = {}
    for row in frame.to_dict("records"):
        name = row["task_name"]
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or any(part in {".", ".."} for part in name.split("/"))
        ):
            raise ValueError(f"{directory}: unsafe task name: {name!r}")
        if "task" in row and row["task"] != name:
            raise ValueError(f"{directory}: conflicting task and task_name")
        rows[name] = row
    return rows


def _window_paths(directory: Path, count: int) -> list[Path]:
    expected = [directory / f"window_{i}.npz" for i in range(count)]
    if set(directory.glob("window_*.npz")) != set(expected):
        raise ValueError(f"{directory}: window counts or indices differ")
    return expected


def ensemble_runs(runs: list[Path], out: Path, model_name: str) -> pd.DataFrame:
    """Reject incompatible evidence before publishing a fresh, independently scored run."""
    if not runs or not model_name:
        raise ValueError("runs and model name must be non-empty")
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError(f"output must be a new or empty directory: {out}")
    if any(out.resolve() == path.resolve() for path in runs):
        raise ValueError("output must differ from every input run")
    rows = [_read_run(Path(path)) for path in runs]
    names = set(rows[0])
    if any(set(run) != names for run in rows[1:]):
        raise ValueError("run task sets differ")
    # Check all definitions and window counts before any scoring or output.
    tasks = {}
    for name in sorted(names):
        task = task_from_row(rows[0][name])
        for directory, run in zip(runs, rows, strict=True):
            other = task_from_row(run[name])
            if other.quantile_levels != task.quantile_levels:
                raise ValueError(f"{name}: quantile levels differ")
            if other.num_windows != task.num_windows:
                raise ValueError(f"{name}: window counts differ")
            if other.to_dict() != task.to_dict():
                raise ValueError(f"{name}: evaluation task definitions differ")
            if run[name].get("dataset_fingerprint") != rows[0][name].get("dataset_fingerprint"):
                raise ValueError(f"{name}: dataset fingerprints differ")
            _window_paths(directory / name, task.num_windows)
        tasks[name] = task

    summaries = []
    for name, task in tasks.items():
        spec = FevTaskSpec(
            name=name,
            dataset_path=task.dataset_path,
            horizon=task.horizon,
            num_windows=task.num_windows,
            quantile_levels=task.quantile_levels,
            target=task.target,
            id_column=task.id_column,
            target_support="real",
        )
        predictions, averaged = [], []
        sources = []
        for index in range(task.num_windows):
            arrays, identities, ids = [], [], None
            for directory in runs:
                path = directory / name / f"window_{index}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    if not {"ids", "arrays", "identity"} <= set(saved.files):
                        raise ValueError(f"{path}: missing ids, arrays or identity")
                    current_ids, current = saved["ids"], saved["arrays"]
                    if "quantile_levels" in saved and not np.array_equal(
                        saved["quantile_levels"], task.quantile_levels
                    ):
                        raise ValueError(f"{path}: quantile levels differ from summary")
                    if ids is not None and not np.array_equal(ids, current_ids):
                        raise ValueError(f"{name} window {index}: series ids differ")
                    ids = current_ids
                    if (
                        ids.ndim != 1
                        or len(set(ids.tolist())) != len(ids)
                        or current.shape != (len(ids), task.horizon, len(task.quantile_levels))
                        or not np.issubdtype(current.dtype, np.floating)
                        or not np.isfinite(current).all()
                    ):
                        raise ValueError(f"{path}: invalid ids, shapes or non-finite forecasts")
                    arrays.append(current)
                    identities.append(str(saved["identity"].item()))
            mean = np.sort(np.mean(np.stack(arrays), axis=0), axis=-1)
            if not np.isfinite(mean).all():
                raise ValueError(f"{name} window {index}: non-finite ensemble")
            past, _ = task.get_window(index).get_input_data()
            predictions.append(assemble_predictions(spec, past, ids.tolist(), mean))
            averaged.append((ids, mean))
            sources.append(identities)
        identity = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
        summary = {
            "task": name,
            **score_predictions(
                task,
                predictions,
                model_name=model_name,
                trained_on_this_dataset=any(
                    str(run[name].get("trained_on_this_dataset", "")).lower() == "true"
                    for run in rows
                ),
                extra_info={"run_identity": identity, "ensemble_size": len(runs)},
            ),
        }
        task_dir = out / name
        task_dir.mkdir(parents=True, exist_ok=True)
        for index, (ids, arrays) in enumerate(averaged):
            np.savez(
                task_dir / f"window_{index}.npz",
                ids=ids,
                arrays=arrays,
                identity=identity,
                quantile_levels=task.quantile_levels,
            )
        summaries.append(summary)
    out.mkdir(parents=True, exist_ok=True)
    _save_summaries(pd.DataFrame(summaries), out / "summaries.csv")
    (out / "ensemble.json").write_text(
        json.dumps({"runs": [str(path.resolve()) for path in runs]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return pd.read_csv(out / "summaries.csv", float_precision="round_trip")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args(argv)
    try:
        ensemble_runs(args.runs, args.out, args.model_name)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
        AssertionError,
        SyntaxError,
        RuntimeError,
        EOFError,
        BadZipFile,
        zlib.error,
    ) as exc:
        print(f"Ensemble refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
