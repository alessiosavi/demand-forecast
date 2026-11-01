"""Profile fev tasks without fitting models; resume completed tasks with --resume."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from demand_forecast.bench.specs import FevTaskSpec, load_task_specs

PROFILE_COLUMNS = [
    "task",
    "n_series",
    "length_min",
    "length_median",
    "length_max",
    "anchor_weekday",
    "nan_share",
    "negative_share",
    "zero_share",
    "zero_scale_items",
    "eligible_items_first_window",
    "static_nan_share",
    "distinct_origins_first_window",
]


def _targets(row: dict, columns: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(row[column], dtype=float) for column in columns])


def _seasonal_scale(target: np.ndarray, seasonality: int) -> float:
    """Mean absolute lag-m difference over observed pairs, without a scale floor."""
    with np.errstate(invalid="ignore"):
        differences = np.abs(target[seasonality:] - target[:-seasonality])
    observed = differences[~np.isnan(differences)]
    return float(observed.mean()) if observed.size else float("nan")


def profile_task(spec: FevTaskSpec) -> dict[str, int | float]:
    """Describe full histories, scale eligibility and first-window origin diversity.

    Target shares include missing values in their denominator. Scale calculations
    omit missing lag pairs; multivariate targets pool pairs within each item.
    These full-data diagnostics must never be used for model selection.
    """
    task = spec.to_task()
    dataset = task.load_full_dataset(num_proc=1)
    # Plain Python values preserve null static strings (the numpy formatter can
    # convert them to text). Read only the columns needed by the diagnostics.
    data = dataset.select_columns(
        [task.timestamp_column, *task.target_columns, *task.static_columns]
    ).with_format(None)
    lengths = []
    total = missing = negative = zeros = zero_scale = static_missing = 0
    for row in data:
        target = _targets(row, task.target_columns)
        lengths.append(len(target))
        total += target.size
        missing += int(np.isnan(target).sum())
        negative += int((target < 0).sum())
        zeros += int((target == 0).sum())
        scale = _seasonal_scale(target, task.seasonality)
        zero_scale += int(not np.isfinite(scale) or scale == 0)
        static_missing += sum(bool(pd.isna(row[column])) for column in task.static_columns)

    eligible = 0
    distinct_origins = 0
    try:
        past, future = task.get_window(0).get_input_data()
    except ValueError as exc:
        # fev rejects an entirely empty window instead of returning an empty
        # dataset. It still has a meaningful diagnostic count of zero.
        if str(exc) != (
            "All time series in the dataset are too short for the chosen cutoff, "
            "horizon and min_context_length"
        ):
            raise
    else:
        distinct_origins = len(
            {
                pd.Timestamp(row[task.timestamp_column][0])
                for row in future.select_columns([task.timestamp_column]).with_format(None)
            }
        )
        for row in past.select_columns(task.target_columns).with_format(None):
            target = _targets(row, task.target_columns)
            scale = _seasonal_scale(target, task.seasonality)
            eligible += int(
                len(target) >= task.min_context_length and np.isfinite(scale) and scale > 0
            )

    static_total = len(data) * len(task.static_columns)
    return {
        "n_series": len(data),
        "length_min": min(lengths),
        "length_median": float(np.median(lengths)),
        "length_max": max(lengths),
        "anchor_weekday": pd.Timestamp(data[0][task.timestamp_column][0]).weekday(),
        "nan_share": missing / total,
        "negative_share": negative / total,
        "zero_share": zeros / total,
        "zero_scale_items": zero_scale,
        "eligible_items_first_window": eligible,
        "static_nan_share": static_missing / static_total if static_total else 0.0,
        "distinct_origins_first_window": distinct_origins,
    }


def _save_profiles(frame: pd.DataFrame, path: Path) -> None:
    """Keep the last completed CSV intact if the next task is interrupted."""
    with NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="weekly",
        help="weekly, heldout, comma-separated weekly names, or task YAML",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output profile CSV")
    parser.add_argument("--profiles", type=Path, help="Override task role profiles YAML")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task names already saved; use a new output for changed definitions or data",
    )
    args = parser.parse_args(argv)
    specs = load_task_specs(args.tasks, args.profiles)
    frame = pd.DataFrame(columns=PROFILE_COLUMNS)
    if args.resume and args.out.exists():
        frame = pd.read_csv(args.out, float_precision="round_trip")
        if list(frame.columns) != PROFILE_COLUMNS or frame["task"].duplicated().any():
            raise ValueError("invalid profile CSV: expected profile columns and unique task names")
    completed = set(frame["task"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if spec.name in completed:
            print(f"Skipping completed task {spec.name}", flush=True)
            continue
        print(f"Profiling {spec.name}", flush=True)
        row = pd.DataFrame([{"task": spec.name, **profile_task(spec)}], columns=PROFILE_COLUMNS)
        frame = row if frame.empty else pd.concat([frame, row], ignore_index=True)
        _save_profiles(frame, args.out)
        completed.add(spec.name)
    if not args.out.exists():
        _save_profiles(frame, args.out)
    print(f"Profiles saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
