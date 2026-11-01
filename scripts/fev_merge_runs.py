"""Merge isolated fev task runs into ROOT/merged without changing their evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from demand_forecast.bench.run import _save_summaries
from demand_forecast.bench.specs import load_task_specs


def merge_runs(root: Path, names: list[str], allow_partial: bool = False) -> Path:
    """Validate all inputs before publishing a CSV and relative artifact links."""
    root = Path(root)
    out = root / "merged"
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError(f"output already exists and is non-empty or unsafe: {out}")
    if not names or len(set(names)) != len(names):
        raise ValueError("task selection must be non-empty and unique")
    frames, present = [], []
    for name in names:
        if name in (".", "..", "merged") or Path(name).name != name or not name:
            raise ValueError(f"unsafe task name: {name!r}")
        path = root / name / "summaries.csv"
        if not path.exists():
            if allow_partial:
                continue
            raise ValueError(f"missing task: {name} ({path})")
        frame = pd.read_csv(path, float_precision="round_trip")
        if not {"task", "model_name"} <= set(frame.columns) or frame.empty:
            raise ValueError(f"{name}: empty summary or missing task/model_name columns")
        if not frame.task.eq(name).all():
            raise ValueError(f"{name}: foreign-task rows")
        if frame.model_name.isna().any():
            raise ValueError(f"{name}: missing model_name")
        if frame.duplicated(["task", "model_name"]).any():
            raise ValueError(f"{name}: duplicate (task, model_name) rows")
        frames.append(frame)
        present.append(name)
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["task", "model_name"])
    )
    out.mkdir(parents=True, exist_ok=True)
    # Publish summary last: readers never observe a completed CSV before its links.
    for name in present:
        (out / name).symlink_to(Path("..") / name / name, target_is_directory=True)
    _save_summaries(combined, out / "summaries.csv")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tasks", default="weekly")
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        specs = load_task_specs(args.tasks, args.profiles)
        out = merge_runs(args.run_root, [spec.name for spec in specs], args.allow_partial)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Merge refused: {exc}", file=sys.stderr)
        return 2
    print(f"Merged evidence: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
