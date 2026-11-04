"""Save validation history before pruning completed tasks' training checkpoints."""

import argparse
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .sources import _atomic_write


def _read(path):
    return json.loads(path.read_text())


def _history(state):
    result = {key: state[key] for key in ("history", "best_step", "best_score", "max_steps")}
    if not isinstance(result["history"], list):
        raise ValueError("history must be a list")

    # Trainer checkpoints may contain NaN diagnostics; JSON records them as null.
    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    return finite(result)


def _finalize_task(directory, summaries, dry_run):
    report_path = directory / "run.json"
    report = _read(report_path) if report_path.exists() else {}
    identity = report.get("run_identity")
    latest = directory / "checkpoints/latest.pt"
    history_path = directory / "history.json"
    history = None
    if latest.exists():
        import torch

        # Trusted files produced by our own trainer only: pickle is required for
        # the optimizer/RNG state. Never use this command on untrusted .pt files.
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if identity is not None and state.get("run_identity") != identity:
            raise ValueError("checkpoint/run identity mismatch")
        history = _history(state)
        if not dry_run:
            _atomic_write(
                history_path, (json.dumps(history, indent=2, allow_nan=False) + "\n").encode()
            )
    elif history_path.exists():
        history = _history(_read(history_path))
    reasons = []
    rows = summaries.loc[summaries.task == directory.name]
    if len(rows) != 1:
        reasons.append("requires one summaries.csv row")
    elif identity is None or rows.iloc[0].get("run_identity") != identity:
        raise ValueError("summary/run identity mismatch or missing identity")
    manifest_path = directory / "bundle/manifest.json"
    if not manifest_path.exists():
        reasons.append("bundle/manifest.json missing")
    elif identity is None or _read(manifest_path).get("run_identity") != identity:
        raise ValueError("bundle/run identity mismatch or missing identity")
    count = report.get("task_definition", {}).get("num_windows")
    if not isinstance(count, int) or count <= 0:
        reasons.append("run.json num_windows missing/invalid")
    else:
        expected = {f"window_{i}.npz" for i in range(count)}
        present = {p.name for p in directory.glob("window_*.npz")}
        if present != expected:
            reasons.append(f"window set incomplete/extra ({len(present)}/{count})")
        for name in sorted(present & expected):
            with np.load(directory / name, allow_pickle=False) as saved:
                if identity is None or saved["identity"].item() != identity:
                    raise ValueError(f"{name}: prediction/run identity mismatch")
                ids, arrays = saved["ids"].tolist(), saved["arrays"]
                if (
                    arrays.ndim != 3
                    or len(ids) != arrays.shape[0]
                    or len(set(ids)) != len(ids)
                    or not np.isfinite(arrays).all()
                ):
                    raise ValueError(f"{name}: invalid predictions")
    if history is None:
        reasons.append("history.json unavailable")
    if history is not None and report:
        for key in ("best_step", "best_score"):
            if history[key] != report.get("fit_result", {}).get(key):
                raise ValueError(f"history/run {key} mismatch")
        if history["max_steps"] != report.get("resolved_budget", {}).get("max_steps"):
            raise ValueError("history/run max_steps mismatch")
    checkpoints = sorted((directory / "checkpoints").glob("*.pt"))
    if reasons:
        return {
            "task": directory.name,
            "status": "incomplete",
            "checkpoints": len(checkpoints),
            "detail": "; ".join(reasons),
        }
    if not dry_run:
        if not history_path.is_file():
            raise ValueError("history.json was not written; refusing prune")
        for checkpoint in checkpoints:
            checkpoint.unlink()
    return {
        "task": directory.name,
        "status": "would-prune" if dry_run and checkpoints else "complete",
        "checkpoints": len(checkpoints),
        "detail": "history saved" if not dry_run else "no files changed",
    }


def finalize(run_dir, dry_run=False):
    run_dir = Path(run_dir)
    summary_path = run_dir / "summaries.csv"
    summaries = (
        pd.read_csv(summary_path, float_precision="round_trip")
        if summary_path.exists()
        else pd.DataFrame(columns=["task"])
    )
    if "task" not in summaries:
        raise ValueError("summaries.csv has no task column")
    directories = {
        p
        for p in run_dir.iterdir()
        if p.is_dir() and ((p / "run.json").exists() or (p / "checkpoints").is_dir())
    }
    for name in summaries.task:
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid task directory name: {name!r}")
        directories.add(run_dir / name)
    if not directories:
        raise ValueError("No task directories found")
    result = []
    for directory in sorted(directories):
        try:
            if directory.is_symlink():
                raise ValueError("Refusing to finalize a symlink task directory")
            result.append(_finalize_task(directory, summaries, dry_run))
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            EOFError,
            pickle.UnpicklingError,
            zipfile.BadZipFile,
        ) as exc:
            result.append(
                {
                    "task": directory.name,
                    "status": "inconsistent",
                    "checkpoints": None,
                    "detail": str(exc),
                }
            )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("finalize")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows = finalize(args.run_dir, args.dry_run)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Finalization failed: {exc}\n")
    print(pd.DataFrame(rows).to_string(index=False))
    if any(row["status"] in {"incomplete", "inconsistent"} for row in rows):
        parser.exit(1)


if __name__ == "__main__":
    main()
