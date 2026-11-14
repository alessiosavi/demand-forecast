"""Average saved forecasts of completed yardstick runs and score them with fev.

Member identities do not hash this repository's evaluation code: callers must
ensure members were scored by the same evaluation code, including adapters and scorer. Cloud scoring
jobs record repo_sha in inputs.json beside the mode folders; the cloud route
will compare those SHAs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Keep argument parsing independent of optional benchmark dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--names", help="comma-separated task names or @file")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--allow-identical-members", action="store_true")
    args = parser.parse_args(argv)
    try:
        from demand_forecast.bench.yardsticks.ensemble import run_ensemble
        from demand_forecast.bench.yardsticks.runner import load_tasks, read_names

        tasks = load_tasks(args.tasks, read_names(args.names))
        frame = run_ensemble(
            tasks,
            args.members,
            args.model_name,
            args.out_dir,
            keep_going=args.keep_going,
            allow_identical_members=args.allow_identical_members,
        )
    except Exception as exc:
        print(f"Ensemble failed: {exc}", file=sys.stderr)
        return 2
    return int(bool(frame.attrs.get("failed_tasks")))


if __name__ == "__main__":
    raise SystemExit(main())
