"""Evaluate a pretrained reference model on fev YAML (run only on the benchmark host)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Parse switches before importing optional benchmark or inference packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["chronos2", "timestral"], required=True)
    parser.add_argument(
        "--weights", required=True, help="Hugging Face model ID or local checkpoint directory"
    )
    parser.add_argument(
        "--revision", help="Hub commit or alias; resolved and recorded before inference"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--names", help="comma-separated task names or @file")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--covariates", choices=["all", "none"], default="all")
    parser.add_argument(
        "--cross-series",
        choices=["on", "off"],
        default="on",
        help="Chronos-2 only: off isolates items and target variates",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Chronos variate budget (100); Timestral item limit (1024)"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)

    from demand_forecast.bench.yardsticks import Options, load_tasks, run_yardstick
    from demand_forecast.bench.yardsticks.runner import read_names

    try:
        options = Options(
            **{
                name: getattr(args, name)
                for name in (
                    "model",
                    "weights",
                    "revision",
                    "model_name",
                    "max_context",
                    "covariates",
                    "cross_series",
                    "batch_size",
                    "device",
                )
            }
        )
        tasks = load_tasks(args.tasks, read_names(args.names))
        frame = run_yardstick(
            tasks, options, args.out_dir, resume=args.resume, keep_going=args.keep_going
        )
    except (ImportError, OSError, ValueError) as exc:
        print(f"Yardstick failed: {exc}", file=sys.stderr)
        return 2
    selected = frame.loc[frame.model_name == args.model_name]
    return int(
        bool(frame.attrs.get("failed_tasks"))
        or not {t.task_name for t in tasks} <= set(selected.task)
    )


if __name__ == "__main__":
    raise SystemExit(main())
