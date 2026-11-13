"""python -m demand_forecast.bench.adaptation --tasks ... --task ... --output ..."""

import argparse
from pathlib import Path

from .data import PHASES, build_task_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one cut-off-bounded fev adaptation bundle")
    parser.add_argument("--tasks", required=True, type=Path, help="Native fev task YAML")
    parser.add_argument("--task", required=True, help="Exact task name (slashes are supported)")
    parser.add_argument("--output", required=True, type=Path, help="New bundle directory")
    parser.add_argument("--phase", choices=PHASES, default="development")
    parser.add_argument("--max-context", type=int, help="Explicit scoring-adapter context override")
    args = parser.parse_args()
    manifest = build_task_manifest(
        args.tasks, args.task, args.output, phase=args.phase, max_context=args.max_context
    )
    print(f"{args.output / 'manifest.json'} {manifest['fingerprint']}")


if __name__ == "__main__":
    main()
