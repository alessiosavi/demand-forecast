"""Operator entry point for the Kaggle bench stages.

Builds the ``RunnerConfig`` for a competition from the pinned workspace
conventions and drives the shared runner:

- ``--slug`` selects the competition (default: store-sales).
- ``--stage`` runs one stage, ``all`` (default) or ``all-until-submit``.
- ``--dry-run`` prints the stage plan without executing anything (exit 0).

Evidence is written under
``.superpowers/sdd/2026-11-07-kaggle-forecasting-competitions/runs/<slug>/``.
See ``docs/kaggle.md`` for prerequisites and stage semantics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import cast

from demand_forecast.kaggle.runner import STAGES, RunnerConfig, Stage, plan_stages, run_stages
from demand_forecast.kaggle.store_sales import SLUG

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_ROOT = _REPO_ROOT / ".superpowers/sdd/2026-11-07-kaggle-forecasting-competitions/runs"
_ASSETS_DATASET = "alessiosavi/kb-assets"
_STAGE_CHOICES = ("all", *STAGES, "all-until-submit")


def _repo_sha() -> str:
    """The pushed repo commit the kernel driver pip-installs."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=_REPO_ROOT).strip()


def _config(slug: str) -> RunnerConfig:
    return RunnerConfig(
        slug=slug,
        evidence_root=_EVIDENCE_ROOT / slug,
        dataset_id=f"alessiosavi/kb-{slug}",
        kernel_id=f"alessiosavi/kb-{slug}-run",
        assets_dataset=_ASSETS_DATASET,
        repo_sha=_repo_sha(),
    )


def _command_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one Kaggle CLI command; ``run_stages`` inspects the return code."""
    return subprocess.run(argv, capture_output=True, text=True)


def _selected_stages(requested: str) -> tuple[Stage, ...]:
    if requested == "all":
        return STAGES
    if requested == "all-until-submit":
        return STAGES[:-1]
    return (cast(Stage, requested),)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=SLUG, help="competition slug (default: store-sales)")
    parser.add_argument("--stage", default="all", choices=_STAGE_CHOICES)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the stage plan and exit without running"
    )
    args = parser.parse_args(argv)

    config = _config(args.slug)
    stages = _selected_stages(args.stage)
    if args.dry_run:
        print(f"evidence root: {config.evidence_root}")
        for stage, commands in zip(STAGES, plan_stages(config)):
            if stage not in stages:
                continue
            print(f"{stage}: {' '.join(commands)}" if commands else f"{stage}: (direct call)")
        return 0

    try:
        run_stages(config, _command_runner, stages=stages)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"kaggle stage failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
