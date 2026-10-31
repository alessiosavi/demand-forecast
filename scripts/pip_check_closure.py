"""Record all pip check output; fail for requirements declared inside our closure."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path


def closure(root: str, extras: Sequence[str]) -> set[str]:
    """Return installed distributions reachable under each requested extra.

    Visit (distribution, extra) pairs so an extra reached later is still expanded,
    even in a cycle. Each dependency enables its own extras, not its parent's.
    """
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError as exc:
        raise ImportError("install packaging to run pip_check_closure.py") from exc

    root = canonicalize_name(root)
    pending = [(root, extra) for extra in ("", *map(canonicalize_name, extras))]
    visited = set()
    reached = set()
    while pending:
        name, extra = pending.pop()
        if (name, extra) in visited:
            continue
        visited.add((name, extra))
        try:
            requirements = importlib.metadata.requires(name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        reached.add(name)
        for text in requirements:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": extra}):
                continue
            dependency = canonicalize_name(requirement.name)
            pending.extend(
                (dependency, child_extra)
                for child_extra in ("", *map(canonicalize_name, requirement.extras))
            )
    return reached


def classify(lines: Iterable[str], closure: set[str]) -> tuple[list[str], list[str]]:
    """Split complaints by their declaring package (the first token)."""
    try:
        from packaging.utils import canonicalize_name
    except ImportError as exc:
        raise ImportError("install packaging to run pip_check_closure.py") from exc

    failures, recorded = [], []
    for line in lines:
        if not line.strip():
            continue
        destination = failures if canonicalize_name(line.split()[0]) in closure else recorded
        destination.append(line)
    return failures, recorded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", nargs="*", default=[], help="Root extras, e.g. bench")
    parser.add_argument("--report", type=Path, required=True, help="Raw pip check output")
    args = parser.parse_args(argv)
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(result.stdout + result.stderr)
    try:
        failures, recorded = classify(
            result.stdout.splitlines(), closure("demand-forecast", args.extras)
        )
    except (ImportError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for heading, lines in (("Failures (dependency closure)", failures), ("Recorded", recorded)):
        print(f"{heading}:")
        print("\n".join(lines) if lines else "(none)")

    # Exit 1 also covers pip execution/metadata errors, not just dependency issues.
    # Accept that status only for pip's known complaint formats, never an error
    # such as 'No module named pip' or a traceback.
    complaint = re.compile(
        r"\S+ \S+ (?:requires .+, which is not installed\."
        r"|has requirement .+, but you have .+\."
        r"|is not supported on this platform)"
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    pip_error = result.returncode not in (0, 1) or (
        result.returncode == 1
        and (
            result.stderr.strip()
            or not lines
            or any(not complaint.fullmatch(line) for line in lines)
        )
    )
    if pip_error:
        print(f"pip check failed to complete reliably; see {args.report}", file=sys.stderr)
    return int(bool(failures or pip_error))


if __name__ == "__main__":
    raise SystemExit(main())
