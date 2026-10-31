#!/usr/bin/env python3
"""Compare exact Colab/EC2 pins for the executing interpreter's installed closure."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# The helper stays in the merged checkout even when a reference venv executes it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pip_check_closure import closure  # noqa: E402


def read_freeze(path: Path) -> dict[str, str | None]:
    """Read freeze pins without normalising versions (including their local tags).

    Direct references and conflicting pins have no unambiguous exact version.
    Keep their names for reporting; they cannot establish parity for a closure
    package, but are harmless outside it.
    """
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e ", "--editable ")):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        specifiers = list(requirement.specifier)
        version = None
        if len(specifiers) == 1:
            specifier = specifiers[0]
            if specifier.operator in {"==", "==="} and "*" not in specifier.version:
                version = specifier.version
        if name in pins and pins[name] != version:
            version = None
        pins[name] = version
    return pins


def table(
    names: set[str], colab: dict[str, str | None], ec2: dict[str, str | None], *, required: bool
) -> tuple[str, int]:
    rows = ["| Package | Colab | EC2 | Status |", "| --- | --- | --- | --- |"]
    failures = 0
    for name in sorted(names):
        issues = []
        for label, pins in (("Colab", colab), ("EC2", ec2)):
            if name not in pins:
                issues.append(f"missing-in-{label}")
            elif pins[name] is None:
                issues.append(f"unpinned-in-{label}")
        if not issues and colab[name] != ec2[name]:
            issues.append("version-mismatch")
        failures += bool(issues) and required
        status = ", ".join(issues) or "equal"
        if not required:
            status = f"informational: {status}"
        values = [pins.get(name, "(missing)") or "(unpinned)" for pins in (colab, ec2)]
        rows.append(f"| {name} | {values[0]} | {values[1]} | {status} |")
    if not names:
        rows.append("| (none) | | | |")
    return "\n".join(rows), failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colab-lock", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--extras", nargs="*", default=[], help="Root extras, e.g. bench")
    parser.add_argument("--out", type=Path, required=True, help="Parity report (also printed)")
    args = parser.parse_args(argv)

    try:
        reached = {canonicalize_name(name) for name in closure("demand-forecast", args.extras)}
        if "demand-forecast" not in reached:
            raise ValueError("demand-forecast is not installed in the executing interpreter")
        required = reached - {"demand-forecast"}
        colab, ec2 = read_freeze(args.colab_lock), read_freeze(args.freeze)
        compared, failures = table(required, colab, ec2, required=True)
        outside, _ = table((set(colab) | set(ec2)) - reached, colab, ec2, required=False)
        report = (
            f"Closure parity: {'FAIL' if failures else 'PASS'} "
            f"({len(required)} packages, {failures} failures)\n\n"
            f"Closure packages (demand-forecast itself excluded):\n{compared}\n\n"
            f"Outside the closure (informational only):\n{outside}\n"
        )
    except (OSError, ValueError, ImportError) as exc:
        report = f"Closure parity: FAIL\n{exc}\n"
        failures = 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report, end="")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
