"""The documented variant table lists exactly the accepted variants, in order."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _accepted() -> tuple[str, ...]:
    # Load the module by path: the names must be readable without the benchmark's optional packages.
    path = ROOT / "demand_forecast" / "bench" / "yardsticks" / "v5_variants.py"
    spec = importlib.util.spec_from_file_location("v5_variants_for_docs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.VARIANT_NAMES)


def test_documented_variants_are_the_accepted_ones():
    text = (ROOT / "docs" / "yardsticks.md").read_text(encoding="utf-8")
    rows = text.split("| Variant | Scoring change |\n|---|---|\n", 1)[1].splitlines()
    documented = []
    for row in rows:
        match = re.match(r"\| `([a-z_]+)` \| \S", row)
        if match is None:
            break
        documented.append(match.group(1))
    assert tuple(documented) == _accepted()
