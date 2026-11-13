"""Freeze the reviewed Simons folder inventory; never infer permission from tier."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import yaml

REPO = "abel-lab/Simons-Pretrain"
REVISION = "5f17ab8cfa3fa00a8e821599b631127ba250c342"
NOTICE = "Research-only use. Original per-source licences apply; no commercial clearance implied."
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/corpus-v4.yaml"


def safe_folder(value):
    """Accept only a single native dataset directory under the two reader roots."""
    path = PurePosixPath(value)
    if (
        len(path.parts) != 2
        or path.parts[0] not in {"univariate", "covariates"}
        or any(part in {".", "..", ""} for part in path.parts)
        or str(path) != value
        or "\\" in value
    ):
        raise ValueError(f"unsafe folder: {value!r}")
    return value


def freeze(draft):
    """Extract deterministic, compact provenance without changing overlap assessments."""
    folders = []
    for source in draft["sources"]:
        if source.get("repo_id") != REPO:
            continue
        if source["revision"] != REVISION:
            raise ValueError("draft revision differs from frozen Simons revision")
        rules = {r: source[r]["status"] for r in ("rule_f", "rule_l")}
        allowed = all(v == "no_identified_overlap" for v in rules.values())
        folders.append(
            dict(
                folder=safe_folder(source["config"]),
                status="allowed" if allowed else "quarantined",
                **rules,
                licence=source.get("licence", {}),
                frequency=source.get("frequency"),
                structure=source.get("structure", {}),
                lineage=source.get("lineage", {}).get("original_dataset", "unknown"),
                domain=source.get("domain", "unknown"),
                reason=source.get("reason", ""),
                uncertain=bool(source.get("uncertain", False)),
                uncertainty=source.get("uncertainty_reasons", []),
            )
        )
    result = dict(
        version="4.0",
        notice=NOTICE,
        repo_id=REPO,
        revision=REVISION,
        default_rule="L",
        folders=sorted(folders, key=lambda x: x["folder"]),
    )
    validate_manifest(result)
    return result


def validate_manifest(data):
    """Fail closed on duplicate paths, mutable revisions or unsupported allow rules."""
    if data.get("repo_id") != REPO or data.get("revision") != REVISION:
        raise ValueError("unapproved repo or revision")
    seen = set()
    for entry in data["folders"]:
        folder = safe_folder(entry["folder"])
        if folder in seen or entry["status"] not in {"allowed", "quarantined"}:
            raise ValueError(f"duplicate folder or invalid status: {folder}")
        seen.add(folder)
        if entry["status"] == "allowed" and any(
            entry[r] != "no_identified_overlap" for r in ("rule_f", "rule_l")
        ):
            raise ValueError(f"allowed folder has unresolved overlap rule: {folder}")
    if not seen:
        raise ValueError("empty manifest")
    return data


def read_manifest(path=DEFAULT_MANIFEST):
    """Read and validate a frozen manifest."""
    return validate_manifest(yaml.safe_load(Path(path).read_text()))


def main(argv=None):
    """Generate byte-stable YAML from the orchestrator's source analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    data = freeze(yaml.safe_load(args.draft.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"# {NOTICE}\n" + yaml.safe_dump(data, sort_keys=False, width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
