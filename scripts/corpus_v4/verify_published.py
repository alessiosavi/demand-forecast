"""Verify an already published corpus offline against the checked-in Hub inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from scripts.corpus_v4.fetch_native import HubFile, inventory, verify_file
from scripts.corpus_v4.freeze_manifest import DEFAULT_MANIFEST, read_manifest, validate_manifest
from scripts.corpus_v4.native import atomic_json, digest, folder_path


class PinnedInventory:
    """Read-only inventory adapter: no Hub client or data download path exists."""

    def __init__(self, repo_id, revision, files):
        self.repo_id, self.revision, self.files = repo_id, revision, files

    def inventory(self, repo_id, revision):
        if (repo_id, revision) != (self.repo_id, self.revision):
            raise ValueError("published corpus differs from pinned inventory revision")
        return self.revision, [HubFile(**f) for f in self.files]


def pinned_inventory():
    """Metadata recorded for 5f17ab8; this is not a new network lookup."""
    return PinnedInventory(
        **json.loads(Path(__file__).with_name("pinned_inventory.json").read_text())
    )


def verify_published(root, manifest, *, pinned_files=None):
    """Reject missing/extra files, stale policy, incomplete receipts and wrong bytes."""
    validate_manifest(manifest)
    root = Path(root)
    hub = (
        pinned_inventory()
        if pinned_files is None
        else PinnedInventory(manifest["repo_id"], manifest["revision"], pinned_files)
    )
    expected = inventory(manifest, hub)
    identity = dict(
        repo_id=manifest["repo_id"], revision=manifest["revision"], folders=manifest["folders"]
    )
    marker = root / "corpus-identity.json"
    if marker.is_symlink() or json.loads(marker.read_text()) != identity:
        raise ValueError("published corpus identity differs from manifest")
    permitted = set()
    receipts = {}
    for entry in manifest["folders"]:
        path = folder_path(root, entry)
        permitted.add(path.relative_to(root).as_posix())
        receipt_path, stats = path / "receipt.json", path / "stats.json"
        if receipt_path.is_symlink() or stats.is_symlink():
            raise ValueError(f"symlink receipt/stats: {entry['folder']}")
        receipt = json.loads(receipt_path.read_text())
        recorded = dict(
            repo_id=manifest["repo_id"],
            revision=manifest["revision"],
            folder=entry["folder"],
            status=entry["status"],
            files=[asdict(f) for f in expected[entry["folder"]]],
        )
        if {k: receipt.get(k) for k in recorded} != recorded:
            raise ValueError(f"receipt differs from pinned inventory/manifest: {entry['folder']}")
        names = {Path(f.path).name for f in expected[entry["folder"]]}
        if {p.name for p in path.iterdir()} != names | {"receipt.json", "stats.json"}:
            raise ValueError(f"incomplete or extra published files: {entry['folder']}")
        for file in expected[entry["folder"]]:
            verify_file(path / Path(file.path).name, file)
        if receipt.get("stats_sha256") != digest(stats):
            raise ValueError(f"published stats hash mismatch: {entry['folder']}")
        receipts[entry["folder"]] = receipt
    for pattern in (
        "univariate/*",
        "covariates/*",
        "quarantine/univariate/*",
        "quarantine/covariates/*",
    ):
        for path in root.glob(pattern):
            if path.relative_to(root).as_posix() not in permitted:
                raise ValueError(f"unlisted or misplaced corpus folder: {path}")
    fingerprint = hashlib.sha256(
        json.dumps(dict(identity=identity, receipts=receipts), sort_keys=True).encode()
    ).hexdigest()
    return dict(
        repo_id=manifest["repo_id"],
        revision=manifest["revision"],
        folders=len(expected),
        files=sum(len(files) for files in expected.values()),
        receipt_fingerprint=fingerprint,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-status", type=Path, required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    verified = verify_published(args.root, read_manifest(args.manifest))
    source = json.loads(args.source_status.read_text())
    fetch_id = source.get("fetch_attempt_id") or source.get("attempt_id")
    if (
        not isinstance(fetch_id, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", fetch_id)
        or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("repo_sha", "")))
    ):
        raise ValueError("published corpus has no source fetch attempt identity")
    result = dict(
        **verified,
        source_prefix=args.source_prefix,
        fetch_attempt_id=fetch_id,
        source_repo_sha=source["repo_sha"],
    )
    if args.provenance.exists() and json.loads(args.provenance.read_text()) != result:
        raise ValueError("restored audit provenance differs from verified corpus/fetch attempt")
    atomic_json(args.provenance, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
