"""Fetch pinned native shards, verify Hub digests, validate and measure every folder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from scripts.corpus_v4.freeze_manifest import DEFAULT_MANIFEST, read_manifest, validate_manifest
from scripts.corpus_v4.native import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    MEASUREMENT_VERSION,
    atomic_json,
    digest,
    folder_path,
    measure_folder,
    memory_budget_arg,
)

SIDECAR_NAMES = {"LICENSE.md", "articles.csv"}


@dataclass(frozen=True)
class HubFile:
    """Hub-recorded content identity (LFS SHA256 or Git blob SHA1)."""

    path: str
    size: int
    hash: str
    algorithm: str


class Hub(Protocol):
    """Only production boundary with the public Hub; offline fakes implement this."""

    def inventory(self, repo_id: str, revision: str) -> tuple[str, list[HubFile]]:
        """Return resolved revision and all files under the two native data roots."""
        ...

    def download(self, repo_id: str, revision: str, file: HubFile, destination: Path):
        """Download only the specified pinned file to the staging path."""
        ...


class PublicHub:
    """Anonymous, revision-pinned Hub access; no credential is required or embedded."""

    def inventory(self, repo_id, revision):
        from huggingface_hub import HfApi

        info = HfApi(token=False).dataset_info(repo_id, revision=revision, files_metadata=True)
        files = []
        for item in info.siblings:
            if not item.rfilename.startswith(("univariate/", "covariates/")):
                continue
            # Keep side files and unknown paths: inventory validates the complete
            # tree against the manifest and reports every offending path together.
            lfs = item.lfs
            files.append(
                HubFile(
                    item.rfilename,
                    lfs.size if lfs else item.size,
                    lfs.sha256 if lfs else item.blob_id,
                    "sha256" if lfs else "git-sha1",
                )
            )
        return info.sha, files

    def download(self, repo_id, revision, file, destination):
        from huggingface_hub import hf_hub_download

        # local_dir writes regular files plus small transfer metadata, no Hub blobs/symlinks.
        downloaded = hf_hub_download(
            repo_id,
            file.path,
            repo_type="dataset",
            revision=revision,
            token=False,
            local_dir=str(destination.parent / ".hub-local"),
        )
        downloaded = Path(downloaded)
        if downloaded.is_symlink():
            raise ValueError(f"Hub returned a symlink: {file.path}")
        downloaded.replace(destination)


def verify_file(path, file):
    """Refuse missing metadata, wrong size and wrong digest, including on restart."""
    path = Path(path)
    if file.algorithm not in {"sha256", "git-sha1"} or not file.hash or file.size is None:
        raise ValueError(f"missing Hub hash metadata: {file.path}")
    if path.is_symlink() or path.stat().st_size != file.size:
        raise ValueError(f"size mismatch: {file.path}")
    if digest(path, file.algorithm) != file.hash:
        raise ValueError(f"hash mismatch: {file.path}")


def inventory(manifest, hub):
    """Resolve and cross-check every folder before making any download."""
    validate_manifest(manifest)
    revision, files = hub.inventory(manifest["repo_id"], manifest["revision"])
    if revision != manifest["revision"]:
        raise ValueError("Hub resolved another revision")
    by_folder = {entry["folder"]: [] for entry in manifest["folders"]}
    seen = set()
    unexpected = []
    for file in sorted(files, key=lambda f: f.path):
        path = PurePosixPath(file.path)
        folder = str(path.parent)
        if (
            folder not in by_folder
            or len(path.parts) != 3
            or ".." in path.parts
            or str(path) != file.path
            or "\\" in file.path
            or file.path in seen
            or not (
                path.name in SIDECAR_NAMES
                or (path.name.startswith(("train-", "validation-")) and path.suffix == ".parquet")
            )
        ):
            unexpected.append(file.path)
            continue
        seen.add(file.path)
        by_folder[folder].append(file)
    if unexpected:
        raise ValueError("unexpected native tree files or folders:\n" + "\n".join(unexpected))
    missing = [
        folder
        for folder, files in by_folder.items()
        if not any(
            PurePosixPath(file.path).name.startswith("train-") and file.path.endswith(".parquet")
            for file in files
        )
    ]
    if missing:
        raise ValueError(
            "manifest folders missing train shards from pinned inventory:\n" + "\n".join(missing)
        )
    return by_folder


def fetch(manifest, root, hub, *, state, memory_budget_bytes=DEFAULT_MEMORY_BUDGET_BYTES):
    """Publish a folder atomically only after all hashes and native rows validate."""
    root, state = Path(root), Path(state)
    files = inventory(manifest, hub)
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(
        repo_id=manifest["repo_id"], revision=manifest["revision"], folders=manifest["folders"]
    )
    marker = root / "corpus-identity.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("existing corpus has another manifest/revision")
    # Refuse stray trainable folders and quarantine remnants from a different policy.
    permitted = {
        (
            Path("quarantine") / e["folder"] if e["status"] == "quarantined" else Path(e["folder"])
        ).as_posix()
        for e in manifest["folders"]
    }
    for pattern in (
        "univariate/*",
        "covariates/*",
        "quarantine/univariate/*",
        "quarantine/covariates/*",
    ):
        for path in root.glob(pattern):
            if path.relative_to(root).as_posix() not in permitted:
                raise ValueError(f"unlisted or misplaced corpus folder: {path}")
    atomic_json(marker, identity)
    # Remove only the obsolete, redundant Hub cache, including failed old attempts.
    downloads = state / "downloads"
    if downloads.exists():
        for cache in downloads.rglob(".hub-cache"):
            if cache.is_symlink():
                cache.unlink()
            elif cache.is_dir():
                shutil.rmtree(cache)
    summary = dict(folders=[], failed_folders=[])
    for entry in manifest["folders"]:
        try:
            status = _fetch_folder(
                manifest, root, hub, state, entry, files[entry["folder"]], memory_budget_bytes
            )
            result = dict(folder=entry["folder"], status=status)
        except InterruptedError:
            raise  # A process interruption is not a corrupt folder.
        except Exception as exc:
            result = dict(
                folder=entry["folder"], status="error", error=f"{type(exc).__name__}: {exc}"
            )
            summary["failed_folders"].append(entry["folder"])
        summary["folders"].append(result)
        atomic_json(state / "fetch-summary.json", summary)
    if summary["failed_folders"]:
        errors = [
            f"{r['folder']}: {r['error']}" for r in summary["folders"] if r["status"] == "error"
        ]
        raise ValueError("failed corpus folders:\n" + "\n".join(errors))
    return root


def _saved_stats(directory, recorded, memory_budget_bytes):
    """Reuse exact measurements only when their receipt digest and limits agree."""
    path = directory / "stats.json"
    try:
        if path.is_symlink() or recorded.get("stats_sha256") != digest(path):
            return None
        stats = json.loads(path.read_text())
        if (
            stats["measurement"]["version"] == MEASUREMENT_VERSION
            and stats["measurement"]["peak_rss_bytes"] <= memory_budget_bytes
            and stats["largest_row_bytes"] <= memory_budget_bytes
        ):
            return stats
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _fetch_folder(manifest, root, hub, state, entry, expected_files, memory_budget_bytes):
    """One folder transaction; validated published data is retained on any failure."""
    folder = entry["folder"]
    destination = folder_path(root, entry)
    expected = expected_files
    receipt = dict(
        repo_id=manifest["repo_id"],
        revision=manifest["revision"],
        folder=folder,
        status=entry["status"],
        files=[asdict(f) for f in expected],
    )
    # Restored object-store snapshots can contain only part of a published folder.
    # Verify all surviving bytes, then stage them outside the reader roots again.
    staging = state / "downloads" / manifest["revision"] / entry["status"] / folder
    cached_stats = None
    if destination.exists():
        names = {p.name for p in destination.iterdir()}
        expected_names = {PurePosixPath(f.path).name for f in expected}
        if names - (expected_names | {"stats.json", "receipt.json"}):
            raise ValueError(f"unexpected files in completed folder: {folder}")
        recorded = (
            json.loads((destination / "receipt.json").read_text())
            if (destination / "receipt.json").exists()
            else {}
        )
        if recorded and {k: recorded.get(k) for k in receipt} != receipt:
            raise ValueError(f"receipt/revision mismatch: {folder}")
        for file in expected:
            local = destination / PurePosixPath(file.path).name
            if local.exists():
                verify_file(local, file)
        cached_stats = _saved_stats(destination, recorded, memory_budget_bytes)
        if names == expected_names | {"stats.json", "receipt.json"}:
            if cached_stats is not None:
                return "resumed"
            stats_path = destination / "stats.json"
            stats = measure_folder(
                destination, folder.split("/")[0], memory_budget_bytes=memory_budget_bytes
            )
            atomic_json(stats_path, stats)
            atomic_json(
                destination / "receipt.json", dict(**receipt, stats_sha256=digest(stats_path))
            )
            return "complete"
        staging.mkdir(parents=True, exist_ok=True)
        for file in expected:
            local = destination / PurePosixPath(file.path).name
            staged = staging / local.name
            if local.exists():
                if staged.exists():
                    verify_file(staged, file)
                    local.unlink()
                else:
                    local.replace(staged)
        for name in ("receipt.json", "stats.json"):
            (destination / name).unlink(missing_ok=True)
        destination.rmdir()
    staging.mkdir(parents=True, exist_ok=True)
    for file in expected:
        local = staging / PurePosixPath(file.path).name
        if not local.exists():
            partial = local.with_suffix(local.suffix + ".partial")
            hub.download(manifest["repo_id"], manifest["revision"], file, partial)
            verify_file(partial, file)
            partial.replace(local)
        verify_file(local, file)
    stats = (
        cached_stats
        if cached_stats is not None
        else measure_folder(staging, folder.split("/")[0], memory_budget_bytes=memory_budget_bytes)
    )
    atomic_json(staging / "stats.json", stats)
    atomic_json(
        staging / "receipt.json", dict(**receipt, stats_sha256=digest(staging / "stats.json"))
    )
    # Move on the same filesystem; eliminate partial/cache files before publish.
    for extra in staging.iterdir():
        if extra.name not in {PurePosixPath(f.path).name for f in expected} | {
            "stats.json",
            "receipt.json",
        }:
            if extra.is_dir():
                shutil.rmtree(extra)
            else:
                extra.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(destination)
    return "complete"


def main(argv=None):
    """CLI deliberately exposes no folder/revision overrides outside the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--memory-budget-gib",
        type=memory_budget_arg,
        default=DEFAULT_MEMORY_BUDGET_BYTES,
        help="process RSS budget in GiB (default: 32)",
    )
    args = parser.parse_args(argv)
    state = args.state or Path(os.environ.get("RUN_DIR", args.root)) / "state"
    try:
        fetch(
            read_manifest(args.manifest),
            args.root,
            PublicHub(),
            state=state,
            memory_budget_bytes=args.memory_budget_gib,
        )
    except (ValueError, OSError, OverflowError, MemoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
