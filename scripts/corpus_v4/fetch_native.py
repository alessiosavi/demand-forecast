"""Fetch pinned native shards, verify Hub digests, validate and measure every folder."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from scripts.corpus_v4.freeze_manifest import DEFAULT_MANIFEST, read_manifest, validate_manifest
from scripts.corpus_v4.native import atomic_json, digest, folder_path, measure_folder

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

        # Put the transfer cache beside staging so downloaded bytes survive restore.
        cached = hf_hub_download(
            repo_id,
            file.path,
            repo_type="dataset",
            revision=revision,
            token=False,
            cache_dir=str(destination.parent / ".hub-cache"),
        )
        shutil.copyfile(cached, destination)


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


def fetch(manifest, root, hub, *, state):
    """Publish a folder atomically only after all hashes and native rows validate."""
    root, state = Path(root), Path(state)
    files = inventory(manifest, hub)
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(
        repo_id=manifest["repo_id"], revision=manifest["revision"], folders=manifest["folders"]
    )
    marker = root / "corpus-identity.json"
    if marker.exists():
        import json

        if json.loads(marker.read_text()) != identity:
            raise ValueError("existing corpus has another manifest/revision")
    # Refuse stray trainable folders and quarantine remnants from a different policy.
    permitted = {folder_path(root, e).relative_to(root).as_posix() for e in manifest["folders"]}
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
    for entry in manifest["folders"]:
        folder = entry["folder"]
        destination = folder_path(root, entry)
        expected = files[folder]
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
        if destination.exists():
            import json

            names = {p.name for p in destination.iterdir()}
            expected_names = {PurePosixPath(f.path).name for f in expected}
            if names - (expected_names | {"stats.json", "receipt.json"}):
                raise ValueError(f"unexpected files in completed folder: {folder}")
            if (destination / "receipt.json").exists() and json.loads(
                (destination / "receipt.json").read_text()
            ) != receipt:
                raise ValueError(f"receipt/revision mismatch: {folder}")
            for file in expected:
                local = destination / PurePosixPath(file.path).name
                if local.exists():
                    verify_file(local, file)
            if names == expected_names | {"stats.json", "receipt.json"}:
                atomic_json(
                    destination / "stats.json", measure_folder(destination, folder.split("/")[0])
                )
                continue
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
        stats = measure_folder(staging, folder.split("/")[0])
        atomic_json(staging / "stats.json", stats)
        atomic_json(staging / "receipt.json", receipt)
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
    return root


def main(argv=None):
    """CLI deliberately exposes no folder/revision overrides outside the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args(argv)
    state = args.state or Path(os.environ.get("RUN_DIR", args.root)) / "state"
    fetch(read_manifest(args.manifest), args.root, PublicHub(), state=state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
