"""Chronos training shards and portable manifests of their content hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

from demand_forecast.data.panel import Panel, Roles, Series

CORPUS_REPO = "autogluon/chronos_datasets"
CHRONOS_REVISION = "eeecad0b82a8c237e212ce6f8d1abecb513e2cec"
TSMIXUP_PREFIX = "training_corpus/tsmixup_10m/"
KERNEL_SYNTH_PREFIX = "training_corpus/kernel_synth_1m/"
PRETRAIN_START = pd.Timestamp("2000-01-02")
PRETRAIN_FREQ = "W-SUN"
CORPUS_ROLES = Roles("target", (), (), (), ())

CORPUS_KINDS = {
    "tsmixup": TSMIXUP_PREFIX.rstrip("/"),
    "kernel_synth": KERNEL_SYNTH_PREFIX.rstrip("/"),
    "chronos_real": "training_corpus/chronos_real",
    "gifteval_pretrain": "training_corpus/gifteval_pretrain",
}


def read_shard(path: Path, *, dtype: DTypeLike = np.float32) -> list[Series]:
    """Read targets in file order, ignoring the corpus's dummy timestamps.

    Preserve all target positions, including NaNs. The weekly start and grid
    are synthetic coordinates for the shared batch builder, not calendar data.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "target"])
    series = []
    for series_id, target in zip(table["id"], table["target"], strict=True):
        values = np.asarray(target.as_py(), dtype=dtype)
        series.append(
            Series(
                series_id=series_id.as_py(),
                start=PRETRAIN_START,
                history_length=len(values),
                target=values,
                dynamic={},
                statics={},
            )
        )
    return series


def shard_panel(series: list[Series]) -> Panel:
    """Wrap corpus series for batching on a synthetic weekly grid."""
    return Panel(series, CORPUS_ROLES, PRETRAIN_FREQ, seasonal_period=1, target_support="real")


@dataclass(frozen=True)
class ShardRef:
    """A shard's location, corpus identity, content digest, and size in bytes."""

    path: Path
    kind: str
    index: int
    sha256: str
    size: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_manifest(root: Path) -> list[ShardRef]:
    """Hash the local corpus shards, returning absolute paths in sorted order.

    Legacy kinds retain their exact shard counts; new kinds accept any
    five-digit count. Hashing reads bounded blocks rather than entire files.
    """
    root = Path(root).resolve()
    refs = []
    for kind, prefix in CORPUS_KINDS.items():
        count = {"tsmixup": "00200", "kernel_synth": "00020"}.get(kind, "[0-9]" * 5)
        pattern = "train-" + "[0-9]" * 5 + f"-of-{count}.parquet"
        for path in (root / prefix).glob(pattern):
            if path.is_file():
                refs.append(
                    ShardRef(
                        path=path,
                        kind=kind,
                        index=int(path.name.split("-")[1]),
                        sha256=_sha256(path),
                        size=path.stat().st_size,
                    )
                )
    return sorted(refs, key=lambda ref: ref.path)


def write_manifest(refs: list[ShardRef], path: Path) -> None:
    """Write JSON beside the corpus's ``training_corpus`` directory.

    The manifest's parent is the corpus root. Paths are stored relative to it,
    so moving that root, including the manifest, preserves shard identities.
    """
    path = Path(path)
    root = path.resolve().parent
    payload = [
        {
            "path": ref.path.resolve().relative_to(root).as_posix(),
            "kind": ref.kind,
            "index": ref.index,
            "sha256": ref.sha256,
            "size": ref.size,
        }
        for ref in refs
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[ShardRef]:
    """Resolve JSON paths against the manifest's parent, without rehashing files."""
    path = Path(path)
    root = path.resolve().parent
    refs = []
    for entry in json.loads(path.read_text(encoding="utf-8")):
        relative = Path(entry["path"])
        resolved = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(root):
            raise ValueError("manifest shard paths must be root-relative")
        refs.append(
            ShardRef(
                path=resolved,
                kind=entry["kind"],
                index=entry["index"],
                sha256=entry["sha256"],
                size=entry["size"],
            )
        )
    return refs


def download_shards(
    root: Path,
    tsmixup: list[int],
    kernel_synth: list[int],
    revision: str | None = CHRONOS_REVISION,
) -> list[Path]:
    """Download only the selected shard filenames; intended for EC2 setup."""
    patterns = [f"{TSMIXUP_PREFIX}train-{index:05d}-of-00200.parquet" for index in tsmixup]
    patterns += [
        f"{KERNEL_SYNTH_PREFIX}train-{index:05d}-of-00020.parquet" for index in kernel_synth
    ]
    if not patterns:
        return []

    from huggingface_hub import snapshot_download

    snapshot_download(
        CORPUS_REPO, repo_type="dataset", revision=revision, local_dir=root, allow_patterns=patterns
    )
    return [Path(root) / name for name in patterns]


def verify_manifest(root: Path) -> dict[str, tuple[int, int]]:
    """Rehash every manifest entry; return (valid, failed) counts by kind."""
    counts = {}
    for ref in read_manifest(Path(root) / "manifest.json"):
        valid, failed = counts.get(ref.kind, (0, 0))
        matches = (
            ref.path.is_file()
            and ref.path.stat().st_size == ref.size
            and _sha256(ref.path) == ref.sha256
        )
        counts[ref.kind] = (valid + int(matches), failed + int(not matches))
    return counts
