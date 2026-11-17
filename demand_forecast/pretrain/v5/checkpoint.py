"""Atomic alternating checkpoints and the guest's preemption request protocol."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import torch

_CHUNK = 1024 * 1024
_FOOTER_BYTES = hashlib.sha256().digest_size


class CheckpointError(RuntimeError):
    """Slots exist, but no checkpoint can safely be restored."""


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(
    destination: Path, write: Callable[[BinaryIO], None], *, staging: Path | None = None
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent if staging is None else staging
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{destination.name}-", dir=staging, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _counters(payload: dict) -> tuple[int, int]:
    try:
        batch, saves = payload["counters"]["batch"], payload["counters"]["saves"]
    except (KeyError, TypeError) as exc:
        raise CheckpointError("checkpoint requires counters.batch and counters.saves") from exc
    if any(type(value) is not int or value < 0 for value in (batch, saves)):
        raise CheckpointError("checkpoint batch and saves must be nonnegative integers")
    return batch, saves


def save_checkpoint(
    directory: Path | str, payload: dict, *, counter: int, staging: Path | str | None = None
) -> tuple[Path, float]:
    """Publish torch bytes followed by their 32-byte binary SHA-256 digest.

    A directory fsync failure raises even after the rename has published the slot.
    """
    directory = Path(directory)
    staging = None if staging is None else Path(staging)
    started = time.perf_counter()
    _counters(payload)
    destination = directory / f"checkpoint-slot-{counter % 2}.pt"

    def write(stream: BinaryIO) -> None:
        torch.save(payload, stream)
        stream.seek(0)
        digest = hashlib.sha256()
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
        stream.write(digest.digest())

    _atomic_write(destination, write, staging=staging)
    return destination, time.perf_counter() - started


class _CheckpointBody(io.RawIOBase):
    """Seekable view of an open checkpoint that hides its integrity footer."""

    def __init__(self, stream: BinaryIO, size: int) -> None:
        self._stream = stream
        self._size = size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_CUR:
            offset += self.tell()
        elif whence == os.SEEK_END:
            offset += self._size
        elif whence != os.SEEK_SET:
            raise ValueError("invalid seek mode")
        if offset < 0:
            raise ValueError("negative seek position")
        return self._stream.seek(offset)

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self._size - self.tell())
        return self._stream.read(remaining if size < 0 else min(size, remaining))

    def readinto(self, buffer) -> int:
        size = min(len(buffer), max(0, self._size - self.tell()))
        return self._stream.readinto(memoryview(buffer)[:size])


def _read_checkpoint(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            body_size = os.fstat(stream.fileno()).st_size - _FOOTER_BYTES
            if body_size <= 0:
                raise CheckpointError("SHA-256 footer missing")
            digest = hashlib.sha256()
            remaining = body_size
            while remaining:
                chunk = stream.read(min(remaining, _CHUNK))
                if not chunk:
                    raise CheckpointError("truncated checkpoint body")
                digest.update(chunk)
                remaining -= len(chunk)
            footer = stream.read(_FOOTER_BYTES)
            if digest.digest() != footer:
                raise CheckpointError("SHA-256 footer mismatch")
            stream.seek(0)
            payload = torch.load(
                _CheckpointBody(stream, body_size), map_location="cpu", weights_only=True
            )
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint payload must be a dictionary")
        _counters(payload)
        return payload
    except Exception as exc:
        raise CheckpointError(f"invalid checkpoint {path}: {exc}") from exc


def latest_checkpoint(directory: Path | str) -> tuple[Path, dict] | None:
    directory = Path(directory)
    latest = None
    errors = []
    for slot in (0, 1):
        path = directory / f"checkpoint-slot-{slot}.pt"
        if not path.exists():
            continue
        try:
            payload = _read_checkpoint(path)
        except CheckpointError as exc:
            errors.append(str(exc))
            continue
        if latest is None or _counters(payload) > _counters(latest[1]):
            latest = path, payload
    if latest is None and errors:
        raise CheckpointError("; ".join(errors))
    return latest


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file(directory: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or ".." in name
        or name.startswith("/")
        or "\\" in name
    ):
        raise ValueError("manifest files require relative names without '..' or backslashes")
    artifact = directory
    for part in Path(name).parts:
        artifact /= part
        if artifact.is_symlink():
            raise ValueError("manifest files must not be symlinks or traverse symlinks")
    if not artifact.is_file():
        raise ValueError("manifest files must name regular files")
    return artifact


def write_manifest(path: Path | str, record: dict) -> None:
    """Atomically publish metadata and the size and SHA-256 of each relative file.

    A directory fsync failure raises even after the rename has published the manifest.
    """
    path = Path(path)
    if not record.get("files"):
        raise ValueError("files must name at least one artifact")
    files = {}
    for name in record["files"]:
        artifact = _manifest_file(path.parent, name)
        files[name] = {"sha256": file_digest(artifact), "bytes": artifact.stat().st_size}
    encoded = json.dumps({**record, "files": files}, allow_nan=False).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded))


def read_manifest(path: Path | str) -> dict | None:
    """Return a published record only while every named artifact still validates."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("files"), dict)
            or not record["files"]
        ):
            return None
        for name, metadata in record["files"].items():
            artifact = _manifest_file(path.parent, name)
            if not isinstance(metadata, dict) or type(metadata.get("bytes")) is not int:
                return None
            if artifact.stat().st_size != metadata["bytes"]:
                return None
            if file_digest(artifact) != metadata.get("sha256"):
                return None
        return record
    except (OSError, ValueError, TypeError):
        return None


class CheckpointRequest:
    def __init__(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir

    def pending(self) -> str | None:
        if self.run_dir is None:
            return None
        try:
            nonce = (self.run_dir / "checkpoint-request").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        if not nonce:
            return None
        try:
            answer = (self.run_dir / "checkpoint-done").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            answer = None
        return None if nonce == answer else nonce

    def answer(self, nonce: str) -> None:
        if self.run_dir is not None:
            _atomic_write(
                self.run_dir / "checkpoint-done", lambda stream: stream.write(nonce.encode())
            )
