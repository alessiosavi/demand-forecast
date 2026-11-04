"""Copy and verify the private bundle using only the Python standard library."""

import argparse
import hashlib
import re
import shutil
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_path(directory, name):
    path = directory / name
    for component in (path, *path.parents):
        if component == directory:
            break
        if component.is_symlink():
            raise ValueError(f"Symlink payload is not allowed: {name}")
    return path


def checksums(directory):
    result = {}
    for line in _payload_path(Path(directory), "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"Invalid SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or name == "SHA256SUMS"
            or name in result
            or "\\" in name
        ):
            raise ValueError(f"Unsafe or duplicate checksum path: {name}")
        result[name] = digest
    if not result:
        raise ValueError("Empty SHA256SUMS")
    return result


def _check_file(directory, name, digest):
    path = _payload_path(directory, name)
    if not path.is_file():
        raise ValueError(f"missing payload: {name}")
    if sha256(path) != digest:
        raise ValueError(f"sha256 mismatch: {name}")


def verify(bundle):
    """Require exactly the listed payloads plus SHA256SUMS, with matching hashes."""
    bundle = Path(bundle)
    expected = checksums(bundle)
    found = {
        p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file() or p.is_symlink()
    }
    missing = set(expected) - found
    extra = found - set(expected) - {"SHA256SUMS"}
    if missing or extra:
        raise ValueError(f"Bundle missing={sorted(missing)}, extra={sorted(extra)}")
    for name, digest in expected.items():
        _check_file(bundle, name, digest)
    return expected


def build(built, out):
    """Copy listed files; refuse conflicting/extra destination files, never overwrite."""
    built, out = Path(built), Path(out)
    expected = checksums(built)
    for name, digest in expected.items():
        _check_file(built, name, digest)
    out.mkdir(parents=True, exist_ok=True)
    found = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() or p.is_symlink()}
    if found - set(expected) - {"SHA256SUMS"}:
        raise ValueError("extra files in bundle destination")
    # Validate every existing destination before any copy. Matching partial
    # copies are resumable; a different bundle needs a different directory.
    for name, digest in {**expected, "SHA256SUMS": sha256(built / "SHA256SUMS")}.items():
        target = _payload_path(out, name)
        if target.exists():
            _check_file(out, name, digest)
    for name in [*expected, "SHA256SUMS"]:
        target = _payload_path(out, name)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(built / name, target)
    return verify(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--built", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build(args.built, args.out) if args.command == "build" else verify(args.bundle)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Bundle verification failed: {exc}\n")
    print(f"Verified {len(result)} payload files")


if __name__ == "__main__":
    main()
