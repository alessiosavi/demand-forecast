import hashlib

import pytest

from scripts.world_signals.colab_bundle import build, verify


def test_colab_bundle_verify_detects_mismatch_missing_extra(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"test")
    (source / "SHA256SUMS").write_text(f"{hashlib.sha256(b'test').hexdigest()}  payload\n")
    (source / "not-payload.log").write_text("do not copy")
    out = tmp_path / "bundle"
    build(source, out)
    assert {p.name for p in out.iterdir()} == {"payload", "SHA256SUMS"}
    verify(out)
    (out / "payload").write_bytes(b"bad")
    with pytest.raises(ValueError, match="mismatch"):
        verify(out)
    (out / "payload").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify(out)
    (out / "payload").write_bytes(b"test")
    (out / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="extra"):
        verify(out)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b", "SHA256SUMS"])
def test_bundle_rejects_unsafe_checksum_paths(tmp_path, name):
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  {name}\n")
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_bundle_build_refuses_conflicting_destination(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "payload").write_bytes(b"test")
    (source / "SHA256SUMS").write_text(f"{hashlib.sha256(b'test').hexdigest()}  payload\n")
    build(source, destination)
    build(source, destination)  # A matching existing copy is idempotent.
    (destination / "payload").write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="mismatch"):
        build(source, destination)
    assert (destination / "payload").read_bytes() == b"do not overwrite"
