"""Bundle asset helpers on tiny dummy files (no parquet engine needed)."""

import json
from pathlib import Path

import pytest

from demand_forecast.kaggle.assets import package_bundle, verify_bundle_identity

# Pinned first-try L3 pretraining identity (plan global constraints).
IDENTITY = "83d6acb1f7f2c1b57ba33c0ca0a870775369d52bf8fb739a9e3f7e28858826c2"

_BUNDLE_FILES = ("config.json", "encoder.json", "panel.json", "manifest.json", "model.pt")


def _write_manifest(bundle_dir: Path, payload: dict) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_verify_bundle_identity_accepts_match(tmp_path: Path):
    _write_manifest(tmp_path, {"pretrain_identity": IDENTITY, "bundle_version": 1})
    assert verify_bundle_identity(tmp_path, IDENTITY) is None


def test_verify_bundle_identity_rejects_mismatch(tmp_path: Path):
    _write_manifest(tmp_path, {"pretrain_identity": "some-other-identity"})
    with pytest.raises(ValueError, match="pretrain_identity"):
        verify_bundle_identity(tmp_path, IDENTITY)


def test_verify_bundle_identity_rejects_missing_key(tmp_path: Path):
    _write_manifest(tmp_path, {"bundle_version": 1})
    with pytest.raises(ValueError, match="pretrain_identity"):
        verify_bundle_identity(tmp_path, IDENTITY)


def test_package_bundle_copies_the_pinned_files(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for name in _BUNDLE_FILES:
        (src / name).write_bytes(f"payload of {name}".encode())
    (src / "extra.txt").write_text("not part of a bundle", encoding="utf-8")

    dest = tmp_path / "dest"
    written = package_bundle(src, dest)

    assert written == list(_BUNDLE_FILES)
    assert not (dest / "extra.txt").exists()
    for name in written:
        assert (dest / name).read_bytes() == (src / name).read_bytes()
