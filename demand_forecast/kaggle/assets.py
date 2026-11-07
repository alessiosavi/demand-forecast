"""Pretrained-bundle packaging and identity verification for kernel runs.

The pretrained bundle travels from the laptop to Kaggle as a private dataset:
``package_bundle`` copies its five portable artifacts into the upload directory,
and ``verify_bundle_identity`` refuses a workload when the mounted bundle does
not carry the expected ``pretrain_identity``.
"""

import json
import shutil
from pathlib import Path

__all__ = ["package_bundle", "verify_bundle_identity"]

_BUNDLE_FILES = ("config.json", "encoder.json", "panel.json", "manifest.json", "model.pt")


def verify_bundle_identity(bundle_dir: Path, expected: str) -> None:
    """Raise ``ValueError`` unless the bundle manifest carries ``expected``."""
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    actual = manifest.get("pretrain_identity")
    if actual != expected:
        raise ValueError(
            f"Bundle {bundle_dir} pretrain_identity mismatch: "
            f"expected {expected!r}, found {actual!r}"
        )


def package_bundle(src: Path, dest: Path) -> list[str]:
    """Copy the five portable bundle artifacts into ``dest``; return their names."""
    src, dest = Path(src), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in _BUNDLE_FILES:
        shutil.copyfile(src / name, dest / name)
    return list(_BUNDLE_FILES)
