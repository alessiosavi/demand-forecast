"""Cut-off-bounded task-local adaptation inputs (no training or model loading)."""

from .data import build_manifest, build_task_manifest

__all__ = ["build_manifest", "build_task_manifest"]
