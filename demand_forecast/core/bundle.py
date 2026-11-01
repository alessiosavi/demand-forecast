"""Versioned, inspectable inference bundles containing JSON and tensor weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from demand_forecast import __version__
from demand_forecast.config.settings import DatasetConfig, ForecastConfig
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import to_offset
from demand_forecast.data.panel import Panel
from demand_forecast.models.base import ForecastModel, ModelSpec
from demand_forecast.models.registry import build_model

BUNDLE_VERSION = 1
_ARTIFACTS = ("model.pt", "config.json", "encoder.json", "panel.json")


@dataclass
class Bundle:
    """Rebuilt inference model, fitted preprocessing and its saved provenance.

    ``panel_metadata`` retains the JSON representation (ISO timestamps, frequency
    string and named roles); it contains no training observations.
    """

    model: ForecastModel
    family: str
    backbone_kwargs: dict
    forecast_cfg: ForecastConfig
    dataset_cfg: DatasetConfig | None
    encoder: StaticEncoder
    panel_metadata: dict
    manifest: dict


def _git_sha() -> str | None:
    """Best-effort source revision; installed wheels need not have a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    # Preserve feature order: ModelSpec vocabularies determine embedding order.
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_bundle(
    directory: Path,
    *,
    model: ForecastModel,
    family: str,
    backbone_kwargs: dict,
    forecast_cfg: ForecastConfig,
    dataset_cfg: DatasetConfig | None,
    encoder: StaticEncoder,
    panel: Panel,
    training_cutoff: pd.Timestamp,
    earliest_origin: pd.Timestamp,
    manifest: dict,
) -> None:
    """Save state_dict weights, reconstruction data and artifact hashes.

    Caller provenance, including seed, run identity and input hashes, is retained.
    Measured artifact fields and the selection cutoff are always written here.
    All artifacts are staged before publication; a failed publication rename
    restores the previous bundle.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"Bundle path is not a directory: {directory}")
    with TemporaryDirectory(dir=directory.parent, prefix=f"{directory.name}.tmp-") as name:
        temporary = Path(name)
        cutoff = pd.Timestamp(training_cutoff).isoformat()
        torch.save(model.state_dict(), temporary / "model.pt")
        _write_json(
            temporary / "config.json",
            {
                "family": family,
                "backbone_kwargs": backbone_kwargs,
                "forecast": forecast_cfg.model_dump(mode="json"),
                "dataset": dataset_cfg.model_dump(mode="json") if dataset_cfg is not None else None,
                "model_spec": asdict(model.spec),
            },
        )
        _write_json(temporary / "encoder.json", encoder.to_json())
        _write_json(
            temporary / "panel.json",
            {
                "freq": to_offset(panel.freq).freqstr,
                "seasonal_period": panel.seasonal_period,
                "target_support": panel.target_support,
                "roles": asdict(panel.roles),
                "training_cutoff": cutoff,
                "earliest_origin": pd.Timestamp(earliest_origin).isoformat(),
                "fallback_level": model.fallback_level.item(),
            },
        )
        _write_json(
            temporary / "manifest.json",
            {
                "seed": None,
                **manifest,
                "bundle_version": BUNDLE_VERSION,
                "package_version": __version__,
                "git_sha": _git_sha(),
                "selection_used_labels_until": cutoff,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sha256": {name: _sha256(temporary / name) for name in _ARTIFACTS},
            },
        )
        backup = None
        if directory.exists():
            backup = temporary.with_name(f"{temporary.name}.old")
            os.replace(directory, backup)
        try:
            os.replace(temporary, directory)
        except OSError:
            if backup is not None:
                os.replace(backup, directory)
            raise
        if backup is not None:
            shutil.rmtree(backup)


def load_bundle_manifest(directory: Path | str) -> dict:
    """Read bundle provenance without loading or verifying model artifacts."""
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def load_bundle(directory: Path | str, device: torch.device | str = "cpu") -> Bundle:
    """Verify artifacts and rebuild the architecture without unpickling Python objects."""
    directory = Path(directory)
    manifest = load_bundle_manifest(directory)
    if manifest["bundle_version"] != BUNDLE_VERSION:
        raise ValueError(f"Unsupported bundle version: {manifest['bundle_version']}")
    for name in _ARTIFACTS:
        if _sha256(directory / name) != manifest["sha256"].get(name):
            raise ValueError(f"Bundle artifact hash mismatch: {name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    encoder = StaticEncoder.from_json(
        json.loads((directory / "encoder.json").read_text(encoding="utf-8"))
    )
    spec_data = config["model_spec"]
    spec = ModelSpec(**{**spec_data, "quantiles": tuple(spec_data["quantiles"])})
    model = build_model(config["family"], spec, config["backbone_kwargs"]).to(device)
    model.load_state_dict(
        torch.load(directory / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return Bundle(
        model=model,
        family=config["family"],
        backbone_kwargs=config["backbone_kwargs"],
        forecast_cfg=ForecastConfig.model_validate(config["forecast"]),
        dataset_cfg=(
            DatasetConfig.model_validate(config["dataset"])
            if config["dataset"] is not None
            else None
        ),
        encoder=encoder,
        panel_metadata=json.loads((directory / "panel.json").read_text(encoding="utf-8")),
        manifest=manifest,
    )
