"""Pinned Kaggle templates and settings composition.

Loads the committed L3 templates, replaces the dataset and horizon with the
competition's values, and writes composed settings back as plain YAML.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # repo venv carries no PyYAML stubs

from demand_forecast.config import DatasetConfig, Settings

__all__ = ["build_settings", "load_template", "write_settings"]


def load_template(path: Path) -> dict[str, Any]:
    """Load a raw settings-template mapping from YAML."""
    with Path(path).open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)
    return raw


def build_settings(template: dict[str, Any], *, data: DatasetConfig, horizon: int) -> Settings:
    """Validate a template with the dataset and horizon replaced.

    The caller's mapping is never mutated; all model and training values from
    the template survive into the validated settings.
    """
    raw = deepcopy(template)
    raw["data"] = data.model_dump(mode="json")
    raw["forecast"]["horizon"] = horizon
    return Settings.model_validate(raw)


def write_settings(settings: Settings, path: Path) -> None:
    """Write composed settings as plain YAML with JSON-compatible values."""
    settings.to_yaml(path)
