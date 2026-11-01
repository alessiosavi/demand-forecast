"""Pretraining configuration and portable identities, without corpus dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from demand_forecast.config.settings import ForecastConfig, ModelConfig


class CorpusConfig(BaseModel):
    """Local corpus shards and independent validation sources."""

    model_config = ConfigDict(extra="forbid")

    root: Path
    tsmixup_shards: list[int]
    kernel_synth_shards: list[int]
    kernel_synth_every: int = 10
    holdout_tsmixup: list[int] = Field(default_factory=lambda: [199])
    holdout_kernel_synth: list[int] = Field(default_factory=lambda: [19])
    val_pairs_per_kind: int = 4096
    external_validation: str | None = "monash_nn5_weekly"


class PretrainTrainingConfig(BaseModel):
    """Fixed-budget pretraining controls, separate from the frozen Trainer budget."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1)
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    eval_every: int = 5000
    checkpoint_every: int = 1000
    amp: bool = True
    val_batch_size: int = 128


class _PretrainForecastConfig(ForecastConfig):
    """Reject pretraining typos without changing the shared Settings schema."""

    model_config = ConfigDict(extra="forbid")


class PretrainConfig(BaseModel):
    """Standalone pretraining configuration; loading never accesses corpus files."""

    model_config = ConfigDict(extra="forbid")

    corpus: CorpusConfig
    forecast: _PretrainForecastConfig
    model: ModelConfig
    training: PretrainTrainingConfig
    seed: int = 0
    device: str | None = None
    log_level: str = "INFO"

    @field_validator("forecast", mode="before")
    @classmethod
    def validate_forecast(cls, value: object) -> object:
        """Also accept existing ForecastConfig instances at the Python API boundary."""
        return value.model_dump() if isinstance(value, ForecastConfig) else value

    @classmethod
    def from_yaml(cls, path: Path) -> PretrainConfig:
        """Load and validate a YAML mapping."""
        import yaml

        with Path(path).open(encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))

    def to_yaml(self, path: Path) -> None:
        """Write JSON-compatible values, including paths, as plain YAML."""
        import yaml

        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.model_dump(mode="json"), stream, sort_keys=False)


def pretrain_identity(
    cfg: PretrainConfig, package_version: str, shard_sha256: dict[str, str]
) -> str:
    """Hash config, package version, and root-relative shard digests canonically."""
    payload = {
        "config": cfg.model_dump(mode="json", exclude={"corpus": {"root"}}),
        "package_version": package_version,
        "shards": shard_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
