"""Pretraining configuration and portable identities, without corpus dependencies."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from demand_forecast.config.settings import ForecastConfig, ModelConfig

LEGACY_RECIPE_DEFAULTS = {
    "recipe": "v2",
    "period_policy": "legacy_one",
    "grid_policy": "legacy_weekly",
}


class SourceShards(BaseModel):
    """Training shards, disjoint hold-outs and a positive mixing weight."""

    model_config = ConfigDict(extra="forbid")

    shards: list[int]
    holdout: list[int] = Field(default_factory=list)
    weight: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_indices(self):
        for indices in (self.shards, self.holdout):
            if len(set(indices)) != len(indices) or any(i < 0 for i in indices):
                raise ValueError("shard indices must be unique and nonnegative")
        if not self.shards or set(self.shards) & set(self.holdout):
            raise ValueError("sources require training shards disjoint from holdout")
        return self


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
    sources: dict[str, SourceShards] | None = None

    @model_validator(mode="before")
    @classmethod
    def reconcile_sources(cls, value):
        if not isinstance(value, dict) or value.get("sources") is None:
            return value
        value = dict(value)
        if not isinstance(value["sources"], dict):
            raise ValueError("sources must be a mapping of kind to source settings")
        sources = {k: SourceShards.model_validate(v) for k, v in value["sources"].items()}
        known = {"tsmixup", "kernel_synth", "chronos_real", "gifteval_pretrain"}
        if "tsmixup" not in sources or set(sources) - known:
            raise ValueError("sources must include tsmixup and contain only registered kinds")
        for kind in ("tsmixup", "kernel_synth"):
            source = sources.get(kind)
            for key, expected in (
                (f"{kind}_shards", source.shards if source else []),
                (f"holdout_{kind}", source.holdout if source else []),
            ):
                if key in value and value[key] != expected:
                    raise ValueError(f"{key} conflicts with sources")
                value[key] = expected
        if "kernel_synth_every" in value:
            every = value["kernel_synth_every"]
            if not isinstance(every, int) or isinstance(every, bool):
                raise ValueError("kernel_synth_every must be an integer")
            kernel = sources.get("kernel_synth")
            # This legacy control can only express a two-kind weight ratio.
            compatible = (
                len(sources) == 1
                and every == 0
                or len(sources) == 2
                and kernel is not None
                and every > 1
                and Fraction(str(sources["tsmixup"].weight))
                == Fraction(str(kernel.weight)) * (every - 1)
            )
            if not compatible:
                raise ValueError("kernel_synth_every conflicts with sources weights")
        return value

    @model_serializer(mode="wrap")
    def serialize_compatible(self, handler):
        payload = handler(self)
        if self.sources is None:
            payload.pop("sources", None)
        elif "kernel_synth_every" not in self.model_fields_set:
            payload.pop("kernel_synth_every", None)
        return payload

    def selections(self) -> dict[str, tuple[list[int], list[int]]]:
        """Return training/hold-out indices without changing legacy defaults."""
        if self.sources is not None:
            return {kind: (source.shards, source.holdout) for kind, source in self.sources.items()}
        return {
            "tsmixup": (self.tsmixup_shards, self.holdout_tsmixup),
            "kernel_synth": (self.kernel_synth_shards, self.holdout_kernel_synth),
        }


class PretrainTrainingConfig(BaseModel):
    """Fixed-budget pretraining controls, separate from the frozen Trainer budget."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1)
    batch_size: int = Field(128, ge=1)
    learning_rate: float = Field(5e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    warmup_steps: int = Field(2000, ge=0)
    grad_clip: float = Field(1.0, gt=0)
    eval_every: int = Field(5000, ge=1)
    checkpoint_every: int = Field(1000, ge=1)
    amp: bool = True
    val_batch_size: int = Field(128, ge=1)


class _PretrainForecastConfig(ForecastConfig):
    """Reject pretraining typos without changing the shared Settings schema."""

    model_config = ConfigDict(extra="forbid")

    # Task-only control: keep pretraining config serialization and identities intact.
    scaler_period: Literal["task"] = Field("task", exclude=True)

    @field_validator("scale_floor_frac")
    @classmethod
    def reject_relative_floor(cls, value: float) -> float:
        """Keep the pretraining protocol unchanged; the relative floor is task-only."""
        if value != 0.0:
            raise ValueError("scale_floor_frac is only supported for task training")
        return value

    @field_validator("calendar_schema")
    @classmethod
    def reject_calendar_schema(cls, value: str) -> str:
        """Keep legacy pretraining batches until the v3 adapter supports named schemas."""
        if value != "legacy":
            raise ValueError(
                "non-default calendar_schema is only supported for task training; "
                "pretraining requires 'legacy'"
            )
        return value

    @field_validator("target_lags")
    @classmethod
    def reject_target_lags(cls, value: str) -> str:
        """Keep pretraining lag-free until the v3 adapter supports target lags."""
        if value != "none":
            raise ValueError(
                "non-default target_lags is only supported for task training; "
                "pretraining requires 'none'"
            )
        return value


class PretrainConfig(BaseModel):
    """Standalone pretraining configuration; loading never accesses corpus files."""

    model_config = ConfigDict(extra="forbid")

    corpus: CorpusConfig
    forecast: _PretrainForecastConfig
    model: ModelConfig
    training: PretrainTrainingConfig
    recipe: Literal["v2", "v3"] = "v2"
    period_policy: Literal["legacy_one", "frequency"] = "legacy_one"
    grid_policy: Literal["legacy_weekly", "source"] = "legacy_weekly"
    seed: int = 0
    device: str | None = None
    log_level: str = "INFO"

    @model_serializer(mode="wrap")
    def serialize_compatible(self, handler):
        """Keep legacy bundle config payloads byte-equivalent at default controls."""
        payload = handler(self)
        for name, default in LEGACY_RECIPE_DEFAULTS.items():
            if payload.get(name) == default:
                payload.pop(name)
        return payload

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
    # The shared forecast schema gained a task-only switch. Do not invalidate
    # existing pretraining checkpoints by hashing its inactive default.
    payload["config"]["forecast"].pop("scale_floor_frac", None)
    for name, default in LEGACY_RECIPE_DEFAULTS.items():
        if payload["config"].get(name) == default:
            payload["config"].pop(name)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
