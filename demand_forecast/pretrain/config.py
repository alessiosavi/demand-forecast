"""Pretraining configuration and portable identities, without corpus dependencies."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from demand_forecast.config.settings import ForecastConfig, ModelConfig

LEGACY_RECIPE_DEFAULTS = {
    "recipe": "v2",
    "period_policy": "legacy_one",
    "grid_policy": "legacy_weekly",
    "horizon_policy": "fixed",
    "shard_visit_quota": {},
}

# Unique horizons in bench/tasks/fev_bench_100_193e4fc.yaml.
FEV_HORIZONS = (5, 8, 12, 13, 14, 24, 28, 39, 48, 52, 60, 61, 96, 144, 168, 288)


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
    selection_shards: dict[str, list[int]] | None = None

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
        if self.selection_shards is None:
            payload.pop("selection_shards", None)
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
    ema_decay: float | None = Field(None, gt=0, lt=1, allow_inf_nan=False)
    keep_eval_snapshots: bool = False

    @model_serializer(mode="wrap")
    def serialize_compatible(self, handler):
        payload = handler(self)
        for name, default in (("ema_decay", None), ("keep_eval_snapshots", False)):
            if payload.get(name) == default:
                payload.pop(name, None)
        return payload


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
    horizon_policy: Literal["fixed", "fev_set"] = "fixed"
    shard_visit_quota: dict[str, Annotated[int, Field(gt=0, strict=True)]] = Field(
        default_factory=dict
    )
    seed: int = 0
    device: str | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_sampling(self):
        if self.recipe != "v3" and (self.horizon_policy != "fixed" or self.shard_visit_quota):
            raise ValueError("horizon_policy and shard_visit_quota require recipe v3")
        if self.horizon_policy == "fev_set" and self.forecast.horizon != max(FEV_HORIZONS):
            raise ValueError("fev_set requires forecast.horizon = 288")
        if self.recipe != "v3" and (
            self.training.ema_decay is not None
            or self.training.keep_eval_snapshots
            or self.corpus.selection_shards is not None
        ):
            raise ValueError(
                "selection_shards, ema_decay and keep_eval_snapshots require recipe v3"
            )
        if self.corpus.selection_shards is not None:
            for kind, indices in self.corpus.selection_shards.items():
                selections = self.corpus.selections()
                if (
                    kind not in ("chronos_real", "gifteval_pretrain")
                    or kind not in selections
                    or not indices
                    or len(set(indices)) != len(indices)
                    or not set(indices) <= set(selections[kind][1])
                ):
                    raise ValueError("selection_shards must name real held-out shards")
            if not self.corpus.selection_shards:
                raise ValueError("selection_shards must not be empty")
        for kind in self.shard_visit_quota:
            if kind not in self.corpus.selections():
                raise ValueError("shard_visit_quota requires configured sources and positive pairs")
        return self

    @model_validator(mode="before")
    @classmethod
    def resolve_input_schema(cls, value):
        """V3 always uses canonical calendar/cycle lags; v2 keeps its old layout."""
        if not isinstance(value, dict):
            return value
        value = dict(value)
        forecast = value.get("forecast", {})
        forecast = forecast.model_dump() if isinstance(forecast, ForecastConfig) else forecast
        if not isinstance(forecast, dict):
            return value
        forecast = dict(forecast)
        if value.get("recipe", "v2") == "v3":
            # Omitted/default task knobs resolve to the fixed v3 layout.
            if forecast.get("calendar_schema", "legacy") == "legacy":
                forecast["calendar_schema"] = "canonical"
            if forecast.get("target_lags", "none") == "none":
                forecast["target_lags"] = "cycle"
        else:
            errors = []
            for name, default in (("calendar_schema", "legacy"), ("target_lags", "none")):
                if forecast.get(name, default) != default:
                    errors.append(
                        dict(
                            type="value_error",
                            loc=("forecast", name),
                            input=forecast[name],
                            ctx={
                                "error": ValueError(
                                    f"{name} is only supported for task training or recipe v3"
                                )
                            },
                        )
                    )
            if errors:
                raise ValidationError.from_exception_data(cls.__name__, errors)
        value["forecast"] = forecast
        return value

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
    if cfg.recipe == "v3":
        from demand_forecast.data.frequency import CANONICAL_CALENDAR_SCHEMA
        from demand_forecast.data.sampler import TARGET_LAG_SCHEMA

        payload["input_schema"] = {
            "calendar_schema": CANONICAL_CALENDAR_SCHEMA,
            "lag_schema": TARGET_LAG_SCHEMA,
            "target_lags": "cycle",
        }
        payload["selection_pair_policy"] = selection_pair_policy_hash()
        payload["sampler_policy"] = {
            "version": 1,
            "pairs": "series_uniform_origin_uniform",
            "window": "full_future_min_origin_max_min_context_m_plus_1",
            "fev_horizons": FEV_HORIZONS if cfg.horizon_policy == "fev_set" else None,
        }
    for name, default in LEGACY_RECIPE_DEFAULTS.items():
        if payload["config"].get(name) == default:
            payload["config"].pop(name)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# Changing construction or aggregation deliberately invalidates v3 continuation.
SELECTION_PAIR_POLICY = {
    "version": 1,
    "quota": "min_4096_configured_per_source",
    "cells": "shard_panels_conventional_period_cross_feasible_fev_horizons_or_fixed",
    "allocation": "sorted_cells_round_robin_capacity_capped",
    "pairs": "seeded_series_without_replacement_final_full_future",
    "window": "origin_ge_max_min_context_m_plus_1",
    "aggregation": "equal_source_mean_of_equal_cell_sql_all_cells_finite",
}


def selection_pair_policy_hash() -> str:
    """Hash the frozen validation-pair construction and aggregation contract."""
    return hashlib.sha256(
        json.dumps(SELECTION_PAIR_POLICY, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
