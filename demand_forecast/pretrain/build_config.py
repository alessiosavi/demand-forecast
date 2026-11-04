"""Offline validation of pinned corpus conversion recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusEntry(_StrictModel):
    """Source attribution and an explicit target-column mapping."""

    config: Name | None = None
    folder: Name | None = None
    licence: str = Field(min_length=1)
    card_licence: str | None = None
    bucket: Literal["yes", "nc", "sa"]
    attribution: str = Field(min_length=1)
    url: str = Field(pattern=r"^https?://")
    licence_url: str | None = Field(default=None, pattern=r"^https?://")
    target_column: str = "target"

    @model_validator(mode="after")
    def one_name(self):
        if (self.config is None) == (self.folder is None):
            raise ValueError("provide exactly one of config or folder")
        return self

    @property
    def name(self) -> str:
        return self.config if self.config is not None else self.folder


class SourceExclusions(_StrictModel):
    """Recorded screening decisions; no implicit downloads outside include."""

    fev_overlap: list[str] = Field(default_factory=list)
    external_validation: list[str] = Field(default_factory=list)
    unlicensed_or_unverified: list[str] = Field(default_factory=list)
    unclear_competition_grant: list[str] = Field(default_factory=list)
    ruled_out: list[str] = Field(default_factory=list)
    existing_corpus: list[str] = Field(default_factory=list)
    duplicate_of_chronos: list[str] = Field(default_factory=list)
    unclear: list[str] = Field(default_factory=list)
    not_relevant: str | None = None


class BuildSource(_StrictModel):
    """A pinned Hub repository, optionally replaced by a local snapshot."""

    name: Name
    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    kind: Literal["chronos_real", "gifteval_pretrain"]
    bundle_terms: str | None = None
    include: list[CorpusEntry] = Field(min_length=1)
    excluded: SourceExclusions
    local_path: Path | None = None

    @model_validator(mode="after")
    def validate_entries(self):
        names = [entry.name for entry in self.include]
        if len(set(names)) != len(names):
            raise ValueError("duplicate included folder/config")
        for entry in self.include:
            if (self.kind == "chronos_real") != (entry.config is not None):
                raise ValueError("chronos_real uses config; gifteval_pretrain uses folder")
        excluded = {
            name
            for values in self.excluded.model_dump().values()
            if isinstance(values, list)
            for name in values
        }
        if set(names) & excluded:
            raise ValueError("included folder/config is also excluded")
        return self

    def allow_patterns(self) -> list[str]:
        if self.kind == "chronos_real":
            return [f"{entry.name}/*.parquet" for entry in self.include]
        return [
            f"{entry.name}/{pattern}"
            for entry in self.include
            for pattern in ("data-*.arrow", "dataset_info.json", "state.json")
        ]


class BuildFilters(_StrictModel):
    min_points: int = Field(default=64, ge=1)
    drop_constant: bool = True
    drop_all_nan: bool = True
    dedupe: Literal["value_hash"] = "value_hash"
    max_series_per_folder: int | None = Field(default=None, ge=1)


class ShardLimits(_StrictModel):
    max_series: int = Field(default=20000, ge=1)
    max_bytes: int = Field(default=268435456, ge=1)


class CorpusBuildConfig(_StrictModel):
    """The conversion recipe; loading it never accesses the Hub."""

    version: Literal[2]
    commercial_use: Literal[False]
    sources: list[BuildSource] = Field(min_length=1)
    filters: BuildFilters = Field(default_factory=BuildFilters)
    shards: ShardLimits = Field(default_factory=ShardLimits)

    @model_validator(mode="after")
    def unique_sources(self):
        if len({s.name for s in self.sources}) != len(self.sources):
            raise ValueError("source names must be unique")
        if len({s.kind for s in self.sources}) != len(self.sources):
            raise ValueError("each kind must have exactly one source")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> CorpusBuildConfig:
        import yaml

        path = Path(path)
        config = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for source in config.sources:
            if source.local_path is not None:
                source.local_path = (path.resolve().parent / source.local_path).resolve()
        return config
