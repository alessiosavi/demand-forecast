"""Portable training settings and identities for native v5 experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from demand_forecast.models.v5.config import V5Config


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    weight: float
    root: str | None = None
    heldout: bool = False
    recipes: tuple[tuple[str, ...], ...] = ()
    freq: str | None = None
    length: tuple[int, int] = (64, 2048)
    max_items: int = 48
    n_nuisance: int = 0
    nuisance_from: tuple[str, ...] | None = None
    singleton_share: float = 0.0
    control: str | None = None
    control_inputs: str = "all"

    def __post_init__(self) -> None:
        if self.kind not in ("typed", "corpus", "upstream"):
            raise ValueError("kind must be typed, corpus or upstream")
        if self.control not in (None, "shuffle", "no_dates", "missing_as_zero"):
            raise ValueError("control must be None, shuffle, no_dates or missing_as_zero")
        if self.control_inputs not in ("all", "attribute", "column"):
            raise ValueError("control_inputs must be all, attribute or column")


@dataclass(frozen=True)
class SamplerConfig:
    episodes_per_batch: int = 32
    episodes_per_panel: int = 8
    origins: int = 4
    origin_stride: int = 2
    contexts: tuple[int, ...] = (256, 512, 1024)
    horizon: tuple[int, int] = (16, 288)
    cutoff_at_origin: float = 0.5
    drop: tuple[tuple[str, float], ...] = (
        ("features", 0.1),
        ("peers", 0.15),
        ("attributes", 0.15),
        ("static_numbers", 0.1),
        ("panel_row", 0.2),
        ("siblings", 0.1),
        ("calendar", 0.1),
        ("rolling", 0.1),
        ("lags", 0.1),
        ("zeros", 0.1),
    )
    drop_always: tuple[str, ...] = ()
    summary_off: tuple[str, ...] = ()
    designate_target: bool = False
    path_share: float = 0.5
    max_cells: int | None = 4_000_000


@dataclass(frozen=True)
class OptimConfig:
    max_steps: int
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float | None = 0.999
    checkpoint_minutes: float = 10.0
    log_every: int = 50


def _plain(value):
    """Turn dataclass tuples into ordinary YAML/JSON sequences."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def _checked_mapping(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} requires a mapping")
    unknown = value.keys() - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"unknown keys in {cls.__name__}: {', '.join(map(str, unknown))}")
    return dict(value)


def _record(cls, value):
    return cls(**{key: _tuples(item) for key, item in _checked_mapping(cls, value).items()})


@dataclass(frozen=True)
class V5TrainConfig:
    model: V5Config
    sources: tuple[SourceConfig, ...]
    sampler: SamplerConfig
    optim: OptimConfig
    seed: int = 0
    workers: int = 0
    device: str = "auto"
    router: str = "learned"
    gates: str = "learned"

    def __post_init__(self) -> None:
        if self.router not in ("learned", "uniform", "rolled"):
            raise ValueError("router must be learned, uniform or rolled")
        if self.gates not in ("learned", "ones"):
            raise ValueError("gates must be learned or ones")
        if any(source.control == "missing_as_zero" for source in self.sources):
            if self.sampler.origins != 1:
                raise ValueError("missing_as_zero requires sampler.origins == 1")
        controlled = (
            any(source.control is not None for source in self.sources)
            or self.sampler.drop_always
            or self.sampler.summary_off
        )
        if controlled and (self.sampler.max_cells is not None or not self.sampler.designate_target):
            raise ValueError("controls require max_cells=None and designate_target=True")

    @classmethod
    def from_yaml(cls, path: Path) -> V5TrainConfig:
        data = _checked_mapping(cls, yaml.safe_load(path.read_text(encoding="utf-8")))
        for name, record_cls in (
            ("model", V5Config),
            ("sampler", SamplerConfig),
            ("optim", OptimConfig),
        ):
            if name in data:
                data[name] = _record(record_cls, data[name])
        if "sources" in data:
            data["sources"] = tuple(_record(SourceConfig, source) for source in data["sources"])
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(_plain(asdict(self)), sort_keys=False), encoding="utf-8")

    def canonical(self) -> dict:
        """Return detached semantics, independent of machine and corpus location."""
        data = _plain(asdict(self))
        del data["workers"], data["device"]
        for source in data["sources"]:
            del source["root"]
        return data


def identity(
    config: V5TrainConfig,
    *,
    code: str,
    corpus: str | None,
    init: str | None,
    env: str | None,
) -> str:
    record = {
        "config": config.canonical(),
        "code": code,
        "corpus": corpus,
        "init": init,
        "env": env,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


T1_MODEL = V5Config(
    d_model=192,
    nhead=4,
    enc_layers=3,
    dec_layers=2,
    dim_feedforward=768,
    n_experts=4,
    dropout=0.1,
)

T1_BASE = V5TrainConfig(
    model=T1_MODEL,
    sources=(),
    sampler=SamplerConfig(contexts=(256, 512), max_cells=None, designate_target=True),
    optim=OptimConfig(max_steps=3000, warmup_steps=200, checkpoint_minutes=10.0),
)
