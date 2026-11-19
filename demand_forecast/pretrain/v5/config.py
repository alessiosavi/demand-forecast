"""Portable training settings and identities for native v5 experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path

import yaml

from demand_forecast.data.episode.builder import DROPPABLE
from demand_forecast.data.episode.schema import MAX_CONTEXT, MAX_HORIZON, PATCH, SUMMARY_FEATURES
from demand_forecast.models.v5.config import V5Config
from demand_forecast.synthetic.typed.draft import INPUT_MECHANISMS
from demand_forecast.synthetic.typed.recipe import MECHANISMS


def _flag(self, name: str) -> None:
    """A switch must be a real boolean: a quoted "false" from yaml is truthy."""
    if not isinstance(getattr(self, name), bool):
        raise ValueError(f"{name} must be a boolean")


def _numeric(value, name: str, *, integer: bool = False, minimum=None, maximum=None):
    """Normalize finite numbers without admitting booleans or truncating counts."""
    try:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError
        normalized = int(value) if integer else float(value)
        if integer and normalized != value:
            raise ValueError
        if minimum is not None and normalized < minimum:
            raise ValueError
        if maximum is not None and normalized > maximum:
            raise ValueError
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError(
            f"{name} must be a finite {'integer' if integer else 'number'} in range"
        ) from exc
    # JSON distinguishes -0.0 from 0.0 even though Python equality does not.
    return normalized if normalized else (0 if integer else 0.0)


def _normalize(record, name: str, **bounds) -> None:
    object.__setattr__(record, name, _numeric(getattr(record, name), name, **bounds))


def _range(value, name: str, *, minimum=1, maximum=None) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} requires two ordered bounds")
    lower, upper = (
        _numeric(v, name, integer=True, minimum=minimum, maximum=maximum) for v in value
    )
    if lower > upper:
        raise ValueError(f"{name} requires ordered bounds")
    return lower, upper


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    weight: float
    root: str | None = None
    heldout: bool = False
    recipes: tuple[tuple[str, ...], ...] = ()
    freq: str | None = None
    length: tuple[int, int] = (64, MAX_CONTEXT)
    max_items: int = 48
    n_nuisance: int = 0
    nuisance_from: tuple[str, ...] | None = None
    singleton_share: float = 0.0
    control: str | None = None
    control_inputs: str = "all"
    folder_power: float | None = None
    plain_share: float = 0.0

    def __post_init__(self) -> None:
        _flag(self, "heldout")
        for name in ("weight", "singleton_share"):
            _normalize(self, name, minimum=0, maximum=1 if name == "singleton_share" else None)
        if self.weight == 0:
            raise ValueError("weight must be positive")
        _normalize(self, "max_items", integer=True, minimum=1)
        _normalize(self, "n_nuisance", integer=True, minimum=0)
        if self.folder_power is not None:
            _normalize(self, "folder_power", minimum=0, maximum=1)
        _normalize(self, "plain_share", minimum=0, maximum=1)
        object.__setattr__(self, "length", _range(self.length, "length", minimum=8))
        if self.kind not in ("typed", "corpus", "upstream"):
            raise ValueError("kind must be typed, corpus or upstream")
        if self.kind == "typed":
            if self.folder_power is not None:
                raise ValueError("folder_power is only valid for corpus sources")
            if self.plain_share != 0.0:
                raise ValueError("plain_share is only valid for upstream sources")
        else:
            source_kinds = {"folder_power": "corpus", "plain_share": "upstream"}
            for field in fields(self):
                if field.name in ("kind", "weight", "root"):
                    continue
                value = getattr(self, field.name)
                # The sampler fills missing history on any panel; the other controls
                # act inside the typed generator and stay typed-only.
                if field.name == "control" and value == "missing_as_zero":
                    continue
                kind = source_kinds.get(field.name, "typed")
                if value != field.default and self.kind != kind:
                    raise ValueError(f"{field.name} is only valid for {kind} sources")
        if self.root is not None and self.kind != "corpus":
            raise ValueError("root is only valid for corpus sources")
        if self.control not in (None, "shuffle", "no_dates", "missing_as_zero"):
            raise ValueError("control must be None, shuffle, no_dates or missing_as_zero")
        if self.control_inputs not in ("all", "attribute", "column"):
            raise ValueError("control_inputs must be all, attribute or column")
        if not self.recipes:
            if self.nuisance_from is not None:
                raise ValueError(
                    "nuisance_from requires recipes and is ignored by the sampled recipe mix"
                )
            if self.n_nuisance > 0:
                raise ValueError(
                    "n_nuisance requires recipes and is ignored by the sampled recipe mix"
                )
            return
        for recipe in self.recipes:
            for mechanism in recipe:
                if mechanism not in MECHANISMS:
                    raise ValueError(f"recipes contains unknown mechanism {mechanism!r}")
        for mechanism in self.nuisance_from or ():
            if mechanism not in INPUT_MECHANISMS:
                raise ValueError(f"nuisance_from contains unsupported mechanism {mechanism!r}")
            for recipe in self.recipes:
                if mechanism not in recipe:
                    raise ValueError(
                        f"nuisance_from mechanism {mechanism!r} is missing from recipe {recipe!r}"
                    )


@dataclass(frozen=True)
class SamplerConfig:
    episodes_per_batch: int = 32
    episodes_per_panel: int = 8
    origins: int = 4
    origin_stride: int = 2
    contexts: tuple[int, ...] = (256, 512, 1024)
    horizon: tuple[int, int] = (PATCH, MAX_HORIZON)
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
    # Zero scores below this level. Within each budget's existing priorities,
    # zero scores rank last and are cut first when the budget binds.
    screen_no_evidence: float = 0.0
    cold_share: float = 0.0
    cold_steps: tuple[int, int] = (3, 4)
    first_event_only: bool = False
    screen_fair_spread: bool = False

    def __post_init__(self) -> None:
        _flag(self, "designate_target")
        _flag(self, "first_event_only")
        _flag(self, "screen_fair_spread")
        for name in ("episodes_per_batch", "episodes_per_panel", "origins", "origin_stride"):
            _normalize(self, name, integer=True, minimum=1)
        if not self.contexts:
            raise ValueError("contexts must be non-empty")
        contexts = tuple(
            _numeric(v, "contexts", integer=True, minimum=1, maximum=MAX_CONTEXT)
            for v in self.contexts
        )
        if any(v % PATCH for v in contexts):
            raise ValueError(f"contexts must be positive multiples of {PATCH}")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "horizon", _range(self.horizon, "horizon", maximum=MAX_HORIZON))
        for name in ("cutoff_at_origin", "path_share", "cold_share"):
            _normalize(self, name, minimum=0, maximum=1)
        object.__setattr__(
            self, "cold_steps", _range(self.cold_steps, "cold_steps", minimum=3, maximum=PATCH - 1)
        )
        _normalize(self, "screen_no_evidence", minimum=0)
        if self.screen_no_evidence >= 1:
            raise ValueError("screen_no_evidence must be in [0, 1)")
        if self.max_cells is not None:
            _normalize(self, "max_cells", integer=True, minimum=1)
        drops = []
        for name, probability in self.drop:
            if name not in DROPPABLE or any(name == previous for previous, _ in drops):
                raise ValueError("drop requires known, unique family names")
            drops.append((name, _numeric(probability, "drop", minimum=0, maximum=1)))
        object.__setattr__(self, "drop", tuple(drops))
        for field, allowed in (("drop_always", DROPPABLE), ("summary_off", SUMMARY_FEATURES)):
            if any(name not in allowed for name in getattr(self, field)):
                raise ValueError(f"{field} requires known names")
            object.__setattr__(self, field, tuple(getattr(self, field)))


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

    def __post_init__(self) -> None:
        _flag(self, "amp")
        for name in ("max_steps", "log_every", "warmup_steps"):
            _normalize(self, name, integer=True, minimum=0 if name == "warmup_steps" else 1)
        for name in ("learning_rate", "weight_decay", "grad_clip", "checkpoint_minutes"):
            _normalize(self, name, minimum=0)
            if name != "weight_decay" and getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if self.ema_decay is not None:
            _normalize(self, "ema_decay", minimum=0, maximum=1)
            if self.ema_decay in (0, 1):
                raise ValueError("ema_decay must be None or in (0, 1)")


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
        _normalize(self, "seed", integer=True, minimum=0)
        _normalize(self, "workers", integer=True, minimum=0)
        if not isinstance(self.device, str) or not re.fullmatch(
            r"auto|cpu|cuda(?::[0-9]+)?|mps", self.device
        ):
            raise ValueError("device must be auto, cpu, cuda, cuda:<n> or mps")
        if self.router not in ("learned", "uniform", "rolled"):
            raise ValueError("router must be learned, uniform or rolled")
        if self.gates not in ("learned", "ones"):
            raise ValueError("gates must be learned or ones")
        if self.model.expert_mode == "dense" and self.router != "learned":
            raise ValueError("the dense expert mode requires router=learned")
        if self.router == "rolled":
            if self.sampler.episodes_per_batch < 2:
                raise ValueError(
                    "the rolled router requires sampler.episodes_per_batch of at least two"
                )
            if self.sampler.max_cells is not None:
                raise ValueError("the rolled router requires sampler.max_cells=None")
        if any(source.control == "missing_as_zero" for source in self.sources):
            if self.sampler.cold_share > 0:
                raise ValueError("missing_as_zero requires sampler.cold_share == 0")
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

    def _serialized(self) -> dict:
        """Omit options at their defaults so that existing runs keep their stored configuration."""
        data = _plain(asdict(self))
        if self.model.expert_mode == "routed":
            del data["model"]["expert_mode"]
        if not self.sampler.screen_fair_spread:
            del data["sampler"]["screen_fair_spread"]
        for source in data["sources"]:
            if source["folder_power"] is None:
                del source["folder_power"]
            if source["plain_share"] == 0.0:
                del source["plain_share"]
        return data

    def to_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self._serialized(), sort_keys=False), encoding="utf-8")

    def canonical(self) -> dict:
        """Return detached semantics, independent of machine and corpus location."""
        data = self._serialized()
        del data["workers"], data["device"]
        if self.sampler.screen_no_evidence == 0.0:
            del data["sampler"]["screen_no_evidence"]
        if self.sampler.cold_share == 0.0:
            del data["sampler"]["cold_share"]
        if self.sampler.cold_steps == (3, 4):
            del data["sampler"]["cold_steps"]
        if not self.sampler.first_event_only:
            del data["sampler"]["first_event_only"]
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
