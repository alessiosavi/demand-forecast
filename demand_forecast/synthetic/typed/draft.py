"""Working panel state, named streams, grouping and latent composition."""

import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from demand_forecast.data.episode import Item, Panel
from demand_forecast.data.frequency import to_offset
from demand_forecast.synthetic.typed.recipe import InputRecord, PanelRecipe

STATIC_MECHANISMS = (
    "static_category",
    "multi_label",
    "hierarchy",
    "static_number",
    "whole_number_code",
    "event_response",
    "label_lag",
)
INPUT_MECHANISMS = (
    *STATIC_MECHANISMS,
    "dynamic_category_mean",
    "dynamic_category_spread",
    "dynamic_category_timing",
    "known_driver",
    "planned_event",
    "delayed_past_driver",
)


@dataclass
class Streams:
    process: np.random.Generator
    inputs: np.random.Generator
    effect: np.random.Generator | None


@dataclass
class Contribution:
    add: list[np.ndarray] | None = None
    mult: list[np.ndarray] | None = None
    occ: list[np.ndarray] | None = None
    rho: np.ndarray | None = None
    stockout: list[np.ndarray] | None = None
    missing: list[np.ndarray] | None = None
    extra: dict[str, list[np.ndarray]] | None = None


@dataclass
class Result:
    inputs: tuple[str, ...]
    process: dict[str, object]
    params: dict[str, object]
    contribution: Contribution | None


@dataclass
class Draft:
    """Mechanisms attach their complete inputs before targets are composed."""

    recipe: PanelRecipe
    freq: str
    epoch: pd.Timestamp
    period: int
    long_lag: int | None
    grid_length: int
    starts: np.ndarray
    lengths: np.ndarray
    short: np.ndarray
    host: np.ndarray
    n_targets: int
    static: list[dict[str, object]]
    dynamic: list[dict[str, np.ndarray]]
    roles: dict[str, str]
    records: list[InputRecord]

    def grid(self, i: int) -> np.ndarray:
        return np.arange(self.starts[i], self.starts[i] + self.lengths[i], dtype=np.int64)

    def timestamps(self, i: int) -> pd.DatetimeIndex:
        offset = to_offset(self.freq)
        return pd.date_range(
            self.epoch + int(self.starts[i]) * offset, periods=int(self.lengths[i]), freq=offset
        )

    def add_static(self, name: str, values: list[object], kind: str, mechanism: str) -> None:
        for attributes, value in zip(self.static, values, strict=True):
            attributes[name] = value
        self.records.append(InputRecord(name, mechanism, "static", kind, None))

    def add_dynamic(
        self, name: str, columns: list[np.ndarray], kind: str, role: str, mechanism: str
    ) -> None:
        for dynamic, column in zip(self.dynamic, columns, strict=True):
            dynamic[name] = column
        self.roles[name] = role
        self.records.append(InputRecord(name, mechanism, "dynamic", kind, role))


RangeFn = Callable[[str], tuple[float, float]]
MechanismFn = Callable[[Draft, Streams, RangeFn], Result]


def stream(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(name.encode())]))


def token(rng: np.random.Generator, prefix: str) -> str:
    return f"{prefix}{int(rng.integers(0, 2**32)):08x}"


def grid(recipe: PanelRecipe) -> Draft:
    """Draw lengths and starts, placing short queries alongside long hosts."""
    rng = stream(recipe.seed, "grid")
    count = recipe.n_items
    short = np.zeros(count, dtype=bool)
    if (
        count >= 3
        and recipe.length[0] > 8
        and any(m in STATIC_MECHANISMS for m in recipe.mechanisms)
    ):
        n_short = min(count - 2, math.ceil(count / 10))
        short[rng.choice(count, n_short, replace=False)] = True
    lengths = np.empty(count, dtype=np.int64)
    lengths[~short] = np.clip(
        np.exp(
            rng.uniform(np.log(recipe.length[0]), np.log(recipe.length[1]), (~short).sum())
        ).astype(np.int64),
        *recipe.length,
    )
    lengths[short] = (
        rng.integers(8, min(63, recipe.length[0] - 1) + 1, short.sum()) if short.any() else []
    )
    grid_length = math.ceil(1.25 * int(lengths.max()))
    starts = np.zeros(count, dtype=np.int64)
    starts[~short] = rng.integers(0, grid_length - lengths[~short] + 1)
    host = np.full(count, -1, dtype=np.int64)
    eligible = np.flatnonzero(~short & (lengths >= 24))
    for i in np.flatnonzero(short):
        if eligible.size:
            h = int(rng.choice(eligible))
            host[i] = h
            starts[i] = rng.integers(starts[h] + 16, starts[h] + lengths[h])
        else:
            starts[i] = rng.integers(0, grid_length - lengths[i] + 1)
    grid_length = max(grid_length, int((starts + lengths).max()))
    offset = to_offset(recipe.freq)
    epoch = offset.rollforward(pd.Timestamp("2000-01-01")) + int(rng.integers(0, 5000)) * offset
    panel = Panel("typed", (), {}, recipe.freq, epoch, True)
    n_targets = (
        int(stream(recipe.seed, "multi_target:process").integers(2, 21))
        if "multi_target" in recipe.mechanisms
        else 1
    )
    return Draft(
        recipe,
        recipe.freq,
        epoch,
        panel.season if panel.season > 1 else panel.long_lag or 8,
        panel.long_lag,
        grid_length,
        starts,
        lengths,
        short,
        host,
        n_targets,
        [{} for _ in range(count)],
        [{} for _ in range(count)],
        {},
        [],
    )


def assign_groups(
    draft: Draft,
    process: np.random.Generator,
    inputs: np.random.Generator,
    *,
    max_labels: int = 12,
    n: int | None = None,
    singletons: bool = True,
) -> tuple[list[str], tuple[str, ...]]:
    """Draw group sizes from the process; assign items and tokens from inputs."""
    long = np.flatnonzero(~draft.short)
    n_single = math.floor(draft.recipe.singleton_share * len(long)) if singletons else 0
    remaining = len(long) - n_single
    if remaining:
        cap = max(1, min(max_labels, remaining // 2))
        n_shared = int(process.integers(min(3, cap), cap + 1)) if n is None else n
        sizes = 1 + process.multinomial(remaining - n_shared, process.dirichlet(np.ones(n_shared)))
    else:
        n_shared, sizes = 0, np.empty(0, dtype=np.int64)
    long = inputs.permutation(long)
    labels = [""] * draft.recipe.n_items
    singleton_labels = []
    for i in long[:n_single]:
        labels[i] = token(inputs, "l")
        singleton_labels.append(labels[i])
    shared_labels = [token(inputs, "l") for _ in range(n_shared)]
    assignments = np.repeat(np.arange(n_shared), sizes)
    for i, group in zip(long[n_single:], assignments, strict=True):
        labels[i] = shared_labels[group]
    for i in np.flatnonzero(draft.short):
        h = int(draft.host[i])
        if h >= 0 and labels[h] in shared_labels:
            labels[i] = labels[h]
        elif n_shared:
            labels[i] = str(inputs.choice(shared_labels, p=sizes / remaining))
        else:
            labels[i] = token(inputs, "l")
            singleton_labels.append(labels[i])
    return labels, tuple(singleton_labels)


def base(draft: Draft) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Draw scalars, innovations and uniforms without consuming any mechanism stream."""
    rng = stream(draft.recipe.seed, "base")
    noise_rng = stream(draft.recipe.seed, "noise")
    occupancy_rng = stream(draft.recipe.seed, "occupancy")
    parts, scalars = {}, {}
    for i, length in enumerate(draft.lengths):
        item_id = f"i{i:04d}"
        s = {
            name: rng.uniform(low, high, draft.n_targets)
            for name, low, high in (
                ("tau", -0.5, 0.5),
                ("a1", 0.05, 0.4),
                ("a2", 0, 0.2),
                ("f1", 0, 2 * np.pi),
                ("f2", 0, 2 * np.pi),
                ("sigma0", 0.05, 0.3),
                ("rho", 0, 0.8),
            )
        }
        s["level"] = np.exp(rng.uniform(0, np.log(1000), draft.n_targets))
        s["rho_effective"] = s["rho"].copy()
        g = draft.grid(i)
        baseline = (
            1
            + s["tau"][:, None] * (g - draft.grid_length / 2) / draft.grid_length
            + s["a1"][:, None] * np.sin(2 * np.pi * g / draft.period + s["f1"][:, None])
            + s["a2"][:, None] * np.sin(4 * np.pi * g / draft.period + s["f2"][:, None])
        )
        shape = (draft.n_targets, int(length))
        parts[item_id] = {
            "base": baseline,
            "xi": noise_rng.standard_normal(shape),
            "noise": np.zeros(shape),
            "u": occupancy_rng.random(shape),
            "occ": np.ones(shape, dtype=bool),
            "mult": np.ones(shape),
            "stockout": np.zeros(shape, dtype=bool),
            "missing": np.zeros(shape, dtype=bool),
        }
        scalars[item_id] = s
    return parts, scalars


def compose(
    draft: Draft, contributions: list[tuple[str, Contribution]]
) -> tuple[Panel, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Merge contribution slots and evaluate the single target identity."""
    parts, scalars = base(draft)
    items = []
    names = (
        ("target",)
        if draft.n_targets == 1
        else tuple(f"target_{j:02d}" for j in range(draft.n_targets))
    )
    for i in range(draft.recipe.n_items):
        item_id = f"i{i:04d}"
        p, s = parts[item_id], scalars[item_id]
        probability = np.ones_like(p["base"])
        latent = p["base"].copy()
        for mechanism, contribution in contributions:
            if contribution.extra is not None:
                for name, values in contribution.extra.items():
                    if name in p or name.startswith("add:"):
                        raise ValueError(
                            f"extra name {name!r} collides with a slot or another mechanism"
                        )
                    p[name] = values[i]
            if contribution.add is not None:
                add = np.broadcast_to(contribution.add[i], latent.shape).astype(
                    np.float64, copy=True
                )
                p[f"add:{mechanism}"] = add
                latent += add
            if contribution.mult is not None:
                p["mult"] *= contribution.mult[i]
            if contribution.occ is not None:
                probability *= contribution.occ[i]
            if contribution.rho is not None:
                s["rho_effective"][:] = contribution.rho[i]
            for name in ("stockout", "missing"):
                values = getattr(contribution, name)
                if values is not None:
                    p[name] |= values[i]
        for j, rho in enumerate(s["rho_effective"]):
            p["noise"][j] = lfilter([np.sqrt(1 - rho**2)], [1, -rho], p["xi"][j])
        p["occ"] = (p["u"] < probability) & ~p["stockout"]
        latent += s["sigma0"][:, None] * p["mult"] * p["noise"]
        y = s["level"][:, None] * p["occ"] * latent
        y[p["missing"]] = np.nan
        items.append(
            Item(
                item_id,
                int(draft.starts[i]),
                dict(zip(names, y, strict=True)),
                draft.dynamic[i],
                draft.static[i],
            )
        )
    return (
        Panel("typed", tuple(items), dict(draft.roles), draft.freq, draft.epoch, True),
        parts,
        scalars,
    )
