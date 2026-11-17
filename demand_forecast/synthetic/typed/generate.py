"""Mechanism orchestration and partial availability of typed inputs."""

import math
from copy import deepcopy
from dataclasses import replace

import numpy as np

from demand_forecast.data.episode import Item, Panel
from demand_forecast.synthetic.typed.category_effects import CATEGORY
from demand_forecast.synthetic.typed.draft import (
    INPUT_MECHANISMS,
    Contribution,
    MechanismFn,
    RangeFn,
    Streams,
    compose,
    grid,
    stream,
)
from demand_forecast.synthetic.typed.driver_effects import DRIVER
from demand_forecast.synthetic.typed.recipe import (
    MECHANISMS,
    RANGES,
    GeneratedPanel,
    InputRecord,
    PanelRecipe,
)
from demand_forecast.synthetic.typed.static_effects import STATIC
from demand_forecast.synthetic.typed.structure import STRUCTURE

REGISTRY: dict[str, MechanismFn] = {**STRUCTURE, **STATIC, **CATEGORY, **DRIVER}


def _range_fn(mechanism: str, heldout: bool) -> RangeFn:
    return lambda key: RANGES[mechanism][key][int(heldout)]


def _availability(
    complete: Panel, recipe: PanelRecipe, records: tuple[InputRecord, ...]
) -> tuple[Panel, dict[str, float]]:
    rng = stream(recipe.seed, "availability")
    shares = {"static": float(rng.uniform(0, 0.3)), "dynamic": float(rng.uniform(0, 0.15))}
    items = [
        Item(
            item.item_id,
            item.start,
            {name: values.copy() for name, values in item.targets.items()},
            {name: values.copy() for name, values in item.dynamic.items()},
            deepcopy(item.static),
        )
        for item in complete.items
    ]
    if not recipe.complete_inputs:
        for record in records:
            rng = stream(recipe.seed, f"availability:{record.name}")
            if record.where == "static":
                selected = rng.choice(
                    len(items), math.floor(shares["static"] * len(items)), replace=False
                )
                for i in selected:
                    items[i].static[record.name] = [] if record.kind == "multilabel" else None
            else:
                selected = rng.choice(len(items), math.ceil(0.2 * len(items)), replace=False)
                for i, item in enumerate(items):
                    column = item.dynamic[record.name]
                    stretch = (
                        max(1, int(rng.uniform(0.1, 0.5) * len(column))) if i in selected else 0
                    )
                    mask = np.zeros(len(column), dtype=bool)
                    mask[:stretch] = True
                    suffix = mask[stretch:]
                    while np.count_nonzero(suffix) < shares["dynamic"] * len(suffix):
                        start = int(rng.integers(len(suffix)))
                        size = int(rng.integers(1, 6))
                        suffix[start : start + size] = True
                    column[mask] = None if column.dtype == object else np.nan
    return replace(complete, items=tuple(items)), shares


def generate_typed_panel(recipe: PanelRecipe, *, without: tuple[str, ...] = ()) -> GeneratedPanel:
    """Generate complete effects first, then hide inputs without changing targets."""
    if "multi_target" in without or any(m not in recipe.mechanisms for m in without):
        raise ValueError("without must name recipe mechanisms other than multi_target")
    draft = grid(recipe)
    params, contributions = {}, []
    for mechanism in MECHANISMS:
        if mechanism not in recipe.mechanisms:
            continue
        result = REGISTRY[mechanism](
            draft,
            Streams(
                stream(recipe.seed, f"{mechanism}:process"),
                stream(recipe.seed, f"{mechanism}:inputs"),
                stream(recipe.seed, f"{mechanism}:effect"),
            ),
            _range_fn(mechanism, recipe.heldout),
        )
        for key in ("process", "inputs"):
            if key in result.params:
                raise ValueError(f"Mechanism {mechanism!r} uses reserved parameter key {key!r}")
        params[mechanism] = {**result.params, "process": result.process, "inputs": result.inputs}
        if result.contribution is not None:
            contribution = (
                Contribution(extra=result.contribution.extra)
                if mechanism in without
                else result.contribution
            )
            contributions.append((mechanism, contribution))
    pool = tuple(
        m
        for m in MECHANISMS
        if m in recipe.mechanisms
        and m in INPUT_MECHANISMS
        and (recipe.nuisance_from is None or m in recipe.nuisance_from)
    )
    rng = stream(recipe.seed, "nuisance")
    if pool:
        for k in range(recipe.n_nuisance):
            mechanism = str(rng.choice(pool))
            first_record = len(draft.records)
            result = REGISTRY[mechanism](
                draft,
                Streams(
                    stream(recipe.seed, f"{mechanism}:process"),
                    stream(recipe.seed, f"nuisance:{mechanism}:{k}"),
                    None,
                ),
                _range_fn(mechanism, recipe.heldout),
            )
            for i in range(first_record, len(draft.records)):
                draft.records[i] = replace(
                    draft.records[i], mechanism="nuisance", twin=mechanism, process=result.process
                )
    complete, parts, scalars = compose(draft, contributions)
    records = tuple(draft.records)
    panel, availability = _availability(complete, recipe, records)
    panel.validate()
    complete.validate()
    return GeneratedPanel(
        panel,
        complete,
        recipe,
        records,
        params,
        parts,
        scalars,
        tuple(f"i{i:04d}" for i in np.flatnonzero(draft.short)),
        {f"i{i:04d}": f"i{draft.host[i]:04d}" for i in np.flatnonzero(draft.host >= 0)},
        availability,
        draft.grid_length,
        draft.period,
        draft.long_lag,
    )
