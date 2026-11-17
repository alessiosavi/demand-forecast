"""Run-based categorical inputs with mean, spread or delayed pulse effects."""

import math
from functools import partial

import numpy as np

from demand_forecast.synthetic.typed.draft import (
    Contribution,
    Draft,
    MechanismFn,
    RangeFn,
    Result,
    Streams,
    token,
)


def _chain(
    length: int,
    rng: np.random.Generator,
    dwell: float,
    weights: np.ndarray,
    *,
    initial: int | None = None,
) -> np.ndarray:
    """Fill geometric runs, choosing each new value with the current one excluded."""
    column = np.empty(length, dtype=np.int64)
    current = (
        int(rng.choice(len(weights), p=weights / weights.sum())) if initial is None else initial
    )
    start = 0
    while start < length:
        end = min(length, start + int(rng.geometric(1 / dwell)))
        column[start:end] = current
        start = end
        if start < length:
            odds = weights.copy()
            odds[current] = 0
            current = int(rng.choice(len(odds), p=odds / odds.sum()))
    return column


def _category_inputs(
    draft: Draft, streams: Streams, mechanism: str
) -> tuple[str, tuple[str, ...], tuple[str, ...], list[np.ndarray], dict[str, object]]:
    """Draw the process once and realise fresh label tokens and chains from inputs."""
    process, inputs = streams.process, streams.inputs
    role = str(process.choice(("known", "past")))
    cap = min(8, draft.period) if mechanism == "dynamic_category_timing" else 8
    n_values = int(process.integers(4, cap + 1))
    dwell = max(2.0, float(process.uniform(0.5, 3) * draft.period))
    rare, late = (int(v) for v in process.choice(n_values, 2, replace=False))
    params = {"n_values": n_values, "dwell": dwell, "role": role, "rare": rare, "late": late}
    name = token(inputs, "c")
    values = tuple(token(inputs, "l") for _ in range(n_values))
    eligible = np.flatnonzero(draft.lengths >= 20)
    late_mask = np.zeros(draft.recipe.n_items, dtype=bool)
    late_mask[inputs.choice(eligible, math.ceil(0.2 * len(eligible)), replace=False)] = True
    late_items = tuple(f"i{i:04d}" for i in np.flatnonzero(late_mask))
    weights = np.ones(n_values)
    weights[rare] = 0.1
    prefix_weights = weights.copy()
    prefix_weights[late] = 0
    columns = []
    for i, length in enumerate(draft.lengths):
        length = int(length)
        if late_mask[i]:
            boundary = 4 * length // 5
            column = np.concatenate(
                (
                    _chain(boundary, inputs, dwell, prefix_weights),
                    _chain(length - boundary, inputs, dwell, weights, initial=late),
                )
            )
        else:
            column = _chain(length, inputs, dwell, weights)
        columns.append(column)
    labels = np.asarray(values, dtype=object)
    draft.add_dynamic(name, [labels[c] for c in columns], "category", role, mechanism)
    return name, values, late_items, columns, params


def _category(draft: Draft, streams: Streams, ranges: RangeFn, *, mechanism: str) -> Result:
    name, values, late_items, columns, process = _category_inputs(draft, streams, mechanism)
    if streams.effect is None:
        return Result((name,), process, {}, None)

    rng = streams.effect
    params: dict[str, object] = {"values": values, "late_items": late_items}
    if mechanism == "dynamic_category_timing":
        pulse = float(rng.uniform(*ranges("pulse")))
        width = 2
        delays = rng.choice(draft.period, len(values), replace=False)
        params.update(
            pulse=pulse,
            width=width,
            delay={value: int(delay) for value, delay in zip(values, delays, strict=True)},
        )
        additions = []
        for column in columns:
            changes = np.flatnonzero(column[1:] != column[:-1]) + 1
            starts = changes + delays[column[changes]]
            starts = starts[starts < len(column)]
            difference = np.zeros(len(column) + 1, dtype=np.int64)
            np.add.at(difference, starts, 1)
            np.add.at(difference, np.minimum(starts + width, len(column)), -1)
            additions.append(pulse * np.cumsum(difference[:-1]))
        contribution = Contribution(add=additions)
    else:
        effect = float(rng.uniform(*ranges("effect")))
        weights = rng.uniform(-1, 1, len(values))
        response = effect * weights / np.max(np.abs(weights))
        spread = mechanism == "dynamic_category_spread"
        if spread:
            response = np.exp(response)
        params.update(
            effect=effect,
            **{
                "factor" if spread else "mean": {
                    value: float(r) for value, r in zip(values, response, strict=True)
                }
            },
        )
        arrays = [response[column] for column in columns]
        contribution = Contribution(mult=arrays) if spread else Contribution(add=arrays)
    return Result((name,), process, params, contribution)


CATEGORY: dict[str, MechanismFn] = {
    mechanism: partial(_category, mechanism=mechanism)
    for mechanism in (
        "dynamic_category_mean",
        "dynamic_category_spread",
        "dynamic_category_timing",
    )
}
