"""Continuous drivers, planned events and responses selected by static labels."""

import math
from collections import Counter

import numpy as np
from scipy.signal import lfilter

from demand_forecast.synthetic.typed.draft import (
    Contribution,
    Draft,
    MechanismFn,
    RangeFn,
    Result,
    Streams,
    assign_groups,
    token,
)


def _standardise(values: np.ndarray) -> np.ndarray:
    deviation = values.std()
    return (values - values.mean()) / deviation if deviation else np.zeros_like(values)


def _ar(rng: np.random.Generator, length: int, rho: float) -> np.ndarray:
    return lfilter([math.sqrt(1 - rho**2)], [1, -rho], rng.standard_normal(length))


def _events(rng: np.random.Generator, length: int, rate: float) -> np.ndarray:
    """Union independently started runs, including starts inside another run."""
    starts = np.flatnonzero(rng.random(length) < rate)
    ends = np.minimum(length, starts + rng.integers(1, 6, len(starts)))
    changes = np.zeros(length + 1, dtype=np.int64)
    np.add.at(changes, starts, 1)
    np.add.at(changes, ends, -1)
    return (np.cumsum(changes[:-1]) > 0).astype(np.float64)


def _signed(rng: np.random.Generator, bounds: tuple[float, float]) -> tuple[float, int]:
    return float(rng.uniform(*bounds)), int(rng.choice((-1, 1)))


def _groups(draft: Draft, streams: Streams) -> tuple[list[str], tuple[str, ...], dict[str, object]]:
    labels, singletons = assign_groups(draft, streams.process, streams.inputs)
    counts = Counter(label for i, label in enumerate(labels) if not draft.short[i])
    # Counts describe the process without tying it to this realisation's tokens.
    process = {"n_labels": len(set(labels)), "group_sizes": tuple(sorted(counts.values()))}
    return labels, singletons, process


def known_driver(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    steps = int(streams.process.integers(0, 4))
    process = {"steps": steps, "rho": 0.9, "role": "known"}
    name = token(streams.inputs, "c")
    columns = []
    for length in draft.lengths:
        values = _ar(streams.inputs, int(length), 0.9)
        for _ in range(steps):
            point = int(streams.inputs.integers(length))
            size, sign = _signed(streams.inputs, (0.5, 2))
            values[point:] += sign * size
        columns.append(values)
    draft.add_dynamic(name, columns, "number", "known", "known_driver")
    if streams.effect is None:
        return Result((name,), process, {}, None)
    beta, sign = _signed(streams.effect, ranges("beta"))
    return Result(
        (name,),
        process,
        {"beta": beta, "beta_sign": sign},
        Contribution(add=[sign * beta * np.tanh(_standardise(x)) for x in columns]),
    )


def planned_event(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    rate = float(streams.process.uniform(0.02, 0.08))
    process = {"rate": rate, "role": "known"}
    name = token(streams.inputs, "c")
    eligible = np.flatnonzero(draft.lengths >= 10)
    late = streams.inputs.choice(eligible, math.ceil(0.1 * len(eligible)), replace=False)
    columns = []
    for i, length in enumerate(draft.lengths):
        boundary = 4 * int(length) // 5 if i in late else 0
        flag = np.zeros(length, dtype=np.float64)
        flag[boundary:] = _events(streams.inputs, int(length) - boundary, rate)
        if i in late:
            point = int(streams.inputs.integers(boundary, length))
            width = int(streams.inputs.integers(1, 6))
            flag[point : point + width] = 1.0
        columns.append(flag)
    draft.add_dynamic(name, columns, "flag", "known", "planned_event")
    if streams.effect is None:
        return Result((name,), process, {}, None)
    size, sign = _signed(streams.effect, ranges("size"))
    carry = float(streams.effect.uniform(-0.5, 0))
    params = {
        "size": size,
        "size_sign": sign,
        "carry": carry,
        "late_items": tuple(f"i{i:04d}" for i in sorted(late.tolist())),
    }
    add = [sign * size * (x + carry * np.r_[0.0, x[:-1]] * (1 - x)) for x in columns]
    return Result((name,), process, params, Contribution(add=add))


def delayed_past_driver(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    process = {"rho": 0.7, "role": "past"}
    name = token(streams.inputs, "c")
    columns = [_ar(streams.inputs, int(length), 0.7) for length in draft.lengths]
    draft.add_dynamic(name, columns, "number", "past", "delayed_past_driver")
    if streams.effect is None:
        return Result((name,), process, {}, None)
    beta, sign = _signed(streams.effect, ranges("beta"))
    lag_periods = float(streams.effect.uniform(*ranges("lag_periods")))
    lag = max(1, math.ceil(lag_periods * draft.period))
    add = []
    for values in columns:
        shifted = np.zeros_like(values)
        shifted[lag:] = _standardise(values)[:-lag]
        add.append(sign * beta * shifted)
    return Result(
        (name,),
        process,
        {"beta": beta, "beta_sign": sign, "lag_periods": lag_periods, "lag": lag},
        Contribution(add=add),
    )


def event_response(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    rate = float(streams.process.uniform(0.02, 0.08))
    attribute, column = token(streams.inputs, "a"), token(streams.inputs, "c")
    labels, singletons, groups = _groups(draft, streams)
    process = {"rate": rate, "role": "known", **groups}
    flag = _events(streams.inputs, draft.grid_length, rate)
    columns = [flag[draft.grid(i)].copy() for i in range(draft.recipe.n_items)]
    draft.add_static(attribute, labels, "category", "event_response")
    draft.add_dynamic(column, columns, "flag", "known", "event_response")
    if streams.effect is None:
        return Result((attribute, column), process, {}, None)
    sizes, signs = {}, {}
    for label in sorted(set(labels)):
        sizes[label], signs[label] = _signed(streams.effect, ranges("size"))
    return Result(
        (attribute, column),
        process,
        {"size": sizes, "size_sign": signs, "singletons": singletons},
        Contribution(
            add=[signs[label] * sizes[label] * x for label, x in zip(labels, columns, strict=True)]
        ),
    )


def label_lag(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    attribute, column = token(streams.inputs, "a"), token(streams.inputs, "c")
    labels, singletons, groups = _groups(draft, streams)
    process = {"rho": 0.7, "role": "past", **groups}
    driver = _ar(streams.inputs, draft.grid_length, 0.7)
    columns = [driver[draft.grid(i)].copy() for i in range(draft.recipe.n_items)]
    draft.add_static(attribute, labels, "category", "label_lag")
    draft.add_dynamic(column, columns, "number", "past", "label_lag")
    if streams.effect is None:
        return Result((attribute, column), process, {}, None)
    beta, sign = _signed(streams.effect, ranges("beta"))
    lags = {
        label: int(streams.effect.integers(1, draft.period + 1)) for label in sorted(set(labels))
    }
    z = _standardise(driver)
    add = []
    for i, label in enumerate(labels):
        values = z[draft.grid(i)]
        shifted = np.zeros_like(values)
        lag = lags[label]
        # Shift within the item's own window; earlier grid history is unavailable.
        shifted[lag:] = values[:-lag]
        add.append(sign * beta * shifted)
    return Result(
        (attribute, column),
        process,
        {
            "beta": beta,
            "beta_sign": sign,
            "lag": lags,
            "singletons": singletons,
            "grid_driver": tuple(float(value) for value in driver),
        },
        Contribution(add=add),
    )


DRIVER: dict[str, MechanismFn] = {
    "known_driver": known_driver,
    "planned_event": planned_event,
    "delayed_past_driver": delayed_past_driver,
    "event_response": event_response,
    "label_lag": label_lag,
}
