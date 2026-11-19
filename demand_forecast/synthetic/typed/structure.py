"""Latent mechanisms driven by the grid and target history, without typed inputs."""

import math

import numpy as np
from scipy.signal import lfilter

from demand_forecast.data.episode import FREQ_CLASSES
from demand_forecast.synthetic.typed.draft import (
    Contribution,
    Draft,
    MechanismFn,
    RangeFn,
    Result,
    Streams,
)


def _multi_target(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    # grid() already drew the shape from this mechanism's process stream.
    process = {"n_targets": draft.n_targets}
    rng = streams.effect
    if rng is None:
        return Result((), process, {}, None)
    loading = rng.uniform(0.3, 1.0, draft.n_targets)
    signs = rng.choice((-1, 1), draft.n_targets)
    lo, hi = ranges("lag")
    lags = rng.integers(int(lo), int(hi) + 1, draft.n_targets)
    innovations, factors, additions = [], [], []
    for length in draft.lengths:
        e = streams.inputs.standard_normal(int(length))
        factor = lfilter([np.sqrt(1 - 0.95**2)], [1, -0.95], e)
        local = np.arange(length)[None, :] - lags[:, None]
        add = loading[:, None] * signs[:, None] * factor[np.maximum(local, 0)]
        add[local < 0] = 0.0
        innovations.append(e)
        factors.append(factor)
        additions.append(add)
    return Result(
        (),
        process,
        {
            "loading": tuple(loading.tolist()),
            "loading_sign": tuple(signs.tolist()),
            "lag": tuple(lags.tolist()),
        },
        Contribution(add=additions, extra={"factor_e": innovations, "factor": factors}),
    )


def _calendar(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    rng = streams.effect
    if rng is None:
        return Result((), {}, {}, None)
    freq_class = FREQ_CLASSES[draft.freq_class]
    fields = {
        "subhourly": ("hour", "dow"),
        "hourly": ("hour", "dow"),
        "daily": ("dow", "dom", "month"),
    }.get(freq_class, ("month",))
    # Keep the original day-of-week and month draws for existing daily seeds.
    draw_order = ("dow", "month", "dom") if freq_class == "daily" else fields
    sizes = {"hour": 24, "dow": 7, "dom": 31, "month": 12}
    amplitudes, tables = {}, {}
    for field in draw_order:
        amplitude = float(rng.uniform(*ranges("amplitude")))
        weights = rng.uniform(-1, 1, sizes[field])
        weights -= weights.mean()
        table = amplitude * weights / np.max(np.abs(weights))
        amplitudes[field] = amplitude
        tables[field] = tuple(table.tolist())
    additions = []
    for i in range(draft.recipe.n_items):
        dates = draft.timestamps(i)
        indices = {"hour": dates.hour, "dow": dates.day_of_week, "month": dates.month - 1}
        if freq_class == "daily":
            indices["dom"] = dates.day - 1
        additions.append(sum(np.array(tables[field])[indices[field]] for field in fields))
    return Result(
        (),
        {},
        {"fields": fields, "amplitude": amplitudes, "tables": tables},
        Contribution(add=additions),
    )


def _seasonal_lag(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    rng = streams.effect
    if rng is None:
        return Result((), {}, {}, None)
    lag = draft.long_lag
    if lag is None:
        raise ValueError("seasonal_lag requires a frequency with a long lag")
    amplitude = float(rng.uniform(*ranges("amplitude")))
    bumps = int(rng.integers(1, 4))
    centres = rng.integers(0, lag, bumps)
    width = max(1.0, lag / 20)
    distance = np.abs(np.arange(lag)[:, None] - centres[None, :])
    distance = np.minimum(distance, lag - distance)
    profile = np.exp(-0.5 * (distance / width) ** 2).max(axis=1)
    return Result(
        (),
        {},
        {
            "amplitude": amplitude,
            "bumps": bumps,
            "centres": tuple(centres.tolist()),
            "width": width,
            "profile": tuple(profile.tolist()),
        },
        Contribution(
            add=[amplitude * profile[draft.grid(i) % lag] for i in range(draft.recipe.n_items)]
        ),
    )


def _intermittent(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    rng = streams.effect
    if rng is None:
        return Result((), {}, {}, None)
    probabilities = rng.uniform(*ranges("p"), draft.recipe.n_items)
    return Result(
        (),
        {},
        {"p": {f"i{i:04d}": float(p) for i, p in enumerate(probabilities)}},
        Contribution(
            occ=[np.full(length, p) for length, p in zip(draft.lengths, probabilities, strict=True)]
        ),
    )


def _level_shift(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    rng = streams.effect
    if rng is None:
        return Result((), {}, {}, None)
    points, sizes, signs, additions = {}, {}, {}, []
    for i, length in enumerate(draft.lengths):
        item_id = f"i{i:04d}"
        n = int(rng.integers(1, 3))
        local = rng.integers(math.ceil(0.1 * length), math.floor(0.9 * length) + 1, n)
        absolute = local + draft.starts[i]
        magnitude = rng.uniform(*ranges("size"), n)
        sign = rng.choice((-1, 1), n)
        points[item_id] = tuple(absolute.tolist())
        sizes[item_id] = tuple(magnitude.tolist())
        signs[item_id] = tuple(sign.tolist())
        additions.append(
            (
                magnitude[:, None] * sign[:, None] * (draft.grid(i)[None, :] >= absolute[:, None])
            ).sum(axis=0)
        )
    return Result(
        (), {}, {"point": points, "size": sizes, "size_sign": signs}, Contribution(add=additions)
    )


def _stockout_runs(length: int, rng: np.random.Generator, rate: float) -> np.ndarray:
    starts = np.flatnonzero(rng.random(length) < rate)
    stops = np.minimum(length, starts + rng.integers(1, 6, len(starts)))
    changes = np.zeros(length + 1, dtype=np.int64)
    np.add.at(changes, starts, 1)
    np.add.at(changes, stops, -1)
    return changes[:-1].cumsum() > 0


def _missing_vs_zero(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    process = {
        "rate": float(streams.process.uniform(0.01, 0.05)),
        "missing_share": float(streams.process.uniform(0.02, 0.15)),
    }
    rng = streams.effect
    if rng is None:
        return Result((), process, {}, None)
    rebound = float(rng.uniform(*ranges("rebound")))
    stockouts, masks, additions = [], [], []
    for length in draft.lengths:
        stockout = _stockout_runs(int(length), streams.inputs, process["rate"])
        missing = np.zeros(length, dtype=bool)
        # Each iteration draws a run, rather than advancing one series step.
        while np.count_nonzero(missing) < process["missing_share"] * length:
            start = int(streams.inputs.integers(length))
            size = int(streams.inputs.integers(1, 6))
            missing[start : start + size] = True
        previous = np.concatenate(([False], stockout[:-1]))
        stockouts.append(stockout)
        masks.append(missing)
        additions.append(rebound * (previous & ~stockout))
    return Result(
        (),
        process,
        {"rebound": rebound},
        Contribution(add=additions, stockout=stockouts, missing=masks),
    )


STRUCTURE: dict[str, MechanismFn] = {
    "multi_target": _multi_target,
    "calendar": _calendar,
    "seasonal_lag": _seasonal_lag,
    "intermittent": _intermittent,
    "level_shift": _level_shift,
    "missing_vs_zero": _missing_vs_zero,
}
