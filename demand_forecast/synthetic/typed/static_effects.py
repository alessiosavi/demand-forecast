"""Static inputs and their equation-defined contributions to the latent panel."""

import math
from collections import Counter
from itertools import combinations

import numpy as np

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


def _group_process(draft: Draft, labels: list[str]) -> dict[str, object]:
    """Record the process's group sizes without the realisation's label tokens."""
    counts = Counter(label for label, short in zip(labels, draft.short, strict=True) if not short)
    return {"group_sizes": tuple(sorted(counts.values()))}


def static_category(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    name = token(streams.inputs, "a")
    labels, singletons = assign_groups(draft, streams.process, streams.inputs)
    draft.add_static(name, labels, "category", "static_category")
    process = _group_process(draft, labels)
    if streams.effect is None:
        return Result((name,), process, {}, None)
    vocabulary = sorted(set(labels))
    amplitude = {label: float(streams.effect.uniform(*ranges("amplitude"))) for label in vocabulary}
    phase = {label: float(streams.effect.uniform(0, 2 * np.pi)) for label in vocabulary}
    add = [
        amplitude[label] * np.sin(2 * np.pi * draft.grid(i) / draft.period + phase[label])
        for i, label in enumerate(labels)
    ]
    return Result(
        (name,),
        process,
        {"amplitude": amplitude, "phase": phase, "singletons": singletons},
        Contribution(add=add),
    )


def multi_label(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    count = draft.recipe.n_items
    vocabulary_size = int(streams.process.integers(4, 11))
    name = token(streams.inputs, "a")
    vocabulary = sorted(token(streams.inputs, "l") for _ in range(vocabulary_size))
    sets = [labels for size in (1, 2, 3) for labels in combinations(vocabulary, size)]
    ordered = [sets[int(k)] for k in streams.inputs.permutation(len(sets))]
    halfway = math.ceil(len(ordered) / 2)
    pool, reserve = tuple(ordered[:halfway]), ordered[halfway:]

    # Held-out sets must be unique, and short items must retain their host's set.
    # Draw candidates among long items that are not hosts, satisfying both promises.
    eligible = ~draft.short.copy()
    eligible[draft.host[draft.host >= 0]] = False
    n_candidates = math.ceil(count / 5) if count > 1 else 0
    candidates = streams.inputs.choice(np.flatnonzero(eligible), n_candidates, replace=False)
    candidate_mask = np.zeros(count, dtype=bool)
    candidate_mask[candidates] = True
    labels = [[] for _ in range(count)]
    for i in np.flatnonzero(~draft.short & ~candidate_mask):
        labels[i] = list(pool[int(streams.inputs.integers(len(pool)))])
    for i in np.flatnonzero(draft.short):
        host = int(draft.host[i])
        labels[i] = (
            labels[host].copy()
            if host >= 0
            else list(pool[int(streams.inputs.integers(len(pool)))])
        )
    observed = {label for values in labels for label in values}
    available = [values for values in reserve if set(values) <= observed]
    heldout = []
    for k, i in enumerate(candidates):
        if k < len(available):
            labels[i] = list(available[k])
            heldout.append(f"i{i:04d}")
        else:
            labels[i] = list(pool[int(streams.inputs.integers(len(pool)))])
    draft.add_static(name, labels, "multilabel", "multi_label")
    process = {"M": vocabulary_size}
    if streams.effect is None:
        return Result((name,), process, {}, None)
    position = {label: int(streams.effect.integers(draft.period)) for label in vocabulary}
    size = {label: float(streams.effect.uniform(*ranges("size"))) for label in vocabulary}
    signs = {label: int(streams.effect.choice((-1, 1))) for label in vocabulary}
    a, b = vocabulary[:2]
    position[b], signs[b] = position[a], -signs[a]
    add = [
        sum(
            signs[label] * size[label] * (draft.grid(i) % draft.period == position[label])
            for label in values
        )
        for i, values in enumerate(labels)
    ]
    params = {
        "position": position,
        "size": size,
        "size_sign": signs,
        "opposed": (a, b),
        "heldout_items": tuple(heldout),
        "pool": pool,
    }
    return Result((name,), process, params, Contribution(add=add))


def hierarchy(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    n_long = int(np.count_nonzero(~draft.short))
    cap = max(1, min(20, n_long // 2))
    n_children = int(streams.process.integers(min(4, cap), cap + 1))
    lower = min(2, n_children)
    upper = max(lower, min(5, n_children // 2))
    n_parents = int(streams.process.integers(lower, upper + 1))
    parent_name, child_name = token(streams.inputs, "a"), token(streams.inputs, "a")
    children, _ = assign_groups(
        draft, streams.process, streams.inputs, n=n_children, singletons=False
    )
    parents = [token(streams.inputs, "l") for _ in range(n_parents)]
    vocabulary = sorted(set(children))
    parent_of = {child: parents[k % n_parents] for k, child in enumerate(vocabulary)}
    values = [parent_of[child] for child in children]
    draft.add_static(parent_name, values, "category", "hierarchy")
    draft.add_static(child_name, children, "category", "hierarchy")
    process = {"n_children": n_children, "n_parents": n_parents, **_group_process(draft, children)}
    names = (parent_name, child_name)
    if streams.effect is None:
        return Result(names, process, {}, None)
    rho = {parent: float(streams.effect.uniform(0, 0.9)) for parent in parents}
    amplitude = {child: float(streams.effect.uniform(*ranges("amplitude"))) for child in vocabulary}
    phase = {child: float(streams.effect.uniform(0, 2 * np.pi)) for child in vocabulary}
    add = [
        amplitude[child] * np.sin(4 * np.pi * draft.grid(i) / draft.period + phase[child])
        for i, child in enumerate(children)
    ]
    params = {"parent_of": parent_of, "rho": rho, "amplitude": amplitude, "phase": phase}
    return Result(
        names,
        process,
        params,
        Contribution(add=add, rho=np.array([rho[parent] for parent in values])),
    )


def static_number(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    name = token(streams.inputs, "a")
    x = np.exp(streams.inputs.uniform(0, np.log(100), draft.recipe.n_items))
    draft.add_static(name, x.tolist(), "number", "static_number")
    if streams.effect is None:
        return Result((name,), {}, {}, None)
    log_x = np.log(x)
    deviation = log_x.std()
    z = (log_x - log_x.mean()) / deviation if deviation != 0 else np.zeros_like(log_x)
    noise = float(streams.effect.uniform(*ranges("noise")))
    trend = float(streams.effect.uniform(*ranges("trend")))
    sign = int(streams.effect.choice((-1, 1)))
    add = [
        sign * trend * z[i] * ((draft.grid(i) - draft.grid_length / 2) / draft.grid_length)
        for i in range(draft.recipe.n_items)
    ]
    mult = [np.full(n, np.exp(-noise * z[i])) for i, n in enumerate(draft.lengths)]
    return Result(
        (name,),
        {},
        {"noise": noise, "trend": trend, "trend_sign": sign},
        Contribution(add=add, mult=mult),
    )


def whole_number_code(draft: Draft, streams: Streams, ranges: RangeFn) -> Result:
    name = token(streams.inputs, "a")
    labels, _ = assign_groups(
        draft, streams.process, streams.inputs, max_labels=6, singletons=False
    )
    code_of = {label: float(k) for k, label in enumerate(sorted(set(labels)))}
    codes = [code_of[label] for label in labels]
    draft.add_static(name, codes, "category", "whole_number_code")
    process = _group_process(draft, labels)
    if streams.effect is None:
        return Result((name,), process, {}, None)
    probability = {code: float(streams.effect.uniform(*ranges("p"))) for code in code_of.values()}
    occ = [np.full(n, probability[code]) for n, code in zip(draft.lengths, codes, strict=True)]
    return Result((name,), process, {"p": probability}, Contribution(occ=occ))


STATIC: dict[str, MechanismFn] = {
    "static_category": static_category,
    "multi_label": multi_label,
    "hierarchy": hierarchy,
    "static_number": static_number,
    "whole_number_code": whole_number_code,
}
