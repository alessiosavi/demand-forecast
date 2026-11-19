"""Benchmark decision figures computed from task errors and task metadata."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

# Dataset family merges and frequency suffix from the stage-1 report (s1_report.py).
_MERGED_PREFIXES = ("boomlet", "epf", "favorita", "rohlik", "uk_covid", "fred", "proenfo", "solar")
_FREQUENCY_SUFFIX = re.compile(r"_\d+[A-Za-z]+(?:-[A-Za-z]+)?$")
_WIN_MARGIN = 0.005

_Scores = dict[str, float]


@dataclass(frozen=True)
class Figures:
    """Win rate, skill, strict firsts, and the tasks with a finite score."""

    w: float
    s: float
    f: int
    scored: frozenset[str]


@dataclass(frozen=True)
class PairedRatio:
    """Geometric error ratio and optional 90% family bootstrap interval."""

    ratio: float
    low: float | None
    high: float | None
    families: int


def relative(errors: _Scores, baseline: _Scores, tasks: Sequence[str]) -> _Scores:
    """Return clipped ratios for the tasks with a finite system error."""
    if not tasks:
        raise ValueError("at least one task is required")
    result = {}
    for task in tasks:
        denominator = baseline.get(task, math.nan)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"baseline error must be finite and positive for task {task!r}")
        error = errors.get(task, math.nan)
        if math.isfinite(error):
            if error < 0:
                # Clipping would turn an invalid negative error into the best possible ratio.
                raise ValueError(f"system error must not be negative for task {task!r}")
            result[task] = min(100.0, max(0.01, error / denominator))
    return result


def _comparison(
    system: _Scores, panel: Sequence[_Scores], baseline: _Scores, tasks: Sequence[str]
) -> tuple[_Scores, list[_Scores]]:
    scores = relative(system, baseline, tasks)
    if not panel:
        raise ValueError("at least one panel member is required")
    members = []
    for index, errors in enumerate(panel, start=1):
        member = relative(errors, baseline, tasks)
        missing = [task for task in tasks if task not in member]
        if missing:
            raise ValueError(
                f"panel member {index} lacks finite errors for tasks: {', '.join(missing)}"
            )
        members.append(member)
    return scores, members


def win_rate(
    system: _Scores, panel: Sequence[_Scores], baseline: _Scores, tasks: Sequence[str]
) -> float:
    """Average wins and half-wins over the entire task-by-panel grid."""
    scores, members = _comparison(system, panel, baseline, tasks)
    # An unscored task supplies no wins and stays in the denominator.
    half_wins = sum(
        2 * (error < member[task]) + (error == member[task])
        for task, error in scores.items()
        for member in members
    )
    return half_wins / (2 * len(tasks) * len(members))


def skill(system: _Scores, baseline: _Scores, tasks: Sequence[str]) -> float:
    """Return one minus the geometric mean of scored relative errors."""
    scores = relative(system, baseline, tasks)
    if not scores:
        raise ValueError("skill requires at least one scored task")
    return 1.0 - math.exp(math.fsum(map(math.log, scores.values())) / len(scores))


def firsts(
    system: _Scores, panel: Sequence[_Scores], baseline: _Scores, tasks: Sequence[str]
) -> int:
    """Count scored tasks with strictly lower error than every panel member."""
    scores, members = _comparison(system, panel, baseline, tasks)
    return sum(all(error < member[task] for member in members) for task, error in scores.items())


def eligible(candidate: Figures, incumbent: Figures) -> bool:
    """Apply the skill, task coverage, win rate, and improvement gates."""
    # Win rates are fractions. Allow one float step for rounding at the inclusive boundary.
    minimum_win = math.nextafter(incumbent.w + _WIN_MARGIN, -math.inf)
    return (
        candidate.s >= incumbent.s - 0.002
        and candidate.scored >= incumbent.scored
        and candidate.w >= incumbent.w
        and (candidate.w >= minimum_win or candidate.f >= incumbent.f + 1)
    )


def choose(incumbent: tuple[str, Figures], candidates: Mapping[str, Figures]) -> str:
    """Select the incumbent or an eligible candidate by W, F, S, and name."""
    incumbent_name, incumbent_figures = incumbent
    contenders = [incumbent]
    contenders.extend(
        (name, figures)
        for name, figures in candidates.items()
        if eligible(figures, incumbent_figures)
    )
    best_win = max(figures.w for _, figures in contenders)
    # Compare by addition to avoid cancellation when the win rates are small.
    minimum_best = math.nextafter(best_win, -math.inf)
    return min(
        (
            (name, figures)
            for name, figures in contenders
            if figures.w + _WIN_MARGIN >= minimum_best
        ),
        key=lambda row: (-row[1].f, -row[1].s, row[0] != incumbent_name, row[0]),
    )[0]


def family(task_name: str) -> str:
    """Strip the variant and frequency before applying dataset family merges."""
    base = _FREQUENCY_SUFFIX.sub("", task_name.split("/", 1)[0])
    return next((prefix for prefix in _MERGED_PREFIXES if base.startswith(prefix)), base)


def paired_ratio(
    a: _Scores,
    b: _Scores,
    baseline: _Scores,
    tasks: Sequence[str],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    draws: ArrayLike | None = None,
) -> PairedRatio:
    """Bootstrap paired relative errors by drawing whole dataset families."""
    scores_a = relative(a, baseline, tasks)
    scores_b = relative(b, baseline, tasks)
    by_family: dict[str, list[float]] = {}
    deltas = []
    for task, error_a in scores_a.items():
        if task in scores_b:
            delta = math.log(error_a / scores_b[task])
            by_family.setdefault(family(task), []).append(delta)
            deltas.append(delta)
    if not deltas:
        raise ValueError("paired ratio requires at least one shared scored task")
    ratio = math.exp(math.fsum(deltas) / len(deltas))
    groups = [by_family[name] for name in sorted(by_family)]
    count = len(groups)
    indices = None
    if draws is not None:
        # Checked before the early return: a malformed matrix is an error for any number of families.
        indices = np.asarray(draws)
        if (
            indices.ndim != 2
            or indices.shape[0] == 0
            or indices.shape[1] != count
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= count)
        ):
            raise ValueError("draws must be a nonempty integer matrix of valid family indices")
    if count < 3:
        return PairedRatio(ratio, None, None, count)

    if indices is None:
        if resamples < 1:
            raise ValueError("resamples must be positive")
        indices = np.random.default_rng(seed).integers(count, size=(resamples, count))

    sums = np.array([math.fsum(group) for group in groups])
    sizes = np.array([len(group) for group in groups])
    # Keep the score task-weighted even when families have different numbers of tasks.
    samples = np.exp(sums[indices].sum(axis=1) / sizes[indices].sum(axis=1))
    low, high = np.percentile(samples, [5, 95])
    return PairedRatio(ratio, float(low), float(high), count)


class _Task(Protocol):
    task_name: str
    target_columns: Sequence[str]
    known_dynamic_columns: Sequence[str]
    past_dynamic_columns: Sequence[str]
    static_columns: Sequence[str]


def slices(tasks: Iterable[_Task]) -> dict[str, list[str]]:
    """Group task names into overlapping slices, preserving the input order."""
    result: dict[str, list[str]] = {
        "all": [],
        "without_features": [],
        "with_features": [],
        "known_future": [],
        "past_only": [],
        "static": [],
        "multi_target": [],
    }
    for task in tasks:
        name = task.task_name
        result["all"].append(name)
        feature_roles = {
            "known_future": task.known_dynamic_columns,
            "past_only": task.past_dynamic_columns,
            "static": task.static_columns,
        }
        for role, columns in feature_roles.items():
            if columns:
                result[role].append(name)
        result["with_features" if any(feature_roles.values()) else "without_features"].append(name)
        targets = task.target_columns or ()
        if isinstance(targets, str):
            targets = (targets,)
        if len(targets) > 1:
            result["multi_target"].append(name)
    return result
