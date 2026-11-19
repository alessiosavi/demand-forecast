"""Forecast panel items in bounded batches using each item's own origin."""

import sys
from collections import defaultdict
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, fields, replace
from itertools import groupby, islice, tee
from operator import itemgetter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from demand_forecast.data.episode import (
    Episode,
    EpisodeBatch,
    EpisodeBuilder,
    EpisodeRequest,
    Panel,
    PanelStatsProvider,
    ScreeningConfig,
    apply_summary_off,
    collate,
)
from demand_forecast.data.episode.schema import MAX_FORECAST_ROWS, STEP_FAMILIES
from demand_forecast.data.episode.screening import Candidate

from .model import Gates, V5Model, prune


@dataclass
class Evidence:
    """Screening and applied gates per request, with sorted candidate columns.

    Status codes are 0 absent, 1 kept, 2 fixed rule, 3 budget, 4 zero and
    retained, and 5 zero and pruned. Rule indices are -1 except for status 2.
    A candidate with multiple rows reports their maximum applied gate and
    has status 4 or 5 only when all those gates are zero.
    A kept candidate without rows or tokens has a NaN gate. Absent and
    removed candidates also have NaN gates. Scores come from the screening
    trace and are NaN for absent candidates and step families.
    """

    requests: list[tuple[int, str | None]]
    candidates: list[str]
    status: np.ndarray
    rules: tuple[str, ...]
    rule: np.ndarray
    gate: np.ndarray
    score: np.ndarray


@dataclass
class ItemForecasts:
    """Original-unit forecasts of shape (horizon, quantiles), keyed by item and target."""

    values: dict[tuple[int, str], np.ndarray]
    fallback: frozenset[tuple[int, str]]
    stats_builds: int
    evidence: Evidence


@dataclass
class _EvidenceInputs:
    trace: tuple[Candidate, ...]
    rows: dict[tuple[str, str], list[int]]
    attributes: dict[str, int]


@dataclass
class _ForecastChunk:
    batch: EpisodeBatch
    evidence: list[_EvidenceInputs]


def _evidence_inputs(episode: Episode) -> _EvidenceInputs:
    key_count, row_count = len(episode.row_keys), len(episode.x)
    if key_count != row_count:
        raise ValueError(f"Episode.row_keys has length {key_count} for {row_count} rows")
    rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, key in enumerate(episode.row_keys):
        if key is not None:
            rows[key].append(i)

    represented = set(episode.row_cand) | set(episode.attr_cand) | set(episode.static_num_cand)
    attributes = {name: i for i, name in enumerate(episode.cand_names) if i in represented}
    return _EvidenceInputs(episode.trace, dict(rows), attributes)


# Each entry contains the status, fixed rule (if any), applied gate and score.
_EvidenceRow = dict[str, tuple[int, str | None, float, float]]


def _evidence_rows(inputs: list[_EvidenceInputs], gates: Gates, pruned: bool) -> list[_EvidenceRow]:
    rows, attributes, families = (
        tensor.cpu().tolist() for tensor in (gates.row, gates.cand, gates.family)
    )
    evidence = []
    for data, row_gates, attr_gates, family_gates in zip(
        inputs, rows, attributes, families, strict=True
    ):
        entries: _EvidenceRow = {
            f"family:{name}": (4 if gate == 0 else 1, None, gate, np.nan)
            for name, gate in zip(STEP_FAMILIES, family_gates, strict=True)
        }
        for candidate in data.trace:
            source, name = candidate.source, candidate.name
            gate = np.nan
            rule = None
            if candidate.removed == "budget":
                status = 3
            elif candidate.removed is not None:
                status, rule = 2, candidate.removed
            else:
                if source == "static":
                    if name in data.attributes:
                        gate = attr_gates[data.attributes[name]]
                else:
                    # An expanded category is removed only if all its rows are.
                    gate = max(
                        (row_gates[i] for i in data.rows.get((source, name), ())), default=np.nan
                    )
                status = (5 if pruned else 4) if gate == 0 else 1
            prefix = "dynamic" if source in ("known", "past") else source
            entries[f"{prefix}:{name}"] = status, rule, gate, candidate.score
        evidence.append(entries)
    return evidence


def _pack_evidence(requests: list[tuple[int, str | None]], rows: list[_EvidenceRow]) -> Evidence:
    candidates = sorted({name for row in rows for name in row})
    rules = tuple(
        sorted({rule for row in rows for _, rule, _, _ in row.values() if rule is not None})
    )
    columns = {name: i for i, name in enumerate(candidates)}
    rule_indices = {name: i for i, name in enumerate(rules)}
    shape = (len(requests), len(candidates))
    status = np.zeros(shape, dtype=np.int8)
    rule = np.full(shape, -1, dtype=np.int8)
    gate = np.full(shape, np.nan, dtype=np.float32)
    score = np.full(shape, np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for name, (disposition, fixed_rule, weight, relevance) in row.items():
            j = columns[name]
            status[i, j] = disposition
            if fixed_rule is not None:
                rule[i, j] = rule_indices[fixed_rule]
            gate[i, j] = weight
            score[i, j] = relevance
    return Evidence(requests, candidates, status, rules, rule, gate, score)


def _jobs_by_origin(
    jobs: Sequence[tuple[int, int]],
) -> Iterator[tuple[int, Iterator[tuple[int, int, int]]]]:
    # Retain the input position for layout lookup, including ties at one origin.
    ordered = sorted((origin, position, index) for position, (index, origin) in enumerate(jobs))
    return groupby(ordered, key=itemgetter(0))


def planned_layouts(
    panel: Panel,
    jobs: Sequence[tuple[int, int]],
    horizon: int,
    *,
    context: int,
    screening: ScreeningConfig = ScreeningConfig(),
    drop: frozenset[str] = frozenset(),
    derive: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the builder's joint or per-target layout in input job order."""
    stats = PanelStatsProvider(panel, limit=None, scale_floor=screening.scale_floor)
    builder = EpisodeBuilder(panel, screening, stats=stats)
    layouts = ["joint"] * len(jobs)
    for origin, same_origin in _jobs_by_origin(jobs):
        try:
            for _, position, index in same_origin:
                requests = builder.plan_requests(
                    index,
                    (origin,),
                    horizon,
                    context=context,
                    cutoff=origin,
                    drop=drop,
                    derive=derive,
                )
                if requests[0].target is not None:
                    layouts[position] = "per_target"
        finally:
            stats.release(origin)
    return layouts


def _validate_joint_layouts(
    builder: EpisodeBuilder,
    jobs: Iterator[tuple[int, int, int]],
    layouts: Sequence[str],
    horizon: int,
    context: int,
    drop: frozenset[str],
    derive: frozenset[str],
) -> None:
    for origin, position, index in jobs:
        if layouts[position] != "joint":
            continue
        # Single targets are already designated, and wide items always split.
        if not 1 < len(builder.panel.items[index].targets) <= MAX_FORECAST_ROWS:
            continue
        requests = builder.plan_requests(
            index, (origin,), horizon, context=context, cutoff=origin, drop=drop, derive=derive
        )
        if requests[0].target is not None:
            raise ValueError(f"item index {index} at origin {origin} must be forecast per target")


def _requests_for_job(
    builder: EpisodeBuilder,
    index: int,
    origin: int,
    horizon: int,
    context: int,
    layout: str | None,
    drop: frozenset[str],
    derive: frozenset[str],
) -> list[EpisodeRequest]:
    if layout is None:
        return builder.plan_requests(
            index, (origin,), horizon, context=context, cutoff=origin, drop=drop, derive=derive
        )
    targets = builder.panel.items[index].targets
    if not targets:
        raise ValueError("a forecast target must exist")
    designated = (
        sorted(targets) if layout == "per_target" or len(targets) > MAX_FORECAST_ROWS else [None]
    )
    return [
        EpisodeRequest(index, (origin,), horizon, context, target, origin, drop, derive=derive)
        for target in designated
    ]


def _validate_workers(workers: int, prefetch: int) -> None:
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    if workers > 0 and sys.platform != "linux":
        raise ValueError("forecast workers are supported only on Linux")
    if type(prefetch) is not int or prefetch < 1:
        raise ValueError("prefetch must be a positive integer")


def usable_workers(requested: int) -> tuple[int, str | None]:
    """Return the worker count and any serial fallback reason for a nonnegative request.

    A positive Linux request may raise the process file-descriptor limit.
    """
    if type(requested) is not int or requested < 0:
        raise ValueError("workers must be a nonnegative integer")
    if requested == 0:
        return 0, None
    if sys.platform != "linux":
        return 0, "forecast workers are supported only on Linux"

    from demand_forecast.pretrain.v5 import sampler

    try:
        sampler._ensure_worker_file_limit(requested)
    except RuntimeError as error:
        return 0, str(error)
    return requested, None


def _collate_chunk(
    builder: EpisodeBuilder, group: list[EpisodeRequest], summary_off: tuple[str, ...]
) -> _ForecastChunk:
    episodes = [apply_summary_off(builder.build(req), summary_off) for req in group]
    evidence = [_evidence_inputs(episode) for episode in episodes]
    return _ForecastChunk(collate(episodes), evidence)


class _EpisodeChunks(Dataset[_ForecastChunk]):
    def __init__(
        self,
        builder: EpisodeBuilder,
        groups: list[list[EpisodeRequest]],
        summary_off: tuple[str, ...],
    ) -> None:
        self.builder = builder
        self.groups = groups
        self.summary_off = summary_off

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> _ForecastChunk:
        return _collate_chunk(self.builder, self.groups[index], self.summary_off)


def _forecast_batches(
    builder: EpisodeBuilder,
    requests: Iterator[EpisodeRequest],
    origin: int,
    batch_episodes: int,
    summary_off: tuple[str, ...],
    workers: int,
    prefetch: int,
) -> Generator[tuple[list[EpisodeRequest], _ForecastChunk], None, None]:
    if not workers:
        while group := list(islice(requests, batch_episodes)):
            yield group, _collate_chunk(builder, group, summary_off)
        return

    groups = []
    targets: set[str] = set()
    while group := list(islice(requests, batch_episodes)):
        groups.append(group)
        for request in group:
            if request.target is not None:
                targets.add(request.target)
                continue
            item_targets = builder.panel.items[request.item_index].targets
            # Joint screening reads the other targets only when static kinds exist.
            stats = builder.stats.get(min(item_targets), origin)
            if stats.static_kinds:
                targets.update(item_targets)
    # Workers inherit the same unbounded provider used for planning.
    for target in sorted(targets):
        builder.stats.get(target, origin)

    if len(groups) <= 1:
        for group in groups:
            yield group, _collate_chunk(builder, group, summary_off)
        return

    from demand_forecast.pretrain.v5 import loop

    iterator = iter(
        DataLoader(
            _EpisodeChunks(builder, groups, summary_off),
            num_workers=min(workers, len(groups)),
            multiprocessing_context="fork",
            batch_size=None,
            prefetch_factor=prefetch,
            generator=torch.Generator(),
        )
    )
    drained = False
    try:
        yield from zip(groups, iterator, strict=True)
        drained = True
    finally:
        try:
            loop._release(iterator)
        except Exception:
            # Preserve a loader error or the forecast error that triggered close().
            if drained:
                raise


def _batch_to_model(batch: EpisodeBatch, device: torch.device, dtype: torch.dtype) -> EpisodeBatch:
    if dtype == torch.float32:
        return batch.to(device)
    tensors = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            # Keep the inverse transform in the builder's float64 frame.
            target_dtype = value.dtype
            if value.is_floating_point() and field.name not in ("loc", "scale"):
                target_dtype = dtype
            tensors[field.name] = value.to(device=device, dtype=target_dtype)
    return replace(batch, **tensors)


def _kept_forecasts(
    batch: EpisodeBatch,
    quantiles: torch.Tensor,
    requests: list[EpisodeRequest],
    horizon: int,
) -> Iterator[tuple[tuple[int, str], np.ndarray, bool]]:
    # Preserve the builder's float64 location and scale for the inverse transform.
    loc = batch.loc[..., None, None]
    original = loc + batch.scale[..., None, None] * quantiles
    original = torch.where(batch.fallback[..., None, None], loc, original)
    values = original[:, :, :horizon].cpu().numpy()
    keep = (batch.keep & batch.forecast_mask).cpu().numpy()
    rows = batch.forecast_rows.cpu().numpy()
    fallback = batch.fallback.cpu().numpy()
    for b, request in enumerate(requests):
        for row in np.flatnonzero(keep[b]):
            name = batch.row_names[b][rows[b, row]]
            yield (request.item_index, name), values[b, row].copy(), bool(fallback[b, row])


@torch.no_grad()
def forecast_items(
    model: V5Model,
    panel: Panel,
    jobs: Sequence[tuple[int, int]],
    horizon: int,
    *,
    context: int,
    screening: ScreeningConfig = ScreeningConfig(),
    layouts: Sequence[str] | None = None,
    drop: frozenset[str] = frozenset(),
    derive: frozenset[str] = frozenset(),
    summary_off: tuple[str, ...] = (),
    gates: str = "learned",
    router: str = "learned",
    pruned: bool = True,
    batch_episodes: int = 64,
    workers: int = 0,
    prefetch: int = 2,
) -> ItemForecasts:
    """Forecast one origin per item in the model's floating-point dtype and device.

    The cutoff is exclusive for observed inputs. Explicit layouts follow the
    input job order and can preserve the full panel's layout through ablations.
    Joint layout compatibility is checked before forecasting each origin.
    On Linux, workers can build and collate chunks for origins with multiple
    chunks, with prefetch chunks queued per worker. Statistics construction
    and model inference run in the caller.
    """
    _validate_workers(workers, prefetch)
    if workers:
        # Pretraining imports the model, so load its helpers only at runtime.
        from demand_forecast.pretrain.v5 import sampler

        sampler._ensure_worker_file_limit(workers)
    if type(batch_episodes) is not int or batch_episodes < 1:
        raise ValueError("batch_episodes must be a positive integer")
    if router not in ("learned", "uniform"):
        raise ValueError(f"unsupported forecasting router {router!r}")
    if layouts is not None and (
        len(layouts) != len(jobs)
        or any(layout not in ("joint", "per_target") for layout in layouts)
    ):
        raise ValueError("layouts must give 'joint' or 'per_target' for every job")
    if len({index for index, _ in jobs}) != len(jobs):
        raise ValueError("forecast jobs must specify one origin per item")
    if any(not 0 <= index < len(panel.items) for index, _ in jobs):
        raise ValueError("item index outside the panel")
    parameter = next(model.parameters())
    dtypes = {p.dtype for p in model.parameters()}
    if len(dtypes) != 1 or not parameter.is_floating_point():
        found = ", ".join(sorted(map(str, dtypes)))
        raise ValueError(
            f"forecast_items requires one floating-point dtype for parameters, got {found}"
        )
    device, dtype = parameter.device, parameter.dtype
    stats = PanelStatsProvider(panel, limit=None, scale_floor=screening.scale_floor)
    builder = EpisodeBuilder(panel, screening, stats=stats)
    values, fallback = {}, set()
    evidence_requests: list[tuple[int, str | None]] = []
    evidence_rows: list[_EvidenceRow] = []
    training = model.training
    try:
        model.eval()
        with torch.autocast(device.type, enabled=False):
            for origin, same_origin in _jobs_by_origin(jobs):
                try:
                    if layouts is not None:
                        # Reuse this origin's validation statistics until its forecasts finish.
                        validation_jobs, same_origin = tee(same_origin)
                        _validate_joint_layouts(
                            builder, validation_jobs, layouts, horizon, context, drop, derive
                        )
                    requests = (
                        request
                        for _, position, index in same_origin
                        for request in _requests_for_job(
                            builder,
                            index,
                            origin,
                            horizon,
                            context,
                            None if layouts is None else layouts[position],
                            drop,
                            derive,
                        )
                    )
                    with closing(
                        _forecast_batches(
                            builder,
                            requests,
                            origin,
                            batch_episodes,
                            summary_off,
                            workers,
                            prefetch,
                        )
                    ) as batches:
                        for group, chunk in batches:
                            batch = _batch_to_model(chunk.batch, device, dtype)
                            selected = model.gates(batch, mode=gates)
                            evidence_requests.extend((req.item_index, req.target) for req in group)
                            evidence_rows.extend(_evidence_rows(chunk.evidence, selected, pruned))
                            if pruned:
                                batch, selected = prune(batch, selected)
                            quantiles = model(batch, selected, router=router).quantiles[:, 0]
                            for key, value, used_fallback in _kept_forecasts(
                                batch, quantiles, group, horizon
                            ):
                                if not np.isfinite(value).all():
                                    item = panel.items[key[0]].item_id
                                    raise ValueError(f"nonfinite forecast for {item!r}, {key[1]!r}")
                                values[key] = value
                                if used_fallback:
                                    fallback.add(key)
                            del chunk, batch, selected, quantiles
                finally:
                    stats.release(origin)
    finally:
        model.train(training)
    evidence = _pack_evidence(evidence_requests, evidence_rows)
    return ItemForecasts(values, frozenset(fallback), stats.builds, evidence)
