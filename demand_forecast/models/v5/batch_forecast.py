"""Forecast panel items in bounded batches using each item's own origin."""

import sys
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, fields, replace
from itertools import groupby, islice
from operator import itemgetter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from demand_forecast.data.episode import (
    EpisodeBatch,
    EpisodeBuilder,
    EpisodeRequest,
    Panel,
    PanelStatsProvider,
    ScreeningConfig,
    apply_summary_off,
    collate,
)
from demand_forecast.data.episode.schema import MAX_FORECAST_ROWS

from .model import V5Model, prune


@dataclass
class ItemForecasts:
    """Original-unit forecasts of shape (horizon, quantiles), keyed by item and target."""

    values: dict[tuple[int, str], np.ndarray]
    fallback: frozenset[tuple[int, str]]
    stats_builds: int


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
) -> list[str]:
    """Return the builder's joint or per-target layout in input job order."""
    stats = PanelStatsProvider(panel, limit=None)
    builder = EpisodeBuilder(panel, screening, stats=stats)
    layouts = ["joint"] * len(jobs)
    for origin, same_origin in _jobs_by_origin(jobs):
        try:
            for _, position, index in same_origin:
                requests = builder.plan_requests(
                    index, (origin,), horizon, context=context, cutoff=origin, drop=drop
                )
                if requests[0].target is not None:
                    layouts[position] = "per_target"
        finally:
            stats.release(origin)
    return layouts


def _validate_joint_layouts(
    builder: EpisodeBuilder,
    jobs: Sequence[tuple[int, int]],
    layouts: Sequence[str],
    horizon: int,
    context: int,
    drop: frozenset[str],
) -> None:
    for origin, same_origin in _jobs_by_origin(jobs):
        for _, position, index in same_origin:
            if layouts[position] != "joint":
                continue
            if not 0 <= index < len(builder.panel.items):
                raise ValueError("item index outside the panel")
            # Single targets are already designated, and wide items always split.
            if not 1 < len(builder.panel.items[index].targets) <= MAX_FORECAST_ROWS:
                continue
            requests = builder.plan_requests(
                index, (origin,), horizon, context=context, cutoff=origin, drop=drop
            )
            if requests[0].target is not None:
                raise ValueError(
                    f"item index {index} at origin {origin} must be forecast per target"
                )


def _requests_for_job(
    builder: EpisodeBuilder,
    index: int,
    origin: int,
    horizon: int,
    context: int,
    layout: str | None,
    drop: frozenset[str],
) -> list[EpisodeRequest]:
    if layout is None:
        return builder.plan_requests(
            index, (origin,), horizon, context=context, cutoff=origin, drop=drop
        )
    if not 0 <= index < len(builder.panel.items):
        raise ValueError("item index outside the panel")
    targets = builder.panel.items[index].targets
    if not targets:
        raise ValueError("a forecast target must exist")
    designated = (
        sorted(targets) if layout == "per_target" or len(targets) > MAX_FORECAST_ROWS else [None]
    )
    return [
        EpisodeRequest(index, (origin,), horizon, context, target, origin, drop)
        for target in designated
    ]


def _validate_workers(workers: int, prefetch: int) -> None:
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    if workers > 0 and sys.platform != "linux":
        raise ValueError("forecast workers are supported only on Linux")
    if type(prefetch) is not int or prefetch < 1:
        raise ValueError("prefetch must be a positive integer")


def _collate_chunk(
    builder: EpisodeBuilder, group: list[EpisodeRequest], summary_off: tuple[str, ...]
) -> EpisodeBatch:
    return collate([apply_summary_off(builder.build(req), summary_off) for req in group])


class _EpisodeChunks(Dataset[EpisodeBatch]):
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

    def __getitem__(self, index: int) -> EpisodeBatch:
        return _collate_chunk(self.builder, self.groups[index], self.summary_off)


def _forecast_batches(
    builder: EpisodeBuilder,
    requests: Iterator[EpisodeRequest],
    origin: int,
    batch_episodes: int,
    summary_off: tuple[str, ...],
    workers: int,
    prefetch: int,
) -> Generator[tuple[list[EpisodeRequest], EpisodeBatch], None, None]:
    if not workers:
        while group := list(islice(requests, batch_episodes)):
            yield group, _collate_chunk(builder, group, summary_off)
        return

    from demand_forecast.pretrain.v5 import loop

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

    iterator = iter(
        DataLoader(
            _EpisodeChunks(builder, groups, summary_off),
            num_workers=workers,
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
    On Linux, workers can build and collate chunks, with prefetch chunks queued
    per worker. Statistics construction and model inference run in the caller.
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
    parameter = next(model.parameters())
    dtypes = {p.dtype for p in model.parameters()}
    if len(dtypes) != 1 or not parameter.is_floating_point():
        found = ", ".join(sorted(map(str, dtypes)))
        raise ValueError(
            f"forecast_items requires one floating-point dtype for parameters, got {found}"
        )
    device, dtype = parameter.device, parameter.dtype
    stats = PanelStatsProvider(panel, limit=None)
    builder = EpisodeBuilder(panel, screening, stats=stats)
    if layouts is not None:
        # Reuse statistics from validation until forecasting at their cutoff is done.
        _validate_joint_layouts(builder, jobs, layouts, horizon, context, drop)
    values, fallback = {}, set()
    training = model.training
    try:
        model.eval()
        with torch.autocast(device.type, enabled=False):
            for origin, same_origin in _jobs_by_origin(jobs):
                try:
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
                        for group, batch in batches:
                            batch = _batch_to_model(batch, device, dtype)
                            selected = model.gates(batch, mode=gates)
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
                            del batch, selected, quantiles
                finally:
                    stats.release(origin)
    finally:
        model.train(training)
    return ItemForecasts(values, frozenset(fallback), stats.builds)
