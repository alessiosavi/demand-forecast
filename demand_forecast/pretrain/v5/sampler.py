"""Index-addressed panel planning and bounded, worker-independent batches."""

from __future__ import annotations

import math
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, Panel, apply_summary_off
from demand_forecast.data.episode.collate import EpisodeBatch, collate
from demand_forecast.data.episode.schema import PATCH
from demand_forecast.data.episode.screening import ScreeningConfig
from demand_forecast.synthetic.typed import necessary_items
from demand_forecast.synthetic.typed.draft import INPUT_MECHANISMS, stream

from .config import SamplerConfig, SourceConfig, V5TrainConfig
from .corpus import CorpusValueError, SamplerSkip, ShortRow
from .sources import SourceMix
from .support import PROTECT, supported

COLD_MECHANISMS = ("static_category", "multi_label")

_FILE_LIMIT_LOCK = Lock()


@dataclass(frozen=True)
class EpisodePlan:
    request: EpisodeRequest
    item_id: str
    mechanism: str | None
    cold: bool = False


def _length(item):
    return len(item.targets[sorted(item.targets)[0]])


def _admissible(item, origin):
    return origin - item.start >= PATCH and item.start + _length(item) > origin


def _window(item, values, origin, horizon):
    return values[max(0, origin - item.start) : max(0, origin + horizon - item.start)]


def _origins(anchor, config, context):
    stride = PATCH * config.origin_stride
    count = min(config.origins, 1 + (context - PATCH) // stride)
    return tuple(anchor - stride * j for j in range(count - 1, -1, -1))


def _cutoff(item, origins, panel_cutoff):
    target = item.targets[sorted(item.targets)[0]]
    prefix = target[: max(0, panel_cutoff - item.start)]
    return panel_cutoff if np.isfinite(prefix).sum() >= PATCH else origins[0]


def plan_panel(
    panel: Panel,
    record: dict,
    config: SamplerConfig,
    source: SourceConfig,
    rng: np.random.Generator,
    *,
    path_quota: int | None = None,
    ordinary_quota: int | None = None,
) -> tuple[Panel, list[EpisodePlan], dict]:
    """Choose requests on the uncontrolled record, within the two open quotas.

    Omitted quotas default to those of an empty batch. Planning and building
    must share one EpisodeBuilder per panel; make_batch retains that builder
    through _plan_panel to preserve screening caches and cutoff consistency.
    """
    # The private builder lets make_batch reuse planning's screening cache.
    return _plan_panel(
        panel, record, config, source, rng, path_quota=path_quota, ordinary_quota=ordinary_quota
    )[:3]


def _plan_panel(panel, record, config, source, rng, *, path_quota=None, ordinary_quota=None):
    # An injected SourceMix or direct planner call can bypass V5TrainConfig.
    if config.cold_share > 0 and source.control == "missing_as_zero":
        raise ValueError("missing_as_zero requires sampler.cold_share == 0")
    target_paths = round(config.path_share * config.episodes_per_batch)
    path_quota = target_paths if path_quota is None else path_quota
    ordinary_quota = (
        config.episodes_per_batch - target_paths if ordinary_quota is None else ordinary_quota
    )
    generated = record.get("generated")
    original = generated.panel if generated is not None else panel
    context = int(rng.choice(config.contexts))
    horizon = int(np.clip(math.exp(rng.uniform(*np.log(config.horizon))), *config.horizon))
    at_origin = rng.random() < config.cutoff_at_origin
    frac = rng.random()
    skips = Counter()
    if config.cold_share > 0:
        skips.update(cold_attempted=0, cold_planned=0)
    mechanism = None
    lead, anchor = None, None
    lead_frame = None
    lead_cold = False
    eligible = [i for i, item in enumerate(original.items) if _length(item) >= PATCH + 1]

    def choose(pool):
        index = int(rng.choice(pool))
        item = original.items[index]
        return index, int(rng.integers(item.start + PATCH, item.start + _length(item)))

    def episode_frame(item, t):
        origins = _origins(t, config, context)
        earliest = origins[0]
        panel_cutoff = (
            earliest if at_origin else max(0, earliest - math.floor(frac * 3 * context / 4))
        )
        own = tuple(o for o in origins if _admissible(item, o))
        return own, _cutoff(item, own, panel_cutoff)

    path_capable = generated is not None and bool(generated.recipe.mechanisms)
    if path_capable and path_quota > 0:
        candidate = str(rng.choice(generated.recipe.mechanisms))
        ids = set(necessary_items(generated, candidate)) if candidate in INPUT_MECHANISMS else None
        pool = [i for i in eligible if ids is None or original.items[i].item_id in ids]
        cold_rng = (
            rng.spawn(1)[0] if candidate in COLD_MECHANISMS and config.cold_share > 0 else None
        )
        for _ in range(8 if pool else 0):
            index, t = choose(pool)
            item = original.items[index]
            is_cold = cold_rng is not None and cold_rng.random() < config.cold_share
            origin = t
            if is_cold:
                low, high = config.cold_steps
                origin = item.start + int(cold_rng.integers(low, high + 1))
                frame = ((origin,), origin)
                skips["cold_attempted"] += 1
            else:
                frame = episode_frame(item, t)
            if supported(
                generated,
                candidate,
                item.item_id,
                origin,
                context=context,
                horizon=horizon,
                cutoff=frame[1],
                first_event_only=config.first_event_only,
            ):
                lead, anchor, mechanism = index, t, candidate
                lead_frame, lead_cold = frame, is_cold
                break
        if lead is None:
            skips["no_support"] += 1
    if lead is None:
        if not eligible:
            skips["no_room"] += 1
            return panel, [], dict(skips), None
        lead, anchor = choose(eligible)
    others = [i for i, item in enumerate(original.items) if i != lead and _admissible(item, anchor)]
    ordered = [int(i) for i in rng.permutation(others)]
    count = 1 if record.get("source") == "corpus" else config.episodes_per_panel
    selected = []
    if mechanism is not None:
        selected.append((lead, mechanism))
        for index in ordered:
            if len(selected) >= min(count, path_quota):
                break
            item = original.items[index]
            _, cutoff = episode_frame(item, anchor)
            if supported(
                generated,
                mechanism,
                item.item_id,
                anchor,
                context=context,
                horizon=horizon,
                cutoff=cutoff,
                first_event_only=config.first_event_only,
            ):
                selected.append((index, mechanism))
        path_items = {index for index, _ in selected}
        remaining = [index for index in ordered if index not in path_items]
        room = min(count - len(selected), ordinary_quota)
    else:
        remaining = [lead, *ordered]
        room = min(count, ordinary_quota + (path_quota if not path_capable else 0))
    selected.extend((index, None) for index in remaining[:room])
    plans = []
    builder = None
    derive = frozenset(config.derive)
    for index, path in selected:
        item = original.items[index]
        cold = index == lead and lead_cold
        origins, cutoff = (
            lead_frame if index == lead and lead_frame is not None else episode_frame(item, anchor)
        )
        if config.designate_target or record.get("source") == "corpus":
            target = record.get("target")
            if target is None:
                target = str(rng.choice(sorted(item.targets)))
            request = EpisodeRequest(
                index, origins, horizon, context, target, cutoff, derive=derive
            )
        else:
            if builder is None:
                builder = EpisodeBuilder(
                    original,
                    ScreeningConfig(
                        no_evidence=config.screen_no_evidence,
                        fair_static_spread=config.screen_fair_spread,
                        scale_floor=config.scale_floor,
                    ),
                )
            requests = builder.plan_requests(
                index, origins, horizon, context=context, cutoff=cutoff, derive=derive
            )
            request = requests[int(rng.integers(len(requests)))]
        drawn = {name for name, probability in config.drop if rng.random() < probability}
        drops = (drawn - set(PROTECT[path] if path else ())) | set(config.drop_always)
        request = replace(request, drop=frozenset(drops))
        targets = (request.target,) if request.target is not None else tuple(sorted(item.targets))
        nonfallback = []
        for name in targets:
            values = item.targets[name]
            prefix = values[: max(0, cutoff - item.start)]
            observed = prefix[np.isfinite(prefix)]
            if len(observed) and np.any(observed != observed[0]):
                nonfallback.append(values)
        if not nonfallback:
            skips["fallback"] += 1
            continue
        if not any(
            np.isfinite(_window(item, values, o, horizon)).any()
            for values in nonfallback
            for o in origins
        ):
            skips["no_labels"] += 1
            continue
        plans.append(EpisodePlan(request, item.item_id, path, cold=cold))
        if cold:
            skips["cold_planned"] += 1
    if source.control == "missing_as_zero":
        items = []
        for item in panel.items:
            targets = {name: values.copy() for name, values in item.targets.items()}
            for values in targets.values():
                prefix = values[: max(0, anchor - item.start)]
                prefix[np.isnan(prefix)] = 0.0
            items.append(replace(item, targets=targets))
        panel = replace(panel, items=tuple(items))
    return panel, plans, dict(skips), builder


def _cells(episodes):
    """Count padded token cells, including known-future rows, not tensor elements."""
    n = len(episodes)
    rows = max(e.x.shape[0] for e in episodes)
    patches = max(e.x.shape[1] for e in episodes)
    origins = max(len(e.origin_patches) for e in episodes)
    forecasts = max(len(e.forecast_rows) for e in episodes)
    known = max(len(e.known_rows) for e in episodes)
    horizon = max(e.labels.shape[-1] // PATCH for e in episodes)
    return n * rows * patches * PATCH + n * origins * (forecasts + known) * horizon * PATCH


def _supervised(episode):
    return int(episode.label_mask[:, ~episode.fallback, : episode.horizon].sum())


def _postprocess(episode, config):
    apply_summary_off(episode, config.summary_off)
    if config.designate_target:
        episode.label_mask[:, ~episode.keep, :] = False


def _skip_reason(error):
    if isinstance(error, ShortRow):
        return "short_row"
    if isinstance(error, CorpusValueError):
        return "bad_values"
    return getattr(error, "reason", "sampler_skip")


def _draw(sources, rng):
    # SourceMix's record omits the selected configuration. Preserve its exact
    # draw while retaining that configuration, also for multiple typed sources.
    index = int(rng.choice(len(sources.sources), p=sources._weights))
    source = sources.sources[index]
    panel, record = source.panel(rng)
    return panel, record, source.config


def make_batch(
    config: V5TrainConfig, k: int, *, sources: SourceMix | None = None
) -> tuple[EpisodeBatch, dict]:
    """Build a nonempty batch, accepting only a SourceMix override.

    The report's ``cells`` counts padded token cells. Exposure counts returned
    panels, retained episodes and their supervised cells by source and folder.
    """
    if sources is not None and not isinstance(sources, SourceMix):
        raise TypeError("sources must be a SourceMix or None")
    try:
        return _make_batch(config, k, sources)
    except Exception as error:
        raise RuntimeError(f"batch {k}: {error}") from error


def _make_batch(config, k, sources):
    rng = stream(config.seed, f"batch:{k}")
    sources = SourceMix(config.sources) if sources is None else sources
    episodes = []
    skips, source_counts, paths, absent = Counter(), Counter(), Counter(), Counter()
    exposure = {}
    target_paths = round(config.sampler.path_share * config.sampler.episodes_per_batch)
    closed = False
    for _ in range(4 * config.sampler.episodes_per_batch):
        try:
            panel, record, source = _draw(sources, rng)
            kind = record.get("source", source.kind)
            counts = [exposure.setdefault(kind, {"panels": 0, "episodes": 0, "cells": 0})]
            if (folder := record.get("folder")) is not None:
                folders = counts[0].setdefault("folders", {})
                counts.append(folders.setdefault(folder, {"panels": 0, "episodes": 0, "cells": 0}))
            for count in counts:
                count["panels"] += 1
            kept_paths = sum(paths.values())
            room = config.sampler.episodes_per_batch - len(episodes)
            path_quota = min(target_paths - kept_paths, room)
            ordinary_quota = max(
                0, config.sampler.episodes_per_batch - target_paths - (len(episodes) - kept_paths)
            )
            panel, plans, planned_skips, builder = _plan_panel(
                panel,
                record,
                config.sampler,
                source,
                rng,
                path_quota=path_quota,
                ordinary_quota=ordinary_quota,
            )
            skips.update(planned_skips)
            if not plans:
                continue
            if builder is None or builder.panel is not panel:
                builder = EpisodeBuilder(
                    panel,
                    ScreeningConfig(
                        no_evidence=config.sampler.screen_no_evidence,
                        fair_static_spread=config.sampler.screen_fair_spread,
                        scale_floor=config.sampler.scale_floor,
                    ),
                )
        except SamplerSkip as error:
            skips[_skip_reason(error)] += 1
            continue
        for plan in plans:
            try:
                episode = builder.build(plan.request)
            except SamplerSkip as error:
                skips[_skip_reason(error)] += 1
                continue
            _postprocess(episode, config.sampler)
            supervised = _supervised(episode)
            if not supervised:
                skips["unsupervised"] += 1
                continue
            budget = config.sampler.max_cells
            if budget is not None:
                if _cells([episode]) > budget:
                    skips["oversized"] += 1
                    continue
                if _cells([*episodes, episode]) > budget:
                    skips["budget"] += 1
                    closed = True
                    break
            episodes.append(episode)
            source_counts[kind] += 1
            for count in counts:
                count["episodes"] += 1
                count["cells"] += supervised
            if plan.mechanism is not None:
                paths[plan.mechanism] += 1
                inputs = record["generated"].inputs_of(plan.mechanism)
                names = set(episode.cand_names) | set(episode.row_names)
                if inputs and not any(r.name in names for r in inputs):
                    absent[plan.mechanism] += 1
            if len(episodes) == config.sampler.episodes_per_batch:
                closed = True
                break
        if closed:
            break
    report = {}
    if config.sampler.cold_share > 0:
        report["cold"] = {
            "attempted": skips.pop("cold_attempted", 0),
            "planned": skips.pop("cold_planned", 0),
        }
    if not episodes:
        cold_details = f"; cold={report['cold']}" if "cold" in report else ""
        raise RuntimeError(f"no episodes after panel limit; skips={dict(skips)}{cold_details}")
    report.update(
        episodes=len(episodes),
        supervised=sum(map(_supervised, episodes)),
        cells=_cells(episodes),
        skips=dict(skips),
        sources=dict(source_counts),
        exposure=exposure,
        paths=dict(paths),
        paths_without_candidate=dict(absent),
    )
    return collate(episodes), report


class _Batches(IterableDataset):
    def __init__(self, config, start, corpus_root):
        self.config, self.start, self.corpus_root = config, start, corpus_root

    def __iter__(self):
        worker = get_worker_info()
        index = self.start + (worker.id if worker else 0)
        stride = worker.num_workers if worker else 1
        sources = SourceMix(self.config.sources, corpus_root=self.corpus_root)
        while True:
            batch, report = make_batch(self.config, index, sources=sources)
            yield index, batch, report
            index += stride


def _ensure_worker_file_limit(workers: int) -> None:
    try:
        import resource
    except ImportError:
        return

    # Allow two prefetched batches and two more in flight per worker.
    needed = 256 + workers * 4 * len(fields(EpisodeBatch))
    preferred = 1_048_576
    # The limit is process-wide. Serialize reads and raises across loader calls.
    with _FILE_LIMIT_LOCK:
        old_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if old_soft == resource.RLIM_INFINITY:
            return
        soft = old_soft
        cap = max(preferred, needed) if hard == resource.RLIM_INFINITY else hard
        if soft < min(preferred, cap):
            candidates = (preferred, 65_536, 16_384, needed)
            for candidate in dict.fromkeys(min(limit, cap) for limit in candidates):
                if candidate <= soft:
                    continue
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (candidate, hard))
                except (ValueError, OSError):
                    continue
                soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                break

    if soft == resource.RLIM_INFINITY or soft > old_soft:
        print(
            f"v5 loader raised RLIMIT_NOFILE soft limit from {old_soft} to {soft} "
            f"(hard limit {hard})",
            file=sys.stderr,
        )
    if soft != resource.RLIM_INFINITY and soft < needed:
        raise RuntimeError(
            f"RLIMIT_NOFILE soft limit {soft} is below the {needed} file descriptors needed "
            f"for {workers} v5 DataLoader workers. Raise ulimit -n or use fewer workers."
        )


def loader(
    config: V5TrainConfig, start: int, *, corpus_root: Path | None = None
) -> Iterator[tuple[int, EpisodeBatch, dict]]:
    """Each process lazily owns its sources; ordered workers partition indices."""
    if config.workers > 0:
        _ensure_worker_file_limit(config.workers)
    generator = torch.Generator().manual_seed(config.seed)
    return iter(
        DataLoader(
            _Batches(config, start, corpus_root),
            batch_size=None,
            num_workers=config.workers,
            persistent_workers=config.workers > 0,
            generator=generator,
        )
    )
