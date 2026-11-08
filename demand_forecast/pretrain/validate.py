"""Deterministic corpus hold-outs and a separate, local external diagnostic."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demand_forecast import metrics
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.panel import Panel, Series
from demand_forecast.data.sampler import BatchBuilder, _items_array, _positive
from demand_forecast.models.base import ForecastModel
from demand_forecast.pretrain.config import FEV_HORIZONS, PretrainConfig, selection_pair_policy_hash
from demand_forecast.pretrain.corpus import (
    CHRONOS_REVISION,
    CORPUS_REPO,
    CORPUS_ROLES,
    PRETRAIN_FREQ,
    ShardPanelGroup,
    ShardRef,
    read_shard,
    read_shard_v3,
    shard_panel,
    shard_panels,
)
from demand_forecast.pretrain.periods import pretrain_period
from demand_forecast.pretrain.sampler import V3BatchBuilder


@dataclass
class V3Pairs:
    """Fixed selected rows with their real grids and per-cell scaling periods."""

    groups: list[ShardPanelGroup]
    items: np.ndarray


@dataclass(frozen=True)
class V3SelectionCell:
    """Immutable pair membership in one predetermined frequency/horizon cell."""

    key: str
    horizon: int
    group: ShardPanelGroup
    items: tuple[tuple[int, int], ...]


def selection_pairs_v3(shards, cfg: PretrainConfig) -> tuple[V3SelectionCell, ...]:
    """Freeze capacity-capped, balanced cell quotas over real hold-out sources.

    A row may appear at several horizons. Within a cell it is sampled once,
    without replacement, with its final full future as the validation window.
    """
    records = []
    for shard in sorted(shards, key=lambda ref: ref.index):
        loaded = read_shard_v3(shard.path, kind=shard.kind, grid_policy=cfg.grid_policy)
        offset = len(records)
        records.extend(replace(row, row_index=offset + i) for i, row in enumerate(loaded.records))
    groups = shard_panels(records, period_policy="frequency")
    horizons = FEV_HORIZONS if cfg.horizon_policy == "fev_set" else (cfg.forecast.horizon,)
    candidates = []
    for group in groups:
        minimum = max(cfg.forecast.min_context, group.panel.seasonal_period + 1)
        key = (
            f"{group.panel.freq.freqstr}|m={group.panel.seasonal_period}"
            f"|calendar={int(group.calendar_available)}"
        )
        for horizon in horizons:
            items = [
                (index, series.length - horizon)
                for index, series in zip(group.row_indices, group.panel.series, strict=True)
                if series.length - horizon >= minimum
            ]
            if items:
                candidates.append((key, horizon, group, items))
    candidates.sort(key=lambda cell: (cell[0], cell[1]))
    budget = min(4096, cfg.corpus.val_pairs_per_kind)
    if not candidates or budget < len(candidates):
        raise ValueError("real selection requires at least one pair in every available cell")
    quotas = [0] * len(candidates)
    remaining = min(budget, sum(len(cell[3]) for cell in candidates))
    while remaining:
        for i, cell in enumerate(candidates):
            if quotas[i] < len(cell[3]) and remaining:
                quotas[i] += 1
                remaining -= 1
    rng = np.random.default_rng(cfg.seed)
    cells = tuple(
        V3SelectionCell(
            f"{key}|h={horizon}",
            horizon,
            group,
            tuple(items[int(i)] for i in rng.choice(len(items), size=quota, replace=False)),
        )
        for (key, horizon, group, items), quota in zip(candidates, quotas, strict=True)
    )
    # Retain only sampled histories, shared across horizons, rather than whole shards.
    selected_rows = {}
    for cell in cells:
        selected_rows.setdefault(id(cell.group), set()).update(index for index, _ in cell.items)
    compact = {}
    for group in groups:
        if id(group) in selected_rows:
            chosen = selected_rows[id(group)]
            retained = [
                (i, series)
                for i, series in zip(group.row_indices, group.panel.series, strict=True)
                if i in chosen
            ]
            compact[id(group)] = replace(
                group,
                row_indices=tuple(i for i, _ in retained),
                panel=replace(group.panel, series=[series for _, series in retained]),
            )
    return tuple(replace(cell, group=compact[id(cell.group)]) for cell in cells)


def evaluate_selection_v3(model, cells, **options) -> dict:
    """Macro-average fixed cells; any non-finite cell invalidates the source."""
    scores = {}
    builders = {}
    for cell in cells:
        key = id(cell.group)
        if key not in builders:
            builders[key] = V3BatchBuilder(
                [cell.group],
                max(c.horizon for c in cells if c.group is cell.group),
                options["max_context"],
                options["encoder"],
                scale_floor_frac=options.get("scale_floor_frac", 0.0),
            )
        result = evaluate_pairs_v3(
            model,
            V3Pairs([cell.group], np.asarray(cell.items, dtype=np.int64)),
            **(
                options | {"horizon": cell.horizon, "slice_horizon": True, "builder": builders[key]}
            ),
        )
        scores[cell.key] = result | {
            "period": cell.group.panel.seasonal_period,
            "horizon": cell.horizon,
        }
    failed = [key for key, score in scores.items() if not np.isfinite(score["SQL"])]
    return {
        "SQL": float(np.mean([score["SQL"] for score in scores.values()]))
        if not failed
        else float("nan"),
        "cells": scores,
        "failed_cells": failed,
        "n_items": sum(score["n_items"] for score in scores.values()),
    }


def holdout_pairs_v3(shards, cfg: PretrainConfig) -> V3Pairs:
    """Keep the legacy pair quota/seed while preserving each selected row's grid."""
    selected = []
    for shard in shards:
        loaded = read_shard_v3(shard.path, kind=shard.kind, grid_policy=cfg.grid_policy)
        eligible = [r for r in loaded.records if r.series.length >= cfg.forecast.horizon + 8]
        indices = np.random.default_rng(cfg.seed).choice(
            len(eligible), size=min(cfg.corpus.val_pairs_per_kind, len(eligible)), replace=False
        )
        for index in indices:
            selected.append(replace(eligible[index], row_index=len(selected)))
    return V3Pairs(
        shard_panels(selected, period_policy=cfg.period_policy),
        np.array(
            [(r.row_index, r.series.length - cfg.forecast.horizon) for r in selected],
            dtype=np.int64,
        ).reshape(-1, 2),
    )


def evaluate_pairs_v3(model, source: V3Pairs, *, scale_floor_frac=0.0, **kwargs) -> dict:
    """Use the training adapter and pool SQL cells with each row's own period."""
    horizon, quantiles = kwargs["horizon"], kwargs["quantiles"]
    batch_size = kwargs["batch_size"]
    builder = kwargs.get("builder")
    if builder is None:
        builder = V3BatchBuilder(
            source.groups,
            horizon,
            kwargs["max_context"],
            kwargs["encoder"],
            scale_floor_frac=scale_floor_frac,
        )
    rows = {
        index: (series, group.panel.seasonal_period)
        for group in source.groups
        for index, series in zip(group.row_indices, group.panel.series, strict=True)
    }
    labels = np.full((len(source.items), horizon), np.nan)
    predictions = np.empty((*labels.shape, len(quantiles)))
    scales = np.empty(len(labels))
    last = np.full(len(labels), np.nan)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                items = source.items[start : start + batch_size]
                batch = (
                    builder.build(items, horizon=horizon)
                    if kwargs.get("slice_horizon")
                    else builder.build(items)
                ).to(kwargs["device"])
                predictions[start : start + batch.size] = model.predict(batch).cpu().numpy()
        for row, (index, origin) in enumerate(source.items):
            series, period = rows[int(index)]
            past = series.target[:origin]
            future = series.target[origin : origin + horizon]
            labels[row, : len(future)] = future
            scales[row] = metrics.seasonal_error_per_item([past], period)[0]
            observed = past[~np.isnan(past)]
            if observed.size:
                last[row] = observed[-1]

        def scaled_errors(pred):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                values = metrics.quantile_loss(labels, pred, quantiles) / scales[:, None, None]
            return np.where(np.isfinite(values), values, np.nan)

        errors = scaled_errors(predictions)
        naive = scaled_errors(np.broadcast_to(last[:, None, None], predictions.shape))

        def score(errors):
            counts = np.isfinite(errors).sum(axis=(0, 1))
            means = np.full(len(quantiles), np.nan)
            np.divide(np.nansum(errors, axis=(0, 1)), counts, out=means, where=counts > 0)
            return float(np.mean(means))

        return {
            "SQL": score(errors),
            "naive_sql": score(naive),
            "n_items": len(labels),
            "n_excluded": int((~np.isfinite(errors).any(axis=(1, 2))).sum()),
        }
    finally:
        model.train(was_training)


def holdout_pairs(
    shard: ShardRef, *, horizon: int, n_pairs: int, seed: int
) -> tuple[Panel, np.ndarray]:
    """Sample eligible series without replacement; reserve their final horizon."""
    _positive(horizon, "horizon")
    _positive(n_pairs + 1, "n_pairs + 1")
    eligible = [series for series in read_shard(shard.path) if series.length >= horizon + 8]
    indices = np.random.default_rng(seed).choice(
        len(eligible), size=min(n_pairs, len(eligible)), replace=False
    )
    selected = [eligible[i] for i in indices]
    origins = np.array([series.length - horizon for series in selected], dtype=np.int64)
    return shard_panel(selected), origins


def download_external(
    config_name: str, root: Path, revision: str | None = CHRONOS_REVISION
) -> Path:
    """Download and persist a Chronos config during EC2 setup, never evaluation."""
    from datasets import load_dataset

    destination = Path(root) / config_name
    dataset = load_dataset(CORPUS_REPO, config_name, split="train", revision=revision)
    dataset.save_to_disk(str(destination))
    return destination


def external_pairs(config_name: str, root: Path, *, horizon: int) -> tuple[Panel, np.ndarray]:
    """Load a saved weekly dataset, retaining real starts and its weekly anchor.

    Every row must have a regular weekly timestamp per target and at least a
    full horizon. No download or calendar fabrication occurs on this path.
    """
    from datasets import load_from_disk

    _positive(horizon, "horizon")
    dataset = load_from_disk(str(Path(root) / config_name))
    series = []
    freq = None
    for row in dataset:
        target = np.asarray(row["target"], dtype=np.float32)
        timestamps = pd.DatetimeIndex(row["timestamp"])
        if target.ndim != 1 or len(target) < horizon or len(timestamps) != len(target):
            raise ValueError("external series require aligned timestamps and a full horizon")
        if timestamps.hasnans:
            raise ValueError("external timestamps must form a regular weekly grid")
        anchor = pd.offsets.Week(weekday=timestamps[0].dayofweek).freqstr
        if not timestamps.equals(pd.date_range(timestamps[0], periods=len(target), freq=anchor)):
            raise ValueError("external timestamps must form a regular weekly grid")
        if freq is not None and anchor != freq:
            raise ValueError("external series must share a weekly anchor")
        freq = anchor
        series.append(Series(str(row["id"]), timestamps[0], len(target), target, {}, {}))
    panel = Panel(series, CORPUS_ROLES, freq or PRETRAIN_FREQ, 1, "real")
    return panel, np.array([item.length - horizon for item in series], dtype=np.int64)


def evaluate_pairs(
    model: ForecastModel,
    panel: Panel,
    pairs: Iterable[tuple[int, int]],
    *,
    horizon: int,
    max_context: int,
    encoder: StaticEncoder,
    quantiles: Sequence[float],
    batch_size: int = 128,
    device: torch.device | str,
) -> dict[str, float | int]:
    """Score ordered pairs in original units using full histories and lag one.

    ``n_excluded`` counts items with no finite scaled quantile-loss cell, using
    the same quotient filtering as SQL. Partial missing labels alone do not
    exclude an item. The naive forecast repeats the last non-NaN past value;
    an entirely unobserved history has an undefined naive forecast.
    """
    _positive(batch_size, "batch_size")
    pairs = _items_array(panel, pairs)
    builder = BatchBuilder(panel, horizon, max_context, encoder, calendar=False)
    y_true = np.full((len(pairs), horizon), np.nan, dtype=np.float64)
    q_pred = np.empty((*y_true.shape, len(quantiles)), dtype=np.float64)
    y_past = []
    last_observed = np.full(len(pairs), np.nan, dtype=np.float64)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = builder.build(pairs[start : start + batch_size]).to(device)
                # predict already denormalises, orders knots and applies support/fallbacks.
                q_pred[start : start + batch.size] = model.predict(batch).cpu().numpy()
            for row, (index, origin) in enumerate(pairs):
                target = panel.series[index].target
                labels = target[origin : origin + horizon]
                y_true[row, : len(labels)] = labels
                past = target[:origin]
                y_past.append(past)
                observed = past[~np.isnan(past)]
                if observed.size:
                    last_observed[row] = observed[-1]
            result = metrics.summarize(y_true, q_pred, y_past, m=1, quantiles=quantiles)
            scales = metrics.seasonal_error_per_item(y_past, m=1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                quotients = metrics.quantile_loss(y_true, q_pred, quantiles) / scales[:, None, None]
            naive = np.broadcast_to(last_observed[:, None, None], q_pred.shape)
            return result | {
                "n_items": len(pairs),
                "n_excluded": int(np.count_nonzero(~np.isfinite(quotients).any(axis=(1, 2)))),
                "naive_sql": metrics.sql(y_true, naive, y_past, m=1, quantiles=quantiles),
            }
    finally:
        model.train(was_training)


@dataclass
class ValidationSet:
    """Panels sampled once per session, with fixed evaluation settings.

    Legacy diagnostics concatenate per-shard quotas in index order. V2 selects
    on the synthetic mean; v3 selects on frozen real frequency/horizon cells.
    Synthetic and external scores remain separately inspectable.
    """

    tsmixup: tuple[Panel, np.ndarray] | None
    kernel_synth: tuple[Panel, np.ndarray] | None
    external: tuple[Panel, np.ndarray] | None
    encoder: StaticEncoder
    horizon: int
    max_context: int
    quantiles: tuple[float, ...]
    batch_size: int = 128
    diagnostics: dict[str, tuple[Panel, np.ndarray] | None] = field(default_factory=dict)
    v3_sources: dict[str, V3Pairs | None] | None = None
    scale_floor_frac: float = 0.0
    selection: dict[str, tuple[V3SelectionCell, ...]] | None = None

    @classmethod
    def build(
        cls, cfg: PretrainConfig, refs: list[ShardRef], encoder: StaticEncoder
    ) -> ValidationSet:
        """Select local hold-outs by kind/index and require configured external data."""
        sets = {}
        v3_sources = {} if cfg.recipe == "v3" else None
        for kind, (_, held_out) in cfg.corpus.selections().items():
            wanted = set(held_out)
            selected = sorted(
                (ref for ref in refs if ref.kind == kind and ref.index in wanted),
                key=lambda ref: ref.index,
            )
            if v3_sources is not None:
                v3_sources[kind] = holdout_pairs_v3(selected, cfg) if selected else None
                sets[kind] = None
                continue
            chunks = [
                holdout_pairs(
                    ref,
                    horizon=cfg.forecast.horizon,
                    n_pairs=cfg.corpus.val_pairs_per_kind,
                    seed=cfg.seed,
                )
                for ref in selected
            ]
            sets[kind] = (
                (
                    shard_panel([series for panel, _ in chunks for series in panel.series]),
                    np.concatenate([origins for _, origins in chunks]),
                )
                if chunks
                else None
            )
        external = None
        config_name = cfg.corpus.external_validation
        if config_name is not None:
            directory = cfg.corpus.root / config_name
            if not directory.is_dir():
                raise FileNotFoundError(f"external validation directory not found: {directory}")
            external = external_pairs(config_name, cfg.corpus.root, horizon=cfg.forecast.horizon)
        if v3_sources is not None:
            v3_sources["external"] = None
            if external is not None:
                panel, origins = external
                panel = replace(
                    panel,
                    seasonal_period=pretrain_period(panel.freq)
                    if cfg.period_policy == "frequency"
                    else 1,
                )
                v3_sources["external"] = V3Pairs(
                    [ShardPanelGroup(panel, tuple(range(len(panel))), True)],
                    np.column_stack((np.arange(len(origins)), origins)),
                )
        selection = None
        if cfg.recipe == "v3":
            selection = {}
            selection_shards = cfg.corpus.selection_shards
            if selection_shards is None:
                selection_shards = {
                    kind: held_out
                    for kind, (_, held_out) in cfg.corpus.selections().items()
                    if kind in ("chronos_real", "gifteval_pretrain") and held_out
                }
            if not selection_shards:
                raise ValueError("recipe v3 selection requires real held-out shards")
            for kind, indices in selection_shards.items():
                selected = [ref for ref in refs if ref.kind == kind and ref.index in indices]
                if {ref.index for ref in selected} != set(indices):
                    raise ValueError(f"missing real selection shards for {kind}")
                selection[kind] = selection_pairs_v3(selected, cfg)
        return cls(
            selection=selection,
            tsmixup=sets.pop("tsmixup", None),
            kernel_synth=sets.pop("kernel_synth", None),
            diagnostics=sets,
            external=external,
            encoder=encoder,
            horizon=cfg.forecast.horizon,
            max_context=cfg.forecast.max_context,
            quantiles=tuple(cfg.forecast.quantiles),
            batch_size=cfg.training.val_batch_size,
            v3_sources=v3_sources,
            scale_floor_frac=cfg.forecast.scale_floor_frac,
        )

    def evaluate(self, model: ForecastModel, *, device: torch.device | str) -> dict:
        """Report corpus selection scores and external diagnostics, including gaps."""
        result = dict.fromkeys(
            (
                "val_sql_tsmixup",
                "val_sql_kernel_synth",
                "val_sql",
                "ext_sql",
                "ext_sql_naive",
                "n_excluded_tsmixup",
                "n_excluded_kernel_synth",
                "n_excluded_external",
            )
        )
        sources = {
            "tsmixup": self.tsmixup,
            "kernel_synth": self.kernel_synth,
            "external": self.external,
            **self.diagnostics,
        }
        if self.v3_sources is not None:
            sources = self.v3_sources
        for kind, source in sources.items():
            if kind in self.diagnostics:
                result[f"val_sql_{kind}"] = result[f"n_excluded_{kind}"] = None
            if source is None:
                continue
            options = dict(
                horizon=self.horizon,
                max_context=self.max_context,
                encoder=self.encoder,
                quantiles=self.quantiles,
                batch_size=self.batch_size,
                device=device,
            )
            if self.v3_sources is not None:
                scores = evaluate_pairs_v3(
                    model, source, scale_floor_frac=self.scale_floor_frac, **options
                )
            else:
                panel, origins = source
                scores = evaluate_pairs(model, panel, enumerate(origins), **options)
            result[f"n_excluded_{kind}"] = scores["n_excluded"]
            if kind == "external":
                result["ext_sql"] = scores["SQL"]
                result["ext_sql_naive"] = scores["naive_sql"]
            else:
                result[f"val_sql_{kind}"] = scores["SQL"]
        corpus_scores = [
            result[key] if result[key] is not None else float("nan")
            for key in ("val_sql_tsmixup", "val_sql_kernel_synth")
        ]
        result["val_sql"] = float(np.mean(corpus_scores))
        if self.selection is not None:
            selection_scores = {
                kind: evaluate_selection_v3(
                    model,
                    cells,
                    max_context=self.max_context,
                    encoder=self.encoder,
                    quantiles=self.quantiles,
                    batch_size=self.batch_size,
                    device=device,
                    scale_floor_frac=self.scale_floor_frac,
                )
                for kind, cells in self.selection.items()
            }
            result["selection_sources"] = selection_scores
            result["selection_pair_policy"] = selection_pair_policy_hash()
            result["val_sql"] = float(
                np.mean([score["SQL"] for score in selection_scores.values()])
            )
            # Label both calendar variants separately from historical v2 measurements.
            result["ext_sql_calendar_on"] = result.pop("ext_sql")
            result["ext_sql_naive_calendar_on"] = result.pop("ext_sql_naive")
            if self.external is not None:
                source = self.v3_sources["external"]
                historical = evaluate_pairs_v3(
                    model,
                    V3Pairs(
                        [replace(group, calendar_available=False) for group in source.groups],
                        source.items,
                    ),
                    horizon=self.horizon,
                    max_context=self.max_context,
                    encoder=self.encoder,
                    quantiles=self.quantiles,
                    batch_size=self.batch_size,
                    device=device,
                )
                result["ext_sql_calendar_off"] = historical["SQL"]
                result["ext_sql_naive_calendar_off"] = historical["naive_sql"]
        return result
