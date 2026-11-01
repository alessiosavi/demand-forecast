"""Deterministic corpus hold-outs and a separate, local external diagnostic."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demand_forecast import metrics
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.panel import Panel, Series
from demand_forecast.data.sampler import BatchBuilder, _items_array, _positive
from demand_forecast.models.base import ForecastModel
from demand_forecast.pretrain.config import PretrainConfig
from demand_forecast.pretrain.corpus import (
    CORPUS_REPO,
    CORPUS_ROLES,
    PRETRAIN_FREQ,
    ShardRef,
    read_shard,
    shard_panel,
)


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


def download_external(config_name: str, root: Path) -> Path:
    """Download and persist a Chronos config during EC2 setup, never evaluation."""
    from datasets import load_dataset

    destination = Path(root) / config_name
    dataset = load_dataset(CORPUS_REPO, config_name, split="train")
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

    Each selected shard contributes up to ``val_pairs_per_kind`` items; multiple
    shards of a kind are concatenated in shard-index order. Only the two corpus
    scores enter model selection. The external scores are diagnostic.
    """

    tsmixup: tuple[Panel, np.ndarray] | None
    kernel_synth: tuple[Panel, np.ndarray] | None
    external: tuple[Panel, np.ndarray] | None
    encoder: StaticEncoder
    horizon: int
    max_context: int
    quantiles: tuple[float, ...]
    batch_size: int = 128

    @classmethod
    def build(
        cls, cfg: PretrainConfig, refs: list[ShardRef], encoder: StaticEncoder
    ) -> ValidationSet:
        """Select local hold-outs by kind/index; an absent external set is optional."""
        sets = {}
        for kind in ("tsmixup", "kernel_synth"):
            wanted = set(getattr(cfg.corpus, f"holdout_{kind}"))
            selected = sorted(
                (ref for ref in refs if ref.kind == kind and ref.index in wanted),
                key=lambda ref: ref.index,
            )
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
        if config_name and (cfg.corpus.root / config_name).is_dir():
            external = external_pairs(config_name, cfg.corpus.root, horizon=cfg.forecast.horizon)
        return cls(
            **sets,
            external=external,
            encoder=encoder,
            horizon=cfg.forecast.horizon,
            max_context=cfg.forecast.max_context,
            quantiles=tuple(cfg.forecast.quantiles),
            batch_size=cfg.training.val_batch_size,
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
        for kind in ("tsmixup", "kernel_synth", "external"):
            source = getattr(self, kind)
            if source is None:
                continue
            panel, origins = source
            scores = evaluate_pairs(
                model,
                panel,
                enumerate(origins),
                horizon=self.horizon,
                max_context=self.max_context,
                encoder=self.encoder,
                quantiles=self.quantiles,
                batch_size=self.batch_size,
                device=device,
            )
            result[f"n_excluded_{kind}"] = scores["n_excluded"]
            if kind == "external":
                result["ext_sql"] = scores["SQL"]
                result["ext_sql_naive"] = scores["naive_sql"]
            else:
                result[f"val_sql_{kind}"] = scores["SQL"]
        corpus_scores = [
            result[key]
            for key in ("val_sql_tsmixup", "val_sql_kernel_synth")
            if result[key] is not None
        ]
        if corpus_scores:
            observed = [score for score in corpus_scores if not np.isnan(score)]
            result["val_sql"] = float(np.mean(observed)) if observed else float("nan")
        return result
