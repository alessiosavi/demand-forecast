"""Corpus-free recipe plans and estimates from measured source/horizon cells."""

from __future__ import annotations

import math

from demand_forecast.data.frequency import CANONICAL_CALENDAR_SCHEMA
from demand_forecast.data.sampler import TARGET_LAG_SCHEMA
from demand_forecast.pretrain.config import FEV_HORIZONS, SELECTION_PAIR_POLICY, PretrainConfig


def horizons(cfg: PretrainConfig) -> tuple[int, ...]:
    return FEV_HORIZONS if cfg.horizon_policy == "fev_set" else (cfg.forecast.horizon,)


def source_mix(cfg: PretrainConfig) -> dict[str, float]:
    if cfg.corpus.sources is not None:
        weights = {kind: source.weight for kind, source in cfg.corpus.sources.items()}
    else:
        every = cfg.corpus.kernel_synth_every
        weights = {
            "tsmixup": 1 - 1 / every if every else 1,
            "kernel_synth": 1 / every if every else 0,
        }
    total = sum(weights.values())
    return {kind: weight / total for kind, weight in weights.items() if weight > 0}


def measured_duration(cfg: PretrainConfig, probe: dict, *, completed_steps: int = 0) -> dict:
    """Weight measured cells by source mix and feasible uniform horizon draws.

    Early live samples may miss cells. Use that source's measured mean there and
    publish the imputed cells, rather than presenting sparse samples as coverage.
    A probe's explicitly infeasible cells are excluded from its horizon support.
    """
    if not isinstance(probe, dict):
        raise ValueError("probe JSON must be a mapping")
    measurements = probe.get("source_horizon_timings", {})
    weighted_step = 0.0
    missing, imputed = [], []
    steps_by_source, distributions = {}, {}
    mix = source_mix(cfg)
    if probe.get("measurement_config") is not None and probe[
        "measurement_config"
    ] != measurement_config(cfg):
        raise ValueError("probe configuration does not match the requested recipe/model/budget")
    for kind, weight in mix.items():
        cells = measurements.get(kind, {})
        observed = [c for c in cells.values() if c.get("steps", 0) > 0]
        if not observed:
            missing.append(kind)
            continue
        mean = sum(c["train_s"] for c in observed) / sum(c["steps"] for c in observed)
        support = [str(h) for h in horizons(cfg) if cells.get(str(h), {}).get("feasible", True)]
        if not support:
            missing.append(kind)
            continue
        distribution = probe.get("horizon_distribution", {}).get(kind)
        if distribution is None:
            distribution = dict.fromkeys(support, 1 / len(support))
        support = [h for h in support if distribution.get(h, 0.0) > 0]
        total = sum(distribution.get(h, 0.0) for h in support)
        if total <= 0 or not math.isfinite(total):
            raise ValueError(f"invalid measured horizon distribution for {kind}")
        distributions[kind] = {h: distribution.get(h, 0.0) / total for h in support}
        source_step = 0.0
        for horizon in support:
            cell = cells.get(horizon, {})
            if cell.get("steps", 0):
                value = cell["train_s"] / cell["steps"]
            else:
                value = mean
                imputed.append(f"{kind}/{horizon}")
            source_step += value * distributions[kind][horizon]
        weighted_step += weight * source_step
        steps_by_source[kind] = cfg.training.max_steps * weight
    if missing:
        return {"status": "collecting", "missing_sources": missing, "imputed_cells": imputed}
    if not math.isfinite(weighted_step) or weighted_step <= 0:
        raise ValueError("duration requires finite positive measured source step times")
    steps = cfg.training.max_steps
    n_evals = math.ceil(steps / cfg.training.eval_every)
    n_ckpts = math.ceil(steps / cfg.training.checkpoint_every)
    # Charge the measured foreground shard wait per sampled batch. No assumed
    # 50k-series shard size: real-source lengths and explicit visit quotas differ.
    shard_per_step = 0.0
    for kind, cells in measurements.items():
        if kind not in mix:
            continue
        observed = {h: c for h, c in cells.items() if c.get("steps", 0)}
        fallback_load = sum(c.get("shard_load_s", 0.0) for c in observed.values()) / sum(
            c["steps"] for c in observed.values()
        )
        source_load = sum(
            probability
            * (
                cells[h].get("shard_load_s", 0.0)
                / max(1, cells[h].get("steps_per_shard", cells[h]["steps"]))
                if h in observed
                else fallback_load
            )
            for h, probability in distributions[kind].items()
        )
        shard_per_step += mix[kind] * source_load
    eval_s = probe.get("eval_s", 0.0)  # Includes raw/EMA and snapshot write.
    checkpoint_s = probe.get("checkpoint_s", 0.0)
    remaining = max(0, steps - completed_steps)
    remaining_evals = n_evals - completed_steps // cfg.training.eval_every
    remaining_ckpts = n_ckpts - completed_steps // cfg.training.checkpoint_every
    if not remaining:
        remaining_evals = remaining_ckpts = 0
    total_s = (
        probe.get("init_s", 0.0)
        + steps * (weighted_step + shard_per_step)
        + n_evals * eval_s
        + n_ckpts * checkpoint_s
        + 300
    ) * 1.15
    remaining_s = (
        remaining * (weighted_step + shard_per_step)
        + max(0, remaining_evals) * eval_s
        + max(0, remaining_ckpts) * checkpoint_s
        + (300 if remaining else 0)
    ) * 1.15
    return dict(
        status="measured" if not imputed and probe.get("evaluation_measured", True) else "partial",
        s_per_step=weighted_step,
        steps_per_hour=3600 / weighted_step,
        source_mix=mix,
        steps_by_source=steps_by_source,
        horizon_distribution=distributions,
        imputed_cells=imputed,
        evaluation_measured=probe.get("evaluation_measured", True),
        eval_s=eval_s,
        shard_load_s_per_step=shard_per_step,
        n_evals=n_evals,
        n_ckpts=n_ckpts,
        total_s=total_s,
        remaining_s=remaining_s,
        minutes=math.ceil(total_s / 60),
        remaining_minutes=math.ceil(remaining_s / 60),
    )


def measurement_config(cfg: PretrainConfig) -> dict:
    """A portable guard against reusing timings from different tensor/eval shapes."""
    return cfg.model_dump(
        mode="json", exclude={"corpus": {"root"}, "device": True, "log_level": True}
    )


def group_horizon_distribution(cfg: PretrainConfig, groups) -> dict[str, float]:
    """Resolve group-weighted feasible horizon probabilities on a resident shard."""
    result = dict.fromkeys((str(h) for h in horizons(cfg)), 0.0)
    total = 0
    for group in groups:
        minimum = max(cfg.forecast.min_context, group.panel.seasonal_period + 1)
        lengths = [s.history_length for s in group.panel.series]
        counts = {h: sum(n >= minimum + h for n in lengths) for h in horizons(cfg)}
        feasible = [h for h, count in counts.items() if count]
        if feasible:
            usable = max(counts.values())
            total += usable
            for h in feasible:
                result[str(h)] += usable / len(feasible)
    return {h: count / total for h, count in result.items()} if total else {}


def pretrain_plan(cfg: PretrainConfig, probe: dict | None = None) -> dict:
    """Resolve semantic controls without opening a manifest or any shard."""
    return dict(
        config=cfg.model_dump(mode="json"),
        recipe=cfg.recipe,
        calendar_schema=list(CANONICAL_CALENDAR_SCHEMA) if cfg.recipe == "v3" else None,
        lag_schema=list(TARGET_LAG_SCHEMA) if cfg.recipe == "v3" else None,
        horizons=list(horizons(cfg)),
        selection_cells_policy=SELECTION_PAIR_POLICY if cfg.recipe == "v3" else None,
        selection_shards=cfg.corpus.selection_shards,
        source_mix=source_mix(cfg),
        duration_estimate=measured_duration(cfg, probe) if probe is not None else None,
    )
