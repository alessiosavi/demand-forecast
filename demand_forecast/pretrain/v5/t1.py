"""Frozen, paired T1 evaluations and their bootstrap decisions.

Training arms fit their own scales. Scoring always uses the true build's real
label cells and affine frame. Admission uses the common base's screening level
on true held-out panels.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest
from demand_forecast.data.episode.collate import collate
from demand_forecast.data.episode.schema import (
    CALENDAR_SIZES,
    CH_STEP_FLAG,
    PATCH,
    ROLE_SIBLING,
    STEP_FAMILIES,
    SUMMARY_FEATURES,
)
from demand_forecast.data.episode.screening import ScreeningConfig
from demand_forecast.models.v5.model import prune
from demand_forecast.synthetic.typed import (
    PanelRecipe,
    generate_typed_panel,
)
from demand_forecast.synthetic.typed.draft import stream

from .config import T1_BASE, SourceConfig, V5TrainConfig
from .loop import load_model
from .sampler import _postprocess, _supervised, plan_panel
from .sources import TypedSource, apply_control
from .support import has_prior_event, supported


def _path(
    sets,
    *,
    both=None,
    arms=None,
    slices=(),
    required=(),
    present="inputs",
    decide_on: str | None = None,
):
    if decide_on is not None and decide_on not in slices:
        raise ValueError(f"decide_on {decide_on!r} must be a registered slice")
    arms = {"null": {"source": {"control": "shuffle"}}} if arms is None else arms
    return {
        "sets": sets,
        "both": {} if both is None else both,
        "arms": {"true": {}, **arms},
        "comparisons": tuple(
            ("true", arm, arm != "null_steps") for arm in arms if arm != "gates_one"
        ),
        "slices": slices,
        "required": required,
        "present": present,
        "decide_on": decide_on,
    }


PATHS: dict[str, dict] = {
    name: _path(
        ((name,),),
        both={"sampler": {"cold_share": 1.0, "cold_steps": (3, 4), "episodes_per_panel": 1}},
    )
    for name in ("static_category", "multi_label")
}
PATHS["static_number"] = _path((("static_number",),))
CATEGORY_SLICES = ("rare", "late_first", "known", "past")
PATHS.update(
    category_mean=_path((("dynamic_category_mean",),), slices=CATEGORY_SLICES),
    category_spread=_path(
        (("dynamic_category_spread",),),
        both={"sampler": {"derive": ("increment_spread",)}},
        slices=CATEGORY_SLICES,
        decide_on="known",
    ),
    category_timing=_path(
        (("dynamic_category_timing",),),
        both={"sampler": {"derive": ("onset_rows",)}},
        slices=CATEGORY_SLICES,
        decide_on="known",
    ),
)
PATHS.update(
    known_driver=_path((("known_driver",),)),
    planned_event=_path((("planned_event",),), slices=("first_event",)),
    past_driver=_path((("delayed_past_driver",),), both={"sampler": {"derive": ("lag_rows",)}}),
    calendar=_path(
        (("calendar",),),
        both={"source": {"freq": "D", "length": (64, 300)}, "sampler": {"contexts": (64, 128)}},
        arms={"null": {"source": {"control": "no_dates"}}},
        present="calendar",
    ),
    several_targets=_path(
        (("multi_target",),),
        # Requested horizons of 1 to 4 steps: the sampler truncates a log-uniform draw, so the
        # upper bound of the range is exclusive.
        both={"sampler": {"designate_target": True, "horizon": (1, 5)}},
        arms={"null": {"sampler": {"drop_always": ("siblings",)}}},
        present="siblings",
    ),
    missing_vs_zero=_path(
        (("missing_vs_zero",),),
        both={"sampler": {"origins": 1}},
        arms={"null": {"source": {"control": "missing_as_zero"}}},
        present="always",
    ),
    bank=_path(
        (("seasonal_lag",), ("intermittent",), ("level_shift",)),
        both={"sampler": {"contexts": (256,)}},
        arms={
            "null_all": {
                "sampler": {
                    "drop_always": ("rolling", "lags", "zeros"),
                    "summary_off": SUMMARY_FEATURES,
                }
            },
            "null_steps": {"sampler": {"drop_always": ("rolling", "lags", "zeros")}},
        },
        slices=("seasonal_lag", "intermittent", "level_shift"),
        required=("seasonal_lag", "intermittent", "level_shift"),
        present="bank",
    ),
    event_response=_path(
        (("event_response",),),
        both={"sampler": {"first_event_only": True}},
        arms={"null": {"source": {"control": "shuffle", "control_inputs": "attribute"}}},
    ),
    label_lag=_path(
        (("label_lag",),),
        both={"sampler": {"derive": ("lag_rows",)}},
        arms={"null": {"source": {"control": "shuffle", "control_inputs": "attribute"}}},
    ),
    routing=_path(
        (("static_category",), ("known_driver",), ("intermittent",)),
        arms={"null_uniform": {"router": "uniform"}, "null_rolled": {"router": "rolled"}},
    ),
    selection=_path(
        (("static_category", "static_number", "delayed_past_driver"),),
        both={
            "source": {
                "n_nuisance": 3,
                "nuisance_from": ("static_category", "static_number", "delayed_past_driver"),
            }
        },
        arms={"null": {"source": {"control": "shuffle"}}, "gates_one": {"gates": "ones"}},
    ),
)


def t1_config(path: str, arm: str, *, seed: int, base: V5TrainConfig = T1_BASE) -> V5TrainConfig:
    """Apply only the registered source, sampler, router and gate overrides."""
    if path not in PATHS or arm not in PATHS[path]["arms"]:
        raise ValueError(f"unknown T1 path/arm: {path}/{arm}")
    entry = PATHS[path]
    source = SourceConfig("typed", 1, recipes=entry["sets"])
    sampler = base.sampler
    switches = {}
    for overrides in (entry["both"], entry["arms"][arm]):
        source = replace(source, **overrides.get("source", {}))
        sampler = replace(sampler, **overrides.get("sampler", {}))
        switches.update({k: overrides[k] for k in ("router", "gates") if k in overrides})
    return replace(base, sources=(source,), sampler=sampler, seed=seed, **switches)


_BANK = {
    "seasonal_lag": ("lags", "acf_long"),
    "intermittent": ("zeros", "adi"),
    "level_shift": ("rolling", "trend_strength"),
}


def _present(path, generated, episode, mechanisms):
    """Check actual inputs and validity flags rather than feature values.

    Bank presence requires at least one context step with a valid flag in the
    mechanism's family (``flags.any()``), plus its named summary entry valid.
    """
    presence = PATHS[path]["present"]
    if presence == "always":
        return True
    if presence == "calendar":
        context = episode.cal_ids.reshape(-1, len(CALENDAR_SIZES))
        horizon = episode.calh_ids.reshape(len(episode.origin_patches), -1, len(CALENDAR_SIZES))[
            :, : episode.horizon
        ]
        return all(
            bool(
                (ids >= 0).any(axis=-1).all() and (ids >= -1).all() and (ids < CALENDAR_SIZES).all()
            )
            for ids in (context, horizon)
        )
    if presence == "siblings":
        return bool((episode.row_role == ROLE_SIBLING).any())
    for mechanism in mechanisms:
        records = generated.inputs_of(mechanism)
        if records:
            if not all(
                r.name in (episode.cand_names if r.where == "static" else episode.row_names)
                for r in records
            ):
                return False
        elif mechanism in _BANK:
            family, summary = _BANK[mechanism]
            span = STEP_FAMILIES[family]
            rows = episode.forecast_rows[episode.keep]
            flags = episode.x[rows, ..., CH_STEP_FLAG + span.start : CH_STEP_FLAG + span.stop]
            if (
                not flags.any()
                or not episode.row_summary_flags[rows, SUMMARY_FEATURES.index(summary)].all()
            ):
                return False
    return True


def _build(generated, entry, config):
    """Replay one frozen request; controls use the recipe seed, as TypedSource does."""
    source = config.sources[0]
    panel = apply_control(generated, source)
    if source.control == "missing_as_zero":
        items = []
        for item in panel.items:
            targets = {name: values.copy() for name, values in item.targets.items()}
            for values in targets.values():
                prefix = values[: max(0, entry["origin"] - item.start)]
                prefix[np.isnan(prefix)] = 0.0
            items.append(replace(item, targets=targets))
        panel = replace(panel, items=tuple(items))
    index = next(i for i, item in enumerate(panel.items) if item.item_id == entry["item_id"])
    request = EpisodeRequest(
        index,
        (entry["origin"],),
        entry["horizon"],
        entry["context"],
        entry["target"],
        entry["cutoff"],
        frozenset(config.sampler.drop_always),
        derive=frozenset(config.sampler.derive),
    )
    screening = ScreeningConfig(
        no_evidence=config.sampler.screen_no_evidence,
        fair_static_spread=config.sampler.screen_fair_spread,
    )
    episode = EpisodeBuilder(panel, screening).build(request)
    _postprocess(episode, replace(config.sampler, designate_target=True))
    return episode


def _slices(path, generated, item_id, origin, horizon):
    item = next(i for i in generated.complete.items if i.item_id == item_id)
    left, right = max(0, origin - item.start), max(0, origin + horizon - item.start)
    slices = []
    for mechanism in generated.recipe.mechanisms:
        if path == "bank":
            slices.append(mechanism)
        if mechanism.startswith("dynamic_category_"):
            record = generated.inputs_of(mechanism)[0]
            values = item.dynamic[record.name]
            params = generated.params[mechanism]
            slices.append(record.role)
            for key, name in (("rare", "rare"), ("late", "late_first")):
                label = params["values"][params["process"][key]]
                positions = np.flatnonzero(values == label)
                if key == "rare" and ((positions >= left) & (positions < right)).any():
                    slices.append(name)
                if key == "late" and len(positions) and left <= positions[0] < right:
                    slices.append(name)
        if mechanism == "planned_event":
            record = generated.inputs_of(mechanism)[0]
            events = item.dynamic[record.name][left:right]
            if not has_prior_event(item, record.name, origin) and (events == 1).any():
                slices.append("first_event")
    return tuple(slices)


def eval_set(
    path: str,
    *,
    base: V5TrainConfig = T1_BASE,
    episodes: int = 200,
    max_panels: int = 1000,
    seed: int = 10_000,
    min_count: int = 30,
) -> list[dict]:
    """Plan on true held-out panels, retaining at most one episode per panel.

    The training base supplies the screening level and optional presentations.
    The held-out sampler uses the fixed T1 settings and path overrides.
    A scoring episode has one origin. Admission, cutoff fallback, target choice
    and the nonconstant-prefix/observed-label checks belong to P3's planner.
    """
    if episodes < 1 or max_panels < 0 or min_count < 1:
        raise ValueError("episodes/min_count must be positive and max_panels nonnegative")
    planning_base = replace(T1_BASE, sampler=replace(T1_BASE.sampler, derive=base.sampler.derive))
    config = t1_config(path, "true", seed=seed, base=planning_base)
    source = replace(config.sources[0], heldout=True)
    sampler = replace(
        config.sampler,
        episodes_per_batch=1,
        episodes_per_panel=1,
        origins=1,
        path_share=1.0,
        drop=(),
        designate_target=True,
        screen_no_evidence=base.sampler.screen_no_evidence,
        screen_fair_spread=base.sampler.screen_fair_spread,
    )
    screening = ScreeningConfig(
        no_evidence=sampler.screen_no_evidence,
        fair_static_spread=sampler.screen_fair_spread,
    )
    results, counts = [], Counter()
    for i in range(max_panels):
        if len(results) >= episodes and all(
            counts[s] >= min_count for s in PATHS[path]["required"]
        ):
            break
        panel, record = TypedSource(source).panel(stream(seed, f"eval:{path}:{i}"))
        generated = record["generated"]
        _, plans, _ = plan_panel(
            panel, record, sampler, source, stream(seed, f"eval-plan:{path}:{i}")
        )
        if not plans or plans[0].mechanism is None:
            continue
        plan = plans[0]
        request = plan.request
        mechanisms = tuple(
            m
            for m in generated.recipe.mechanisms
            if supported(
                generated,
                m,
                plan.item_id,
                request.origins[0],
                context=request.context,
                horizon=request.horizon,
                cutoff=request.cutoff,
                first_event_only=sampler.first_event_only,
            )
        )
        if path != "selection" and len(mechanisms) != len(generated.recipe.mechanisms):
            continue
        built = EpisodeBuilder(panel, screening).build(request)
        _postprocess(built, sampler)
        if not _supervised(built) or not _present(path, generated, built, mechanisms):
            continue
        entry = {
            "recipe": asdict(generated.recipe),
            "item_id": plan.item_id,
            "target": request.target,
            "origin": request.origins[0],
            "horizon": request.horizon,
            "context": request.context,
            "cutoff": request.cutoff,
            "slices": _slices(path, generated, plan.item_id, request.origins[0], request.horizon),
            "drivers": tuple(asdict(r) for r in generated.inputs if r.mechanism in mechanisms),
            "nuisance": tuple(asdict(r) for r in generated.inputs if r.twin is not None),
        }
        results.append(entry)
        counts.update(entry["slices"])
    return results


def eval_digest(episodes: list[dict]) -> str:
    """Identify the complete ordered replay specification, independent of key order."""
    return hashlib.sha256(
        json.dumps(episodes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _losses(output, arm, true, quantiles):
    losses = []
    q = output.quantiles.new_tensor(quantiles)
    for b in range(arm.x.shape[0]):
        ai = (arm.keep[b] & arm.forecast_mask[b]).nonzero().flatten().item()
        ti = (true.keep[b] & true.forecast_mask[b]).nonzero().flatten().item()
        horizon = int(true.horizon[b])
        prediction = output.quantiles[b, :, ai, :horizon].double()
        prediction = (
            prediction * arm.scale[b, ai] + arm.loc[b, ai] - true.loc[b, ti]
        ) / true.scale[b, ti]
        error = true.labels[b, :, ti, :horizon, None].double() - prediction
        pinball = torch.maximum(q * error, (q - 1) * error)
        mask = true.label_mask[b, :, ti, :horizon] & true.origin_mask[b, :, None]
        if not mask.any():
            raise ValueError("frozen episode has no true label cells")
        losses.append(float(pinball[mask].mean()))
    return losses


def _sizes(batch):
    n, rows, patches = batch.x.shape[:3]
    origins, forecasts, horizon = batch.labels.shape[1:]
    cells = (
        n * rows * patches * PATCH + n * origins * (forecasts + batch.known_rows.shape[1]) * horizon
    )
    return {
        "rows": int(batch.row_mask.sum()),
        "tokens": cells // PATCH + int(batch.attr_mask.sum() + batch.static_num_mask.sum()),
        "cells": cells,
    }


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    path: str,
    arm: str,
    episodes: list[dict],
    *,
    batch_episodes: int = 32,
    forced: bool = False,
    derive_override: frozenset[str] | None = None,
) -> dict:
    """Score in frozen order, optionally overriding presentations for the scored arm."""
    if batch_episodes < 1:
        raise ValueError("batch_episodes must be positive")
    model, saved = load_model(checkpoint, weights="export")
    model.eval()
    # A null checkpoint already contains its training overrides. They cannot
    # contaminate the reference build or a different evaluation arm.
    base = replace(
        saved,
        sources=(),
        router="learned",
        gates="learned",
        sampler=replace(
            saved.sampler,
            drop=(),
            drop_always=(),
            summary_off=(),
            designate_target=True,
            max_cells=None,
            origins=1,
        ),
    )
    config = t1_config(path, arm, seed=saved.seed, base=base)
    true_config = t1_config(path, "true", seed=saved.seed, base=base)
    derived = [] if config.sampler.derive else None
    if derive_override is not None:
        config = replace(config, sampler=replace(config.sampler, derive=tuple(derive_override)))
    if config.router == "rolled":
        if len(episodes) < 2:
            raise ValueError("the rolled null needs at least two episodes")
        batch_episodes = max(2, batch_episodes)
    starts = list(range(0, len(episodes), batch_episodes))
    if config.router == "rolled" and len(starts) > 1 and len(episodes) - starts[-1] == 1:
        starts.pop()  # Merge the singleton tail into the preceding batch.
    mechanisms = sorted({r["mechanism"] for e in episodes for r in e["drivers"]})
    result = {
        "loss": [],
        "gates": [],
        "pruned": {"rows": 0, "tokens": 0, "cells": 0},
        "without_candidate": 0,
        "eval_digest": eval_digest(episodes),
    }
    if derived is not None:
        result["derive"] = list(config.sampler.derive)
        result["derived"] = derived
    if forced:
        result["forced"] = {m: [] for m in mechanisms}
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(episodes)
        entries = episodes[start:stop]
        true_episodes, arm_episodes = [], []
        for entry in entries:
            generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
            reference = _build(generated, entry, true_config)
            built = _build(generated, entry, config)
            if derived is not None:
                derived.append(
                    sorted(key[1] for key in built.row_keys if key and key[0] == "derived")
                )
            own = {r["mechanism"] for r in entry["drivers"]} or set(generated.recipe.mechanisms)
            result["without_candidate"] += not _present(path, generated, built, own)
            true_episodes.append(reference)
            arm_episodes.append(built)
        true_batch, batch = collate(true_episodes), collate(arm_episodes)
        gates = model.gates(batch, mode=config.gates)
        output = model(batch, gates, router=config.router)
        result["loss"].extend(_losses(output, batch, true_batch, model.config.quantiles))
        for b in range(len(entries)):
            named = {name: float(gates.cand[b, j]) for j, name in enumerate(batch.cand_names[b])}
            named.update(
                {name: float(gates.row[b, j]) for j, name in enumerate(batch.row_names[b])}
            )
            result["gates"].append(named)
        smaller, _ = prune(batch, gates)
        before, after = _sizes(batch), _sizes(smaller)
        for key in before:
            result["pruned"][key] += before[key] - after[key]
        if forced:
            for mechanism in mechanisms:
                candidates, rows = (
                    torch.zeros_like(batch.cand_mask),
                    torch.zeros_like(batch.row_mask),
                )
                active = []
                for b, entry in enumerate(entries):
                    names = {r["name"] for r in entry["drivers"] if r["mechanism"] == mechanism}
                    for j, name in enumerate(batch.cand_names[b]):
                        candidates[b, j] = name in names
                    for j, name in enumerate(batch.row_names[b]):
                        rows[b, j] = name in names
                    active.append(bool(candidates[b].any() or rows[b].any()))
                forced_gates = model.gates(
                    batch, mode=config.gates, zero_candidates=candidates, zero_rows=rows
                )
                losses = _losses(
                    model(batch, forced_gates, router=config.router),
                    batch,
                    true_batch,
                    model.config.quantiles,
                )
                result["forced"][mechanism].extend(
                    loss if present else None for loss, present in zip(losses, active)
                )
    return result


def _outcome(outcomes):
    outcomes = tuple(outcomes)
    return (
        "fail"
        if "fail" in outcomes
        else "inconclusive"
        if not outcomes or "inconclusive" in outcomes
        else "pass"
    )


def _interval(numerator, denominator=None, *, resamples=10_000, seed=0, min_count=30):
    """Paired percentile bootstrap; bounded allocation even for extended sets."""
    numerator = np.asarray(numerator, dtype=np.float64)
    n = len(numerator)
    denominator = np.ones(n) if denominator is None else np.asarray(denominator, dtype=np.float64)
    total = denominator.sum()
    record = {
        "gain": float(numerator.sum() / total) if total > 0 else None,
        "low": None,
        "high": None,
        "n": n,
    }
    if n < min_count or total <= 0:
        return record
    rng, draws = stream(seed, "bootstrap"), []
    for offset in range(0, resamples, 128):
        indices = rng.integers(n, size=(min(128, resamples - offset), n))
        bottom = denominator[indices].sum(axis=1)
        if (bottom <= 0).any():
            return record
        chunk = numerator[indices].sum(axis=1) / bottom
        draws.append(chunk)
    record["low"], record["high"] = (
        float(v) for v in np.quantile(np.concatenate(draws), (0.05, 0.95))
    )
    return record


def _loss_arrays(true, null):
    true, null = np.asarray(true, dtype=np.float64), np.asarray(null, dtype=np.float64)
    if true.ndim != 1 or null.ndim != 1 or true.shape != null.shape:
        raise ValueError("paired losses must have equal one-dimensional shapes")
    if (
        not np.isfinite(true).all()
        or not np.isfinite(null).all()
        or (true < 0).any()
        or (null < 0).any()
    ):
        raise ValueError("losses must be finite and nonnegative")
    return true, null


def _cohort_indices(episodes: list[dict], name: str) -> np.ndarray:
    return np.asarray(
        [i for i, episode in enumerate(episodes) if name in episode["slices"]], dtype=int
    )


def compare(
    true: list[float],
    null: list[float],
    *,
    episodes: list[dict] | None = None,
    required: tuple[str, ...] = (),
    decide_on: str | None = None,
    resamples: int = 10_000,
    seed: int = 0,
    min_count: int = 30,
) -> dict:
    """Compare paired losses and optionally decide on one named cohort."""
    true, null = _loss_arrays(true, null)
    if resamples < 1 or min_count < 1:
        raise ValueError("resamples and min_count must be positive")
    if episodes is not None and len(episodes) != len(true):
        raise ValueError("episodes and losses must have equal lengths")
    if required and episodes is None:
        raise ValueError("required slices need episodes")
    if decide_on is not None and episodes is None:
        raise ValueError("a decision cohort needs episodes")

    def decision(indices):
        record = _interval(
            null[indices] - true[indices],
            null[indices],
            resamples=resamples,
            seed=seed,
            min_count=min_count,
        )
        record["outcome"] = (
            "inconclusive"
            if record["low"] is None
            else "pass"
            if record["gain"] >= 0.05 and record["low"] > 0
            else "fail"
        )
        return record

    names = sorted(set(required) | {s for e in episodes or [] for s in e["slices"]})
    if decide_on is not None and decide_on not in names:
        names = sorted([*names, decide_on])
    slices = {name: decision(_cohort_indices(episodes or [], name)) for name in names}
    overall = decision(np.arange(len(true)))
    if decide_on is not None:
        return {"overall": overall, "slices": slices, "outcome": slices[decide_on]["outcome"]}
    return {
        "all": overall,
        "slices": slices,
        "outcome": _outcome([overall["outcome"], *(slices[s]["outcome"] for s in required)]),
    }


def _removal(on, off, episodes: list[dict], *, decide_on: str | None) -> dict:
    """Paired gain with the derived-off replay as the loss baseline."""
    on, off = _loss_arrays(on, off)
    if decide_on is not None:
        indices = _cohort_indices(episodes, decide_on)
        on, off = on[indices], off[indices]
    return _interval(off - on, off)


def selection_report(
    true: dict,
    null: dict,
    gates_one: dict,
    episodes: list[dict],
    *,
    min_count: int = 30,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """R2 decisions over present inputs, as candidates or rows, and live drivers.

    Every decision uses ``min_count`` episodes as its evidence floor; all
    bootstrap intervals use the requested ``resamples`` and ``seed``.
    """
    if resamples < 1 or min_count < 1:
        raise ValueError("resamples and min_count must be positive")
    learned, shuffled = _loss_arrays(true["loss"], null["loss"])
    _, ones = _loss_arrays(true["loss"], gates_one["loss"])
    if len(episodes) != len(learned) or len(true["gates"]) != len(learned):
        raise ValueError("selection episodes, losses and gates must be aligned")
    pruning = {}
    for kind, key, threshold in (("nuisance", "nuisance", 0.5), ("drivers", "drivers", 0.05)):
        values, indices = [], set()
        for i, (entry, gates) in enumerate(zip(episodes, true["gates"])):
            for record in entry[key]:
                if record["name"] in gates:
                    values.append(gates[record["name"]] == 0)
                    indices.add(i)
        share = float(np.mean(values)) if values else None
        passed = share is not None and (
            share >= threshold if kind == "nuisance" else share <= threshold
        )
        pruning[kind] = {
            "share_zero": share,
            "inputs": len(values),
            "n": len(indices),
            "outcome": "inconclusive" if len(indices) < min_count else "pass" if passed else "fail",
        }
    pruning["outcome"] = _outcome(pruning[k]["outcome"] for k in ("nuisance", "drivers"))
    forced = {}
    mechanisms = sorted({r["mechanism"] for e in episodes for r in e["drivers"]})
    for mechanism in mechanisms:
        losses = true.get("forced", {}).get(mechanism, [None] * len(episodes))
        if len(losses) != len(episodes):
            raise ValueError("forced losses must be aligned with episodes")
        differences = []
        for i, entry in enumerate(episodes):
            gates = true["gates"][i]
            live = any(
                r["mechanism"] == mechanism and gates.get(r["name"], 0) != 0
                for r in entry["drivers"]
            )
            if live and losses[i] is not None:
                if not np.isfinite(losses[i]):
                    raise ValueError("forced losses must be finite")
                differences.append(losses[i] - learned[i])
        record = _interval(differences, min_count=min_count, resamples=resamples, seed=seed)
        record["outcome"] = (
            "inconclusive" if record["low"] is None else "pass" if record["low"] > 0 else "fail"
        )
        forced[mechanism] = record
    forced["outcome"] = _outcome(r["outcome"] for r in forced.values())
    advantage = _interval(learned - ones, ones, min_count=min_count, resamples=resamples, seed=seed)
    advantage["outcome"] = (
        "inconclusive"
        if advantage["high"] is None
        else "pass"
        if advantage["high"] < 0.01
        else "fail"
    )
    report = {
        "pruning": pruning,
        "forced": forced,
        "shuffle": compare(learned, shuffled, min_count=min_count, resamples=resamples, seed=seed),
        "gates_one": advantage,
        "pruned": dict(true["pruned"]),
    }
    report["outcome"] = _outcome(
        report[k]["outcome"] for k in ("pruning", "forced", "shuffle", "gates_one")
    )
    return report
