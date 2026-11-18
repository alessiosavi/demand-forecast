"""P6: paired inputs, common scoring frames, and evidence-based T1 verdicts."""

import copy
import importlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from demand_forecast.data.episode.schema import (
    CH_STEP_FLAG,
    N_SUMMARY,
    ROLE_PEER,
    STEP_FAMILIES,
    SUMMARY_FEATURES,
)
from demand_forecast.models.v5 import V5Model
from demand_forecast.pretrain.v5 import loop
from demand_forecast.pretrain.v5.support import supported
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
from tests.helpers.v5_loop import config

PATH_NAMES = (
    "static_category",
    "multi_label",
    "static_number",
    "category_mean",
    "category_spread",
    "category_timing",
    "known_driver",
    "planned_event",
    "past_driver",
    "calendar",
    "several_targets",
    "missing_vs_zero",
    "bank",
    "event_response",
    "label_lag",
    "routing",
    "selection",
)


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.t1")


def tiny():
    base = config(steps=8)
    return replace(
        base,
        model=replace(base.model, n_experts=2, top_k=2, expert_enc_layers=1),
        sampler=replace(base.sampler, designate_target=True, horizon=(16, 32), contexts=(64,)),
    )


@pytest.fixture
def small(monkeypatch):
    module = api()
    monkeypatch.setattr(module, "T1_BASE", tiny())
    paths = copy.deepcopy(module.PATHS)
    for path, entry in paths.items():
        source = entry["both"].setdefault("source", {})
        source.update(max_items=20, length=(96, 192))
        if path == "bank":
            source.update(freq="h", length=(400, 480))
    monkeypatch.setattr(module, "PATHS", paths)
    return module


def changed(left, right, prefix=""):
    result = set()
    for key in left:
        name = f"{prefix}{key}"
        if isinstance(left[key], dict):
            result |= changed(left[key], right[key], name + ".")
        elif left[key] != right[key]:
            result.add(name)
    return result


def test_paths_recipes_37_valid_arms_and_exact_overrides():
    module = api()
    assert set(module.PATHS) == set(PATH_NAMES)
    assert sum(len(entry["arms"]) for entry in module.PATHS.values()) == 37
    for path, entry in module.PATHS.items():
        for mechanisms in entry["sets"]:
            PanelRecipe(mechanisms=mechanisms)
        true = module.t1_config(path, "true", seed=23, base=tiny())
        assert true.seed == 23 and true.model == tiny().model and true.optim == tiny().optim
        assert true.sources[0].recipes == entry["sets"]
        assert true.sampler.max_cells is None and true.sampler.designate_target
        for arm, overrides in entry["arms"].items():
            built = module.t1_config(path, arm, seed=23, base=tiny())
            expected = set()
            for section, values in overrides.items():
                if section in ("source", "sampler"):
                    expected.update(
                        f"{'sources' if section == 'source' else section}.{key}" for key in values
                    )
                else:
                    expected.add(section)
            before, after = asdict(true), asdict(built)
            before["sources"], after["sources"] = before["sources"][0], after["sources"][0]
            assert changed(before, after) == expected
        assert all(a in entry["arms"] and b in entry["arms"] for a, b, _ in entry["comparisons"])
    bank = module.t1_config("bank", "null_all", seed=0)
    assert bank.sampler.summary_off == SUMMARY_FEATURES
    assert (
        module.t1_config("event_response", "null", seed=0).sources[0].control_inputs == "attribute"
    )
    assert module.t1_config("missing_vs_zero", "null", seed=0).sampler.origins == 1
    with pytest.raises(ValueError):
        module.t1_config("bank", "typo", seed=0)


@pytest.mark.parametrize("path", PATH_NAMES)
def test_eval_set_supported_present_labelled_deterministic(small, path):
    episodes = small.eval_set(path, episodes=2, max_panels=60, min_count=1)
    assert len(episodes) >= 2
    assert episodes == small.eval_set(path, episodes=2, max_panels=60, min_count=1)
    cfg = small.t1_config(path, "true", seed=0, base=tiny())
    for entry in episodes:
        generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
        assert generated.recipe.heldout
        mechanisms = {r["mechanism"] for r in entry["drivers"]} or set(generated.recipe.mechanisms)
        assert all(
            supported(
                generated,
                m,
                entry["item_id"],
                entry["origin"],
                context=entry["context"],
                horizon=entry["horizon"],
                cutoff=entry["cutoff"],
            )
            for m in mechanisms
        )
        built = small._build(generated, entry, cfg)
        assert built.label_mask.any() and not built.fallback[built.keep].any()
        assert small._present(path, generated, built, mechanisms)
        assert not built.label_mask[:, ~built.keep].any()


def test_eval_limit_required_bank_quota_and_digest_order(small):
    short = small.eval_set("known_driver", episodes=40, max_panels=1)
    assert len(short) <= 1
    episodes = small.eval_set("bank", episodes=2, min_count=2, max_panels=60)
    assert len(episodes) >= 6
    for name in ("seasonal_lag", "intermittent", "level_shift"):
        assert sum(name in e["slices"] for e in episodes) >= 2
    assert small.eval_digest(episodes) != small.eval_digest(episodes[::-1])


def test_forty_episodes_and_default_required_slice_minimum(small):
    assert len(small.eval_set("known_driver", episodes=40, max_panels=100)) == 40
    episodes = small.eval_set("bank", episodes=40, min_count=30, max_panels=500)
    assert len(episodes) >= 90
    assert all(
        sum(m in e["slices"] for e in episodes) >= 30
        for m in ("seasonal_lag", "intermittent", "level_shift")
    )


def test_recorded_category_and_event_slices(small):
    generated = generate_typed_panel(
        PanelRecipe(("dynamic_category_mean",), n_items=2, length=(96, 96), seed=5)
    )
    record = generated.inputs_of("dynamic_category_mean")[0]
    params = generated.params["dynamic_category_mean"]
    labels = params["values"]
    rare, late = (labels[params["process"][k]] for k in ("rare", "late"))
    ordinary = next(v for v in labels if v not in (rare, late))
    item = generated.complete.items[0]
    values = item.dynamic[record.name]
    values[:] = ordinary
    values[17], values[18] = late, rare
    assert set(small._slices("category_mean", generated, item.item_id, item.start + 16, 8)) == {
        record.role,
        "rare",
        "late_first",
    }
    values[5] = late
    assert set(small._slices("category_mean", generated, item.item_id, item.start + 16, 8)) == {
        record.role,
        "rare",
    }
    event = generate_typed_panel(
        PanelRecipe(("planned_event",), n_items=2, length=(96, 96), seed=6)
    )
    item = event.complete.items[0]
    values = item.dynamic[event.inputs_of("planned_event")[0].name]
    values[:] = 0
    values[17] = 1
    assert small._slices("planned_event", event, item.item_id, item.start + 16, 8) == (
        "first_event",
    )
    values[5] = 1
    assert small._slices("planned_event", event, item.item_id, item.start + 16, 8) == ()


@pytest.mark.parametrize("field", ["cal_ids", "calh_ids"])
def test_calendar_presence_requires_both_arrays_and_valid_steps(small, field):
    entry = small.eval_set("calendar", episodes=1, min_count=1, max_panels=30)[0]
    generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
    built = small._build(generated, entry, small.t1_config("calendar", "true", seed=0, base=tiny()))
    ids = getattr(built, field)
    ids.reshape(-1, 6)[0] = -1
    assert not small._present("calendar", generated, built, {"calendar"})


def test_null_checkpoint_controls_do_not_contaminate_true_reference(small, monkeypatch):
    entry = small.eval_set("bank", episodes=1, min_count=1, max_panels=60)[0]
    saved = small.t1_config("bank", "null_all", seed=17, base=tiny())
    monkeypatch.setattr(small, "load_model", lambda *a, **kw: (V5Model(tiny().model), saved))
    build = small._build
    observed = []

    def observe(generated, episode, config):
        result = build(generated, episode, config)
        observed.append(result)
        return result

    monkeypatch.setattr(small, "_build", observe)
    small.evaluate(Path("unused"), "bank", "null_all", [entry])
    assert observed[0].row_summary_flags.any()
    assert not observed[1].row_summary_flags.any()


def test_absent_screened_candidate_is_counted_and_still_scored(small, monkeypatch):
    episodes = small.eval_set("known_driver", episodes=1, min_count=1, max_panels=30)
    monkeypatch.setattr(small, "load_model", lambda *a, **kw: (V5Model(tiny().model), tiny()))
    build = small._build

    def screened(generated, entry, config):
        result = build(generated, entry, config)
        if config.sources[0].control:
            result.row_names = ["screened"] * len(result.row_names)
        return result

    monkeypatch.setattr(small, "_build", screened)
    result = small.evaluate(Path("unused"), "known_driver", "null", episodes)
    assert result["without_candidate"] == 1
    assert len(result["loss"]) == 1 and np.isfinite(result["loss"][0])


@pytest.mark.parametrize("path", PATH_NAMES)
def test_evaluate_all_arms_same_requests_cells_and_finite_losses(small, monkeypatch, path):
    episodes = small.eval_set(path, episodes=1, max_panels=60, min_count=1)
    assert episodes
    model = V5Model(tiny().model)
    # Loading is replaced here only; builds, gates, model, pruning and scoring are real.
    monkeypatch.setattr(small, "load_model", lambda *a, **kw: (model, tiny()))
    real_build = small.EpisodeBuilder.build
    requests = []

    def observe(builder, request):
        assert request.drop in (
            frozenset(),
            frozenset(("siblings",)),
            frozenset(("rolling", "lags", "zeros")),
        )
        item = builder.panel.items[request.item_index]
        built = real_build(builder, request)
        kept = int(np.flatnonzero(built.keep)[0])
        requests.append(
            (
                item.item_id,
                request.target,
                request.origins,
                request.cutoff,
                built.label_mask[:, kept].copy(),
            )
        )
        return built

    monkeypatch.setattr(small.EpisodeBuilder, "build", observe)
    forward = model.forward
    batches = []

    def observe_batch(batch, *args, **kwargs):
        for b in range(len(batch.context_target)):
            kept = (batch.keep[b] & batch.forecast_mask[b]).nonzero().item()
            batches.append(
                (
                    batch.context_target[b],
                    batch.origin_patches[b].tolist(),
                    batch.label_mask[b, :, kept].tolist(),
                )
            )
        return forward(batch, *args, **kwargs)

    monkeypatch.setattr(model, "forward", observe_batch)
    reference = None
    batch_reference = None
    for arm in small.PATHS[path]["arms"]:
        requests.clear()
        batches.clear()
        result = small.evaluate(Path("unused"), path, arm, episodes, batch_episodes=2)
        assert len(result["loss"]) == len(episodes) and np.isfinite(result["loss"]).all()
        assert result["eval_digest"] == small.eval_digest(episodes)
        assert set(result["pruned"]) == {"rows", "tokens", "cells"}
        observed = [(a, b, c, d, mask.tolist()) for a, b, c, d, mask in requests]
        if reference is None:
            reference = observed
            batch_reference = copy.deepcopy(batches)
        assert observed == reference
        assert batches == batch_reference


def test_real_export_checkpoint_and_rolled_repeat(small, tmp_path):
    trained = loop.train(tiny(), tmp_path)
    episodes = small.eval_set("routing", episodes=3, min_count=1, max_panels=30)
    left = small.evaluate(trained.checkpoint, "routing", "null_rolled", episodes)
    assert left == small.evaluate(trained.checkpoint, "routing", "null_rolled", episodes)


def test_missing_zero_erases_history_and_scores_in_true_frame(small, monkeypatch):
    entry = small.eval_set("missing_vs_zero", episodes=1, min_count=1, max_panels=30)[0]
    generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
    true_cfg = small.t1_config("missing_vs_zero", "true", seed=0, base=tiny())
    null_cfg = small.t1_config("missing_vs_zero", "null", seed=0, base=tiny())
    true = small._build(generated, entry, true_cfg)
    null = small._build(generated, entry, null_cfg)
    zeroed = copy.deepcopy(generated)
    for item in zeroed.panel.items:
        for values in item.targets.values():
            prefix = values[: max(0, entry["origin"] - item.start)]
            prefix[np.isnan(prefix)] = 0
    same = small._build(zeroed, entry, null_cfg)
    for field in (
        "x",
        "row_summary",
        "row_summary_flags",
        "cand_summary",
        "cand_flags",
        "attr_stats",
        "loc",
        "scale",
    ):
        np.testing.assert_array_equal(getattr(null, field), getattr(same, field))
    assert not np.array_equal(true.scale, null.scale) or not np.array_equal(true.loc, null.loc)
    model = V5Model(tiny().model)
    monkeypatch.setattr(small, "load_model", lambda *a, **kw: (model, tiny()))

    def oracle(batch, gates, **kwargs):
        original = (
            torch.as_tensor(true.labels[:, :1], dtype=torch.float64) * true.scale[0] + true.loc[0]
        )
        prediction = (original - batch.loc[:, None, :1, None]) / batch.scale[:, None, :1, None]
        quantiles = prediction[..., None].expand(-1, -1, -1, -1, len(model.config.quantiles))
        return type("Output", (), {"quantiles": quantiles})()

    monkeypatch.setattr(model, "forward", oracle)
    for arm in ("true", "null"):
        assert small.evaluate(Path("unused"), "missing_vs_zero", arm, [entry])[
            "loss"
        ] == pytest.approx([0], abs=1e-12)


def test_forced_drivers_zero_inherited_peer_rows(small, monkeypatch):
    episodes = small.eval_set("selection", episodes=3, min_count=1, max_panels=50)
    model = V5Model(tiny().model)
    monkeypatch.setattr(small, "load_model", lambda *a, **kw: (model, tiny()))
    real_gates = model.gates
    checked = []

    def gates(batch, **kwargs):
        result = real_gates(batch, **kwargs)
        forced = kwargs.get("zero_candidates")
        if forced is not None and forced.any():
            for b, names in enumerate(batch.cand_names):
                for c in range(len(names)):
                    if forced[b, c]:
                        peers = (
                            (batch.row_cand[b] == c)
                            & (batch.row_role[b] == ROLE_PEER)
                            & batch.row_mask[b]
                        )
                        assert (result.row[b, peers] == 0).all()
                        checked.append(int(peers.sum()))
        return result

    monkeypatch.setattr(model, "gates", gates)
    result = small.evaluate(Path("unused"), "selection", "true", episodes, forced=True)
    assert checked and sum(checked) > 0
    assert set(result["forced"]) == {r["mechanism"] for e in episodes for r in e["drivers"]}
    assert all(len(values) == len(episodes) for values in result["forced"].values())


def test_pruning_reports_real_rows_tokens_and_padded_cells_removed(small, monkeypatch):
    entry = small.eval_set("selection", episodes=1, min_count=1, max_panels=60)[0]
    generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
    built = small._build(
        generated, entry, small.t1_config("selection", "true", seed=0, base=tiny())
    )
    model = V5Model(tiny().model)
    with torch.no_grad():
        model.gate_net.net[-1].bias.fill_(-100)
    monkeypatch.setattr(small, "load_model", lambda *a, **kw: (model, tiny()))
    result = small.evaluate(Path("unused"), "selection", "true", [entry])
    # Selection has only a forecast row, past inputs, peers and a panel row.
    # Every independent input gate is zero, leaving exactly the forecast row.
    removed_rows = len(built.row_names) - 1
    assert removed_rows > 0
    assert result["pruned"] == {
        "rows": removed_rows,
        "tokens": removed_rows * (entry["context"] // 16)
        + len(built.attr_stats)
        + len(built.static_num),
        "cells": removed_rows * entry["context"],
    }


@pytest.mark.parametrize("path", PATH_NAMES)
def test_presence_can_reject_missing_inputs(small, path):
    entry = small.eval_set(path, episodes=1, min_count=1, max_panels=60)[0]
    generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
    cfg = small.t1_config(path, "true", seed=0, base=tiny())
    built = small._build(generated, entry, cfg)
    mechanisms = {r["mechanism"] for r in entry["drivers"]} or set(generated.recipe.mechanisms)
    assert small._present(path, generated, built, mechanisms)
    if path == "missing_vs_zero":
        item = next(i for i in generated.panel.items if i.item_id == entry["item_id"])
        for values in item.targets.values():
            values[np.isnan(values)] = 0
        assert not supported(
            generated,
            "missing_vs_zero",
            item.item_id,
            entry["origin"],
            context=entry["context"],
            horizon=entry["horizon"],
            cutoff=entry["cutoff"],
        )
        return
    # Change the panel, then let real screening and feature construction reject
    # it; clearing reporting names on an already-built episode is insufficient.
    missing = copy.deepcopy(generated)
    if path == "calendar":
        missing = replace(missing, panel=replace(missing.panel, epoch=None))
    elif path == "several_targets":
        items = tuple(
            replace(i, targets={entry["target"]: i.targets[entry["target"]]})
            for i in missing.panel.items
        )
        missing = replace(missing, panel=replace(missing.panel, items=items))
    elif path == "bank" or (path == "routing" and mechanisms == {"intermittent"}):
        for item in missing.panel.items:
            for values in item.targets.values():
                values[: max(0, entry["origin"] - item.start)] = np.nan
    else:
        # Screening can retain an all-missing named column. Presence is a names
        # contract, so the negative panel omits the recorded inputs entirely.
        items = tuple(replace(i, static={}, dynamic={}) for i in missing.panel.items)
        missing = replace(missing, panel=replace(missing.panel, items=items, roles={}))
    broken = small._build(missing, entry, cfg)
    assert not small._present(path, missing, broken, mechanisms)


@pytest.mark.parametrize(
    "mechanism,family,summary",
    [
        ("seasonal_lag", "lags", "acf_long"),
        ("intermittent", "zeros", "adi"),
        ("level_shift", "rolling", "trend_strength"),
    ],
)
def test_bank_presence_requires_both_step_and_summary(small, mechanism, family, summary):
    entries = small.eval_set("bank", episodes=1, min_count=1, max_panels=60)
    entry = next(e for e in entries if mechanism in e["slices"])
    generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
    cfg = small.t1_config("bank", "true", seed=0, base=tiny())
    built = small._build(generated, entry, cfg)
    missing_summary = copy.deepcopy(built)
    missing_summary.row_summary_flags[
        built.forecast_rows[built.keep], SUMMARY_FEATURES.index(summary)
    ] = False
    assert not small._present("bank", generated, missing_summary, {mechanism})
    span = STEP_FAMILIES[family]
    built.x[..., CH_STEP_FLAG + span.start : CH_STEP_FLAG + span.stop] = 0
    assert not small._present("bank", generated, built, {mechanism})


def test_bank_null_all_clears_every_summary_and_keeps_target(small):
    entry = small.eval_set("bank", episodes=1, min_count=1, max_panels=60)[0]
    generated = generate_typed_panel(PanelRecipe(**entry["recipe"]))
    cfg = small.t1_config("bank", "null_all", seed=0, base=tiny())
    built = small._build(generated, entry, cfg)
    for name in ("row_summary", "row_summary_flags", "cand_summary", "cand_flags"):
        assert not getattr(built, name).any()
    assert not built.attr_stats[..., : 2 * N_SUMMARY].any()
    assert built.label_mask.any()


@pytest.mark.parametrize(
    "true,count,outcome",
    [(0.9, 50, "pass"), (1.0, 50, "fail"), (0.98, 50, "fail"), (0.9, 10, "inconclusive")],
)
def test_compare_hand_calculated_gains(true, count, outcome):
    result = api().compare([true] * count, [1.0] * count)
    assert result["outcome"] == outcome
    assert result["all"]["gain"] == pytest.approx(1 - true)
    assert result["all"]["n"] == count


def test_compare_paired_bootstrap_seed_and_required_slice():
    module = api()
    true = [0.8] * 40 + [1.1] * 30
    episodes = [{"slices": ("weak",) if i >= 40 else ("good",)} for i in range(70)]
    result = module.compare(
        true, [1.0] * 70, episodes=episodes, required=("weak",), resamples=300, seed=7
    )
    assert result == module.compare(
        true, [1.0] * 70, episodes=episodes, required=("weak",), resamples=300, seed=7
    )
    assert result["all"]["outcome"] == "pass"
    assert result["outcome"] == result["slices"]["weak"]["outcome"] == "fail"
    assert (
        module.compare([0.8] * 40, [1.0] * 40, episodes=episodes[:40], required=("absent",))[
            "outcome"
        ]
        == "inconclusive"
    )
    with pytest.raises(ValueError):
        module.compare([1.0], [])
    with pytest.raises(ValueError):
        module.compare([float("nan")] * 30, [1.0] * 30)
    assert module.compare([0.0] * 40, [0.0] * 40)["outcome"] == "inconclusive"


def test_bootstrap_pairs_episodes_and_gain_is_ratio_of_sums():
    # Every resampled pair has the same 10% gain, despite unequal episode weights.
    null = [0.1, 10.0] * 25
    result = api().compare([v * 0.9 for v in null], null, resamples=200)
    assert result["all"]["low"] == pytest.approx(0.1)
    assert result["all"]["high"] == pytest.approx(0.1)
    weighted = api().compare([0.0, 10.0] * 25, null, resamples=200)
    assert weighted["all"]["gain"] == pytest.approx(0.1 / 10.1)
    assert weighted["outcome"] == "fail"


def selection_inputs():
    driver = {"name": "signal", "mechanism": "static_category", "where": "static"}
    nuisance = {
        "name": "noise",
        "mechanism": "nuisance",
        "where": "static",
        "twin": "static_category",
    }
    episodes = [{"drivers": (driver,), "nuisance": (nuisance,), "slices": ()} for _ in range(40)]
    true = {
        "loss": [0.9] * 40,
        "gates": [{"signal": 1.0, "noise": 0.0} for _ in range(40)],
        "forced": {"static_category": [1.2] * 40},
        "pruned": {"rows": 10, "tokens": 20, "cells": 30},
    }
    return true, {"loss": [1.0] * 40}, {"loss": [0.9] * 40}, episodes


@pytest.mark.parametrize(
    "violation,failed",
    [
        (None, None),
        ("nuisance", "pruning"),
        ("drivers", "pruning"),
        ("forced", "forced"),
        ("shuffle", "shuffle"),
        ("gates_one", "gates_one"),
    ],
)
def test_selection_report_isolates_each_decision(violation, failed):
    true, null, ones, episodes = selection_inputs()
    if violation in ("nuisance", "drivers"):
        key, value = ("noise", 1.0) if violation == "nuisance" else ("signal", 0.0)
        for gates in true["gates"]:
            gates[key] = value
        if violation == "drivers":
            # Enough other live episodes for the forced decision; just over 5% drivers pruned.
            for gates in true["gates"][3:]:
                gates["signal"] = 1.0
            # Interventions on already-zero gates must not enter decision (ii).
            true["forced"]["static_category"] = [0.0] * 3 + [0.91] * 37
    elif violation == "forced":
        true["forced"]["static_category"] = [0.8] * 40
    elif violation == "shuffle":
        null["loss"] = [0.9] * 40
    elif violation == "gates_one":
        ones["loss"] = [0.8] * 40
    result = api().selection_report(true, null, ones, episodes)
    assert {
        key
        for key in ("pruning", "forced", "shuffle", "gates_one")
        if result[key]["outcome"] == "fail"
    } == ({failed} if failed else set())
    assert result["outcome"] == ("fail" if failed else "pass")
    assert result["pruned"] == true["pruned"]


def test_selection_absent_candidates_and_short_decisions_inconclusive():
    true, null, ones, episodes = selection_inputs()
    for gates in true["gates"]:
        gates.clear()
    result = api().selection_report(true, null, ones, episodes)
    assert result["pruning"]["outcome"] == result["forced"]["outcome"] == "inconclusive"
    true, null, ones, episodes = selection_inputs()
    true["loss"], true["gates"] = true["loss"][:10], true["gates"][:10]
    true["forced"]["static_category"] = true["forced"]["static_category"][:10]
    null["loss"], ones["loss"] = null["loss"][:10], ones["loss"][:10]
    result = api().selection_report(true, null, ones, episodes[:10])
    assert all(
        result[k]["outcome"] == "inconclusive"
        for k in ("pruning", "forced", "shuffle", "gates_one")
    )
