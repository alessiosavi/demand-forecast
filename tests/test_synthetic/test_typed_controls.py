"""Null controls, candidate queries and the typed generator's model path."""

import os
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from demand_forecast.data.episode import (
    CH_SUPPORT,
    KIND_CATEGORY,
    KIND_FLAG,
    KIND_NUMBER,
    PATCH,
    ROLE_FORECAST,
    ROLE_KNOWN,
    ROLE_PAST,
    ROLE_PEER,
    EpisodeBuilder,
    Item,
    infer_kind,
    is_missing,
)
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.screening import screen
from demand_forecast.models.v5 import V5Config, V5Model, forecast
from demand_forecast.synthetic import typed

INPUT_RECIPES = (
    (
        "static_category",
        "multi_label",
        "hierarchy",
        "static_number",
        "whole_number_code",
        "known_driver",
        "planned_event",
        "delayed_past_driver",
        "event_response",
        "label_lag",
        "dynamic_category_mean",
    ),
    ("dynamic_category_spread",),
    ("dynamic_category_timing",),
)
TINY = V5Config(
    d_model=48,
    nhead=4,
    enc_layers=2,
    dec_layers=1,
    dim_feedforward=96,
    n_experts=2,
    top_k=1,
    expert_enc_layers=1,
    expert_dec_layers=1,
    dropout=0.0,
)


def targets_of(item):
    return next(iter(item.targets.values()))


def assert_array_equal(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert np.array_equal(left, right, equal_nan=left.dtype != object)


def assert_inputs_equal(left, right, records):
    assert (left.name, left.freq, left.epoch, left.aligned) == (
        right.name,
        right.freq,
        right.epoch,
        right.aligned,
    )
    for record in records:
        if record.where == "dynamic":
            assert left.roles[record.name] == right.roles[record.name]
    for a, b in zip(left.items, right.items, strict=True):
        assert (a.item_id, a.start) == (b.item_id, b.start)
        for record in records:
            if record.where == "static":
                assert a.static[record.name] == b.static[record.name]
            else:
                assert_array_equal(a.dynamic[record.name], b.dynamic[record.name])


def assert_diagnostics_equal(left, right):
    for field in ("parts", "scalars"):
        for item_id, values in getattr(left, field).items():
            other = getattr(right, field)[item_id]
            assert values.keys() == other.keys()
            for key in values:
                assert_array_equal(values[key], other[key])


def assert_targets_equal(left, right):
    for a, b in zip(left.items, right.items, strict=True):
        assert a.targets.keys() == b.targets.keys()
        for name in a.targets:
            assert_array_equal(a.targets[name], b.targets[name])


def assert_identity(generated):
    for item in generated.panel.items:
        parts = generated.parts[item.item_id]
        scalars = generated.scalars[item.item_id]
        latent = parts["base"].copy()
        for key in sorted(parts):
            if key.startswith("add:"):
                latent += parts[key]
        latent += scalars["sigma0"][:, None] * parts["mult"] * parts["noise"]
        expected = scalars["level"][:, None] * parts["occ"] * latent
        expected[parts["missing"]] = np.nan
        actual = np.stack([item.targets[name] for name in sorted(item.targets)])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mechanisms", INPUT_RECIPES)
@pytest.mark.parametrize("complete_inputs", (False, True))
def test_shuffle_keeps_everything_else(mechanisms, complete_inputs):
    g = typed.generate_typed_panel(
        typed.PanelRecipe(
            mechanisms,
            n_items=30,
            length=(100, 300),
            complete_inputs=complete_inputs,
            n_nuisance=3,
            seed=9,
        )
    )
    before = deepcopy(g)
    for record in g.inputs:
        shuffled = typed.shuffle_assignment(g, record.name, seed=13)
        repeated = typed.shuffle_assignment(g, record.name, seed=13)
        if record.mechanism == "hierarchy":
            moved = tuple(r.name for r in g.inputs_of("hierarchy"))
        elif record.twin == "hierarchy":
            moved = tuple(
                r.name for r in g.inputs if r.twin == "hierarchy" and r.process is record.process
            )
        else:
            moved = (record.name,)
        assert shuffled.shuffled == moved
        assert tuple(shuffled.donors) == moved
        assert shuffled.donors == repeated.donors
        assert shuffled.inputs == g.inputs and shuffled.params == g.params
        assert shuffled.recipe == g.recipe and shuffled.hosts == g.hosts
        assert shuffled.short_items == g.short_items and shuffled.availability == g.availability
        assert_diagnostics_equal(shuffled, g)
        remaining = tuple(r for r in g.inputs if r.name not in moved)
        for original, changed, again in (
            (g.panel, shuffled.panel, repeated.panel),
            (g.complete, shuffled.complete, repeated.complete),
        ):
            changed.validate()
            assert_inputs_equal(original, changed, remaining)
            assert_inputs_equal(changed, again, g.inputs)
            assert_targets_equal(original, changed)
        originals = {item.item_id: item for item in g.complete.items}
        for name in moved:
            assert shuffled.donors[name] == shuffled.donors[record.name]
            mapping = shuffled.donors[name]
            assert {donor for donor, _ in mapping.values()} == set(originals)
            for source, masked, complete in zip(
                g.panel.items, shuffled.panel.items, shuffled.complete.items, strict=True
            ):
                donor_id, offset = mapping[source.item_id]
                assert donor_id != source.item_id
                donor = originals[donor_id]
                if record.where == "static":
                    assert offset == 0
                    assert complete.static[name] == donor.static[name]
                    expected = (
                        source.static[name]
                        if is_missing(source.static[name])
                        else donor.static[name]
                    )
                    assert masked.static[name] == expected
                else:
                    values = donor.dynamic[name]
                    assert 0 <= offset < len(values)
                    expected = values[(np.arange(len(targets_of(source))) + offset) % len(values)]
                    assert_array_equal(complete.dynamic[name], expected)
                    missing = np.frompyfunc(is_missing, 1, 1)(source.dynamic[name]).astype(bool)
                    actual = masked.dynamic[name]
                    assert_array_equal(actual[~missing], expected[~missing])
                    assert_array_equal(actual[missing], source.dynamic[name][missing])
        if record.mechanism == "hierarchy":
            parent, child = moved
            parent_of = g.params["hierarchy"]["parent_of"]
            for item in shuffled.complete.items:
                assert parent_of[item.static[child]] == item.static[parent]
    assert_inputs_equal(g.panel, before.panel, g.inputs)
    assert_inputs_equal(g.complete, before.complete, g.inputs)
    assert_targets_equal(g.panel, before.panel)
    assert_diagnostics_equal(g, before)
    assert g.shuffled == () and g.donors == {}


def test_shuffle_refusals():
    g = typed.generate_typed_panel(typed.PanelRecipe(("known_driver",), n_items=2))
    with pytest.raises(ValueError):
        typed.shuffle_assignment(g, "unknown", seed=0)
    one = typed.generate_typed_panel(replace(g.recipe, n_items=1))
    with pytest.raises(ValueError):
        typed.shuffle_assignment(one, one.inputs[0].name, seed=0)


def test_shuffle_keeps_separate_nuisance_hierarchy_runs():
    g = typed.generate_typed_panel(
        typed.PanelRecipe(("hierarchy",), n_items=4, length=(32, 32), n_nuisance=2)
    )
    a, b, c, d = (r for r in g.inputs if r.twin == "hierarchy")
    assert a.process == c.process and a.process is not c.process
    for pair, other in (((a, b), (c, d)), ((c, d), (a, b))):
        for record in pair:
            shuffled = typed.shuffle_assignment(g, record.name, seed=0)
            assert shuffled.shuffled == tuple(r.name for r in pair)
            assert shuffled.donors[pair[0].name] == shuffled.donors[pair[1].name]
            assert_inputs_equal(g.complete, shuffled.complete, (*g.inputs_of("hierarchy"), *other))


@pytest.mark.parametrize(
    "invalid",
    (
        "different_process",
        "nonadjacent",
        "different_twin",
        "different_mechanism",
        "missing_twin",
        "extra_twin",
    ),
)
def test_shuffle_rejects_invalid_nuisance_hierarchy_pair(invalid):
    g = typed.generate_typed_panel(
        typed.PanelRecipe(("hierarchy",), n_items=4, length=(32, 32), n_nuisance=2)
    )
    original = g.inputs_of("hierarchy")
    a, b, c, d = (r for r in g.inputs if r.twin == "hierarchy")
    twins = {
        "different_process": (a, replace(b, process=deepcopy(b.process)), c, d),
        "nonadjacent": (a, c, b, d),
        "different_twin": (a, replace(b, twin="static_category"), c, d),
        "different_mechanism": (a, replace(b, mechanism="hierarchy"), c, d),
        "missing_twin": (a, c, d),
        "extra_twin": (a, b, replace(c, process=a.process), d),
    }[invalid]
    g = replace(g, inputs=(*original, *twins))
    with pytest.raises(ValueError, match=a.name):
        typed.shuffle_assignment(g, a.name, seed=0)


def test_two_items_and_successive_shuffles():
    g = typed.generate_typed_panel(
        typed.PanelRecipe(("known_driver", "multi_label"), n_items=2, length=(8, 12))
    )
    first, second = (r.name for r in g.inputs)
    a = typed.shuffle_assignment(g, first, seed=0)
    b = typed.shuffle_assignment(a, second, seed=0)
    assert b.shuffled == (first, second)
    assert b.donors[first] == a.donors[first]
    assert_inputs_equal(a.complete, b.complete, (g.inputs[0],))
    for mapping in b.donors.values():
        assert mapping["i0000"][0] == "i0001" and mapping["i0001"][0] == "i0000"


def continuous_effect(generated, mechanism, item):
    params = generated.params[mechanism]
    values = item.dynamic[generated.inputs_of(mechanism)[0].name]
    deviation = values.std(ddof=0)
    z = (values - values.mean()) / deviation if deviation else np.zeros_like(values)
    if mechanism == "known_driver":
        response = np.tanh(z)
    else:
        response = np.zeros_like(z)
        lag = params["lag"]
        response[lag:] = z[:-lag]
    return params["beta_sign"] * params["beta"] * response


@pytest.mark.parametrize("mechanism", ("known_driver", "delayed_past_driver"))
def test_shuffle_breaks_a_continuous_driver(mechanism):
    g = typed.generate_typed_panel(
        typed.PanelRecipe((mechanism,), n_items=60, length=(300, 600), complete_inputs=True)
    )
    shuffled = typed.shuffle_assignment(g, g.inputs_of(mechanism)[0].name, seed=7)
    actual, null = [], []
    for item, changed in zip(g.complete.items, shuffled.complete.items, strict=True):
        original = g.parts[item.item_id][f"add:{mechanism}"][0]
        np.testing.assert_allclose(continuous_effect(g, mechanism, item), original, atol=1e-12)
        actual.append(original)
        null.append(continuous_effect(shuffled, mechanism, changed))
    correlation = np.corrcoef(np.concatenate(actual), np.concatenate(null))[0, 1]
    assert abs(correlation) < 0.2, correlation


def test_products_of_two_suppliers():
    g = typed.generate_typed_panel(
        typed.PanelRecipe(
            ("static_number", "dynamic_category_spread"), n_items=30, complete_inputs=True
        )
    )
    attribute = g.inputs_of("static_number")[0].name
    column = g.inputs_of("dynamic_category_spread")[0].name
    log_x = np.log([item.static[attribute] for item in g.complete.items])
    z = (log_x - log_x.mean()) / log_x.std()
    for i, item in enumerate(g.complete.items):
        factors = np.array(
            [g.params["dynamic_category_spread"]["factor"][v] for v in item.dynamic[column]]
        )
        expected = np.exp(-g.params["static_number"]["noise"] * z[i]) * factors
        np.testing.assert_allclose(g.parts[item.item_id]["mult"], expected[None], atol=1e-12)
    for mechanisms in (
        ("intermittent", "whole_number_code"),
        ("intermittent", "whole_number_code", "missing_vs_zero"),
    ):
        g = typed.generate_typed_panel(
            typed.PanelRecipe(mechanisms, n_items=30, complete_inputs=True)
        )
        attribute = g.inputs_of("whole_number_code")[0].name
        for item in g.complete.items:
            p = g.parts[item.item_id]
            probability = (
                g.params["intermittent"]["p"][item.item_id]
                * g.params["whole_number_code"]["p"][item.static[attribute]]
            )
            assert_array_equal(p["occ"], (p["u"] < probability) & ~p["stockout"])
        assert_identity(g)


@pytest.mark.parametrize("mechanisms", INPUT_RECIPES)
def test_without_over_all_mechanisms_with_inputs(mechanisms):
    recipe = typed.PanelRecipe(mechanisms, n_items=30, length=(100, 300), n_nuisance=3, seed=9)
    full = typed.generate_typed_panel(recipe)
    for mechanism in mechanisms:
        cut = typed.generate_typed_panel(recipe, without=(mechanism,))
        assert cut.inputs == full.inputs and cut.params == full.params
        assert cut.hosts == full.hosts and cut.availability == full.availability
        assert_inputs_equal(cut.panel, full.panel, full.inputs)
        assert_inputs_equal(cut.complete, full.complete, full.inputs)
        changed_parts = {f"add:{mechanism}"}
        if mechanism in ("static_number", "dynamic_category_spread"):
            changed_parts.add("mult")
        if mechanism == "whole_number_code":
            changed_parts.add("occ")
        if mechanism == "hierarchy":
            changed_parts.add("noise")
        for item_id, parts in full.parts.items():
            assert f"add:{mechanism}" not in cut.parts[item_id]
            assert set(cut.parts[item_id]) == set(parts) - {f"add:{mechanism}"}
            for name in parts.keys() - changed_parts:
                assert_array_equal(parts[name], cut.parts[item_id][name])
            for name, value in full.scalars[item_id].items():
                if name == "rho_effective" and mechanism == "hierarchy":
                    assert_array_equal(cut.scalars[item_id][name], full.scalars[item_id]["rho"])
                else:
                    assert_array_equal(value, cut.scalars[item_id][name])
        assert_identity(cut)


def expected_necessary(g, mechanism):
    """Evaluate the table directly, independently of the controls implementation."""
    records = g.inputs_of(mechanism)
    if mechanism == "static_number":
        return g.short_items
    if mechanism == "delayed_past_driver":
        return tuple(
            i.item_id
            for i in g.complete.items
            if len(targets_of(i)) > g.params[mechanism]["lag"] + 16
        )
    attributes = tuple(r for r in records if r.where == "static")
    if not attributes:
        dynamic = next(r for r in records if r.where == "dynamic")
        minimum = 20 if dynamic.role == "past" else 0
        return tuple(i.item_id for i in g.complete.items if len(targets_of(i)) >= minimum)
    if mechanism == "hierarchy":
        children = set(g.params["hierarchy"]["parent_of"])
        attribute = next(
            r.name
            for r in attributes
            if {item.static[r.name] for item in g.complete.items} == children
        )
    else:
        attribute = next(r.name for r in records if r.where == "static")
    result = []
    for item in g.complete.items:
        if item.item_id not in g.short_items:
            continue
        peers = [
            peer
            for peer in g.complete.items
            if peer.item_id not in g.short_items
            and peer.start <= item.start - 16
            and peer.start + len(targets_of(peer)) > item.start
        ]
        if mechanism == "multi_label":
            supported = any(set(item.static[attribute]) & set(p.static[attribute]) for p in peers)
        else:
            supported = any(item.static[attribute] == p.static[attribute] for p in peers)
        if supported:
            result.append(item.item_id)
    return tuple(result)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("heldout", (False, True))
@pytest.mark.parametrize("singleton_share", (0.0, 0.25))
def test_necessary_items(seed, heldout, singleton_share):
    for mechanism in (*INPUT_RECIPES[0], *INPUT_RECIPES[1], *INPUT_RECIPES[2]):
        g = typed.generate_typed_panel(
            typed.PanelRecipe(
                (mechanism,),
                n_items=40,
                length=(128, 400),
                heldout=heldout,
                singleton_share=singleton_share,
                seed=seed,
            )
        )
        actual = typed.necessary_items(g, mechanism)
        assert actual == expected_necessary(g, mechanism), (mechanism, seed)
        assert len(g.hosts) == len(g.short_items)
        if mechanism == "static_category":
            assert actual
            attribute = g.inputs[0].name
            for item_id in actual:
                item = next(i for i in g.complete.items if i.item_id == item_id)
                assert item_id in g.short_items
                assert any(
                    p.item_id not in g.short_items
                    and p.static[attribute] == item.static[attribute]
                    and p.start + 16 <= item.start < p.start + len(targets_of(p))
                    for p in g.complete.items
                )
        assert actual == typed.necessary_items(
            replace(g, panel=replace(g.panel, items=())), mechanism
        )  # Availability is irrelevant: read complete.
    for mechanism in typed.MECHANISMS[:6]:
        g = typed.generate_typed_panel(typed.PanelRecipe((mechanism,), n_items=2))
        with pytest.raises(ValueError):
            typed.necessary_items(g, mechanism)


def test_necessary_items_boundaries_and_non_host_peer():
    g = typed.generate_typed_panel(
        typed.PanelRecipe(("static_category",), n_items=4, length=(32, 32))
    )
    name = g.inputs[0].name
    items = (
        Item("short", 20, {"target": np.ones(8)}, {}, {name: "shared"}),
        Item("host", 0, {"target": np.ones(64)}, {}, {name: "other"}),
        Item("peer", 4, {"target": np.ones(17)}, {}, {name: "shared"}),
    )
    g = replace(
        g,
        complete=replace(g.complete, items=items),
        short_items=("short",),
        hosts={"short": "host"},
    )
    assert typed.necessary_items(g, "static_category") == ("short",)
    for peer in (
        replace(items[2], start=5),  # Only 15 preceding steps.
        replace(items[2], targets={"target": np.ones(16)}),  # Ends exactly at the start.
    ):
        changed = replace(g, complete=replace(g.complete, items=(*items[:2], peer)))
        assert typed.necessary_items(changed, "static_category") == ()
    delayed = typed.generate_typed_panel(
        typed.PanelRecipe(("delayed_past_driver",), n_items=2, length=(32, 32))
    )
    boundary = delayed.params["delayed_past_driver"]["lag"] + 16
    items = tuple(
        replace(i, targets={"target": np.ones(boundary + k)})
        for k, i in enumerate(delayed.complete.items)
    )
    assert typed.necessary_items(
        replace(delayed, complete=replace(delayed.complete, items=items)), "delayed_past_driver"
    ) == (items[1].item_id,)


def test_quiet_origins_separate_fitting_from_context():
    found = False
    for seed in range(5):
        g = typed.generate_typed_panel(
            typed.PanelRecipe(
                ("event_response",),
                n_items=24,
                length=(600, 900),
                singleton_share=0.25,
                complete_inputs=True,
                seed=seed,
            )
        )
        attribute, column = (r.name for r in g.inputs_of("event_response"))
        counts = Counter(i.static[attribute] for i in g.complete.items)
        assert max(counts.values()) >= 2 and 1 in counts.values()
        singleton_labels = g.params["event_response"]["singletons"]
        assert singleton_labels and all(counts[label] == 1 for label in singleton_labels)
        for k, item in enumerate(g.complete.items):
            if item.static[attribute] not in singleton_labels:
                continue
            flag = item.dynamic[column]
            actual = typed.quiet_origins(g, item.item_id, context=64, horizon=16)
            expected = tuple(
                item.start + o
                for o in range(64, len(flag) - 16 + 1)
                if not flag[o - 64 : o].any() and flag[o : o + 16].any() and flag[: o - 64].any()
            )
            assert actual == expected
            for origin in actual:
                local = origin - item.start
                assert not flag[local - 64 : local].any()
                assert flag[local : local + 16].any() and flag[: local - 64].any()
                assert item.start <= origin - 64 and origin + 16 <= item.start + len(flag)
            cutoff = actual[0] if actual else item.start + len(flag) - 16
            stats = PanelStats(g.panel, "target", cutoff)
            candidate = next(
                c for c in screen(g.panel, stats, k, ("target",), cutoff) if c.name == attribute
            )
            assert candidate.removed is None
            found |= bool(actual)
    assert found
    other = typed.generate_typed_panel(typed.PanelRecipe(("static_category",)))
    with pytest.raises(ValueError):
        typed.quiet_origins(other, other.panel.items[0].item_id, context=64, horizon=16)


def test_quiet_origins_exact_boundaries():
    g = typed.generate_typed_panel(
        typed.PanelRecipe(("event_response",), n_items=1, length=(8, 8), complete_inputs=True)
    )
    name = g.inputs_of("event_response")[1].name
    flag = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float64)
    item = replace(g.complete.items[0], start=100, dynamic={name: flag})
    g = replace(
        g,
        complete=replace(g.complete, items=(item,)),
        panel=replace(g.panel, items=(deepcopy(item),)),
    )
    assert typed.quiet_origins(g, item.item_id, context=3, horizon=3) == (104, 105)
    assert typed.quiet_origins(g, item.item_id, context=8, horizon=1) == ()
    with pytest.raises(ValueError):
        typed.quiet_origins(g, "unknown", context=3, horizon=3)
    for context, horizon in ((0, 3), (3, 0), (-1, 3), (3, -1)):
        with pytest.raises(ValueError):
            typed.quiet_origins(g, item.item_id, context=context, horizon=horizon)


@pytest.mark.parametrize("seed", range(4))
def test_quiet_origins_with_partial_availability(seed):
    g = typed.generate_typed_panel(
        typed.PanelRecipe(("event_response",), n_items=24, length=(600, 900), seed=seed)
    )
    column = next(r.name for r in g.inputs_of("event_response") if r.where == "dynamic")
    exposed = {item.item_id: item for item in g.panel.items}
    count = old_count = 0
    for item in g.complete.items:
        truth = item.dynamic[column] == 1
        visible = np.array(
            [
                not is_missing(value) and value == 1
                for value in exposed[item.item_id].dynamic[column]
            ]
        )
        actual = typed.quiet_origins(g, item.item_id, context=64, horizon=16)
        for origin in actual:
            local = origin - item.start
            assert item.start <= origin - 64 and origin + 16 <= item.start + len(truth)
            assert not truth[local - 64 : local].any()
            assert visible[local : local + 16].any()
            assert visible[: local - 64].any()
        expected = tuple(
            item.start + o
            for o in range(64, len(truth) - 16 + 1)
            if not truth[o - 64 : o].any() and visible[o : o + 16].any() and visible[: o - 64].any()
        )
        assert actual == expected
        count += len(actual)
        old_count += sum(
            not truth[o - 64 : o].any() and truth[o : o + 16].any() and truth[: o - 64].any()
            for o in range(64, len(truth) - 16 + 1)
        )
    if seed == 0:
        assert 0 < count < old_count


@pytest.mark.parametrize("hidden", (0, 2, 5), ids=("prefix", "context", "horizon"))
def test_quiet_origins_truth_in_context_and_visible_events_elsewhere(hidden):
    g = typed.generate_typed_panel(typed.PanelRecipe(("event_response",), n_items=1, length=(8, 8)))
    column = next(r.name for r in g.inputs_of("event_response") if r.where == "dynamic")
    truth = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float64)
    if hidden == 2:
        truth[2] = 1
    visible = truth.copy()
    visible[hidden] = np.nan
    item = replace(g.complete.items[0], start=100, dynamic={column: truth})
    g = replace(
        g,
        complete=replace(g.complete, items=(item,)),
        panel=replace(g.panel, items=(replace(item, dynamic={column: visible}),)),
    )
    assert typed.quiet_origins(g, item.item_id, context=3, horizon=3) == ()


def last_aligned_origin(panel, k):
    item = panel.items[k]
    origin = (item.start + len(targets_of(item)) - 16) // PATCH * PATCH
    assert origin - item.start >= 64
    assert origin + 16 <= item.start + len(targets_of(item))
    return origin


def forecast_targets(episode):
    return {
        episode.row_names[row]
        for row in episode.forecast_rows
        if episode.row_role[row] == ROLE_FORECAST
    }


def candidate_of(episode, name):
    # Episode has static candidates and dynamic rows in separate name lists.
    if name in episode.cand_names:
        return "static", episode.cand_names.index(name)
    if name in episode.row_names:
        return "dynamic", episode.row_names.index(name)
    return None


@pytest.mark.parametrize("m", typed.MECHANISMS)
def test_every_mechanism_runs_through_the_model(m):
    g = typed.generate_typed_panel(
        typed.PanelRecipe((m,), n_items=12, length=(128, 400), n_nuisance=1, seed=7)
    )
    k = max(range(12), key=lambda i: len(targets_of(g.panel.items[i])))
    origin = last_aligned_origin(g.panel, k)
    builder = EpisodeBuilder(g.panel)
    requests = builder.plan_requests(k, origins=(origin,), horizon=16)
    assert len(requests) >= 1
    covered = set()
    for request in requests:
        episode = builder.build(request)
        covered |= forecast_targets(episode)
        for record in g.inputs_of(m):
            assert candidate_of(episode, record.name) is not None, (m, record.name)
            if record.where == "dynamic":
                row = episode.row_names.index(record.name)
                assert episode.row_role[row] == (
                    ROLE_KNOWN if record.role == "known" else ROLE_PAST
                )
        out = forecast(V5Model(TINY), g.panel, request, builder=builder, gates="ones")
        assert out.quantiles.shape[-2] == 16 and np.isfinite(out.quantiles).all(), m
        assert set(out.names) == forecast_targets(episode)
    assert covered == set(g.panel.items[k].targets), m
    for record in g.inputs_of(m):
        if record.where == "dynamic":
            prefix = g.panel.items[k].dynamic[record.name][: origin - g.panel.items[k].start]
            distinct = len({value for value in prefix if not is_missing(value)})
            if record.kind == "category" and distinct >= 3:
                assert infer_kind(prefix) == KIND_CATEGORY
            if record.kind == "flag" and distinct == 2:
                assert infer_kind(prefix) == KIND_FLAG
            if record.kind == "number":
                assert infer_kind(prefix) == KIND_NUMBER


@pytest.mark.parametrize("seed", range(5))
def test_a_supported_short_query_has_peers(seed):
    g = typed.generate_typed_panel(
        typed.PanelRecipe(
            ("static_category",), n_items=40, length=(128, 400), complete_inputs=True, seed=seed
        )
    )
    supported = typed.necessary_items(g, "static_category")
    assert supported
    k = next(k for k, item in enumerate(g.panel.items) if item.item_id == supported[0])
    origin = g.panel.items[k].start + 8
    builder = EpisodeBuilder(g.panel)
    episode = builder.build(builder.plan_requests(k, (origin,), horizon=16, context=64)[0])
    attribute = g.inputs[0].name
    tokens = [i for i, name in enumerate(episode.attr_names) if name.startswith(attribute + "=")]
    assert tokens and any(episode.attr_stats[i, -1] > 0 for i in tokens)
    peers = [
        i
        for i, name in enumerate(episode.row_names)
        if name.startswith(attribute + "=") and episode.row_role[i] == ROLE_PEER
    ]
    assert peers and any((episode.x[i, ..., CH_SUPPORT] > 0).any() for i in peers)


@pytest.mark.parametrize("seed", range(25))
def test_sampled_recipes_build_and_run(seed):
    recipe = typed.sample_recipe(seed)
    recipe = replace(recipe, n_items=min(recipe.n_items, 12), length=(128, 300))
    g = typed.generate_typed_panel(recipe)
    g.panel.validate()
    g.complete.validate()
    k = max(range(recipe.n_items), key=lambda i: len(targets_of(g.panel.items[i])))
    builder = EpisodeBuilder(g.panel)
    request = builder.plan_requests(k, (last_aligned_origin(g.panel, k),), horizon=16)[0]
    out = forecast(V5Model(TINY), g.panel, request, builder=builder, gates="ones")
    assert out.quantiles.shape[-2] == 16 and np.isfinite(out.quantiles).all()


@pytest.mark.parametrize("mechanisms", INPUT_RECIPES)
def test_nuisance_and_availability_preserve_all_original_effects(mechanisms):
    recipe = typed.PanelRecipe(
        (*typed.MECHANISMS[:6], *mechanisms), n_items=30, length=(100, 300), seed=4
    )
    original = typed.generate_typed_panel(recipe)
    for changed in (
        typed.generate_typed_panel(replace(recipe, n_nuisance=4)),
        typed.generate_typed_panel(replace(recipe, complete_inputs=True)),
    ):
        assert changed.inputs[: len(original.inputs)] == original.inputs
        assert changed.params == original.params
        assert changed.hosts == original.hosts and changed.short_items == original.short_items
        assert_diagnostics_equal(original, changed)
        assert_inputs_equal(original.complete, changed.complete, original.inputs)
        if not changed.recipe.complete_inputs:
            assert_inputs_equal(original.panel, changed.panel, original.inputs)
        assert_targets_equal(original.panel, changed.panel)
        assert_identity(changed)
        for item in changed.panel.items:
            parts = changed.parts[item.item_id]
            target = np.stack(list(item.targets.values()))
            assert np.isnan(target[parts["missing"]]).all()
            assert (target[parts["stockout"] & ~parts["missing"]] == 0).all()


@pytest.mark.parametrize("n_items", (1, 2))
@pytest.mark.parametrize("freq", ("D", "W-MON", "YE"))
def test_controls_on_short_panels(n_items, freq):
    mechanisms = (*typed.MECHANISMS[:2], *typed.MECHANISMS[3:12], *typed.MECHANISMS[14:])
    g = typed.generate_typed_panel(
        typed.PanelRecipe(mechanisms, n_items=n_items, length=(8, 8), freq=freq, heldout=True)
    )
    assert_identity(g)
    for m in mechanisms[5:]:
        assert typed.necessary_items(g, m) == expected_necessary(g, m)
    if n_items == 2:
        for record in g.inputs:
            shuffled = typed.shuffle_assignment(g, record.name, seed=1)
            assert_diagnostics_equal(g, shuffled)
            assert_targets_equal(g.panel, shuffled.panel)


def test_shuffle_deterministic_across_hash_seeds():
    script = """
import hashlib
import json
from dataclasses import asdict
import numpy as np
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel, shuffle_assignment

def normalise(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)

g = generate_typed_panel(PanelRecipe(
    ("hierarchy", "multi_label", "known_driver", "event_response"),
    n_items=12, length=(64, 100), n_nuisance=2, seed=9,
))
for record in g.inputs:
    g = shuffle_assignment(g, record.name, seed=5)
encoded = json.dumps(asdict(g), sort_keys=True, default=normalise).encode()
print(hashlib.sha256(encoded).hexdigest())
"""
    digests = []
    for seed in ("0", "723"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True
        )
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1] and len(digests[0]) == 64
