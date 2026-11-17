"""Equations, input structure and isolation of the five static mechanisms."""

import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest

from demand_forecast.data.episode import infer_static_kind
from demand_forecast.synthetic.typed import RANGES, PanelRecipe, generate_typed_panel
from demand_forecast.synthetic.typed.draft import (
    Contribution,
    Result,
    Streams,
    compose,
    grid,
    stream,
)
from demand_forecast.synthetic.typed.generate import REGISTRY
from demand_forecast.synthetic.typed.static_effects import STATIC

ALL = ("static_category", "multi_label", "hierarchy", "static_number", "whole_number_code")


def targets(item):
    return np.stack([item.targets[name] for name in sorted(item.targets)])


def rebuild(g, item):
    p, s = g.parts[item.item_id], g.scalars[item.item_id]
    latent = p["base"].copy()
    for name in sorted(p):
        if name.startswith("add:"):
            latent += p[name]
    latent += s["sigma0"][:, None] * p["mult"] * p["noise"]
    result = s["level"][:, None] * p["occ"] * latent
    result[p["missing"]] = np.nan
    return result


def recompute_noise(g, item):
    xi = g.parts[item.item_id]["xi"]
    rho = g.scalars[item.item_id]["rho_effective"]
    noise = np.empty_like(xi)
    previous = np.zeros(len(rho))
    for t in range(xi.shape[1]):
        previous = rho * previous + np.sqrt(1 - rho**2) * xi[:, t]
        noise[:, t] = previous
    return noise


def assert_array_equal(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert np.array_equal(left, right, equal_nan=left.dtype != object)


def assert_inputs_equal(left, right, records):
    for a, b in zip(left.items, right.items, strict=True):
        assert (a.item_id, a.start) == (b.item_id, b.start)
        for record in records:
            assert a.static[record.name] == b.static[record.name]


def assert_parts_equal(left, right):
    for field in ("parts", "scalars"):
        for item_id, values in getattr(left, field).items():
            other = getattr(right, field)[item_id]
            assert values.keys() == other.keys()
            for name, value in values.items():
                assert_array_equal(value, other[name])


def assert_identity(g):
    g.panel.validate()
    g.complete.validate()
    for item in g.panel.items:
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)


def test_each_static_equation():
    for mechanism in ALL:
        recipe = PanelRecipe(
            (mechanism,), n_items=40, length=(64, 300), complete_inputs=True, seed=5
        )
        g = generate_typed_panel(recipe)
        params = g.params[mechanism]
        period = g.panel.season if g.panel.season > 1 else g.panel.long_lag or 8
        grid_length = max(
            math.ceil(1.25 * max(targets(i).shape[1] for i in g.complete.items)),
            max(i.start + targets(i).shape[1] for i in g.complete.items),
        )
        if mechanism == "static_number":
            x = np.array([i.static[params["inputs"][0]] for i in g.complete.items])
            log_x = np.log(x)
            z = (log_x - log_x.mean()) / log_x.std()
        for k, item in enumerate(g.complete.items):
            p = g.parts[item.item_id]
            absolute = np.arange(item.start, item.start + targets(item).shape[1])
            value = item.static[params["inputs"][0]]
            if mechanism == "static_category":
                expected = params["amplitude"][value] * np.sin(
                    2 * np.pi * absolute / period + params["phase"][value]
                )
            elif mechanism == "multi_label":
                expected = sum(
                    params["size_sign"][label]
                    * params["size"][label]
                    * (absolute % period == params["position"][label])
                    for label in value
                )
            elif mechanism == "hierarchy":
                child = item.static[params["inputs"][1]]
                assert params["parent_of"][child] == value
                assert (g.scalars[item.item_id]["rho_effective"] == params["rho"][value]).all()
                expected = params["amplitude"][child] * np.sin(
                    4 * np.pi * absolute / period + params["phase"][child]
                )
                np.testing.assert_allclose(
                    p["noise"], recompute_noise(g, item), rtol=1e-12, atol=1e-12
                )
            elif mechanism == "static_number":
                np.testing.assert_allclose(
                    p["mult"], np.exp(-params["noise"] * z[k]), rtol=1e-12, atol=1e-12
                )
                expected = (
                    params["trend_sign"]
                    * params["trend"]
                    * z[k]
                    * ((absolute - grid_length / 2) / grid_length)
                )
            else:
                np.testing.assert_array_equal(
                    p["occ"], (p["u"] < params["p"][value]) & ~p["stockout"]
                )
                continue
            np.testing.assert_allclose(p[f"add:{mechanism}"][0], expected, rtol=1e-12, atol=1e-12)
        assert_identity(g)
        if mechanism == "hierarchy":
            cut = generate_typed_panel(recipe, without=("hierarchy",))
            for item in cut.complete.items:
                s, p = cut.scalars[item.item_id], cut.parts[item.item_id]
                assert_array_equal(s["rho_effective"], s["rho"])
                assert_array_equal(p["xi"], g.parts[item.item_id]["xi"])
                np.testing.assert_allclose(
                    p["noise"], recompute_noise(cut, item), rtol=1e-12, atol=1e-12
                )
            assert_identity(cut)


def test_inferred_kinds_match_the_records():
    g = generate_typed_panel(PanelRecipe(ALL, n_items=40, seed=5))
    expected = {
        "static_category": "category",
        "multi_label": "multilabel",
        "hierarchy": "category",
        "static_number": "number",
        "whole_number_code": "category",
    }
    for record in g.inputs:
        assert record.where == "static" and record.role is None
        assert record.kind == expected[record.mechanism]
        assert infer_static_kind([i.static[record.name] for i in g.complete.items]) == record.kind


@pytest.mark.parametrize("n_items", [1, 2, 3, 5, 8, 12, 96])
def test_small_and_large_panels_are_feasible(n_items):
    for seed in range(10):
        g = generate_typed_panel(PanelRecipe(ALL, n_items=n_items, length=(64, 200), seed=seed))
        assert_identity(g)
        parent, child = g.params["hierarchy"]["inputs"]
        mapping = g.params["hierarchy"]["parent_of"]
        observed = {}
        for item in g.complete.items:
            c, p = item.static[child], item.static[parent]
            assert c not in observed or observed[c] == p
            observed[c] = p
            labels = item.static[g.params["multi_label"]["inputs"][0]]
            assert 1 <= len(labels) <= 3 and labels == sorted(set(labels))
            code = item.static[g.params["whole_number_code"]["inputs"][0]]
            assert type(code) is float and code == math.floor(code)
        assert observed == mapping
        assert set(observed.values()) == set(g.params["hierarchy"]["rho"])
        codes = {i.static[g.params["whole_number_code"]["inputs"][0]] for i in g.complete.items}
        assert codes == set(range(len(codes)))


def check_heldout_sets(items, name, params):
    sets = {item.item_id: tuple(item.static[name]) for item in items}
    heldout = params["heldout_items"]
    assert len(heldout) <= math.ceil(len(items) / 5)
    outside = {
        label for item_id, labels in sets.items() if item_id not in heldout for label in labels
    }
    counts = Counter(sets.values())
    for item_id in heldout:
        assert counts[sets[item_id]] == 1
        assert sets[item_id] not in params["pool"]
        assert set(sets[item_id]) <= outside
    assert all(
        labels in params["pool"] for item_id, labels in sets.items() if item_id not in heldout
    )


def test_multi_label_structure():
    successful = 0
    for seed in range(10):
        g = generate_typed_panel(
            PanelRecipe(("multi_label",), n_items=40, length=(64, 200), seed=seed)
        )
        params = g.params["multi_label"]
        a, b = params["opposed"]
        assert a != b and params["position"][a] == params["position"][b]
        assert params["size_sign"][a] == -params["size_sign"][b]
        check_heldout_sets(g.complete.items, params["inputs"][0], params)
        successful += bool(params["heldout_items"])
    assert successful >= 8, successful

    # The review's six-item/vocabulary-four case must never duplicate a reserve set.
    class VocabularyFour:
        def integers(self, low, high):
            assert (low, high) == (4, 11)
            return 4

    for seed in range(100):
        draft = grid(PanelRecipe(("multi_label",), n_items=6, length=(8, 12)))
        result = STATIC["multi_label"](
            draft,
            Streams(VocabularyFour(), stream(seed, "inputs"), stream(seed, "effect")),
            lambda key: RANGES["multi_label"][key][0],
        )
        panel, _, _ = compose(draft, [("multi_label", result.contribution)])
        check_heldout_sets(panel.items, result.inputs[0], result.params)
    one = generate_typed_panel(PanelRecipe(("multi_label",), n_items=1))
    assert one.params["multi_label"]["heldout_items"] == ()


def test_short_items_share_their_label_with_their_host():
    for mechanism in ("static_category", "hierarchy", "whole_number_code", "multi_label"):
        for seed in range(5):
            g = generate_typed_panel(
                PanelRecipe((mechanism,), n_items=40, length=(64, 200), seed=seed)
            )
            name = g.params[mechanism]["inputs"][int(mechanism == "hierarchy")]
            items = {i.item_id: i for i in g.complete.items}
            long_counts = Counter(
                tuple(i.static[name]) if mechanism == "multi_label" else i.static[name]
                for i in g.complete.items
                if i.item_id not in g.short_items
            )
            shared = 0
            assert len(g.short_items) == 4
            for item_id in g.short_items:
                item, host = items[item_id], items[g.hosts[item_id]]
                assert 8 <= targets(item).shape[1] <= 63
                assert host.start + 16 <= item.start < host.start + targets(host).shape[1]
                host_value = (
                    tuple(host.static[name]) if mechanism == "multi_label" else host.static[name]
                )
                if mechanism == "multi_label" or long_counts[host_value] > 1:
                    assert item.static[name] == host.static[name]
                    shared += 1
            assert shared >= 1


def test_singletons():
    g = generate_typed_panel(
        PanelRecipe(("static_category",), n_items=40, singleton_share=0.25, seed=5)
    )
    params = g.params["static_category"]
    counts = Counter(i.static[params["inputs"][0]] for i in g.complete.items)
    assert len(params["singletons"]) == math.floor(0.25 * (40 - len(g.short_items)))
    assert all(counts[label] == 1 for label in params["singletons"])


def test_labels_and_names_are_never_shared_between_seeds():
    names, labels = [], []
    for seed in (1, 2):
        g = generate_typed_panel(PanelRecipe(ALL, n_items=40, seed=seed))
        names.append({record.name for record in g.inputs})
        # Opaque string labels are episode-local; numeric codes deliberately use 0 .. K-1.
        vocabulary = set()
        for item in g.complete.items:
            for value in item.static.values():
                if isinstance(value, str):
                    vocabulary.add(value)
                elif isinstance(value, list):
                    vocabulary.update(value)
        labels.append(vocabulary)
    assert names[0].isdisjoint(names[1]) and labels[0].isdisjoint(labels[1])


def test_nuisance_static_inputs(monkeypatch):
    # G3 builds on G1 only; calendar is G2's independent task.
    def calendar(draft, streams, ranges):
        return Result((), {}, {}, Contribution(add=[np.full(n, 0.5) for n in draft.lengths]))

    monkeypatch.setitem(REGISTRY, "calendar", calendar)
    recipe = PanelRecipe(
        ("static_category", "static_number", "calendar"), n_items=40, length=(64, 200), seed=9
    )
    full = generate_typed_panel(recipe)
    noisy = generate_typed_panel(replace(recipe, n_nuisance=3))
    assert full.params == noisy.params and full.inputs == noisy.inputs[: len(full.inputs)]
    assert_parts_equal(full, noisy)
    for a, b in ((full.panel, noisy.panel), (full.complete, noisy.complete)):
        assert_inputs_equal(a, b, full.inputs)
        for ia, ib in zip(a.items, b.items, strict=True):
            assert_array_equal(targets(ia), targets(ib))
    twins = noisy.inputs_of("nuisance")
    assert len(twins) == 3
    for twin in twins:
        assert twin.twin in {"static_category", "static_number"}
        assert twin.process == full.params[twin.twin]["process"]
        original = full.inputs_of(twin.twin)[0]
        assert twin.kind == original.kind
        if twin.twin == "static_category":
            a = {i.static[original.name] for i in full.complete.items}
            b = {i.static[twin.name] for i in noisy.complete.items}
            assert len(a) == len(b) and a.isdisjoint(b)


def test_availability_never_changes_targets():
    recipe = PanelRecipe(ALL, n_items=40, length=(64, 200), seed=3)
    g = generate_typed_panel(recipe)
    complete = generate_typed_panel(replace(recipe, complete_inputs=True))
    assert_parts_equal(g, complete)
    assert_inputs_equal(g.complete, complete.panel, g.inputs)
    missing = math.floor(g.availability["static"] * recipe.n_items)
    for record in g.inputs:
        assert (
            sum(
                i.static[record.name] == []
                if record.kind == "multilabel"
                else i.static[record.name] is None
                for i in g.panel.items
            )
            == missing
        )
        assert all(
            i.static[record.name] is not None and i.static[record.name] != []
            for i in g.complete.items
        )
    for a, b, c in zip(g.panel.items, g.complete.items, complete.panel.items, strict=True):
        assert_array_equal(targets(a), targets(b))
        assert_array_equal(targets(a), targets(c))


def leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from leaves(child)
    elif isinstance(value, tuple):
        for child in value:
            yield from leaves(child)
    else:
        yield value


def test_recorded_parameters_lie_in_the_range_of_the_setting():
    for heldout in (False, True):
        for seed in range(5):
            g = generate_typed_panel(
                PanelRecipe(ALL, n_items=40, length=(64, 200), heldout=heldout, seed=seed)
            )
            json.dumps(g.params)
            json.dumps([asdict(record) for record in g.inputs])
            for mechanism in ALL:
                params = g.params[mechanism]
                for key, ranges in RANGES[mechanism].items():
                    lo, hi = ranges[int(heldout)]
                    values = list(leaves(params[key]))
                    assert values and all(type(v) in (int, float) and lo <= v <= hi for v in values)
                for key, value in params.items():
                    if key.endswith("_sign"):
                        assert all(type(v) is int and v in (-1, 1) for v in leaves(value))


def test_deterministic_across_hash_seeds():
    script = """
import hashlib, json
from dataclasses import asdict
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
g = generate_typed_panel(PanelRecipe(("static_category", "multi_label", "hierarchy", "static_number", "whole_number_code"), n_items=12, seed=5, n_nuisance=3))
h = hashlib.sha256()
for panel in (g.panel, g.complete):
    h.update(str((panel.freq, panel.epoch, panel.aligned, panel.roles)).encode())
    for item in panel.items:
        h.update(str((item.item_id, item.start, item.static)).encode())
        for arrays in (item.targets, item.dynamic):
            for name, values in arrays.items():
                h.update(name.encode()); h.update(values.tobytes())
for mapping in (g.parts, g.scalars):
    for values in mapping.values():
        for name, array in values.items():
            h.update(name.encode()); h.update(array.tobytes())
h.update(json.dumps(g.params, sort_keys=True).encode())
h.update(json.dumps([asdict(r) for r in g.inputs], sort_keys=True).encode())
print(h.hexdigest())
"""
    results = [
        subprocess.run(
            ["/private/tmp/v5-venv/bin/python", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout.strip()
        for seed in (1, 2)
    ]
    assert len(results[0]) == hashlib.sha256().digest_size * 2 and results[0] == results[1]


def test_without_keeps_other_mechanisms_and_missing_cells():
    recipe = PanelRecipe(ALL, n_items=12, length=(64, 200), seed=7)
    full = generate_typed_panel(recipe)
    for mechanism in ALL:
        cut = generate_typed_panel(recipe, without=(mechanism,))
        assert full.params == cut.params and full.inputs == cut.inputs
        assert_inputs_equal(full.panel, cut.panel, full.inputs)
        assert_inputs_equal(full.complete, cut.complete, full.inputs)
        for item in cut.panel.items:
            p, f = cut.parts[item.item_id], full.parts[item.item_id]
            for name in ("base", "xi", "u", "stockout", "missing"):
                assert_array_equal(p[name], f[name])
            for name in f:
                if name.startswith("add:") and name != f"add:{mechanism}":
                    assert_array_equal(p[name], f[name])
            assert f"add:{mechanism}" not in p
            if mechanism != "hierarchy":
                assert_array_equal(p["noise"], f["noise"])
            if mechanism != "static_number":
                assert_array_equal(p["mult"], f["mult"])
            else:
                assert (p["mult"] == 1).all()
            if mechanism != "whole_number_code":
                assert_array_equal(p["occ"], f["occ"])
            else:
                assert p["occ"].all()
        assert_identity(cut)


@pytest.mark.parametrize("freq", ["h", "D", "W-MON", "ME", "YE"])
@pytest.mark.parametrize(
    "n_items,length", [(1, (8, 12)), (2, (8, 12)), (3, (9, 20)), (12, (9, 20))]
)
def test_no_eligible_host_short_periods_and_multiple_targets(freq, n_items, length):
    recipe = PanelRecipe(ALL, n_items=n_items, length=length, freq=freq, singleton_share=1, seed=4)
    g = generate_typed_panel(recipe)
    assert g.hosts == {}
    assert_identity(g)
    if n_items == 1:
        item = g.complete.items[0]
        assert (g.parts[item.item_id]["mult"] == 1).all()
        assert (g.parts[item.item_id]["add:static_number"] == 0).all()
    draft = grid(recipe)
    draft.n_targets = 3
    contributions = []
    for mechanism in ALL:
        result = STATIC[mechanism](
            draft,
            Streams(
                *(
                    stream(recipe.seed, f"{mechanism}:{suffix}")
                    for suffix in ("process", "inputs", "effect")
                )
            ),
            lambda key: RANGES[mechanism][key][0],
        )
        contributions.append((mechanism, result.contribution))
    panel, parts, scalars = compose(draft, contributions)
    generated = replace(g, panel=panel, complete=panel, parts=parts, scalars=scalars)
    assert_identity(generated)
    for item in panel.items:
        assert targets(item).shape == (3, len(next(iter(item.targets.values()))))


def test_nuisance_all_static_processes_and_recipe_order():
    for mechanism in ALL:
        recipe = PanelRecipe((mechanism,), n_items=40, length=(64, 200), n_nuisance=2, seed=6)
        g = generate_typed_panel(recipe)
        original = g.inputs_of(mechanism)
        for twin in g.inputs_of("nuisance"):
            assert twin.process == g.params[mechanism]["process"]
            assert twin.kind in {r.kind for r in original}
        for item_id in g.parts:
            assert "add:nuisance" not in g.parts[item_id]
    recipe = PanelRecipe(ALL, n_items=12, length=(64, 200), n_nuisance=3, seed=6)
    a = generate_typed_panel(recipe)
    b = generate_typed_panel(replace(recipe, mechanisms=ALL[::-1]))
    assert a.inputs == b.inputs and a.params == b.params
    assert_inputs_equal(a.panel, b.panel, a.inputs)
    assert_inputs_equal(a.complete, b.complete, a.inputs)
    assert_parts_equal(a, b)
