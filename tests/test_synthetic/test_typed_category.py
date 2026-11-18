"""Equations, transition laws and stream isolation of time-varying categories."""

import hashlib
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pytest

from demand_forecast.data.episode import KIND_CATEGORY, infer_kind
from demand_forecast.synthetic.typed import RANGES, PanelRecipe, generate_typed_panel
from demand_forecast.synthetic.typed.category_effects import CATEGORY
from demand_forecast.synthetic.typed.draft import (
    Contribution,
    Result,
    Streams,
    grid,
    stream,
    token,
)
from demand_forecast.synthetic.typed.generate import REGISTRY

ALL = ("dynamic_category_mean", "dynamic_category_spread", "dynamic_category_timing")


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


def assert_parts_equal(a, b):
    for mapping in ("parts", "scalars"):
        left, right = getattr(a, mapping), getattr(b, mapping)
        assert left.keys() == right.keys()
        for item_id, arrays in left.items():
            assert arrays.keys() == right[item_id].keys()
            for name, array in arrays.items():
                np.testing.assert_array_equal(array, right[item_id][name])


def runs(column):
    starts = np.r_[0, np.flatnonzero(column[1:] != column[:-1]) + 1]
    return starts, column[starts], np.diff(np.r_[starts, len(column)])


def timing_equation(column, params):
    expected = np.zeros(len(column))
    changes = np.flatnonzero(column[1:] != column[:-1]) + 1
    for change in changes:
        start = change + params["delay"][column[change]]
        expected[start : start + 2] += params["pulse"]
    return expected


@pytest.mark.parametrize("mechanism", ALL)
def test_each_category_equation(mechanism):
    recipe = PanelRecipe((mechanism,), n_items=12, length=(200, 600), complete_inputs=True, seed=9)
    g = generate_typed_panel(recipe)
    cut = generate_typed_panel(recipe, without=(mechanism,))
    params = g.params[mechanism]
    name = g.inputs_of(mechanism)[0].name
    for item in g.complete.items:
        column, parts = item.dynamic[name], g.parts[item.item_id]
        if mechanism == "dynamic_category_spread":
            expected = [params["factor"][value] for value in column]
            np.testing.assert_allclose(parts["mult"][0], expected, rtol=1e-12, atol=1e-12)
            assert "add:dynamic_category_spread" not in parts
        else:
            expected = (
                [params["mean"][value] for value in column]
                if mechanism == "dynamic_category_mean"
                else timing_equation(column, params)
            )
            np.testing.assert_allclose(
                parts[f"add:{mechanism}"][0], expected, rtol=1e-12, atol=1e-12
            )
        np.testing.assert_array_equal(parts["noise"], cut.parts[item.item_id]["noise"])
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)


def test_timing_values_share_the_pulse_and_differ_in_delay():
    mechanism = "dynamic_category_timing"
    g = generate_typed_panel(
        PanelRecipe((mechanism,), n_items=40, length=(400, 800), complete_inputs=True, seed=9)
    )
    params = g.params[mechanism]
    period = g.panel.season if g.panel.season > 1 else g.panel.long_lag or 8
    assert isinstance(params["pulse"], float) and params["width"] == 2
    assert set(params["delay"]) == set(params["values"])
    assert len(set(params["delay"].values())) == params["process"]["n_values"]
    assert all(0 <= delay < period for delay in params["delay"].values())
    name = g.inputs_of(mechanism)[0].name
    interior, overlapping, clipped = 0, 0, 0
    for item in g.complete.items:
        column = item.dynamic[name]
        changes = np.flatnonzero(column[1:] != column[:-1]) + 1
        ends = np.array([c + params["delay"][column[c]] + 2 for c in changes])
        add = g.parts[item.item_id][f"add:{mechanism}"][0]
        if len(changes) and (ends <= len(column)).all():
            interior += 1
            assert add.sum() == pytest.approx(params["pulse"] * 2 * len(changes))
        overlapping += bool((add >= 2 * params["pulse"]).any())
        clipped += bool((ends > len(column)).any())
        np.testing.assert_allclose(add, timing_equation(column, params), rtol=1e-12, atol=1e-12)
    assert interior > 0 and overlapping > 0 and clipped > 0


@pytest.mark.parametrize("mechanism", ALL)
def test_chain_structure(mechanism):
    roles = set()
    for seed in range(20):
        recipe = PanelRecipe(
            (mechanism,), n_items=5, length=(100, 200), complete_inputs=True, seed=seed
        )
        g = generate_typed_panel(recipe)
        params = g.params[mechanism]
        process = params["process"]
        record = g.inputs_of(mechanism)[0]
        assert set(process) == {"n_values", "dwell", "role", "rare", "late"}
        assert 4 <= process["n_values"] <= 8
        assert len(set(params["values"])) == process["n_values"]
        assert 0 <= process["rare"] < process["n_values"]
        assert 0 <= process["late"] < process["n_values"]
        assert process["rare"] != process["late"]
        assert record.where == "dynamic" and record.kind == "category"
        assert g.panel.roles[record.name] == record.role == process["role"]
        roles.add(record.role)
        assert re.fullmatch(r"c[0-9a-f]{8}", record.name)
        assert all(re.fullmatch(r"l[0-9a-f]{8}", v) for v in params["values"])
        period = g.panel.season if g.panel.season > 1 else g.panel.long_lag or 8
        assert max(2, 0.5 * period) <= process["dwell"] <= max(2, 3 * period)
        if mechanism == "dynamic_category_timing":
            assert process["n_values"] <= period
        draft = grid(recipe)
        result = CATEGORY[mechanism](
            draft,
            Streams(
                stream(seed, f"{mechanism}:process"),
                stream(seed, f"{mechanism}:inputs"),
                stream(seed, f"{mechanism}:effect"),
            ),
            lambda key: RANGES[mechanism][key][0],
        )
        assert draft.roles[record.name] == record.role
        assert result.process == process
        for item in g.complete.items:
            column = item.dynamic[record.name]
            assert column.dtype == object and set(column) <= set(params["values"])
            if len(set(column)) >= 3:
                assert infer_kind(column) == KIND_CATEGORY
            if item.item_id not in params["late_items"]:
                _, values, _ = runs(column)
                assert (values[1:] != values[:-1]).all()
    assert roles == {"known", "past"}


@pytest.mark.parametrize("mechanism", ALL)
def test_transition_law(mechanism):
    for seed in range(3):
        g = generate_typed_panel(
            PanelRecipe(
                (mechanism,), n_items=40, length=(400, 800), complete_inputs=True, seed=seed
            )
        )
        params = g.params[mechanism]
        name = g.inputs_of(mechanism)[0].name
        lengths = []
        entries = dict.fromkeys(params["values"], 0)
        for item in g.complete.items:
            if item.item_id in params["late_items"]:
                continue
            _, values, sizes = runs(item.dynamic[name])
            lengths.extend(sizes[:-1])  # The final run is right-censored.
            assert (values[1:] != values[:-1]).all()
            for value in values[1:]:
                entries[value] += 1
        dwell = params["process"]["dwell"]
        assert abs(np.mean(lengths) / dwell - 1) <= 0.25, (seed, np.mean(lengths), dwell)
        rare = params["values"][params["process"]["rare"]]
        ratio = entries[rare] / np.mean([n for v, n in entries.items() if v != rare])
        assert ratio < 0.5, (seed, ratio)
        late = params["values"][params["process"]["late"]]
        assert all(n > 0 for v, n in entries.items() if v != late)


@pytest.mark.parametrize("mechanism", ALL)
def test_late_value(mechanism):
    g = generate_typed_panel(
        PanelRecipe((mechanism,), n_items=40, length=(400, 800), complete_inputs=True, seed=9)
    )
    params = g.params[mechanism]
    name = g.inputs_of(mechanism)[0].name
    late = params["values"][params["process"]["late"]]
    eligible = [i for i in g.complete.items if len(i.dynamic[name]) >= 20]
    assert len(set(params["late_items"])) == math.ceil(0.2 * len(eligible))
    suffix_runs, reentries = [], 0
    for item in g.complete.items:
        if item.item_id in params["late_items"]:
            column = item.dynamic[name]
            boundary = 4 * len(column) // 5
            assert not (column[:boundary] == late).any()
            assert column[boundary] == late
            _, values, _ = runs(column[boundary:])
            suffix_runs.append(len(values))
            reentries += np.count_nonzero(values[1:] == late)
    assert np.mean(suffix_runs) > 1 and reentries > 0
    exact = generate_typed_panel(
        PanelRecipe((mechanism,), n_items=1, length=(20, 20), complete_inputs=True, seed=9)
    )
    params = exact.params[mechanism]
    column = exact.complete.items[0].dynamic[exact.inputs_of(mechanism)[0].name]
    late = params["values"][params["process"]["late"]]
    assert params["late_items"] == ("i0000",)
    assert column[16] == late and not (column[:16] == late).any()
    small = generate_typed_panel(PanelRecipe((mechanism,), n_items=2, length=(8, 12)))
    assert small.params[mechanism]["late_items"] == ()


@pytest.mark.parametrize("mechanism", ALL)
def test_nuisance_category(mechanism):
    recipe = PanelRecipe((mechanism,), n_items=12, length=(100, 300), seed=9)
    original = generate_typed_panel(recipe)
    noisy = generate_typed_panel(replace(recipe, n_nuisance=2))
    assert noisy.params == original.params
    assert noisy.inputs[:1] == original.inputs
    assert_parts_equal(original, noisy)
    twins = noisy.inputs_of("nuisance")
    assert len(twins) == 2
    assert len({r.name for r in noisy.inputs}) == 3
    labels = set(original.params[mechanism]["values"])
    record = original.inputs_of(mechanism)[0]
    for twin in twins:
        assert twin.twin == mechanism and twin.role == record.role
        assert twin.process == original.params[mechanism]["process"]
        realised = set().union(*(set(i.dynamic[twin.name]) for i in noisy.complete.items))
        assert realised.isdisjoint(labels)
        labels.update(realised)
    for left, right in ((original.panel, noisy.panel), (original.complete, noisy.complete)):
        for a, b in zip(left.items, right.items, strict=True):
            np.testing.assert_array_equal(a.dynamic[record.name], b.dynamic[record.name])
            np.testing.assert_array_equal(targets(a), targets(b))

    def no_effect_range(key):
        pytest.fail(f"Nuisance run requested effect range {key}")

    result = CATEGORY[mechanism](
        grid(recipe),
        Streams(stream(9, f"{mechanism}:process"), stream(9, "nuisance-test"), None),
        no_effect_range,
    )
    assert result.params == {} and result.contribution is None
    assert result.process == original.params[mechanism]["process"]


@pytest.mark.parametrize("mechanism", ALL)
def test_partial_availability_of_a_category(mechanism):
    recipe = PanelRecipe((mechanism,), n_items=20, length=(100, 200), seed=9)
    g = generate_typed_panel(recipe)
    full = generate_typed_panel(replace(recipe, complete_inputs=True))
    g.panel.validate()
    g.complete.validate()
    assert g.params == full.params
    assert_parts_equal(g, full)
    name = g.inputs_of(mechanism)[0].name
    rng = stream(recipe.seed, f"availability:{name}")
    selected = set(rng.choice(20, math.ceil(0.2 * 20), replace=False).tolist())
    assert len(selected) == 4
    for i, (observed, complete, unmasked) in enumerate(
        zip(g.panel.items, g.complete.items, full.panel.items, strict=True)
    ):
        column = observed.dynamic[name]
        mask = column == None  # noqa: E711
        stretch = max(1, int(rng.uniform(0.1, 0.5) * len(column))) if i in selected else 0
        if i in selected:
            assert int(0.1 * len(column)) <= stretch <= int(0.5 * len(column))
            assert mask[:stretch].all()
        suffix = np.zeros(len(column) - stretch, dtype=bool)
        while suffix.mean() < g.availability["dynamic"]:
            start = int(rng.integers(len(suffix)))
            size = int(rng.integers(1, 6))
            suffix[start : start + size] = True
        np.testing.assert_array_equal(mask[stretch:], suffix)
        assert all(value is None for value in column[mask])
        np.testing.assert_array_equal(column[~mask], complete.dynamic[name][~mask])
        assert all(value is not None for value in complete.dynamic[name])
        np.testing.assert_array_equal(complete.dynamic[name], unmasked.dynamic[name])
        np.testing.assert_array_equal(targets(observed), targets(complete))
        np.testing.assert_array_equal(targets(observed), targets(unmasked))


def assert_plain(value):
    if type(value) is dict:
        for key, child in value.items():
            assert type(key) is str
            assert_plain(child)
    elif type(value) is tuple:
        for child in value:
            assert_plain(child)
    else:
        assert type(value) in (str, float, int, type(None))


@pytest.mark.parametrize("mechanism", ALL)
def test_recorded_parameters_lie_in_the_range_of_the_setting(mechanism):
    for seed in range(5):
        for heldout in (False, True):
            recipe = PanelRecipe(
                (mechanism,), n_items=5, length=(64, 100), heldout=heldout, seed=seed
            )
            g = generate_typed_panel(recipe)
            params = g.params[mechanism]
            assert_plain(params)
            for record in g.inputs:
                assert_plain(asdict(record))
            for key, ranges in RANGES[mechanism].items():
                low, high = ranges[int(heldout)]
                assert low <= params[key] <= high
            for key in params:
                if key.endswith("_sign"):
                    assert params[key] in (-1, 1)
            if mechanism == "dynamic_category_mean":
                assert set(params["mean"]) == set(params["values"])
                assert max(abs(v) for v in params["mean"].values()) == pytest.approx(
                    params["effect"]
                )
            elif mechanism == "dynamic_category_spread":
                assert set(params["factor"]) == set(params["values"])
                assert max(abs(np.log(v)) for v in params["factor"].values()) == pytest.approx(
                    params["effect"]
                )


@pytest.mark.parametrize("mechanism", ALL)
def test_deterministic_across_hash_seeds(mechanism):
    script = """
import hashlib, json, sys
from dataclasses import asdict
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
g = generate_typed_panel(PanelRecipe((sys.argv[1],), n_items=12, seed=5, n_nuisance=2))
h = hashlib.sha256()
for panel in (g.panel, g.complete):
    h.update(str((panel.freq, panel.epoch, panel.aligned, panel.roles)).encode())
    for item in panel.items:
        h.update(str((item.item_id, item.start, item.static)).encode())
        for arrays in (item.targets, item.dynamic):
            for name, values in arrays.items():
                h.update(name.encode())
                h.update(json.dumps(values.tolist()).encode() if values.dtype == object else values.tobytes())
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
            [sys.executable, "-c", script, mechanism],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout.strip()
        for seed in (1, 2)
    ]
    assert len(results[0]) == hashlib.sha256().digest_size * 2 and results[0] == results[1]


@pytest.mark.parametrize("mechanism", ALL)
@pytest.mark.parametrize("freq", ["h", "D", "W-MON", "ME", "YE"])
@pytest.mark.parametrize("n_items", [1, 2])
def test_small_panels_and_short_series(mechanism, freq, n_items):
    recipe = PanelRecipe((mechanism,), n_items=n_items, length=(8, 12), freq=freq, seed=3)
    g = generate_typed_panel(recipe)
    assert g.params[mechanism]["late_items"] == ()
    for item in g.panel.items:
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)
        assert np.isfinite(targets(item)).all()


@pytest.mark.parametrize("mechanism", ALL)
def test_without_changes_only_the_category_effect(mechanism):
    recipe = PanelRecipe((mechanism,), n_items=6, length=(64, 200), seed=11)
    full = generate_typed_panel(recipe)
    cut = generate_typed_panel(recipe, without=(mechanism,))
    assert full.params == cut.params and full.inputs == cut.inputs
    for left, right in ((full.panel, cut.panel), (full.complete, cut.complete)):
        for a, b in zip(left.items, right.items, strict=True):
            for name in a.dynamic:
                np.testing.assert_array_equal(a.dynamic[name], b.dynamic[name])
    for item in cut.panel.items:
        p, q = full.parts[item.item_id], cut.parts[item.item_id]
        for name in p:
            if name not in (f"add:{mechanism}", "mult"):
                np.testing.assert_array_equal(p[name], q[name])
        assert f"add:{mechanism}" not in q and (q["mult"] == 1).all()
        if mechanism != "dynamic_category_spread":
            level = full.scalars[item.item_id]["level"][:, None]
            np.testing.assert_allclose(
                targets(full.panel.items[int(item.item_id[1:])]) - targets(item),
                level * p["occ"] * p[f"add:{mechanism}"],
                rtol=1e-9,
                atol=1e-9,
            )
        np.testing.assert_allclose(targets(item), rebuild(cut, item), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mechanism", ALL)
def test_repeated_generation_and_heldout_keep_inputs(mechanism):
    recipe = PanelRecipe((mechanism,), n_items=5, length=(64, 100), seed=4)
    full = generate_typed_panel(recipe)
    again = generate_typed_panel(recipe)
    heldout = generate_typed_panel(replace(recipe, heldout=True))
    assert_parts_equal(full, again)
    assert full.params == again.params and full.inputs == again.inputs
    assert full.params[mechanism]["process"] == heldout.params[mechanism]["process"]
    for left, right, other in zip(
        full.panel.items, again.panel.items, heldout.panel.items, strict=True
    ):
        np.testing.assert_array_equal(targets(left), targets(right))
        for name in left.dynamic:
            np.testing.assert_array_equal(left.dynamic[name], right.dynamic[name])
            np.testing.assert_array_equal(left.dynamic[name], other.dynamic[name])


@pytest.mark.parametrize("mechanism", ALL)
def test_category_preserves_other_mechanisms(monkeypatch, mechanism):
    def fake_driver(draft, streams, ranges):
        name = token(streams.inputs, "c")
        column = [streams.inputs.standard_normal(n) for n in draft.lengths]
        draft.add_dynamic(name, column, "number", "known", "known_driver")
        return Result((name,), {}, {}, Contribution(add=column))

    monkeypatch.setitem(REGISTRY, "known_driver", fake_driver)
    recipe = PanelRecipe((mechanism, "known_driver"), n_items=5, length=(64, 100), seed=4)
    full = generate_typed_panel(recipe)
    cut = generate_typed_panel(recipe, without=(mechanism,))
    plain = generate_typed_panel(replace(recipe, mechanisms=("known_driver",)))
    reversed_recipe = generate_typed_panel(replace(recipe, mechanisms=recipe.mechanisms[::-1]))
    assert_parts_equal(full, reversed_recipe)
    name = full.inputs_of("known_driver")[0].name
    assert full.inputs_of("known_driver") == plain.inputs_of("known_driver")
    for a, b, c in zip(full.panel.items, cut.panel.items, plain.panel.items, strict=True):
        np.testing.assert_array_equal(a.dynamic[name], b.dynamic[name])
        np.testing.assert_array_equal(a.dynamic[name], c.dynamic[name])
        for key in ("base", "xi", "u", "noise", "add:known_driver"):
            np.testing.assert_array_equal(full.parts[a.item_id][key], plain.parts[a.item_id][key])
        np.testing.assert_array_equal(targets(b), targets(c))
