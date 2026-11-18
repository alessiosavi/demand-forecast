"""Equations, input processes and stream isolation of the five driver mechanisms."""

import hashlib
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from demand_forecast.data.episode import (
    KIND_FLAG,
    KIND_NUMBER,
    infer_kind,
    infer_static_kind,
)
from demand_forecast.synthetic.typed import RANGES, PanelRecipe, generate_typed_panel
from demand_forecast.synthetic.typed.draft import Streams, grid, stream, token
from demand_forecast.synthetic.typed.driver_effects import DRIVER

ALL = (
    "known_driver",
    "planned_event",
    "delayed_past_driver",
    "event_response",
    "label_lag",
)


def targets(item):
    return np.stack([item.targets[name] for name in sorted(item.targets)])


def standardise(values):
    deviation = values.std(ddof=0)
    return (values - values.mean()) / deviation if deviation else np.zeros_like(values)


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


def assert_inputs_equal(left, right, records):
    for a, b in zip(left.items, right.items, strict=True):
        assert (a.item_id, a.start) == (b.item_id, b.start)
        for record in records:
            if record.where == "static":
                assert a.static[record.name] == b.static[record.name]
            else:
                np.testing.assert_array_equal(a.dynamic[record.name], b.dynamic[record.name])


def assert_parts_equal(left, right):
    for field in ("parts", "scalars"):
        for item_id, values in getattr(left, field).items():
            other = getattr(right, field)[item_id]
            assert values.keys() == other.keys()
            for name, array in values.items():
                np.testing.assert_array_equal(array, other[name])


def assert_plain(value):
    assert type(value) in (float, int, str, tuple, dict)
    if isinstance(value, dict):
        for key, entry in value.items():
            assert_plain(key)
            assert_plain(entry)
    elif isinstance(value, tuple):
        for entry in value:
            assert_plain(entry)


@pytest.mark.parametrize("mechanism", ALL)
def test_each_driver_equation(mechanism):
    g = generate_typed_panel(
        PanelRecipe((mechanism,), n_items=12, length=(200, 600), complete_inputs=True, seed=9)
    )
    p = g.params[mechanism]
    assert_plain(p)
    assert p["inputs"] == tuple(r.name for r in g.inputs_of(mechanism))
    column = g.inputs_of(mechanism)[-1].name
    if mechanism == "label_lag":
        assert type(p["grid_driver"]) is tuple
        assert all(type(value) is float for value in p["grid_driver"])
        raw = np.asarray(p["grid_driver"], dtype=np.float64)
        assert len(raw) == grid(g.recipe).grid_length
        z_grid = standardise(raw)
    for item in g.complete.items:
        values = item.dynamic[column]
        assert values.dtype == np.float64
        if mechanism == "known_driver":
            expected = p["beta_sign"] * p["beta"] * np.tanh(standardise(values))
        elif mechanism == "planned_event":
            previous = np.r_[0.0, values[:-1]]
            expected = p["size_sign"] * p["size"] * (values + p["carry"] * previous * (1 - values))
        elif mechanism == "delayed_past_driver":
            period = grid(g.recipe).period
            assert p["lag"] == max(1, math.ceil(p["lag_periods"] * period))
            shifted = np.zeros_like(values)
            shifted[p["lag"] :] = standardise(values)[: -p["lag"]]
            expected = p["beta_sign"] * p["beta"] * shifted
        else:
            label = item.static[g.inputs_of(mechanism)[0].name]
            if mechanism == "event_response":
                expected = p["size_sign"][label] * p["size"][label] * values
            else:
                np.testing.assert_array_equal(values, raw[item.start : item.start + len(values)])
                indices = np.arange(item.start, item.start + len(values)) - p["lag"][label]
                shifted = np.zeros_like(values)
                shifted[indices >= item.start] = z_grid[indices[indices >= item.start]]
                expected = p["beta_sign"] * p["beta"] * shifted
        np.testing.assert_allclose(
            g.parts[item.item_id][f"add:{mechanism}"], expected[None, :], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)
    if mechanism in ("event_response", "label_lag"):
        for a in g.complete.items:
            for b in g.complete.items:
                lo, hi = (
                    max(a.start, b.start),
                    min(a.start + len(a.dynamic[column]), b.start + len(b.dynamic[column])),
                )
                if lo < hi:
                    np.testing.assert_array_equal(
                        a.dynamic[column][lo - a.start : hi - a.start],
                        b.dynamic[column][lo - b.start : hi - b.start],
                    )


def test_roles_and_inferred_kinds():
    g = generate_typed_panel(PanelRecipe(ALL, n_items=40, length=(200, 600), complete_inputs=True))
    for mechanism in ALL:
        records = g.inputs_of(mechanism)
        dynamic = records[-1]
        role = "past" if mechanism in ("delayed_past_driver", "label_lag") else "known"
        flag = mechanism in ("planned_event", "event_response")
        assert dynamic.where == "dynamic" and dynamic.role == g.panel.roles[dynamic.name] == role
        assert dynamic.kind == ("flag" if flag else "number")
        checked = 0
        for item in g.complete.items:
            values = item.dynamic[dynamic.name]
            if flag:
                assert set(values) <= {0.0, 1.0}
                if len(set(values)) < 2:
                    continue
            assert infer_kind(values) == (KIND_FLAG if flag else KIND_NUMBER)
            checked += 1
        assert checked > 0
        if len(records) == 2:
            attribute = records[0]
            assert attribute.where == "static" and attribute.role is None
            assert (
                attribute.kind
                == infer_static_kind([item.static[attribute.name] for item in g.complete.items])
                == "category"
            )


def test_planned_event_late_items():
    for length in ((200, 600), (10, 10), (8, 9)):
        g = generate_typed_panel(
            PanelRecipe(("planned_event",), n_items=20, length=length, complete_inputs=True, seed=9)
        )
        late = g.params["planned_event"]["late_items"]
        assert len(late) == math.ceil(0.1 * sum(len(targets(i)[0]) >= 10 for i in g.panel.items))
        column = g.inputs_of("planned_event")[0].name
        for item in g.complete.items:
            if item.item_id in late:
                flag = item.dynamic[column]
                boundary = 4 * len(flag) // 5
                assert not flag[:boundary].any() and flag[boundary:].any()
        if length == (8, 9):
            assert late == ()


@pytest.mark.parametrize("heldout", [False, True])
def test_heldout_lags_and_effects(heldout):
    for seed in range(5):
        g = generate_typed_panel(
            PanelRecipe(ALL, n_items=12, length=(200, 600), heldout=heldout, seed=seed)
        )
        period = grid(g.recipe).period
        for mechanism in ALL:
            p = g.params[mechanism]
            assert_plain(p)
            for key, bounds in RANGES[mechanism].items():
                lo, hi = bounds[int(heldout)]
                values = p[key].values() if isinstance(p[key], dict) else (p[key],)
                assert all(lo <= value <= hi for value in values)
                if f"{key}_sign" in p:
                    signs = p[f"{key}_sign"]
                    signs = signs.values() if isinstance(signs, dict) else (signs,)
                    assert set(signs) <= {-1, 1}
        lag = g.params["delayed_past_driver"]["lag"]
        assert (period < lag <= 2 * period) if heldout else (1 <= lag <= period)
        assert -0.5 <= g.params["planned_event"]["carry"] <= 0
        assert all(1 <= lag <= period for lag in g.params["label_lag"]["lag"].values())


def test_event_response_differs_by_label():
    g = generate_typed_panel(
        PanelRecipe(("event_response",), n_items=40, length=(200, 600), complete_inputs=True)
    )
    p = g.params["event_response"]
    signed_sizes = {label: p["size_sign"][label] * size for label, size in p["size"].items()}
    assert len(set(signed_sizes.values())) >= 2
    attribute, column = (r.name for r in g.inputs_of("event_response"))
    responses = {}
    for item in g.complete.items:
        label = item.static[attribute]
        flag = item.dynamic[column].astype(bool)
        if flag.any():
            response = g.parts[item.item_id]["add:event_response"][0, flag]
            np.testing.assert_allclose(response, signed_sizes[label], rtol=1e-12, atol=1e-12)
            if label in responses:
                assert responses[label] == response[0]
            responses[label] = response[0]
    assert len(responses) >= 2


@pytest.mark.parametrize("mechanism", ["planned_event", "event_response"])
def test_promised_events_exist(mechanism):
    for seed in range(3):
        g = generate_typed_panel(
            PanelRecipe(
                (mechanism,), n_items=20, length=(300, 600), complete_inputs=True, seed=seed
            )
        )
        column = g.inputs_of(mechanism)[-1].name
        assert any(item.dynamic[column].any() for item in g.complete.items)
        if mechanism == "planned_event":
            late = g.params[mechanism]["late_items"]
            for item in g.complete.items:
                if item.item_id in late:
                    assert item.dynamic[column].any(), (seed, item.item_id)
            # Late items get a forced run; the others depend on the event rate alone.
            assert any(
                item.dynamic[column].any() for item in g.complete.items if item.item_id not in late
            ), seed


@pytest.mark.parametrize("pool", [None, ("known_driver",), ("planned_event",), ("label_lag",)])
def test_nuisance_drivers(pool):
    recipe = PanelRecipe(
        ("known_driver", "planned_event", "label_lag"),
        n_items=20,
        length=(100, 300),
        nuisance_from=pool,
        seed=9,
    )
    full = generate_typed_panel(recipe)
    noisy = generate_typed_panel(replace(recipe, n_nuisance=4))
    assert full.params == noisy.params and noisy.inputs[: len(full.inputs)] == full.inputs
    for a, b in ((full.panel, noisy.panel), (full.complete, noisy.complete)):
        assert_inputs_equal(a, b, full.inputs)
        for ia, ib in zip(a.items, b.items, strict=True):
            np.testing.assert_array_equal(targets(ia), targets(ib))
    assert_parts_equal(full, noisy)
    twins = noisy.inputs_of("nuisance")
    assert len(twins) == 4 + sum(t.where == "static" for t in twins)
    assert len({r.name for r in noisy.inputs}) == len(noisy.inputs)
    for twin in twins:
        original = next(r for r in full.inputs_of(twin.twin) if r.where == twin.where)
        assert (twin.kind, twin.role) == (original.kind, original.role)
        assert twin.process == full.params[twin.twin]["process"]
        assert twin.name != original.name
        if twin.where == "dynamic":
            assert any(
                not np.array_equal(a.dynamic[original.name], b.dynamic[twin.name])
                for a, b in zip(full.complete.items, noisy.complete.items, strict=True)
            )
            if twin.kind == "flag":
                assert 0.02 <= twin.process["rate"] <= 0.08
        else:
            assert twin.twin == "label_lag"
            assert set(i.static[twin.name] for i in noisy.complete.items).isdisjoint(
                i.static[original.name] for i in full.complete.items
            )
    reordered = generate_typed_panel(
        replace(recipe, mechanisms=recipe.mechanisms[::-1], n_nuisance=4)
    )
    assert noisy.params == reordered.params and noisy.inputs == reordered.inputs
    assert_inputs_equal(noisy.panel, reordered.panel, noisy.inputs)
    assert_parts_equal(noisy, reordered)


def test_partial_availability_of_drivers():
    recipe = PanelRecipe(ALL, n_items=20, length=(100, 300), seed=3)
    g = generate_typed_panel(recipe)
    full = generate_typed_panel(replace(recipe, complete_inputs=True))
    assert_inputs_equal(g.complete, full.panel, g.inputs)
    assert g.params == full.params
    assert_parts_equal(g, full)
    for record in g.inputs:
        if record.where == "static":
            assert sum(i.static[record.name] is None for i in g.panel.items) == math.floor(
                g.availability["static"] * recipe.n_items
            )
            continue
        rng = stream(recipe.seed, f"availability:{record.name}")
        selected = set(rng.choice(recipe.n_items, math.ceil(0.2 * recipe.n_items), replace=False))
        for k, (item, complete) in enumerate(zip(g.panel.items, g.complete.items, strict=True)):
            values, original = item.dynamic[record.name], complete.dynamic[record.name]
            stretch = max(1, int(rng.uniform(0.1, 0.5) * len(values))) if k in selected else 0
            if stretch:
                assert 0.1 * len(values) - 1 < stretch <= 0.5 * len(values)
                assert np.isnan(values[:stretch]).all()
            mask = np.zeros(len(values) - stretch, dtype=bool)
            while mask.mean() < g.availability["dynamic"]:
                start, size = int(rng.integers(len(mask))), int(rng.integers(1, 6))
                mask[start : start + size] = True
            np.testing.assert_array_equal(
                np.isnan(values), np.r_[np.ones(stretch, dtype=bool), mask]
            )
            assert np.isfinite(original).all()
            np.testing.assert_array_equal(values[~np.isnan(values)], original[~np.isnan(values)])
    for a, b in zip(g.panel.items, full.panel.items, strict=True):
        np.testing.assert_array_equal(targets(a), targets(b))
    g.panel.validate()
    g.complete.validate()


def test_deterministic_across_hash_seeds():
    script = """
import hashlib, json
from dataclasses import asdict
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
names = ('known_driver', 'planned_event', 'delayed_past_driver', 'event_response', 'label_lag')
g = generate_typed_panel(PanelRecipe(names, n_items=12, length=(100, 300), n_nuisance=4, seed=5))
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
    digests = [
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout.strip()
        for seed in (1, 2)
    ]
    assert len(digests[0]) == 2 * hashlib.sha256().digest_size and digests[0] == digests[1]


@pytest.mark.parametrize("n_items", [1, 2, 3, 12])
@pytest.mark.parametrize("freq", ["h", "D", "W-MON", "ME", "YE"])
def test_small_panels_and_series_shorter_than_lags(n_items, freq):
    recipe = PanelRecipe(ALL, n_items=n_items, length=(8, 12), freq=freq, heldout=True, seed=2)
    g = generate_typed_panel(recipe)
    for item in g.complete.items:
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)
        lag = g.params["delayed_past_driver"]["lag"]
        assert not g.parts[item.item_id]["add:delayed_past_driver"][:, :lag].any()
        attribute = g.inputs_of("label_lag")[0].name
        lag = g.params["label_lag"]["lag"][item.static[attribute]]
        assert not g.parts[item.item_id]["add:label_lag"][:, :lag].any()


def test_without_preserves_other_inputs_and_contributions():
    recipe = PanelRecipe(ALL, n_items=12, length=(100, 300), n_nuisance=4, seed=7)
    g = generate_typed_panel(recipe)
    repeated = generate_typed_panel(recipe)
    assert g.inputs == repeated.inputs and g.params == repeated.params
    assert_inputs_equal(g.panel, repeated.panel, g.inputs)
    assert_parts_equal(g, repeated)
    for mechanism in ALL:
        cut = generate_typed_panel(recipe, without=(mechanism,))
        assert cut.params == g.params and cut.inputs == g.inputs
        assert_inputs_equal(cut.panel, g.panel, g.inputs)
        assert_inputs_equal(cut.complete, g.complete, g.inputs)
        for item in cut.panel.items:
            actual, original = cut.parts[item.item_id], g.parts[item.item_id]
            assert actual.keys() == original.keys() - {f"add:{mechanism}"}
            for key in actual:
                np.testing.assert_array_equal(actual[key], original[key])
            np.testing.assert_allclose(targets(item), rebuild(cut, item), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mechanism", ["event_response", "label_lag"])
@pytest.mark.parametrize("share", [0.25, 1.0])
def test_singletons_and_hosted_short_items(mechanism, share):
    g = generate_typed_panel(
        PanelRecipe(
            (mechanism,),
            n_items=40,
            length=(100, 300),
            singleton_share=share,
            complete_inputs=True,
        )
    )
    attribute = g.inputs_of(mechanism)[0].name
    counts = Counter(i.static[attribute] for i in g.complete.items)
    singletons = g.params[mechanism]["singletons"]
    assert all(counts[label] == 1 for label in singletons)
    by_id = {i.item_id: i for i in g.complete.items}
    n_long = g.recipe.n_items - len(g.short_items)
    assert len(singletons) == (g.recipe.n_items if share == 1 else math.floor(share * n_long))
    for item_id, host_id in g.hosts.items():
        item, host = by_id[item_id], by_id[host_id]
        assert host.start + 16 <= item.start < host.start + len(targets(host)[0])
        if host.static[attribute] not in singletons:
            assert item.static[attribute] == host.static[attribute]


@pytest.mark.parametrize("mechanism", ALL)
def test_nuisance_has_no_effect_and_same_process(mechanism):
    recipe = PanelRecipe((mechanism,), n_items=20, length=(100, 300), singleton_share=0.25)
    draft = grid(recipe)
    result = DRIVER[mechanism](
        draft,
        Streams(stream(0, f"{mechanism}:process"), stream(0, "twin"), None),
        lambda key: pytest.fail("a nuisance run read an effect range"),
    )
    full = generate_typed_panel(recipe)
    assert result.contribution is None and result.params == {}
    assert result.process == full.params[mechanism]["process"]
    assert len(result.inputs) == len(full.inputs_of(mechanism))


def test_ar_processes_and_known_step_changes():
    # Recover innovations from the inputs stream and evaluate the recursion directly.
    for mechanism, rho in (("known_driver", 0.9), ("delayed_past_driver", 0.7)):
        recipe = PanelRecipe((mechanism,), n_items=2, length=(8, 12), seed=9, complete_inputs=True)
        g = generate_typed_panel(recipe)
        rng = stream(recipe.seed, f"{mechanism}:inputs")
        name = token(rng, "c")
        steps = g.params[mechanism]["process"].get("steps", 0)
        for item in g.complete.items:
            values = item.dynamic[name]
            innovations = rng.standard_normal(len(values))
            expected = np.empty(len(values))
            previous = 0.0
            for t, innovation in enumerate(innovations):
                previous = rho * previous + math.sqrt(1 - rho**2) * innovation
                expected[t] = previous
            for _ in range(steps):
                point = int(rng.integers(len(values)))
                size = float(rng.uniform(0.5, 2))
                sign = int(rng.choice((-1, 1)))
                expected[point:] += sign * size
            np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)


def test_all_drivers_can_be_combined():
    mechanisms = ALL
    recipe = PanelRecipe(mechanisms, n_items=20, length=(100, 300), seed=3)
    full = generate_typed_panel(recipe)
    available = generate_typed_panel(replace(recipe, complete_inputs=True))
    heldout = generate_typed_panel(replace(recipe, heldout=True))
    noisy = generate_typed_panel(replace(recipe, n_nuisance=4))
    assert full.params == noisy.params
    assert_inputs_equal(full.panel, noisy.panel, full.inputs)
    assert_inputs_equal(full.complete, noisy.complete, full.inputs)
    assert_inputs_equal(full.complete, available.panel, full.inputs)
    assert_inputs_equal(full.panel, heldout.panel, full.inputs)
    assert_parts_equal(full, noisy)
    for twin in noisy.inputs_of("nuisance"):
        original = next(r for r in full.inputs_of(twin.twin) if r.where == twin.where)
        assert twin.process == full.params[twin.twin]["process"]
        assert (twin.kind, twin.role) == (original.kind, original.role)
    for a, b, c in zip(full.panel.items, available.panel.items, noisy.panel.items, strict=True):
        np.testing.assert_array_equal(targets(a), targets(b))
        np.testing.assert_array_equal(targets(a), targets(c))
        np.testing.assert_allclose(targets(a), rebuild(full, a), rtol=1e-12, atol=1e-12)
    for mechanism in mechanisms:
        cut = generate_typed_panel(recipe, without=(mechanism,))
        assert cut.params == full.params
        assert_inputs_equal(cut.panel, full.panel, full.inputs)
        assert_inputs_equal(cut.complete, full.complete, full.inputs)
        for item in cut.panel.items:
            actual, original = cut.parts[item.item_id], full.parts[item.item_id]
            assert actual.keys() == original.keys() - {f"add:{mechanism}"}
            for key in actual:
                np.testing.assert_array_equal(actual[key], original[key])
            np.testing.assert_allclose(targets(item), rebuild(cut, item), rtol=1e-12, atol=1e-12)
