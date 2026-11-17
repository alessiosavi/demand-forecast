"""Composition, stream isolation and availability contracts of the typed generator."""

import hashlib
import json
import math
import os
import subprocess
import sys
import zlib
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.frequency import to_offset
from demand_forecast.synthetic.typed import (
    MECHANISMS,
    RANGES,
    SHARES,
    PanelRecipe,
    generate_typed_panel,
    sample_recipe,
)
from demand_forecast.synthetic.typed.draft import (
    Contribution,
    Draft,
    Result,
    assign_groups,
    grid,
    stream,
    token,
)
from demand_forecast.synthetic.typed.generate import REGISTRY


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


def recompute_base(g, item):
    s = g.scalars[item.item_id]
    grid_length = max(
        math.ceil(1.25 * max(targets(i).shape[1] for i in g.panel.items)),
        max(i.start + targets(i).shape[1] for i in g.panel.items),
    )
    grid = np.arange(item.start, item.start + targets(item).shape[1])
    period = g.panel.season if g.panel.season > 1 else g.panel.long_lag or 8
    return (
        1
        + s["tau"][:, None] * (grid - grid_length / 2) / grid_length
        + s["a1"][:, None] * np.sin(2 * np.pi * grid / period + s["f1"][:, None])
        + s["a2"][:, None] * np.sin(4 * np.pi * grid / period + s["f2"][:, None])
    )


def recompute_noise(g, item):
    xi = g.parts[item.item_id]["xi"]
    rho = g.scalars[item.item_id]["rho_effective"]
    noise = np.empty_like(xi)
    previous = np.zeros(len(rho))
    # The test deliberately evaluates the recursion directly, independently of lfilter.
    for t in range(xi.shape[1]):
        previous = rho * previous + np.sqrt(1 - rho**2) * xi[:, t]
        noise[:, t] = previous
    return noise


def assert_array_equal(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert np.array_equal(left, right, equal_nan=left.dtype != object)


def assert_inputs_equal(left, right, records=None):
    for a, b in zip(left.items, right.items, strict=True):
        assert (a.item_id, a.start) == (b.item_id, b.start)
        for record in records or ():
            if record.where == "static":
                assert a.static[record.name] == b.static[record.name]
            else:
                assert_array_equal(a.dynamic[record.name], b.dynamic[record.name])


def assert_generated_equal(left, right):
    assert left.inputs == right.inputs and left.params == right.params
    assert left.short_items == right.short_items and left.hosts == right.hosts
    assert left.availability == right.availability
    for a, b in ((left.panel, right.panel), (left.complete, right.complete)):
        assert (a.freq, a.epoch, a.aligned, a.roles) == (b.freq, b.epoch, b.aligned, b.roles)
        assert_inputs_equal(a, b, left.inputs)
        for ia, ib in zip(a.items, b.items, strict=True):
            assert_array_equal(targets(ia), targets(ib))
    for field in ("parts", "scalars"):
        for item_id, values in getattr(left, field).items():
            other = getattr(right, field)[item_id]
            assert values.keys() == other.keys()
            for name in values:
                assert_array_equal(values[name], other[name])


def fake_add(draft, streams, ranges):
    return Result((), {}, {}, Contribution(add=[np.full(n, 0.5) for n in draft.lengths]))


def fake_number(draft, streams, ranges):
    name = token(streams.inputs, "a")
    draft.add_static(
        name,
        streams.inputs.uniform(1, 100, draft.recipe.n_items).tolist(),
        "number",
        "static_number",
    )
    contribution = (
        None
        if streams.effect is None
        else Contribution(mult=[np.full(n, 2.0) for n in draft.lengths])
    )
    return Result((name,), {}, {}, contribution)


def fake_spread(draft, streams, ranges):
    return Result((), {}, {}, Contribution(mult=[np.full(n, 3.0) for n in draft.lengths]))


def fake_occ(draft, streams, ranges):
    return Result((), {}, {}, Contribution(occ=[np.full(n, 0.5) for n in draft.lengths]))


def fake_code(draft, streams, ranges):
    name = token(streams.inputs, "a")
    labels, singletons = assign_groups(draft, streams.process, streams.inputs)
    draft.add_static(name, labels, "category", "whole_number_code")
    contribution = (
        None
        if streams.effect is None
        else Contribution(occ=[np.full(n, 0.5) for n in draft.lengths])
    )
    return Result(
        (name,), {}, {"singletons": singletons} if streams.effect is not None else {}, contribution
    )


def fake_rho(draft, streams, ranges):
    return Result((), {}, {}, Contribution(rho=np.full(draft.recipe.n_items, 0.5)))


def fake_driver(draft, streams, ranges):
    rate = float(streams.process.uniform(0.1, 0.4))
    name = token(streams.inputs, "c")
    columns = [(streams.inputs.random(n) < rate).astype(np.float64) for n in draft.lengths]
    draft.add_dynamic(name, columns, "flag", "known", "known_driver")
    contribution = None if streams.effect is None else Contribution(add=[0.3 * x for x in columns])
    return Result((name,), {"rate": rate}, {}, contribution)


def fake_zero(draft, streams, ranges):
    return Result(
        (),
        {},
        {},
        Contribution(
            stockout=[np.arange(n) % 5 == 0 for n in draft.lengths],
            missing=[np.arange(n) % 7 == 0 for n in draft.lengths],
        ),
    )


FAKES = {
    "calendar": fake_add,
    "static_number": fake_number,
    "dynamic_category_spread": fake_spread,
    "intermittent": fake_occ,
    "whole_number_code": fake_code,
    "hierarchy": fake_rho,
    "known_driver": fake_driver,
    "missing_vs_zero": fake_zero,
}


def register_fakes(monkeypatch):
    for name, fn in FAKES.items():
        monkeypatch.setitem(REGISTRY, name, fn)


@pytest.mark.parametrize("freq", ["h", "D", "W-MON", "ME"])
def test_base_and_noise_without_mechanisms(freq):
    g = generate_typed_panel(PanelRecipe((), n_items=9, length=(40, 300), freq=freq, seed=3))
    g.panel.validate()
    g.complete.validate()
    for item in g.panel.items:
        p = g.parts[item.item_id]
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p["base"], recompute_base(g, item), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p["noise"], recompute_noise(g, item), rtol=1e-10, atol=1e-10)
        assert (
            p["occ"].all()
            and not p["stockout"].any()
            and not p["missing"].any()
            and (p["mult"] == 1).all()
        )
        for name in ("base", "xi", "noise", "u", "mult"):
            assert p[name].dtype == np.float64 and p[name].shape == targets(item).shape
        for name in ("occ", "stockout", "missing"):
            assert p[name].dtype == bool and p[name].shape == targets(item).shape


def test_merge_rules(monkeypatch):
    register_fakes(monkeypatch)
    g = generate_typed_panel(PanelRecipe(tuple(FAKES), n_items=12, length=(64, 200), seed=2))
    driver = g.inputs_of("known_driver")[0]
    for item, complete in zip(g.panel.items, g.complete.items, strict=True):
        p, s = g.parts[item.item_id], g.scalars[item.item_id]
        assert (p["add:calendar"] == 0.5).all() and (p["mult"] == 6).all()
        np.testing.assert_array_equal(p["add:known_driver"][0], 0.3 * complete.dynamic[driver.name])
        np.testing.assert_array_equal(p["occ"], (p["u"] < 0.25) & ~p["stockout"])
        assert (s["rho_effective"] == 0.5).all()
        np.testing.assert_allclose(p["noise"], recompute_noise(g, item), rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(np.isnan(targets(item)), p["missing"])
        assert p["missing"][0, 0] and p["stockout"][0, 0]
        assert (targets(item)[p["stockout"] & ~p["missing"]] == 0.0).all()
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)


def test_without_changes_only_that_mechanism(monkeypatch):
    register_fakes(monkeypatch)
    r = PanelRecipe(tuple(FAKES), n_items=12, length=(64, 200), seed=2)
    full = generate_typed_panel(r)
    for mechanism in FAKES:
        cut = generate_typed_panel(r, without=(mechanism,))
        assert cut.params == full.params and cut.inputs == full.inputs
        assert_inputs_equal(cut.panel, full.panel, full.inputs)
        assert_inputs_equal(cut.complete, full.complete, full.inputs)
        for item in cut.panel.items:
            p, f = cut.parts[item.item_id], full.parts[item.item_id]
            for name in ("base", "xi", "u"):
                assert_array_equal(p[name], f[name])
            assert f"add:{mechanism}" not in p
            for name in f:
                if name.startswith("add:") and name != f"add:{mechanism}":
                    assert_array_equal(p[name], f[name])
            multiplier = {"static_number": 3, "dynamic_category_spread": 2}.get(mechanism, 6)
            assert (p["mult"] == multiplier).all()
            probability = 0.5 if mechanism in ("intermittent", "whole_number_code") else 0.25
            stockout = (
                np.zeros_like(f["stockout"]) if mechanism == "missing_vs_zero" else f["stockout"]
            )
            missing = (
                np.zeros_like(f["missing"]) if mechanism == "missing_vs_zero" else f["missing"]
            )
            assert_array_equal(p["stockout"], stockout)
            assert_array_equal(p["missing"], missing)
            assert_array_equal(p["occ"], (p["u"] < probability) & ~stockout)
            if mechanism == "hierarchy":
                assert_array_equal(
                    cut.scalars[item.item_id]["rho_effective"], full.scalars[item.item_id]["rho"]
                )
            else:
                assert_array_equal(p["noise"], f["noise"])
            np.testing.assert_allclose(
                p["noise"], recompute_noise(cut, item), rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(targets(item), rebuild(cut, item), rtol=1e-12, atol=1e-12)


def test_without_refusals():
    with pytest.raises(ValueError):
        generate_typed_panel(PanelRecipe(()), without=("calendar",))
    with pytest.raises(ValueError):
        generate_typed_panel(PanelRecipe(("multi_target",)), without=("multi_target",))


@pytest.mark.parametrize("key", ["process", "inputs"])
@pytest.mark.parametrize("without", [(), ("calendar",)])
def test_params_reject_reserved_keys(monkeypatch, key, without):
    def fake(draft, streams, ranges):
        return Result((), {}, {key: "reserved"}, None)

    monkeypatch.setitem(REGISTRY, "calendar", fake)
    with pytest.raises(ValueError, match=rf"calendar.*{key}"):
        generate_typed_panel(PanelRecipe(("calendar",), n_items=2, length=(8, 12)), without=without)


def test_extra_arrays_survive_without(monkeypatch):
    def fake(draft, streams, ranges):
        result = fake_driver(draft, streams, ranges)
        if result.contribution is None:
            return result
        columns = [item[result.inputs[0]] for item in draft.dynamic]
        return replace(
            result,
            contribution=Contribution(
                add=result.contribution.add,
                extra={
                    "diagnostic": columns,
                    "diagnostic_matrix": [np.stack((column, -column)) for column in columns],
                },
            ),
        )

    monkeypatch.setitem(REGISTRY, "known_driver", fake)
    monkeypatch.setitem(REGISTRY, "calendar", fake_add)
    recipe = PanelRecipe(("calendar", "known_driver"), n_items=5, length=(40, 80), seed=3)
    full = generate_typed_panel(recipe)
    cut = generate_typed_panel(recipe, without=("known_driver",))
    baseline = generate_typed_panel(replace(recipe, mechanisms=("calendar",)))
    assert cut.params == full.params and cut.inputs == full.inputs
    assert_inputs_equal(cut.panel, full.panel, full.inputs)
    assert_inputs_equal(cut.complete, full.complete, full.inputs)
    name = full.inputs_of("known_driver")[0].name
    for item, removed, plain in zip(
        full.complete.items, cut.complete.items, baseline.complete.items, strict=True
    ):
        p, q = full.parts[item.item_id], cut.parts[item.item_id]
        column = item.dynamic[name]
        for parts in (p, q):
            assert_array_equal(parts["diagnostic"], column)
            assert_array_equal(parts["diagnostic_matrix"], np.stack((column, -column)))
        assert "add:known_driver" not in q
        assert_array_equal(targets(removed), targets(plain))
        np.testing.assert_allclose(targets(item), rebuild(full, item), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "name",
    [
        "base",
        "xi",
        "noise",
        "u",
        "occ",
        "mult",
        "stockout",
        "missing",
        "add:calendar",
        "add:absent",
        "add:",
    ],
)
@pytest.mark.parametrize("without", [(), ("calendar",)])
def test_extra_rejects_slot_names(monkeypatch, name, without):
    def fake(draft, streams, ranges):
        return Result((), {}, {}, Contribution(extra={name: [np.zeros(n) for n in draft.lengths]}))

    monkeypatch.setitem(REGISTRY, "calendar", fake)
    with pytest.raises(ValueError, match="collid"):
        generate_typed_panel(PanelRecipe(("calendar",), n_items=2, length=(8, 12)), without=without)


@pytest.mark.parametrize(
    "without", [(), ("calendar",), ("level_shift",), ("calendar", "level_shift")]
)
def test_extra_rejects_duplicate_mechanism_names(monkeypatch, without):
    def fake(draft, streams, ranges):
        return Result(
            (), {}, {}, Contribution(extra={"diagnostic": [np.zeros(n) for n in draft.lengths]})
        )

    for mechanism in ("calendar", "level_shift"):
        monkeypatch.setitem(REGISTRY, mechanism, fake)
    with pytest.raises(ValueError, match="collid"):
        generate_typed_panel(
            PanelRecipe(("level_shift", "calendar"), n_items=2, length=(8, 12)), without=without
        )


def test_nuisance_inputs(monkeypatch):
    register_fakes(monkeypatch)
    r = PanelRecipe(
        ("known_driver", "static_number", "calendar"), n_items=20, length=(64, 200), seed=9
    )
    full = generate_typed_panel(r)
    noisy = generate_typed_panel(replace(r, n_nuisance=3))
    assert {m: noisy.params[m] for m in r.mechanisms} == full.params
    assert noisy.inputs[: len(full.inputs)] == full.inputs
    for a, b in ((full.panel, noisy.panel), (full.complete, noisy.complete)):
        assert_inputs_equal(a, b, full.inputs)
        for ia, ib in zip(a.items, b.items, strict=True):
            assert_array_equal(targets(ia), targets(ib))
    for item_id, values in full.parts.items():
        assert values.keys() == noisy.parts[item_id].keys()
        for key, value in values.items():
            assert_array_equal(value, noisy.parts[item_id][key])
    twins = noisy.inputs_of("nuisance")
    assert len(twins) == 3 and len({r.name for r in noisy.inputs}) == len(noisy.inputs)
    for twin in twins:
        assert twin.twin in {"known_driver", "static_number"}
        assert twin.process == full.params[twin.twin]["process"]
        if twin.twin == "known_driver":
            assert twin.role == "known" and twin.kind == "flag"
            original = full.inputs_of("known_driver")[0]
            assert any(
                not np.array_equal(a.dynamic[original.name], b.dynamic[twin.name])
                for a, b in zip(full.complete.items, noisy.complete.items, strict=True)
            )
    static = generate_typed_panel(replace(r, n_nuisance=3, nuisance_from=("static_number",)))
    assert all(
        t.twin == "static_number" and t.kind == "number" for t in static.inputs_of("nuisance")
    )
    assert_generated_equal(
        noisy, generate_typed_panel(replace(r, mechanisms=r.mechanisms[::-1], n_nuisance=3))
    )
    empty = generate_typed_panel(PanelRecipe(("calendar",), n_items=2, n_nuisance=3))
    assert empty.inputs_of("nuisance") == ()


@pytest.mark.parametrize("n_nuisance", [0, 2])
def test_mechanism_inputs_keep_attribute_before_column(monkeypatch, n_nuisance):
    def fake(draft, streams, ranges):
        attribute, column = token(streams.inputs, "z"), token(streams.inputs, "a")
        draft.add_static(attribute, ["l1"] * draft.recipe.n_items, "category", "static_category")
        draft.add_dynamic(
            column,
            [np.arange(n, dtype=np.float64) for n in draft.lengths],
            "number",
            "past",
            "static_category",
        )
        return Result((attribute, column), {}, {}, None)

    monkeypatch.setitem(REGISTRY, "static_category", fake)
    g = generate_typed_panel(
        PanelRecipe(("static_category",), n_items=3, length=(64, 64), n_nuisance=n_nuisance)
    )
    attribute, column = g.inputs_of("static_category")
    assert attribute.where == "static" and column.where == "dynamic"
    assert g.params["static_category"]["inputs"] == (attribute.name, column.name)


def test_availability(monkeypatch):
    register_fakes(monkeypatch)
    r = PanelRecipe(("static_number", "known_driver"), n_items=20, length=(100, 200), seed=3)
    g = generate_typed_panel(r)
    all_inputs = generate_typed_panel(replace(r, complete_inputs=True))
    availability_rng = stream(r.seed, "availability")
    assert g.availability == {
        "static": float(availability_rng.uniform(0, 0.3)),
        "dynamic": float(availability_rng.uniform(0, 0.15)),
    }
    static, dynamic = g.inputs_of("static_number")[0], g.inputs_of("known_driver")[0]
    assert sum(i.static[static.name] is None for i in g.panel.items) == math.floor(
        g.availability["static"] * 20
    )
    assert all(i.static[static.name] is not None for i in g.complete.items)
    # Recover the chosen initial stretches from their named stream. Later runs can touch a stretch,
    # so the final contiguous missing prefix need not end at the chosen stretch's boundary.
    rng = stream(r.seed, f"availability:{dynamic.name}")
    selected = set(rng.choice(20, math.ceil(0.2 * 20), replace=False).tolist())
    assert len(selected) == 4
    for k, (item, complete, unmasked) in enumerate(
        zip(g.panel.items, g.complete.items, all_inputs.panel.items, strict=True)
    ):
        values = item.dynamic[dynamic.name]
        stretch = max(1, int(rng.uniform(0.1, 0.5) * len(values))) if k in selected else 0
        assert stretch <= len(values) // 2
        assert np.isnan(values[:stretch]).all()
        suffix = np.isnan(values[stretch:])
        assert (
            g.availability["dynamic"]
            <= suffix.mean()
            <= g.availability["dynamic"] + 5 / len(suffix)
        )
        # Consume the run draws to reach the next item's independent initial-stretch draw.
        mask = np.zeros(len(suffix), dtype=bool)
        while mask.mean() < g.availability["dynamic"]:
            start = int(rng.integers(len(mask)))
            size = int(rng.integers(1, 6))
            mask[start : start + size] = True
        assert np.isfinite(complete.dynamic[dynamic.name]).all()
        observed = ~np.isnan(values)
        assert_array_equal(values[observed], complete.dynamic[dynamic.name][observed])
        if item.static[static.name] is not None:
            assert item.static[static.name] == complete.static[static.name]
        assert_array_equal(complete.dynamic[dynamic.name], unmasked.dynamic[dynamic.name])
        assert_array_equal(targets(item), targets(unmasked))
        for name, values in g.parts[item.item_id].items():
            assert_array_equal(values, all_inputs.parts[item.item_id][name])
    assert_inputs_equal(all_inputs.panel, all_inputs.complete, all_inputs.inputs)


@pytest.mark.parametrize(
    ("freq", "period", "long_lag"),
    [("D", 7, 364), ("W-MON", 52, 52), ("ME", 12, 24), ("h", 24, 168), ("YE", 8, None)],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_generated_grid_metadata(monkeypatch, freq, period, long_lag, seed):
    register_fakes(monkeypatch)
    recipe = PanelRecipe(("whole_number_code",), n_items=3, length=(64, 64), freq=freq, seed=seed)
    g = generate_typed_panel(recipe)
    assert g.grid_length == max(
        math.ceil(1.25 * max(targets(item).shape[1] for item in g.panel.items)),
        max(item.start + targets(item).shape[1] for item in g.panel.items),
    )
    assert g.period == period
    assert g.long_lag == long_lag


@pytest.mark.parametrize("freq", ["D", "W-MON", "ME", "h", "YE"])
def test_draft_frequency_class(monkeypatch, freq):
    observed = []

    def capture(draft, streams, ranges):
        observed.append(draft.freq_class)
        return fake_add(draft, streams, ranges)

    monkeypatch.setitem(REGISTRY, "calendar", capture)
    g = generate_typed_panel(PanelRecipe(("calendar",), n_items=2, length=(8, 12), freq=freq))
    assert observed == [g.panel.freq_class]


def test_grid_hosts_and_short_items(monkeypatch):
    register_fakes(monkeypatch)
    plain = generate_typed_panel(PanelRecipe((), n_items=12, length=(64, 200)))
    assert plain.short_items == () and plain.hosts == {}
    for freq in ("h", "D", "W-MON", "ME"):
        for count, expected in ((1, 0), (2, 0), (3, 1), (12, 2), (64, 7)):
            recipe = PanelRecipe(("whole_number_code",), n_items=count, length=(64, 200), freq=freq)
            draft = grid(recipe)
            g = generate_typed_panel(recipe)
            assert len(g.short_items) == expected
            assert len({i.item_id for i in g.panel.items}) == count
            assert [i.item_id for i in g.panel.items] == [f"i{k:04d}" for k in range(count)]
            offset = to_offset(freq)
            assert offset.is_on_offset(g.panel.epoch)
            rolled = offset.rollforward(pd.Timestamp("2000-01-01"))
            assert g.panel.epoch in pd.date_range(rolled, periods=5000, freq=offset)
            initial_grid_length = math.ceil(1.25 * max(targets(i).shape[1] for i in g.panel.items))
            assert draft.grid_length == max(
                initial_grid_length,
                max(i.start + targets(i).shape[1] for i in g.panel.items),
            )
            by_id = {i.item_id: i for i in g.panel.items}
            for item in g.panel.items:
                length = targets(item).shape[1]
                assert 0 <= item.start and item.start + length <= draft.grid_length
                if item.item_id in g.short_items:
                    assert 8 <= length <= 63
                    host = by_id[g.hosts[item.item_id]]
                    assert host.item_id not in g.short_items
                    assert host.start + 16 <= item.start <= host.start + targets(host).shape[1] - 1
                else:
                    assert 64 <= length <= 200
                    assert item.start + length <= initial_grid_length
    g = generate_typed_panel(PanelRecipe(("whole_number_code",), n_items=12, length=(8, 20)))
    assert g.short_items == () and g.hosts == {}


def test_grid_clipping_preserves_host_history(monkeypatch):
    """The grid must grow rather than clip a hosted item's 16-step history."""
    register_fakes(monkeypatch)
    recipe = PanelRecipe(("whole_number_code",), n_items=3, length=(64, 64), seed=1)
    draft = grid(recipe)
    g = generate_typed_panel(recipe)
    by_id = {item.item_id: item for item in g.panel.items}
    for item_id, host_id in g.hosts.items():
        item, host = by_id[item_id], by_id[host_id]
        assert host.start + 16 <= item.start <= host.start + targets(host).shape[1] - 1
    assert draft.grid_length > math.ceil(1.25 * max(targets(i).shape[1] for i in g.panel.items))
    assert draft.grid_length == max(i.start + targets(i).shape[1] for i in g.panel.items)
    for item in g.panel.items:
        assert item.start + targets(item).shape[1] <= draft.grid_length
        np.testing.assert_allclose(g.parts[item.item_id]["base"], recompute_base(g, item))


def group_draft(n_items=40, singleton_share=0.25):
    short = np.arange(n_items) >= 36 if n_items == 40 else np.zeros(n_items, dtype=bool)
    host = np.full(n_items, -1, dtype=np.int64)
    host[short] = np.arange(short.sum())
    return Draft(
        recipe=PanelRecipe(("static_category",), n_items=n_items, singleton_share=singleton_share),
        freq="D",
        epoch=pd.Timestamp("2000-01-01"),
        period=7,
        long_lag=364,
        grid_length=160,
        starts=np.where(short, 16, 0).astype(np.int64),
        lengths=np.where(short, 8, 128).astype(np.int64),
        short=short,
        host=host,
        n_targets=1,
        static=[{} for _ in range(n_items)],
        dynamic=[{} for _ in range(n_items)],
        roles={},
        records=[],
        freq_class=3,
    )


def test_assign_groups():
    draft = group_draft()
    labels, singletons = assign_groups(draft, stream(4, "process"), stream(4, "inputs"))
    long_counts = Counter(
        label for label, short in zip(labels, draft.short, strict=True) if not short
    )
    assert len(singletons) == math.floor(0.25 * 36)
    assert all(long_counts[label] == 1 and labels.count(label) == 1 for label in singletons)
    assert set(labels) - set(singletons) <= set(long_counts)
    for i in np.flatnonzero(draft.short):
        if labels[draft.host[i]] not in singletons:
            assert labels[i] == labels[draft.host[i]]
    all_single, unique = assign_groups(
        group_draft(singleton_share=1), stream(4, "process"), stream(4, "inputs")
    )
    assert len(set(all_single)) == 40 and len(unique) == 40
    one, unique = assign_groups(
        group_draft(n_items=1, singleton_share=0), stream(4, "process"), stream(4, "inputs")
    )
    assert len(one) == 1 and len(set(one)) == 1
    fixed, unique = assign_groups(
        draft, stream(4, "process"), stream(4, "inputs"), n=5, singletons=False
    )
    assert len(set(fixed)) == 5 and unique == ()
    for n in (None, 5):
        a, sa = assign_groups(draft, stream(9, "process"), stream(1, "inputs"), n=n)
        b, sb = assign_groups(draft, stream(9, "process"), stream(2, "inputs"), n=n)
        assert len(set(a)) == len(set(b)) and len(sa) == len(sb)
        assert sorted(Counter(a[:36]).values()) == sorted(Counter(b[:36]).values())
        assert set(a).isdisjoint(b)


def test_recipe_validation():
    for kwargs in (
        {"mechanisms": ("unknown",)},
        {"mechanisms": ("calendar", "calendar")},
        {"mechanisms": ("dynamic_category_mean", "dynamic_category_spread")},
        {"mechanisms": ("seasonal_lag",), "freq": "YE"},
        {"n_items": 0},
        {"length": (4, 10)},
        {"length": (100, 50)},
        {"singleton_share": 1.5},
        {"mechanisms": ("calendar",), "nuisance_from": ("static_number",)},
    ):
        with pytest.raises(ValueError):
            PanelRecipe(**kwargs)


def test_unregistered_mechanism_names_itself(monkeypatch):
    monkeypatch.delitem(REGISTRY, "calendar", raising=False)
    with pytest.raises(KeyError, match="calendar"):
        generate_typed_panel(PanelRecipe(("calendar",)))


def test_deterministic_bit_for_bit(monkeypatch):
    register_fakes(monkeypatch)
    recipe = PanelRecipe(tuple(FAKES), n_items=12, length=(64, 200), n_nuisance=4, seed=12)
    assert_generated_equal(generate_typed_panel(recipe), generate_typed_panel(recipe))


def test_deterministic_across_hash_seeds():
    script = """
import hashlib, json
from dataclasses import asdict
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
g = generate_typed_panel(PanelRecipe((), n_items=12, seed=5))
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
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout.strip()
        for seed in (1, 2)
    ]
    assert len(results[0]) == hashlib.sha256().digest_size * 2 and results[0] == results[1]


def test_one_item_and_short_series():
    for n_items in (1, 2):
        g = generate_typed_panel(PanelRecipe((), n_items=n_items, length=(8, 12)))
        for item in g.panel.items:
            np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)


def test_ranges_table():
    expected = {
        "multi_target": {"lag": ((0, 3), (4, 6))},
        "calendar": {"amplitude": ((0.1, 0.4), (0.5, 0.8))},
        "seasonal_lag": {"amplitude": ((0.5, 1.5), (1.7, 2.3))},
        "intermittent": {"p": ((0.15, 0.7), (0.05, 0.12))},
        "level_shift": {"size": ((0.15, 0.4), (0.5, 0.8))},
        "missing_vs_zero": {"rebound": ((0.5, 1.5), (1.7, 2.3))},
        "static_category": {"amplitude": ((0.2, 0.6), (0.7, 1.0))},
        "multi_label": {"size": ((0.2, 0.5), (0.6, 0.9))},
        "hierarchy": {"amplitude": ((0.15, 0.4), (0.5, 0.8))},
        "static_number": {"noise": ((0.3, 0.7), (0.8, 1.1)), "trend": ((0.3, 0.8), (0.9, 1.3))},
        "whole_number_code": {"p": ((0.3, 0.9), (0.1, 0.25))},
        "dynamic_category_mean": {"effect": ((0.3, 0.6), (0.7, 1.0))},
        "dynamic_category_spread": {"effect": ((0.4, 1.0), (1.2, 1.6))},
        "dynamic_category_timing": {"pulse": ((0.5, 1.2), (1.4, 2.0))},
        "known_driver": {"beta": ((0.3, 0.8), (0.9, 1.3))},
        "planned_event": {"size": ((0.4, 1.2), (1.4, 2.0))},
        "delayed_past_driver": {"beta": ((0.4, 0.9), (1.0, 1.4)), "lag_periods": ((0, 1), (1, 2))},
        "event_response": {"size": ((0.3, 1.0), (1.2, 1.8))},
        "label_lag": {"beta": ((0.4, 0.9), (1.0, 1.4))},
    }
    assert RANGES == expected and set(RANGES) == set(MECHANISMS)
    for params in RANGES.values():
        for (lo, hi), (held_lo, held_hi) in params.values():
            assert hi <= held_lo or held_hi <= lo


def test_sample_recipe():
    recipes = [sample_recipe(seed) for seed in range(4000)]
    counts, frequencies = Counter(), Counter()
    for seed, recipe in enumerate(recipes):
        recipe.__post_init__()
        assert recipe == sample_recipe(seed) and recipe.seed == seed
        assert recipe.heldout is False and sample_recipe(seed, heldout=True).heldout is True
        assert 8 <= recipe.n_items <= 96 and 0 <= recipe.n_nuisance <= 3
        assert recipe.singleton_share == 0 or 0.1 <= recipe.singleton_share <= 0.5
        assert recipe.length == {"ME": (24, 240), "W-MON": (52, 1040)}.get(recipe.freq, (64, 2048))
        assert sum(m.startswith("dynamic_category_") for m in recipe.mechanisms) <= 1
        counts.update(
            "dynamic_category" if m.startswith("dynamic_category_") else m
            for m in recipe.mechanisms
        )
        frequencies[recipe.freq] += 1
    for mechanism, share in SHARES.items():
        measured = counts[mechanism] / len(recipes)
        assert abs(measured - 0.9 * share) <= 0.035, (mechanism, measured, 0.9 * share)
    empty_share = sum(not r.mechanisms for r in recipes) / len(recipes)
    assert 0.085 <= empty_share <= 0.120, empty_share
    for freq, share in {"h": 0.25, "D": 0.40, "W-MON": 0.20, "ME": 0.15}.items():
        assert abs(frequencies[freq] / len(recipes) - share) <= 0.035


def test_period_fallback_and_draft_helpers(monkeypatch):
    observed = []

    def capture(draft, streams, ranges):
        observed.append((draft.period, draft.long_lag))
        for i in range(draft.recipe.n_items):
            assert draft.grid(i).dtype == np.int64
            np.testing.assert_array_equal(
                draft.grid(i), np.arange(draft.starts[i], draft.starts[i] + draft.lengths[i])
            )
            assert len(draft.timestamps(i)) == draft.lengths[i]
        return fake_add(draft, streams, ranges)

    monkeypatch.setitem(REGISTRY, "calendar", capture)
    for freq in ("W-MON", "YE"):
        g = generate_typed_panel(PanelRecipe(("calendar",), n_items=2, length=(8, 12), freq=freq))
        for i in g.panel.items:
            np.testing.assert_allclose(g.parts[i.item_id]["base"], recompute_base(g, i))
    assert observed == [(52, 52), (8, None)]


def test_no_eligible_host_and_tiny_static_panels(monkeypatch):
    register_fakes(monkeypatch)
    for n_items in (1, 2, 3, 12):
        g = generate_typed_panel(
            PanelRecipe(("whole_number_code",), n_items=n_items, length=(9, 20))
        )
        assert g.hosts == {}
        for item in g.panel.items:
            np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)


def test_object_and_multilabel_availability(monkeypatch):
    def fake(draft, streams, ranges):
        attribute, column = token(streams.inputs, "a"), token(streams.inputs, "c")
        draft.add_static(
            attribute, [["l1", "l2"] for _ in draft.lengths], "multilabel", "multi_label"
        )
        draft.add_dynamic(
            column,
            [np.full(n, "l1", dtype=object) for n in draft.lengths],
            "category",
            "past",
            "multi_label",
        )
        return Result((attribute, column), {}, {}, None)

    monkeypatch.setitem(REGISTRY, "multi_label", fake)
    g = generate_typed_panel(PanelRecipe(("multi_label",), n_items=40, length=(64, 100), seed=4))
    attribute, column = g.inputs
    assert sum(not i.static[attribute.name] for i in g.panel.items) == math.floor(
        40 * g.availability["static"]
    )
    assert all(i.static[attribute.name] == ["l1", "l2"] for i in g.complete.items)
    assert any(None in i.dynamic[column.name] for i in g.panel.items)
    assert all(None not in i.dynamic[column.name] for i in g.complete.items)


def test_multitarget_shapes_masks_and_last_rho(monkeypatch):
    register_fakes(monkeypatch)

    def fake_multi(draft, streams, ranges):
        assert draft.n_targets == int(streams.process.integers(2, 21))
        assert ranges("lag") == ((4, 6) if draft.recipe.heldout else (0, 3))
        return Result(
            (),
            {},
            {},
            Contribution(
                add=[
                    np.arange(draft.n_targets * n, dtype=np.float64).reshape(draft.n_targets, n)
                    for n in draft.lengths
                ]
            ),
        )

    def later(draft, streams, ranges):
        return Result(
            (),
            {},
            {},
            Contribution(
                rho=np.full(draft.recipe.n_items, 0.2),
                stockout=[np.arange(n) % 3 == 0 for n in draft.lengths],
                missing=[np.arange(n) % 4 == 0 for n in draft.lengths],
            ),
        )

    monkeypatch.setitem(REGISTRY, "multi_target", fake_multi)
    monkeypatch.setitem(REGISTRY, "known_driver", later)
    for heldout in (False, True):
        g = generate_typed_panel(
            PanelRecipe(
                ("known_driver", "hierarchy", "missing_vs_zero", "multi_target", "calendar"),
                n_items=2,
                length=(8, 12),
                heldout=heldout,
            )
        )
        for item in g.panel.items:
            p = g.parts[item.item_id]
            assert 2 <= len(item.targets) <= 20
            assert list(item.targets) == [f"target_{j:02d}" for j in range(len(item.targets))]
            assert (g.scalars[item.item_id]["rho_effective"] == 0.2).all()
            local = np.arange(targets(item).shape[1])
            np.testing.assert_array_equal(p["stockout"][0], (local % 3 == 0) | (local % 5 == 0))
            np.testing.assert_array_equal(p["missing"][0], (local % 4 == 0) | (local % 7 == 0))
            np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)


def test_stream_contract_base_draw_order_and_global_rng():
    expected_rng = np.random.default_rng(np.random.SeedSequence([3, zlib.crc32(b"base")]))
    recipe = PanelRecipe((), n_items=3, length=(8, 12), seed=3)
    np.random.seed(105)
    expected_global = np.random.random(5)
    np.random.seed(105)
    g = generate_typed_panel(recipe)
    np.testing.assert_array_equal(np.random.random(5), expected_global)
    for item in g.panel.items:
        scalars = g.scalars[item.item_id]
        for name, low, high in (
            ("tau", -0.5, 0.5),
            ("a1", 0.05, 0.4),
            ("a2", 0, 0.2),
            ("f1", 0, 2 * np.pi),
            ("f2", 0, 2 * np.pi),
            ("sigma0", 0.05, 0.3),
            ("rho", 0, 0.8),
        ):
            np.testing.assert_array_equal(scalars[name], expected_rng.uniform(low, high, 1))
        np.testing.assert_array_equal(
            scalars["level"], np.exp(expected_rng.uniform(0, np.log(1000), 1))
        )
    assert np.array_equal(
        stream(5, "grid").random(10),
        np.random.default_rng(np.random.SeedSequence([5, zlib.crc32(b"grid")])).random(10),
    )


def test_parameters_and_records_are_plain_python(monkeypatch):
    register_fakes(monkeypatch)
    g = generate_typed_panel(PanelRecipe(tuple(FAKES), n_items=4, length=(8, 12), n_nuisance=3))
    # JSON round-tripping rejects arrays and NumPy scalar values in recorded parameters.
    json.dumps(g.params)
    json.dumps([asdict(record) for record in g.inputs])
