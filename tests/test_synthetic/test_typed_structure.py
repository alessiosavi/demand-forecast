"""Equations, removal controls and edge cases of mechanisms without inputs."""

import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from demand_forecast.data.episode import FREQ_CLASSES
from demand_forecast.synthetic.typed import RANGES, PanelRecipe, generate_typed_panel
from demand_forecast.synthetic.typed.draft import Streams, grid, stream
from demand_forecast.synthetic.typed.structure import STRUCTURE

ALL = (
    "multi_target",
    "calendar",
    "seasonal_lag",
    "intermittent",
    "level_shift",
    "missing_vs_zero",
)


def test_calendar_uses_draft_frequency_class():
    draft = grid(PanelRecipe(("calendar",), n_items=2, length=(32, 32), freq="D"))
    draft = replace(draft, freq_class=FREQ_CLASSES.index("hourly"))
    streams = Streams(stream(0, "process"), stream(0, "inputs"), stream(0, "effect"))
    result = STRUCTURE["calendar"](draft, streams, lambda _: (0.1, 0.2))
    assert result.params["fields"] == ("hour", "dow")


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
    absolute = np.arange(item.start, item.start + targets(item).shape[1])
    period = g.panel.season if g.panel.season > 1 else g.panel.long_lag or 8
    return (
        1
        + s["tau"][:, None] * (absolute - grid_length / 2) / grid_length
        + s["a1"][:, None] * np.sin(2 * np.pi * absolute / period + s["f1"][:, None])
        + s["a2"][:, None] * np.sin(4 * np.pi * absolute / period + s["f2"][:, None])
    )


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


@pytest.mark.parametrize("freq", ["h", "D", "W-MON", "ME"])
def test_identity_holds_for_every_item_and_target(freq):
    g = generate_typed_panel(PanelRecipe(ALL, n_items=9, length=(40, 300), freq=freq, seed=3))
    assert tuple(STRUCTURE) == ALL
    assert g.inputs == () and g.panel.roles == {}
    g.panel.validate()
    g.complete.validate()
    for item, complete in zip(g.panel.items, g.complete.items, strict=True):
        p = g.parts[item.item_id]
        assert item.static == item.dynamic == {}
        assert targets(item).dtype == np.float64
        np.testing.assert_allclose(
            targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(p["base"], recompute_base(g, item), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p["noise"], recompute_noise(g, item), rtol=1e-10, atol=1e-10)
        assert np.isnan(targets(item)[p["missing"]]).all()
        assert np.isfinite(targets(item)[~p["missing"]]).all()
        assert (targets(item)[p["stockout"] & ~p["missing"]] == 0.0).all()
        assert_array_equal(targets(item), targets(complete))


@pytest.mark.parametrize("freq", ["h", "D", "W-MON", "ME"])
def test_calendar_equation(freq):
    g = generate_typed_panel(
        PanelRecipe(("calendar",), n_items=9, length=(40, 300), freq=freq, seed=3)
    )
    q = g.params["calendar"]
    assert q["fields"] == {"h": ("hour", "dow"), "D": ("dow", "month")}.get(freq, ("month",))
    sizes = {"hour": 24, "dow": 7, "month": 12}
    for field in q["fields"]:
        table = np.array(q["tables"][field])
        assert len(table) == sizes[field]
        assert abs(table.mean()) <= 1e-12
        assert np.max(np.abs(table)) == pytest.approx(q["amplitude"][field], abs=1e-12)
    for item in g.complete.items:
        dates = g.complete.timestamps(item.start, targets(item).shape[1])
        values = {"hour": dates.hour, "dow": dates.day_of_week, "month": dates.month - 1}
        expected = sum(np.array(q["tables"][field])[values[field]] for field in q["fields"])
        np.testing.assert_allclose(g.parts[item.item_id]["add:calendar"][0], expected)


def test_each_structure_equation():
    for mechanism in ALL:
        g = generate_typed_panel(PanelRecipe((mechanism,), n_items=5, length=(64, 300), seed=4))
        q = g.params[mechanism]
        if mechanism == "seasonal_lag":
            lag = g.panel.long_lag
            profile = np.array(q["profile"])
            assert len(profile) == lag and profile.max() == 1
            assert 1 <= len(q["centres"]) <= 3
            assert all(0 <= centre < lag for centre in q["centres"])
            assert q["width"] == max(1, lag / 20)
            distances = np.abs(np.arange(lag)[:, None] - np.array(q["centres"])[None, :])
            distances = np.minimum(distances, lag - distances)
            expected_profile = np.exp(-0.5 * (distances / q["width"]) ** 2).max(axis=1)
            np.testing.assert_allclose(profile, expected_profile, rtol=1e-12, atol=1e-12)
        for item in g.panel.items:
            p = g.parts[item.item_id]
            length = targets(item).shape[1]
            absolute = np.arange(item.start, item.start + length)
            if mechanism == "seasonal_lag":
                np.testing.assert_allclose(
                    p["add:seasonal_lag"][0], q["amplitude"] * profile[absolute % lag]
                )
            elif mechanism == "level_shift":
                points = q["point"][item.item_id]
                sizes = q["size"][item.item_id]
                signs = q["size_sign"][item.item_id]
                assert 1 <= len(points) <= 2 and len(points) == len(sizes) == len(signs)
                assert all(0.1 * length <= point - item.start <= 0.9 * length for point in points)
                expected = sum(
                    sign * size * (absolute >= point)
                    for point, size, sign in zip(points, sizes, signs, strict=True)
                )
                np.testing.assert_allclose(p["add:level_shift"][0], expected)
            elif mechanism == "missing_vs_zero":
                previous = np.concatenate(([False], p["stockout"][0, :-1]))
                np.testing.assert_allclose(
                    p["add:missing_vs_zero"][0], q["rebound"] * (previous & ~p["stockout"][0])
                )
            elif mechanism == "intermittent":
                assert_array_equal(p["occ"], (p["u"] < q["p"][item.item_id]) & ~p["stockout"])
            elif mechanism == "multi_target":
                n_targets = targets(item).shape[0]
                assert 2 <= n_targets <= 20
                assert q["process"]["n_targets"] == n_targets
                assert len(q["loading"]) == len(q["loading_sign"]) == len(q["lag"]) == n_targets
                assert all(0.3 <= loading <= 1.0 for loading in q["loading"])
                assert p["factor_e"].shape == p["factor"].shape == (length,)
                factor = np.empty(length)
                previous = 0.0
                for t, innovation in enumerate(p["factor_e"]):
                    previous = 0.95 * previous + np.sqrt(1 - 0.95**2) * innovation
                    factor[t] = previous
                np.testing.assert_allclose(p["factor"], factor, rtol=1e-12, atol=1e-12)
                for j, (loading, sign, delay) in enumerate(
                    zip(q["loading"], q["loading_sign"], q["lag"], strict=True)
                ):
                    expected = np.zeros(length)
                    expected[delay:] = sign * loading * factor[: length - delay]
                    np.testing.assert_allclose(p["add:multi_target"][j], expected)


def test_promised_structure_exists():
    for seed in range(3):
        g = generate_typed_panel(
            PanelRecipe(
                ("missing_vs_zero", "intermittent"), n_items=20, length=(300, 600), seed=seed
            )
        )
        assert any(p["stockout"].any() for p in g.parts.values())
        assert any(p["missing"].any() for p in g.parts.values())
        assert any((p["add:missing_vs_zero"] > 0).any() for p in g.parts.values())
        missing_share = sum(p["missing"].sum() for p in g.parts.values()) / sum(
            p["missing"].size for p in g.parts.values()
        )
        requested = g.params["missing_vs_zero"]["process"]["missing_share"]
        assert requested <= missing_share <= requested + 0.02, (seed, missing_share, requested)
        for item in g.panel.items:
            p = g.parts[item.item_id]
            y = targets(item)
            zero_share = np.mean(y[np.isfinite(y)] == 0)
            expected = 1 - g.params["intermittent"]["p"][item.item_id] * (1 - p["stockout"].mean())
            assert abs(zero_share - expected) <= 0.15, (seed, item.item_id, zero_share, expected)


def test_without_on_real_mechanisms():
    r = PanelRecipe(ALL, n_items=6, length=(64, 400), seed=11)
    full = generate_typed_panel(r)
    for mechanism in ALL[1:]:
        cut = generate_typed_panel(r, without=(mechanism,))
        assert cut.params == full.params and cut.inputs == full.inputs
        assert cut.availability == full.availability
        assert_inputs_equal(cut.panel, full.panel, full.inputs)
        assert_inputs_equal(cut.complete, full.complete, full.inputs)
        for a, b in zip(full.panel.items, cut.panel.items, strict=True):
            p, q = full.parts[a.item_id], cut.parts[a.item_id]
            s = full.scalars[a.item_id]
            changed = {f"add:{mechanism}"}
            if mechanism == "missing_vs_zero":
                changed |= {"stockout", "missing", "occ"}
                assert not q["stockout"].any() and not q["missing"].any()
                assert f"add:{mechanism}" not in q
                probability = full.params["intermittent"]["p"][a.item_id]
                assert_array_equal(q["occ"], q["u"] < probability)
                assert np.isfinite(targets(b)).all()
            elif mechanism == "intermittent":
                changed.add("occ")
                assert_array_equal(q["occ"], ~q["stockout"])
            else:
                delta = s["level"][:, None] * p["occ"] * p[f"add:{mechanism}"]
                observed = ~p["missing"]
                np.testing.assert_allclose(
                    (targets(a) - targets(b))[observed], delta[observed], rtol=0, atol=1e-9
                )
            assert set(q) == set(p) - ({f"add:{mechanism}"} & set(p))
            for name in set(p) - changed:
                assert_array_equal(p[name], q[name])
            for name in s:
                assert_array_equal(s[name], cut.scalars[a.item_id][name])
            np.testing.assert_allclose(targets(b), rebuild(cut, b), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        generate_typed_panel(r, without=("multi_target",))


def leaves(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from leaves(nested)
    elif isinstance(value, tuple):
        for nested in value:
            yield from leaves(nested)
    else:
        yield value


def assert_plain_parameters(value):
    assert type(value) in (float, int, str, tuple, dict), type(value)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert type(key) is str
            assert_plain_parameters(nested)
    elif isinstance(value, tuple):
        for nested in value:
            assert_plain_parameters(nested)


def test_recorded_parameters_lie_in_the_range_of_the_setting():
    for heldout in (False, True):
        for seed in range(5):
            g = generate_typed_panel(
                PanelRecipe(ALL, n_items=5, length=(64, 300), heldout=heldout, seed=seed)
            )
            assert_plain_parameters(g.params)
            for mechanism in ALL:
                q = g.params[mechanism]
                assert q["inputs"] == ()
                for key, bounds in RANGES[mechanism].items():
                    lo, hi = bounds[int(heldout)]
                    magnitudes = tuple(leaves(q[key]))
                    assert magnitudes and all(lo <= magnitude <= hi for magnitude in magnitudes)
                for key, value in q.items():
                    if key.endswith("_sign"):
                        assert all(sign in (-1, 1) for sign in leaves(value))
            process = g.params["missing_vs_zero"]["process"]
            assert 0.01 <= process["rate"] <= 0.05
            assert 0.02 <= process["missing_share"] <= 0.15


def test_small_panels_and_hash_seeds():
    for n_items in (1, 2):
        g = generate_typed_panel(PanelRecipe(ALL, n_items=n_items, length=(8, 12), freq="D"))
        for item in g.panel.items:
            np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)
    script = f"""
import hashlib, json
from dataclasses import asdict
from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel
g = generate_typed_panel(PanelRecipe({ALL!r}, n_items=12, seed=5))
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


def test_structure_stream_isolation_and_availability():
    recipe = PanelRecipe(ALL, n_items=6, length=(40, 160), seed=7)
    full = generate_typed_panel(recipe)
    for modified in (
        recipe,
        replace(recipe, mechanisms=ALL[::-1]),
        replace(recipe, n_nuisance=3),
        replace(recipe, complete_inputs=True),
    ):
        assert_generated_equal(full, generate_typed_panel(modified))
    bare = generate_typed_panel(replace(recipe, mechanisms=("multi_target",)))
    for item_id, p in full.parts.items():
        for key in ("base", "xi", "noise", "u", "factor", "factor_e"):
            assert_array_equal(p[key], bare.parts[item_id][key])
    heldout = generate_typed_panel(replace(recipe, heldout=True))
    assert full.params["missing_vs_zero"]["process"] == heldout.params["missing_vs_zero"]["process"]
    for item_id, p in full.parts.items():
        for key in ("base", "xi", "noise", "u", "factor", "factor_e", "stockout", "missing"):
            assert_array_equal(p[key], heldout.parts[item_id][key])


@pytest.mark.parametrize("freq", ["30min", "W-MON", "YE"])
def test_period_fallback_and_calendar_frequency_classes(freq):
    mechanisms = tuple(m for m in ALL if m != "seasonal_lag")
    recipe = PanelRecipe(mechanisms, n_items=2, length=(8, 12), freq=freq)
    draft = grid(recipe)
    g = generate_typed_panel(recipe)
    assert draft.period == {"30min": 48, "W-MON": 52, "YE": 8}[freq]
    assert g.params["calendar"]["fields"] == (("hour", "dow") if freq == "30min" else ("month",))
    for item in g.panel.items:
        np.testing.assert_allclose(g.parts[item.item_id]["base"], recompute_base(g, item))
        np.testing.assert_allclose(targets(item), rebuild(g, item), rtol=1e-12, atol=1e-12)
    if freq == "YE":
        with pytest.raises(ValueError, match="long lag"):
            PanelRecipe(("seasonal_lag",), freq=freq)


def test_seeded_processes_and_nonwrapping_lags():
    recipe = PanelRecipe(ALL, n_items=3, length=(8, 8), heldout=True, seed=4)
    g = generate_typed_panel(recipe)
    n_targets = int(stream(recipe.seed, "multi_target:process").integers(2, 21))
    assert g.params["multi_target"]["process"]["n_targets"] == n_targets
    factor_rng = stream(recipe.seed, "multi_target:inputs")
    process_rng = stream(recipe.seed, "missing_vs_zero:process")
    assert g.params["missing_vs_zero"]["process"] == {
        "rate": float(process_rng.uniform(0.01, 0.05)),
        "missing_share": float(process_rng.uniform(0.02, 0.15)),
    }
    for item in g.panel.items:
        p = g.parts[item.item_id]
        assert_array_equal(p["factor_e"], factor_rng.standard_normal(8))
        for j, delay in enumerate(g.params["multi_target"]["lag"]):
            assert (p["add:multi_target"][j, :delay] == 0).all()
        assert p["add:missing_vs_zero"][0, 0] == 0
        assert p["missing"].any()


def test_rebound_does_not_wrap_around_the_item():
    # A stock-out on the item's last step must not create a rebound on its first step.
    g = generate_typed_panel(
        PanelRecipe(
            ("missing_vs_zero",), n_items=6, length=(64, 400), seed=11, complete_inputs=True
        )
    )
    rebound = g.params["missing_vs_zero"]["rebound"]
    wrapping = 0
    for item in g.panel.items:
        parts = g.parts[item.item_id]
        stockout = parts["stockout"]
        previous = np.concatenate([np.zeros_like(stockout[:, :1]), stockout[:, :-1]], axis=1)
        np.testing.assert_allclose(
            parts["add:missing_vs_zero"], rebound * (previous & ~stockout), rtol=0, atol=1e-12
        )
        if stockout[0, -1] and not stockout[0, 0]:
            wrapping += 1
            assert parts["add:missing_vs_zero"][0, 0] == 0.0
    assert wrapping >= 1  # the fixture must contain the case a wrap-around would get wrong
