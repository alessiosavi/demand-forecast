"""Task 6 contracts: availability, typed rows, budgets and batch padding."""

from collections import Counter
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import (
    EpisodeBuilder,
    EpisodeRequest,
    Item,
    Panel,
    collate,
    derived,
    panel_stats,
)
from demand_forecast.data.episode import builder as episode_builder
from demand_forecast.data.episode.bank import step_features, summary_vectors
from demand_forecast.data.episode.builder import DROPPABLE
from demand_forecast.data.episode.kinds import flag_codes
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.scale import compress, fit_scale, to_scaled
from demand_forecast.data.episode.schema import (
    CH_OBSERVED,
    CH_STEP,
    CH_STEP_FLAG,
    CH_SUPPORT,
    CH_VALUE,
    CH_VALUE_STATS,
    KIND_CATEGORY,
    KIND_FLAG,
    KIND_NUMBER,
    MAX_FEATURE_ROWS,
    N_CHANNELS,
    ROLE_KNOWN,
    ROLE_PANEL,
    ROLE_PAST,
    ROLE_PEER,
    ROLE_SIBLING,
    STEP_FAMILIES,
    SUMMARY_FEATURES,
)
from demand_forecast.data.episode.screening import ScreeningConfig, relevance_number
from demand_forecast.data.episode.value_stats import category_step_stats
from demand_forecast.models import v5
from tests.helpers.episodes import lag_panel, perturb_after, toy_panel
from tests.helpers.lag_scan_reference import scan_delay as reference_scan
from tests.helpers.v5_loop import config


def _plain(panel):
    return replace(
        panel, items=tuple(replace(i, static={}, dynamic={}) for i in panel.items), roles={}
    )


def _equal_inputs(a, b, exclude=()):
    for field in fields(a):
        if field.name in exclude:
            continue
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-6, equal_nan=True, err_msg=field.name)
        else:
            assert x == y, field.name


def _equal_exact(a, b, exclude=()):
    for field in fields(a):
        if field.name in exclude:
            continue
        expected, actual = getattr(a, field.name), getattr(b, field.name)
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected, err_msg=field.name)
        else:
            assert actual == expected, field.name


def test_onset_option_preserves_a_panel_without_categories():
    panel = toy_panel(n_targets=2)
    roles = {name: role for name, role in panel.roles.items() if name != "event"}
    items = tuple(
        replace(item, dynamic={name: item.dynamic[name] for name in roles}, static={})
        for item in panel.items
    )
    panel = replace(panel, items=items, roles=roles)
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(1, (320, 336), 16, 64, "y", 304)
    assert request == replace(request, derive=frozenset())
    baseline = builder.build(request)
    assert KIND_CATEGORY not in baseline.row_kind
    enabled = builder.build(replace(request, derive=frozenset({"onset_rows"})))
    _equal_exact(baseline, enabled)


def test_derive_reaches_every_request(monkeypatch):
    builder = EpisodeBuilder(toy_panel(n_targets=2))
    real_validate = builder._validate
    probes = []

    def validate(request):
        probes.append(request)
        return real_validate(request)

    monkeypatch.setattr(builder, "_validate", validate)
    derive = frozenset({"increment_spread"})
    requests = builder.plan_requests(0, (336,), 5, context=64, derive=derive)
    assert [r.target for r in requests] == ["y", "y1"]
    assert [r.derive for r in [*probes, *requests]] == [derive] * 3
    with pytest.raises(ValueError, match="unknown derive"):
        builder.plan_requests(0, (336,), 5, derive=frozenset({"unknown"}))


def test_increment_spread_preserves_feature_admission_and_other_channels():
    panel = toy_panel(n_items=1)
    item = panel.items[0]
    event = np.resize(np.repeat(np.array(["a", "b", "c"], dtype=object), 4), len(item.targets["y"]))
    dynamic = {
        "event": event,
        **{f"number_{i:02}": item.dynamic["temp"] + i for i in range(MAX_FEATURE_ROWS)},
    }
    roles = dict.fromkeys(dynamic, "past")
    roles["event"] = "known"
    panel = replace(panel, items=(replace(item, dynamic=dynamic),), roles=roles)
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (320, 336), 16, 64, cutoff=304)
    enabled = replace(request, derive=frozenset({"increment_spread"}))
    original, actual = builder.build(request), builder.build(enabled)
    index = original.row_names.index("event")
    assert np.isin(actual.row_role, [ROLE_KNOWN, ROLE_PAST]).sum() == MAX_FEATURE_ROWS
    spread = CH_VALUE_STATS + 1
    assert np.any(original.x[index, ..., spread] != actual.x[index, ..., spread])
    assert np.any(original.xh[..., spread] != actual.xh[..., spread])
    _equal_exact(actual, original, exclude=("x", "xh"))
    for field in ("x", "xh"):
        base, changed = getattr(original, field), getattr(actual, field)
        assert base.dtype == changed.dtype
        np.testing.assert_array_equal(base[..., :spread], changed[..., :spread])
        np.testing.assert_array_equal(base[..., spread + 1 :], changed[..., spread + 1 :])
    for drop in (frozenset({"features"}), frozenset({"value_stats"})):
        plain, derived = (builder.build(replace(r, drop=drop)) for r in (request, enabled))
        np.testing.assert_array_equal(plain.x, derived.x)
        np.testing.assert_array_equal(plain.xh, derived.xh)
        assert plain.row_names == derived.row_names

    # The earlier horizon can overlap the later origin's visible context.
    changed = EpisodeBuilder(perturb_after(panel, request.origins[0], keep_known=True)).build(
        enabled
    )
    first_origin_patches = actual.origin_patches[0]
    np.testing.assert_array_equal(
        changed.x[:, :first_origin_patches], actual.x[:, :first_origin_patches]
    )
    np.testing.assert_array_equal(changed.xh[0], actual.xh[0])


def test_inputs_ignore_what_is_not_available():
    p = toy_panel()
    req = EpisodeRequest(0, (320, 336), 28, context=256)
    e = EpisodeBuilder(p).build(req)
    e2 = EpisodeBuilder(perturb_after(p, 336, value=1e6, keep_known=True)).build(req)
    for f in (
        "x",
        "xh",
        "attr_stats",
        "cand_summary",
        "row_summary",
        "row_score",
        "static_num",
        "loc",
        "scale",
    ):
        assert np.array_equal(getattr(e, f), getattr(e2, f)), f


def test_statistics_stop_at_the_cutoff():
    p = toy_panel()
    req = EpisodeRequest(0, (320, 336), 28, context=256, cutoff=304)
    changed = perturb_after(p, 304, keep_known=True)
    items = []
    for original, modified in zip(p.items, changed.items):
        targets = {name: a.copy() for name, a in modified.targets.items()}
        dynamic = {name: a.copy() for name, a in modified.dynamic.items()}
        for name, values in targets.items():
            values[max(0, 336 - original.start) :] = original.targets[name][
                max(0, 336 - original.start) :
            ]
        dynamic["temp"][max(0, 336 - original.start) :] = original.dynamic["temp"][
            max(0, 336 - original.start) :
        ]
        items.append(replace(modified, targets=targets, dynamic=dynamic))
    e = EpisodeBuilder(p).build(req)
    e2 = EpisodeBuilder(replace(p, items=tuple(items))).build(req)
    for name in (
        "row_summary",
        "row_summary_flags",
        "row_score",
        "cand_summary",
        "cand_flags",
        "cand_score",
        "attr_stats",
        "static_num",
        "loc",
        "scale",
    ):
        np.testing.assert_array_equal(getattr(e, name), getattr(e2, name), err_msg=name)
    assert e.row_names == e2.row_names
    assert not np.array_equal(e.x, e2.x)


def test_known_future_reaches_xh_only():
    p = toy_panel()
    req = EpisodeRequest(0, (320, 336), 28, context=256)
    promo = p.items[0].dynamic["promo"].copy()
    promo[336:] = 1 - promo[336:]
    item = replace(p.items[0], dynamic={**p.items[0].dynamic, "promo": promo})
    a = EpisodeBuilder(p).build(req)
    b = EpisodeBuilder(replace(p, items=(item, *p.items[1:]))).build(req)
    assert not np.array_equal(a.xh, b.xh)
    _equal_inputs(a, b, exclude=("xh",))


def test_known_category_horizon_follows_its_labels():
    p = toy_panel()
    req = EpisodeRequest(0, (336,), 16, context=64)
    a = EpisodeBuilder(p).build(req)
    event = p.items[0].dynamic["event"].copy()
    event[336:352] = "holiday"
    item = replace(p.items[0], dynamic={**p.items[0].dynamic, "event": event})
    b = EpisodeBuilder(replace(p, items=(item, *p.items[1:]))).build(req)
    row = list(a.known_rows).index(a.row_names.index("event"))
    assert not np.array_equal(
        a.xh[:, row, ..., CH_VALUE_STATS:], b.xh[:, row, ..., CH_VALUE_STATS:]
    )
    _equal_inputs(a, b, exclude=("xh",))


def test_channel_table():
    p = toy_panel()
    items = []
    for item in p.items:
        target = item.targets["y"].copy()
        temp = item.dynamic["temp"].copy()
        known = 4 * temp + np.arange(len(temp)) * 0.25
        target[280 - item.start] = np.nan
        temp[281 - item.start] = np.nan
        known[np.array([283, 322, 340]) - item.start] = np.nan
        items.append(
            replace(
                item,
                targets={"y": target},
                dynamic={**item.dynamic, "temp": temp, "known_number": known},
            )
        )
    p = replace(p, items=tuple(items), roles={**p.roles, "known_number": "known"})
    req = EpisodeRequest(1, (320, 336), 28, context=64, cutoff=304)
    e = EpisodeBuilder(p).build(req)
    item = p.items[req.item_index]
    end = req.cutoff - item.start
    columns = {**item.targets, **item.dynamic}
    stats = PanelStats(p, "y", req.cutoff)
    assert {"y", "temp", "known_number"} <= set(e.row_names)
    for row, (name, kind, role) in enumerate(zip(e.row_names, e.row_kind, e.row_role)):
        x = e.x[row].reshape(64, 39)
        if role in (ROLE_PEER, ROLE_PANEL):
            if role == ROLE_PEER:
                attribute, label = name.split("=", 1)
                values, support = stats.peer_row(attribute, label, req.item_index, 272, 336)
                assert e.row_score[row] == e.cand_score[e.row_cand[row]]
            else:
                values, support = stats.panel_row(req.item_index, 272, 336)
            np.testing.assert_allclose(x[:, CH_VALUE], values)
            np.testing.assert_allclose(x[:, CH_SUPPORT], np.log1p(support), rtol=1e-6)
            assert not x[:, CH_STEP:].any()
            assert not e.row_summary[row].any()
        elif kind == KIND_FLAG:
            np.testing.assert_array_equal(
                x[:, CH_VALUE],
                flag_codes(item.dynamic[name], end)[272 - item.start : 336 - item.start],
            )
            assert not x[:, CH_SUPPORT:].any()
        elif kind == KIND_CATEGORY:
            assert not x[:, CH_VALUE].any()
            assert not x[:, CH_SUPPORT:CH_VALUE_STATS].any()
            assert not e.row_summary_flags[row].any()
        else:
            scale = fit_scale(columns[name], end, p.season)
            raw = columns[name][272 - item.start : 336 - item.start]
            observed = np.isfinite(raw)
            scaled = (raw - scale.loc) / scale.scale
            assert np.max(np.abs(scaled[observed])) > 1
            np.testing.assert_array_equal(x[:, CH_OBSERVED], observed)
            np.testing.assert_allclose(
                x[:, CH_VALUE], np.where(observed, np.arcsinh(scaled), 0.0), rtol=1e-6, atol=1e-6
            )
            assert not x[:, CH_SUPPORT].any()
            assert not x[:, CH_VALUE_STATS:].any()
            if role == ROLE_KNOWN:
                known_row = list(e.known_rows).index(row)
                for oi, origin in enumerate(req.origins):
                    raw = columns[name][origin - item.start : origin - item.start + req.horizon]
                    observed = np.isfinite(raw)
                    scaled = (raw - scale.loc) / scale.scale
                    assert np.max(np.abs(scaled[observed])) > 1
                    expected = np.zeros(e.xh.shape[2] * 16)
                    expected[: req.horizon] = np.where(observed, np.arcsinh(scaled), 0.0)
                    horizon_observed = np.zeros(len(expected), dtype=bool)
                    horizon_observed[: req.horizon] = observed
                    xh = e.xh[oi, known_row].reshape(-1, 39)
                    np.testing.assert_array_equal(xh[:, CH_OBSERVED], horizon_observed)
                    np.testing.assert_allclose(xh[:, CH_VALUE], expected, rtol=1e-6, atol=1e-6)
    assert not e.xh[..., CH_STEP:CH_VALUE_STATS].any()


def test_forecast_row_summary_matches_panel_stats_on_grid():
    p = toy_panel(n_targets=2)
    req = EpisodeRequest(1, (320, 336), 16, context=64, target="y", cutoff=304)
    assert p.items[req.item_index].start % p.season != 0
    e = EpisodeBuilder(p).build(req)
    for row in e.forecast_rows:
        stats = PanelStats(p, e.row_names[row], req.cutoff)
        np.testing.assert_allclose(
            e.row_summary[row], stats.summaries[req.item_index], rtol=0, atol=1e-6
        )
        np.testing.assert_array_equal(e.row_summary_flags[row], stats.summary_flags[req.item_index])


def test_seasonal_profiles_are_invariant_to_full_season_grid_shift():
    p = toy_panel()
    req = EpisodeRequest(1, (320, 336), 16, context=64, cutoff=304)
    e = EpisodeBuilder(p).build(req)
    items = list(p.items)
    items[req.item_index] = replace(
        items[req.item_index], start=items[req.item_index].start + p.season
    )
    shifted = EpisodeBuilder(replace(p, items=tuple(items))).build(
        replace(req, origins=tuple(o + p.season for o in req.origins), cutoff=req.cutoff + p.season)
    )
    profiles = [j for j, name in enumerate(SUMMARY_FEATURES) if name.startswith("profile_")]
    row, shifted_row = e.row_names.index("y"), shifted.row_names.index("y")
    assert e.row_summary_flags[row, profiles].any()
    np.testing.assert_allclose(
        e.row_summary[row, profiles], shifted.row_summary[shifted_row, profiles], rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(
        e.row_summary_flags[row, profiles], shifted.row_summary_flags[shifted_row, profiles]
    )


@pytest.mark.parametrize("role", ["known", "past"])
def test_number_row_summary_uses_grid_cutoff(role):
    p = toy_panel()
    p = replace(p, roles={**p.roles, "temp": role})
    req = EpisodeRequest(1, (320, 336), 16, context=64, cutoff=304)
    item = p.items[req.item_index]
    end = req.cutoff - item.start
    raw = item.dynamic["temp"][:end]
    scaled = to_scaled(raw, fit_scale(item.dynamic["temp"], end, p.season))
    expected, flags = summary_vectors(
        scaled[None], raw[None], season=p.season, long_lag=p.long_lag, end=req.cutoff
    )
    e = EpisodeBuilder(p).build(req)
    row = e.row_names.index("temp")
    np.testing.assert_allclose(e.row_summary[row], expected[0], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(e.row_summary_flags[row], flags[0])


@pytest.mark.parametrize("mode", ["flag", "effect"])
def test_expanded_category_summary_uses_grid_cutoff(mode):
    p = toy_panel()
    req = EpisodeRequest(1, (320, 336), 16, context=64, cutoff=304, drop=frozenset({"value_stats"}))
    item = p.items[req.item_index]
    end = req.cutoff - item.start
    event = item.dynamic["event"]
    observed = np.array([label is not None for label in event])
    e = EpisodeBuilder(p).build(req)
    names = (
        [name for name in e.row_names if name.startswith("event=")]
        if mode == "flag"
        else ["event~effect"]
    )
    assert names
    for name in names:
        if mode == "flag":
            raw = np.where(observed[:end], event[:end] == name.split("=", 1)[1], np.nan)
        else:
            target = to_scaled(item.targets["y"], fit_scale(item.targets["y"], end, p.season))
            effects, _ = category_step_stats(event, target, left=0, right=end)
            raw = np.where(observed[:end], effects[:, 0], np.nan)
        expected, flags = summary_vectors(
            raw[None], raw[None], season=p.season, long_lag=p.long_lag, end=req.cutoff
        )
        row = e.row_names.index(name)
        np.testing.assert_allclose(e.row_summary[row], expected[0], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(e.row_summary_flags[row], flags[0])


def test_overflow_item():
    p = toy_panel(n_targets=20)
    p = replace(
        p,
        items=tuple(
            replace(i, targets={f"y{j:02}": a for j, a in enumerate(i.targets.values())})
            for i in p.items
        ),
    )
    e = EpisodeBuilder(p).build(EpisodeRequest(0, (336,), 5, context=64, target="y07"))
    assert len(e.forecast_rows) == 16
    assert e.keep.sum() == 1
    assert e.row_names[e.forecast_rows[np.flatnonzero(e.keep)[0]]] == "y07"
    assert (e.row_role == ROLE_SIBLING).sum() == 15
    with pytest.raises(ValueError):
        EpisodeBuilder(p).build(EpisodeRequest(0, (336,), 5, context=64))
    joint = EpisodeBuilder(_plain(toy_panel(n_targets=3))).build(
        EpisodeRequest(0, (336,), 5, context=64)
    )
    assert len(joint.forecast_rows) == 3 and joint.keep.all()
    assert joint.context_target is None
    assert not len(joint.cand_names)


def test_plan_requests():
    p = toy_panel(n_targets=3)
    builder = EpisodeBuilder(p)
    requests = builder.plan_requests(
        0, (336,), 5, context=64, cutoff=320, drop=frozenset({"calendar"})
    )
    assert [r.target for r in requests] == ["y", "y1", "y2"]
    episodes = [builder.build(r) for r in requests]
    assert all(len(e.forecast_rows) == 3 and e.keep.sum() == 1 for e in episodes)
    for r, e in zip(requests, episodes):
        assert e.context_target == r.target
        assert r.cutoff == 320 and r.drop == frozenset({"calendar"})
        attribute, label = e.attr_names[0].split("=", 1)
        expected, _, _ = PanelStats(p, r.target, 320).value_token(attribute, label, 0)
        np.testing.assert_array_equal(e.attr_stats[0], expected)
    assert not np.array_equal(episodes[0].attr_stats, episodes[1].attr_stats)
    with pytest.raises(ValueError):
        builder.build(EpisodeRequest(0, (336,), 5, context=64))
    plain = EpisodeBuilder(_plain(p)).plan_requests(0, (336,), 5, context=64)
    assert len(plain) == 1 and plain[0].target is None
    assert len(EpisodeBuilder(toy_panel(n_targets=20)).plan_requests(0, (336,), 5)) == 20


@pytest.mark.parametrize(
    "changes",
    [
        {"origins": (320, 330)},
        {"cutoff": 321},
        {"origins": (16, 336)},
        {"context": 250},
        {"context": 0},
        {"context": 2064},
        {"horizon": 0},
        {"horizon": 289},
        {"origins": ()},
        {"origins": (336, 320)},
        {"origins": (320, 320)},
        {"item_index": -1},
        {"item_index": 6},
        {"target": "absent"},
        {"drop": frozenset({"typo"})},
    ],
)
def test_request_validation(changes):
    request = replace(EpisodeRequest(0, (320, 336), 28, context=256), **changes)
    with pytest.raises(ValueError):
        EpisodeBuilder(toy_panel()).build(request)


def test_short_history_and_empty_target():
    p = _plain(toy_panel(n_items=1, length=10))
    e = EpisodeBuilder(p).build(EpisodeRequest(0, (10,), 5, context=64))
    np.testing.assert_array_equal(e.patch_valid, [[False, False, False, True]])
    assert not e.x[:, :3].any()
    assert e.origin_patches.tolist() == [4]
    p = replace(p, items=(replace(p.items[0], start=20),))
    e = EpisodeBuilder(p).build(EpisodeRequest(0, (10,), 5, context=64))
    assert e.fallback.all() and not e.x.any() and not e.patch_valid.any()
    assert not e.label_mask.any()


def test_constant_history_is_a_fallback_row():
    p = _plain(toy_panel(n_items=1, length=20))
    p = replace(p, items=(replace(p.items[0], targets={"y": np.full(20, 7.0)}),))
    e = EpisodeBuilder(p).build(EpisodeRequest(0, (17,), 3, context=32))
    assert e.fallback.tolist() == [True] and e.loc.tolist() == [7.0]
    assert np.isfinite(e.x).all()


def test_model_inputs_are_finite():
    p = toy_panel()
    item = p.items[1]
    y, temp, event = (
        item.targets["y"].copy(),
        item.dynamic["temp"].copy(),
        item.dynamic["event"].copy(),
    )
    y[::7], temp[::5], event[::3] = np.nan, np.nan, None
    item = replace(
        item,
        targets={"y": y},
        dynamic={**item.dynamic, "temp": temp, "event": event, "empty": np.full(len(y), np.nan)},
    )
    p = replace(
        p,
        items=tuple(
            replace(i, dynamic={**i.dynamic, "empty": np.full(len(i.targets["y"]), np.nan)})
            if n != 1
            else item
            for n, i in enumerate(p.items)
        ),
        roles={**p.roles, "empty": "known"},
    )
    e = EpisodeBuilder(p).build(EpisodeRequest(1, (33,), 19, context=64))
    b = collate([e, EpisodeBuilder(toy_panel()).build(EpisodeRequest(0, (336,), 33, context=96))])
    for record in (e, b):
        for field in fields(record):
            values = getattr(record, field.name)
            if field.name == "labels":
                assert not np.isinf(np.asarray(values)).any()
                assert np.isfinite(np.asarray(values)[np.asarray(record.label_mask)]).all()
            elif isinstance(values, (np.ndarray, torch.Tensor)):
                assert np.isfinite(np.asarray(values)).all(), field.name
        for name in ("x", "xh"):
            values = np.asarray(getattr(record, name))
            assert not values[values[..., CH_OBSERVED] == 0].any()
    assert not e.x[e.row_names.index("empty")].any()


def test_horizon_not_a_multiple_of_16():
    p = toy_panel()
    req = EpisodeRequest(0, (337,), 5, context=64)
    a = EpisodeBuilder(p).build(req)
    assert a.xh.shape[2] == 1
    assert a.label_mask[..., :5].all() and not a.label_mask[..., 5:].any()
    assert not a.xh[..., 5:, :].any()
    assert not a.calh_phases[..., 5:, :].any()
    assert (a.calh_ids[..., 5:, :] == -1).all()
    b = EpisodeBuilder(perturb_after(p, 342, keep_known=False)).build(req)
    _equal_inputs(a, b)


def test_unsupported_peer_rows_are_not_built():
    p = toy_panel()
    items = tuple(
        replace(i, static={"group": "alone" if n == 0 else "shared"}) for n, i in enumerate(p.items)
    )
    e = EpisodeBuilder(replace(p, items=items)).build(EpisodeRequest(0, (336,), 5, context=64))
    assert "group=alone" not in e.row_names
    assert not (e.row_role == ROLE_PEER).any()
    assert (e.row_role == ROLE_PANEL).any()


@pytest.mark.parametrize(
    "dated,aligned,frequency",
    [
        (True, True, "D"),
        (False, True, None),
        (False, False, None),
        (False, True, "D"),
        (False, False, "h"),
    ],
)
def test_dated_and_aligned_cases(dated, aligned, frequency):
    p = replace(toy_panel(dated=dated), aligned=aligned, freq=frequency)
    e = EpisodeBuilder(p).build(EpisodeRequest(0, (336,), 5, context=64))
    assert bool((e.cal_ids >= 0).any()) == dated
    assert bool((e.row_role == ROLE_PEER).any()) == aligned
    assert bool((e.row_role == ROLE_PANEL).any()) == aligned
    assert len(e.attr_stats) > 0
    assert e.freq_class == p.freq_class
    if not dated:
        assert not e.cal_phases.any() and not e.calh_phases.any()
        assert (e.calh_ids == -1).all()


def test_no_feature_single_item_panel():
    p = _plain(toy_panel(n_items=1))
    e = EpisodeBuilder(p).build(EpisodeRequest(0, (336,), 5, context=64))
    assert len(e.x) == 1
    assert len(e.attr_stats) == len(e.cand_names) == len(e.static_num) == 0


def test_budgets_after_expansion():
    p = toy_panel(n_items=20, length=96)
    items = []
    for n, item in enumerate(p.items):
        dynamic = {
            f"c{j:02}": np.array(
                [f"label{(t + j) % 3}" for t in range(len(item.targets["y"]))], dtype=object
            )
            for j in range(20)
        }
        static = {"huge": [f"shared{j}" for j in range(17)]}
        static.update({f"s{j:02}": float((n + 0.25) ** (1 + j / 100)) for j in range(20)})
        items.append(replace(item, dynamic=dynamic, static=static))
    p = replace(p, items=tuple(items), roles={f"c{j:02}": "known" for j in range(20)})
    e = EpisodeBuilder(p).build(
        EpisodeRequest(0, (64,), 5, context=64, drop=frozenset({"value_stats"}))
    )
    feature = np.isin(e.row_role, [ROLE_KNOWN, ROLE_PAST])
    assert feature.sum() == 16
    assert any(name.endswith("~effect") for name in e.row_names)
    assert not any(name.startswith("huge=") for name in e.attr_names)
    assert "huge" not in e.cand_names
    dispositions = {(c.source, c.name): c.removed for c in e.trace}
    assert dispositions["static", "huge"] == "budget"
    for name in p.roles:
        has_row = any(row.startswith((f"{name}=", f"{name}~")) for row in e.row_names)
        assert dispositions["known", name] == (None if has_row else "budget")
    assert len(e.attr_stats) <= 16 and len(e.static_num) <= 16
    assert np.isin(e.row_role, [ROLE_PEER, ROLE_PANEL]).sum() <= 8


def test_row_keys_follow_builder_rows_after_feature_budget(monkeypatch):
    steps = np.arange(48, dtype=np.float64)
    category = np.resize(np.array(["a", "b", "c"], dtype=object), len(steps))
    items = tuple(
        Item(
            str(i),
            0,
            {"y": steps + i, "sibling": steps[::-1] + i},
            {"number": steps**2, "category": category},
            {"brand": "shared"},
        )
        for i in range(2)
    )
    panel = Panel("row keys", items, {"number": "known", "category": "past"}, None, None, True)
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (40,), 3, context=32, target="y", drop=frozenset({"value_stats"}))
    category_key = ("past", "category")
    expected = [
        ("y", None),
        ("sibling", ("sibling", "sibling")),
        ("number", ("known", "number")),
        ("category=a", category_key),
        ("category=b", category_key),
        ("category=c", category_key),
        ("category~effect", category_key),
        ("brand=shared", None),
        ("__panel__", None),
    ]

    full = builder.build(request)
    assert len(full.row_keys) == len(full.x)
    assert list(zip(full.row_names, full.row_keys, strict=True)) == expected
    assert not hasattr(collate([full]), "row_keys")

    monkeypatch.setattr(episode_builder, "MAX_FEATURE_ROWS", 2)
    limited = builder.build(request)
    assert len(limited.row_keys) == len(limited.x)
    limited_rows = list(zip(limited.row_names, limited.row_keys, strict=True))
    assert limited_rows == expected[:4] + expected[-2:]


@pytest.mark.parametrize("drop", DROPPABLE)
def test_drop_switches(drop):
    p = toy_panel(n_targets=2)
    req = EpisodeRequest(0, (336,), 5, context=64, target="y")
    a = EpisodeBuilder(p).build(req)
    b = EpisodeBuilder(p).build(replace(req, drop=frozenset({drop})))
    roles = {
        "features": (ROLE_KNOWN, ROLE_PAST),
        "siblings": (ROLE_SIBLING,),
        "peers": (ROLE_PEER,),
        "panel_row": (ROLE_PANEL,),
    }
    if drop in roles:
        assert not np.isin(b.row_role, roles[drop]).any()
        assert b.row_names == [
            name for name, role in zip(a.row_names, a.row_role) if role not in roles[drop]
        ]
    elif drop == "attributes":
        assert not len(b.attr_stats)
        assert b.cand_names == a.cand_names and b.row_names == a.row_names
    elif drop == "static_numbers":
        assert not len(b.static_num)
        assert b.row_names == a.row_names
    elif drop == "calendar":
        assert not b.cal_phases.any() and (b.cal_ids == -1).all()
        assert not b.calh_phases.any() and (b.calh_ids == -1).all()
        np.testing.assert_array_equal(a.x, b.x)
    elif drop in STEP_FAMILIES:
        family = STEP_FAMILIES[drop]
        assert not b.x[..., CH_STEP + family.start : CH_STEP + family.stop].any()
        assert not b.x[..., CH_STEP_FLAG + family.start : CH_STEP_FLAG + family.stop].any()
        np.testing.assert_array_equal(a.x[..., CH_VALUE], b.x[..., CH_VALUE])
    elif drop == "value_effects":
        _equal_inputs(a, b)
    else:
        assert not (b.row_kind == KIND_CATEGORY).any()
        assert "event~effect" in b.row_names
        assert not b.x[..., CH_VALUE_STATS:].any()


def test_order_independence():
    p = toy_panel(n_targets=2)
    req = EpisodeRequest(0, (320, 336), 19, context=64, target="y")
    a = EpisodeBuilder(p).build(req)
    items = tuple(
        replace(
            i,
            targets=dict(reversed(list(i.targets.items()))),
            dynamic=dict(reversed(list(i.dynamic.items()))),
            static={
                k: list(reversed(v)) if isinstance(v, list) else v
                for k, v in reversed(list(i.static.items()))
            },
        )
        for i in reversed(p.items)
    )
    b = EpisodeBuilder(replace(p, items=items, roles=dict(reversed(list(p.roles.items()))))).build(
        replace(req, item_index=5)
    )
    _equal_inputs(a, b)


def test_statistics_are_cached(monkeypatch):
    calls = []
    original = PanelStats.__init__

    def counted(self, panel, target, cutoff):
        calls.append((target, cutoff))
        original(self, panel, target, cutoff)

    monkeypatch.setattr(PanelStats, "__init__", counted)
    builder = EpisodeBuilder(toy_panel(n_targets=2))
    for i in range(6):
        for target in ("y", "y1"):
            builder.build(EpisodeRequest(i, (336,), 5, context=64, target=target))
    assert calls == [("y", 336), ("y1", 336)]
    for cutoff in range(300, 316):
        builder.build(EpisodeRequest(0, (336,), 5, context=64, cutoff=cutoff, target="y"))
    builder.build(EpisodeRequest(0, (336,), 5, context=64, target="y"))
    assert calls.count(("y", 336)) == 2


@pytest.mark.parametrize("use_default", [True, False], ids=["default", "unbounded"])
def test_statistics_are_constructed_once_per_target_and_cutoff(monkeypatch, use_default):
    panel = _plain(toy_panel(n_items=1, length=96, n_targets=9))
    panel = replace(panel, items=(replace(panel.items[0], static={"id": "only"}),))
    targets = sorted(panel.items[0].targets)
    calls = Counter()
    original = PanelStats.__init__

    def counted(self, panel, target, cutoff):
        calls[target, cutoff] += 1
        original(self, panel, target, cutoff)

    monkeypatch.setattr(PanelStats, "__init__", counted)
    provider = None if use_default else panel_stats.PanelStatsProvider(panel, limit=None)
    builder = EpisodeBuilder(panel, stats=provider)
    (joint,) = builder.plan_requests(0, (64,), 5, context=64)
    assert joint.target is None
    builder.build(joint)
    for target in targets:
        builder.build(replace(joint, target=target))

    builds = sum(calls.values())
    assert builds == 9, f"Expected 9 statistics constructions, got {builds} (B0 built 17)"
    assert builder.stats.builds == builds
    assert calls == {(target, 64): 1 for target in targets}


def test_default_builder_is_bounded_at_sixteen():
    p = _plain(toy_panel(n_items=1, length=96, n_targets=16))
    builder = EpisodeBuilder(p)
    (joint,) = builder.plan_requests(0, (64,), 5, context=64)
    assert builder.stats.builds == 1
    targets = sorted(p.items[0].targets)
    for target in targets:
        builder.build(replace(joint, target=target))
    assert builder.stats.builds == 16

    builder.build(replace(joint, target=targets[0]))  # Refresh the first pair.
    builder.build(replace(joint, target=targets[0], cutoff=63))
    builder.build(replace(joint, target=targets[0]))
    assert builder.stats.builds == 17
    builder.build(replace(joint, target=targets[1]))
    assert builder.stats.builds == 18


def test_release_frees_a_cutoff():
    provider = panel_stats.PanelStatsProvider(toy_panel(n_targets=2), limit=None)
    released = [provider.get(target, 320) for target in ("y", "y1")]
    retained = provider.get("y", 336)
    provider.release(320)
    provider.release(320)
    assert provider.get("y", 336) is retained
    for target, old in zip(("y", "y1"), released, strict=True):
        assert provider.get(target, 320) is not old
    assert provider.builds == 5


def test_collate_pads():
    a = EpisodeBuilder(_plain(toy_panel(n_items=1))).build(EpisodeRequest(0, (337,), 5, context=32))
    b = EpisodeBuilder(toy_panel(n_targets=2)).build(
        EpisodeRequest(0, (320, 336), 33, context=96, target="y")
    )
    batch = collate([a, b])
    assert batch.x.shape == (2, len(b.x), 6, 16, 39)
    assert batch.row_mask.sum(1).tolist() == [len(a.x), len(b.x)]
    assert batch.forecast_mask.sum(1).tolist() == [1, 2]
    assert batch.known_mask.sum(1).tolist() == [len(a.known_rows), len(b.known_rows)]
    assert batch.cand_mask.sum(1).tolist() == [0, len(b.cand_names)]
    assert batch.attr_mask.sum(1).tolist() == [0, len(b.attr_stats)]
    assert batch.static_num_mask.sum(1).tolist() == [0, len(b.static_num)]
    assert batch.origin_mask.tolist() == [[True, False], [True, True]]
    assert batch.horizon_mask.tolist() == [[True, False, False], [True, True, True]]
    assert batch.origin_patches[0, 0] == a.origin_patches[0] + 4
    assert not batch.patch_valid[0, :, :4].any()
    np.testing.assert_array_equal(batch.x[0, :1, 4:].numpy(), a.x)
    np.testing.assert_array_equal(batch.x[1].numpy(), b.x)
    for name in (
        "row_kind",
        "row_role",
        "row_cand",
        "forecast_rows",
        "known_rows",
        "cal_ids",
        "calh_ids",
        "cand_kind",
        "attr_cand",
        "static_num_cand",
        "origin_patches",
        "freq_class",
        "horizon",
    ):
        assert getattr(batch, name).dtype == torch.long, name
    assert batch.to("cpu").x.device.type == "cpu"
    assert not batch.x[~batch.row_mask].any()
    assert not batch.xh[0].any()
    assert batch.row_names == [a.row_names, b.row_names]
    with pytest.raises(ValueError):
        collate([])


def test_expanded_feature_budget_breaks_score_ties_by_row_name():
    p = _plain(toy_panel(n_items=1, length=96))
    steps = np.arange(96)
    dynamic = {f"a{j:02}": np.sin(steps + j) for j in range(15)}
    dynamic["z"] = np.array(["x", "a", "m"] * 32, dtype=object)
    p = replace(
        p, items=(replace(p.items[0], dynamic=dynamic),), roles={name: "known" for name in dynamic}
    )
    e = EpisodeBuilder(p, ScreeningConfig(min_steps=1000)).build(
        EpisodeRequest(0, (80,), 5, context=64, drop=frozenset({"value_stats"}))
    )
    assert [e.row_names[i] for i in e.known_rows] == [*(f"a{j:02}" for j in range(15)), "z=a"]
    assert all(c.removed is None for c in e.trace)


def test_keep_columns_filters_before_allocating_budgets():
    p = toy_panel(n_targets=3)
    req = EpisodeRequest(
        0, (336,), 5, context=64, target="y", keep_columns=frozenset({"promo", "y2"})
    )
    e = EpisodeBuilder(p).build(req)
    assert e.row_names[:3] == ["y", "y2", "promo"]
    assert "temp" not in e.row_names and "event" not in e.row_names and "y1" not in e.row_names
    assert len(e.cand_names) > 0


def test_joint_episode_scores_features_against_every_kept_target():
    p = toy_panel(n_items=1, n_targets=3)
    item = p.items[0]
    p = replace(
        p,
        items=(replace(item, static={}, dynamic={"temp": item.dynamic["temp"]}),),
        roles={"temp": "past"},
    )
    req = EpisodeRequest(0, (336,), 5, context=64)
    e = EpisodeBuilder(p).build(req)
    expected = max(
        relevance_number(item.dynamic["temp"][:336], y[:336], (0, 1, 7), 16)
        for y in item.targets.values()
    )
    assert e.row_score[e.row_names.index("temp")] == pytest.approx(expected)
    assert not np.isin(e.row_role, [ROLE_PEER, ROLE_PANEL]).any()
    assert len(EpisodeBuilder(p).plan_requests(0, (336,), 5, context=64)) == 1


def test_category_horizon_statistics_freeze_at_each_origin():
    p = toy_panel()
    req = EpisodeRequest(0, (320, 336), 16, context=64, cutoff=304)
    e = EpisodeBuilder(p).build(req)
    row = list(e.known_rows).index(e.row_names.index("event"))
    target = (p.items[0].targets["y"] - e.loc[0]) / e.scale[0]
    for oi, origin in enumerate(req.origins):
        expected, _ = category_step_stats(
            p.items[0].dynamic["event"], target, left=origin, right=origin + 16, frozen_at=origin
        )
        np.testing.assert_array_equal(e.xh[oi, row, ..., CH_VALUE_STATS:].reshape(16, 6), expected)
    expanded = EpisodeBuilder(p).build(replace(req, drop=frozenset({"value_stats"})))
    effect = list(expanded.known_rows).index(expanded.row_names.index("event~effect"))
    np.testing.assert_array_equal(
        expanded.xh[:, effect, ..., CH_VALUE], e.xh[:, row, ..., CH_VALUE_STATS]
    )


def test_category_effect_number_has_causal_bank_channels():
    p = toy_panel()
    req = EpisodeRequest(0, (336,), 16, context=64, drop=frozenset({"value_stats"}))
    e = EpisodeBuilder(p).build(req)
    row = e.row_names.index("event~effect")
    assert e.x[row, ..., CH_STEP_FLAG:CH_VALUE_STATS].any()
    assert not e.xh[..., CH_STEP:CH_VALUE_STATS].any()
    changed = EpisodeBuilder(perturb_after(p, 336, keep_known=True)).build(req)
    np.testing.assert_array_equal(e.x[row], changed.x[row])


def test_plan_requests_splits_for_dynamic_categories_without_statics():
    p = toy_panel(n_targets=3)
    p = replace(p, items=tuple(replace(i, static={}) for i in p.items))
    requests = EpisodeBuilder(p).plan_requests(0, (336,), 5, context=64)
    assert [r.target for r in requests] == ["y", "y1", "y2"]


def test_collate_preserves_all_unpadded_fields():
    p = toy_panel(n_targets=2)
    req = EpisodeRequest(
        0,
        (337,),
        5,
        context=32,
        target="y",
        drop=frozenset({"siblings", "static_numbers"}),
        keep_columns=frozenset({"promo"}),
    )
    a = EpisodeBuilder(p).build(req)
    b = EpisodeBuilder(p).build(EpisodeRequest(0, (320, 336), 33, context=96, target="y"))
    assert a.trace and b.trace
    reasons = {(c.source, c.name): c.removed for c in a.trace}
    assert reasons["sibling", "y1"] == "siblings"
    assert reasons["known", "event"] == "keep_columns"
    batch = collate([a, b])
    assert not hasattr(batch, "trace")
    for index, episode in enumerate((a, b)):
        for field in fields(episode):
            name = field.name
            value = getattr(episode, name)
            if not isinstance(value, np.ndarray):
                continue
            restored = getattr(batch, name)[index]
            if name in ("x", "patch_valid"):
                restored = restored[: len(episode.x), -episode.x.shape[1] :]
            elif name in ("cal_phases", "cal_ids"):
                restored = restored[-episode.x.shape[1] :]
            else:
                restored = restored[tuple(slice(0, length) for length in value.shape)]
            expected = value + 4 if name == "origin_patches" and index == 0 else value
            np.testing.assert_array_equal(restored.numpy(), expected, err_msg=name)
    assert batch.loc.dtype == batch.scale.dtype == torch.float64
    assert batch.x.dtype == batch.xh.dtype == torch.float32


def test_value_effects_can_be_dropped_from_the_category_fallback():
    p = toy_panel()
    builder = EpisodeBuilder(p)
    request = EpisodeRequest(0, (336,), 16, context=64, drop=frozenset({"value_stats"}))
    both_drops = replace(request, drop=request.drop | {"value_effects"})
    full = builder.build(request)
    flags = builder.build(both_drops)
    assert "value_effects" in DROPPABLE
    assert "event~effect" in full.row_names
    assert flags.row_names == [name for name in full.row_names if name != "event~effect"]
    assert any(name.startswith("event=") for name in flags.row_names)
    for index, name in enumerate(flags.row_names):
        np.testing.assert_array_equal(flags.x[index], full.x[full.row_names.index(name)])
    assert not flags.x[..., CH_VALUE_STATS:].any()


def test_known_category_with_no_observed_prefix_has_no_removal_disposition():
    event = np.array([None] * 16 + ["a", "b", "c"] * 6, dtype=object)
    item = Item("only", 0, {"y": np.arange(len(event), dtype=np.float64)}, {"event": event}, {})
    panel = Panel("late event", (item,), {"event": "known"}, None, None, False)
    request = EpisodeRequest(
        0, (16,), 16, context=16, drop=frozenset({"value_stats", "value_effects"})
    )

    episode = EpisodeBuilder(panel).build(request)

    assert episode.row_names == ["y"]
    (candidate,) = episode.trace
    assert (candidate.source, candidate.name, candidate.kind) == ("known", "event", KIND_CATEGORY)
    assert candidate.removed is None


def test_apply_summary_off_returns_the_transformed_episode():
    episode = EpisodeBuilder(toy_panel()).build(EpisodeRequest(0, (336,), 5, context=64))
    original = episode.row_summary.copy()
    i = SUMMARY_FEATURES.index("log_length")
    assert original[..., i].any()
    result = episode_builder.apply_summary_off(episode, ("log_length",))
    original[..., i] = 0
    np.testing.assert_array_equal(result.row_summary, original)
    assert not result.cand_flags[..., i].any()
    assert not result.row_summary_flags[..., i].any()


def test_scan_is_computed_once_per_item_column_target_cutoff(monkeypatch):
    panel = lag_panel(length=730)
    item = panel.items[0]
    item = replace(
        item,
        targets={"z": item.targets["y"], "y": item.dynamic["driver"].copy()},
        dynamic={**item.dynamic, "second": item.dynamic["driver"] + item.targets["y"] / 10},
    )
    panel = replace(
        panel,
        items=(item, replace(item, item_id="second")),
        roles={"driver": "past", "second": "past"},
    )
    calls = []
    real_scan = getattr(derived, "scan_delay", None)

    def record(x, y, **kwargs):
        calls.append((x.copy(), y.copy()))
        return real_scan(x, y, **kwargs)

    monkeypatch.setattr(derived, "scan_delay", record, raising=False)
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (700,), 12, 128, target="z", cutoff=696)
    builder.build(request)
    assert calls == []

    enabled = replace(request, derive=frozenset({"lag_rows"}))
    (planned,) = builder.plan_requests(
        0, enabled.origins, 12, context=128, cutoff=696, derive=enabled.derive
    )
    assert planned.target is None
    assert (
        "driver~lag" not in builder.build(planned).row_names
    )  # Sorted first target is contemporaneous.
    assert len(calls) == 2
    assert "driver~lag" in builder.build(enabled).row_names
    assert len(calls) == 4
    assert {x.tobytes() for x, _ in calls[2:]} == {
        values[184:696].tobytes() for values in item.dynamic.values()
    }
    for _, y in calls[2:]:
        np.testing.assert_array_equal(y, item.targets["z"][184:696])
    builder.build(enabled)
    builder.build(replace(enabled, target="y"))
    assert len(calls) == 4  # Hits include rejected scans and joint/designated requests for y.
    strict = EpisodeBuilder(panel, ScreeningConfig(no_evidence=1.0), stats=builder.stats)
    assert "driver~lag" not in strict.build(enabled).row_names
    assert len(calls) == 4

    builder.build(replace(enabled, item_index=1))
    builder.build(replace(enabled, cutoff=700))
    assert len(calls) == 8
    builder.stats.release(696)
    builder.build(replace(enabled, cutoff=700))
    builder.build(enabled)
    assert len(calls) == 10


def test_lag_row_layout():
    panel = lag_panel(start=19)
    item = panel.items[0]
    origin, cutoff, delay = 531, 523, 5
    request = EpisodeRequest(0, (origin,), 12, 544, cutoff=cutoff, derive=frozenset({"lag_rows"}))
    episode = EpisodeBuilder(panel).build(request)
    index = episode.row_names.index("driver~lag")
    assert episode.row_kind[index] == KIND_NUMBER
    assert episode.row_role[index] == ROLE_KNOWN
    assert episode.row_cand[index] == -1
    assert episode.row_keys[index] == ("derived", "driver~lag")
    end = cutoff - item.start
    found, score, _ = reference_scan(item.dynamic["driver"][:end], item.targets["y"][:end], 1)
    assert found == delay
    assert episode.row_score[index] == pytest.approx(score)
    assert episode.trace[-1] == episode_builder.Candidate(
        "driver~lag", "derived", KIND_NUMBER, pytest.approx(score), None
    )

    shifted = np.full_like(item.dynamic["driver"], np.nan)
    shifted[delay:] = item.dynamic["driver"][:-delay]
    shifted_panel = replace(
        panel, items=(replace(item, dynamic={"shifted": shifted}),), roles={"shifted": "known"}
    )
    ordinary = EpisodeBuilder(shifted_panel).build(replace(request, derive=frozenset()))
    reference = ordinary.row_names.index("shifted")
    for name in ("x", "row_summary", "row_summary_flags"):
        np.testing.assert_allclose(
            getattr(episode, name)[index], getattr(ordinary, name)[reference], atol=1e-6
        )
    context = episode.x[index].reshape(request.context, -1)
    assert not context[: item.start + delay - (origin - request.context)].any()


def test_horizon_cells_per_origin():
    delay = 5
    panel = lag_panel(start=19, delay=delay)
    driver = panel.items[0].dynamic["driver"]
    origins = (531, 547)
    driver[origins[0] - 2 - panel.items[0].start] = np.nan
    request = EpisodeRequest(0, origins, 12, 544, derive=frozenset({"lag_rows"}))
    episode = EpisodeBuilder(panel).build(request)
    index = episode.row_names.index("driver~lag")
    (known,) = np.flatnonzero(episode.known_rows == index)
    shifted = np.r_[np.full(delay, np.nan), driver]
    scale = fit_scale(shifted, origins[0] - panel.items[0].start, 1)
    for oi, origin in enumerate(origins):
        local = origin - panel.items[0].start
        actual = episode.xh[oi, known].reshape(-1, episode.xh.shape[-1])
        expected = np.zeros_like(actual)
        source = driver[local - delay : local]
        observed = np.isfinite(source)
        expected[:delay, CH_OBSERVED] = observed
        expected[:delay, CH_VALUE] = np.where(observed, compress(to_scaled(source, scale)), 0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_nothing_at_or_after_an_origin_is_read():
    panel = lag_panel(start=19)
    request = EpisodeRequest(0, (531, 547), 12, 544, derive=frozenset({"lag_rows"}))
    original = EpisodeBuilder(panel).build(request)
    altered = EpisodeBuilder(perturb_after(panel, request.origins[0])).build(request)
    assert "driver~lag" in original.row_names
    _equal_exact(original, altered, exclude=("x", "xh", "labels"))
    prefix = original.origin_patches[0]
    np.testing.assert_array_equal(original.x[:, :prefix], altered.x[:, :prefix])
    np.testing.assert_array_equal(original.xh[0], altered.xh[0])
    assert not np.array_equal(original.xh[1], altered.xh[1])


def _fill_legacy_slots(panel, count):
    item = panel.items[0]
    columns = {f"known{i}": item.targets["y"] + i for i in range(count)}
    return replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, **columns}),),
        roles={**panel.roles, **dict.fromkeys(columns, "known")},
    )


def test_bases_first_under_the_cap():
    panel = _fill_legacy_slots(lag_panel(), MAX_FEATURE_ROWS - 2)
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (512,), 12, 128)
    enabled = replace(request, derive=frozenset({"lag_rows"}))
    legacy, actual = builder.build(request), builder.build(enabled)
    assert len(actual.row_names) == len(legacy.row_names) + 1
    assert set(actual.row_names) == {*legacy.row_names, "driver~lag"}
    for i, name in enumerate(legacy.row_names):
        np.testing.assert_array_equal(actual.x[actual.row_names.index(name)], legacy.x[i])
    assert actual.trace[:-1] == legacy.trace
    (planned,) = builder.plan_requests(0, (512,), 12, context=128, derive=enabled.derive)
    _equal_exact(builder.build(planned), actual)
    for restriction in (
        {"drop": frozenset({"features"})},
        {"keep_columns": frozenset({"known0"})},
    ):
        restricted = builder.build(replace(enabled, **restriction))
        assert "driver" not in restricted.row_names
        assert all(c.source != "derived" for c in restricted.trace)


def test_name_collision_builds_nothing():
    panel = lag_panel()
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (512,), 12, 128, derive=frozenset({"lag_rows"}))
    assert "driver~lag" in builder.build(request).row_names
    item = panel.items[0]
    collision = np.zeros_like(item.targets["y"])  # A screened-out column still reserves its name.
    panel = replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, "driver~lag": collision}),),
        roles={**panel.roles, "driver~lag": "past"},
    )
    actual = EpisodeBuilder(panel).build(request)
    assert "driver~lag" not in actual.row_names
    assert all(c.source != "derived" for c in actual.trace)


def test_no_scan_without_a_free_slot(monkeypatch):
    panel = _fill_legacy_slots(lag_panel(), MAX_FEATURE_ROWS - 1)
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (512,), 12, 128)
    calls = 0
    real_scan = derived.scan_delay

    def record(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(derived, "scan_delay", record)
    legacy = builder.build(request)
    assert "driver" in legacy.row_names and len(legacy.row_names) == MAX_FEATURE_ROWS + 1
    actual = builder.build(replace(request, derive=frozenset({"lag_rows"})))
    assert calls == 0
    _equal_exact(legacy, actual)


def test_overflow_is_traced_as_budget():
    panel = _fill_legacy_slots(lag_panel(), MAX_FEATURE_ROWS - 3)
    item = panel.items[0]
    second = item.dynamic["driver"] + np.random.default_rng(829).normal(scale=0.3, size=560)
    panel = replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, "a_second": second}),),
        roles={**panel.roles, "a_second": "past"},
    )
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (512,), 12, 128, derive=frozenset({"lag_rows"}))
    actual = builder.build(request)
    assert "driver~lag" in actual.row_names
    assert "a_second" in actual.row_names and "a_second~lag" not in actual.row_names
    kept, budget = [c for c in actual.trace if c.source == "derived"]
    assert kept.name == "driver~lag" and kept.removed is None
    assert budget.name == "a_second~lag" and budget.removed == "budget"
    assert kept.score > budget.score


def test_a_delay_longer_than_the_horizon_covers_it():
    panel = lag_panel(delay=11)
    request = EpisodeRequest(0, (512,), 4, 128, derive=frozenset({"lag_rows"}))
    actual = EpisodeBuilder(panel).build(request)
    (i,) = np.flatnonzero(actual.known_rows == actual.row_names.index("driver~lag"))
    np.testing.assert_array_equal(actual.xh[0, i, 0, :4, CH_OBSERVED], np.ones(4))
    assert not actual.xh[0, i, 0, 4:].any()


def _onset_panel(role="known", *, run_lengths=(8, 8, 8)):
    labels = np.resize(np.repeat(["a", "b", "c"], run_lengths), 192).astype(object)
    target = np.resize(np.repeat([1.0, 9.0, 17.0], run_lengths), len(labels))
    target[np.r_[True, labels[1:] != labels[:-1]]] += 0.25
    item = Item("item", 19, {"y": target}, {"event": labels}, {})
    return Panel("onsets", (item,), {"event": role}, "D", None, True)


def test_onset_option_preserves_legacy_rows_in_place():
    panel = _onset_panel()
    item = panel.items[0]
    dynamic = {"amount": np.arange(len(item.targets["y"]), dtype=float), **item.dynamic}
    panel = replace(
        panel,
        items=(replace(item, dynamic=dynamic),),
        roles={**panel.roles, "amount": "known"},
    )
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (115, 147), 13, 128, cutoff=102)
    baseline = builder.build(request)
    enabled = builder.build(replace(request, derive=frozenset({"onset_rows"})))
    assert baseline.row_names == ["y", "amount", "event"]
    assert enabled.row_names == [*baseline.row_names, "event~onset"]
    assert enabled.row_keys[:-1] == baseline.row_keys
    assert enabled.trace[:-1] == baseline.trace
    np.testing.assert_array_equal(enabled.forecast_rows, baseline.forecast_rows)
    np.testing.assert_array_equal(enabled.known_rows[:-1], baseline.known_rows)
    for field in (
        "x",
        "patch_valid",
        "row_kind",
        "row_role",
        "row_summary",
        "row_summary_flags",
        "row_score",
        "row_cand",
        "xh",
    ):
        expected, actual = getattr(baseline, field), getattr(enabled, field)
        actual = actual[:, :-1] if field == "xh" else actual[:-1]
        assert actual.shape == expected.shape, field
        assert actual.dtype == expected.dtype, field
        assert actual.tobytes() == expected.tobytes(), field


def test_each_onset_horizon_freezes_at_its_own_origin():
    panel = _onset_panel(run_lengths=(2, 3, 4))
    item = panel.items[0]
    origins = (item.start + 96, item.start + 128)
    request = EpisodeRequest(
        0, origins, 37, 128, cutoff=origins[0] - 7, derive=frozenset({"onset_rows"})
    )
    episode = EpisodeBuilder(panel).build(request)
    row = episode.row_names.index("event~onset")
    (known,) = np.flatnonzero(episode.known_rows == row)
    raw = item.targets["y"]
    target = to_scaled(raw, fit_scale(raw, request.cutoff - item.start, panel.season))
    labels = item.dynamic["event"]
    for oi, origin in enumerate(origins):
        local = origin - item.start
        left, right = local, local + request.horizon
        actual = episode.xh[oi, known].reshape(-1, N_CHANNELS)
        values, observed = derived.onset_profile(
            labels, target, left=left, right=right, frozen_at=local, season=panel.season
        )
        unfrozen, _ = derived.onset_profile(
            labels, target, left=left, right=right, frozen_at=None, season=panel.season
        )
        expected = np.zeros_like(actual)
        expected[: request.horizon, CH_VALUE] = values
        expected[: request.horizon, CH_OBSERVED] = observed
        assert np.any(expected[: request.horizon, CH_VALUE] != unfrozen.astype(expected.dtype))
        np.testing.assert_array_equal(actual, expected, err_msg=f"origin {origin}")

    altered = EpisodeBuilder(perturb_after(panel, origins[0], keep_known=True)).build(request)
    np.testing.assert_array_equal(altered.xh[0, known], episode.xh[0, known])


def test_a_change_on_the_origin_cell_counts_in_the_known_role():
    panel = _onset_panel()
    item = panel.items[0]
    local = 96
    request = EpisodeRequest(0, (item.start + local,), 13, 128, derive=frozenset({"onset_rows"}))
    builder = EpisodeBuilder(panel)
    episode = builder.build(request)
    assert episode.row_names == ["y", "event", "event~onset"]
    index = episode.row_names.index("event~onset")
    assert episode.row_kind[index] == KIND_NUMBER
    assert episode.row_role[index] == ROLE_KNOWN
    assert episode.row_cand[index] == -1
    assert episode.row_keys[index] == ("derived", "event~onset")
    assert episode.row_score[index] == episode.row_score[episode.row_names.index("event")]
    (known,) = np.flatnonzero(episode.known_rows == index)
    actual = episode.xh[0, known].reshape(-1, N_CHANNELS)
    labels = item.dynamic["event"]
    target = to_scaled(item.targets["y"], fit_scale(item.targets["y"], local, panel.season))
    values, observed = derived.onset_profile(
        labels,
        target,
        left=local,
        right=local + request.horizon,
        frozen_at=local,
        season=panel.season,
    )
    expected = np.zeros((16, N_CHANNELS), dtype=np.float32)
    expected[: request.horizon, CH_VALUE] = values
    expected[: request.horizon, CH_OBSERVED] = observed
    assert observed[0] and values[0] < 0 and values[8] > 0  # Ages reset at the two changes.
    np.testing.assert_array_equal(actual, expected)

    values, observed = derived.onset_profile(
        labels, target, left=0, right=local, frozen_at=None, season=panel.season
    )
    series = np.where(observed, values, np.nan)
    steps, flags = step_features(
        series, series, season=panel.season, long_lag=panel.long_lag, left=0, right=local
    )
    expected = np.zeros((request.context, N_CHANNELS), dtype=np.float32)
    expected[-local:, CH_VALUE] = values
    expected[-local:, CH_OBSERVED] = observed
    expected[-local:, CH_STEP:CH_STEP_FLAG] = np.where(observed[:, None], steps, 0)
    expected[-local:, CH_STEP_FLAG:CH_VALUE_STATS] = observed[:, None] & flags
    np.testing.assert_array_equal(episode.x[index].reshape(-1, N_CHANNELS), expected)


@pytest.mark.parametrize("role", ["known", "past"])
def test_only_hidden_targets_varied_nothing_visible_moves(role):
    panel = _onset_panel(role)
    item = panel.items[0]
    origins = (item.start + 96, item.start + 128)
    request = EpisodeRequest(0, origins, 12, 64, derive=frozenset({"onset_rows"}))
    hidden = perturb_after(panel, origins[0], keep_known=True)
    hidden = replace(hidden, items=(replace(hidden.items[0], dynamic=item.dynamic),))
    original = EpisodeBuilder(panel).build(request)
    changed = EpisodeBuilder(hidden).build(request)
    index = original.row_names.index("event~onset")
    assert original.row_role[index] == (ROLE_KNOWN if role == "known" else ROLE_PAST)
    assert (index in original.known_rows) == (role == "known")
    _equal_exact(original, changed, exclude=("x", "xh", "labels"))
    prefix = original.origin_patches[0]
    np.testing.assert_array_equal(original.x[:, :prefix], changed.x[:, :prefix])
    np.testing.assert_array_equal(original.xh[0], changed.xh[0])
    assert not np.array_equal(original.x[index, prefix:], changed.x[index, prefix:])
    if role == "known":
        assert not np.array_equal(original.xh[1], changed.xh[1])
    assert not np.array_equal(original.labels, changed.labels)

    model = v5.V5Model(config().model)
    before = v5.forecast(model, panel, request, pruned=False)
    after = v5.forecast(model, hidden, request, pruned=False)
    assert before.gates == after.gates
    np.testing.assert_array_equal(before.quantiles[0], after.quantiles[0])

    with torch.no_grad():
        model.gate_net.net[-1].weight.zero_()
        model.gate_net.net[-1].bias.fill_(-1000)
    evidence = v5.forecast_items(
        model, panel, [(0, origins[0])], 12, context=64, derive=request.derive, pruned=True
    ).evidence
    base = evidence.candidates.index("dynamic:event")
    onset = evidence.candidates.index("derived:event~onset")
    expected_gate = np.float32(model.gate_net.known_gate_floor if role == "known" else 0)
    expected_status = 1 if role == "known" else 5
    np.testing.assert_array_equal(evidence.status[0, [base, onset]], [expected_status] * 2)
    np.testing.assert_array_equal(evidence.gate[0, [base, onset]], [expected_gate] * 2)


def test_no_row_for_a_flag_or_when_value_stats_is_dropped():
    panel = _onset_panel()
    item = panel.items[0]
    flag = np.where(item.dynamic["event"] == "a", "a", "b").astype(object)
    panel = replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, "flag": flag}),),
        roles={**panel.roles, "flag": "known"},
    )
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (115,), 12, 64, derive=frozenset({"onset_rows"}))
    episode = builder.build(request)
    assert episode.row_kind[episode.row_names.index("flag")] == KIND_FLAG
    assert "event~onset" in episode.row_names
    assert "flag~onset" not in episode.row_names
    request = replace(request, drop=frozenset({"value_stats"}))
    dropped = builder.build(request)
    assert "event~effect" in dropped.row_names
    _equal_exact(dropped, builder.build(replace(request, derive=frozenset())))


@pytest.mark.parametrize(
    "case", ["constant", "single_change", "support_after_cutoff", "support_before_context"]
)
def test_no_onset_row_without_support_in_context_before_cutoff(case):
    panel = _onset_panel(run_lengths=(2, 3, 4))
    item = panel.items[0]
    labels = item.dynamic["event"].copy()
    if case == "constant":
        labels[:] = "a"
    elif case == "single_change":
        # The isolated third label keeps the column categorical, with one adjacent change.
        labels[:2] = ["c", None]
        labels[2:48], labels[48:] = "a", "b"
    elif case == "support_after_cutoff":
        labels[:64] = "a"
    else:
        labels[32], labels[33:] = None, "a"
    panel = replace(panel, items=(replace(item, dynamic={"event": labels}),))
    request = EpisodeRequest(0, (item.start + 128,), 12, 96, cutoff=item.start + 64)
    builder = EpisodeBuilder(panel)
    enabled = builder.build(replace(request, derive=frozenset({"onset_rows"})))
    assert enabled.row_kind[enabled.row_names.index("event")] == KIND_CATEGORY
    assert "event~onset" not in enabled.row_names
    assert all(candidate.source != "derived" for candidate in enabled.trace)
    _equal_exact(builder.build(request), enabled)


def test_an_unsupported_onset_leaves_room_for_a_lower_scored_lag(monkeypatch):
    panel = _onset_panel(run_lengths=(2, 3, 4))
    item = panel.items[0]
    labels = item.dynamic["event"].copy()
    labels[32], labels[33:] = None, "a"
    driver = lag_panel(length=len(labels)).items[0].dynamic["driver"]
    panel = replace(
        panel,
        items=(replace(item, dynamic={"event": labels, "driver": driver}),),
        roles={**panel.roles, "driver": "past"},
    )
    panel = _fill_legacy_slots(panel, MAX_FEATURE_ROWS - 3)
    builder = EpisodeBuilder(panel, ScreeningConfig(no_evidence=0.0))
    request = EpisodeRequest(0, (item.start + 128,), 12, 96, cutoff=item.start + 64)
    baseline = builder.build(request)
    assert len(baseline.row_names) == MAX_FEATURE_ROWS
    score = next(c.score for c in baseline.trace if c.name == "event")
    assert score > 0
    monkeypatch.setattr(derived, "scan_delay", lambda *a, **k: (5, score / 2, 0.0))

    enabled = builder.build(replace(request, derive=frozenset({"onset_rows", "lag_rows"})))
    assert set(enabled.row_names) == {*baseline.row_names, "driver~lag"}
    assert [(c.name, c.removed) for c in enabled.trace if c.source == "derived"] == [
        ("driver~lag", None)
    ]


def test_summary_obeys_the_cutoff():
    panel = _onset_panel()
    item = panel.items[0]
    end = 83
    request = EpisodeRequest(
        0, (115, 131), 12, 128, cutoff=item.start + end, derive=frozenset({"onset_rows"})
    )
    original = EpisodeBuilder(panel).build(request)
    index = original.row_names.index("event~onset")
    target = to_scaled(item.targets["y"], fit_scale(item.targets["y"], end, panel.season))
    values, observed = derived.onset_profile(
        item.dynamic["event"], target, left=0, right=end, frozen_at=None, season=panel.season
    )
    series = np.where(observed, values, np.nan)[None]
    expected, flags = summary_vectors(
        series, series, season=panel.season, long_lag=panel.long_lag, end=request.cutoff
    )
    np.testing.assert_array_equal(original.row_summary[index], expected[0])
    np.testing.assert_array_equal(original.row_summary_flags[index], flags[0])

    altered = EpisodeBuilder(perturb_after(panel, request.cutoff, keep_known=True)).build(request)
    np.testing.assert_array_equal(original.row_summary[index], altered.row_summary[index])
    np.testing.assert_array_equal(
        original.row_summary_flags[index], altered.row_summary_flags[index]
    )
    assert not np.array_equal(original.x[index], altered.x[index])


@pytest.mark.parametrize("lag_margin", [-0.01, 0, 0.01])
def test_onset_and_lag_compete_for_the_same_free_slot(monkeypatch, lag_margin):
    panel = _onset_panel()
    item = panel.items[0]
    driver = lag_panel(length=len(item.targets["y"])).items[0].dynamic["driver"]
    panel = replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, "driver": driver}),),
        roles={**panel.roles, "driver": "past"},
    )
    panel = _fill_legacy_slots(panel, MAX_FEATURE_ROWS - 3)
    builder = EpisodeBuilder(panel, ScreeningConfig(no_evidence=0.0))
    request = EpisodeRequest(0, (115,), 12, 64)
    legacy = builder.build(request)
    assert len(legacy.row_names) == MAX_FEATURE_ROWS
    score = next(c.score for c in legacy.trace if c.name == "event")
    monkeypatch.setattr(derived, "scan_delay", lambda *a, **k: (5, score + lag_margin, 0.0))
    enabled = replace(request, derive=frozenset({"onset_rows", "lag_rows"}))
    episode = builder.build(enabled)
    names = ["driver~lag", "event~onset"]
    if lag_margin < 0:
        names.reverse()
    assert set(episode.row_names) == {*legacy.row_names, names[0]}
    assert [(c.name, c.removed) for c in episode.trace[len(legacy.trace) :]] == [
        (names[0], None),
        (names[1], "budget"),
    ]
    for i, name in enumerate(legacy.row_names):
        np.testing.assert_array_equal(episode.x[episode.row_names.index(name)], legacy.x[i])
    (planned,) = builder.plan_requests(0, (115,), 12, context=64, derive=enabled.derive)
    _equal_exact(builder.build(planned), episode)
    for restricted in (
        replace(enabled, drop=frozenset({"features"})),
        replace(enabled, keep_columns=frozenset({"driver"})),
    ):
        assert "event~onset" not in builder.build(restricted).row_names

    full = EpisodeBuilder(_fill_legacy_slots(panel, MAX_FEATURE_ROWS - 2))
    capped = full.build(request)
    assert len(capped.row_names) == MAX_FEATURE_ROWS + 1
    _equal_exact(capped, full.build(enabled))


def test_onset_name_collision_builds_nothing():
    panel = _onset_panel()
    item = panel.items[0]
    request = EpisodeRequest(0, (115,), 12, 64, derive=frozenset({"onset_rows"}))
    assert "event~onset" in EpisodeBuilder(panel).build(request).row_names
    panel = replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, "event~onset": np.zeros(192)}),),
        roles={**panel.roles, "event~onset": "past"},
    )
    actual = EpisodeBuilder(panel).build(request)
    assert "event~onset" not in actual.row_names
    assert all(c.source != "derived" for c in actual.trace)


_NAIVE = frozenset({"naive_rows"})
_NAIVE_HOURLY_NAMES = {"y~last", "y~season", "y~season2", "y~drift"}


def _naive_panel(freq="h", *, start=71, target=None):
    steps = np.arange(420, dtype=np.float64)
    if target is None:
        target = steps / 23 + np.sin(steps / 5)
    return Panel("naive", (Item("one", start, {"y": target}, {}, {}),), {}, freq, None, False)


def _naive_horizon(episode, name, origin=0):
    known = episode.known_rows.tolist().index(episode.row_names.index(name))
    return episode.xh[origin, known].reshape(-1, N_CHANNELS)[: episode.horizon]


@pytest.mark.parametrize("origin_index", [0, 1])
@pytest.mark.parametrize("replacement", [1e6, np.nan])
def test_naive_horizon_reads_nothing_at_or_after_its_origin(origin_index, replacement):
    panel = _naive_panel()
    item = panel.items[0]
    origins = tuple(item.start + local for local in (192, 224))
    request = EpisodeRequest(0, origins, 45, 64, cutoff=origins[0] - 16, derive=_NAIVE)
    changed = item.targets["y"].copy()
    changed[origins[origin_index] - item.start :] = replacement
    hidden = replace(panel, items=(replace(item, targets={"y": changed}),))
    original = EpisodeBuilder(panel).build(request)
    altered = EpisodeBuilder(hidden).build(request)

    assert set(original.row_names[1:]) == _NAIVE_HOURLY_NAMES
    _equal_exact(original, altered, exclude=("x", "xh", "labels", "label_mask"))
    prefix = original.origin_patches[origin_index]
    np.testing.assert_array_equal(original.x[:, :prefix], altered.x[:, :prefix])
    np.testing.assert_array_equal(original.xh[origin_index], altered.xh[origin_index])
    assert not np.array_equal(original.labels[origin_index], altered.labels[origin_index])
    if np.isnan(replacement):
        assert not altered.label_mask[origin_index].any()

    model = v5.V5Model(config().model)
    before = v5.forecast(model, panel, request)
    after = v5.forecast(model, hidden, request)
    assert before.names == after.names
    assert before.gates == pytest.approx(after.gates, abs=1e-6)
    np.testing.assert_allclose(
        before.quantiles[origin_index], after.quantiles[origin_index], atol=1e-6, rtol=1e-6
    )


def test_new_history_before_an_origin_is_used():
    panel = _naive_panel()
    item = panel.items[0]
    request = EpisodeRequest(0, (263, 295), 45, 64, cutoff=247, derive=_NAIVE)
    values = item.targets["y"].copy()
    values[request.origins[0] - item.start : request.origins[1] - item.start] += 20
    changed = replace(panel, items=(replace(item, targets={"y": values}),))
    original = EpisodeBuilder(panel).build(request)
    altered = EpisodeBuilder(changed).build(request)
    _equal_exact(original, altered, exclude=("x", "xh", "labels"))
    np.testing.assert_array_equal(original.xh[0], altered.xh[0])
    patch = original.origin_patches[0]
    row = original.row_names.index("y~last")
    # The changed target at the grid cell can first serve the following context cell.
    np.testing.assert_array_equal(original.x[row, patch, 0], altered.x[row, patch, 0])
    assert original.x[row, patch, 1, CH_VALUE] != altered.x[row, patch, 1, CH_VALUE]
    for name in ("y~last", "y~season", "y~drift"):
        assert not np.allclose(
            _naive_horizon(original, name, 1)[:, CH_VALUE],
            _naive_horizon(altered, name, 1)[:, CH_VALUE],
        )


@pytest.mark.parametrize(
    ("freq", "name", "season", "slope"),
    [
        ("D", "season", 7, 0),
        ("h", "season", 24, 0),
        (None, "drift", 1, 0.2),
        ("D", "drift", 7, 0.2),
        ("h", "drift", 24, 0.2),
        ("h", "season2", 168, 0),
    ],
)
def test_naive_rows_recover_planted_values(freq, name, season, slope):
    steps = np.arange(420, dtype=np.float64)
    raw = 50 + np.sin(2 * np.pi * (steps % season) / season) + slope * steps
    origin = 192
    if name == "drift":
        drift_width = season * ((64 + season - 1) // season)
        # A level shift outside both endpoints of the slope window cannot enter the fit.
        raw[: origin - drift_width - season] += 20
    panel = _naive_panel(freq, target=raw)
    item = panel.items[0]
    request = EpisodeRequest(0, (item.start + origin,), 45, 16, derive=_NAIVE)
    episode = EpisodeBuilder(panel).build(request)
    horizon = _naive_horizon(episode, f"y~{name}")
    assert horizon[:, CH_OBSERVED].all()
    np.testing.assert_allclose(
        np.sinh(horizon[:, CH_VALUE]), episode.labels[0, 0, : request.horizon], atol=1e-6, rtol=1e-6
    )
    if freq is None:
        assert set(episode.row_names[1:]) == {"y~last", "y~drift"}
    if freq == "D":
        assert set(episode.row_names[1:]) == {"y~last", "y~season", "y~drift"}


def test_drift_row_survives_a_short_history():
    origin = 14  # Seven daily seasonal pairs, the minimum in the 70-cell window.
    panel = _naive_panel("D", target=np.arange(30, dtype=np.float64))
    item = panel.items[0]
    request = EpisodeRequest(0, (item.start + origin,), 8, 16, derive=_NAIVE)
    episode = EpisodeBuilder(panel).build(request)
    horizon = _naive_horizon(episode, "y~drift")

    assert horizon[:, CH_OBSERVED].all()
    np.testing.assert_allclose(
        np.sinh(horizon[:, CH_VALUE]), episode.labels[0, 0, : request.horizon], atol=1e-6, rtol=1e-6
    )


def test_naive_row_layout_uses_the_first_origin_even_with_missing_history():
    panel = _naive_panel(target=np.full(420, np.nan))
    item = panel.items[0]
    early = EpisodeRequest(0, (item.start + 167, item.start + 183), 17, 192, derive=_NAIVE)
    builder = EpisodeBuilder(panel)
    actual = builder.build(early)
    assert set(actual.row_names[1:]) == _NAIVE_HOURLY_NAMES - {"y~season2"}
    assert not actual.x[1:].any()
    assert not actual.xh.any()
    enough = builder.build(replace(early, origins=(item.start + 168,)))
    assert set(enough.row_names[1:]) == _NAIVE_HOURLY_NAMES
    assert not enough.xh.any()


def test_naive_rows_are_reserved_under_the_cap():
    panel = _fill_legacy_slots(_naive_panel(), 20)
    item = panel.items[0]
    roles = {name: "past" if int(name[5:]) % 3 == 0 else "known" for name in panel.roles}
    builder = EpisodeBuilder(replace(panel, roles=roles))
    request = EpisodeRequest(0, (item.start + 192,), 21, 64)
    baseline = builder.build(request)
    enabled = builder.build(replace(request, derive=_NAIVE | {"lag_rows", "onset_rows"}))

    by_rank = sorted(
        (c for c in baseline.trace if c.removed is None),
        key=lambda c: (roles[c.name] != "known", -c.score, c.name),
    )
    assert len(by_rank) == MAX_FEATURE_ROWS
    preserved, displaced = by_rank[:-4], by_rank[-4:]
    assert enabled.row_names == [
        "y",
        *sorted(_NAIVE_HOURLY_NAMES | {c.name for c in preserved if roles[c.name] == "known"}),
        *sorted(c.name for c in preserved if roles[c.name] == "past"),
    ]
    trace = {(c.source, c.name): c for c in enabled.trace}
    for candidate in preserved:
        assert trace[candidate.source, candidate.name] == candidate
        np.testing.assert_array_equal(
            enabled.x[enabled.row_names.index(candidate.name)],
            baseline.x[baseline.row_names.index(candidate.name)],
        )
    for candidate in displaced:
        assert trace[candidate.source, candidate.name] == replace(candidate, removed="budget")
    for name in _NAIVE_HOURLY_NAMES:
        c = trace["derived", name]
        assert (c.kind, c.score, c.removed) == (KIND_NUMBER, 1.0, None)
        row = enabled.row_names.index(name)
        assert enabled.row_keys[row] == ("derived", name)
        assert (enabled.row_kind[row], enabled.row_role[row], enabled.row_score[row]) == (
            KIND_NUMBER,
            ROLE_KNOWN,
            1.0,
        )
    _equal_exact(baseline, builder.build(replace(request, derive=frozenset())))


def test_naive_rows_leave_free_slots_for_existing_derived_rows(monkeypatch):
    panel = _onset_panel()
    item = panel.items[0]
    driver = lag_panel(length=len(item.targets["y"])).items[0].dynamic["driver"]
    panel = replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, "driver": driver}),),
        roles={**panel.roles, "driver": "past"},
    )
    panel = _fill_legacy_slots(panel, MAX_FEATURE_ROWS - 6)
    request = EpisodeRequest(0, (115,), 12, 64, derive=_NAIVE | {"onset_rows", "lag_rows"})
    monkeypatch.setattr(derived, "scan_delay", lambda *a, **k: (5, 1.0, 0.0))
    episode = EpisodeBuilder(panel).build(request)
    expected = {
        "y~last": None,
        "y~season": None,
        "y~drift": None,
        "driver~lag": None,
        "event~onset": "budget",
    }
    assert {c.name: c.removed for c in episode.trace if c.source == "derived"} == expected
    assert {name for name in episode.row_names if "~" in name} == {
        name for name, reason in expected.items() if reason is None
    }


def test_naive_row_name_collision_does_not_reserve_a_slot():
    panel = _fill_legacy_slots(_naive_panel(), 20)
    item = panel.items[0]
    panel = replace(
        panel,
        items=(replace(item, dynamic={**item.dynamic, "y~last": np.full(420, np.nan)}),),
        roles={**panel.roles, "y~last": "past"},
    )
    actual = EpisodeBuilder(panel).build(EpisodeRequest(0, (263,), 12, 64, derive=_NAIVE))
    assert "y~last" not in actual.row_names
    assert len(actual.row_names) == MAX_FEATURE_ROWS + 1
    assert sum(c.removed is None and c.source == "known" for c in actual.trace) == 13


def test_naive_rows_use_only_the_kept_forecast_target():
    panel = _naive_panel()
    item = panel.items[0]
    targets = {**item.targets, "actual": item.targets["y"] ** 2}
    builder = EpisodeBuilder(replace(panel, items=(replace(item, targets=targets),)))
    (joint,) = builder.plan_requests(0, (263,), 21, context=64, derive=_NAIVE)
    assert joint.target is None
    _equal_exact(builder.build(joint), builder.build(replace(joint, derive=frozenset())))
    episode = builder.build(replace(joint, target="actual"))
    assert episode.keep.sum() == 1
    assert {name for name in episode.row_names if "~" in name} == {
        name.replace("y~", "actual~") for name in _NAIVE_HOURLY_NAMES
    }
    scaled = to_scaled(targets["actual"], fit_scale(targets["actual"], 192, panel.season))
    np.testing.assert_allclose(
        _naive_horizon(episode, "actual~last")[:, CH_VALUE], np.arcsinh(scaled[191]), atol=1e-6
    )


def test_builder_rejects_unknown_derive_names_with_naive_anchor():
    t = np.arange(64, dtype=float)
    panel = Panel("anchor", (Item("one", 0, {"y": t}, {}, {}),), {}, None, None, False)
    builder = EpisodeBuilder(panel)
    request = EpisodeRequest(0, (32,), 5, context=32, derive=frozenset({"naive_anchor"}))
    assert builder.build(request).naive_seen[..., :5, 1].all()
    with pytest.raises(ValueError, match="unknown derive switch"):
        builder.plan_requests(0, (32,), 5, context=32, derive=frozenset({"naive_anchors"}))
