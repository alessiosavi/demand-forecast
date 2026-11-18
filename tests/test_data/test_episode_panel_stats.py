"""Causal typed-context statistics, leave-one-out groups and bounded memory."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from demand_forecast.data.episode import Item, Panel
from demand_forecast.data.episode import panel_stats as ps
from demand_forecast.data.episode.bank import summary_vectors
from demand_forecast.data.episode.calendar import calendar_arrays
from demand_forecast.data.episode.kinds import is_missing, label_of
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.scale import Scale, fit_scale, to_scaled
from demand_forecast.data.episode.schema import BANK_WINDOW, N_SUMMARY, N_VALUE_TOKEN
from demand_forecast.data.episode.value_stats import category_step_stats
from tests.helpers.episodes import perturb_after, toy_panel


def test_summary_chunks_preserve_summaries_and_flags(monkeypatch):
    panel = toy_panel(n_items=8)
    default = PanelStats(panel, "y", 300)
    calls = []

    def checked_summary(scaled, raw, *, season, long_lag, end):
        calls.append(raw.shape)
        assert raw.shape == scaled.shape and raw.shape[0] <= 3
        return summary_vectors(scaled, raw, season=season, long_lag=long_lag, end=end)

    monkeypatch.setattr(PanelStats, "SUMMARY_CHUNK", 3)
    monkeypatch.setattr(ps, "summary_vectors", checked_summary)
    chunked = PanelStats(panel, "y", 300)
    np.testing.assert_array_equal(chunked.summaries, default.summaries)
    np.testing.assert_array_equal(chunked.summary_flags, default.summary_flags)
    assert calls == [(3, 300), (3, 291), (2, 282)]


def small_panel(values, *, starts=None, statics=None):
    """Ragged raw targets without unrelated dynamic columns."""
    if starts is None:
        starts = [0] * len(values)
    if statics is None:
        statics = [{"brand": "shared"} for _ in values]
    return Panel(
        "small",
        tuple(
            Item(str(i), start, {"y": np.asarray(y, dtype=np.float64)}, {}, static)
            for i, (y, start, static) in enumerate(zip(values, starts, statics))
        ),
        {},
        None,
        None,
        True,
    )


def test_leave_one_out():
    p = toy_panel(n_items=8)
    s = PanelStats(p, "y", 300)
    q = 2
    lab = s.labels("brand", q)[0]
    rest = Panel(
        p.name,
        tuple(i for k, i in enumerate(p.items) if k != q),
        p.roles,
        p.freq,
        p.epoch,
        p.aligned,
    )
    r = PanelStats(rest, "y", 300)
    assert np.allclose(
        s.value_token("brand", lab, q)[0], r.value_token("brand", lab, -1)[0], atol=1e-6
    )
    assert s.value_token("brand", lab, q)[2] == r.value_token("brand", lab, -1)[2]
    assert np.allclose(
        s.peer_row("brand", lab, q, 200, 300)[0],
        r.peer_row("brand", lab, -1, 200, 300)[0],
        atol=1e-6,
    )


def test_nothing_after_the_cutoff_is_read():
    p = toy_panel()
    s = PanelStats(p, "y", 300)
    p2 = perturb_after(p, 300, value=1e6)
    s2 = PanelStats(p2, "y", 300)
    assert s.scales == s2.scales and np.array_equal(s.summaries, s2.summaries)
    np.testing.assert_array_equal(s.summary_flags, s2.summary_flags)
    assert np.array_equal(
        s.peer_row("brand", s.labels("brand", 0)[0], 0, 250, 300)[0],
        s2.peer_row("brand", s2.labels("brand", 0)[0], 0, 250, 300)[0],
    )
    for attribute in s.static_kinds:
        assert s.labels(attribute, 0) == s2.labels(attribute, 0)
        for label in s.labels(attribute, 0):
            a, flags, support = s.value_token(attribute, label, 0)
            b, other_flags, other_support = s2.value_token(attribute, label, 0)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(flags, other_flags)
            assert support == other_support


def category_reference(labels, target, left, right, frozen_at=None):
    """Deliberately scalar reference: only completed, observed responses enter."""
    values = np.zeros((right - left, 6), dtype=np.float32)
    flags = np.zeros(right - left, dtype=bool)
    for row, t in enumerate(range(left, right)):
        if not 0 <= t < len(labels) or is_missing(labels[t]):
            continue
        bound = t if frozen_at is None else min(t, frozen_at)
        responses = []
        for offset in range(4):
            responses.append(
                [
                    target[u + offset]
                    for u in range(len(labels))
                    if not is_missing(labels[u])
                    and label_of(labels[u]) == label_of(labels[t])
                    and u + offset < bound
                    and u + offset < len(target)
                    and np.isfinite(target[u + offset])
                ]
            )
        n = len(responses[0])
        if not n:
            continue
        shrink = n / (n + 4)
        values[row, 0] = np.mean(responses[0]) * shrink
        values[row, 1] = np.std(responses[0])
        for offset in range(1, 4):
            if responses[offset]:
                values[row, offset + 1] = np.mean(responses[offset]) * shrink
        values[row, 5] = np.log1p(n)
        flags[row] = True
    return values, flags


def test_category_step_stats_bounds():
    labels = np.array(["a", "a", "b", None, "a", "new", "a", "b", "a"], dtype=object)
    target = np.array([2.0, np.nan, 6.0, 9.0, 10.0, 12.0, 14.0, -2.0, 1.0])
    baseline, flags = category_step_stats(labels, target, left=0, right=len(labels))
    for t in range(len(labels)):
        changed = target.copy()
        changed[t:] = 1e6
        actual, actual_flags = category_step_stats(labels, changed, left=0, right=len(labels))
        np.testing.assert_array_equal(actual[: t + 1], baseline[: t + 1])
        np.testing.assert_array_equal(actual_flags[: t + 1], flags[: t + 1])
    frozen, frozen_flags = category_step_stats(labels, target, left=0, right=9, frozen_at=4)
    changed = target.copy()
    changed[4:] = -1e6
    actual, actual_flags = category_step_stats(labels, changed, left=0, right=9, frozen_at=4)
    np.testing.assert_array_equal(actual, frozen)
    np.testing.assert_array_equal(actual_flags, frozen_flags)
    assert not flags[[0, 2, 3, 5]].any()
    assert not baseline[[0, 2, 3, 5]].any()


@pytest.mark.parametrize("frozen_at", [None, -1, 0, 5, 100])
def test_category_step_stats_against_a_loop_reference(frozen_at):
    labels = np.empty(12, dtype=object)
    labels[:] = [3, "3", True, None, "True", np.nan, "", [], 3, "new", True, "3"]
    target = np.array([2, 4, np.nan, 8, 10, -2, 3, 6, 1, 4, 5, 7], dtype=np.float64)
    actual, flags = category_step_stats(labels, target, left=-2, right=15, frozen_at=frozen_at)
    expected, expected_flags = category_reference(labels, target, -2, 15, frozen_at)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_array_equal(flags, expected_flags)
    assert actual.dtype == np.float32 and flags.dtype == bool and np.isfinite(actual).all()


def test_category_response_profile_uses_completed_responses_and_base_support():
    values, flags = category_step_stats(
        np.array(["a"] * 5, dtype=object), np.array([1, 3, 5, 7, 9]), left=4, right=5
    )
    np.testing.assert_allclose(values[0], [2, np.sqrt(5), 2.5, 3, 3.5, np.log(5)], atol=1e-6)
    assert flags[0]


def test_category_spread_preserves_small_variation_at_large_level():
    values, flags = category_step_stats(
        np.array(["a"] * 4, dtype=object), 1e8 + np.arange(4), left=0, right=4
    )
    assert flags[-1]
    assert values[-1, 1] == pytest.approx(np.sqrt(2 / 3), abs=1e-6)


def test_category_large_responses_are_finite_without_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values, flags = category_step_stats(
            np.array(["a"] * 4, dtype=object),
            np.array([1e155, 1.1e155, 1.2e155, 1.3e155]),
            left=0,
            right=4,
        )
    assert values.dtype == np.float32 and np.isfinite(values).all()
    np.testing.assert_array_equal(flags, [False, True, True, True])


@pytest.mark.parametrize("item_index", [-1, 2])
def test_item_accessors_reject_out_of_range_indices(item_index):
    stats = PanelStats(small_panel([[1, 2], [3, 4]]), "y", 2)
    for attribute in ("brand", "unknown"):
        with pytest.raises(IndexError):
            stats.labels(attribute, item_index)
    with pytest.raises(IndexError):
        stats.static_numbers(item_index)


def test_category_stats_slices_and_label_renaming():
    labels = np.array(["a", "b", "a", "c", "b", "a", "a", "b"], dtype=object)
    target = np.arange(8, dtype=np.float64)
    full, full_flags = category_step_stats(labels, target, left=0, right=8)
    renamed = np.array([{"a": "z", "b": "x", "c": "q"}[x] for x in labels], dtype=object)
    sliced, flags = category_step_stats(renamed, target, left=3, right=7)
    np.testing.assert_array_equal(sliced, full[3:7])
    np.testing.assert_array_equal(flags, full_flags[3:7])


def peer_reference(panel, scales, members, exclude, left, right):
    """Explicit loop over grid dates and peer members, independent of caches."""
    values = np.zeros(right - left, dtype=np.float32)
    support = np.zeros(right - left, dtype=np.int64)
    for t in range(left, right):
        observed = []
        for i in members:
            item = panel.items[i]
            y = item.targets.get("y")
            if i != exclude and y is not None and 0 <= t - item.start < len(y):
                raw = y[t - item.start]
                if np.isfinite(raw):
                    observed.append(np.arcsinh((raw - scales[i].loc) / scales[i].scale))
        support[t - left] = len(observed)
        if observed:
            values[t - left] = np.mean(observed)
    return values, support


def test_peer_row_against_a_loop_reference():
    clean = small_panel([np.arange(10), np.arange(8) + 10, np.arange(6) + 20], starts=[0, 2, 4])
    stats = PanelStats(clean, "y", 10)
    _, support = stats.peer_row("brand", "shared", 0, 0, 10)
    np.testing.assert_array_equal(support, [0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    clean.items[1].targets["y"][1] = np.nan
    clean.items[2].targets["y"][2] = np.nan
    stats = PanelStats(clean, "y", 8)
    # Grow and revisit the cache in both directions; peers can be read past the statistics cutoff.
    for left, right in ((2, 7), (-3, 10), (0, 10), (4, 6), (12, 15), (0, 0)):
        for exclude in (-1, 0, 1, 2):
            actual, support = stats.peer_row("brand", "shared", exclude, left, right)
            expected, counts = peer_reference(clean, stats.scales, range(3), exclude, left, right)
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            np.testing.assert_array_equal(support, counts)
            assert actual.dtype == np.float32 and np.isfinite(actual).all()
            whole, whole_counts = stats.panel_row(exclude, left, right)
            np.testing.assert_allclose(whole, expected, atol=1e-6)
            np.testing.assert_array_equal(whole_counts, counts)


@pytest.mark.parametrize("n_items", [1, 4])
def test_group_of_one_is_empty(n_items):
    panel = toy_panel(n_items=n_items)
    panel = replace(
        panel,
        items=tuple(
            replace(item, static={"brand": f"unique{i}"}) for i, item in enumerate(panel.items)
        ),
    )
    stats = PanelStats(panel, "y", 300)
    token, flags, support = stats.value_token("brand", "unique0", 0)
    assert support == 0 and not token.any() and not flags.any()
    values, support = stats.peer_row("brand", "unique0", 0, 200, 300)
    assert not values.any() and not support.any()
    if n_items == 1:
        values, support = stats.panel_row(0, 200, 300)
        assert not values.any() and not support.any()


def test_renaming_labels():
    panel = toy_panel(n_items=8)
    renamed = replace(
        panel,
        items=tuple(
            replace(item, static={**item.static, "brand": f"renamed_{item.static['brand']}"})
            for item in panel.items
        ),
    )
    before, after = PanelStats(panel, "y", 300), PanelStats(renamed, "y", 300)
    for q in range(len(panel.items)):
        label = before.labels("brand", q)[0]
        new_label = after.labels("brand", q)[0]
        a, flags, support = before.value_token("brand", label, q)
        b, new_flags, new_support = after.value_token("brand", new_label, q)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(flags, new_flags)
        assert support == new_support
        for old, new in zip(
            before.peer_row("brand", label, q, 200, 300),
            after.peer_row("brand", new_label, q, 200, 300),
        ):
            np.testing.assert_array_equal(old, new)


def test_mixed_label_types():
    labels = [3, "3", True, None, float("nan"), "", [], [True, 3, "3", None, ""]]
    panel = small_panel([np.arange(10)] * 8, statics=[{"dept": x} for x in labels])
    stats = PanelStats(panel, "y", 8)
    np.testing.assert_array_equal(stats.members("dept", "3"), [0, 1, 7])
    np.testing.assert_array_equal(stats.members("dept", "True"), [2, 7])
    assert stats.labels("dept", 0) == stats.labels("dept", 1) == ("3",)
    assert stats.labels("dept", 2) == ("True",)
    assert stats.labels("dept", 7) == ("3", "True")
    for i in (3, 4, 5, 6):
        assert stats.labels("dept", i) == ()
    assert stats.members("dept", "None").size == 0
    assert stats.value_token("dept", "3", 0)[2] == 2


@pytest.mark.parametrize("freq", [None, "h"])
def test_unaligned_panel_has_no_peers(freq):
    panel = replace(toy_panel(), epoch=None, freq=freq, aligned=False)
    panel.validate()
    stats = PanelStats(panel, "y", 300)
    label = stats.labels("brand", 0)[0]
    assert stats.value_token("brand", label, 0)[2] > 0
    for values, support in (
        stats.peer_row("brand", label, 0, 200, 300),
        stats.panel_row(0, 200, 300),
    ):
        assert not values.any() and not support.any()


@pytest.mark.parametrize("freq", [None, "h"])
def test_undated_aligned_panel_keeps_peers_without_calendar(freq):
    panel = replace(toy_panel(), epoch=None, freq=freq)
    panel.validate()
    stats = PanelStats(panel, "y", 300)
    assert calendar_arrays(panel, 200, 100) is None
    label = stats.labels("brand", 0)[0]
    assert stats.peer_row("brand", label, 0, 200, 300)[1].min() > 0
    assert stats.panel_row(0, 200, 300)[1].min() > 0


def test_value_token_uses_only_valid_summary_entries():
    panel = small_panel([np.arange(10), np.ones(10), np.full(10, np.nan), [1, 2, np.nan]])
    stats = PanelStats(panel, "y", 10)
    members = [1, 3]
    token, flags, support = stats.value_token("brand", "shared", 0)
    expected = np.zeros(N_VALUE_TOKEN)
    for j in range(N_SUMMARY):
        observed = [stats.summaries[i, j] for i in members if stats.summary_flags[i, j]]
        assert flags[j] == bool(observed)
        if observed:
            expected[j] = np.mean(observed)
            expected[N_SUMMARY + j] = np.std(observed)
    expected[-1] = np.log1p(2)
    np.testing.assert_allclose(token, expected, atol=1e-6)
    assert support == 2 and token.dtype == np.float32 and flags.dtype == bool
    empty, empty_flags, support = stats.value_token("unknown", "missing", -1)
    assert empty.shape == (57,) and empty_flags.shape == (28,)
    assert support == 0 and not empty.any() and not empty_flags.any()


def test_items_without_observed_target_before_cutoff_are_in_no_group():
    panel = small_panel([[1, 2, 3, 4, 5], [np.nan] * 4 + [9], [1, 2], [3, 4]], starts=[0, 0, 0, 4])
    panel = replace(
        panel, items=(*panel.items[:2], replace(panel.items[2], targets={}), panel.items[3])
    )
    stats = PanelStats(panel, "y", 4)
    np.testing.assert_array_equal(stats.members("brand", "shared"), [0])
    assert stats.scales[1:] == [Scale(0, 1)] * 3
    assert not stats.summaries[1:].any() and not stats.summary_flags[1:].any()
    assert stats.value_token("brand", "shared", -1)[2] == 1
    assert not stats.peer_row("brand", "shared", 0, 0, 6)[1].any()
    assert not stats.panel_row(0, 0, 6)[1].any()


def test_static_numbers_and_quantile_groups():
    values = np.arange(16, dtype=np.float64) + 0.25
    panel = small_panel(
        [np.arange(10)] * 17,
        statics=[{"size": x, "constant": 1.5} for x in [*values, np.nan]],
    )
    stats = PanelStats(panel, "y", 8)
    assert stats.static_kinds == {"constant": "number", "size": "number"}
    for i, raw in enumerate(values):
        names, numbers, flags = stats.static_numbers(i)
        assert names == ["constant", "size"]
        np.testing.assert_allclose(numbers, [0, (raw - 7.75) / np.sqrt(21.25)], atol=1e-6)
        assert flags.all() and numbers.dtype == np.float32 and flags.dtype == bool
        assert stats.labels("size", i) == (f"bin{i // 2}",)
    assert stats.labels("constant", 0) == ("bin0",)
    assert stats.labels("size", 16) == ()
    _, numbers, flags = stats.static_numbers(16)
    np.testing.assert_array_equal(numbers, [0, 0])
    np.testing.assert_array_equal(flags, [True, False])


def test_item_and_label_order_do_not_change_context():
    panel = toy_panel(n_items=8)
    order = [6, 3, 1, 7, 0, 2, 5, 4]
    reordered = replace(
        panel,
        items=tuple(
            replace(
                panel.items[i],
                item_id=f"new{i}",
                static={
                    **dict(reversed(list(panel.items[i].static.items()))),
                    "tags": list(reversed(panel.items[i].static["tags"])),
                },
            )
            for i in order
        ),
    )
    a, b = PanelStats(panel, "y", 300), PanelStats(reordered, "y", 300)
    np.testing.assert_array_equal(a.summaries[order], b.summaries)
    for new_q, old_q in enumerate(order):
        for attribute in a.static_kinds:
            assert a.labels(attribute, old_q) == b.labels(attribute, new_q)
            for label in a.labels(attribute, old_q):
                old, flags, support = a.value_token(attribute, label, old_q)
                new, new_flags, new_support = b.value_token(attribute, label, new_q)
                np.testing.assert_allclose(old, new, atol=1e-6)
                np.testing.assert_array_equal(flags, new_flags)
                assert support == new_support


def test_summaries_use_grid_cutoff_bank_window_and_own_scale(monkeypatch):
    panel = small_panel([np.arange(8300) + 0.5, np.arange(10) + 1.25], starts=[-5, 8200])
    panel = replace(panel, freq="h")
    cutoff = 8212
    monkeypatch.setattr(PanelStats, "SUMMARY_CHUNK", 1)
    shapes = []

    def checked_summary(scaled, raw, *, season, long_lag, end):
        shapes.append(raw.shape)
        assert raw.shape[1] <= BANK_WINDOW and season == 24 and long_lag == 168 and end == cutoff
        return summary_vectors(scaled, raw, season=season, long_lag=long_lag, end=end)

    monkeypatch.setattr(ps, "summary_vectors", checked_summary)
    stats = PanelStats(panel, "y", cutoff)
    for i, item in enumerate(panel.items):
        scale = fit_scale(item.targets["y"], cutoff - item.start, 24)
        assert stats.scales[i] == scale
        left = max(item.start, cutoff - BANK_WINDOW)
        raw = np.full(cutoff - left, np.nan)
        stop = min(cutoff, item.start + len(item.targets["y"]))
        raw[: stop - left] = item.targets["y"][left - item.start : stop - item.start]
        expected, flags = summary_vectors(
            to_scaled(raw, scale)[None], raw[None], season=24, long_lag=168, end=cutoff
        )
        np.testing.assert_array_equal(stats.summaries[i], expected[0])
        np.testing.assert_array_equal(stats.summary_flags[i], flags[0])
    assert shapes == [(1, BANK_WINDOW), (1, 12)]


def test_empty_panel_and_empty_cutoff_are_finite():
    empty = PanelStats(small_panel([]), "y", 0)
    assert empty.summaries.shape == (0, 28) and empty.scales == []
    with pytest.raises(IndexError):
        empty.static_numbers(0)
    with pytest.raises(IndexError):
        empty.labels("brand", 0)
    stats = PanelStats(small_panel([[1, 2], [3, 4]]), "y", -1)
    assert stats.scales == [Scale(0, 1), Scale(0, 1)]
    assert not stats.summaries.any() and not stats.summary_flags.any()
    assert stats.members("brand", "shared").size == 0
    assert not stats.panel_row(-1, 0, 2)[1].any()


def test_spread_at_a_large_level_keeps_small_variation():
    # Raw moments cancel at this level (float64 ulp of 1e16 is 2); the spread must still be sqrt(2/3).
    values, flags = category_step_stats(
        np.array(["a"] * 4, dtype=object), 1e8 + np.arange(4), left=0, right=4
    )
    assert flags[-1]
    assert values[-1, 1] == pytest.approx(np.sqrt(2 / 3), abs=1e-6)


def test_huge_responses_stay_finite():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values, flags = category_step_stats(
            np.array(["a"] * 4, dtype=object),
            np.array([1e155, 1.1e155, 1.2e155, 1.3e155]),
            left=0,
            right=4,
        )
    assert values.dtype == np.float32 and np.isfinite(values).all()
    np.testing.assert_array_equal(flags, [False, True, True, True])
