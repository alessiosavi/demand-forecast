"""Cutoff-only screening, label-independent relevance and expansion budgets."""

import gc
import weakref
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, Item, Panel, collate
from demand_forecast.data.episode import builder as episode_builder
from demand_forecast.data.episode import screening as sc
from demand_forecast.data.episode.panel_stats import PanelStats
from demand_forecast.data.episode.schema import KIND_FLAG, KIND_NUMBER
from demand_forecast.data.episode.screening import (
    Candidate,
    ScreeningConfig,
    adjusted_share,
    relevance_category,
    relevance_number,
    relevance_static_category,
    relevance_static_number,
    screen,
    select,
)
from demand_forecast.models.v5 import V5Config, V5Model
from tests.helpers.episodes import perturb_after, toy_panel


@pytest.fixture
def panel_with_candidate_budget(monkeypatch):
    steps = np.arange(96, dtype=np.float64)
    driver = np.sin(steps / 3)
    items = tuple(
        Item(
            f"item{i}",
            0,
            {"y": driver + i},
            {"driver": driver, "duplicate": driver.copy(), "empty": np.full_like(steps, np.nan)},
            {"driver": f"group{i % 2}", "id": f"item{i}", "z_cut": f"group{i // 4}"},
        )
        for i in range(8)
    )
    panel = Panel(
        "trace", items, {"driver": "past", "duplicate": "past", "empty": "past"}, None, None, False
    )
    request = EpisodeRequest(0, (64,), 5, context=64)
    config = ScreeningConfig(min_items=9)

    def select_one_attribute(candidates, **budgets):
        return select(candidates, **(budgets | {"attribute_budget": 1}))

    monkeypatch.setattr(episode_builder, "select", select_one_attribute)
    return panel, request, config


def test_trace_lists_every_candidate_with_its_disposition(panel_with_candidate_budget):
    panel, request, config = panel_with_candidate_budget
    episode = EpisodeBuilder(panel, config).build(request)
    assert isinstance(episode.trace, tuple)
    assert [(c.source, c.name, c.removed) for c in episode.trace] == [
        ("past", "driver", None),
        ("past", "duplicate", "duplicate"),
        ("past", "empty", "empty"),
        ("static", "driver", None),
        ("static", "id", "entity"),
        ("static", "z_cut", "budget"),
    ]
    assert episode.trace[0].score == pytest.approx(1)
    assert tuple(c.score for c in episode.trace[1:]) == (0,) * 5
    assert episode.row_names == ["y", "driver"]
    assert episode.cand_names == ["driver"]


def test_dropped_candidate_does_not_count_in_sparsity(panel_with_candidate_budget):
    panel, request, config = panel_with_candidate_budget
    episode = EpisodeBuilder(panel, config).build(request)
    cut = next(c for c in episode.trace if (c.source, c.name) == ("static", "z_cut"))
    assert cut.removed == "budget"
    assert len(episode.cand_names) == sum(
        c.removed is None and c.source == "static" for c in episode.trace
    )
    reference = replace(
        panel,
        items=tuple(
            replace(
                item,
                static={name: value for name, value in item.static.items() if name != cut.name},
            )
            for item in panel.items
        ),
    )
    batch = collate([episode])
    clean_batch = collate([EpisodeBuilder(reference, config).build(request)])
    assert batch.cand_mask.tolist() == [[True]]
    model = V5Model(
        V5Config(
            d_model=16,
            nhead=2,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=32,
            expert_enc_layers=0,
            expert_dec_layers=0,
        )
    ).eval()
    with torch.no_grad():
        # Distinguish family from candidate probabilities in the normalized L0.
        model.family_logits.bias.zero_()
        torch.testing.assert_close(
            model.gates(batch).expected_l0, model.gates(clean_batch).expected_l0, rtol=0, atol=0
        )


def promo_never_seen_before_cutoff():
    panel = toy_panel(length=80)
    panel = replace(
        panel,
        items=tuple(
            replace(
                item,
                dynamic={
                    "promo": (np.arange(len(item.targets["y"])) + item.start >= 48).astype(float),
                    "flat_past": np.zeros_like(item.targets["y"]),
                },
            )
            for item in panel.items
        ),
        roles={"promo": "known", "flat_past": "past"},
    )
    return panel, PanelStats(panel, "y", 48), 0, ("y",), 48


def test_known_column_constant_in_history_is_kept():
    c = {x.name: x for x in screen(*promo_never_seen_before_cutoff())}
    assert c["promo"].removed is None and c["flat_past"].removed == "constant"


def test_spread_only_category_scores():
    rng = np.random.default_rng(3)
    lab = np.array(["a", "b"] * 200, dtype=object)
    y = rng.normal(size=400) * np.where(lab == "a", 1.0, 3.0)
    assert adjusted_share(y, lab) < 0.05 and relevance_category(lab, y, (0,), 16) > 0.2


def test_nuisance_scores_low():
    rng = np.random.default_rng(4)
    assert relevance_number(rng.normal(size=400), rng.normal(size=400), (0, 1, 7), 16) < 0.2


def test_duplicates_do_not_depend_on_column_order():
    panel = toy_panel(length=80)
    for names in (("b", "a"), ("a", "b")):
        changed = replace(
            panel,
            items=tuple(
                replace(item, dynamic={name: item.dynamic["temp"].copy() for name in names})
                for item in panel.items
            ),
            roles={name: "past" for name in names},
        )
        candidates = screen(changed, PanelStats(changed, "y", 60), 0, ("y",), 60)
        assert [c.name for c in candidates if c.source == "past"] == ["a", "b"]
        assert {c.name: c.removed for c in candidates if c.source == "past"} == {
            "a": None,
            "b": "duplicate",
        }


def test_entity_like_static_is_removed():
    panel = toy_panel(n_items=12, length=80)
    panel = replace(
        panel,
        items=tuple(
            replace(item, static={"entity": f"id{i}", "private": [f"p{i}", f"q{i}"]})
            for i, item in enumerate(panel.items)
        ),
    )
    candidates = {c.name: c for c in screen(panel, PanelStats(panel, "y", 60), 0, ("y",), 60)}
    assert candidates["entity"].removed == "entity"
    assert candidates["private"].removed == "unshared"


def test_too_little_support_scores_zero():
    x = np.arange(10, dtype=float)
    assert relevance_number(x, x, (0,), 16) == 0.0
    assert relevance_category(np.array(["a", "b"] * 5), x, (0,), 16) == 0.0
    summaries = np.arange(5, dtype=float)[:, None]
    flags = np.ones_like(summaries, dtype=bool)
    assert relevance_static_number(np.arange(5), summaries, flags, 8) == 0.0
    assert relevance_static_category([("a",)] * 2 + [("b",)] * 3, summaries, flags, 8) == 0.0


def test_nothing_after_the_cutoff_is_read():
    panel = toy_panel(n_items=12, n_targets=2)
    changed = perturb_after(panel, 200)
    before = screen(panel, PanelStats(panel, "y", 200), 2, ("y", "y1"), 200)
    after = screen(changed, PanelStats(changed, "y", 200), 2, ("y", "y1"), 200)
    assert before == after


def test_select_order():
    candidates = [
        Candidate("past", "past", KIND_NUMBER, 1.0, None),
        Candidate("known_b", "known", KIND_FLAG, 0.1, None),
        Candidate("known_a", "known", KIND_FLAG, 0.1, None),
        Candidate("removed", "past", KIND_NUMBER, 1.0, "constant"),
        Candidate("sibling_b", "sibling", KIND_NUMBER, 0.5, None),
        Candidate("sibling_a", "sibling", KIND_NUMBER, 0.5, None),
        Candidate("big", "static", "multilabel", 1.0, None, 3),
        Candidate("small", "static", "category", 0.5, None, 2),
        Candidate("number", "static", "number", 0.2, None),
    ]
    kept = select(candidates, feature_budget=2, sibling_budget=1, attribute_budget=2)
    assert [c.name for c in kept] == ["known_a", "known_b", "sibling_a", "small"]
    assert kept == select(
        list(reversed(candidates)), feature_budget=2, sibling_budget=1, attribute_budget=2
    )
    assert select(candidates, feature_budget=0, sibling_budget=0, attribute_budget=0) == []


def test_adjusted_share_matches_hand_calculation_and_filters_missing():
    # eta2 = 16/20; adjusted = 1 - (1 - .8) * 3/2 = .7.
    assert adjusted_share(np.array([0, 2, 4, 6]), np.array(["a", "a", "b", "b"])) == pytest.approx(
        0.7
    )
    assert adjusted_share(
        np.array([0, 2, 4, 6, np.nan, 1e6]),
        np.array(["a", "a", "b", "b", "a", None], dtype=object),
    ) == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("values", "groups"),
    [([], []), ([1, 1], ["a", "b"]), ([1, 2], ["a", "a"]), ([1, 2], ["a", "b"])],
)
def test_adjusted_share_degenerate_cases(values, groups):
    assert adjusted_share(np.array(values), np.array(groups)) == 0.0


def test_numeric_relevance_uses_positive_lags_and_common_observations():
    x = np.random.default_rng(8).normal(size=128)
    y = np.r_[np.full(7, np.nan), -x[:-7]]
    assert relevance_number(x, y, (7,), 16) == pytest.approx(1.0)
    assert relevance_number(x, y, (0,), 16) < 0.2
    x[20:40] = np.nan
    y[70:80] = np.inf
    assert relevance_number(x, y, (7,), 16) == pytest.approx(1.0)
    assert relevance_number(np.ones(128), y, (0, 1, 7), 16) == 0.0
    assert relevance_number(x, np.ones(128), (0,), 16) == 0.0
    assert relevance_number(x, y, (200,), 16) == 0.0


def test_category_lags_renaming_and_multilabel_membership():
    lab = np.array(["a", "a", "b", "b"] * 40, dtype=object)
    y = np.r_[np.nan, np.where(lab[:-1] == "a", -2.0, 2.0)]
    assert relevance_category(lab, y, (1,), 16) == pytest.approx(1.0)
    renamed = np.where(lab == "a", "z", "x")
    assert relevance_category(renamed, y, (1,), 16) == relevance_category(lab, y, (1,), 16)
    multi = np.empty(160, dtype=object)
    multi[:] = [["a", "shared"] if label == "a" else ["shared"] for label in lab]
    assert relevance_category(multi, y, (1,), 16) == pytest.approx(1.0)
    multi[::9] = None
    assert relevance_category(multi, y, (1,), 16) == pytest.approx(1.0)


def test_static_relevance_uses_flags_and_rank_correlation():
    values = np.arange(12, dtype=float)
    summaries = np.column_stack([values**2, np.full(12, 42.0)])
    flags = np.ones_like(summaries, dtype=bool)
    summaries[-1, 0], flags[-1, 0] = -1e6, False
    assert relevance_static_number(values, summaries, flags, 8) == pytest.approx(1.0)
    flags[:5, 0] = False
    assert relevance_static_number(values, summaries, flags, 8) == 0.0
    assert relevance_static_number(np.ones(12), summaries, flags, 8) == 0.0


def test_static_category_scores_mean_spread_and_multilabel():
    labels = [("a",)] * 8 + [("b",)] * 8
    mean = np.r_[np.zeros(8), np.ones(8)][:, None]
    flags = np.ones_like(mean, dtype=bool)
    assert relevance_static_category(labels, mean, flags, 8) == pytest.approx(1.0)
    spread = np.array([-1, 1] * 4 + [-3, 3] * 4, dtype=float)[:, None]
    assert relevance_static_category(labels, spread, flags, 8) == pytest.approx(1.0)
    multi = [("a", "shared")] * 8 + [("shared",)] * 8
    assert relevance_static_category(multi, spread, flags, 8) == pytest.approx(1.0)
    flags[:10] = False
    assert relevance_static_category(labels, mean, flags, 8) == 0.0


def test_empty_past_known_and_duplicates_use_only_history():
    x = np.arange(80, dtype=float) + 0.25
    x[5] = np.nan
    b = x.copy()
    b[60:] = -100
    item = Item(
        "q",
        10,
        {"y": np.arange(80, dtype=float)},
        {"b": b, "a": x, "empty": np.full(80, np.nan), "known": x.copy()},
        {},
    )
    panel = Panel(
        "p",
        (item,),
        {"a": "past", "b": "past", "empty": "past", "known": "known"},
        None,
        None,
        True,
    )
    c = {c.name: c for c in screen(panel, PanelStats(panel, "y", 70), 0, ("y",), 70)}
    assert c["b"].removed == "duplicate" and c["a"].removed is None
    assert c["empty"].removed == "empty" and c["known"].removed is None
    assert c["a"].score == pytest.approx(1.0)
    assert c["a"].kind == KIND_NUMBER


def test_flag_scores_use_codes_and_siblings_use_only_lag_zero():
    x = np.tile([0.0, 1.0], 80)
    rng = np.random.default_rng(33)
    y = rng.normal(size=160)
    item = Item(
        "q",
        0,
        {"y": y, "other": x, "lagged": np.r_[y[1:], np.nan]},
        {"flag": np.where(x == 1, "z", "a").astype(object)},
        {},
    )
    panel = Panel("p", (item,), {"flag": "known"}, None, None, True)
    stats = PanelStats(panel, "y", 140)
    c = {c.name: c for c in screen(panel, stats, 0, ("y", "other"), 140)}
    assert c["flag"].kind == KIND_FLAG and c["flag"].score == pytest.approx(1.0)
    assert c["lagged"].source == "sibling" and c["lagged"].score < 0.2
    assert "y" not in c and "other" not in c


def test_static_duplicates_use_raw_values_and_count_query_labels():
    panel = toy_panel(n_items=12, length=80)
    panel = replace(
        panel,
        items=tuple(
            replace(
                item,
                static={
                    "a": ["shared", f"tag{i % 2}"],
                    "b": [f"tag{i % 2}", "shared"],
                    "n": i + 0.25,
                    "offset": i + 1.25,
                },
            )
            for i, item in enumerate(panel.items)
        ),
    )
    c = {c.name: c for c in screen(panel, PanelStats(panel, "y", 60), 0, ("y",), 60)}
    assert c["a"].size == 2 and c["b"].removed == "duplicate"
    assert c["n"].removed is None and c["offset"].removed is None
    assert c["n"].size == c["offset"].size == 1


def test_screen_is_invariant_to_item_column_and_label_order():
    panel = toy_panel(n_items=12, length=100, n_targets=2)
    order = np.random.default_rng(31).permutation(12)
    changed = replace(
        panel,
        items=tuple(
            replace(
                panel.items[i],
                item_id=f"renamed{i}",
                targets=dict(reversed(list(panel.items[i].targets.items()))),
                dynamic=dict(reversed(list(panel.items[i].dynamic.items()))),
                static={
                    **panel.items[i].static,
                    "brand": f"new{panel.items[i].static['brand']}",
                    "tags": list(reversed(panel.items[i].static["tags"])),
                },
            )
            for i in order
        ),
    )
    before = screen(panel, PanelStats(panel, "y", 80), 2, ("y", "y1"), 80)
    after = screen(
        changed, PanelStats(changed, "y", 80), int(np.flatnonzero(order == 2)[0]), ("y1", "y"), 80
    )
    assert [(c.source, c.name, c.kind, c.removed, c.size) for c in before] == [
        (c.source, c.name, c.kind, c.removed, c.size) for c in after
    ]
    np.testing.assert_allclose([c.score for c in before], [c.score for c in after], atol=1e-12)
    assert [(c.source, c.name) for c in before] == sorted((c.source, c.name) for c in before)


@pytest.mark.parametrize("n_targets", [1, 16])
def test_static_scores_are_cached_across_query_items(monkeypatch, n_targets):
    panel = toy_panel(n_items=12, length=80, n_targets=n_targets)
    stats = PanelStats(panel, "y", 60)
    targets = tuple(panel.items[0].targets)
    first = screen(panel, stats, 0, targets, 60)

    def unexpected_recomputation(*args, **kwargs):
        pytest.fail("static relevance was recomputed for another query item")

    monkeypatch.setattr(sc, "relevance_static_category", unexpected_recomputation)
    monkeypatch.setattr(sc, "relevance_static_number", unexpected_recomputation)
    second = screen(panel, stats, 1, targets, 60)
    assert {c.name: c.score for c in first if c.source == "static"} == {
        c.name: c.score for c in second if c.source == "static"
    }


def test_screen_cache_releases_panel_with_stats():
    panel = toy_panel(n_items=12, length=80, n_targets=16)
    stats = PanelStats(panel, "y", 60)
    reference = weakref.ref(panel)
    screen(panel, stats, 0, tuple(panel.items[0].targets), 60)
    del stats, panel
    gc.collect()
    assert reference() is None


def test_screen_reuses_each_other_target_with_sixteen_targets(monkeypatch):
    panel = toy_panel(n_items=12, length=80, n_targets=16)
    stats = PanelStats(panel, "y", 60)
    targets = tuple(panel.items[0].targets)
    constructions = Counter()
    original_init = PanelStats.__init__

    def counted_init(self, panel, target, cutoff):
        constructions[target] += 1
        original_init(self, panel, target, cutoff)

    monkeypatch.setattr(PanelStats, "__init__", counted_init)
    results = [screen(panel, stats, 0, targets, 60) for _ in range(3)]
    assert results[0] == results[1] == results[2]
    assert constructions == {target: 1 for target in targets if target != "y"}


def test_static_cache_respects_min_items():
    panel = toy_panel(n_items=12, length=80)
    stats = PanelStats(panel, "y", 60)
    first = screen(panel, stats, 0, ("y",), 60, ScreeningConfig(min_items=8))
    unsupported = screen(panel, stats, 0, ("y",), 60, ScreeningConfig(min_items=13))
    repeated = screen(panel, stats, 0, ("y",), 60, ScreeningConfig(min_items=8))
    assert any(c.score > 0 for c in first if c.source == "static")
    assert all(c.score == 0 for c in unsupported if c.source == "static")
    assert first == repeated


@pytest.mark.parametrize(
    ("attribute", "missing"), [("brand", None), ("tags", []), ("tags", [None])]
)
def test_missing_static_labels_are_not_candidates(attribute, missing):
    panel = toy_panel(n_items=12, length=80)
    panel = replace(
        panel,
        items=(
            replace(panel.items[0], static={**panel.items[0].static, attribute: missing}),
            *panel.items[1:],
        ),
    )
    stats = PanelStats(panel, "y", 60)
    absent = screen(panel, stats, 0, ("y",), 60)
    present = screen(panel, stats, 1, ("y",), 60)
    assert attribute not in {c.name for c in absent if c.source == "static"}
    assert next(c for c in present if c.source == "static" and c.name == attribute).size > 0


def test_empty_static_dictionary_keeps_only_number_candidates():
    panel = toy_panel(n_items=12, length=80)
    panel = replace(panel, items=(replace(panel.items[0], static={}), *panel.items[1:]))
    stats = PanelStats(panel, "y", 60)
    candidates = screen(panel, stats, 0, ("y",), 60)
    assert [(c.name, c.kind, c.size, c.removed) for c in candidates if c.source == "static"] == [
        ("size", "number", 1, None)
    ]
    names, numbers, flags = stats.static_numbers(0)
    assert names == ["size"]
    np.testing.assert_array_equal(numbers, [0.0])
    np.testing.assert_array_equal(flags, [False])


def test_zero_attribute_budget_selects_no_static_category_for_missing_labels():
    panel = toy_panel(n_items=12, length=80)
    panel = replace(panel, items=(replace(panel.items[0], static={}), *panel.items[1:]))
    candidates = screen(panel, PanelStats(panel, "y", 60), 0, ("y",), 60)
    assert not any(c.source == "static" for c in select(candidates, attribute_budget=0))


def test_select_limits_static_numbers_and_tries_smaller_candidates():
    candidates = [Candidate(f"n{i:02}", "static", "number", 0.5, None) for i in range(20)]
    candidates.append(Candidate("category", "static", "category", 0.1, None, 2))
    kept = select(candidates, attribute_budget=30)
    assert [c.name for c in kept] == [f"n{i:02}" for i in range(16)] + ["category"]
    small = [
        Candidate("big", "static", "multilabel", 1.0, None, 4),
        Candidate("small", "static", "category", 0.5, None, 1),
    ]
    assert [c.name for c in select(small, attribute_budget=2)] == ["small"]


def test_no_evidence_threshold_zeroes_scores_without_removing_known():
    panel, stats, item_index, targets, cutoff = promo_never_seen_before_cutoff()
    c = screen(panel, stats, item_index, targets, cutoff, ScreeningConfig(no_evidence=1.0))
    assert all(x.score == 0.0 for x in c)
    assert next(x for x in c if x.name == "promo").removed is None
