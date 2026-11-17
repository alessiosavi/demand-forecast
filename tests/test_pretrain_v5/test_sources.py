"""Uniform corpus variates and typed controls with identical random streams."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.synthetic.typed import sample_recipe, shuffle_assignment
from tests.helpers.corpus_v4_fixture import write_corpus, write_rows


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.sources")


def equal_panels(left, right):
    assert (left.freq, left.epoch, left.aligned, left.roles) == (
        right.freq,
        right.epoch,
        right.aligned,
        right.roles,
    )
    assert len(left.items) == len(right.items)
    for a, b in zip(left.items, right.items, strict=True):
        assert (a.item_id, a.start, a.static) == (b.item_id, b.start, b.static)
        for family in ("targets", "dynamic"):
            assert getattr(a, family).keys() == getattr(b, family).keys()
            for name in getattr(a, family):
                np.testing.assert_array_equal(getattr(a, family)[name], getattr(b, family)[name])


def typed_config(**kwargs):
    return SourceConfig(
        "typed",
        1.0,
        recipes=(("event_response",),),
        max_items=12,
        length=(64, 96),
        freq="h",
        **kwargs,
    )


def test_explicit_recipes_honor_all_configured_fields():
    config = replace(
        typed_config(n_nuisance=2, nuisance_from=("event_response",), singleton_share=0.2),
        recipes=(("event_response",), ("static_number", "known_driver")),
        nuisance_from=None,
        heldout=True,
    )
    source = api().TypedSource(config)
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(12):
        panel, record = source.panel(rng)
        panel.validate()
        recipe = record["recipe"]
        seen.add(recipe.mechanisms)
        assert recipe.mechanisms in config.recipes
        assert (recipe.freq, recipe.length, recipe.n_items, recipe.n_nuisance) == (
            "h",
            (64, 96),
            12,
            2,
        )
        assert recipe.heldout and recipe.singleton_share == 0.2 and not recipe.complete_inputs
        assert record["source"] == "typed"
        assert record["index"] == {item.item_id: i for i, item in enumerate(panel.items)}
    assert seen == set(config.recipes)


def test_sampled_recipe_only_caps_items_and_passes_heldout():
    config = SourceConfig("typed", 1.0, max_items=3, heldout=True)
    rng = np.random.default_rng(9)
    duplicate = np.random.default_rng(9)
    seed = int(duplicate.integers(0, 2**63))
    _, record = api().TypedSource(config).panel(rng)
    assert record["recipe"] == replace(sample_recipe(seed, heldout=True), n_items=3)
    assert rng.bit_generator.state == duplicate.bit_generator.state


@pytest.mark.parametrize("selection", ["all", "attribute", "column"])
def test_shuffle_preserves_targets_uncontrolled_record_and_rng(selection):
    config = typed_config(n_nuisance=2, nuisance_from=("event_response",))
    true_rng, null_rng = np.random.default_rng(20), np.random.default_rng(20)
    true_panel, true_record = api().TypedSource(config).panel(true_rng)
    null_panel, null_record = (
        api()
        .TypedSource(replace(config, control="shuffle", control_inputs=selection))
        .panel(null_rng)
    )
    assert true_rng.bit_generator.state == null_rng.bit_generator.state
    assert true_record["recipe"] == null_record["recipe"]
    equal_panels(true_panel, null_record["generated"].panel)
    equal_panels(true_record["generated"].complete, null_record["generated"].complete)
    generated = true_record["generated"]
    expected = generated
    for record in generated.inputs:
        if (
            selection == "all"
            or record.where == {"attribute": "static", "column": "dynamic"}[selection]
        ):
            expected = shuffle_assignment(expected, record.name, seed=generated.recipe.seed)
    equal_panels(null_panel, expected.panel)
    for true, null in zip(true_panel.items, null_panel.items, strict=True):
        for name in true.targets:
            np.testing.assert_array_equal(true.targets[name], null.targets[name])
        if selection == "attribute":
            for name in true.dynamic:
                np.testing.assert_array_equal(true.dynamic[name], null.dynamic[name])
    assert null_record["generated"].shuffled == ()
    assert (
        any(a.static != b.static for a, b in zip(true_panel.items, null_panel.items, strict=True))
        if selection != "column"
        else any(
            not np.array_equal(a.dynamic[name], b.dynamic[name], equal_nan=True)
            for a, b in zip(true_panel.items, null_panel.items, strict=True)
            for name in a.dynamic
        )
    )


def test_hierarchy_and_its_twins_are_each_shuffled_once():
    config = replace(typed_config(n_nuisance=2), recipes=(("hierarchy",),), control="shuffle")
    panel, record = api().TypedSource(config).panel(np.random.default_rng(3))
    expected = record["generated"]
    for input_record in expected.inputs:
        if input_record.name not in expected.shuffled:
            expected = shuffle_assignment(expected, input_record.name, expected.recipe.seed)
    equal_panels(panel, expected.panel)


@pytest.mark.parametrize("control", ["no_dates", "missing_as_zero"])
def test_other_controls_preserve_uncontrolled_record_and_rng(control):
    config = typed_config()
    left, right = np.random.default_rng(2), np.random.default_rng(2)
    panel, record = api().TypedSource(replace(config, control=control)).panel(left)
    true, _ = api().TypedSource(config).panel(right)
    assert left.bit_generator.state == right.bit_generator.state
    equal_panels(record["generated"].panel, true)
    assert record["control"] == control
    equal_panels(panel, replace(true, epoch=None) if control == "no_dates" else true)


def test_all_corpus_variates_reachable_and_panel_limits(tmp_path):
    pytest.importorskip("pyarrow")
    target = [np.arange(40, dtype=np.float64) + j for j in range(20)]
    write_rows(tmp_path / "univariate/multi/train-000.parquet", [target])
    source = api().CorpusSource(SourceConfig("corpus", 1.0, root=str(tmp_path)))
    rng = np.random.default_rng(3)
    designated = set()
    for _ in range(400):
        panel, record = source.panel(rng)
        designated.add(record["target"])
        item = panel.items[0]
        assert len(item.targets) == 16 and record["target"] in item.targets
        for name, values in item.targets.items():
            np.testing.assert_array_equal(values, target[int(name.removeprefix("target_"))])
    assert designated == {f"target_{i:03d}" for i in range(20)}


def test_corpus_roles_vary_covariates_are_sampled_and_grid_has_season(tmp_path):
    pytest.importorskip("pyarrow")
    first = np.arange(40, dtype=np.float64)
    write_rows(
        tmp_path / "covariates/a/train-000.parquet",
        [first],
        kind="covariates",
        covariates=[[first + j for j in range(40)]],
        freqs=["h"],
    )
    source = api().CorpusSource(SourceConfig("corpus", 1.0), root=tmp_path)
    rng = np.random.default_rng(6)
    assignments = {}
    seen = set()
    for _ in range(10):
        panel, record = source.panel(rng)
        panel.validate()
        assert panel.epoch is None and panel.aligned is False and panel.season == 24
        assert len(panel.items) == 1 and panel.items[0].static == {}
        assert len(panel.roles) == len(panel.items[0].dynamic) == 32
        assert record["source"] == "corpus" and record["row"] == 0
        assert record["file"] == "covariates/a/train-000.parquet"
        assert record["target"] == "target_000"
        for name, role in panel.roles.items():
            assert role in ("known", "past")
            assignments.setdefault(name, set()).add(role)
            np.testing.assert_array_equal(panel.items[0].dynamic[name], first + int(name[4:]))
            seen.add(name)
    assert any(len(roles) == 2 for roles in assignments.values())
    assert len(seen) == 40


def test_mix_weights_and_corpus_digest(tmp_path):
    configs = tuple(
        SourceConfig("typed", weight, recipes=((),), freq=freq, max_items=1, length=(8, 8))
        for weight, freq in ((0.25, "D"), (0.75, "h"))
    )
    mix = api().SourceMix(configs)
    assert mix.corpus_digest is None
    rng = np.random.default_rng(7)
    share = sum(mix.panel(rng)[0].freq == "D" for _ in range(2000)) / 2000
    assert abs(share - 0.25) < 0.05
    pytest.importorskip("pyarrow")
    write_corpus(tmp_path)
    corpus = api().SourceMix((SourceConfig("corpus", 1.0),), corpus_root=tmp_path)
    index_api = importlib.import_module("demand_forecast.pretrain.v5.corpus")
    assert corpus.corpus_digest == index_api.CorpusIndex.build(tmp_path).digest
    assert corpus.panel(rng)[1]["source"] == "corpus"
