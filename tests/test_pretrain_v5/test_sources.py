"""Uniform corpus variates and typed controls with identical random streams."""

import hashlib
import importlib
from dataclasses import replace

import numpy as np
import pytest

from demand_forecast.data.episode import is_missing
from demand_forecast.data.episode.schema import FREQ_CLASSES
from demand_forecast.data.frequency import to_offset
from demand_forecast.pretrain.v5.config import SourceConfig
from demand_forecast.pretrain.v5.sampler import make_batch
from demand_forecast.synthetic.typed import PanelRecipe, sample_recipe, shuffle_assignment
from tests.helpers.corpus_v4_fixture import write_corpus, write_malformed_row, write_rows
from tests.helpers.v5_sampler import config


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.sources")


def equal_panels(left, right):
    assert (left.name, left.freq, left.epoch, left.aligned, left.roles) == (
        right.name,
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
                x, y = getattr(a, family)[name], getattr(b, family)[name]
                np.testing.assert_array_equal(x, y)
                assert (x.dtype, x.shape) == (y.dtype, y.shape)
                if not x.dtype.hasobject:
                    assert x.tobytes() == y.tobytes()


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


@pytest.mark.parametrize("recipes", [(), (("event_response",),)])
def test_shuffle_rejects_too_few_items_at_construction(recipes):
    config = SourceConfig("typed", 1.0, recipes=recipes, max_items=1, control="shuffle")
    with pytest.raises(ValueError, match=r"shuffle.*max_items.*2"):
        api().TypedSource(config)
    for invalid in (-1, 0):  # refused earlier, by the configuration itself
        with pytest.raises(ValueError, match="max_items"):
            SourceConfig("typed", 1.0, recipes=recipes, max_items=invalid, control="shuffle")


@pytest.mark.parametrize("mechanisms", [(), ("static_category",)])
@pytest.mark.parametrize("items", [0, 1])
def test_apply_shuffle_rejects_short_panels_even_without_inputs(mechanisms, items):
    module = api()
    generated = module.generate_typed_panel(
        PanelRecipe(mechanisms=mechanisms, n_items=1, length=(64, 64), seed=7)
    )
    generated = replace(
        generated, panel=replace(generated.panel, items=generated.panel.items[:items])
    )
    with pytest.raises(ValueError, match="shuffle"):
        module.apply_control(generated, SourceConfig("typed", 1, control="shuffle"))


def test_shuffle_raises_sampled_single_item_to_two_without_extra_rng_draws(monkeypatch):
    def one_item_recipe(seed, *, heldout):
        return PanelRecipe(
            mechanisms=("event_response",),
            n_items=1,
            length=(64, 64),
            seed=seed,
            heldout=heldout,
        )

    module = api()
    monkeypatch.setattr(module, "sample_recipe", one_item_recipe)
    config = SourceConfig("typed", 1.0, max_items=2, heldout=True, control="shuffle")
    rng, duplicate = np.random.default_rng(9), np.random.default_rng(9)
    seed = int(duplicate.integers(0, 2**63))
    panel, record = module.TypedSource(config).panel(rng)
    panel.validate()
    assert len(panel.items) == record["recipe"].n_items == 2
    assert record["recipe"] == replace(one_item_recipe(seed, heldout=True), n_items=2)
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


@pytest.mark.parametrize("selection", ["all", "attribute"])
def test_hierarchy_and_its_twins_are_each_shuffled_once(monkeypatch, selection):
    config = replace(
        typed_config(n_nuisance=2, nuisance_from=("hierarchy",)),
        recipes=(("hierarchy",),),
        max_items=64,
    )
    module = api()
    calls = []

    def capture_shuffle(generated, name, seed):
        result = shuffle_assignment(generated, name, seed)
        calls.append(result)
        return result

    monkeypatch.setattr(module, "shuffle_assignment", capture_shuffle)
    true_rng, null_rng = np.random.default_rng(3), np.random.default_rng(3)
    true, true_record = module.TypedSource(config).panel(true_rng)
    panel, record = module.TypedSource(
        replace(config, control="shuffle", control_inputs=selection)
    ).panel(null_rng)
    assert true_rng.bit_generator.state == null_rng.bit_generator.state
    equal_panels(true, record["generated"].panel)
    equal_panels(true_record["generated"].complete, record["generated"].complete)
    assert record["generated"].shuffled == ()
    generated = record["generated"]
    assert len(generated.inputs) == 6  # hierarchy pair and two nuisance twin pairs
    assert len(calls) == 3
    assert [len(call.shuffled) for call in calls] == [2, 4, 6]
    controlled = calls[-1]
    originals = {item.item_id: item for item in generated.complete.items}
    for i in (0, 2, 4):
        first, second = generated.inputs[i : i + 2]
        assert controlled.donors[first.name] == controlled.donors[second.name]
    for input_record in generated.inputs:
        name = input_record.name
        assert any(a.static[name] != b.static[name] for a, b in zip(true.items, panel.items))
        for original, moved in zip(true.items, panel.items, strict=True):
            donor, _ = controlled.donors[name][original.item_id]
            assert donor != original.item_id
            expected = original.static[name]
            if not is_missing(expected):
                expected = originals[donor].static[name]
            assert moved.static[name] == expected
    for original, moved in zip(true.items, panel.items, strict=True):
        for name, target in original.targets.items():
            assert target.dtype == moved.targets[name].dtype
            assert target.shape == moved.targets[name].shape
            assert target.tobytes() == moved.targets[name].tobytes()


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


@pytest.mark.parametrize("mechanism", ["hierarchy", "event_response"])
@pytest.mark.parametrize(
    "control,selection",
    [
        (None, "all"),
        ("shuffle", "all"),
        ("shuffle", "attribute"),
        ("shuffle", "column"),
        ("no_dates", "all"),
        ("missing_as_zero", "all"),
    ],
)
def test_apply_control_matches_recipe_seed_and_input_selection(mechanism, control, selection):
    module = api()
    config = replace(
        typed_config(n_nuisance=2, nuisance_from=(mechanism,)),
        recipes=((mechanism,),),
        max_items=64,
        control=control,
        control_inputs=selection,
    )
    rng, duplicate = np.random.default_rng(20), np.random.default_rng(20)
    _, original = module.TypedSource(replace(config, control=None)).panel(duplicate)
    generated = original["generated"]
    expected = generated
    if control == "shuffle":
        for record in generated.inputs:
            if record.name in expected.shuffled:
                continue
            if selection == "attribute" and record.where != "static":
                continue
            if selection == "column" and record.where != "dynamic":
                continue
            expected = shuffle_assignment(expected, record.name, seed=generated.recipe.seed)
    panel = replace(expected.panel, epoch=None) if control == "no_dates" else expected.panel
    equal_panels(module.apply_control(generated, config), panel)
    exposed, record = module.TypedSource(config).panel(rng)
    equal_panels(exposed, panel)
    equal_panels(record["generated"].panel, generated.panel)
    assert rng.bit_generator.state == duplicate.bit_generator.state


@pytest.mark.parametrize(
    "label",
    [
        "30T",
        "S",
        "5T",
        "D",
        "H",
        "20S",
        "T",
        "native_irregular_seconds",
        "3H",
        "10T",
        "12500us",
        "irregular",
        "native_sample_index",
        "h",
        "30min",
        "4S",
        "W",
        "4ms",
        "W-SAT",
        "3min",
        "MS",
        "5ms",
        "15T",
        "2ms",
        "33ms",
        "100ms",
        "6min",
        "s",
        "min",
        "1m",
        "6h",
        "3h",
        "M",
    ],
)
def test_every_v41_corpus_frequency_label_yields_a_valid_panel(tmp_path, label):
    pytest.importorskip("pyarrow")
    target = np.arange(40)
    write_rows(tmp_path / "univariate/a/train-000.parquet", [[target]], freqs=[label])
    source = api().CorpusSource(SourceConfig("corpus", 1.0), root=tmp_path)
    assert source.index.rows == 1
    row = api().read_values(source.index, 0, source.cache)
    np.testing.assert_array_equal(row["targets"], [target])
    panel, record = source.panel(np.random.default_rng(0))
    panel.validate()
    assert panel.epoch is None and panel.aligned is False
    assert record["source"] == "corpus" and record["row"] == row["row"] == 0
    if label in {"irregular", "native_irregular_seconds", "native_sample_index", "1m"}:
        assert row["freq"] is None and panel.freq is None
        assert source.index.unparsed_frequencies == {label: 1}
        assert panel.season == 1 and panel.long_lag is None
        assert FREQ_CLASSES[panel.freq_class] == "undated"
        assert panel.timestamps(0, len(target)) is None
    else:
        assert row["freq"] == panel.freq == label
        seconds = {"S": 1, "20S": 20, "4S": 4}
        if label in seconds:
            assert to_offset(panel.freq).nanos == seconds[label] * 1_000_000_000
            assert FREQ_CLASSES[panel.freq_class] == "subhourly"


def test_corpus_only_batch_without_frequencies(tmp_path):
    pytest.importorskip("pyarrow")
    write_rows(
        tmp_path / "univariate/a/train-000.parquet",
        [[np.arange(96)], [np.arange(96) + 1]],
        freqs=["irregular", None],
    )
    settings = replace(config(), sources=(SourceConfig("corpus", 1, root=str(tmp_path)),))
    batch, report = make_batch(settings, 0)
    assert report["episodes"] == settings.sampler.episodes_per_batch
    assert report["sources"] == {"corpus": report["episodes"]}
    assert report["supervised"] > 0
    assert (batch.freq_class == FREQ_CLASSES.index("undated")).all()
    assert (batch.cal_phases == 0).all() and (batch.calh_phases == 0).all()
    assert (batch.cal_ids == -1).all() and (batch.calh_ids == -1).all()


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


@pytest.mark.parametrize(
    "problem", ["null_variate", "null_channel", "ragged_channels", "zero_variates", "bad_shape"]
)
def test_corpus_panel_propagates_malformed_row_skip_unchanged(tmp_path, monkeypatch, problem):
    pytest.importorskip("pyarrow")
    write_malformed_row(tmp_path, problem)
    module = api()
    corpus = importlib.import_module("demand_forecast.pretrain.v5.corpus")
    source = module.CorpusSource(SourceConfig("corpus", 1.0, root=str(tmp_path)))
    read = module.read_values
    skips = []

    def capture_skip(*args):
        try:
            return read(*args)
        except corpus.SamplerSkip as skip:
            skips.append(skip)
            raise

    monkeypatch.setattr(module, "read_values", capture_skip)
    with pytest.raises(corpus.CorpusValueError, match=r"train-bad.parquet: row 1") as caught:
        source.panel(np.random.default_rng(0))  # seed 0 selects the malformed second row
    assert skips == [caught.value]


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


def test_corpus_roots_only_fill_missing_configuration(tmp_path):
    pytest.importorskip("pyarrow")
    roots = [tmp_path / name for name in ("first", "second", "fallback")]
    for i, root in enumerate(roots):
        write_rows(root / "univariate/a/train-000.parquet", [[np.arange(32) + i]])
    configs = tuple(SourceConfig("corpus", 1.0, root=str(root)) for root in roots[:2])
    mix = api().SourceMix(configs, corpus_root=roots[2])
    assert [source.index.root for source in mix.sources] == roots[:2]
    assert api().CorpusSource(configs[0], roots[2]).index.root == roots[0]
    digests = [source.index.digest for source in mix.sources]
    assert len(set(digests)) == 2
    expected = hashlib.sha256(
        "".join(f"{i}\t0\t{digest}\n" for i, digest in enumerate(digests)).encode()
    ).hexdigest()
    assert mix.corpus_digest == expected
    fallback = api().SourceMix((SourceConfig("corpus", 1.0),), corpus_root=roots[2])
    assert fallback.sources[0].index.root == roots[2]
    with pytest.raises(ValueError, match="corpus source requires a root"):
        api().SourceMix((SourceConfig("corpus", 1.0),))
