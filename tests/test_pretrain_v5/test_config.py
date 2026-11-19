"""Portable configuration, experiment identity, and comparison constraints."""

import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import yaml

from demand_forecast.data.episode.schema import MAX_CONTEXT, MAX_HORIZON, PATCH
from demand_forecast.models.v5 import V5Config, V5Model
from demand_forecast.synthetic.typed.draft import INPUT_MECHANISMS
from demand_forecast.synthetic.typed.recipe import MECHANISMS


def config_module():
    return importlib.import_module("demand_forecast.pretrain.v5.config")


def training_config():
    cfg = config_module()
    return cfg.V5TrainConfig(
        model=V5Config(),
        sources=(
            cfg.SourceConfig(
                "typed",
                0.5,
                recipes=(("static_number", "known_driver"),),
                freq="D",
                length=(128, 256),
                nuisance_from=("static_number",),
            ),
            cfg.SourceConfig("corpus", 0.5, root="/corpus"),
        ),
        sampler=cfg.SamplerConfig(),
        optim=cfg.OptimConfig(max_steps=10),
    )


def test_yaml_round_trip_is_plain_and_restores_tuples(tmp_path):
    cfg = training_config()
    path = tmp_path / "train.yaml"
    cfg.to_yaml(path)
    raw = yaml.safe_load(path.read_text())
    assert raw["sources"][0]["recipes"] == [["static_number", "known_driver"]]
    assert raw["sampler"]["contexts"] == [256, 512, 1024]
    assert raw["model"]["quantiles"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    restored = type(cfg).from_yaml(path)
    assert restored == cfg
    assert isinstance(restored.sources, tuple)
    assert isinstance(restored.sources[0].recipes[0], tuple)
    assert isinstance(restored.sampler.drop[0], tuple)
    assert isinstance(restored.model.quantiles, tuple)


@pytest.mark.parametrize(
    "value", [-0.01, 1.0, 1.01, False, True, "0.5", float("nan"), float("inf")]
)
def test_screen_no_evidence_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="screen_no_evidence"):
        config_module().SamplerConfig(screen_no_evidence=value)


def test_screen_no_evidence_yaml_round_trip_and_identity(tmp_path):
    base = training_config()
    cfg = replace(base, sampler=replace(base.sampler, screen_no_evidence=0.999))
    path = tmp_path / "train.yaml"
    cfg.to_yaml(path)
    raw = yaml.safe_load(path.read_text())
    assert raw["sampler"]["screen_no_evidence"] == 0.999
    assert type(cfg).from_yaml(path) == cfg
    assert cfg.canonical()["sampler"]["screen_no_evidence"] == 0.999
    parts = dict(code="0" * 40, corpus=None, init=None, env="e" * 64)
    identity = config_module().identity
    assert identity(cfg, **parts) != identity(base, **parts)

    del raw["sampler"]["screen_no_evidence"]
    path.write_text(yaml.safe_dump(raw))
    assert type(cfg).from_yaml(path) == base


@pytest.mark.parametrize(
    "with_corpus,expected",
    [
        (False, "5864f0742ee6db2d2e4f82229c7265f1d99f880767d2336f5686caf8c3e9213c"),
        (True, "d2f2e89d63f935af94d8f198f02105097d1169803a26ffbcdb0cff19203e01d2"),
    ],
)
def test_default_screen_no_evidence_keeps_base_identities(with_corpus, expected):
    api = config_module()
    sources = (api.SourceConfig("typed", 1.0),)
    if with_corpus:
        sources += (api.SourceConfig("corpus", 1.0, root="/corpus"),)
    cfg = replace(api.T1_BASE, sources=sources)
    assert "screen_no_evidence" not in cfg.canonical()["sampler"]
    assert api.identity(cfg, code="0" * 40, corpus=None, init=None, env="e" * 64) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cold_share", 1.2),
        ("cold_steps", (2, 4)),
        ("cold_steps", (5, 3)),
        ("cold_steps", (3, 16)),
    ],
)
def test_cold_sampler_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        config_module().SamplerConfig(**{field: value})


@pytest.mark.parametrize("overrides", [{"cold_share": 1.0}, {"cold_steps": (3, 5)}])
def test_cold_sampler_yaml_and_identity(tmp_path, overrides):
    base = training_config()
    default = base.canonical()["sampler"]
    assert "cold_share" not in default and "cold_steps" not in default
    cfg = replace(base, sampler=replace(base.sampler, **overrides))
    path = tmp_path / "cold.yaml"
    cfg.to_yaml(path)
    raw = yaml.safe_load(path.read_text())["sampler"]
    assert raw["cold_share"] == cfg.sampler.cold_share
    assert raw["cold_steps"] == list(cfg.sampler.cold_steps)
    assert type(cfg).from_yaml(path) == cfg
    actual = cfg.canonical()["sampler"]
    assert {field: actual[field] for field in ("cold_share", "cold_steps") if field in actual} == {
        field: raw[field] for field in overrides
    }
    identity = config_module().identity
    parts = dict(code="0" * 40, corpus=None, init=None, env="e" * 64)
    assert identity(cfg, **parts) != identity(base, **parts)


def test_cold_share_refuses_missing_as_zero():
    base = training_config()
    sampler = replace(
        base.sampler, origins=1, designate_target=True, max_cells=None, cold_share=0.5
    )
    sources = (replace(base.sources[0], control="missing_as_zero"),)
    with pytest.raises(
        ValueError, match=r"cold_share.*missing_as_zero|missing_as_zero.*cold_share"
    ):
        replace(base, sampler=sampler, sources=sources)


@pytest.mark.parametrize("section", [None, "model", "sources", "sampler", "optim"])
def test_yaml_rejects_unknown_keys_at_every_level(tmp_path, section):
    cfg = training_config()
    path = tmp_path / "train.yaml"
    cfg.to_yaml(path)
    raw = yaml.safe_load(path.read_text())
    target = raw if section is None else raw[section]
    if section == "sources":
        target = target[0]
    target["unknown_option"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises((TypeError, ValueError), match="unknown_option"):
        type(cfg).from_yaml(path)


def test_yaml_preserves_model_validation(tmp_path):
    cfg = training_config()
    path = tmp_path / "train.yaml"
    cfg.to_yaml(path)
    raw = yaml.safe_load(path.read_text())
    raw["model"]["d_model"] = 193
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="divisible"):
        type(cfg).from_yaml(path)


def test_all_configuration_sections_are_frozen():
    cfg = training_config()
    for record, name in (
        (cfg, "seed"),
        (cfg.sources[0], "kind"),
        (cfg.sampler, "origins"),
        (cfg.optim, "max_steps"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, name, None)


def test_canonical_ignores_only_location_and_execution_settings():
    cfg = training_config()
    moved = replace(
        cfg,
        workers=5,
        device="cuda",
        sources=tuple(
            replace(source, root="/elsewhere") if source.kind == "corpus" else source
            for source in cfg.sources
        ),
    )
    assert cfg.canonical() == moved.canonical()
    canonical = cfg.canonical()
    assert "workers" not in canonical and "device" not in canonical
    assert all("root" not in source for source in canonical["sources"])
    canonical["model"]["d_model"] = 1
    assert cfg.model.d_model == 384


@pytest.mark.parametrize(
    "section", ["model", "sources", "sampler", "optim", "seed", "router", "gates"]
)
def test_canonical_changes_with_training_semantics(section):
    cfg = training_config()
    changed = {
        "model": replace(cfg.model, dropout=0.2),
        "sources": (replace(cfg.sources[0], weight=0.75), cfg.sources[1]),
        "sampler": replace(cfg.sampler, contexts=(128, 256)),
        "optim": replace(cfg.optim, learning_rate=1e-3),
        "seed": 12,
        "router": "uniform",
        "gates": "ones",
    }[section]
    assert cfg.canonical() != replace(cfg, **{section: changed}).canonical()


def test_identity_hashes_all_five_parts_in_canonical_json():
    api = config_module()
    cfg = training_config()
    parts = {"code": "rev", "corpus": "corpus", "init": "weights", "env": "env"}
    identity = api.identity(cfg, **parts)
    expected = {"config": cfg.canonical(), **parts}
    assert (
        identity
        == hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert api.identity(replace(cfg, seed=4), **parts) != identity
    for name in parts:
        assert api.identity(cfg, **{**parts, name: "different"}) != identity
        assert api.identity(cfg, **{**parts, name: None}) != identity
    assert api.identity(replace(cfg, workers=7, device="cpu"), **parts) == identity


def test_t1_model_and_base_are_valid_and_small():
    api = config_module()
    assert isinstance(api.T1_MODEL, V5Config)
    assert sum(p.numel() for p in V5Model(api.T1_MODEL).parameters()) < 6_000_000
    base = api.T1_BASE
    assert base.model == api.T1_MODEL and base.sources == ()
    assert base.sampler.contexts == (256, 512)
    assert base.sampler.episodes_per_batch == 32 and base.sampler.origins == 4
    assert base.sampler.path_share == 0.5
    assert base.sampler.max_cells is None and base.sampler.designate_target
    assert base.optim.max_steps == 3000 and base.optim.warmup_steps == 200
    assert base.optim.checkpoint_minutes == 10


def test_every_t1_arm_constructs_with_default_base():
    t1 = importlib.import_module("demand_forecast.pretrain.v5.t1")
    for path, entry in t1.PATHS.items():
        for arm in entry["arms"]:
            config = t1.t1_config(path, arm, seed=0)
            assert config.sources[0].recipes == entry["sets"]


def test_t1_null_rolled_arm_uses_unbudgeted_multi_episode_batches():
    t1 = importlib.import_module("demand_forecast.pretrain.v5.t1")
    config = t1.t1_config("routing", "null_rolled", seed=7)
    assert config.router == "rolled"
    assert config.sampler.episodes_per_batch >= 2
    assert config.sampler.max_cells is None


def test_missing_as_zero_requires_one_origin():
    api = config_module()
    base = api.T1_BASE
    source = api.SourceConfig("typed", 1.0, control="missing_as_zero")
    with pytest.raises(ValueError, match="origins"):
        replace(base, sources=(source,))
    cfg = replace(base, sources=(source,), sampler=replace(base.sampler, origins=1))
    assert cfg.sampler.origins == 1


@pytest.mark.parametrize("control", ["shuffle", "no_dates", "missing_as_zero"])
@pytest.mark.parametrize("violation", ["budget", "target"])
def test_source_controls_require_identical_unbudgeted_designated_episodes(control, violation):
    api = config_module()
    base = api.T1_BASE
    sampler = replace(base.sampler, origins=1)
    sampler = replace(
        sampler, **({"max_cells": 100} if violation == "budget" else {"designate_target": False})
    )
    with pytest.raises(ValueError, match="max_cells|designate_target"):
        replace(base, sources=(api.SourceConfig("typed", 1.0, control=control),), sampler=sampler)


@pytest.mark.parametrize("removal", ["drop_always", "summary_off"])
@pytest.mark.parametrize("violation", ["budget", "target"])
def test_fixed_removals_require_unbudgeted_designated_episodes(removal, violation):
    base = config_module().T1_BASE
    name = "skewness" if removal == "summary_off" else "calendar"
    changes = {removal: (name,)}
    changes.update({"max_cells": 100} if violation == "budget" else {"designate_target": False})
    with pytest.raises(ValueError, match="max_cells|designate_target"):
        replace(base, sampler=replace(base.sampler, **changes))
    assert replace(base, sampler=replace(base.sampler, **{removal: (name,)}))


@pytest.mark.parametrize("field", ["router", "gates"])
def test_router_and_gates_reject_unknown_modes(field):
    with pytest.raises(ValueError, match=field):
        replace(config_module().T1_BASE, **{field: "unrecognised"})


_INVALID_ROLLED_SAMPLER_OPTIONS = [("episodes_per_batch", 1), ("max_cells", 100)]


@pytest.mark.parametrize("field,value", _INVALID_ROLLED_SAMPLER_OPTIONS)
def test_rolled_config_rejects_sampler_options_that_allow_singletons(field, value):
    base = config_module().T1_BASE
    sampler = replace(base.sampler, **{field: value})
    with pytest.raises(ValueError, match=rf"rolled.*{field}"):
        replace(base, sampler=sampler, router="rolled")


@pytest.mark.parametrize("field,value", _INVALID_ROLLED_SAMPLER_OPTIONS)
def test_rolled_yaml_rejects_sampler_options_that_allow_singletons(tmp_path, field, value):
    base = config_module().T1_BASE
    raw = base.canonical()
    raw["router"] = "rolled"
    raw["sampler"][field] = value
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=rf"rolled.*{field}"):
        type(base).from_yaml(path)


_TYPED_OPTIONS = (
    ("control", "shuffle"),
    ("control", "no_dates"),
    ("control", False),
    ("control_inputs", "attribute"),
    ("control_inputs", "column"),
    ("heldout", True),
    ("recipes", (("static_number",),)),
    ("freq", "D"),
    ("length", (64, 128)),
    ("max_items", 2),
    ("n_nuisance", 1),
    ("nuisance_from", ("static_number",)),
    ("singleton_share", 0.5),
)


@pytest.mark.parametrize("kind", ["corpus", "upstream"])
@pytest.mark.parametrize("field,value", _TYPED_OPTIONS)
def test_non_typed_sources_reject_typed_options(kind, field, value):
    with pytest.raises(ValueError, match=field):
        config_module().SourceConfig(kind, 1.0, **{field: value})


@pytest.mark.parametrize("field,value", [row for row in _TYPED_OPTIONS if row[1] is not False])
def test_typed_sources_accept_typed_options(field, value):
    options = {"recipes": (("static_number",),)} if field in ("n_nuisance", "nuisance_from") else {}
    options[field] = value
    assert getattr(config_module().SourceConfig("typed", 1.0, **options), field) == value


_UNKNOWN_MECHANISMS = (
    ("recipes", (("no_such",),)),
    ("nuisance_from", ("no_such",)),
)


@pytest.mark.parametrize("field,value", _UNKNOWN_MECHANISMS)
def test_typed_sources_reject_unknown_mechanism_names(field, value):
    options = {"recipes": (("static_number",),), field: value}
    with pytest.raises(ValueError, match=rf"{field}.*no_such"):
        config_module().SourceConfig("typed", 1.0, **options)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_typed_recipes_accept_every_mechanism(mechanism):
    recipes = ((mechanism,),)
    assert config_module().SourceConfig("typed", 1.0, recipes=recipes).recipes == recipes


@pytest.mark.parametrize("mechanism", INPUT_MECHANISMS)
def test_typed_nuisance_from_accepts_every_input_mechanism(mechanism):
    names = (mechanism,)
    source = config_module().SourceConfig(
        "typed", 1.0, recipes=(names,), n_nuisance=1, nuisance_from=names
    )
    assert source.nuisance_from == names


def test_typed_nuisance_from_rejects_a_mechanism_without_inputs():
    with pytest.raises(ValueError, match=r"nuisance_from.*level_shift"):
        config_module().SourceConfig(
            "typed", 1.0, recipes=(("level_shift",),), n_nuisance=1, nuisance_from=("level_shift",)
        )


def test_typed_nuisance_from_requires_membership_in_every_recipe():
    with pytest.raises(ValueError, match=r"nuisance_from.*static_number.*recipe.*known_driver"):
        config_module().SourceConfig(
            "typed",
            1.0,
            recipes=(("static_number",), ("known_driver",)),
            n_nuisance=1,
            nuisance_from=("static_number",),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_nuisance", 1),
        ("nuisance_from", ("static_number",)),
        ("nuisance_from", ("no_such",)),
        ("nuisance_from", ()),
    ],
)
def test_typed_nuisance_options_require_explicit_recipes(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*requires recipes.*ignored"):
        config_module().SourceConfig("typed", 1.0, **{field: value})


@pytest.mark.parametrize("kind", ["corpus", "upstream"])
@pytest.mark.parametrize("field,value", _UNKNOWN_MECHANISMS)
def test_non_typed_sources_reject_mechanism_fields_before_checking_names(kind, field, value):
    with pytest.raises(ValueError, match=rf"^{field} is only valid for typed sources$"):
        config_module().SourceConfig(kind, 1.0, **{field: value})


@pytest.mark.parametrize("kind", ["typed", "corpus", "upstream"])
def test_missing_as_zero_is_valid_for_every_source_kind(kind):
    """The sampler fills missing history on any panel, so this control is not typed-only."""
    api = config_module()
    source = api.SourceConfig(kind, 1.0, control="missing_as_zero")
    assert source.control == "missing_as_zero"
    base = api.T1_BASE
    with pytest.raises(ValueError, match="origins"):
        replace(base, sources=(source,))
    cfg = replace(base, sources=(source,), sampler=replace(base.sampler, origins=1))
    assert cfg.sources[0].control == "missing_as_zero"


@pytest.mark.parametrize("kind", ["corpus", "upstream"])
def test_missing_as_zero_does_not_admit_other_typed_options(kind):
    with pytest.raises(ValueError, match="control_inputs"):
        config_module().SourceConfig(
            kind, 1.0, control="missing_as_zero", control_inputs="attribute"
        )


@pytest.mark.parametrize("kind", ["typed", "upstream"])
@pytest.mark.parametrize("root", ["/corpus", ""])
def test_root_is_only_valid_for_corpus_sources(kind, root):
    with pytest.raises(ValueError, match="root"):
        config_module().SourceConfig(kind, 1.0, root=root)


@pytest.mark.parametrize("kind", ["typed", "corpus", "upstream"])
def test_source_defaults_are_valid_for_every_kind(kind):
    assert config_module().SourceConfig(kind, 1.0).kind == kind


def test_sampler_rejects_context_beyond_schema_limit():
    with pytest.raises(ValueError, match="contexts"):
        config_module().SamplerConfig(contexts=(MAX_CONTEXT + PATCH,))
    assert config_module().SamplerConfig(contexts=(MAX_CONTEXT,)).contexts == (MAX_CONTEXT,)
    assert config_module().SamplerConfig(horizon=(PATCH, MAX_HORIZON)).horizon == (
        PATCH,
        MAX_HORIZON,
    )


def test_negative_seed_is_rejected_before_training():
    with pytest.raises(ValueError, match="seed"):
        replace(training_config(), seed=-1)
    assert replace(training_config(), seed=0).seed == 0


def test_source_length_respects_panel_recipe_minimum():
    with pytest.raises(ValueError, match="length"):
        config_module().SourceConfig("typed", 1.0, length=(4, 8))
    assert config_module().SourceConfig("typed", 1.0, length=(8, 8)).length == (8, 8)


# Each row catches a missing range/type guard, before invalid values reach the sampler/loop.
_NUMERIC_FIELDS = (
    ("source", "weight"),
    ("source", "max_items"),
    ("source", "n_nuisance"),
    ("source", "singleton_share"),
    ("source", "length"),
    ("sampler", "episodes_per_batch"),
    ("sampler", "episodes_per_panel"),
    ("sampler", "origins"),
    ("sampler", "origin_stride"),
    ("sampler", "contexts"),
    ("sampler", "horizon"),
    ("sampler", "cutoff_at_origin"),
    ("sampler", "path_share"),
    ("sampler", "drop"),
    ("sampler", "max_cells"),
    ("optim", "max_steps"),
    ("optim", "learning_rate"),
    ("optim", "weight_decay"),
    ("optim", "warmup_steps"),
    ("optim", "grad_clip"),
    ("optim", "ema_decay"),
    ("optim", "checkpoint_minutes"),
    ("optim", "log_every"),
    ("train", "seed"),
    ("train", "workers"),
)


def config_section(section):
    cfg = training_config()
    return {"source": cfg.sources[0], "sampler": cfg.sampler, "optim": cfg.optim, "train": cfg}[
        section
    ]


@pytest.mark.parametrize("section,field", _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_numeric_fields_reject_nonfinite_and_nonnumeric_values(section, field, value):
    if field in ("length", "horizon"):
        value = (value, 128)
    elif field == "contexts":
        value = (value,)
    elif field == "drop":
        value = (("features", value),)
    with pytest.raises(ValueError, match=field):
        replace(config_section(section), **{field: value})


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("source", "weight", 0),
        ("source", "weight", -1),
        ("source", "max_items", 0),
        ("source", "n_nuisance", -1),
        ("source", "singleton_share", -0.1),
        ("source", "singleton_share", 1.1),
        ("source", "length", (0, 64)),
        ("source", "length", (128, 64)),
        ("source", "length", (64,)),
        ("sampler", "episodes_per_batch", 0),
        ("sampler", "episodes_per_panel", 0),
        ("sampler", "origins", 0),
        ("sampler", "origin_stride", 0),
        ("sampler", "contexts", ()),
        ("sampler", "contexts", (0,)),
        ("sampler", "contexts", (17,)),
        ("sampler", "horizon", (0, 16)),
        ("sampler", "horizon", (16, 289)),
        ("sampler", "horizon", (32, 16)),
        ("sampler", "horizon", (16,)),
        ("sampler", "cutoff_at_origin", -0.1),
        ("sampler", "cutoff_at_origin", 1.1),
        ("sampler", "path_share", -0.1),
        ("sampler", "path_share", 1.1),
        ("sampler", "drop", (("features", -0.1),)),
        ("sampler", "drop", (("features", 1.1),)),
        ("sampler", "max_cells", 0),
        ("optim", "max_steps", 0),
        ("optim", "learning_rate", 0),
        ("optim", "weight_decay", -1),
        ("optim", "warmup_steps", -1),
        ("optim", "grad_clip", 0),
        ("optim", "ema_decay", 0),
        ("optim", "ema_decay", 1),
        ("optim", "checkpoint_minutes", 0),
        ("optim", "log_every", 0),
        ("train", "workers", -1),
    ],
)
def test_numeric_fields_reject_values_outside_contract(section, field, value):
    with pytest.raises(ValueError, match=field):
        replace(config_section(section), **{field: value})


@pytest.mark.parametrize(
    "section,field",
    [
        ("source", "max_items"),
        ("source", "n_nuisance"),
        ("source", "length"),
        ("sampler", "episodes_per_batch"),
        ("sampler", "episodes_per_panel"),
        ("sampler", "origins"),
        ("sampler", "origin_stride"),
        ("sampler", "contexts"),
        ("sampler", "horizon"),
        ("sampler", "max_cells"),
        ("optim", "max_steps"),
        ("optim", "warmup_steps"),
        ("optim", "log_every"),
        ("train", "seed"),
        ("train", "workers"),
    ],
)
def test_integer_fields_do_not_silently_truncate(section, field):
    value = (
        (16.5, 32) if field in ("length", "horizon") else (16.5,) if field == "contexts" else 1.5
    )
    with pytest.raises(ValueError, match=field):
        replace(config_section(section), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("drop", (("typo", 0.1),)),
        ("drop", (("features", 0.1), ("features", 0.2))),
        ("drop_always", ("typo",)),
        ("summary_off", ("mean",)),
    ],
)
def test_sampler_rejects_unknown_or_duplicate_family_names(field, value):
    with pytest.raises(ValueError, match=field):
        replace(config_module().T1_BASE.sampler, **{field: value})


@pytest.mark.parametrize(
    "device", ["banana", "cuda:-1", "cuda:", "cuda:1.5", "cuda:1x", "CPU", None]
)
def test_train_rejects_invalid_device(device):
    with pytest.raises(ValueError, match="device"):
        replace(training_config(), device=device)


@pytest.mark.parametrize(
    "section,field", [pair for pair in _NUMERIC_FIELDS if pair[1] != "ema_decay"]
)
def test_numeric_normalization_keeps_canonical_and_identity_in_agreement(section, field):
    api = config_module()
    cfg = training_config()
    record = config_section(section)
    if field == "drop":
        integral, floating = (("features", 1),), (("features", 1.0),)
        expected_type = float
    elif field in ("length", "horizon", "contexts"):
        integral, floating = (16, 32), (16.0, 32.0)
        expected_type = int
    else:
        integral, floating = 1, 1.0
        expected_type = (
            int
            if field
            in {
                "max_items",
                "n_nuisance",
                "episodes_per_batch",
                "episodes_per_panel",
                "origins",
                "origin_stride",
                "max_cells",
                "max_steps",
                "warmup_steps",
                "log_every",
                "seed",
                "workers",
            }
            else float
        )
    left = replace(record, **{field: integral})
    right = replace(record, **{field: floating})
    value = getattr(left, field)
    if field == "drop":
        value = value[0][1]
    elif isinstance(value, tuple):
        value = value[0]
    if value is not None:
        assert type(value) is expected_type
        other = getattr(right, field)
        if field == "drop":
            other = other[0][1]
        elif isinstance(other, tuple):
            other = other[0]
        assert type(other) is expected_type
    if section == "train":
        a, b = left, right
    elif section == "source":
        a, b = replace(cfg, sources=(left,)), replace(cfg, sources=(right,))
    else:
        a, b = replace(cfg, **{section: left}), replace(cfg, **{section: right})
    parts = dict(code="rev", corpus=None, init=None, env=None)
    assert a.canonical() == b.canonical()
    assert api.identity(a, **parts) == api.identity(b, **parts)


@pytest.mark.parametrize(
    ("section", "field"),
    [("source", "heldout"), ("sampler", "designate_target"), ("optim", "amp")],
)
@pytest.mark.parametrize("value", ["false", "yes", 0, 1, None])
def test_flags_must_be_real_booleans(section, field, value):
    # A quoted "false" in a yaml is truthy: it must fail instead of silently switching a control on.
    cfg = config_module()
    base = {
        "source": cfg.SourceConfig(kind="typed", weight=1.0),
        "sampler": cfg.SamplerConfig(),
        "optim": cfg.OptimConfig(max_steps=10),
    }[section]
    with pytest.raises(ValueError, match=field):
        replace(base, **{field: value})
