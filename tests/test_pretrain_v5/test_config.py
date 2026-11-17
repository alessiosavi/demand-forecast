"""Portable configuration, experiment identity, and comparison constraints."""

import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import yaml

from demand_forecast.models.v5 import V5Config, V5Model


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
                recipes=(("trend", "seasonality"),),
                freq="D",
                length=(128, 256),
                nuisance_from=("noise",),
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
    assert raw["sources"][0]["recipes"] == [["trend", "seasonality"]]
    assert raw["sampler"]["contexts"] == [256, 512, 1024]
    assert raw["model"]["quantiles"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    restored = type(cfg).from_yaml(path)
    assert restored == cfg
    assert isinstance(restored.sources, tuple)
    assert isinstance(restored.sources[0].recipes[0], tuple)
    assert isinstance(restored.sampler.drop[0], tuple)
    assert isinstance(restored.model.quantiles, tuple)


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
        sources=tuple(replace(source, root="/elsewhere") for source in cfg.sources),
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
