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
    changes = {removal: ("calendar",)}
    changes.update({"max_cells": 100} if violation == "budget" else {"designate_target": False})
    with pytest.raises(ValueError, match="max_cells|designate_target"):
        replace(base, sampler=replace(base.sampler, **changes))
    assert replace(base, sampler=replace(base.sampler, **{removal: ("calendar",)}))


@pytest.mark.parametrize("field", ["router", "gates"])
def test_router_and_gates_reject_unknown_modes(field):
    with pytest.raises(ValueError, match=field):
        replace(config_module().T1_BASE, **{field: "unrecognised"})
