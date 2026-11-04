from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def config(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_seed1_configs_differ_only_in_seed():
    for base, seeded in [
        ("world_signals/probe-scratch.yaml", "world_signals/probe-scratch-seed1.yaml"),
        ("fev-v2-pretrained-finetune.yaml", "world_signals/l3-finetune-seed1.yaml"),
    ]:
        original, copy = config(base), config(seeded)
        assert original.pop("seed") == 0
        assert copy.pop("seed") == 1
        assert original == copy


def test_smoke_config_is_not_cpu():
    smoke, quick = config("world_signals/probe-smoke.yaml"), config("fev-v2-quick.yaml")
    assert smoke["device"] is None
    assert smoke["training"]["max_steps_cap"] == 150
    quick["training"]["max_steps_cap"] = 150
    quick["device"] = None
    assert smoke == quick


def test_scratch_capacity_and_budget():
    scratch = config("world_signals/probe-scratch.yaml")
    assert {
        k: scratch["model"][k]
        for k in ("d_model", "num_encoder_layers", "num_decoder_layers", "dim_feedforward")
    } == dict(d_model=64, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256)
    assert {
        k: scratch["training"][k] for k in ("base_steps", "max_steps_cap", "eval_every", "patience")
    } == dict(base_steps=1500, max_steps_cap=1500, eval_every=100, patience=3)
    assert (
        "500 steps / eval 100"
        in (ROOT / "configs/world_signals/probe-scratch.yaml").read_text().splitlines()[0]
    )
