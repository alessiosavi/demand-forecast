"""Portable pretraining configuration, run identity, and shared LR schedule."""

import hashlib
import json
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from demand_forecast.config.settings import ForecastConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = ROOT / "configs/pretrain-v2-first-try.yaml"


@pytest.mark.parametrize("section", [None, "corpus", "forecast", "model", "training"])
def test_pretrain_config_loads_shipped_yaml_and_rejects_unknown_keys(tmp_path, section):
    from demand_forecast.pretrain.config import PretrainConfig

    cfg = PretrainConfig.from_yaml(SHIPPED_CONFIG)
    assert cfg.corpus.root == Path("/home/ubuntu/fev/corpus")
    assert cfg.corpus.tsmixup_shards == list(range(199))
    assert cfg.corpus.kernel_synth_shards == list(range(19))
    assert cfg.corpus.model_dump(exclude={"root", "tsmixup_shards", "kernel_synth_shards"}) == {
        "kernel_synth_every": 10,
        "holdout_tsmixup": [199],
        "holdout_kernel_synth": [19],
        "val_pairs_per_kind": 4096,
        "external_validation": "monash_nn5_weekly",
    }
    assert isinstance(cfg.forecast, ForecastConfig)
    assert (
        cfg.forecast.model_dump()
        == ForecastConfig(horizon=52, max_context=512, min_context=8).model_dump()
    )
    assert cfg.model == ModelConfig(
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        hidden_continuous_size=256,
    )
    assert cfg.training.model_dump() == {
        "max_steps": 150000,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "warmup_steps": 2000,
        "grad_clip": 1.0,
        "eval_every": 5000,
        "checkpoint_every": 1000,
        "amp": True,
        "val_batch_size": 128,
    }
    assert (cfg.seed, cfg.device, cfg.log_level) == (0, None, "INFO")

    payload = cfg.model_dump(mode="json")
    target = payload if section is None else payload[section]
    target["unknown_option"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="unknown_option"):
        PretrainConfig.from_yaml(invalid)


def test_pretrain_config_yaml_roundtrip_accepts_shared_models(tmp_path):
    from demand_forecast.pretrain.config import CorpusConfig, PretrainConfig, PretrainTrainingConfig

    cfg = PretrainConfig(
        corpus=CorpusConfig(
            root=tmp_path / "absent-corpus",
            tsmixup_shards=[3, 1],
            kernel_synth_shards=[2],
            external_validation=None,
        ),
        forecast=ForecastConfig(horizon=4, max_context=16, min_context=8),
        model=ModelConfig(d_model=16, nhead=2),
        training=PretrainTrainingConfig(max_steps=1),
        seed=17,
        device="cpu",
        log_level="DEBUG",
    )
    path = tmp_path / "pretrain.yaml"
    cfg.to_yaml(path)
    assert PretrainConfig.from_yaml(path) == cfg
    assert yaml.safe_load(path.read_text())["corpus"]["root"] == str(cfg.corpus.root)


@pytest.mark.parametrize("max_steps", [0, -1])
def test_pretrain_training_rejects_nonpositive_max_steps(max_steps):
    from demand_forecast.pretrain.config import PretrainTrainingConfig

    with pytest.raises(ValidationError, match="max_steps"):
        PretrainTrainingConfig(max_steps=max_steps)


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 0),
        ("learning_rate", -1e-4),
        ("grad_clip", 0),
        ("grad_clip", -1),
        ("weight_decay", -0.01),
        ("warmup_steps", -1),
        ("batch_size", 0),
        ("eval_every", 0),
        ("checkpoint_every", 0),
        ("val_batch_size", 0),
    ],
)
def test_pretrain_training_rejects_invalid_controls(field, value):
    from demand_forecast.pretrain.config import PretrainTrainingConfig

    with pytest.raises(ValidationError, match=field):
        PretrainTrainingConfig(max_steps=1, **{field: value})


def test_pretrain_training_accepts_boundary_controls():
    from demand_forecast.pretrain.config import PretrainTrainingConfig

    cfg = PretrainTrainingConfig(
        max_steps=1,
        learning_rate=1e-8,
        grad_clip=1e-8,
        weight_decay=0,
        warmup_steps=0,
        batch_size=1,
        eval_every=1,
        checkpoint_every=1,
        val_batch_size=1,
    )
    assert cfg.weight_decay == cfg.warmup_steps == 0
    assert cfg.learning_rate == cfg.grad_clip == 1e-8


def test_pretrain_identity_ignores_root_and_tracks_shards():
    from demand_forecast.pretrain.config import PretrainConfig, pretrain_identity

    cfg = PretrainConfig.from_yaml(SHIPPED_CONFIG)
    shards = {
        "training_corpus/tsmixup_10m/train-00000-of-00200.parquet": "a" * 64,
        "training_corpus/kernel_synth_1m/train-00000-of-00020.parquet": "b" * 64,
    }
    before = cfg.model_dump(mode="json")
    shards_before = shards.copy()
    portable = deepcopy(before)
    del portable["corpus"]["root"]
    payload = {"config": portable, "package_version": "2.0.0", "shards": shards}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = pretrain_identity(cfg, "2.0.0", shards)
    assert identity == expected
    assert pretrain_identity(cfg, "2.0.0", dict(reversed(list(shards.items())))) == identity

    moved = cfg.model_copy(deep=True)
    moved.corpus.root = Path("/another/machine/corpus")
    assert pretrain_identity(moved, "2.0.0", shards) == identity
    changed_shards = shards | {next(iter(shards)): "c" * 64}
    assert pretrain_identity(cfg, "2.0.0", changed_shards) != identity
    assert pretrain_identity(cfg, "2.0.1", shards) != identity
    changed = cfg.model_copy(deep=True)
    changed.training.max_steps += 1
    assert pretrain_identity(changed, "2.0.0", shards) != identity
    assert cfg.model_dump(mode="json") == before
    assert shards == shards_before


@pytest.mark.parametrize(
    "max_steps,warmup_steps,expected",
    [
        (6, 2, [0.0, 0.5, 1.0, 0.8535533905932737, 0.5, 0.1464466094067262, 0.0, 0.0]),
        (4, 0, [1.0, 0.8535533905932737, 0.5, 0.1464466094067262, 0.0, 0.0]),
        (1, 0, [1.0, 0.0, 0.0]),
        (1, 1, [0.0, 0.0, 0.0]),
        (2, 4, [0.0, 0.25, 0.0, 0.0]),
    ],
)
def test_lr_factor_matches_trainer_schedule(max_steps, warmup_steps, expected):
    from demand_forecast.core.training import Trainer, lr_factor

    # The schedule only reads these two attributes; no model or training is needed.
    trainer = Trainer.__new__(Trainer)
    trainer.max_steps = max_steps
    trainer.warmup_steps = warmup_steps
    actual = [
        lr_factor(step, max_steps=max_steps, warmup_steps=warmup_steps)
        for step in range(len(expected))
    ]
    assert actual == pytest.approx(expected, abs=1e-15, rel=0)
    assert [trainer._lr_factor(step) for step in range(len(expected))] == actual


def test_pretrain_config_imports_without_bench():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import sys

                for name in ("pyarrow", "datasets", "huggingface_hub", "fev", "statsforecast"):
                    sys.modules[name] = None

                from demand_forecast.pretrain.config import PretrainConfig, pretrain_identity

                cfg = PretrainConfig.from_yaml("configs/pretrain-v2-first-try.yaml")
                assert cfg.training.max_steps == 150000
                assert len(pretrain_identity(cfg, "2.0.0", {})) == 64
            """),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
