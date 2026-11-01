"""Offline, CPU-only fixtures shared by pretraining integration tests."""

from dataclasses import asdict

import pytest
import yaml

from demand_forecast.pretrain.config import PretrainConfig
from tests.helpers.settings import tiny_settings


@pytest.fixture
def tiny_pretrain_cfg(tmp_path):
    pytest.importorskip("pyarrow")
    from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest
    from tests.helpers.corpus_fixture import write_corpus_fixture

    root = tmp_path / "corpus"
    for kind in ("tsmixup", "kernel_synth"):
        for index in range(2):
            write_corpus_fixture(root, kind=kind, index=index, seed=index)
    write_manifest(corpus_manifest(root), root / "manifest.json")
    return PretrainConfig(
        corpus=dict(
            root=root,
            tsmixup_shards=[0],
            kernel_synth_shards=[0],
            holdout_tsmixup=[1],
            holdout_kernel_synth=[1],
            val_pairs_per_kind=8,
            external_validation=None,
        ),
        forecast=dict(horizon=4, max_context=16, min_context=8),
        model=tiny_settings().model,
        training=dict(
            max_steps=12,
            batch_size=4,
            eval_every=6,
            checkpoint_every=3,
            amp=False,
            warmup_steps=2,
        ),
        device="cpu",
    )


@pytest.fixture
def tiny_pretrain_yaml(tmp_path, tiny_pretrain_cfg):
    path = tmp_path / "pretrain.yaml"
    tiny_pretrain_cfg.to_yaml(path)
    return path


@pytest.fixture
def nonfinite_cfg(tiny_pretrain_cfg):
    cfg = tiny_pretrain_cfg.model_copy(deep=True)
    cfg.training.max_steps = 40
    return cfg


@pytest.fixture
def fixture_tasks_yaml(tmp_path):
    pytest.importorskip("fev")
    from tests.helpers.fev_fixture import write_fev_fixture
    from tests.helpers.fev_spec import fixture_spec

    spec = asdict(fixture_spec(write_fev_fixture(tmp_path / "fev", n_series=8)))
    del spec["categorical_statics"], spec["target_support"]
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump({"tasks": [spec]}))
    return path


@pytest.fixture
def fixture_profiles_yaml(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        yaml.safe_dump(
            {"fixture": {"target_support": "nonnegative", "categorical_statics": ["cat"]}}
        )
    )
    return path


@pytest.fixture
def tiny_experiment_yaml(tmp_path):
    path = tmp_path / "experiment.yaml"
    tiny_settings().to_yaml(path)
    return path
