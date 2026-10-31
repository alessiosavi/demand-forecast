"""Configuration migration and validation of the public v2 schema."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from demand_forecast.config import (
    DatasetConfig,
    ForecastConfig,
    ModelConfig,
    Settings,
    TrainingConfig,
    TuningConfig,
)
from tests.helpers.run_config import settings_with

MINIMAL_YAML = """
data:
  path: data/long.csv
  id_columns: [id]
  target: y
  target_support: nonnegative
forecast:
  horizon: 4
"""
LEGACY_HACKATHON_YAML = """
data:
  catalogue_path: data/products.csv
  input_path: data/sales.csv
  resample_period: W-MON
timeseries:
  window: 52
  n_out: 13
output:
  model_dir: models
"""


def test_settings_schema():
    with pytest.raises(ValidationError):
        Settings(**yaml.safe_load(LEGACY_HACKATHON_YAML))
    with pytest.raises(ValidationError, match="patch"):
        ModelConfig(use_patch_embedding=True)
    settings = Settings(**yaml.safe_load(MINIMAL_YAML))
    assert ModelConfig().d_model == 64 and settings.training.refit == "none"
    assert settings.seed == 0 and settings.device is None
    assert settings.data.path == Path("data/long.csv")


def test_model_defaults_match_standard_backbone():
    from demand_forecast.models.v2 import V2_STANDARD

    config = ModelConfig()
    assert config.backbone_kwargs() == asdict(V2_STANDARD)
    assert config.cat_emb_dim == 16
    assert config.family == "v2"
    assert not config.use_patch_embedding and not config.use_decomposition


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"use_decomposition": True}, "decomposition"),
        ({"d_model": 14, "nhead": 4}, "divisible"),
        ({"d_model": 12, "nhead": 4}, "even"),
        ({"family": "standard"}, "family"),
        ({"num_encoder_layers": 0}, "num_encoder_layers"),
    ],
)
def test_model_invalid_options(options, message):
    with pytest.raises(ValidationError, match=message):
        ModelConfig(**options)


def test_training_defaults_match_trainer():
    from demand_forecast.core.training import TrainConfig

    config = TrainingConfig()
    assert config.model_dump(exclude={"refit", "min_train_history", "val_items_max"}) == asdict(
        TrainConfig()
    )
    assert config.min_train_history is None and config.refit == "none"
    assert "seed" not in TrainingConfig.model_fields


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_steps", 0),
        ("batch_size", 0),
        ("learning_rate", 0),
        ("warmup_steps", -1),
        ("checkpoint_every", 0),
        ("patience", 0),
        ("eval_every", 0),
        ("min_train_history", 0),
    ],
)
def test_training_invalid_options(field, value):
    with pytest.raises(ValidationError, match=field):
        TrainingConfig(**{field: value})


def test_unknown_legacy_controls_are_rejected():
    with pytest.raises(ValidationError, match="timeseries"):
        Settings(**yaml.safe_load(MINIMAL_YAML), timeseries={"window": 10})
    with pytest.raises(ValidationError, match="model_type"):
        ModelConfig(model_type="advanced")
    with pytest.raises(ValidationError, match="seed"):
        TrainingConfig(seed=7)


def test_yaml_roundtrip(tmp_path):
    original = Settings(**yaml.safe_load(MINIMAL_YAML), seed=999)
    path = tmp_path / "config.yaml"
    original.to_yaml(path)
    assert Settings.from_yaml(path) == original


@pytest.mark.parametrize("path", ["config.example.yaml", "configs/example-long.yaml"])
def test_example_configs(path):
    settings = Settings.from_yaml(Path(__file__).resolve().parents[1] / path)
    assert settings.data.source == "long"
    assert settings.model == ModelConfig()
    assert settings.training == TrainingConfig()


def test_legacy_tuning_settings_preserved():
    # The Pydantic schema and the live legacy tuner have separate configs.
    from demand_forecast.core.tuning import HyperparameterTuner

    assert HyperparameterTuner is not None
    assert TuningConfig().model_dump() == {
        "enabled": False,
        "n_trials": 50,
        "timeout": None,
        "direction": "minimize",
        "metric": "mse",
        "pruner": "median",
        "sampler": "tpe",
        "study_name": "demand_forecast_tuning",
        "storage": None,
        "n_jobs": 1,
        "epochs_per_trial": 5,
        "early_stop_patience": 2,
    }
    assert TuningConfig(n_trials=0).n_trials == 0


def test_refit_full_is_explicitly_unimplemented(tmp_path):
    from demand_forecast.core.run import train_from_settings

    with pytest.raises(NotImplementedError, match="refit=full arrives with the phase-4 plan"):
        train_from_settings(settings_with(refit="full"), tmp_path)
    assert not list(tmp_path.iterdir())


def test_dataset_config_requires_target_support():
    with pytest.raises(ValidationError, match="target_support"):
        DatasetConfig(id_columns=["id"], target="y")


def test_forecast_config_quantiles():
    for quantiles in (
        [0.1, 0.9],
        [0.9, 0.5, 0.1],
        [0.1, 0.5, 0.5, 0.9],
        [0.0, 0.5, 0.9],
        [0.1, 0.5, 1.0],
        [-0.1, 0.5, 0.9],
        [0.1, 0.5, 1.1],
        [float("nan"), 0.5],
        [0.5, float("inf")],
        [],
    ):
        with pytest.raises(ValidationError):
            ForecastConfig(horizon=4, quantiles=quantiles)
    assert ForecastConfig(horizon=4).median_index == 4
    assert ForecastConfig(horizon=4, quantiles=[0.1, 0.5, 0.9]).median_index == 1
    assert ForecastConfig(horizon=4, quantiles=[0.5]).median_index == 0


def test_frozen_fev_standard_config_matches_protocol():
    import re

    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs/protocol.md").read_text()
    frozen = yaml.safe_load(re.search(r"```yaml\n(.*?)\n```", protocol, re.DOTALL).group(1))
    settings = Settings.from_yaml(root / "configs/fev-v2-standard.yaml")
    actual = settings.model_dump()
    for key, expected in frozen.items():
        if isinstance(expected, dict):
            assert {name: actual[key][name] for name in expected} == expected
        else:
            assert actual[key] == expected
    assert settings.model.d_model == 64 and settings.training.base_steps == 3000
    assert settings.seed == 0 and settings.device is None and settings.training.amp
    assert settings.training.checkpoint_every == 200


def test_quick_config_differs_from_frozen_only_in_declared_fields():
    root = Path(__file__).resolve().parents[1]
    frozen = Settings.from_yaml(root / "configs/fev-v2-standard.yaml").model_dump()
    quick = Settings.from_yaml(root / "configs/fev-v2-quick.yaml").model_dump()
    assert quick["training"].pop("max_steps_cap") == 300
    assert frozen["training"].pop("max_steps_cap") is None
    assert quick["training"].pop("val_items_max") == 1024
    assert frozen["training"].pop("val_items_max") is None
    for key, values in {
        "eval_every": (200, 50),
        "patience": (5, 2),
        "checkpoint_every": (200, 50),
    }.items():
        assert (frozen["training"].pop(key), quick["training"].pop(key)) == values
    assert quick == frozen


@pytest.mark.parametrize("field", ["max_steps_cap", "val_items_max"])
@pytest.mark.parametrize("value", [0, -1])
def test_training_config_rejects_nonpositive_quick_fields(field, value):
    with pytest.raises(ValidationError, match=field):
        TrainingConfig(**{field: value})
