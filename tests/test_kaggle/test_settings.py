from copy import deepcopy
from pathlib import Path

import pytest

from demand_forecast.config import DatasetConfig
from demand_forecast.kaggle.settings import build_settings, load_template, write_settings

TEMPLATE = Path("configs/kaggle-v2-l3-finetune.yaml")
ZERO_SHOT = Path("configs/kaggle-v2-l3-zero-shot.yaml")

DATASET = DatasetConfig(
    source="long",
    id_columns=["store_nbr", "family"],
    target="sales",
    target_support="nonnegative",
    freq="D",
    missing_target="zero",
)


def test_finetune_template_loads_with_pinned_recipe():
    settings = build_settings(
        load_template(TEMPLATE),
        data=DatasetConfig(
            source="long",
            id_columns=["store_nbr", "family"],
            target="sales",
            target_support="nonnegative",
            freq="D",
            missing_target="zero",
        ),
        horizon=16,
    )
    assert settings.model.d_model == 320
    assert settings.training.learning_rate == pytest.approx(0.0002558484623288223)
    assert settings.training.refit == "full"
    assert settings.training.zero_shot is False
    assert settings.forecast.horizon == 16


def test_zero_shot_template_sets_zero_shot():
    settings = build_settings(
        load_template(ZERO_SHOT),
        data=DatasetConfig(
            source="long",
            id_columns=["store_nbr", "family"],
            target="sales",
            target_support="nonnegative",
            freq="D",
            missing_target="zero",
        ),
        horizon=16,
    )
    assert settings.training.zero_shot is True
    assert settings.training.init_from is None  # supplied by the workload


def test_write_settings_round_trips(tmp_path):
    settings = build_settings(
        load_template(TEMPLATE),
        data=DatasetConfig(
            source="long",
            id_columns=["store_nbr", "family"],
            target="sales",
            target_support="nonnegative",
            freq="D",
            missing_target="zero",
        ),
        horizon=16,
    )
    path = tmp_path / "settings.yaml"
    write_settings(settings, path)
    assert (
        build_settings(load_template(path), data=settings.data, horizon=16).training.refit == "full"
    )


def test_build_settings_does_not_mutate_the_template():
    template = load_template(TEMPLATE)
    snapshot = deepcopy(template)
    build_settings(template, data=DATASET, horizon=16)
    assert template == snapshot


def test_build_settings_overrides_horizon_and_data():
    settings = build_settings(load_template(TEMPLATE), data=DATASET, horizon=7)
    assert settings.forecast.horizon == 7
    assert settings.data == DATASET
