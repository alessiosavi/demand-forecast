"""Small panels and forecast origins for the new core training path."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core.training import TrainConfig, Trainer
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import to_offset
from demand_forecast.models.base import ModelSpec
from tests.helpers.models import build_v2
from tests.helpers.panels import weekly_panel


@pytest.fixture
def panel29():
    panel = weekly_panel([29] * 3)
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin29(panel29):
    return panel29.timestamps(0)[-1] + panel29.freq


@pytest.fixture
def panel8():
    panel = weekly_panel([8] * 2)
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin8(panel8):
    return panel8.timestamps(0)[-1] + panel8.freq


@pytest.fixture
def panel_constant_long():
    panel = weekly_panel([60] * 2)
    for series in panel.series:
        series.target[:] = 6
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin_cl(panel_constant_long):
    return panel_constant_long.timestamps(0)[-1] + panel_constant_long.freq


@pytest.fixture
def panel_mixed():
    panel = weekly_panel([60, 20])
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def origin_mixed(panel_mixed):
    return panel_mixed.timestamps(0)[-1] + panel_mixed.freq


@pytest.fixture
def panel_long():
    panel = weekly_panel([80] * 20)
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def plan29(panel29, origin29):
    return plan_inner_validation(panel29, origin29, 13)


@pytest.fixture
def plan8(panel8, origin8):
    return plan_inner_validation(panel8, origin8, 8)


@pytest.fixture
def plan_long(panel_long):
    origin = panel_long.timestamps(0)[-1] + panel_long.freq
    return plan_inner_validation(panel_long, origin, 4)


@pytest.fixture
def plan_cl(panel_constant_long, origin_cl):
    return plan_inner_validation(panel_constant_long, origin_cl, 4)


@pytest.fixture
def encoder29(panel29, plan29):
    return StaticEncoder().fit(panel29, plan29.fit_ids)


@pytest.fixture
def encoder8(panel8, plan8):
    return StaticEncoder().fit(panel8, plan8.fit_ids)


@pytest.fixture
def encoder_long(panel_long, plan_long):
    return StaticEncoder().fit(panel_long, plan_long.fit_ids)


@pytest.fixture
def encoder_cl(panel_constant_long, plan_cl):
    return StaticEncoder().fit(panel_constant_long, plan_cl.fit_ids)


@pytest.fixture
def panel_singletons():
    panel = weekly_panel([1, 1])
    panel.series[0].target[:] = 5.0
    panel.series[1].target[:] = 7.0
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def plan_singletons(panel_singletons):
    origin = panel_singletons.series[0].start + panel_singletons.freq
    return plan_inner_validation(panel_singletons, origin, 4)


@pytest.fixture
def encoder_singletons(panel_singletons, plan_singletons):
    return StaticEncoder().fit(panel_singletons, plan_singletons.fit_ids)


@pytest.fixture
def panel_all_missing():
    panel = weekly_panel([8, 8])
    for series in panel.series:
        series.target[:] = np.nan
    return replace(panel, freq=to_offset(panel.freq))


@pytest.fixture
def cutoff_am(panel_all_missing):
    return panel_all_missing.timestamps(0)[-1]


@pytest.fixture
def origin_am(panel_all_missing, cutoff_am):
    return cutoff_am + panel_all_missing.freq


@pytest.fixture
def plan_all_missing(panel_all_missing, origin_am):
    return plan_inner_validation(panel_all_missing, origin_am, 4)


@pytest.fixture
def encoder_all_missing(panel_all_missing, plan_all_missing):
    return StaticEncoder().fit(panel_all_missing, plan_all_missing.fit_ids)


@pytest.fixture
def forecast29():
    return ForecastConfig(horizon=13, max_context=64)


@pytest.fixture
def spec29(panel29, encoder29, forecast29):
    return ModelSpec.from_panel(panel29, encoder29, forecast29, 16, 4)


@pytest.fixture
def cutoff29(plan29):
    return plan29.train_upto


@pytest.fixture
def tiny_v2_trained(spec29, panel29, plan29, forecast29, encoder29):
    torch.manual_seed(3)
    model = build_v2(spec29)
    result = Trainer(
        model, TrainConfig(base_steps=30, batch_size=4, amp=False), "cpu", "test-run", 3
    ).fit(panel29, plan29, forecast29, encoder29)
    assert result.steps == 30
    return model


@pytest.fixture
def bundle_dir(tmp_path, tiny_v2_trained, forecast29, encoder29, panel29, cutoff29, origin29):
    from demand_forecast.core.bundle import save_bundle

    directory = tmp_path / "bundle"
    save_bundle(
        directory,
        model=tiny_v2_trained,
        family="v2",
        backbone_kwargs=asdict(tiny_v2_trained.backbone.cfg),
        forecast_cfg=forecast29,
        dataset_cfg=None,
        encoder=encoder29,
        panel=panel29,
        training_cutoff=cutoff29,
        earliest_origin=origin29,
        manifest={"seed": 3, "run_identity": "test-run"},
    )
    return directory
