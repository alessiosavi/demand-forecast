"""The latency CLI builds inputs matching each supported bundle schema."""

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from demand_forecast.config import ForecastConfig
from demand_forecast.core.bundle import save_bundle
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.models.base import ModelSpec
from tests.helpers.models import build_v2
from tests.helpers.panels import weekly_panel


@pytest.mark.parametrize(
    "calendar_schema,target_lags",
    [("legacy", "none"), ("canonical", "none"), ("legacy", "cycle"), ("canonical", "cycle")],
)
def test_latency_honours_bundle_input_schema(tmp_path, calendar_schema, target_lags):
    panel = weekly_panel([64] * 3)
    encoder = StaticEncoder().fit(panel, [series.series_id for series in panel.series])
    forecast = ForecastConfig(
        horizon=4, max_context=64, calendar_schema=calendar_schema, target_lags=target_lags
    )
    model = build_v2(ModelSpec.from_panel(panel, encoder, forecast, 8, 4), d_model=8)
    directory = tmp_path / "bundle"
    save_bundle(
        directory,
        model=model,
        family="v2",
        backbone_kwargs=asdict(model.backbone.cfg),
        forecast_cfg=forecast,
        dataset_cfg=None,
        encoder=encoder,
        panel=panel,
        training_cutoff=panel.timestamps(0)[-1],
        earliest_origin=panel.timestamps(0, 65)[-1],
        manifest={},
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/latency.py"),
            "--bundle",
            str(directory),
            "--n-series",
            "3",
            "--repeats",
            "1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "family=v2 device=cpu threads=1 n_series=3 context=64 horizon=4" in result.stdout
    assert "warmups=5 repeats=1" in result.stdout
    assert "ms/batch;" in result.stdout and "ms/series" in result.stdout
