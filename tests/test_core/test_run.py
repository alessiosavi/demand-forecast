import json

import pandas as pd
import pytest

from demand_forecast.core.bundle import load_bundle
from demand_forecast.core.run import train_panel
from demand_forecast.data.frequency import to_offset
from tests.helpers.settings import tiny_settings


@pytest.fixture
def tiny_panel_and_settings(panel29, origin29):
    return panel29, origin29, tiny_settings(base_steps=2, d_model=8)


def test_train_panel_records_earliest_origin_and_checks_it_on_reuse(
    tmp_path, tiny_panel_and_settings
):
    panel, origin, settings = tiny_panel_and_settings
    step = to_offset(panel.freq)
    bundle = train_panel(panel, origin, settings, tmp_path / "a", dataset_fingerprint="fp")
    assert load_bundle(bundle).panel_metadata["earliest_origin"] == origin.isoformat()
    assert (
        json.loads((tmp_path / "a/run.json").read_text())["earliest_origin"] == origin.isoformat()
    )
    earlier = origin - 3 * step
    bundle = train_panel(
        panel, origin, settings, tmp_path / "b", dataset_fingerprint="fp", earliest_origin=earlier
    )
    loaded = load_bundle(bundle)
    assert loaded.panel_metadata["earliest_origin"] == earlier.isoformat()
    assert loaded.manifest["run"]["earliest_origin"] == earlier.isoformat()
    assert (
        json.loads((tmp_path / "b/run.json").read_text())["earliest_origin"] == earlier.isoformat()
    )
    with pytest.raises(ValueError, match="^earliest_origin after origin$"):
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "c",
            dataset_fingerprint="fp",
            earliest_origin=origin + step,
        )
    before = {path: path.read_bytes() for path in (tmp_path / "b").rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="^bundle earliest_origin mismatch$"):
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "b",
            dataset_fingerprint="fp",
            resume=True,
            earliest_origin=origin,
        )
    assert (
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "b",
            dataset_fingerprint="fp",
            resume=True,
            earliest_origin=earlier,
        )
        == bundle
    )
    assert (
        train_panel(
            panel,
            origin,
            settings,
            tmp_path / "b",
            dataset_fingerprint="fp",
            resume=True,
        )
        == bundle
    )
    assert before == {
        path: path.read_bytes() for path in (tmp_path / "b").rglob("*") if path.is_file()
    }


def test_train_panel_rejects_off_grid_earliest_origin(tmp_path, tiny_panel_and_settings):
    panel, origin, settings = tiny_panel_and_settings
    with pytest.raises(ValueError, match="off the .* grid"):
        train_panel(
            panel,
            origin,
            settings,
            tmp_path,
            dataset_fingerprint="fp",
            earliest_origin=origin - pd.Timedelta(days=1),
        )
