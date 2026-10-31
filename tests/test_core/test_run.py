import json

import numpy as np
import pandas as pd
import pytest

from demand_forecast.core.bundle import load_bundle
from demand_forecast.core.run import _identity_settings, run_identity, train_panel
from demand_forecast.core.training import resolve_budget
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.frequency import to_offset
from tests.helpers.panels import weekly_panel
from tests.helpers.settings import tiny_settings


@pytest.fixture
def tiny_panel_and_settings(panel29, origin29):
    return panel29, origin29, tiny_settings(base_steps=2, d_model=8)


@pytest.fixture
def quick_panel_and_settings():
    panel = weekly_panel([40] * 6)
    settings = tiny_settings(base_steps=20, d_model=8)
    settings.seed = 23
    return panel, panel.timestamps(0, 41)[40], settings


def test_quick_fields_subsample_validation_and_are_reported(
    tmp_path, quick_panel_and_settings, caplog
):
    panel, origin, settings = quick_panel_and_settings
    eligible = plan_inner_validation(panel, origin, settings.forecast.horizon)
    assert len(eligible.val_items) >= 4
    assert resolve_budget(settings.training.base_steps, len(eligible.fit_ids), 5)[0] > 7
    quick = settings.model_copy(deep=True)
    quick.training.val_items_max = 2
    quick.training.max_steps_cap = 7
    with caplog.at_level("INFO", logger="demand_forecast.core.run"):
        bundle = train_panel(panel, origin, quick, tmp_path / "q", dataset_fingerprint="fp")
    report = json.loads((tmp_path / "q/run.json").read_text())
    assert report["val_items_max"] == 2 and report["n_val_items"] == 2
    assert report["n_val_items_eligible"] == len(eligible.val_items)
    assert report["resolved_budget"] == {"max_steps": 7, "eval_every": 1}
    assert report["fit_result"]["steps"] <= 7
    assert report["plan"]["train_upto"] == eligible.train_upto.isoformat()
    assert report["plan"]["val_origin"] == eligible.val_origin.isoformat()
    assert report["plan"]["fit_ids"] == eligible.fit_ids
    assert report["plan"]["exception"] == eligible.exception
    assert report["plan"]["val_items"] == sorted(
        np.random.default_rng(quick.seed).choice(eligible.val_items, size=2, replace=False).tolist()
    )
    assert "val_items=2/6" in caplog.text
    assert load_bundle(bundle).manifest["run"] == report


@pytest.mark.parametrize("field", ["max_steps_cap", "val_items_max"])
def test_identity_ignores_unset_quick_fields_and_changes_when_set(field):
    settings = tiny_settings()
    dumped = settings.model_dump(mode="json")
    base = run_identity(settings, 0, "fp", "2.0.0", "2023-01-01", None)
    assert dumped["training"]["max_steps_cap"] is None
    assert dumped["training"]["val_items_max"] is None
    expected = settings.model_dump(mode="json")
    del expected["training"]["max_steps_cap"], expected["training"]["val_items_max"]
    assert _identity_settings(settings) == expected
    assert settings.model_dump(mode="json") == dumped
    quick = settings.model_copy(deep=True)
    setattr(quick.training, field, 300)
    assert _identity_settings(quick)["training"][field] == 300
    assert run_identity(quick, 0, "fp", "2.0.0", "2023-01-01", None) != base


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
