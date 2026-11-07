import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.run import run_fev  # noqa: E402
from demand_forecast.data.loader_fev import panel_from_window  # noqa: E402
from tests.helpers.fev_fixture import write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402
from tests.helpers.settings import tiny_settings  # noqa: E402


def assert_panels_equal(actual, expected):
    assert actual.roles == expected.roles
    assert actual.freq == expected.freq
    assert actual.seasonal_period == expected.seasonal_period
    assert actual.target_support == expected.target_support
    assert len(actual.series) == len(expected.series)
    for left, right in zip(actual.series, expected.series, strict=True):
        assert (left.series_id, left.start, left.history_length) == (
            right.series_id,
            right.start,
            right.history_length,
        )
        assert left.statics.keys() == right.statics.keys()
        for key in left.statics:
            np.testing.assert_equal(left.statics[key], right.statics[key])
        np.testing.assert_array_equal(left.target, right.target)
        assert left.target.dtype == right.target.dtype
        assert left.dynamic.keys() == right.dynamic.keys()
        for key in left.dynamic:
            np.testing.assert_array_equal(left.dynamic[key], right.dynamic[key])
            assert left.dynamic[key].dtype == right.dynamic[key].dtype


@pytest.mark.parametrize("cutoff", [None, 12, "2022-05-01"])
@pytest.mark.parametrize("freq", ["W-SUN", "MS", "5min"])
@pytest.mark.parametrize("events,multiple", [(False, False), (True, True)])
def test_cached_panels_equal_window_rebuilds(tmp_path, cutoff, freq, events, multiple):
    from datasets import Dataset

    from demand_forecast.bench.panels import TaskPanelCache

    parquet = write_fev_fixture(
        tmp_path,
        n_series=3,
        lengths=[40, 35, 18],
        freq=freq,
        start="2022-01-01" if freq == "MS" else "2022-01-02",
        text_dynamic=("promo", "oil") if events else (),
    )
    data = Dataset.from_parquet(str(parquet))

    def missing_and_targets(row):
        row["target"][9:12] = [None] * 3
        row["other"] = [-value if value is not None else None for value in row["target"]]
        row["number"] = None if row["id"] == "0" else 7.5
        return row

    data.map(missing_and_targets).to_parquet(parquet)
    # Calendar cutoffs must leave enough future observations on this frequency.
    if isinstance(cutoff, str):
        cutoff = pd.date_range(
            "2022-01-01" if freq == "MS" else "2022-01-02", periods=13, freq=freq
        )[-1].isoformat()
    spec = fixture_spec(
        parquet,
        initial_cutoff=cutoff,
        window_step_size=3,
        static_columns=["cat", "number"],
        target=["target", "other"] if multiple else "target",
        target_support={"target": "nonnegative", "other": "real"} if multiple else "nonnegative",
        event_dynamics={"promo": ["0"], "oil": ["0"]} if events else {},
    )
    task = spec.to_task()
    cache = TaskPanelCache(task.load_full_dataset(), spec, task.freq)
    first_panel = None
    for window in task.iter_windows():
        past, future = window.get_input_data()
        actual = cache.for_window(window, past)
        expected = panel_from_window(past, future, spec, task.freq)
        assert_panels_equal(actual, expected)
        if first_panel is None:
            first_panel = actual
            first_targets = [series.target.copy() for series in actual.series]
        # Later windows must not change an earlier window's visible history.
        for series, target in zip(first_panel.series, first_targets, strict=True):
            np.testing.assert_array_equal(series.target, target)


@pytest.mark.parametrize("policy", ["first_window", "per_window"])
def test_cached_run_writes_same_forecasts_as_rebuild(tmp_path, monkeypatch, policy):
    from demand_forecast.bench.panels import TaskPanelCache

    spec = fixture_spec(write_fev_fixture(tmp_path, lengths=[40, 38, 40, 28, 38, 40]))
    settings = tiny_settings(base_steps=2, d_model=8)
    cached = tmp_path / "cached"
    reference = tmp_path / "reference"
    build = TaskPanelCache.__init__
    builds = []

    def count_build(self, *args, **kwargs):
        builds.append(1)
        build(self, *args, **kwargs)

    monkeypatch.setattr(TaskPanelCache, "__init__", count_build)
    run_fev([spec], settings, cached, policy=policy)
    assert len(builds) == 1

    def rebuild(self, window, past):
        _, future = window.get_input_data()
        return panel_from_window(past, future, spec, self.panel.freq)

    monkeypatch.setattr(TaskPanelCache, "for_window", rebuild)
    run_fev([spec], settings, reference, policy=policy)
    for index in range(spec.num_windows):
        with (
            np.load(cached / spec.name / f"window_{index}.npz", allow_pickle=False) as actual,
            np.load(reference / spec.name / f"window_{index}.npz", allow_pickle=False) as expected,
        ):
            for key in ("ids", "arrays", "identity"):
                np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("policy", ["first_window", "per_window"])
def test_completed_resume_skips_before_panel_build(tmp_path, monkeypatch, capsys, policy):
    from demand_forecast.bench.panels import TaskPanelCache
    from demand_forecast.data import loader_fev

    spec = fixture_spec(write_fev_fixture(tmp_path))
    settings = tiny_settings(base_steps=2, d_model=8)
    out = tmp_path / "out"
    run_fev([spec], settings, out, policy=policy)
    capsys.readouterr()
    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("completed resume must not build a panel")

    monkeypatch.setattr(loader_fev, "panel_from_window", forbidden)
    monkeypatch.setattr(TaskPanelCache, "__init__", forbidden)
    run_fev([spec], settings, out, policy=policy, resume=True)
    assert f"Skipping completed task {spec.name}" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in out.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("max_context", [None, 6])
def test_coarser_item_grids_and_context_limits_match_rebuild(tmp_path, max_context):
    from datasets import Dataset

    from demand_forecast.bench.panels import TaskPanelCache

    parquet = write_fev_fixture(tmp_path, n_series=2)
    data = Dataset.from_parquet(str(parquet))

    def coarser_grid(row):
        if row["id"] == "1":
            for column in ("timestamp", "target", "promo", "oil"):
                row[column] = row[column][::2]
        return row

    data.map(coarser_grid).to_parquet(parquet)
    spec = fixture_spec(parquet)
    task = spec.to_task()
    task.max_context_length = max_context
    cache = TaskPanelCache(task.load_full_dataset(), spec, task.freq)
    for window in task.iter_windows():
        past, future = window.get_input_data()
        assert_panels_equal(
            cache.for_window(window, past), panel_from_window(past, future, spec, task.freq)
        )
