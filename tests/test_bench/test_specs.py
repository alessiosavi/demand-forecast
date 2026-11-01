from dataclasses import fields
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

fev = pytest.importorskip("fev")
pytestmark = pytest.mark.bench

from demand_forecast.bench.specs import FevTaskSpec, load_task_specs  # noqa: E402
from tests.helpers.fev_fixture import fixture_task, write_fev_fixture  # noqa: E402
from tests.helpers.fev_spec import fixture_spec  # noqa: E402

WEEKLY_NAMES = [
    "m5_1W",
    "favorita_stores_1W",
    "rossmann_1W",
    "rohlik_sales_1W",
    "walmart",
    "hermes",
    "hierarchical_sales_1W",
]

HELDOUT_NAMES = [
    "favorita_transactions_1W",
    "rohlik_orders_1W",
    "hospital_admissions_1W",
    "ercot_1W",
    "uk_covid_utla_1W",
]


def test_load_heldout_specs_preserve_preregistered_tasks():
    specs = load_task_specs("heldout")
    assert [spec.name for spec in specs] == HELDOUT_NAMES
    assert [spec.horizon for spec in specs] == [13, 8, 13, 13, 13]
    assert [spec.num_windows for spec in specs] == [10, 5, 16, 20, 5]
    assert [spec.target for spec in specs] == [
        "transactions",
        "orders",
        "target",
        "target",
        "cumulative_cases",
    ]
    assert [len(spec.known_dynamic) for spec in specs] == [0, 4, 0, 0, 0]
    assert [len(spec.past_dynamic) for spec in specs] == [1, 9, 0, 0, 0]
    assert [len(spec.static_columns) for spec in specs] == [5, 0, 0, 0, 0]
    assert [spec.initial_cutoff for spec in specs] == [None, "2023-05-01", None, None, None]


def test_heldout_profiles_declare_support():
    profiles = yaml.safe_load(
        files("demand_forecast.bench").joinpath("tasks/profiles.yaml").read_text()
    )
    for name in HELDOUT_NAMES:
        categorical_statics = (
            ["store_nbr", "city", "state", "type", "cluster"]
            if name == "favorita_transactions_1W"
            else []
        )
        assert profiles[name] == {
            "target_support": "nonnegative",
            "categorical_statics": categorical_statics,
        }
    for spec in load_task_specs("heldout"):
        assert spec.target_support == profiles[spec.name]["target_support"]
        assert spec.categorical_statics == profiles[spec.name]["categorical_statics"]


def test_weekly_selector_still_seven():
    specs = load_task_specs("weekly")
    assert len(specs) == 7
    assert [spec.name for spec in specs] == WEEKLY_NAMES
    assert not set(HELDOUT_NAMES) & {spec.name for spec in specs}


def test_heldout_to_task_matches_upstream_definition_except_task_name():
    # Independent upstream excerpts retain the UK task_name that the local set drops.
    upstream = yaml.safe_load(
        (Path(__file__).parent / "fixtures/fev_heldout_193e4fc.yaml").read_text()
    )["tasks"]
    definitions = yaml.safe_load(
        files("demand_forecast.bench").joinpath("tasks/fev_bench_weekly_heldout.yaml").read_text()
    )["tasks"]
    assert [definition["dataset_config"] for definition in upstream] == HELDOUT_NAMES
    for spec, definition, original in zip(
        load_task_specs("heldout"), definitions, upstream, strict=True
    ):
        assert definition == {key: value for key, value in original.items() if key != "task_name"}
        actual, expected = spec.to_task(), fev.Task(**original)
        assert actual.task_name == spec.name
        for field in fields(fev.Task):
            if field.name != "task_name":
                assert getattr(actual, field.name) == getattr(expected, field.name), (
                    spec.name,
                    field.name,
                )


def test_load_weekly_specs_have_profiles():
    specs = load_task_specs("weekly")
    assert [s.name for s in specs] == WEEKLY_NAMES
    assert {s.horizon for s in specs} == {8, 13, 39, 52}
    assert all(s.target_support == "nonnegative" for s in specs)
    assert "StoreType" in next(s for s in specs if s.name == "rossmann_1W").categorical_statics
    expected_numeric = {
        "rossmann_1W": {
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
        },
        "walmart": {"Size"},
    }
    for spec in specs:
        assert set(spec.static_columns) - set(spec.categorical_statics) == expected_numeric.get(
            spec.name, set()
        )


def test_weekly_tasks_preserve_every_pinned_yaml_field():
    definitions = yaml.safe_load(
        files("demand_forecast.bench").joinpath("tasks/fev_bench_weekly.yaml").read_text()
    )["tasks"]
    for spec, definition in zip(load_task_specs("weekly"), definitions, strict=True):
        actual, expected = spec.to_task(), fev.Task(**definition)
        for field in fields(fev.Task):
            assert getattr(actual, field.name) == getattr(expected, field.name), field.name


def test_select_named_tasks_and_reject_unknown_names():
    assert [s.name for s in load_task_specs("hermes, m5_1W")] == ["hermes", "m5_1W"]
    with pytest.raises(ValueError, match="unknown task"):
        load_task_specs("m5_1W,typo")


def write_spec_files(tmp_path, task, profile):
    tasks_path = tmp_path / "tasks.yaml"
    profiles_path = tmp_path / "profiles.yaml"
    tasks_path.write_text(yaml.safe_dump({"tasks": [task]}))
    profiles_path.write_text(yaml.safe_dump(profile))
    return tasks_path, profiles_path


def test_custom_yaml_uses_fev_defaults_and_requires_profile(tmp_path):
    task = {"dataset_path": "example/datasets", "dataset_config": "custom"}
    path, profiles = write_spec_files(tmp_path, task, {})
    with pytest.raises(ValueError, match="^profile missing$"):
        load_task_specs(path, profiles)
    profiles.write_text("custom:\n  target_support: real\n  categorical_statics: []\n")
    spec = load_task_specs(str(path), profiles)[0]
    assert spec.target_support == "real"
    assert spec.target == "target"
    assert spec.horizon == spec.num_windows == spec.seasonality == spec.min_context_length == 1
    assert spec.initial_cutoff is None
    assert spec.eval_metric == "MASE"
    assert spec.quantile_levels == spec.extra_metrics == []
    assert spec.to_task().to_dict() == fev.Task(**task).to_dict()


def test_local_task_maps_roles_and_ignores_dataset_config(tmp_path):
    parquet = write_fev_fixture(tmp_path)
    task = fixture_spec(parquet, dataset_config="ignored-for-local-file").to_task()
    assert task.dataset_config is None
    assert task.task_name == "fixture"
    expected = fixture_task(parquet)
    for key in expected.to_dict():
        if key != "task_name":
            assert task.to_dict()[key] == expected.to_dict()[key], key
    assert task.load_full_dataset() is not None


def test_custom_yaml_keeps_window_and_column_settings(tmp_path):
    task = {
        "name": "local",
        "dataset_path": "data.parquet",
        "dataset_config": "unused",
        "horizon": 4,
        "num_windows": 2,
        "initial_cutoff": -12,
        "window_step_size": 2,
        "min_context_length": 3,
        "id_column": "item",
        "timestamp_column": "date",
        "target": "sales",
        "known_dynamic_columns": ["promotion"],
        "past_dynamic_columns": ["price"],
        "static_columns": ["category", "size"],
    }
    path, profiles = write_spec_files(
        tmp_path, task, {"local": {"target_support": "real", "categorical_statics": ["category"]}}
    )
    spec = load_task_specs(path, profiles)[0]
    converted = spec.to_task()
    assert converted.dataset_path == str(tmp_path / "data.parquet")
    assert converted.dataset_config is None
    assert converted.cutoffs == [-12, -10]
    assert converted.id_column == "item" and converted.timestamp_column == "date"
    assert converted.target == "sales" and converted.min_context_length == 3
    assert spec.known_dynamic == ["promotion"] and spec.past_dynamic == ["price"]
    assert spec.categorical_statics == ["category"]


@pytest.mark.parametrize("support", [None, "positive"])
def test_profile_cannot_default_or_invent_target_support(tmp_path, support):
    profile = {"categorical_statics": []}
    if support is not None:
        profile["target_support"] = support
    path, profiles = write_spec_files(
        tmp_path, {"dataset_path": "example/data", "dataset_config": "custom"}, {"custom": profile}
    )
    with pytest.raises(ValueError, match="target_support"):
        load_task_specs(path, profiles)


def test_categorical_statics_must_belong_to_static_columns(tmp_path):
    with pytest.raises(ValueError, match="categorical_statics"):
        fixture_spec(tmp_path / "data.parquet", categorical_statics=["typo"])
    with pytest.raises(TypeError, match="target_support"):
        FevTaskSpec(name="local", dataset_path="data.parquet")
