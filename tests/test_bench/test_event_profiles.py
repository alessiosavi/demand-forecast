"""Pinned event profiles and identities need no optional benchmark imports."""

import copy
import json
import pickle
from dataclasses import asdict, replace
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import yaml

from demand_forecast import __version__
from demand_forecast.bench.run import settings_for_task
from demand_forecast.bench.specs import FEV_DATASETS_REVISION, load_task_specs
from demand_forecast.core.run import run_identity
from tests.helpers.fev_spec import fixture_spec
from tests.helpers.settings import tiny_settings


@pytest.mark.parametrize(
    "events",
    [None, [], {"promo": "0"}, {"promo": [0]}, {7: []}, {"target": []}, {"typo": []}, {"cat": []}],
)
def test_invalid_event_profiles_rejected(tmp_path, events):
    tasks = tmp_path / "tasks.yaml"
    profiles = tmp_path / "profiles.yaml"
    tasks.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "name": "fixture",
                        "dataset_path": "unused.parquet",
                        "known_dynamic_columns": ["promo"],
                        "past_dynamic_columns": ["oil"],
                        "static_columns": ["cat"],
                    }
                ]
            }
        )
    )
    profiles.write_text(
        yaml.safe_dump({"fixture": {"target_support": "real", "event_dynamics": events}})
    )
    with pytest.raises(ValueError, match="fixture: event_dynamics"):
        load_task_specs(tasks, profiles)


def test_event_profiles_cover_exactly_nonnumeric_declared_dynamics():
    resources = files("demand_forecast.bench").joinpath("tasks")
    schema = yaml.safe_load(resources.joinpath("schema_fev100.yaml").read_text())
    definitions = yaml.safe_load(resources.joinpath("fev_bench_100_193e4fc.yaml").read_text())[
        "tasks"
    ]
    profiles = yaml.safe_load(resources.joinpath("profiles.yaml").read_text())
    profiles.update(yaml.safe_load(resources.joinpath("profiles_fev100.yaml").read_text()))
    assert schema["revision"] == FEV_DATASETS_REVISION
    assert len(definitions) == 100
    expected = {}
    for task in definitions:
        name = task.get("task_name", task["dataset_config"])
        features = schema["configs"][task["dataset_config"]]["features"]
        nonnumeric = set()
        for column in task.get("known_dynamic_columns", []) + task.get("past_dynamic_columns", []):
            dtype = features[column]["sequence"]
            if dtype in ("string", "large_string") or np.dtype(dtype).kind not in "biuf":
                nonnumeric.add(column)
        assert set(profiles[name].get("event_dynamics", {})) == nonnumeric, name
        if nonnumeric:
            expected[name] = nonnumeric
    assert expected == {
        "rossmann_1D": {"StateHoliday"},
        "favorita_stores_1D": {"holiday"},
        "favorita_transactions_1D": {"holiday"},
    }
    actual = {
        spec.name: spec.event_dynamics for spec in load_task_specs("fev100") if spec.event_dynamics
    }
    assert actual == {
        "rossmann_1D": {"StateHoliday": ["0"]},
        "favorita_stores_1D": {"holiday": []},
        "favorita_transactions_1D": {"holiday": []},
    }


@pytest.mark.parametrize("policy", ["first_window", "per_window"])
def test_event_tokens_participate_in_identity(policy):
    spec = load_task_specs("rossmann_1D")[0]
    changed = replace(spec, event_dynamics={"StateHoliday": []})
    settings = settings_for_task(spec, tiny_settings(), "D")

    def identity(task):
        return run_identity(
            settings,
            0,
            "fixed-fingerprint",
            __version__,
            pd.Timestamp("2024-01-01"),
            task.task_definition(),
            policy=policy,
        )

    assert identity(spec) != identity(changed)
    assert spec.to_dict()["event_dynamics"] == {"StateHoliday": ["0"]}
    assert spec.to_dict() != changed.to_dict()  # tuning resume inputs
    assert replace(spec, name="renamed").event_dynamics == spec.event_dynamics
    assert spec.task_definition()["event_dynamics"] == spec.event_dynamics
    assert spec.event_dynamics == {"StateHoliday": ["0"]}


def test_empty_profiles_preserve_legacy_serialization(tmp_path):
    spec = fixture_spec(tmp_path / "unused.parquet")
    explicit = replace(spec, event_dynamics={})
    assert spec.event_dynamics == {}
    assert spec.to_dict() == asdict(spec)
    assert json.dumps(spec.task_definition()) == json.dumps(explicit.task_definition())
    assert "event_dynamics" not in spec.task_definition()
    assert "event_dynamics" not in asdict(spec)


def test_all_fev100_specs_preserve_undeclared_serialization():
    declared = {"rossmann_1D", "favorita_stores_1D", "favorita_transactions_1D"}
    specs = load_task_specs("fev100")
    assert len(specs) == 100
    assert declared <= {spec.name for spec in specs}
    for spec in specs:
        has_events = spec.name in declared
        assert ("event_dynamics" in spec.task_definition()) == has_events, spec.name
        assert ("event_dynamics" in spec.to_dict()) == has_events, spec.name
        assert "event_dynamics" not in asdict(spec), spec.name
        if not has_events:
            assert spec.to_dict() == asdict(spec), spec.name


@pytest.mark.parametrize("name", ["rossmann_1D", "rossmann_1W"])
@pytest.mark.parametrize("method", ["deepcopy", "pickle"])
def test_copy_and_pickle_preserve_event_profiles(name, method):
    spec = load_task_specs(name)[0]
    restored = copy.deepcopy(spec) if method == "deepcopy" else pickle.loads(pickle.dumps(spec))
    expected = {"StateHoliday": ["0"]} if name == "rossmann_1D" else {}
    assert restored is not spec
    assert restored == spec
    assert restored.event_dynamics == spec.event_dynamics == expected
    if expected:
        restored.event_dynamics["StateHoliday"].append("no holiday")
        assert spec.event_dynamics == {"StateHoliday": ["0"]}
        assert restored != spec


def test_event_profiles_participate_in_equality_and_repr():
    spec = load_task_specs("rossmann_1D")[0]
    identical = replace(spec, event_dynamics={"StateHoliday": ["0"]})
    changed = replace(spec, event_dynamics={"StateHoliday": []})
    empty = replace(spec, event_dynamics={})
    assert spec == identical
    assert spec != changed
    assert spec != empty
    assert spec != replace(spec, name="renamed")
    assert spec.__eq__(object()) is NotImplemented
    assert repr(spec) == repr(identical)
    assert repr(spec) != repr(changed)
    assert "event_dynamics={'StateHoliday': ['0']}" in repr(spec)
    assert "event_dynamics" not in repr(empty)
    assert asdict(spec) == asdict(identical) == asdict(changed) == asdict(empty)
    assert "event_dynamics" not in asdict(spec)
    assert type(spec).__hash__ is None
    with pytest.raises(TypeError, match="unhashable"):
        hash(spec)
