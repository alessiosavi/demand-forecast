"""Offline study orchestration: real Optuna, fake fev inputs and training."""

import csv
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd
import pytest
import torch
import yaml
from optuna.trial import TrialState
from typer.testing import CliRunner

from demand_forecast.bench import tune
from demand_forecast.bench.run import settings_for_task
from demand_forecast.bench.specs import FevTaskSpec
from demand_forecast.cli import app
from demand_forecast.config import Settings
from demand_forecast.core.training import NonFiniteTraining
from demand_forecast.data.loader_fev import panel_from_window

RECIPE = {
    "learning_rate": 1e-4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "batch_size": 128,
    "patience": 5,
    "base_steps": 3000,
}
SOURCE = Path(__file__).resolve().parents[2] / "configs/fev-v2-pretrained-finetune.yaml"


def read_json(path):
    return json.loads(path.read_text())


def study_at(out):
    return optuna.load_study(study_name=tune.STUDY_NAME, storage=f"sqlite:///{out}/study.db")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Only window zero is available; test targets and prediction are tripwires."""

    def forbidden(*args, **kwargs):
        pytest.fail("tuning must not predict, evaluate fev, or access test targets")

    from demand_forecast.core import run
    from demand_forecast.inference import predictor

    monkeypatch.setattr(predictor, "Predictor", forbidden)
    monkeypatch.setattr(run, "Predictor", forbidden)
    monkeypatch.setattr("demand_forecast.bench.run.predictions_for_fev", forbidden)

    class FutureRow(dict):
        def __getitem__(self, key):
            if key == "target":
                forbidden()
            return super().__getitem__(key)

    h = SimpleNamespace(
        settings=Settings.from_yaml(SOURCE),
        init=tmp_path / "pretrained",
        out=tmp_path / "study",
        builds=Counter(),
        loads=Counter(),
        calls=[],
        panels={},
        score=lambda number, name: (2.0, 2.0, 0) if name == "unchanged" else (2.0, 1.0, 200),
        fail=None,
    )
    h.settings.seed = 123
    h.init.mkdir()
    h.specs = [
        FevTaskSpec(
            name=name,
            dataset_path="fake/local",
            horizon=2,
            num_windows=2,
            quantile_levels=[0.1, 0.5, 0.9],
            target_support="nonnegative",
        )
        for name in ("large", "skipped", "unchanged", "cheap")
    ]
    h.sizes = {"cheap": 1, "unchanged": 2, "skipped": 3, "large": 4}

    def inputs(spec):
        past, future = [], []
        for i in range(h.sizes[spec.name]):
            # Ragged origins test that the latest origin and earliest floor survive.
            dates = pd.date_range("2020-01-05", periods=82 - i, freq="W-SUN")
            past.append(
                {
                    "id": str(i),
                    "timestamp": dates[:-2],
                    "target": [1.0] * (len(dates) - 2)
                    if spec.name == "skipped"
                    else list(range(1, len(dates) - 1)),
                    **{
                        name: [
                            "0" if step % 2 == 0 else "holiday" for step in range(len(dates) - 2)
                        ]
                        for name in spec.known_dynamic
                    },
                }
            )
            future.append(
                FutureRow(
                    id=str(i),
                    timestamp=dates[-2:],
                    **{name: [None, "holiday"] for name in spec.known_dynamic},
                )
            )
        return past, future

    def to_task(spec):
        def windows():
            yield SimpleNamespace(get_input_data=lambda: inputs(spec), get_test_data=forbidden)
            pytest.fail("tuning must not build later fev windows")

        return SimpleNamespace(
            name=spec.name, freq="W-SUN", iter_windows=windows, evaluate=forbidden
        )

    def fingerprint(task):
        h.loads[task.name] += 1
        return f"fingerprint-{task.name}"

    def build(past, future, spec, freq):
        h.builds[spec.name] += 1
        panel = panel_from_window(past, future, spec, freq)
        h.panels[spec.name] = panel
        return panel

    def train(panel, origin, settings, out_dir, **kwargs):
        name, number = out_dir.name, int(out_dir.parent.name)
        h.calls.append((number, name, panel, origin, settings, kwargs))
        assert panel is h.panels[name]
        (out_dir / "bundle").mkdir(parents=True)
        (out_dir / "bundle/model.pt").write_bytes(b"discard weights")
        (out_dir / "checkpoints").mkdir()
        if h.fail is not None:
            raise h.fail
        step0, best, best_step = h.score(number, name)
        report = {
            "n_val_items": 0 if name == "skipped" else h.sizes[name],
            "n_fit_ids": h.sizes[name],
            "fit_result": {
                "best_score": best,
                "best_step": best_step,
                "steps": 600,
                "stop_reason": "exception_budget" if name == "skipped" else "patience",
            },
        }
        (out_dir / "run.json").write_text(json.dumps(report))
        history = [] if step0 is None or name == "skipped" else [{"step": 0, "SQL": step0}]
        torch.save({"history": history}, out_dir / "checkpoints/latest.pt")
        return out_dir / "bundle"

    monkeypatch.setattr(FevTaskSpec, "to_task", to_task)
    monkeypatch.setattr(tune, "dataset_fingerprint", fingerprint)
    monkeypatch.setattr(tune, "panel_from_window", build)
    monkeypatch.setattr(tune, "train_panel", train)
    return h


def test_baseline_ratios_cache_order_settings_and_cleanup(harness):
    h = harness
    original = h.settings.model_dump()
    summary = tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=2, seed=7)
    study = study_at(h.out)
    assert study.trials[0].params == RECIPE
    assert study.trials[0].user_attrs["baseline"] is True
    assert all(t.value == pytest.approx((0.5 + 1.0 + 0.5) / 3) for t in study.trials)
    assert summary["baseline_objective"] == summary["best_objective"] == study.best_value
    assert read_json(h.out / "tune.json") == summary
    assert h.settings.model_dump() == original
    assert h.builds == h.loads == Counter({s.name: 1 for s in h.specs})
    assert [(n, name) for n, name, *_ in h.calls] == [
        (n, name) for n in range(2) for name in ("cheap", "unchanged", "skipped", "large")
    ]
    for number, name, panel, origin, settings, kwargs in h.calls:
        expected = settings_for_task(
            next(s for s in h.specs if s.name == name), h.settings, "W-SUN"
        )
        expected.training = expected.training.model_copy(
            update={
                **study.trials[number].params,
                "init_from": str(h.init),
                "eval_at_start": True,
                "zero_shot": False,
            }
        )
        assert settings == expected
        assert settings.seed == 123
        assert kwargs["dataset_fingerprint"] == f"fingerprint-{name}"
        assert kwargs["task_definition"]["name"] == name
        assert kwargs["earliest_origin"] == origin - pd.Timedelta(weeks=h.sizes[name] - 1)
        task_dir = h.out / "trials" / str(number) / name
        assert {p.name for p in task_dir.iterdir()} == {"run.json", "score.json"}
        score = read_json(task_dir / "score.json")
        if name == "skipped":
            assert score["status"] == "skipped_no_inner_validation"
            assert score["ratio"] is None
        else:
            assert score["ratio"] == (1.0 if name == "unchanged" else 0.5)
            assert score["step0_score"] == 2.0
            assert score["steps"] == 600
    assert study.trials[0].intermediate_values == pytest.approx({0: 0.5, 1: 0.75, 2: 2 / 3})
    with (h.out / "trials.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["state"] == "COMPLETE"
    assert json.loads(rows[0]["params"]) == RECIPE
    assert float(rows[0]["duration_s"]) >= 0
    assert json.loads(rows[0]["tasks"])["unchanged"]["best_step"] == 0


def test_yaml_changes_only_tuned_keys_and_keeps_source_fields(harness):
    h = harness
    # Source YAML is sparse: omitted default controls must stay omitted.
    raw = yaml.safe_load(SOURCE.read_text())
    h.settings = Settings.from_yaml(SOURCE)
    h.score = lambda number, name: (2.0, 1.5 if number == 0 else 1.0, 200)
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=2)
    best = read_json(h.out / "best-params.json")
    assert best != RECIPE
    expected = yaml.safe_load(SOURCE.read_text())
    expected["training"].update(best)
    assert yaml.safe_load((h.out / "best-finetune.yaml").read_text()) == expected
    loaded = Settings.from_yaml(h.out / "best-finetune.yaml")
    assert loaded.training.init_from is None
    assert loaded.training.model_dump(include=set(RECIPE)) == best
    assert yaml.safe_load(SOURCE.read_text()) == raw


def test_actual_median_pruner_after_four_startup_trials(harness):
    h = harness
    # Four scored tasks, so the bad trial can prune before the expensive last task.
    h.specs[1].name = "fourth"
    h.sizes["fourth"] = 5
    h.score = lambda number, name: (2.0, 1.0, 200) if number < 4 else (2.0, 2.0, 0)
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=5)
    study = study_at(h.out)
    assert [t.state for t in study.trials] == [TrialState.COMPLETE] * 4 + [TrialState.PRUNED]
    assert list(study.trials[4].intermediate_values) == [0, 1, 2]
    assert [name for n, name, *_ in h.calls if n == 4] == ["cheap", "unchanged", "large"]
    assert not (h.out / "trials/4/fourth").exists()
    assert not list((h.out / "trials").rglob("*.pt"))
    h.calls.clear()
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=6, resume=True)
    assert {n for n, *_ in h.calls} == {5}
    assert len(study_at(h.out).trials) == 6


def test_resume_total_budget_and_abandoned_running_trial(harness):
    h = harness
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=2)
    before = {p: p.read_bytes() for p in (h.out / "trials").rglob("*.json")}
    h.calls.clear()
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=4, resume=True)
    assert {n for n, *_ in h.calls} == {2, 3}
    assert all(p.read_bytes() == content for p, content in before.items())
    study = study_at(h.out)
    abandoned = study.ask()
    task_dir = h.out / "trials" / str(abandoned.number) / "cheap"
    (task_dir / "checkpoints").mkdir(parents=True)
    (task_dir / "checkpoints/latest.pt").write_bytes(b"partial")
    (task_dir / "bundle").mkdir()
    h.calls.clear()
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=6, resume=True)
    trials = study_at(h.out).trials
    assert len(trials) == 6
    assert trials[abandoned.number].state == TrialState.FAIL
    assert {n for n, *_ in h.calls} == {5}
    assert not (task_dir / "checkpoints").exists()
    assert not (task_dir / "bundle").exists()
    h.calls.clear()
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=6, resume=True)
    assert h.calls == []
    assert sum(t.user_attrs.get("baseline", False) for t in trials) == 1
    with pytest.raises(optuna.exceptions.DuplicatedStudyError):
        tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=7)


@pytest.mark.parametrize("step0", [None, float("nan"), 0.0])
def test_invalid_step_zero_fails_instead_of_silently_dropping_task(harness, step0):
    h = harness
    h.score = lambda number, name: (step0, 1.0, 200)
    with pytest.raises(ValueError, match="step.0"):
        tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=3)
    assert [trial.state for trial in study_at(h.out).trials] == [TrialState.FAIL]
    assert not list((h.out / "trials").rglob("*.pt"))
    assert read_json(h.out / "tune.json")["best_objective"] is None
    assert not (h.out / "best-finetune.yaml").exists()


def test_zero_initial_score_best_at_zero_has_ratio_one(harness):
    h = harness
    h.score = lambda number, name: (0.0, 0.0, 0)
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=1)
    assert study_at(h.out).best_value == 1.0


def test_no_inner_validation_never_produces_a_winner(harness):
    h = harness
    h.specs = [s for s in h.specs if s.name == "skipped"]
    with pytest.raises(ValueError, match="no inner.validation"):
        tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=1)
    assert (
        read_json(h.out / "trials/0/skipped/score.json")["status"] == "skipped_no_inner_validation"
    )
    assert read_json(h.out / "best-params.json") is None


def test_nonfinite_training_records_failure_and_continues(harness, monkeypatch):
    h = harness
    fake_train = tune.train_panel
    message = "non-finite losses on unchanged"

    def fail_one_task(panel, origin, settings, out_dir, **kwargs):
        h.fail = (
            NonFiniteTraining(message)
            if int(out_dir.parent.name) == 1 and out_dir.name == "unchanged"
            else None
        )
        return fake_train(panel, origin, settings, out_dir, **kwargs)

    monkeypatch.setattr(tune, "train_panel", fail_one_task)
    # The failed trial's completed first task looks best; it must not win.
    h.score = lambda number, name: (2.0, 0.01 if number == 1 else 1.0, 200)
    summary = tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=3)
    study = study_at(h.out)
    assert [trial.state for trial in study.trials] == [
        TrialState.COMPLETE,
        TrialState.FAIL,
        TrialState.COMPLETE,
    ]
    assert {n for n, *_ in h.calls} == {0, 1, 2}
    assert [name for n, name, *_ in h.calls if n == 1] == ["cheap", "unchanged"]
    failed = study.trials[1]
    failure = {"task": "unchanged", "message": message}
    assert failed.user_attrs["failure"] == failure
    assert failed.value is None
    assert failed.intermediate_values[0] < study.best_value
    assert study.best_trial.number == summary["best_trial"] == 0
    assert summary["best_objective"] == 0.5
    assert summary["n_trials"] == 3
    assert summary["states"] == {"COMPLETE": 2, "FAIL": 1}
    assert summary["failures"] == [{"trial": 1, **failure}]
    assert read_json(h.out / "tune.json") == summary
    assert read_json(h.out / "best-params.json") == study.best_params
    with (h.out / "trials.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert json.loads(rows[1]["failure"]) == failure
    assert json.loads(rows[0]["failure"]) is None
    assert not list((h.out / "trials").rglob("bundle"))
    assert not list((h.out / "trials").rglob("checkpoints"))
    h.calls.clear()
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=3, resume=True)
    assert h.calls == []  # FAIL consumed one slot of the total budget.


@pytest.mark.parametrize("error", [ValueError, OSError, RuntimeError])
def test_training_failure_still_exports_and_cleans_weights(harness, error):
    h = harness
    h.fail = error("fake training failed")
    with pytest.raises(error, match="fake training failed"):
        tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=3)
    assert [trial.state for trial in study_at(h.out).trials] == [TrialState.FAIL]
    assert {n for n, *_ in h.calls} == {0}
    assert not list((h.out / "trials").rglob("*.pt"))
    assert read_json(h.out / "tune.json")["best_objective"] is None


def test_resume_rejects_changed_settings(harness):
    h = harness
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=1)
    h.settings.seed += 1
    with pytest.raises(ValueError, match="study inputs"):
        tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=2, resume=True)
    assert len(study_at(h.out).trials) == 1


def test_resume_rejects_changed_event_tokens(harness):
    h = harness
    h.specs = [
        replace(spec, known_dynamic=["holiday"], event_dynamics={"holiday": ["0"]})
        for spec in h.specs
    ]
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=1)
    for panel in h.panels.values():
        for series in panel.series:
            assert series.dynamic["holiday"].tolist() == [i % 2 for i in range(series.length)] + [
                0,
                1,
            ]
    h.specs = [replace(spec, event_dynamics={"holiday": []}) for spec in h.specs]
    with pytest.raises(ValueError, match="study inputs"):
        tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=2, resume=True)
    assert len(study_at(h.out).trials) == 1


def test_resume_preserves_sampler_stream_instead_of_repeating_recipes(harness):
    h = harness
    uninterrupted = h.out.with_name("uninterrupted")
    # Cross the default ten startup trials to exercise both random and TPE RNGs.
    tune.tune_finetune(h.specs, h.settings, h.init, uninterrupted, n_trials=12, seed=7)
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=11, seed=7)
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=12, seed=7, resume=True)
    assert [t.params for t in study_at(h.out).trials] == [
        t.params for t in study_at(uninterrupted).trials
    ]


def test_timeout_stops_new_trials_and_resume_uses_total_budget(harness, monkeypatch):
    h = harness
    # Advance only Optuna's scheduling clock; no sleeping or training-time race.
    ticks = iter([0, 0, 120, 120])
    clock = SimpleNamespace(now=lambda: datetime(2026, 1, 1) + timedelta(seconds=next(ticks)))
    with monkeypatch.context() as patch:
        patch.setattr(optuna.study._optimize, "datetime", SimpleNamespace(datetime=clock))
        tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=3, timeout_s=60)
    assert len(study_at(h.out).trials) == 1
    assert {n for n, *_ in h.calls} == {0}
    h.calls.clear()
    tune.tune_finetune(h.specs, h.settings, h.init, h.out, n_trials=3, resume=True)
    assert len(study_at(h.out).trials) == 3
    assert {n for n, *_ in h.calls} == {1, 2}


def test_cli_tune_wiring(tmp_path, monkeypatch):
    seen = {}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    specs = [object()]

    def capture(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return {"best_objective": 0.8}

    def load(selector):
        assert selector == "weekly"
        return specs

    monkeypatch.setattr(tune, "tune_finetune", capture)
    monkeypatch.setattr("demand_forecast.bench.specs.load_task_specs", load)
    result = CliRunner().invoke(
        app,
        [
            "fev",
            "tune",
            "--tasks",
            "weekly",
            "--config",
            str(SOURCE),
            "--init-from",
            str(bundle),
            "--out-dir",
            str(tmp_path / "out"),
            "--trials",
            "9",
            "--timeout-s",
            "12",
            "--seed",
            "7",
            "--resume",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert seen["args"] == (specs, Settings.from_yaml(SOURCE), bundle, tmp_path / "out")
    assert seen["kwargs"] == {"n_trials": 9, "timeout_s": 12, "seed": 7, "resume": True}
