"""Study one shared fine-tune recipe using only each task's inner validation.

No fev predictions or test-window evaluations are made here. The only fev data
adapter is the same window-zero input path used by run_fev. A study directory
belongs to one process; resume assumes its previous process has stopped.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import pickle
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd
import torch
import yaml
from optuna.trial import TrialState

from demand_forecast.bench.run import resolved_origin, settings_for_task, window_origins
from demand_forecast.bench.specs import FevTaskSpec, dataset_fingerprint
from demand_forecast.config import Settings
from demand_forecast.core.run import train_panel
from demand_forecast.core.training import NonFiniteTraining
from demand_forecast.core.validation import plan_inner_validation
from demand_forecast.data.loader_fev import panel_from_window
from demand_forecast.data.panel import Panel

STUDY_NAME = "fev-finetune"
BASELINE_RECIPE = {
    "learning_rate": 1e-4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "batch_size": 128,
    "patience": 5,
    "base_steps": 3000,
}


@dataclass
class _Task:
    spec: FevTaskSpec
    panel: Panel
    settings: Settings
    origin: pd.Timestamp
    earliest_origin: pd.Timestamp
    fingerprint: str
    definition: dict
    n_fit_ids: int


def _prepare_tasks(specs: Iterable[FevTaskSpec], settings: Settings) -> list[_Task]:
    tasks = []
    names = set()
    for spec in specs:
        if spec.name in names or Path(spec.name).name != spec.name or spec.name in ("", ".", ".."):
            raise ValueError(f"task names must be unique directory names: {spec.name!r}")
        names.add(spec.name)
        task = spec.to_task()
        fingerprint = dataset_fingerprint(task)
        task_settings = settings_for_task(spec, settings, task.freq)
        window = next(iter(task.iter_windows()), None)
        if window is None:
            raise ValueError(f"task {spec.name} has no windows")
        past, future = window.get_input_data()
        panel = panel_from_window(past, future, spec, task.freq)
        origins = window_origins(future, spec.id_column, spec.timestamp_column)
        origin = resolved_origin(origins)
        plan = plan_inner_validation(
            panel, origin, task_settings.forecast.horizon, task_settings.training.min_train_history
        )
        tasks.append(
            _Task(
                spec,
                panel,
                task_settings,
                origin,
                min(origins.values()),
                fingerprint,
                spec.task_definition(task.freq),
                len(plan.fit_ids),
            )
        )
    if not tasks:
        raise ValueError("at least one task is required")
    return sorted(tasks, key=lambda task: task.n_fit_ids)


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [0, 100, 300, 600]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "patience": trial.suggest_categorical("patience", [3, 5, 8]),
        "base_steps": trial.suggest_categorical("base_steps", [1500, 3000, 6000]),
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _clean_weights(task_dir: Path) -> None:
    for name in ("bundle", "checkpoints"):
        path = task_dir / name
        if path.exists():
            shutil.rmtree(path)


def _read_score(task_dir: Path) -> dict:
    report = json.loads((task_dir / "run.json").read_text(encoding="utf-8"))
    fit = report["fit_result"]
    score = {
        "status": "scored",
        "n_val_items": report["n_val_items"],
        "steps": fit["steps"],
        "best_step": fit["best_step"],
        "best_score": fit["best_score"],
        "stop_reason": fit["stop_reason"],
        "step0_score": None,
        "ratio": None,
    }
    if report["n_val_items"] == 0:
        score["status"] = "skipped_no_inner_validation"
        return score

    # This checkpoint was written by our own train_panel call. mmap avoids
    # copying model/optimizer tensors into RAM just to read its small history.
    state = torch.load(
        task_dir / "checkpoints/latest.pt", map_location="cpu", weights_only=False, mmap=True
    )
    initial = [entry["SQL"] for entry in state["history"] if entry.get("step") == 0]
    del state
    if len(initial) != 1 or initial[0] is None or not math.isfinite(initial[0]) or initial[0] < 0:
        raise ValueError(f"{task_dir}: missing or invalid step-0 inner SQL")
    step0 = float(initial[0])
    best = fit["best_score"]
    if best is None or not math.isfinite(best) or best < 0:
        raise ValueError(f"{task_dir}: invalid best inner SQL")
    if fit["best_step"] == 0:
        ratio = 1.0
    elif step0 == 0:
        raise ValueError(f"{task_dir}: step-0 inner SQL must be positive for a ratio")
    else:
        ratio = best / step0
    if not math.isfinite(ratio):
        raise ValueError(f"{task_dir}: non-finite inner SQL ratio")
    return {**score, "step0_score": step0, "ratio": ratio}


def _write_outputs(study: optuna.Study, out_dir: Path, n_trials: int) -> dict:
    trials = study.get_trials(deepcopy=False)
    complete = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    best = min(complete, key=lambda trial: trial.value) if complete else None
    baseline = next((trial for trial in trials if trial.user_attrs.get("baseline")), None)
    with (out_dir / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["number", "state", "params", "objective", "tasks", "failure", "duration_s"],
        )
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "params": json.dumps(trial.params, sort_keys=True),
                    "objective": trial.value,
                    "tasks": json.dumps(trial.user_attrs.get("tasks", {}), sort_keys=True),
                    "failure": json.dumps(trial.user_attrs.get("failure"), sort_keys=True),
                    "duration_s": trial.duration.total_seconds() if trial.duration else None,
                }
            )
    params = best.params if best is not None else None
    _write_json(out_dir / "best-params.json", params)
    if best is not None:
        source = study.user_attrs["source_config"]
        source.setdefault("training", {}).update(params)
        Settings.model_validate(source)
        (out_dir / "best-finetune.yaml").write_text(
            yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
        )
    summary = {
        "study_name": STUDY_NAME,
        "objective": "mean(best_inner_SQL / step0_inner_SQL)",
        "direction": "minimize",
        "n_trials_requested": n_trials,
        "n_trials": len(trials),
        "states": dict(Counter(trial.state.name for trial in trials)),
        "failures": [
            {"trial": trial.number, **trial.user_attrs["failure"]}
            for trial in trials
            if "failure" in trial.user_attrs
        ],
        "task_order": [task["spec"]["name"] for task in study.user_attrs["inputs"]["tasks"]],
        "baseline_trial": baseline.number if baseline is not None else None,
        "baseline_objective": baseline.value if baseline is not None else None,
        "best_trial": best.number if best is not None else None,
        "best_objective": best.value if best is not None else None,
        "best_params": params,
    }
    _write_json(out_dir / "tune.json", summary)
    return summary


def tune_finetune(
    specs: Iterable[FevTaskSpec],
    settings: Settings,
    init_from: Path,
    out_dir: Path,
    *,
    n_trials: int,
    timeout_s: float | None = None,
    seed: int = 0,
    resume: bool = False,
) -> dict:
    """Minimize mean inner SQL ratios, retaining only reports and study state.

    ``n_trials`` is the total attempted-trial budget, including FAIL and PRUNED.
    NonFiniteTraining records a task failure and continues; other errors propagate.
    Timeout follows Optuna semantics: it stops launching trials, not an ongoing
    train_panel call. Best-recipe files require at least one COMPLETE trial.
    Settings loaded from YAML retain their explicitly supplied fields when the
    winning YAML is exported; only the six tuned training fields are replaced.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s must be finite and positive")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{out_dir / 'study.db'}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2),
        load_if_exists=resume,
    )
    tasks = _prepare_tasks(specs, settings)
    # Compare the data/configuration across resumes, independently of trial
    # parameters. The caller must keep the pretrained bundle at this path fixed.
    inputs = {
        "settings": settings.model_dump(mode="json"),
        "init_from": str(Path(init_from).resolve()),
        "sampler_seed": seed,
        "optuna_version": optuna.__version__,
        "tasks": [
            {
                "spec": task.spec.to_dict(),
                "fingerprint": task.fingerprint,
                "origin": task.origin.isoformat(),
                "earliest_origin": task.earliest_origin.isoformat(),
                "n_fit_ids": task.n_fit_ids,
            }
            for task in tasks
        ],
    }
    inputs = json.loads(json.dumps(inputs, default=str))
    if "inputs" in study.user_attrs and study.user_attrs["inputs"] != inputs:
        raise ValueError("study inputs differ; use the original inputs or a new out_dir")
    study.set_user_attr("inputs", inputs)
    if "source_config" not in study.user_attrs:
        study.set_user_attr("source_config", settings.model_dump(mode="json", exclude_unset=True))
    if "sampler_state" in study.user_attrs:
        # Like our trainer checkpoints, this is trusted state from this local
        # study. Optuna's SQL storage does not persist the sampler's RNG stream.
        study.sampler = pickle.loads(base64.b64decode(study.user_attrs["sampler_state"]))
    for trial in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
        for task_dir in (out_dir / "trials" / str(trial.number)).glob("*"):
            _clean_weights(task_dir)
    if not study.trials:
        study.enqueue_trial(BASELINE_RECIPE, user_attrs={"baseline": True})

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        # Commit the advanced sampler before expensive training, so even a
        # killed RUNNING trial's recipe is not sampled again on resume.
        study.set_user_attr(
            "sampler_state", base64.b64encode(pickle.dumps(study.sampler)).decode("ascii")
        )
        scores, ratios = {}, []
        for task in tasks:
            task_dir = out_dir / "trials" / str(trial.number) / task.spec.name
            task_settings = task.settings.model_copy(deep=True)
            task_settings.training = task_settings.training.model_copy(
                update={
                    **params,
                    "init_from": str(init_from),
                    "eval_at_start": True,
                    "zero_shot": False,
                }
            )
            try:
                train_panel(
                    task.panel,
                    task.origin,
                    task_settings,
                    task_dir,
                    dataset_fingerprint=task.fingerprint,
                    task_definition=task.definition,
                    earliest_origin=task.earliest_origin,
                )
                score = _read_score(task_dir)
                _write_json(task_dir / "score.json", score)
                scores[task.spec.name] = score
                trial.set_user_attr("tasks", scores)
            except NonFiniteTraining as exc:
                trial.set_user_attr("failure", {"task": task.spec.name, "message": str(exc)})
                raise
            finally:
                _clean_weights(task_dir)
            if score["status"] == "skipped_no_inner_validation":
                continue
            ratios.append(score["ratio"])
            trial.report(sum(ratios) / len(ratios), step=len(ratios) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after {len(ratios)} scored tasks")
        if not ratios:
            raise ValueError("tasks have no inner-validation items; no objective can be scored")
        return sum(ratios) / len(ratios)

    attempted = sum(trial.state != TrialState.WAITING for trial in study.trials)
    try:
        study.optimize(
            objective,
            n_trials=max(0, n_trials - attempted),
            timeout=timeout_s,
            catch=(NonFiniteTraining,),
            callbacks=[lambda study, trial: _write_outputs(study, out_dir, n_trials)],
        )
    finally:
        summary = _write_outputs(study, out_dir, n_trials)
    return summary
