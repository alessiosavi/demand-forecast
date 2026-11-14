"""Score the fixed, equal-weight mean of aligned saved yardstick quantile forecasts.

The mean keeps quantiles ordered when each member's quantiles are ordered. It
is not the quantile function of the mixture distribution. Scores are computed
from the averaged forecasts, never averaged from member scores. No weights are
fitted and no missing member is replaced by a per-task fallback.

Scoring still needs the original task data: truth, history for scaled metrics,
item order and the dataset fingerprint. For disconnected use, supply local
data or a populated Hub/datasets cache and set HF_DATASETS_OFFLINE=1 and
HF_HUB_OFFLINE=1. Checkpoints and inference packages are never loaded.

Member records identify packages and upstream implementations, but do not hash
this repository's evaluation code. The caller must ensure all members were
scored by the same evaluation code, including adapters and scorer. Cloud jobs record repo_sha
in inputs.json beside the mode folders; the cloud route will compare those SHAs.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.bench.run import _save_failures, _save_summaries, score_predictions
from demand_forecast.bench.specs import dataset_fingerprint

from .runner import (
    PROTOCOL_VERSION,
    Options,
    _safe_name,
    _save_window,
    _summary_matches,
    digest,
    fev_predictions,
    file_digest,
    prepare_windows,
    read_window,
)

# Allow only these member-specific differences. All other configuration keys,
# including unknown future keys, remain part of the evaluation identity.
MEMBER_OPTION_KEYS = frozenset({"weights", "revision", "weights_revision", "model_name"})
EVALUATION_OPTION_KEYS = frozenset(
    {"model", "processing", "covariates", "cross_series", "max_context", "batch_size", "device"}
)


def _evaluation_configuration(configuration: dict) -> dict:
    """Remove member labels/checkpoints and derived hashes, keeping evaluation keys."""
    return {
        "configuration": {
            key: value
            for key, value in configuration.items()
            if key not in {"configuration_identity", "options"}
        },
        "options": {
            key: value
            for key, value in configuration["options"].items()
            if key not in MEMBER_OPTION_KEYS
        },
    }


def evaluation_identity(record: dict) -> dict:
    """Task-specific evaluation keys; configurations are checked once per run."""
    return {
        "task": record["task"],
        "dataset_fingerprint": record["dataset_fingerprint"],
        "levels": record["levels"],
        "input_metadata": record.get("input_metadata", {}),
    }


def _window_paths(directory: Path, count: int) -> list[Path]:
    paths = [directory / f"window_{index}.npz" for index in range(count)]
    if set(directory.glob("window_*.npz")) != set(paths):
        raise ValueError(f"{directory}: window counts or indices differ")
    return paths


def _window_hashes(directory: Path, count: int) -> dict[str, str]:
    return {path.name: file_digest(path) for path in _window_paths(directory, count)}


def _read_configuration(directory: Path) -> dict:
    path = directory / "yardstick.json"
    if not path.is_file():
        raise ValueError(f"{path}: missing root manifest")
    configuration = json.loads(path.read_text())
    required = {
        "options",
        "configuration_identity",
        "packages",
        "protocol_version",
        "published_adapter_source",
    }
    if not isinstance(configuration, dict) or not required <= configuration.keys():
        raise ValueError(f"{path}: incomplete evaluation identity in configuration")
    options = configuration["options"]
    required = (EVALUATION_OPTION_KEYS - {"processing"}) | MEMBER_OPTION_KEYS
    if not isinstance(options, dict) or not required <= options.keys():
        raise ValueError(f"{path}: incomplete evaluation identity in options")
    if options["model"] == "simons" and (
        "processing" not in options
        or not {"simons_source", "simons_implementation"} <= configuration.keys()
    ):
        raise ValueError(f"{path}: incomplete evaluation identity for Simons")
    if not isinstance(options.get("weights_revision"), str) or not options["weights_revision"]:
        raise ValueError(f"{path}: missing immutable weights digest")
    expected_configuration = digest(
        {key: value for key, value in configuration.items() if key != "configuration_identity"}
    )
    if configuration["configuration_identity"] != expected_configuration:
        raise ValueError(f"{path}: inconsistent configuration identity")
    return configuration


def _bind_members(members: list[Path]) -> list[dict]:
    """Reject mixed directories before creating output or scoring any task."""
    configurations = [_read_configuration(member) for member in members]
    for directory, configuration in zip(members, configurations, strict=True):
        for path in directory.rglob("run.json"):
            record = json.loads(path.read_text())
            if (
                not isinstance(record, dict)
                or record.get("configuration") != configuration
                or record.get("configuration_identity") != configuration["configuration_identity"]
            ):
                raise ValueError(f"{path}: task configuration differs from root manifest")
    evaluation = _evaluation_configuration(configurations[0])
    for directory, configuration in zip(members[1:], configurations[1:], strict=True):
        if _evaluation_configuration(configuration) != evaluation:
            raise ValueError(f"{directory}: evaluation identity differs")
    return configurations


def _read_member(directory: Path, task, configuration: dict) -> dict:
    path = directory / task.task_name / "run.json"
    if not path.is_file():
        raise ValueError(f"{path}: missing completed task")
    record = json.loads(path.read_text())
    required = {
        "task",
        "configuration",
        "configuration_identity",
        "run_identity",
        "dataset_fingerprint",
        "levels",
        "window_sha256",
    }
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError(f"{path}: incomplete evaluation identity in task record")
    if record["task"] != task.to_dict():
        raise ValueError(f"{path}: task definition differs from --tasks")
    if (
        record["configuration"] != configuration
        or record["configuration_identity"] != configuration["configuration_identity"]
        or record["run_identity"]
        != digest(
            {key: record[key] for key in ("configuration_identity", "task", "dataset_fingerprint")}
        )
    ):
        raise ValueError(f"{path}: inconsistent run or configuration identity")
    expected_levels = sorted(set(task.quantile_levels) | {0.5})
    if record["levels"] != expected_levels:
        raise ValueError(f"{path}: quantile levels differ from --tasks")
    summary = record.get("summary")
    summary_path = directory / "summaries.csv"
    if not summary or not summary_path.is_file():
        raise ValueError(f"{path}: task is not complete (missing summary)")
    frame = pd.read_csv(summary_path, float_precision="round_trip")
    options = configuration["options"]
    rows = frame.loc[(frame.task == task.task_name) & (frame.model_name == options["model_name"])]
    if (
        len(rows) != 1
        or summary.get("run_identity") != record["run_identity"]
        or not _summary_matches(rows.iloc[0], summary)
    ):
        raise ValueError(f"{path}: task is not complete (summary mismatch)")
    failure_path = directory / "failures.json"
    if failure_path.exists() and task.task_name in json.loads(failure_path.read_text()):
        raise ValueError(f"{path}: task has a recorded failure")
    paths = _window_paths(path.parent, task.num_windows)
    if set(record["window_sha256"]) != {window.name for window in paths}:
        raise ValueError(f"{path}: window sha256 entries differ")
    return record


def _completed(directory: Path, identity: str, rows: pd.DataFrame, count: int) -> bool:
    path = directory / "run.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get("run_identity") != identity:
        raise ValueError(f"{path}: ensemble identity mismatch; use a new out-dir")
    if len(rows) != 1 or not record.get("summary"):
        return False
    if not _summary_matches(rows.iloc[0], record["summary"]):
        return False
    try:
        return _window_hashes(directory, count) == record.get("window_sha256")
    except (OSError, ValueError):
        return False


def _average_windows(task, members, records, windows, levels):
    """Read through the runner's validator, and retain only one member at a time."""
    averaged, durations = [], []
    for index, window in enumerate(windows):
        mean, elapsed = None, 0.0
        for directory, record in zip(members, records, strict=True):
            path = directory / task.task_name / f"window_{index}.npz"
            try:
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != record["window_sha256"][path.name]:
                    raise ValueError("window sha256 mismatch")
                array, duration = read_window(
                    BytesIO(raw), window, levels, record["run_identity"], task.horizon
                )
            except (ValueError, KeyError, OSError) as exc:
                raise ValueError(f"{path}: {exc}") from exc
            if not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"{path}: forecasts must be floating point")
            # Divide before summing to avoid overflowing for large finite members.
            # For two identical float64 members, halving and adding is exact
            # except at subnormal magnitudes; fev's scorer is otherwise identical.
            contribution = array.astype(np.float64) / len(members)
            if mean is None:
                mean = contribution
            else:
                mean += contribution
            elapsed += duration
        if not np.isfinite(mean).all() or not np.isfinite(elapsed):
            raise ValueError(f"{task.task_name} window {index}: non-finite ensemble or timing")
        averaged.append(mean)
        durations.append(elapsed)
    return averaged, durations


def run_ensemble(
    tasks,
    members: list[Path],
    model_name: str,
    out_dir: Path,
    *,
    keep_going: bool = False,
    allow_identical_members: bool = False,
) -> pd.DataFrame:
    """Average complete, comparable tasks; persist failures just like run_yardstick.

    A completed matching output is left byte-for-byte unchanged. Changed member
    directories or identities require a new output directory. Members are
    ordered by directory spelling so permutations have the same identity and
    floating-point accumulation order; directory spellings are retained.
    """
    tasks = list(tasks)
    members = sorted((Path(member) for member in members), key=str)
    if len(members) < 2 or not model_name.strip() or not tasks:
        raise ValueError("at least two members, tasks and a nonempty model-name are required")
    names = [task.task_name for task in tasks]
    for name in names:
        _safe_name(name)
    if len(set(names)) != len(names):
        raise ValueError("task names must be unique")
    out_dir = Path(out_dir)
    for member in members:
        if out_dir.resolve().is_relative_to(member.resolve()) or member.resolve().is_relative_to(
            out_dir.resolve()
        ):
            raise ValueError("out-dir must not overlap a member directory")
    configurations = _bind_members(members)
    revisions = [configuration["options"]["weights_revision"] for configuration in configurations]
    if not allow_identical_members and len(set(revisions)) != len(revisions):
        raise ValueError("identical weights digests; use --allow-identical-members to reproduce")
    evaluation_configuration = _evaluation_configuration(configurations[0])
    weights = [1 / len(members)] * len(members)
    run = {
        "ensemble_version": 1,
        "members": [str(p) for p in members],
        "member_configurations": {
            str(member): configuration["configuration_identity"]
            for member, configuration in zip(members, configurations, strict=True)
        },
        "weights": weights,
        "model_name": model_name,
    }
    run["configuration_identity"] = digest(run)
    manifest_path = out_dir / "yardstick.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != run:
            raise ValueError("ensemble members or model identity mismatch; use a new out-dir")
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("out-dir is nonempty and has no ensemble manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _save_failures(manifest_path, run)
    summary_path, failure_path = out_dir / "summaries.csv", out_dir / "failures.json"
    frame = (
        pd.read_csv(summary_path, float_precision="round_trip")
        if summary_path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    for task in tasks:
        name, identity = task.task_name, None
        selected = (frame.task == name) & (frame.model_name == model_name)
        directory = out_dir / name
        try:
            records = [
                _read_member(member, task, configuration)
                for member, configuration in zip(members, configurations, strict=True)
            ]
            task_evaluation = evaluation_identity(records[0])
            for member, record in zip(members[1:], records[1:], strict=True):
                if evaluation_identity(record) != task_evaluation:
                    raise ValueError(f"{name}: {member}: evaluation identity differs")
            evaluation = {**evaluation_configuration, **task_evaluation}
            sources = [
                {
                    "directory": str(member),
                    "run_identity": record["run_identity"],
                    "window_sha256": record["window_sha256"],
                }
                for member, record in zip(members, records, strict=True)
            ]
            configuration = {**run, "evaluation": evaluation, "members": sources}
            configuration.pop("configuration_identity")
            configuration["configuration_identity"] = digest(configuration)
            payload = {
                "configuration_identity": configuration["configuration_identity"],
                "task": task.to_dict(),
                "dataset_fingerprint": evaluation["dataset_fingerprint"],
            }
            identity = digest(payload)
        except Exception as exc:
            _record_failure(failures, failure_path, name, identity, exc)
            if not keep_going:
                raise
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        # An output identity conflict is a refusal, not a failed evaluation that
        # should remove an already successful row or alter its audit evidence.
        completed = _completed(directory, identity, frame.loc[selected], task.num_windows)
        try:
            if completed:
                # No deserialization is needed on a no-op resume, but corrupt
                # sources must still fail without altering the completed output.
                for member, record in zip(members, records, strict=True):
                    if _window_hashes(member / name, task.num_windows) != record["window_sha256"]:
                        raise ValueError(f"{member / name}: window sha256 mismatch")
                print(f"Skipping completed task {name}", flush=True)
                if name in failures:
                    del failures[name]
                    _save_failures(failure_path, failures)
                continue
            if dataset_fingerprint(task) != evaluation["dataset_fingerprint"]:
                raise ValueError(f"{name}: task dataset fingerprint differs from members")
            options = records[0]["configuration"]["options"]
            windows = prepare_windows(task, Options(**options))
            if len(windows) != task.num_windows:
                raise ValueError(f"{name}: task window count differs")
            levels = evaluation["levels"]
            arrays, durations = _average_windows(task, members, records, windows, levels)
            summary = score_predictions(
                task,
                [
                    fev_predictions(array, levels, window.targets)
                    for array, window in zip(arrays, windows, strict=True)
                ],
                model_name=model_name,
                training_time_s=0.0,
                inference_time_s=sum(durations),
                trained_on_this_dataset=False,
            )
            summary.update(
                task=name,
                run_identity=identity,
                yardstick_protocol=PROTOCOL_VERSION,
                configuration_identity=configuration["configuration_identity"],
                **{
                    **options,
                    "model": "ensemble",
                    "model_name": model_name,
                    "weights": json.dumps([str(p) for p in members]),
                    "revision": None,
                    "weights_revision": digest({"members": revisions}),
                },
                **evaluation["input_metadata"],
            )
            directory.mkdir(parents=True, exist_ok=True)
            paths = [directory / f"window_{index}.npz" for index in range(len(windows))]
            for path, window, array, duration in zip(
                paths, windows, arrays, durations, strict=True
            ):
                _save_window(path, window, array, levels, identity, duration)
            record = {
                **payload,
                "configuration": configuration,
                "run_identity": identity,
                "summary": summary,
                "levels": levels,
                "members": sources,
                "weights": weights,
                "window_sha256": {path.name: file_digest(path) for path in paths},
            }
            if evaluation["input_metadata"]:
                record["input_metadata"] = evaluation["input_metadata"]
            _save_failures(directory / "run.json", record)
            row = pd.DataFrame([summary])
            frame = frame.loc[~selected]
            frame = row if frame.empty else pd.concat([frame, row], ignore_index=True)
            _save_summaries(frame, summary_path)
            if name in failures:
                del failures[name]
                _save_failures(failure_path, failures)
        except Exception as exc:
            _record_failure(failures, failure_path, name, identity, exc)
            if not keep_going:
                raise
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    if not summary_path.exists():
        _save_summaries(frame, summary_path)
    result = pd.read_csv(summary_path, float_precision="round_trip")
    result.attrs["failed_tasks"] = [name for name in names if name in failures]
    return result


def _record_failure(failures, path, name, identity, exc):
    failures[name] = {
        "task": name,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc().splitlines()[-20:],
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "run_identity": identity,
    }
    _save_failures(path, failures)
