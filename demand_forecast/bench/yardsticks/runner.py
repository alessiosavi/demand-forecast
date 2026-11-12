"""Shared fev inputs, inference ablations, and atomic evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

from demand_forecast.bench.run import _save_failures, _save_summaries, score_predictions
from demand_forecast.bench.specs import dataset_fingerprint

PROTOCOL_VERSION = 1
FEV_SOURCE = "9cf943c388e5cfaf352175ea3c69e408b197d997"
DEFAULT_WEIGHTS = {
    "chronos2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
    "timestral": ("mistralai/Timestral-5M", "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"),
}


@dataclass(frozen=True)
class Options:
    """Every inference switch, including the resolved immutable checkpoint identity."""

    model: str
    weights: str
    model_name: str
    revision: str | None = None
    max_context: int | None = None
    covariates: str = "all"
    cross_series: str = "on"
    batch_size: int | None = None
    device: str = "cuda"
    weights_revision: str | None = None

    def resolved(self) -> Options:
        """Validate choices and fill only published batching/context defaults."""
        if self.model not in DEFAULT_WEIGHTS:
            raise ValueError("model must be chronos2 or timestral")
        if not self.model_name.strip():
            raise ValueError("model-name must be nonempty")
        if self.covariates not in ("all", "none") or self.cross_series not in ("on", "off"):
            raise ValueError("invalid covariates or cross-series switch")
        if self.model == "timestral" and self.cross_series == "off":
            raise ValueError("cross-series off is supported only for Chronos-2")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for name in ("max_context", "batch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be positive")
        return replace(
            self, batch_size=self.batch_size or (100 if self.model == "chronos2" else 1024)
        )


@dataclass
class SeriesInput:
    """One model group; covariates include known history and future separately."""

    item_index: int
    target_indices: tuple[int, ...]
    target: np.ndarray
    past: dict[str, np.ndarray]
    future: dict[str, np.ndarray]


@dataclass
class WindowInput:
    """Positional identity and model groups of one fev evaluation window."""

    ids: list[str]
    targets: list[str]
    items: list[SeriesInput]


class Adapter(Protocol):
    """Adapters return one (variates, horizon, quantiles) array per model group."""

    def predict(
        self, windows: list[WindowInput], horizon: int, levels: list[float]
    ) -> list[list[np.ndarray]]:
        """Forecast all windows, preserving group order and using original units."""
        ...


def digest(value: dict) -> str:
    """Hash canonical, finite JSON for both manifests and resume identities."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without loading checkpoint tensors into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def resolve_weights(options: Options) -> Options:
    """Pin Hub aliases before model loading; content-hash local checkpoint files."""
    path = Path(options.weights).expanduser()
    if path.is_dir():
        files = {
            str(p.relative_to(path)): file_digest(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(path).parts
        }
        if not files:
            raise ValueError("local weights directory is empty")
        return replace(
            options, weights=str(path.resolve()), weights_revision="local-sha256:" + digest(files)
        )
    if options.weights == DEFAULT_WEIGHTS[options.model][0] and options.revision is None:
        revision = DEFAULT_WEIGHTS[options.model][1]
    else:
        revision = options.revision
    if revision is None or not re.fullmatch("[0-9a-f]{40}", revision):
        from huggingface_hub import HfApi

        revision = HfApi().model_info(options.weights, revision=revision).sha
    if not revision or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("weights revision could not be resolved to a commit SHA")
    return replace(options, weights_revision=revision)


def runtime_identity(options: Options) -> dict:
    """Include the inference implementation and numerical runtime in resume checks."""
    packages = {}
    for name in (
        "fev",
        "datasets",
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "chronos-forecasting" if options.model == "chronos2" else "timestral",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "options": asdict(options),
        "packages": packages,
        "protocol_version": PROTOCOL_VERSION,
        "published_adapter_source": FEV_SOURCE,
    }


def _safe_name(name: str) -> None:
    if (
        not name
        or Path(name).is_absolute()
        or any(p in (".", "..", "") for p in name.split("/"))
        or "\\" in name
    ):
        raise ValueError(f"unsafe task name: {name!r}")


def read_names(value: str | None) -> list[str] | None:
    """Parse CSV names or @file selections without importing repository scripts."""
    if value is None:
        return None
    if value.startswith("@"):
        value = Path(value[1:]).read_text(encoding="utf-8")
    names = [p.strip() for line in value.splitlines() for p in line.split(",") if p.strip()]
    if not names or len(set(names)) != len(names):
        raise ValueError("names must be nonempty and unique")
    return names


def _summary_matches(row, expected):
    for key, value in expected.items():
        actual = row.get(key)
        if value is None:
            matches = pd.isna(actual)
        elif isinstance(value, bool):
            matches = str(actual).lower() == str(value).lower()
        elif isinstance(value, (int, float)):
            matches = pd.to_numeric(actual, errors="coerce") == value
        else:
            matches = actual == str(value)
        if not matches:
            return False
    return True


def load_tasks(path: Path, names: list[str] | None = None) -> list:
    """Load native fev YAML, including custom tasks without our training profiles."""
    import fev

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise ValueError("task YAML must contain a nonempty tasks list")
    tasks = {}
    for raw in data["tasks"]:
        values = dict(raw)
        if "name" in values:
            values["task_name"] = values.pop("name")
        source = str(values["dataset_path"])
        dataset = Path(source).expanduser()
        if not urlparse(source).scheme and (
            dataset.suffix in (".parquet", ".arrow") or source.startswith(("./", "../", "~", "/"))
        ):
            values["dataset_path"] = str((path.parent / dataset).resolve())
            values["dataset_config"] = None
        task = fev.Task(**values)
        name = task.task_name
        _safe_name(name)
        if name in tasks:
            raise ValueError(f"duplicate task name: {name}")
        tasks[name] = task
    if names is not None and (
        not names or len(set(names)) != len(names) or set(names) - tasks.keys()
    ):
        raise ValueError("names must be unique and match tasks in the YAML")
    return [tasks[name] for name in (names if names is not None else tasks)]


def prepare_windows(task, options: Options) -> list[WindowInput]:
    """Apply history/role/group ablations before an adapter sees the input."""
    windows = []
    targets = list(task.target_columns)
    for window in task.iter_windows():
        past, future = window.get_input_data()
        future_by_id = {str(row[task.id_column]): row for row in future}
        ids, items = [], []
        for i, row in enumerate(past):
            key = str(row[task.id_column])
            ids.append(key)
            target = np.stack([np.asarray(row[t], dtype=np.float64) for t in targets])
            # Default caps are checkpoint settings, enforced by each adapter. An
            # explicit ablation is applied here to every dynamic history column.
            start = max(0, target.shape[-1] - options.max_context) if options.max_context else 0
            history, known = {}, {}
            if options.covariates == "all":
                history = {
                    c: np.asarray(row[c])[start:]
                    for c in task.past_dynamic_columns + task.known_dynamic_columns
                }
                known = {c: np.asarray(future_by_id[key][c]) for c in task.known_dynamic_columns}
            if options.cross_series == "off":
                for j in range(len(targets)):
                    items.append(
                        SeriesInput(
                            i, (j,), target[j : j + 1, start:], history.copy(), known.copy()
                        )
                    )
            else:
                items.append(
                    SeriesInput(i, tuple(range(len(targets))), target[:, start:], history, known)
                )
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate fev item ids")
        windows.append(WindowInput(ids, targets, items))
    return windows


def assemble(
    window: WindowInput, outputs: list[np.ndarray], horizon: int, levels: list[float]
) -> np.ndarray:
    """Restore item/target order and reject malformed or non-finite predictions."""
    arrays = np.full((len(window.ids), len(window.targets), horizon, len(levels)), np.nan)
    for item, output in zip(window.items, outputs, strict=True):
        output = np.asarray(output, dtype=np.float64)
        if output.shape != (len(item.target_indices), horizon, len(levels)):
            raise ValueError("adapter forecast shape mismatch")
        arrays[item.item_index, list(item.target_indices)] = output
    if not np.isfinite(arrays).all():
        raise ValueError("adapter returned non-finite forecasts")
    return arrays


def fev_predictions(arrays: np.ndarray, levels: list[float], targets: list[str]) -> dict:
    """Use medians as point forecasts, including when task quantiles omit 0.5."""
    median = levels.index(0.5)
    return {
        target: [
            {
                "predictions": a[:, median].tolist(),
                **{str(q): a[:, j].tolist() for j, q in enumerate(levels)},
            }
            for a in arrays[:, v]
        ]
        for v, target in enumerate(targets)
    }


def _save_window(path, window, arrays, levels, identity, elapsed):
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(
                stream,
                ids=np.asarray(window.ids, dtype=str),
                targets=np.asarray(window.targets, dtype=str),
                arrays=arrays,
                quantiles=np.asarray(levels),
                identity=identity,
                inference_time_s=float(elapsed),
            )
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_window(
    path: Path, window: WindowInput, levels: list[float], identity: str, horizon: int
) -> tuple[np.ndarray, float]:
    """Validate audit evidence before resuming or recomputing official scores."""
    with np.load(path, allow_pickle=False) as saved:
        if saved["identity"].item() != identity:
            raise ValueError("run identity mismatch in window")
        if (
            saved["ids"].tolist() != window.ids
            or saved["targets"].tolist() != window.targets
            or saved["quantiles"].tolist() != levels
        ):
            raise ValueError("window input identity mismatch")
        arrays = saved["arrays"]
        if (
            arrays.shape != (len(window.ids), len(window.targets), horizon, len(levels))
            or not np.isfinite(arrays).all()
        ):
            raise ValueError("invalid saved forecasts")
        elapsed = saved["inference_time_s"].item()
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid saved inference timing")
        return arrays, elapsed


def _adapter(options, task):
    from .adapters import Chronos2Adapter, TimestralAdapter

    return (Chronos2Adapter if options.model == "chronos2" else TimestralAdapter)(options, task)


def run_yardstick(
    tasks, options: Options, out_dir: Path, *, resume=False, keep_going=False, adapter_factory=None
) -> pd.DataFrame:
    """Evaluate tasks atomically; resume requires matching settings and dataset content.

    A directory holds one yardstick configuration. Use a separate directory for
    each ablation. Existing baseline rows are retained. Model loading is lazy.
    """
    options = resolve_weights(options.resolved())
    run = runtime_identity(options)
    run["configuration_identity"] = digest(run)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "yardstick.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != run:
        raise ValueError(
            "run identity mismatch; use a new out-dir for changed settings, weights or runtime"
        )
    if not manifest_path.exists():
        _save_failures(manifest_path, run)
    summary_path, failure_path = out_dir / "summaries.csv", out_dir / "failures.json"
    frame = (
        pd.read_csv(summary_path, float_precision="round_trip")
        if summary_path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    adapter = None
    attempted = []
    for task in tasks:
        name, identity = task.task_name, None
        attempted.append(name)
        _safe_name(name)
        directory = out_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / "run.json"
        try:
            fingerprint = dataset_fingerprint(task)
            payload = {
                "configuration_identity": run["configuration_identity"],
                "task": task.to_dict(),
                "dataset_fingerprint": fingerprint,
            }
            identity = digest(payload)
            if (
                record_path.exists()
                and json.loads(record_path.read_text()).get("run_identity") != identity
            ):
                raise ValueError(f"{name}: run identity mismatch; task or dataset changed")
            windows = prepare_windows(task, options)
            # Include the median internally even for tasks requesting other levels.
            levels = sorted(set(task.quantile_levels) | {0.5})
            paths = [directory / f"window_{i}.npz" for i in range(len(windows))]
            rows = frame.loc[(frame.task == name) & (frame.model_name == options.model_name)]
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            valid = (
                resume
                and len(rows) == 1
                and rows.iloc[0].get("run_identity") == identity
                and record.get("summary") is not None
            )
            valid = valid and all(
                p.exists() and file_digest(p) == record.get("window_sha256", {}).get(p.name)
                for p in paths
            )
            if valid:
                # Detect altered summary scores as well as altered NPZ files.
                valid = _summary_matches(rows.iloc[0], record["summary"])
            if valid:
                print(f"Skipping completed task {name}", flush=True)
                if name in failures:
                    del failures[name]
                    _save_failures(failure_path, failures)
                continue
            # A failed rerun must never leave a stale successful row.
            frame = frame.loc[~((frame.task == name) & (frame.model_name == options.model_name))]
            if summary_path.exists():
                _save_summaries(frame, summary_path)
            arrays, elapsed = [], 0.0
            cached = resume and all(
                p.exists() and file_digest(p) == record.get("window_sha256", {}).get(p.name)
                for p in paths
            )
            if cached:
                for p, w in zip(paths, windows, strict=True):
                    a, duration = read_window(p, w, levels, identity, task.horizon)
                    arrays.append(a)
                    elapsed += duration
            else:
                if adapter_factory is not None:
                    adapter = adapter_factory(options)
                else:
                    # Timestral classifies dynamic types per task; weights are
                    # retained across tasks by the adapter's prepare method.
                    if adapter is None:
                        adapter = _adapter(options, task)
                    else:
                        adapter.prepare(task)
                start = time.perf_counter()
                outputs = adapter.predict(windows, task.horizon, levels)
                elapsed = time.perf_counter() - start
                if len(outputs) != len(windows):
                    raise ValueError("adapter window count mismatch")
                for p, w, output in zip(paths, windows, outputs, strict=True):
                    a = assemble(w, output, task.horizon, levels)
                    arrays.append(a)
                    _save_window(p, w, a, levels, identity, elapsed / len(windows))
            predictions = [
                fev_predictions(a, levels, w.targets) for a, w in zip(arrays, windows, strict=True)
            ]
            summary = score_predictions(
                task,
                predictions,
                model_name=options.model_name,
                training_time_s=0.0,
                inference_time_s=elapsed,
                trained_on_this_dataset=False,
            )
            summary.update(
                task=name,
                run_identity=identity,
                yardstick_protocol=PROTOCOL_VERSION,
                configuration_identity=run["configuration_identity"],
                **asdict(options),
            )
            # model is the adapter selector; model_name remains the user's label.
            record = {
                **payload,
                "configuration": run,
                "run_identity": identity,
                "summary": summary,
                "levels": levels,
                "window_sha256": {p.name: file_digest(p) for p in paths},
            }
            _save_failures(record_path, record)
            row = pd.DataFrame([summary])
            frame = row if frame.empty else pd.concat([frame, row], ignore_index=True)
            _save_summaries(frame, summary_path)
            if name in failures:
                del failures[name]
                _save_failures(failure_path, failures)
        except Exception as exc:
            failures[name] = {
                "task": name,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-20:],
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "run_identity": identity,
            }
            _save_failures(failure_path, failures)
            if not keep_going:
                raise
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    if not summary_path.exists():
        _save_summaries(frame, summary_path)
    result = pd.read_csv(summary_path, float_precision="round_trip")
    result.attrs["failed_tasks"] = [name for name in attempted if name in failures]
    return result
