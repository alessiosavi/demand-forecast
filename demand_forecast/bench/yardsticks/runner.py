"""Shared fev inputs, inference ablations, and atomic evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

from demand_forecast.bench.run import _save_failures, _save_summaries, score_predictions
from demand_forecast.bench.specs import dataset_fingerprint

if TYPE_CHECKING:
    from demand_forecast.bench.v5_panels import ParsedFrequency, TimeAxis
    from demand_forecast.models.v5 import Evidence

PROTOCOL_VERSION = 1
FEV_SOURCE = "9cf943c388e5cfaf352175ea3c69e408b197d997"
SIMONS_SOURCE = "8e338e915d26c8960713889ef12ab1cf5c08d591"
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
    processing: str | None = None
    variant: str | None = None

    def resolved(self) -> Options:
        """Validate choices and fill only published batching/context defaults."""
        if self.model not in (*DEFAULT_WEIGHTS, "simons", "v5"):
            raise ValueError("model must be chronos2, timestral, simons or v5")
        if self.model != "v5" and self.variant is not None:
            raise ValueError("--variant is supported only for v5")
        if self.model == "v5":
            from demand_forecast.data.episode.schema import PATCH

            from .v5_variants import VARIANT_NAMES

            if self.processing != "raw":
                raise ValueError("v5 requires --processing raw")
            if self.revision is not None:
                raise ValueError("v5 uses a local export directory; --revision is unsupported")
            if self.cross_series != "on" or self.covariates != "all":
                raise ValueError("v5 requires --cross-series on and --covariates all")
            if self.variant is not None and self.variant not in VARIANT_NAMES:
                raise ValueError(f"unknown v5 variant {self.variant!r}")
            if self.max_context is not None and (
                type(self.max_context) is not int or self.max_context < PATCH
            ):
                raise ValueError(f"v5 --max-context must be an integer of at least {PATCH}")
        elif self.model == "simons":
            if self.processing not in ("raw", "published"):
                raise ValueError("Simons requires --processing raw|published")
            if self.revision is not None:
                raise ValueError(
                    "Simons uses a local exported directory; --revision is unsupported"
                )
        elif self.processing is not None:
            raise ValueError("--processing is supported only for Simons and v5")
        if not self.model_name.strip():
            raise ValueError("model-name must be nonempty")
        if self.covariates not in ("all", "none") or self.cross_series not in ("on", "off"):
            raise ValueError("invalid covariates or cross-series switch")
        if self.model != "chronos2" and self.cross_series == "off":
            raise ValueError("cross-series off is supported only for Chronos-2")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for name in ("max_context", "batch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be positive")
        return replace(
            self,
            batch_size=self.batch_size
            or {"chronos2": 100, "timestral": 1024, "simons": 256, "v5": 64}[self.model],
        )


@dataclass
class SeriesInput:
    """One model group; covariates include known history and future separately."""

    item_index: int
    target_indices: tuple[int, ...]
    target: np.ndarray
    past: dict[str, np.ndarray]
    future: dict[str, np.ndarray]
    axis: TimeAxis | None = None
    static: dict[str, object] | None = None


@dataclass
class WindowInput:
    """Positional identity and model groups of one fev evaluation window."""

    ids: list[str]
    targets: list[str]
    items: list[SeriesInput]
    freq: str | None = None
    declared: dict[str, str] | None = None
    class_labels: dict[str, tuple[str, ...]] | None = None
    parsed_frequency: ParsedFrequency | None = None


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
    if options.model == "v5":
        from demand_forecast.models.v5.export import export_digest

        if not path.is_dir():
            raise ValueError("v5 requires a local export directory")
        return replace(
            options,
            weights=str(path.resolve()),
            weights_revision="local-sha256:" + export_digest(path),
        )
    if options.model == "simons":
        if not path.is_dir():
            raise ValueError("Simons requires a local exported checkpoint directory")
        names = ("config.json", "model.safetensors")
        if any(not (path / name).is_file() for name in names):
            raise ValueError("Simons export must contain config.json and model.safetensors")
        return replace(
            options,
            weights=str(path.resolve()),
            weights_revision="local-sha256:"
            + digest({name: file_digest(path / name) for name in names}),
        )
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


def _options_payload(options: Options) -> dict:
    values = asdict(options)
    # Keep the exact serialization used by existing configuration/window hashes.
    if options.model != "v5":
        values.pop("variant")
    if options.model not in ("simons", "v5"):
        values.pop("processing")
    return values


def _simons_implementation() -> dict:
    # Distribution versions alone do not distinguish upstream commits or edits
    # in editable installs. Inspect source without importing/loading the model.
    commit = None
    try:
        direct_url = distribution("simons").read_text("direct_url.json")
        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except PackageNotFoundError:
        pass
    try:
        spec = find_spec("simons")
    except ValueError:
        # Recording modules injected by offline tests may have no __spec__.
        spec = None
    files = {}
    if spec is not None and spec.origin is not None:
        root = Path(spec.origin).parent
        files = {
            str(path.relative_to(root)): file_digest(path)
            for path in sorted(root.rglob("*.py"))
            if path.is_file()
        }
    return {"vcs_commit": commit, "source_sha256": digest(files) if files else None}


def _source_sha256(root: Path) -> str:
    result = hashlib.sha256()
    files = (path for path in root.rglob("*.py") if path.is_file())
    for path in sorted(files, key=lambda path: path.relative_to(root).as_posix()):
        result.update(path.relative_to(root).as_posix().encode())
        result.update(b"\0")
        result.update(path.read_bytes())
        result.update(b"\0")
    return result.hexdigest()


def runtime_identity(options: Options) -> dict:
    """Include the inference implementation and numerical runtime in resume checks."""
    model_packages = {
        "chronos2": ("chronos-forecasting",),
        "timestral": ("timestral",),
        "simons": ("simons",),
        "v5": ("scipy", "safetensors"),
    }
    packages = {}
    for name in (
        "fev",
        "datasets",
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        *model_packages[options.model],
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    identity = {
        "options": _options_payload(options),
        "packages": packages,
        "protocol_version": PROTOCOL_VERSION,
        "published_adapter_source": FEV_SOURCE,
    }
    if options.model == "simons":
        identity["simons_source"] = SIMONS_SOURCE
        identity["simons_implementation"] = _simons_implementation()
    if options.model == "v5":
        identity["v5_implementation"] = {
            "source_sha256": _source_sha256(Path(__file__).resolve().parents[2])
        }
    return identity


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
    is_v5 = options.model == "v5"
    freq = declared = class_labels = None
    if is_v5:
        from datasets import ClassLabel

        from demand_forecast.bench.v5_panels import _positive_offset, _prove_time_axis

    for window in task.iter_windows():
        past, future = window.get_input_data()
        if is_v5 and declared is None:
            try:
                freq = task.freq
            except ValueError:
                freq = None
            declared, class_labels = {}, {}
            columns = task.past_dynamic_columns + task.known_dynamic_columns + task.static_columns
            # fev reconstructs the split datasets and loses ClassLabel metadata.
            for col in columns:
                feature = window.full_dataset.features[col]
                while hasattr(feature, "feature"):
                    feature = feature.feature
                dtype = getattr(feature, "dtype", "")
                if isinstance(feature, ClassLabel):
                    declared[col] = "class_label"
                    class_labels[col] = tuple(feature.names)
                elif dtype == "bool":
                    declared[col] = "boolean"
                elif dtype.startswith(("int", "uint", "float", "decimal")):
                    declared[col] = "number"
                elif dtype.startswith(("string", "large_string")):
                    declared[col] = "string"
        parsed_frequency = _positive_offset(freq) if is_v5 else None
        future_by_id = {str(row[task.id_column]): row for row in future}
        ids, items = [], []
        for i, row in enumerate(past):
            key = str(row[task.id_column])
            ids.append(key)
            target = np.stack([np.asarray(row[t], dtype=np.float64) for t in targets])
            # Reference adapters receive truncated histories for an explicit cap.
            # v5 needs the full histories for statistics and caps encoder input.
            start = 0
            if not is_v5 and options.max_context:
                start = max(0, target.shape[-1] - options.max_context)
            axis = static = None
            if parsed_frequency is not None:
                axis = _prove_time_axis(
                    np.asarray(row[task.timestamp_column]),
                    parsed_frequency,
                    future=np.asarray(future_by_id[key][task.timestamp_column]),
                    horizon=task.horizon,
                )
                static = {col: row[col] for col in task.static_columns}
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
                            i,
                            (j,),
                            target[j : j + 1, start:],
                            history.copy(),
                            known.copy(),
                            axis,
                            static,
                        )
                    )
            else:
                items.append(
                    SeriesInput(
                        i,
                        tuple(range(len(targets))),
                        target[:, start:],
                        history,
                        known,
                        axis,
                        static,
                    )
                )
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate fev item ids")
        windows.append(
            WindowInput(ids, targets, items, freq, declared, class_labels, parsed_frequency)
        )
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


def _save_v5_evidence(path: Path, evidence: Evidence, identity: str) -> None:
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(
                stream,
                item_position=np.asarray([i for i, _ in evidence.requests], dtype=np.int64),
                target=np.asarray([target or "" for _, target in evidence.requests], dtype=str),
                joint=np.asarray([target is None for _, target in evidence.requests], dtype=bool),
                candidates=np.asarray(evidence.candidates, dtype=str),
                rules=np.asarray(evidence.rules, dtype=str),
                status=evidence.status,
                rule=evidence.rule,
                gate=evidence.gate,
                score=evidence.score,
                run_identity=np.asarray(identity, dtype=str),
            )
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _adapter(options, task):
    if options.model == "v5":
        from .v5_adapter import V5Adapter

        return V5Adapter(options, task)
    from .adapters import Chronos2Adapter, SimonsAdapter, TimestralAdapter

    return {"chronos2": Chronos2Adapter, "timestral": TimestralAdapter, "simons": SimonsAdapter}[
        options.model
    ](options, task)


def run_yardstick(
    tasks, options: Options, out_dir: Path, *, resume=False, keep_going=False, adapter_factory=None
) -> pd.DataFrame:
    """Evaluate tasks atomically; resume requires matching settings and dataset content.

    A directory holds one yardstick configuration. Use a separate directory for
    each ablation. Existing baseline rows are retained. Model loading is lazy.
    """
    options = resolve_weights(options.resolved())
    is_v5 = options.model == "v5"
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
            input_metadata = {}
            if options.model == "simons":
                # Count column occurrences per item/window, before the adapter
                # splits oversized groups; no model load is needed on resume.
                input_metadata["dropped_past_only_covariates"] = sum(
                    len(item.past.keys() - item.future.keys())
                    for window in windows
                    for item in window.items
                )
            # Include the median internally even for tasks requesting other levels.
            levels = sorted(set(task.quantile_levels) | {0.5})
            paths = [directory / f"window_{i}.npz" for i in range(len(windows))]
            rows = frame.loc[(frame.task == name) & (frame.model_name == options.model_name)]
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            if is_v5:
                report_path = directory / "v5-report.json"
                evidence_paths = [
                    directory / f"evidence_window_{i}.npz" for i in range(len(windows))
                ]
                evidence_files = [report_path, *evidence_paths]
                evidence_digests = record.get("evidence_sha256", {})
            evidence_valid = not is_v5 or (
                resume
                and all(
                    p.is_file() and file_digest(p) == evidence_digests.get(p.name)
                    for p in evidence_files
                )
            )
            valid = (
                resume
                and evidence_valid
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
            cached = (
                resume
                and evidence_valid
                and all(
                    p.exists() and file_digest(p) == record.get("window_sha256", {}).get(p.name)
                    for p in paths
                )
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
                if is_v5:
                    evidence = adapter.evidence()
                    if len(evidence) != len(windows):
                        raise ValueError("adapter evidence window count mismatch")
                    for path, details in zip(evidence_paths, evidence, strict=True):
                        _save_v5_evidence(path, details, identity)
                    _save_failures(report_path, {**adapter.metadata(), "run_identity": identity})
                for p, w, output in zip(paths, windows, outputs, strict=True):
                    a = assemble(w, output, task.horizon, levels)
                    arrays.append(a)
                    _save_window(p, w, a, levels, identity, elapsed / len(windows))
            if is_v5:
                input_metadata = json.loads(report_path.read_text())
                del input_metadata["run_identity"]
                input_metadata.pop("v5_worker_reason", None)
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
                **_options_payload(options),
                **input_metadata,
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
            if options.model == "simons":
                record["input_metadata"] = input_metadata
            if is_v5:
                record["evidence_sha256"] = {p.name: file_digest(p) for p in evidence_files}
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
