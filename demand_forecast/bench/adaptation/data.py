"""Export only permitted task-local histories; schema version 2 is in docs/adaptation.md.

No dataset fingerprint is used: fev's full-data hash includes outer labels.
The public task loader and the scoring adapter are read, never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from demand_forecast.bench.yardsticks.runner import FEV_SOURCE, SIMONS_SOURCE, load_tasks

SCHEMA_VERSION = 2
MAX_VARIATES = 32
PHASES = ("development", "confirmation", "official")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isinf(value):
            raise ValueError("infinite values are unsupported")
        return None if math.isnan(value) else value
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp(times, index):
    return times[index].isoformat() if 0 <= index < len(times) else None


@dataclass
class _Item:
    item_id: str
    row: dict
    times: pd.DatetimeIndex
    outer: int
    boundary: int
    train_end: int = 0
    reason: str | None = None
    training_input_sha256: str | None = None

    def origin(self, index):
        return {"index": index, "timestamp": _timestamp(self.times, index)}


def _resolve_items(task, dataset, targets, known) -> list[_Item]:
    # Project away dropped columns before materializing rows. Their values and
    # vocabulary must never influence adaptation, including its identity.
    # Raw cells preserve timestamp zones, scoring precision and source hashes.
    # Model/training slices separately apply the unchanged scorer's formatter.
    dataset = dataset.select_columns(
        [task.id_column, task.timestamp_column, *targets, *known]
    ).with_format(None)
    items = []
    cutoff = task.cutoffs[0]
    for row in dataset:
        key = str(row[task.id_column])
        times = pd.DatetimeIndex(row[task.timestamp_column])
        if times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
            raise ValueError(f"{key}: timestamps must be strictly monotonic and non-null")
        if len(times) and not times.equals(
            pd.date_range(times[0], periods=len(times), freq=task.freq)
        ):
            raise ValueError(f"{key}: timestamps do not follow task frequency {task.freq}")
        n = len(times)
        if any(len(row[c]) != n for c in [*targets, *known]):
            raise ValueError(f"{key}: target/covariate lengths must match timestamps")
        if isinstance(cutoff, (int, np.integer)) and not isinstance(cutoff, bool):
            outer = max(0, min(n, int(cutoff if cutoff >= 0 else n + cutoff)))
        else:
            import pyarrow as pa
            from fev.utils import _compute_cutoff_indices

            try:
                # Use the pinned splitter's inclusive count and Arrow cast,
                # including source timestamp units/zones. No grid snapping or
                # step offset: task.cutoffs[0] is fev's first window unchanged.
                timestamps = pa.array(
                    [row[task.timestamp_column]],
                    type=dataset.features[task.timestamp_column].pa_type,
                )
                outer = int(
                    _compute_cutoff_indices(timestamps, cutoff, np.array([n]), np.array([0, n]))[0]
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key}: unresolvable date cut-off {cutoff!r}") from exc
        item = _Item(key, row, times, outer, outer - 2 * task.horizon)
        if outer < task.min_context_length or n - outer < task.horizon or outer > n:
            item.reason = "insufficient_outer_history"
        elif item.boundary < max(1, task.min_context_length):
            item.reason = "insufficient_inner_history"
        items.append(item)
    items.sort(key=lambda item: item.item_id)
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("duplicate item ids")
    return items


def _numeric(values, *, dtype=np.float64):
    result = np.asarray(values, dtype=dtype)
    if np.isinf(result).any():
        raise ValueError("infinite target/covariate values are unsupported")
    return result


def _scorer_slice(dataset, row, columns, start, end):
    """Format only permitted cells, as fev does after slicing each window.

    Formatting the full series first could let forbidden cells affect dtype
    inference. fev retains the format type, but resets its keyword arguments.
    """
    values = {c: row[c][start:end] for c in columns}
    formatter = getattr(dataset, "format", {}).get("type")
    if formatter is None or not values:
        return values
    import datasets

    bounded = datasets.Dataset.from_dict(
        {c: [v] for c, v in values.items()},
        features=datasets.Features({c: dataset.features[c] for c in columns}),
    ).with_format(formatter)
    return next(iter(bounded))


def _encode_training(values):
    """Native covariate rows are numeric; categorical vocabulary is history-only."""
    values = list(values)
    observed = [v for v in values if v is not None and not pd.isna(v)]
    if all(isinstance(v, (int, float, bool, np.number, np.bool_)) for v in observed):
        return _numeric(
            _numeric([np.nan if v is None or pd.isna(v) else v for v in values]), dtype=np.float32
        ), {"kind": "numeric"}
    if not all(isinstance(v, str) for v in observed):
        raise ValueError("known covariates must contain numeric, boolean or string scalars")
    categories = sorted(set(observed))
    codes = {v: i for i, v in enumerate(categories)}
    encoded = np.array(
        [codes.get(v, np.nan) if v is not None else np.nan for v in values], dtype=np.float32
    )
    return encoded, {"kind": "categorical", "categories": categories, "unknown": None}


def _native_tables(rows, cov_rows):
    import pyarrow as pa

    identity = [("item_id", pa.string()), ("source", pa.string()), ("freq", pa.string())]
    vector = pa.list_(pa.float32())
    matrix = pa.list_(vector)
    return (
        pa.Table.from_pylist(rows, schema=pa.schema([*identity, ("target", matrix)])),
        pa.Table.from_pylist(
            cov_rows, schema=pa.schema([*identity, ("target", vector), ("covariates", matrix)])
        ),
    )


def build_manifest(
    task,
    *,
    task_yaml: str | Path,
    output: str | Path,
    phase: str = "development",
    max_context: int | None = None,
) -> dict[str, Any]:
    """Write one immutable bundle; return its manifest (also saved as manifest.json).

    ``task`` implements the fev 0.10 Task interface. ``task_yaml`` identifies the
    exact source YAML; ``build_task_manifest`` additionally verifies selection.
    Context overrides affect model input only, not fev's scoring history.
    Output must not exist, and is published only after validation and hashing.
    """
    import fev.metrics
    import fev.task
    import pyarrow as pa
    import pyarrow.parquet as pq

    output, task_yaml = Path(output), Path(task_yaml)
    if output.exists():
        raise FileExistsError(output)
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if max_context is not None and (
        isinstance(max_context, bool) or not isinstance(max_context, int) or max_context < 1
    ):
        raise ValueError("max_context must be a positive integer")
    if isinstance(task.horizon, bool) or not isinstance(task.horizon, int) or task.horizon < 1:
        raise ValueError("task horizon must be a positive integer")
    yaml_hash = _sha(task_yaml.read_bytes())
    dataset = task.load_full_dataset()
    targets, known = list(task.target_columns), list(task.known_dynamic_columns)
    items = _resolve_items(task, dataset, targets, known)
    candidates = [it for it in items if it.reason is None]
    shared = min((it.times[it.boundary] for it in candidates), default=None)
    for item in candidates:
        item.train_end = min(item.boundary, int(item.times.searchsorted(shared, side="left")))
        if item.train_end < max(1, task.min_context_length):
            item.reason = "insufficient_shared_training_history"
    eligible = [it for it in items if it.reason is None]
    scoring = {
        "eval_metric": task.eval_metric,
        "extra_metrics": list(task.extra_metrics),
        "quantile_levels": list(task.quantile_levels),
        "seasonality": task.seasonality,
        "target_columns": targets,
        "aggregation": "fev default item/time reduction, then mean over target channels",
    }
    inners = [
        {"schema_version": SCHEMA_VERSION, "origin_number": i, "scoring": scoring, "items": []}
        for i in range(2)
    ]
    main_rows, cov_rows, groups, encodings = [], [], [], []
    observed_targets = 0
    for item in eligible:
        end = item.train_end
        raw_train = _numeric([item.row[c][:end] for c in targets])
        item.training_input_sha256 = _sha(
            _json_bytes(
                {"target": raw_train, "known_covariates": {c: item.row[c][:end] for c in known}}
            )
        )
        training = _scorer_slice(dataset, item.row, [*targets, *known], 0, end)
        # The runner casts targets to float64 before Simons casts to float32.
        train = _numeric(_numeric([training[c] for c in targets]), dtype=np.float32)
        observed_targets += int(np.isfinite(train).sum())
        covariates, encoding = [], {}
        for column in known:
            array, spec = _encode_training(training[column])
            covariates.append(array.tolist())
            encoding[column] = spec
        encodings.append({"item_id": item.item_id, "columns": encoding})
        for start in range(0, len(targets), MAX_VARIATES):
            indices = list(range(start, min(start + MAX_VARIATES, len(targets))))
            # JSON tuples avoid collisions between raw IDs and block suffixes.
            native_id = json.dumps([item.item_id, start // MAX_VARIATES], separators=(",", ":"))
            main_rows.append(
                {
                    "item_id": native_id,
                    "source": task.task_name,
                    "freq": task.freq,
                    "target": train[indices].tolist(),
                }
            )
            groups.append(
                {
                    "item_id": item.item_id,
                    "native_item_id": native_id,
                    "target_indices": indices,
                    "target_columns": [targets[j] for j in indices],
                }
            )
        if known:
            for j, column in enumerate(targets):
                cov_rows.append(
                    {
                        "item_id": json.dumps([item.item_id, j], separators=(",", ":")),
                        "source": task.task_name,
                        "freq": task.freq,
                        "target": train[j].tolist(),
                        "covariates": covariates,
                    }
                )
        for i, data in enumerate(inners):
            origin = item.boundary + i * task.horizon
            score_start = max(0, origin - task.max_context_length) if task.max_context_length else 0
            model_start = max(score_start, origin - max_context) if max_context else score_start
            # Slice BEFORE converting: forbidden outer values never affect even
            # dtype/category inference or finite-value validation.
            score_past = _numeric([item.row[c][score_start:origin] for c in targets])
            score_target = _numeric([item.row[c][origin : origin + task.horizon] for c in targets])
            # fev formats capped history and future separately; the runner's
            # optional cap comes afterwards. Mirror that order before joining.
            past = _scorer_slice(dataset, item.row, [*targets, *known], score_start, origin)
            future = _scorer_slice(dataset, item.row, known, origin, origin + task.horizon)
            offset = model_start - score_start
            model_target = _numeric(_numeric([past[c][offset:] for c in targets]), dtype=np.float32)
            model_known = {
                c: _jsonable(
                    np.concatenate(
                        [
                            np.asarray(np.asarray(past[c])[offset:], dtype=object),
                            np.asarray(np.asarray(future[c]), dtype=object),
                        ]
                    )
                )
                for c in known
            }
            data["items"].append(
                {
                    "item_id": item.item_id,
                    "origin": item.origin(origin),
                    "history_start_index": model_start,
                    "score_history_start_index": score_start,
                    "history_timestamps": [t.isoformat() for t in item.times[model_start:origin]],
                    "horizon_timestamps": [
                        t.isoformat() for t in item.times[origin : origin + task.horizon]
                    ],
                    "model_input": [
                        {
                            "target_indices": list(range(j, min(j + MAX_VARIATES, len(targets)))),
                            "target": model_target[j : j + MAX_VARIATES].tolist(),
                            "known_covariates": model_known or None,
                        }
                        for j in range(0, len(targets), MAX_VARIATES)
                    ],
                    "score_past": score_past.tolist(),
                    "score_target": score_target.tolist(),
                }
            )
    main_table, cov_table = _native_tables(main_rows, cov_rows)
    adapter_dir = Path(__file__).resolve().parents[1] / "yardsticks"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "status": "ready" if eligible else "frozen_fallback",
        "task": {
            "name": task.task_name,
            "yaml_sha256": yaml_hash,
            "definition": task.to_dict(),
            "freq": task.freq,
            "horizon": task.horizon,
            "target_columns": targets,
        },
        "versions": {
            "fev": version("fev"),
            "fev_source_reference": FEV_SOURCE,
            "fev_task_sha256": _sha(Path(fev.task.__file__).read_bytes()),
            "fev_metrics_sha256": _sha(Path(fev.metrics.__file__).read_bytes()),
            "simons_source": SIMONS_SOURCE,
            "adapter": "SimonsAdapter",
            "adapter_sha256": _sha((adapter_dir / "adapters.py").read_bytes()),
            "runner_sha256": _sha((adapter_dir / "runner.py").read_bytes()),
            "builder_sha256": _sha(Path(__file__).read_bytes()),
            "pyarrow": pa.__version__,
        },
        "cutoffs": {
            "fev_initial_cutoff": task.cutoffs[0],
            "convention": "origins are first forecast observations; all end indices exclusive",
            "inner_offsets": [-2 * task.horizon, -task.horizon],
            "shared_training_boundary": shared.isoformat() if shared is not None else None,
        },
        "scoring": scoring,
        "model_input": {
            "dtype": "float32",
            "max_variates": MAX_VARIATES,
            "max_context": max_context,
            "fev_max_context_length": task.max_context_length,
            "covariates": "all",
            "past_covariates": None,
            "shared_categories": False,
        },
        "items": [
            {
                "item_id": it.item_id,
                "outer_origin": it.origin(it.outer),
                "inner_origins": [it.origin(it.boundary + i * task.horizon) for i in range(2)],
                "training_boundary": it.origin(it.boundary),
                "training_end_index": it.train_end,
                "training_end_timestamp": _timestamp(it.times, it.train_end),
                "training_input_sha256": it.training_input_sha256,
                "eligible": it.reason is None,
            }
            for it in items
        ],
        "excluded_items": [
            {"item_id": it.item_id, "reason": it.reason, "fallback": "frozen_model"}
            for it in items
            if it.reason
        ],
        "groups": groups,
        "covariates": {
            "known_future": known,
            "past_only_dropped": list(task.past_dynamic_columns),
            "static_dropped": list(task.static_columns),
            "dropped_value_counts": {
                "past_only": sum(max(0, min(it.outer, len(it.times))) for it in items)
                * len(task.past_dynamic_columns),
                "static": len(items) * len(task.static_columns),
            },
            "training_encodings": encodings,
        },
        "counts": {
            "items": len(items),
            "eligible_items": len(eligible),
            "excluded_items": len(items) - len(eligible),
            "channels": len(items) * len(targets),
            "eligible_channels": len(eligible) * len(targets),
            "training_groups": len(main_rows),
            "training_covariate_rows": len(cov_rows),
            "training_target_values": sum(it.train_end * len(targets) for it in eligible),
            "training_observed_target_values": observed_targets,
            "training_known_covariate_values": sum(it.train_end * len(known) for it in eligible),
            "inner_target_values_per_origin": len(eligible) * len(targets) * task.horizon,
        },
        "files": {},
    }
    # Validate canonical metadata and all inner values before creating output.
    inner_bytes = [_json_bytes(data) for data in inners]
    _json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".adaptation-", dir=output.parent) as temporary:
        root = Path(temporary) / "bundle"
        (root / "univariate").mkdir(parents=True)
        pq.write_table(
            main_table,
            root / "univariate/train-00000.parquet",
            compression="zstd",
            use_dictionary=False,
            version="2.6",
        )
        if cov_rows:
            (root / "covariates").mkdir()
            pq.write_table(
                cov_table,
                root / "covariates/train-00000.parquet",
                compression="zstd",
                use_dictionary=False,
                version="2.6",
            )
        for i, contents in enumerate(inner_bytes):
            (root / f"inner-{i}.json").write_bytes(contents)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                contents = path.read_bytes()
                manifest["files"][path.relative_to(root).as_posix()] = {
                    "sha256": _sha(contents),
                    "bytes": len(contents),
                }
        manifest = _jsonable(manifest)
        manifest["fingerprint"] = _sha(_json_bytes(manifest))
        (root / "manifest.json").write_bytes(_json_bytes(manifest))
        if output.exists():
            raise FileExistsError(output)
        root.rename(output)
    return manifest


def build_task_manifest(
    task_yaml: str | Path,
    task_name: str,
    output: str | Path,
    *,
    phase: str = "development",
    max_context: int | None = None,
) -> dict[str, Any]:
    """Select exactly one native fev task, resolving local data relative to YAML."""
    task_yaml = Path(task_yaml)
    task = load_tasks(task_yaml, [task_name])[0]
    return build_manifest(
        task, task_yaml=task_yaml, output=output, phase=phase, max_context=max_context
    )
