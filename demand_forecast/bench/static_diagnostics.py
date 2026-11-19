"""Stage D measurements on the history side of fev's benchmark windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from zlib import crc32

import numpy as np
import yaml

from demand_forecast.bench.v5_panels import (
    Part,
    WindowPanels,
    _cell,
    _positive_offset,
    _prove_time_axis,
    describe,
    window_panels,
)
from demand_forecast.bench.yardsticks.runner import SeriesInput, WindowInput, read_names
from demand_forecast.data.episode.kinds import is_missing
from demand_forecast.data.episode.panel_stats import PanelStatsProvider
from demand_forecast.data.episode.schema import (
    KIND_CATEGORY,
    KIND_FLAG,
    KIND_NUMBER,
    MAX_FORECAST_ROWS,
    Panel,
)
from demand_forecast.data.episode.screening import ScreeningConfig, _static_candidates, screen

if TYPE_CHECKING:
    import fev

SHUFFLES = 20
SEED = 0
MAX_CUTOFFS = 3
MIN_WINDOW_ITEMS = 8
MIN_PANEL_ITEMS = 8
MIN_POOLED_SCORES = 200
PERCENTILE = 95
ROUNDING_DIGITS = 9
LEVEL_DIGITS = 2
MAX_LEVEL = 0.99
SUPPORT_SIZES = (2, 4, 8, 16, 32)
SCREENING = ScreeningConfig()
KIND_NAMES = {KIND_CATEGORY: "category", KIND_FLAG: "flag", KIND_NUMBER: "number"}

SUPPORT_COLUMNS = (
    "kind",
    "missing_share",
    "labels",
    "label_size_min",
    "label_size_median",
    "label_size_max",
    *(f"share_at_least_{size}" for size in SUPPORT_SIZES),
    "constant",
    "identifier",
    "unshared",
    "determined_by",
    "copied_from",
)
ATTRIBUTE_COLUMNS = (
    "task",
    "window",
    "cutoff",
    "attribute",
    "window_items",
    "panel_items",
    *SUPPORT_COLUMNS,
    "screen_kind",
    "removed",
    "score",
    "shuffled_scores",
)
TASK_COLUMNS = (
    "task",
    "window",
    "items",
    "targets_per_item",
    "history_min",
    "history_median",
    "history_max",
    "short_items",
    "on_grid",
    "off_grid",
    "items_alone",
    "items_without_time_axis",
    "grid_reasons",
    "distinct_origins",
    "over_origin_budget",
    "new_items",
)


def _column_types(window: fev.EvaluationWindow) -> tuple[dict, dict]:
    from datasets import ClassLabel

    declared, labels = {}, {}
    # fev's split preserves the codes but drops ClassLabel metadata.
    for name in window.known_dynamic_columns + window.past_dynamic_columns + window.static_columns:
        feature = window.full_dataset.features[name]
        while hasattr(feature, "feature"):
            feature = feature.feature
        dtype = getattr(feature, "dtype", "")
        if isinstance(feature, ClassLabel):
            declared[name], labels[name] = "class_label", tuple(feature.names)
        elif dtype == "bool":
            declared[name] = "boolean"
        elif dtype.startswith(("int", "uint", "float", "decimal")):
            declared[name] = "number"
        elif dtype.startswith(("string", "large_string")):
            declared[name] = "string"
    return declared, labels


def _history_panels(task: fev.Task) -> Iterator[tuple[int, WindowPanels, dict[int, dict]]]:
    freq = None
    for index, window in enumerate(task.iter_windows()):
        history, future = window.get_input_data()
        if index == 0:
            try:
                freq = task.freq
            except ValueError:
                freq = None
        parsed_frequency = _positive_offset(freq)
        declared, class_labels = _column_types(window)
        ids, items = [], []
        targets = list(window.target_columns)
        columns = window.past_dynamic_columns + window.known_dynamic_columns
        # fev filters both splits together and returns the rows in the same order.
        future_timestamps = future[task.timestamp_column]
        for position, (row, horizon_timestamps) in enumerate(
            zip(history, future_timestamps, strict=True)
        ):
            ids.append(str(row[window.id_column]))
            items.append(
                SeriesInput(
                    position,
                    tuple(range(len(targets))),
                    np.stack([np.asarray(row[name], dtype=np.float64) for name in targets]),
                    {name: np.asarray(row[name]) for name in columns},
                    {name: np.array([None], dtype=object) for name in window.known_dynamic_columns},
                    axis=_prove_time_axis(
                        np.asarray(row[task.timestamp_column]),
                        parsed_frequency,
                        future=np.asarray(horizon_timestamps),
                        horizon=task.horizon,
                    ),
                    static={name: row[name] for name in window.static_columns},
                )
            )
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate fev item ids")
        panels = window_panels(
            WindowInput(ids, targets, items, freq, declared, class_labels, parsed_frequency),
            name=task.task_name,
            horizon=1,
            known=window.known_dynamic_columns,
            past=window.past_dynamic_columns,
        )
        # B1 strips statics from fallback panels. Keep normalized values for
        # descriptive statistics on the full window, never for relevance.
        fallbacks = {
            position: {
                name: _cell(value, declared.get(name), class_labels.get(name, ()), name)
                for name, value in items[position].static.items()
            }
            for position in panels.reasons
        }
        yield index, panels, fallbacks


def past_panels(task: fev.Task) -> Iterator[tuple[int, WindowPanels]]:
    """Build B1 panels with a synthetic, missing horizon of one step."""
    for index, panels, _ in _history_panels(task):
        yield index, panels


def _prefix_panel(part: Part, *, statics: bool) -> Panel:
    items = []
    for item, origin in zip(part.panel.items, part.origins, strict=True):
        end = origin - item.start
        items.append(
            replace(
                item,
                targets={
                    name: item.targets[name][:end]
                    for name in sorted(item.targets)[:MAX_FORECAST_ROWS]
                },
                dynamic={} if statics else {name: raw[:end] for name, raw in item.dynamic.items()},
                static=item.static if statics else {},
            )
        )
    return replace(part.panel, items=tuple(items), roles={} if statics else part.panel.roles)


def _screen_statics(panel: Panel, cutoff: int, only: str | None = None) -> dict:
    targets = tuple(sorted(panel.items[0].targets))
    provider = PanelStatsProvider(panel)
    stats = provider.get(targets[0], cutoff)
    # Static score and removal are panel-wide. The index only controls the size
    # of the emitted token expansion, and missing categories emit no candidate.
    indices = defaultdict(set)
    candidates = {}
    attributes = (only,) if only is not None else stats.static_kinds
    for name in attributes:
        if stats.static_kinds[name] == "number":
            indices[0].add(name)
        else:
            first = next((i for i in range(len(panel.items)) if stats.labels(name, i)), None)
            if first is None:
                # No item can emit this candidate. Read the same fixed-rule
                # trace that screen uses, including duplicate empty columns.
                candidates[name] = next(
                    c for c in _static_candidates(stats, SCREENING.min_items) if c.name == name
                )
            else:
                indices[first].add(name)
    for index, names in indices.items():
        candidates.update(
            (c.name, c)
            for c in screen(panel, stats, index, targets, cutoff, SCREENING, provider=provider)
            if c.source == "static" and c.name in names
        )
    return candidates


def attribute_scores(
    part: Part, *, shuffles: int = SHUFFLES, seed: int = SEED, name: str, window: int = 0
) -> list[dict]:
    """Return fixed-rule traces and panel-wide relevance at the selected dated origins."""
    if part.alone or part.panel.epoch is None or not any(item.static for item in part.panel.items):
        return []
    panel = _prefix_panel(part, statics=True)
    counts = Counter(part.origins)
    cutoffs = sorted(counts, key=lambda origin: (counts[origin], origin), reverse=True)[
        :MAX_CUTOFFS
    ]
    rows = {}
    for cutoff in sorted(cutoffs):
        for attribute, candidate in sorted(_screen_statics(panel, cutoff).items()):
            rows[cutoff, attribute] = {
                "task": name,
                "window": window,
                "cutoff": cutoff,
                "attribute": attribute,
                "window_items": len(panel.items),
                "panel_items": len(panel.items),
                "screen_kind": candidate.kind,
                "removed": candidate.removed,
                "score": candidate.score,
                "shuffled_scores": [],
            }
    for attribute in sorted({attr for _, attr in rows}):
        kept = [
            row for (_, attr), row in rows.items() if attr == attribute and row["removed"] is None
        ]
        if not kept:
            continue
        cells = [item.static.get(attribute) for item in panel.items]
        entropy = [seed, crc32(name.encode("utf-8")), window, crc32(attribute.encode("utf-8"))]
        for k in range(shuffles):
            permutation = np.random.default_rng([*entropy, k]).permutation(len(cells))
            items = tuple(
                replace(item, static={**item.static, attribute: cells[j]})
                for item, j in zip(panel.items, permutation, strict=True)
            )
            shuffled = replace(panel, items=items)
            for row in kept:
                candidate = _screen_statics(shuffled, row["cutoff"], only=attribute)[attribute]
                row["shuffled_scores"].append(candidate.score)
    return list(rows.values())


def _distribution(values) -> tuple[int | None, float | None, int | None]:
    return (min(values), float(np.median(values)), max(values)) if values else (None, None, None)


def _determines(source: tuple, target: tuple) -> bool:
    mapping = {}
    for left, right in zip(source, target, strict=True):
        if left is not None and right is not None and mapping.setdefault(left, right) != right:
            return False
    return bool(mapping)


def _static_profile(name: str, items: list) -> tuple[dict, list]:
    # Label counts use the builder's category normalization and numeric bins.
    # No target histories are needed for either operation.
    metadata = Panel(
        name,
        tuple(replace(item, start=0, targets={}, dynamic={}) for item in items),
        {},
        None,
        None,
        False,
    )
    stats = PanelStatsProvider(metadata).get("", 0)
    profiles, signatures = {}, {}
    for attribute, kind in stats.static_kinds.items():
        labels = [stats.labels(attribute, i) for i in range(len(items))]
        if kind == "number":
            signatures[attribute] = tuple(
                float(item.static[attribute]) if cell else None
                for item, cell in zip(items, labels, strict=True)
            )
        else:
            signatures[attribute] = tuple(cell or None for cell in labels)
        observed = set(signatures[attribute]) - {None}
        counts = Counter(label for cell in labels for label in cell)
        low, median, high = _distribution(list(counts.values()))
        support = [max((counts[label] for label in cell), default=0) for cell in labels]
        profiles[attribute] = {
            "kind": kind,
            "missing_share": labels.count(()) / len(items),
            "labels": len(counts),
            "label_size_min": low,
            "label_size_median": median,
            "label_size_max": high,
            **{
                f"share_at_least_{size}": sum(v >= size for v in support) / len(items)
                for size in SUPPORT_SIZES
            },
            "constant": len(observed) == 1,
            "identifier": kind != "number"
            and all(len(cell) == 1 for cell in labels)
            and len(counts) == len(items),
            "unshared": kind != "number" and (high is None or high < 2),
        }
    for attribute, profile in profiles.items():
        cells = signatures[attribute]
        profile["determined_by"] = [
            other
            for other, source in signatures.items()
            if other != attribute and _determines(source, cells)
        ]
        profile["copied_from"] = [
            other for other, source in signatures.items() if other != attribute and source == cells
        ]
    groups = _equal_groups(items, zip(*signatures.values(), strict=True)) if signatures else []
    return profiles, groups


def _equal_groups(items, signatures) -> list[list[str]]:
    groups = defaultdict(list)
    for item, signature in zip(items, signatures, strict=True):
        groups[signature].append(item.item_id)
    return sorted(sorted(group) for group in groups.values() if len(group) > 1)


def _target_signature(item, length: int) -> tuple:
    result = []
    for name, values in sorted(item.targets.items()):
        history = values[:length].copy()
        # Canonical NaNs and signed zeroes have the same value for this comparison.
        history[np.isnan(history)] = np.nan
        history[history == 0] = 0
        result.append((name, hashlib.sha256(history.tobytes()).digest()))
    return tuple(result)


def _dynamic_profile(parts: tuple[Part, ...]) -> dict:
    columns, missing, total = {}, Counter(), Counter()
    for part in parts:
        if not part.panel.roles:
            continue
        panel = _prefix_panel(part, statics=False)
        targets = tuple(sorted(panel.items[0].targets))
        provider = PanelStatsProvider(panel)
        for i, (item, cutoff) in enumerate(zip(panel.items, part.origins, strict=True)):
            stats = provider.get(targets[0], cutoff)
            for candidate in screen(panel, stats, i, targets, cutoff, SCREENING, provider=provider):
                name = candidate.name
                history = item.dynamic[name]
                missing[name] += sum(is_missing(cell) for cell in history)
                total[name] += len(history)
                row = columns.setdefault(
                    name, {"role": candidate.source, "kinds": set(), "score": 0.0}
                )
                row["kinds"].add(KIND_NAMES[candidate.kind])
                row["score"] = max(row["score"], candidate.score)
    for name, row in columns.items():
        row["kinds"] = sorted(row["kinds"])
        row["missing_share"] = missing[name] / total[name] if total[name] else None
    return columns


def diagnose_task(task: fev.Task) -> dict:
    """Measure each benchmark window and shuffle the endpoint windows."""
    windows, seen = [], set()
    for index, panels, fallbacks in _history_panels(task):
        entries = sorted(
            (position, origin - item.start, item)
            for part in panels.parts
            for position, origin, item in zip(
                part.positions, part.origins, part.panel.items, strict=True
            )
        )
        lengths = [length for _, length, _ in entries]
        items = [
            replace(item, static=fallbacks.get(position, item.static))
            for position, _, item in entries
        ]
        ids = {item.item_id for item in items}
        counts = describe([panels])
        low, median, high = _distribution(lengths)
        attributes, equal_statics = _static_profile(task.task_name, items)
        shuffles = SHUFFLES if index in (0, task.num_windows - 1) else 0
        scores = [
            row | {"window_items": len(items)}
            for part in panels.parts
            for row in attribute_scores(part, shuffles=shuffles, name=task.task_name, window=index)
        ]
        windows.append(
            {
                "task": task.task_name,
                "window": index,
                "items": len(items),
                "targets_per_item": len(items[0].targets) if items else 0,
                "history_min": low,
                "history_median": median,
                "history_max": high,
                "short_items": counts["v5_short_items"],
                "on_grid": len(items) - counts["v5_items_off_grid"],
                "off_grid": counts["v5_items_off_grid"],
                "items_alone": counts["v5_items_alone"],
                "items_without_time_axis": counts["v5_items_without_time_axis"],
                "grid_reasons": dict(Counter(panels.reasons.values())),
                "distinct_origins": panels.distinct_origins,
                "over_origin_budget": bool(counts["v5_windows_over_origin_budget"]),
                "new_items": len(ids - seen) if index else 0,
                "attributes": attributes,
                "scores": scores,
                "dynamic": _dynamic_profile(panels.parts),
                "equal_static_groups": equal_statics,
                "equal_target_groups": _equal_groups(
                    items, (_target_signature(item, length) for _, length, item in entries)
                ),
            }
        )
        seen.update(ids)
    return {"task": task.task_name, "windows": windows}


def _eligible_rows(rows: Iterable[Mapping]) -> Iterator[Mapping]:
    for row in rows:
        if (
            int(row["window_items"]) < MIN_WINDOW_ITEMS
            or int(row["panel_items"]) < MIN_PANEL_ITEMS
            or row["removed"] not in (None, "")
        ):
            continue
        scores = row["shuffled_scores"]
        if isinstance(scores, str):
            scores = json.loads(scores)
        if len(scores):
            yield {**row, "shuffled_scores": scores}


def _level(pool: list[float]) -> float:
    if len(pool) < MIN_POOLED_SCORES:
        return 0.0
    percentile = round(float(np.percentile(pool, PERCENTILE)), ROUNDING_DIGITS)
    unit = Decimal(1).scaleb(-LEVEL_DIGITS)
    ceiling = Decimal(str(percentile)).quantize(unit, rounding=ROUND_CEILING)
    return min(MAX_LEVEL, float(ceiling))


def no_evidence_level(rows: Iterable[Mapping]) -> float:
    """Use the stage D rule on attribute records or rows from csv.DictReader."""
    return _level([score for row in _eligible_rows(rows) for score in row["shuffled_scores"]])


def _level_report(rows: list[dict]) -> dict:
    eligible = list(_eligible_rows(rows))
    level = no_evidence_level(eligible)
    return {
        "screen_no_evidence": level,
        "pooled_scores": sum(len(row["shuffled_scores"]) for row in eligible),
        "tasks": len({row["task"] for row in eligible}),
        "windows": len({(row["task"], row["window"]) for row in eligible}),
        "attributes": len({(row["task"], row["attribute"]) for row in eligible}),
        "share_real_above_level": sum(row["score"] > level for row in eligible) / len(eligible)
        if eligible
        else None,
        "rule": {
            "percentile": PERCENTILE,
            "method": "linear",
            "pre_round_digits": ROUNDING_DIGITS,
            "ceil_digits": LEVEL_DIGITS,
            "cap": MAX_LEVEL,
            "min_window_items": MIN_WINDOW_ITEMS,
            "min_panel_items": MIN_PANEL_ITEMS,
            "min_pooled_scores": MIN_POOLED_SCORES,
            "insufficient_pool_level": 0.0,
            "windows": "first_and_last",
            "kept_only": True,
            "max_cutoffs": MAX_CUTOFFS,
            "max_targets": MAX_FORECAST_ROWS,
            "shuffles_per_attribute_cutoff": SHUFFLES,
            "seed": SEED,
        },
    }


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)


def _csv_cell(value):
    if isinstance(value, (bool, dict, list)):
        return _json(value)
    return repr(value) if isinstance(value, float) else value


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    def write(stream):
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows([_csv_cell(row[field]) for field in fields] for row in rows)

    _atomic(path, write)


def _safe_name(name: str) -> None:
    """Validate a task name for both local output and guest publication."""
    if (
        not isinstance(name, str)
        or not name
        or any(c in name for c in "\\*?[]#\r\n")
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError(f"unsafe diagnostic task path: {name!r}")
    filename = name + ".json"
    if filename.split("/", 1)[0] in {
        "attributes.csv",
        "tasks.csv",
        "level.json",
        "tasks.yaml",
        "names.txt",
        "inputs.json",
        "completed.json",
    }:
        raise ValueError(f"reserved diagnostic report path: {filename!r}")


def write_run(tasks: Iterable[fev.Task], out: Path) -> None:
    """Run tasks in turn and write deterministic per-task and aggregate reports."""
    windows, attributes = [], []
    for task in tasks:
        _safe_name(task.task_name)
        report_path = out / f"{task.task_name}.json"
        report = diagnose_task(task)
        _atomic(report_path, lambda stream: stream.write(_json(report) + "\n"))
        for window in report["windows"]:
            windows.append({field: window[field] for field in TASK_COLUMNS})
            attributes.extend(
                window["attributes"][row["attribute"]] | row for row in window["scores"]
            )
    windows.sort(key=lambda row: (row["task"], row["window"]))
    attributes.sort(key=lambda row: (row["task"], row["window"], row["cutoff"], row["attribute"]))
    _write_csv(out / "tasks.csv", TASK_COLUMNS, windows)
    _write_csv(out / "attributes.csv", ATTRIBUTE_COLUMNS, attributes)
    _atomic(
        out / "level.json", lambda stream: stream.write(_json(_level_report(attributes)) + "\n")
    )


def _load_tasks(path: Path, names: list[str] | None) -> Iterator[fev.Task]:
    import fev

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise ValueError("task YAML must contain a nonempty tasks list")
    selected = {}
    for raw in data["tasks"]:
        values = dict(raw)
        alias = values.pop("name", None)
        explicit_name = values.get("task_name")
        if alias is not None:
            if explicit_name not in (None, alias):
                raise ValueError("conflicting name and task_name")
            explicit_name = alias
        if names is not None and explicit_name is not None and explicit_name not in names:
            continue
        source = values["dataset_path"]
        dataset = Path(source).expanduser()
        if not urlparse(source).scheme and (
            dataset.suffix in (".parquet", ".arrow") or source.startswith(("./", "../", "~", "/"))
        ):
            values["dataset_path"] = str((path.parent / dataset).resolve())
            values["dataset_config"] = None
        name = explicit_name if explicit_name is not None else values.get("dataset_config")
        if name is None:
            name = "/".join(Path(values["dataset_path"]).parts[-3:-1])
        if names is not None and name not in names:
            continue
        _safe_name(name)
        if name in selected:
            raise ValueError(f"duplicate task name: {name}")
        selected[name] = dict(values, task_name=name)
    if names is not None and set(names) - selected.keys():
        raise ValueError("names must match tasks in the YAML")
    for name in sorted(selected):
        yield fev.Task(**selected[name])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--names", metavar="@FILE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.names is not None and not args.names.startswith("@"):
        parser.error("--names requires @FILE")
    write_run(_load_tasks(args.tasks, read_names(args.names)), args.out)


if __name__ == "__main__":
    main()
