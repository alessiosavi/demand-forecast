"""Score a parity-checked v5 export on benchmark episode panels."""

from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from demand_forecast.bench.v5_panels import describe, window_panels
from demand_forecast.data.episode import ScreeningConfig
from demand_forecast.data.episode.schema import MAX_CONTEXT, PATCH, SUMMARY_FEATURES
from demand_forecast.models.v5 import Evidence, forecast_items, planned_layouts, usable_workers
from demand_forecast.models.v5.export import load_export, require_parity

from .runner import Options, WindowInput


@dataclass(frozen=True)
class Variant:
    without: frozenset[str] = frozenset()
    drop: frozenset[str] = frozenset()
    per_target: bool = False
    summary_off: tuple[str, ...] = ()
    gates: str = "learned"
    router: str = "learned"
    pruned: bool = True
    derive: frozenset[str] | None = None
    zero_head_gain: bool = False


VARIANTS: dict[str, Variant] = {
    "full": Variant(),
    **{
        f"no_{column}": Variant(without=frozenset({column}))
        for column in (
            "features",
            "known",
            "past",
            "dynamic_categories",
            "statics",
            "static_categories",
            "static_numbers",
        )
    },
    "no_peers": Variant(drop=frozenset({"peers"})),
    "no_panel_row": Variant(drop=frozenset({"panel_row"})),
    "no_typed_context": Variant(
        drop=frozenset(
            {"attributes", "static_numbers", "peers", "panel_row", "value_stats", "value_effects"}
        ),
    ),
    "no_value_stats": Variant(drop=frozenset({"value_stats"})),
    "no_derived": Variant(derive=frozenset()),
    "no_calendar": Variant(drop=frozenset({"calendar"})),
    "no_rolling": Variant(drop=frozenset({"rolling"})),
    "no_lags": Variant(drop=frozenset({"lags"})),
    "no_zeros": Variant(drop=frozenset({"zeros"})),
    "no_step_families": Variant(drop=frozenset({"rolling", "lags", "zeros"})),
    "no_summary": Variant(summary_off=SUMMARY_FEATURES),
    "per_target": Variant(per_target=True),
    "no_siblings": Variant(per_target=True, drop=frozenset({"siblings"})),
    "gates_one": Variant(gates="ones", pruned=False),
    "screening_alone": Variant(gates="evidence"),
    "router_uniform": Variant(router="uniform"),
    "no_head_gain": Variant(zero_head_gain=True),
}


def _worker_count(device: str) -> int:
    return (
        16
        if device == "cuda" and sys.platform == "linux" and len(os.sched_getaffinity(0)) >= 18
        else 0
    )


def _peak_rss_bytes(who: int) -> int:
    return resource.getrusage(who).ru_maxrss * (1024 if sys.platform == "linux" else 1)


def _levels(values: np.ndarray, native: tuple[float, ...], levels: list[float]) -> np.ndarray:
    if all(level in native for level in levels):
        return values[..., [native.index(level) for level in levels]]
    return np.stack(
        [
            values[..., native.index(level)]
            if level in native
            else np.apply_along_axis(lambda row: np.interp(level, native, row), -1, values)
            for level in levels
        ],
        axis=-1,
    )


def _window_evidence(parts: list[tuple[tuple[int, ...], Evidence]]) -> Evidence:
    """Restore runner positions and share a candidate and rule axis across parts."""
    candidates = sorted({name for _, part in parts for name in part.candidates})
    rules = tuple(sorted({name for _, part in parts for name in part.rules}))
    candidate_index = {name: i for i, name in enumerate(candidates)}
    rule_index = {name: i for i, name in enumerate(rules)}
    shape = (sum(len(part.requests) for _, part in parts), len(candidates))
    status = np.zeros(shape, dtype=np.int8)
    rule = np.full(shape, -1, dtype=np.int8)
    gate = np.full(shape, np.nan, dtype=np.float32)
    score = np.full(shape, np.nan, dtype=np.float32)
    requests = []
    for positions, part in parts:
        start = len(requests)
        requests.extend((positions[i], target) for i, target in part.requests)
        rows = slice(start, len(requests))
        columns = [candidate_index[name] for name in part.candidates]
        # Put the sentinel last so NumPy's index -1 retains the meaning "no rule".
        rule_map = np.asarray([*(rule_index[name] for name in part.rules), -1], dtype=np.int8)
        status[rows, columns] = part.status
        rule[rows, columns] = rule_map[part.rule]
        gate[rows, columns] = part.gate
        score[rows, columns] = part.score
    return Evidence(requests, candidates, status, rules, rule, gate, score)


class V5Adapter:
    """Reuse a verified model across tasks, building one window's panels at a time."""

    def __init__(self, options: Options, task):
        self.options = options.resolved()
        export = Path(self.options.weights)
        model, metadata = load_export(export, device="cpu")
        require_parity(export, device="cpu")
        self.model = model.to(self.options.device)
        require_parity(export, device=self.options.device)

        sampler = metadata["training"]["sampler"]
        kind = sampler.get("scale_kind")
        scale_options = {} if kind is None else {"scale_kind": kind}
        self.context = (
            max(sampler["contexts"])
            if self.options.max_context is None
            else min(MAX_CONTEXT, self.options.max_context // PATCH * PATCH)
        )
        self.screening = ScreeningConfig(
            no_evidence=sampler.get("screen_no_evidence", 0.0),
            fair_static_spread=sampler.get("screen_fair_spread", False),
            scale_floor=sampler.get("scale_floor"),
            **scale_options,
        )
        self.variant = VARIANTS[self.options.variant or "full"]
        self._training_drop = frozenset(sampler["drop_always"])
        self._drop = self._training_drop | self.variant.drop
        self._training_derive = frozenset(sampler.get("derive", ()))
        if self.variant.zero_head_gain:
            if not (self.model.config.mix_evidence and "naive_evidence" in self._training_derive):
                raise ValueError(
                    "no_head_gain requires model.mix_evidence and sampler naive_evidence"
                )
            with torch.no_grad():
                self.model.head_gain.zero_()
        self._derive = self._training_derive if self.variant.derive is None else self.variant.derive
        self._summary_off = tuple(
            dict.fromkeys((*sampler["summary_off"], *self.variant.summary_off))
        )
        count = _worker_count(self.options.device)
        if "naive_evidence" in self._derive:
            self._workers, self._worker_reason = usable_workers(
                count, naive_anchor=True, naive_evidence=True
            )
        elif "naive_anchor" in self._derive:
            self._workers, self._worker_reason = usable_workers(count, naive_anchor=True)
        else:
            self._workers, self._worker_reason = usable_workers(count)
        self.prepare(task)

    def prepare(self, task) -> None:
        self._name = task.task_name
        self._known = task.known_dynamic_columns
        self._past = task.past_dynamic_columns
        self._counts = {**describe(()), "v5_fallback_rows": 0}
        self._evidence = []

    def _forecast_window(self, window: WindowInput, horizon: int, levels: list[float]):
        panel_options = {
            "name": self._name,
            "horizon": horizon,
            "known": self._known,
            "past": self._past,
        }
        full = window_panels(window, **panel_options)
        layouts = {
            part.positions: (
                ["per_target"] * len(part.positions)
                if self.variant.per_target
                else planned_layouts(
                    part.panel,
                    list(enumerate(part.origins)),
                    horizon,
                    context=self.context,
                    screening=self.screening,
                    drop=self._training_drop,
                    derive=self._training_derive,
                )
            )
            for part in full.parts
        }
        if self.variant.without:
            del full
            panels = window_panels(window, **panel_options, without=self.variant.without)
        else:
            panels = full

        outputs, evidence, fallback = {}, [], 0
        for part in panels.parts:
            forecasts = forecast_items(
                self.model,
                part.panel,
                list(enumerate(part.origins)),
                horizon,
                context=self.context,
                screening=self.screening,
                layouts=layouts[part.positions],
                drop=self._drop,
                derive=self._derive,
                summary_off=self._summary_off,
                gates=self.variant.gates,
                router=self.variant.router,
                pruned=self.variant.pruned,
                batch_episodes=self.options.batch_size,
                workers=self._workers,
            )
            for i, position in enumerate(part.positions):
                targets = window.items[position].target_indices
                values = np.stack([forecasts.values[i, window.targets[j]] for j in targets])
                outputs[position] = _levels(values, self.model.config.quantiles, levels)
            evidence.append((part.positions, forecasts.evidence))
            fallback += len(forecasts.fallback)

        return (
            [outputs[position] for position in range(len(window.items))],
            _window_evidence(evidence),
            {**describe((panels,)), "v5_fallback_rows": fallback},
        )

    def predict(
        self, windows: list[WindowInput], horizon: int, levels: list[float]
    ) -> list[list[np.ndarray]]:
        outputs, evidence = [], []
        counts = dict.fromkeys(self._counts, 0)
        for window in windows:
            output, details, window_counts = self._forecast_window(window, horizon, levels)
            outputs.append(output)
            evidence.append(details)
            for key, value in window_counts.items():
                if key == "v5_max_distinct_origins":
                    counts[key] = max(counts[key], value)
                else:
                    counts[key] += value
        self._counts, self._evidence = counts, evidence
        return outputs

    def metadata(self) -> dict:
        return {
            **self._counts,
            "v5_workers": self._workers,
            "v5_worker_reason": self._worker_reason,
            "peak_rss_bytes": _peak_rss_bytes(resource.RUSAGE_SELF),
            "peak_rss_children_bytes": _peak_rss_bytes(resource.RUSAGE_CHILDREN),
        }

    def evidence(self) -> list[Evidence]:
        return self._evidence


def _parity_tasks():
    import fev
    import pandas as pd
    from datasets import Dataset

    for name, lengths, targets, statics in (
        ("dated-ragged", [49, 51, 54, 49, 51, 54, 49, 51], ["y"], True),
        ("split-two", [50] * 8, ["a", "b"], True),
        ("origin-budget", list(range(36, 53)), ["y"], False),
    ):
        columns = {
            "id": [f"sku-{i:02}" for i in range(len(lengths))],
            "timestamp": [pd.date_range("2024-01-01", periods=n, freq="D") for n in lengths],
        }
        for j, target in enumerate(targets):
            columns[target] = [
                (i + 1) * np.sin(np.arange(n) / (2.3 + j / 3)) + 10 * j
                for i, n in enumerate(lengths)
            ]
        columns["planned"] = [
            [None if step % 11 == 0 else step % 7 for step in range(n)] for n in lengths
        ]
        if statics:
            columns["group"] = ["east" if i % 2 else "west" for i in range(len(lengths))]
            columns["size"] = [0.5 + i // 2 for i in range(len(lengths))]
        task = fev.Task(
            dataset_path="in-memory",
            task_name=name,
            target=targets if len(targets) > 1 else targets[0],
            known_dynamic_columns=["planned"],
            static_columns=["group", "size"] if statics else [],
            horizon=3,
            min_context_length=1,
        )
        # Use fev's real window splitter without a dataset download.
        task._full_dataset = Dataset.from_dict(columns).with_format("numpy")
        task._freq = "D"
        yield task


def _store_parity_window(arrays, name, window):
    items = []
    for i, item in enumerate(window.items):
        prefix = f"{name}/item_{i}"
        arrays[f"{prefix}/target"] = item.target
        for role, values in (("past", item.past), ("future", item.future)):
            for column, value in values.items():
                arrays[f"{prefix}/{role}/{column}"] = value
        axis = item.axis
        items.append(
            {
                "item_index": item.item_index,
                "target_indices": item.target_indices,
                "past": list(item.past),
                "future": list(item.future),
                "static": item.static,
                "axis": {
                    "first": axis.first.isoformat() if axis.first is not None else None,
                    "on_grid": axis.on_grid,
                    "reason": axis.reason,
                    "missing_timestamps": axis.missing_timestamps,
                }
                if axis is not None
                else None,
            }
        )
    return {
        "ids": window.ids,
        "targets": window.targets,
        "freq": window.freq,
        "declared": window.declared,
        "class_labels": window.class_labels,
        "items": items,
    }


def _read_parity_window(arrays, name, data) -> WindowInput:
    import pandas as pd

    from demand_forecast.bench.v5_panels import TimeAxis

    from .runner import SeriesInput

    items = []
    for i, item in enumerate(data["items"]):
        prefix = f"{name}/item_{i}"
        axis = item["axis"]
        if axis is not None:
            if axis["first"] is not None:
                axis["first"] = pd.Timestamp(axis["first"])
            axis = TimeAxis(**axis)
        items.append(
            SeriesInput(
                item["item_index"],
                tuple(item["target_indices"]),
                arrays[f"{prefix}/target"],
                {column: arrays[f"{prefix}/past/{column}"] for column in item["past"]},
                {column: arrays[f"{prefix}/future/{column}"] for column in item["future"]},
                axis,
                item["static"],
            )
        )
    return WindowInput(
        data["ids"],
        data["targets"],
        items,
        data["freq"],
        data["declared"],
        {column: tuple(labels) for column, labels in (data["class_labels"] or {}).items()},
    )


def _selftest(export: Path, path: Path) -> None:
    import json

    from .runner import assemble, prepare_windows

    options = Options("v5", str(export), "parity", device="cpu", processing="raw")
    levels = [0.05, 0.1, 0.35, 0.5, 0.9, 0.95]
    arrays, cases = {}, []
    adapter = None
    for task in _parity_tasks():
        if adapter is None:
            adapter = V5Adapter(options, task)
        else:
            adapter.prepare(task)
        (window,) = prepare_windows(task, options)
        (outputs,) = adapter.predict([window], task.horizon, levels)
        name = task.task_name
        arrays[f"{name}/forecast"] = assemble(window, outputs, task.horizon, levels)
        cases.append({"task": task.to_dict(), "window": _store_parity_window(arrays, name, window)})

        report = adapter.metadata()
        # Explicit checks: these prove which code path each stored window took, and a bare
        # assert would vanish under ``python -O``.
        took_its_path = {
            "dated-ragged": report["v5_items_alone"] == 0
            and report["v5_max_distinct_origins"] > 1
            and all(item.static for item in window.items),
            "split-two": {t for i, t in adapter.evidence()[0].requests if i == 0} == {"a", "b"},
            "origin-budget": report["v5_windows_over_origin_budget"] == 1
            and report["v5_items_alone"] == 17,
        }[name]
        if not took_its_path:
            raise SystemExit(f"v5 adapter parity case {name!r} did not take its code path")
    if any(array.dtype.hasobject for array in arrays.values()):
        raise TypeError("v5 adapter parity cannot store object arrays")
    np.savez(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(
            {"levels": levels, "cases": cases},
            allow_nan=False,
            default=lambda value: value.item(),  # NumPy scalars in fev static cells.
        ),
        encoding="utf-8",
    )


def _check_selftest(export: Path, path: Path) -> None:
    import json

    import fev

    from .runner import assemble

    options = Options("v5", str(export), "parity", device="cpu", processing="raw")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    levels = metadata["levels"]
    adapter = None
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        for case in metadata["cases"]:
            task = fev.Task(**case["task"])
            name = task.task_name
            if adapter is None:
                adapter = V5Adapter(options, task)
            else:
                adapter.prepare(task)
            window = _read_parity_window(arrays, name, case["window"])
            (outputs,) = adapter.predict([window], task.horizon, levels)
            actual = assemble(window, outputs, task.horizon, levels)
            expected = arrays[f"{name}/forecast"]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4, err_msg=name)
            print(
                f"adapter {name}: ok (max absolute error {np.max(np.abs(actual - expected)):.6g})"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("selftest", "check"))
    parser.add_argument("export", type=Path)
    parser.add_argument("path", type=Path, help="path stem for the .npz and .json parity files")
    args = parser.parse_args()
    command = _selftest if args.verb == "selftest" else _check_selftest
    command(args.export, args.path)


if __name__ == "__main__":
    main()
