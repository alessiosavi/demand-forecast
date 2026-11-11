"""Derive named official or shifted-backtest fev task YAML without loading data."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from pandas.tseries.frequencies import to_offset

from demand_forecast.bench.specs import fev_bench_tasks_path, load_task_specs


def read_names(value: str | None) -> list[str] | None:
    """Read comma-separated names or a file containing one name per line."""
    if value is None:
        return None
    if value.startswith("@"):
        value = Path(value[1:]).read_text(encoding="utf-8")
    names = [part.strip() for line in value.splitlines() for part in line.split(",")]
    names = [name for name in names if name]
    if not names or len(names) != len(set(names)):
        raise ValueError("names must be non-empty and unique")
    return names


def shift_tasks(
    names: list[str],
    out: Path,
    *,
    tasks: str | Path = "fev100",
    mode: str = "shifted",
    windows: dict[str, int] | None = None,
    cutoffs: dict[str, str] | None = None,
    steps: dict[str, str] | None = None,
) -> None:
    """Preserve source definitions, changing only cutoff, window count and step."""
    if mode not in {"shifted", "official"}:
        raise ValueError(f"unknown mode: {mode}")
    specs = {spec.name: spec for spec in load_task_specs(tasks)}
    if not names or len(names) != len(set(names)):
        raise ValueError("names must be non-empty and unique")
    for name in names:
        if name not in specs:
            raise ValueError(f"unknown task: {name}")
    windows = windows or {}
    cutoffs = cutoffs or {}
    steps = steps or {}
    if mode == "official" and windows:
        raise ValueError("--windows cannot modify official task definitions")
    if mode == "official" and (cutoffs or steps):
        raise ValueError("--cutoff and --step cannot modify official task definitions")
    for name, count in windows.items():
        if name not in names or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"invalid window override: {name}={count}")
    for flag, overrides in (("--cutoff", cutoffs), ("--step", steps)):
        for name in overrides:
            if name not in names:
                raise ValueError(f"{name}: {flag} override requires a selected task")
            if not isinstance(specs[name].initial_cutoff, str):
                raise ValueError(f"{name}: {flag} requires an official timestamp cutoff")

    source = yaml.safe_load(fev_bench_tasks_path(tasks).read_text(encoding="utf-8"))
    definitions = {}
    for definition in source["tasks"]:
        name = definition.get("name") or definition.get("task_name")
        name = name or definition.get("dataset_config") or Path(definition["dataset_path"]).stem
        definitions[name] = definition
    result = []
    for name in names:
        spec = specs[name]
        definition = deepcopy(definitions[name])
        if mode == "shifted":
            count = windows.get(name, min(max(spec.num_windows, 4), 8))
            if spec.horizon < 1 or spec.num_windows < 1:
                raise ValueError(f"{name}: horizon and window count must be positive")
            if isinstance(spec.initial_cutoff, str):
                if name not in cutoffs or name not in steps:
                    raise ValueError(f"{name}: timestamp cutoffs require both --cutoff and --step")
                try:
                    cutoff = pd.Timestamp(cutoffs[name])
                    official_cutoff = pd.Timestamp(spec.initial_cutoff)
                    offset = to_offset(steps[name])
                    if pd.isna(cutoff) or pd.isna(official_cutoff):
                        raise ValueError("cutoffs must be valid timestamps")
                    if offset is None or offset.n < 1:
                        raise ValueError("--step must be a positive pandas offset")
                    end = cutoff + count * offset
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError(f"{name}: invalid timestamp override: {exc}") from exc
                if end != official_cutoff:
                    raise ValueError(
                        f"{name}: shifted backtest ends at {end.isoformat()}, "
                        f"but official cutoff is {official_cutoff.isoformat()}"
                    )
                definition.update(
                    initial_cutoff=cutoffs[name],
                    window_step_size=steps[name],
                    num_windows=count,
                )
                result.append(definition)
                continue
            step = spec.window_step_size if spec.window_step_size is not None else spec.horizon
            if isinstance(step, str):
                raise ValueError(
                    f"{name}: integer cutoffs require integer steps; load_task_specs "
                    "exposes no sampling frequency to convert an offset-string step"
                )
            if step < 1:
                raise ValueError(f"{name}: step must be positive")
            cutoff = spec.initial_cutoff
            if cutoff is None:
                # fev 0.10.0's default, resolved using the ORIGINAL window count.
                cutoff = -spec.horizon - (spec.num_windows - 1) * step
            shifted_cutoff = cutoff - ((count - 1) * step + spec.horizon)
            if cutoff >= 0 and shifted_cutoff <= 0:
                raise ValueError(
                    f"{name}: shifted cutoff has insufficient history; reduce --windows "
                    "or record the backtest as infeasible"
                )
            definition["num_windows"] = count
            definition["initial_cutoff"] = shifted_cutoff
        result.append(definition)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"tasks": result}, sort_keys=False), encoding="utf-8")
    load_task_specs(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["shifted", "official"], default="shifted")
    parser.add_argument("--tasks", default="fev100")
    parser.add_argument("--windows", nargs="+", action="extend", default=[])
    parser.add_argument("--cutoff", action="append", default=[], metavar="NAME=TIMESTAMP")
    parser.add_argument("--step", action="append", default=[], metavar="NAME=OFFSET")
    args = parser.parse_args(argv)
    try:
        windows = {}
        for value in args.windows:
            name, count = value.rsplit("=", 1)
            if name in windows:
                raise ValueError(f"duplicate window override: {name}")
            windows[name] = int(count)
        cutoffs, steps = {}, {}
        for flag, values, overrides in (
            ("--cutoff", args.cutoff, cutoffs),
            ("--step", args.step, steps),
        ):
            for value in values:
                name, override = value.rsplit("=", 1)
                if name in overrides:
                    raise ValueError(f"{name}: duplicate {flag} override")
                if not name or not override:
                    raise ValueError(f"invalid {flag} override: {value}")
                overrides[name] = override
        shift_tasks(
            read_names(args.names),
            args.out,
            tasks=args.tasks,
            mode=args.mode,
            windows=windows,
            cutoffs=cutoffs,
            steps=steps,
        )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"Task derivation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
