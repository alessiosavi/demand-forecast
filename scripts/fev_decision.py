"""Report benchmark decision figures from local error CSVs and fev task YAML."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from demand_forecast.bench import decision

_RUNTIME_NOTE = "The registration's runtime criterion is unavailable."


def _assignment(value: str) -> tuple[str, str]:
    name, _, target = value.partition("=")
    if not name.strip() or not target.strip():
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name, target


def _unique(entries: list[tuple[str, str]], flag: str) -> dict[str, str]:
    result = dict(entries)
    if len(result) != len(entries):
        raise ValueError(f"duplicate name in {flag}")
    return result


def _read_scores(path: Path, label: str | None) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        if "test_error" not in headers or not {"task", "task_name"} & headers:
            raise ValueError(f"{path}: expected task or task_name and test_error columns")
        if label is not None and "model_name" not in headers:
            raise ValueError(f"{path}: --model-name needs a model_name column")

        scores = {}
        for row in reader:
            if label is not None and row["model_name"] != label:
                continue
            task = row.get("task") or row.get("task_name")
            if not task:
                raise ValueError(f"{path}:{reader.line_num}: missing task name")
            if task in scores:
                hint = "select a label with --model-name" if label is None else f"label {label!r}"
                raise ValueError(f"{path}: multiple rows for task {task!r} ({hint})")
            error = (row["test_error"] or "").strip()
            try:
                scores[task] = float(error) if error else math.nan
            except ValueError as exc:
                raise ValueError(f"{path}:{reader.line_num}: invalid test_error {error!r}") from exc

        if label is not None and not scores:
            raise ValueError(f"{path}: no rows for model_name {label!r}")
        return scores


def _require_complete(systems: dict[str, dict[str, float]], tasks: list[str]) -> None:
    for name, errors in systems.items():
        missing = [task for task in tasks if not math.isfinite(errors.get(task, math.nan))]
        if missing:
            raise ValueError(
                f"--require-complete: {name!r} lacks finite errors for tasks: {', '.join(missing)}"
            )


def _slice_report(tasks, baseline, panel, systems, incumbent, against):
    figures = {}
    ratios = {}
    if tasks:
        against_scored = decision.relative(against, baseline, tasks) if against is not None else {}
        for name, errors in systems.items():
            scored = decision.relative(errors, baseline, tasks)
            figures[name] = {
                "w": decision.win_rate(errors, panel, baseline, tasks),
                "s": decision.skill(errors, baseline, tasks) if scored else None,
                "f": decision.firsts(errors, panel, baseline, tasks),
                "scored": sorted(scored),
            }
            if against is not None and name != incumbent:
                ratios[name] = (
                    asdict(decision.paired_ratio(errors, against, baseline, tasks))
                    if scored.keys() & against_scored.keys()
                    else None
                )
    return {"tasks": tasks, "figures": figures, "paired_ratios": ratios}


def _selection(incumbent, rows):
    figures = {
        name: decision.Figures(row["w"], row["s"], row["f"], frozenset(row["scored"]))
        for name, row in rows.items()
        if row["s"] is not None
    }
    reference = figures.pop(incumbent, None)
    candidates = [name for name in rows if name != incumbent]
    if reference is None:
        return dict.fromkeys(candidates), None
    eligibility = {
        name: name in figures and decision.eligible(figures[name], reference) for name in candidates
    }
    return eligibility, decision.choose((incumbent, reference), figures)


def _markdown(report: dict) -> str:
    lines = [
        "# Benchmark decision",
        "",
        f"Panel: {', '.join(report['panel'])}",
        f"Incumbent: {report['incumbent']}",
        "",
        "## Selection on all tasks",
        "",
        report["runtime_note"],
    ]
    for name, eligible in report["eligibility"].items():
        status = "unavailable" if eligible is None else "eligible" if eligible else "ineligible"
        lines.append(f"- {name}: {status}")
    lines.append(f"- Choice: {report['choice'] or 'unavailable'}")
    for name, data in report["slices"].items():
        lines.extend(
            [
                "",
                f"## {name} ({len(data['tasks'])} tasks)"
                if data["tasks"]
                else f"## {name} (empty)",
                "",
            ]
        )
        if not data["tasks"]:
            continue
        lines.extend(["| System | W | S | F | Scored |", "| --- | ---: | ---: | ---: | ---: |"])
        for system, figures in data["figures"].items():
            skill = "unscored" if figures["s"] is None else f"{figures['s']:.4f}"
            lines.append(
                f"| {system} | {figures['w']:.4f} | {skill} | {figures['f']} | {len(figures['scored'])} |"
            )
        if data["paired_ratios"]:
            lines.extend(
                [
                    "",
                    f"Paired ratios against {report['against']}",
                    "",
                    "| Candidate | Ratio | 90% interval | Families |",
                    "| --- | ---: | --- | ---: |",
                ]
            )
            for system, paired in data["paired_ratios"].items():
                if paired is None:
                    lines.append(f"| {system} | unavailable | — | — |")
                    continue
                interval = (
                    "—" if paired["low"] is None else f"[{paired['low']:.4f}, {paired['high']:.4f}]"
                )
                lines.append(
                    f"| {system} | {paired['ratio']:.4f} | {interval} | {paired['families']} |"
                )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", type=Path, help="file listing the selected task names")
    parser.add_argument("baseline", type=Path, help="baseline error CSV")
    parser.add_argument("tasks", type=Path, help="fev task YAML")
    parser.add_argument(
        "--panel", action="append", type=_assignment, metavar="N=FILE", required=True
    )
    parser.add_argument("--incumbent", type=_assignment, metavar="N=FILE", required=True)
    parser.add_argument(
        "--candidate", action="append", type=_assignment, metavar="N=FILE", default=[]
    )
    parser.add_argument("--against", type=_assignment, metavar="N=FILE")
    parser.add_argument(
        "--model-name",
        action="append",
        type=_assignment,
        metavar="N=LABEL",
        default=[],
        help="select a CSV model_name for N (use baseline for the baseline CSV)",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--out", type=Path, metavar="DIR")
    args = parser.parse_args(argv)

    try:
        from demand_forecast.bench.yardsticks.runner import load_tasks, read_names

        incumbent, incumbent_file = args.incumbent
        candidate_files = _unique(args.candidate, "--candidate")
        if incumbent in candidate_files:
            raise ValueError(f"incumbent and candidate share the name {incumbent!r}")
        panel_files = _unique(args.panel, "--panel")
        model_names = _unique(args.model_name, "--model-name")
        against_name = args.against[0] if args.against else None
        unknown = model_names.keys() - {
            "baseline",
            incumbent,
            *panel_files,
            *candidate_files,
            against_name,
        }
        if unknown:
            raise ValueError(f"unknown names in --model-name: {', '.join(sorted(unknown))}")

        def read(name: str, path: str | Path) -> dict[str, float]:
            return _read_scores(Path(path), model_names.get(name))

        tasks = load_tasks(args.tasks, read_names(f"@{args.names}"))
        cohorts = decision.slices(tasks)
        baseline = read("baseline", args.baseline)
        panel = [read(name, path) for name, path in panel_files.items()]
        systems = {incumbent: read(incumbent, incumbent_file)}
        systems.update({name: read(name, path) for name, path in candidate_files.items()})
        if args.require_complete:
            _require_complete(systems, cohorts["all"])
        against = read(*args.against) if args.against else None

        slices = {
            key: _slice_report(names, baseline, panel, systems, incumbent, against)
            for key, names in cohorts.items()
        }
        eligibility, choice = _selection(incumbent, slices["all"]["figures"])
        report = {
            "panel": list(panel_files),
            "incumbent": incumbent,
            "against": against_name,
            "runtime_note": _RUNTIME_NOTE,
            "eligibility": eligibility,
            "choice": choice,
            "slices": slices,
        }
        markdown = _markdown(report)
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "decision.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            (args.out / "decision.md").write_text(markdown, encoding="utf-8")
        print(markdown, end="")
    except (ImportError, OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
