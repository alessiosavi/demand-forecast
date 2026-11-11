"""Select recipes on backtests and report dataset standings against published fev rows."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import pandas as pd

from scripts.fev_shift_tasks import read_names


def read_rows(path: Path) -> list[dict]:
    """Read fev cells, keeping task_name as identity (never dataset_config)."""
    path = path / "summaries.csv" if path.is_dir() else path
    frame = pd.read_csv(path, float_precision="round_trip")
    if "task_name" not in frame and "task" in frame:
        frame["task_name"] = frame["task"]
    if not {"task_name", "test_error", "model_name"} <= set(frame):
        raise ValueError(f"{path}: missing task_name, test_error or model_name")
    if frame[["task_name", "model_name"]].isna().any().any():
        raise ValueError(f"{path}: missing task or model name")
    if "task" in frame and not frame.task.eq(frame.task_name).all():
        raise ValueError(f"{path}: conflicting task and task_name")
    frame["test_error"] = pd.to_numeric(frame.test_error, errors="raise")
    if (frame.test_error < 0).any():
        raise ValueError(f"{path}: negative test_error")
    return frame.to_dict("records")


def _by_task(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        name = row["task_name"]
        if name in result:
            raise ValueError(f"{label}: more than one row for task {name}")
        result[name] = row
    return result


def _score(row: dict | None) -> float | None:
    value = row["test_error"] if row else None
    return float(value) if value is not None and math.isfinite(value) else None


def select_arms(
    arms: dict[str, Path], reference: str, min_gain: float, names: list[str] | None = None
) -> dict:
    """Make the registered decision using only the supplied backtest rows."""
    if reference not in arms:
        raise ValueError(f"reference arm is not supplied: {reference}")
    if not math.isfinite(min_gain) or not 0 <= min_gain < 1:
        raise ValueError("min-gain must be finite and in [0, 1)")
    cells = {name: _by_task(read_rows(path), name) for name, path in arms.items()}
    tasks = (
        names if names is not None else sorted({task for rows in cells.values() for task in rows})
    )
    choices = {}
    for task in tasks:
        scores = {arm: _score(rows.get(task)) for arm, rows in cells.items()}
        baseline = scores[reference]
        eligible = {arm: score for arm, score in scores.items() if score is not None}
        if baseline is None:
            chosen, reason = (
                None,
                f"missing or non-finite reference row; retain {reference} operationally",
            )
        else:
            best = min(eligible, key=lambda arm: (eligible[arm], arm != reference, arm))
            chosen = best if eligible[best] <= (1 - min_gain) * baseline else reference
            reason = "minimum gain met" if chosen != reference else "reference retained"
        choices[task] = {
            "chosen": chosen,
            "reason": reason,
            "reference_error": baseline,
            "chosen_error": scores[chosen] if chosen is not None else None,
            "scores": scores,
        }
    return {"reference": reference, "min_gain": min_gain, "tasks": choices}


def _leaking(row: dict) -> bool:
    value = str(row.get("trained_on_this_dataset", "")).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{row['task_name']}/{row['model_name']}: missing or invalid leak flag")
    return value in {"true", "1"}


def _require_sql(row: dict) -> None:
    metric = row.get("eval_metric", "SQL")
    if isinstance(metric, str) and metric.startswith("{"):
        metric = ast.literal_eval(metric)
    if isinstance(metric, dict):
        metric = metric.get("name")
    if metric != "SQL":
        raise ValueError(f"{row['task_name']}/{row['model_name']}: test_error is not SQL")


def standing(
    entries: list[Path],
    published: Path,
    names: list[str] | None = None,
    include_leaking: bool = False,
) -> dict:
    """Use competition ranks for ties; first place requires a strict improvement."""
    ours = _by_task([row for path in entries for row in read_rows(path)], "entry")
    paths = sorted(published.glob("*.csv"))
    if not paths:
        raise ValueError(f"no published CSV files in {published}")
    cells = {}
    published_tasks = set()
    for path in paths:
        for row in read_rows(path):
            key = (row["task_name"], row["model_name"])
            if key in cells:
                raise ValueError(f"duplicate published task/model cell: {key}")
            _require_sql(row)
            leak = _leaking(row)
            published_tasks.add(row["task_name"])
            cells[key] = row if include_leaking or not leak else None
    tasks = names if names is not None else sorted(set(ours) | published_tasks)
    rows = []
    for task in tasks:
        our_row = ours.get(task)
        if our_row is not None:
            _require_sql(our_row)
        score = _score(our_row)
        eligible = [
            row
            for (name, _), row in cells.items()
            if name == task and row is not None and _score(row) is not None
        ]
        eligible.sort(key=lambda row: (row["test_error"], row["model_name"]))
        best = float(eligible[0]["test_error"]) if eligible else None
        seasonal = [
            row["test_error"]
            for row in eligible
            if "".join(c for c in row["model_name"].lower() if c.isalnum()) == "seasonalnaive"
        ]
        comparable = score is not None and best is not None
        gap = 100 * (score / best - 1) if comparable and best > 0 else None
        if comparable and best == score == 0:
            gap = 0.0
        rows.append(
            {
                "task_name": task,
                "our_model": our_row["model_name"] if our_row else None,
                "our_sql": score,
                "seasonal_naive_sql": min(seasonal) if seasonal else None,
                "best_published_model": eligible[0]["model_name"] if eligible else None,
                "best_published_sql": best,
                "gap_percent": gap,
                "rank": 1 + sum(row["test_error"] < score for row in eligible)
                if comparable
                else None,
                "first": bool(comparable and score < best),
                "published_within_5pct": sum(row["test_error"] <= 1.05 * best for row in eligible)
                if best is not None
                else 0,
                "eligible_published_models": len(eligible),
                "reason": "missing or non-finite entry"
                if score is None
                else "no eligible published cells"
                if best is None
                else None,
            }
        )
    totals = {
        "tasks": len(rows),
        "scored": sum(row["rank"] is not None for row in rows),
        "first": sum(row["first"] for row in rows),
        "top_3": sum(row["rank"] is not None and row["rank"] <= 3 for row in rows),
        "top_5": sum(row["rank"] is not None and row["rank"] <= 5 for row in rows),
    }
    return {"include_leaking": include_leaking, "totals": totals, "tasks": rows}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _markdown(report: dict) -> str:
    def cell(value):
        if value is None:
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = report["tasks"]
    totals = report["totals"]
    text = "# Dataset standing\n\n"
    text += f"First: {totals['first']}; top 3: {totals['top_3']}; top 5: {totals['top_5']}. "
    text += f"Comparable: {totals['scored']}/{totals['tasks']}.\n\n"
    text += f"Include leaking published cells: {report['include_leaking']}.\n\n"
    if rows:
        columns = list(rows[0])
        text += "| " + " | ".join(columns) + " |\n"
        text += "| " + " | ".join("---" for _ in columns) + " |\n"
        text += "".join(
            "| " + " | ".join(cell(row[key]) for key in columns) + " |\n" for row in rows
        )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--arm", nargs="+", action="extend", required=True)
    select.add_argument("--reference", default="A0")
    select.add_argument("--min-gain", type=float, default=0.01)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--names")
    compare = commands.add_parser("standing")
    compare.add_argument("--entry", nargs="+", action="extend", type=Path, required=True)
    compare.add_argument("--published", type=Path, required=True)
    compare.add_argument("--out-dir", type=Path, required=True)
    compare.add_argument("--names")
    compare.add_argument("--include-leaking", action="store_true")
    args = parser.parse_args(argv)
    try:
        names = read_names(args.names)
        if args.command == "select":
            arms = {}
            for value in args.arm:
                name, path = value.split("=", 1)
                if not name or not path or name in arms:
                    raise ValueError(f"empty or duplicate arm: {value}")
                arms[name] = Path(path)
            report = select_arms(arms, args.reference, args.min_gain, names)
            if args.out.suffix.lower() != ".json":
                raise ValueError("selection --out must have a .json extension")
            _write_json(args.out, report)
            pd.DataFrame(
                [{"task_name": name, **row} for name, row in report["tasks"].items()]
            ).to_csv(args.out.with_suffix(".csv"), index=False)
        else:
            report = standing(args.entry, args.published, names, args.include_leaking)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            _write_json(args.out_dir / "standing.json", report)
            pd.DataFrame(report["tasks"]).to_csv(args.out_dir / "standing.csv", index=False)
            (args.out_dir / "standing.md").write_text(_markdown(report), encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(f"Dataset wins refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
