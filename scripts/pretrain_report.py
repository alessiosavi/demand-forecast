"""Compare Phase B runs with the pinned fev-bench results, offline.

spend.json accepts {"runs": {"frozen": {"spend_usd": 1.2, "wall_time_s": 300},
"zero-shot": {...}, "finetune": {...}, "wide-random": {...}, "heldout-frozen": {...}}.
Optional top-level spend_usd/wall_time_s are experiment totals (including pretraining).
Run totals are displayed unchanged, including the zero-shot total in both blocks.
Source pretraining provenance is read from settings.training.init_from in run.json;
keep that bundle available to report its best step and external diagnostic.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import fev_report
else:
    import fev_report

BASELINE = "seasonal_naive"
PUBLISHED_MODELS = 43
FEV_COMMIT = "193e4fc"
RUN_NAMES = ("frozen", "zero-shot", "finetune", "wide-random", "heldout-frozen")


def _normalize(frame: pd.DataFrame, *, published: bool = False) -> pd.DataFrame:
    # Pinned fev 0.10.0 owns both the task keys and their CSV normalization.
    from fev.analysis import TASK_DEF_COLUMNS, _load_summaries

    frame = frame.copy()
    for column in ("dataset_fingerprint", "num_forecasts"):
        if published:
            if column not in frame:
                frame[column] = None
        elif column not in frame or frame[column].isna().any():
            raise ValueError(f"Our summaries require {column} on every row")
    required = {*TASK_DEF_COLUMNS, "model_name"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    if frame.empty or frame.model_name.isna().any():
        raise ValueError("Empty summaries or missing model_name")
    return _load_summaries(frame).reset_index(drop=True)


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    from fev.analysis import TASK_DEF_COLUMNS

    return frame[TASK_DEF_COLUMNS]


def _one_model(frame: pd.DataFrame) -> str:
    if frame.model_name.nunique() != 1:
        raise ValueError("Expected exactly one candidate model_name per run set")
    if _keys(frame).duplicated().any():
        raise ValueError("duplicate candidate pivot definitions")
    return str(frame.model_name.iloc[0])


def load_published(directory: Path) -> pd.DataFrame:
    """Load local CSVs, discard the upstream baseline, require 43 other models."""
    paths = sorted(Path(directory).glob("*.csv"))
    if not paths:
        raise ValueError("Expected 43 published models and the published seasonal-naive CSV")
    frame = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths], ignore_index=True
    )
    frame = _normalize(frame, published=True)
    frame = frame.loc[frame.model_name != "Seasonal Naive"].reset_index(drop=True)
    if frame.model_name.nunique() != PUBLISHED_MODELS or BASELINE in set(frame.model_name):
        raise ValueError("Expected exactly 43 published models after dropping Seasonal Naive")
    return frame


def _align(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str, *, published: bool = False
) -> pd.DataFrame:
    """Select by fev's pivot definition, then fail on incomplete model coverage."""
    columns = list(_keys(candidate).columns)
    selected = reference.merge(_keys(candidate), on=columns, how="inner", validate="many_to_one")
    if selected.duplicated([*columns, "model_name"]).any():
        raise ValueError(f"{label}: duplicate model/pivot definition")
    counts = (
        selected.groupby("model_name").size().reindex(reference.model_name.unique(), fill_value=0)
    )
    if not (counts == len(candidate)).all():
        raise ValueError(
            f"{label}: pivot definition / full task coverage mismatch: {counts.to_dict()}"
        )
    matched = selected.merge(
        candidate[[*columns, "dataset_fingerprint", "num_forecasts"]],
        on=columns,
        suffixes=("", "_candidate"),
        validate="many_to_one",
    )
    for key in ("num_forecasts",) if published else ("dataset_fingerprint", "num_forecasts"):
        agrees = matched[key].notna() & matched[key].eq(matched[f"{key}_candidate"])
        if published:
            # Older published summaries omit counts; check every reported count.
            agrees = agrees | matched[key].isna()
        if not agrees.fillna(False).all():
            mismatches = (
                matched.loc[
                    ~agrees.fillna(False),
                    ["model_name", "dataset_path", "dataset_config", key, f"{key}_candidate"],
                ]
                .head(5)
                .to_dict("records")
            )
            raise ValueError(f"{label}: {key} mismatch or missing: {mismatches}")
    return selected


def comparison_board(
    candidate: pd.DataFrame, baseline: pd.DataFrame, published: pd.DataFrame
) -> pd.DataFrame:
    """One candidate versus 43 published models and the one canonical baseline."""
    import fev

    candidate, baseline = map(_normalize, (candidate, baseline))
    published = _normalize(published, published=True)
    model = _one_model(candidate)
    if set(baseline.model_name) != {BASELINE}:
        raise ValueError("--baseline must supply the canonical seasonal_naive model")
    if published.model_name.nunique() != PUBLISHED_MODELS:
        raise ValueError("Expected exactly 43 published models")
    if {model, BASELINE} & set(published.model_name) or model == BASELINE:
        raise ValueError(
            "Candidate, published models and baseline must have distinct model_name values"
        )
    if not (candidate.eval_metric == "SQL").all():
        raise ValueError("Comparison requires eval_metric=SQL")
    summaries = [
        candidate,
        _align(baseline, candidate, "baseline"),
        _align(published, candidate, "published", published=True),
    ]
    for frame in summaries:
        if not np.isfinite(frame.test_error).all():
            raise ValueError("Missing/non-finite test_error; full coverage is required")
    board = fev.leaderboard(
        summaries,
        metric_column="test_error",
        baseline_model=BASELINE,
        missing_strategy="error",
        n_resamples=1000,
        seed=123,
    )
    if len(board) != PUBLISHED_MODELS + 2:
        raise ValueError("Expected 45 models: 43 published + candidate + baseline")
    return board


def comparability_rows(tasks: pd.DataFrame, published: pd.DataFrame) -> list[dict]:
    """Report published metadata over the disjoint union of our two task blocks."""
    tasks = _normalize(tasks)
    published = _normalize(published, published=True)
    selected = _align(published, tasks, "published", published=True)
    columns = list(_keys(tasks).columns)
    matched = selected.merge(
        tasks[[*columns, "dataset_fingerprint"]],
        on=columns,
        suffixes=("", "_ours"),
        validate="many_to_one",
    )
    rows = []
    for model, group in matched.groupby("model_name", sort=True):
        matches = group.dataset_fingerprint.eq(group.dataset_fingerprint_ours).fillna(False)
        rows.append(
            {
                "model_name": model,
                "fev_version": ", ".join(sorted(group.fev_version.dropna().unique())),
                "fingerprint matches": f"{int(matches.sum())}/{len(tasks)}",
                "num_forecasts": "unreported" if group.num_forecasts.isna().any() else "equal",
            }
        )
    return rows


def rank_labels(board: pd.DataFrame, model: str) -> tuple[str, str]:
    """Preserve fev's returned order; equal scores share their first position (k=)."""
    win_ties = board.win_rate.eq(board.loc[model, "win_rate"])
    position = int(np.flatnonzero(win_ties.to_numpy())[0]) + 1
    skill = board.loc[model, "skill_score"]
    skill_position = int((board.skill_score > skill).sum()) + 1
    return (
        f"{position}{'=' if win_ties.sum() > 1 else ''} of {len(board)}",
        f"{skill_position}{'=' if board.skill_score.eq(skill).sum() > 1 else ''} of {len(board)}",
    )


def _task(row: pd.Series):
    import fev

    values = row[
        list(_keys(pd.DataFrame([row])).columns) + ["num_windows", "window_step_size"]
    ].to_dict()
    for key, value in values.items():
        if pd.isna(value):
            values[key] = None
        elif isinstance(value, str):
            # CSV task fields include Python-literal lists/dicts and integer cutoffs.
            try:
                values[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
    return fev.Task(**values, task_name=str(row["task"]))


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def _provenance(run_dir: Path, row: pd.Series) -> dict:
    from demand_forecast.core.bundle import load_bundle

    task_dir = run_dir / row["task"]
    report = _json(task_dir / "run.json")
    bundle_dir = task_dir / "bundle"
    manifest = _json(bundle_dir / "manifest.json")
    transfer = report.get("transfer", {})
    source_path = report.get("settings", {}).get("training", {}).get("init_from")
    source = _json(Path(source_path) / "manifest.json") if source_path else {}
    if source and source.get("pretrain_identity") != transfer.get("pretrain_identity"):
        raise ValueError(f"{task_dir}: pretrain_identity mismatch")
    params = None
    if (bundle_dir / "model.pt").exists():
        bundle = load_bundle(bundle_dir, device="cpu")
        params = sum(parameter.numel() for parameter in bundle.model.parameters())
        if bundle.manifest.get("run_identity") != row.get("run_identity"):
            raise ValueError(f"{task_dir}: bundle run_identity mismatch")
    diagnostic = source.get("external_diagnostic") or {}
    return {
        "task": row["task"],
        "pretrain identity": transfer.get("pretrain_identity", "n/a"),
        "pretrain best step": source.get("best_step"),
        "task best step": report.get("fit_result", {}).get("best_step"),
        "params": params,
        "external diagnostic": diagnostic.get("name"),
        "external SQL": diagnostic.get("ext_sql"),
        "external naive SQL": diagnostic.get("ext_sql_naive"),
        "pretrain commit": source.get("git_sha"),
        "run commit": manifest.get("git_sha"),
    }


def build_report(
    runs: dict[str, Path],
    baseline: Path,
    published: Path,
    fev_commit: str,
    spend: dict,
    leaked: list[str],
) -> str:
    """Build the two comparison blocks; no network or benchmark selection."""
    import fev

    if fev_commit != FEV_COMMIT:
        raise ValueError(f"Published results must be pinned to fev commit {FEV_COMMIT}")
    frames = {}
    for name, directory in runs.items():
        frame = pd.read_csv(directory / "summaries.csv", float_precision="round_trip")
        frame = _normalize(frame.loc[frame.model_name != BASELINE])
        _one_model(frame)
        if "task" not in frame or frame.task.isna().any() or frame.task.duplicated().any():
            raise ValueError(f"{name}: expected unique local task names for saved evidence")
        frames[name] = frame
    retail, heldout = frames["frozen"], frames["heldout-frozen"]
    union = pd.concat([retail, heldout], ignore_index=True)
    if _keys(union).duplicated().any():
        raise ValueError("Frozen and held-out task sets overlap")
    for name, expected in (("zero-shot", union), ("finetune", retail), ("wide-random", retail)):
        if len(frames[name]) != len(expected):
            raise ValueError(f"{name}: full task coverage mismatch")
        _align(frames[name], expected, name)
    reference = pd.read_csv(baseline / "summaries.csv", float_precision="round_trip")
    reference = reference.loc[reference.model_name == BASELINE]
    published_frame = load_published(published)
    comparability = comparability_rows(union, published_frame)
    rows, provenance, cached_metrics = [], [], {}
    for block, expected, names in (
        ("frozen tasks", retail, ("frozen", "zero-shot", "finetune", "wide-random")),
        ("held-out tasks", heldout, ("zero-shot", "heldout-frozen")),
    ):
        for name in names:
            candidate = _align(frames[name], expected, name)
            model = _one_model(candidate)
            board = comparison_board(candidate, reference, published_frame)
            ranked = board.loc[model]
            fev_rank, skill_rank = rank_labels(board, model)
            task_metrics = []
            for _, row in candidate.iterrows():
                cache_key = (name, row["task"])
                if cache_key not in cached_metrics:
                    metrics = fev_report.window_metrics(runs[name], _task(row))
                    for key in ("dataset_fingerprint", "num_forecasts", "run_identity"):
                        if metrics[key] != row.get(key):
                            raise ValueError(f"{name}/{row['task']}: saved evidence {key} mismatch")
                    cached_metrics[cache_key] = metrics
                    provenance.append({"run": name, **_provenance(runs[name], row)})
                task_metrics.append(cached_metrics[cache_key])
            cost = spend.get("runs", {}).get(name, {})
            rows.append(
                {
                    "block": block,
                    "run": name,
                    "model_name": model,
                    "per-task SQL": "; ".join(
                        f"{row.task}: {row.test_error:.9g}" for row in candidate.itertuples()
                    ),
                    "skill ± bounds": f"{ranked.skill_score:.9g} [{ranked.skill_score_lower:.9g}, {ranked.skill_score_upper:.9g}]",
                    "win rate": ranked.win_rate,
                    "fev rank": fev_rank,
                    "skill rank": skill_rank,
                    "mean MASE": float(np.mean([np.mean(m["MASE"]) for m in task_metrics])),
                    "mean coverage_80": float(
                        np.mean([np.mean(m["coverage_80"]) for m in task_metrics])
                    ),
                    "training time s": candidate.training_time_s.sum(min_count=len(candidate)),
                    "inference time s": candidate.inference_time_s.sum(min_count=len(candidate)),
                    "spend USD": cost.get("spend_usd"),
                    "wall time s": cost.get("wall_time_s"),
                }
            )
    return "\n".join(
        [
            "# Pretraining comparison\n",
            f"post_hoc: no\n\nfev version: {fev.__version__}; published fev commit: {fev_commit}.\n",
            "Each row compares one candidate with 43 published models and our canonical seasonal_naive "
            "baseline (45 models, full coverage). The published Seasonal Naive is excluded. "
            "fev rank follows the returned leaderboard order (win rate); skill rank sorts skill_score "
            "descending. Ties share the first position, marked k=. Skill bounds are the 95% bootstrap "
            "interval, n_resamples=1000, seed=123. Task matching uses fev pivot definitions, never task_name.\n",
            "MASE and coverage_80 are recomputed per window, averaged equally within each task, then "
            "equally across tasks. Training/inference times are sums for each block. Spend and wall time "
            "are whole-run totals from spend.json; zero-shot totals repeat across blocks and must not be "
            "added twice. Missing cost/provenance is unknown. External SQL is diagnostic only.\n",
            "wide-random is control (i); heldout-frozen is the frozen-config control (ii).\n",
            "Published rows were produced with fev 0.6.1–0.11.0.dev0; fev's dataset fingerprint "
            "is sensitive to fev/pyarrow versions (seven values for m5_1W across the 44 pinned files), "
            "so fingerprint equality is enforced only within our own runs, while structural identity "
            "(pivot definition and num_forecasts where reported) is enforced for every row.\n",
            "Limitation: if a published model was evaluated on a different revision of a dataset, "
            "its error is not comparable; this risk is mitigated by identical num_forecasts where "
            "reported and by fev-bench's own leaderboard mixing these files.\n",
            "## Comparison\n",
            fev_report._table(rows),
            "## Comparability\n",
            "Fingerprint matches are k/n over the union of frozen and held-out tasks; "
            "unreported means at least one selected task lacks num_forecasts.\n",
            fev_report._table(comparability),
            "## Leakage\n",
            "\n".join(f"- {task}" for task in leaked) if leaked else "None declared.",
            "\n## Provenance\n",
            fev_report._table(provenance),
            "## Experiment totals\n",
            fev_report._table(
                [
                    {
                        "spend USD": spend.get("spend_usd"),
                        "wall time s": spend.get("wall_time_s"),
                    }
                ]
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*RUN_NAMES, "baseline", "published", "spend", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fev-commit", required=True, choices=[FEV_COMMIT])
    parser.add_argument("--leaked", nargs="*", default=[])
    args = parser.parse_args(argv)
    try:
        report = build_report(
            {name: getattr(args, name.replace("-", "_")) for name in RUN_NAMES},
            args.baseline,
            args.published,
            args.fev_commit,
            json.loads(args.spend.read_text(encoding="utf-8")),
            args.leaked,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    fev_report._write_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
