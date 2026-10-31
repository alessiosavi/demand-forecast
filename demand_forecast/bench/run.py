"""Resumable reference forecasts and official fev leaderboard aggregation."""

from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from demand_forecast.bench.specs import FevTaskSpec


def run_baseline(specs: Iterable[FevTaskSpec], out_dir: Path) -> pd.DataFrame:
    """Save each completed task; skip existing seasonal-naive task rows.

    Resume is keyed by task name and model. Use a new output directory when
    changing a task definition or its dataset. The returned frame uses the
    CSV representation, including string-valued lists and metric definitions.
    """
    try:
        from demand_forecast.bench.baselines import SeasonalNaiveBaseline, baseline_summary
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries.csv"
    summaries = (
        pd.read_csv(path, float_precision="round_trip")
        if path.exists()
        else pd.DataFrame(columns=["task", "model_name"])
    )
    completed = set(
        summaries.loc[summaries["model_name"] == SeasonalNaiveBaseline.model_name, "task"]
    )
    for spec in specs:
        if spec.name in completed:
            continue
        summary = {"task": spec.name, **baseline_summary(spec.to_task())}
        row = pd.DataFrame([summary])
        summaries = row if summaries.empty else pd.concat([summaries, row], ignore_index=True)
        _save_summaries(summaries, path)
        completed.add(spec.name)
    if not path.exists():
        _save_summaries(summaries, path)
    return pd.read_csv(path, float_precision="round_trip")


def _save_summaries(summaries: pd.DataFrame, path: Path) -> None:
    """Replace the CSV only once its complete contents have been written."""
    with NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        summaries.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def leaderboard(summaries_csv: Path, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Return fev's model-indexed skill scores with 1,000 bootstrap resamples."""
    try:
        import fev
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc

    summaries = pd.read_csv(summaries_csv, float_precision="round_trip")
    return fev.leaderboard(summaries, baseline_model=baseline, n_resamples=1000)
