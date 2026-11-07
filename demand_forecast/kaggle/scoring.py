"""Mirror-window scoring for the store-sales bench.

The competition test labels are hidden, so every run is scored on the mirror
window: the last ``horizon`` permitted train days, cut out of the mirror panel
and kept in ``truth.parquet``. This module rebuilds the mirror panel from the
prepared ``settings-finetune.yaml``, computes the seasonal-naive baseline at
the mirror origin from ``prepare.json``, and scores the baseline plus every
prediction frame with the competition metric.

Prediction frames carry the loader's ``series_id`` (``store_nbr|family``),
``timestamp`` and ``point``. Every frame — the baseline included — must cover
exactly the truth row set, so all runs are compared on one shared, fully
observed row set: the join never drops rows and the metric's silent NaN-pair
exclusion never masks an incomplete run. Point values are validated loudly
(finite and non-negative) before scoring; the metric's NaN skipping is not a
guard. Join-key normalization is shared with ``submission`` so the writer and
the scorer cannot drift apart.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

from demand_forecast.config import Settings
from demand_forecast.data.loader_long import load_long, read_table
from demand_forecast.kaggle.baselines import seasonal_naive_points
from demand_forecast.kaggle.metrics import METRIC_FUNCTIONS
from demand_forecast.kaggle.submission import _JOIN_KEYS, _catalog_keys, _prediction_keys

__all__ = ["score_mirror"]


def _key_set(keys: pd.DataFrame) -> set[tuple[object, object, object]]:
    """The Python-object key set, stable across pandas dtypes and datetime units."""
    return set(zip(keys["store_nbr"].tolist(), keys["family"].tolist(), keys["date"].tolist()))


def _require_valid_points(frame: pd.DataFrame, *, name: str) -> None:
    """Raise unless every point is finite and non-negative (nonnegative target support)."""
    points = frame["point"].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(points)).sum())
    if non_finite:
        raise ValueError(f"{name}: {non_finite} non-finite point values")
    negative = int((points < 0).sum())
    if negative:
        raise ValueError(f"{name}: {negative} negative point values for a nonnegative target")


def _load_truth(path: Path) -> pd.DataFrame:
    """Read the mirror truth table and reject non-finite labels."""
    truth = read_table(path)
    missing = [column for column in (*_JOIN_KEYS, "sales") if column not in truth.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    sales = truth["sales"].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(sales)).sum())
    if non_finite:
        raise ValueError(f"{path.name}: {non_finite} non-finite truth values")
    return truth


def score_mirror(
    prepared_mirror_dir: Path,
    prediction_frames: dict[str, pd.DataFrame],
    metric_name: str = "rmsle",
) -> dict[str, float]:
    """Score the seasonal-naive baseline and every prediction frame on the mirror window.

    ``prepared_mirror_dir`` is a ``mirror`` variant written by
    ``demand_forecast.kaggle.store_sales.prepare``: its ``settings-finetune.yaml``
    rebuilds the panel, its ``prepare.json`` holds ``mirror_origin`` and its
    ``truth.parquet`` holds the cut labels. Keys of the returned mapping are
    ``baseline`` followed by ``prediction_frames`` in insertion order.

    Raises ``ValueError`` when a frame's row set differs from the truth
    (missing or extra keys), when any point is non-finite or negative, when the
    truth carries non-finite labels, or when ``metric_name`` is unknown.
    """
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(
            f"unknown metric {metric_name!r}; expected one of {sorted(METRIC_FUNCTIONS)}"
        )
    if "baseline" in prediction_frames:
        raise ValueError("prediction_frames must not reuse the reserved 'baseline' key")
    mirror = Path(prepared_mirror_dir)
    settings = Settings.from_yaml(mirror / "settings-finetune.yaml")
    panel = load_long(settings.data)
    report = json.loads((mirror / "prepare.json").read_text(encoding="utf-8"))
    origin = pd.Timestamp(str(report["mirror_origin"]))

    frames: dict[str, pd.DataFrame] = {
        "baseline": seasonal_naive_points(panel, origin, settings.forecast.horizon),
        **prediction_frames,
    }
    keys = {name: _prediction_keys(frame, name=name) for name, frame in frames.items()}
    truth = _catalog_keys(_load_truth(mirror / "truth.parquet"), ["sales"], name="truth.parquet")
    truth_keys = _key_set(truth)
    metric = METRIC_FUNCTIONS[metric_name]

    scores: dict[str, float] = {}
    for name, frame in keys.items():
        _require_valid_points(frame, name=name)
        frame_keys = _key_set(frame)
        if frame_keys != truth_keys:
            raise ValueError(
                f"{name}: row set differs from truth.parquet "
                f"(missing {len(truth_keys - frame_keys)}, extra {len(frame_keys - truth_keys)})"
            )
        joined = frame.merge(truth, on=_JOIN_KEYS, how="inner", validate="one_to_one")
        joined = joined.sort_values(_JOIN_KEYS, kind="stable").reset_index(drop=True)
        scores[name] = metric(
            joined["sales"].to_numpy(dtype=np.float64),
            joined["point"].to_numpy(dtype=np.float64),
        )
    return scores
