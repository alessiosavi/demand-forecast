"""Convert the raw Kaggle mlcspp files into prepared long tables.

One module per competition (the design's hybrid runner): ``prepare`` reads the
raw per-stock CSVs, verifies their pinned contracts, and writes two prepared
variants under ``out_dir``:

- ``official`` — one ``"0056"`` series on a dense daily grid through the last
  raw date, with ``0056_close_y`` NaN exactly on the pinned final ``horizon``
  dates (the competition's 157-step test window).
- ``mirror`` — the same panel with the target NaN from the mirror cutoff (the
  first of the last ``horizon`` observed days) on; those cut closes are split
  out to ``truth.parquet`` and never enter the panel's targets.

Design quirks (controller ruling, recorded in the mlcspp inspection record):

- **Dense day index**: raw ``date`` values are an integer-day index with
  non-trading gaps. They are sorted ascending and mapped to consecutive
  synthetic days (``2000-01-01`` + rank), so the one-series panel is a
  gap-free ``freq="D"`` grid. The official variant carries ``date_map.parquet``
  (raw ``date_int`` -> synthetic ``date``) so the submission builder can map
  forecasts back to the original integers.
- **Covariate lag**: the description's rule is "use yesterday's complete data
  to predict today", so every constituent column enters the panel lagged by one
  day: the covariate at synthetic day ``i`` holds the raw value of day ``i-1``
  (NaN at day 0). All 90 constituent fields (the 10 stocks' 9 fields, raw
  names) are ``known_dynamic``; the frozen model's decoder consumes them per
  horizon step, so step ``t`` sees exactly the data of ``t-1``. There are no
  past-dynamic and no static columns.

``prepare`` fails loudly — like ``store_sales`` — when a raw file deviates
from its pinned column contract, when two files disagree on the shared date
set, when the target's missing block is not exactly the final ``horizon``
dates, or when fewer than ``horizon`` days are observed before that block.
"""

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

from demand_forecast.config import DatasetConfig, Settings
from demand_forecast.data.loader_long import load_long, read_table
from demand_forecast.kaggle.baselines import seasonal_naive_points
from demand_forecast.kaggle.metrics import METRIC_FUNCTIONS
from demand_forecast.kaggle.settings import build_settings, load_template, write_settings
from demand_forecast.kaggle.store_sales import PrepareReport

__all__ = ["SLUG", "build_submission", "prepare", "score_mirror", "validate_submission"]

SLUG = "mlcspp"
TICKER = "0056"
_TARGET = "0056_close_y"
_STOCKS = ("1101", "2327", "2352", "2385", "2449", "2915", "3005", "3532", "6176", "9945")
_FIELDS = (
    "Turnover",
    "Volume",
    "close",
    "eps",
    "high",
    "low",
    "open",
    "ups_and_downs",
    "ups_percent",
)
_KNOWN_DYNAMIC = [f"{stock}_{field}" for stock in _STOCKS for field in _FIELDS]
_EPOCH = "2000-01-01"
_FREQ = "D"
_JOIN_KEYS = ["series_id", "date"]
_ID_WIDTH = len(TICKER)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = {
    "finetune": _REPO_ROOT / "configs/kaggle-v2-l3-finetune.yaml",
    "zero_shot": _REPO_ROOT / "configs/kaggle-v2-l3-zero-shot.yaml",
}


def _read_csv(path: Path, expected: Sequence[str]) -> pd.DataFrame:
    """Read a raw CSV and reject any deviation from the pinned column contract."""
    frame = pd.read_csv(path)
    missing = [name for name in expected if name not in frame.columns]
    unexpected = [name for name in frame.columns if name not in expected]
    if missing or unexpected:
        raise ValueError(
            f"{path.name} columns deviate: missing {missing}, unexpected {unexpected}; "
            f"expected {list(expected)}"
        )
    return frame


def _check_dates(frame: pd.DataFrame, reference: np.ndarray, *, name: str) -> None:
    """Require the frame's integer dates to be unique and equal to the reference set."""
    if frame["date"].duplicated().any():
        raise ValueError(f"{name}: duplicate dates")
    found = {int(value) for value in frame["date"]}
    expected = {int(value) for value in reference}
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing or unexpected:
        raise ValueError(f"{name}: date set deviates: missing {missing}, unexpected {unexpected}")


def _dataset_config(directory: Path) -> DatasetConfig:
    """The long-only roles the mlcspp templates consume."""
    return DatasetConfig(
        source="long",
        path=directory / "long.parquet",
        id_columns=["ticker"],
        timestamp_column="date",
        target=_TARGET,
        target_support="nonnegative",
        known_dynamic=list(_KNOWN_DYNAMIC),
        freq=_FREQ,
        missing_target="nan",
        duplicates="error",
    )


def _write_report(report: PrepareReport, directory: Path) -> None:
    with (directory / "prepare.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(report), stream, indent=2, sort_keys=False)


def prepare(raw_dir: Path, out_dir: Path, *, horizon: int = 157) -> PrepareReport:
    """Convert the raw mlcspp files into the official and mirror variants."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    raw_dir, out_dir = Path(raw_dir), Path(out_dir).resolve()

    target_frame = _read_csv(raw_dir / f"{TICKER}.csv", ("date", _TARGET))
    dates_sorted = np.sort(target_frame["date"].to_numpy(dtype=np.int64))
    _check_dates(target_frame, dates_sorted, name=f"{TICKER}.csv")
    if len(dates_sorted) < horizon:
        raise ValueError(f"horizon {horizon} exceeds the {len(dates_sorted)} raw dates")

    constituent = {}
    for stock in _STOCKS:
        expected = ("date", *(f"{stock}_{field}" for field in _FIELDS))
        frame = _read_csv(raw_dir / f"{stock}.csv", expected)
        _check_dates(frame, dates_sorted, name=f"{stock}.csv")
        constituent[stock] = frame

    calendar = pd.date_range(_EPOCH, periods=len(dates_sorted), freq=_FREQ)
    close = target_frame.set_index("date")[_TARGET].reindex(dates_sorted)
    missing_positions = np.flatnonzero(close.isna().to_numpy())
    expected_positions = np.arange(len(dates_sorted) - horizon, len(dates_sorted))
    if not np.array_equal(missing_positions, expected_positions):
        raise ValueError(
            f"{TICKER}.csv: {_TARGET} must be missing on exactly the final {horizon} "
            f"dates; found {len(missing_positions)} missing"
        )
    observed_count = len(dates_sorted) - horizon
    if observed_count < horizon:
        raise ValueError(f"horizon {horizon} exceeds the {observed_count} observed days")

    official = pd.DataFrame(
        {
            "ticker": np.full(len(dates_sorted), TICKER, dtype=object),
            "date": calendar,
            _TARGET: close.to_numpy(dtype=np.float64),
        }
    )
    for stock in _STOCKS:
        frame = constituent[stock].set_index("date")
        for field in _FIELDS:
            column = f"{stock}_{field}"
            official[column] = (
                frame[column].reindex(dates_sorted).shift(1).to_numpy(dtype=np.float64)
            )
    official = official[["ticker", "date", _TARGET, *_KNOWN_DYNAMIC]]

    mirror_cutoff = calendar[observed_count - horizon]
    mirror = official.copy()
    mirror[_TARGET] = official[_TARGET].mask(official["date"] >= mirror_cutoff)
    truth = pd.DataFrame(
        {
            "series_id": pd.Series([TICKER] * horizon, dtype="string"),
            "date": calendar[observed_count - horizon : observed_count],
            "close": close.to_numpy(dtype=np.float64)[observed_count - horizon : observed_count],
        }
    )
    date_map = pd.DataFrame({"date_int": dates_sorted, "date": calendar})

    official_dir = out_dir / "official"
    mirror_dir = out_dir / "mirror"
    official_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    official.to_parquet(official_dir / "long.parquet", index=False)
    mirror.to_parquet(mirror_dir / "long.parquet", index=False)
    truth.to_parquet(mirror_dir / "truth.parquet", index=False)
    date_map.to_parquet(official_dir / "date_map.parquet", index=False)
    shutil.copyfile(raw_dir / "sample_submission.csv", official_dir / "sample_submission.csv")

    report = PrepareReport(
        slug=SLUG,
        series=1,
        train_days=observed_count,
        test_days=horizon,
        mirror_origin=mirror_cutoff.date().isoformat(),
        official_origin=calendar[observed_count].date().isoformat(),
        quirks={
            "covariate_lag_days": 1,
            "dense_day_index": True,
            "missing_block": "contiguous_final",
        },
    )
    for directory in (official_dir, mirror_dir):
        for mode, template_path in _TEMPLATES.items():
            settings = build_settings(
                load_template(template_path),
                data=_dataset_config(directory),
                horizon=horizon,
            )
            write_settings(settings, directory / f"settings-{mode.replace('_', '-')}.yaml")
        _write_report(report, directory)
    return report


def _normalize_series_id(values: pd.Series) -> pd.Series:
    """Canonical ``series_id`` strings for the pinned ticker.

    The loader emits the four-character ticker (``"0056"``), but a CSV
    round-trip reads a bare ``0056`` column as the integer ``56``; both sides
    are normalized to the zero-padded ticker width so the join cannot miss.
    """
    if pd.api.types.is_numeric_dtype(values):
        values = values.astype("int64").astype("string")
    return values.astype("string").str.zfill(_ID_WIDTH)


def _prediction_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Normalized ``series_id``/``date``/``point`` columns of a prediction frame."""
    missing = [
        column for column in ("series_id", "timestamp", "point") if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    dates = pd.to_datetime(frame["timestamp"], format="mixed")
    if dates.isna().any():
        raise ValueError(f"{name}: null timestamp values")
    return pd.DataFrame(
        {
            "series_id": _normalize_series_id(frame["series_id"]).reset_index(drop=True),
            "date": dates.reset_index(drop=True),
            "point": frame["point"].astype("float64").reset_index(drop=True),
        }
    )


def _key_set(keys: pd.DataFrame) -> set[tuple[object, object]]:
    """The Python-object key set, stable across pandas dtypes and datetime units."""
    return set(zip(keys["series_id"].tolist(), keys["date"].tolist()))


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
    missing = [column for column in (*_JOIN_KEYS, "close") if column not in truth.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    closes = truth["close"].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(closes)).sum())
    if non_finite:
        raise ValueError(f"{path.name}: {non_finite} non-finite truth values")
    dates = pd.to_datetime(truth["date"], format="mixed")
    if dates.isna().any():
        raise ValueError(f"{path.name}: null date values")
    return pd.DataFrame(
        {
            "series_id": _normalize_series_id(truth["series_id"]).reset_index(drop=True),
            "date": dates.reset_index(drop=True),
            "close": closes,
        }
    )


def score_mirror(
    prepared_mirror_dir: Path,
    prediction_frames: dict[str, pd.DataFrame],
    metric_name: str = "rmse",
) -> dict[str, float]:
    """Score the seasonal-naive baseline and every prediction frame on the mirror window.

    ``prepared_mirror_dir`` is a ``mirror`` variant written by
    ``demand_forecast.kaggle.mlcspp.prepare``: its ``settings-finetune.yaml``
    rebuilds the panel, its ``prepare.json`` holds the synthetic
    ``mirror_origin`` and its ``truth.parquet`` holds the cut closes. Keys of
    the returned mapping are ``baseline`` followed by ``prediction_frames`` in
    insertion order.

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
    truth = _load_truth(mirror / "truth.parquet")
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
            joined["close"].to_numpy(dtype=np.float64),
            joined["point"].to_numpy(dtype=np.float64),
        )
    return scores


def build_submission(
    predictions: pd.DataFrame, prepared_official_dir: Path, out: Path
) -> pd.DataFrame:
    """Map official-window predictions back onto the sample dates and write them.

    The predictions carry the synthetic loader ``timestamp``; the prepared
    ``date_map.parquet`` translates each to the original raw integer date.
    Every sample date must receive exactly one finite prediction and the
    predictions must not carry dates outside the sample — a partial, duplicated
    or mixed candidate raises instead of being written. The returned frame is
    the written submission, in sample order (raw dates descending).
    """
    directory = Path(prepared_official_dir)
    sample = pd.read_csv(directory / "sample_submission.csv")
    missing = [column for column in ("date", _TARGET) if column not in sample.columns]
    if missing:
        raise ValueError(f"sample_submission.csv: missing columns {missing}")
    date_map = read_table(directory / "date_map.parquet")
    missing = [column for column in ("date_int", "date") if column not in date_map.columns]
    if missing:
        raise ValueError(f"date_map.parquet: missing columns {missing}")

    keys = _prediction_keys(predictions, name="predictions")
    mapped = keys.merge(
        date_map[["date_int", "date"]], on="date", how="left", validate="many_to_one"
    )
    unmapped = int(mapped["date_int"].isna().sum())
    if unmapped:
        raise ValueError(f"predictions: {unmapped} timestamps absent from date_map.parquet")
    mapped["date_int"] = mapped["date_int"].astype("int64")
    duplicated = int(mapped["date_int"].duplicated().sum())
    if duplicated:
        raise ValueError(f"predictions: {duplicated} rows duplicate a sample date")
    sample_dates = sample["date"].astype("int64")
    extras = int((~mapped["date_int"].isin(sample_dates)).sum())
    if extras:
        raise ValueError(f"predictions: {extras} rows carry dates outside the sample")

    merged = sample.loc[:, ["date"]].merge(
        mapped.loc[:, ["date_int", "point"]],
        left_on="date",
        right_on="date_int",
        how="left",
        validate="one_to_one",
    )
    unmatched = int(merged["point"].isna().sum())
    if unmatched:
        raise ValueError(f"predictions missing for {unmatched} of {len(sample)} sample dates")
    points = merged["point"].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(points)).sum())
    if non_finite:
        raise ValueError(f"predictions: {non_finite} non-finite point values")

    submission = pd.DataFrame(
        {"date": sample["date"].astype("int64").reset_index(drop=True), _TARGET: points}
    )
    submission.to_csv(out, index=False)
    return submission


def validate_submission(candidate: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise ``ValueError`` unless ``candidate`` is a valid submission.

    The candidate must carry the sample's columns in the same order and the
    same raw dates in the same order, with finite non-negative closes — the
    last check a submission passes before any upload.
    """
    if list(candidate.columns) != list(sample.columns):
        raise ValueError(
            f"submission columns {list(candidate.columns)} differ from sample "
            f"{list(sample.columns)}"
        )
    if candidate["date"].astype("int64").tolist() != sample["date"].astype("int64").tolist():
        raise ValueError("submission date order differs from the sample")
    closes = candidate[_TARGET].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(closes)).sum())
    if non_finite:
        raise ValueError(f"submission {_TARGET} contains {non_finite} non-finite values")
    negative = int((closes < 0).sum())
    if negative:
        raise ValueError(f"submission {_TARGET} contains {negative} negative values")
