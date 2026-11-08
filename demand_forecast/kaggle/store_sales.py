"""Convert the raw Kaggle store-sales files into prepared long/static tables.

One module per competition (the design's hybrid runner): ``prepare`` reads the
raw CSVs, verifies their pinned column contracts, and writes two prepared
variants under ``out_dir``:

- ``official`` — every observed ``(store_nbr, family)`` series on its daily
  grid through the last test day, with the target ``log1p_sales`` NaN from the
  test period on.
- ``mirror`` — the same panel with the target NaN from the mirror cutoff (the
  first of the last ``horizon`` train days) on; those cut targets are split
  out to ``truth.parquet`` (log-space as well) and never enter the panel's
  targets.

The target is the pinned ``log1p`` transform of raw sales: observed rows carry
``log1p_sales = log1p(sales)``, while missing sale rows inside the history keep
the ``missing_target == "zero"`` policy in raw space first (their log1p then
reads zero, exactly mirroring the historical raw-target policy) and window
rows are NaN. Mirror truth is log-space too, so scoring the mirror window with
plain ``rmse`` equals the historical RMSLE on raw sales.

Covariate availability is pinned per variant: the known covariates
(``onpromotion``, ``is_national_holiday``, the payday flags ``is_day_15`` and
``is_month_end``, and ``oil_price``) extend into the forecast window, from
``test.csv``, the holiday calendar, the calendar itself and ``oil.csv``
respectively; ``oil_price`` is filled for every date the raw file covers —
mirror and official windows included — and NaN exactly where it has no row.
The past-dynamic covariates (``transactions`` and the raw-sales lag/rolling
block) stop at the variant's origin: the block's lags are strict shifts of the
panel's sales series and its rolls are trailing means of the one-day-shifted
series (``min_periods=1``), so no value at date ``t`` ever sees same-day
sales; missing history rows participate exactly as they participate in the
target (as zero-filled sales). Past-dynamic values at or after the origin
would extend each series' loaded history past the origin, and the trainer
would then publish a bundle whose earliest permissible origin outruns the
window origin the kernel workload forecasts at. The mirror variant therefore
masks the block from the mirror cutoff and the official variant from the
official test start, so no in-window sales information is carried anywhere in
the prepared files. Official test rows carry NaN targets.
"""

import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

from demand_forecast.config import DatasetConfig
from demand_forecast.data.frequency import to_offset
from demand_forecast.kaggle import scoring
from demand_forecast.kaggle.settings import build_settings, load_template, write_settings

__all__ = ["PrepareReport", "SLUG", "prepare", "score_mirror"]

SLUG = "store-sales-time-series-forecasting"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = {
    "finetune": _REPO_ROOT / "configs/kaggle-v2-l3-finetune.yaml",
    "zero_shot": _REPO_ROOT / "configs/kaggle-v2-l3-zero-shot.yaml",
}

_TRAIN_COLUMNS = ("id", "date", "store_nbr", "family", "sales", "onpromotion")
_TEST_COLUMNS = ("id", "date", "store_nbr", "family", "onpromotion")
_SAMPLE_COLUMNS = ("id", "sales")
_STORES_COLUMNS = ("store_nbr", "city", "state", "type", "cluster")
_OIL_COLUMNS = ("date", "dcoilwtico")
_TRANSACTIONS_COLUMNS = ("date", "store_nbr", "transactions")
_HOLIDAY_COLUMNS = ("date", "type", "locale", "locale_name", "description", "transferred")

_TARGET = "log1p_sales"
_KNOWN_DYNAMIC = ["onpromotion", "is_national_holiday", "is_day_15", "is_month_end", "oil_price"]
_PAST_DYNAMIC = ["transactions", "sales_lag7", "sales_lag28", "sales_roll7", "sales_roll28"]
_STATIC_CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]
_FREQ = "D"
_LONG_COLUMNS = ["store_nbr", "family", "date", _TARGET, *_KNOWN_DYNAMIC, *_PAST_DYNAMIC]
_NATIONAL_TYPES = ("Holiday", "Transfer", "Bridge", "Additional")


@dataclass
class PrepareReport:
    """What ``prepare`` wrote; the mlcspp converter reuses this shape."""

    slug: str
    series: int
    train_days: int
    test_days: int
    mirror_origin: str
    official_origin: str
    quirks: dict[str, int | str]


def _read_csv(path: Path, expected: tuple[str, ...]) -> pd.DataFrame:
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


def _parse_date(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _require_unique_keys(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    duplicated = int(frame.duplicated(subset=list(keys)).sum())
    if duplicated:
        raise ValueError(f"duplicate keys: {duplicated} rows in {name} on {list(keys)}")


def _series_first_dates(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per observed series with its earliest observed day."""
    observed = pd.concat(
        [train[["store_nbr", "family", "date"]], test[["store_nbr", "family", "date"]]]
    )
    return (
        observed.drop_duplicates()
        .groupby(["store_nbr", "family"], sort=False, as_index=False)["date"]
        .min()
        .rename(columns={"date": "first_date"})
        .sort_values(["store_nbr", "family"], kind="stable")
        .reset_index(drop=True)
    )


def _union_grid(first_dates: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The series x daily-dates union window, starting at each series' first day."""
    grid = first_dates.merge(pd.DataFrame({"date": dates}), how="cross")
    return grid.loc[grid["date"] >= grid["first_date"], ["store_nbr", "family", "date"]]


def _payday_flags(dates: pd.Series) -> dict[str, pd.Series]:
    """The pinned known payday flags for every date: the 15th and month ends."""
    day = dates.dt.day
    return {
        "is_day_15": (day == 15).astype("int64"),
        "is_month_end": (day == dates.dt.days_in_month).astype("int64"),
    }


def _sales_covariates(long: pd.DataFrame) -> pd.DataFrame:
    """Add the raw-sales lag/rolling block, computed per series from the past only.

    The block derives from the same sales series the target is built from, so
    missing history rows participate exactly like the target does (they carry
    the zero-fill policy by the time this runs). Lags are strict shifts; rolls
    are trailing means of the one-day-shifted series with ``min_periods=1``,
    so a value at date ``t`` never sees same-day sales and a series' first day
    leaves the rolls NaN.
    """
    frame = long.sort_values(["store_nbr", "family", "date"], kind="stable").reset_index(drop=True)
    keys = [frame["store_nbr"], frame["family"]]
    sales = frame["sales"]
    frame["sales_lag7"] = sales.groupby(keys, sort=False).shift(7)
    frame["sales_lag28"] = sales.groupby(keys, sort=False).shift(28)
    prior = sales.groupby(keys, sort=False).shift(1)
    for window, column in ((7, "sales_roll7"), (28, "sales_roll28")):
        frame[column] = prior.groupby(keys, sort=False).transform(
            lambda values, window=window: values.rolling(window, min_periods=1).mean()
        )
    return frame


def _clip_at_origin(
    frame: pd.DataFrame, origin: pd.Timestamp, *, mask_target: bool
) -> pd.DataFrame:
    """Drop past-dynamic values (and optionally the target) from ``origin`` on."""
    clipped = frame.copy()
    after = clipped["date"] >= origin
    for column in _PAST_DYNAMIC:
        clipped[column] = clipped[column].mask(after)
    if mask_target:
        clipped[_TARGET] = clipped[_TARGET].mask(after)
    return clipped


def _static_table(first_dates: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Store metadata replicated per observed ``(store_nbr, family)`` key."""
    static = first_dates[["store_nbr", "family"]].merge(
        stores[["store_nbr", "city", "state", "type", "cluster"]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )
    return (
        static[_STATIC_CATEGORICAL]
        .sort_values(["store_nbr", "family"], kind="stable")
        .reset_index(drop=True)
    )


def _dataset_config(directory: Path) -> DatasetConfig:
    """The long/static roles the store-sales templates consume."""
    return DatasetConfig(
        source="long",
        path=directory / "long.parquet",
        id_columns=["store_nbr", "family"],
        timestamp_column="date",
        target=_TARGET,
        target_support="nonnegative",
        known_dynamic=list(_KNOWN_DYNAMIC),
        past_dynamic=list(_PAST_DYNAMIC),
        static_path=directory / "static.parquet",
        static_categorical=list(_STATIC_CATEGORICAL),
        freq=_FREQ,
        missing_target="zero",
        duplicates="error",
    )


def _write_report(report: PrepareReport, directory: Path) -> None:
    with (directory / "prepare.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(report), stream, indent=2, sort_keys=False)


def prepare(raw_dir: Path, out_dir: Path, *, horizon: int = 16) -> PrepareReport:
    """Convert the raw store-sales files into the official and mirror variants."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    raw_dir, out_dir = Path(raw_dir), Path(out_dir).resolve()

    train = _parse_date(_read_csv(raw_dir / "train.csv", _TRAIN_COLUMNS))
    test = _parse_date(_read_csv(raw_dir / "test.csv", _TEST_COLUMNS))
    sample = _read_csv(raw_dir / "sample_submission.csv", _SAMPLE_COLUMNS)
    stores = _read_csv(raw_dir / "stores.csv", _STORES_COLUMNS)
    oil = _parse_date(_read_csv(raw_dir / "oil.csv", _OIL_COLUMNS))
    transactions = _parse_date(_read_csv(raw_dir / "transactions.csv", _TRANSACTIONS_COLUMNS))
    holidays = _parse_date(_read_csv(raw_dir / "holidays_events.csv", _HOLIDAY_COLUMNS))

    _require_unique_keys(train, ("store_nbr", "family", "date"), "train.csv")
    _require_unique_keys(test, ("store_nbr", "family", "date"), "test.csv")
    _require_unique_keys(oil, ("date",), "oil.csv")
    _require_unique_keys(transactions, ("date", "store_nbr"), "transactions.csv")
    _require_unique_keys(sample, ("id",), "sample_submission.csv")

    negative = int((train["sales"] < 0).sum())
    if negative:
        raise ValueError(f"negative sales: {negative} rows — stop and decide policy")

    train_dates = pd.DatetimeIndex(train["date"].unique()).sort_values()
    test_dates = pd.DatetimeIndex(test["date"].unique()).sort_values()
    overlap = train_dates.intersection(test_dates)
    if len(overlap):
        raise ValueError(f"test dates overlap train dates: {len(overlap)} dates")
    last_train_date = train_dates.max()
    official_origin = test_dates.min()
    mirror_cutoff = last_train_date - to_offset(_FREQ) * (horizon - 1)

    first_dates = _series_first_dates(train, test)
    union_dates = pd.date_range(first_dates["first_date"].min(), test_dates.max(), freq=_FREQ)
    long = _union_grid(first_dates, union_dates)
    long = long.merge(
        train[["store_nbr", "family", "date", "sales", "onpromotion"]],
        on=["store_nbr", "family", "date"],
        how="left",
        validate="one_to_one",
    )
    long = long.merge(
        test[["store_nbr", "family", "date", "onpromotion"]].rename(
            columns={"onpromotion": "test_promotion"}
        ),
        on=["store_nbr", "family", "date"],
        how="left",
        validate="one_to_one",
    )
    long["onpromotion"] = long["onpromotion"].fillna(long["test_promotion"])
    long = long.drop(columns="test_promotion")

    in_history = long["date"] <= last_train_date
    missing_history = in_history & long["sales"].isna()
    zero_filled = int(missing_history.sum())
    long["sales"] = long["sales"].mask(missing_history, 0.0)

    national = holidays.loc[
        (holidays["locale"] == "National")
        & holidays["type"].isin(_NATIONAL_TYPES)
        & holidays["transferred"].eq(False)
    ]
    long["is_national_holiday"] = (
        long["date"].isin(pd.DatetimeIndex(national["date"].unique())).astype("int64")
    )
    long = long.merge(
        oil[["date", "dcoilwtico"]].rename(columns={"dcoilwtico": "oil_price"}),
        on="date",
        how="left",
        validate="many_to_one",
    )
    long = long.merge(
        transactions[["date", "store_nbr", "transactions"]],
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
    )

    long = _sales_covariates(long)
    long[_TARGET] = np.log1p(long["sales"])
    for name, column in _payday_flags(long["date"]).items():
        long[name] = column
    long = long[_LONG_COLUMNS].reset_index(drop=True)

    official_dir = out_dir / "official"
    mirror_dir = out_dir / "mirror"
    official_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    official = _clip_at_origin(long, official_origin, mask_target=False)
    mirror = _clip_at_origin(long, mirror_cutoff, mask_target=True)
    truth = (
        long.loc[
            (long["date"] >= mirror_cutoff) & long[_TARGET].notna(),
            ["store_nbr", "family", "date", _TARGET],
        ]
        .sort_values(["store_nbr", "family", "date"], kind="stable")
        .reset_index(drop=True)
    )
    static = _static_table(first_dates, stores)

    official.to_parquet(official_dir / "long.parquet", index=False)
    mirror.to_parquet(mirror_dir / "long.parquet", index=False)
    truth.to_parquet(mirror_dir / "truth.parquet", index=False)
    static.to_parquet(official_dir / "static.parquet", index=False)
    static.to_parquet(mirror_dir / "static.parquet", index=False)

    test_ids = sample[["id"]].merge(
        test[["id", "date", "store_nbr", "family"]], on="id", how="left", validate="one_to_one"
    )
    if len(test_ids) != len(test) or test_ids[["date", "store_nbr", "family"]].isna().any().any():
        raise ValueError("sample submission ids do not match test.csv ids")
    test_ids.to_parquet(official_dir / "test_ids.parquet", index=False)
    shutil.copyfile(raw_dir / "sample_submission.csv", official_dir / "sample_submission.csv")

    report = PrepareReport(
        slug=SLUG,
        series=len(first_dates),
        train_days=len(train_dates),
        test_days=len(test_dates),
        mirror_origin=mirror_cutoff.date().isoformat(),
        official_origin=official_origin.date().isoformat(),
        quirks={
            "missing_target": "zero",
            "negative_sales": negative,
            "zero_filled_rows": zero_filled,
            "target_transform": "log1p",
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


def score_mirror(
    prepared_mirror_dir: Path, prediction_frames: dict[str, pd.DataFrame]
) -> dict[str, float]:
    """Score the store-sales mirror window in log space (RMSLE-equivalent).

    The prepared mirror tree stores its labels as ``log1p_sales`` while the
    generic scorer contracts on a raw ``sales`` label column, so this wrapper
    stages a throwaway mirror directory — the prepared settings and report
    plus the truth table with its label column renamed — and scores it with
    the plain ``rmse`` metric. The settings carry the prepared panel's
    absolute path, so the original ``long.parquet`` is read in place, never
    copied or modified. Log-space RMSE is exactly RMSLE on raw sales, keeping
    new mirror scores comparable with the historical raw-target runs; the
    generic scorer stays untouched.
    """
    mirror = Path(prepared_mirror_dir)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "mirror"
        staged.mkdir()
        for name in ("settings-finetune.yaml", "prepare.json"):
            shutil.copyfile(mirror / name, staged / name)
        truth = pd.read_parquet(mirror / "truth.parquet")
        if _TARGET not in truth.columns:
            raise ValueError(f"truth.parquet: missing label column {_TARGET!r}")
        truth.rename(columns={_TARGET: "sales"}).to_parquet(staged / "truth.parquet", index=False)
        return scoring.score_mirror(staged, prediction_frames, metric_name="rmse")
