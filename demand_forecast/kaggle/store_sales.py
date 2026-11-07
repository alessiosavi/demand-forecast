"""Convert the raw Kaggle store-sales files into prepared long/static tables.

One module per competition (the design's hybrid runner): ``prepare`` reads the
raw CSVs, verifies their pinned column contracts, and writes two prepared
variants under ``out_dir``:

- ``official`` — every observed ``(store_nbr, family)`` series on its daily
  grid through the last test day, with ``sales`` NaN from the test period on.
- ``mirror`` — the same panel with ``sales`` NaN from the mirror cutoff (the
  first of the last ``horizon`` train days) on; those cut targets are split
  out to ``truth.parquet`` and never enter the panel's targets.

Covariate availability is pinned per variant: the known covariates
(``onpromotion``, ``is_national_holiday``) extend into the forecast window,
from ``test.csv`` and the holiday calendar respectively, while the
past-dynamic covariates (``oil_price``, ``transactions``) stop at the
variant's origin. Past-dynamic values at or after the origin would extend
each series' loaded history past the origin, and the trainer would then
publish a bundle whose earliest permissible origin outruns the window origin
the kernel workload forecasts at. Missing sale rows inside the history carry
the pinned ``missing_target == "zero"`` policy (the competition convention);
official test rows carry NaN targets.
"""

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

from demand_forecast.config import DatasetConfig
from demand_forecast.data.frequency import to_offset
from demand_forecast.kaggle.settings import build_settings, load_template, write_settings

__all__ = ["PrepareReport", "SLUG", "prepare"]

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

_KNOWN_DYNAMIC = ["onpromotion", "is_national_holiday"]
_PAST_DYNAMIC = ["oil_price", "transactions"]
_STATIC_CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]
_FREQ = "D"
_LONG_COLUMNS = ["store_nbr", "family", "date", "sales", *_KNOWN_DYNAMIC, *_PAST_DYNAMIC]
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


def _clip_at_origin(
    frame: pd.DataFrame, origin: pd.Timestamp, *, mask_target: bool
) -> pd.DataFrame:
    """Drop past-dynamic values (and optionally the target) from ``origin`` on."""
    clipped = frame.copy()
    after = clipped["date"] >= origin
    for column in _PAST_DYNAMIC:
        clipped[column] = clipped[column].mask(after)
    if mask_target:
        clipped["sales"] = clipped["sales"].mask(after)
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
        target="sales",
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
    long = (
        long[_LONG_COLUMNS]
        .sort_values(["store_nbr", "family", "date"], kind="stable")
        .reset_index(drop=True)
    )

    official_dir = out_dir / "official"
    mirror_dir = out_dir / "mirror"
    official_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    official = _clip_at_origin(long, official_origin, mask_target=False)
    mirror = _clip_at_origin(long, mirror_cutoff, mask_target=True)
    truth = (
        long.loc[
            (long["date"] >= mirror_cutoff) & long["sales"].notna(),
            ["store_nbr", "family", "date", "sales"],
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
