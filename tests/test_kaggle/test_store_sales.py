"""Store-sales converter tests on a synthetic mini raw fixture."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand_forecast.config import Settings
from demand_forecast.kaggle import store_sales
from demand_forecast.kaggle.store_sales import SLUG, PrepareReport, _payday_flags, prepare

# The dev-only CI lap installs no parquet engine: skip this module, never fail it.
pytest.importorskip("pyarrow")


def _mini_raw(root: Path, *, negative_sale: bool = False) -> None:
    """Write a miniature stand-in for the raw competition files.

    Layout: 2 stores x 2 families over 2017-07-20..2017-08-03 (15 train days)
    plus 2017-08-04..05 (2 test days). Store 2 / AUTOMOTIVE is missing the
    2017-07-25 row (deliberate gap); store 1 / BABY CARE sells zero every day.
    Oil covers the test period with two dates absent; transactions stop at the
    last train day; the sample-submission ids are deliberately shuffled.
    Holiday rows cover every branch of the pinned national filter.
    ``negative_sale=True`` flips one train value to -1.0.
    """
    root.mkdir(parents=True, exist_ok=True)
    train_dates = pd.date_range("2017-07-20", "2017-08-03", freq="D")
    test_dates = pd.date_range("2017-08-04", "2017-08-05", freq="D")
    families = ("AUTOMOTIVE", "BABY CARE")

    train_rows = []
    for store_nbr in (1, 2):
        for family in families:
            for date in train_dates:
                if (store_nbr, family, date) == (2, "AUTOMOTIVE", pd.Timestamp("2017-07-25")):
                    continue  # deliberate gap row
                sales = 0.0 if family == "BABY CARE" else float(store_nbr * 100 + date.day)
                train_rows.append(
                    {
                        "date": date,
                        "store_nbr": store_nbr,
                        "family": family,
                        "sales": sales,
                        "onpromotion": int(family == "AUTOMOTIVE" and date.day % 2 == 0),
                    }
                )
    train = pd.DataFrame(train_rows).sort_values(["store_nbr", "family", "date"])
    train["id"] = range(len(train))
    if negative_sale:
        train.loc[train.index[0], "sales"] = -1.0
    train[["id", "date", "store_nbr", "family", "sales", "onpromotion"]].to_csv(
        root / "train.csv", index=False
    )

    test_rows = []
    for store_nbr in (1, 2):
        for family in families:
            for date in test_dates:
                test_rows.append(
                    {
                        "date": date,
                        "store_nbr": store_nbr,
                        "family": family,
                        "onpromotion": int(date.day == 5),
                    }
                )
    test = pd.DataFrame(test_rows)
    test["id"] = range(900, 900 + len(test))
    test[["id", "date", "store_nbr", "family", "onpromotion"]].to_csv(
        root / "test.csv", index=False
    )

    sample_ids = [903, 900, 907, 901, 906, 902, 905, 904]
    pd.DataFrame({"id": sample_ids, "sales": 0.0}).to_csv(
        root / "sample_submission.csv", index=False
    )

    pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "B"],
            "cluster": [3, 7],
        }
    ).to_csv(root / "stores.csv", index=False)

    oil_dates = pd.date_range("2017-07-20", "2017-08-05", freq="D").difference(
        pd.DatetimeIndex(["2017-07-22", "2017-07-23"])
    )
    pd.DataFrame(
        {"date": oil_dates, "dcoilwtico": [40.0 + index * 0.5 for index in range(len(oil_dates))]}
    ).to_csv(root / "oil.csv", index=False)

    transaction_dates = pd.date_range("2017-07-20", "2017-08-03", freq="D")
    pd.DataFrame(
        [
            {"date": date, "store_nbr": store_nbr, "transactions": 1000 + store_nbr}
            for date in transaction_dates
            for store_nbr in (1, 2)
        ]
    ).to_csv(root / "transactions.csv", index=False)

    holidays = pd.DataFrame(
        [
            ("2017-07-24", "Holiday", "National", "Ecuador", "National holiday", False),
            ("2017-07-25", "Transfer", "National", "Ecuador", "Transferred holiday date", False),
            ("2017-07-26", "Bridge", "National", "Ecuador", "Bridge day", False),
            ("2017-07-27", "Additional", "National", "Ecuador", "Additional holiday", False),
            ("2017-07-28", "Holiday", "Local", "Quito", "Local holiday", False),
            ("2017-07-29", "Holiday", "National", "Ecuador", "Moved holiday", True),
            ("2017-07-30", "Work Day", "National", "Ecuador", "Work day", False),
            ("2017-08-04", "Holiday", "National", "Ecuador", "Test-period holiday", False),
        ],
        columns=["date", "type", "locale", "locale_name", "description", "transferred"],
    )
    holidays["date"] = pd.to_datetime(holidays["date"])
    holidays.to_csv(root / "holidays_events.csv", index=False)


def test_official_test_targets_all_nan(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    report = prepare(raw, tmp_path / "out", horizon=2)
    long = pd.read_parquet(tmp_path / "out/official/long.parquet")
    test_rows = long[long.date >= report.official_origin]
    assert test_rows.log1p_sales.isna().all()
    assert report.test_days == 2
    assert "sales" not in long.columns  # the raw target is transformed away, not kept


def test_target_is_log1p_of_raw_sales(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    official = pd.read_parquet(tmp_path / "out/official/long.parquet")

    auto = official[(official.store_nbr == 1) & (official.family == "AUTOMOTIVE")]
    by_date = auto.set_index("date")["log1p_sales"]
    assert by_date.loc["2017-07-20"] == pytest.approx(np.log1p(120.0))
    assert by_date.loc["2017-08-03"] == pytest.approx(np.log1p(103.0))
    baby = official[(official.store_nbr == 1) & (official.family == "BABY CARE")]
    history = baby[baby.date <= "2017-08-03"]
    assert (history.log1p_sales == 0.0).all()  # log1p(0) == 0: zero-sales stays observed


def test_mirror_no_targets_after_cutoff(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    long = pd.read_parquet(tmp_path / "out/mirror/long.parquet")
    assert long[long.date >= "2017-08-02"].log1p_sales.isna().all()
    truth = pd.read_parquet(tmp_path / "out/mirror/truth.parquet")
    assert list(truth.columns) == ["store_nbr", "family", "date", "log1p_sales"]
    assert (truth.date >= "2017-08-02").all()
    assert truth.log1p_sales.notna().all()  # labels are log-space numbers, never NaN
    labels = truth.set_index(["store_nbr", "family", "date"])["log1p_sales"]
    assert labels.loc[(1, "AUTOMOTIVE", pd.Timestamp("2017-08-02"))] == pytest.approx(
        np.log1p(102.0)
    )
    assert labels.loc[(2, "AUTOMOTIVE", pd.Timestamp("2017-08-03"))] == pytest.approx(
        np.log1p(203.0)
    )


def test_zero_only_series_kept(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)  # includes one series whose sales are all 0/absent
    report = prepare(raw, tmp_path / "out", horizon=2)
    assert report.series == 4


def test_negative_sales_stop_conversion(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw, negative_sale=True)
    with pytest.raises(ValueError, match="negative sales"):
        prepare(raw, tmp_path / "out", horizon=2)


def test_static_table_replicates_store_metadata(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    static = pd.read_parquet(tmp_path / "out/official/static.parquet")
    assert set(static.columns) == {"store_nbr", "family", "city", "state", "type", "cluster"}
    assert len(static) == static[["store_nbr", "family"]].drop_duplicates().shape[0]


def test_is_national_holiday_filter(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    official = pd.read_parquet(tmp_path / "out/official/long.parquet")
    by_date = official.drop_duplicates(subset="date").set_index("date")

    def flag(date: str) -> int:
        return int(by_date.loc[pd.Timestamp(date), "is_national_holiday"])

    assert [flag(day) for day in ("2017-07-24", "2017-07-25", "2017-07-26", "2017-07-27")] == [
        1,
        1,
        1,
        1,
    ]
    # 07-28 local, 07-29 transferred, 07-30 work day, 07-31 no event
    assert [flag(day) for day in ("2017-07-28", "2017-07-29", "2017-07-30", "2017-07-31")] == [
        0,
        0,
        0,
        0,
    ]
    assert flag("2017-08-04") == 1  # test-period national holiday


def test_payday_flags_mark_the_15th_and_month_end() -> None:
    dates = pd.Series(pd.date_range("2017-08-13", "2017-09-02", freq="D"))

    flags = _payday_flags(dates)

    assert dates[flags["is_day_15"].eq(1)].tolist() == [pd.Timestamp("2017-08-15")]
    assert dates[flags["is_month_end"].eq(1)].tolist() == [pd.Timestamp("2017-08-31")]
    assert set(flags) == {"is_day_15", "is_month_end"}


def test_window_covariates_follow_pinned_availability(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    official = pd.read_parquet(tmp_path / "out/official/long.parquet")
    mirror = pd.read_parquet(tmp_path / "out/mirror/long.parquet")

    # Past dynamics — transactions and the raw-sales lag/rolling block — stop
    # at each variant's origin, so the loaded history ends exactly at origin - 1.
    past = ["transactions", "sales_lag7", "sales_lag28", "sales_roll7", "sales_roll28"]
    for column in past:
        assert official.loc[official.date >= "2017-08-04", column].isna().all()
        assert mirror.loc[mirror.date >= "2017-08-02", column].isna().all()

    # oil_price is a known covariate now: filled for every date the raw file
    # covers — both windows included — and NaN exactly where the file is absent.
    all_store = official[(official.store_nbr == 1) & (official.family == "AUTOMOTIVE")]
    raw_oil = pd.read_csv(raw / "oil.csv", parse_dates=["date"]).set_index("date")["dcoilwtico"]
    for day in ("2017-08-01", "2017-08-04", "2017-08-05"):
        window_rows = all_store[all_store.date == day]
        assert window_rows.oil_price.tolist() == pytest.approx([raw_oil.loc[day]])
    assert official[official.date >= "2017-08-04"].oil_price.notna().all()
    assert mirror[mirror.date >= "2017-08-02"].oil_price.notna().all()
    assert all_store[all_store.date == "2017-07-22"].oil_price.isna().all()  # market hole

    # transactions keep their observed values on the last history day.
    assert all_store[all_store.date == "2017-08-03"].transactions.tolist() == [1001.0]

    # Known covariates fill the mirror window and the official test window.
    for frame, cutoff in ((mirror, "2017-08-02"), (official, "2017-08-04")):
        window = frame[frame.date >= cutoff]
        for column in (
            "onpromotion",
            "is_national_holiday",
            "is_day_15",
            "is_month_end",
            "oil_price",
        ):
            assert window[column].notna().all(), column


def test_sales_lag_and_rolling_block_semantics(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    official = pd.read_parquet(tmp_path / "out/official/long.parquet")

    def series(store_nbr: int, family: str) -> pd.DataFrame:
        rows = official[(official.store_nbr == store_nbr) & (official.family == family)]
        return rows.set_index("date").sort_index()

    auto = series(1, "AUTOMOTIVE")
    # Lags are strict: the value at t is raw sales at t-7 / t-28, NaN before start.
    assert auto.loc["2017-07-27", "sales_lag7"] == pytest.approx(120.0)
    assert auto.loc["2017-08-03", "sales_lag7"] == pytest.approx(127.0)
    assert pd.isna(auto.loc["2017-07-26", "sales_lag7"])
    assert auto["sales_lag28"].isna().all()  # the 15-day fixture is shorter than 28 days

    # Rolls are trailing means of the one-day-shifted series with min_periods=1.
    assert pd.isna(auto.loc["2017-07-20", "sales_roll7"])
    assert auto.loc["2017-07-21", "sales_roll7"] == pytest.approx(120.0)
    assert auto.loc["2017-07-26", "sales_roll7"] == pytest.approx(122.5)
    assert auto.loc["2017-07-27", "sales_roll7"] == pytest.approx(123.0)
    # 08-03 rolls over every available prior day: July sales run 120..131 and
    # August resets the day-of-month formula, so August 1-2 sell 101 and 102.
    available = [120.0 + offset for offset in range(12)] + [101.0, 102.0]
    assert auto.loc["2017-08-03", "sales_roll28"] == pytest.approx(sum(available) / 14)

    # The zero-filled gap row participates in lags and rolls like the target:
    # store 2 / AUTOMOTIVE has no 2017-07-25 row, so its raw sales read 0.0 there.
    glove = series(2, "AUTOMOTIVE")
    assert glove.loc["2017-08-01", "sales_lag7"] == 0.0
    assert glove.loc["2017-07-26", "sales_roll7"] == pytest.approx(185.0)


def test_union_grid_is_dense_and_zero_fills_missing_rows(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    official = pd.read_parquet(tmp_path / "out/official/long.parquet")

    assert list(official.columns) == [
        "store_nbr",
        "family",
        "date",
        "log1p_sales",
        "onpromotion",
        "is_national_holiday",
        "is_day_15",
        "is_month_end",
        "oil_price",
        "transactions",
        "sales_lag7",
        "sales_lag28",
        "sales_roll7",
        "sales_roll28",
    ]
    assert len(official) == 4 * 17  # 4 series x (15 train + 2 test) union-window days
    assert not official.duplicated(subset=["store_nbr", "family", "date"]).any()

    gap = official[
        (official.store_nbr == 2)
        & (official.family == "AUTOMOTIVE")
        & (official.date == "2017-07-25")
    ]
    assert gap.log1p_sales.tolist() == [0.0]  # missing sale row -> zero-target policy
    assert gap.onpromotion.isna().all()  # unknown promotion stays unknown
    assert gap.sales_lag7.isna().all()  # 2017-07-18 precedes the series start

    month_end = official[(official.date == "2017-07-31")]
    assert month_end.is_month_end.tolist() == [1] * 4
    assert month_end.is_day_15.tolist() == [0] * 4


def test_settings_files_compose_for_both_windows(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    for window in ("official", "mirror"):
        settings = Settings.from_yaml(tmp_path / f"out/{window}/settings-finetune.yaml")
        assert settings.forecast.horizon == 2
        assert settings.data.path == (tmp_path / f"out/{window}/long.parquet").resolve()
        assert settings.data.static_path == (tmp_path / f"out/{window}/static.parquet").resolve()
        assert settings.data.id_columns == ["store_nbr", "family"]
        assert settings.data.timestamp_column == "date"
        assert settings.data.target == "log1p_sales"
        assert settings.data.target_support == "nonnegative"
        assert settings.data.known_dynamic == [
            "onpromotion",
            "is_national_holiday",
            "is_day_15",
            "is_month_end",
            "oil_price",
        ]
        assert settings.data.past_dynamic == [
            "transactions",
            "sales_lag7",
            "sales_lag28",
            "sales_roll7",
            "sales_roll28",
        ]
        assert settings.data.static_categorical == [
            "store_nbr",
            "family",
            "city",
            "state",
            "type",
            "cluster",
        ]
        assert settings.data.freq == "D"
        assert settings.data.missing_target == "zero"
        assert settings.data.duplicates == "error"
        assert settings.training.refit == "full"
    zero_shot = Settings.from_yaml(tmp_path / "out/official/settings-zero-shot.yaml")
    assert zero_shot.training.zero_shot is True


def test_test_ids_follow_sample_order_and_report_round_trips(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    report = prepare(raw, tmp_path / "out", horizon=2)

    official_dir = tmp_path / "out/official"
    mirror_dir = tmp_path / "out/mirror"
    assert {path.name for path in official_dir.iterdir()} == {
        "long.parquet",
        "static.parquet",
        "settings-finetune.yaml",
        "settings-zero-shot.yaml",
        "test_ids.parquet",
        "sample_submission.csv",
        "prepare.json",
    }
    assert {path.name for path in mirror_dir.iterdir()} == {
        "long.parquet",
        "static.parquet",
        "settings-finetune.yaml",
        "settings-zero-shot.yaml",
        "truth.parquet",
        "prepare.json",
    }

    test_ids = pd.read_parquet(official_dir / "test_ids.parquet")
    sample = pd.read_csv(official_dir / "sample_submission.csv")
    assert list(test_ids.columns) == ["id", "date", "store_nbr", "family"]
    assert test_ids.id.tolist() == sample.id.tolist()  # submission order follows the sample
    assert sorted(test_ids.id.tolist()) == list(range(900, 908))
    assert (official_dir / "sample_submission.csv").read_bytes() == (
        raw / "sample_submission.csv"
    ).read_bytes()

    assert isinstance(report, PrepareReport)
    data = json.loads((official_dir / "prepare.json").read_text(encoding="utf-8"))
    assert data["slug"] == SLUG == "store-sales-time-series-forecasting"
    assert data["series"] == report.series == 4
    assert data["train_days"] == report.train_days == 15
    assert data["test_days"] == report.test_days == 2
    assert data["official_origin"] == report.official_origin == "2017-08-04"
    assert data["mirror_origin"] == report.mirror_origin == "2017-08-02"
    assert data["quirks"] == {
        "missing_target": "zero",
        "negative_sales": 0,
        "zero_filled_rows": 1,
        "target_transform": "log1p",
    }
    mirrored = json.loads((mirror_dir / "prepare.json").read_text(encoding="utf-8"))
    assert mirrored == data


def _truth_predictions(truth: pd.DataFrame) -> pd.DataFrame:
    """Log-space prediction rows matching the mirror truth row set."""
    return pd.DataFrame(
        {
            "series_id": truth["store_nbr"].astype(str) + "|" + truth["family"],
            "timestamp": truth["date"],
            "point": truth["log1p_sales"],
        }
    )


def test_store_sales_score_mirror_scores_log_space_truth(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    mirror = tmp_path / "out/mirror"
    truth = pd.read_parquet(mirror / "truth.parquet")

    scores = store_sales.score_mirror(mirror, {"finetune": _truth_predictions(truth)})

    assert set(scores) == {"baseline", "finetune"}
    assert scores["finetune"] == 0.0  # identical log-space points: rmse hits zero
    assert np.isfinite(scores["baseline"])
    # Scoring stages a private renamed copy; the prepared tree keeps its label.
    assert "log1p_sales" in pd.read_parquet(mirror / "truth.parquet").columns


def test_store_sales_score_mirror_uses_plain_rmse(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=2)
    mirror = tmp_path / "out/mirror"
    truth = pd.read_parquet(mirror / "truth.parquet")
    shifted = _truth_predictions(truth).assign(point=lambda frame: frame["point"] + 0.5)

    scores = store_sales.score_mirror(mirror, {"finetune": shifted})

    assert scores["finetune"] == pytest.approx(0.5)  # rmse of a constant log-space offset
