"""mlcspp converter/scorer/submission tests on a synthetic mini raw fixture."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from demand_forecast.config import Settings
from demand_forecast.kaggle.mlcspp import (
    SLUG,
    build_submission,
    prepare,
    score_mirror,
    validate_submission,
)

# The dev-only CI lap installs no parquet engine: skip this module, never fail it.
pytest.importorskip("pyarrow")

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
_RAW_DATES = (101, 102, 104, 105, 106, 108, 109, 110, 112, 113, 114, 116)
_SORTED_DATES = sorted(_RAW_DATES)
_HORIZON = 4
_OBSERVED = len(_RAW_DATES) - _HORIZON  # 8
_EPOCH = pd.Timestamp("2000-01-01")
_CALENDAR = pd.date_range(_EPOCH, periods=len(_RAW_DATES), freq="D")
_MIRROR_DATES = _CALENDAR[_OBSERVED - _HORIZON : _OBSERVED]
_TRUTH_CLOSES = [300.0 + rank for rank in range(_OBSERVED - _HORIZON, _OBSERVED)]
_BASE_ORDER = (105, 116, 101, 110, 104, 113, 108, 102, 114, 109, 106, 112)


def _file_order(offset: int) -> list[int]:
    """A deterministic per-file permutation of ``_RAW_DATES`` (never sorted order)."""
    offset %= len(_BASE_ORDER)
    return list(_BASE_ORDER[offset:] + _BASE_ORDER[:offset])


def _mini_raw(root: Path, *, drop_row: tuple[str, int] | None = None) -> None:
    """Write a miniature stand-in for the raw mlcspp files.

    The twelve ``_RAW_DATES`` are permuted independently per file, so alignment
    must sort on ``date``; ``0056.csv`` is missing its close on exactly the
    final ``_HORIZON`` sorted dates. Constituent values are
    ``100 + 10 * stock + field + rank``, checkable arithmetically after the
    lag-1 shift. ``drop_row`` removes one row from the named file to exercise
    the date-set guard.
    """
    root.mkdir(parents=True, exist_ok=True)
    rank_of = {date: rank for rank, date in enumerate(_SORTED_DATES)}

    order = _file_order(0)
    target = pd.DataFrame(
        {
            "date": order,
            "0056_close_y": [
                300.0 + rank_of[date] if rank_of[date] < _OBSERVED else float("nan")
                for date in order
            ],
        }
    )
    if drop_row is not None and drop_row[0] == "0056.csv":
        target = target.loc[target["date"] != drop_row[1]]
    target.to_csv(root / "0056.csv", index=False)

    for stock_index, stock in enumerate(_STOCKS):
        order = _file_order(stock_index + 1)
        data = {"date": order}
        for field_index, field in enumerate(_FIELDS):
            data[f"{stock}_{field}"] = [
                100.0 + 10.0 * stock_index + field_index + rank_of[date] for date in order
            ]
        frame = pd.DataFrame(data)
        if drop_row is not None and drop_row[0] == f"{stock}.csv":
            frame = frame.loc[frame["date"] != drop_row[1]]
        frame.to_csv(root / f"{stock}.csv", index=False)

    pd.DataFrame(
        {
            "date": list(reversed(_SORTED_DATES[-_HORIZON:])),
            "0056_close_y": [0.0] * _HORIZON,
        }
    ).to_csv(root / "sample_submission.csv", index=False)


def _prepare_mini(tmp_path: Path) -> tuple[Path, Path]:
    """Prepare the mini fixture; return the official and mirror variant dirs."""
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=_HORIZON)
    return tmp_path / "out" / "official", tmp_path / "out" / "mirror"


def _predictions(dates, values) -> pd.DataFrame:
    """A loader-shaped prediction frame: ``series_id``, ``timestamp``, ``point``."""
    points = [float(value) for value in values]
    return pd.DataFrame(
        {
            "series_id": "0056",
            "timestamp": dates,
            "point": points,
            "q0.5": points,
        }
    )


# --------------------------------------------------------------------------- #
# prepare: alignment, dense index, lag, variants, artifacts
# --------------------------------------------------------------------------- #


def test_shuffled_inputs_align_on_sorted_dates(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=_HORIZON)

    assert pd.read_csv(raw / "0056.csv")["date"].tolist() != _SORTED_DATES  # fixture is shuffled
    official = pd.read_parquet(tmp_path / "out/official/long.parquet")
    assert official["date"].tolist() == _CALENDAR.tolist()
    closes = official["0056_close_y"]
    assert closes.iloc[:_OBSERVED].tolist() == [300.0 + rank for rank in range(_OBSERVED)]
    assert closes.iloc[_OBSERVED:].isna().all()  # NaN only in the missing block


def test_dense_day_index_and_date_map(tmp_path):
    official_dir, _ = _prepare_mini(tmp_path)

    date_map = pd.read_parquet(official_dir / "date_map.parquet")
    assert list(date_map.columns) == ["date_int", "date"]
    assert date_map["date_int"].dtype == "int64"
    assert date_map["date_int"].tolist() == _SORTED_DATES
    assert date_map["date"].tolist() == _CALENDAR.tolist()


def test_covariates_are_lagged_by_one_day(tmp_path):
    official_dir, _ = _prepare_mini(tmp_path)
    official = pd.read_parquet(official_dir / "long.parquet")

    expected_columns = ["ticker", "date", "0056_close_y"] + [
        f"{stock}_{field}" for stock in _STOCKS for field in _FIELDS
    ]
    assert list(official.columns) == expected_columns
    assert official["ticker"].tolist() == ["0056"] * len(_RAW_DATES)

    lagged = official["1101_close"]
    assert pd.isna(lagged.iloc[0])  # rank 0 has no yesterday
    assert lagged.iloc[1] == pytest.approx(102.0)  # 100 + 10*0 + 2 + raw rank 0
    assert lagged.iloc[-1] == pytest.approx(112.0)  # ... + raw rank 10
    assert official["9945_ups_percent"].iloc[5] == pytest.approx(202.0)  # 100 + 90 + 8 + rank 4

    raw = pd.read_csv(tmp_path / "raw" / "1101.csv").set_index("date")
    expected = raw["1101_close"].reindex(_SORTED_DATES).shift(1)
    assert lagged.tolist()[1:] == expected.tolist()[1:]


def test_mirror_cut_truth_and_panel(tmp_path):
    _, mirror_dir = _prepare_mini(tmp_path)
    cutoff_rank = _OBSERVED - _HORIZON

    mirror = pd.read_parquet(mirror_dir / "long.parquet")
    assert len(mirror) == len(_RAW_DATES)
    target = mirror["0056_close_y"]
    assert target.iloc[:cutoff_rank].tolist() == [300.0 + rank for rank in range(cutoff_rank)]
    assert target.iloc[cutoff_rank:].isna().all()

    truth = pd.read_parquet(mirror_dir / "truth.parquet")
    assert list(truth.columns) == ["series_id", "date", "close"]
    assert truth["series_id"].tolist() == ["0056"] * _HORIZON
    assert truth["date"].tolist() == _MIRROR_DATES.tolist()
    assert truth["close"].tolist() == _TRUTH_CLOSES


def test_prepare_report_counts_and_quirks(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    report = prepare(raw, tmp_path / "out", horizon=_HORIZON)

    assert report.slug == SLUG == "mlcspp"
    assert report.series == 1
    assert report.train_days == _OBSERVED == 8
    assert report.test_days == _HORIZON == 4
    assert report.mirror_origin == "2000-01-05"
    assert report.official_origin == "2000-01-09"
    assert report.quirks == {
        "covariate_lag_days": 1,
        "dense_day_index": True,
        "missing_block": "contiguous_final",
    }

    official_json = json.loads((tmp_path / "out/official/prepare.json").read_text(encoding="utf-8"))
    assert official_json == asdict(report)
    mirrored = json.loads((tmp_path / "out/mirror/prepare.json").read_text(encoding="utf-8"))
    assert mirrored == official_json


def test_artifacts_and_settings_composition(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw)
    prepare(raw, tmp_path / "out", horizon=_HORIZON)
    official_dir = tmp_path / "out/official"
    mirror_dir = tmp_path / "out/mirror"

    assert {path.name for path in official_dir.iterdir()} == {
        "long.parquet",
        "date_map.parquet",
        "settings-finetune.yaml",
        "settings-zero-shot.yaml",
        "sample_submission.csv",
        "prepare.json",
    }
    assert {path.name for path in mirror_dir.iterdir()} == {
        "long.parquet",
        "settings-finetune.yaml",
        "settings-zero-shot.yaml",
        "truth.parquet",
        "prepare.json",
    }
    assert (official_dir / "sample_submission.csv").read_bytes() == (
        raw / "sample_submission.csv"
    ).read_bytes()

    expected_dynamic = [f"{stock}_{field}" for stock in _STOCKS for field in _FIELDS]
    for directory in (official_dir, mirror_dir):
        settings = Settings.from_yaml(directory / "settings-finetune.yaml")
        assert settings.forecast.horizon == _HORIZON
        assert settings.data.path == (directory / "long.parquet").resolve()
        assert settings.data.id_columns == ["ticker"]
        assert settings.data.timestamp_column == "date"
        assert settings.data.target == "0056_close_y"
        assert settings.data.target_support == "nonnegative"
        assert settings.data.known_dynamic == expected_dynamic
        assert settings.data.past_dynamic == []
        assert settings.data.static_path is None
        assert settings.data.static_categorical == []
        assert settings.data.static_numeric == []
        assert settings.data.freq == "D"
        assert settings.data.missing_target == "nan"
        assert settings.data.duplicates == "error"
    zero_shot = Settings.from_yaml(official_dir / "settings-zero-shot.yaml")
    assert zero_shot.training.zero_shot is True
    assert zero_shot.forecast.horizon == _HORIZON


def test_date_set_deviation_raises(tmp_path):
    raw = tmp_path / "raw"
    _mini_raw(raw, drop_row=("1101.csv", 104))
    with pytest.raises(ValueError, match=r"1101\.csv: date set deviates"):
        prepare(raw, tmp_path / "out", horizon=_HORIZON)


# --------------------------------------------------------------------------- #
# score_mirror: shared row set, baseline, loud point validation
# --------------------------------------------------------------------------- #


def test_score_mirror_scores_baseline_and_every_frame(tmp_path):
    _, mirror_dir = _prepare_mini(tmp_path)

    exact = _predictions(_MIRROR_DATES, _TRUTH_CLOSES).iloc[[3, 1, 0, 2]].reset_index(drop=True)
    offset = _predictions(_MIRROR_DATES, [value + 1.0 for value in _TRUTH_CLOSES])
    scores = score_mirror(mirror_dir, {"finetune": exact, "zero_shot": offset})

    assert list(scores) == ["baseline", "finetune", "zero_shot"]
    assert scores["finetune"] == 0.0
    assert scores["zero_shot"] == pytest.approx(1.0)
    # Seasonal-naive: last observed close (303.0 at rank 3) for steps 4-6, and
    # the one-season-back value at rank 0 (300.0) for step 7.
    assert scores["baseline"] == pytest.approx(math.sqrt(63.0 / 4.0))
    assert all(math.isfinite(value) for value in scores.values())


def test_score_mirror_row_set_must_match_truth(tmp_path):
    _, mirror_dir = _prepare_mini(tmp_path)

    short = _predictions(_MIRROR_DATES[:3], _TRUTH_CLOSES[:3])
    with pytest.raises(ValueError, match="row set differs from truth"):
        score_mirror(mirror_dir, {"finetune": short})

    extra_dates = _MIRROR_DATES.append(pd.DatetimeIndex([_CALENDAR[_OBSERVED]]))
    extra = _predictions(extra_dates, [*_TRUTH_CLOSES, 400.0])
    with pytest.raises(ValueError, match="row set differs from truth"):
        score_mirror(mirror_dir, {"finetune": extra})


def test_score_mirror_rejects_invalid_points(tmp_path):
    _, mirror_dir = _prepare_mini(tmp_path)

    non_finite = _predictions(_MIRROR_DATES, [*_TRUTH_CLOSES[:3], float("nan")])
    with pytest.raises(ValueError, match="non-finite point values"):
        score_mirror(mirror_dir, {"finetune": non_finite})

    negative = _predictions(_MIRROR_DATES, [*_TRUTH_CLOSES[:3], -1.0])
    with pytest.raises(ValueError, match="negative point values"):
        score_mirror(mirror_dir, {"finetune": negative})


def test_score_mirror_rejects_reserved_key_and_unknown_metric(tmp_path):
    _, mirror_dir = _prepare_mini(tmp_path)

    with pytest.raises(ValueError, match="reserved 'baseline'"):
        score_mirror(mirror_dir, {"baseline": _predictions(_MIRROR_DATES, _TRUTH_CLOSES)})
    with pytest.raises(ValueError, match="unknown metric"):
        score_mirror(mirror_dir, {}, metric_name="mape")


# --------------------------------------------------------------------------- #
# build_submission / validate_submission
# --------------------------------------------------------------------------- #


def test_build_submission_maps_back_to_raw_dates_in_sample_order(tmp_path):
    official_dir, _ = _prepare_mini(tmp_path)

    window_dates = _CALENDAR[_OBSERVED:]
    points = [400.0, 401.0, 402.0, 403.0]
    predictions = _predictions(window_dates, points).iloc[[2, 0, 3, 1]].reset_index(drop=True)
    out = tmp_path / "submission.csv"

    submission = build_submission(predictions, official_dir, out)

    sample = pd.read_csv(official_dir / "sample_submission.csv")
    assert list(submission.columns) == ["date", "0056_close_y"]
    assert submission["date"].tolist() == sample["date"].tolist()
    assert submission["date"].tolist() == list(reversed(_SORTED_DATES[-_HORIZON:]))
    expected = dict(zip(_SORTED_DATES[-_HORIZON:], points))
    assert submission["0056_close_y"].tolist() == [expected[date] for date in submission["date"]]
    assert pd.read_csv(out).equals(submission)


def test_build_submission_raises_on_missing_extra_duplicate_and_nonfinite_dates(tmp_path):
    official_dir, _ = _prepare_mini(tmp_path)
    window_dates = _CALENDAR[_OBSERVED:]
    out = tmp_path / "submission.csv"

    short = _predictions(window_dates[:3], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="missing"):
        build_submission(short, official_dir, out)

    extra = _predictions(
        pd.DatetimeIndex([_CALENDAR[_OBSERVED - 1], *window_dates]),
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    with pytest.raises(ValueError, match="outside the sample"):
        build_submission(extra, official_dir, out)

    duplicated = _predictions(
        pd.DatetimeIndex([*window_dates, window_dates[0]]), [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_submission(duplicated, official_dir, out)

    non_finite = _predictions(window_dates, [1.0, 2.0, float("inf"), 4.0])
    with pytest.raises(ValueError, match="non-finite"):
        build_submission(non_finite, official_dir, out)

    unmapped = _predictions(pd.DatetimeIndex(["2000-02-01"]), [1.0])
    with pytest.raises(ValueError, match="date_map"):
        build_submission(unmapped, official_dir, out)


def test_validate_submission_rejects_columns_order_and_values(tmp_path):
    official_dir, _ = _prepare_mini(tmp_path)
    sample = pd.read_csv(official_dir / "sample_submission.csv")
    candidate = pd.read_csv(official_dir / "sample_submission.csv")
    validate_submission(candidate, sample)

    with pytest.raises(ValueError, match="columns"):
        validate_submission(candidate[["0056_close_y", "date"]], sample)
    with pytest.raises(ValueError, match="order"):
        validate_submission(candidate.iloc[::-1].reset_index(drop=True), sample)

    non_finite = candidate.copy()
    non_finite.loc[0, "0056_close_y"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_submission(non_finite, sample)

    negative = candidate.copy()
    negative.loc[0, "0056_close_y"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        validate_submission(negative, sample)
