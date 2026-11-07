"""Store-sales submission writer and validator tests (pure: no parquet engine)."""

import pandas as pd
import pytest

from demand_forecast.kaggle.submission import build_store_sales_submission, validate_submission


def _test_ids() -> pd.DataFrame:
    """Test ids in sample order, carrying the raw integer store_nbr column."""
    return pd.DataFrame(
        {
            "id": [3, 1, 2],
            "date": pd.to_datetime(["2017-08-16", "2017-08-16", "2017-08-17"]),
            "store_nbr": [1, 2, 1],
            "family": ["AUTOMOTIVE", "AUTOMOTIVE", "BABY CARE"],
        }
    )


def _predictions() -> pd.DataFrame:
    """All three keys in a shuffled row order, with loader-style series ids."""
    rows = [
        {"series_id": "1|AUTOMOTIVE", "timestamp": "2017-08-16", "point": 5.0},
        {"series_id": "1|BABY CARE", "timestamp": "2017-08-17", "point": 9.0},
        {"series_id": "2|AUTOMOTIVE", "timestamp": "2017-08-16", "point": 7.0},
    ]
    return pd.DataFrame(rows).iloc[[2, 0, 1]].reset_index(drop=True)


def test_store_sales_submission_order_follows_test_ids(tmp_path):
    test_ids = _test_ids()
    # Integer store numbers must join with the string parts split out of series_id.
    assert pd.api.types.is_integer_dtype(test_ids["store_nbr"])
    out = tmp_path / "submission.csv"
    submission = build_store_sales_submission(_predictions(), test_ids, out)

    assert list(submission.columns) == ["id", "sales"]
    assert submission["id"].tolist() == [3, 1, 2]
    assert submission["sales"].tolist() == [5.0, 7.0, 9.0]

    sample = pd.DataFrame({"id": test_ids["id"], "sales": 0.0})
    validate_submission(submission, sample)  # must not raise

    written = pd.read_csv(out)
    assert list(written.columns) == ["id", "sales"]
    assert written["id"].tolist() == [3, 1, 2]
    assert written["sales"].tolist() == [5.0, 7.0, 9.0]


def test_submission_missing_rows_rejected(tmp_path):
    incomplete = _predictions().iloc[[0, 1]]  # drops the (1, BABY CARE) row
    with pytest.raises(ValueError, match="missing"):
        build_store_sales_submission(incomplete, _test_ids(), tmp_path / "submission.csv")


def test_submission_ignores_untested_prediction_rows(tmp_path):
    extra = pd.DataFrame([{"series_id": "9|GROCERY", "timestamp": "2017-08-16", "point": 1.0}])
    predictions = pd.concat([_predictions(), extra], ignore_index=True)
    submission = build_store_sales_submission(predictions, _test_ids(), tmp_path / "submission.csv")
    assert submission["id"].tolist() == [3, 1, 2]
    assert submission["sales"].tolist() == [5.0, 7.0, 9.0]


def test_submission_reordering_rejected():
    sample = pd.DataFrame({"id": [3, 1, 2], "sales": [0.0, 0.0, 0.0]})
    swapped = pd.DataFrame({"id": [1, 3, 2], "sales": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="order"):
        validate_submission(swapped, sample)


def test_validate_submission_rejects_column_mismatch():
    sample = pd.DataFrame({"id": [1], "sales": [0.0]})
    candidate = pd.DataFrame({"id": [1], "sales": [0.0], "extra": [1]})
    with pytest.raises(ValueError, match="columns"):
        validate_submission(candidate, sample)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_validate_submission_rejects_bad_sales(bad):
    sample = pd.DataFrame({"id": [1, 2], "sales": [0.0, 0.0]})
    candidate = pd.DataFrame({"id": [1, 2], "sales": [1.0, bad]})
    with pytest.raises(ValueError, match="sales"):
        validate_submission(candidate, sample)
