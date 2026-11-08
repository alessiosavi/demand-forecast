"""Store-sales submission assembly and validation.

The competition's submission is ``id,sales`` in the sample-submission row
order. Predictions arrive keyed by the loader's ``series_id`` (the ``"|"``-join
of the configured id columns) and ``timestamp``; the writer joins them back
onto the test ids on ``(store_nbr, family, date)``.

Prediction points are log1p-space — the converter's pinned target transform —
so the writer inverts them with ``expm1`` before writing the raw ``sales``
column: ``expm1`` maps every finite point to a non-negative value, so the
unchanged validator still holds.

Join keys are normalized to pandas' string dtype on both sides: splitting
``series_id`` yields strings while ``test_ids.store_nbr`` and
``truth.store_nbr`` are integer columns, so ``"1"`` must be made to match ``1``.
``scoring`` imports the same normalizers, keeping writer and scorer identical.
"""

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # repo venv carries no pandas stubs

__all__ = ["build_store_sales_submission", "validate_submission"]

_JOIN_KEYS = ["store_nbr", "family", "date"]


def _as_datetime(values: pd.Series, *, name: str) -> pd.Series:
    """Parse a date or timestamp column, rejecting null values."""
    stamps = pd.to_datetime(values, format="mixed")
    if stamps.isna().any():
        raise ValueError(f"{name}: null date values")
    return stamps


def _prediction_keys(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Normalized join keys plus the ``point`` column of a prediction frame.

    The frame must carry ``series_id`` (``store_nbr|family``), ``timestamp`` and
    ``point``; the returned ``store_nbr``/``family`` columns are string-typed so
    they join with the raw integer store tables.
    """
    missing = [
        column for column in ("series_id", "timestamp", "point") if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    parts = frame["series_id"].astype("string").str.split("|", regex=False, expand=True)
    if parts.shape[1] != 2 or parts.isna().any().any():
        raise ValueError(f"{name}: series_id must be 'store_nbr|family'")
    return pd.DataFrame(
        {
            "store_nbr": parts[0].astype("string").reset_index(drop=True),
            "family": parts[1].astype("string").reset_index(drop=True),
            "date": _as_datetime(frame["timestamp"], name=name).reset_index(drop=True),
            "point": frame["point"].astype("float64").reset_index(drop=True),
        }
    )


def _catalog_keys(frame: pd.DataFrame, carry: list[str], *, name: str) -> pd.DataFrame:
    """Normalized join keys of a catalog-style frame, plus ``carry`` columns.

    Used for ``test_ids`` and for the mirror ``truth.parquet``; their
    ``store_nbr`` is an integer column while prediction keys are strings, so
    both sides are pinned to pandas' string dtype for the join.
    """
    needed = [*_JOIN_KEYS, *carry]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    keys = pd.DataFrame(
        {
            "store_nbr": frame["store_nbr"].astype("string").reset_index(drop=True),
            "family": frame["family"].astype("string").reset_index(drop=True),
            "date": _as_datetime(frame["date"], name=name).reset_index(drop=True),
        }
    )
    for column in carry:
        keys[column] = frame[column].reset_index(drop=True)
    return keys


def build_store_sales_submission(
    predictions: pd.DataFrame, test_ids: pd.DataFrame, out: Path
) -> pd.DataFrame:
    """Join predictions onto the sample order and write ``id,sales`` to ``out``.

    Prediction points are log1p-space (the converter's target transform), so
    they are inverted with ``expm1`` before the raw ``sales`` column is
    written. Every test id must receive exactly one prediction; unmatched ids
    raise instead of writing a partial file. Prediction rows without a test id
    are ignored, because the loaded panel may carry series absent from
    ``test.csv``. The returned frame is the written submission, in ``test_ids``
    order.
    """
    keys = _prediction_keys(predictions, name="predictions")
    test = _catalog_keys(test_ids, ["id"], name="test_ids")
    merged = test.merge(keys, on=_JOIN_KEYS, how="left", validate="one_to_one")
    unmatched = int(merged["point"].isna().sum())
    if unmatched:
        raise ValueError(f"predictions missing or NaN for {unmatched} of {len(test)} test ids")
    submission = merged.assign(sales=np.expm1(merged["point"]))[["id", "sales"]]
    if len(submission) != len(test_ids) or submission["id"].tolist() != test_ids["id"].tolist():
        raise ValueError("submission row count or id order differs from test_ids")
    submission.to_csv(out, index=False)
    return submission


def validate_submission(candidate: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise ``ValueError`` unless ``candidate`` is a valid submission.

    The candidate must carry the sample's columns in the same order and the
    same ids in the same order, with finite non-negative ``sales`` — the last
    check a submission passes before any upload.
    """
    if list(candidate.columns) != list(sample.columns):
        raise ValueError(
            f"submission columns {list(candidate.columns)} differ from sample {list(sample.columns)}"
        )
    if candidate["id"].tolist() != sample["id"].tolist():
        raise ValueError("submission id order differs from the sample")
    sales = candidate["sales"].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(sales)).sum())
    if non_finite:
        raise ValueError(f"submission sales contain {non_finite} non-finite values")
    negative = int((sales < 0).sum())
    if negative:
        raise ValueError(f"submission sales contain {negative} negative values")
