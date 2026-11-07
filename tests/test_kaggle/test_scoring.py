"""Mirror scoring tests on a synthetic prepared mirror tree (needs pyarrow)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand_forecast.config import DatasetConfig
from demand_forecast.kaggle.scoring import score_mirror
from demand_forecast.kaggle.settings import build_settings, load_template, write_settings

# The dev-only CI lap installs no parquet engine: skip this module, never fail it.
pytest.importorskip("pyarrow")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO_ROOT / "configs/kaggle-v2-l3-finetune.yaml"
_HORIZON = 2
# 2017-08-01..04; the mirror window cuts the last 2 train days (origin 08-03),
# so history is observed through 08-02 and truth covers 08-03..04.
_DAYS = pd.date_range("2017-08-01", periods=4, freq="D")
_MIRROR_ORIGIN = "2017-08-03"
_HISTORY = {(1, "AUTOMOTIVE"): [8.0, 10.0], (2, "BABY CARE"): [25.0, 30.0]}
_TRUTH = {(1, "AUTOMOTIVE"): [12.0, 13.0], (2, "BABY CARE"): [20.0, 21.0]}
_TRUTH_VALUES = np.array([12.0, 13.0, 20.0, 21.0])
# Short history (< the 7-day season) makes the baseline fall back to the last
# observed target at or before each step: 08-02 sales for both steps.
_BASELINE_VALUES = np.array([10.0, 10.0, 30.0, 30.0])


def _dataset_config(directory: Path) -> DatasetConfig:
    """The long/static roles the store-sales templates consume (as Task 4 pins)."""
    return DatasetConfig(
        source="long",
        path=directory / "long.parquet",
        id_columns=["store_nbr", "family"],
        timestamp_column="date",
        target="sales",
        target_support="nonnegative",
        known_dynamic=["onpromotion", "is_national_holiday"],
        past_dynamic=["oil_price", "transactions"],
        static_path=directory / "static.parquet",
        static_categorical=["store_nbr", "family", "city", "state", "type", "cluster"],
        freq="D",
        missing_target="zero",
        duplicates="error",
    )


def _long_frame() -> pd.DataFrame:
    """Mirror-cut panel: target and past dynamics stop at 2017-08-02."""
    rows = []
    for index, day in enumerate(_DAYS):
        for (store_nbr, family), values in _HISTORY.items():
            observed = index < _HORIZON
            rows.append(
                {
                    "store_nbr": store_nbr,
                    "family": family,
                    "date": day,
                    "sales": values[index] if observed else float("nan"),
                    "onpromotion": int(day.day % 2 == 0),
                    "is_national_holiday": 0,
                    "oil_price": 40.0 + index if observed else float("nan"),
                    "transactions": 100.0 * store_nbr if observed else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _static_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "family": ["AUTOMOTIVE", "BABY CARE"],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "B"],
            "cluster": [3, 7],
        }
    )


def _truth_frame() -> pd.DataFrame:
    rows = [
        {"store_nbr": store_nbr, "family": family, "date": _DAYS[_HORIZON + step], "sales": value}
        for (store_nbr, family), values in _TRUTH.items()
        for step, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def _tiny_mirror(root: Path) -> Path:
    """Write the mirror variant of a prepared tree, settings built from the real template."""
    mirror = root / "mirror"
    mirror.mkdir(parents=True)
    _long_frame().to_parquet(mirror / "long.parquet", index=False)
    _static_frame().to_parquet(mirror / "static.parquet", index=False)
    _truth_frame().to_parquet(mirror / "truth.parquet", index=False)
    settings = build_settings(
        load_template(_TEMPLATE), data=_dataset_config(mirror), horizon=_HORIZON
    )
    write_settings(settings, mirror / "settings-finetune.yaml")
    (mirror / "prepare.json").write_text(
        json.dumps({"mirror_origin": _MIRROR_ORIGIN}), encoding="utf-8"
    )
    return mirror


def _points(values: dict[tuple[int, str], list[float]], *, strings: bool = False) -> pd.DataFrame:
    """Prediction rows in a shuffled order; ``strings`` mimics a plain read_csv."""
    rows = [
        {"series_id": f"{store_nbr}|{family}", "timestamp": _DAYS[_HORIZON + step], "point": value}
        for (store_nbr, family), series_values in values.items()
        for step, value in enumerate(series_values)
    ]
    frame = pd.DataFrame(rows).iloc[[3, 0, 2, 1]].reset_index(drop=True)
    if strings:
        frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    return frame


def _rmsle(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(labels) - np.log1p(predictions)) ** 2)))


def test_score_mirror_uses_identical_rows_and_adds_baseline(tmp_path):
    mirror = _tiny_mirror(tmp_path)
    zero_shot = _points(_TRUTH)
    offset = {key: [value + 1.0 for value in values] for key, values in _TRUTH.items()}
    finetune = _points(offset, strings=True)

    scores = score_mirror(mirror, {"zero_shot": zero_shot, "finetune": finetune})

    assert set(scores) == {"baseline", "zero_shot", "finetune"}
    assert list(scores) == ["baseline", "zero_shot", "finetune"]
    assert all(np.isfinite(list(scores.values())))
    # Exact row alignment: identical points score exactly zero; a misaligned join cannot.
    assert scores["zero_shot"] == 0.0
    assert scores["finetune"] == pytest.approx(_rmsle(_TRUTH_VALUES, _TRUTH_VALUES + 1.0))
    assert scores["baseline"] == pytest.approx(_rmsle(_TRUTH_VALUES, _BASELINE_VALUES))


def test_score_mirror_row_mismatch_raises(tmp_path):
    mirror = _tiny_mirror(tmp_path)
    zero_shot = _points(_TRUTH)
    finetune = zero_shot.iloc[[0, 1, 2]].reset_index(drop=True)  # one row dropped
    with pytest.raises(ValueError, match="row set differs"):
        score_mirror(mirror, {"zero_shot": zero_shot, "finetune": finetune})


@pytest.mark.parametrize(
    ("bad_point", "match"),
    [(-1.0, "negative"), (float("nan"), "non-finite"), (float("inf"), "non-finite")],
)
def test_score_mirror_rejects_invalid_points(tmp_path, bad_point, match):
    mirror = _tiny_mirror(tmp_path)
    zero_shot = _points(_TRUTH)
    zero_shot.loc[1, "point"] = bad_point
    with pytest.raises(ValueError, match=match):
        score_mirror(mirror, {"zero_shot": zero_shot})


def test_score_mirror_rejects_unknown_metric(tmp_path):
    with pytest.raises(ValueError, match="unknown metric"):
        score_mirror(tmp_path, {}, metric_name="mape")


def test_score_mirror_rejects_reserved_baseline_key(tmp_path):
    with pytest.raises(ValueError, match="baseline"):
        score_mirror(tmp_path, {"baseline": pd.DataFrame()})
