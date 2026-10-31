"""Small, reproducible local datasets for fev integration tests."""

from pathlib import Path

import numpy as np
import pandas as pd


def write_fev_fixture(
    directory: Path,
    *,
    n_series: int = 6,
    length: int = 40,
    freq: str = "W-SUN",
    start: str = "2022-01-02",
    known: tuple[str, ...] = ("promo",),
    past: tuple[str, ...] = ("oil",),
    statics: tuple[str, ...] = ("cat",),
    seed: int = 0,
) -> Path:
    """Write fev list-column data with a 50% promotion uplift and 15% zeros.

    The zero count is rounded to the nearest observation per series. The
    promotion indicator is ``promo`` when present, otherwise the first known
    covariate. Randomness is local to this call and controlled by ``seed``.
    """
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=length, freq=freq).to_numpy()
    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "timestamp": datasets.Sequence(datasets.Value("timestamp[ns]")),
            "target": datasets.Sequence(datasets.Value("float32")),
            **{name: datasets.Sequence(datasets.Value("int8")) for name in known},
            **{name: datasets.Sequence(datasets.Value("float32")) for name in past},
            **{name: datasets.Value("string") for name in statics},
        }
    )
    rows = []
    for index in range(n_series):
        target = rng.uniform(10.0, 20.0, size=length)
        known_values = {name: rng.integers(0, 2, size=length) for name in known}
        if known:
            promo = known_values["promo" if "promo" in known else known[0]]
            target *= 1.0 + 0.5 * promo
        target[rng.choice(length, size=round(0.15 * length), replace=False)] = 0.0
        rows.append(
            {
                "id": str(index),
                "timestamp": timestamps,
                "target": target.astype(np.float32),
                **known_values,
                **{name: rng.normal(50.0, 5.0, size=length).astype(np.float32) for name in past},
                **{name: "A" if index % 2 == 0 else "B" for name in statics},
            }
        )

    directory.mkdir(parents=True, exist_ok=True)
    parquet = directory / "data.parquet"
    datasets.Dataset.from_list(rows, features=features).to_parquet(parquet)
    return parquet


def fixture_task(
    parquet: Path,
    *,
    horizon: int = 4,
    num_windows: int = 2,
    known: tuple[str, ...] = ("promo",),
    past: tuple[str, ...] = ("oil",),
    statics: tuple[str, ...] = ("cat",),
) -> "fev.Task":  # noqa: F821 -- fev must remain a function-local optional import.
    """Create a local fev task with the benchmark quantiles and metric settings."""
    try:
        import fev
    except ImportError as exc:
        raise ImportError("install demand-forecast[bench]") from exc

    return fev.Task(
        dataset_path=str(parquet),
        dataset_config=None,
        horizon=horizon,
        num_windows=num_windows,
        quantile_levels=[i / 10 for i in range(1, 10)],
        eval_metric="SQL",
        extra_metrics=[
            "MASE",
            {"name": "WAPE", "epsilon": 1.0},
            {"name": "WQL", "epsilon": 1.0},
        ],
        known_dynamic_columns=list(known),
        past_dynamic_columns=list(past),
        static_columns=list(statics),
        seasonality=1,
    )
