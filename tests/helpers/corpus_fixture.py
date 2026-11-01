"""Tiny, reproducible shards with the Chronos training corpus schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def write_corpus_fixture(
    directory: Path,
    *,
    n_series: int = 12,
    lengths: tuple[int, int] = (70, 90),
    seed: int = 0,
    kind: Literal["tsmixup", "kernel_synth"] = "tsmixup",
    index: int = 0,
    constant: bool = False,
    all_nan: bool = False,
) -> Path:
    """Write a shard below a corpus root, using dummy epoch timestamps.

    Lengths are sampled from the inclusive bounds. Constant targets are ones;
    ``all_nan`` takes precedence when both target switches are enabled.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    prefix, count = {
        "tsmixup": ("training_corpus/tsmixup_10m", 200),
        "kernel_synth": ("training_corpus/kernel_synth_1m", 20),
    }[kind]
    rng = np.random.default_rng(seed)
    rows = []
    for series_index in range(n_series):
        length = int(rng.integers(lengths[0], lengths[1] + 1))
        if all_nan:
            target = np.full(length, np.nan)
        elif constant:
            target = np.ones(length)
        else:
            target = rng.normal(size=length)
        rows.append(
            {
                "id": f"{kind}-{index}-{series_index}",
                "timestamp": np.zeros(length, dtype="datetime64[ms]"),
                "target": target,
            }
        )
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("timestamp", pa.list_(pa.timestamp("ms"))),
            ("target", pa.list_(pa.float64())),
        ]
    )
    path = directory / prefix / f"train-{index:05d}-of-{count:05d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path
