"""Tiny local shards with the native corpus v4.1 schema."""

from pathlib import Path

import numpy as np


def write_rows(
    path: Path,
    targets,
    *,
    kind="univariate",
    covariates=None,
    freqs=None,
    row_group_size=2,
    item_id_size=8,
    compression="snappy",
    value_type=None,
    domains=None,
    subdomains=None,
    covariate_names=None,
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    numeric = value_type or pa.float64()
    schema = [
        ("item_id", pa.string()),
        ("source", pa.string()),
        ("freq", pa.string()),
        ("target", pa.list_(numeric) if kind == "covariates" else pa.list_(pa.list_(numeric))),
    ]
    if kind == "covariates":
        schema.extend(
            [
                ("covariates", pa.list_(pa.list_(numeric))),
                ("covariate_names", pa.list_(pa.string())),
            ]
        )
    optional_labels = {"domain": domains, "subdomain": subdomains}
    schema.extend(
        (name, pa.string()) for name, labels in optional_labels.items() if labels is not None
    )
    rows = []
    for i, target in enumerate(targets):
        row = {
            "item_id": f"{i:04d}".ljust(item_id_size, "x"),
            "source": "local-fixture",
            "freq": freqs[i] if freqs is not None else "D",
            "target": target,
        }
        if kind == "covariates":
            row["covariates"] = covariates[i]
            row["covariate_names"] = (
                covariate_names[i]
                if covariate_names is not None
                else [f"driver_{j}" for j in range(len(covariates[i]))]
            )
        row.update(
            (name, labels[i]) for name, labels in optional_labels.items() if labels is not None
        )
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=pa.schema(schema)),
        path,
        row_group_size=row_group_size,
        compression=compression,
        use_dictionary=True,
    )
    return path


def write_corpus(root: Path) -> tuple[Path, ...]:
    """Two univariate folders, one covariate folder; groups of two and three."""
    first = np.arange(40, dtype=np.float64)
    missing = first.copy()
    missing[4] = np.nan
    return (
        write_rows(
            root / "univariate/alpha/train-000.parquet",
            [[first], [missing], [first + 2], [first + 3]],
            freqs=["D", "h", "ME", "W-MON"],
        ),
        write_rows(
            root / "univariate/beta/validation-000.parquet",
            [[first + j for j in range(20)], [first], [first]],
            row_group_size=3,
        ),
        write_rows(
            root / "covariates/drivers/train-000.parquet",
            [first, missing, first],
            kind="covariates",
            covariates=[[first, missing], [first, first], [missing, first]],
            freqs=["h", "D", "ME"],
            row_group_size=3,
        ),
    )


def write_malformed_row(root: Path, problem: str) -> Path:
    """A valid row followed by a malformed row, in separate row groups."""
    first = np.arange(32, dtype=np.float64)
    if problem == "bad_shape":
        return write_rows(
            root / "covariates/a/train-bad.parquet",
            [first, first],
            kind="covariates",
            covariates=[[first], [first[:-1]]],
            row_group_size=1,
        )
    target = {
        "null_variate": None,
        "null_channel": [first, None],
        "ragged_channels": [first, first[:-1]],
        "zero_variates": [],
    }[problem]
    return write_rows(root / "univariate/a/train-bad.parquet", [[first], target], row_group_size=1)
