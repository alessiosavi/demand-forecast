"""Feature engineering utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CategoricalEncoder:
    """Stage-fitted integer vocabularies (unknown=0) and log continuous features."""

    encoders: dict[str, dict[str, int]] = field(default_factory=dict)
    encoded_columns: list[str] = field(default_factory=list)
    vocab_sizes: dict[str, int] = field(default_factory=dict)
    continuous_stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, df, categorical_columns, continuous_columns=()):
        self.encoders = {}
        self.encoded_columns = []
        self.vocab_sizes = {}
        self.continuous_stats = {}
        for column in categorical_columns:
            values = sorted(df[column].dropna().astype(str).unique())
            self.encoders[column] = {value: i + 1 for i, value in enumerate(values)}
            encoded = f"encoded_{column}"
            self.encoded_columns.append(encoded)
            self.vocab_sizes[encoded] = len(values) + 1
        for column in continuous_columns:
            values = self._log_values(df[column])
            self.continuous_stats[column] = (float(values.mean()), float(values.std()) or 1.0)
        return self

    @staticmethod
    def _log_values(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Continuous static features must be finite and non-negative")
        return np.log1p(values)

    def fit_transform(self, df, categorical_columns, onehot_columns=None):
        self.fit(df, categorical_columns)
        return self.transform(df)

    def transform(self, df):
        df = df.copy()
        for column, mapping in self.encoders.items():
            df[f"encoded_{column}"] = (
                df[column].astype("string").map(mapping).fillna(0).astype("int64")
            )
        for column, (mean, std) in self.continuous_stats.items():
            df[f"continuous_{column}"] = (self._log_values(df[column]) - mean) / std
        return df

    def get_encoded_columns(self):
        return self.encoded_columns

    def save(self, path):
        joblib.dump(vars(self), path)

    @classmethod
    def load(cls, path):
        return cls(**joblib.load(path))


def extract_metafeatures(
    df: pd.DataFrame,
    sku_code_column: str = "sku_code",
    store_column: str = "store_id",
    quantity_column: str = "qty",
    date_column: str = "date",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Extract time series meta-features using TSFresh.

    Args:
        df: DataFrame with time series data.
        sku_code_column: Column with SKU codes.
        store_column: Column with store IDs.
        quantity_column: Column with quantity values.
        date_column: Column with date/time values for sorting.
        cache_path: Optional path to cache/load features.

    Returns:
        DataFrame with extracted meta-features.
    """
    import tsfresh
    from tsfresh.feature_extraction import MinimalFCParameters

    logger.info("Extracting TSFresh meta-features")

    # One extraction call, no worker pool per SKU; cache is output-only, never blindly reused.
    keys = [sku_code_column, store_column]
    data = df.reset_index().copy()
    data["_series_id"] = data.groupby(keys, sort=True).ngroup()
    identities = data.drop_duplicates("_series_id").set_index("_series_id")[keys]
    features = tsfresh.extract_features(
        data[["_series_id", date_column, quantity_column]],
        column_id="_series_id",
        column_value=quantity_column,
        column_sort=date_column,
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0,
        disable_progressbar=True,
    )
    feature_df = identities.join(features).reset_index(drop=True)

    if cache_path:
        feature_df.to_csv(cache_path, index=False)
        logger.info(f"Cached metafeatures to {cache_path}")

    return feature_df


def get_categorical_columns(
    df: pd.DataFrame,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """Identify categorical columns in a DataFrame.

    Args:
        df: DataFrame to analyze.
        exclude_patterns: List of patterns to exclude from categorical columns.

    Returns:
        List of categorical column names.
    """
    if exclude_patterns is None:
        exclude_patterns = ["sku"]

    numeric_cols = set(df.select_dtypes(np.number).columns)

    categorical_cols = [
        c
        for c in df.columns
        if c not in numeric_cols and not any(pattern in c for pattern in exclude_patterns)
    ]

    return categorical_cols
