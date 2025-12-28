"""Feature engineering utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CategoricalEncoder:
    """Encapsulates categorical encoding logic.

    Replaces the global `label_encoders` dict pattern with a proper
    encapsulated data structure.
    """

    encoders: dict[str, MultiLabelBinarizer | OneHotEncoder] = field(default_factory=dict)
    encoded_columns: list[str] = field(default_factory=list)

    def fit_transform(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
        onehot_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fit encoders and transform categorical columns.

        Args:
            df: DataFrame to transform.
            categorical_columns: Columns to encode with MultiLabelBinarizer.
            onehot_columns: Columns to encode with OneHotEncoder.

        Returns:
            DataFrame with encoded columns added.
        """
        if onehot_columns is None:
            onehot_columns = []

        for column in tqdm(categorical_columns, desc="Encoding categoricals"):
            if column in onehot_columns:
                encoder = OneHotEncoder(sparse_output=False, dtype=int)
                encoded = encoder.fit_transform(df[column].reset_index(drop=True).to_frame())
            else:
                encoder = MultiLabelBinarizer()
                encoded = encoder.fit_transform(df[column].reset_index(drop=True))

            self.encoders[column] = encoder
            encoded_col = f"encoded_{column}"
            df[encoded_col] = encoded.tolist()
            self.encoded_columns.append(encoded_col)

        logger.info(f"Encoded {len(categorical_columns)} categorical columns")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted encoders.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with encoded columns added.
        """
        for column, encoder in self.encoders.items():
            if isinstance(encoder, OneHotEncoder):
                encoded = encoder.transform(df[column].reset_index(drop=True).to_frame())
            else:
                encoded = encoder.transform(df[column].reset_index(drop=True))

            encoded_col = f"encoded_{column}"
            df[encoded_col] = encoded.tolist()

        return df

    def get_encoded_columns(self) -> list[str]:
        """Get list of encoded column names.

        Returns:
            List of column names for encoded features.
        """
        return self.encoded_columns

    def save(self, path: Path) -> None:
        """Save encoders to file.

        Args:
            path: Path to save encoders.
        """
        data = {
            "encoders": self.encoders,
            "encoded_columns": self.encoded_columns,
        }
        joblib.dump(data, path)
        logger.info(f"Saved {len(self.encoders)} encoders to {path}")

    @classmethod
    def load(cls, path: Path) -> "CategoricalEncoder":
        """Load encoders from file.

        Args:
            path: Path to load encoders from.

        Returns:
            CategoricalEncoder instance with loaded encoders.
        """
        data = joblib.load(path)
        instance = cls()
        instance.encoders = data["encoders"]
        instance.encoded_columns = data["encoded_columns"]
        logger.info(f"Loaded {len(instance.encoders)} encoders from {path}")
        return instance


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
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached metafeatures from {cache_path}")
        return pd.read_csv(cache_path)

    import tsfresh
    from tsfresh.feature_extraction import MinimalFCParameters

    logger.info("Extracting TSFresh meta-features")

    default_fc_params = MinimalFCParameters()
    metafeatures = {}

    grouped = df.groupby([sku_code_column, store_column])
    for label, group in tqdm(grouped, desc="Extracting features"):
        metafeatures[label] = tsfresh.extract_features(
            group.reset_index(),
            column_id=sku_code_column,
            column_value=quantity_column,
            column_sort=date_column,
            disable_progressbar=True,
            default_fc_parameters=default_fc_params,
        )

    feature_df = pd.DataFrame(metafeatures.keys())
    tmp = pd.concat(metafeatures.values()).reset_index(drop=True)

    feature_df = pd.concat([feature_df, tmp], axis=1, ignore_index=True)
    feature_df.columns = [sku_code_column, store_column] + list(tmp.columns)

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
