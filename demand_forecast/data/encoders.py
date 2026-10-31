"""Training-only static vocabularies and masked numeric standardisation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from demand_forecast.data.panel import Panel, Series


def _vocabulary(values: Iterable[object]) -> dict[str, int]:
    observed = sorted({str(value) for value in values if not pd.isna(value)})
    return {value: code for code, value in enumerate(observed, start=2)}


class StaticEncoder:
    """Encode statics using only training series, reserving 0/1 for unknown/missing.

    Numeric columns use the observed training mean and population standard
    deviation (ddof=0). A constant column uses std 1; an entirely missing column
    or an empty training split uses mean 0 and std 1. Numeric outputs follow
    the fitted role order, with missing values zeroed and masked out.
    """

    def __init__(self) -> None:
        self.static_categorical: tuple[str, ...] = ()
        self.static_numeric: tuple[str, ...] = ()
        self.vocabularies: dict[str, dict[str, int]] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def fit(self, panel: Panel, train_ids: Iterable[str]) -> StaticEncoder:
        """Fit once per distinct training id; an id absent from the panel raises KeyError."""
        if "__series__" in panel.roles.static_categorical:
            raise ValueError("__series__ is reserved for the series id vocabulary")
        by_id = {series.series_id: series for series in panel.series}
        training = [by_id[series_id] for series_id in sorted(set(train_ids))]
        vocabularies = {
            name: _vocabulary(series.statics.get(name) for series in training)
            for name in panel.roles.static_categorical
        }
        vocabularies["__series__"] = _vocabulary(series.series_id for series in training)

        means, stds = {}, {}
        for name in panel.roles.static_numeric:
            values = [series.statics.get(name) for series in training]
            observed = np.asarray([value for value in values if not pd.isna(value)], dtype=float)
            means[name] = float(observed.mean()) if observed.size else 0.0
            stds[name] = (float(observed.std()) or 1.0) if observed.size else 1.0

        self.static_categorical = tuple(panel.roles.static_categorical)
        self.static_numeric = tuple(panel.roles.static_numeric)
        self.vocabularies = vocabularies
        self.means = means
        self.stds = stds
        return self

    @property
    def vocab_sizes(self) -> dict[str, int]:
        """Embedding sizes, including the unknown and missing slots."""
        return {name: len(vocabulary) + 2 for name, vocabulary in self.vocabularies.items()}

    @property
    def n_numeric(self) -> int:
        """Number of numeric features in each transformed series."""
        return len(self.static_numeric)

    def _require_fitted(self) -> None:
        if not self.vocabularies:
            raise RuntimeError("StaticEncoder must be fitted before use")

    def transform(self, series: Series) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
        """Return categorical codes, float32 numerics and a boolean observed mask."""
        self._require_fitted()
        codes = {}
        for name, vocabulary in self.vocabularies.items():
            value = series.series_id if name == "__series__" else series.statics.get(name)
            codes[name] = 1 if pd.isna(value) else vocabulary.get(str(value), 0)

        numeric = np.zeros(self.n_numeric, dtype=np.float32)
        mask = np.zeros(self.n_numeric, dtype=bool)
        for i, name in enumerate(self.static_numeric):
            value = series.statics.get(name)
            if not pd.isna(value):
                numeric[i] = (float(value) - self.means[name]) / self.stds[name]
                mask[i] = True
        return codes, numeric, mask

    def to_json(self) -> dict:
        """Return JSON-compatible fitted state, including explicit column order."""
        self._require_fitted()
        return {
            "static_categorical": list(self.static_categorical),
            "static_numeric": list(self.static_numeric),
            "vocabularies": {name: dict(values) for name, values in self.vocabularies.items()},
            "means": dict(self.means),
            "stds": dict(self.stds),
        }

    @classmethod
    def from_json(cls, d: dict) -> StaticEncoder:
        """Restore fitted state without depending on JSON object key order."""
        encoder = cls()
        encoder.static_categorical = tuple(d["static_categorical"])
        encoder.static_numeric = tuple(d["static_numeric"])
        encoder.vocabularies = {
            name: dict(d["vocabularies"][name])
            for name in (*encoder.static_categorical, "__series__")
        }
        encoder.means = {name: float(d["means"][name]) for name in encoder.static_numeric}
        encoder.stds = {name: float(d["stds"][name]) for name in encoder.static_numeric}
        return encoder
