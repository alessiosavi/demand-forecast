"""Fixed kind rules and spelling-independent flag codes at a statistics cutoff."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .schema import KIND_CATEGORY, KIND_FLAG, KIND_NUMBER


def is_missing(value: object) -> bool:
    """Recognise missing scalar cells without confusing zero, False or labels."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, str)):
        return len(value) == 0
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def label_of(value: object) -> str:
    """Compare non-missing category values by their string representation."""
    return str(value)


def _observed(values: np.ndarray) -> np.ndarray:
    # Object cells can contain lists; pandas.isna cannot reduce those to a scalar.
    return ~np.frompyfunc(is_missing, 1, 1)(values).astype(bool)


def _labels(values: np.ndarray) -> np.ndarray:
    return np.frompyfunc(label_of, 1, 1)(values)


def infer_kind(values: np.ndarray) -> int:
    """Infer a dynamic kind from only the caller-supplied permitted prefix."""
    values = np.asarray(values)
    observed = values[_observed(values)]
    distinct = len(np.unique(_labels(observed)))
    if distinct == 2:
        return KIND_FLAG
    if values.dtype.kind in "OUSb":
        return KIND_CATEGORY
    if (
        3 <= distinct <= 16
        and len(observed) >= 4 * distinct
        and np.isfinite(observed).all()
        and np.all(observed == np.floor(observed))
    ):
        return KIND_CATEGORY
    return KIND_NUMBER


def infer_static_kind(values: Sequence[object]) -> str:
    """Infer a static kind across items, preserving empty multilabel cells."""
    cells = np.empty(len(values), dtype=object)
    cells[:] = values
    if np.frompyfunc(lambda value: isinstance(value, (list, tuple)), 1, 1)(cells).any():
        return "multilabel"
    observed = cells[_observed(cells)]
    if np.frompyfunc(lambda value: isinstance(value, (str, bool, np.bool_)), 1, 1)(observed).any():
        return "category"
    numbers = np.asarray(observed, dtype=np.float64)
    if (
        len(numbers)
        and np.all(numbers == np.floor(numbers))
        and 2 * len(np.unique(numbers)) <= len(values)
    ):
        return "category"
    return "number"


def flag_codes(values: np.ndarray, cutoff: int) -> np.ndarray:
    """Code the prefix majority as zero; ties favour its earliest occurrence.

    The rarer (or later tied) value and any unseen label are one. Missing cells
    remain NaN. Factorisation preserves occurrence order rather than spelling.
    """
    values = np.asarray(values)
    observed = _observed(values)
    labels = _labels(values)
    prefix_codes, prefix_labels = pd.factorize(labels[:cutoff][observed[:cutoff]], sort=False)
    codes = np.full(values.shape, np.nan, dtype=np.float64)
    codes[observed] = 1.0
    if len(prefix_labels):
        majority = prefix_labels[np.argmax(np.bincount(prefix_codes))]
        codes[observed & (labels == majority)] = 0.0
    return codes
