"""Fixed kind rules and spelling-independent flag codes at a statistics cutoff."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .schema import KIND_CATEGORY, KIND_FLAG, KIND_NUMBER


def is_missing(value: object) -> bool:
    """Recognise missing scalar cells without confusing zero, False or labels."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (list, tuple, str)):
        return len(value) == 0
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def label_of(value: object) -> str:
    """Compare non-missing category values by their string representation."""
    return str(value)


_SIMPLE_OBJECT_TYPES = {id(str), id(type(None)), id(float)}


def _simple_object_types(values: np.ndarray) -> set[int] | None:
    # Type identities also avoid invoking custom metaclass equality or hashing.
    types = set(map(id, map(type, values.flat)))
    return types if types <= _SIMPLE_OBJECT_TYPES else None


def observed_mask(values: np.ndarray) -> np.ndarray:
    """Return the cells that the scalar missingness rule considers observed."""
    values = np.asarray(values)
    if values.ndim == 0:
        return observed_mask(values.reshape(1)).reshape(())
    kind = values.dtype.kind
    if kind == "f":
        return ~np.isnan(values)
    if kind in "biuS":
        # Even empty byte strings are observed by is_missing.
        return np.ones(values.shape, dtype=bool)
    if kind == "U":
        return values != ""
    if kind == "O":
        types = _simple_object_types(values)
        if types is not None:
            if types <= {id(str)}:
                return values != ""
            return (values != "") & np.not_equal(values, None) & (values == values)
    # Lists and tuples have scalar missingness rules that pandas.isna cannot apply.
    return ~np.asarray(np.frompyfunc(is_missing, 1, 1)(values), dtype=bool)


def _distinct_labels(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(keys, return_inverse=True)
    # Feed frompyfunc a typed array to use the same scalar boxing as the reference.
    labels = np.frompyfunc(label_of, 1, 1)(unique.view(values.dtype))
    return labels[inverse.ravel()].reshape(values.shape)


def labels_of(values: np.ndarray) -> np.ndarray:
    """Return exact scalar string labels in an object array of the input shape."""
    values = np.asarray(values)
    kind = values.dtype.kind
    if kind == "f":
        # Float equality merges signed zeros. Deduplicate the original bits instead.
        size = values.itemsize
        bits = values.view(f"u{size}" if size in (2, 4, 8) else f"V{size}")
        return _distinct_labels(values, bits)
    if kind in "iuS":
        return _distinct_labels(values, values)
    if kind == "b":
        return np.where(values, "True", "False").astype(object)
    if kind == "U":
        return values.astype(object)
    if kind == "O":
        types = _simple_object_types(values)
        if types is not None:
            labels = values.copy()
            if id(type(None)) in types:
                labels[np.equal(values, None)] = "None"
            if id(float) in types:
                floats = np.equal(np.frompyfunc(type, 1, 1)(values), float)
                labels[floats] = labels_of(values[floats].astype(np.float64))
            return labels
    return np.asarray(np.frompyfunc(label_of, 1, 1)(values), dtype=object)


def infer_kind(values: np.ndarray) -> int:
    """Infer a dynamic kind from only the caller-supplied permitted prefix."""
    values = np.asarray(values)
    observed = values[observed_mask(values)]
    distinct = len(np.unique(labels_of(observed)))
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
    observed = cells[observed_mask(cells)]
    if np.frompyfunc(lambda value: isinstance(value, (str, bool, np.bool_)), 1, 1)(observed).any():
        return "category"
    numbers = np.asarray(observed, dtype=np.float64)
    if (
        len(numbers)
        and np.isfinite(numbers).all()
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
    cutoff = max(0, cutoff)
    observed = observed_mask(values)
    labels = labels_of(values)
    prefix_codes, prefix_labels = pd.factorize(labels[:cutoff][observed[:cutoff]], sort=False)
    codes = np.full(values.shape, np.nan, dtype=np.float64)
    codes[observed] = 1.0
    if len(prefix_labels):
        majority = prefix_labels[np.argmax(np.bincount(prefix_codes))]
        codes[observed & (labels == majority)] = 0.0
    return codes
