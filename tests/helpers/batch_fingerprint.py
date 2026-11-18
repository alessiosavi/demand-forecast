"""Compare dataclass batches with tolerance for floating-point rounding."""

import hashlib
from dataclasses import fields
from math import isclose

import numpy as np
import torch


def _fingerprint(name: str, value: object) -> object:
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, (np.ndarray, torch.Tensor)) for item in value
    ):
        return [_fingerprint(name, item) for item in value]

    dtype = None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        try:
            value = tensor.resolve_neg().numpy()
        except TypeError:
            if not tensor.is_floating_point():
                raise
            # Some Torch floating formats, such as bfloat16, have no NumPy dtype.
            dtype = str(tensor.dtype)
            value = tensor.double().numpy()

    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        dtype = value.dtype.str if dtype is None else dtype
        flat = value.astype(np.float64, copy=False).ravel()
        nan = np.isnan(flat)
        flat = np.where(nan, 0.0, flat)
        weights = np.arange(1, flat.size + 1, dtype=np.float64) / max(1, flat.size)
        return {
            "dtype": dtype,
            "shape": list(value.shape),
            "nan": int(nan.sum()),
            "sums": [float(flat.sum()), float(np.abs(flat).sum()), float((flat * weights).sum())],
        }

    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(repr((name, value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    else:
        digest = hashlib.sha256(repr((name, value)).encode())
    return {"sha256": digest.hexdigest()}


def batch_fingerprint(batch) -> dict[str, object]:
    """Record each field in declaration order, using C order for array elements."""
    return {
        field.name: _fingerprint(field.name, getattr(batch, field.name)) for field in fields(batch)
    }


def _matches(actual: object, expected: object) -> bool:
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _matches(a, e) for a, e in zip(actual, expected, strict=True)
        )
    if (
        isinstance(actual, dict)
        and isinstance(expected, dict)
        and "sums" in actual
        and "sums" in expected
    ):
        tolerance = 1e-5 * (1 + expected["sums"][1])
        return (
            ("dtype" not in expected or actual["dtype"] == expected["dtype"])
            and actual["shape"] == expected["shape"]
            and actual["nan"] == expected["nan"]
            and all(
                isclose(a, e, rel_tol=0, abs_tol=tolerance)
                for a, e in zip(actual["sums"], expected["sums"], strict=True)
            )
        )
    return actual == expected


def assert_fingerprint(batch, expected: dict[str, object]) -> None:
    """Report all fields whose fingerprints differ from the reference."""
    actual = batch_fingerprint(batch)
    differences = []
    for name in actual | expected:
        before = expected.get(name, "<missing>")
        after = actual.get(name, "<missing>")
        if not _matches(after, before):
            differences.append(f"{name}: expected {before!r}, got {after!r}")
    assert not differences, "\n".join(differences)
