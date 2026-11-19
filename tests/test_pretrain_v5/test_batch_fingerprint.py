"""The portable batch pin detects meaningful changes and tolerates rounding."""

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from tests.helpers.batch_fingerprint import assert_fingerprint, batch_fingerprint


@dataclass
class Batch:
    ids: torch.Tensor
    name: str
    values: np.ndarray | torch.Tensor
    parts: list[torch.Tensor]


@pytest.fixture
def batch():
    return Batch(
        ids=torch.tensor([1, 2]),
        name="target",
        values=np.array([1.0, -2.0, np.nan, 4.0]),
        parts=[torch.tensor([1.0, 2.0]), torch.tensor([True, False])],
    )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("ids", torch.tensor([1, 3])),
        ("name", "other"),
        ("values", np.array([1.0, -2.0, np.nan, 4.004])),
        ("values", np.array([-2.0, 1.0, np.nan, 4.0])),
        ("values", np.array([1.0, np.nan, -2.0, 4.0])),
        ("values", np.array([[1.0, -2.0], [np.nan, 4.0]])),
        ("values", np.array([1.0, -2.0, 0.0, 4.0])),
        ("parts", [torch.tensor([True, False]), torch.tensor([1.0, 2.0])]),
    ],
    ids=["integer", "string", "float", "swap", "nan-moved", "shape", "nan-count", "list-order"],
)
def test_detects_changes(batch, field, changed):
    expected = batch_fingerprint(batch)
    with pytest.raises(AssertionError, match=rf"\b{field}\b"):
        assert_fingerprint(replace(batch, **{field: changed}), expected)


def test_tolerates_float_rounding(batch):
    rounded = batch.values.copy()
    rounded[3] *= 1 + 1e-9
    assert_fingerprint(replace(batch, values=rounded), batch_fingerprint(batch))


@pytest.mark.parametrize("as_array", [np.asarray, torch.from_numpy], ids=["numpy", "torch"])
def test_detects_float_dtype_changes_and_accepts_legacy_entries(batch, as_array):
    values = batch.values.astype(np.float32)
    original = replace(batch, values=as_array(values))
    promoted = replace(original, values=as_array(values.astype(np.float64)))
    expected = batch_fingerprint(original)
    with pytest.raises(AssertionError, match=r"^values:"):
        assert_fingerprint(promoted, expected)

    del expected["values"]["dtype"]
    assert_fingerprint(original, expected)
    assert_fingerprint(promoted, expected)


def test_records_fields_in_order_with_float64_summaries(batch):
    result = batch_fingerprint(batch)
    assert list(result) == ["ids", "name", "values", "parts"]
    assert result["values"] == {
        "dtype": np.dtype("float64").str,
        "shape": [4],
        "nan": 1,
        "sums": [3.0, 7.0, 3.25],
    }
    floating, discrete = result["parts"]
    assert floating == {
        "dtype": np.dtype("float32").str,
        "shape": [2],
        "nan": 0,
        "sums": [3.0, 3.0, 2.5],
    }
    assert len(discrete["sha256"]) == 64


def test_reports_every_difference_with_both_values(batch):
    expected = batch_fingerprint(batch)
    del expected["name"]
    expected["unknown"] = {"sha256": "unknown"}
    changed = replace(batch, values=np.zeros_like(batch.values))
    actual = batch_fingerprint(changed)
    with pytest.raises(AssertionError) as error:
        assert_fingerprint(changed, expected)

    message = str(error.value)
    for field in ("name", "unknown", "values"):
        assert field in message
    for entry in (actual["name"], expected["unknown"], actual["values"], expected["values"]):
        assert repr(entry) in message
