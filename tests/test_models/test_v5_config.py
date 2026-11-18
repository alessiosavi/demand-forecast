"""Configuration values have finite, canonical numeric representations."""

import json
from dataclasses import asdict, fields, replace

import pytest

from demand_forecast.models.v5 import V5Config

NUMERIC_FIELDS = [field for field in fields(V5Config) if field.type in (int, float)]
INTEGER_FIELDS = [field.name for field in NUMERIC_FIELDS if field.type is int]


def test_quantiles_require_median():
    with pytest.raises(ValueError, match="quantiles.*0.5"):
        V5Config(quantiles=(0.1, 0.9))


@pytest.mark.parametrize("field", [field.name for field in NUMERIC_FIELDS])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "4"])
def test_numeric_fields_reject_nonfinite_and_wrong_kinds(field, value):
    with pytest.raises(ValueError, match=field):
        replace(V5Config(), **{field: value})


@pytest.mark.parametrize("field", INTEGER_FIELDS)
def test_integer_fields_reject_fractions(field):
    with pytest.raises(ValueError, match=field):
        replace(V5Config(), **{field: getattr(V5Config(), field) + 0.5})


@pytest.mark.parametrize("field", INTEGER_FIELDS)
def test_integral_floats_are_stored_as_ints(field):
    original = V5Config()
    canonical = replace(original, **{field: float(getattr(original, field))})
    assert type(getattr(canonical, field)) is int
    assert canonical == original
    assert json.dumps(asdict(canonical)) == json.dumps(asdict(original))


@pytest.mark.parametrize("field", [field.name for field in NUMERIC_FIELDS if field.type is float])
def test_float_fields_store_ints_as_floats(field):
    value = 2 if field in ("width_factor", "gate_init") else 0
    original = replace(V5Config(), **{field: float(value)})
    canonical = replace(original, **{field: value})
    assert type(getattr(canonical, field)) is float
    assert canonical == original
    assert json.dumps(asdict(canonical)) == json.dumps(asdict(original))


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), float("inf")])
def test_quantile_levels_reject_wrong_kinds_and_nonfinite(value):
    with pytest.raises(ValueError, match="quantiles"):
        V5Config(quantiles=(value, 0.5, 0.9))
