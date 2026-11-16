"""Exact, dependency-free validation of fixed forecast-average weights."""

import re
from collections.abc import Sequence
from fractions import Fraction


def parse_member_weights(values: Sequence[str] | None, count: int) -> list[Fraction] | None:
    """Accept decimal text only, requiring strictly positive weights summing to one."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or len(values) != count:
        raise ValueError("member weight count must equal member count")
    fractions = []
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(
            r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value
        ):
            raise ValueError("member weights must be finite decimal text (no fractions syntax)")
        weight = Fraction(value)
        if weight <= 0:
            raise ValueError("member weights must be strictly positive")
        fractions.append(weight)
    if sum(fractions) != 1:
        raise ValueError("member weights must sum exactly to 1")
    return fractions
