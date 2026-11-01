"""Frozen dynamic helper from 2863356, used only as a batch byte-identity oracle."""

import numpy as np


def _dynamic_values(
    source: np.ndarray | None, start: int, origin: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice one dynamic column and standardise using only its observed past."""
    values = np.full(stop - start, np.nan, dtype=np.float64)
    if source is None:
        return np.zeros(stop - start, dtype=np.float32), np.zeros(stop - start, dtype=bool)
    end = min(stop, len(source))
    if end > start:
        values[: end - start] = source[start:end]
    past = np.asarray(source[:origin], dtype=np.float64)
    past = past[~np.isnan(past)]
    # An unobserved past uses the identity transform. Classify indicators from
    # the past as well, so future values cannot change the context transform.
    if past.size and not np.all((past == 0) | (past == 1)):
        values = (values - past.mean()) / (past.std() or 1.0)
    mask = ~np.isnan(values)
    return np.where(mask, values, 0).astype(np.float32), mask


def assert_batch_matches_reference(monkeypatch, panel, items, horizon, max_context, encoder):
    from dataclasses import fields

    from demand_forecast.data import sampler

    actual = sampler.make_batch(panel, items, horizon, max_context, encoder)

    def previous_helper(*args):
        values, mask = _dynamic_values(*args)
        return values, mask, sampler.DynamicDiagnostics(cold=False, clipped=0, floored=False)

    with monkeypatch.context() as patch:
        patch.setattr(sampler, "_dynamic_values", previous_helper)
        expected = sampler.make_batch(panel, items, horizon, max_context, encoder)
    assert not actual.cold_covariates.any() and not actual.clipped_covariates.any()
    assert not actual.floored_covariates.any()
    for field in fields(actual):
        if field.name in ("cold_covariates", "clipped_covariates", "floored_covariates"):
            continue
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        pairs = (
            [(left[key], right[key]) for key in left] if isinstance(left, dict) else [(left, right)]
        )
        for left_tensor, right_tensor in pairs:
            assert left_tensor.dtype == right_tensor.dtype, field.name
            assert left_tensor.shape == right_tensor.shape, field.name
            assert left_tensor.numpy().tobytes() == right_tensor.numpy().tobytes(), field.name
