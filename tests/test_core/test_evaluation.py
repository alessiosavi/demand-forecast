"""Frozen-reference parity and ownership of reusable CPU validation inputs."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core import evaluation
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.sampler import Batch
from demand_forecast.models.base import ModelSpec
from tests.helpers.models import build_v2
from tests.helpers.panels import weekly_panel
from tests.helpers.reference_batch import assert_batches_equal, reference_evaluate


@pytest.fixture
def validation_case():
    panel = weekly_panel([8, 3, 9])
    panel.series[2].target[7] = np.nan
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    forecast = ForecastConfig(horizon=4, max_context=2)
    model = build_v2(ModelSpec.from_panel(panel, encoder, forecast, 16, 4))
    args = (panel, np.array([2, 1, 0]), panel.timestamps(0)[6], 4, 2, encoder)
    return model, forecast.quantiles, args


def test_evaluate_prepared_matches_reference(validation_case):
    model, quantiles, args = validation_case
    prepared = evaluation.prepare_validation(*args, batch_size=2)
    assert [batch.size for batch in prepared.batches] == [2, 1]
    assert prepared.y_true.dtype == np.float64
    assert np.isnan(prepared.y_true[1]).all()  # ends before the origin
    assert np.isnan(prepared.y_true[2, 2:]).all()  # partial horizon
    expected = reference_evaluate(model, *args, quantiles, batch_size=2)
    np.testing.assert_equal(evaluation.evaluate_prepared(model, prepared, quantiles), expected)
    np.testing.assert_equal(evaluation.evaluate(model, *args, quantiles, batch_size=2), expected)
    assert model.training


def test_prepared_batches_survive_cpu_evaluation(validation_case):
    model, quantiles, args = validation_case
    prepared = evaluation.prepare_validation(*args, batch_size=2)
    copies = [batch.clone() for batch in prepared.batches]
    true = prepared.y_true.copy()
    past = [values.copy() for values in prepared.y_past]
    first = evaluation.evaluate_prepared(model, prepared, quantiles)
    second = evaluation.evaluate_prepared(model, prepared, quantiles)
    np.testing.assert_equal(first, second)
    np.testing.assert_equal(prepared.y_true, true)
    for actual, expected in zip(prepared.y_past, past):
        np.testing.assert_equal(actual, expected)
    nbytes = prepared.y_true.nbytes + sum(values.nbytes for values in prepared.y_past)
    for batch, clone in zip(prepared.batches, copies):
        assert_batches_equal(batch, clone)
        for field in fields(Batch):
            value = getattr(batch, field.name)
            tensors = value.values() if isinstance(value, dict) else [value]
            for tensor in tensors:
                assert tensor.device.type == "cpu"
                nbytes += tensor.numel() * tensor.element_size()
    assert prepared.nbytes == nbytes


def test_prepared_default_chunks_keep_item_order(validation_case, monkeypatch):
    model, quantiles, args = validation_case
    panel, _, origin, horizon, context, encoder = args
    items = np.tile([2, 0, 1], 171)
    prepared = evaluation.prepare_validation(panel, items, origin, horizon, context, encoder)
    calls = []

    def predict(batch):
        assert not torch.is_grad_enabled() and not model.training
        calls.extend(batch.series_index.tolist())
        return torch.ones(batch.size, horizon, len(quantiles))

    monkeypatch.setattr(model, "predict", predict)
    evaluation.evaluate_prepared(model, prepared, quantiles)
    assert [batch.size for batch in prepared.batches] == [512, 1]
    assert calls == items.tolist()


def test_evaluate_prepared_restores_training_after_failure(validation_case, monkeypatch):
    model, quantiles, args = validation_case
    prepared = evaluation.prepare_validation(*args)

    def fail(batch):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(RuntimeError, match="prediction failed"):
        evaluation.evaluate_prepared(model, prepared, quantiles)
    assert model.training
