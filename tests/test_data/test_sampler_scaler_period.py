"""Independent frame periods and context-only metric loss weights."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.sampler import BatchBuilder, make_batch
from demand_forecast.data.scaling import seasonal_error
from tests.helpers.panels import weekly_panel
from tests.helpers.reference_batch import assert_batches_equal, reference_make_batch


def test_explicit_task_period_is_byte_identical_to_frozen_batch():
    panel = weekly_panel([40, 32])
    panel.seasonal_period = 7
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    items = [(0, 28), (1, 24)]
    default = make_batch(panel, items, 4, 8, encoder)
    explicit = make_batch(panel, items, 4, 8, encoder, scaler_period=7)
    reference = reference_make_batch(panel, items, 4, 8, encoder)
    assert_batches_equal(default, reference)
    assert_batches_equal(explicit, reference)
    for field in fields(default):
        left, right = getattr(explicit, field.name), getattr(reference, field.name)
        pairs = zip(left.values(), right.values()) if isinstance(left, dict) else [(left, right)]
        for left, right in pairs:
            assert left.numpy().tobytes() == right.numpy().tobytes(), field.name


def test_matching_period_has_unit_weights_and_identical_loss():
    from demand_forecast.models.losses import pinball_loss

    panel = weekly_panel([40, 32])
    panel.seasonal_period = 7
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    batch = make_batch(panel, [(0, 28), (1, 24)], 4, 8, encoder, loss_weight="metric_scale")
    assert torch.equal(batch.loss_weight, torch.ones(2))
    pred = torch.zeros(2, 4, 3)
    args = (
        pred,
        batch.labels,
        batch.label_mask,
        batch.loss_eligible,
        torch.tensor([0.1, 0.5, 0.9]),
    )
    plain, n = pinball_loss(*args)
    weighted, mass = pinball_loss(*args, weights=batch.loss_weight)
    assert torch.equal(plain, weighted) and mass == n


def test_frame_and_weights_use_whole_past_and_ignore_labels():
    panel = weekly_panel([40])
    panel.seasonal_period = 7
    panel.series[0].target = np.arange(40, dtype=float)
    encoder = StaticEncoder().fit(panel, [panel.series[0].series_id])
    builder = BatchBuilder(panel, 4, 3, encoder, scaler_period=1, loss_weight="metric_scale")
    before = builder.build([(0, 28)])
    assert before.scale.item() == 1 and before.loc.item() == 26
    assert before.loss_weight.item() == pytest.approx(1 / 7)
    assert seasonal_error(panel.series[0].target[:28], 7) == 7
    panel.series[0].target[28:] = 1e10
    after = builder.build([(0, 28)])
    assert torch.equal(before.target, after.target)
    assert torch.equal(before.scale, after.scale)
    assert torch.equal(before.loss_weight, after.loss_weight)


def test_builder_counts_clipped_and_zero_or_undefined_task_scales():
    panel = weekly_panel([40] * 3)
    panel.seasonal_period = 7
    seasonal = np.resize(np.array([0, 10, 0, 10, 0, 10, 0], dtype=float), 40)
    panel.series[0].target = seasonal + np.arange(40) * 1e-4
    panel.series[1].target = seasonal
    panel.series[2].target = np.arange(40, dtype=float)
    encoder = StaticEncoder().fit(panel, [s.series_id for s in panel.series])
    builder = BatchBuilder(panel, 2, 8, encoder, scaler_period=1, loss_weight="metric_scale")
    batch = builder.build([(0, 21), (1, 21), (2, 3)])
    assert batch.loss_weight.tolist() == [10, 0, 0]
    assert builder.loss_weight_clipped == 1 and builder.loss_weight_excluded == 2
    # The task-scale exclusion is additional to the frame's eligibility.
    assert batch.loss_eligible.all()
