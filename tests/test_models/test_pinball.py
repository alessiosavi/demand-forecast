"""Tests for doubled pinball loss over observed, loss-eligible label cells."""

import numpy as np
import pytest
import torch

from demand_forecast.data.scaling import sample_frame, seasonal_error
from demand_forecast.models.losses import pinball_loss


def test_pinball_equals_sql_when_scale_is_seasonal_error():
    loss, n = pinball_loss(
        torch.tensor([[[13.0 / 4]]]),
        torch.tensor([[14.0 / 4]]),
        torch.tensor([[True]]),
        torch.tensor([True]),
        torch.tensor([0.5]),
    )
    assert loss.item() == pytest.approx(0.25)
    assert n == 1


def test_metric_scale_pinball_cells_align_with_sql_but_weighted_mean_has_own_denominator():
    past = np.stack([np.arange(28), np.arange(28) ** 2]).astype(float)
    frames = [sample_frame(p[-4:], p, 1) for p in past]
    scales = torch.tensor([frame.scale for frame in frames], dtype=torch.float64)
    task_scales = torch.tensor([seasonal_error(p, 7) for p in past], dtype=torch.float64)
    weights = scales / task_scales
    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    errors = torch.tensor([[2.0, -3], [-5, 4]], dtype=torch.float64)
    pred = torch.zeros(2, 2, 3, dtype=torch.float64, requires_grad=True)
    labels = errors / scales[:, None]
    loss, mass = pinball_loss(
        pred,
        labels,
        torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        quantiles,
        weights=weights,
    )
    raw_errors = errors[:, :, None]
    sql_cells = 2 * torch.maximum(quantiles * raw_errors, (quantiles - 1) * raw_errors)
    sql_cells /= task_scales[:, None, None]
    # Weighting restores each SQL numerator cell, while the specified weighted
    # mean divides by weight mass rather than SQL's selected-cell count.
    reference = sql_cells.sum() / (2 * weights.sum() * len(quantiles))
    assert loss.item() == pytest.approx(reference.item(), abs=1e-6)
    assert mass == pytest.approx(2 * weights.sum().item())
    assert loss.item() != pytest.approx(sql_cells.mean().item())
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_zero_weight_excludes_nan_cells_and_all_zero_has_finite_zero_gradients():
    for weights in (torch.tensor([1.0, 0.0]), torch.zeros(2)):
        pred = torch.tensor([[[0.0]], [[float("nan")]]], requires_grad=True)
        loss, mass = pinball_loss(
            pred,
            torch.tensor([[2.0], [float("nan")]]),
            torch.ones(2, 1, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.tensor([0.5]),
            weights=weights,
        )
        assert loss.item() == (2 if weights[0] else 0)
        assert mass == weights[0].item()
        loss.backward()
        assert torch.isfinite(pred.grad).all() and pred.grad[1].item() == 0


@pytest.mark.parametrize(
    ("label_mask", "loss_eligible"),
    [
        pytest.param(
            [[True, True, True], [True, True, True]],
            [False, False],
            id="no-eligible-samples",
        ),
        pytest.param(
            [[False, False, False], [False, False, False]],
            [True, True],
            id="no-observed-labels",
        ),
        pytest.param(
            [[True, True, True], [False, False, False]],
            [False, True],
            id="no-overlap",
        ),
    ],
)
def test_no_eligible_cells_gives_zero_loss_with_grad_path(label_mask, loss_eligible):
    pred = torch.zeros(2, 3, 1, requires_grad=True)
    loss, n = pinball_loss(
        pred,
        torch.zeros(2, 3),
        torch.tensor(label_mask),
        torch.tensor(loss_eligible),
        torch.tensor([0.5]),
    )
    assert loss.item() == 0.0
    assert n == 0
    assert loss.requires_grad
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("excluded_value", [float("nan"), 1_000_000.0], ids=["nan", "large"])
def test_pinball_pools_selected_cells_and_knots_with_masked_gradients(excluded_value):
    pred = torch.tensor(
        [
            [[0.0, 1.0, 4.0], [0.0, 2.0, 7.0]],
            [[2.0, 6.0, 7.0], [excluded_value] * 3],
            [[excluded_value] * 3, [excluded_value] * 3],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor(
        [[2.0, 3.0], [4.0, excluded_value], [excluded_value, excluded_value]],
        dtype=torch.float64,
    )
    loss, n = pinball_loss(
        pred,
        labels,
        torch.tensor([[True, True], [True, False], [True, True]]),
        torch.tensor([True, True, False]),
        torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64),
    )
    # Selected cells contribute [0.4, 1, 0.4], [0.6, 1, 0.8], [0.4, 2, 0.6].
    assert loss.item() == pytest.approx(7.2 / 9)
    assert n == 3
    assert isinstance(n, int)
    loss.backward()
    expected_grad = (
        torch.tensor(
            [
                [[-0.2, -1.0, 0.2], [-0.2, -1.0, 0.2]],
                [[-0.2, 1.0, 0.2], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        / 9
    )
    torch.testing.assert_close(pred.grad, expected_grad)
