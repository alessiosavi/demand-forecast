"""Tests for doubled pinball loss over observed, loss-eligible label cells."""

import pytest
import torch

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
