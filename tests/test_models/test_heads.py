"""Tests for the median-centred, non-crossing quantile head."""

import math

import pytest
import torch

from demand_forecast.models.heads import MonotoneQuantileHead, monotone_quantiles


def test_head_outputs_sorted_and_median_is_free():
    head = MonotoneQuantileHead(8, (0.1, 0.5, 0.9))
    out = head(torch.randn(5, 3, 8))
    assert out.shape == (5, 3, 3)
    assert (out.diff(dim=-1) >= 0).all()

    mono = monotone_quantiles(torch.tensor([[-3.0, 0.7, 2.0]]), (0.1, 0.5, 0.9), 2.0)
    assert mono[0, 1].item() == pytest.approx(0.7)
    assert mono[0, 2] > mono[0, 1] > mono[0, 0]


def test_head_gradients_reach_all_parameters():
    head = MonotoneQuantileHead(4, (0.1, 0.5, 0.9))
    head(torch.randn(2, 4)).sum().backward()
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert (parameter.grad.abs().sum(dim=-1) > 0).all()


@pytest.mark.parametrize("width_factor", [0.5, 2.0, 4.0])
def test_head_respects_unequal_spacing_and_configured_width(width_factor):
    head = MonotoneQuantileHead(4, [0.1, 0.2, 0.5, 0.9], width_factor).double()
    assert head.quantiles == (0.1, 0.2, 0.5, 0.9)
    assert isinstance(head.quantiles, tuple)
    assert all(isinstance(level, float) for level in head.quantiles)
    assert isinstance(head.projection, torch.nn.Linear)
    with torch.no_grad():
        head.projection.weight.zero_()
        head.projection.bias.copy_(torch.tensor([0.0, 0.0, 0.7, 0.0], dtype=torch.float64))

    out = head(torch.ones(4, dtype=torch.float64))
    expected = 0.7 + width_factor * math.log(2) * torch.tensor(
        [-0.4, -0.3, 0.0, 0.4], dtype=torch.float64
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("levels", [(0.5,), (0.1, 0.5), (0.5, 0.6, 0.9)])
def test_median_can_be_the_only_level_or_at_either_edge(levels):
    raw = torch.randn(2, 3, len(levels), dtype=torch.float64, requires_grad=True)
    out = monotone_quantiles(raw, levels, 2.0)
    assert out.shape == raw.shape
    assert out.dtype == raw.dtype
    assert out.device == raw.device
    assert (out.diff(dim=-1) >= 0).all()
    assert torch.equal(out[..., levels.index(0.5)], raw[..., levels.index(0.5)])
    out.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert (raw.grad.abs() > 0).all()


def test_median_level_is_required():
    with pytest.raises(ValueError):
        monotone_quantiles(torch.zeros(2, 2), (0.1, 0.9), 2.0)


def test_extreme_finite_widths_saturate_without_crossing():
    big = torch.finfo(torch.float32).max
    raw = torch.tensor([[big, big, 0.7, big, big]])
    out = monotone_quantiles(raw, (0.1, 0.2, 0.5, 0.6, 0.9), 10.0)
    assert torch.isfinite(out).all()
    assert (out.diff(dim=-1) >= 0).all()
    assert out[0, 0].item() == -big
    assert out[0, -1].item() == big
    assert out[0, 2].item() == pytest.approx(0.7)
