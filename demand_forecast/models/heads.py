# SPDX-License-Identifier: Apache-2.0
# monotone_quantiles ported from Timestral src/timestral/model.py at commit a534bdb.
# Source: https://github.com/mistralai/timestral
# Adaptation: retain the quantile transform and add a linear projection wrapper.
# See THIRD_PARTY_NOTICES.md for attribution and the Apache-2.0 license text.
"""Monotone quantile outputs with an unconstrained median."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def monotone_quantiles(raw: Tensor, levels: tuple[float, ...], width_factor: float) -> Tensor:
    """Non-crossing quantiles from raw head outputs along the last axis (`monotone_cumsum`).

    The median channel is free. The others fan out from it as
    `median -/+ cumsum(width_factor * dp * softplus(raw))`, `dp` being the probability gap
    to the next level inward, so unequal knot spacing is respected.
    """
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    widths_t = torch.tensor(widths, device=raw.device, dtype=raw.dtype)
    inc = width_factor * widths_t * nn.functional.softplus(raw)
    median = raw[..., m : m + 1]
    up = torch.cumsum(inc[..., m + 1 :], dim=-1)
    down = torch.flip(torch.cumsum(torch.flip(inc[..., :m], dims=[-1]), dim=-1), dims=[-1])
    big = torch.finfo(raw.dtype).max
    return torch.cat([median - down, median, median + up], dim=-1).clamp(min=-big, max=big)


class MonotoneQuantileHead(nn.Module):
    """Project hidden states to sorted quantiles at increasing knots containing 0.5."""

    def __init__(self, d_model: int, quantiles: Sequence[float], width_factor: float = 2.0) -> None:
        super().__init__()
        self.quantiles = tuple(float(level) for level in quantiles)
        self.width_factor = width_factor
        self.projection = nn.Linear(d_model, len(self.quantiles))

    def forward(self, x: Tensor) -> Tensor:
        """Map `(..., d_model)` hidden states to `(..., Q)` quantile values."""
        return monotone_quantiles(self.projection(x), self.quantiles, self.width_factor)
