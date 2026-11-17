"""Calendar and kind-specific patch encoders, with no dataset vocabulary."""

import torch
from torch import Tensor, nn

from demand_forecast.data.episode.schema import (
    CALENDAR_SIZES,
    CH_OBSERVED,
    CH_VALUE_STATS,
    KIND_CATEGORY,
    N_PHASES,
    PATCH,
    ROLE_PANEL,
    VALUE_STATS,
)

from .config import V5Config


class CalendarEmbedding(nn.Module):
    """Combine phase projections with learned tables for fixed calendar fields."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.phase_projection = nn.Linear(N_PHASES, config.d_calendar)
        self.tables = nn.ModuleList(
            nn.Embedding(size, config.d_calendar) for size in CALENDAR_SIZES
        )

    def forward(self, phases: Tensor, ids: Tensor) -> Tensor:
        """Encode every step; an inactive field (id -1) contributes zero."""
        result = self.phase_projection(phases)
        for field, table in enumerate(self.tables):
            index = ids[..., field]
            result = result + table(index.clamp_min(0)).masked_fill((index < 0).unsqueeze(-1), 0)
        return result


class PatchEncoder(nn.Module):
    """Read each kind's permitted channels and preserve within-patch timing."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.value_embedding = nn.Sequential(nn.Linear(1 + VALUE_STATS, config.d_step), nn.GELU())
        self.encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(
                    PATCH
                    * (
                        (config.d_step if kind == KIND_CATEGORY else CH_VALUE_STATS)
                        + config.d_calendar
                    ),
                    config.d_model,
                ),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            for kind in range(KIND_CATEGORY + 1)
        )
        self.kind_embedding = nn.Embedding(KIND_CATEGORY + 1, config.d_model)
        self.role_embedding = nn.Embedding(ROLE_PANEL + 1, config.d_model)

    def forward(self, x: Tensor, cal: Tensor, kind: Tensor, role: Tensor) -> Tensor:
        """Encode patches [..., 16, 39] into tokens [..., d_model]."""
        category = self.value_embedding(
            torch.cat((x[..., CH_OBSERVED : CH_OBSERVED + 1], x[..., CH_VALUE_STATS:]), dim=-1)
        )
        number_patch = torch.cat((x[..., :CH_VALUE_STATS], cal), dim=-1).flatten(-2)
        category_patch = torch.cat((category, cal), dim=-1).flatten(-2)
        tokens = torch.zeros_like(self.kind_embedding(kind))
        for index, encoder in enumerate(self.encoders):
            encoded = encoder(category_patch if index == KIND_CATEGORY else number_patch)
            tokens = tokens + encoded.masked_fill((kind != index).unsqueeze(-1), 0)
        return tokens + self.kind_embedding(kind) + self.role_embedding(role)
