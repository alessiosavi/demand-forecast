"""Independent candidate gates and a shared sparsity mask for summaries."""

import math

import torch
from torch import Tensor, nn

from demand_forecast.data.episode.schema import (
    KIND_CATEGORY,
    N_SUMMARY,
    ROLE_FORECAST,
    ROLE_KNOWN,
    ROLE_PANEL,
)

from .config import V5Config


class HardConcreteGate(nn.Module):
    """Stretched binary concrete samples with deterministic clipped eval gates."""

    def forward(self, logits: Tensor) -> Tensor:
        """Return gates in [0, 1], with exact zeros and ones after clipping."""
        if self.training:
            eps = torch.finfo(logits.dtype).eps
            uniform = torch.rand_like(logits).clamp(eps, 1 - eps)
            concrete = torch.sigmoid((logits + uniform.log() - torch.log1p(-uniform)) / (2 / 3))
        else:
            concrete = torch.sigmoid(logits)
        return (concrete * 1.2 - 0.1).clamp(0, 1)

    def expected_l0(self, logits: Tensor) -> Tensor:
        """Probability that the gate is not zero."""
        return torch.sigmoid(logits - (2 / 3) * math.log(0.1 / 1.1))


class GateNet(nn.Module):
    """Score each candidate without reading any other candidate or network state."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.known_gate_floor = config.known_gate_floor
        self.kind_embedding = nn.Embedding(KIND_CATEGORY + 1, 8)
        self.role_embedding = nn.Embedding(ROLE_PANEL + 1, 8)
        self.net = nn.Sequential(
            nn.Linear(4 * N_SUMMARY + 1 + 8 + 8, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.Linear(config.d_model, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, config.gate_init)
        self.gate = HardConcreteGate()

    def forward(
        self,
        summary: Tensor,
        flags: Tensor,
        score: Tensor,
        kind: Tensor,
        role: Tensor,
        ref_summary: Tensor,
        ref_flags: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return candidate gates and their expected sparsity cost."""
        while ref_summary.ndim < summary.ndim:
            ref_summary = ref_summary.unsqueeze(-2)
            ref_flags = ref_flags.unsqueeze(-2)
        inputs = torch.cat(
            (
                summary,
                flags.to(summary.dtype),
                score.unsqueeze(-1),
                self.kind_embedding(kind),
                self.role_embedding(role),
                ref_summary.expand_as(summary),
                ref_flags.expand_as(flags).to(summary.dtype),
            ),
            dim=-1,
        )
        logits = self.net(inputs).squeeze(-1)
        gate = self.gate(logits)
        known = self.known_gate_floor + (1 - self.known_gate_floor) * torch.sigmoid(logits)
        gate = torch.where(role == ROLE_KNOWN, known, gate)
        gate = torch.where(role == ROLE_FORECAST, torch.ones_like(gate), gate)
        protected = (role == ROLE_FORECAST) | (role == ROLE_KNOWN)
        l0 = self.gate.expected_l0(logits).masked_fill(protected, 0)
        return gate, l0


class GlobalMask(nn.Module):
    """One learned gate per summary coordinate, shared across all summaries."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.full((N_SUMMARY,), config.gate_init))
        self.gate = HardConcreteGate()

    def forward(self, summary: Tensor, flags: Tensor) -> tuple[Tensor, Tensor]:
        """Sample once per coordinate and clear flags for exactly zero gates."""
        mask = self.gate(self.logits)
        return summary * mask, flags & (mask > 0)

    def expected_l0(self) -> Tensor:
        """Return nonzero probabilities for all summary coordinates."""
        return self.gate.expected_l0(self.logits)
