"""Gated static pooling, identity-initialised conditioning and per-row routing."""

import math

import torch
from torch import Tensor, nn

from demand_forecast.data.episode.schema import FREQ_CLASSES, N_SUMMARY, N_VALUE_TOKEN

from .config import V5Config


def _mlp(width: int, d_model: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, d_model), nn.GELU(), nn.Linear(d_model, d_model))


class StaticContext(nn.Module):
    """Pool value tokens and identity-free static numbers with candidate weights."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.attribute_encoder = _mlp(N_VALUE_TOKEN, config.d_model)
        self.number_encoder = _mlp(2, config.d_model)
        self.reference_encoder = _mlp(2 * N_SUMMARY, config.d_model)
        self.frequency_embedding = nn.Embedding(len(FREQ_CLASSES), config.d_model)

    def forward(
        self,
        attr_stats: Tensor,
        attr_weight: Tensor,
        static_num: Tensor,
        static_num_flags: Tensor,
        static_num_weight: Tensor,
        ref_summary: Tensor,
        ref_flags: Tensor,
        freq_class: Tensor,
    ) -> Tensor:
        """Return [B, d_model], including a zero pooled term for empty context."""
        attributes = self.attribute_encoder(attr_stats)
        numbers = self.number_encoder(
            torch.stack((static_num, static_num_flags.to(static_num.dtype)), dim=-1)
        )
        total = (attributes * attr_weight.unsqueeze(-1)).sum(dim=-2)
        total = total + (numbers * static_num_weight.unsqueeze(-1)).sum(dim=-2)
        weight = attr_weight.sum(dim=-1) + static_num_weight.sum(dim=-1)
        pooled = total / weight.clamp_min(torch.finfo(total.dtype).tiny).unsqueeze(-1)
        pooled = pooled.masked_fill((weight == 0).unsqueeze(-1), 0)
        reference = self.reference_encoder(
            torch.cat((ref_summary, ref_flags.to(ref_summary.dtype)), dim=-1)
        )
        return pooled + reference + self.frequency_embedding(freq_class)


class ScaleShift(nn.Module):
    """Condition tokens by a residual affine map that initially is the identity."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.g = nn.Linear(config.d_model, config.d_model)
        self.b = nn.Linear(config.d_model, config.d_model)
        for layer in (self.g, self.b):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        """Broadcast one context vector over all token axes within each episode."""
        shape = (context.shape[0],) + (1,) * (tokens.ndim - 2) + (context.shape[-1],)
        return tokens * (1 + self.g(context).reshape(shape)) + self.b(context).reshape(shape)


class ExpertRouter(nn.Module):
    """Choose and mix top-k experts once per row using cutoff-only inputs."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.net = nn.Sequential(
            _mlp(config.d_model + 2 * N_SUMMARY, config.d_model),
            nn.Linear(config.d_model, config.n_experts),
        )

    def forward(
        self, context: Tensor, row_summary: Tensor, flags: Tensor, row_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return sparse row weights and KL(mean routing || uniform) on real rows."""
        inputs = torch.cat(
            (
                context.unsqueeze(-2).expand(*row_summary.shape[:-1], context.shape[-1]),
                row_summary,
                flags.to(row_summary.dtype),
            ),
            dim=-1,
        )
        logits = self.net(inputs)
        selected, indices = logits.topk(self.top_k, dim=-1)
        # Under autocast softmax returns float32 for half-precision logits; scatter needs one dtype.
        probabilities = selected.softmax(dim=-1)
        weights = torch.zeros_like(logits, dtype=probabilities.dtype).scatter(
            -1, indices, probabilities
        )
        real_weights = weights.masked_fill(~row_mask.unsqueeze(-1), 0)
        mean = real_weights.sum(dim=(0, 1)) / row_mask.sum().clamp_min(1)
        # Clamp only the log argument: empty rows and unused experts contribute 0.
        balance = (
            mean * (mean.clamp_min(torch.finfo(mean.dtype).tiny).log() + math.log(self.n_experts))
        ).sum()
        return weights, balance
