"""Multi-head attention with explicit allowed keys and source gate biases."""

import torch
from torch import Tensor, nn

from demand_forecast.models.v2 import RotaryPositionEmbedding


class MaskedAttention(nn.Module):
    """Attention over the last sequence axis; fully masked queries return zero."""

    def __init__(self, d_model: int, nhead: int, dropout: float, rope: bool) -> None:
        super().__init__()
        if nhead < 1 or d_model % nhead:
            raise ValueError("d_model must be divisible by a positive nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if rope and self.head_dim % 2:
            raise ValueError("RoPE requires an even dimension per attention head")
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim) if rope else None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_positions: Tensor | None = None,
        key_positions: Tensor | None = None,
        mask: Tensor | None = None,
        key_bias: Tensor | None = None,
    ) -> Tensor:
        """Use bool masks where True permits attention and additive key biases.

        Leading batch axes broadcast. Positions have shape [..., L] and masks
        [..., Lq, Lk], without a head axis.
        """
        dead_keys = ~mask.any(dim=-2) if mask is not None else None
        if key_bias is not None:
            blocked = torch.isneginf(key_bias)
            dead_keys = blocked if dead_keys is None else dead_keys | blocked
        if dead_keys is not None:
            # Sanitize before projection: even zero gradients times NaN are NaN.
            key = key.masked_fill(dead_keys.unsqueeze(-1), 0)
            value = value.masked_fill(dead_keys.unsqueeze(-1), 0)
        q = self.q_proj(query).unflatten(-1, (self.nhead, self.head_dim)).transpose(-3, -2)
        k = self.k_proj(key).unflatten(-1, (self.nhead, self.head_dim)).transpose(-3, -2)
        v = self.v_proj(value).unflatten(-1, (self.nhead, self.head_dim)).transpose(-3, -2)
        if self.rope is not None:
            if query_positions is None or key_positions is None:
                raise ValueError("RoPE requires both query_positions and key_positions")
            q = self.rope(q, query_positions.unsqueeze(-2))
            k = self.rope(k, key_positions.unsqueeze(-2))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim**-0.5
        if key_bias is not None:
            scores = scores + key_bias.unsqueeze(-2).unsqueeze(-2)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-3), -torch.inf)
        all_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        weights = scores.masked_fill(all_masked, 0).softmax(dim=-1)
        weights = self.dropout(weights.masked_fill(all_masked, 0))
        attended = torch.matmul(weights, v).transpose(-3, -2).flatten(-2)
        # Suppress the output bias too, including when no keys exist at all.
        return self.out_proj(attended).masked_fill(all_masked.all(dim=-3), 0)
