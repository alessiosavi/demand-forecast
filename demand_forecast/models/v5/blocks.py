"""Causal time/row encoder and per-origin horizon decoder blocks."""

import torch
from torch import Tensor, nn

from .attention import MaskedAttention
from .config import V5Config


def _ffn(config: V5Config, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.d_model, width),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(width, config.d_model),
    )


def _gate_bias(gate: Tensor) -> Tensor:
    # Avoid log(0)'s undefined backward, while keeping exact hard masking.
    return gate.clamp_min(torch.finfo(gate.dtype).tiny).log().masked_fill(gate == 0, -torch.inf)


def _rows(values: Tensor, indices: Tensor) -> Tensor:
    shape = (*indices.shape, *values.shape[2:])
    index = indices.reshape(*indices.shape, *((1,) * (values.ndim - 2))).expand(shape)
    return values.gather(1, index)


class ExpertFFN(nn.Module):
    """Mix narrow experts' hidden states with one fixed weight vector per row."""

    def __init__(self, config: V5Config) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            _ffn(config, config.dim_feedforward // config.n_experts)
            for _ in range(config.n_experts)
        )

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        """Apply [B, R, E] weights across any token axes of [B, R, ..., d]."""
        result = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            weight = weights[..., index].reshape(*weights.shape[:2], *((1,) * (x.ndim - 2)))
            result = result + expert(x) * weight
        return result


class EncoderBlock(nn.Module):
    """Pre-norm causal time attention, gated row attention and feedforward."""

    def __init__(self, config: V5Config, *, expert: bool = False) -> None:
        super().__init__()
        self.time_attention = MaskedAttention(config.d_model, config.nhead, config.dropout, True)
        self.row_attention = MaskedAttention(config.d_model, config.nhead, config.dropout, False)
        self.norms = nn.ModuleList(nn.LayerNorm(config.d_model) for _ in range(3))
        self.feedforward = ExpertFFN(config) if expert else _ffn(config, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tokens: Tensor,
        patch_valid: Tensor,
        row_mask: Tensor,
        row_gate: Tensor,
        expert_weights: Tensor | None,
    ) -> Tensor:
        """Encode [B, R, P, d] without reading any patch after the query."""
        valid = patch_valid & row_mask[..., None] & (row_gate > 0)[..., None]
        tokens = tokens.masked_fill(~valid[..., None], 0)
        positions = torch.arange(tokens.shape[2], device=tokens.device)
        causal = positions[:, None] >= positions[None, :]
        mask = valid[..., :, None] & valid[..., None, :] & causal
        normed = self.norms[0](tokens)
        tokens = tokens + self.dropout(
            self.time_attention(
                normed,
                normed,
                normed,
                query_positions=positions,
                key_positions=positions,
                mask=mask,
            )
        )
        normed = self.norms[1](tokens).transpose(1, 2)
        row_valid = valid.transpose(1, 2)
        attended = self.row_attention(
            normed,
            normed,
            normed,
            mask=row_valid[..., :, None] & row_valid[..., None, :],
            key_bias=_gate_bias(row_gate)[:, None, :],
        )
        tokens = tokens + self.dropout(attended.transpose(1, 2))
        normed = self.norms[2](tokens)
        if isinstance(self.feedforward, ExpertFFN):
            if expert_weights is None:
                raise ValueError("expert weights are required for an expert block")
            hidden = self.feedforward(normed, expert_weights)
        else:
            hidden = self.feedforward(normed)
        return (tokens + self.dropout(hidden)).masked_fill(~valid[..., None], 0)


class DecoderBlock(nn.Module):
    """Horizon attention and same-row cross-attention, independently per origin."""

    def __init__(self, config: V5Config, *, expert: bool = False) -> None:
        super().__init__()
        self.horizon_attention = MaskedAttention(config.d_model, config.nhead, config.dropout, True)
        self.cross_attention = MaskedAttention(config.d_model, config.nhead, config.dropout, True)
        self.row_attention = MaskedAttention(config.d_model, config.nhead, config.dropout, False)
        self.norms = nn.ModuleList(nn.LayerNorm(config.d_model) for _ in range(4))
        self.feedforward = ExpertFFN(config) if expert else _ffn(config, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        h: Tensor,
        memory: Tensor,
        dec_rows: Tensor,
        origin_patches: Tensor,
        patch_valid: Tensor,
        horizon_mask: Tensor,
        dec_mask: Tensor,
        dec_gate: Tensor,
        expert_weights: Tensor | None,
    ) -> Tensor:
        """Decode [B, O, Rd, H, d]; origins count from the padded memory edge."""
        valid = dec_mask[:, None, :, None] & horizon_mask[:, None, None, :]
        valid = valid & (dec_gate > 0)[:, None, :, None]
        h = h.masked_fill(~valid[..., None], 0)
        positions = origin_patches[..., None] + torch.arange(h.shape[3], device=h.device)
        positions = positions[:, :, None, :]
        normed = self.norms[0](h)
        h = h + self.dropout(
            self.horizon_attention(
                normed,
                normed,
                normed,
                query_positions=positions,
                key_positions=positions,
                mask=valid[..., :, None] & valid[..., None, :],
            )
        )
        row_memory = _rows(memory, dec_rows)[:, None]
        memory_positions = torch.arange(memory.shape[2], device=h.device)
        visible = _rows(patch_valid, dec_rows)[:, None]
        visible = visible & (memory_positions < origin_patches[..., None])[:, :, None, :]
        mask = valid[..., :, None] & visible[..., None, :]
        h = h + self.dropout(
            self.cross_attention(
                self.norms[1](h),
                row_memory,
                row_memory,
                query_positions=positions,
                key_positions=memory_positions,
                mask=mask,
            )
        )
        normed = self.norms[2](h).transpose(2, 3)
        row_valid = valid.transpose(2, 3)
        attended = self.row_attention(
            normed,
            normed,
            normed,
            mask=row_valid[..., :, None] & row_valid[..., None, :],
            key_bias=_gate_bias(dec_gate)[:, None, None, :],
        )
        h = h + self.dropout(attended.transpose(2, 3))
        normed = self.norms[3](h)
        if isinstance(self.feedforward, ExpertFFN):
            if expert_weights is None:
                raise ValueError("expert weights are required for an expert block")
            # ExpertFFN's row axis precedes every token axis.
            hidden = self.feedforward(normed.transpose(1, 2), expert_weights).transpose(1, 2)
        else:
            hidden = self.feedforward(normed)
        return (h + self.dropout(hidden)).masked_fill(~valid[..., None], 0)
