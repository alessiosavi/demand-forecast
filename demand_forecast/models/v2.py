"""V2 temporal backbone for the shared forecasting skeleton.

Layers are copied/adapted from this package's ``transformer_v2.py`` and
``components.py``. The legacy modules remain independent and unchanged.
Input projections, static embeddings and the quantile head belong to ForecastModel.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@dataclass(frozen=True)
class V2Config:
    """Standard V2 architecture, independent of horizon and context length."""

    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    hidden_continuous_size: int = 64
    stochastic_depth_rate: float = 0.0
    use_rope: bool = True

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "hidden_continuous_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if self.use_rope and (self.d_model // self.nhead) % 2:
            raise ValueError("RoPE requires an even dimension per attention head")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        if not 0 <= self.stochastic_depth_rate < 1:
            raise ValueError("stochastic_depth_rate must be in [0, 1)")


V2_STANDARD = V2Config()


class RotaryPositionEmbedding(nn.Module):
    """Rotate Q/K at explicit signed positions, without a fixed-length cache."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        angles = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        angles = torch.cat([angles, angles], dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        return x * cos + rotated * sin


class InterpretableMultiHeadAttention(nn.Module):
    """Legacy attention adapted to explicit Q/K positions and padding masks.

    Retains Task 19's zero-output semantics for fully masked rows, including
    suppressing the output projection bias. Only hidden states are returned.
    """

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.nhead = cfg.nhead
        self.head_dim = cfg.d_model // cfg.nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim) if cfg.use_rope else None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_positions: Tensor,
        key_positions: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        if key_padding_mask is not None:
            # Zero before projection: even NaNs in padded cells must not enter
            # the value matmul (0 * NaN is NaN) or its backward pass.
            key = key.masked_fill(key_padding_mask.unsqueeze(-1), 0)
            value = value.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        q = self.q_proj(query).view(batch_size, tgt_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q, query_positions)
            k = self.rope(k, key_positions)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], -torch.inf)
        all_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        weights = scores.masked_fill(all_masked, 0).softmax(dim=-1)
        weights = self.dropout(weights.masked_fill(all_masked, 0))
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        return self.out_proj(attended).masked_fill(all_masked.all(dim=1), 0)


class FiLMConditioning(nn.Module):
    """Legacy feature-wise affine modulation by a projected static vector."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: Tensor, static: Tensor) -> Tensor:
        return self.gamma(static).unsqueeze(1) * x + self.beta(static).unsqueeze(1)


class StochasticDepth(nn.Module):
    """Drop each sample's residual branch during training, preserving its mean."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0:
            return residual + x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(x.new_full(shape, keep_prob))
        return residual + x * mask / keep_prob


class GatedResidualNetwork(nn.Module):
    """Legacy GRN specialised to equal input/output widths and static context."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.context_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        hidden = F.elu(self.fc1(x) + self.context_proj(context))
        hidden = self.dropout(self.fc2(hidden))
        return self.layer_norm(x + torch.sigmoid(self.gate(hidden)) * hidden)


class StaticEnrichmentLayer(nn.Module):
    """Enrich encoded time steps with the static vector before decoding."""

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(cfg.d_model, cfg.hidden_continuous_size, cfg.dropout)

    def forward(self, temporal: Tensor, static: Tensor) -> Tensor:
        return self.grn(temporal, static.unsqueeze(1))


class PreNormEncoderLayer(nn.Module):
    """Pre-norm self-attention and GELU feedforward with masked encoder keys."""

    def __init__(self, cfg: V2Config, depth_rate: float) -> None:
        super().__init__()
        self.self_attn = InterpretableMultiHeadAttention(cfg)
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.stochastic_depth = StochasticDepth(depth_rate)

    def forward(self, src: Tensor, positions: Tensor, key_padding_mask: Tensor) -> Tensor:
        normed = self.norm1(src)
        attended = self.self_attn(normed, normed, normed, positions, positions, key_padding_mask)
        src = self.stochastic_depth(self.dropout1(attended), src)
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(src)))))
        return self.stochastic_depth(self.dropout2(feedforward), src)


class PreNormDecoderLayer(nn.Module):
    """Bidirectional known-future self-attention and masked cross-attention."""

    def __init__(self, cfg: V2Config, depth_rate: float) -> None:
        super().__init__()
        self.self_attn = InterpretableMultiHeadAttention(cfg)
        self.cross_attn = InterpretableMultiHeadAttention(cfg)
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)
        self.stochastic_depth = StochasticDepth(depth_rate)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_positions: Tensor,
        memory_positions: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        normed = self.norm1(tgt)
        attended = self.self_attn(normed, normed, normed, tgt_positions, tgt_positions)
        tgt = self.stochastic_depth(self.dropout1(attended), tgt)
        attended = self.cross_attn(
            self.norm2(tgt), memory, memory, tgt_positions, memory_positions, key_padding_mask
        )
        tgt = self.stochastic_depth(self.dropout2(attended), tgt)
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(tgt)))))
        return self.stochastic_depth(self.dropout3(feedforward), tgt)


class V2Backbone(nn.Module):
    """Transform projected context and known future to (B, H, d_model) states."""

    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.encoder_film = FiLMConditioning(cfg.d_model)
        self.decoder_film = FiLMConditioning(cfg.d_model)
        self.encoder_layers = nn.ModuleList(
            PreNormEncoderLayer(cfg, cfg.stochastic_depth_rate * (i + 1) / cfg.num_encoder_layers)
            for i in range(cfg.num_encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        self.static_enrichment = StaticEnrichmentLayer(cfg)
        self.decoder_layers = nn.ModuleList(
            PreNormDecoderLayer(cfg, cfg.stochastic_depth_rate * (i + 1) / cfg.num_decoder_layers)
            for i in range(cfg.num_decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, enc_in: Tensor, enc_valid: Tensor, dec_in: Tensor, static: Tensor) -> Tensor:
        key_padding_mask = ~enc_valid
        # Anchor time at the forecast origin: the last real context is -1 and
        # the first future step is 0. Left padding cannot shift real positions,
        # including the relative distances used by cross-attention.
        enc_positions = torch.arange(-enc_in.shape[1], 0, device=enc_in.device)
        dec_positions = torch.arange(dec_in.shape[1], device=dec_in.device)
        enc_in = enc_in.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        memory = self.encoder_film(enc_in, static)
        for layer in self.encoder_layers:
            memory = layer(memory, enc_positions, key_padding_mask)
        memory = self.static_enrichment(self.encoder_norm(memory), static)
        decoded = self.decoder_film(dec_in, static)
        for layer in self.decoder_layers:
            decoded = layer(decoded, memory, dec_positions, enc_positions, key_padding_mask)
        return self.decoder_norm(decoded)
