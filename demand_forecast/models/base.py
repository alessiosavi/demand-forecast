"""Shared input projections, static conditioning and quantile prediction semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.frequency import calendar_feature_names
from demand_forecast.data.panel import Panel
from demand_forecast.data.sampler import Batch
from demand_forecast.models.heads import MonotoneQuantileHead


@dataclass
class ModelSpec:
    """Dimensions and output conventions shared by every temporal backbone."""

    d_model: int
    quantiles: tuple[float, ...]
    enc_channels: int
    dec_channels: int
    vocab_sizes: dict[str, int]
    n_numeric: int
    cat_emb_dim: int
    target_support: str
    width_factor: float = 2.0
    use_vsn: bool = True

    @classmethod
    def from_panel(
        cls,
        panel: Panel,
        encoder: StaticEncoder,
        forecast_cfg: ForecastConfig,
        d_model: int,
        cat_emb_dim: int,
    ) -> ModelSpec:
        """Match the calendar, known/past covariate and fitted static batch layout."""
        n_calendar = len(calendar_feature_names(panel.freq))
        n_dec = n_calendar + len(panel.roles.known_dynamic)
        n_enc = n_dec + len(panel.roles.past_dynamic)
        return cls(
            d_model=d_model,
            quantiles=tuple(forecast_cfg.quantiles),
            enc_channels=2 + 2 * n_enc,
            dec_channels=2 * n_dec,
            vocab_sizes=dict(encoder.vocab_sizes),
            n_numeric=encoder.n_numeric,
            cat_emb_dim=cat_emb_dim,
            target_support=panel.target_support,
        )


class Backbone(Protocol):
    """Temporal module: projected inputs and statics produce horizon hidden states.

    Implementations are ``nn.Module`` instances with a ``d_model`` attribute.
    Encoder/decoder tensors have shape (B, L/H, d_model), statics (B, d_model),
    and ``enc_valid`` (B, L) is True at real positions, even if unobserved.
    """

    d_model: int

    def forward(
        self, enc_in: Tensor, enc_valid: Tensor, dec_in: Tensor, static: Tensor
    ) -> Tensor: ...

    def __call__(
        self, enc_in: Tensor, enc_valid: Tensor, dec_in: Tensor, static: Tensor
    ) -> Tensor: ...


class ForecastModel(nn.Module):
    """Wrap a backbone with label-free inputs, a monotone head and frame fallbacks."""

    def __init__(self, spec: ModelSpec, backbone: Backbone) -> None:
        super().__init__()
        self.spec = spec
        self.backbone = backbone
        self.enc_proj = nn.Linear(spec.enc_channels, spec.d_model)
        self.dec_proj = nn.Linear(spec.dec_channels, spec.d_model)
        # Preserve vocabulary order without restricting user column names to
        # valid PyTorch module names (column names may contain periods).
        self.statics = nn.ModuleList(
            nn.Embedding(size, spec.cat_emb_dim) for size in spec.vocab_sizes.values()
        )
        static_channels = len(spec.vocab_sizes) * spec.cat_emb_dim + 2 * spec.n_numeric
        n_blocks = len(spec.vocab_sizes) + int(spec.n_numeric > 0)
        self.static_gate = nn.Linear(static_channels, n_blocks) if spec.use_vsn else None
        self.static_proj = nn.Linear(static_channels, spec.d_model)
        self.head = MonotoneQuantileHead(spec.d_model, spec.quantiles, spec.width_factor)
        self.register_buffer("fallback_level", torch.tensor(0.0))

    def encoder_inputs(self, batch: Batch) -> Tensor:
        """Concatenate [normalised target, observed, covariates, covariate masks]."""
        return torch.cat(
            [
                batch.target.unsqueeze(-1),
                batch.observed.to(batch.target.dtype).unsqueeze(-1),
                batch.enc_cov.masked_fill(~batch.enc_cov_mask, 0),
                batch.enc_cov_mask.to(batch.enc_cov.dtype),
            ],
            dim=-1,
        )

    def decoder_inputs(self, batch: Batch) -> Tensor:
        """Concatenate known-future covariates and their float masks."""
        values = batch.dec_cov.masked_fill(~batch.dec_cov_mask, 0)
        return torch.cat([values, batch.dec_cov_mask.to(batch.dec_cov.dtype)], dim=-1)

    def static_vector(self, batch: Batch) -> Tensor:
        """Embed all categorical columns, append [num, num_mask], then project.

        The optional minimal VSN is a linear softmax gate over blocks: one per
        categorical embedding (including __series__), plus one combined numeric
        value/mask block when present. Each block is weighted before concatenation
        and projection; disabling VSN leaves the concatenated inputs unweighted.
        """
        blocks = [
            embedding(batch.cat[name])
            for name, embedding in zip(self.spec.vocab_sizes, self.statics)
        ]
        if self.spec.n_numeric:
            blocks.append(torch.cat([batch.num, batch.num_mask.to(batch.num.dtype)], dim=-1))
        combined = torch.cat(blocks, dim=-1)
        if self.static_gate is not None:
            weights = self.static_gate(combined).softmax(dim=-1)
            combined = torch.cat(
                [block * weights[:, i : i + 1] for i, block in enumerate(blocks)], dim=-1
            )
        return self.static_proj(combined)

    def forward(self, batch: Batch) -> Tensor:
        """Return (B, H, Q) sorted quantiles in the batch's normalised frame."""
        hidden = self.backbone(
            self.enc_proj(self.encoder_inputs(batch)),
            batch.padding,
            self.dec_proj(self.decoder_inputs(batch)),
            self.static_vector(batch),
        )
        return self.head(hidden)

    def fallback_flags(self, batch: Batch) -> dict[str, Tensor]:
        """Disjoint item masks used by prediction and its diagnostics."""
        return {
            "no_context": ~batch.has_context,
            "undefined_scale": batch.has_context & ~batch.eligible,
        }

    @torch.no_grad()
    def predict(self, batch: Batch) -> Tensor:
        """Denormalise once, apply context/scale fallbacks, then enforce support."""
        quantiles = self(batch) * batch.scale[:, None, None] + batch.loc[:, None, None]
        flags = self.fallback_flags(batch)
        for row in torch.nonzero(flags["undefined_scale"], as_tuple=True)[0]:
            observed = batch.target[row, batch.observed[row]]
            values = observed * batch.scale[row] + batch.loc[row]
            quantiles[row] = torch.quantile(values, values.new_tensor(self.spec.quantiles))
        empty_level = 0.0 if self.spec.target_support == "nonnegative" else self.fallback_level
        quantiles = torch.where(flags["no_context"][:, None, None], empty_level, quantiles)
        return quantiles.clamp_min(0) if self.spec.target_support == "nonnegative" else quantiles
