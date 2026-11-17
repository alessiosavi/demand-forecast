"""Architecture and regularisation settings for v5."""

from dataclasses import dataclass

from demand_forecast.data.episode.schema import QUANTILES


@dataclass(frozen=True)
class V5Config:
    """Fixed dimensions, independent of the dataset and episode shape."""

    d_model: int = 384
    nhead: int = 6
    enc_layers: int = 5
    dec_layers: int = 3
    dim_feedforward: int = 1536
    n_experts: int = 4
    top_k: int = 2
    expert_enc_layers: int = 2
    expert_dec_layers: int = 2
    d_calendar: int = 16
    d_step: int = 32
    d_out: int = 32
    dropout: float = 0.1
    quantiles: tuple[float, ...] = QUANTILES
    width_factor: float = 2.0
    known_gate_floor: float = 0.05
    sparsity_weight: float = 1e-3
    balance_weight: float = 1e-2
    gate_init: float = 3.0
