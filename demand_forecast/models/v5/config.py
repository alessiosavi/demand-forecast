"""Architecture and regularisation settings for v5."""

import math
from dataclasses import dataclass, fields
from numbers import Real

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
    expert_mode: str = "routed"
    horizon_seed: bool = False
    naive_mix: bool = False

    def __post_init__(self) -> None:
        """Reject invalid dimensions and regularisation before constructing layers."""

        def numeric(value: object, name: str, kind: type) -> int | float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite {kind.__name__}")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite")
            if kind is int and value != int(value):
                raise ValueError(f"{name} must be an integer")
            # Adding zero turns -0.0 into 0.0, so equal configurations serialise identically.
            return kind(value) + 0

        for field in fields(self):
            if field.type in (int, float):
                object.__setattr__(
                    self, field.name, numeric(getattr(self, field.name), field.name, field.type)
                )
        object.__setattr__(
            self, "quantiles", tuple(numeric(q, "quantiles", float) for q in self.quantiles)
        )
        for name in (
            "d_model",
            "nhead",
            "enc_layers",
            "dec_layers",
            "dim_feedforward",
            "n_experts",
            "d_calendar",
            "d_step",
            "d_out",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.nhead or (self.d_model // self.nhead) % 2:
            raise ValueError("d_model must be divisible by nhead with an even dimension per head")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        for name, maximum in (
            ("expert_enc_layers", self.enc_layers),
            ("expert_dec_layers", self.dec_layers),
        ):
            if not 0 <= getattr(self, name) <= maximum:
                raise ValueError(f"{name} must be in [0, {maximum}]")
        if not isinstance(self.expert_mode, str) or self.expert_mode not in ("routed", "dense"):
            raise ValueError("expert_mode must be 'routed' or 'dense'")
        if self.expert_mode == "dense" and self.expert_enc_layers == self.expert_dec_layers == 0:
            raise ValueError("dense expert mode requires at least one expert block")
        for name in ("horizon_seed", "naive_mix"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("dropout", "known_gate_floor"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("sparsity_weight", "balance_weight"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be nonnegative")
        if not self.width_factor > 0:
            raise ValueError("width_factor must be positive")
        if (
            not self.quantiles
            or any(not 0 < q < 1 for q in self.quantiles)
            or any(b <= a for a, b in zip(self.quantiles, self.quantiles[1:]))
        ):
            raise ValueError("quantiles must be strictly increasing inside (0, 1)")
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must contain 0.5")
