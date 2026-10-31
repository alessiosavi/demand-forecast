"""Tiny temporal backbone for CPU-only skeleton tests."""

from torch import Tensor, nn


class ToyBackbone(nn.Module):
    """Project the valid encoder mean plus statics, then add decoder features."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, enc_in: Tensor, enc_valid: Tensor, dec_in: Tensor, static: Tensor) -> Tensor:
        total = enc_in.masked_fill(~enc_valid.unsqueeze(-1), 0).sum(dim=1)
        count = enc_valid.sum(dim=1, keepdim=True).clamp_min(1)
        context = self.projection(total / count + static)
        return context.unsqueeze(1) + dec_in
