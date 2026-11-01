"""Initialise task-shaped V2 models from calendar-off univariate pretraining.

ForecastModel.encoder_inputs always places the normalised target and observation
mask at columns 0 and 1. Columns 2 onward contain calendar/known/past covariate
values, then their masks; ModelSpec reserves these even with calendar disabled.
The pretrained weekly model thus has 10 encoder channels: target, observed,
four calendar values and four masks. Copy source columns 0:2 to target columns
0:2 and zero all others. Decoder inputs contain calendar/known covariate values
then masks, all inactive in pretraining, so zero the entire decoder projection.

Collapse the source's constant code-0 static contribution into static_proj.bias.
For identical targets, normalisation and horizon in evaluation mode, forward()
then matches the source up to floating-point rounding of the static path
(<= 5e-7 measured at d_model=16), hence atol=rtol=1e-5 rather than bitwise equality.
The source must be a calendar-off, covariate-free pretrained bundle; the static
shape checks alone do not establish how an arbitrary bundle was trained.
"""

from dataclasses import dataclass, fields

import torch

from demand_forecast.core.bundle import Bundle
from demand_forecast.models.base import ForecastModel
from demand_forecast.models.v2 import V2Backbone, V2Config


class TransferError(ValueError):
    """Source and target cannot share the pretrained forecasting function."""


@dataclass
class TransferReport:
    """Sorted parameter names and the norm of the collapsed static contribution.

    enc_proj.weight appears in both copied and zeroed because different columns
    receive those operations. copied includes the derived static_proj.bias.
    Buffers (RoPE and fallback_level) are copied too, but these lists name only
    parameters. fresh parameters retain their existing task initialisation.
    """

    copied: list[str]
    zeroed: list[str]
    fresh: list[str]
    static_bias_norm: float


def _require_equal(name: str, source: object, target: object) -> None:
    if source != target:
        raise TransferError(f"{name}: source {source}, target {target}")


@torch.no_grad()
def init_from_pretrained(model: ForecastModel, source: Bundle) -> TransferReport:
    """Copy shared weights in place, preserving trainability and the source model.

    Validate compatibility before changing any target tensors. Static vocabulary
    sizes, embedding widths, numeric features and temporal channel counts may
    differ in the task model. Task embeddings and gates are left untouched.
    """
    pretrained = source.model
    _require_equal("family", source.family, "v2")
    if not isinstance(pretrained.backbone, V2Backbone) or not isinstance(
        model.backbone, V2Backbone
    ):
        raise TransferError(
            f"backbone: source {type(pretrained.backbone).__name__}, "
            f"target {type(model.backbone).__name__}; expected V2Backbone for both"
        )
    _require_equal("d_model", pretrained.spec.d_model, model.spec.d_model)
    for field in fields(V2Config):
        _require_equal(
            field.name,
            getattr(pretrained.backbone.cfg, field.name),
            getattr(model.backbone.cfg, field.name),
        )
    _require_equal("quantiles", pretrained.spec.quantiles, model.spec.quantiles)
    # This non-tensor head setting is not transferred by load_state_dict.
    _require_equal("width_factor", pretrained.head.width_factor, model.head.width_factor)
    if len(pretrained.spec.vocab_sizes) != 1:
        raise TransferError(
            f"vocab_sizes: source {pretrained.spec.vocab_sizes}, "
            f"target {model.spec.vocab_sizes}; expected one source categorical vocabulary"
        )
    if pretrained.spec.n_numeric != 0:
        raise TransferError(
            f"n_numeric: source {pretrained.spec.n_numeric}, "
            f"target {model.spec.n_numeric}; expected source 0"
        )

    # One categorical block makes the source's optional softmax gate identity.
    s0 = pretrained.static_proj(pretrained.statics[0].weight[0:1])[0]
    model.backbone.load_state_dict(pretrained.backbone.state_dict())
    model.head.load_state_dict(pretrained.head.state_dict())
    model.enc_proj.weight.zero_()
    model.enc_proj.weight[:, :2].copy_(pretrained.enc_proj.weight[:, :2])
    model.enc_proj.bias.copy_(pretrained.enc_proj.bias)
    model.dec_proj.weight.zero_()
    model.dec_proj.bias.copy_(pretrained.dec_proj.bias)
    model.static_proj.weight.zero_()
    model.static_proj.bias.copy_(s0)
    model.fallback_level.copy_(pretrained.fallback_level)

    shared = [
        name for name, _ in model.named_parameters() if name.startswith(("backbone.", "head."))
    ]
    return TransferReport(
        copied=sorted(
            [*shared, "enc_proj.weight", "enc_proj.bias", "dec_proj.bias", "static_proj.bias"]
        ),
        zeroed=sorted(["enc_proj.weight", "dec_proj.weight", "static_proj.weight"]),
        fresh=sorted(
            name
            for name, _ in model.named_parameters()
            if name.startswith(("statics.", "static_gate."))
        ),
        static_bias_norm=s0.norm().item(),
    )
