"""V2 backbone contracts: masking, relative positions and known-future attention."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest
import torch

from demand_forecast.data.sampler import make_batch
from demand_forecast.models.base import ForecastModel
from demand_forecast.models.registry import FAMILIES, build_model
from demand_forecast.models.v2 import V2_STANDARD, V2Backbone, V2Config
from tests.helpers.models import build_v2


@pytest.mark.parametrize("horizon", [8, 13, 39, 52])
def test_v2_shapes(horizon, spec29_for, batch29_for):
    out = build_v2(spec29_for(horizon))(batch29_for(horizon))
    assert out.shape == (2, horizon, 9)
    assert torch.isfinite(out).all()
    assert (out.diff(dim=-1) >= 0).all()


def test_padding_content_and_length_invariance(
    spec29, batch_short, batch_short_items, panel29, encoder29
):
    model = build_v2(spec29).eval()
    a = model.predict(batch_short)
    b = batch_short.clone()
    b.target[0, :22] = 123.0
    b.enc_cov[0, :22] = -5.0
    assert torch.allclose(model.predict(b), a)
    c = make_batch(panel29, [batch_short_items[0]], 13, 64, encoder29)
    assert c.target.shape[1] == 7 and batch_short.target.shape[1] == 29
    assert torch.allclose(model.predict(c)[0], a[0], atol=1e-5)


def test_later_known_input_affects_earlier_step(spec29, batch29):
    model = build_v2(spec29).eval()
    base = model(batch29)
    b = batch29.clone()
    assert b.dec_cov_mask[0, -1, 4]
    b.dec_cov[0, -1, 4] += 1.0
    assert not torch.allclose(model(b)[0, 0], base[0, 0])


def test_future_known_nan_value_does_not_leak_through_fill(spec29, batch_future_nan):
    model = build_v2(spec29).eval()
    b = batch_future_nan.clone()
    assert not b.dec_cov_mask[0, 2, 4]
    b.dec_cov[0, 2, 4] = 7.0
    out = model(b)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, model(batch_future_nan))


def test_labels_and_future_past_only_values_do_not_change_outputs(
    spec29, batch29, panel29, panel29_with_future_oil, encoder29
):
    model = build_v2(spec29).eval()
    b = batch29.clone()
    b.labels += 7.0
    assert torch.allclose(model(b), model(batch29))
    c = make_batch(panel29_with_future_oil, [(0, 20)], 13, 64, encoder29)
    d = make_batch(panel29, [(0, 20)], 13, 64, encoder29)
    assert torch.allclose(model(c), model(d))


def test_standard_config_is_frozen():
    assert asdict(V2_STANDARD) == dict(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        hidden_continuous_size=64,
        stochastic_depth_rate=0.0,
        use_rope=True,
    )
    with pytest.raises(FrozenInstanceError):
        V2_STANDARD.d_model = 16


def test_registry_uses_spec_width_and_rejects_mismatches(spec29):
    model = build_model("v2", spec29, {})
    assert isinstance(model, ForecastModel)
    assert isinstance(model.backbone, V2Backbone)
    assert model.backbone.d_model == spec29.d_model
    assert FAMILIES["v2"](spec29, {}).d_model == spec29.d_model
    kw = {"d_model": 16, "nhead": 2}
    build_model("v2", spec29, kw)
    assert kw == {"d_model": 16, "nhead": 2}
    with pytest.raises(ValueError, match="d_model"):
        build_model("v2", spec29, {"d_model": 8})
    with pytest.raises(ValueError, match="d_model"):
        FAMILIES["v2"](spec29, {"d_model": 8})
    with pytest.raises(ValueError, match="family"):
        build_model("unknown", spec29, {})


@pytest.mark.parametrize("use_rope", [True, False])
def test_backbone_masks_padding_and_backpropagates_with_empty_memory(use_rope):
    model = V2Backbone(
        V2Config(
            d_model=16,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=32,
            dropout=0,
            use_rope=use_rope,
        )
    ).eval()
    enc = torch.randn(2, 7, 16, requires_grad=True)
    valid = torch.tensor([[False] * 4 + [True] * 3, [False] * 7])
    dec = torch.randn(2, 5, 16, requires_grad=True)
    static = torch.randn(2, 16, requires_grad=True)
    out = model(enc, valid, dec, static)
    poisoned = enc.detach().masked_fill(~valid.unsqueeze(-1), torch.nan)
    torch.testing.assert_close(model(poisoned, valid, dec, static), out)
    torch.testing.assert_close(model(enc[:1, -3:], valid[:1, -3:], dec[:1], static[:1]), out[:1])
    assert out.shape == (2, 5, 16) and torch.isfinite(out).all()
    out.square().mean().backward()
    for tensor in (enc, dec, static):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(enc.grad[~valid]) == 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    empty = model(enc[:, :0], valid[:, :0], dec, static)
    assert torch.isfinite(empty).all()


def test_fallbacks_and_zero_context_are_finite(
    spec29, batch_with_empty_and_constant, panel29, encoder29
):
    model = build_v2(spec29).eval()
    out = model.predict(batch_with_empty_and_constant)
    assert (out[0] == 0).all() and (out[1] == 5).all()
    empty = make_batch(panel29, [(0, 0), (1, 0)], 13, 64, encoder29)
    assert torch.isfinite(model(empty)).all()
    assert (model.predict(empty) == 0).all()


def test_dropout_and_stochastic_depth_follow_training_mode(spec29, batch29):
    model = build_v2(spec29, dropout=0.25, stochastic_depth_rate=0.5)
    assert not torch.equal(model(batch29), model(batch29))
    model(batch29).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    torch.testing.assert_close(model(batch29), model(batch29), rtol=0, atol=0)
    restored = build_v2(spec29, dropout=0.25, stochastic_depth_rate=0.5).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(batch29), model(batch29), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kw",
    [
        {"d_model": 0},
        {"nhead": 0},
        {"nhead": 3},
        {"d_model": 6, "nhead": 2},
        {"num_encoder_layers": 0},
        {"num_decoder_layers": 0},
        {"dim_feedforward": 0},
        {"hidden_continuous_size": 0},
        {"dropout": -0.1},
        {"stochastic_depth_rate": 1.0},
    ],
)
def test_invalid_configuration_is_rejected(kw):
    with pytest.raises(ValueError):
        V2Backbone(replace(V2_STANDARD, **{"d_model": 16, "nhead": 2, **kw}))
