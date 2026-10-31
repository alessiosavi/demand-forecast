"""Contracts for projected inputs, normalised quantiles and label-free fallbacks."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.sampler import make_batch
from demand_forecast.models.base import ForecastModel, ModelSpec
from tests.helpers.panels import synthetic_panel, weekly_panel
from tests.helpers.toy_backbone import ToyBackbone


def test_forward_shapes_and_sorted_output(spec29, batch29):
    out = ForecastModel(spec29, ToyBackbone(16))(batch29)
    assert out.shape == (2, 13, 9) and (out.diff(dim=-1) >= 0).all()


def test_predict_clamps_only_when_nonnegative(spec29, batch29):
    real = ForecastModel(replace(spec29, target_support="real"), ToyBackbone(16))
    nonneg = ForecastModel(spec29, ToyBackbone(16))
    nonneg.load_state_dict(real.state_dict())
    with torch.no_grad():
        real.head.projection.bias.fill_(-50.0)
        nonneg.head.projection.bias.fill_(-50.0)
    assert (real.predict(batch29) < 0).any() and (nonneg.predict(batch29) >= 0).all()


def test_predictions_do_not_depend_on_labels_and_do_depend_on_weights(spec29, batch29):
    model = ForecastModel(spec29, ToyBackbone(16)).eval()
    b = batch29.clone()
    b.labels.zero_()
    b.label_mask.zero_()
    b.loss_eligible.zero_()
    assert b.eligible.any() and torch.allclose(model.predict(b), model.predict(batch29))
    other = ForecastModel(spec29, ToyBackbone(16)).eval()
    other.load_state_dict(model.state_dict())
    with torch.no_grad():
        other.head.projection.bias.add_(1.0)
    assert not torch.allclose(
        other.predict(b)[batch29.eligible], model.predict(b)[batch29.eligible]
    )

    # The plan fixture has no future labels already; make forbidden reads fail too.
    unlabelled = replace(batch29, labels=None, label_mask=None, loss_eligible=None)
    torch.testing.assert_close(model(unlabelled), model(batch29))
    torch.testing.assert_close(model.predict(unlabelled), model.predict(batch29))


def test_fallbacks_for_empty_and_zero_scale_context(spec29, batch_with_empty_and_constant):
    out = ForecastModel(spec29, ToyBackbone(16)).predict(batch_with_empty_and_constant)
    assert (out[0] == 0).all() and torch.allclose(out[1], torch.full_like(out[1], 5.0))


@pytest.mark.parametrize("freq,n_calendar", [("W-SUN", 4), ("D", 10), ("h", 12), ("MS", 4)])
def test_spec_matches_sampler_channels_without_optional_statics(freq, n_calendar):
    panel = synthetic_panel(2, 8, freq, seed=0)
    encoder = StaticEncoder().fit(panel, [series.series_id for series in panel.series])
    cfg = ForecastConfig(horizon=3, quantiles=[0.1, 0.5, 0.95])
    spec = ModelSpec.from_panel(panel, encoder, cfg, 8, 4)
    assert spec.enc_channels == 2 + 2 * n_calendar
    assert spec.dec_channels == 2 * n_calendar
    assert spec.vocab_sizes == {"__series__": 4} and spec.n_numeric == 0
    assert spec.quantiles == (0.1, 0.5, 0.95)
    assert spec.target_support == "nonnegative"
    assert spec.width_factor == 2.0 and spec.use_vsn is True
    batch = make_batch(panel, [(0, 8), (1, 5)], 3, 8, encoder)
    model = ForecastModel(spec, ToyBackbone(8))
    assert model(batch).shape == (2, 3, 3)
    assert torch.isfinite(model.predict(batch)).all()


def test_input_channels_projections_and_padding_mask(spec29, panel29, encoder29):
    panel29.series[1].target[3] = np.nan
    batch = make_batch(panel29, [(0, 12), (1, 6)], 13, 64, encoder29)
    model = ForecastModel(spec29, ToyBackbone(16))
    enc, dec = model.encoder_inputs(batch), model.decoder_inputs(batch)
    assert enc.shape == (2, 12, 14) and dec.shape == (2, 13, 10)
    assert enc.dtype == dec.dtype == torch.float32
    torch.testing.assert_close(enc[..., 0], batch.target)
    torch.testing.assert_close(enc[..., 1], batch.observed.float())
    torch.testing.assert_close(enc[..., 2:8], batch.enc_cov)
    torch.testing.assert_close(enc[..., 8:], batch.enc_cov_mask.float())
    torch.testing.assert_close(dec[..., :5], batch.dec_cov)
    torch.testing.assert_close(dec[..., 5:], batch.dec_cov_mask.float())

    seen = []
    handle = model.backbone.register_forward_pre_hook(lambda _module, args: seen.append(args))
    model(batch)
    handle.remove()
    enc_in, enc_valid, dec_in, static = seen[0]
    torch.testing.assert_close(enc_in, model.enc_proj(enc))
    torch.testing.assert_close(dec_in, model.dec_proj(dec))
    torch.testing.assert_close(static, model.static_vector(batch))
    assert torch.equal(enc_valid, batch.padding)
    assert (enc_valid & ~batch.observed).any()  # A real but missing observation is valid.

    poisoned = batch.clone()
    poisoned.target[~batch.padding] = 1e6
    poisoned.enc_cov[~batch.padding] = -1e6
    torch.testing.assert_close(model(poisoned), model(batch))


def test_masked_covariate_values_do_not_change_outputs(spec29, batch_future_nan, batch_short):
    model = ForecastModel(spec29, ToyBackbone(16)).eval()
    padded = batch_short.clone()
    # Include a missing covariate at a real step: padding alone is already
    # ignored by ToyBackbone and would not catch missing encoder-cell masking.
    assert padded.padding[0, -1] and (~padded.padding).any()
    padded.enc_cov[0, -1, -1] = 0.0
    padded.enc_cov_mask[0, -1, -1] = False

    for batch, values_name, mask_name in (
        (batch_future_nan, "dec_cov", "dec_cov_mask"),
        (padded, "enc_cov", "enc_cov_mask"),
    ):
        masked = ~getattr(batch, mask_name)
        assert masked.any()
        poisoned = batch.clone()
        getattr(poisoned, values_name).masked_fill_(masked, 7.0)
        torch.testing.assert_close(model(poisoned), model(batch), rtol=0, atol=0)


@pytest.mark.parametrize("use_vsn", [False, True])
def test_statics_and_gradients_reach_the_forecast(spec29, batch29, use_vsn):
    model = ForecastModel(replace(spec29, use_vsn=use_vsn), ToyBackbone(16))
    static = model.static_vector(batch29)
    assert static.shape == (2, 16)
    for name in spec29.vocab_sizes:
        changed = batch29.clone()
        changed.cat[name].zero_()  # Unknown category, including the series id.
        assert not torch.allclose(model.static_vector(changed), static)
    for field in ("num", "num_mask"):
        changed = batch29.clone()
        getattr(changed, field).zero_()
        assert not torch.allclose(model.static_vector(changed), static)
    model(batch29).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_static_selection_can_be_disabled(spec29, batch29):
    selected = ForecastModel(spec29, ToyBackbone(16))
    plain = ForecastModel(replace(spec29, use_vsn=False), ToyBackbone(16))
    # Share every common parameter so only variable selection can cause a difference.
    common = {
        name: value for name, value in selected.state_dict().items() if name in plain.state_dict()
    }
    plain.load_state_dict(common)
    assert not torch.allclose(selected.static_vector(batch29), plain.static_vector(batch29))


def test_predict_denormalises_once(spec29, batch29):
    model = ForecastModel(replace(spec29, target_support="real"), ToyBackbone(16))
    batch = batch29.clone()
    batch.loc = torch.tensor([-13.0, 100.0])
    batch.scale = torch.tensor([2.0, 7.0])
    assert batch.eligible.all() and batch.has_context.all()
    torch.testing.assert_close(model(batch), model(batch29))
    torch.testing.assert_close(
        model.predict(batch), model(batch) * batch.scale[:, None, None] + batch.loc[:, None, None]
    )


@pytest.mark.parametrize("support", ["nonnegative", "real"])
@pytest.mark.parametrize("empty_eligible", [False, True])
def test_fallbacks_use_observed_values_and_flags(spec29, encoder29, support, empty_eligible):
    panel = weekly_panel([6, 8])
    panel.target_support = support
    panel.seasonal_period = 2
    panel.series[0].target[:] = np.nan
    panel.series[1].target[:] = [-4, 8, -4, np.nan, np.nan, 8, np.nan, np.nan]
    batch = make_batch(panel, [(0, 6), (1, 8)], 13, 64, encoder29)
    assert batch.has_context.tolist() == [False, True]
    assert not batch.eligible.any()
    batch.eligible[0] = empty_eligible  # No-context fallback takes precedence over scale.
    knots = (0.1, 0.5, 0.75, 0.95)
    model = ForecastModel(replace(spec29, target_support=support, quantiles=knots), ToyBackbone(16))
    assert model.fallback_level.item() == 0.0
    assert "fallback_level" in dict(model.named_buffers())
    model.fallback_level.fill_(-7.5)
    out = model.predict(batch)
    expected = torch.tensor([-4.0, 2.0, 8.0, 8.0]).expand(13, -1)
    if support == "nonnegative":
        expected = expected.clamp_min(0)
    torch.testing.assert_close(
        out[0], torch.full_like(out[0], 0 if support == "nonnegative" else -7.5)
    )
    torch.testing.assert_close(out[1], expected)
    assert torch.isfinite(out).all() and (out.diff(dim=-1) >= 0).all()

    restored = ForecastModel(model.spec, ToyBackbone(16))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.predict(batch), out)
    with torch.no_grad():
        restored.head.projection.bias.add_(100)
    torch.testing.assert_close(restored.predict(batch), out)


def test_zero_length_context_is_finite(spec29, panel29, encoder29):
    batch = make_batch(panel29, [(0, 0), (1, 0)], 13, 64, encoder29)
    model = ForecastModel(spec29, ToyBackbone(16))
    assert torch.isfinite(model(batch)).all()
    assert (model.predict(batch) == 0).all()
