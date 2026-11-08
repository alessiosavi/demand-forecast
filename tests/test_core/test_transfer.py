"""Pretrained transfer preserves forecasts while allowing new features to learn."""

from dataclasses import replace

import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.panel import Panel, Roles
from demand_forecast.data.sampler import make_batch
from demand_forecast.models.base import ModelSpec
from demand_forecast.models.losses import pinball_loss
from tests.helpers.models import build_v2

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench


def strip_covariates(panel: Panel) -> Panel:
    """Preserve the target frame and support while removing all task features."""
    return replace(
        panel,
        roles=Roles(panel.roles.target, (), (), (), ()),
        series=[replace(series, dynamic={}, statics={}) for series in panel.series],
    )


def test_transfer_matches_source_forward_to_rounding(
    panel29, encoder29, pretrained_univariate_bundle
):
    from demand_forecast.core.transfer import TransferReport, init_from_pretrained

    source = pretrained_univariate_bundle
    spec = ModelSpec.from_panel(panel29, encoder29, ForecastConfig(horizon=4), 16, 4)
    target = build_v2(spec, **source.backbone_kwargs)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    source_before = {name: value.clone() for name, value in source.model.state_dict().items()}
    with torch.no_grad():
        # RoPE and fallback buffers otherwise start identical without transfer.
        for buffer in target.buffers():
            buffer.fill_(-99)
    report = init_from_pretrained(target, source)

    assert isinstance(report, TransferReport)
    parameters = dict(target.named_parameters())
    shared = [name for name in parameters if name.startswith(("backbone.", "head."))]
    assert report.copied == sorted(
        [*shared, "enc_proj.weight", "enc_proj.bias", "dec_proj.bias", "static_proj.bias"]
    )
    assert report.zeroed == ["dec_proj.weight", "enc_proj.weight", "static_proj.weight"]
    assert report.fresh == sorted(
        name for name in parameters if name.startswith(("statics.", "static_gate."))
    )
    assert set(report.copied + report.zeroed + report.fresh) == set(parameters)
    state = target.state_dict()
    for name, value in state.items():
        if name.startswith(("backbone.", "head.")) or name == "fallback_level":
            assert torch.equal(value, source_before[name]), name
        elif name in report.fresh:
            assert torch.equal(value, before[name]), name
    assert any(name.endswith("rope.inv_freq") for name in state)
    assert torch.equal(target.enc_proj.weight[:, :2], source.model.enc_proj.weight[:, :2])
    assert torch.equal(target.enc_proj.bias, source.model.enc_proj.bias)
    assert torch.equal(target.dec_proj.bias, source.model.dec_proj.bias)
    assert torch.count_nonzero(target.enc_proj.weight[:, 2:]) == 0
    assert torch.count_nonzero(target.dec_proj.weight) == 0
    assert torch.count_nonzero(target.static_proj.weight) == 0
    with torch.no_grad():
        s0 = source.model.static_proj(source.model.statics[0].weight[0:1])[0]
    assert torch.equal(target.static_proj.bias, s0)
    assert report.static_bias_norm == pytest.approx(s0.norm().item())

    batch = make_batch(panel29, [(0, 29), (1, 29)], 4, 64, encoder29)
    uni = strip_covariates(panel29)
    uni_batch = make_batch(
        uni, [(0, 29), (1, 29)], 4, 64, StaticEncoder().fit(uni, []), calendar=False
    )
    assert source.model.spec.enc_channels == 10
    assert source.model.spec.dec_channels == 8
    assert source.model.spec.vocab_sizes == {"__series__": 2}
    assert source.model.spec.n_numeric == 0
    assert torch.count_nonzero(source.model.encoder_inputs(uni_batch)[..., 2:]) == 0
    assert torch.count_nonzero(source.model.decoder_inputs(uni_batch)) == 0
    assert torch.equal(
        target.encoder_inputs(batch)[..., :2], source.model.encoder_inputs(uni_batch)[..., :2]
    )
    assert batch.enc_cov_mask.any() and batch.dec_cov_mask.any()
    assert torch.equal(batch.loc, uni_batch.loc)
    assert torch.equal(batch.scale, uni_batch.scale)
    with torch.no_grad():
        out, ref = target.eval()(batch), source.model.eval()(uni_batch)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, source_before[name]), name


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_encoder_layers", 2),
        ("dropout", 0.2),
        ("quantiles", (0.1, 0.5, 0.9)),
        ("d_model", 32),
        ("width_factor", 3.0),
    ],
)
def test_transfer_rejects_architecture_and_quantile_mismatch(
    panel29, encoder29, pretrained_univariate_bundle, field, value
):
    from demand_forecast.core.transfer import TransferError, init_from_pretrained

    source = pretrained_univariate_bundle
    spec = ModelSpec.from_panel(panel29, encoder29, ForecastConfig(horizon=4), 16, 4)
    options = dict(source.backbone_kwargs)
    if field in ("quantiles", "d_model", "width_factor"):
        spec = replace(spec, **{field: value})
        old = getattr(source.model.spec, field)
    else:
        old = getattr(source.model.backbone.cfg, field)
    if field in options:
        options[field] = value
    target = build_v2(spec, **options)
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
    with pytest.raises(TransferError, match=field) as error:
        init_from_pretrained(target, source)
    assert f"source {old}" in str(error.value)
    assert f"target {value}" in str(error.value)
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name]), name


@pytest.mark.parametrize(
    "field, value", [("vocab_sizes", {"cat": 3, "__series__": 2}), ("n_numeric", 1)]
)
def test_transfer_rejects_nonunivariate_source(
    panel29, encoder29, pretrained_univariate_bundle, field, value
):
    from demand_forecast.core.transfer import TransferError, init_from_pretrained

    source = pretrained_univariate_bundle
    invalid = replace(
        source,
        model=build_v2(replace(source.model.spec, **{field: value}), **source.backbone_kwargs),
    )
    spec = ModelSpec.from_panel(panel29, encoder29, ForecastConfig(horizon=4), 16, 4)
    target = build_v2(spec, **source.backbone_kwargs)
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
    with pytest.raises(TransferError, match=field) as error:
        init_from_pretrained(target, invalid)
    assert f"source {value}" in str(error.value)
    assert f"target {getattr(spec, field)}" in str(error.value)
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_zeroed_weights_get_gradients_and_move_after_a_step(
    panel29, encoder29, pretrained_univariate_bundle
):
    from demand_forecast.core.transfer import init_from_pretrained

    source = pretrained_univariate_bundle
    spec = ModelSpec.from_panel(panel29, encoder29, ForecastConfig(horizon=4), 16, 4)
    target = build_v2(spec, **source.backbone_kwargs)
    init_from_pretrained(target, source)
    source_before = {name: tensor.clone() for name, tensor in source.model.state_dict().items()}
    batch = make_batch(panel29, [(0, 25), (1, 25)], 4, 64, encoder29)
    optimizer = torch.optim.SGD(target.parameters(), lr=1e-2)
    weights = [target.enc_proj.weight[:, 2:], target.dec_proj.weight, target.static_proj.weight]
    assert all(torch.count_nonzero(weight) == 0 for weight in weights)
    loss, _ = pinball_loss(
        target.train()(batch),
        batch.labels,
        batch.label_mask,
        batch.loss_eligible,
        torch.tensor(spec.quantiles),
    )
    loss.backward()
    gradients = [
        target.enc_proj.weight.grad[:, 2:],
        target.dec_proj.weight.grad,
        target.static_proj.weight.grad,
    ]
    assert all(
        torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients
    )
    optimizer.step()
    assert all(torch.count_nonzero(weight) > 0 for weight in weights)
    for name, tensor in source.model.state_dict().items():
        assert torch.equal(tensor, source_before[name]), name


def _named_source(source, panel):
    """Give every named projection column a distinct deterministic weight."""
    cfg = ForecastConfig(horizon=4, calendar_schema="canonical", target_lags="cycle")
    uni = strip_covariates(panel)
    encoder = StaticEncoder().fit(uni, [])
    spec = ModelSpec.from_panel(uni, encoder, cfg, 16, 4)
    model = build_v2(spec, **source.backbone_kwargs)
    with torch.no_grad():
        for weight in (model.enc_proj.weight, model.dec_proj.weight):
            weight.copy_(torch.arange(weight.numel()).reshape_as(weight) / 1000 + 0.1)
    return replace(source, model=model, forecast_cfg=cfg, encoder=encoder)


@pytest.mark.parametrize("freq", ["W-MON", "h", "MS"])
@pytest.mark.parametrize("calendar_schema", ["legacy", "canonical"])
@pytest.mark.parametrize("target_lags", ["none", "cycle"])
def test_named_transfer_maps_values_and_masks_with_dynamic_columns(
    panel29, encoder29, pretrained_univariate_bundle, freq, calendar_schema, target_lags
):
    from demand_forecast.core.transfer import init_from_pretrained
    from demand_forecast.data.frequency import CANONICAL_CALENDAR_SCHEMA, calendar_feature_names
    from demand_forecast.data.sampler import TARGET_LAG_SCHEMA, target_lag_names

    panel = replace(panel29, freq=freq)
    source = _named_source(pretrained_univariate_bundle, panel)
    forecast = ForecastConfig(horizon=4, calendar_schema=calendar_schema, target_lags=target_lags)
    target = build_v2(
        ModelSpec.from_panel(panel, encoder29, forecast, 16, 4), **source.backbone_kwargs
    )
    before = {name: value.clone() for name, value in source.model.state_dict().items()}
    report = init_from_pretrained(target, source, target_panel=panel, target_forecast=forecast)
    src_names = CANONICAL_CALENDAR_SCHEMA + TARGET_LAG_SCHEMA
    dest_names = tuple(calendar_feature_names(freq, calendar_schema)) + target_lag_names(
        target_lags
    )
    n_enc = len(dest_names) + len(panel.roles.known_dynamic) + len(panel.roles.past_dynamic)
    n_dec = len(dest_names) + len(panel.roles.known_dynamic)
    expected_enc = torch.zeros_like(target.enc_proj.weight)
    expected_dec = torch.zeros_like(target.dec_proj.weight)
    expected_enc[:, :2] = source.model.enc_proj.weight[:, :2]
    for dest, name in enumerate(dest_names):
        src = src_names.index(name)
        for offset in (0, 1):
            expected_enc[:, 2 + dest + offset * n_enc] = source.model.enc_proj.weight[
                :, 2 + src + offset * len(src_names)
            ]
            expected_dec[:, dest + offset * n_dec] = source.model.dec_proj.weight[
                :, src + offset * len(src_names)
            ]
    assert torch.equal(target.enc_proj.weight, expected_enc)
    assert torch.equal(target.dec_proj.weight, expected_dec)
    assert "dec_proj.weight" in report.copied
    assert not target.static_proj.weight.any()
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_named_transfer_weekly_legacy_forward_equivalence(
    panel29, encoder29, pretrained_univariate_bundle
):
    from demand_forecast.core.transfer import init_from_pretrained

    source = _named_source(pretrained_univariate_bundle, panel29)
    forecast = ForecastConfig(horizon=4, target_lags="cycle")
    target = build_v2(
        ModelSpec.from_panel(panel29, encoder29, forecast, 16, 4), **source.backbone_kwargs
    )
    init_from_pretrained(target, source, target_panel=panel29, target_forecast=forecast)
    uni = strip_covariates(panel29)
    source_batch = make_batch(
        uni,
        [(0, 25), (1, 25)],
        4,
        64,
        source.encoder,
        calendar_schema="canonical",
        target_lags="cycle",
    )
    task_batch = make_batch(panel29, [(0, 25), (1, 25)], 4, 64, encoder29, target_lags="cycle")
    with torch.no_grad():
        assert torch.allclose(
            target.eval()(task_batch), source.model.eval()(source_batch), atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize(
    "defect",
    [
        "calendar_order",
        "missing_lags",
        "channels",
        "forecast",
        "target_schema",
        "target_channels",
        "missing_layout",
    ],
)
def test_named_transfer_rejects_inconsistent_schema_before_mutation(
    panel29, encoder29, pretrained_univariate_bundle, defect
):
    from demand_forecast.core.transfer import TransferError, init_from_pretrained

    source = _named_source(pretrained_univariate_bundle, panel29)
    forecast = ForecastConfig(horizon=4, target_lags="cycle")
    target = build_v2(
        ModelSpec.from_panel(panel29, encoder29, forecast, 16, 4), **source.backbone_kwargs
    )
    if defect == "calendar_order":
        source.model.spec.calendar_schema = tuple(reversed(source.model.spec.calendar_schema))
    elif defect == "missing_lags":
        source.model.spec.lag_schema = None
    elif defect == "channels":
        source.model.spec.enc_channels += 2
    elif defect == "forecast":
        source.forecast_cfg.calendar_schema = "legacy"
    elif defect == "target_schema":
        target.spec.lag_schema = None
    elif defect == "target_channels":
        target.spec.dec_channels += 2
    before = {name: value.clone() for name, value in target.state_dict().items()}
    kwargs = (
        {} if defect == "missing_layout" else dict(target_panel=panel29, target_forecast=forecast)
    )
    with pytest.raises(TransferError, match="schema metadata|layout metadata"):
        init_from_pretrained(target, source, **kwargs)
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_task_initialization_supplies_layout_for_named_bundle(
    tmp_path, panel29, encoder29, pretrained_univariate_bundle
):
    from demand_forecast.core.bundle import save_bundle
    from demand_forecast.core.run import _initial_model
    from tests.helpers.settings import tiny_settings

    source = _named_source(pretrained_univariate_bundle, panel29)
    settings = tiny_settings()
    settings.forecast.target_lags = "cycle"
    settings.training.init_from = tmp_path / "source"
    save_bundle(
        settings.training.init_from,
        model=source.model,
        family="v2",
        backbone_kwargs=source.backbone_kwargs,
        forecast_cfg=source.forecast_cfg,
        dataset_cfg=None,
        encoder=source.encoder,
        panel=strip_covariates(panel29),
        training_cutoff=panel29.series[0].start,
        earliest_origin=panel29.series[0].start,
        manifest={"pretrain_identity": "tiny-v3"},
    )
    spec = ModelSpec.from_panel(panel29, encoder29, settings.forecast, 16, 4)
    target, report = _initial_model(settings, spec, panel29)
    assert report["pretrain_identity"] == "tiny-v3"
    assert "dec_proj.weight" in report["copied"]
    # Weekly month_sin is source slot 12 and destination slot 2.
    assert torch.equal(target.dec_proj.weight[:, 2], source.model.dec_proj.weight[:, 12])
