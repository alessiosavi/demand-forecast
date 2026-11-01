"""Pretrained transfer preserves forecasts while allowing new features to learn."""

from dataclasses import replace

import pytest
import torch

from demand_forecast.config.settings import ForecastConfig
from demand_forecast.core.bundle import load_bundle
from demand_forecast.data.encoders import StaticEncoder
from demand_forecast.data.panel import Panel, Roles
from demand_forecast.data.sampler import make_batch
from demand_forecast.models.base import ModelSpec
from demand_forecast.models.losses import pinball_loss
from tests.helpers.models import build_v2
from tests.helpers.settings import tiny_settings

pyarrow = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.bench


@pytest.fixture(scope="session")
def pretrained_univariate_bundle(tmp_path_factory):
    from demand_forecast.pretrain.config import PretrainConfig
    from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest
    from demand_forecast.pretrain.train import pretrain
    from tests.helpers.corpus_fixture import write_corpus_fixture

    directory = tmp_path_factory.mktemp("transfer-pretrain")
    root = directory / "corpus"
    for kind in ("tsmixup", "kernel_synth"):
        for index in range(2):
            write_corpus_fixture(root, kind=kind, index=index, seed=index)
    write_manifest(corpus_manifest(root), root / "manifest.json")
    cfg = PretrainConfig(
        corpus=dict(
            root=root,
            tsmixup_shards=[0],
            kernel_synth_shards=[0],
            holdout_tsmixup=[1],
            holdout_kernel_synth=[1],
            val_pairs_per_kind=8,
            external_validation=None,
        ),
        forecast=dict(horizon=4, max_context=16, min_context=8),
        model=tiny_settings().model,
        training=dict(
            max_steps=12,
            batch_size=4,
            eval_every=6,
            checkpoint_every=3,
            amp=False,
            warmup_steps=2,
        ),
        device="cpu",
    )
    result = pretrain(cfg, directory / "out")
    assert result.steps == 12
    return load_bundle(directory / "out/bundle")


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
