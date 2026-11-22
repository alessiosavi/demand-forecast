"""Compressed quantiles, loss and evidence composition."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import collate
from demand_forecast.data.episode.derived import NAIVE_LIMIT, NaiveHistory
from demand_forecast.data.episode.scale import Scale
from demand_forecast.data.episode.schema import PATCH
from demand_forecast.models.v5 import V5Config, v5_loss
from demand_forecast.models.v5.batch_forecast import _batch_to_model
from tests.test_models.test_v5_mix_evidence import DERIVE, NONE, two_measured_paths
from tests.test_models.test_v5_naive_mix import TINY, episode, model, panel


def _net(*, fan_mode="head", space="asinh", mix=True, evidence=True, quantiles=TINY.quantiles):
    return model(
        mix=mix,
        config=replace(
            TINY,
            naive_mix=mix,
            mix_evidence=evidence,
            fan_mode=fan_mode,
            mix_bound=2.0 if evidence else None,
            output_space=space,
            quantiles=quantiles,
        ),
    )


@pytest.mark.parametrize("value", ["log", True, None])
def test_output_space_validation(value):
    with pytest.raises(ValueError, match="output_space"):
        V5Config(output_space=value)


@pytest.mark.parametrize("fan_mode", ["head", "residual"])
@pytest.mark.parametrize("precision", ["float32", "bfloat16", "float64"])
def test_asinh_formulas_match_mixed_paths_and_the_existing_masks(fan_mode, precision):
    if precision == "bfloat16" and not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("CPU autocast is unavailable")
    first = episode(panel(), DERIVE, horizon=17, origins=(189, 205))
    second = episode(panel(), DERIVE, horizon=17, origins=(205,))
    batch = collate([first, second, second])
    seen = batch.naive_seen.clone()
    measured = batch.naive_error_seen.clone()
    forecast_mask = batch.forecast_mask.clone()
    # Step 1 sees only an unmeasured path in the first patch.
    seen[0, 0, 0, 1] = False
    seen[0, 0, 0, 1, -1] = True
    measured[0, 0, 0, 0, -1] = False
    seen[0, 0, 0, 2] = False
    measured[0, 0, 0, 1] = False
    forecast_mask[-1] = False
    batch = replace(batch, naive_seen=seen, naive_error_seen=measured, forecast_mask=forecast_mask)
    dtype = torch.float64 if precision == "float64" else torch.float32
    net = _net(fan_mode=fan_mode).to(dtype)
    batch = _batch_to_model(batch, torch.device("cpu"), dtype)
    heads = []
    with (
        net.head.register_forward_hook(lambda _, _inputs, head: heads.append(head)),
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bfloat16"),
    ):
        result = net(batch)
    (head,) = heads
    assert head.abs().max() > 0.1
    real = batch.origin_mask[:, :, None, None] & batch.forecast_mask[:, None, :, None]
    steps = torch.arange(head.shape[-2]) < 17
    with_path = seen.any(-1) & real & steps
    assert result.evidence[0, 0, 0, 0]
    assert with_path[0, 0, 0, 1] and not result.evidence[0, 0, 0, 1]
    assert not with_path[0, 0, 0, 2]
    assert not result.evidence[..., 17:].any()
    assert not real.all()
    assert result.compressed.dtype == result.quantiles.dtype == dtype
    gain = net.head_gain.abs()
    terms = gain * head
    if fan_mode == "residual":
        median = head[..., 4:5]
        terms = gain * median + 1e-3 * (head - median)
    expected = torch.where(with_path[..., None], result.anchor[..., None].asinh() + head, head)
    evidence_value = (result.anchor[..., None] + result.fan).asinh() + terms
    expected = torch.where(result.evidence[..., None], evidence_value, expected)
    expected = expected.masked_fill(~real[..., None], 0)
    torch.testing.assert_close(result.compressed, expected)
    torch.testing.assert_close(result.quantiles, expected.sinh())
    assert (result.quantiles.diff(dim=-1) >= 0).all()


@pytest.mark.parametrize("mix", [False, True])
def test_asinh_head_and_anchor_work_with_three_quantiles(mix):
    batch = collate([episode(panel(), DERIVE, horizon=17)])
    net = _net(mix=mix, evidence=False, quantiles=(0.1, 0.5, 0.9))
    heads = []
    with net.head.register_forward_hook(lambda _, _inputs, head: heads.append(head)):
        output = net(batch)
    (expected,) = heads
    assert expected.abs().max() > 0.1
    if mix:
        with_path = batch.naive_seen.any(-1)
        assert with_path.any()
        expected = expected + output.anchor[..., None].asinh()
    torch.testing.assert_close(output.compressed, expected)
    torch.testing.assert_close(output.quantiles, expected.sinh())


@pytest.mark.parametrize("fan_mode", ["head", "residual"])
def test_start_matches_linear_within_head_terms_in_compressed_units(fan_mode):
    batch = two_measured_paths()
    net = _net(fan_mode=fan_mode)
    linear = _net(fan_mode=fan_mode, space="linear")
    heads = []
    with (
        torch.no_grad(),
        net.head.register_forward_hook(lambda _, _inputs, head: heads.append(head)),
    ):
        before, after = linear(batch), net(batch)
    assert before.compressed is None
    assert after.evidence.any()
    base = (after.anchor[..., None] + after.fan)[after.evidence]
    head = heads[0][after.evidence]
    terms = 1e-3 * head
    if fan_mode == "residual":
        terms = 1e-3 * head[..., 4:5] + 1e-3 * (head - head[..., 4:5])
    torch.testing.assert_close(
        after.compressed[after.evidence] - base.asinh(), terms, atol=1e-6, rtol=1e-5
    )
    difference = (after.compressed[after.evidence] - before.quantiles[after.evidence].asinh()).abs()
    assert (difference <= terms.abs() + 1e-6).all()


@pytest.mark.parametrize("fan_mode", ["head", "residual"])
@pytest.mark.parametrize(("options", "gain"), [(2, 0.5), (1, 0.5), (2, 0.0)])
def test_asinh_pinball_value_and_evidence_gradients(fan_mode, options, gain):
    batch = two_measured_paths(corrections=(0, 3))
    measured = batch.naive_error_seen.clone()
    measured[..., options:] = False
    seen = batch.naive_seen.clone()
    seen[..., options:] = False
    mask = torch.zeros_like(batch.label_mask)
    mask[..., 0] = True
    batch = replace(
        batch,
        naive_error_seen=measured,
        naive_seen=seen,
        labels=torch.full_like(batch.labels, 1e6),
        label_mask=mask,
    )
    net = _net(fan_mode=fan_mode)
    with torch.no_grad():
        net.head_gain.fill_(gain)
    output = net(batch)
    assert output.evidence[mask].all()
    assert not output.tilt_saturated.any()
    error = batch.labels[mask].asinh()[:, None]
    error = error - output.compressed[mask]
    levels = torch.tensor(net.config.quantiles)
    expected = (2 * torch.maximum(levels * error, (levels - 1) * error)).mean()
    loss, parts = v5_loss(output, batch, net.config)
    assert parts["pinball"] == pytest.approx(expected.item())
    assert parts["n_labels"] == pytest.approx(1)
    assert torch.isfinite(loss)
    loss.backward()
    for grad in (param.grad for param in net.parameters() if param.grad is not None):
        assert torch.isfinite(grad).all()
    assert net.head_gain.grad.abs() > 0
    assert bool(net.mix_projection.weight.grad.abs().max() > 0) is (options == 2)
    assert bool(net.evidence_bias.grad.abs().max() > 0) is (options == 2)
    grad = net.head.projection.bias.grad
    assert bool(grad[4].abs() > 0) is (gain != 0)
    assert bool(grad[torch.arange(9) != 4].abs().max() > 0) is (gain != 0 and fan_mode == "head")

    capped, parts = v5_loss(output, batch, net.config, label_cap=1000)
    assert torch.isfinite(capped)
    assert parts["capped"] == 1
    assert parts["n_labels"] == pytest.approx(0)
    assert parts["pinball"] == pytest.approx(0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_asinh_inverse_clamps_but_loss_uses_unclamped_knots(sign):
    batch = collate([episode(panel(), frozenset(), horizon=17)])
    net = _net(mix=False, evidence=False)
    with torch.no_grad():
        net.head.projection.weight.zero_()
        net.head.projection.bias[4] = 95 * sign
    output = net(batch)
    median = output.compressed[..., 4]
    assert (median.abs() > 80).all()
    torch.testing.assert_close(output.quantiles[..., 4], torch.full_like(median, 80 * sign).sinh())
    loss, parts = v5_loss(output, batch, net.config)
    assert torch.isfinite(loss) and parts["pinball"] > 80
    loss.backward()
    assert net.head.projection.bias.grad[4].abs() > 0


@pytest.mark.parametrize("injected", [False, True], ids=["backtest-limit", "injected-1e15"])
def test_asinh_large_backtest_residuals_are_finite_and_ordered(injected):
    batch = collate([episode(panel(), DERIVE, horizon=PATCH)])
    seen = torch.zeros_like(batch.naive_seen)
    seen[..., NONE] = True
    measured = torch.zeros_like(batch.naive_error_seen)
    measured[..., NONE] = True
    residuals = torch.zeros_like(batch.naive_resq)
    if injected:
        # Float32 can resolve knots spaced 2e13 apart at a centre of 1e15.
        residuals[..., NONE, :] = 1e15 + 2e13 * torch.arange(-4, 5)
    else:
        n = 4 * PATCH
        # A ramp makes every interpolated knot distinct. The pulse uses nearly the entire budget.
        raw = np.r_[np.zeros(PATCH), np.arange(1, n, dtype=float) * 100, n * NAIVE_LIMIT - 210_000]
        history = NaiveHistory(raw, season=7, long_lag=None, end=len(raw))
        error, admitted, knots = history.backtest(raw, Scale(0, 1), origin=len(raw), horizon=PATCH)
        assert admitted[0, NONE] and 0.99 * NAIVE_LIMIT < error[0, NONE] <= NAIVE_LIMIT
        assert raw[-1] > 0.99 * n * NAIVE_LIMIT
        residuals[..., NONE, :] = torch.from_numpy(knots[0, NONE])
    batch = replace(batch, naive_seen=seen, naive_error_seen=measured, naive_resq=residuals)
    out = _net()(batch)
    assert out.quantiles.dtype == torch.float32 and out.evidence.all()
    assert torch.isfinite(out.quantiles).all()
    assert (out.quantiles.diff(dim=-1) >= 0).all()
    if not injected:
        assert (out.quantiles.diff(dim=-1) > 0).all()
