"""Causal backtest carriers and the quantiles of the evidence-enabled mix."""

import sys
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, ScreeningConfig, collate
from demand_forecast.data.episode.collate import EpisodeBatchWithRowKeys
from demand_forecast.data.episode.schema import NAIVE_OPTIONS, PATCH, QUANTILES, ROLE_SIBLING
from demand_forecast.models.v5 import V5Config, v5_loss
from demand_forecast.models.v5.batch_forecast import _batch_to_model
from demand_forecast.models.v5.model import prune
from tests.test_models.test_v5_naive_mix import ANCHOR, TINY, episode, model, panel

DERIVE = ANCHOR | {"naive_evidence"}
NONE, LAST, LEVEL, SEASON, SEASON2, DRIFT = range(len(NAIVE_OPTIONS))


def evidence_model():
    return model(config=replace(TINY, naive_mix=True, mix_evidence=True))


def with_values(values, *, freq=None):
    data = panel(freq=freq)
    return replace(data, items=(replace(data.items[0], targets={"y": values}),))


@pytest.mark.parametrize("origin_index", [0, 1])
@pytest.mark.parametrize("future", [1e6, np.nan])
def test_two_origins_only_use_their_own_past(origin_index, future):
    values = panel().items[0].targets["y"]
    values = values + np.random.default_rng(3).normal(0, 2, len(values))
    origins = (218, 234)
    local = origins[origin_index] - 13
    horizon = 32

    def build(raw):
        return episode(with_values(raw, freq="h"), DERIVE, origins=origins, horizon=horizon)

    later = values.copy()
    later[local:] = future
    earlier = values.copy()
    # Only the newest window can provide season2 evidence at either origin.
    earlier[local - horizon : local - 15] = np.nan
    original, untouched, changed = map(build, (values, later, earlier))
    for name in ("naive_error", "naive_error_seen", "naive_resq"):
        np.testing.assert_array_equal(
            getattr(original, name)[origin_index], getattr(untouched, name)[origin_index]
        )
    for name in ("naive_error", "naive_resq"):
        assert not np.allclose(
            getattr(original, name)[origin_index], getattr(changed, name)[origin_index]
        )
    assert original.naive_error_seen[origin_index, ..., SEASON2].all()
    assert not changed.naive_error_seen[origin_index, ..., SEASON2].any()
    net = evidence_model()
    with torch.no_grad():
        a, b, c = (net(collate([sample])) for sample in (original, untouched, changed))
    for name in ("mix", "fan", "quantiles"):
        expected, actual, moved = (getattr(out, name)[:, origin_index] for out in (a, b, c))
        torch.testing.assert_close(actual, expected)
        assert not torch.allclose(moved, expected)
    assert torch.equal(a.evidence[:, origin_index], b.evidence[:, origin_index])


@pytest.mark.parametrize("horizon", [1, 16, 17, 288])
@pytest.mark.parametrize("windows", [0, 1, 2, 5])
def test_window_bounds_and_last_residuals(horizon, windows):
    # The next older window begins before the item. Nonlinear outcomes distinguish
    # taking the newest four windows from pooling the entire observed history.
    local = windows * horizon + horizon - 1
    time = np.arange(local + horizon, dtype=float)
    values = (time + 1) ** 2 * np.where(time % 3 == 0, -1, 1)
    sample = episode(with_values(values), DERIVE, horizon=horizon, origins=(13 + local,))
    shape = (1, 1, (horizon + PATCH - 1) // PATCH, len(NAIVE_OPTIONS))
    assert sample.naive_error.shape == sample.naive_error_seen.shape == shape
    assert sample.naive_resq.shape == (*shape, len(QUANTILES))
    assert sample.naive_error.dtype == sample.naive_resq.dtype == np.float32
    assert sample.naive_error_seen.dtype == bool
    # No path is defined at cell 0, so skipping that window gives the same residuals.
    residuals = [
        (values[start : start + horizon] - values[start - 1]) / sample.scale[0]
        for k in range(1, min(4, windows) + 1)
        if (start := local - k * horizon) > 0
    ]
    if not residuals:
        assert not sample.naive_error_seen.any()
    else:
        assert sample.naive_error_seen[0, 0, :, LAST].all()
        for p, left in enumerate(range(0, horizon, PATCH)):
            pairs = np.asarray(residuals)[:, left : left + PATCH].flatten()
            np.testing.assert_allclose(sample.naive_error[0, 0, p, LAST], np.abs(pairs).mean())
            np.testing.assert_allclose(
                sample.naive_resq[0, 0, p, LAST], np.quantile(pairs, QUANTILES)
            )


def test_partial_patches_count_pooled_pairs_and_incomplete_windows_are_dropped():
    values = np.zeros(85)
    values[17:34] = 2
    values[18:30] = np.nan  # Five finite outcomes, so this whole window is dropped.
    values[34:39] = 4
    values[39:50] = np.nan
    values[50] = 4  # Six finite outcomes, so this window is dropped too.
    values[51:58] = 6
    values[58:67] = np.nan
    values[67] = 6
    values[68:] = 8
    # Newest window has nine outcomes: eight in patch 0 and one in patch 1.
    values[59] = 6
    sample = episode(with_values(values), DERIVE, horizon=17, origins=(81,))
    assert sample.naive_error_seen[0, 0, :, NONE].tolist() == [True, True]
    np.testing.assert_allclose(sample.naive_error[0, 0, :, NONE] * sample.scale[0], [6, 6])
    values[59] = np.nan
    values[58] = 6
    values[60] = 6
    values[67] = np.nan
    # Nine outcomes in patch 0, no pair in the last patch.
    sample = episode(with_values(values), DERIVE, horizon=17, origins=(81,))
    assert sample.naive_error_seen[0, 0, :, NONE].tolist() == [True, False]

    values[:] = 0
    values[27:33] = values[44:50] = np.nan
    # H=17, both windows have enough outcomes. Their last patch has a pair
    # in only the older window, which suffices for the one-cell patch.
    values[50] = np.nan
    sample = episode(with_values(values), DERIVE, horizon=17, origins=(64,))
    assert sample.naive_error_seen[0, 0, :, NONE].tolist() == [True, True]


def test_full_patch_needs_eight_pairs_across_windows():
    values = np.arange(128, dtype=float)
    values[49:64] = values[87:96] = np.nan
    # Both windows have at least 16 finite outcomes, but neither supplies
    # eight pairs to the second patch by itself.
    sample = episode(with_values(values), DERIVE, horizon=32, origins=(109,))
    assert sample.naive_error_seen[0, 0, :, LAST].tolist() == [True, True]
    values[48] = np.nan
    sample = episode(with_values(values), DERIVE, horizon=32, origins=(109,))
    assert sample.naive_error_seen[0, 0, :, LAST].tolist() == [True, False]
    assert not sample.naive_resq[0, 0, 1, LAST].any()


def test_windows_extending_beyond_the_stored_item_are_dropped():
    values = np.arange(40, dtype=float) ** 2
    sample = episode(with_values(values), DERIVE, horizon=16, origins=(61,))
    # [32, 48) has eight observations but is outside the stored history.
    pairs = (values[16:32] - values[15]) / sample.scale[0]
    assert sample.naive_error_seen[0, 0, 0, LAST]
    np.testing.assert_allclose(sample.naive_error[0, 0, 0, LAST], pairs.mean())
    np.testing.assert_allclose(sample.naive_resq[0, 0, 0, LAST], np.quantile(pairs, QUANTILES))


def test_none_uses_the_median_at_the_window_start():
    values = np.r_[np.zeros(16), np.full(32, 10.0)]
    sample = episode(
        with_values(values),
        DERIVE,
        horizon=32,
        origins=(61,),
        screening=ScreeningConfig(scale_floor=0.15),
    )
    assert sample.loc[0] == 10
    assert sample.scale[0] == pytest.approx(1)
    assert sample.naive_error_seen[..., NONE].all()
    np.testing.assert_allclose(sample.naive_error[..., NONE], 10)
    np.testing.assert_allclose(sample.naive_resq[..., NONE, :], 10)


def test_line_in_episode_units_with_a_scale_floor():
    values = 1000 + np.arange(140) * 0.2
    sample = episode(
        with_values(values),
        DERIVE,
        horizon=17,
        origins=(113,),
        screening=ScreeningConfig(scale_floor=0.3),
    )
    assert sample.scale[0] == pytest.approx(0.3 * values[:100].mean())
    for p, leads in enumerate((np.arange(1, 17), np.array([17]))):
        residuals = 0.2 * leads / sample.scale[0]
        np.testing.assert_allclose(sample.naive_error[0, 0, p, LAST], residuals.mean())
        np.testing.assert_allclose(
            sample.naive_resq[0, 0, p, LAST], np.quantile(np.tile(residuals, 4), QUANTILES)
        )


def test_exact_season_and_long_lag_history_requirement():
    sample = episode(
        with_values(np.tile(np.arange(24, dtype=float), 12), freq="h"),
        DERIVE,
        horizon=17,
        origins=(205,),
    )
    assert sample.naive_error_seen[..., SEASON].all()
    assert not sample.naive_error[..., SEASON].any()
    # Only one backtest origin has 168 earlier cells. Its signed residuals
    # vanish on this 24-period series, because 168 is a multiple of 24.
    assert sample.naive_error_seen[..., SEASON2].all()
    assert not sample.naive_resq[..., SEASON2, :].any()
    early = episode(
        with_values(np.tile(np.arange(24, dtype=float), 12), freq="h"),
        DERIVE,
        horizon=17,
        origins=(181,),
    )
    assert early.naive_seen[..., SEASON2].any()
    assert not early.naive_error_seen[..., SEASON2].any()
    assert torch.isfinite(evidence_model()(collate([sample])).mix).all()


def test_long_lag_evidence_is_admitted_per_backtest_origin():
    values = np.tile(np.arange(24, dtype=float), 12)
    data = with_values(values, freq="h")
    sample = EpisodeBuilder(data).build(EpisodeRequest(0, (153, 217), 17, 96, derive=DERIVE))
    # N6's forecast paths retain the first-origin rule. The later backtest
    # itself has enough earlier history to measure the long lag.
    assert not sample.naive_seen[..., SEASON2].any()
    assert not sample.naive_error_seen[0, ..., SEASON2].any()
    assert sample.naive_error_seen[1, ..., SEASON2].all()
    assert not sample.naive_error[1, ..., SEASON2].any()


def test_large_errors_and_large_paths_are_unseen():
    values = np.r_[np.zeros(16), np.full(16, 2e6), np.zeros(32)]
    sample = EpisodeBuilder(with_values(values)).build(
        EpisodeRequest(0, (45, 61), 16, 48, cutoff=29, derive=DERIVE)
    )
    assert not sample.naive_error_seen[0, ..., LAST].any()  # Residual beyond cap.
    assert not sample.naive_error_seen[1, ..., LAST].any()  # Path beyond cap.
    assert not sample.naive_resq[..., LAST, :].any()


def test_untrained_mix_equals_the_batch_reference_and_legacy_fallback():
    batch = collate([episode(panel(), DERIVE, horizon=17)])
    seen = batch.naive_error_seen.clone()
    seen[..., 0, LAST] = False  # A path without evidence retains the floor.
    seen[..., 1, :] = False  # The last real cell uses the N6 bias and head.
    batch = replace(batch, naive_error_seen=seen)
    net, legacy = evidence_model(), model()
    state, legacy_state = net.state_dict(), legacy.state_dict()
    additions = tuple(name for name in state if name not in legacy_state)
    assert additions == ("evidence_bias", "beta_raw", "gamma_raw", "head_gain")
    assert sum(state[name].numel() for name in additions) == 9
    assert net.head_gain.ndim == 0
    assert net.head_gain.item() == pytest.approx(1e-3)
    with torch.no_grad():
        out, old = net(batch), legacy(batch)
    paths = batch.naive_paths[0, 0, 0, :16]
    available = batch.naive_seen[0, 0, 0, :16]
    present = available & seen[0, 0, 0, 0]
    err = batch.naive_error[0, 0, 0, 0]
    prior = -4 * (err + 1e-3).log()
    prior = (prior - prior.masked_fill(~present, -torch.inf).amax(-1, keepdim=True)).clamp_min(-6)
    weights = prior.masked_fill(~present, -6).masked_fill(~available, -torch.inf).softmax(-1)
    fan_weights = weights * present
    fan_weights /= fan_weights.sum(-1, keepdim=True)
    fan = fan_weights @ batch.naive_resq[0, 0, 0, 0]
    anchor = (weights * paths).sum(-1)
    head = legacy(replace(batch, naive_seen=torch.zeros_like(batch.naive_seen))).quantiles
    knots = anchor[:, None] + fan + 1e-3 * head[0, 0, 0, :16]
    for actual, expected in (
        (out.mix[0, 0, 0, :16], weights),
        (out.fan[0, 0, 0, :16], fan),
        (out.anchor[0, 0, 0, :16], anchor),
        (out.quantiles[0, 0, 0, :16], knots),
    ):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
    assert (out.mix[0, 0, 0, :16, LAST] > 0).all()
    assert out.evidence[0, 0, 0].tolist() == [True] * 16 + [False] * 16
    assert not out.fan[..., 16:, :].any()
    torch.testing.assert_close(
        out.quantiles[..., 16:, :], old.quantiles[..., 16:, :], atol=0, rtol=0
    )
    no_arrays = collate([episode(panel(), ANCHOR)])
    with torch.no_grad():
        net.mix_projection.weight.normal_(std=0.5, generator=torch.Generator().manual_seed(11))
        legacy.mix_projection.weight.copy_(net.mix_projection.weight)
        no_evidence = net(no_arrays)
        torch.testing.assert_close(no_evidence.quantiles, legacy(no_arrays).quantiles)
        torch.testing.assert_close(no_evidence.fan, torch.zeros_like(no_evidence.quantiles))
        assert not no_evidence.evidence.any()
        torch.testing.assert_close(
            net(batch).quantiles[..., 16:, :],
            legacy(batch).quantiles[..., 16:, :],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("at_zero", [False, True])
def test_training_loss_has_a_nonzero_head_gain_gradient(at_zero):
    net = evidence_model()
    if at_zero:
        with torch.no_grad():
            net.head_gain.zero_()
    batch = collate([episode(panel(), DERIVE)])
    loss, _ = v5_loss(net(batch), batch, net.config)
    (gradient,) = torch.autograd.grad(loss, net.head_gain)
    assert torch.isfinite(gradient)
    assert gradient.abs().item() > 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantiles_stay_sorted_with_random_masks_and_explicit_precision(dtype):
    batch = _batch_to_model(collate([episode(panel(), DERIVE)]), torch.device("cpu"), dtype)
    gen = torch.Generator().manual_seed(3)
    seen = torch.rand(batch.naive_seen.shape, generator=gen) > 0.4
    errors_seen = torch.rand(batch.naive_error_seen.shape, generator=gen) > 0.4
    resq = torch.rand(batch.naive_resq.shape, generator=gen).sort(dim=-1).values
    batch = replace(batch, naive_seen=seen, naive_error_seen=errors_seen, naive_resq=resq)
    net = evidence_model().to(dtype)
    with torch.no_grad():
        net.head_gain.fill_(-0.5)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.float32):
        out = net(batch)
    for tensor in (out.quantiles, out.mix, out.anchor, out.fan):
        assert tensor.dtype == dtype
        assert torch.isfinite(tensor).all()
    assert (out.quantiles.diff(dim=-1) >= 0).all()
    assert not out.evidence[..., 19:].any()


@pytest.mark.parametrize("errors", [(0, 1), (1e-9, 1e6)])
def test_prior_caps_ratios_and_fan_does_not_backpropagate_into_logits(errors):
    net = evidence_model()
    batch = collate([episode(panel(), DERIVE)])
    paths, error_seen = torch.zeros_like(batch.naive_seen), torch.zeros_like(batch.naive_error_seen)
    paths[..., :2] = error_seen[..., :2] = True
    error = torch.zeros_like(batch.naive_error)
    error[..., :2] = torch.tensor(errors)
    batch = replace(batch, naive_seen=paths, naive_error_seen=error_seen, naive_error=error)
    out = net(batch)
    weights = out.mix[0, 0, 0, 0, :2]
    assert (weights[0] / weights[1]).item() == pytest.approx(np.exp(6), rel=1e-6)
    (gradient,) = torch.autograd.grad(weights[1], net.evidence_bias, retain_graph=True)
    assert torch.isfinite(gradient).all() and gradient[1] > 0
    grad_mix, grad_beta, grad_gamma = torch.autograd.grad(
        out.fan.sum(), (net.evidence_bias, net.beta_raw, net.gamma_raw), allow_unused=True
    )
    assert grad_mix is grad_beta is None
    assert torch.isfinite(grad_gamma) and grad_gamma != 0


@pytest.mark.parametrize("keyed", [False, True])
def test_mixed_padding_prune_and_device_dtype_moves(keyed):
    from demand_forecast.data.episode.collate import EpisodeBatchWithNaiveEvidence

    first = episode(panel(sibling=True), DERIVE, origins=(205, 221), horizon=17)
    second = episode(panel(), ANCHOR, horizon=37)
    batch = collate([first, second], include_row_keys=keyed).to("cpu")
    assert isinstance(batch, EpisodeBatchWithNaiveEvidence)
    assert isinstance(batch, EpisodeBatchWithRowKeys) is keyed
    assert batch.naive_error.shape == (2, 2, 2, 3, 6)
    assert batch.naive_resq.shape == (2, 2, 2, 3, 6, 9)
    assert not batch.naive_error_seen[1].any()
    assert not batch.naive_error_seen[0, :, 0, 2].any()
    assert not batch.naive_error_seen[0, :, 1].any()
    assert not batch.naive_resq[1].any()
    net = evidence_model()
    with torch.no_grad():
        gates = net.gates(batch, zero_rows=batch.row_role == ROLE_SIBLING)
        small, small_gates = prune(batch, gates)
        full, trimmed = net(batch, gates), net(small, small_gates)
    assert type(small) is type(batch)
    for name in ("naive_error", "naive_error_seen", "naive_resq"):
        torch.testing.assert_close(getattr(small, name), getattr(batch, name)[:, :, :1])
    torch.testing.assert_close(trimmed.quantiles, full.quantiles[:, :, :1], atol=1e-5, rtol=1e-5)
    half = _batch_to_model(small, torch.device("cpu"), torch.float16)
    assert half.x.dtype == torch.float16
    assert half.naive_error.dtype == half.naive_resq.dtype == torch.float32
    assert half.naive_error_seen.dtype == torch.bool


@pytest.mark.skipif(
    sys.platform != "linux", reason="evidence batch transport requires Linux workers"
)
@pytest.mark.parametrize("keyed", [False, True])
def test_evidence_batch_loader_worker_transport(keyed):
    from torch.utils.data import DataLoader

    batch = collate([episode(panel(), DERIVE)], include_row_keys=keyed)
    (actual,) = list(DataLoader([batch], batch_size=None, num_workers=1, timeout=10))
    assert type(actual) is type(batch)
    for field in fields(batch):
        expected, value = getattr(batch, field.name), getattr(actual, field.name)
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(value, expected)
        else:
            assert value == expected


def test_mix_evidence_config_requires_the_mix_and_fixed_quantiles():
    with pytest.raises(ValueError, match="naive_mix"):
        V5Config(mix_evidence=True)
    with pytest.raises(ValueError, match="quantiles"):
        V5Config(naive_mix=True, mix_evidence=True, quantiles=(0.1, 0.5, 0.9))


def guarded_model(*, fan_mode="residual", mix_bound=2.0):
    return model(
        config=replace(
            TINY, naive_mix=True, mix_evidence=True, fan_mode=fan_mode, mix_bound=mix_bound
        )
    )


def two_measured_paths(corrections=(0, 3)):
    batch = collate([episode(panel(), DERIVE)])
    seen = torch.zeros_like(batch.naive_seen)
    seen[..., :3] = True
    measured = torch.zeros_like(batch.naive_error_seen)
    measured[..., :2] = True
    paths = torch.zeros_like(batch.naive_paths)
    paths[..., :2] = torch.tensor([1.0, 4.0])
    paths[..., 2] = 2
    errors = torch.zeros_like(batch.naive_error)
    errors[..., :2] = torch.tensor([0.1, 0.2])
    residuals = torch.zeros_like(batch.naive_resq)
    spread = torch.tensor([[1.0], [2.0]]) * torch.arange(-4.0, 5.0)
    residuals[..., :2, :] = torch.tensor(corrections)[:, None] + spread
    return replace(
        batch,
        naive_seen=seen,
        naive_paths=paths,
        naive_error_seen=measured,
        naive_error=errors,
        naive_resq=residuals,
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"fan_mode": "learned"}, "fan_mode"),
        ({"fan_mode": True}, "fan_mode"),
        ({"fan_mode": "residual", "mix_evidence": False}, "requires mix_evidence"),
        ({"mix_bound": 2, "mix_evidence": False}, "requires mix_evidence"),
        *[
            ({"mix_bound": value}, "mix_bound")
            for value in (True, "2", 0, -1, 0.09, 10.01, float("inf"), float("-inf"), float("nan"))
        ],
    ],
)
def test_evidence_guards_reject_invalid_configurations(options, message):
    with pytest.raises(ValueError, match=message):
        V5Config(**({"naive_mix": True, "mix_evidence": True} | options))


def test_guarded_start_matches_corrected_paths_and_the_residual_fan():
    batch = two_measured_paths()
    guarded, n7 = guarded_model(), evidence_model()
    assert tuple(guarded.state_dict()) == tuple(n7.state_dict())
    guarded.load_state_dict(n7.state_dict(), strict=True)
    with torch.no_grad():
        out, old = guarded(batch), n7(batch)
        head = n7(replace(batch, naive_seen=torch.zeros_like(batch.naive_seen))).quantiles
    index = (0, 0, 0, 0)
    prior = torch.tensor([0.0, -4 * np.log(0.201 / 0.101), -6.0], dtype=out.mix.dtype)
    mix = prior.softmax(0)
    measured_mix = prior[:2].softmax(0)
    centre = mix[1] * 3
    spread = (measured_mix[0] + 2 * measured_mix[1]) * torch.arange(-4.0, 5.0)
    anchor = (mix * torch.tensor([1.0, 4.0, 2.0])).sum()
    torch.testing.assert_close(out.mix[index][:3], mix)
    torch.testing.assert_close(out.anchor[index], anchor)
    torch.testing.assert_close(out.fan[index], centre + spread)
    torch.testing.assert_close(out.quantiles[index], anchor + centre + spread + 1e-3 * head[index])
    # N7 gives the median the entire measured softmax mass. G1 keeps the mixed
    # mask's measured mass while preserving the spread around that median.
    torch.testing.assert_close(old.fan[index][4], measured_mix[1] * 3)
    torch.testing.assert_close(out.fan[index][4], mix[:2].sum() * old.fan[index][4])
    torch.testing.assert_close(
        out.fan[index] - out.fan[index][4], old.fan[index] - old.fan[index][4]
    )
    assert out.evidence[index]
    assert not out.tilt_saturated.any()
    assert old.tilt_saturated is None


@pytest.mark.parametrize(("correction", "factor"), [(3, 2), (-3, 0)])
def test_residual_centre_can_reinforce_or_cancel_the_anchor_gradient(correction, factor):
    batch = two_measured_paths((0, correction))
    # Only these two paths are seen, so both corrected paths equal 1 when
    # the second correction is -3.
    seen = batch.naive_seen.clone()
    seen[..., 2] = False
    net = guarded_model()
    out = net(replace(batch, naive_seen=seen))
    weight = net.mix_projection.weight
    index = (0, 0, 0, 0)
    (anchor_grad,) = torch.autograd.grad(out.anchor[index], weight, retain_graph=True)
    (median_grad,) = torch.autograd.grad(out.quantiles[index][4], weight, retain_graph=True)
    (spread_grad,) = torch.autograd.grad(out.fan[index][-1] - out.fan[index][0], weight)
    assert anchor_grad.norm() > 1e-6
    torch.testing.assert_close(median_grad, factor * anchor_grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(spread_grad, torch.zeros_like(spread_grad), atol=1e-7, rtol=0)


@pytest.mark.parametrize("fan_mode", ["head", "residual"])
@pytest.mark.parametrize("bound", [None, 2.0])
def test_evidence_calibration_gradients_follow_each_guard(fan_mode, bound):
    net = guarded_model(fan_mode=fan_mode, mix_bound=bound)
    out = net(two_measured_paths())
    grad_beta, grad_gamma = torch.autograd.grad(
        out.quantiles[0, 0, 0, 0].sum(), (net.beta_raw, net.gamma_raw), allow_unused=True
    )
    if bound is None:
        assert grad_beta is not None and torch.isfinite(grad_beta) and grad_beta.abs() > 1e-6
    else:
        assert grad_beta is None
    if fan_mode == "head":
        assert grad_gamma is not None and torch.isfinite(grad_gamma) and grad_gamma.abs() > 1e-6
    else:
        assert grad_gamma is None


def test_fixed_head_spread_breaks_flat_evidence_ties_without_training_the_head_tails():
    batch = two_measured_paths()
    # One measured option at zero residual, with all other available options
    # at the prior floor. The bounded network tilt is zero on this cell.
    measured = batch.naive_error_seen.clone()
    measured[..., 1] = False
    batch = replace(batch, naive_error_seen=measured, naive_resq=torch.zeros_like(batch.naive_resq))
    net = guarded_model()
    index = (0, 0, 0, 0)
    out = net(batch)
    head = net.head.projection.bias
    assert out.evidence[index]
    assert not out.tilt_saturated[index]
    torch.testing.assert_close(
        out.mix[index][0] / out.mix[index][1], torch.tensor(np.exp(6)).float()
    )
    torch.testing.assert_close(out.fan[index], torch.zeros_like(out.fan[index]))
    assert (out.quantiles[index].diff() > 0).all()
    (spread_grad,) = torch.autograd.grad(out.quantiles[index].diff().sum(), head, retain_graph=True)
    torch.testing.assert_close(spread_grad, torch.zeros_like(spread_grad), atol=1e-7, rtol=0)
    grad_head, grad_mix = torch.autograd.grad(
        out.quantiles[index].sum(), (head, net.mix_projection.weight)
    )
    assert torch.isfinite(grad_head).all() and torch.isfinite(grad_mix).all()
    assert grad_head[4].abs() > 1e-6
    with torch.no_grad():
        net.head_gain.zero_()
    (gain_grad,) = torch.autograd.grad(net(batch).quantiles[index][4], net.head_gain)
    assert torch.isfinite(gain_grad) and gain_grad.abs() > 1e-6


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_guarded_quantiles_stay_sorted_with_random_masks_and_extreme_decoder_steps(dtype):
    batch = _batch_to_model(two_measured_paths(), torch.device("cpu"), dtype)
    rng = torch.Generator().manual_seed(21)
    batch = replace(
        batch,
        naive_seen=torch.rand(batch.naive_seen.shape, generator=rng) > 0.4,
        naive_error_seen=torch.rand(batch.naive_error_seen.shape, generator=rng) > 0.4,
    )
    net = guarded_model().to(dtype)
    with torch.no_grad():
        net.head_gain.fill_(-0.5)
        net.step_projection.bias[::2].fill_(1e4)
        net.step_projection.bias[1::2].fill_(-1e4)
        net.mix_projection.weight.normal_(generator=rng)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.float32):
        out = net(batch)
    for tensor in (out.quantiles, out.mix, out.anchor, out.fan):
        assert tensor.dtype == dtype
        assert torch.isfinite(tensor).all()
    assert (out.quantiles.diff(dim=-1) >= 0).all()
    assert not out.evidence[..., 19:].any()
    assert not out.tilt_saturated[..., 19:].any()


@pytest.mark.parametrize("bound", [0.1, 10])
@pytest.mark.parametrize("amplitude", [0.0, 1e4])
def test_bounded_logits_center_only_measured_options_and_keep_the_floor(bound, amplitude):
    net = guarded_model(mix_bound=bound)
    batch = two_measured_paths()
    steps = batch.naive_paths.new_ones((*batch.naive_paths.shape[:-1], net.config.d_out))
    with torch.no_grad():
        net.mix_projection.weight[0, 0] = amplitude
        net.mix_projection.weight[1, 0] = -amplitude
        net.evidence_bias[2] = 1e4  # Unmeasured, so it cannot affect the mean or saturate.
    logits, options, evidence, saturated = net._evidence_logits(steps, batch, batch.naive_seen)
    tilt = bound * np.tanh(amplitude / bound)
    expected = logits.new_tensor([tilt, -4 * np.log(0.201 / 0.101) - tilt, -6])
    torch.testing.assert_close(logits[0, 0, 0, 0, :3], expected, atol=1e-6, rtol=1e-6)
    assert options[..., :2].all() and not options[..., 2:].any()
    assert evidence.all()
    assert saturated.dtype == torch.bool
    assert saturated.shape == batch.naive_paths.shape[:-1]
    assert saturated.all() if amplitude else not saturated.any()
    grads = torch.autograd.grad(
        logits[..., 0].sum(), (net.mix_projection.weight, net.evidence_bias)
    )
    assert all(torch.isfinite(grad).all() for grad in grads)
    with torch.no_grad():
        net.evidence_bias.add_(123)
        shifted, _, _, shifted_saturated = net._evidence_logits(steps, batch, batch.naive_seen)
    torch.testing.assert_close(shifted, logits)
    assert torch.equal(shifted_saturated, saturated)


def test_mix_bound_saturation_uses_the_centred_logit_before_tanh():
    net = guarded_model(mix_bound=2.0)
    batch = two_measured_paths()
    with torch.no_grad():
        net.evidence_bias[:2] = torch.tensor([5.0, 5.0 + 2 * 1.89])
        below = net(batch).tilt_saturated
        net.evidence_bias[:2] = torch.tensor([5.0, 5.0 + 2 * 1.91])
        above = net(batch)
    assert not below.any()
    assert torch.equal(above.tilt_saturated, above.evidence)


@pytest.mark.parametrize("keyed", [False, True])
def test_guards_leave_mixed_pruned_and_padded_cells_without_evidence_on_the_old_path(keyed):
    first = episode(panel(sibling=True), DERIVE, origins=(205, 221), horizon=17)
    second = episode(panel(), ANCHOR, horizon=37)
    batch = collate([first, second], include_row_keys=keyed)
    net, old = guarded_model(), evidence_model()
    with torch.no_grad():
        gates = net.gates(batch, zero_rows=batch.row_role == ROLE_SIBLING)
        small, small_gates = prune(batch, gates)
        result, expected, trimmed = net(batch, gates), old(batch, gates), net(small, small_gates)
        plain = net(collate([second], include_row_keys=keyed))
    assert not plain.tilt_saturated.any()
    assert not result.evidence[1].any()
    assert not result.evidence[0, :, :, 17:].any()
    assert not result.tilt_saturated[~result.evidence].any()
    assert not result.fan[~result.evidence].any()
    for key in ("anchor", "mix", "quantiles"):
        torch.testing.assert_close(
            getattr(result, key)[~result.evidence], getattr(expected, key)[~result.evidence]
        )
    torch.testing.assert_close(trimmed.quantiles, result.quantiles[:, :, :1], atol=1e-5, rtol=1e-5)
    assert torch.equal(trimmed.tilt_saturated, result.tilt_saturated[:, :, :1])
