"""Causal backtest carriers and the quantiles of the evidence-enabled mix."""

import sys
from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from torch.nn.functional import softplus

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, ScreeningConfig, collate
from demand_forecast.data.episode.collate import EpisodeBatchWithRowKeys
from demand_forecast.data.episode.schema import NAIVE_OPTIONS, PATCH, QUANTILES, ROLE_SIBLING
from demand_forecast.models.v5 import V5Config
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
    additions = net.state_dict().keys() - legacy.state_dict().keys()
    assert sum(net.state_dict()[name].numel() for name in additions) == 9
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
    knots = anchor[:, None] + fan + softplus(torch.tensor(-7.0)) * head[0, 0, 0, :16]
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
        torch.testing.assert_close(
            net(no_arrays).quantiles, legacy(no_arrays).quantiles, atol=0, rtol=0
        )
        torch.testing.assert_close(
            net(batch).quantiles[..., 16:, :],
            legacy(batch).quantiles[..., 16:, :],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantiles_stay_sorted_with_random_masks_and_explicit_precision(dtype):
    batch = _batch_to_model(collate([episode(panel(), DERIVE)]), torch.device("cpu"), dtype)
    gen = torch.Generator().manual_seed(3)
    seen = torch.rand(batch.naive_seen.shape, generator=gen) > 0.4
    errors_seen = torch.rand(batch.naive_error_seen.shape, generator=gen) > 0.4
    resq = torch.rand(batch.naive_resq.shape, generator=gen).sort(dim=-1).values
    batch = replace(batch, naive_seen=seen, naive_error_seen=errors_seen, naive_resq=resq)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.float32):
        out = evidence_model().to(dtype)(batch)
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
