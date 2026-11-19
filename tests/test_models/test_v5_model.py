"""Task 8 network, gating, loss, pruning and forecast contracts."""

import dataclasses
from dataclasses import replace

import numpy as np
import pytest
import torch

from demand_forecast.data.episode import EpisodeBuilder, EpisodeRequest, collate
from demand_forecast.data.episode.schema import (
    CH_STEP,
    CH_STEP_FLAG,
    CH_VALUE_STATS,
    KIND_NUMBER,
    N_SUMMARY,
    ROLE_FORECAST,
    ROLE_KNOWN,
    ROLE_PANEL,
    ROLE_PAST,
    ROLE_PEER,
    ROLE_SIBLING,
    STEP_FAMILIES,
)
from demand_forecast.models.losses import pinball_loss
from demand_forecast.models.v5 import V5Config, V5Model, forecast, v5_loss
from demand_forecast.models.v5.blocks import DecoderBlock, EncoderBlock, ExpertFFN
from demand_forecast.models.v5.model import Gates, V5Output, prune
from tests.helpers.episodes import toy_panel

TINY = V5Config(d_model=64, nhead=4, enc_layers=2, dec_layers=2, dim_feedforward=128)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        yield
    torch.set_num_threads(previous)


def toy_batch():
    return collate(
        [EpisodeBuilder(toy_panel()).build(EpisodeRequest(0, (320, 336), 28, context=256))]
    )


def bare_panel(values=None, *, dated=False, n_targets=1):
    panel = toy_panel(n_items=1, dated=dated, n_targets=n_targets)
    item = replace(panel.items[0], dynamic={}, static={})
    if values is not None:
        item = replace(item, targets={"y": np.asarray(values, dtype=np.float64)})
    return replace(panel, items=(item,), roles={})


def test_default_size():
    n = sum(p.numel() for p in V5Model().parameters())
    assert 15_000_000 <= n <= 30_000_000


def test_shapes_and_monotone_quantiles():
    batch = toy_batch()
    out = V5Model(TINY).eval()(batch)
    assert out.quantiles.shape == (1, 2, 1, 32, 9)
    assert (out.quantiles.diff(dim=-1) >= 0).all()


def test_overfits_one_batch():
    cfg = dataclasses.replace(TINY, dropout=0.0)
    torch.manual_seed(0)
    batch = toy_batch()
    model = V5Model(cfg)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    ones = model.gates(batch, mode="ones")

    def eval_loss():
        model.eval()
        with torch.no_grad():
            return v5_loss(model(batch, ones), batch, cfg)[0].item()

    first = eval_loss()
    for _ in range(40):
        model.train()
        loss, _ = v5_loss(model(batch, ones), batch, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = eval_loss()
    assert last < 0.7 * first, (first, last)


def test_loss_ignores_padding():
    small = EpisodeBuilder(bare_panel(n_targets=2)).build(EpisodeRequest(0, (320,), 5, context=32))
    large = EpisodeBuilder(bare_panel(n_targets=3)).build(
        EpisodeRequest(0, (304, 320), 35, context=64)
    )
    large.label_mask[:] = False
    model = V5Model(TINY).eval()
    alone, packed = collate([small]), collate([small, large])
    a = model(alone, model.gates(alone, mode="ones"))
    b = model(packed, model.gates(packed, mode="ones"))
    for output, batch in ((a, alone), (b, packed)):
        eligible = (
            batch.origin_mask[:, :, None]
            & batch.forecast_mask[:, None, :]
            & ~batch.fallback[:, None, :]
        )
        direct, count = pinball_loss(
            output.quantiles.flatten(0, 2),
            batch.labels.flatten(0, 2),
            batch.label_mask.flatten(0, 2),
            eligible.flatten(),
            torch.tensor(TINY.quantiles),
        )
        assert count == 10
        loss, terms = v5_loss(output, batch, TINY)
        assert terms["pinball"] == pytest.approx(direct.item())
        torch.testing.assert_close(loss, direct + TINY.balance_weight * output.aux["balance"])
    assert v5_loss(a, alone, TINY)[1]["pinball"] == pytest.approx(
        v5_loss(b, packed, TINY)[1]["pinball"], abs=1e-6
    )


def test_loss_masks_horizon_and_fallback_and_supervises_siblings():
    batch = toy_batch()
    batch.forecast_mask[:] = True
    batch.keep[:] = False  # siblings are still supervised
    batch.label_mask[:] = True
    batch.labels[:] = 1.0
    batch.horizon[:] = 5
    batch.labels[..., 5:] = float("nan")
    gates = V5Model(TINY).gates(batch, mode="ones")
    predictions = torch.zeros(1, 2, 1, 32, 9, requires_grad=True)
    output = V5Output(predictions, gates, torch.empty(0), {"balance": torch.tensor(0.0)})
    loss, terms = v5_loss(output, batch, TINY)
    assert terms["pinball"] == pytest.approx(1.0)
    loss.backward()
    assert predictions.grad[..., :5, :].abs().sum() > 0
    assert predictions.grad[..., 5:, :].count_nonzero() == 0
    batch.fallback[:] = True
    assert v5_loss(output, batch, TINY)[1]["pinball"] == 0.0


def test_horizon_patches_differ():
    panel = bare_panel()
    batch = collate([EpisodeBuilder(panel).build(EpisodeRequest(0, (320,), 48, context=64))])
    out = V5Model(TINY).eval()(batch).quantiles
    assert not torch.allclose(out[..., :16, :], out[..., 16:32, :], atol=1e-6)
    assert not torch.allclose(out[..., 16:32, :], out[..., 32:48, :], atol=1e-6)


def test_gates_do_not_encode():
    model = V5Model(TINY).eval()
    blocks = [module for module in model.modules() if isinstance(module, EncoderBlock)]
    calls = []
    hook = blocks[0].register_forward_hook(lambda *args: calls.append(True))
    try:
        gates = model.gates(toy_batch())
        assert not calls
        assert gates.row.shape == toy_batch().row_mask.shape
    finally:
        hook.remove()


def test_supplied_gates_are_used():
    model, batch = V5Model(TINY).eval(), toy_batch()
    ones = model.gates(batch, mode="ones")
    row = ones.row.clone()
    row[batch.row_role == ROLE_PAST] = 0
    gates = Gates(row, ones.cand * 0, ones.family * 0, ones.summary_mask, ones.expected_l0)
    out = model(batch, gates)
    assert out.gates is gates
    assert not torch.allclose(out.quantiles, model(batch, ones).quantiles, atol=1e-6)


@pytest.mark.parametrize("training", [False, True])
def test_peer_rows_share_candidate_gates(training):
    batch, model = toy_batch(), V5Model(TINY).train(training)
    gates = model.gates(batch)
    peers = batch.row_mask & (batch.row_role == ROLE_PEER)
    assert peers.any()
    expected = gates.cand.gather(1, batch.row_cand.clamp_min(0))
    torch.testing.assert_close(gates.row[peers], expected[peers], rtol=0, atol=0)


def test_zero_candidate_matches_removed_peers_and_attributes():
    batch, model = toy_batch(), V5Model(TINY).eval()
    learned = model.gates(batch)
    ci = batch.cand_names[0].index("tags")
    cand = learned.cand.clone()
    cand[0, ci] = 0
    row = torch.where(
        batch.row_role == ROLE_PEER,
        cand.gather(1, batch.row_cand.clamp_min(0)),
        learned.row,
    )
    gates = replace(learned, row=row, cand=cand)
    remove_rows = (batch.row_role[0] == ROLE_PEER) & (batch.row_cand[0] == ci)
    remove_attrs = batch.attr_cand[0] == ci
    assert remove_rows.any() and remove_attrs.any()
    assert not (batch.static_num_cand == ci).any()
    rows = (~remove_rows).nonzero().flatten()
    attrs = (~remove_attrs).nonzero().flatten()
    row_map = torch.full_like(batch.row_cand[0], -1)
    row_map[rows] = torch.arange(len(rows))
    without = replace(
        batch,
        **{
            name: getattr(batch, name)[:, rows]
            for name in (
                "x",
                "patch_valid",
                "row_kind",
                "row_role",
                "row_summary",
                "row_summary_flags",
                "row_score",
                "row_cand",
                "row_mask",
            )
        },
        **{
            name: getattr(batch, name)[:, attrs]
            for name in ("attr_stats", "attr_flags", "attr_cand", "attr_mask")
        },
        forecast_rows=row_map[batch.forecast_rows],
        known_rows=row_map[batch.known_rows],
        row_names=[[batch.row_names[0][i] for i in rows.tolist()]],
        attr_names=[[batch.attr_names[0][i] for i in attrs.tolist()]],
    )
    without_gates = replace(gates, row=gates.row[:, rows])
    torch.testing.assert_close(
        model(batch, gates).quantiles,
        model(without, without_gates).quantiles,
        atol=1e-5,
        rtol=0,
    )


def test_sibling_horizon_tokens_are_gated_before_conditioning():
    panel = bare_panel(n_targets=2)
    batch = collate([EpisodeBuilder(panel).build(EpisodeRequest(0, (320,), 5, 32, target="y"))])
    model = V5Model(TINY).eval()
    gates = model.gates(batch, mode="ones")
    gates.row[batch.row_role == ROLE_SIBLING] = 0.5
    inputs = []
    hook = model.scale_shift.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    try:
        model(batch, gates)
    finally:
        hook.remove()
    torch.testing.assert_close(inputs[1][:, :, 1], 0.5 * inputs[1][:, :, 0])


def test_summary_mask_is_carried_once_and_ones_bypasses_it():
    batch, model = toy_batch(), V5Model(TINY).eval()
    from demand_forecast.models.v5.gates import GlobalMask

    global_mask = next(m for m in model.modules() if isinstance(m, GlobalMask))
    global_mask.logits.data.fill_(-10)
    gates = model.gates(batch)
    assert torch.equal(gates.summary_mask, torch.zeros(N_SUMMARY))
    ones = model.gates(batch, mode="ones")
    assert torch.equal(ones.summary_mask, torch.ones(N_SUMMARY))
    assert ones.expected_l0 == 0
    first = model(batch, gates)
    global_mask.logits.data.fill_(10)
    second = model(batch, gates)
    torch.testing.assert_close(first.quantiles, second.quantiles, rtol=0, atol=0)
    changed = replace(
        batch,
        row_summary=batch.row_summary + 100,
        cand_summary=batch.cand_summary + 100,
    )
    torch.testing.assert_close(first.router, model(changed, gates).router, rtol=0, atol=0)


def test_training_summary_mask_is_sampled_once():
    from demand_forecast.models.v5.gates import GlobalMask

    model, batch = V5Model(TINY).train(), toy_batch()
    global_mask = next(m for m in model.modules() if isinstance(m, GlobalMask))
    seen = []
    hook = global_mask.gate.register_forward_hook(lambda module, args, out: seen.append(out))
    try:
        gates = model.gates(batch)
        model(batch, gates)
    finally:
        hook.remove()
    assert len(seen) == 1
    torch.testing.assert_close(gates.summary_mask, seen[0])


def test_expected_l0_counts_only_independent_real_candidates():
    model, batch = V5Model(TINY).eval(), toy_batch()
    gates = model.gates(batch)
    probability = torch.sigmoid(torch.tensor(TINY.gate_init + (2 / 3) * np.log(11)))
    assert gates.expected_l0.item() == pytest.approx(2 * probability.item())
    assert (gates.row[batch.row_role == 0] == 1).all()
    assert (gates.row[batch.row_role == 2] >= TINY.known_gate_floor).all()


def test_prune_remaps():
    panel = toy_panel(n_items=9, n_targets=2)
    episode = EpisodeBuilder(panel).build(EpisodeRequest(0, (320, 336), 28, 128, target="y"))
    batch = collate([episode, EpisodeBuilder(bare_panel()).build(EpisodeRequest(0, (320,), 5, 32))])
    model = V5Model(TINY).eval()
    gates = model.gates(batch, mode="ones")
    gates.row[0, batch.row_role[0] == ROLE_PAST] = 0
    gates.row[0, batch.row_role[0] == ROLE_SIBLING] = 0
    assert "size" in batch.cand_names[0]
    ci = batch.cand_names[0].index("size")
    gates.cand[0, ci] = 0
    gates.row[0, (batch.row_role[0] == ROLE_PEER) & (batch.row_cand[0] == ci)] = 0
    before = model(batch, gates)
    reduced, new_gates = prune(batch, gates)
    assert "temp" not in reduced.row_names[0]
    assert "y1" not in reduced.row_names[0]
    assert "size" not in reduced.cand_names[0]
    assert all(not name.startswith("size=") for name in reduced.attr_names[0])
    assert all(not name.startswith("size=") for name in reduced.row_names[0])
    assert reduced.static_num_mask[0].sum() == 0
    for b in range(2):
        for rows, mask in (
            (reduced.forecast_rows, reduced.forecast_mask),
            (reduced.known_rows, reduced.known_mask),
        ):
            assert (rows[b, mask[b]] < reduced.row_mask[b].sum()).all()
        for mapping, mask in (
            (reduced.attr_cand, reduced.attr_mask),
            (reduced.static_num_cand, reduced.static_num_mask),
        ):
            assert (mapping[b, mask[b]] < reduced.cand_mask[b].sum()).all()
    after = model(reduced, new_gates)
    torch.testing.assert_close(
        before.quantiles[:, :, :1], after.quantiles[:, :, :1], atol=1e-5, rtol=0
    )


@pytest.mark.parametrize(
    "index_name,flag_name",
    [("attr_cand", "attr_flags"), ("static_num_cand", "static_num_flags")],
)
def test_prune_preserves_negative_candidate_indices(index_name, flag_name):
    batch, model = toy_batch(), V5Model(TINY).eval()
    batch = replace(
        batch,
        **{name: getattr(batch, name).clone() for name in (index_name, flag_name)},
    )
    getattr(batch, index_name)[0, 0] = -1
    getattr(batch, flag_name)[0, 0] = False
    gates = model.gates(batch, mode="ones")
    gates.cand[0, 0] = 0
    gates.row[(batch.row_role == ROLE_PEER) & (batch.row_cand == 0)] = 0
    before = model(batch, gates).quantiles
    reduced, selected = prune(batch, gates)
    torch.testing.assert_close(before, model(reduced, selected).quantiles, atol=1e-5, rtol=0)
    assert getattr(reduced, index_name)[0, 0].item() == -1


def test_forecast_units_and_length():
    panel = bare_panel(n_targets=2)
    request = EpisodeRequest(0, (319, 335), 5, context=64, target="y")
    builder, model = EpisodeBuilder(panel), V5Model(TINY).eval()
    batch = collate([builder.build(request)])
    out = model(batch, model.gates(batch, mode="ones"))
    expected = batch.loc[0, :, None, None] + batch.scale[0, :, None, None] * out.quantiles[0]
    result = forecast(model, panel, request, builder=builder, gates="ones", pruned=False)
    assert result.quantiles.shape == (2, 1, 5, 9)
    assert result.names == ["y"]
    np.testing.assert_allclose(result.quantiles, expected.detach().numpy()[:, :1, :5], atol=1e-6)
    assert {f"row:{name}" for name in batch.row_names[0]} <= result.gates.keys()
    assert {f"family:{name}" for name in STEP_FAMILIES} <= result.gates.keys()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("failure", [None, "gates", "forward"])
def test_forecast_restores_model_mode(training, failure, monkeypatch):
    panel, model = bare_panel(), V5Model(TINY).train(training)
    request = EpisodeRequest(0, (320,), 5, context=32)
    if failure is None:
        forecast(model, panel, request)
    else:

        def fail(*args, **kwargs):
            assert not model.training
            raise RuntimeError("forecast failed")

        monkeypatch.setattr(model, failure, fail)
        with pytest.raises(RuntimeError, match="forecast failed"):
            forecast(model, panel, request)
    assert model.training is training
    assert all(module.training is training for module in model.modules())


def test_forecast_reports_colliding_gate_names():
    panel = toy_panel()
    panel = replace(
        panel,
        items=tuple(
            replace(
                item,
                static={
                    "promo" if name == "dept" else name: value
                    for name, value in item.static.items()
                },
            )
            for item in panel.items
        ),
    )
    request = EpisodeRequest(0, (320,), 5, context=32)
    builder, model = EpisodeBuilder(panel), V5Model(TINY).eval()
    batch = collate([builder.build(request)])
    assert "promo" in batch.row_names[0] and "promo" in batch.cand_names[0]
    with torch.no_grad():
        gates = model.gates(batch)
    result = forecast(model, panel, request, builder=builder)
    expected = {f"row:{name}": float(gates.row[0, i]) for i, name in enumerate(batch.row_names[0])}
    expected.update(
        {f"candidate:{name}": float(gates.cand[0, i]) for i, name in enumerate(batch.cand_names[0])}
    )
    expected.update(
        {f"family:{name}": float(gates.family[0, i]) for i, name in enumerate(STEP_FAMILIES)}
    )
    assert result.gates == expected
    assert result.gates["row:promo"] != result.gates["candidate:promo"]


@pytest.mark.parametrize(
    "values,origin,expected",
    [
        (np.full(20, np.nan), 17, 0.0),
        (np.full(20, 7.0), 17, 7.0),
        (np.array([1.0, 3.0, 2.0]), 3, None),
    ],
)
def test_fallback_rows_are_finite(values, origin, expected):
    panel, model = bare_panel(values), V5Model(TINY)
    result = forecast(model, panel, EpisodeRequest(0, (origin,), 5, context=32))
    assert result.quantiles.shape == (1, 1, 5, 9)
    assert np.isfinite(result.quantiles).all()
    if expected is not None:
        assert (result.quantiles == expected).all()


@pytest.mark.parametrize("dated", [True, False])
def test_no_feature_episode_runs(dated):
    panel = bare_panel(dated=dated)
    batch = collate([EpisodeBuilder(panel).build(EpisodeRequest(0, (320,), 5, 32))])
    assert batch.x.shape[1] == 1 and batch.attr_stats.shape[1] == 0
    assert torch.isfinite(V5Model(TINY).eval()(batch).quantiles).all()


def test_overflow_item_decodes_siblings():
    panel = bare_panel(n_targets=20)
    request = EpisodeRequest(0, (320,), 5, 32, target="y")
    batch = collate([EpisodeBuilder(panel).build(request)])
    model = V5Model(TINY).eval()
    assert model(batch).quantiles.shape == (1, 1, 16, 16, 9)
    result = forecast(model, panel, request)
    assert result.quantiles.shape == (1, 1, 5, 9) and result.names == ["y"]


def test_eval_is_deterministic():
    batch, model = toy_batch(), V5Model(TINY).eval()
    torch.testing.assert_close(model(batch).quantiles, model(batch).quantiles, rtol=0, atol=0)


def routing_batch(episodes=3):
    builder = EpisodeBuilder(toy_panel(n_items=9))
    examples = [builder.build(EpisodeRequest(i, (320,), 5, 32)) for i in range(episodes)]
    if episodes > 1:
        examples[-1] = EpisodeBuilder(bare_panel()).build(EpisodeRequest(0, (320,), 5, 32))
    return collate(examples)


def test_uniform_router_uses_all_experts_in_encoder_and_decoder():
    batch = routing_batch()
    model = V5Model(replace(TINY, n_experts=3)).eval()
    learned = model(batch)
    encoder_weights, decoder_weights = [], []
    hooks = [
        block.register_forward_pre_hook(lambda module, args: encoder_weights.append(args[-1]))
        for block in model.encoder
    ] + [
        block.register_forward_pre_hook(lambda module, args: decoder_weights.append(args[-1]))
        for block in model.decoder
    ]
    rng = torch.random.get_rng_state()
    try:
        uniform = model(batch, router="uniform")
    finally:
        for hook in hooks:
            hook.remove()
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert (~batch.row_mask).any()  # Padding also receives uniform routing.
    for weights in [uniform.router, *encoder_weights, *decoder_weights]:
        torch.testing.assert_close(weights, torch.full_like(weights, 1 / 3), rtol=0, atol=0)
    assert not torch.allclose(uniform.quantiles, learned.quantiles, atol=1e-6)
    torch.testing.assert_close(uniform.aux["balance"], learned.aux["balance"], rtol=0, atol=0)


@pytest.mark.parametrize("episodes,shift", [(2, 1), (3, 1), (4, 2)])
def test_rolled_router_is_a_deterministic_batch_roll(episodes, shift):
    batch, model = routing_batch(episodes), V5Model(TINY).eval()
    learned = model(batch)
    rng = torch.random.get_rng_state()
    first = model(batch, router="rolled")
    second = model(batch, router="rolled")
    assert torch.equal(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(first.router, learned.router.roll(shift, 0), rtol=0, atol=0)
    torch.testing.assert_close(first.router, second.router, rtol=0, atol=0)
    torch.testing.assert_close(first.quantiles, second.quantiles, rtol=0, atol=0)
    torch.testing.assert_close(first.aux["balance"], learned.aux["balance"], rtol=0, atol=0)
    assert not torch.equal(first.router, learned.router)
    assert not torch.allclose(first.quantiles, learned.quantiles, atol=1e-6)


def test_rolled_router_rejects_one_episode():
    with pytest.raises(ValueError, match="^the rolled router needs at least two episodes$"):
        V5Model(TINY).eval()(routing_batch(1), router="rolled")


def test_unknown_router_raises():
    with pytest.raises(ValueError, match="router"):
        V5Model(TINY).eval()(routing_batch(1), router="typo")


def test_default_router_matches_explicit_learned_router_bitwise():
    batch, model = routing_batch(), V5Model(TINY).eval()
    # The default router is the learned router.
    default = model(batch)
    explicit = model(batch, router="learned")
    torch.testing.assert_close(explicit.quantiles, default.quantiles, rtol=0, atol=0)
    torch.testing.assert_close(explicit.router, default.router, rtol=0, atol=0)
    for field in dataclasses.fields(Gates):
        assert torch.equal(getattr(explicit.gates, field.name), getattr(default.gates, field.name))
    for name in default.aux:
        assert torch.equal(explicit.aux[name], default.aux[name])


@pytest.mark.parametrize("mode", ["learned", "ones"])
def test_forced_candidates_zero_all_uses_and_match_pruning(mode):
    batch, model = routing_batch(1), V5Model(replace(TINY, gate_init=0.0)).eval()
    baseline = model.gates(batch, mode=mode)
    zero = torch.zeros_like(batch.cand_mask)
    indices = [batch.cand_names[0].index(name) for name in ("tags", "size")]
    zero[0, indices] = True
    gates = model.gates(batch, mode=mode, zero_candidates=zero)
    peers = batch.row_mask & (batch.row_role == ROLE_PEER)
    forced_peers = peers & zero.gather(1, batch.row_cand.clamp_min(0))
    forced_attrs = batch.attr_mask & zero.gather(1, batch.attr_cand.clamp_min(0))
    forced_numbers = batch.static_num_mask & zero.gather(1, batch.static_num_cand.clamp_min(0))
    assert forced_peers.any() and forced_attrs.any() and forced_numbers.any()
    assert (gates.cand[zero] == 0).all()
    assert (gates.row[forced_peers] == 0).all()
    assert torch.equal(gates.cand[~zero], baseline.cand[~zero])
    assert torch.equal(gates.row[~forced_peers], baseline.row[~forced_peers])
    assert torch.equal(gates.expected_l0, baseline.expected_l0)
    assert torch.equal(gates.family, baseline.family)
    assert torch.equal(gates.summary_mask, baseline.summary_mask)
    context_inputs = []
    hook = model.static_context.register_forward_pre_hook(
        lambda module, args: context_inputs.append(args)
    )
    try:
        before = model(batch, gates)
    finally:
        hook.remove()
    assert (context_inputs[0][1][forced_attrs] == 0).all()
    assert (context_inputs[0][4][forced_numbers] == 0).all()
    reduced, reduced_gates = prune(batch, gates)
    assert all(name not in reduced.cand_names[0] for name in ("tags", "size"))
    assert not any(name.startswith("tags=") for name in reduced.row_names[0])
    assert not any(name.startswith(("tags=", "size=")) for name in reduced.attr_names[0])
    assert reduced.static_num_mask.sum() == 0
    after = model(reduced, reduced_gates)
    kept = (batch.keep & batch.forecast_mask)[:, None, :, None, None].expand_as(before.quantiles)
    torch.testing.assert_close(before.quantiles[kept], after.quantiles[kept], rtol=0, atol=1e-5)


@pytest.mark.parametrize("mode", ["learned", "ones"])
@pytest.mark.parametrize("role", [ROLE_PAST, ROLE_PEER, ROLE_KNOWN])
def test_forced_rows_match_pruning_without_changing_candidate_gates(mode, role):
    batch, model = routing_batch(1), V5Model(replace(TINY, gate_init=0.0)).eval()
    baseline = model.gates(batch, mode=mode)
    zero = batch.row_mask & (batch.row_role == role)
    assert zero.any() and (baseline.row[zero] > 0).all()
    gates = model.gates(batch, mode=mode, zero_rows=zero)
    assert (gates.row[zero] == 0).all()
    assert torch.equal(gates.row[~zero], baseline.row[~zero])
    assert torch.equal(gates.cand, baseline.cand)
    assert torch.equal(gates.expected_l0, baseline.expected_l0)
    before = model(batch, gates)
    reduced, reduced_gates = prune(batch, gates)
    after = model(reduced, reduced_gates)
    kept = batch.keep[0] & batch.forecast_mask[0]
    reduced_kept = reduced.keep[0] & reduced.forecast_mask[0]
    torch.testing.assert_close(
        before.quantiles[:, :, kept], after.quantiles[:, :, reduced_kept], rtol=0, atol=1e-5
    )


@pytest.mark.parametrize("mode", ["learned", "ones"])
def test_forced_siblings_keep_forecast_slots_until_pruning(mode):
    panel = bare_panel(n_targets=20)
    request = EpisodeRequest(0, (320,), 5, 32, target="y")
    batch = collate([EpisodeBuilder(panel).build(request)])
    model = V5Model(replace(TINY, gate_init=0.0)).eval()
    baseline = model.gates(batch, mode=mode)
    zero = batch.row_mask & (batch.row_role == ROLE_SIBLING)
    sibling_slots = batch.forecast_mask & (
        batch.row_role.gather(1, batch.forecast_rows) == ROLE_SIBLING
    )
    kept = batch.keep[0] & batch.forecast_mask[0]
    assert sibling_slots.sum() == zero.sum() > 0
    assert kept.sum() == 1 and not batch.keep[sibling_slots].any()
    assert (baseline.row[zero] > 0).all()
    gates = model.gates(batch, mode=mode, zero_rows=zero)
    assert (gates.row[zero] == 0).all()
    before = model(batch, gates)
    natural_zero = replace(baseline, row=baseline.row.masked_fill(zero, 0))
    torch.testing.assert_close(
        before.quantiles, model(batch, natural_zero).quantiles, rtol=0, atol=0
    )
    assert before.quantiles.shape[2] == batch.forecast_rows.shape[1]
    reduced, reduced_gates = prune(batch, gates)
    assert reduced.forecast_mask.sum() == 1
    assert not (reduced.row_mask & (reduced.row_role == ROLE_SIBLING)).any()
    after = model(reduced, reduced_gates)
    reduced_kept = reduced.keep[0] & reduced.forecast_mask[0]
    torch.testing.assert_close(
        before.quantiles[:, :, kept], after.quantiles[:, :, reduced_kept], rtol=0, atol=1e-5
    )
    zero_kept = batch.row_mask & (batch.row_role == ROLE_FORECAST)
    assert zero_kept.sum() == kept.sum()
    with pytest.raises(ValueError, match="forecast"):
        model.gates(batch, mode=mode, zero_rows=zero_kept)


@pytest.mark.parametrize("mode", ["learned", "ones"])
def test_forcing_forecast_rows_raises(mode):
    batch = routing_batch(1)
    zero = batch.row_mask & (batch.row_role == ROLE_FORECAST)
    assert zero.any()
    with pytest.raises(ValueError, match="forecast"):
        V5Model(TINY).gates(batch, mode=mode, zero_rows=zero)


@pytest.mark.parametrize("argument", ["zero_candidates", "zero_rows"])
@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_forcing_masks_require_boolean_batch_shapes(argument, invalid):
    batch = routing_batch(1)
    mask = batch.cand_mask if argument == "zero_candidates" else batch.row_mask
    mask = torch.zeros_like(mask)
    mask = mask[:, :-1] if invalid == "shape" else mask.float()
    with pytest.raises(ValueError, match=argument):
        V5Model(TINY).gates(batch, **{argument: mask})


def test_padded_horizon_inputs_are_inert():
    batch = collate([EpisodeBuilder(toy_panel()).build(EpisodeRequest(0, (319,), 5, 32))])
    changed = replace(
        batch,
        xh=batch.xh.clone(),
        calh_phases=batch.calh_phases.clone(),
        calh_ids=batch.calh_ids.clone(),
    )
    changed.xh[..., 5:, :] = 1e4
    changed.calh_phases[..., 5:, :] = 1e4
    changed.calh_ids[..., 5:, :] = 0
    model = V5Model(TINY).eval()
    gates = model.gates(batch, mode="ones")
    torch.testing.assert_close(
        model(batch, gates).quantiles[..., :5, :],
        model(changed, gates).quantiles[..., :5, :],
        rtol=0,
        atol=0,
    )


def test_encoder_is_causal_and_masks_zero_rows():
    block = EncoderBlock(replace(TINY, dropout=0.0)).eval()
    x = torch.randn(1, 3, 4, 64)
    valid, rows, gates = (
        torch.ones(1, 3, 4, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[1.0, 0.5, 0.0]]),
    )
    before = block(x, valid, rows, gates, None)
    changed = x.clone()
    changed[:, :, 2:] += 100
    changed[:, 2] = float("nan")
    after = block(changed, valid, rows, gates, None)
    torch.testing.assert_close(before[:, :2, :2], after[:, :2, :2], rtol=0, atol=0)
    assert torch.isfinite(after).all()


def test_decoder_memory_is_cut_off_per_origin():
    block = DecoderBlock(replace(TINY, dropout=0.0)).eval()
    h, memory = torch.randn(1, 2, 2, 2, 64), torch.randn(1, 3, 4, 64)
    args = (
        torch.tensor([[0, 2]]),
        torch.tensor([[2, 4]]),
        torch.ones(1, 3, 4, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2),
        None,
    )
    first = block(h, memory, *args)
    changed = memory.clone()
    changed[:, :, 2:] += 100 * torch.randn_like(changed[:, :, 2:])
    second = block(h, changed, *args)
    torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
    assert not torch.allclose(first[:, 1], second[:, 1])


def test_expert_ffn_mixes_hidden_states_per_row():
    module = ExpertFFN(replace(TINY, dropout=0.0)).eval()
    x = torch.randn(2, 3, 4, 64)
    a = torch.zeros(2, 3, 4)
    b = a.clone()
    a[..., 0], b[..., 1] = 1, 1
    ya, yb = module(x, a), module(x, b)
    torch.testing.assert_close(module(x, 0.25 * a + 0.75 * b), 0.25 * ya + 0.75 * yb)
    assert not torch.allclose(ya, yb)


def test_family_gates_mask_values_and_flags():
    batch, model = toy_batch(), V5Model(TINY).eval()
    gates = model.gates(batch, mode="ones")
    gates.family[:] = 0
    changed = replace(batch, x=batch.x.clone())
    numeric = (batch.row_kind == KIND_NUMBER)[..., None, None, None]
    step_values = changed.x[..., CH_STEP:CH_STEP_FLAG]
    step_flags = changed.x[..., CH_STEP_FLAG:CH_VALUE_STATS]
    step_values.copy_(torch.where(numeric, 200, step_values))
    step_flags.copy_(torch.where(numeric, 1, step_flags))
    channels = torch.arange(batch.x.shape[-1], device=batch.x.device)
    allowed = numeric & (channels >= CH_STEP) & (channels < CH_VALUE_STATS)
    different = changed.x != batch.x
    assert different.any() and not (different & ~allowed).any()
    torch.testing.assert_close(
        model(batch, gates).quantiles, model(changed, gates).quantiles, rtol=0, atol=0
    )


def test_gate_mode_is_validated():
    with pytest.raises(ValueError):
        V5Model(TINY).gates(toy_batch(), mode="typo")


def test_learned_gates_train_with_finite_gradients():
    model, batch = V5Model(replace(TINY, dropout=0.0)).train(), toy_batch()
    loss, _ = v5_loss(model(batch), batch, model.config)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.global_mask.logits.grad.abs().sum() > 0
    assert model.family_logits.bias.grad.abs().sum() > 0
    assert model.gate_net.net[-1].bias.grad.abs().sum() > 0


def test_empty_pruning_and_all_attributes_removed():
    batch, model = toy_batch(), V5Model(TINY).eval()
    gates = model.gates(batch, mode="ones")
    gates.cand[:] = 0
    gates.row[
        (batch.row_role == ROLE_PEER)
        | (batch.row_role == ROLE_PAST)
        | (batch.row_role == ROLE_PANEL)
    ] = 0
    reduced, selected = prune(batch, gates)
    assert reduced.attr_stats.shape[1] == 0
    assert reduced.cand_summary.shape[1] == 0
    assert reduced.static_num.shape[1] == 0
    torch.testing.assert_close(
        model(batch, gates).quantiles, model(reduced, selected).quantiles, atol=1e-5, rtol=0
    )
