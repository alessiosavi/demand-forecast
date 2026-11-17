"""Contracts for the dataset-independent building blocks of v5."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from demand_forecast.data.episode.schema import (
    CALENDAR_SIZES,
    CH_OBSERVED,
    CH_VALUE_STATS,
    KIND_CATEGORY,
    KIND_FLAG,
    KIND_NUMBER,
    N_CHANNELS,
    N_PHASES,
    N_SUMMARY,
    N_VALUE_TOKEN,
    PATCH,
    ROLE_FORECAST,
    ROLE_KNOWN,
    ROLE_PANEL,
    ROLE_PAST,
    ROLE_PEER,
    ROLE_SIBLING,
)
from demand_forecast.models.v5.attention import MaskedAttention
from demand_forecast.models.v5.config import V5Config
from demand_forecast.models.v5.context import ExpertRouter, ScaleShift, StaticContext
from demand_forecast.models.v5.gates import GateNet, GlobalMask, HardConcreteGate
from demand_forecast.models.v5.tokens import CalendarEmbedding, PatchEncoder

TINY = V5Config(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0)


def test_scale_shift_is_identity_at_init():
    t = torch.randn(2, 3, 5, 384)
    assert torch.equal(ScaleShift(V5Config())(t, torch.randn(2, 384)), t)


def test_bias_of_minus_infinity_removes_a_key():
    a = MaskedAttention(64, 4, 0.0, rope=False).eval()
    q, k = torch.randn(1, 3, 64), torch.randn(1, 5, 64)
    bias = torch.zeros(1, 5)
    bias[0, 2] = -torch.inf
    keep = [0, 1, 3, 4]
    assert torch.allclose(a(q, k, k, key_bias=bias), a(q, k[:, keep], k[:, keep]), atol=1e-6)


@pytest.mark.parametrize("blocked_by", ["mask", "bias"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("contaminated", ["key", "value", "both"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train_dropout"])
def test_dead_nonfinite_key_is_equivalent_to_removal(blocked_by, nonfinite, contaminated, training):
    a = MaskedAttention(64, 4, 0.2, rope=False).train(training)
    q = torch.randn(2, 3, 64, requires_grad=True)
    k, v = torch.randn(2, 5, 64), torch.randn(2, 5, 64)
    dead_indices = torch.tensor([2, 4])
    batch_indices = torch.arange(2)
    if contaminated in ("key", "both"):
        k[batch_indices, dead_indices] = nonfinite
    if contaminated in ("value", "both"):
        v[batch_indices, dead_indices] = nonfinite
    k.requires_grad_()
    v.requires_grad_()
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    bias = torch.zeros(2, 5)
    if blocked_by == "mask":
        mask[batch_indices, :, dead_indices] = False
    else:
        bias[batch_indices, dead_indices] = -torch.inf

    sampled_dropout = []
    with a.dropout.register_forward_hook(
        lambda module, inputs, output: sampled_dropout.append(output.detach())
    ):
        out = a(q, k, v, mask=mask, key_bias=bias)
    assert torch.isfinite(out).all()
    for batch, dead in enumerate(dead_indices):
        keep = torch.arange(5) != dead
        # Compare removal under the same dropout draw on the surviving keys.
        scale = (sampled_dropout[0][batch : batch + 1, ..., keep] != 0).to(q.dtype)
        if training:
            scale = scale / (1 - a.dropout.p)
        with (
            torch.no_grad(),
            a.dropout.register_forward_hook(lambda module, inputs, output: inputs[0] * scale),
        ):
            without = a(
                q[batch : batch + 1],
                k[batch : batch + 1, keep],
                v[batch : batch + 1, keep],
            )
        torch.testing.assert_close(out[batch : batch + 1], without)

    out.square().sum().backward()
    assert torch.isfinite(q.grad).all()
    for tensor in (k, v):
        assert torch.isfinite(tensor.grad).all()
        assert (tensor.grad[batch_indices, dead_indices] == 0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())


def test_attention_does_not_hide_nan_in_a_live_key():
    a = MaskedAttention(64, 4, 0.0, rope=False).eval()
    q, k, v = (torch.randn(1, length, 64) for length in (2, 3, 3))
    k[:, 0] = torch.nan
    mask = torch.tensor([[[True, True, False], [False, True, False]]])
    out = a(q, k, v, mask=mask)
    assert torch.isnan(out[:, 0]).any()


def test_gate_classes():
    g = GateNet(V5Config()).eval()
    s = torch.randn(4, 28)
    f = torch.ones(4, 28, dtype=torch.bool)
    role = torch.tensor([ROLE_FORECAST, ROLE_KNOWN, ROLE_PAST, ROLE_PEER])
    kind = torch.zeros(4, dtype=torch.long)
    gate, l0 = g(s, f, torch.rand(4), kind, role, s, f)
    assert gate[0] == 1.0 and gate[1] >= 0.05
    assert ((gate[2:] >= 0) & (gate[2:] <= 1)).all()
    assert l0[0] == 0 and l0[1] == 0


def test_gate_net_starts_open():
    g = GateNet(TINY).eval()
    summary, reference = torch.randn(2, 6, N_SUMMARY), torch.randn(2, N_SUMMARY)
    flags = torch.rand_like(summary) > 0.5
    ref_flags = torch.rand_like(reference) > 0.5
    role = torch.tensor(
        [ROLE_SIBLING, ROLE_PAST, ROLE_PEER, ROLE_PANEL, ROLE_FORECAST, ROLE_KNOWN]
    ).expand(2, 6)
    kind = torch.randint(KIND_CATEGORY + 1, (2, 6))
    gates, _ = g(summary, flags, torch.rand(2, 6), kind, role, reference, ref_flags)
    assert torch.equal(gates[:, :5], torch.ones(2, 5))
    expected_known = TINY.known_gate_floor + (1 - TINY.known_gate_floor) * torch.sigmoid(
        torch.tensor(3.0)
    )
    torch.testing.assert_close(gates[:, 5], expected_known.expand(2), rtol=0, atol=0)


def test_global_mask_starts_open():
    summary = torch.randn(2, 3, N_SUMMARY)
    flags = torch.rand_like(summary) > 0.5
    masked, masked_flags = GlobalMask(TINY).eval()(summary, flags)
    assert torch.equal(masked, summary)
    assert torch.equal(masked_flags, flags)


def test_gate_initial_logits_are_configurable():
    cfg = replace(TINY, gate_init=0.0)
    summary = torch.randn(2, N_SUMMARY)
    flags = torch.ones_like(summary, dtype=torch.bool)
    gates, _ = GateNet(cfg).eval()(
        summary,
        flags,
        torch.rand(2),
        torch.tensor([KIND_NUMBER, KIND_CATEGORY]),
        torch.tensor([ROLE_PAST, ROLE_PEER]),
        summary,
        flags,
    )
    torch.testing.assert_close(gates, torch.full((2,), 0.5), rtol=0, atol=0)
    masked, masked_flags = GlobalMask(cfg).eval()(summary, flags)
    assert torch.equal(masked, summary * 0.5)
    assert torch.equal(masked_flags, flags)


def test_hard_concrete_reaches_zero_and_one():
    g = HardConcreteGate().eval()
    logits = torch.tensor([-10.0, 0.0, 10.0])
    assert torch.equal(g(logits), torch.tensor([0.0, 0.5, 1.0]))
    probability = g.expected_l0(logits)
    assert ((probability > 0) & (probability < 1)).all()
    # P(stretched gate > 0) at logit zero, temperature 2/3 and stretch [-.1, 1.1].
    assert probability[1].item() == pytest.approx(0.8318222)
    assert torch.equal(g(logits), g(logits))


def test_hard_concrete_training_samples_and_backpropagates():
    g = HardConcreteGate().train()
    logits = torch.zeros(512, requires_grad=True)
    first, second = g(logits), g(logits)
    assert ((first >= 0) & (first <= 1)).all()
    assert (first == 0).any() and (first == 1).any()
    assert not torch.equal(first, second)
    (first.mean() + g.expected_l0(logits).mean()).backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad != 0).any()


@pytest.mark.parametrize("blocked_by", ["mask", "bias", "both", "empty"])
def test_masked_query_returns_zeros(blocked_by):
    a = MaskedAttention(64, 4, 0.0, rope=False).eval()
    q = torch.randn(2, 3, 64, requires_grad=True)
    k = torch.randn(2, 5 if blocked_by != "empty" else 0, 64, requires_grad=True)
    mask = torch.ones(2, 3, k.shape[1], dtype=torch.bool)
    bias = torch.zeros(2, k.shape[1])
    if blocked_by in ("mask", "both"):
        mask[:, 1] = False
    if blocked_by in ("bias", "both"):
        bias[0] = -torch.inf
    out = a(q, k, k, mask=mask, key_bias=bias)
    if blocked_by in ("bias", "both", "empty"):
        assert torch.equal(out[0], torch.zeros_like(out[0]))
    if blocked_by in ("mask", "both", "empty"):
        assert torch.equal(out[:, 1], torch.zeros_like(out[:, 1]))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())


def test_causal_mask():
    a = MaskedAttention(64, 4, 0.0, rope=False).eval()
    q, k, v = (torch.randn(2, 5, 64) for _ in range(3))
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    original = a(q, k, v, mask=mask)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, 3] += 100
    changed_v[:, 3] -= 100
    changed = a(q, changed_k, changed_v, mask=mask)
    torch.testing.assert_close(original[:, :3], changed[:, :3], rtol=0, atol=0)
    assert not torch.allclose(original[:, 3:], changed[:, 3:])


def test_rope_positions_matter():
    a = MaskedAttention(64, 4, 0.0, rope=True).eval()
    without_rope = MaskedAttention(64, 4, 0.0, rope=False).eval()
    # Identity projections make the positional effect independent of random weights.
    with torch.no_grad():
        for attention in (a, without_rope):
            for layer in attention.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.copy_(torch.eye(64))
                    layer.bias.zero_()
    q = torch.ones(1, 1, 64)
    k = torch.ones(1, 2, 64)
    v = torch.stack((torch.zeros(64), torch.ones(64))).unsqueeze(0)
    query_positions, key_positions = torch.tensor([0]), torch.tensor([0, 3])
    first = a(q, k, v, query_positions=query_positions, key_positions=key_positions)
    swapped = a(q, k, v, query_positions=query_positions, key_positions=key_positions.flip(0))
    assert not torch.allclose(first, swapped)
    without = without_rope(q, k, v)
    torch.testing.assert_close(without, torch.full_like(without, 0.5))
    torch.testing.assert_close(without, without_rope(q, k.flip(1), v.flip(1)))
    shifted = a(q, k, v, query_positions=query_positions + 10, key_positions=key_positions + 10)
    torch.testing.assert_close(first, shifted)


@pytest.mark.parametrize("has_query,has_key", [(False, False), (False, True), (True, False)])
def test_rope_requires_both_position_arguments(has_query, has_key):
    a = MaskedAttention(64, 4, 0.0, rope=True).eval()
    q, k = torch.randn(1, 3, 64), torch.randn(1, 5, 64)
    with pytest.raises(ValueError, match="query_positions.*key_positions"):
        a(
            q,
            k,
            k,
            query_positions=torch.arange(3) if has_query else None,
            key_positions=torch.arange(5) if has_key else None,
        )


def test_attention_supports_episode_batch_axes():
    a = MaskedAttention(64, 4, 0.0, rope=True).eval()
    q, k, v = (torch.randn(2, 3, length, 64) for length in (4, 5, 5))
    qp = torch.arange(4).expand(2, 3, 4)
    kp = torch.arange(5).expand(2, 3, 5)
    mask = torch.rand(2, 3, 4, 5) > 0.3
    bias = torch.randn(2, 3, 5)
    together = a(q, k, v, query_positions=qp, key_positions=kp, mask=mask, key_bias=bias)
    separately = torch.stack(
        [
            a(
                q[b],
                k[b],
                v[b],
                query_positions=qp[b],
                key_positions=kp[b],
                mask=mask[b],
                key_bias=bias[b],
            )
            for b in range(2)
        ]
    )
    assert together.shape == q.shape
    torch.testing.assert_close(together, separately)


def test_gate_is_per_candidate():
    g = GateNet(TINY).eval()
    nn.init.normal_(g.net[-1].weight, std=0.1)
    nn.init.zeros_(g.net[-1].bias)
    s, ref = torch.randn(4, N_SUMMARY), torch.randn(4, N_SUMMARY)
    f = torch.ones_like(s, dtype=torch.bool)
    score = torch.rand(4)
    kind = torch.tensor([KIND_NUMBER, KIND_CATEGORY, KIND_FLAG, KIND_NUMBER])
    role = torch.tensor([ROLE_PAST, ROLE_PEER, ROLE_SIBLING, ROLE_PANEL])
    gate, l0 = g(s, f, score, kind, role, ref, f)
    permutation = torch.tensor([2, 0, 3, 1])
    perm_gate, perm_l0 = g(*(t[permutation] for t in (s, f, score, kind, role, ref, f)))
    torch.testing.assert_close(perm_gate, gate[permutation])
    torch.testing.assert_close(perm_l0, l0[permutation])
    changed_s = s.clone()
    changed_s[1] += 3
    changed_gate, changed_l0 = g(changed_s, f, score, kind, role, ref, f)
    torch.testing.assert_close(changed_gate[[0, 2, 3]], gate[[0, 2, 3]], rtol=0, atol=0)
    torch.testing.assert_close(changed_l0[[0, 2, 3]], l0[[0, 2, 3]], rtol=0, atol=0)
    assert not torch.allclose(changed_gate[1], gate[1])
    kept = torch.tensor([0, 2, 3])
    removed_gate, _ = g(*(t[kept] for t in (s, f, score, kind, role, ref, f)))
    torch.testing.assert_close(removed_gate, gate[kept])


def test_known_gate_is_floored_and_deterministic_during_training():
    cfg = replace(TINY, known_gate_floor=0.2)
    g = GateNet(cfg).train()
    with torch.no_grad():
        for layer in g.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.fill_(-100)
    s = torch.zeros(2, N_SUMMARY)
    f = torch.zeros_like(s, dtype=torch.bool)
    role, kind = torch.tensor([ROLE_FORECAST, ROLE_KNOWN]), torch.zeros(2, dtype=torch.long)
    first, l0 = g(s, f, torch.zeros(2), kind, role, s, f)
    second, _ = g(s, f, torch.zeros(2), kind, role, s, f)
    torch.testing.assert_close(first, torch.tensor([1.0, 0.2]), rtol=0, atol=0)
    assert torch.equal(first, second) and torch.equal(l0, torch.zeros(2))


def test_static_context_pooling():
    ctx = StaticContext(TINY).eval()
    attrs, numbers = torch.randn(2, 3, N_VALUE_TOKEN), torch.randn(2, 4)
    attr_weight = torch.tensor([[0.4, 0.0, 0.8], [0.2, 0.0, 0.3]])
    num_weight = torch.tensor([[0.3, 0.7, 0.0, 0.4], [0.8, 0.6, 0.0, 0.1]])
    flags = torch.ones_like(numbers, dtype=torch.bool)
    ref = torch.randn(2, N_SUMMARY)
    ref_flags = torch.ones_like(ref, dtype=torch.bool)
    freq = torch.tensor([0, 3])

    def call(a, aw, n, nf, nw):
        return ctx(a, aw, n, nf, nw, ref, ref_flags, freq)

    original = call(attrs, attr_weight, numbers, flags, num_weight)
    removed = call(
        attrs[:, [0, 2]],
        attr_weight[:, [0, 2]],
        numbers[:, [0, 1, 3]],
        flags[:, [0, 1, 3]],
        num_weight[:, [0, 1, 3]],
    )
    torch.testing.assert_close(original, removed)
    ap, np = [2, 0, 1], [3, 1, 0, 2]
    permuted = call(
        attrs[:, ap], attr_weight[:, ap], numbers[:, np], flags[:, np], num_weight[:, np]
    )
    torch.testing.assert_close(original, permuted)
    padded = call(
        torch.cat((attrs, torch.randn(2, 2, N_VALUE_TOKEN)), dim=1),
        torch.cat((attr_weight, torch.zeros(2, 2)), dim=1),
        torch.cat((numbers, torch.randn(2, 2)), dim=1),
        torch.cat((flags, torch.ones(2, 2, dtype=torch.bool)), dim=1),
        torch.cat((num_weight, torch.zeros(2, 2)), dim=1),
    )
    torch.testing.assert_close(original, padded)
    torch.testing.assert_close(
        original, call(attrs, 7 * attr_weight, numbers, flags, 7 * num_weight)
    )
    zeros = call(attrs, torch.zeros_like(attr_weight), numbers, flags, torch.zeros_like(num_weight))
    empty = call(attrs[:, :0], attr_weight[:, :0], numbers[:, :0], flags[:, :0], num_weight[:, :0])
    torch.testing.assert_close(zeros, empty)
    assert original.shape == (2, TINY.d_model) and torch.isfinite(empty).all()
    changed = call(attrs + 3, attr_weight, numbers + 3, flags, num_weight)
    assert not torch.allclose(original, changed)


def test_global_mask():
    g = GlobalMask(TINY).eval()
    with torch.no_grad():
        g.logits.fill_(10)
        g.logits[2] = -10
    summary = torch.randn(2, 3, N_SUMMARY)
    flags = torch.ones_like(summary, dtype=torch.bool)
    flags[..., 5] = False
    masked, masked_flags = g(summary, flags)
    assert torch.equal(masked[..., 2], torch.zeros(2, 3))
    assert not masked_flags[..., 2].any()
    assert not masked_flags[..., 5].any()
    keep = torch.arange(N_SUMMARY) != 2
    assert torch.equal(masked[..., keep], summary[..., keep])
    assert torch.equal(masked_flags[..., keep], flags[..., keep])
    l0 = g.expected_l0()
    assert l0.shape == (N_SUMMARY,) and ((l0 > 0) & (l0 < 1)).all()


def test_global_mask_training_shares_one_sample_across_summaries():
    g = GlobalMask(TINY).train()
    summary = torch.ones(2, 3, N_SUMMARY)
    flags = torch.ones_like(summary, dtype=torch.bool)
    masked, masked_flags = g(summary, flags)
    torch.testing.assert_close(masked, masked[0, 0].expand_as(masked), rtol=0, atol=0)
    assert torch.equal(masked_flags, flags & (masked > 0))
    (masked.sum() + g.expected_l0().sum()).backward()
    assert torch.isfinite(g.logits.grad).all() and (g.logits.grad != 0).any()


def test_router():
    router = ExpertRouter(TINY).eval()
    context = torch.randn(2, TINY.d_model)
    summary = torch.randn(2, 3, N_SUMMARY)
    flags = torch.ones_like(summary, dtype=torch.bool)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    weights, balance = router(context, summary, flags, mask)
    assert weights.shape == (2, 3, TINY.n_experts)
    assert ((weights != 0).sum(dim=-1) == TINY.top_k).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))
    logits = router.net(
        torch.cat(
            (context[:, None].expand(-1, summary.shape[1], -1), summary, flags.to(summary.dtype)),
            dim=-1,
        )
    )
    top_logits, top_indices = logits.topk(TINY.top_k, dim=-1)
    selected = torch.zeros_like(weights, dtype=torch.bool).scatter(-1, top_indices, True)
    assert torch.equal(weights != 0, selected)
    torch.testing.assert_close(weights.gather(-1, top_indices), top_logits.softmax(dim=-1))
    mean = weights[mask].mean(0)
    positive = mean > 0
    expected_balance = (mean[positive] * (mean[positive] * TINY.n_experts).log()).sum()
    torch.testing.assert_close(balance, expected_balance)
    padded_weights, padded_balance = router(
        context,
        torch.cat((summary, torch.randn(2, 2, N_SUMMARY)), dim=1),
        torch.cat((flags, torch.zeros(2, 2, N_SUMMARY, dtype=torch.bool)), dim=1),
        torch.cat((mask, torch.zeros(2, 2, dtype=torch.bool)), dim=1),
    )
    torch.testing.assert_close(weights, padded_weights[:, :3])
    torch.testing.assert_close(balance, padded_balance)
    _, empty_balance = router(context, summary, flags, torch.zeros_like(mask))
    assert empty_balance == 0 and torch.isfinite(empty_balance)
    (weights.square().sum() + balance + empty_balance).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters())


def test_calendar_embedding_ignores_inactive_ids():
    embedding = CalendarEmbedding(TINY).eval()
    phases = torch.randn(2, 3, PATCH, N_PHASES)
    ids = torch.stack([torch.randint(size, (2, 3, PATCH)) for size in CALENDAR_SIZES], dim=-1)
    for field, table in enumerate(embedding.tables):
        inactive = ids.clone()
        inactive[..., field] = -1
        original = embedding(phases, inactive)
        saved = table.weight.detach().clone()
        with torch.no_grad():
            table.weight.zero_()
        torch.testing.assert_close(original, embedding(phases, inactive), rtol=0, atol=0)
        with torch.no_grad():
            table.weight.copy_(saved)
    assert original.shape == (2, 3, PATCH, TINY.d_calendar)
    absent_ids = torch.full_like(ids, -1)
    absent = embedding(phases, absent_ids)
    assert not torch.allclose(absent, embedding(phases, ids))
    assert not torch.allclose(absent, embedding(phases + 1, absent_ids))


@pytest.mark.parametrize("number_kind", [KIND_NUMBER, KIND_FLAG])
def test_encoders_read_their_own_channels(number_kind):
    encoder = PatchEncoder(TINY).eval()
    x = torch.randn(2, 3, PATCH, N_CHANNELS)
    cal = torch.randn(2, 3, PATCH, TINY.d_calendar)
    kind = torch.tensor([KIND_CATEGORY, number_kind, KIND_CATEGORY]).expand(2, 3)
    role = torch.tensor([ROLE_KNOWN, ROLE_PAST, ROLE_PEER]).expand(2, 3)
    original = encoder(x, cal, kind, role)
    assert original.shape == (2, 3, TINY.d_model)
    changed_stats = x.clone()
    changed_stats[..., CH_VALUE_STATS:] += 3
    stats_tokens = encoder(changed_stats, cal, kind, role)
    torch.testing.assert_close(stats_tokens[:, 1], original[:, 1], rtol=0, atol=0)
    assert not torch.allclose(stats_tokens[:, [0, 2]], original[:, [0, 2]])
    changed_numbers = x.clone()
    changed_numbers[..., :CH_OBSERVED] += 3
    changed_numbers[..., CH_OBSERVED + 1 : CH_VALUE_STATS] += 3
    number_tokens = encoder(changed_numbers, cal, kind, role)
    torch.testing.assert_close(number_tokens[:, [0, 2]], original[:, [0, 2]], rtol=0, atol=0)
    assert not torch.allclose(number_tokens[:, 1], original[:, 1])
    changed_observed = x.clone()
    changed_observed[..., CH_OBSERVED] += 3
    assert not torch.allclose(original, encoder(changed_observed, cal, kind, role))
    assert not torch.allclose(original, encoder(x, cal + 3, kind, role))
    assert not torch.allclose(original, encoder(x, cal, kind, (role + 1) % 6))
