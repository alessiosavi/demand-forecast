"""Causal naive paths and their use as a per-cell quantile anchor."""

import sys
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
import torch

from demand_forecast.data.episode import (
    CH_OBSERVED,
    CH_VALUE,
    EpisodeBatch,
    EpisodeBuilder,
    EpisodeRequest,
    Item,
    Panel,
    ScreeningConfig,
    collate,
)
from demand_forecast.data.episode.collate import (
    EpisodeBatchWithNaive,
    EpisodeBatchWithNaiveAndRowKeys,
    EpisodeBatchWithRowKeys,
)
from demand_forecast.data.episode.derived import naive_series
from demand_forecast.data.episode.schema import ROLE_SIBLING
from demand_forecast.models.v5 import V5Model, forecast_items, planned_layouts, usable_workers
from demand_forecast.models.v5.model import prune
from tests.helpers.v5_loop import config

ANCHOR = frozenset({"naive_anchor"})
BOTH = ANCHOR | {"naive_rows"}
TINY = replace(config().model, d_out=32, horizon_seed=True)
OPTIONS = ("none", "last", "level", "season", "season2", "drift")
N6_BATCH_FIELDS = (
    "x",
    "patch_valid",
    "row_kind",
    "row_role",
    "row_names",
    "row_summary",
    "row_summary_flags",
    "row_score",
    "row_cand",
    "forecast_rows",
    "keep",
    "labels",
    "label_mask",
    "loc",
    "scale",
    "fallback",
    "known_rows",
    "xh",
    "cal_phases",
    "cal_ids",
    "calh_phases",
    "calh_ids",
    "cand_names",
    "cand_kind",
    "cand_summary",
    "cand_flags",
    "cand_score",
    "attr_stats",
    "attr_flags",
    "attr_cand",
    "attr_names",
    "static_num",
    "static_num_flags",
    "static_num_cand",
    "freq_class",
    "horizon",
    "origin_patches",
    "context_target",
    "row_mask",
    "forecast_mask",
    "known_mask",
    "cand_mask",
    "attr_mask",
    "static_num_mask",
    "origin_mask",
    "horizon_mask",
)


def panel(*, freq="h", sibling=False, collision=False, size=256):
    time = np.arange(size, dtype=float)
    target = 20 + 6 * np.sin(time * np.pi / 12) + time / 40
    targets = {"y": target}
    if sibling:
        targets["y~season"] = target * 2 + 30
    dynamic = {"y~last": target * 3 - 43} if collision else {}
    return Panel(
        "paths",
        (Item("item", 13, targets, dynamic, {}),),
        dict.fromkeys(dynamic, "known"),
        freq,
        pd.Timestamp("2024-01-01") if freq else None,
        freq is not None,
    )


def episode(data, derive=BOTH, *, origins=(205,), horizon=19, screening=ScreeningConfig()):
    return EpisodeBuilder(data, screening).build(
        EpisodeRequest(0, origins, horizon, 48, "y", derive=derive)
    )


def model(*, mix=True, zero_head=False, config=TINY):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        net = V5Model(replace(config, naive_mix=mix)).eval()
    if zero_head:
        with torch.no_grad():
            net.step_projection.weight.zero_()
            net.step_projection.bias.zero_()
            net.head.projection.bias.zero_()
    return net


@pytest.mark.parametrize(
    ("derive", "keyed", "batch_type", "extra_fields"),
    [
        ("naive_rows", False, EpisodeBatch, ()),
        ("naive_rows", True, EpisodeBatchWithRowKeys, ("row_keys",)),
        ("naive_anchor", False, EpisodeBatchWithNaive, ("naive_paths", "naive_seen")),
        (
            "naive_anchor",
            True,
            EpisodeBatchWithNaiveAndRowKeys,
            ("row_keys", "naive_paths", "naive_seen"),
        ),
    ],
)
def test_earlier_derives_preserve_the_n6_batch_contract(derive, keyed, batch_type, extra_fields):
    batch = collate([episode(panel(), frozenset({derive}))], include_row_keys=keyed)
    assert type(batch) is batch_type
    assert tuple(field.name for field in fields(batch)) == N6_BATCH_FIELDS + extra_fields
    assert not {"naive_error", "naive_error_seen", "naive_resq"}.intersection(dir(batch))
    if derive == "naive_anchor":
        assert batch.naive_paths.shape == batch.naive_seen.shape == (*batch.labels.shape, 6)
        assert batch.naive_paths.dtype == torch.float32
        assert batch.naive_seen.dtype == torch.bool


def test_later_targets_do_not_change_an_earlier_anchor_or_forecast():
    data = panel()
    item = data.items[0]
    changed = item.targets["y"].copy()
    changed[192:] = np.arange(len(changed) - 192) * -50
    shifted = replace(data, items=(replace(item, targets={"y": changed}),))
    left, right = (episode(p, origins=(205, 221)) for p in (data, shifted))
    np.testing.assert_array_equal(left.naive_seen[0], right.naive_seen[0])
    np.testing.assert_allclose(left.naive_paths[0], right.naive_paths[0], atol=1e-7)
    assert not np.allclose(left.naive_paths[1], right.naive_paths[1])
    net = model()
    with torch.no_grad():
        net.mix_projection.weight.normal_(std=0.25)
        a, b = (net(collate([e])) for e in (left, right))
    torch.testing.assert_close(a.anchor[:, 0], b.anchor[:, 0])
    torch.testing.assert_close(a.quantiles[:, 0], b.quantiles[:, 0])


def test_paths_come_from_target_history_even_when_names_collide():
    data = panel(sibling=True, collision=True)
    example = episode(data)
    item = data.items[0]
    local = 205 - item.start
    kept = np.flatnonzero(example.keep).item()
    scaled = (item.targets["y"] - example.loc[kept]) / example.scale[kept]
    expected = naive_series(
        scaled, season=24, long_lag=168, left=local, right=local + 19, origin=local
    )
    assert "y~last" in example.row_names
    assert ("derived", "y~last") not in example.row_keys
    assert ("sibling", "y~season") in example.row_keys
    assert ("derived", "y~season") not in example.row_keys
    assert not example.naive_seen[:, ~example.keep].any()
    for name, (values, seen) in expected.items():
        j = OPTIONS.index(name)
        np.testing.assert_array_equal(example.naive_seen[0, kept, :19, j], seen)
        np.testing.assert_allclose(example.naive_paths[0, kept, :19, j], values, rtol=1e-6)
    assert example.naive_paths[0, kept, :19, 2] == pytest.approx(scaled[local - 24 : local].mean())


def test_paths_are_uncompressed_label_units_and_cover_missing_labels():
    data = panel(size=200)
    screening = ScreeningConfig(scale_floor=0.5)
    example = episode(data, screening=screening)
    assert example.naive_paths.dtype == np.float32
    assert example.naive_seen.dtype == bool
    assert example.naive_paths.shape == example.naive_seen.shape == (*example.labels.shape, 6)
    assert not example.naive_seen[..., 19:, :].any()
    assert not example.naive_paths[..., 19:, :].any()
    assert not example.label_mask[..., 8:].any()
    assert example.naive_seen[..., 8:19, 0].all()
    assert not example.naive_paths[..., 0].any()
    for slot, row in enumerate(example.known_rows):
        option = OPTIONS.index(example.row_names[row].split("~")[1])
        observed = example.xh[0, slot, ..., CH_OBSERVED].reshape(-1).astype(bool)
        assert observed.any()
        np.testing.assert_allclose(
            example.naive_paths[0, 0, observed, option],
            np.sinh(example.xh[0, slot, ..., CH_VALUE].reshape(-1)[observed]),
            rtol=1e-5,
        )
    raw = data.items[0].targets["y"]
    np.testing.assert_allclose(
        example.labels[0, 0, :8] - example.naive_paths[0, 0, :8, 1],
        (raw[192:] - raw[191]) / example.scale[0],
        atol=1e-7,
    )
    anchor_only = episode(data, ANCHOR, screening=screening)
    assert not any("~" in name for name in anchor_only.row_names)
    np.testing.assert_array_equal(anchor_only.naive_paths, example.naive_paths)
    np.testing.assert_array_equal(anchor_only.naive_seen, example.naive_seen)


def test_initial_parameters_and_forward_are_identical_without_active_arrays():
    plain, enabled = model(mix=False), model()
    extra = enabled.state_dict().keys() - plain.state_dict().keys()
    assert extra == {"mix_projection.weight", "mix_projection.bias"}
    assert sum(enabled.state_dict()[name].numel() for name in extra) == 198
    for name, parameter in plain.state_dict().items():
        assert torch.equal(parameter, enabled.state_dict()[name]), name
    no_arrays = collate([episode(panel(), frozenset({"naive_rows"}))])
    arrays = collate([episode(panel())])
    with torch.no_grad():
        expected, off, missing = plain(no_arrays), plain(arrays), enabled(no_arrays)
    for output in (off, missing):
        assert output.anchor is output.mix is None
        assert torch.equal(output.quantiles, expected.quantiles)


@pytest.mark.parametrize(
    ("freq", "origin", "weights"),
    [
        ("h", 205, {3: 0.9658}),
        ("h", 77, {3: 0.9763}),
        (None, 205, {1: 0.4576, 2: 0.4576}),
    ],
)
def test_initial_weights_and_zero_residual_median(freq, origin, weights):
    batch = collate([episode(panel(freq=freq), origins=(origin,))])
    with torch.no_grad():
        output = model(zero_head=True)(batch)
    for option, expected in weights.items():
        torch.testing.assert_close(
            output.mix[0, 0, 0, :19, option], torch.full((19,), expected), atol=1e-4, rtol=0
        )
    logits = torch.tensor([-3, 0, 0, 4.5, 0, -2])
    mix = logits.masked_fill(~batch.naive_seen[0, 0, 0, :19], -torch.inf).softmax(-1)
    expected = (mix * batch.naive_paths[0, 0, 0, :19]).sum(-1)
    torch.testing.assert_close(output.mix[0, 0, 0, :19], mix)
    torch.testing.assert_close(output.quantiles[0, 0, 0, :19, 4], expected)
    torch.testing.assert_close(output.anchor[0, 0, 0, :19], expected)
    torch.testing.assert_close(
        output.quantiles[..., 1:] >= output.quantiles[..., :-1],
        torch.ones_like(output.quantiles[..., 1:], dtype=torch.bool),
    )


def test_per_target_layout_is_selected_only_for_anchor():
    data = panel(sibling=True)
    jobs = [(0, 205)]
    for derive, layout in ((BOTH, "per_target"), (frozenset({"naive_rows"}), "joint")):
        assert planned_layouts(data, jobs, 19, context=48, derive=derive) == [layout]
    builder = EpisodeBuilder(data)
    requests = builder.plan_requests(0, (205,), 19, context=48, derive=BOTH)
    assert [request.target for request in requests] == ["y", "y~season"]
    for request in requests:
        example = builder.build(request)
        assert example.naive_seen[:, example.keep, :19, 0].all()
        assert not example.naive_seen[:, ~example.keep].any()
    with pytest.raises(ValueError, match="per target"):
        forecast_items(model(), data, jobs, 19, context=48, derive=BOTH, layouts=["joint"])


@pytest.mark.parametrize("keyed", [False, True])
def test_mixed_episodes_padding_and_pruned_sibling(keyed):
    from demand_forecast.data.episode.collate import EpisodeBatchWithNaive

    first = episode(panel(sibling=True), origins=(205, 221))
    second = episode(panel(), frozenset(), horizon=3)
    batch = collate([first, second], include_row_keys=keyed).to("cpu")
    assert isinstance(batch, EpisodeBatchWithNaive)
    assert isinstance(batch, EpisodeBatchWithRowKeys) is keyed
    assert batch.naive_paths.shape == batch.naive_seen.shape == (*batch.labels.shape, 6)
    assert not batch.naive_seen[1].any()
    assert not batch.naive_paths[1].any()
    net = model()
    with torch.no_grad():
        gates = net.gates(batch, zero_rows=batch.row_role == ROLE_SIBLING)
        before = net(batch, gates)
        smaller, smaller_gates = prune(batch, gates)
        after = net(smaller, smaller_gates)
        head = model(mix=False)(batch, gates)
    assert type(smaller) is type(batch)
    assert smaller.forecast_rows.shape[1] == 1 < batch.forecast_rows.shape[1]
    torch.testing.assert_close(smaller.naive_paths, batch.naive_paths[:, :, :1])
    torch.testing.assert_close(smaller.naive_seen, batch.naive_seen[:, :, :1])
    torch.testing.assert_close(before.quantiles[:, :, :1], after.quantiles)
    torch.testing.assert_close(before.quantiles[..., 19:, :], head.quantiles[..., 19:, :])
    assert torch.equal(before.quantiles[1], head.quantiles[1])
    assert torch.count_nonzero(head.quantiles[1, 0, 0, 3:])
    assert not before.anchor[1].any()
    assert not before.mix[1].any()


def test_empty_naive_cell_has_finite_softmax_backward():
    batch = collate([episode(panel(), ANCHOR)])
    paths, seen = torch.full_like(batch.naive_paths, torch.nan), torch.zeros_like(batch.naive_seen)
    paths[0, 0, 0, 0, 1] = 17
    seen[0, 0, 0, 0, 1] = True
    batch = replace(batch, naive_paths=paths, naive_seen=seen)
    net = model(zero_head=True, config=replace(TINY, expert_mode="dense", expert_enc_layers=1))
    with torch.autograd.set_detect_anomaly(True):
        result = net(batch)
        assert result.mix[0, 0, 0, 0, 1].item() == pytest.approx(1)
        assert result.quantiles[0, 0, 0, 0, 4].item() == pytest.approx(17)
        assert not result.mix[0, 0, 0, 1:].any()
        assert not result.anchor[0, 0, 0, 1:].any()
        assert result.quantiles.isfinite().all()
        result.quantiles[0, 0, 0, 1].sum().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
def test_anchor_precision_under_autocast_and_explicit_model_dtypes(dtype, expected_dtype):
    from demand_forecast.models.v5.batch_forecast import _batch_to_model

    batch = collate([episode(panel(), ANCHOR)])
    paths, seen = torch.zeros_like(batch.naive_paths), torch.zeros_like(batch.naive_seen)
    paths[0, 0, 0, :19, 1] = 1000.125
    seen[0, 0, 0, :19, 1] = True
    batch = _batch_to_model(
        replace(batch, naive_paths=paths, naive_seen=seen), torch.device("cpu"), dtype
    )
    net = model(zero_head=True).to(dtype)
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.float32),
    ):
        output = net(batch)
    for value in (output.anchor, output.mix, output.quantiles):
        assert value.dtype == expected_dtype
    torch.testing.assert_close(
        output.quantiles[0, 0, 0, :19, 4],
        torch.full((19,), 1000.125, dtype=expected_dtype),
        atol=0,
        rtol=0,
    )


def test_oversized_paths_are_unseen_before_float32_conversion():
    data = panel(freq=None)
    item = data.items[0]
    values = np.ones(256)
    values[191] = 1e31
    data = replace(data, items=(replace(item, targets={"y": values}),))
    example = EpisodeBuilder(data).build(
        EpisodeRequest(0, (205,), 19, 48, cutoff=204, derive=ANCHOR)
    )
    assert not example.naive_seen.any()
    assert not example.naive_paths.any()


def test_output_hook_sees_the_pruned_batch_and_preserves_forecasts(monkeypatch):
    data = panel(sibling=True)
    net = model(zero_head=True)
    options = dict(context=48, derive=ANCHOR)
    jobs = [(0, 205)]
    original_gates = net.gates

    def gates(batch, *, mode):
        return original_gates(batch, mode=mode, zero_rows=batch.row_role == ROLE_SIBLING)

    monkeypatch.setattr(net, "gates", gates)
    plain = forecast_items(net, data, jobs, 19, **options)
    observed = []
    explicit = forecast_items(net, data, jobs, 19, output_hook=None, **options)
    hooked = forecast_items(
        net,
        data,
        jobs,
        19,
        output_hook=lambda batch, output: observed.append((batch, output)),
        **options,
    )
    [(batch, output)] = observed
    assert isinstance(batch, EpisodeBatchWithRowKeys)
    assert batch.forecast_rows.shape == (2, 1)
    assert batch.naive_paths.shape[:4] == output.anchor.shape
    for i, key in enumerate(plain.values):
        assert plain.values[key].tobytes() == explicit.values[key].tobytes()
        assert plain.values[key].tobytes() == hooked.values[key].tobytes()
        expected_median = batch.loc[i, 0] + batch.scale[i, 0] * output.anchor[i, 0, 0, :19]
        np.testing.assert_allclose(hooked.values[key][..., 4], expected_median.numpy())


@pytest.mark.skipif(sys.platform != "linux", reason="naive batch transport requires Linux workers")
@pytest.mark.parametrize("keyed", [False, True])
def test_naive_batch_types_transport_through_loader_workers(keyed):
    from torch.utils.data import DataLoader

    batch = collate([episode(panel())], include_row_keys=keyed)
    iterator = iter(DataLoader([batch], batch_size=None, num_workers=1, timeout=10))
    (actual,) = list(iterator)
    assert type(actual) is type(batch)
    for field in fields(batch):
        expected, value = getattr(batch, field.name), getattr(actual, field.name)
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(value, expected)
        else:
            assert value == expected


def test_file_descriptor_budget_accounts_for_two_extra_tensors(monkeypatch):
    import resource

    base_budget = 256 + 4 * len(fields(EpisodeBatch))
    monkeypatch.setattr(resource, "getrlimit", lambda key: (base_budget, base_budget))
    monkeypatch.setattr(sys, "platform", "linux")
    assert usable_workers(1) == (1, None)
    workers, reason = usable_workers(1, naive_anchor=True)
    assert workers == 0
    assert reason is not None and f"below the {base_budget + 8} file descriptors" in reason


def test_evidence_file_limit_matches_forecast_preflight(monkeypatch):
    import resource

    from demand_forecast.data.episode.collate import EpisodeBatchWithNaive
    from demand_forecast.models.v5 import batch_forecast

    anchor_limit = 256 + 4 * len(fields(EpisodeBatchWithNaive))
    monkeypatch.setattr(resource, "getrlimit", lambda key: (anchor_limit, anchor_limit))
    monkeypatch.setattr(batch_forecast.sys, "platform", "linux")
    derive = ANCHOR | {"naive_evidence"}
    assert usable_workers(1, naive_anchor=True) == (1, None)
    chosen, reason = usable_workers(1, naive_anchor=True, naive_evidence=True)
    assert chosen == 0
    assert f"below the {anchor_limit + 12} file descriptors" in reason
    with pytest.raises(RuntimeError) as failure:
        forecast_items(model(), panel(), [(0, 205)], 19, context=48, workers=1, derive=derive)
    assert str(failure.value) == reason
