"""T0: panel-to-forecast correctness, availability and selection contracts."""

from collections.abc import Callable, Iterator
from dataclasses import replace

import numpy as np
import pytest
import torch
from tsfresh.feature_extraction import feature_calculators as fc

from demand_forecast.data.episode import (
    EpisodeBatch,
    EpisodeBuilder,
    EpisodeRequest,
    Panel,
    collate,
    fit_scale,
    infer_kind,
    infer_static_kind,
    is_missing,
    label_of,
)
from demand_forecast.data.episode.bank import TSFRESH_PARITY
from demand_forecast.data.episode.schema import (
    CH_STEP,
    CH_STEP_FLAG,
    CH_VALUE_STATS,
    KIND_CATEGORY,
    KIND_FLAG,
    ROLE_KNOWN,
    ROLE_PANEL,
    ROLE_PAST,
    ROLE_PEER,
    ROLE_SIBLING,
    SUMMARY_FEATURES,
)
from demand_forecast.models.v5 import V5Config, V5Model, forecast
from demand_forecast.models.v5.model import Gates, V5Output, prune
from tests.helpers.episodes import perturb_after, toy_panel

TINY = V5Config(d_model=64, nhead=4, enc_layers=2, dec_layers=2, dim_feedforward=128)
REQUEST = EpisodeRequest(0, (336,), 28, context=256)
GateFactory = Callable[[EpisodeBatch, dict[str, float] | None], Gates]


@pytest.fixture(autouse=True)
def deterministic_eval() -> Iterator[None]:
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(0)
        yield


@pytest.fixture
def model() -> V5Model:
    result = V5Model(TINY).eval()
    # Defaults initialise these outputs to constant, saturated gates. Randomise
    # them too, so equality checks can detect leakage into the learned inputs.
    result.gate_net.net[-1].reset_parameters()
    result.family_logits.reset_parameters()
    return result


@pytest.fixture(name="mixed_gates")
def mixed_gate_factory(model: V5Model) -> GateFactory:
    def mixed_gates(batch: EpisodeBatch, values: dict[str, float] | None = None) -> Gates:
        """Fix independent gates by name, retaining learned forecast/known gates."""
        gates = model.gates(batch, mode="learned")
        row, cand = gates.row.clone(), gates.cand.clone()
        for b, names in enumerate(batch.row_names):
            independent = {
                name: i
                for i, name in enumerate(names)
                if batch.row_role[b, i] in (ROLE_SIBLING, ROLE_PAST, ROLE_PANEL)
            }
            candidates = {name: i for i, name in enumerate(batch.cand_names[b])}
            defaults = {
                name: (0.0, 1.0, 0.5)[i % 3]
                for i, name in enumerate(sorted(set(independent) | set(candidates)))
            }
            selected = defaults if values is None else {**defaults, **values}
            for name, i in independent.items():
                row[b, i] = selected[name]
            for name, i in candidates.items():
                cand[b, i] = selected[name]
            for i in range(len(names)):
                if batch.row_role[b, i] == ROLE_PEER:
                    row[b, i] = cand[b, batch.row_cand[b, i]]
        return replace(
            gates,
            row=row,
            cand=cand,
            family=batch.x.new_tensor([[1, 0.5, 0]]).expand(batch.x.shape[0], -1),
        )

    return mixed_gates


def _batch(panel: Panel, request: EpisodeRequest = REQUEST) -> EpisodeBatch:
    return collate([EpisodeBuilder(panel).build(request)])


def _equal(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=0, atol=1e-5), (a, b)


def _kept(output: V5Output, batch: EpisodeBatch, index: int = 0) -> np.ndarray:
    quantiles = output.quantiles[index, batch.origin_mask[index]]
    keep = batch.keep[index] & batch.forecast_mask[index]
    return quantiles[:, keep, : int(batch.horizon[index])].numpy()


def _gate_names(
    batch: EpisodeBatch, gates: Gates, index: int = 0, rename: Callable[[str], str] = str
) -> dict[str, float]:
    result = {
        f"row:{rename(name)}": float(gates.row[index, i])
        for i, name in enumerate(batch.row_names[index])
    }
    result.update(
        {
            f"candidate:{name}": float(gates.cand[index, i])
            for i, name in enumerate(batch.cand_names[index])
        }
    )
    result.update({f"family:{i}": float(value) for i, value in enumerate(gates.family[index])})
    return result


def _equal_learned_gates(
    model: V5Model,
    before: EpisodeBatch,
    after: EpisodeBatch,
    *,
    index: int = 0,
    before_index: int = 0,
    rename: Callable[[str], str] = str,
    removed: bool = False,
) -> None:
    a, b = model.gates(before, mode="learned"), model.gates(after, mode="learned")
    left = _gate_names(before, a, before_index, rename=rename)
    right = _gate_names(after, b, index)
    assert right.keys() <= left.keys() if removed else left.keys() == right.keys()
    _equal(np.array([left[name] for name in right]), np.array(list(right.values())))
    _equal(a.summary_mask, b.summary_mask)


def _gate_values(batch: EpisodeBatch, gates: Gates) -> dict[str, float]:
    """Carry surviving independent values through removal without cycling again."""
    return {
        **{name: float(gates.row[0, i]) for i, name in enumerate(batch.row_names[0])},
        **{name: float(gates.cand[0, i]) for i, name in enumerate(batch.cand_names[0])},
    }


def _change_cells(
    panel: Panel,
    left: int,
    *,
    right: int | None = None,
    targets: bool = False,
    columns: tuple[str, ...] = (),
    items: tuple[int, ...] | None = None,
    value: float = 1e6,
) -> Panel:
    changed = []
    for i, item in enumerate(panel.items):
        target, dynamic = item.targets.copy(), item.dynamic.copy()
        if items is None or i in items:
            for source, names in ((target, tuple(target) if targets else ()), (dynamic, columns)):
                for name in names:
                    cells = source[name].copy()
                    grid = item.start + np.arange(len(cells))
                    mask = grid >= left
                    if right is not None:
                        mask &= grid < right
                    cells[mask] = value
                    source[name] = cells
        changed.append(replace(item, targets=target, dynamic=dynamic))
    return replace(panel, items=tuple(changed))


def _rename(panel: Panel, cutoff: int) -> tuple[Panel, Callable[[str], str]]:
    """Reverse label spelling/order, preserving missing values and numeric statics."""
    static_maps, dynamic_maps = {}, {}
    for column in panel.items[0].static:
        cells = [item.static[column] for item in panel.items]
        if infer_static_kind(cells) == "number":
            continue
        labels = {
            label_of(value)
            for cell in cells
            for value in (cell if isinstance(cell, (list, tuple)) else [cell])
            if not is_missing(value)
        }
        static_maps[column] = {
            label: f"renamed_{len(labels) - i}" for i, label in enumerate(sorted(labels))
        }
    for column in panel.roles:
        if infer_kind(panel.items[0].dynamic[column][:cutoff]) not in (KIND_FLAG, KIND_CATEGORY):
            continue
        labels = {
            label_of(value)
            for item in panel.items
            for value in item.dynamic[column]
            if not is_missing(value)
        }
        dynamic_maps[column] = {
            label: f"value_{len(labels) - i}" for i, label in enumerate(sorted(labels))
        }

    def cell(value: object, mapping: dict[str, str]) -> object:
        if is_missing(value):
            return value
        if isinstance(value, (list, tuple)):
            return [cell(v, mapping) for v in value]
        return mapping[label_of(value)]

    renamed = replace(
        panel,
        items=tuple(
            replace(
                item,
                item_id=f"renamed_item_{len(panel.items) - i}",
                static={
                    name: cell(v, static_maps[name]) if name in static_maps else v
                    for name, v in item.static.items()
                },
                dynamic={
                    name: np.array([cell(v, dynamic_maps[name]) for v in values], dtype=object)
                    if name in dynamic_maps
                    else values.copy()
                    for name, values in item.dynamic.items()
                },
            )
            for i, item in enumerate(panel.items)
        ),
    )

    def row_name(name: str) -> str:
        column, separator, label = name.partition("=")
        return (
            f"{column}={static_maps[column][label]}"
            if separator and column in static_maps
            else name
        )

    return renamed, row_name


def test_renaming_ids_and_labels(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel(n_items=12, n_targets=2)
    request = replace(REQUEST, target="y")
    renamed, rename = _rename(panel, 336)
    a, b = _batch(panel, request), _batch(renamed, request)
    assert panel.items[0].item_id != renamed.items[0].item_id
    assert panel.items[0].static["tags"] != renamed.items[0].static["tags"]
    _equal_learned_gates(model, a, b, rename=rename)
    _equal(_kept(model(a, mixed_gates(a)), a), _kept(model(b, mixed_gates(b)), b))


def test_reordering(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel(n_items=12, n_targets=3)
    order = np.random.default_rng(7).permutation(len(panel.items))
    shuffled = replace(
        panel,
        items=tuple(
            replace(
                panel.items[i],
                targets=dict(reversed(list(panel.items[i].targets.items()))),
                dynamic=dict(reversed(list(panel.items[i].dynamic.items()))),
                static={
                    name: list(reversed(value)) if isinstance(value, list) else value
                    for name, value in reversed(list(panel.items[i].static.items()))
                },
            )
            for i in order
        ),
        roles=dict(reversed(list(panel.roles.items()))),
    )
    request = replace(REQUEST, target="y")
    a = _batch(panel, request)
    b = _batch(shuffled, replace(request, item_index=int(np.flatnonzero(order == 0)[0])))
    _equal_learned_gates(model, a, b)
    _equal(_kept(model(a, mixed_gates(a)), a), _kept(model(b, mixed_gates(b)), b))


@pytest.mark.parametrize("horizon", [5, 28])
@pytest.mark.parametrize("index", [0, 1])
def test_batch_packing(model: V5Model, mixed_gates: GateFactory, horizon: int, index: int) -> None:
    panel = toy_panel(n_targets=2)
    small = EpisodeBuilder(panel).build(replace(REQUEST, context=128, horizon=horizon, target="y"))
    large = EpisodeBuilder(toy_panel(n_items=12, n_targets=5)).build(
        EpisodeRequest(0, (304, 320, 336), 53, context=384, target="y")
    )
    alone = collate([small])
    packed = collate([small, large] if index == 0 else [large, small])
    assert len(large.x) > len(small.x)
    assert packed.x.shape[2] > alone.x.shape[2]
    _equal_learned_gates(model, alone, packed, index=index)
    _equal(
        _kept(model(alone, mixed_gates(alone)), alone),
        _kept(model(packed, mixed_gates(packed)), packed, index),
    )


@pytest.mark.parametrize("origins", [(336,), (320, 336)])
def test_hidden_targets(model: V5Model, mixed_gates: GateFactory, origins: tuple[int, ...]) -> None:
    panel = toy_panel(n_targets=3)
    changed = _change_cells(panel, origins[0], targets=True, items=(0,))
    request = replace(REQUEST, origins=origins, target="y")
    a, b = _batch(panel, request), _batch(changed, request)
    assert not torch.equal(a.labels, b.labels)
    _equal_learned_gates(model, a, b)
    _equal(_kept(model(a, mixed_gates(a)), a)[:1], _kept(model(b, mixed_gates(b)), b)[:1])


@pytest.mark.parametrize("origins", [(336,), (320, 336)])
def test_past_only_after_origin(
    model: V5Model, mixed_gates: GateFactory, origins: tuple[int, ...]
) -> None:
    panel = toy_panel()
    changed = _change_cells(panel, origins[0], columns=("temp",), items=(0,))
    request = replace(REQUEST, origins=origins)
    a, b = _batch(panel, request), _batch(changed, request)
    _equal_learned_gates(model, a, b)
    _equal(_kept(model(a, mixed_gates(a)), a)[:1], _kept(model(b, mixed_gates(b)), b)[:1])


@pytest.mark.parametrize("origins", [(336,), (320, 336)])
def test_peers_after_origin(
    model: V5Model, mixed_gates: GateFactory, origins: tuple[int, ...]
) -> None:
    panel = toy_panel()
    changed = _change_cells(
        panel, origins[0], targets=True, items=tuple(range(1, len(panel.items)))
    )
    request = replace(REQUEST, origins=origins)
    a, b = _batch(panel, request), _batch(changed, request)
    assert (a.row_role == ROLE_PEER).any() and (a.row_role == ROLE_PANEL).any()
    _equal_learned_gates(model, a, b)
    first, second = model(a, mixed_gates(a)), model(b, mixed_gates(b))
    _equal(_kept(first, a)[:1], _kept(second, b)[:1])
    _equal(first.router, second.router)


def test_after_cutoff_statistics(model: V5Model) -> None:
    panel = toy_panel(n_items=12, n_targets=3)
    changed = _change_cells(panel, 304, right=336, targets=True, columns=tuple(panel.roles))
    request = replace(REQUEST, cutoff=304, target="y")
    a, b = _batch(panel, request), _batch(changed, request)
    assert not torch.equal(a.x, b.x)
    _equal_learned_gates(model, a, b)
    for name in (
        "row_summary",
        "row_summary_flags",
        "row_score",
        "cand_summary",
        "cand_flags",
        "cand_score",
        "attr_stats",
        "static_num",
        "loc",
        "scale",
    ):
        _equal(getattr(a, name), getattr(b, name))
    _equal(model(a).router, model(b).router)
    # Positive control: a changed permitted prefix must reach the learned gates.
    prefix = _batch(_change_cells(panel, 272, right=304, targets=True), request)
    assert not torch.allclose(model.gates(a).family, model.gates(prefix).family, rtol=0, atol=1e-5)


def test_known_future_changes_forecast(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel()
    changed = _change_cells(panel, 336, columns=("promo",), items=(0,), value=1)
    a, b = _batch(panel), _batch(changed)
    _equal_learned_gates(model, a, b)
    assert (
        np.max(np.abs(_kept(model(a, mixed_gates(a)), a) - _kept(model(b, mixed_gates(b)), b)))
        > 1e-6
    )


def test_known_future_never_seen_before(model: V5Model, mixed_gates: GateFactory) -> None:
    zero = _change_cells(toy_panel(), 0, columns=("promo",), value=0)
    future = _change_cells(zero, 336, columns=("promo",), value=1)
    a, b = _batch(zero), _batch(future)
    _equal_learned_gates(model, a, b)
    row = b.row_names[0].index("promo")
    assert b.row_role[0, row] == ROLE_KNOWN
    assert model.gates(b).row[0, row] > 0
    assert (
        np.max(np.abs(_kept(model(a, mixed_gates(a)), a) - _kept(model(b, mixed_gates(b)), b)))
        > 1e-6
    )


def test_swapped_futures(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel()
    items, swapped = [], []
    for item in panel.items:
        t = item.start + np.arange(len(item.targets["y"]))
        a = np.sin(t / 9) + t / 100
        b = np.cos(t / 13) - t / 80
        a[t >= 336], b[t >= 336] = 8 + np.sin(t[t >= 336]), -6 + np.cos(t[t >= 336])
        dynamic = {**item.dynamic, "driver_a": a, "driver_b": b}
        items.append(replace(item, dynamic=dynamic))
        aa, bb = a.copy(), b.copy()
        aa[t >= 336], bb[t >= 336] = b[t >= 336], a[t >= 336]
        swapped.append(replace(item, dynamic={**dynamic, "driver_a": aa, "driver_b": bb}))
    panel = replace(
        panel, items=tuple(items), roles={**panel.roles, "driver_a": "known", "driver_b": "known"}
    )
    a, b = _batch(panel), _batch(replace(panel, items=tuple(swapped)))
    _equal_learned_gates(model, a, b)
    assert (
        np.max(np.abs(_kept(model(a, mixed_gates(a)), a) - _kept(model(b, mixed_gates(b)), b)))
        > 1e-6
    )


def test_past_value_changes_forecast(model: V5Model) -> None:
    panel = toy_panel()
    changed = _change_cells(panel, 304, right=336, columns=("temp",), items=(0,))
    request = replace(REQUEST, cutoff=304)
    a, b = _batch(panel, request), _batch(changed, request)
    _equal_learned_gates(model, a, b)
    assert (
        np.max(
            np.abs(
                _kept(model(a, model.gates(a, mode="ones")), a)
                - _kept(model(b, model.gates(b, mode="ones")), b)
            )
        )
        > 1e-6
    )


def test_origins_in_one_pass(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel(n_targets=3)
    request = replace(REQUEST, origins=(304, 320, 336), cutoff=304, target="y")
    combined = _batch(panel, request)
    gates = mixed_gates(combined)
    predictions = _kept(model(combined, gates), combined)
    for i, origin in enumerate(request.origins):
        single = _batch(panel, replace(request, origins=(origin,), context=256 - (336 - origin)))
        _equal_learned_gates(model, combined, single)
        _equal(
            predictions[i : i + 1],
            _kept(model(single, mixed_gates(single, _gate_values(combined, gates))), single),
        )


@pytest.mark.parametrize("dated", [False, True])
@pytest.mark.parametrize("horizon", [5, 19])
def test_inference_origin_anywhere(model: V5Model, dated: bool, horizon: int) -> None:
    panel = toy_panel(n_items=1, length=11, dated=dated)
    request = EpisodeRequest(0, (11,), horizon, context=256)
    batch = _batch(panel, request)
    assert batch.patch_valid.sum() == batch.row_mask.sum()
    result = forecast(model, panel, request)
    assert result.quantiles.shape == (1, 1, horizon, 9)
    assert np.isfinite(result.quantiles).all()


@pytest.mark.parametrize("packed", [False, True])
def test_pruning_changes_nothing(model: V5Model, mixed_gates: GateFactory, packed: bool) -> None:
    panel = toy_panel(n_items=12, n_targets=3)
    episode = EpisodeBuilder(panel).build(replace(REQUEST, target="y"))
    episodes = [episode]
    if packed:
        episodes.append(
            EpisodeBuilder(toy_panel(n_items=1)).build(replace(REQUEST, context=64, horizon=5))
        )
    batch = collate(episodes)
    gates = mixed_gates(batch)
    assert {0.0, 0.5, 1.0} <= set(gates.row.flatten().tolist()) | set(gates.cand.flatten().tolist())
    before = model(batch, gates)
    reduced, selected = prune(batch, gates)
    assert reduced.row_mask.sum() < batch.row_mask.sum()
    assert reduced.cand_mask.sum() < batch.cand_mask.sum()
    assert reduced.static_num_mask.sum() < batch.static_num_mask.sum()
    after = model(reduced, selected)
    for i in range(len(episodes)):
        _equal(_kept(before, batch, i), _kept(after, reduced, i))
        _equal_learned_gates(model, batch, reduced, index=i, before_index=i, removed=True)


def test_zero_gate_equals_absence(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel(n_targets=3)
    request = replace(REQUEST, target="y")
    a = _batch(panel, request)
    # Exclude the past column itself, retaining the same known and sibling columns.
    b = _batch(
        panel,
        replace(request, keep_columns=frozenset((set(panel.roles) | {"y1", "y2"}) - {"temp"})),
    )
    assert "temp" in a.row_names[0] and "temp" not in b.row_names[0]
    gates = mixed_gates(a, {"temp": 0.0})
    _equal_learned_gates(model, a, b, removed=True)
    _equal(_kept(model(a, gates), a), _kept(model(b, mixed_gates(b, _gate_values(a, gates))), b))


def test_group_of_one(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel(n_items=1)
    a = _batch(panel)
    b = _batch(panel, replace(REQUEST, drop=frozenset({"attributes", "peers", "panel_row"})))
    _equal_learned_gates(model, a, b)
    _equal(_kept(model(a, mixed_gates(a)), a), _kept(model(b, mixed_gates(b)), b))

    panel = toy_panel(n_items=12)
    panel = replace(
        panel,
        items=tuple(
            replace(item, static={"group": "alone" if i == 0 else "shared"})
            for i, item in enumerate(panel.items)
        ),
    )
    without = replace(panel, items=tuple(replace(item, static={}) for item in panel.items))
    a, b = _batch(panel), _batch(without)
    assert "group" in a.cand_names[0]
    assert not a.attr_flags.any() and not (a.row_role == ROLE_PEER).any()
    _equal_learned_gates(model, a, b, removed=True)
    _equal(
        _kept(model(a, model.gates(a, mode="ones")), a),
        _kept(model(b, model.gates(b, mode="ones")), b),
    )


def test_padded_horizon_is_inert(model: V5Model, mixed_gates: GateFactory) -> None:
    panel = toy_panel()
    changed = _change_cells(panel, 341, columns=("promo", "event"), items=(0,))
    request = replace(REQUEST, horizon=5)
    a, b = _batch(panel, request), _batch(changed, request)
    _equal_learned_gates(model, a, b)
    _equal(_kept(model(a, mixed_gates(a)), a), _kept(model(b, mixed_gates(b)), b))


def test_no_dataset_parameters(model: V5Model) -> None:
    other = V5Model(TINY).eval()
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    panel = toy_panel(n_items=19, n_targets=3)
    renamed, _ = _rename(panel, 336)
    forecast(model, toy_panel(n_items=1), REQUEST)
    forecast(other, renamed, replace(REQUEST, target="y"))
    assert shapes == {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert shapes == {name: tuple(value.shape) for name, value in other.state_dict().items()}


def test_feature_bank_is_causal(model: V5Model) -> None:
    panel = toy_panel()
    request = replace(REQUEST, cutoff=160)
    a = _batch(panel, request)
    b = _batch(perturb_after(panel, 200), request)
    _equal_learned_gates(model, a, b)
    for name in ("y", "temp"):
        i = a.row_names[0].index(name)
        x, changed = a.x[0, i].flatten(0, 1), b.x[0, i].flatten(0, 1)
        # The feature at 200 may use 199, but never the changed value at 200.
        _equal(x[:121, CH_STEP:CH_VALUE_STATS], changed[:121, CH_STEP:CH_VALUE_STATS])
        assert not torch.equal(
            x[121:, CH_STEP:CH_VALUE_STATS], changed[121:, CH_STEP:CH_VALUE_STATS]
        )
    event = a.row_names[0].index("event")
    _equal(
        a.x[0, event].flatten(0, 1)[:120, CH_VALUE_STATS:],
        b.x[0, event].flatten(0, 1)[:120, CH_VALUE_STATS:],
    )


@pytest.mark.parametrize("length", [64, 200, 400])
def test_feature_bank_matches_tsfresh(model: V5Model, length: int) -> None:
    panel = toy_panel(length=length)
    request = EpisodeRequest(0, (length,), 5, context=256)
    batch = _batch(panel, request)
    for name in ("y", "temp"):
        raw = (panel.items[0].targets if name == "y" else panel.items[0].dynamic)[name]
        scale = fit_scale(raw, length, panel.season)
        scaled = (raw - scale.loc) / scale.scale
        row = batch.row_names[0].index(name)
        for feature, (calculator, arguments) in TSFRESH_PARITY.items():
            kwargs = {**arguments}
            if feature == "acf_long":
                kwargs["lag"] = panel.long_lag
            result = getattr(fc, calculator)(scaled, **kwargs)
            if calculator == "linear_trend":
                result = next(iter(result))[1]
                result = result * length if feature == "trend_slope" else result**2
            elif calculator == "cid_ce":
                result /= np.sqrt(length)
            elif calculator.startswith("longest_strike_"):
                result /= length
            column = SUMMARY_FEATURES.index(feature)
            assert bool(batch.row_summary_flags[0, row, column]) == bool(np.isfinite(result))
            if np.isfinite(result):
                assert float(batch.row_summary[0, row, column]) == pytest.approx(
                    np.clip(result, -10, 10), abs=1e-4
                )
    assert np.isfinite(_kept(model(batch), batch)).all()


@pytest.mark.parametrize("case", ["constant", "short", "missing", "ragged"])
def test_feature_bank_edge_cases(model: V5Model, case: str) -> None:
    raw = {
        "constant": np.full(64, 7.0),
        "short": np.array([1.0, 2.0, 4.0]),
        "missing": np.full(64, np.nan),
        "ragged": np.array([np.nan, 1.0, np.nan, 4.0, 2.0]),
    }[case]
    panel = toy_panel(n_items=1, length=len(raw))
    item = replace(panel.items[0], start=17, targets={"y": raw}, dynamic={}, static={})
    panel = replace(panel, items=(item,), roles={})
    request = EpisodeRequest(0, (17 + len(raw),), 5, context=256)
    batch = _batch(panel, request)
    assert torch.isfinite(batch.x).all() and torch.isfinite(batch.row_summary).all()
    assert not batch.row_summary[~batch.row_summary_flags].any()
    flags = batch.row_summary_flags[0, 0]
    if case == "missing":
        assert not flags.any() and not batch.x.any()
    if case in ("constant", "short"):
        for name in ("acf_season", "acf_long", "spec_period", "spec_strength"):
            assert not flags[SUMMARY_FEATURES.index(name)]
    if case == "constant":
        for name in ("acf_1", "trend_slope", "trend_strength"):
            assert not flags[SUMMARY_FEATURES.index(name)]
    assert not batch.x[0, 0, 0, :, CH_STEP_FLAG:CH_VALUE_STATS].any()
    result = forecast(model, panel, request)
    assert result.quantiles.shape == (1, 1, 5, 9) and np.isfinite(result.quantiles).all()
    if case in ("constant", "missing"):
        assert (result.quantiles == (7.0 if case == "constant" else 0.0)).all()


@pytest.mark.parametrize("dated", [False, True])
def test_no_feature_episode_runs(model: V5Model, dated: bool) -> None:
    panel = toy_panel(n_items=1, dated=dated)
    panel = replace(panel, items=(replace(panel.items[0], dynamic={}, static={}),), roles={})
    batch = _batch(panel)
    assert batch.row_mask.sum() == 1 and not batch.cand_mask.any()
    result = forecast(model, panel, REQUEST)
    assert result.quantiles.shape == (1, 1, 28, 9) and np.isfinite(result.quantiles).all()
