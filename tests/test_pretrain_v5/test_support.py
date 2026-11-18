"""Support follows exposed records and the true mechanism's grid windows."""

import importlib
from copy import deepcopy

import numpy as np
import pytest

from demand_forecast.synthetic.typed import PanelRecipe, generate_typed_panel, necessary_items
from demand_forecast.synthetic.typed.recipe import MECHANISMS


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.support")


def test_supported_rejects_unknown_item_with_id():
    generated = generate_typed_panel(PanelRecipe(("calendar",), n_items=2, length=(32, 32), seed=4))
    with pytest.raises(ValueError, match="absent-item"):
        api().supported(generated, "calendar", "absent-item", 16, context=16, horizon=16, cutoff=16)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_supported_table_against_generated_columns(mechanism):
    generated = generate_typed_panel(
        PanelRecipe(
            (mechanism,), n_items=24, length=(128, 160), freq="h", complete_inputs=True, seed=4
        )
    )
    records = generated.inputs_of(mechanism)
    ids = set(necessary_items(generated, mechanism)) if records else None
    found = None
    for item in generated.panel.items:
        if ids is not None and item.item_id not in ids:
            continue
        length = len(next(iter(item.targets.values())))
        for local in range(16, length):
            origin = item.start + local
            if api().supported(
                generated, mechanism, item.item_id, origin, context=64, horizon=32, cutoff=origin
            ):
                found = item, origin
                break
        if found:
            break
    # Hourly long lag is 168; the seasonal rule is checked explicitly below.
    if mechanism == "seasonal_lag":
        assert found is None
        item = generated.panel.items[0]
        origin = item.start + generated.panel.long_lag + 16
        assert api().supported(
            generated, mechanism, item.item_id, origin, context=64, horizon=32, cutoff=origin
        )
        return
    assert found is not None, mechanism
    item, origin = found
    left = max(0, origin - 64 - item.start)
    stop = origin - item.start
    if mechanism in ("planned_event", "event_response"):
        column = next(r.name for r in records if r.where == "dynamic")
        assert (item.dynamic[column][stop : stop + 32] == 1).any()
    if mechanism == "missing_vs_zero":
        values = item.targets[sorted(item.targets)[0]][left:stop]
        assert np.isnan(values).any() and (values == 0).any()
    if mechanism.startswith("dynamic_category_"):
        column = item.dynamic[records[0].name]
        end = stop if records[0].role == "past" else min(len(column), stop + 32)
        assert (column[left + 1 : end] != column[left : end - 1]).any()
    if records:
        hidden = deepcopy(generated)
        exposed = next(i for i in hidden.panel.items if i.item_id == item.item_id)
        for record in records:
            if record.where == "static":
                exposed.static[record.name] = None
            else:
                exposed.dynamic[record.name][:] = None if record.kind == "category" else np.nan
        assert not api().supported(
            hidden, mechanism, item.item_id, origin, context=64, horizon=32, cutoff=origin
        )


def test_only_peer_must_have_sixteen_observations():
    g = generate_typed_panel(
        PanelRecipe(
            ("static_category",), n_items=24, length=(128, 160), complete_inputs=True, seed=4
        )
    )
    item_id = necessary_items(g, "static_category")[0]
    item = next(i for i in g.panel.items if i.item_id == item_id)
    origin = item.start + 16
    attribute = g.inputs_of("static_category")[0].name
    peers = [peer for peer in g.panel.items if peer.item_id != item_id]
    for peer in peers:
        peer.static[attribute] = f"unshared:{peer.item_id}"
    only_peer = peers[0]
    only_peer.static[attribute] = item.static[attribute]
    for values in only_peer.targets.values():
        values[15:] = np.nan
    assert not api().supported(
        g, "static_category", item_id, origin, context=64, horizon=16, cutoff=origin
    )


def test_protected_drop_table():
    protect = api().PROTECT
    assert set(protect) == set(MECHANISMS)
    assert set(protect["static_category"]) == {"attributes", "peers", "value_stats"}
    assert "static_numbers" in protect["static_number"]
    assert set(protect["event_response"]) == {"attributes", "peers", "features", "value_stats"}
    assert protect["missing_vs_zero"] == ()
    assert set(protect["seasonal_lag"]) == {"rolling", "lags", "zeros"}


def test_common_support_requires_target_and_column_observed_together():
    g = generate_typed_panel(
        PanelRecipe(("known_driver",), n_items=2, length=(96, 96), complete_inputs=True, seed=2)
    )
    item = g.panel.items[0]
    name = g.inputs_of("known_driver")[0].name
    origin = item.start + 64
    assert api().supported(
        g, "known_driver", item.item_id, origin, context=64, horizon=16, cutoff=origin
    )
    # Each series has 32 observations, but their intersection is empty.
    item.targets["target"][:32] = np.nan
    item.dynamic[name][32:64] = np.nan
    assert not api().supported(
        g, "known_driver", item.item_id, origin, context=64, horizon=16, cutoff=origin
    )


def test_events_require_true_one_and_exposed_availability():
    g = generate_typed_panel(
        PanelRecipe(("planned_event",), n_items=2, length=(96, 96), complete_inputs=True, seed=2)
    )
    item, truth = g.panel.items[0], g.complete.items[0]
    name = g.inputs_of("planned_event")[0].name
    origin = item.start + 64
    truth.dynamic[name][64:80] = 0
    item.dynamic[name][64:80] = 1
    assert not api().supported(
        g, "planned_event", item.item_id, origin, context=64, horizon=16, cutoff=origin
    )
    truth.dynamic[name][64] = 1
    assert api().supported(
        g, "planned_event", item.item_id, origin, context=64, horizon=16, cutoff=origin
    )
    item.dynamic[name][64] = np.nan
    assert not api().supported(
        g, "planned_event", item.item_id, origin, context=64, horizon=16, cutoff=origin
    )


def test_past_category_change_must_be_adjacent_and_before_origin():
    g = generate_typed_panel(
        PanelRecipe(
            ("dynamic_category_mean",), n_items=2, length=(96, 96), complete_inputs=True, seed=2
        )
    )
    item = g.panel.items[0]
    record = g.inputs_of("dynamic_category_mean")[0]
    column = item.dynamic[record.name]
    column[:] = "a"
    column[64:] = "b"
    origin = item.start + 64
    if record.role == "past":
        assert not api().supported(
            g, record.mechanism, item.item_id, origin, context=64, horizon=16, cutoff=origin
        )
    column[40:64] = "b"
    column[39] = None
    column[64:] = "b"
    assert not api().supported(
        g, record.mechanism, item.item_id, origin, context=64, horizon=16, cutoff=origin
    )
    column[39] = "a"
    assert api().supported(
        g, record.mechanism, item.item_id, origin, context=64, horizon=16, cutoff=origin
    )
