"""Small deterministic v5 panels and copies for statistics-cutoff checks."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd

from demand_forecast.data.episode import Item, Panel


def toy_panel(
    n_items: int = 6,
    length: int = 400,
    seed: int = 0,
    dated: bool = True,
    n_targets: int = 1,
) -> Panel:
    """Build ragged daily items with flags, numbers, categories and statics."""
    rng = np.random.default_rng(seed)
    steps = np.arange(length)[None, :]
    positions = np.arange(n_items)[:, None]
    promo = ((steps + positions) % 11 == 0).astype(np.float64)
    temp = 15 + np.sin(2 * np.pi * steps / 30) + rng.normal(size=(n_items, length))
    event = np.where(
        (steps + positions) % 29 == 0,
        "holiday",
        np.where(
            (steps + positions) % 13 == 0,
            "sale",
            np.where((steps + positions) % 37 == 0, "clearance", None),
        ),
    )
    level = 10 + positions + 2 * np.sin(2 * np.pi * steps / 7) + 3 * promo
    targets = (
        level[:, None, :]
        + np.arange(n_targets)[None, :, None]
        + rng.normal(scale=0.5, size=(n_items, n_targets, length))
    )
    starts = np.minimum(np.arange(n_items) * 3, max(0, length - 1))
    # Records are ragged dictionaries; the time-series arithmetic above is batched.
    items = tuple(
        Item(
            item_id=f"item_{i}",
            start=int(start),
            targets={
                "y" if j == 0 else f"y{j}": targets[i, j, start:].copy() for j in range(n_targets)
            },
            dynamic={
                "promo": promo[i, start:].copy(),
                "temp": temp[i, start:].copy(),
                "event": event[i, start:].copy(),
            },
            static={
                "brand": f"brand_{i % 3}",
                "tags": [f"tag_{i % 2}", f"tier_{i % 3}"],
                "size": 1.5 + 0.3 * i,
                "dept": i % 2,
            },
        )
        for i, start in enumerate(starts)
    )
    return Panel(
        name="toy",
        items=items,
        roles={"promo": "known", "temp": "past", "event": "known"},
        freq="D" if dated else None,
        epoch=pd.Timestamp("2020-01-01") if dated else None,
        aligned=True,
    )


def perturb_after(
    panel: Panel,
    index: int,
    *,
    value: float = 1e6,
    keep_known: bool = False,
    only_items: Sequence[int] | None = None,
) -> Panel:
    """Copy a panel, replacing permitted cells at absolute grid indices >= index."""
    selected = None if only_items is None else set(only_items)
    items = []
    for position, item in enumerate(panel.items):
        targets = {name: values.copy() for name, values in item.targets.items()}
        dynamic = {name: values.copy() for name, values in item.dynamic.items()}
        if selected is None or position in selected:
            for columns, is_dynamic in ((targets, False), (dynamic, True)):
                for name, values in columns.items():
                    if is_dynamic and keep_known and panel.roles[name] == "known":
                        continue
                    values[np.arange(len(values)) + item.start >= index] = value
        items.append(replace(item, targets=targets, dynamic=dynamic, static=deepcopy(item.static)))
    return replace(panel, items=tuple(items), roles=panel.roles.copy())
