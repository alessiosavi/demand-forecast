"""Input-assignment null controls and candidate queries for typed panels."""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from demand_forecast.data.episode import Item, is_missing
from demand_forecast.data.episode.kinds import observed_mask
from demand_forecast.synthetic.typed.draft import INPUT_MECHANISMS, stream
from demand_forecast.synthetic.typed.recipe import GeneratedPanel, InputRecord


def _length(item: Item) -> int:
    return len(next(iter(item.targets.values())))


def _moving_inputs(generated: GeneratedPanel, record: InputRecord) -> tuple[InputRecord, ...]:
    if record.twin == "hierarchy":
        twins = tuple(
            (i, r)
            for i, r in enumerate(generated.inputs)
            if r.twin == record.twin and r.process is record.process
        )
        if (
            record.mechanism != "nuisance"
            or record.process is None
            or len(twins) != 2
            or twins[1][0] != twins[0][0] + 1
            or any(r.mechanism != "nuisance" for _, r in twins)
        ):
            raise ValueError(f"Invalid hierarchy twin pair for input: {record.name!r}")
        return tuple(r for _, r in twins)
    if record.mechanism == "hierarchy":
        return generated.inputs_of("hierarchy")
    return (record,)


def shuffle_assignment(generated: GeneratedPanel, name: str, seed: int) -> GeneratedPanel:
    """Move complete donor inputs between items, retaining each recipient's missing cells."""
    record = next((r for r in generated.inputs if r.name == name), None)
    if record is None:
        raise ValueError(f"Unknown input: {name!r}")
    items = generated.complete.items
    if len(items) < 2:
        raise ValueError("Shuffling requires at least two items")
    moving = _moving_inputs(generated, record)
    rng = stream(seed, f"shuffle:{name}")
    indices = np.arange(len(items))
    permutation = rng.permutation(len(items))
    while (permutation == indices).any():
        permutation = rng.permutation(len(items))

    panel, complete = deepcopy(generated.panel), deepcopy(generated.complete)
    assignment = {}
    for i, donor_index in enumerate(permutation):
        donor = items[donor_index]
        offset = int(rng.integers(_length(donor))) if record.where == "dynamic" else 0
        assignment[items[i].item_id] = (donor.item_id, offset)
        for moved in moving:
            if moved.where == "static":
                complete.items[i].static[moved.name] = deepcopy(donor.static[moved.name])
                if not is_missing(generated.panel.items[i].static[moved.name]):
                    panel.items[i].static[moved.name] = deepcopy(donor.static[moved.name])
            else:
                column = donor.dynamic[moved.name]
                values = column[(np.arange(_length(items[i])) + offset) % len(column)]
                complete.items[i].dynamic[moved.name] = values
                original = generated.panel.items[i].dynamic[moved.name]
                missing = ~observed_mask(original)
                masked = values.copy()
                masked[missing] = original[missing]
                panel.items[i].dynamic[moved.name] = masked
    names = tuple(r.name for r in moving)
    return replace(
        generated,
        panel=panel,
        complete=complete,
        shuffled=tuple(dict.fromkeys((*generated.shuffled, *names))),
        donors={**generated.donors, **{name: assignment.copy() for name in names}},
    )


def necessary_items(generated: GeneratedPanel, mechanism: str) -> tuple[str, ...]:
    """Return the table's candidate items; episode support still depends on its origin."""
    records = generated.inputs_of(mechanism)
    if mechanism not in INPUT_MECHANISMS or not records:
        raise ValueError(f"No input-bearing mechanism: {mechanism!r}")
    items = generated.complete.items
    if mechanism == "static_number":
        return generated.short_items
    if mechanism == "delayed_past_driver":
        lag = generated.params[mechanism]["lag"]
        return tuple(item.item_id for item in items if _length(item) > lag + 16)
    if records[0].where == "dynamic":
        minimum = 20 if records[0].role == "past" else 0
        return tuple(item.item_id for item in items if _length(item) >= minimum)

    attribute = records[1].name if mechanism == "hierarchy" else records[0].name
    short = set(generated.short_items)
    long_items = tuple(item for item in items if item.item_id not in short)
    candidates = []
    for item in items:
        if item.item_id not in short:
            continue
        value = item.static[attribute]
        for peer in long_items:
            if not peer.start + 16 <= item.start < peer.start + _length(peer):
                continue
            other = peer.static[attribute]
            shared = bool(set(value) & set(other)) if mechanism == "multi_label" else value == other
            if shared:
                candidates.append(item.item_id)
                break
    return tuple(candidates)


def quiet_origins(
    generated: GeneratedPanel, item_id: str, *, context: int, horizon: int
) -> tuple[int, ...]:
    """Find origins with no true context event and visible prefix and horizon events."""
    records = generated.inputs_of("event_response")
    if not records:
        raise ValueError("quiet_origins requires event_response")
    if context < 1 or horizon < 1:
        raise ValueError("context and horizon must be positive")
    item = next((item for item in generated.complete.items if item.item_id == item_id), None)
    if item is None:
        raise ValueError(f"Unknown item: {item_id!r}")
    exposed = next(item for item in generated.panel.items if item.item_id == item_id)
    column = next(r.name for r in records if r.where == "dynamic")
    events = item.dynamic[column] == 1
    prefix = np.r_[0, np.cumsum(events)]
    visible = np.r_[0, np.cumsum(exposed.dynamic[column] == 1)]
    origins = np.arange(context, len(events) - horizon + 1)
    left = origins - context
    quiet = prefix[origins] == prefix[left]
    future = visible[origins + horizon] > visible[origins]
    fitted = visible[left] > 0
    return tuple(int(o) + item.start for o in origins[quiet & future & fitted])
