#!/usr/bin/env python3
"""Seconds-only synthetic batch profile/benchmark; never loads fev data or trains.

The profile calls the frozen reference unchanged. A temporary Tensor.__setitem__
shim exposes assignment time to cProfile (implicit special-method calls
otherwise sit inside make_batch's self time). A no-op control estimates shim
overhead, subtracted from assignment time but retained in the denominator.
"""

import argparse
import cProfile
import pstats
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_forecast.data.encoders import StaticEncoder  # noqa: E402
from demand_forecast.data.panel import Panel, Roles, Series  # noqa: E402
from demand_forecast.data.sampler import build_origins  # noqa: E402
from tests.helpers.reference_batch import reference_make_batch  # noqa: E402

HORIZON, MAX_CONTEXT, BATCH_SIZE, N_BATCHES = 13, 512, 128, 20


def synthetic_case():
    """2,000 × 277 weekly observations, eight knowns and five categorical statics."""
    rng = np.random.default_rng(0)
    length = 277
    known = tuple(f"known_{i}" for i in range(8))
    cats = tuple(f"cat_{i}" for i in range(5))
    roles = Roles("target", known, (), cats, ())
    start = pd.Timestamp("2011-01-02")
    series = [
        Series(
            str(i),
            start,
            length,
            rng.uniform(0, 100, length),
            {name: rng.normal(size=length + HORIZON) for name in known},
            {name: str(i % (j + 2)) for j, name in enumerate(cats)},
        )
        for i in range(2000)
    ]
    panel = Panel(series, roles, "W-SUN", 1, "nonnegative")
    encoder = StaticEncoder().fit(panel, [s.series_id for s in series])
    cutoff = panel.timestamps(0)[length - HORIZON - 1]
    origins = build_origins(panel, HORIZON, 1, upto=cutoff)
    draws = np.random.default_rng(0).integers(len(origins), size=(N_BATCHES, BATCH_SIZE))
    return panel, encoder, cutoff, origins[draws]


def profile_reference(panel, encoder, cutoff, batches):
    native_setitem = torch.Tensor.__setitem__

    def profiled_setitem(tensor, key, value):
        native_setitem(tensor, key, value)

    profiler = cProfile.Profile()
    with patch.object(torch.Tensor, "__setitem__", profiled_setitem):
        profiler.enable()
        for items in batches:
            reference_make_batch(panel, items, HORIZON, MAX_CONTEXT, encoder, cutoff)
        profiler.disable()
    stats = pstats.Stats(profiler)

    def cumulative(name):
        return sum(value[3] for key, value in stats.stats.items() if key[2] == name)

    total = cumulative("reference_make_batch")
    assignment = cumulative("profiled_setitem")
    calls = sum(value[1] for key, value in stats.stats.items() if key[2] == "profiled_setitem")

    # CPython does not profile the native slot even when called explicitly.
    # Make the same shim calls with a Python no-op in its place as a control.
    def no_op(tensor, key, value):
        pass

    native_setitem = no_op
    tensor = torch.zeros(1)
    control = cProfile.Profile()
    control.enable()
    for _ in range(calls):
        profiled_setitem(tensor, 0, 0)
    control.disable()
    overhead = sum(
        value[3]
        for key, value in pstats.Stats(control).stats.items()
        if key[2] == "profiled_setitem"
    )
    assignment = max(assignment - overhead, 0.0)
    components = {
        "grid_range + calendar_features": cumulative("grid_range")
        + cumulative("calendar_features"),
        "encoder.transform": cumulative("transform"),
        "per-row torch assignment": assignment,
        "sample_frame": cumulative("sample_frame"),
        "_dynamic_values": cumulative("reference_dynamic_values"),
    }
    print(f"Reference cProfile: {total:.6f} s, {len(batches)} batches of {BATCH_SIZE}")
    print("Component                              cumulative_s   share")
    for name, duration in components.items():
        print(f"{name:39} {duration:11.6f} {duration / total:7.2%}")
    cacheable = sum(list(components.values())[:3]) / total
    print(
        f"Cacheable + assignment: {cacheable:.2%}; gate: {'GO' if cacheable >= 0.60 else 'STOP'} (60%)"
    )
    print("Assignment includes retained loc/scale stores; estimated via shim minus no-op control.")
    print(f"Subtracted shim control: {overhead:.6f} s across {calls} assignments")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        f"Python {sys.version.split()[0]}, torch {torch.__version__}, numpy {np.__version__}, pandas {pd.__version__}; torch threads=2"
    )
    panel, encoder, cutoff, batches = synthetic_case()
    reference_make_batch(panel, batches[0], HORIZON, MAX_CONTEXT, encoder, cutoff)
    if args.profile:
        profile_reference(panel, encoder, cutoff, batches)
        return

    from demand_forecast.data.sampler import BatchBuilder

    builder = BatchBuilder(panel, HORIZON, MAX_CONTEXT, encoder)
    # Cold builder: the measured loop includes lazy cache construction.
    timings = []
    for build in (
        lambda items: reference_make_batch(panel, items, HORIZON, MAX_CONTEXT, encoder, cutoff),
        lambda items: builder.build(items, cutoff),
    ):
        started = perf_counter()
        for items in batches:
            build(items)
        timings.append((perf_counter() - started) * 1000 / len(batches))
    print(f"Reference: {timings[0]:.3f} ms/batch; builder: {timings[1]:.3f} ms/batch")
    print(f"Reference / builder: {timings[0] / timings[1]:.3f}x (20 × 128; cold builder)")


if __name__ == "__main__":
    main()
