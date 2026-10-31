"""Time warmed model.predict calls on a synthetic CPU batch from a saved bundle.

Loading, panel generation and batch preparation are excluded. Uses one PyTorch
CPU thread, five warmups, and the bundle's context, horizon and column roles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from demand_forecast.core.bundle import Bundle, load_bundle
from demand_forecast.data.frequency import to_offset
from demand_forecast.data.panel import Panel, Roles, Series
from demand_forecast.data.sampler import make_batch


def synthetic_panel(bundle: Bundle, n_series: int) -> Panel:
    """Generate observed histories and future known covariates on the bundle grid."""
    metadata = bundle.panel_metadata
    roles = Roles(
        **{
            key: value if key == "target" else tuple(value)
            for key, value in metadata["roles"].items()
        }
    )
    cfg = bundle.forecast_cfg
    # Enough history for a nonzero seasonal scale, including very short contexts.
    length = max(cfg.max_context, metadata["seasonal_period"] + 1, 2)
    start = pd.date_range(
        end=pd.Timestamp(metadata["earliest_origin"]),
        periods=length + 1,
        freq=to_offset(metadata["freq"]),
    )[0]
    rng = np.random.default_rng(0)
    series = []
    for i in range(n_series):
        target = rng.normal(10.0, 2.0, length)
        if metadata["target_support"] == "nonnegative":
            target = target.clip(min=0)
        dynamic = {
            name: rng.integers(0, 2, length + cfg.horizon).astype(float)
            for name in roles.known_dynamic
        }
        dynamic.update({name: rng.normal(size=length) for name in roles.past_dynamic})
        statics = {}
        for name in roles.static_categorical:
            values = tuple(bundle.encoder.vocabularies[name])
            statics[name] = values[i % len(values)] if values else None
        statics.update(
            {
                name: bundle.encoder.means[name] + bundle.encoder.stds[name] * rng.normal()
                for name in roles.static_numeric
            }
        )
        series.append(Series(f"latency-{i}", start, length, target, dynamic, statics))
    return Panel(
        series, roles, metadata["freq"], metadata["seasonal_period"], metadata["target_support"]
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> None:
    """Load once, prepare one batch, warm up, and report mean CPU prediction latency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Trained bundle directory")
    parser.add_argument("--n-series", type=_positive_int, default=512)
    parser.add_argument("--repeats", type=_positive_int, default=20)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    bundle = load_bundle(args.bundle, device="cpu")
    panel = synthetic_panel(bundle, args.n_series)
    cfg = bundle.forecast_cfg
    batch = make_batch(
        panel,
        [(i, series.history_length) for i, series in enumerate(panel.series)],
        cfg.horizon,
        cfg.max_context,
        bundle.encoder,
    )
    model = bundle.model.eval()
    with torch.inference_mode():
        for _ in range(5):
            predictions = model.predict(batch)
        if not torch.isfinite(predictions).all():
            raise ValueError("bundle produced nonfinite predictions during warmup")
        start = perf_counter()
        for _ in range(args.repeats):
            model.predict(batch)
        ms_per_batch = (perf_counter() - start) * 1000 / args.repeats

    print(
        f"family={bundle.family} device=cpu threads={torch.get_num_threads()} "
        f"n_series={args.n_series} context={batch.target.shape[1]} "
        f"horizon={cfg.horizon} warmups=5 repeats={args.repeats}"
    )
    print(f"{ms_per_batch:.3f} ms/batch; {ms_per_batch / args.n_series:.6f} ms/series")


if __name__ == "__main__":
    main()
