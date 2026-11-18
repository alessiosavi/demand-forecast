"""Measurements for stage Q; no qualification threshold or training choice is applied.

Only qualify.json is retained. Training states and checkpoint timing files live
in temporary directories below out, keeping guest writes inside its state tree.
Qualification temporarily observes the trainer's loader and telemetry; run it
serially in the process that owns training (as train's signal handler requires).
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import Counter
from dataclasses import replace
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from demand_forecast.core.training import resolve_amp
from demand_forecast.data.episode import (
    EpisodeBatch,
    EpisodeBuilder,
    EpisodeRequest,
    Item,
    Panel,
    collate,
)
from demand_forecast.data.episode.schema import (
    MAX_ATTRIBUTE_TOKENS,
    MAX_FEATURE_ROWS,
    MAX_FORECAST_ROWS,
    PATCH,
)
from demand_forecast.models.v5 import V5Config, V5Model, v5_loss
from demand_forecast.models.v5.model import prune
from demand_forecast.synthetic.typed.draft import stream

from . import loop
from .checkpoint import _atomic_write, _read_checkpoint, save_checkpoint
from .config import T1_MODEL, V5TrainConfig
from .corpus import CorpusIndex
from .loop import load_model, train
from .sampler import loader


@torch.no_grad()
def forecast_panel(
    model: V5Model,
    panel: Panel,
    origin: int,
    horizon: int,
    *,
    context: int = 512,
    batch_episodes: int = 64,
    gates: str = "learned",
    pruned: bool = True,
    device: str | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Forecast bounded batches on the model's device, without moving the model.

    An explicit device must match the model's device. Restore its training mode
    on success or failure.
    """
    if type(batch_episodes) is not int or batch_episodes < 1:
        raise ValueError("batch_episodes must be a positive integer")
    model_device = next(model.parameters()).device
    if device is not None:
        requested = torch.device(device)
        if requested.type == model_device.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        if requested != model_device:
            raise ValueError(
                f"requested device {requested} differs from model device {model_device}"
            )
    builder = EpisodeBuilder(panel)

    def requests():
        for index, item in enumerate(panel.items):
            for request in builder.plan_requests(
                index, (origin,), horizon, context=context, cutoff=origin
            ):
                yield item.item_id, request

    training = model.training
    try:
        model.eval()
        planned = requests()
        result = {}
        while group := list(islice(planned, batch_episodes)):
            # The comprehension releases every Episode before forwarding. Never
            # retain a previous group while building the next one's episodes.
            batch = collate([builder.build(request) for _, request in group]).to(model_device)
            selected = model.gates(batch, mode=gates)
            if pruned:
                batch, selected = prune(batch, selected)
            quantiles = model(batch, selected).quantiles[:, 0]
            quantiles = batch.loc[..., None, None] + batch.scale[..., None, None] * quantiles
            quantiles = torch.where(
                batch.fallback[..., None, None], batch.loc[..., None, None], quantiles
            )
            quantiles = quantiles[:, :, :horizon].cpu().numpy()
            for b, (item_id, _) in enumerate(group):
                kept = (batch.keep[b] & batch.forecast_mask[b]).nonzero().flatten().tolist()
                for row in kept:
                    name = batch.row_names[b][int(batch.forecast_rows[b, row])]
                    values = quantiles[b, row].copy()
                    if not np.isfinite(values).all():
                        raise ValueError(f"nonfinite forecast for {item_id!r}, {name!r}")
                    result[item_id, name] = values
            del batch, selected, quantiles
        return result
    finally:
        model.train(training)


def worst_batch(config: V5TrainConfig) -> EpisodeBatch:
    """Build the rectangular stress case, independent of sampled sources/budget.

    Use all architectural row/token slots, the largest configured context and
    horizon, and the greatest origin count that fits P3's stride. The fixed
    aligned panel supplies 16 targets, 16 known rows and 16 numeric attributes
    (each also has its value token and peers). One real episode is repeated to
    the configured batch size; no drop or stochastic screening can shrink it.
    This is deliberately allowed to exceed max_cells: Q measures that limit.
    """
    sampler = config.sampler
    context, horizon = max(sampler.contexts), max(sampler.horizon)
    origins = min(sampler.origins, 1 + (context - PATCH) // (PATCH * sampler.origin_stride))
    positions = tuple(
        context - PATCH * sampler.origin_stride * j for j in range(origins - 1, -1, -1)
    )
    rng = stream(config.seed, "qualification:worst")
    length = context + horizon
    items = tuple(
        Item(
            f"stress_{i:02d}",
            0,
            {f"target_{j:02d}": rng.normal(size=length) for j in range(MAX_FORECAST_ROWS)},
            {f"known_{j:02d}": rng.normal(size=length) for j in range(MAX_FEATURE_ROWS)},
            {f"number_{j:02d}": float(rng.normal()) for j in range(MAX_ATTRIBUTE_TOKENS)},
        )
        for i in range(32)
    )
    panel = Panel(
        "qualification-worst",
        items,
        {f"known_{j:02d}": "known" for j in range(MAX_FEATURE_ROWS)},
        "D",
        None,
        True,
    )
    episode = EpisodeBuilder(panel).build(
        EpisodeRequest(0, positions, horizon, context, "target_00", positions[0])
    )
    if sampler.designate_target:
        episode.label_mask[:, ~episode.keep] = False
    return collate([episode] * sampler.episodes_per_batch)


def _device(config):
    return torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device, function):
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    return result, time.perf_counter() - started


def _reset_gpu(device):
    if device.type == "cuda":
        _sync(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gpu(device):
    _sync(device)
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None


def _host_rss(proc: Path = Path("/proc")) -> int | None:
    """Sum RSS for this process and its descendants, including loader workers."""
    if not (proc / str(os.getpid()) / "status").is_file():
        return None
    pending, visited, total = [os.getpid()], set(), 0
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        try:
            status = (proc / str(pid) / "status").read_text()
            total += next(
                int(line.split()[1]) * 1024
                for line in status.splitlines()
                if line.startswith("VmRSS:")
            )
            # A child may have been created by any thread of the process.
            for task in (proc / str(pid) / "task").iterdir():
                pending.extend(int(child) for child in (task / "children").read_text().split())
        except (OSError, StopIteration):
            # Exiting workers can disappear between the metadata reads.
            continue
    return total


class _HostPeak:
    """Sample summed RSS every 10 ms and at every observed batch boundary."""

    def __init__(self):
        self.peak = None
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        value = _host_rss()
        if value is not None:
            self.peak = value if self.peak is None else max(value, self.peak)

    def __enter__(self):
        self.sample()
        if self.peak is not None:

            def poll():
                while not self._stop.wait(0.01):
                    self.sample()

            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def _trial(config, directory, device):
    """Observe real train() batches, synchronizing before/after each GPU update.

    The two temporary observers preserve the trainer's artifacts and release its
    actual iterator on success or exception. log_every=1 supplies successful vs
    skipped update counts; no loss/optimiser logic is duplicated here.
    The per-batch clock excludes periodic checkpoint writes, measured separately
    in qualification section (e).
    """
    reports, durations = [], []
    real_loader, real_line = loop.loader, loop._json_line
    started = None

    def observed_loader(*args, **kwargs):
        nonlocal started
        iterator = real_loader(*args, **kwargs)
        try:
            while True:
                _sync(device)
                started = time.perf_counter()
                value = next(iterator)
                reports.append(value[2])
                yield value
        finally:
            loop._release(iterator)

    def observed_line(path, record):
        if record.get("event") == "batch":
            _sync(device)
            durations.append(time.perf_counter() - started)
            memory.sample()
        real_line(path, record)

    _reset_gpu(device)
    with _HostPeak() as memory:
        with (
            patch.object(loop, "loader", observed_loader),
            patch.object(loop, "_json_line", observed_line),
        ):
            result, elapsed = _timed(device, lambda: train(config, directory))
    if result.batch != config.optim.max_steps or result.stop_reason != "max_steps":
        raise RuntimeError(f"qualification training stopped with {result.stop_reason}")
    if not (len(reports) == len(durations) == len(result.history) == result.batch):
        raise RuntimeError("qualification did not observe every training batch")
    tail_start = len(durations) // 5
    tail = durations[tail_start:]
    updated = [entry["updates"] for entry in result.history]
    successful = [i for i in range(tail_start, len(durations)) if updated[i]]
    skips = Counter()
    for report in reports:
        skips.update(report["skips"])
    if result.skipped:
        skips["nonfinite"] += result.skipped
    metrics = {
        "batches": result.batch,
        "updates": result.updates,
        "tail_batches": len(tail),
        "batches_per_second": statistics.median(1 / seconds for seconds in tail),
        "updates_per_second": sum(updated[tail_start:]) / sum(tail),
        "supervised_cells_per_batch": statistics.mean(report["supervised"] for report in reports),
        "padded_cells_per_batch": statistics.mean(report["cells"] for report in reports),
        "supervised_cells_per_update": statistics.mean(reports[i]["supervised"] for i in successful)
        if successful
        else None,
        "seconds_per_update": statistics.median(durations[i] for i in successful)
        if successful
        else None,
        "skips": dict(skips),
        "host_memory_bytes": memory.peak,
        "peak_gpu_bytes": _peak_gpu(device),
        "total_seconds": elapsed,
    }
    metrics["updates_per_hour"] = 3600 * metrics["updates_per_second"]
    return metrics, result.checkpoint


def _corpus_measure(config, device):
    sources = [source for source in config.sources if source.kind == "corpus"]
    if not sources:
        return None, None
    roots = sorted({source.root for source in sources if source.root is not None})
    if len(roots) != len({source.root for source in sources}):
        raise ValueError("qualification corpus sources require source.root")
    indices, seconds = _timed(device, lambda: [CorpusIndex.build(Path(root)) for root in roots])
    admitted, excluded = sum(index.rows for index in indices), 0
    # excluded stores (relative file, row-group id, reason), not row counts.
    # Use footer metadata; never decode an excluded group's numeric values.
    import pyarrow.parquet as pq

    for index in indices:
        for relative, group, _ in index.excluded:
            with pq.ParquetFile(index.root / relative) as parquet:
                excluded += parquet.metadata.row_group(group).num_rows
    total = admitted + excluded
    return seconds, excluded / total if total else None


def _checkpoint_measure(path, directory, device):
    saved = _read_checkpoint(path)
    if not saved["optimizer"]["state"] or saved["ema"] is None:
        raise RuntimeError("checkpoint qualification requires populated optimiser and EMA state")

    def placed(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: placed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [placed(item) for item in value]
        if isinstance(value, tuple):
            return tuple(placed(item) for item in value)
        return value

    # Loading maps tensors to CPU. Restore training-device state before timing
    # so torch.save includes the device-to-host copies of a real GPU checkpoint.
    saved = placed(saved)
    durations, size = [], 0
    for counter in range(5):
        (written, _), seconds = _timed(
            device, lambda: save_checkpoint(directory, saved, counter=counter)
        )
        durations.append(seconds)
        size = written.stat().st_size
    return {"seconds": statistics.median(durations), "bytes": size, "writes": 5}


def _worst_measure(config, device):
    """Time forward and backward only, without optimiser state or EMA.

    The batch-shape decision uses the training trial's peak, which includes
    optimiser and EMA state, rather than this forward/backward peak.
    """
    _reset_gpu(device)
    model = V5Model(config.model).to(device).train()
    batch = worst_batch(config).to(device)
    amp, dtype, use_scaler = resolve_amp(config.optim.amp, device)
    scaler = loop.make_scaler(use_scaler)

    def forward_loss():
        with torch.autocast(device.type, dtype=dtype, enabled=amp):
            output = model(batch, model.gates(batch, mode="ones"), router=config.router)
            return v5_loss(output, batch, config.model)[0]

    loss, forward_seconds = _timed(device, forward_loss)
    _, backward_seconds = _timed(
        device, lambda: (loss if scaler is None else scaler.scale(loss)).backward()
    )
    return {
        "forward_backward_seconds": forward_seconds + backward_seconds,
        "forward_backward_peak_gpu_bytes": _peak_gpu(device),
    }


def _inference_panel(*, seed: int, items: int, length: int) -> Panel:
    """Build aligned daily series with a weekly target, category and known flag."""
    rng = stream(seed, "qualification:inference")
    steps = np.arange(length)
    weekly = 2 * np.sin(2 * np.pi * steps / 7)
    flag = (steps % 7 == 0).astype(np.float64)
    return Panel(
        "qualification-inference",
        tuple(
            Item(
                f"item_{i:05d}",
                0,
                {"target": 10 + i + weekly + rng.normal(0, 0.1, size=length)},
                {"flag": flag},
                {"category": f"group_{i % 10}"},
            )
            for i in range(items)
        ),
        {"flag": "known"},
        "D",
        pd.Timestamp("2000-01-01"),
        True,
    )


def qualify(
    config: V5TrainConfig,
    out: Path,
    *,
    models: dict[str, V5Config] | None = None,
    updates: int = 200,
    workers: tuple[int, ...] = (8, 16, 32, 46),
    panel_items: int = 30_490,
    panel_length: int = 1_941,
) -> dict:
    """Measure Q (a) through (g); retain only a finite/null JSON report.

    Discard floor(20%) of batch timings. Batch rate is a median; update rate is
    tail updates divided by total tail seconds, including skipped batches.
    seconds_per_update is the median over successful tail updates. Batch timings
    include loader, transfer, forward, backward, optimiser and history logging;
    total_seconds also includes startup and checkpoint writes. CPU GPU-memory
    fields and host-memory fields without /proc are null, never estimates.
    """
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer")
    if not workers or any(type(worker) is not int or worker < 0 for worker in workers):
        raise ValueError("workers requires nonnegative integer counts")
    if (
        type(panel_items) is not int
        or panel_items < 1
        or type(panel_length) is not int
        or panel_length < 29
    ):
        raise ValueError("panel_items must be positive and panel_length must exceed 28")
    models = {"full": config.model, "t1": T1_MODEL} if models is None else dict(models)
    if not models or any(not isinstance(name, str) or not name for name in models):
        raise ValueError("models requires nonempty string names")
    device = _device(config)
    # Q always measures the bank and all inputs, even if passed a T1 null arm.
    base = replace(
        config,
        gates="ones",
        sampler=replace(config.sampler, drop=(), drop_always=(), summary_off=()),
        optim=replace(
            config.optim, max_steps=updates, log_every=1, ema_decay=config.optim.ema_decay or 0.999
        ),
    )
    index_seconds, excluded_share = _corpus_measure(base, device)
    first_iterator = None

    def first_batch():
        nonlocal first_iterator
        first_iterator = loader(replace(base, workers=workers[0]), 0)
        next(first_iterator)

    try:
        _, first_seconds = _timed(device, first_batch)
    finally:
        if first_iterator is not None:
            loop._release(first_iterator)
        first_iterator = None
    cold = {
        "corpus_index_seconds": index_seconds,
        "first_batch_seconds": first_seconds,
        "restore_seconds": {},
        "excluded_row_share": excluded_share,
    }
    report = {"cold_start": cold, "models": {}}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".qualify-", dir=out) as temporary:
        scratch = Path(temporary)
        for name, architecture in models.items():
            model_config = replace(base, model=architecture)
            measured = {"workers": {}}
            for worker in workers:
                trial_config = replace(model_config, workers=worker)
                with TemporaryDirectory(dir=scratch) as trial:
                    metrics, checkpoint = _trial(trial_config, Path(trial) / "state", device)
                    measured["workers"][str(worker)] = metrics
                    if worker == workers[0]:
                        measured["checkpoint"] = _checkpoint_measure(
                            checkpoint, Path(trial) / "write", device
                        )

                        def restore():
                            model, _ = load_model(checkpoint)
                            return model.to(device)

                        restored, seconds = _timed(device, restore)
                        cold["restore_seconds"][name] = seconds
                        del restored
            configured_origins = str(model_config.sampler.origins)
            measured["origins"] = {configured_origins: measured["workers"][str(workers[0])].copy()}
            if configured_origins != "1":
                single = replace(
                    model_config,
                    workers=workers[0],
                    sampler=replace(model_config.sampler, origins=1),
                )
                with TemporaryDirectory(dir=scratch) as trial:
                    measured["origins"]["1"], _ = _trial(single, Path(trial) / "state", device)
            measured["worst_batch"] = _worst_measure(model_config, device)
            panel = _inference_panel(seed=config.seed, items=panel_items, length=panel_length)
            origin = panel_length - 28
            fallback_forecasts = 0
            # Match EpisodeBuilder's fallback rule on each target's finite prefix.
            for item in panel.items:
                for values in item.targets.values():
                    observed = values[: max(0, origin - item.start)]
                    observed = observed[np.isfinite(observed)]
                    fallback_forecasts += int(not observed.size or np.all(observed == observed[0]))
            model = V5Model(architecture).to(device)
            _reset_gpu(device)
            with _HostPeak() as memory:
                forecasts, seconds = _timed(
                    device,
                    lambda model=model, panel=panel: forecast_panel(
                        model,
                        panel,
                        origin,
                        28,
                        context=512,
                        gates="ones",
                        pruned=False,
                    ),
                )
                if len(forecasts) != panel_items or fallback_forecasts:
                    raise RuntimeError(
                        f"qualification inference requires {panel_items} forecasts and zero "
                        f"fallbacks; got {len(forecasts)} forecasts and {fallback_forecasts} fallbacks"
                    )
                measured["inference"] = {
                    "seconds": seconds,
                    "peak_gpu_bytes": _peak_gpu(device),
                    "origin": origin,
                    "items": len(panel.items),
                    "forecasts": len(forecasts),
                    "fallback_forecasts": fallback_forecasts,
                }
            measured["inference"]["host_memory_bytes"] = memory.peak
            parts = Counter()
            for parameter_name, parameter in model.named_parameters():
                parts[parameter_name.split(".", 1)[0]] += parameter.numel()
            measured["parameters"] = {"total": sum(parts.values()), "by_part": dict(parts)}
            report["models"][name] = measured
            del model, panel, forecasts
    encoded = json.dumps(report, allow_nan=False, sort_keys=True).encode()
    _atomic_write(out / "qualify.json", lambda output: output.write(encoded))
    return report
