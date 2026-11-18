"""Index-seeded v5 training with durable checkpoints and exact CPU continuation."""

from __future__ import annotations

import json
import math
import shutil
import signal
import time
import zlib
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from demand_forecast.core.training import NONFINITE_STOP_AFTER, lr_factor, resolve_amp
from demand_forecast.models.v5 import V5Config, V5Model, v5_loss

from .checkpoint import (
    CheckpointError,
    CheckpointRequest,
    _atomic_write,
    _counters,
    _read_checkpoint,
    file_digest,
    read_manifest,
    read_payload,
    save_checkpoint,
    write_manifest,
)
from .config import (
    OptimConfig,
    SamplerConfig,
    SourceConfig,
    V5TrainConfig,
    _checked_mapping,
    _plain,
    _record,
    identity,
)
from .sampler import loader
from .sources import SourceMix


class IdentityError(RuntimeError):
    """Existing state belongs to a different training identity."""


@dataclass
class TrainResult:
    batch: int
    updates: int
    skipped: int
    stop_reason: str
    checkpoint: Path
    history: list[dict]


class _PretrainEMA:
    """Device-local shadow copied from pretrain.train, without importing its loop."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def restore(self, state, device):
        self.shadow = {key: value.to(device).clone() for key, value in state.items()}


def parameter_groups(model: V5Model, weight_decay: float) -> list[dict]:
    """Decay matrices except embeddings, normalisation and gate-logit projections."""
    excluded = set()
    gate_layers = (model.family_logits, model.gate_net.net[-1])
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if (
                name == "bias"
                or name == "logits"
                or isinstance(module, (nn.LayerNorm, nn.Embedding))
                or any(module is layer for layer in gate_layers)
            ):
                excluded.add(id(parameter))
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if id(parameter) in excluded else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_scaler(enabled: bool):
    """Factory kept separate so CPU tests can exercise the fp16 skip protocol."""
    return torch.amp.GradScaler("cuda", enabled=True) if enabled else None


def _seed(seed: int, name: str, index: int | None = None) -> None:
    entropy = [seed, zlib.crc32(name.encode())]
    if index is not None:
        entropy.append(index)
    # SeedSequence never touches either NumPy's or torch's default data generator.
    value = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
    torch.manual_seed(value)


def _configuration(data: dict) -> V5TrainConfig:
    data = _checked_mapping(V5TrainConfig, data)
    for name, cls in (("model", V5Config), ("sampler", SamplerConfig), ("optim", OptimConfig)):
        data[name] = _record(cls, data[name])
    data["sources"] = tuple(_record(SourceConfig, source) for source in data["sources"])
    return V5TrainConfig(**data)


def load_model(path: Path, *, weights: str = "export") -> tuple[V5Model, V5TrainConfig]:
    """Validate the footer and load raw, EMA, or the checkpoint's chosen export."""
    saved = read_payload(Path(path))
    key = saved["export"] if weights == "export" else {"raw": "model", "ema": "ema"}.get(weights)
    if key not in ("model", "ema") or saved[key] is None:
        raise ValueError(f"unavailable checkpoint weights: {weights}")
    config = _configuration(saved["config"])
    model = V5Model(config.model)
    model.load_state_dict(saved[key])
    return model, config


def _pending() -> dict:
    return {
        "loss": 0.0,
        "parts": {},
        "supervised": 0,
        "episodes": 0,
        "batches": 0,
        "updates": 0,
        "skipped": 0,
    }


def _json_line(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def _rewrite_history(path: Path, history: list[dict]) -> None:
    encoded = "".join(
        json.dumps(entry, allow_nan=False, sort_keys=True) + "\n" for entry in history
    ).encode()
    _atomic_write(path, lambda stream: stream.write(encoded))


@torch.no_grad()
def _zero_gates(model, batch, mode) -> dict:
    model.eval()
    try:
        gates = model.gates(batch, mode=mode)
        selected = {
            "row": gates.row[batch.row_mask],
            "cand": gates.cand[batch.cand_mask],
            "family": gates.family.flatten(),
        }
        return {
            name: float((values == 0).float().mean()) if values.numel() else 0.0
            for name, values in selected.items()
        }
    finally:
        model.train()


def _release(iterator) -> None:
    # P3 returns a DataLoader iterator, whose public API has no close(). Its
    # persistent processes must also stop while an exception retains our frame.
    close = getattr(iterator, "_shutdown_workers", None) or getattr(iterator, "close", None)
    if close is not None:
        close()


def train(
    config: V5TrainConfig,
    state_dir: Path,
    *,
    code: str = "dev",
    env: str | None = None,
    run_dir: Path | None = None,
    staging: Path | None = None,
    corpus_root: Path | None = None,
    stop_after: int | None = None,
    init: Path | None = None,
) -> TrainResult:
    """Consume at most max_steps batches; only a completed run publishes train.json.

    stop_after counts batches in this invocation. Training counters and history
    are deterministic; saves counts physical writes, including interruptions.
    A terminal nonfinite streak wins even at max_steps; retries return the saved
    outcome without writing another checkpoint or constructing the loader.
    """
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive integer")
    state_dir = Path(state_dir)
    staging = None if staging is None else Path(staging)
    requested = identity(
        config,
        code=code,
        corpus=SourceMix(config.sources, corpus_root=corpus_root).corpus_digest,
        init=None if init is None else file_digest(Path(init)),
        env=env,
    )

    def check_identity(value):
        if value != requested:
            raise IdentityError(f"training identity mismatch: requested {requested}, found {value}")

    # (1) Establish identity before consulting any completion or checkpoint.
    sidecar = state_dir / "identity.json"
    try:
        sidecar_identity = json.loads(sidecar.read_text(encoding="utf-8"))["identity"]
        if not isinstance(sidecar_identity, str):
            raise ValueError("sidecar identity must be a string")
    except (OSError, ValueError, KeyError, TypeError):
        encoded = json.dumps({"identity": requested}).encode()
        _atomic_write(sidecar, lambda stream: stream.write(encoded))
    else:
        check_identity(sidecar_identity)

    def result(counters, reason, checkpoint, history):
        return TrainResult(
            counters["batch"], counters["updates"], counters["skipped"], reason, checkpoint, history
        )

    # (2) Completed work wins over even damaged slots. Validate the payload too.
    manifest = read_manifest(state_dir / "train.json")
    if manifest is not None:
        check_identity(manifest.get("identity"))
        final = state_dir / "checkpoint-final.pt"
        saved = _read_checkpoint(final)
        check_identity(saved.get("identity"))
        if saved["counters"]["batch"] != config.optim.max_steps:
            raise CheckpointError("finished manifest has an unfinished batch counter")
        return result(saved["counters"], "max_steps", final, saved["history"])

    # (3) Every valid slot read must belong to this identity, including the older one.
    latest, errors = None, []
    for slot_number in (0, 1):
        slot = state_dir / f"checkpoint-slot-{slot_number}.pt"
        if not slot.exists():
            continue
        try:
            candidate = _read_checkpoint(slot)
        except CheckpointError as exc:
            errors.append(str(exc))
            continue
        check_identity(candidate.get("identity"))
        if latest is None or _counters(candidate) > _counters(latest[1]):
            latest = slot, candidate
        del candidate
    if latest is None and errors:
        raise CheckpointError("; ".join(errors))
    saved = None if latest is None else latest[1]
    if saved is not None:
        if not 0 <= saved["counters"]["batch"] <= config.optim.max_steps:
            raise CheckpointError("checkpoint batch counter outside training budget")
        if saved["counters"]["consecutive_nonfinite"] >= NONFINITE_STOP_AFTER:
            return result(saved["counters"], "nonfinite", latest[0], saved["history"])
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )
    _seed(config.seed, "init")
    model = V5Model(config.model).to(device)
    if saved is not None:
        model.load_state_dict(saved["model"])
    elif init is not None:
        initial, _ = load_model(init)
        model.load_state_dict(initial.state_dict())
        del initial
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.optim.weight_decay), lr=config.optim.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(
            step, max_steps=config.optim.max_steps, warmup_steps=config.optim.warmup_steps
        ),
    )
    amp, dtype, use_scaler = resolve_amp(config.optim.amp, device)
    scaler = make_scaler(use_scaler)
    ema = None if config.optim.ema_decay is None else _PretrainEMA(model, config.optim.ema_decay)
    counters = {"batch": 0, "updates": 0, "skipped": 0, "consecutive_nonfinite": 0, "saves": 0}
    history, pending = [], _pending()
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if scaler is not None and saved["scaler"] is not None:
            scaler.load_state_dict(saved["scaler"])
        if ema is not None:
            ema.restore(saved["ema"], device)
        counters, history, pending = saved["counters"], saved["history"], saved["pending"]
    _rewrite_history(state_dir / "history.jsonl", history)
    del saved, latest

    def checkpoint_payload():
        return {
            "identity": requested,
            "config": _plain(asdict(config)),
            "model": model.state_dict(),
            "ema": None if ema is None else ema.shadow,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "counters": counters,
            "history": history,
            "pending": pending,
            "export": "model" if ema is None else "ema",
        }

    last_write, last_save = 0.0, time.perf_counter()

    def save():
        nonlocal last_write, last_save
        counters["saves"] += 1
        path, last_write = save_checkpoint(
            state_dir, checkpoint_payload(), counter=counters["saves"], staging=staging
        )
        last_save = time.perf_counter()
        _json_line(
            state_dir / "telemetry.jsonl",
            {
                "event": "checkpoint",
                "batch": counters["batch"],
                "path": path.name,
                "seconds": last_write,
                "time": time.time(),
            },
        )
        return path

    def finish():
        slot = save()
        final = state_dir / "checkpoint-final.pt"

        def copy(stream):
            with slot.open("rb") as source:
                shutil.copyfileobj(source, stream)

        _atomic_write(final, copy, staging=staging)
        write_manifest(
            state_dir / "train.json",
            {
                "files": [final.name],
                "batch": counters["batch"],
                "updates": counters["updates"],
                "skipped": counters["skipped"],
                "identity": requested,
            },
        )
        return result(counters, "max_steps", final, history)

    # (4) Recover a completed cursor without ever constructing the infinite loader.
    if counters["batch"] == config.optim.max_steps:
        return finish()

    terminated = False

    def terminate(signum, frame):
        nonlocal terminated
        terminated = True

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    iterator = None
    try:
        # (5) Measure a save before the first update of every active invocation.
        checkpoint = save()
        request = CheckpointRequest(run_dir)
        durations = deque(maxlen=20)
        call_batches = 0
        iterator = loader(config, counters["batch"], corpus_root=corpus_root)
        model.train()
        while counters["batch"] < config.optim.max_steps:
            started = time.perf_counter()
            k, batch, report = next(iterator)
            if k != counters["batch"]:
                raise RuntimeError(f"loader returned batch {k}, expected {counters['batch']}")
            batch = batch.to(device)
            _seed(config.seed, "update", k)
            with torch.autocast(device.type, dtype=dtype, enabled=amp):
                gates = model.gates(batch, mode=config.gates)
                output = model(batch, gates, router=config.router)
                loss, parts = v5_loss(output, batch, config.model)
            value = float(loss.detach())
            updated = False
            if math.isfinite(value):
                optimizer.zero_grad(set_to_none=True)
                (loss if scaler is None else scaler.scale(loss)).backward()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                if bool(torch.isfinite(norm)):
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                    if ema is not None:
                        ema.update(model)
                    updated = True
                else:
                    optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
            else:
                optimizer.zero_grad(set_to_none=True)
            if updated:
                counters["updates"] += 1
                counters["consecutive_nonfinite"] = 0
                pending["loss"] += value
                for name, part in parts.items():
                    pending["parts"][name] = pending["parts"].get(name, 0.0) + float(part)
            else:
                counters["skipped"] += 1
                counters["consecutive_nonfinite"] += 1
            pending["updates" if updated else "skipped"] += 1
            pending["supervised"] += int(report["supervised"])
            pending["episodes"] += int(report["episodes"])
            pending["batches"] += 1
            counters["batch"] = k + 1
            # Batches, including skipped ones, advance the schedule.
            scheduler.step()
            call_batches += 1
            if counters["batch"] % config.optim.log_every == 0:
                count = pending["updates"]
                entry = {
                    "batch": counters["batch"],
                    "loss": pending["loss"] / count if count else None,
                    "parts": {name: total / count for name, total in pending["parts"].items()},
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "supervised": pending["supervised"] / pending["batches"],
                    "episodes": pending["episodes"] / pending["batches"],
                    "updates": count,
                    "skipped": pending["skipped"],
                    "zero_gates": _zero_gates(model, batch, config.gates),
                }
                history.append(entry)
                _json_line(state_dir / "history.jsonl", entry)
                pending = _pending()
            durations.append(time.perf_counter() - started)
            _json_line(
                state_dir / "telemetry.jsonl",
                {"event": "batch", "batch": counters["batch"], "seconds": durations[-1]},
            )
            nonce = request.pending()
            if nonce is not None and last_write + max(durations) <= 6.0:
                checkpoint = save()
                request.answer(nonce)
            if counters["consecutive_nonfinite"] >= NONFINITE_STOP_AFTER:
                return result(counters, "nonfinite", save(), history)
            if counters["batch"] == config.optim.max_steps:
                return finish()
            if terminated or (stop_after is not None and call_batches >= stop_after):
                return result(
                    counters, "terminated" if terminated else "interrupted", save(), history
                )
            if time.perf_counter() - last_save >= config.optim.checkpoint_minutes * 60:
                checkpoint = save()
        raise RuntimeError(f"training exited without an outcome; last checkpoint {checkpoint}")
    finally:
        try:
            if iterator is not None:
                _release(iterator)
        except Exception as exc:
            try:
                _json_line(
                    state_dir / "telemetry.jsonl",
                    {"event": "loader_release_failed", "error": str(exc)},
                )
            except Exception:
                # Even a telemetry failure must preserve the run's outcome.
                pass
        finally:
            iterator = None
            signal.signal(signal.SIGTERM, previous_handler)
