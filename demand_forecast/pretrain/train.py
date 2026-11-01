"""Fixed-budget corpus pretraining, exact continuation and portable bundles."""

from __future__ import annotations

import json
import math
import os
import random
import resource
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from demand_forecast import __version__
from demand_forecast.core.bundle import save_bundle
from demand_forecast.core.run import _finite_json, _write_run
from demand_forecast.core.training import (
    NONFINITE_STOP_AFTER,
    InterruptedRun,
    NonFiniteTraining,
    lr_factor,
    resolve_amp,
)
from demand_forecast.models.base import ModelSpec
from demand_forecast.models.losses import pinball_loss
from demand_forecast.models.registry import build_model
from demand_forecast.pretrain.config import PretrainConfig, pretrain_identity


@dataclass
class PretrainResult:
    """Selected checkpoint, per-step history and cumulative run measurements."""

    identity: str
    steps: int
    best_step: int
    best_score: float | None
    history: list[dict]
    stop_reason: str
    timings: dict


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_rss() -> int:
    # macOS reports bytes; Linux reports KiB. This is the process lifetime peak.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _cpu_state(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validate_controls(cfg: PretrainConfig) -> None:
    for name in ("max_steps", "batch_size", "eval_every", "checkpoint_every", "val_batch_size"):
        if getattr(cfg.training, name) < 1:
            raise ValueError(f"{name} must be positive")
    if cfg.training.warmup_steps < 0 or cfg.corpus.kernel_synth_every < 0:
        raise ValueError("warmup_steps and kernel_synth_every must be nonnegative")


class _PretrainLoop:
    """Session state shared by the run and the timing probe.

    Optional corpus dependencies remain behind construction. Shard time measures
    foreground wait/setup, including the first batch at a transition; subsequent
    batch construction and optimizer work are charged to training.
    """

    def __init__(self, cfg: PretrainConfig) -> None:
        from demand_forecast.pretrain.corpus import read_manifest, read_shard, shard_panel
        from demand_forecast.pretrain.sampler import CorpusSampler
        from demand_forecast.pretrain.validate import ValidationSet

        _validate_controls(cfg)
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        _seed(cfg.seed)
        self.refs = read_manifest(cfg.corpus.root / "manifest.json")
        self.shards = {
            ref.path.relative_to(cfg.corpus.root.resolve()).as_posix(): ref.sha256
            for ref in self.refs
        }
        self.identity = pretrain_identity(cfg, __version__, self.shards)
        selected = {}
        holdout = []
        for kind in ("tsmixup", "kernel_synth"):
            wanted = set(getattr(cfg.corpus, f"{kind}_shards"))
            held_out = set(getattr(cfg.corpus, f"holdout_{kind}"))
            available = {ref.index for ref in self.refs if ref.kind == kind}
            if missing := (wanted | held_out) - available:
                raise ValueError(f"missing {kind} shards in manifest: {sorted(missing)}")
            selected[kind] = [ref for ref in self.refs if ref.kind == kind and ref.index in wanted]
            holdout.extend(ref for ref in self.refs if ref.kind == kind and ref.index in held_out)
        if not selected["tsmixup"]:
            raise ValueError("pretraining requires a tsmixup shard for the model specification")
        self.sampler = CorpusSampler(
            **selected,
            horizon=cfg.forecast.horizon,
            max_context=cfg.forecast.max_context,
            min_context=cfg.forecast.min_context,
            batch_size=cfg.training.batch_size,
            seed=cfg.seed,
            kernel_synth_every=cfg.corpus.kernel_synth_every,
            holdout=holdout,
        )
        try:
            # Match the primary stream's first seeded pass, not manifest order.
            first = np.random.default_rng((cfg.seed, 0, 0)).permutation(len(selected["tsmixup"]))[0]
            self.first_ref = selected["tsmixup"][first]
            started = time.perf_counter()
            self.panel = shard_panel(read_shard(self.first_ref.path))
            initial_load_s = time.perf_counter() - started
            self.encoder = self.sampler.encoder
            spec = ModelSpec.from_panel(
                self.panel, self.encoder, cfg.forecast, cfg.model.d_model, cfg.model.cat_emb_dim
            )
            self.model = build_model("v2", spec, cfg.model.backbone_kwargs()).to(self.device)
            self.validation = ValidationSet.build(cfg, self.refs, self.encoder)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                partial(
                    lr_factor,
                    max_steps=cfg.training.max_steps,
                    warmup_steps=cfg.training.warmup_steps,
                ),
            )
            self.amp, self.amp_dtype, use_scaler = resolve_amp(cfg.training.amp, self.device)
            if hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
            self.quantiles = torch.tensor(cfg.forecast.quantiles, device=self.device)
            self.step = self.best_step = self.consecutive_nonfinite = 0
            self.best_score = None
            self.best_state = _cpu_state(self.model)
            self.history: list[dict] = []
            self.timings = dict(
                train_s=0.0,
                eval_s=0.0,
                checkpoint_s=0.0,
                shard_load_s=initial_load_s,
                peak_cuda_mem_bytes=None if self.device.type != "cuda" else 0,
                peak_rss_bytes=_peak_rss(),
                nonfinite_losses=0,
                scaler_skips=0,
            )
            self.batches = self.sampler.batches()
            self.model.train()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
        except BaseException:
            self.sampler.close()
            raise

    def next_batch(self):
        """Return a CPU batch and split foreground transition/build wall times."""
        before = self.sampler.state()
        started = time.perf_counter()
        batch = next(self.batches)
        elapsed = time.perf_counter() - started
        old = getattr(before, self.sampler.last_kind)
        new = getattr(self.sampler.state(), self.sampler.last_kind)
        transition = old.batch_cursor == 0 or (old.pass_index, old.shard_cursor) != (
            new.pass_index,
            new.shard_cursor,
        )
        load_s = elapsed if transition else 0.0
        self.timings["shard_load_s"] += load_s
        return batch, elapsed - load_s, load_s

    def train_step(self, batch, batch_s: float) -> None:
        """Mirror Trainer's finite-loss, scaling, clipping and scheduler contract."""
        _sync(self.device)
        started = time.perf_counter()
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.optimizer.param_groups[0]["lr"]
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            loss, _ = pinball_loss(
                self.model(batch),
                batch.labels,
                batch.label_mask,
                batch.loss_eligible,
                self.quantiles,
            )
        loss_value = float(loss.detach().item())
        finite = math.isfinite(loss_value)
        if finite:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
            self.scaler.step(self.optimizer)
            scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else None
            self.scaler.update()
            if scale_before is not None and self.scaler.get_scale() < scale_before:
                self.timings["scaler_skips"] += 1
        self.scheduler.step()
        self.step += 1
        self.consecutive_nonfinite = 0 if finite else self.consecutive_nonfinite + 1
        self.timings["nonfinite_losses"] += int(not finite)
        self.history.append(
            dict(step=self.step, loss=loss_value, lr=lr, kind=self.sampler.last_kind)
        )
        _sync(self.device)
        self.timings["train_s"] += batch_s + time.perf_counter() - started

    def evaluate(self) -> None:
        _sync(self.device)
        started = time.perf_counter()
        scores = self.validation.evaluate(self.model, device=self.device)
        self.history[-1].update(scores)
        score = scores["val_sql"]
        if (
            score is not None
            and math.isfinite(score)
            and (self.best_score is None or score < self.best_score)
        ):
            self.best_score, self.best_step = score, self.step
            self.best_state = _cpu_state(self.model)
        _sync(self.device)
        self.timings["eval_s"] += time.perf_counter() - started

    def sample_peak(self) -> None:
        self.timings["peak_rss_bytes"] = max(self.timings["peak_rss_bytes"], _peak_rss())
        if self.device.type == "cuda":
            self.timings["peak_cuda_mem_bytes"] = max(
                self.timings["peak_cuda_mem_bytes"], torch.cuda.max_memory_allocated(self.device)
            )

    def checkpoint(self, path: Path, stop_reason: str | None = None) -> None:
        """Atomically save live weights and continuation state, then progress."""
        _sync(self.device)
        started = time.perf_counter()
        self.sample_peak()
        state = {
            "identity": self.identity,
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "sampler_state": asdict(self.sampler.state()),
            **{
                key: getattr(self, key)
                for key in (
                    "history",
                    "best_state",
                    "best_score",
                    "best_step",
                    "consecutive_nonfinite",
                    "timings",
                )
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(state, temporary)
            self.timings["checkpoint_s"] += time.perf_counter() - started
            # As in Trainer, the footer retains this serialization's measured
            # duration across resume; footer/rename/report writes are excluded.
            with ZipFile(temporary, "a") as archive:
                root = archive.namelist()[0].split("/")[0]
                archive.writestr(f"{root}/timings.json", json.dumps(self.timings, allow_nan=False))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        _write_run(
            path.with_name("progress.json"),
            {
                "identity": self.identity,
                "step": self.step,
                "max_steps": self.cfg.training.max_steps,
                "stop_reason": stop_reason,
                "best_step": self.best_step,
                "best_score": self.best_score,
                "timings": self.timings,
                "device": str(self.device),
                "amp_dtype": str(self.amp_dtype).removeprefix("torch.") if self.amp else None,
            },
        )

    def restore(self, path: Path) -> None:
        from demand_forecast.pretrain.sampler import CorpusSamplerState, StreamState

        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["identity"] != self.identity:
            raise ValueError(
                f"pretrain identity mismatch: checkpoint {state['identity']}; requested {self.identity}"
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        sampler = state["sampler_state"]
        self.sampler.load_state(
            CorpusSamplerState(
                step=sampler["step"],
                tsmixup=StreamState(**sampler["tsmixup"]),
                kernel_synth=StreamState(**sampler["kernel_synth"]),
            )
        )
        for key in (
            "step",
            "history",
            "best_state",
            "best_score",
            "best_step",
            "consecutive_nonfinite",
            "timings",
        ):
            setattr(self, key, state[key])
        with ZipFile(path) as archive:
            root = archive.namelist()[0].split("/")[0]
            member = f"{root}/timings.json"
            if member in archive.namelist():
                self.timings.update(json.loads(archive.read(member)))
        # Construction and sampler restoration must finish before RNG restore.
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])


def pretrain(
    cfg: PretrainConfig, out_dir: Path, *, resume: bool = False, stop_after: int | None = None
) -> PretrainResult:
    """Train to the fixed budget or checkpoint and raise a terminal/session signal."""
    from demand_forecast.pretrain.corpus import PRETRAIN_START

    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    out_dir = Path(out_dir)
    checkpoint = out_dir / "checkpoints/latest.pt"
    loop = _PretrainLoop(cfg)
    try:
        if resume:
            loop.restore(checkpoint)
        session_steps = 0
        while loop.step < cfg.training.max_steps:
            if loop.consecutive_nonfinite >= NONFINITE_STOP_AFTER:
                break
            batch, batch_s, _ = loop.next_batch()
            loop.train_step(batch, batch_s)
            session_steps += 1
            if loop.consecutive_nonfinite >= NONFINITE_STOP_AFTER:
                break
            finished = loop.step == cfg.training.max_steps
            if loop.step % cfg.training.eval_every == 0 or finished:
                loop.evaluate()
            interrupted = not finished and stop_after is not None and session_steps >= stop_after
            if interrupted:
                loop.checkpoint(checkpoint, "interrupted")
                raise InterruptedRun(
                    f"Pretraining interrupted at step {loop.step}; resume from {checkpoint}"
                )
            if not finished and loop.step % cfg.training.checkpoint_every == 0:
                loop.checkpoint(checkpoint)
        if loop.consecutive_nonfinite >= NONFINITE_STOP_AFTER:
            loop.checkpoint(checkpoint, "nonfinite_losses")
            raise NonFiniteTraining(
                f"Pretraining stopped after {NONFINITE_STOP_AFTER} consecutive non-finite losses "
                f"at step {loop.step}; checkpoint {checkpoint}"
            )
        loop.checkpoint(checkpoint, "max_steps")
        loop.model.load_state_dict(loop.best_state)
        external = None
        if loop.validation.external is not None:
            record = loop.history[loop.best_step - 1] if loop.best_step else {}
            external = {
                "name": cfg.corpus.external_validation,
                "ext_sql": record.get("ext_sql"),
                "ext_sql_naive": record.get("ext_sql_naive"),
                "n_items": len(loop.validation.external[1]),
            }
        save_bundle(
            out_dir / "bundle",
            model=loop.model,
            family="v2",
            backbone_kwargs=cfg.model.backbone_kwargs(),
            forecast_cfg=cfg.forecast,
            dataset_cfg=None,
            encoder=loop.encoder,
            panel=loop.panel,
            training_cutoff=PRETRAIN_START,
            earliest_origin=PRETRAIN_START,
            manifest=_finite_json(
                {
                    "pretrain_identity": loop.identity,
                    "best_step": loop.best_step,
                    "best_score": loop.best_score,
                    "steps": loop.step,
                    "config": cfg.model_dump(mode="json"),
                    "shards": loop.shards,
                    "external_diagnostic": external,
                }
            ),
        )
        result = PretrainResult(
            loop.identity,
            loop.step,
            loop.best_step,
            loop.best_score,
            deepcopy(loop.history),
            "max_steps",
            deepcopy(loop.timings),
        )
        _write_run(out_dir / "pretrain.json", asdict(result))
        pd.DataFrame(result.history).to_csv(out_dir / "history.csv", index=False)
        return result
    finally:
        loop.sampler.close()


def pretrain_duration(cfg: PretrainConfig, probe: dict) -> dict:
    """Estimate the complete run, including publication and a 15% reserve."""
    _validate_controls(cfg)
    steps = cfg.training.max_steps
    n_evals = math.ceil(steps / cfg.training.eval_every)
    n_ckpts = math.ceil(steps / cfg.training.checkpoint_every)
    every = cfg.corpus.kernel_synth_every
    steps_ks = steps // every if every else 0
    steps_ts = steps - steps_ks
    n_shard_loads = (
        2
        + math.ceil(steps_ts * cfg.training.batch_size / 50_000)
        + math.ceil(steps_ks * cfg.training.batch_size / 50_000)
    )
    total_s = (
        probe["init_s"]
        + steps * probe["s_per_step"]
        + n_evals * probe["eval_s"]
        + n_ckpts * probe["checkpoint_s"]
        + n_shard_loads * probe["shard_load_s"]
        + 300
    ) * 1.15
    return dict(
        n_evals=n_evals,
        n_ckpts=n_ckpts,
        n_shard_loads=n_shard_loads,
        steps_ts=steps_ts,
        steps_ks=steps_ks,
        total_s=total_s,
        minutes=math.ceil(total_s / 60),
    )
