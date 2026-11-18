"""P4: exact CPU continuation, durable completion, and bounded guest requests."""

import importlib
import json
import signal
import subprocess
import sys
import time
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from demand_forecast.models.v5 import V5Model, v5_loss
from demand_forecast.pretrain.v5.checkpoint import (
    CheckpointError,
    _read_checkpoint,
    latest_checkpoint,
    read_manifest,
    save_checkpoint,
)
from demand_forecast.pretrain.v5.sampler import make_batch
from tests.helpers.v5_loop import config, equal, training_equal


def api():
    return importlib.import_module("demand_forecast.pretrain.v5.loop")


def payload(result):
    return _read_checkpoint(result.checkpoint)


def unfinished(directory):
    assert not (directory / "train.json").exists()
    assert not (directory / "checkpoint-final.pt").exists()


@pytest.fixture
def full(tmp_path):
    c = config()
    result = api().train(c, tmp_path / "full")
    return c, result, payload(result)


def test_full_run_and_fixed_batch_loss_decreases(tmp_path, full):
    c, result, saved = full
    assert (result.batch, result.updates, result.skipped, result.stop_reason) == (
        12,
        12,
        0,
        "max_steps",
    )
    assert result.checkpoint.name == "checkpoint-final.pt"
    manifest = read_manifest(result.checkpoint.parent / "train.json")
    assert manifest["batch"] == 12 and manifest["identity"] == saved["identity"]
    assert list(manifest["files"]) == ["checkpoint-final.pt"]
    assert [entry["batch"] for entry in result.history] == [3, 6, 9, 12]
    assert all("seconds" not in entry for entry in result.history)
    # The first slot of this separate call holds exactly the seeded fresh model.
    api().train(c, tmp_path / "initial", stop_after=1)
    initial = min(
        (_read_checkpoint(p) for p in (tmp_path / "initial").glob("checkpoint-slot-*.pt")),
        key=lambda p: p["counters"]["batch"],
    )
    model = V5Model(c.model).eval()
    batch, _ = make_batch(c, 0)
    with torch.no_grad():
        model.load_state_dict(initial["model"])
        before = float(v5_loss(model(batch), batch, c.model)[0])
        model.load_state_dict(saved["model"])
        after = float(v5_loss(model(batch), batch, c.model)[0])
    assert after < before


@pytest.mark.parametrize("ema,workers", [(None, 0), (0.9, 0), (0.9, 2)])
def test_resume_is_bitwise_equal_including_partial_history(tmp_path, monkeypatch, ema, workers):
    loop = api()
    c = config(ema=ema, workers=workers)
    if workers:
        sampler = importlib.import_module("demand_forecast.pretrain.v5.sampler")
        real = sampler.DataLoader
        monkeypatch.setattr(sampler, "DataLoader", lambda *a, **kw: real(*a, timeout=10, **kw))
    reference = payload(loop.train(c, tmp_path / "full"))
    state = tmp_path / "resume"
    interrupted = loop.train(c, state, stop_after=5)
    assert (interrupted.batch, interrupted.stop_reason) == (5, "interrupted")
    unfinished(state)
    assert payload(interrupted)["pending"]
    # Discard history ahead of the durable slot, including a torn last line.
    with (state / "history.jsonl").open("a") as stream:
        stream.write('{"batch":999}\n{"torn":')
    resumed = loop.train(c, state)
    training_equal(reference, payload(resumed))
    assert resumed.history == reference["history"]
    assert [
        json.loads(line) for line in (state / "history.jsonl").read_text().splitlines()
    ] == resumed.history


def inject_faults(monkeypatch, faults):
    loop = api()
    real_loader, real_loss = loop.loader, loop.v5_loss
    current = [None]

    def batches(*args, **kwargs):
        iterator = real_loader(*args, **kwargs)
        try:
            for k, batch, report in iterator:
                current[0] = k
                yield k, batch, report
        finally:
            if hasattr(iterator, "close"):
                iterator.close()

    def loss(*args):
        value, parts = real_loss(*args)
        fault = faults.get(current[0])
        if fault == "loss":
            value = value * float("nan")
        if fault == "gradient":
            value.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        return value, parts

    monkeypatch.setattr(loop, "loader", batches)
    monkeypatch.setattr(loop, "v5_loss", loss)


@pytest.mark.parametrize("stop", [4, 7])
def test_nonfinite_skips_resume_schedule_and_ema(tmp_path, monkeypatch, stop):
    loop = api()
    inject_faults(monkeypatch, {3: "loss", 6: "gradient"})
    c = config(ema=0.9)
    reference = payload(loop.train(c, tmp_path / "full"))
    result = loop.train(c, tmp_path / "resume", stop_after=stop - 1)
    before = payload(result)
    result = loop.train(c, tmp_path / "resume", stop_after=1)
    after = payload(result)
    equal(before["model"], after["model"])
    equal(before["ema"], after["ema"])
    assert after["counters"]["batch"] == stop
    assert after["counters"]["updates"] == before["counters"]["updates"]
    assert after["counters"]["skipped"] == before["counters"]["skipped"] + 1
    assert after["scheduler"] != before["scheduler"]
    resumed = loop.train(c, tmp_path / "resume")
    assert (resumed.updates, resumed.skipped) == (10, 2)
    training_equal(reference, payload(resumed))


@pytest.mark.parametrize("steps", [20, 30])
def test_nonfinite_streak_is_terminal_without_completion(tmp_path, monkeypatch, steps):
    loop = api()
    inject_faults(monkeypatch, dict.fromkeys(range(20), "loss"))
    c = config(steps=steps)
    first = loop.train(c, tmp_path, stop_after=9)
    assert first.skipped == 9
    result = loop.train(c, tmp_path)
    assert (result.batch, result.updates, result.skipped, result.stop_reason) == (
        20,
        0,
        20,
        "nonfinite",
    )
    assert payload(result)["counters"]["consecutive_nonfinite"] == 20
    unfinished(tmp_path)
    slots = {path: path.read_bytes() for path in tmp_path.glob("checkpoint-slot-*.pt")}
    telemetry = (tmp_path / "telemetry.jsonl").read_bytes()
    monkeypatch.setattr(loop, "loader", lambda *a, **kw: pytest.fail("built terminal loader"))
    for _ in range(3):
        assert loop.train(c, tmp_path) == result
        assert {path: path.read_bytes() for path in tmp_path.glob("checkpoint-slot-*.pt")} == slots
        assert (tmp_path / "telemetry.jsonl").read_bytes() == telemetry
        unfinished(tmp_path)


def test_sigterm_finishes_current_update_and_checkpoints(tmp_path):
    api()
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.helpers.v5_loop", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready").exists() and process.poll() is None:
            assert time.monotonic() < deadline, "child did not enter first update"
            time.sleep(0.01)
        assert process.poll() is None
        process.send_signal(signal.SIGTERM)
        (tmp_path / "go").write_text("go")
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert json.loads((tmp_path / "result.json").read_text()) == {
        "reason": "terminated",
        "batch": 1,
    }
    assert latest_checkpoint(tmp_path / "state")[1]["counters"]["batch"] == 1
    unfinished(tmp_path / "state")


@pytest.mark.parametrize("write_seconds,answered", [(0.01, True), (7.0, False)])
def test_guest_request_acknowledges_new_slot_and_keeps_running(
    tmp_path, monkeypatch, write_seconds, answered
):
    loop = api()
    real_save, real_loss = loop.save_checkpoint, loop.v5_loss
    real_answer = loop.CheckpointRequest.answer
    acknowledgements = []
    calls = [0]

    def save(*args, **kwargs):
        path, _ = real_save(*args, **kwargs)
        return path, write_seconds

    def loss(*args):
        calls[0] += 1
        if calls[0] == 2:
            (tmp_path / "checkpoint-request").write_text("nonce-2")
        return real_loss(*args)

    def answer(self, nonce):
        acknowledgements.append(latest_checkpoint(tmp_path / "state")[1]["counters"]["batch"])
        real_answer(self, nonce)

    monkeypatch.setattr(loop, "save_checkpoint", save)
    monkeypatch.setattr(loop, "v5_loss", loss)
    monkeypatch.setattr(loop.CheckpointRequest, "answer", answer)
    result = loop.train(
        config(), tmp_path / "state", run_dir=tmp_path, staging=tmp_path / "staging"
    )
    assert result.batch == 12
    assert acknowledgements == ([2] if answered else [])
    assert (tmp_path / "checkpoint-done").exists() is answered
    assert list((tmp_path / "staging").iterdir()) == []


def test_damaged_newest_slot_replays_from_other_slot(tmp_path, full):
    c, _, reference = full
    interrupted = api().train(c, tmp_path / "resume", stop_after=5)
    interrupted.checkpoint.write_bytes(b"damaged")
    assert latest_checkpoint(tmp_path / "resume")[1]["counters"]["batch"] == 0
    resumed = api().train(c, tmp_path / "resume")
    training_equal(reference, payload(resumed))


def test_two_damaged_slots_raise(tmp_path):
    api().train(config(), tmp_path, stop_after=5)
    for path in tmp_path.glob("checkpoint-slot-*.pt"):
        path.write_bytes(b"damaged")
    with pytest.raises(CheckpointError):
        api().train(config(), tmp_path)
    unfinished(tmp_path)


@pytest.mark.parametrize(
    "damaged", [b'{"identity":', b"not JSON", b"[]", b"null", b"{}", b'{"identity": []}', b"\xff"]
)
@pytest.mark.parametrize("state_kind", ["fresh", "slot", "manifest"])
def test_unreadable_sidecar_is_rewritten(tmp_path, damaged, state_kind):
    loop = api()
    c = config(steps=3)
    previous = None
    if state_kind != "fresh":
        previous = loop.train(c, tmp_path, stop_after=None if state_kind == "manifest" else 1)
    (tmp_path / "identity.json").write_bytes(damaged)
    result = loop.train(c, tmp_path)
    assert result.stop_reason == "max_steps"
    assert json.loads((tmp_path / "identity.json").read_text()) == {
        "identity": payload(result)["identity"]
    }
    if state_kind == "manifest":
        assert result == previous


@pytest.mark.parametrize("state_kind", ["slot", "manifest"])
def test_unreadable_sidecar_does_not_hide_foreign_embedded_identity(tmp_path, state_kind):
    loop = api()
    c = config(steps=3)
    previous = loop.train(c, tmp_path, stop_after=None if state_kind == "manifest" else 1)
    foreign_identity = payload(previous)["identity"]
    (tmp_path / "identity.json").write_text('{"identity":')
    with pytest.raises(loop.IdentityError, match="identity"):
        loop.train(replace(c, seed=c.seed + 1), tmp_path)
    assert json.loads((tmp_path / "identity.json").read_text())["identity"] != foreign_identity


@pytest.mark.parametrize("order", ["batch", "saves", "damaged", "both_damaged", "foreign"])
def test_slots_are_opened_once_and_selected_by_batch_then_saves(tmp_path, monkeypatch, order):
    loop = api()
    c = config(steps=6)
    interrupted = loop.train(c, tmp_path, stop_after=1)
    older = _read_checkpoint(tmp_path / "checkpoint-slot-1.pt")
    newer = payload(interrupted)
    if order == "saves":
        older = payload(interrupted)
    # Write the selected payload to slot 0, which is visited first. In the batch
    # case it must win even though the older payload has a larger saves counter.
    older["counters"]["saves"] = 10
    newer["counters"]["saves"] = 20 if order == "saves" else 2
    if order == "foreign":
        older["identity"] = "foreign"
    save_checkpoint(tmp_path, newer, counter=0)
    save_checkpoint(tmp_path, older, counter=1)
    if order in ("damaged", "both_damaged"):
        (tmp_path / "checkpoint-slot-1.pt").write_bytes(b"damaged")
    if order == "both_damaged":
        (tmp_path / "checkpoint-slot-0.pt").write_bytes(b"damaged")
    opened = []
    real_open = Path.open

    def counted_open(path, mode="r", *args, **kwargs):
        if path.name.startswith("checkpoint-slot-") and mode == "rb":
            opened.append(path.name)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    if order in ("both_damaged", "foreign"):
        with pytest.raises(loop.IdentityError if order == "foreign" else CheckpointError):
            loop.train(c, tmp_path)
    else:
        result = loop.train(c, tmp_path, stop_after=1)
        assert (result.batch, result.updates) == (2, 2)
    assert sorted(opened) == ["checkpoint-slot-0.pt", "checkpoint-slot-1.pt"]
    if order not in ("both_damaged", "foreign"):
        assert payload(result)["counters"]["saves"] == newer["counters"]["saves"] + 2


@pytest.mark.parametrize("changed", ["code", "corpus", "env", "init", "config"])
@pytest.mark.parametrize("state_kind", ["sidecar", "slot", "manifest"])
def test_identity_mismatch_rejected_even_without_sidecar(
    tmp_path, monkeypatch, changed, state_kind
):
    loop = api()
    c = config()
    state = tmp_path / "state"
    loop.train(c, state, stop_after=None if state_kind == "manifest" else 2)
    if state_kind != "sidecar":
        (state / "identity.json").unlink()
    kwargs = {}
    if changed == "config":
        c = replace(c, seed=c.seed + 1)
    elif changed == "corpus":
        monkeypatch.setattr(loop.SourceMix, "corpus_digest", property(lambda self: "other"))
    elif changed == "init":
        init = loop.train(c, tmp_path / "init").checkpoint
        kwargs["init"] = init
    else:
        kwargs[changed] = "other"
    with pytest.raises(loop.IdentityError, match="identity"):
        loop.train(c, state, **kwargs)


def test_foreign_older_slot_is_also_rejected(tmp_path):
    loop = api()
    loop.train(config(), tmp_path, stop_after=2)
    older = min(
        tmp_path.glob("checkpoint-slot-*.pt"),
        key=lambda p: _read_checkpoint(p)["counters"]["batch"],
    )
    saved = _read_checkpoint(older)
    saved["identity"] = "foreign"
    save_checkpoint(tmp_path, saved, counter=int(older.stem[-1]))
    with pytest.raises(loop.IdentityError):
        loop.train(config(), tmp_path)


def test_slot_at_max_steps_finishes_without_loading_batches(tmp_path, monkeypatch, full):
    c, _, reference = full
    state = tmp_path / "resume"
    save_checkpoint(state, reference, counter=0)
    monkeypatch.setattr(api(), "loader", lambda *a, **kw: pytest.fail("consumed a batch"))
    result = api().train(c, state)
    assert result.stop_reason == "max_steps" and read_manifest(state / "train.json")
    training_equal(reference, payload(result))


def test_stub_scaler_finite_gradient_norm_overflow_never_steps(tmp_path, monkeypatch):
    loop = api()

    class Scaler:
        def __init__(self):
            self.updated = 0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.fill_(1e20)
                        assert torch.isfinite(parameter.grad).all()

        def step(self, optimizer):
            pytest.fail("scaler.step called on overflowed norm")

        def update(self):
            self.updated += 1

        def state_dict(self):
            return {}

    scaler = Scaler()
    monkeypatch.setattr(loop, "make_scaler", lambda *a, **kw: scaler)
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("optimizer stepped")
    )
    result = loop.train(config(ema=0.9), tmp_path, stop_after=1)
    initial = min(
        (_read_checkpoint(p) for p in tmp_path.glob("checkpoint-slot-*.pt")),
        key=lambda p: p["counters"]["batch"],
    )
    saved = payload(result)
    equal(initial["model"], saved["model"])
    equal(initial["ema"], saved["ema"])
    assert (result.updates, result.skipped, scaler.updated) == (0, 1, 1)


def test_finished_manifest_returns_before_reading_damaged_slots(tmp_path, monkeypatch, full):
    c, result, _ = full
    for slot in result.checkpoint.parent.glob("checkpoint-slot-*.pt"):
        slot.write_bytes(b"damaged")
    monkeypatch.setattr(api(), "loader", lambda *a, **kw: pytest.fail("consumed a batch"))
    repeated = api().train(c, result.checkpoint.parent)
    assert repeated == result


def test_parameter_groups_cover_once_and_exclude_special_weights():
    loop = api()
    model = V5Model(config().model)
    groups = loop.parameter_groups(model, 0.1)
    flattened = [parameter for group in groups for parameter in group["params"]]
    assert len(flattened) == len({id(p) for p in flattened}) == len(list(model.parameters()))
    no_decay = {id(p) for group in groups if group["weight_decay"] == 0 for p in group["params"]}
    special = set()
    for name, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            if (
                local_name == "bias"
                or isinstance(module, (nn.LayerNorm, nn.Embedding))
                or "logits" in name
                or local_name == "logits"
                or module is model.gate_net.net[-1]
            ):
                special.add(id(parameter))
    assert special <= no_decay
    assert id(model.patch_encoder.encoders[0][0].weight) not in no_decay


def test_package_reexports_loop_api():
    package = importlib.import_module("demand_forecast.pretrain.v5")
    for name in ("train", "TrainResult", "load_model", "parameter_groups", "IdentityError"):
        assert name in package.__all__
        assert getattr(package, name) is getattr(api(), name)


def test_init_loads_export_and_load_model_selects_weights(tmp_path):
    loop = api()
    c = config(ema=0.9)
    trained = loop.train(c, tmp_path / "source")
    saved = payload(trained)
    for weights, key in (("raw", "model"), ("ema", "ema"), ("export", "ema")):
        model, restored = loop.load_model(trained.checkpoint, weights=weights)
        assert restored == c
        equal(model.state_dict(), saved[key])
    loop.train(c, tmp_path / "dest", init=trained.checkpoint, stop_after=1)
    initial = min(
        (_read_checkpoint(p) for p in (tmp_path / "dest").glob("checkpoint-slot-*.pt")),
        key=lambda p: p["counters"]["batch"],
    )
    equal(initial["model"], saved["ema"])
    damaged = tmp_path / "bad.pt"
    damaged.write_bytes(trained.checkpoint.read_bytes()[:-1])
    with pytest.raises(CheckpointError):
        loop.load_model(damaged)


def test_uniform_router_and_ones_gates_train(tmp_path):
    result = api().train(config(router="uniform", gates="ones"), tmp_path)
    assert result.updates == 12
    assert all(
        entry["zero_gates"] == {"row": 0.0, "cand": 0.0, "family": 0.0} for entry in result.history
    )


def test_history_means_only_successful_updates(tmp_path, monkeypatch):
    loop = api()
    observations = []
    real_loss = loop.v5_loss

    def observed_loss(*args):
        loss, parts = real_loss(*args)
        observations.append((float(loss.detach()), parts))
        if len(observations) == 2:
            loss = loss * float("nan")
        return loss, parts

    monkeypatch.setattr(loop, "v5_loss", observed_loss)
    result = loop.train(config(), tmp_path)
    first = result.history[0]
    assert (first["batch"], first["updates"], first["skipped"], first["episodes"]) == (3, 2, 1, 1.0)
    assert first["loss"] == (observations[0][0] + observations[2][0]) / 2
    for name, value in first["parts"].items():
        assert value == (observations[0][1][name] + observations[2][1][name]) / 2
    assert result.history[-1]["lr"] == 0.0


def test_periodic_saves_alternate_and_manifest_is_last(tmp_path, monkeypatch):
    loop = api()
    c = config()
    c = replace(c, optim=replace(c.optim, checkpoint_minutes=1e-9))
    real_manifest = loop.write_manifest

    def manifest(path, record):
        assert not path.exists()
        saved = _read_checkpoint(path.parent / record["files"][0])
        assert saved["counters"]["batch"] == record["batch"] == 12
        real_manifest(path, record)

    monkeypatch.setattr(loop, "write_manifest", manifest)
    loop.train(c, tmp_path / "state", staging=tmp_path / "staging")
    saves = [
        json.loads(line)
        for line in (tmp_path / "state/telemetry.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "checkpoint"
    ]
    assert [entry["batch"] for entry in saves] == list(range(13))
    assert all(left["path"] != right["path"] for left, right in zip(saves, saves[1:]))
    assert all(entry["seconds"] >= 0 for entry in saves)
    assert not list((tmp_path / "staging").iterdir())


def test_guest_request_uses_longest_of_last_twenty_updates(tmp_path, monkeypatch):
    loop = api()
    real_loss, real_save = loop.v5_loss, loop.save_checkpoint
    clock, calls, answered = [0.0], [0], []

    def save(*args, **kwargs):
        path, _ = real_save(*args, **kwargs)
        return path, 0.01

    def loss(*args):
        calls[0] += 1
        if calls[0] == 1:
            clock[0] += 7.0
        if calls[0] == 2:
            (tmp_path / "checkpoint-request").write_text("after-slow-update")
        return real_loss(*args)

    real_answer = loop.CheckpointRequest.answer

    def answer(self, nonce):
        answered.append(calls[0])
        real_answer(self, nonce)

    monkeypatch.setattr(loop.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(loop, "save_checkpoint", save)
    monkeypatch.setattr(loop, "v5_loss", loss)
    monkeypatch.setattr(loop.CheckpointRequest, "answer", answer)
    result = loop.train(config(steps=24), tmp_path / "state", run_dir=tmp_path)
    assert result.batch == 24
    assert answered == [21]


def test_final_publication_failure_recovers_completed_slot_without_data(tmp_path, monkeypatch):
    loop = api()
    real_manifest = loop.write_manifest

    def fail(*args, **kwargs):
        raise OSError("injected manifest write failure")

    monkeypatch.setattr(loop, "write_manifest", fail)
    with pytest.raises(OSError, match="injected"):
        loop.train(config(), tmp_path)
    assert not (tmp_path / "train.json").exists()
    assert latest_checkpoint(tmp_path)[1]["counters"]["batch"] == 12
    monkeypatch.setattr(loop, "write_manifest", real_manifest)
    monkeypatch.setattr(loop, "loader", lambda *a, **kw: pytest.fail("consumed a batch"))
    assert loop.train(config(), tmp_path).stop_reason == "max_steps"


@pytest.mark.parametrize("raises", [False, True])
def test_release_failure_preserves_run_outcome_and_is_logged(tmp_path, monkeypatch, raises):
    loop = api()
    original = RuntimeError("original loop failure")
    handler = signal.getsignal(signal.SIGTERM)
    real_release = loop._release

    def release(iterator):
        real_release(iterator)
        raise RuntimeError("injected release failure")

    def failed_loss(*args):
        raise original

    monkeypatch.setattr(loop, "_release", release)
    if raises:
        monkeypatch.setattr(loop, "v5_loss", failed_loss)
        with pytest.raises(RuntimeError) as caught:
            loop.train(config(steps=3), tmp_path)
        assert caught.value is original
        unfinished(tmp_path)
    else:
        result = loop.train(config(steps=3), tmp_path)
        assert result.stop_reason == "max_steps"
        assert read_manifest(tmp_path / "train.json")["batch"] == result.batch == 3
    assert signal.getsignal(signal.SIGTERM) == handler
    events = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    failures = [event for event in events if event["event"] == "loader_release_failed"]
    assert len(failures) == 1
    assert "injected release failure" in failures[0]["error"]


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_loader_released_and_signal_handler_restored(tmp_path, monkeypatch, raises, workers):
    loop = api()
    real_loader = loop.loader
    references, processes = [], []
    handler = signal.getsignal(signal.SIGTERM)

    def tracked_loader(*args, **kwargs):
        iterator = real_loader(*args, **kwargs)
        references.append(weakref.ref(iterator))
        processes.extend(getattr(iterator, "_workers", []))
        return iterator

    monkeypatch.setattr(loop, "loader", tracked_loader)
    if workers:
        sampler = importlib.import_module("demand_forecast.pretrain.v5.sampler")
        real = sampler.DataLoader
        monkeypatch.setattr(sampler, "DataLoader", lambda *a, **kw: real(*a, timeout=10, **kw))
    if raises:

        def failed_loss(*args):
            raise RuntimeError("injected loss failure")

        monkeypatch.setattr(loop, "v5_loss", failed_loss)
        with pytest.raises(RuntimeError, match="injected"):
            loop.train(config(workers=workers), tmp_path)
        unfinished(tmp_path)
    else:
        loop.train(config(workers=workers), tmp_path, stop_after=1)
    assert signal.getsignal(signal.SIGTERM) == handler
    assert all(not process.is_alive() for process in processes)
    assert all(reference() is None for reference in references)
