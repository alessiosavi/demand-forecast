"""Durable slot selection, integrity, and the guest nonce protocol."""

import hashlib
import importlib
import io
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import torch


def checkpoint_module():
    return importlib.import_module("demand_forecast.pretrain.v5.checkpoint")


def payload(batch=1, saves=1):
    return {"counters": {"batch": batch, "saves": saves}, "model": {"weight": torch.arange(3)}}


def test_read_payload_exported_from_package():
    from demand_forecast.pretrain.v5 import read_payload

    assert read_payload is checkpoint_module().read_payload


def test_read_payload_keys_and_damaged_footer(tmp_path):
    api = checkpoint_module()
    original = payload()
    path, _ = api.save_checkpoint(tmp_path, original, counter=0)
    restored = api.read_payload(path)
    assert restored.keys() == original.keys()
    assert restored["counters"] == original["counters"]
    assert torch.equal(restored["model"]["weight"], original["model"]["weight"])

    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(api.CheckpointError, match="SHA-256 footer mismatch"):
        api.read_payload(path)


def test_private_reader_name_still_works(tmp_path):
    api = checkpoint_module()
    original = {"counters": {"batch": 0, "saves": 0}}
    path, _ = api.save_checkpoint(tmp_path, original, counter=0)
    assert api._read_checkpoint(path) == original
    assert api._read_checkpoint is api.read_payload


def test_slots_alternate_and_round_trip_tensors(tmp_path):
    api = checkpoint_module()
    for counter in range(4):
        path, seconds = api.save_checkpoint(tmp_path, payload(counter, counter), counter=counter)
        assert path == tmp_path / f"checkpoint-slot-{counter % 2}.pt"
        assert seconds >= 0
        selected, restored = api.latest_checkpoint(tmp_path)
        assert selected == path and restored["counters"] == {"batch": counter, "saves": counter}
        assert torch.equal(restored["model"]["weight"], torch.arange(3))
        assert restored["model"]["weight"].device.type == "cpu"
    assert {p.name for p in tmp_path.iterdir()} == {"checkpoint-slot-0.pt", "checkpoint-slot-1.pt"}


@pytest.mark.parametrize("value", [np.float64(0.5), np.int64(3), np.arange(3), {1, 2}, object()])
@pytest.mark.parametrize("existing", [False, True])
def test_checkpoint_rejects_unsafe_values_before_writing(tmp_path, value, existing):
    api = checkpoint_module()
    directory, staging = tmp_path / "state", tmp_path / "staging"
    if existing:
        api.save_checkpoint(directory, payload(), counter=0, staging=staging)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    unsafe = {**payload(2, 2), "history": [{}, {}, {}, {"loss": value}]}
    with pytest.raises(TypeError, match=r"history\[3\]\.loss"):
        api.save_checkpoint(directory, unsafe, counter=0, staging=staging)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    if not existing:
        assert not directory.exists() and not staging.exists()
    else:
        assert list(staging.iterdir()) == []


def test_checkpoint_accepts_native_integer_keys_anywhere(tmp_path):
    api = checkpoint_module()
    api.save_checkpoint(tmp_path / "state", {**payload(), "metadata": {3: "value"}}, counter=0)
    assert api.latest_checkpoint(tmp_path / "state")[1]["metadata"] == {3: "value"}


@pytest.mark.parametrize("key", [True, np.int64(2), 1.5, ("a", "b")])
def test_checkpoint_rejects_nonstring_mapping_keys(tmp_path, key):
    with pytest.raises(TypeError, match="metadata"):
        checkpoint_module().save_checkpoint(
            tmp_path / "state", {**payload(), "metadata": {key: "value"}}, counter=0
        )
    assert not (tmp_path / "state").exists()


def test_checkpoint_validates_values_under_optimizer_parameter_ids(tmp_path):
    unsafe = {**payload(), "optimizer": {"state": {0: {"step": np.float64(1.0)}}}}
    with pytest.raises(TypeError, match=r"optimizer\.state\[0\]\.step"):
        checkpoint_module().save_checkpoint(tmp_path / "state", unsafe, counter=0)
    assert not (tmp_path / "state").exists()


def test_training_payload_remains_weights_only_loadable(tmp_path):
    api = checkpoint_module()
    # Construct the loop's payload shape here so checkpoint tests can run before
    # the loop and its fixtures exist when this fix is replayed earlier in the chain.
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    original = {
        **payload(),
        "identity": "training-identity",
        "config": {"seed": 17, "sources": [{"kind": "typed", "recipes": [["trend"]]}]},
        "model": model.state_dict(),
        "ema": {name: value.clone() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": torch.amp.GradScaler("cpu").state_dict(),
        "history": [{"batch": 1, "loss": 0.5, "parts": {"pinball": 0.4}}],
        "pending": {"loss": 0.0, "parts": {}, "updates": 0},
        "export": "ema",
        "values": [None, True, 1, 0.5, "text", (2, 3)],
    }
    path, _ = api.save_checkpoint(tmp_path, original, counter=0)
    restored = torch.load(io.BytesIO(path.read_bytes()[:-32]), weights_only=True)
    assert restored.keys() == original.keys()
    for key in original.keys() - {"model", "ema", "optimizer"}:
        assert restored[key] == original[key]
    for key in ("model", "ema"):
        for name, value in original[key].items():
            assert torch.equal(restored[key][name], value)
    assert restored["optimizer"]["param_groups"] == original["optimizer"]["param_groups"]
    for parameter, state in original["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(restored["optimizer"]["state"][parameter][name], value)
    optimizer.load_state_dict(restored["optimizer"])


def test_checkpoint_staging_keeps_temporary_files_out_of_state(tmp_path, monkeypatch):
    api = checkpoint_module()
    directory, staging = tmp_path / "state", tmp_path / "staging"
    api.save_checkpoint(directory, payload(), counter=0, staging=staging)
    old_slot = (directory / "checkpoint-slot-0.pt").read_bytes()
    original_replace = os.replace

    def inspect_replace(source, destination):
        assert source.parent == staging
        assert destination == directory / "checkpoint-slot-0.pt"
        assert {p.name for p in directory.iterdir()} == {"checkpoint-slot-0.pt"}
        assert destination.read_bytes() == old_slot
        original_replace(source, destination)

    monkeypatch.setattr(api.os, "replace", inspect_replace)
    api.save_checkpoint(directory, payload(5, 2), counter=2, staging=staging)
    assert api.latest_checkpoint(directory)[1]["counters"]["batch"] == 5
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize(
    "batches,saves,expected",
    [((8, 7), (1, 9), 0), ((7, 8), (9, 1), 1), ((8, 8), (9, 10), 1), ((8, 8), (10, 9), 0)],
)
def test_latest_orders_by_batch_then_saves(tmp_path, batches, saves, expected):
    api = checkpoint_module()
    for slot in (0, 1):
        api.save_checkpoint(tmp_path, payload(batches[slot], saves[slot]), counter=slot)
    assert api.latest_checkpoint(tmp_path)[0].name == f"checkpoint-slot-{expected}.pt"


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_damaged_newer_slot_falls_back_to_valid_slot(tmp_path, damage):
    api = checkpoint_module()
    good, _ = api.save_checkpoint(tmp_path, payload(1, 1), counter=0)
    bad, _ = api.save_checkpoint(tmp_path, payload(2, 2), counter=1)
    data = bytearray(bad.read_bytes())
    if damage == "truncate":
        data = data[:-10]
    else:
        data[len(data) // 2] ^= 1
    bad.write_bytes(data)
    assert api.latest_checkpoint(tmp_path)[0] == good


def test_two_damaged_slots_raise_instead_of_starting_over(tmp_path):
    api = checkpoint_module()
    for slot in (0, 1):
        (tmp_path / f"checkpoint-slot-{slot}.pt").write_bytes(b"corrupt")
    with pytest.raises(api.CheckpointError):
        api.latest_checkpoint(tmp_path)


def test_empty_directory_and_unpublished_temporary_file_are_not_checkpoints(tmp_path):
    api = checkpoint_module()
    assert api.latest_checkpoint(tmp_path) is None
    (tmp_path / ".checkpoint-in-progress.pt").write_bytes(b"partial")
    assert api.latest_checkpoint(tmp_path) is None
    assert api.latest_checkpoint(tmp_path / "absent") is None


def test_failed_publish_preserves_previous_slot_and_cleans_staging(tmp_path, monkeypatch):
    api = checkpoint_module()
    directory, staging = tmp_path / "state", tmp_path / "staging"
    path, _ = api.save_checkpoint(directory, payload(), counter=0, staging=staging)
    before = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated failed publish")

    monkeypatch.setattr(api.os, "replace", fail_replace)
    with pytest.raises(OSError, match="failed publish"):
        api.save_checkpoint(directory, payload(2, 2), counter=2, staging=staging)
    assert path.read_bytes() == before
    assert list(staging.iterdir()) == []


def test_checkpoint_requires_only_batch_and_save_counters(tmp_path):
    api = checkpoint_module()
    minimal = {"counters": {"batch": 0, "saves": 0}}
    api.save_checkpoint(tmp_path, minimal, counter=0)
    assert api.latest_checkpoint(tmp_path)[1] == minimal


def test_file_digest_hashes_file_bytes(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"abc")
    assert (
        checkpoint_module().file_digest(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_manifest_round_trip_preserves_metadata_and_input(tmp_path):
    api = checkpoint_module()
    directory = tmp_path / "unit"
    (directory / "artifacts").mkdir(parents=True)
    (directory / "artifacts/weights.pt").write_bytes(b"abc")
    (directory / "empty").write_bytes(b"")
    record = {
        "files": ["artifacts/weights.pt", "empty"],
        "identity": "revision",
        "details": {"files": ["unchanged"], "values": [1, None, True, "é"]},
    }
    expected = {
        **record,
        "files": {
            "artifacts/weights.pt": {
                "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "bytes": 3,
            },
            "empty": {
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "bytes": 0,
            },
        },
    }
    path = directory / "unit.json"
    api.write_manifest(path, record)
    assert json.loads(path.read_text()) == expected
    assert api.read_manifest(path) == expected
    assert record["files"] == ["artifacts/weights.pt", "empty"]


@pytest.mark.parametrize("record", [{"status": "done"}, {"files": []}, {"files": {}}])
def test_manifest_without_named_files_is_rejected(tmp_path, record):
    path = tmp_path / "new/unit.json"
    with pytest.raises(ValueError, match="files"):
        checkpoint_module().write_manifest(path, record)
    assert not path.exists()


def test_manifest_paths_remain_valid_when_unit_directory_moves(tmp_path):
    api = checkpoint_module()
    directory = tmp_path / "original"
    directory.mkdir()
    (directory / "weights.pt").write_bytes(b"abc")
    api.write_manifest(directory / "unit.json", {"files": ["weights.pt"]})
    expected = api.read_manifest(directory / "unit.json")
    directory.rename(tmp_path / "moved")
    assert api.read_manifest(tmp_path / "moved/unit.json") == expected


@pytest.mark.parametrize("damage", ["same_size", "different_size", "missing"])
def test_manifest_rejects_changed_or_missing_artifact(tmp_path, damage):
    api = checkpoint_module()
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"abc")
    path = tmp_path / "unit.json"
    api.write_manifest(path, {"files": ["weights.pt"]})
    if damage == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"abd" if damage == "same_size" else b"abcd")
    assert api.read_manifest(path) is None


@pytest.mark.parametrize(
    "contents",
    [None, b"", b'{"files":', b"not json", b"\xff", b"null", b"[]", b"{}"],
)
def test_missing_or_invalid_manifest_returns_none(tmp_path, contents):
    path = tmp_path / "unit.json"
    if contents is not None:
        path.write_bytes(contents)
    assert checkpoint_module().read_manifest(path) is None


@pytest.mark.parametrize(
    "files",
    [
        [],
        {},
        None,
        {"weights.pt": None},
        {"weights.pt": {}},
        {"weights.pt": {"sha256": "wrong", "bytes": 3}},
        {
            "weights.pt": {
                "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "bytes": 2,
            }
        },
    ],
)
def test_manifest_rejects_invalid_artifact_records(tmp_path, files):
    (tmp_path / "weights.pt").write_bytes(b"abc")
    path = tmp_path / "unit.json"
    path.write_text(json.dumps({"files": files}))
    assert checkpoint_module().read_manifest(path) is None


@pytest.mark.parametrize(
    "name",
    ["../weights.pt", "nested/../weights.pt", "/weights.pt", r"nested\weights.pt", "weights..pt"],
)
def test_manifest_writer_rejects_unsafe_names(tmp_path, name):
    path = tmp_path / "unit.json"
    with pytest.raises(ValueError):
        checkpoint_module().write_manifest(path, {"files": [name]})
    assert not path.exists()


@pytest.mark.parametrize("kind", ["parent", "absolute", "backslash"])
def test_manifest_reader_rejects_unsafe_names(tmp_path, kind):
    directory = tmp_path / "unit"
    directory.mkdir()
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"abc")
    (directory / r"nested\weights.pt").write_bytes(b"abc")
    name = {
        "parent": "../weights.pt",
        "absolute": str(artifact),
        "backslash": r"nested\weights.pt",
    }[kind]
    path = directory / "unit.json"
    path.write_text(
        json.dumps(
            {
                "files": {
                    name: {
                        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                        "bytes": 3,
                    }
                }
            }
        )
    )
    assert checkpoint_module().read_manifest(path) is None


def test_manifest_is_synced_before_replace_and_directory_synced_after(tmp_path, monkeypatch):
    api = checkpoint_module()
    path = tmp_path / "unit.json"
    (tmp_path / "artifact").write_bytes(b"abc")
    api.write_manifest(path, {"files": ["artifact"], "version": 1})
    files = api.read_manifest(path)["files"]
    events = []
    original_sync, original_replace = os.fsync, os.replace

    def sync(descriptor):
        events.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        original_sync(descriptor)

    def replace(source, destination):
        assert events == ["file"]
        assert Path(source).parent == path.parent
        assert Path(source) != path and destination == path
        assert json.loads(path.read_text())["version"] == 1
        assert json.loads(Path(source).read_text()) == {"files": files, "version": 2}
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(api.os, "fsync", sync)
    monkeypatch.setattr(api.os, "replace", replace)
    api.write_manifest(path, {"files": ["artifact"], "version": 2})
    assert events == ["file", "replace", "directory"]
    assert api.read_manifest(path) == {"files": files, "version": 2}
    assert {p.name for p in tmp_path.iterdir()} == {"unit.json", "artifact"}


def test_manifest_failed_replace_preserves_previous_record(tmp_path, monkeypatch):
    api = checkpoint_module()
    path = tmp_path / "unit.json"
    (tmp_path / "artifact").write_bytes(b"abc")
    api.write_manifest(path, {"files": ["artifact"], "version": 1})
    before = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated failed publish")

    monkeypatch.setattr(api.os, "replace", fail_replace)
    with pytest.raises(OSError, match="failed publish"):
        api.write_manifest(path, {"files": ["artifact"], "version": 2})
    assert path.read_bytes() == before
    assert {p.name for p in tmp_path.iterdir()} == {"unit.json", "artifact"}


def test_manifest_requires_json_serialisable_metadata(tmp_path):
    api = checkpoint_module()
    path = tmp_path / "unit.json"
    (tmp_path / "artifact").write_bytes(b"abc")
    api.write_manifest(path, {"files": ["artifact"], "version": 1})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        api.write_manifest(path, {"files": ["artifact"], "invalid": {1, 2}})
    assert path.read_bytes() == before
    assert {p.name for p in tmp_path.iterdir()} == {"unit.json", "artifact"}


def test_request_disabled_is_never_pending():
    request = checkpoint_module().CheckpointRequest(None)
    assert request.pending() is None


def test_nonce_remains_pending_until_exact_answer_and_survives_restart(tmp_path):
    api = checkpoint_module()
    request = api.CheckpointRequest(tmp_path)
    assert request.pending() is None
    (tmp_path / "checkpoint-request").write_text("first-nonce\n")
    (tmp_path / "checkpoint-done").write_text("older-nonce")
    assert request.pending() == "first-nonce"
    assert request.pending() == "first-nonce"
    request.answer("first-nonce")
    assert (tmp_path / "checkpoint-done").read_bytes() == b"first-nonce"
    assert request.pending() is None
    assert api.CheckpointRequest(tmp_path).pending() is None
    (tmp_path / "checkpoint-request").write_text("second-nonce\n")
    assert request.pending() == "second-nonce"
    request.answer("second-nonce")
    assert request.pending() is None
    assert (tmp_path / "checkpoint-done").read_bytes() == b"second-nonce"
    assert {p.name for p in tmp_path.iterdir()} == {"checkpoint-request", "checkpoint-done"}


@pytest.mark.parametrize("damage", ["tensor_storage", "footer", "foreign_footer"])
@pytest.mark.parametrize("fallback", [False, True])
def test_sha256_footer_rejects_loadable_corruption(tmp_path, damage, fallback):
    api = checkpoint_module()
    # Unique uint8 storage bytes let us flip tensor data, never ZIP metadata.
    pattern = bytes(range(32, 224))
    newer = {**payload(2, 2), "pattern": torch.tensor(list(pattern), dtype=torch.uint8)}
    bad, _ = api.save_checkpoint(tmp_path, newer, counter=1)
    data = bytearray(bad.read_bytes())
    if damage == "tensor_storage":
        assert data.count(pattern) == 1
        data[data.index(pattern) + 17] ^= 1
    elif damage == "footer":
        data[-1] ^= 1
    else:
        other, _ = api.save_checkpoint(tmp_path / "other", payload(9, 9), counter=0)
        foreign = other.read_bytes()[-32:]
        assert foreign != data[-32:]
        data[-32:] = foreign
    bad.write_bytes(data)
    # Prove deserialization alone accepts these bytes (including wrong tensor values).
    loadable = torch.load(io.BytesIO(data[:-32]), map_location="cpu", weights_only=True)
    assert loadable["counters"] == {"batch": 2, "saves": 2}
    assert torch.equal(loadable["pattern"], newer["pattern"]) == (damage != "tensor_storage")
    if fallback:
        good, _ = api.save_checkpoint(tmp_path, payload(), counter=0)
        selected, restored = api.latest_checkpoint(tmp_path)
        assert selected == good
        assert torch.equal(restored["model"]["weight"], torch.arange(3))
    else:
        with pytest.raises(api.CheckpointError, match="footer"):
            api.latest_checkpoint(tmp_path)


@pytest.mark.parametrize("fallback", [False, True])
def test_checkpoint_payload_without_counters_is_rejected(tmp_path, fallback):
    api = checkpoint_module()
    # Bypass the writer's counter check to model a malformed, correctly hashed slot.
    body = io.BytesIO()
    torch.save({"model": {"weight": torch.arange(3)}}, body)
    data = body.getvalue()
    (tmp_path / "checkpoint-slot-1.pt").write_bytes(data + hashlib.sha256(data).digest())
    if fallback:
        good, _ = api.save_checkpoint(tmp_path, payload(), counter=0)
        assert api.latest_checkpoint(tmp_path)[0] == good
    else:
        with pytest.raises(api.CheckpointError, match="counters"):
            api.latest_checkpoint(tmp_path)


def test_checkpoint_load_streams_body_and_hides_footer(tmp_path, monkeypatch):
    api = checkpoint_module()
    saved = {**payload(), "large": torch.arange(600_000)}
    path, _ = api.save_checkpoint(tmp_path, saved, counter=0)
    body_size = path.stat().st_size - 32
    original_open, original_load = Path.open, torch.load
    reads, handles = [], []
    loading = False

    class TrackedFile:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            # Hashing must never materialize the entire checkpoint.
            if not loading:
                assert 0 <= size <= 1024 * 1024
                reads.append(size)
            return self.stream.read(size)

    def tracked_open(self, *args, **kwargs):
        stream = TrackedFile(original_open(self, *args, **kwargs))
        handles.append(stream)
        return stream

    def checked_load(stream, *args, **kwargs):
        nonlocal loading
        loading = True
        assert len(handles) == 1
        assert not isinstance(stream, io.BytesIO)
        assert kwargs["weights_only"] is True
        assert stream.seek(0, os.SEEK_END) == body_size
        assert stream.read(32) == b""
        stream.seek(-16, os.SEEK_END)
        assert len(stream.read(64)) == 16
        buffer = bytearray(64)
        stream.seek(-8, os.SEEK_END)
        assert stream.readinto(buffer) == 8
        stream.seek(0)
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(api.torch, "load", checked_load)
    restored = api.latest_checkpoint(tmp_path)[1]
    assert len(reads) >= 5
    assert torch.equal(restored["large"], saved["large"])


@pytest.mark.parametrize(
    "operation", ["save_checkpoint", "latest_checkpoint", "write_manifest", "read_manifest"]
)
def test_persistence_accepts_string_paths(tmp_path, operation):
    api = checkpoint_module()
    path = tmp_path / "unit.json"
    (tmp_path / "artifact").write_bytes(b"abc")
    if operation == "save_checkpoint":
        staging = tmp_path / "staging"
        slot, _ = api.save_checkpoint(str(tmp_path), payload(), counter=0, staging=str(staging))
        assert api.latest_checkpoint(tmp_path)[0] == slot
    elif operation == "latest_checkpoint":
        slot, _ = api.save_checkpoint(tmp_path, payload(), counter=0)
        assert api.latest_checkpoint(str(tmp_path))[0] == slot
    elif operation == "write_manifest":
        api.write_manifest(str(path), {"files": ["artifact"]})
        assert api.read_manifest(path)["files"]["artifact"]["bytes"] == 3
    else:
        api.write_manifest(path, {"files": ["artifact"]})
        assert api.read_manifest(str(path))["files"]["artifact"]["bytes"] == 3


@pytest.mark.parametrize("kind", ["symlink", "directory", "symlink_parent"])
def test_manifest_rejects_nonregular_artifact_names(tmp_path, kind):
    api = checkpoint_module()
    directory = tmp_path / "unit"
    directory.mkdir()
    target = tmp_path / "outside"
    target.write_bytes(b"abc")
    name = "artifact"
    artifact = directory / name
    if kind == "symlink":
        artifact.symlink_to(target)
    elif kind == "directory":
        artifact.mkdir()
    else:
        artifact.symlink_to(tmp_path, target_is_directory=True)
        name += "/outside"
    path = directory / "unit.json"
    path.write_text(
        json.dumps({"files": {name: {"sha256": hashlib.sha256(b"abc").hexdigest(), "bytes": 3}}})
    )
    assert api.read_manifest(path) is None
    path.unlink()
    with pytest.raises(ValueError):
        api.write_manifest(path, {"files": [name]})
    assert not path.exists()


@pytest.mark.parametrize("operation", ["checkpoint", "manifest"])
def test_directory_fsync_failure_raises_after_publishing(tmp_path, monkeypatch, operation):
    api = checkpoint_module()
    original_sync = os.fsync

    def fail_directory_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        original_sync(descriptor)

    monkeypatch.setattr(api.os, "fsync", fail_directory_sync)
    with pytest.raises(OSError, match="directory fsync failed"):
        if operation == "checkpoint":
            api.save_checkpoint(tmp_path, payload(), counter=0)
        else:
            (tmp_path / "artifact").write_bytes(b"abc")
            api.write_manifest(tmp_path / "unit.json", {"files": ["artifact"]})
    if operation == "checkpoint":
        assert api.latest_checkpoint(tmp_path)[1]["counters"] == {"batch": 1, "saves": 1}
    else:
        assert api.read_manifest(tmp_path / "unit.json")["files"]["artifact"]["bytes"] == 3
    assert not list(tmp_path.glob(".*-*"))


@pytest.mark.parametrize("files", ["a", True, 3, {"a"}])
def test_manifest_files_must_be_a_list_of_names(tmp_path, files):
    (tmp_path / "a").write_bytes(b"x")
    with pytest.raises(ValueError, match="files"):
        checkpoint_module().write_manifest(tmp_path / "unit.json", {"files": files})


def test_manifest_record_must_be_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="record"):
        checkpoint_module().write_manifest(tmp_path / "unit.json", ["a"])
