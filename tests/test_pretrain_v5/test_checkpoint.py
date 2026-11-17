"""Durable slot selection, integrity, and the guest nonce protocol."""

import importlib
import json
import os
import stat
from pathlib import Path

import pytest
import torch


def checkpoint_module():
    return importlib.import_module("demand_forecast.pretrain.v5.checkpoint")


def payload(batch=1, saves=1):
    return {"counters": {"batch": batch, "saves": saves}, "model": {"weight": torch.arange(3)}}


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


@pytest.mark.parametrize("record", [{"status": "done"}, {"files": [], "status": "done"}])
def test_manifest_without_named_files_publishes_empty_mapping(tmp_path, record):
    api = checkpoint_module()
    path = tmp_path / "new/unit.json"
    api.write_manifest(path, record)
    assert api.read_manifest(path) == {**record, "files": {}}


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
    path.write_text('{"files": {}, "version": 1}')
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
        assert json.loads(Path(source).read_text()) == {"files": {}, "version": 2}
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(api.os, "fsync", sync)
    monkeypatch.setattr(api.os, "replace", replace)
    api.write_manifest(path, {"version": 2})
    assert events == ["file", "replace", "directory"]
    assert api.read_manifest(path) == {"files": {}, "version": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_manifest_failed_replace_preserves_previous_record(tmp_path, monkeypatch):
    api = checkpoint_module()
    path = tmp_path / "unit.json"
    api.write_manifest(path, {"version": 1})
    before = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated failed publish")

    monkeypatch.setattr(api.os, "replace", fail_replace)
    with pytest.raises(OSError, match="failed publish"):
        api.write_manifest(path, {"version": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_manifest_requires_json_serialisable_metadata(tmp_path):
    api = checkpoint_module()
    path = tmp_path / "unit.json"
    api.write_manifest(path, {"version": 1})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        api.write_manifest(path, {"invalid": {1, 2}})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


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
