"""Malformed restored JSON must replay work, never crash or invent progress."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.corpus_v4 import audit_sketch as s
from tests.test_corpus_v4.test_audit_sketch_restart import fixture
from tests.test_corpus_v4.test_manifest_fetch import manifest, native

SHAPES = (
    "evidence_list",
    "units_null",
    "units_mapping",
    "units_member",
    "counts_missing",
    "counts_list",
    "counts_value",
)


def damage(marker, shape):
    record = json.loads(marker.read_text())
    result = record["result"]
    if shape == "evidence_list":
        result["evidence"] = []
    elif shape.startswith("units_"):
        result["evidence"]["units"] = {
            "units_null": None,
            "units_mapping": {},
            "units_member": [None],
        }[shape]
    elif shape.endswith("_missing"):
        result.pop(shape.removesuffix("_missing"), None)
    elif shape == "counts_list":
        result["counts"] = []
    elif shape == "counts_value":
        result["counts"]["values"] = "512"
    elif shape == "last_row_string":
        result["last_row"] = "0"
    record["result_digest"] = s._fingerprint(result)
    marker.write_text(json.dumps(record))


@pytest.mark.parametrize("skip_finished", [False, True])
def test_subset_reuses_unselected_and_rescans_selected_unless_skipped(
    tmp_path, monkeypatch, skip_finished
):
    root, state, out, benchmarks = fixture(tmp_path)
    native(root / "univariate/other/train-000.parquet", target=benchmarks[0].values.tolist())
    policy = manifest()
    policy["folders"] += manifest("univariate/other")["folders"]
    monkeypatch.setattr(s, "_execute", s.scan_folder)
    before, code = s.run_sketch(root, out, policy, benchmarks, state=state)
    assert code == 0
    computed = []
    original = s.sketch

    def observe(values):
        computed.append(len(values))
        return original(values)

    monkeypatch.setattr(s, "sketch", observe)
    after, code = s.run_sketch(
        root,
        out,
        policy,
        state=state,
        only_folders=["univariate/demo"],
        skip_finished=skip_finished,
    )
    assert code == 0 and after == before
    assert computed == ([] if skip_finished else [512])
    progress = json.loads((state / "sketch-progress.json").read_text())
    assert progress["values_reused"] == (1024 if skip_finished else 512)


@pytest.mark.parametrize(
    ("kind", "shape"),
    [(kind, shape) for kind in ("folder", "unit", "part", "controls") for shape in SHAPES]
    + [
        (kind, shape)
        for kind in ("unit", "part")
        for shape in ("last_row_missing", "last_row_string")
    ],
)
def test_malformed_completed_or_partial_marker_is_replayed(tmp_path, monkeypatch, kind, shape):
    root, state, out, benchmarks = fixture(tmp_path)
    monkeypatch.setattr(s, "_execute", s.scan_folder)
    before, code = s.run_sketch(root, out, manifest(), benchmarks, state=state)
    assert code == 0
    folder = out / "evidence/univariate--demo.json"
    unit = next((out / "evidence/units").glob("*-rg000000.json"))
    part = next(unit.parent.glob("*-part000000.json"))
    marker = dict(folder=folder, unit=unit, part=part, controls=state / "sketch-controls.json")[
        kind
    ]
    original = json.loads(marker.read_text())
    damage(marker, shape)
    if kind in {"unit", "part"}:
        folder.unlink()
    if kind == "part":
        unit.unlink()
    after, code = s.run_sketch(root, out, manifest(), state=state)
    assert code == 0 and after == before
    assert json.loads(marker.read_text()) == original


@pytest.mark.parametrize("shape", [*SHAPES, "last_row_missing", "last_row_string"])
def test_progress_ignores_malformed_partial_marker(tmp_path, shape):
    state, directory = tmp_path / "state", tmp_path / "units"
    directory.mkdir()
    evidence = directory / "unit-part000000.jsonl.gz"
    evidence.write_bytes(b"closed evidence")
    marker = directory / "unit-part000000.json"
    result = dict(
        counts=dict(values=512, channels=1),
        last_row=0,
        corpus_frequencies=["h"],
        corpus_sources=["fixture"],
        evidence=dict(file=evidence.name, sha256=s.digest(evidence)),
    )
    marker.write_text(
        json.dumps(dict(identity="identity", result=result, result_digest=s._fingerprint(result)))
    )
    progress = s._Progress(state)
    try:
        with progress.lock:
            progress.record.update(
                phase="scanning",
                units_total=1,
                folders={"demo": dict(values_done=0, channels_done=0, values_estimated=512)},
            )
            progress.pending[("demo", "unit")] = (str(directory), "unit", "identity")
        progress.publish()
        assert json.loads(progress.path.read_text())["values_done"] == 512
        damage(marker, shape)
        progress.publish()
        snapshot = json.loads(progress.path.read_text())
        assert snapshot["values_done"] == 0 and snapshot["channels_done"] == 0
        assert snapshot["folders"]["demo"]["values_done"] == 0
    finally:
        progress.close()


@pytest.mark.parametrize("destination", ["state", "out", "scratch", "evidence", "units"])
def test_separate_filesystem_is_rejected_before_preparation(tmp_path, monkeypatch, destination):
    root, state, out, benchmarks = fixture(tmp_path)
    paths = dict(
        state=state,
        out=out,
        scratch=state.parent / ("." + state.name + "-sketch-work"),
        evidence=out / "evidence",
        units=out / "evidence/units",
    )
    actual_stat = Path.stat

    def other_device(path, *args, **kwargs):
        stat = actual_stat(path, *args, **kwargs)
        if path == paths[destination]:
            return SimpleNamespace(st_mode=stat.st_mode, st_dev=stat.st_dev + 1)
        return stat

    monkeypatch.setattr(Path, "stat", other_device)
    monkeypatch.setattr(s, "prepare_benchmarks", lambda *a, **kw: pytest.fail("prepared too soon"))
    with pytest.raises(ValueError, match="same filesystem"):
        s.run_sketch(root, out, manifest(), benchmarks, state=state)
    assert not (state / "sketch-input.json").exists()
    assert not (state / "sketch-progress.json").exists()
