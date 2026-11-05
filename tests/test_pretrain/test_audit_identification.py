"""Offline report, resume, and serial/pool/chunk equivalence for all statuses."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

# The audit needs the corpus extras; the dev-only CI job must skip this module.
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("datasets")

from demand_forecast.pretrain import audit, audit_info  # noqa: E402
from demand_forecast.pretrain.corpus import corpus_manifest, write_manifest  # noqa: E402
from tests.test_pretrain.test_audit import _parquet  # noqa: E402
from tests.test_pretrain.test_audit import local_audit as local_audit  # noqa: E402


@pytest.fixture
def identification_audit(local_audit, tmp_path):
    root, tasks = local_audit
    # Separate unrelated ramps share shape; a short periodic motif provides
    # low-information matches in the same native scan as literal random copies.
    ramp = np.arange(32, dtype=float)
    motif = np.tile([0.0, 1.0], 16)
    sparse = np.array([0.0] * 6 + [3] + [0.0] * 5 + [1] + [0.0] * 7 + [2] + [0.0] * 4 + [5])
    corpus_path = next((root / "training_corpus/chronos_real").glob("*.parquet"))
    corpus = pq.read_table(corpus_path).to_pylist()
    for name, values in (("ramp", 7 * ramp + 13), ("motif", motif), ("sparse", 2 * sparse + 1)):
        corpus.append({**corpus[0], "id": name, "target": values.tolist(), "folder": name})
    _parquet(corpus_path, corpus)
    write_manifest(corpus_manifest(root), root / "manifest.json")
    path = tmp_path / "copies.parquet"
    rows = pq.read_table(path).to_pylist()
    for name, values in (("ramp", ramp), ("motif", motif)):
        rows.append(
            {
                "id": name,
                "timestamp": pd.date_range("2020-01-01", periods=len(values)).tolist(),
                "target": values.tolist(),
                "other_target": values.tolist(),
            }
        )
    _parquet(path, rows)
    definitions = yaml.safe_load(tasks.read_text())
    for name, values in (("ramps", ramp), ("sparse", sparse)):
        _parquet(
            tmp_path / f"{name}.parquet",
            [
                {
                    "id": name,
                    "timestamp": pd.date_range("2020-01-01", periods=len(values)).tolist(),
                    "target": values.tolist(),
                }
            ],
        )
        definitions["tasks"].append(
            {"task_name": name, "dataset_path": "fixture/fev", "local_path": f"{name}.parquet"}
        )
    tasks.write_text(yaml.safe_dump(definitions))
    return root, tasks


def _assert_report(out, report):
    declaration = json.loads((out / "declaration.json").read_text())
    assert declaration["copies"] == [
        p for p in report["copies"] if p["identification"] == "identifying"
    ]
    assert all(p["values_target"] and p["values_index"] for p in declaration["copies"])
    assert all(
        type(p[field]) is int
        for p in declaration["copies"]
        for field in ("info_bits_target", "info_bits_index")
    )
    assert declaration["identification_summary"] == report["identification_summary"]
    assert not any(p["task"] == "negative" for p in report["copies"])
    assert report["identification_summary"]["negative"] == {
        role: dict.fromkeys(audit_info.STATUSES, 0) for role in ("target", "covariate")
    }
    assert (
        declaration["tasks_with_only_low_information_matches"]
        == report["tasks_with_only_low_information_matches"]
    )
    assert declaration["method"]["info_version"] == 1
    assert (
        declaration["method"]["detection_limits"]["production_positive_controls_identifying"] == 21
    )
    for corpus_id in ("literal", "affine", "weekly"):
        assert any(p["corpus_id"] == corpus_id for p in declaration["copies"])
    assert {"copies", "covariate-only"} <= set(report["trained_on_datasets"])
    ramps = [p for p in report["copies"] if p["task"] == "ramps"]
    assert ramps and all(p["identification"] == "interpolation_ramp" for p in ramps)
    assert "ramps" in report["tasks_with_only_low_information_matches"]
    assert "ramps" not in report["trained_on_datasets"]
    assert all("_info_values" not in p for p in report["copies"])
    markdown = (out / "audit.md").read_text()
    assert "## Identification" in markdown and "not identifiable as a copy" in markdown
    assert "not proven unrelated" in markdown and '"ramps"' in markdown
    assert json.loads((out / "audit.json").read_text())["copies"] == report["copies"]


def test_all_statuses_and_values_byte_identical_across_execution_paths(
    identification_audit, tmp_path, monkeypatch
):
    root, tasks = identification_audit
    expected = None
    for label, workers, threshold, size in (
        ("serial", 1, 256, 32),
        ("whole", 2, 256, 32),
        ("chunk1", 2, 1, 1),
        ("chunk2", 3, 1, 2),
        ("chunk_all", 2, 1, 32),
    ):
        monkeypatch.setattr(audit, "SCAN_CHUNK_ROWS", threshold)
        monkeypatch.setattr(audit, "SCAN_CHUNK_MIN_GROUPS", size)
        out = tmp_path / label
        report = audit.run_audit(root, out, tasks=tasks, workers=workers, progress=lambda _: None)
        _assert_report(out, report)
        native = next(
            s
            for s in report["scans"]
            if s["task"] == "copies" and s["column"] == "target" and s["variant"] == "native"
        )
        assert {p["identification"] for p in native["copies"]} == set(audit_info.STATUSES)
        assert all(
            "values_target" in p
            for p in native["copies"]
            if p["identification"] != "interpolation_ramp"
        )
        for scan in report["scans"]:
            for status in audit_info.STATUSES:
                assert scan["counts"][status + "_copies"] == sum(
                    p["identification"] == status for p in scan["copies"]
                )
            index = report["indices"][scan["kind"] + "::" + scan["frequency_class"]]
            assert scan["info"]["index_windows"] == sum(index["windows"].values())
            assert scan["info"]["target_windows"] == sum(scan["counts"]["windows"].values())
        actual = {p.name: p.read_bytes() for p in (out / "scans").glob("*.json")}
        for name, content in actual.items():
            scan = json.loads(content)
            job = {"result_path": str(out / "scans" / name), "fingerprint": scan["fingerprint"]}
            assert audit._resume_result(job) == scan
        if expected is None:
            expected = actual
        assert actual == expected


def test_raised_margin_preserves_low_information_evidence(
    identification_audit, tmp_path, monkeypatch
):
    root, tasks = identification_audit
    before = audit.run_audit(
        root, tmp_path / "before", tasks=tasks, workers=1, progress=lambda _: None
    )
    assert "sparse" in before["trained_on_datasets"]
    monkeypatch.setattr(audit_info, "INFO_MARGIN_BITS", 100)
    out = tmp_path / "raised"
    report = audit.run_audit(root, out, tasks=tasks, workers=1, progress=lambda _: None)
    _assert_report(out, report)
    assert "sparse" not in report["trained_on_datasets"]
    assert "sparse" in report["tasks_with_only_low_information_matches"]
    records = [p for p in report["copies"] if p["task"] == "sparse"]
    assert records and all(p["identification"] == "low_information" for p in records)
    assert all(p["values_target"] and p["values_index"] for p in records)
    assert report["identification_summary"]["sparse"]["target"]["low_information"] == len(records)
    assert report["summary"]["copies"] == before["summary"]["copies"]
    assert report["input_fingerprint"] != before["input_fingerprint"]
    assert report["constants"]["INFO_MARGIN_BITS"] == 100


def test_previous_method_scans_are_not_resumed(local_audit, tmp_path, monkeypatch):
    root, tasks = local_audit
    out = tmp_path / "out"
    assert audit.METHOD_VERSION == "run12-corpus-v1"
    with monkeypatch.context() as old:
        old.setattr(audit, "METHOD_VERSION", "run11-corpus-v1")
        previous = audit.run_audit(root, out, tasks=tasks, workers=1, progress=lambda _: None)
    messages = []
    current = audit.run_audit(
        root, out, tasks=tasks, workers=1, resume=True, progress=messages.append
    )
    assert previous["input_fingerprint"] != current["input_fingerprint"]
    assert not any(m.startswith("Resume:") for m in messages)
    assert sum(m.startswith("Scanned ") for m in messages) == len(current["scans"])


@pytest.mark.parametrize("name", ["INFO_VERSION", "INFO_MARGIN_BITS"])
def test_info_constants_change_fingerprint(monkeypatch, name):
    before = audit._fingerprint({"constants": audit.constants()})
    monkeypatch.setattr(audit_info, name, getattr(audit_info, name) + 1)
    assert audit._fingerprint({"constants": audit.constants()}) != before


_MISSING = object()


@pytest.mark.parametrize(
    "keys, replacement",
    [
        (("info",), _MISSING),
        (("info",), {}),
        (("info",), []),
        (("info",), None),
        *[
            (("info", field), _MISSING)
            for field in ("version", "margin_bits", "target_windows", "index_windows")
        ],
        (("copies",), {}),
        (("copies",), None),
        *[(("copies", 0), value) for value in (7, None, [], "bad")],
        *[(("copies", 0, "identification"), value) for value in (_MISSING, "bad", None, [])],
        *[
            (("copies", 0, field), value)
            for field in ("info_bits_target", "info_bits_index")
            for value in (_MISSING, None, "20", 20.0, True)
        ],
        *[(("copies", 0, "info_strict_ramp"), value) for value in (_MISSING, None, "false", 0)],
    ],
)
def test_incomplete_identification_scan_is_rescanned(local_audit, tmp_path, keys, replacement):
    root, tasks = local_audit
    out = tmp_path / "out"
    audit.run_audit(root, out, tasks=tasks, workers=1, progress=lambda _: None)
    path = next(
        p for p in sorted((out / "scans").glob("*.json")) if json.loads(p.read_text())["copies"]
    )
    original = path.read_bytes()
    scan = json.loads(original)
    job = {"result_path": str(path), "fingerprint": scan["fingerprint"]}
    assert audit._resume_result(job) == scan
    parent = scan
    for key in keys[:-1]:
        parent = parent[key]
    if replacement is _MISSING:
        del parent[keys[-1]]
    else:
        parent[keys[-1]] = replacement
    path.write_text(json.dumps(scan))
    assert audit._resume_result(job) is None
    messages = []
    audit.run_audit(root, out, tasks=tasks, workers=1, resume=True, progress=messages.append)
    assert sum(m.startswith("Scanned ") for m in messages) == 1
    assert path.read_bytes() == original


def test_resume_checks_structure_without_rederiving_status(local_audit, tmp_path):
    root, tasks = local_audit
    out = tmp_path / "out"
    audit.run_audit(root, out, tasks=tasks, workers=1, progress=lambda _: None)
    path = next(
        p for p in sorted((out / "scans").glob("*.json")) if json.loads(p.read_text())["copies"]
    )
    scan = json.loads(path.read_text())
    job = {"result_path": str(path), "fingerprint": scan["fingerprint"]}
    # Even an inconsistent but known status remains outside structural validation.
    for status in audit_info.STATUSES:
        scan["copies"][0]["identification"] = status
        path.write_text(json.dumps(scan))
        assert audit._resume_result(job) == scan
