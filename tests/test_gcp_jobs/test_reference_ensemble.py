"""Mix native references with consistently processed Simons saved forecasts."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fev")

from demand_forecast.bench.yardsticks import runner  # noqa: E402
from tests.test_bench.test_fev_ensemble import (  # noqa: E402
    ensemble,
    rewrite_window,
)
from tests.test_bench.test_fev_ensemble import (
    make_member as make_member,
)
from tests.test_bench.test_yardsticks import (  # noqa: E402
    offline_only as offline_only,
)
from tests.test_bench.test_yardsticks import (
    scenario as scenario,
)


@pytest.mark.parametrize("model,batch_size", [("timestral", 1024), ("chronos2", 100)])
@pytest.mark.parametrize("simons_first", [True, False])
def test_mixed_raw_reference_mean(scenario, make_member, tmp_path, model, batch_size, simons_first):
    tasks, _, _ = scenario
    simons = make_member(int(not simons_first), model="simons", processing="raw")
    reference = make_member(int(simons_first), 2, model=model, batch_size=batch_size)
    expected = make_member(2, 1)
    out = tmp_path / "average"
    frame = ensemble(tasks, [simons, reference], out)
    scores = pd.read_csv(expected / "summaries.csv")
    assert frame.test_error.iloc[0] == scores.test_error.iloc[0]
    record = json.loads((out / tasks[0].task_name / "run.json").read_text())
    assert record["weights"] == [0.5, 0.5]
    manifest = json.loads((out / "yardstick.json").read_text())
    assert set(manifest) == {
        "ensemble_version",
        "members",
        "member_configurations",
        "weights",
        "model_name",
        "configuration_identity",
    }
    assert "member_processing" not in record and "member_input_metadata" not in record
    assert "weight_spec" not in record
    for index in range(tasks[0].num_windows):
        filename = tasks[0].task_name + f"/window_{index}.npz"
        with np.load(out / filename) as actual, np.load(simons / filename) as source:
            np.testing.assert_allclose(actual["arrays"], source["arrays"] + 1, rtol=0, atol=1e-14)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    ensemble(tasks, [reference, simons], out)
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("change", ["covariates", "context", "fev", "datasets"])
def test_mixed_members_keep_evaluation_guards(scenario, make_member, tmp_path, monkeypatch, change):
    tasks, _, _ = scenario
    simons = make_member(
        0, model="simons", processing="published" if change == "published" else "raw"
    )
    changes = {}
    if change == "covariates":
        changes["covariates"] = "none"
    elif change == "context":
        changes["max_context"] = 5
    elif change in ("fev", "datasets"):
        original = runner.runtime_identity

        def runtime(options):
            identity = original(options)
            identity["packages"][change] = "changed"
            return identity

        monkeypatch.setattr(runner, "runtime_identity", runtime)
    reference = make_member(1, model="timestral", **changes)
    with pytest.raises(ValueError, match="evaluation.*identity|raw"):
        ensemble(tasks, [simons, reference], tmp_path / "rejected")


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
@pytest.mark.parametrize("processing", ["raw", "published"])
@pytest.mark.parametrize(
    "change", ["simons_source", "simons_implementation", "simons", "batch_size"]
)
def test_mixed_members_reject_different_simons_configurations(
    scenario, make_member, tmp_path, monkeypatch, model, change, processing
):
    tasks, _, _ = scenario
    reference = make_member(0, model=model)
    first = make_member(1, model="simons", processing=processing)
    original = runner.runtime_identity

    def runtime(options):
        identity = original(options)
        if change == "simons":
            identity["packages"]["simons"] = "changed"
        elif change == "simons_implementation":
            identity["simons_implementation"]["source_sha256"] = "changed"
        elif change == "simons_source":
            identity["simons_source"] = "changed"
        return identity

    monkeypatch.setattr(runner, "runtime_identity", runtime)
    changes = {"batch_size": 7} if change == "batch_size" else {}
    second = make_member(2, model="simons", processing=processing, **changes)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="evaluation.*identity"):
        ensemble(tasks, [reference, first, second], out)
    assert not out.exists()


@pytest.mark.parametrize(
    "model,package", [("timestral", "timestral"), ("chronos2", "chronos-forecasting")]
)
def test_mixed_members_reject_different_same_model_reference_packages(
    scenario, make_member, tmp_path, monkeypatch, model, package
):
    tasks, _, _ = scenario
    simons = make_member(0, model="simons", processing="raw")
    first = make_member(1, model=model)
    original = runner.runtime_identity

    def runtime(options):
        identity = original(options)
        identity["packages"][package] = "changed"
        return identity

    monkeypatch.setattr(runner, "runtime_identity", runtime)
    second = make_member(2, model=model)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="evaluation.*identity"):
        ensemble(tasks, [simons, first, second], out)
    assert not out.exists()


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
def test_mixed_members_accept_matching_simons_configurations(
    scenario, make_member, tmp_path, model
):
    tasks, _, _ = scenario
    first = make_member(0, model="simons", processing="raw")
    reference = make_member(1, 4, model=model)
    second = make_member(2, 2, model="simons", processing="raw")
    out = tmp_path / "average"
    frame = ensemble(tasks, [first, reference, second], out)
    expected = pd.read_csv(second / "summaries.csv")
    np.testing.assert_allclose(frame.test_error, expected.test_error, rtol=0, atol=1e-14)
    record = json.loads((out / tasks[0].task_name / "run.json").read_text())
    assert record["weights"] == [1 / 3] * 3


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
def test_mixed_members_reject_different_same_model_covariate_reports(
    scenario, make_member, tmp_path, model
):
    tasks, _, _ = scenario
    reference = make_member(0, model=model)
    first = make_member(1, model="simons", processing="raw")
    second = make_member(2, model="simons", processing="raw")
    path = second / tasks[0].task_name / "run.json"
    record = json.loads(path.read_text())
    key = "dropped_past_only_covariates"
    record["input_metadata"][key] += 1
    record["summary"][key] += 1
    path.write_text(json.dumps(record))
    summary_path = second / "summaries.csv"
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    summary[key] += 1
    summary.to_csv(summary_path, index=False)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="evaluation.*identity"):
        ensemble(tasks, [reference, first, second], out)
    assert not (out / tasks[0].task_name / "run.json").exists()


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
@pytest.mark.parametrize("change", ["fingerprint", "window_ids"])
def test_mixed_members_reject_different_task_windows(
    scenario, make_member, tmp_path, monkeypatch, model, change
):
    tasks, _, _ = scenario
    simons = make_member(0, model="simons", processing="raw")
    with monkeypatch.context() as patch:
        if change == "fingerprint":
            patch.setattr(runner, "dataset_fingerprint", lambda _: "different-dataset")
        reference = make_member(1, model=model)
    if change == "window_ids":
        with np.load(reference / tasks[0].task_name / "window_0.npz") as saved:
            ids = saved["ids"][::-1].copy()
        rewrite_window(reference, tasks[0], ids=ids)
    match = "evaluation.*identity" if change == "fingerprint" else "window input identity"
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match=match):
        ensemble(tasks, [simons, reference], out)
    assert not (out / tasks[0].task_name / "run.json").exists()


@pytest.mark.parametrize("model", ["timestral", "chronos2"])
@pytest.mark.parametrize("reference_first", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_published_mixed_ensemble_provenance(
    scenario, make_member, tmp_path, model, reference_first, weighted
):
    tasks, _, _ = scenario
    reference = make_member(0 if reference_first else 2, 6, model=model)
    simons = [
        make_member(i, offset, model="simons", processing="published")
        for i, offset in zip(([1, 2] if reference_first else [0, 1]), [0, 2], strict=True)
    ]
    out = tmp_path / "published-mixed"
    frame = ensemble(
        tasks, [*simons, reference], out, member_weights=[".25", ".25", ".5"] if weighted else None
    )
    assert frame.processing.iloc[0] == "published"
    manifest = json.loads((out / "yardstick.json").read_text())
    record = json.loads((out / tasks[0].task_name / "run.json").read_text())
    expected = {str(member): {"model": "simons", "processing": "published"} for member in simons}
    expected[str(reference)] = {"model": model, "processing": "native"}
    assert manifest["member_processing"] == record["member_processing"] == expected
    for member in [*simons, reference]:
        metadata = json.loads((member / tasks[0].task_name / "run.json").read_text()).get(
            "input_metadata", {}
        )
        assert record["member_input_metadata"][str(member)] == metadata
        assert manifest["member_input_metadata"][str(member)][tasks[0].task_name] == metadata
    assert record["member_input_metadata"][str(simons[0])]["dropped_past_only_covariates"] == 6
    assert "dropped_past_only_covariates" not in record["member_input_metadata"][str(reference)]
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    ensemble(
        tasks, [reference, *simons], out, member_weights=[".5", ".25", ".25"] if weighted else None
    )
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_mixed_ensemble_rejects_raw_and_published_simons(scenario, make_member, tmp_path):
    tasks, _, _ = scenario
    members = [
        make_member(0, model="timestral"),
        make_member(1, model="simons", processing="raw"),
        make_member(2, model="simons", processing="published"),
    ]
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="raw and published Simons"):
        ensemble(tasks, members, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "change,match", [("unknown-model", "models"), ("reference-processing", "native")]
)
def test_mixed_ensemble_rejects_unsupported_processing(
    scenario, make_member, tmp_path, monkeypatch, change, match
):
    tasks, _, _ = scenario
    simons = make_member(0, model="simons", processing="published")
    original = runner.runtime_identity

    def runtime(options):
        configuration = original(options)
        configuration["options"]["model" if change == "unknown-model" else "processing"] = (
            "unknown" if change == "unknown-model" else "published"
        )
        return configuration

    monkeypatch.setattr(runner, "runtime_identity", runtime)
    reference = make_member(1, model="timestral")
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        ensemble(tasks, [simons, reference], out)
    assert not out.exists()
