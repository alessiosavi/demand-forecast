"""Variation fidelity: synthetic arrays, including properties measured by refit2."""

import json

import numpy as np
import pytest

from scripts.corpus_v4 import audit_sketch as s
from scripts.corpus_v4.audit_fev_overlap import BenchmarkSeries
from tests.test_corpus_v4.test_manifest_fetch import manifest, native


def sparse_pair():
    # refit.json records[10]: modes 0, benchmark levels 0/1, 1804 pairs;
    # these are off-modal positions, not copied corpus/benchmark observations.
    benchmark = np.zeros(1804)
    benchmark[
        [207, 335, 344, 571, 694, 708, 935, 1058, 1068, 1299, 1422, 1434, 1670, 1786, 1803]
    ] = 1
    corpus = np.zeros(1804)
    corpus[[207, 607, 937, 1310]] = [20, 10, 30, 50]
    return corpus, benchmark, s.Candidate(0, "prefix/256")


def price_pair():
    # refit.json records[5]: synthesize both binary series from levels, modes
    # and off-modal positions; no original data files or arrays are used.
    benchmark = np.full(95, 98.80999755859375)
    benchmark[
        [
            2,
            12,
            13,
            19,
            20,
            22,
            25,
            30,
            33,
            37,
            40,
            42,
            43,
            52,
            53,
            57,
            58,
            60,
            61,
            62,
            64,
            66,
            68,
            69,
            72,
            75,
            76,
            79,
            81,
            82,
            83,
            86,
            88,
            89,
            90,
            93,
            94,
        ]
    ] = 98.81999969482422
    corpus = np.full(95, 45.0)
    corpus[
        [
            2,
            12,
            13,
            19,
            20,
            22,
            24,
            27,
            28,
            29,
            31,
            32,
            34,
            35,
            36,
            37,
            41,
            44,
            46,
            48,
            49,
            50,
            52,
            55,
            57,
            58,
            60,
            61,
            63,
            64,
            67,
            68,
            69,
            70,
            72,
            76,
            84,
            85,
            86,
            87,
            88,
            90,
            92,
        ]
    ] = 46
    corpus[71] = np.nan  # The one nonfinite joint position in the refit.
    return corpus, benchmark, s.Candidate(0, "prefix/24")


@pytest.mark.parametrize(
    "make_pair", [sparse_pair, price_pair], ids=["refit-sparse", "refit-price"]
)
def test_refit_shapes_old_rule_accepts_but_fidelity_rejects(make_pair):
    corpus, benchmark, candidate = make_pair()
    old = s.old.fit_segment(corpus, benchmark, 0, len(corpus), 0)
    assert old["is_copy"] and old["r2"] < 0.08
    result = s.verify_candidate(corpus, benchmark, candidate, opportunities=1 << 80)
    assert result["old_rule_accepted"]
    assert not result["verified"]
    assert result["reason"] == "variation_fidelity_rejected"
    assert not result["fidelity"]["passed"]
    json.dumps(result, allow_nan=False)


def shapes():
    flag = np.zeros(4096)
    flag[
        [
            17,
            113,
            207,
            340,
            511,
            700,
            897,
            1101,
            1377,
            1599,
            1833,
            2099,
            2391,
            2617,
            2903,
            3201,
            3527,
            3777,
        ]
    ] = 1
    runs = np.random.default_rng(822).integers(4, 31, 400)
    price = 45.24 + 0.01000213623046875 * np.repeat(np.arange(400) % 2, runs)[:4096]
    return {"sparse_flag": flag, "cent_step": price}


@pytest.mark.parametrize("kind", ["sparse_flag", "cent_step"])
@pytest.mark.parametrize(
    "variant", ["exact", "affine_positive", "affine_negative", "edited", "affine_edited"]
)
def test_fidelity_preserves_variation_copies(kind, variant):
    benchmark = shapes()[kind]
    corpus = benchmark.copy()
    if variant.startswith("affine"):
        corpus = (-2.5 if variant == "affine_negative" else 2.5) * corpus + 7
    if variant.endswith("edited"):
        mode = np.unique(benchmark, return_counts=True)
        modal = np.flatnonzero(benchmark == mode[0][mode[1].argmax()])
        edits = modal[(modal >= 256) & (modal < len(benchmark) - 256)][: len(benchmark) // 100]
        # Large raw edits wreck OLS while preserving every benchmark event.
        # Sparse affine edits stay inside the old relative residual tolerance;
        # cent-step affine outliers pass its absolute floor, then need the
        # robust map to show that the benchmark events really are reproduced.
        size = (0.001 if kind == "sparse_flag" else 2) if variant == "affine_edited" else 20
        corpus[edits] += np.ptp(benchmark) * size
    result = s.verify_candidate(
        corpus, benchmark, s.Candidate(0, "prefix/24"), opportunities=1 << 80
    )
    assert result["verified"], result
    assert result["fidelity"]["passed"]
    assert result["fidelity"]["off_modal_outside"] > 0
    if variant == "edited":
        assert result["fit"]["r2"] < 0.5
    if variant == "affine_edited" and kind == "cent_step":
        assert result["fit"]["r2"] < 0.999
        assert result["fidelity"]["maps"][-1]["method"] == "modal_median"
        assert result["fidelity"]["maps"][-1]["passed"]


def test_fidelity_statistics_exist_for_an_old_rejection_and_zero_range():
    a = np.random.default_rng(71).normal(size=400)
    b = np.random.default_rng(72).normal(size=400)
    result = s.verify_candidate(a, b, s.Candidate(0, "summary"), opportunities=1)
    assert result["fit"] and result["fidelity"]
    assert not result["old_rule_accepted"]
    constant = s.verify_candidate(a, np.ones(400), s.Candidate(0, "raw"), opportunities=1)
    assert not constant["fidelity"]["passed"]
    json.dumps(constant, allow_nan=False)


def test_fidelity_cannot_override_old_rejection_of_an_affine_copy_with_outliers():
    b = shapes()["cent_step"]
    a = b * 2.5 + 7
    levels, counts = np.unique(b, return_counts=True)
    modal = np.flatnonzero(b == levels[counts.argmax()])
    edits = modal[(modal >= 256) & (modal < len(b) - 256)][: len(b) // 100]
    a[edits] += 20 * np.ptp(b)
    result = s.verify_candidate(a, b, s.Candidate(0, "prefix/24"), opportunities=1 << 80)
    assert result["fidelity"]["passed"]
    assert not result["old_rule_accepted"] and not result["verified"]
    assert result["reason"] == "numerical_fit_rejected"


def test_outside_window_cannot_be_diluted_by_many_reproduced_inside_events():
    benchmark = np.zeros(8192)
    benchmark[1:255:2] = 1
    benchmark[4096] = 1
    corpus = benchmark.copy()
    corpus[4096] = 0
    result = s.verify_candidate(corpus, benchmark, s.Candidate(0, "prefix/256"), opportunities=1)
    assert result["old_rule_accepted"]
    assert not result["verified"]
    identity = next(m for m in result["fidelity"]["maps"] if m["method"] == "identity")
    assert identity["off_modal_fraction"] > 0.99
    assert identity["inside_off_modal_fraction"] == 1
    assert identity["outside_off_modal_fraction"] == 0


def test_suffix_window_uses_unmasked_overlap_positions_and_offset():
    benchmark = shapes()["sparse_flag"]
    corpus = np.r_[np.full(37, np.nan), benchmark]
    corpus[40:50] = np.nan
    result = s.verify_candidate(corpus, benchmark, s.Candidate(37, "suffix/256"), opportunities=1)
    assert result["verified"]
    f = result["fidelity"]
    assert f["retrieval_window"] == [3840, 4096]
    assert f["off_modal_inside"] == 0  # Undefined recall is not an invented success.
    assert all(m["inside_off_modal_fraction"] is None for m in f["maps"])


def test_shape_controls_use_production_path_and_report_each_class(tmp_path):
    root, out, state = (tmp_path / name for name in ("corpus", "out", "state"))
    values = shapes()
    native(root / "univariate/demo/train-000.parquet", target=values["sparse_flag"].tolist())
    report, code = s.run_sketch(
        root,
        out,
        manifest(),
        [BenchmarkSeries(kind, kind, "value", "h", b) for kind, b in values.items()],
        state=state,
    )
    assert code == 0 and report["controls"]["ok"]
    cases = {c["class"]: c for c in report["controls"]["cases"]}
    for kind in values:
        for variant in (
            "exact",
            "affine_positive",
            "affine_negative",
            "edited_1pct",
            "affine_edited_1pct",
            "coincidence",
        ):
            case = cases[f"fidelity_{kind}_{variant}"]
            assert case["ok"], case
            assert case["expected"] == (variant != "coincidence")
            if not case["expected"]:
                assert case["rejection_reasons"]["variation_fidelity_rejected"] > 0
    assert report["method"] == "provisional-endpoints-fidelity-2"
    # The negative plants really reach and pass all old verification gates.
    control = json.loads((state / "sketch-controls.json").read_text())["result"]
    for kind in values:
        case = cases[f"fidelity_{kind}_coincidence"]
        records = control["control_evidence"][case["item_id"]]
        assert any(r["old_rule_accepted"] and not r["verified"] for r in records)
    assert control["counts"]["fidelity_rejections"] >= 2
    assert control["fidelity_rejections"]
    assert "Fidelity rejected" in (out / "sketch-report.md").read_text()


def test_fidelity_rejections_have_their_own_capped_sample(tmp_path, monkeypatch):
    # Exercise actual unit journals and folder assembly, rather than fabricated
    # evidence records or a replacement verifier.
    root, out, state = (tmp_path / name for name in ("corpus", "out", "state"))
    corpus, benchmark, _ = sparse_pair()
    for i in range(3):
        native(root / f"univariate/demo/train-{i:03d}.parquet", target=corpus.tolist())
    monkeypatch.setattr(s, "MATCH_SAMPLE", 1)
    monkeypatch.setattr(s, "_execute", s.scan_folder)
    report, _ = s.run_sketch(
        root, out, manifest(), [BenchmarkSeries("flag", "b", "value", "h", benchmark)], state=state
    )
    folder = report["folders"][0]
    assert folder["counts"]["fidelity_rejections"] >= 3
    assert folder["matches"] == []
    assert len(folder["fidelity_rejections"]) == folder["match_sample_limit"] == 1
    assert folder["fidelity_rejections"][0]["old_rule_accepted"]


def test_edited_plants_change_one_percent_of_the_finite_points():
    # Missing benchmark values shrink the compared overlap; 1% of the raw length
    # would exceed the verifier's 99% fraction and fail a correct scanner's control.
    values = np.zeros(2000)
    values[[300, 700, 1100, 1500]] = 1.0
    values[[10, 20, 30, 40, 50]] = np.nan
    kind, variants = s._fidelity_plants(values)
    finite = np.isfinite(values)
    assert kind == "sparse_flag"
    for name in ("edited_1pct", "affine_edited_1pct"):
        planted, expected = variants[name]
        reference = values if name == "edited_1pct" else values * 2.5 + 7
        changed = int((planted[finite] != reference[finite]).sum())
        assert expected and 0 < changed <= int(finite.sum()) // 100
