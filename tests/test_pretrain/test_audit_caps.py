"""Deterministic limits on candidate work, with uncapped differential oracles."""

import json
from copy import deepcopy

import numpy as np
import pytest

from demand_forecast.pretrain import audit
from demand_forecast.pretrain import audit_match as match
from tests.helpers.audit_mask_reference import verify as mask_verify

CAP_COUNTERS = (
    "groups_capped_rows",
    "groups_dropped",
    "offsets_capped_candidates",
    "offsets_dropped",
)


def _scan(targets, index):
    cov, pairs, records = {}, [], []
    match.scan("caps", "target", targets, index, cov, pairs, records)
    return cov["caps::target"], pairs, records


@pytest.mark.parametrize(
    "row_cap,offset_cap,kept,counts",
    [
        (4, 3, {(0, 0), (1, 0), (0, -1), (0, 1)}, (1, 158, 0, 0)),
        (1000, 3, {(m, off) for m in (0, 1) for off in (0, -1, 1)}, (0, 0, 2, 152)),
        (10, 3, {(m, off) for m in (0, 1) for off in (0, -1, 1)}, (1, 152, 2, 4)),
    ],
)
def test_scan_caps_rank_groups_and_merge_duplicate_counters(
    monkeypatch, row_cap, offset_cap, kept, counts
):
    target = np.arange(48, dtype=float)
    index = match.Index([target, target.copy()], ["first", "second"])
    targets = [target, target.copy()]
    # Each series has 81 groups from affine/runs8 hits; runs exceed CAP["runs"].
    # Offset 0 has 123 hits, +/-1 have 80, +/-2 have 78. The two extreme
    # offsets per series fail the minimum-hit/span check, leaving 79 for verify.
    monkeypatch.setattr(match, "ROW_GROUP_CAP", 1000)
    monkeypatch.setattr(match, "VERIFY_OFFSET_CAP", 1000)
    uncapped_cov, uncapped_pairs, uncapped_records = _scan(targets, index)
    assert uncapped_records == []

    monkeypatch.setattr(match, "ROW_GROUP_CAP", row_cap)
    monkeypatch.setattr(match, "VERIFY_OFFSET_CAP", offset_cap)
    capped = _scan(targets, index)
    cov, pairs, records = capped
    assert {(p["index_series_index"], p["offset"]) for p in pairs} == kept
    assert pairs == [p for p in uncapped_pairs if (p["index_series_index"], p["offset"]) in kept]
    assert records == []
    assert cov["unique_series"] == 1
    assert cov["n_series"] == 2
    assert cov["candidates"] == cov["copies"] == uncapped_cov["copies"] == 4
    assert tuple(cov[key] for key in CAP_COUNTERS) == tuple(2 * n for n in counts)
    assert all(type(cov[key]) is int for key in CAP_COUNTERS)
    assert json.loads(json.dumps(cov)) == cov
    assert _scan(targets, index) == capped


def test_fallback_postpass_still_applies_candidate_caps(monkeypatch):
    target = np.arange(48, dtype=float)
    monkeypatch.setattr(match, "CAP", dict.fromkeys(match.FAMILIES, 0))
    monkeypatch.setattr(match, "FALLBACK_TOPK", 1)
    monkeypatch.setattr(match, "ROW_GROUP_CAP", 4)
    monkeypatch.setattr(match, "VERIFY_OFFSET_CAP", 1)
    index = match.Index([target, target.copy()], ["first", "second"])
    cov, pairs, records = _scan([target], index)
    assert cov["initial_truncated_rows"] == cov["postpass_rows"] == 1
    assert cov["unresolved_rows"] == 0
    # Fallback includes the four-run hits: 85 eligible offsets per series.
    assert cov["fallback_candidates"] == 170
    assert tuple(cov[key] for key in CAP_COUNTERS) == (1, 166, 1, 2)
    assert [(p["index_series_index"], p["offset"]) for p in pairs] == [(0, 0), (1, 0)]
    assert all(p["is_copy"] for p in pairs)
    assert records == []


def _copy_offsets():
    target = np.random.default_rng(42).normal(size=64)
    source = np.tile(target, 7)
    # The strongest copy comes late; three four-hit offsets tie at the cutoff.
    counts = [3, 6, 4, 4, 4, 8, 3]
    hits = [
        ("exact", pos, pos + 64 * block, 8)
        for block, count in enumerate(counts)
        for pos in range(0, 8 * count, 8)
    ]
    return target, source, hits


@pytest.mark.parametrize("reverse", [False, True])
def test_verify_caps_offsets_before_fitting_and_keeps_strong_copy(monkeypatch, reverse):
    target, source, hits = _copy_offsets()
    if reverse:
        hits.reverse()
    monkeypatch.setattr(match, "VERIFY_OFFSET_CAP", 3)
    # Eight hits win, then six, then the lowest offset among the four-hit ties.
    kept = [-320, -64, -256]
    restricted = [hit for off in kept for hit in hits if hit[1] - hit[2] == off]
    expected = mask_verify(target, source, restricted)
    assert expected[0]["is_copy"] and expected[0]["offset"] == -320

    # Observe the actual fits as well as records: discarded offsets must never
    # incur fitting cost, even if their records could later be filtered away.
    fitted = []
    fit_segment = match.fit_segment

    def track_fit(t, s, a0, a1, off, dim=None):
        fitted.append(off)
        return fit_segment(t, s, a0, a1, off, dim)

    monkeypatch.setattr(match, "fit_segment", track_fit)
    assert match.verify(target, source, hits) == expected
    assert fitted == kept
    stats = {"unrelated": 17}
    for _ in range(2):
        assert match.verify(target, source, hits, stats=stats) == expected
    assert stats == {"unrelated": 17, "offsets_capped_candidates": 2, "offsets_dropped": 8}
    assert all(type(value) is int for value in stats.values())


@pytest.mark.parametrize("cap", [7, 8])
def test_verify_at_or_below_cap_matches_frozen_reference(monkeypatch, cap):
    target, source, hits = _copy_offsets()
    monkeypatch.setattr(match, "VERIFY_OFFSET_CAP", cap)
    expected = mask_verify(target, source, hits)
    stats = {"offsets_capped_candidates": 5, "offsets_dropped": 11}
    assert match.verify(target, source, hits, stats=stats) == expected
    assert stats == {"offsets_capped_candidates": 5, "offsets_dropped": 11}
    assert match.verify(target, source, hits) == expected


@pytest.mark.parametrize("row_cap,offset_cap", [(2, 1), (3, 2)])
def test_scan_at_or_below_caps_matches_frozen_reference(monkeypatch, row_cap, offset_cap):
    target = np.random.default_rng(1234).normal(size=32)
    index = match.Index([target.copy(), 3 * target + 10], ["exact", "affine"])
    monkeypatch.setattr(match, "ROW_GROUP_CAP", row_cap)
    monkeypatch.setattr(match, "VERIFY_OFFSET_CAP", offset_cap)
    actual = _scan([target, target.copy()], index)
    assert actual[1] and all(p["is_copy"] for p in actual[1])
    assert all(actual[0][key] == 0 for key in CAP_COUNTERS)

    def reference(t, s, hits, dim=None, stats=None):
        return mask_verify(t, s, hits, dim)

    monkeypatch.setattr(match, "verify", reference)
    assert actual == _scan([target, target.copy()], index)


@pytest.mark.parametrize("name", ["ROW_GROUP_CAP", "VERIFY_OFFSET_CAP"])
def test_caps_are_recorded_in_method_fingerprint(monkeypatch, name):
    before = deepcopy(audit.constants())
    assert name in before
    monkeypatch.setattr(match, name, before[name] + 1)
    after = audit.constants()
    assert after[name] == before[name] + 1
    assert audit._fingerprint(after) != audit._fingerprint(before)
