"""Independently refit reported pairs from native corpus and benchmark spool bytes."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from scripts.corpus_v4.audit_sketch import _open_spool, _series
from scripts.corpus_v4.native import atomic_json, digest, iter_rows

STRONG_ASSOCIATION = dict(min_phi=0.5, min_shared=4, min_finite_pairs=32)
AFFINE_RTOL = AFFINE_ATOL = 1e-10


def _records(report):
    for folder in report["folders"]:
        for record in folder.get("matches", []):
            yield folder, record


def _shard_path(folder, record):
    name, shard = folder["folder"], record["corpus"]["shard"]
    if not re.fullmatch(r"(?:univariate|covariates)/[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("invalid native folder")
    if not re.fullmatch(r"(?:train|validation)-[A-Za-z0-9_-]+[.]parquet", shard):
        raise ValueError("invalid native shard")
    prefix = "quarantine/" if folder.get("status") == "quarantined" else ""
    return prefix + name + "/" + shard


def needed_shards(report):
    """Canonical, deduplicated paths used by the guest's selective download."""
    return sorted({_shard_path(f, r) for f, r in _records(report)})


def _integer(value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"expected integer, got {value!r}")
    return value


def _load_corpus(root, records):
    """Read each needed row group once, stopping at its last requested row."""
    import pyarrow.parquet as pq

    groups, loaded, errors = defaultdict(list), {}, {}
    for i, (folder, record) in enumerate(records):
        try:
            path = Path(root) / _shard_path(folder, record)
            row = _integer(record["corpus"]["row"])
            if row < 0:
                raise ValueError("negative corpus row")
            groups[path].append((i, folder, record, row))
        except Exception as exc:
            errors[i] = str(exc)
    for path, requests in groups.items():
        try:
            by_group = defaultdict(list)
            with pq.ParquetFile(path) as parquet:
                bounds = np.cumsum(
                    [parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)]
                )
            for request in requests:
                i, _, _, row = request
                group = int(np.searchsorted(bounds, row, side="right"))
                if group == len(bounds):
                    errors[i] = (
                        f"corpus row {row} outside shard ({int(bounds[-1]) if len(bounds) else 0} rows)"
                    )
                else:
                    by_group[group].append(request)
            for group, selected in by_group.items():
                wanted = defaultdict(list)
                for i, folder, record, row in selected:
                    wanted[row].append((i, record))
                last = max(wanted)
                try:
                    for rec in iter_rows(
                        path.parent,
                        selected[0][1]["folder"].split("/")[0],
                        shard=path.name,
                        row_group=group,
                    ):
                        row = rec["row_index"]
                        for i, record in wanted.get(row, []):
                            try:
                                meta = record["corpus"]
                                for key in ("item_id", "freq", "source"):
                                    if key in meta and rec["row"][key] != meta[key]:
                                        raise ValueError(f"corpus {key} mismatch at row {row}")
                                column, channel = meta["column"], _integer(meta["channel"])
                                if column not in {"target", "covariates"} or channel < 0:
                                    raise ValueError("invalid corpus channel")
                                loaded[i] = np.asarray(
                                    rec[column][channel], dtype=np.float64
                                ).copy()
                            except Exception as exc:
                                errors[i] = str(exc)
                        if row >= last:
                            break
                except Exception as exc:
                    for i, _, _, _ in selected:
                        if i not in loaded:
                            errors[i] = str(exc)
        except Exception as exc:
            for i, _, _, _ in requests:
                errors[i] = str(exc)
    return loaded, errors


def _owner_key(meta):
    return tuple(meta[k] for k in ("dataset", "item_id", "column"))


def _owners(db, records):
    # One streaming scan: the spool has no metadata index. Never assume rowid
    # is the report position, and never silently pick an ambiguous owner.
    wanted = {_owner_key(r["benchmark"]) for _, r in records}
    owners = defaultdict(list)
    for owner, encoded in db.execute("SELECT rowid, meta FROM series"):
        key = _owner_key(json.loads(encoded))
        if key in wanted:
            owners[key].append(owner)
    return owners


def _align(record, x, y, output):
    """corpus index = benchmark index + offset; bounds are half-open.

    audit_sketch.verify_candidate defines the full overlap as
    [max(0, offset), min(len(corpus), len(benchmark) + offset)).
    audit_match.fit_segment records t_* on corpus and index_* on benchmark,
    with segment counting the joint finite mask. No verifier is called here.
    """
    fit, offset = record["fit"], _integer(record["offset"])
    ts, te, bs, be, segment = [
        _integer(fit[k]) for k in ("t_start", "t_end", "index_start", "index_end", "segment")
    ]
    output.update(
        offset=offset,
        t_start=ts,
        t_end=te,
        index_start=bs,
        index_end=be,
        expected_pairs=segment,
        corpus_length=len(x),
        benchmark_length=len(y),
    )
    if not (0 <= ts < te <= len(x) and 0 <= bs < be <= len(y)):
        raise ValueError("stored alignment bounds outside original arrays")
    if ts - bs != offset or te - be != offset:
        raise ValueError("stored alignment bounds contradict offset")
    if (ts, te) != (max(0, offset), min(len(x), len(y) + offset)):
        raise ValueError("stored alignment is not the fingerprint pass's full overlap")
    a, b = x[ts:te], y[bs:be]
    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    output.update(
        raw_pairs=len(a),
        finite_pairs=n,
        dropped_nonfinite=len(a) - n,
        position_convention="zero-based position in the unmasked overlap",
    )
    if n != segment:
        raise ValueError(f"joint finite overlap {n} differs from stored segment {segment}")
    if n < 2:
        raise ValueError("fewer than two finite aligned pairs")
    return a[finite], b[finite], np.flatnonzero(finite)


def _fit(x, y):
    """Centered ordinary least squares, with no floor on target variance.

    Constant targets have undefined r²; Pearson is undefined if either side
    is constant. Exact modes and equality use the original float64 values.
    """
    xm, ym = float(x.mean()), float(y.mean())
    dx, dy = x - xm, y - ym
    xx, yy, xy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    slope = xy / xx if xx > 0 else 0.0
    residual = dy - slope * dx
    return dict(
        coef=[slope, ym - slope * xm],
        r2=float(1 - (residual @ residual) / yy) if yy > 0 else None,
        pearson=float(np.clip(xy / np.sqrt(xx * yy), -1, 1)) if xx > 0 and yy > 0 else None,
        max_resid=float(np.abs(residual).max()),
        target_range=float(np.ptp(y)),
        within_range_frac=float(np.mean(np.abs(residual) <= 0.005 * np.ptp(y))),
    )


def _stats(values):
    levels, counts = np.unique(values, return_counts=True)
    # A tie deterministically chooses the smallest value.
    mode = int(np.argmax(counts))
    return dict(
        distinct=len(levels),
        mode=float(levels[mode]),
        mode_count=int(counts[mode]),
        mode_share=float(counts[mode] / len(values)),
    )


def _association(ao, bo):
    """Fixed full-overlap modes; the tail is a reference, not a calibrated test."""
    n = len(ao)
    a, b, k = int(ao.sum()), int(bo.sum()), int((ao & bo).sum())
    denominator = math.sqrt(a * b * (n - a) * (n - b))
    return dict(
        finite_pairs=n,
        corpus_count=a,
        benchmark_count=b,
        intersection_count=k,
        expected_intersection=a * b / n if n else None,
        phi=(n * k - a * b) / denominator if denominator else None,
        hypergeom_upper_tail=float(hypergeom.sf(k - 1, n, a, b)) if n else None,
    )


def _retrieval_window(proposal, positions, raw_pairs, ao, bo):
    match = re.fullmatch(r"(prefix|suffix)/([1-9][0-9]*)", proposal or "")
    window = dict(proposal=proposal, inside=None, outside=None)
    if match is None:
        window["note"] = "whole overlap only: proposal has no recognized retrieval window"
        return window
    end, length = match[1], int(match[2])
    start, stop = (
        (0, min(length, raw_pairs)) if end == "prefix" else (max(0, raw_pairs - length), raw_pairs)
    )
    inside = (positions >= start) & (positions < stop)
    window.update(
        end=end,
        requested_pairs=length,
        position_start=start,
        position_end=stop,
        inside=_association(ao[inside], bo[inside]),
        outside=_association(ao[~inside], bo[~inside]),
    )
    return window


def _retrieval_end_run(a, b, positions, raw_pairs, window):
    """Maximal endpoint-connected agreement on finite pairs, with gaps retained."""
    if "end" not in window:
        return None
    suffix = window["end"] == "suffix"
    x, y = (a[::-1], b[::-1]) if suffix else (a, b)
    different = np.flatnonzero(x != x[0])
    slope = float((y[different[0]] - y[0]) / (x[different[0]] - x[0])) if len(different) else 0.0
    predicted = y[0] + slope * (x - x[0])

    def span(agrees):
        failures = np.flatnonzero(~agrees)
        count = int(failures[0]) if len(failures) else len(x)
        return dict(
            finite_pairs=count,
            position_start=int(positions[len(x) - count - 1]) + 1
            if suffix and count < len(x)
            else 0,
            position_end=int(positions[count]) if not suffix and count < len(x) else raw_pairs,
        )

    affine = span(np.isclose(y, predicted, rtol=AFFINE_RTOL, atol=AFFINE_ATOL))
    affine["coef"] = [slope, float(y[0] - slope * x[0])]
    return dict(
        end=window["end"],
        position_convention="half-open positions in the unmasked overlap; nonfinite pairs skipped",
        affine_rtol=AFFINE_RTOL,
        affine_atol=AFFINE_ATOL,
        affine=affine,
        exact=span(x == y),
    )


def _measure(record, x, y, result):
    a, b, positions = _align(record, x, y, result["alignment"])
    astat, bstat = _stats(a), _stats(b)
    ao, bo = a != astat["mode"], b != bstat["mode"]
    nb, common = int(bo.sum()), int((ao & bo).sum())
    association = _association(ao, bo)
    window = _retrieval_window(
        record.get("proposal"), positions, result["alignment"]["raw_pairs"], ao, bo
    )

    def pairs(indices):
        return [
            dict(
                position=int(positions[i]),
                corpus_index=int(positions[i]) + record["fit"]["t_start"],
                benchmark_index=int(positions[i]) + record["fit"]["index_start"],
                corpus=float(a[i]),
                benchmark=float(b[i]),
            )
            for i in indices
        ]

    forward, reverse = _fit(a, b), _fit(b, a)
    reported = result["reported_fit"]["r2"]
    result.update(
        alignment_ok=True,
        benchmark_on_corpus=forward,
        corpus_on_benchmark=reverse,
        corpus_stats=astat,
        benchmark_stats=bstat,
        off_modal=dict(
            **association,
            corpus_positions=positions[ao].tolist(),
            benchmark_positions=positions[bo].tolist(),
            benchmark_reproduced_frac=common / nb if nb else None,
        ),
        retrieval_window=window,
        retrieval_end_run=_retrieval_end_run(
            a, b, positions, result["alignment"]["raw_pairs"], window
        ),
        exact_equal_frac=float(np.mean(a == b)),
        first_pairs=pairs(range(min(120, len(a)))),
        benchmark_off_modal_pairs=pairs(np.flatnonzero(bo)[:200]),
        r2_delta=forward["r2"] - reported
        if forward["r2"] is not None and reported is not None
        else None,
    )


def _markdown(payload):
    s = payload["summary"]
    lines = [
        f"Pairs with r² ≥ 0.5 in either direction: {s['r2_at_least_half']}/{s['records']}",
        f"Pairs with strong off-modal association outside retrieval window: {s['strong_outside_retrieval_window']}/{s['records_with_retrieval_window']}"
        f" records with a recognized window ({s['records'] - s['records_with_retrieval_window']} of {s['records']} have none or did not align and are not counted here)",
        f"Pairs differing from reported r² by > 0.01: {s['r2_differs_over_001']}/{s['records']}",
        "",
        "Strong association is descriptive, not a significance test: φ ≥ 0.5 (large positive association), shared ≥ 4 (exclude single-event coincidences), and n ≥ 32 finite outside-window pairs (exclude tiny remainders).",
        "E = a·b/n; φ = (n·k − a·b)/√(a·b·(n−a)·(n−b)); hypergeometric upper tail = P(K ≥ k | n,a,b). Serial dependence and retrieval selection prevent interpreting this reference tail as calibrated significance.",
        "Full-overlap exact modes define off-modal events in every region; degenerate φ and empty-region statistics are undefined. prefix/N excludes the first N unmasked overlap positions; suffix/N excludes the last N. Other proposals: whole overlap only, excluded from the outside-window headline.",
        "The window is that of the one proposal the fingerprint pass recorded for the offset; if several key sizes matched, positions outside the recorded window can still lie inside another matched key.",
        "",
        f"artifact_type: {payload['artifact_type']}",
        "provisional: true",
        "release_eligible: false",
        f"input_fingerprint: {json.dumps(payload['input_fingerprint'], sort_keys=True)}",
        f"benchmark_revision: {json.dumps(payload['benchmark_revision'], sort_keys=True)}",
        f"method (source report): {json.dumps(payload['method'], sort_keys=True)}",
        f"repository_commit: {payload['repository_commit'] or 'not supplied'}",
        "",
        "Input files (SHA256):",
        "```json",
        json.dumps(payload["provenance"], sort_keys=True, indent=2),
        "```",
        "",
        f"Alignment failures: {s['alignment_failures']}. Off-modal overlap counts shared event positions, not matching amplitudes.",
        "Modes use exact values (ties: smallest value); r²/Pearson are undefined for constant targets/series. Positions retain missing-value gaps.",
        "",
        "| # | Corpus (shard, row, column, channel) | Benchmark (dataset, item, column) | n | r² B←C | r² C←B | Pearson | Off C / B | Shared / chance | B reproduced | Equal | Within 0.5% | Report r² / Δ | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def number(value):
        return "undefined" if value is None else f"{value:.6g}"

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    for i, rec in enumerate(payload["records"], 1):
        c, b = rec["corpus"], rec["benchmark"]
        row = [
            str(i),
            cell(f"{rec['folder']}/{c['shard']}:{c['row']}:{c['column']}:{c['channel']}"),
            cell(f"{b['dataset']}:{b['item_id']}:{b['column']}"),
        ]
        if rec["alignment_ok"]:
            fwd, rev, off = rec["benchmark_on_corpus"], rec["corpus_on_benchmark"], rec["off_modal"]
            row += [
                str(rec["alignment"]["finite_pairs"]),
                number(fwd["r2"]),
                number(rev["r2"]),
                number(fwd["pearson"]),
                f"{off['corpus_count']} / {off['benchmark_count']}",
                f"{off['intersection_count']} / {number(off['expected_intersection'])}",
                number(off["benchmark_reproduced_frac"]),
                number(rec["exact_equal_frac"]),
                number(fwd["within_range_frac"]),
                f"{number(rec['reported_fit']['r2'])} / {number(rec['r2_delta'])}",
                "OK",
            ]
        else:
            row += ["—"] * 10 + ["ERROR: " + cell(rec["error"])]
        lines.append("| " + " | ".join(row) + " |")
    for i, rec in enumerate(payload["records"], 1):
        if not rec["alignment_ok"]:
            continue
        window = rec["retrieval_window"]
        lines += [
            "",
            f"Pair {i}, proposal: {cell(window['proposal'])}",
            "",
            "| Region | n | Off C (a) | Off B (b) | Shared (k) | Expected (E) | φ | Hypergeometric upper tail |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, stats in (
            ("whole overlap", rec["off_modal"]),
            ("inside retrieval window", window["inside"]),
            ("outside retrieval window", window["outside"]),
        ):
            if stats is not None:
                lines.append(
                    "| "
                    + " | ".join(
                        [label]
                        + [
                            number(stats[k])
                            for k in (
                                "finite_pairs",
                                "corpus_count",
                                "benchmark_count",
                                "intersection_count",
                                "expected_intersection",
                                "phi",
                                "hypergeom_upper_tail",
                            )
                        ]
                    )
                    + " |"
                )
        run = rec["retrieval_end_run"]
        if run is None:
            lines.append(window["note"])
            continue
        lines.append(
            f"Retrieval window positions [{window['position_start']}, {window['position_end']}); longest runs from the {run['end']} end on finite pairs (half-open unmasked positions, gaps retained):"
        )
        for label in ("affine", "exact"):
            span = run[label]
            lines.append(
                f"{label}: [{span['position_start']}, {span['position_end']}), {span['finite_pairs']} finite pairs."
            )
        lines.append(
            f"Affine B←C coefficients: {run['affine']['coef']}; rtol={run['affine_rtol']}, atol={run['affine_atol']} (floating-point agreement). Exact equality has no tolerance."
        )
    return "\n".join(lines) + "\n"


def _file_evidence(path, *, name=None):
    evidence = dict(path=name or str(path), sha256=None)
    try:
        evidence["sha256"] = digest(path)
    except OSError as exc:
        evidence["error"] = str(exc)
    return evidence


def run_refit(report, corpus, spool, out, *, report_path=None, repo_commit=None):
    records = list(_records(report))
    shards = set()
    for folder, record in records:
        try:
            shards.add(_shard_path(folder, record))
        except ValueError:
            pass  # _load_corpus records invalid paths as per-record failures.
    provenance = dict(
        report=_file_evidence(report_path)
        if report_path is not None
        else dict(sha256=None, error="report path not supplied"),
        spool_sqlite=_file_evidence(spool),
        spool_values=_file_evidence(Path(spool).with_suffix(".values")),
        shards=[_file_evidence(Path(corpus) / path, name=path) for path in sorted(shards)],
    )
    channels, errors = _load_corpus(corpus, records)
    results = []
    db = None
    try:
        try:
            db = _open_spool(spool)
            owners = _owners(db, records)
            spool_error = None
        except Exception as exc:
            owners, spool_error = {}, str(exc)
        for i, (folder, record) in enumerate(records):
            result = dict(
                folder=folder["folder"],
                corpus=record["corpus"],
                benchmark=record["benchmark"],
                offset=record["offset"],
                alignment_ok=False,
                alignment={},
                reported_fit={k: record["fit"].get(k) for k in ("r2", "exact_frac", "max_resid")},
            )
            try:
                if i in errors:
                    raise ValueError(errors[i])
                if spool_error is not None:
                    raise ValueError(spool_error)
                found = owners[_owner_key(record["benchmark"])]
                if len(found) != 1:
                    raise ValueError(f"expected one benchmark owner, found {len(found)}")
                meta, y, _ = _series(db, spool, found[0], calendar_dimensions=False)
                if meta.get("freq") != record["benchmark"].get("freq"):
                    raise ValueError("benchmark frequency mismatch")
                _measure(record, channels[i], y, result)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
    finally:
        if db is not None:
            db.close()
    good = [r for r in results if r["alignment_ok"]]
    summary = dict(
        records=len(results),
        alignment_failures=len(results) - len(good),
        r2_at_least_half=sum(
            any(
                r[d]["r2"] is not None and r[d]["r2"] >= 0.5
                for d in ("benchmark_on_corpus", "corpus_on_benchmark")
            )
            for r in good
        ),
        strong_outside_retrieval_window=sum(
            outside is not None
            and outside["phi"] is not None
            and outside["phi"] >= STRONG_ASSOCIATION["min_phi"]
            and outside["intersection_count"] >= STRONG_ASSOCIATION["min_shared"]
            and outside["finite_pairs"] >= STRONG_ASSOCIATION["min_finite_pairs"]
            for r in good
            for outside in [r["retrieval_window"]["outside"]]
        ),
        records_with_retrieval_window=sum(
            r["retrieval_window"]["outside"] is not None for r in good
        ),
        strong_association_thresholds=STRONG_ASSOCIATION,
        r2_differs_over_001=sum(
            r["r2_delta"] is not None and abs(r["r2_delta"]) > 0.01 for r in good
        ),
    )
    payload = dict(
        artifact_type="corpus-match-refit",
        provisional=True,
        release_eligible=False,
        input_fingerprint=report.get("input_fingerprint"),
        benchmark_revision=report.get("benchmark_revision"),
        method=report.get("method"),
        repository_commit=repo_commit,
        association_interpretation=(
            "Off-modal associations and strong-pair thresholds are descriptive, not a significance test. "
            "All regions use full-overlap exact modes. Hypergeometric upper tails are fixed-marginal "
            "references, uncalibrated under serial dependence and retrieval selection."
        ),
        provenance=provenance,
        summary=summary,
        records=results,
    )
    out = Path(out)
    atomic_json(out / "refit.json", payload)
    temporary = out / "refit.md.tmp"
    temporary.write_text(_markdown(payload))
    temporary.replace(out / "refit.md")
    return 1 if summary["alignment_failures"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("spool", type=Path, help="benchmark.sqlite")
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo-commit", help="repository commit supplied by the pinned job")
    args = parser.parse_args(argv)
    return run_refit(
        json.loads(args.report.read_text()),
        args.corpus,
        args.spool,
        args.out,
        report_path=args.report,
        repo_commit=args.repo_commit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
