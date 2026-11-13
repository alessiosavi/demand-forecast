"""Render a separate provisional-triage guest job; never touch the corpus gate."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.corpus_v4.make_corpus_job import ENVIRONMENT, REPOSITORY
from scripts.corpus_v4.native import memory_budget_arg


def _prefix(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", value
    ):
        raise ValueError("prefix must be a plain gs:// bucket path")
    if any(part in {"", ".", ".."} for part in value[5:].split("/")):
        raise ValueError("prefix must be canonical without empty/dot path components")
    return value.rstrip("/")


def _overlap(a, b):
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def render(
    repo_sha,
    corpus_prefix,
    report_prefix,
    *,
    state_prefix=None,
    benchmark_state_prefix=None,
    workers=16,
    memory_budget_gib=104,
):
    """Generate reviewable shell for the RUN_DIR/STATE_URI host runner contract.

    Host restores state before invocation and periodically/preemption-copies it.
    Corpus is outside state: immutable published bytes are fetched from GCS only.
    A legacy benchmark spool without byte/revision identity is rebuilt from the
    pinned benchmark loader; corpus data has no Hub download operation.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("repo SHA must be a full lowercase commit hash")
    corpus_prefix, report_prefix = _prefix(corpus_prefix), _prefix(report_prefix)
    if _overlap(corpus_prefix, report_prefix):
        raise ValueError("report prefix must be separate from the published corpus")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    memory_budget_arg(memory_budget_gib)
    if state_prefix is not None:
        state_prefix = _prefix(state_prefix)
        if _overlap(state_prefix, corpus_prefix) or _overlap(state_prefix, report_prefix):
            raise ValueError("state, report and corpus prefixes must be separate")
    benchmark_state_prefix = _prefix(benchmark_state_prefix) if benchmark_state_prefix else ""
    expected_state = state_prefix or ""
    return f"""#!/usr/bin/env bash
# PROVISIONAL triage only. This job cannot certify training or release.
set -euo pipefail
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export a separate STATE_URI}}"
REPO_SHA='{repo_sha}'
CORPUS_URI='{corpus_prefix}'
REPORT_URI='{report_prefix}'
EXPECTED_STATE_URI='{expected_state}'
BENCHMARK_STATE_URI='{benchmark_state_prefix}'
# Check external destinations before installation, transfer or publication.
python3 - "$CORPUS_URI" "$REPORT_URI" "$STATE_URI" "$EXPECTED_STATE_URI" <<'SKETCH_DESTINATIONS'
import re, sys
uris = [s.rstrip("/") for s in sys.argv[1:4]]
for uri in uris:
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", uri):
        raise ValueError("invalid sketch destination")
    if any(p in ("", ".", "..") for p in uri[5:].split("/")):
        raise ValueError("noncanonical sketch destination")
for i, a in enumerate(uris):
    for b in uris[i+1:]:
        if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
            raise ValueError("sketch destinations overlap; refusing to touch an audit/corpus gate")
if sys.argv[4] and uris[2] != sys.argv[4]:
    raise ValueError("STATE_URI differs from the reviewed job")
SKETCH_DESTINATIONS
mkdir -p "$RUN_DIR/state"
export HOME="${{HOME:-$RUN_DIR/home}}"
export HF_HOME="$RUN_DIR/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
mkdir -p "$HOME" "$HF_HOME"
CORPUS="$RUN_DIR/sketch-corpus"
REPORT="$RUN_DIR/state/sketch"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$REPORT"
rm -f "$REPORT/sketch-report.json" "$REPORT/sketch-report.md"
write_status() {{
    python3 - "$REPORT/sketch-status.json" "$ATTEMPT_ID" "$REPO_SHA" "$1" <<'SKETCH_STATUS'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
record = dict(artifact_type="corpus-sketch-triage", provisional=True, release_eligible=False,
              attempt_id=sys.argv[2], repo_sha=sys.argv[3], process_exit_code=int(sys.argv[4]),
              execution_ok=sys.argv[4] == "0")
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(record, sort_keys=True) + "\\n")
tmp.replace(path)
SKETCH_STATUS
}}
finish() {{
    status=$?
    trap - EXIT
    set +e
    # Evidence precedes the final completion status. Upload failures stay failures.
    gcloud storage rsync --recursive --exclude='[.]tmp$|sketch-status[.]json$' "$REPORT" "$REPORT_URI" || status=3
    gcloud storage rsync --recursive --exclude='[.]tmp$|sketch-status[.]json$' "$RUN_DIR/state" "$STATE_URI" || status=3
    write_status "$status" || status=3
    gcloud storage cp "$REPORT/sketch-status.json" "$REPORT_URI/sketch-status.json" || status=3
    gcloud storage cp "$REPORT/sketch-status.json" "$STATE_URI/sketch/sketch-status.json" || status=3
    if (( status != 0 )); then
        write_status "$status"
        # A final marker copy can fail after its peer published success.
        # Best-effort replace both with failure; retain the nonzero process exit.
        gcloud storage cp "$REPORT/sketch-status.json" "$REPORT_URI/sketch-status.json" || true
        gcloud storage cp "$REPORT/sketch-status.json" "$STATE_URI/sketch/sketch-status.json" || true
    fi
    exit "$status"
}}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_status -1
gcloud storage cp "$REPORT/sketch-status.json" "$REPORT_URI/sketch-status.json"
gcloud storage cp "$REPORT/sketch-status.json" "$STATE_URI/sketch/sketch-status.json"
CHECKOUT="$RUN_DIR/corpus-repo-$REPO_SHA"
if [[ ! -d "$CHECKOUT/.git" ]]; then
    git clone --no-checkout '{REPOSITORY}' "$CHECKOUT"
fi
cd "$CHECKOUT"
git fetch origin "$REPO_SHA"
git checkout --detach "$REPO_SHA"
test "$(git rev-parse HEAD)" = "$REPO_SHA"
{ENVIRONMENT}\
# There is no corpus Hub path: restore the published prefix, then hash every object.
gcloud storage rsync --recursive "$CORPUS_URI" "$CORPUS"
# Benchmark downloads alone may use the pinned revision loader on a cold cache.
# Reuse old spool bytes only with a revision-bound identity marker.
if [[ -n "$BENCHMARK_STATE_URI" && ! -f "$RUN_DIR/state/benchmark.sketch-identity.json" ]]; then
    if gcloud storage ls "$BENCHMARK_STATE_URI/benchmark.sketch-identity.json" >/dev/null 2>&1; then
        for name in benchmark.sqlite benchmark.values benchmark.stamps benchmark.sketch-identity.json; do
            gcloud storage cp "$BENCHMARK_STATE_URI/$name" "$RUN_DIR/state/$name"
        done
    else
        echo 'Benchmark spool has no verifiable identity marker; rebuild from pinned cache/loader.' >&2
        gcloud storage rsync --recursive "$BENCHMARK_STATE_URI/benchmark-cache" "$RUN_DIR/state/benchmark-cache"
    fi
fi
# audit_sketch CLI invokes verify_published BEFORE any benchmark loading/scanning.
python -m scripts.corpus_v4.audit_sketch "$CORPUS" "$REPORT" --state "$RUN_DIR/state" \\
    --workers {workers} --memory-budget-gib {float(memory_budget_gib):g}
test -f "$REPORT/sketch-report.json"
test -f "$REPORT/sketch-report.md"
python - "$REPORT/sketch-report.json" <<'SKETCH_CONTRACT'
import json, sys
report = json.load(open(sys.argv[1]))
if (report.get("artifact_type") != "corpus-sketch-triage" or report.get("provisional") is not True
        or report.get("release_eligible") is not False or not report.get("execution_ok")
        or "valid" in report or "exit_code" in report):
    raise RuntimeError("invalid provisional sketch report")
SKETCH_CONTRACT
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_sha")
    parser.add_argument("corpus_prefix")
    parser.add_argument("report_prefix")
    parser.add_argument("out", type=Path)
    parser.add_argument("--state-prefix")
    parser.add_argument("--benchmark-state-prefix")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--memory-budget-gib", type=float, default=104)
    args = parser.parse_args(argv)
    source = render(
        args.repo_sha,
        args.corpus_prefix,
        args.report_prefix,
        state_prefix=args.state_prefix,
        benchmark_state_prefix=args.benchmark_state_prefix,
        workers=args.workers,
        memory_budget_gib=args.memory_budget_gib,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
