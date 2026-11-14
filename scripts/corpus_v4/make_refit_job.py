"""Render a small raw-array refit guest under the RUN_DIR/STATE_URI contract."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.corpus_v4.make_corpus_job import ENVIRONMENT, REPOSITORY
from scripts.corpus_v4.make_sketch_job import _overlap, _prefix


def render(repo_sha, report_uri, corpus_prefix, spool_state_prefix, output_prefix):
    if not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("repo SHA must be a full lowercase commit hash")
    report_uri, corpus_prefix, spool_state_prefix, output_prefix = map(
        _prefix, (report_uri, corpus_prefix, spool_state_prefix, output_prefix)
    )
    if any(_overlap(output_prefix, uri) for uri in (report_uri, corpus_prefix, spool_state_prefix)):
        raise ValueError("refit output must be separate from every input")
    return f"""#!/usr/bin/env bash
set -euo pipefail
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export a separate STATE_URI}}"
REPO_SHA='{repo_sha}'
REPORT_URI='{report_uri}'
CORPUS_URI='{corpus_prefix}'
SPOOL_URI='{spool_state_prefix}'
OUTPUT_URI='{output_prefix}'
# Validate the host's write destination before installation or transfers.
python3 - "$STATE_URI" "$REPORT_URI" "$CORPUS_URI" "$SPOOL_URI" "$OUTPUT_URI" <<'REFIT_DESTINATIONS'
import re, sys
state = sys.argv[1].rstrip("/")
if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", state):
    raise ValueError("invalid STATE_URI")
if any(p in ("", ".", "..") for p in state[5:].split("/")):
    raise ValueError("noncanonical STATE_URI")
for other in sys.argv[2:]:
    if state == other or state.startswith(other + "/") or other.startswith(state + "/"):
        raise ValueError("STATE_URI overlaps refit inputs/output")
REFIT_DESTINATIONS
mkdir -p "$RUN_DIR/state"
export HOME="${{HOME:-$RUN_DIR/home}}"
export HF_HOME="$RUN_DIR/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
mkdir -p "$HOME" "$HF_HOME"
WORK="$RUN_DIR/refit-work"
CORPUS="$WORK/corpus"
mkdir -p "$CORPUS"
# Keep only this attempt's completed evidence for the guard's periodic copy.
rm -f "$RUN_DIR/state/refit.json" "$RUN_DIR/state/refit.md"
finish() {{
    status=$?
    trap - EXIT
    set +e
    for name in refit.json refit.md; do
        if [[ -f "$RUN_DIR/state/$name" ]]; then
            gcloud storage cp "$RUN_DIR/state/$name" "$OUTPUT_URI/$name" || status=3
            gcloud storage cp "$RUN_DIR/state/$name" "$STATE_URI/$name" || status=3
        fi
    done
    exit "$status"
}}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
CHECKOUT="$RUN_DIR/corpus-repo-$REPO_SHA"
if [[ ! -d "$CHECKOUT/.git" ]]; then
    git clone --no-checkout '{REPOSITORY}' "$CHECKOUT"
fi
cd "$CHECKOUT"
git fetch origin "$REPO_SHA"
git checkout --detach "$REPO_SHA"
test "$(git rev-parse HEAD)" = "$REPO_SHA"
{ENVIRONMENT}\
gcloud storage cp "$REPORT_URI" "$WORK/report.json"
for name in benchmark.sqlite benchmark.values benchmark.stamps; do
    gcloud storage cp "$SPOOL_URI/$name" "$WORK/$name"
done
python - "$WORK/report.json" > "$WORK/shards.txt" <<'REFIT_SHARDS'
import json, sys
from scripts.corpus_v4.refit_matches import needed_shards
for path in needed_shards(json.load(open(sys.argv[1]))):
    print(path)
REFIT_SHARDS
while IFS= read -r shard; do
    mkdir -p "$CORPUS/$(dirname "$shard")"
    gcloud storage cp "$CORPUS_URI/$shard" "$CORPUS/$shard"
done < "$WORK/shards.txt"
# Nonzero alignment status still reaches finish(), which publishes both reports.
python -m scripts.corpus_v4.refit_matches "$WORK/report.json" "$CORPUS" "$WORK/benchmark.sqlite" "$RUN_DIR/state" --repo-commit "$REPO_SHA"
test -f "$RUN_DIR/state/refit.json"
test -f "$RUN_DIR/state/refit.md"
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo_sha", "report_uri", "corpus_prefix", "spool_state_prefix", "output_prefix"):
        parser.add_argument(name)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    source = render(
        args.repo_sha,
        args.report_uri,
        args.corpus_prefix,
        args.spool_state_prefix,
        args.output_prefix,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
