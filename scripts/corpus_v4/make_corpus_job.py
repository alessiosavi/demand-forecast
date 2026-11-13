"""Render a pinned, credential-free corpus guest job (no cloud operations here)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPOSITORY = "https://github.com/alessiosavi/demand-forecast.git"


def render(repo_sha, corpus_prefix):
    """Generate a bash job that preserves incomplete work and uploads failure evidence."""
    if not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("repo SHA must be a full lowercase commit hash")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", corpus_prefix):
        raise ValueError("corpus prefix must be a plain gs:// bucket path")
    return f"""#!/usr/bin/env bash
# Research-only use. Original source licences apply.
# Full corpus and audit state are durable below RUN_DIR/state.
set -euo pipefail
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
REPO_SHA='{repo_sha}'
CORPUS_URI='{corpus_prefix.rstrip("/")}'
mkdir -p "$RUN_DIR/state"
CORPUS="$RUN_DIR/state/corpus"
AUDIT="$RUN_DIR/state/audit"
SYNC_PID=''
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export CORPUS_ATTEMPT_ID="$ATTEMPT_ID" CORPUS_REPO_SHA="$REPO_SHA"
write_status() {{
    local code="$1" complete=false
    if [[ "$code" == 0 ]]; then complete=true; fi
    printf '{{"attempt_id":"%s","repo_sha":"%s","exit_code":%d,"complete":%s}}\\n' \
      "$ATTEMPT_ID" "$REPO_SHA" "$code" "$complete" > "$RUN_DIR/state/attempt-status.json.tmp"
    mv "$RUN_DIR/state/attempt-status.json.tmp" "$RUN_DIR/state/attempt-status.json"
}}
finish() {{
    status=$?
    trap - EXIT
    set +e
    if [[ -n "$SYNC_PID" ]]; then
        kill "$SYNC_PID" 2>/dev/null
        wait "$SYNC_PID" 2>/dev/null
    fi
    if [[ -d "$CORPUS" ]]; then
        gcloud storage rsync --recursive "$CORPUS" "$CORPUS_URI" || status=3
    fi
    if [[ -f "$AUDIT/audit.json" ]]; then
        gcloud storage cp "$AUDIT/audit.json" "$CORPUS_URI/audit.json" || status=3
    fi
    gcloud storage rsync --recursive "$RUN_DIR/state" "$STATE_URI" || status=3
    write_status "$status"
    gcloud storage cp "$RUN_DIR/state/attempt-status.json" "$CORPUS_URI/attempt-status.json" || status=3
    exit "$status"
}}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Invalidate previous success before provisioning or fetching anything.
rm -f "$AUDIT/audit.json"
write_status -1
gcloud storage cp "$RUN_DIR/state/attempt-status.json" "$CORPUS_URI/attempt-status.json"
# The host restores state before invoking us. Upload every ten minutes as well.
(
    while sleep 600; do
        gcloud storage rsync --recursive "$RUN_DIR/state" "$STATE_URI" || true
    done
) >/dev/null 2>&1 &
SYNC_PID=$!
CHECKOUT="$RUN_DIR/corpus-repo-$REPO_SHA"
if [[ ! -d "$CHECKOUT/.git" ]]; then
    git clone --no-checkout '{REPOSITORY}' "$CHECKOUT"
fi
cd "$CHECKOUT"
git fetch origin "$REPO_SHA"
git checkout --detach "$REPO_SHA"
test "$(git rev-parse HEAD)" = "$REPO_SHA"
python3 -m venv "$RUN_DIR/corpus-venv"
source "$RUN_DIR/corpus-venv/bin/activate"
python -m pip install -e '.[bench]'
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -m scripts.corpus_v4.fetch_native "$CORPUS" --state "$RUN_DIR/state"
# Each folder contains stats.json and a verified per-file receipt.
python -m scripts.corpus_v4.audit_fev_overlap "$CORPUS" "$AUDIT" --state "$RUN_DIR/state"
"""


def main(argv=None):
    """Write one reviewable job.sh for an existing cloud guest runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_sha")
    parser.add_argument("corpus_prefix")
    parser.add_argument("out", type=Path, help="job.sh path")
    args = parser.parse_args(argv)
    source = render(args.repo_sha, args.corpus_prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
