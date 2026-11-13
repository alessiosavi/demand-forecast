"""Render a pinned, credential-free corpus guest job (no cloud operations here)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.corpus_v4.native import memory_budget_arg

REPOSITORY = "https://github.com/alessiosavi/demand-forecast.git"

ENVIRONMENT = r"""# Preserve the image's CUDA-matched torch, including its local version suffix.
scratch=$(mktemp -d "$RUN_DIR/corpus-install-XXXXXX")
python3 - "$scratch/image-torch.json" "$scratch/torch-constraint.txt" <<'CORPUS_IMAGE'
import json
import sys
from pathlib import Path


import torch

image = {"version": torch.__version__, "path": str(Path(torch.__file__).resolve())}
Path(sys.argv[1]).write_text(json.dumps(image))
Path(sys.argv[2]).write_text("torch==" + image["version"] + "\n")
CORPUS_IMAGE
if ! python3 -m venv --clear --system-site-packages "$RUN_DIR/corpus-venv"; then
    # The image may lack ensurepip. uv seeds pip and clears any interrupted venv.
    python3 -m pip install --target "$scratch/uv-bootstrap" uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --system-site-packages \
        --python "$(command -v python3)" --no-python-downloads "$RUN_DIR/corpus-venv"
fi
source "$RUN_DIR/corpus-venv/bin/activate"
python -m pip install --constraint "$scratch/torch-constraint.txt" -e '.[bench]'
python -m pip freeze > "$RUN_DIR/state/environment.txt"
python - "$CHECKOUT" "$REPO_SHA" "$scratch/image-torch.json" <<'CORPUS_ENVIRONMENT'
import json
import subprocess
import sys
from importlib import import_module, metadata
from pathlib import Path

from urllib.parse import unquote, urlparse

root = Path(sys.argv[1]).resolve()
# stdin gives the checkout an implicit import path. Prove the installation
# works without it, so cwd cannot hide a stale package in site-packages.
if sys.path[0] == "":
    sys.path.pop(0)
# An editable's .pth may also put the checkout on sys.path. Ignore its source
# egg-info for metadata lookup, preserving precedence among installed copies.
installed = next(metadata.distributions(
    name="demand-forecast", path=[p for p in sys.path if Path(p).resolve() != root]
), None)
direct = json.loads(installed.read_text("direct_url.json") or "{}") if installed else {}
url = urlparse(direct.get("url", ""))
if (url.scheme != "file" or url.netloc not in ("", "localhost")
        or not Path(unquote(url.path)).is_absolute()
        or Path(unquote(url.path)).resolve() != root
        or direct.get("dir_info", {}).get("editable") is not True):
    raise RuntimeError("installed demand-forecast is not the pinned editable checkout")

import demand_forecast
import torch

image = json.loads(Path(sys.argv[3]).read_text())
if torch.__version__ != image["version"] or str(Path(torch.__file__).resolve()) != image["path"]:
    raise RuntimeError("torch is not the image's CUDA-matched package")
if Path(demand_forecast.__file__).resolve() != root / "demand_forecast/__init__.py":
    raise RuntimeError("installed demand-forecast is not the pinned editable checkout")
# Include lazy imports used by the fetcher and audit, without loading any data.
for name in (
    "huggingface_hub", "datasets", "fev", "numpy", "pandas", "pyarrow.parquet", "yaml",
):
    import_module(name)
# scripts/ is deliberately outside the installed package; load it from checkout.
sys.path.insert(0, str(root))
for name in ("scripts.corpus_v4.fetch_native", "scripts.corpus_v4.audit_fev_overlap"):
    if Path(import_module(name).__file__).resolve() != root / (name.replace(".", "/") + ".py"):
        raise RuntimeError("corpus module is not from the pinned checkout: " + name)
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if head != sys.argv[2]:
    raise RuntimeError("installed checkout HEAD does not match REPO_SHA")
CORPUS_ENVIRONMENT
"""


def render(repo_sha, corpus_prefix, *, memory_budget_gib=32):
    """Generate a bash job that preserves incomplete work and uploads failure evidence."""
    if not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("repo SHA must be a full lowercase commit hash")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9._/-]+)?", corpus_prefix):
        raise ValueError("corpus prefix must be a plain gs:// bucket path")
    memory_budget_arg(memory_budget_gib)
    memory_budget_gib = float(memory_budget_gib)
    return f"""#!/usr/bin/env bash
# Research-only use. Original source licences apply.
# Full corpus and audit state are durable below RUN_DIR/state.
set -euo pipefail
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
REPO_SHA='{repo_sha}'
CORPUS_URI='{corpus_prefix.rstrip("/")}'
mkdir -p "$RUN_DIR/state"
export HOME="${{HOME:-$RUN_DIR/home}}"
export HF_HOME="$RUN_DIR/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HOME" "$HF_HOME"
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
{ENVIRONMENT}\
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
MEMORY_BUDGET_GIB="${{CORPUS_MEMORY_BUDGET_GIB:-{memory_budget_gib:g}}}"
fetch_status=0
python -m scripts.corpus_v4.fetch_native "$CORPUS" --state "$RUN_DIR/state" \\
    --memory-budget-gib "$MEMORY_BUDGET_GIB" || fetch_status=$?
# Each complete folder has stats.json and a verified receipt.
# Audit it even when another folder failed to fetch/measure.
audit_status=0
python -m scripts.corpus_v4.audit_fev_overlap "$CORPUS" "$AUDIT" --state "$RUN_DIR/state" \\
    --memory-budget-gib "$MEMORY_BUDGET_GIB" || audit_status=$?
if (( fetch_status != 0 || audit_status != 0 )); then
    echo "Corpus attempt failed: fetch=$fetch_status audit=$audit_status; see fetch-summary.json and audit.json" >&2
    exit 2
fi
"""


def main(argv=None):
    """Write one reviewable job.sh for an existing cloud guest runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_sha")
    parser.add_argument("corpus_prefix")
    parser.add_argument("out", type=Path, help="job.sh path")
    parser.add_argument("--memory-budget-gib", type=float, default=32)
    args = parser.parse_args(argv)
    source = render(args.repo_sha, args.corpus_prefix, memory_budget_gib=args.memory_budget_gib)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    args.out.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
