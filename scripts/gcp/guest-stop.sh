#!/bin/bash
# Best-effort marker fallback only. Real Spot power-off hooks may get no CPU.
# The startup runner's metadata watcher saves state before power-off starts.
set -euo pipefail
if [[ -x /usr/local/sbin/gcp-guest-run.sh ]]; then
    exec /usr/local/sbin/gcp-guest-run.sh --preempt
fi
