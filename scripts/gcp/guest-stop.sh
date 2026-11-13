#!/bin/bash
# Compute Engine's platform hook; the startup runner installs this local helper.
set -euo pipefail
if [[ -x /usr/local/sbin/gcp-guest-run.sh ]]; then
    exec /usr/local/sbin/gcp-guest-run.sh --preempt
fi
