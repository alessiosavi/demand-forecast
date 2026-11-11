#!/bin/bash
set -u
[[ $# == 2 ]] || { echo 'usage: wins-guest.sh <stage> <manifest.json>' >&2; exit 2; }
# shellcheck source=/dev/null
source "${GUEST_ENV:-/home/ubuntu/fev/env.sh}" || exit 2
cd "${TOOLS:?guest checkout}" || exit 2
exec "${PY:-python3}" "$TOOLS/scripts/ec2/ops/guest/wins_queue.py" "$@"
