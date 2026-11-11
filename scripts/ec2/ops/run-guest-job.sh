#!/bin/bash
set -u
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
source "$(dirname -- "$0")/common.sh" || exit 2
exec "$PYTHON" "$OPS_DIR/ops_control.py" run
