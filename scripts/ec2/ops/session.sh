#!/bin/bash
set -u
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
source "$(dirname -- "$0")/common.sh" || exit 2
[[ $# -ge 1 ]] || fail 'usage: session.sh start <label> <minutes> [retries] | host | stop | status'
operation=$1
shift
case "$operation" in
    start)
        [[ $# -ge 2 && $# -le 3 ]] || fail 'usage: session.sh start <label> <minutes> [retries]'
        exec "$PYTHON" "$OPS_DIR/ops_control.py" session-start "$@"
        ;;
    host)
        [[ $# == 0 || ( $# == 1 && $1 == --fallback ) ]] || fail 'host takes only --fallback'
        load_host "${1:-}" || exit $?
        if [[ $# == 1 ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$KEY" "$HOST" "$KNOWN_HOSTS" "$DEADLINE" "$SESSION_LABEL" "$INSTANCE" "$TXN"
        else
            printf '%s\t%s\t%s\t%s\n' "$KEY" "$HOST" "$KNOWN_HOSTS" "$DEADLINE"
        fi
        ;;
    status)
        [[ $# == 0 ]] || fail 'status takes no arguments'
        out=$(guard status)
        rc=$?
        [[ $rc == 0 ]] || exit "$rc"
        printf '%s\n' "$out" | "$PYTHON" "$OPS_DIR/ops_control.py" status
        ;;
    stop)
        [[ $# == 0 ]] || fail 'stop takes no arguments'
        exec "$PYTHON" "$OPS_DIR/ops_control.py" session-stop
        ;;
    *) fail "Unknown operation: $operation" ;;
esac
