#!/bin/bash
set -u

OPS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
if [[ -n ${OPERATOR_ENV:-} ]]; then
    # shellcheck source=/dev/null
    source "$OPERATOR_ENV" || exit 2
fi
# Operator env files sometimes enable errexit; stop/reconcile must handle errors.
set +e
LOCAL_CODE=${LOCAL_CODE:-$(cd "$OPS_DIR/../../.." && pwd -P)}
PYTHON=${PYTHON:-python3}
: "${EXPERIMENT:?absolute evidence directory outside the repository}"
LEDGER=${LEDGER:-$EXPERIMENT/ops/cost-ledger.json}
GUEST_USER=${GUEST_USER:-ubuntu}
GUEST_HOME=${GUEST_HOME:-/home/ubuntu}
GUEST_CODE=${GUEST_CODE:-$GUEST_HOME/fev/code}
GUEST_ENV=${GUEST_ENV:-$GUEST_HOME/fev/env.sh}
GUEST_DISPATCH=${GUEST_DISPATCH:-$GUEST_HOME/dispatch-job.sh}
GUEST_WAIT=${GUEST_WAIT:-$GUEST_HOME/wait-job.sh}
GUEST_LOG_ROOT=${GUEST_LOG_ROOT:-$GUEST_HOME/fev/host-logs}
export LOCAL_CODE PYTHON EXPERIMENT LEDGER GUEST_USER GUEST_HOME GUEST_CODE GUEST_ENV
export GUEST_DISPATCH GUEST_WAIT GUEST_LOG_ROOT
export AWS_DEFAULT_OUTPUT=json AWS_PAGER=''
mkdir -p "$EXPERIMENT/ops" || exit 2

fail() { echo "$*" >&2; exit 2; }
bounded() { "$PYTHON" "$OPS_DIR/ops_control.py" bounded "$@"; }
guard() { bounded "${GUARD_TIMEOUT_SECONDS:-180}" "$PYTHON" "$LOCAL_CODE/scripts/ec2/cost_guard.py" --ledger "$LEDGER" "$@"; }
launch() { bash "$LOCAL_CODE/scripts/ec2/pretrain_launch.sh" "$@" --ledger "$LEDGER"; }
quote_command() {
    local separator='' item
    for item in "$@"; do
        printf '%s%q' "$separator" "$item"
        separator=' '
    done
}
load_host() {
    local fields fallback=${1:-}
    fields=$("$PYTHON" "$OPS_DIR/ops_control.py" host "$fallback") || return $?
    # These fields are also consumed by scripts sourcing this library.
    # shellcheck disable=SC2034
    IFS=$'\t' read -r KEY HOST KNOWN_HOSTS DEADLINE SESSION_LABEL INSTANCE TXN <<< "$fields"
    [[ -s $KEY && -s $KNOWN_HOSTS ]] || { echo 'Missing ledger key or pinned known-hosts file' >&2; return 2; }
    ssh-keygen -F "$HOST" -f "$KNOWN_HOSTS" >/dev/null || {
        echo "No ledger-pinned host key for $HOST; inspect the launch evidence" >&2; return 2;
    }
    SSH_OPTS=(-o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=yes
        -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
}
refresh_transport_failure() {
    local rc=$1
    if [[ $rc == 255 ]]; then
        bounded 45 "$PYTHON" "$OPS_DIR/ops_control.py" ingress >&2 || true
    fi
    return "$rc"
}
guest_ssh() {
    bounded "${SSH_TIMEOUT_SECONDS:-120}" ssh -i "$KEY" "${SSH_OPTS[@]}" "$GUEST_USER@$HOST" "$@"
    refresh_transport_failure "$?"
}
# rsync parses -e itself, without shell backslash escaping. Double quotes and
# doubled embedded quotes are understood by both openrsync and upstream rsync.
rsync_transport() {
    "$PYTHON" "$OPS_DIR/ops_control.py" rsync-transport ssh -i "$KEY" "${SSH_OPTS[@]}"
}
