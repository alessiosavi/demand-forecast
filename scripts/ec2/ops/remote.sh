#!/bin/bash
set -u
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
source "$(dirname -- "$0")/common.sh" || exit 2
[[ $# -ge 1 ]] || fail 'usage: remote.sh ssh|dispatch|wait|push|pull ...'
operation=$1
shift
load_host || exit $?
case "$operation" in
    ssh)
        if [[ $# == 0 ]]; then
            guest_ssh
        elif [[ $# == 1 ]]; then
            guest_ssh "$1"
        else
            guest_ssh "$(quote_command "$@")"
        fi
        ;;
    dispatch)
        [[ $# -ge 2 && $1 =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'usage: remote.sh dispatch <label> <cmd...>'
        # Never retry dispatch: a lost reply leaves an unknown running job.
        SSH_TIMEOUT_SECONDS=${DISPATCH_TIMEOUT_SECONDS:-60} guest_ssh "$(quote_command env "HOST_LOGS=$GUEST_LOG_ROOT" bash "$GUEST_DISPATCH" "$@")"
        rc=$?
        [[ $rc == 0 ]] || echo "UNKNOWN dispatch (exit $rc); inspect guest host-logs before any new dispatch" >&2
        exit "$rc"
        ;;
    wait)
        [[ $# == 3 && $1 == /* && $2 =~ ^[0-9]+$ && $3 =~ ^[0-9]+$ ]] || fail 'usage: remote.sh wait <job> <expected-exit> <deadline-epoch>'
        job=$1 expected=$2 deadline=$3
        while :; do
            now=$(date -u +%s)
            remaining=$((deadline - now))
            (( remaining > 0 )) || { echo "UNKNOWN: deadline reached for $job" >&2; exit 124; }
            chunk=$remaining
            (( chunk <= 600 )) || chunk=600
            command=$(quote_command bash "$GUEST_WAIT" "$job" "$expected" "$chunk")
            "$PYTHON" "$OPS_DIR/ops_control.py" bounded "$chunk" ssh -i "$KEY" "${SSH_OPTS[@]}" "$GUEST_USER@$HOST" "$command"
            rc=$?
            refresh_transport_failure "$rc"
            case "$rc" in
                255|124)
                    echo "Wait interrupted (exit $rc); re-polling $job" >&2
                    remaining=$((deadline - $(date -u +%s)))
                    (( remaining > 0 )) || exit 124
                    delay=5
                    (( remaining >= delay )) || delay=$remaining
                    sleep "$delay" || exit 130
                    ;;
                *) exit "$rc" ;;
            esac
        done
        ;;
    push)
        [[ $# -ge 2 ]] || fail 'usage: remote.sh push <file...> <remote-dir>'
        files=()
        while [[ $# -gt 1 ]]; do
            [[ -f $1 ]] || fail "Missing input file: $1"
            files+=("$1")
            shift
        done
        [[ $1 == /* ]] || fail 'Remote destination must be absolute'
        guest_ssh "$(quote_command mkdir -p -- "$1")" || exit $?
        bounded "${PUSH_TIMEOUT_SECONDS:-180}" scp -q -i "$KEY" "${SSH_OPTS[@]}" "${files[@]}" "$GUEST_USER@$HOST:${1%/}/"
        refresh_transport_failure "$?"
        ;;
    pull)
        [[ $# -ge 2 && $1 == /* ]] || fail 'usage: remote.sh pull <remote-dir> <local-dir> [--exclude PATTERN ...]'
        remote_dir=$1 local_dir=$2
        shift 2
        excludes=(--exclude 'bundle/' --exclude 'checkpoints/')
        while [[ $# -gt 0 ]]; do
            [[ $1 == --exclude && $# -ge 2 ]] || fail 'Expected --exclude PATTERN'
            excludes+=(--exclude "$2")
            shift 2
        done
        mkdir -p "$local_dir" || exit 2
        evidence=$(mktemp -d "$EXPERIMENT/ops/pull.XXXXXXXX") || exit 2
        # Reserve an upper bound, including excluded files and protocol overhead.
        size=$(guest_ssh "set -o pipefail; $(quote_command du -sb -- "$remote_dir") | cut -f1") || exit $?
        "$PYTHON" "$OPS_DIR/ops_control.py" transfer-manifest "$size" "$evidence/manifest.json" || exit $?
        guard transfer-begin --manifest "$evidence/manifest.json" --attempt 1 >> "$evidence/guard.log" 2>&1 || {
            cat "$evidence/guard.log" >&2; exit 2;
        }
        transport=$(rsync_transport) || exit $?
        source_path=$(quote_command "${remote_dir%/}/")
        LC_ALL=C bounded "${COPY_TIMEOUT_SECONDS:-180}" rsync -az --stats -e "$transport" "${excludes[@]}" "$GUEST_USER@$HOST:$source_path" "${local_dir%/}/" > "$evidence/rsync.log" 2>&1
        rsync_rc=$?
        refresh_transport_failure "$rsync_rc"
        received=$("$PYTHON" "$OPS_DIR/rsync_received.py" "$evidence/rsync.log")
        stats_rc=$?
        end_rc=2
        if [[ $stats_rc == 0 ]]; then
            guard transfer-end --manifest "$evidence/manifest.json" --attempt 1 --bytes "$received" >> "$evidence/guard.log" 2>&1
            end_rc=$?
        else
            echo "Transfer statistics missing; hold retained. Inspect $evidence" >&2
        fi
        echo "PULL rsync_exit=$rsync_rc accounting_exit=$end_rc evidence=$evidence" >&2
        [[ $rsync_rc == 0 ]] || exit "$rsync_rc"
        exit "$end_rc"
        ;;
    *) fail "Unknown operation: $operation" ;;
esac
