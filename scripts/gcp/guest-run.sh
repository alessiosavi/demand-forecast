#!/bin/bash
# GCE owns the authoritative DELETE deadline; this runner only finishes early.
# Functions called from traps are reachable; ShellCheck cannot follow EXIT traps.
# shellcheck disable=SC2329
set -euo pipefail
umask 077
BOOT_TIME=$(date -u +%FT%TZ)

# Bound the whole platform hook, including metadata and marker/log uploads.
if [[ ${1:-} == --preempt && ${2:-} != --bounded ]]; then
    exec timeout --kill-after=2s 25s bash "$0" --preempt --bounded
fi
CLOUD_PID=
RESTORE_LIMIT=${GCP_RESTORE_TIMEOUT:-1800s}
JOB_FETCH_LIMIT=${GCP_JOB_FETCH_TIMEOUT:-120s}
PERIODIC_COPY_LIMIT=${GCP_PERIODIC_COPY_TIMEOUT:-540s}
FINAL_COPY_LIMIT=${GCP_FINAL_COPY_TIMEOUT:-480s}
MARKER_LIMIT=${GCP_MARKER_TIMEOUT:-60s}
LOG_LIMIT=${GCP_LOG_TIMEOUT:-60s}
METADATA_LIMIT=10
if [[ ${1:-} == --preempt ]]; then
    MARKER_LIMIT=3s
    LOG_LIMIT=3s
    METADATA_LIMIT=3
fi

metadata() {
    curl --fail --silent --show-error --connect-timeout 3 --max-time "$METADATA_LIMIT" \
        -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

cloud() {
    local limit=$1 result=0
    shift
    timeout --signal=TERM --kill-after=2s "$limit" gcloud "$@" --format=json --quiet &
    CLOUD_PID=$!
    wait "$CLOUD_PID" || result=$?
    CLOUD_PID=
    return "$result"
}
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

# Failures before usable metadata still end the VM. GCE's deadline remains armed.
early_exit() {
    # Suppress the platform hook's copy/marker if startup has not been admitted.
    if [[ -n ${FINISHED:-} ]]; then touch "$FINISHED"; fi
    shutdown -h now
}
trap early_exit EXIT
LABEL=$(metadata guard-label)
STATE_URI=$(metadata guard-state-uri)
JOB_URI=$(metadata guard-job-uri)
RESERVED_END=$(metadata guard-reserved-end)
[[ $LABEL =~ ^[a-z][a-z0-9-]{0,39}$ ]]
[[ $STATE_URI =~ ^gs://[a-z0-9][a-z0-9._-]+/[A-Za-z0-9_./:@+=-]+$ ]]
[[ $JOB_URI =~ ^gs://[a-z0-9][a-z0-9._-]+/[A-Za-z0-9_./:@+=-]+$ ]]
RUN_DIR="${GCP_RUN_ROOT:-/var/lib/gcp-run}/$LABEL"
export RUN_DIR STATE_URI
mkdir -p "$RUN_DIR/state"
JOB_LOG="$RUN_DIR/job.log"
PERIODIC_PID=
JOB_PID=
FINISHED="$RUN_DIR/.finished"
PREEMPTING="$RUN_DIR/.preempting"

copy_state() {
    # Upload in-place; the job must publish restartable checkpoint generations.
    cloud "$1" storage rsync "$RUN_DIR/state/" "$STATE_URI/" --recursive
}

marker() {
    local key=$1 status=$2
    printf 'label=%s\njob_uri=%s\nexit_status=%s\ntime=%s\n' \
        "$LABEL" "$JOB_URI" "$status" "$(date -u +%FT%TZ)" > "$RUN_DIR/.$key"
    if [[ $key == STARTED ]]; then printf 'boot_time=%s\n' "$BOOT_TIME" >> "$RUN_DIR/.$key"; fi
    cloud "$MARKER_LIMIT" storage cp "$RUN_DIR/.$key" "$STATE_URI/$key"
}

stop_periodic() {
    if [[ -n $PERIODIC_PID ]]; then
        kill "$PERIODIC_PID" 2>/dev/null || true
        wait "$PERIODIC_PID" 2>/dev/null || true
        PERIODIC_PID=
    fi
}

stop_job() {
    if [[ -n $JOB_PID ]]; then
        local killer
        # GNU timeout owns a process group. Signal it, then force the whole
        # group to end if the job ignores TERM, before copying its checkpoint.
        kill -TERM -- "-$JOB_PID" 2>/dev/null || true
        ( sleep 2; kill -KILL -- "-$JOB_PID" 2>/dev/null || true ) &
        killer=$!
        wait "$JOB_PID" 2>/dev/null || true
        # Reap the bounded escalation helper too; no writer survives finish.
        wait "$killer" 2>/dev/null || true
        JOB_PID=
    fi
}

if [[ ${1:-} == --preempt ]]; then
    trap - EXIT
    # A normal job ending powers off too; do not mislabel that as preemption.
    [[ ! -e $FINISHED ]] || exit 0
    touch "$PREEMPTING"
    # The platform hook has a short best-effort window. Bound each storage call.
    timeout --kill-after=2s 10s gcloud storage rsync "$RUN_DIR/state/" "$STATE_URI/" --recursive --format=json --quiet || log 'preemption copy failed'
    marker PREEMPTED interrupted || log 'preemption marker failed'
    if [[ -e $JOB_LOG ]]; then
        cloud "$LOG_LIMIT" storage cp "$JOB_LOG" "$STATE_URI/logs/$LABEL.log" || log 'preemption log copy failed'
    fi
    exit 0
fi

finish() {
    local status=$1 key=FAILED
    trap - EXIT TERM INT
    set +e
    stop_periodic
    stop_job
    # The platform hook handles preemption. Do not publish a conflicting result.
    if [[ -e $PREEMPTING ]]; then
        exit "$status"
    fi
    if [[ $status == 124 || $status == 137 ]]; then key=TIMEOUT; fi
    if copy_state "$FINAL_COPY_LIMIT"; then
        if [[ $status == 0 ]]; then key=DONE; fi
    else
        log 'final copy failed; refusing DONE'
        [[ $status != 0 ]] || status=74
    fi
    marker "$key" "$status" || { log 'final marker failed'; status=74; }
    if [[ -e $JOB_LOG ]]; then
        cloud "$LOG_LIMIT" storage cp "$JOB_LOG" "$STATE_URI/logs/$LABEL.log" || { log 'log copy failed'; status=74; }
    fi
    touch "$FINISHED"
    shutdown -h now
    exit "$status"
}
# Install before fetching the job, so the platform hook can flush local state.
install -m 0755 -- "$0" /usr/local/sbin/gcp-guest-run.sh

read_marker() {
    local key=$1
    MARKER_JOB_URI=
    # Describe distinguishes an absent object from cat's unmatched-URL error.
    if cloud "$MARKER_LIMIT" storage objects describe "$STATE_URI/$key" \
        > "$RUN_DIR/.marker.json" 2> "$RUN_DIR/.marker-error"; then
        if ! cloud "$MARKER_LIMIT" storage cat "$STATE_URI/$key" \
            > "$RUN_DIR/.$key.previous" 2> "$RUN_DIR/.marker-error"; then
            log "cannot read $key body; refusing to run"
            exit 1
        fi
        MARKER_JOB_URI=$(sed -n 's/^job_uri=//p' "$RUN_DIR/.$key.previous")
        return 0
    elif ! grep -Eq '403|PERMISSION_DENIED|Permission denied' "$RUN_DIR/.marker-error" \
        && grep -Eq '404|NOT_FOUND|not found|No such object' "$RUN_DIR/.marker-error"; then
        return 1
    else
        log "cannot read $key or verify its absence"
        exit 1
    fi
}

if read_marker DONE; then
    if [[ -z $MARKER_JOB_URI ]]; then
        log 'DONE has no job URI (legacy format); refusing to run. Use a new state prefix, or verify the completed job and archive the legacy DONE before launching a different job.'
        exit 0
    elif [[ $MARKER_JOB_URI == "$JOB_URI" ]]; then
        log 'job URI is DONE; refusing to run'
        exit 0
    fi
    HAS_DONE=1
else
    HAS_DONE=0
fi

# Preserve prior outcomes before publishing this attempt. Interrupted/failed
# attempts of the same job are archived too, so resume cannot leave stale results.
# Keep the early-exit trap until admission ends: an archive failure must not run
# finish(), sync local state, overwrite a result or invoke the preemption copy.
ARCHIVE_TIME=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
for key in DONE FAILED TIMEOUT PREEMPTED; do
    if [[ $key == DONE ]]; then
        [[ $HAS_DONE == 1 ]] || continue
    elif ! read_marker "$key"; then
        continue
    fi
    cloud "$MARKER_LIMIT" storage cp "$STATE_URI/$key" "$STATE_URI/history/$ARCHIVE_TIME-$LABEL-$key" \
        || { log "archive copy failed for $key; refusing to run"; exit 1; }
    cloud "$MARKER_LIMIT" storage rm "$STATE_URI/$key" \
        || { log "archive delete failed for $key; refusing to run"; exit 1; }
done

trap 'finish $?' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
marker STARTED started
cloud "$RESTORE_LIMIT" storage rsync "$STATE_URI/" "$RUN_DIR/state/" --recursive \
    --exclude='(^|/)(STARTED|DONE|FAILED|TIMEOUT|PREEMPTED|HEARTBEAT|logs|history)(/|$)'
cloud "$JOB_FETCH_LIMIT" storage cp "$JOB_URI" "$RUN_DIR/job.sh"

remaining=$(( $(date -u -d "$RESERVED_END" +%s) - $(date -u +%s) - 600 ))
if (( remaining <= 0 )); then
    log 'no training time remains before the final-copy window'
    exit 124
fi

periodic() {
    local timer=
    trap 'if [[ -n $timer ]]; then kill "$timer" 2>/dev/null || true; wait "$timer" 2>/dev/null || true; fi;
          if [[ -n $CLOUD_PID ]]; then kill "$CLOUD_PID" 2>/dev/null || true; wait "$CLOUD_PID" 2>/dev/null || true; fi;
          exit 0' TERM INT
    while true; do
        sleep 600 &
        timer=$!
        wait "$timer" || return
        timer=
        if copy_state "$PERIODIC_COPY_LIMIT"; then
            marker HEARTBEAT running || log 'heartbeat copy failed'
        else
            log 'periodic copy failed; retrying next round'
        fi
    done
}

# GNU timeout signals the process group and escalates after 30 seconds.
timeout --signal=TERM --kill-after=30s "${remaining}s" bash "$RUN_DIR/job.sh" > "$JOB_LOG" 2>&1 &
JOB_PID=$!
periodic &
PERIODIC_PID=$!
set +e
wait "$JOB_PID"
status=$?
JOB_PID=
set -e
exit "$status"
