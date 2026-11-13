#!/bin/bash
# GCE owns the authoritative DELETE deadline; this runner only finishes early.
# Functions called from traps are reachable; ShellCheck cannot follow EXIT traps.
# shellcheck disable=SC2329
set -euo pipefail
umask 077
BOOT_TIME=$(date -u +%FT%TZ)

# The hook is only a cheap fallback; the in-job watcher owns the save window.
if [[ ${1:-} == --preempt && ${2:-} != --bounded ]]; then
    exec timeout --kill-after=0.2s 5s bash "$0" --preempt --bounded
fi
CLOUD_PID=
RESTORE_LIMIT=${GCP_RESTORE_TIMEOUT:-1800s}
JOB_FETCH_LIMIT=${GCP_JOB_FETCH_TIMEOUT:-120s}
FINAL_COPY_LIMIT=${GCP_FINAL_COPY_TIMEOUT:-480s}
MARKER_LIMIT=${GCP_MARKER_TIMEOUT:-60s}
LOG_LIMIT=${GCP_LOG_TIMEOUT:-60s}
METADATA_LIMIT=10
if [[ ${1:-} == --preempt ]]; then
    MARKER_LIMIT=3s
    LOG_LIMIT=3s
    METADATA_LIMIT=0.25
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
if [[ ${1:-} != --preempt ]]; then trap early_exit EXIT; fi
LABEL=$(metadata guard-label)
STATE_URI=$(metadata guard-state-uri)
JOB_URI=$(metadata guard-job-uri)
RESERVED_END=$(metadata guard-reserved-end)
[[ $LABEL =~ ^[a-z][a-z0-9-]{0,39}$ ]]
[[ $STATE_URI =~ ^gs://[a-z0-9][a-z0-9._-]+/[A-Za-z0-9_./:@+=-]+$ ]]
[[ $JOB_URI =~ ^gs://[a-z0-9][a-z0-9._-]+/[A-Za-z0-9_./:@+=-]+$ ]]
RUN_DIR="${GCP_RUN_ROOT:-/var/lib/gcp-run}/$LABEL"
export RUN_DIR STATE_URI LABEL JOB_URI
command -v python3 >/dev/null
mkdir -p "$RUN_DIR/state"
JOB_LOG="$RUN_DIR/job.log"
PERIODIC_PID=
JOB_PID=
FINISHED="$RUN_DIR/.finished"

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

# The generated helper is private guest-local code, shared by monitor and hook.
if [[ ${1:-} != --preempt || ! -e $RUN_DIR/control.py ]]; then
    cat > "$RUN_DIR/control.py" <<'GCP_CONTROL_PY'
"""Guest-local monitor; all cloud children have a hard deadline and process group."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["RUN_DIR"])
state_uri = os.environ["STATE_URI"]
label = os.environ["LABEL"]
job_uri = os.environ["JOB_URI"]
stop = threading.Event()
children = set()
children_lock = threading.RLock()
copy_lock = threading.Lock()


def log(message):
    print(f"gcp preemption: {message}", file=sys.stderr, flush=True)


def command(argv, budget, capture=False, cancel_if_stopped=False):
    if budget is not None and budget <= 0:
        return None
    with children_lock:
        if cancel_if_stopped and stop.is_set():
            return None
        child = subprocess.Popen(
            argv,
            start_new_session=True,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        children.add(child)
    try:
        out, _ = child.communicate(timeout=budget)
        return (
            out
            if capture and child.returncode == 0
            else child.returncode == 0
            if not capture
            else None
        )
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        return None
    finally:
        with children_lock:
            children.discard(child)


def cancel_children():
    with children_lock:
        for child in list(children):
            try:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=0.2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass


def terminate(*_):
    stop.set()
    cancel_children()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)


def cloud(args, budget):
    return command(["gcloud", *args, "--format=json", "--quiet"], budget)


def marker(key, source, budget=3, status="interrupted"):
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    body = (
        f"label={label}\njob_uri={job_uri}\nexit_status={status}\n"
        f"time={timestamp}\nsource={source}\n"
    )
    path = root / ("." + key)
    path.write_text(body)
    return upload(path, key, budget)


def upload(path, key, budget):
    deadline = time.monotonic() + budget
    # JSON media upload avoids gcloud startup (measured 0.13 s versus 0.9 s).
    token = command(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "0.5",
            "--max-time",
            "1",
            "-H",
            "Metadata-Flavor: Google",
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        ],
        min(1, max(0, deadline - time.monotonic())),
        True,
    )
    try:
        token = json.loads(token)["access_token"]
        if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
            raise ValueError("invalid access token")
    except (TypeError, ValueError, KeyError):
        token = None
    if token:
        bucket, key_path = state_uri[5:].split("/", 1)
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={urllib.parse.quote(key_path + '/' + key, safe='')}"
        if (
            command(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "0.5",
                    "--max-time",
                    "1.5",
                    "--output",
                    "/dev/null",
                    "--write-out",
                    "%{http_code}",
                    "-X",
                    "POST",
                    "-H",
                    f"Authorization: Bearer {token}",
                    "-H",
                    "Content-Type: text/plain",
                    "--data-binary",
                    "@" + str(path),
                    url,
                ],
                min(1.5, max(0, deadline - time.monotonic())),
                True,
            )
            == b"200"
        ):
            return True
    return cloud(
        ["storage", "cp", str(path), state_uri + "/" + key], max(0, deadline - time.monotonic())
    )


def claim(source):
    with (root / ".outcome-lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source == "normal":
            if (root / ".preempting").exists():
                return False
            (root / ".finished").touch()
            return True
        if (root / ".finished").exists() or (root / ".job-ended").exists():
            return False
        owner = root / ".preempting"
        if owner.exists() and owner.read_text() != "hook":
            return False
        owner.write_text(source)
        return True


def periodic():
    interval = os.environ["COPY_INTERVAL"]
    while not stop.is_set():
        command(["sleep", interval], float(interval) + 1, cancel_if_stopped=True)
        if stop.is_set():
            return
        with copy_lock:
            if stop.is_set():
                return
            limit = os.environ.get("GCP_PERIODIC_COPY_TIMEOUT", "540s")
            result = command(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=2s",
                    limit,
                    "gcloud",
                    "storage",
                    "rsync",
                    str(root / "state") + "/",
                    state_uri + "/",
                    "--recursive",
                    "--format=json",
                    "--quiet",
                ],
                None,
                cancel_if_stopped=True,
            )
            if stop.is_set():
                return
            if result:
                heartbeat = root / ".HEARTBEAT"
                heartbeat.write_text(
                    f"label={label}\njob_uri={job_uri}\nexit_status=running\ntime={datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}\n"
                )
                limit = os.environ.get("GCP_MARKER_TIMEOUT", "60s")
                command(
                    [
                        "timeout",
                        "--signal=TERM",
                        "--kill-after=2s",
                        limit,
                        "gcloud",
                        "storage",
                        "cp",
                        str(heartbeat),
                        state_uri + "/HEARTBEAT",
                        "--format=json",
                        "--quiet",
                    ],
                    None,
                    cancel_if_stopped=True,
                )
            else:
                log("periodic copy failed; retrying next round")


def preempt(save_deadline):
    # The alarm also bounds directory traversal and local checkpoint operations.
    started = time.monotonic()
    trace_path = root / ".preempt-trace.jsonl"
    trace_path.write_text("")

    def trace(step, outcome, **details):
        record = dict(
            time=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            elapsed_s=round(time.monotonic() - started, 6),
            step=step, outcome=outcome, **details,
        )
        with trace_path.open("a") as out:
            out.write(json.dumps(record) + "\n")

    def diagnostics(final=False):
        # Small diagnostics precede the state queue and get a protected refresh.
        path = root / "job.log"
        if path.exists():
            result = upload(path, f"logs/{label}.log", min(2.5 if final else 3, max(0, save_deadline - time.monotonic() - 1.5)))
            trace("job_log", "uploaded" if result else "failed", final=final)
        trace("trace_upload", "attempting", final=final)
        upload(trace_path, f"logs/{label}.preempt.jsonl", min(1.5, max(0, save_deadline - time.monotonic())))

    def expired(*_):
        trace("deadline", "expired")
        log("24 s save deadline reached")
        terminate()

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, save_deadline - time.monotonic()))
    try:
        trace("preemption", "observed", label=label, job_uri=job_uri)
        if marker("PREEMPTED", "watcher"):
            (root / ".PREEMPTED.published").touch()
            trace("marker", "uploaded")
        else:
            log("preemption marker failed")
            trace("marker", "failed")
        # Stop periodic writers only AFTER the first marker attempt; never the job.
        stop.set()
        cancel_children()
        acquired = copy_lock.acquire(timeout=0.5)
        trace("periodic_stop", "stopped" if acquired else "lock_unavailable")
        try:
            nonce = uuid.uuid4().hex
            request = root / "checkpoint-request"
            temporary = root / ".checkpoint-request"
            temporary.write_text(nonce + "\n")
            temporary.replace(request)
            trace("checkpoint_request", "published", nonce=nonce)
            try:
                wait = min(8, max(0, float(os.environ.get("GCP_CHECKPOINT_WAIT", "8"))))
            except ValueError:
                wait = 8
            deadline = time.monotonic() + wait
            outcome = "timed_out"
            while time.monotonic() < deadline:
                try:
                    if (root / "checkpoint-done").read_text().strip() == nonce:
                        outcome = "acknowledged"
                        break
                except (OSError, UnicodeError):
                    pass
                if (root / ".job-ended").exists():
                    outcome = "job_ended"
                    break
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            trace("checkpoint_wait", outcome)
            diagnostics()
            # Newest immutable generations first. Reserve seven seconds for
            # reconciliation (<=3 s) and a final fast log/trace refresh (<=4 s).
            deadline = min(time.monotonic() + 8, save_deadline - 7)
            copied = True
            files = inventory()
            if files is None:
                files = []
                copied = False
            if acquired:
                for path, _, _ in files:
                    key = path.relative_to(root / "state").as_posix()
                    trace("state_file", "attempting", key=key)
                    result = cloud(
                        ["storage", "cp", str(path), state_uri + "/" + key],
                        max(0, deadline - time.monotonic()),
                    )
                    trace("state_file", "uploaded" if result else "failed", key=key)
                    if not result:
                        copied = False
                        if time.monotonic() >= deadline:
                            break
            else:
                copied = False
            trace("state_copy", "complete" if copied else "incomplete")
            if not copied:
                log("preemption state copy incomplete")
            ended = root / ".job-ended"
            status = ended.read_text().strip() if ended.exists() else "running"
            trace("job_status", status)
            log(f"save state_complete={copied}; job_exit_status={status}; job left running for immutable checkpoint copies")
            if copied and ended.exists() and inventory() == files:
                reconcile(status)
                trace("reconcile", "finished" if (root / ".finished").exists() else "unconfirmed")
            diagnostics(final=True)
        finally:
            if acquired:
                copy_lock.release()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        (root / ".preempt-done").touch()
        cancel_children()


def inventory():
    try:
        files = []
        for path in (root / "state").rglob("*"):
            if path.is_file() and not path.is_symlink():
                stat = path.stat()
                files.append((path, stat.st_mtime_ns, stat.st_size))
        return sorted(files, key=lambda row: (row[1], str(row[0])), reverse=True)
    except OSError:
        log("state inventory changed or could not be read")
        return None


def reconcile(status):
    # One owner transitions a saved event into a verified final job outcome.
    # Preserve the event first; never leave two conflicting root markers.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    archive = f"history/{timestamp}-{label}-PREEMPTED"
    if not upload(root / ".PREEMPTED", archive, 0.6):
        return
    if not cloud(["storage", "rm", state_uri + "/PREEMPTED"], 1.2):
        return
    key = "DONE" if status == "0" else "TIMEOUT" if status in ("124", "137") else "FAILED"
    if marker(key, "watcher", 0.8, status):
        (root / ".finished").touch()
    else:
        log("completion marker failed; restoring PREEMPTED")
        upload(root / ".PREEMPTED", "PREEMPTED", 0.4)


mode = sys.argv[1]
if mode == "claim-normal":
    sys.exit(0 if claim("normal") else 1)
if mode == "hook":
    if not (root / ".finished").exists() and not (root / ".PREEMPTED.published").exists():
        with (root / ".outcome-lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (
                not (root / ".finished").exists()
                and not (root / ".PREEMPTED.published").exists()
                and not (root / ".job-ended").exists()
                and not (root / ".preempting").exists()
            ):
                (root / ".preempting").write_text("hook")
                if marker("PREEMPTED", "hook"):
                    (root / ".PREEMPTED.published").touch()
                else:
                    log("preemption marker failed")
    sys.exit(0)

threading.Thread(target=periodic, daemon=True).start()
while not stop.is_set():
    response = command(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "0.5",
            "--max-time",
            "1.5",
            "-H",
            "Metadata-Flavor: Google",
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
        ],
        1.5,
        True,
    )
    if response in (b"TRUE", b"TRUE\n"):
        save_deadline = time.monotonic() + 24
        if claim("watcher"):
            preempt(save_deadline)
        break
    command(["sleep", "1"], 2)
terminate()
GCP_CONTROL_PY
fi

if [[ ${1:-} == --preempt ]]; then
    trap - EXIT
    [[ ! -e $FINISHED ]] || exit 0
    if [[ -e $RUN_DIR/control.py ]]; then
        exec python3 "$RUN_DIR/control.py" hook
    fi
    # No admitted runner: there is no job to classify or state to save.
    exit 0
fi

COPY_INTERVAL=$(metadata guard-copy-interval)
[[ $COPY_INTERVAL =~ ^[0-9]+$ ]] && (( COPY_INTERVAL >= 60 && COPY_INTERVAL <= 3600 ))
export COPY_INTERVAL

finish() {
    local status=$1 key=FAILED
    trap - EXIT TERM INT
    set +e
    # Decide the owner before a normal copy can race the metadata watcher.
    if ! python3 "$RUN_DIR/control.py" claim-normal; then
        # The watcher remains bounded even if all storage calls hang. Do not
        # cancel it when the job finishes during checkpoint negotiation/copy.
        wait "$PERIODIC_PID" 2>/dev/null || true
        exit "$status"
    fi
    stop_periodic
    stop_job
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
# Install before fetching the job; the platform hook is a marker-only fallback.
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

# GNU timeout signals the process group and escalates after 30 seconds.
# The inner Bash expands these variables after timeout establishes its group.
# shellcheck disable=SC2016
timeout --signal=TERM --kill-after=30s "${remaining}s" bash -c '
    status=0
    bash "$RUN_DIR/job.sh" || status=$?
    printf "%s\n" "$status" > "$RUN_DIR/.job-ended.tmp"
    mv -f "$RUN_DIR/.job-ended.tmp" "$RUN_DIR/.job-ended"
    exit "$status"
' > "$JOB_LOG" 2>&1 &
JOB_PID=$!
python3 "$RUN_DIR/control.py" monitor &
PERIODIC_PID=$!
set +e
wait "$JOB_PID"
status=$?
JOB_PID=
set -e
exit "$status"
