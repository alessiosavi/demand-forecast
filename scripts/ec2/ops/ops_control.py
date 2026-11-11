#!/usr/bin/env python3
"""Portable local plumbing for the Bash 3.2 operator entry points (stdlib only)."""

import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from process_control import cleanup_signals, install_interrupts, run_process
from session_control import ingress, owned_session, start, start_check, stop, unresolved

HERE = Path(__file__).resolve().parent


def ledger() -> dict:
    return json.loads(Path(os.environ["LEDGER"]).read_text())


def host(fallback: bool = False) -> None:
    data = ledger()
    sessions = [s for s in data["sessions"] if s["status"] == "open"]
    if fallback:
        sessions = [s for s in data["sessions"] if s["status"] in ("open", "pending")]
        if not sessions:
            sessions = [s for s in data["sessions"] if s.get("public_ip")][-1:]
    if len(sessions) != 1:
        raise ValueError("Expected one open session; inspect the ledger and reconcile")
    session = sessions[0]
    fields = [
        data["config"]["key_path"],
        session["public_ip"],
        data["config"]["known_hosts_path"],
        session["deadline"],
        session["label"],
        session["instance_id"],
        session["txn_id"],
    ]
    if any(not isinstance(s, str) or not s or any(c in s for c in "\r\n\t") for s in fields):
        raise ValueError("Incomplete or malformed session connection in ledger")
    print("\t".join(fields))


def stop_evidence(label: str, txn: str, at: str, rc: str, failed: str) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
        raise ValueError("Invalid session label")
    path = Path(os.environ["EXPERIMENT"]) / "ops" / f"stop.{label}.json"
    evidence = dict(
        label=label,
        txn=txn,
        shutdown_at=at,
        shutdown_ssh_exit=int(rc),
        first_failed_ssh_at=failed or None,
        note="SSH unreachability is evidence for operator review, not proof of an EC2 stop",
    )
    # Keep the first observed failure across repeated stop attempts.
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("first_failed_ssh_at"):
            evidence["first_failed_ssh_at"] = previous["first_failed_ssh_at"]
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"SSH stop evidence: {path}; after login, reconcile and settle estimated times with evidence",
        file=sys.stderr,
    )


def command(script: str, *args: str, seconds=None) -> None:
    if seconds is None:
        operation = args[0] if args else ""
        name, default = {
            "ssh": ("CHECKOUT_TIMEOUT_SECONDS", 180),
            "push": ("PUSH_TIMEOUT_SECONDS", 180),
            "pull": ("COPY_TIMEOUT_SECONDS", 180),
        }.get(operation, ("COMMAND_TIMEOUT_SECONDS", 180))
        if operation == "wait":
            seconds = max(0.01, int(args[-1]) - time.time() + 2)
        else:
            seconds = float(os.environ.get(name, default))
    result = run_process(["bash", str(HERE / script), *args], seconds, text=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def json_list(name: str) -> list:
    value = json.loads(os.environ.get(name, "[]"))
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def strings(values: object) -> bool:
    return isinstance(values, list) and all(isinstance(v, str) and "\0" not in v for v in values)


def validate_inputs() -> tuple[list, list, list]:
    env = os.environ
    for key in ("LABEL", "MINUTES", "SHA", "GUEST_SCRIPT", "EXPERIMENT", "LOCAL_CODE"):
        if not env.get(key):
            raise ValueError(f"Missing {key}")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,24}", env["LABEL"]):
        raise ValueError("Invalid LABEL")
    if not re.fullmatch(r"[0-9a-f]{40}", env["SHA"]):
        raise ValueError("SHA must be a full lowercase 40-character commit id")
    experiment, code = Path(env["EXPERIMENT"]), Path(env["LOCAL_CODE"])
    if not experiment.is_absolute() or not code.is_absolute():
        raise ValueError("EXPERIMENT and LOCAL_CODE must be absolute")
    if experiment.resolve().is_relative_to(code.resolve()):
        raise ValueError("EXPERIMENT must be outside LOCAL_CODE")
    if not Path(env["GUEST_SCRIPT"]).is_file():
        raise ValueError("GUEST_SCRIPT must name a local file")
    if env.get("SKIP_SETUP", "0") not in ("0", "1"):
        raise ValueError("SKIP_SETUP must be 0 or 1")
    expected = int(env.get("EXPECTED_EXIT", "0"))
    if not 0 <= expected <= 255:
        raise ValueError("EXPECTED_EXIT must be between 0 and 255")
    for key, default in (("SETUP_TIMEOUT_SECONDS", "2400"), ("COPY_MARGIN_SECONDS", "120")):
        if int(env.get(key, default)) < 1:
            raise ValueError(f"{key} must be positive")
    args, pushes, pulls = json_list("GUEST_ARGS"), json_list("PUSH"), json_list("PULL")
    if not strings(args):
        raise ValueError("GUEST_ARGS must be a JSON array of strings")
    for entry in pushes:
        if not isinstance(entry, dict) or not strings(entry.get("files")) or not entry["files"]:
            raise ValueError("PUSH entries need a nonempty files array")
        if any(not Path(p).is_file() for p in entry["files"]):
            raise ValueError("PUSH input file missing")
    for entry in pushes + pulls:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("remote_dir"), str)
            or not entry["remote_dir"].startswith("/")
        ):
            raise ValueError("PUSH/PULL entries need an absolute remote_dir")
    for entry in pulls:
        if (
            not isinstance(entry.get("local_dir"), str)
            or not Path(entry["local_dir"]).is_absolute()
        ):
            raise ValueError("PULL entries need an absolute local_dir")
        if not strings(entry.get("exclude", [])):
            raise ValueError("PULL exclude must be an array of strings")
    start_check(env["LABEL"], env["MINUTES"])
    return args, pushes, pulls


def run_chain() -> int:
    args, pushes, pulls = validate_inputs()
    env = os.environ
    label, sha = env["LABEL"], env["SHA"]
    ops = Path(env["EXPERIMENT"]) / "ops"
    guest_code, guest_env = env["GUEST_CODE"], env["GUEST_ENV"]
    quote = shlex.quote
    jobs: list[tuple[str, str]] = []
    workload_started = False
    owned = set()
    phase_deadline = None
    rc = 0

    install_interrupts()

    def work(script, *argv):
        name, default = {
            "ssh": ("CHECKOUT_TIMEOUT_SECONDS", 180),
            "push": ("PUSH_TIMEOUT_SECONDS", 180),
        }.get(argv[0], ("COMMAND_TIMEOUT_SECONDS", 180))
        seconds = float(env.get(name, default))
        if phase_deadline is not None:
            seconds = min(seconds, max(0.01, phase_deadline - time.time()))
        command(script, *argv, seconds=seconds)

    def dispatch(name: str, argv: list[str], record: Path) -> str:
        record.write_text("UNKNOWN dispatch; inspect guest host-logs before retrying\n")
        seconds = float(env.get("DISPATCH_TIMEOUT_SECONDS", "60"))
        if phase_deadline is not None:
            seconds = min(seconds, max(0.01, phase_deadline - time.time()))
        reply = run_process(
            ["bash", str(HERE / "remote.sh"), "dispatch", name, *argv],
            seconds,
            text=True,
            stdout=subprocess.PIPE,
        )
        result = reply.stdout.strip()
        record.write_text(result + "\n")
        if reply.returncode:
            raise subprocess.CalledProcessError(reply.returncode, reply.args, output=reply.stdout)
        root = env["GUEST_LOG_ROOT"].rstrip("/") + "/"
        if (
            not result.startswith(root)
            or "\n" in result
            or "/" in result[len(root) :]
            or not result[len(root) :].startswith(name + ".")
        ):
            raise ValueError(
                f"UNKNOWN dispatch reply saved to {record}; inspect guest host-logs before retrying"
            )
        jobs.append((name, result))
        return result

    def wait(job: str, expected: str, deadline: int) -> None:
        command("remote.sh", "wait", job, expected, str(deadline))

    try:
        current = start(label, env["MINUTES"], env.get("START_RETRIES", "24"), owned)
        label = current["label"]
        (ops / f"session.{env['LABEL']}.json").write_text(json.dumps(current, indent=2) + "\n")
        deadline = int(
            datetime.fromisoformat(current["deadline"].replace("Z", "+00:00")).timestamp()
        )
        work_deadline = deadline - int(env.get("COPY_MARGIN_SECONDS", "120"))
        phase_deadline = work_deadline
        # Preserve a dirty guest checkout rather than silently overwriting it.
        checkout = f"""set -eu
test -f {quote(guest_env)}
if test ! -d {quote(guest_code)}/.git; then
    git clone {quote(env.get("REPO_URL", "https://github.com/alessiosavi/demand-forecast.git"))} {quote(guest_code)}
fi
cd {quote(guest_code)}
test -z "$(git status --porcelain)"
git fetch -q origin {quote(sha)}
git checkout -q --detach {quote(sha)}
test "$(git rev-parse HEAD)" = {quote(sha)}
test -z "$(git status --porcelain)"
printf '\\nexport FEV_SHA=%s\\n' {quote(sha)} >> {quote(guest_env)}
"""
        work("remote.sh", "ssh", checkout)
        for name, destination in (
            ("dispatch-job.sh", env["GUEST_DISPATCH"]),
            ("wait-job.sh", env["GUEST_WAIT"]),
        ):
            parent = str(Path(destination).parent)
            work("remote.sh", "push", str(HERE / "guest" / name), parent)
            if Path(destination).name != name:
                work("remote.sh", "ssh", shlex.join(["mv", "--", parent + "/" + name, destination]))
        if env.get("SKIP_SETUP", "0") != "1":
            setup = f"set -eu; export GUEST_ENV={quote(guest_env)}; source {quote(guest_env)}; cd {quote(guest_code)}; bash scripts/ec2/pretrain-setup.sh"
            job = dispatch(f"gates-{label}", ["bash", "-c", setup], ops / f"setup-job.{label}.txt")
            setup_deadline = min(
                work_deadline,
                int(datetime.now(timezone.utc).timestamp())
                + int(env.get("SETUP_TIMEOUT_SECONDS", "2400")),
            )
            wait(job, "0", setup_deadline)
            work("remote.sh", "ssh", f"grep -q '^SETUP_VERIFIED ' {quote(job + '/run.log')}")
        for entry in pushes:
            work("remote.sh", "push", *entry["files"], entry["remote_dir"])
        work("remote.sh", "push", env["GUEST_SCRIPT"], env["GUEST_HOME"])
        if int(datetime.now(timezone.utc).timestamp()) >= work_deadline:
            raise ValueError("Session has no work time remaining; no job dispatched")
        guest_script = env["GUEST_HOME"].rstrip("/") + "/" + Path(env["GUEST_SCRIPT"]).name
        guest_command = (
            f"set -eu; export GUEST_ENV={quote(guest_env)}; source {quote(guest_env)}; cd {quote(guest_code)}; "
            + shlex.join(["bash", guest_script, *args])
        )
        workload_started = True
        job = dispatch(label, ["bash", "-c", guest_command], ops / f"outer-job.{env['LABEL']}.txt")
        wait(job, env.get("EXPECTED_EXIT", "0"), work_deadline)
    except (
        OSError,
        ValueError,
        StopIteration,
        subprocess.SubprocessError,
        InterruptedError,
    ) as exc:
        print(f"Operator chain failed: {exc}", file=sys.stderr)
        rc = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 2
    finally:
        copy_end = time.monotonic() + float(env.get("COPY_BUDGET_SECONDS", "240"))

        def pull(remote_dir, local_dir, *extra):
            remaining = copy_end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Overall copy-back budget exhausted")
            command(
                "remote.sh",
                "pull",
                remote_dir,
                local_dir,
                *extra,
                seconds=min(remaining, float(env.get("COPY_TIMEOUT_SECONDS", "180"))),
            )

        try:
            if workload_started:
                for entry in pulls:
                    extra = [
                        arg
                        for pattern in entry.get("exclude", [])
                        for arg in ("--exclude", pattern)
                    ]
                    try:
                        pull(entry["remote_dir"], entry["local_dir"], *extra)
                    except InterruptedError:
                        raise
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        print(f"Result pull failed: {exc}", file=sys.stderr)
                        rc = rc or 2
            for name, job in jobs:
                try:
                    pull(job, str(ops / "jobs" / name))
                except InterruptedError:
                    raise
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    print(f"Job evidence pull failed: {exc}", file=sys.stderr)
                    rc = rc or 2
        except InterruptedError as exc:
            print(f"Copy-back interrupted: {exc}", file=sys.stderr)
            rc = rc or 130
        finally:
            with cleanup_signals():
                try:
                    owned_session(ledger(), owned)
                    if owned and any(s["txn_id"] in owned for s in unresolved(ledger())):
                        rc = stop(owned) or rc
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    print(f"Stop/reconcile requires attention: {exc}", file=sys.stderr)
                    rc = rc or 2
    return rc


def main() -> int:
    action, *args = sys.argv[1:]
    if action == "host":
        host("--fallback" in args)
    elif action == "start-check":
        start_check(*args)
    elif action == "status":
        data, _end = json.JSONDecoder().raw_decode(sys.stdin.read().lstrip())
        print(json.dumps(data, indent=2))
    elif action == "stop-evidence":
        stop_evidence(*args)
    elif action == "transfer-manifest":
        size, path = args
        if not re.fullmatch(r"[0-9]+", size.strip()):
            raise ValueError("Guest du did not return a byte count; transfer refused")
        Path(path).write_text(json.dumps({"bytes": int(size) + 4 * 1024 * 1024}) + "\n")
    elif action == "session-start":
        install_interrupts()
        start(*args)
    elif action == "session-stop":
        return stop()
    elif action == "ingress":
        install_interrupts()
        ingress()
    elif action == "rsync-transport":
        print(" ".join('"' + arg.replace('"', '""') + '"' for arg in args))
    elif action == "bounded":
        install_interrupts()
        try:
            result = run_process(args[1:], float(args[0]))
            return result.returncode if result.returncode >= 0 else 128 - result.returncode
        except subprocess.TimeoutExpired:
            return 124
    elif action == "run":
        return run_chain()
    else:
        raise ValueError(f"Unknown action: {action}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        subprocess.SubprocessError,
        InterruptedError,
    ) as exc:
        print(f"Operator error: {exc}", file=sys.stderr)
        sys.exit(2)
