"""Guest bootstrap contracts; no bench dependencies or live system services."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EC2 = ROOT / "scripts/ec2"


def embedded(source, marker):
    return (
        source.split(f"<<'{marker}'\n".encode(), 1)[1].split(f"\n{marker}\n".encode(), 1)[0] + b"\n"
    )


@pytest.mark.parametrize(
    ("name", "marker"),
    [("arm-deadline.sh", "ARM_DEADLINE"), ("budget-deadline.service", "BUDGET_DEADLINE")],
)
def test_embedded_files_match_standalone(name, marker):
    assert (
        embedded((EC2 / "pretrain-user-data.sh").read_bytes(), marker) == (EC2 / name).read_bytes()
    )


@pytest.mark.parametrize("name", ["arm-deadline.sh", "pretrain-user-data.sh", "pretrain-setup.sh"])
def test_shell_syntax(name):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    subprocess.run([bash, "-n", str(EC2 / name)], check=True, capture_output=True)


def test_only_deadline_helper_invokes_shutdown():
    for path in (ROOT / "scripts").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".service"}:
            continue
        source = path.read_text()
        if path.name == "arm-deadline.sh":
            continue
        if path.name == "pretrain-user-data.sh":
            source = source.replace(embedded(path.read_bytes(), "ARM_DEADLINE").decode(), "")
        assert not re.search(r"(?<![\w-])shutdown(?:\s+|[\"']\s*,)", source), path


def test_boot_unit_and_standalone_bootstrap():
    unit = (EC2 / "budget-deadline.service").read_text()
    for directive in (
        "Type=oneshot",
        "After=multi-user.target",
        "WantedBy=multi-user.target",
        "ExecStart=/usr/local/sbin/arm-deadline.sh boot",
        "DefaultDependencies=no",
    ):
        assert directive in unit.splitlines()
    source = (EC2 / "pretrain-user-data.sh").read_text()
    assert "systemctl enable --now budget-deadline.service" in source
    assert "/usr/local/sbin/arm-deadline.sh boot" not in source.splitlines()
    for directory in ("/home/ubuntu/fev", "/home/ubuntu/fev/ops"):
        assert f"install -d -o ubuntu -g ubuntu {directory}" in source.splitlines()
    assert not re.search(r"\b(python\d*|curl|wget|git|pip)\b", source)


@pytest.fixture
def guest(tmp_path):
    bash = shutil.which("bash")
    date = shutil.which("gdate") or shutil.which("date")
    if not bash or not date:
        pytest.skip("bash and GNU date required")
    if subprocess.run([date, "-d", "@0", "+%s"], capture_output=True).returncode:
        pytest.skip("GNU date required for Ubuntu deadline semantics")
    scripts = {
        "date": f'#!/bin/bash\nif [[ $# == 1 && $1 == +%s ]]; then echo 1793620800; else exec "{date}" "$@"; fi\n',
        "shutdown": (
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n'
            "if [[ $1 == --show ]]; then\n"
            '    printf "%s\\n" "$SHOW" >&2\n'
            '    exit "$SHOW_EXIT"\n'
            "fi\necho noisy-shutdown\n"
        ),
        "busctl": '#!/bin/bash\n[[ ${BUS_FAIL:-0} == 0 ]] || exit 1\nprintf \'(st) "poweroff" %s\\n\' "$SCHEDULED_US"\n',
        "flock": '#!/bin/bash\n[[ $1 == 9 && -e /dev/fd/9 ]] || exit 1\nprintf "%s\\n" "$1" >> "$LOCK_CALLS"\n',
    }
    for name, source in scripts.items():
        path = tmp_path / name
        path.write_text(source)
        path.chmod(0o755)
    calls = tmp_path / "calls"
    marker = tmp_path / "budget-deadline.booted"
    marker.touch()  # Existing set tests model a boot that already completed.
    lock = tmp_path / "budget-deadline.lock"
    lock_calls = tmp_path / "lock-calls"
    helper = tmp_path / "arm-deadline.sh"
    helper.write_text(
        (EC2 / "arm-deadline.sh")
        .read_text()
        .replace("/run/lock/budget-deadline.lock", str(lock))
        .replace("/run/budget-deadline.booted", str(marker))
    )

    def invoke(
        *args, scheduled="1793622600000000", bus_fail="0", show="", show_exit="0", fresh_boot=False
    ):
        if fresh_boot:
            marker.unlink(missing_ok=True)
        env = dict(
            os.environ,
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            CALLS=str(calls),
            LOCK_CALLS=str(lock_calls),
            SCHEDULED_US=scheduled,
            BUS_FAIL=bus_fail,
            SHOW=show,
            SHOW_EXIT=show_exit,
            TZ="Pacific/Honolulu",
        )
        result = subprocess.run([bash, str(helper), *args], env=env, capture_output=True, text=True)
        return result, calls.read_text().splitlines() if calls.exists() else []

    invoke.marker = marker
    invoke.lock = lock
    invoke.lock_calls = lock_calls
    return invoke


def test_set_requires_current_boot_before_changing_shutdown(guest):
    result, calls = guest("set", "2026-11-02T12:30:45Z", fresh_boot=True)
    assert result.returncode == 3
    assert result.stdout == "DEADLINE_NOT_READY 2026-11-02T12:30:45Z\n"
    assert calls == []
    result, _ = guest("boot", scheduled="1793622000000000")
    assert result.returncode == 0, result.stderr
    assert guest.marker.exists()
    result, calls = guest("set", "2026-11-02T12:30:45Z")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "DEADLINE_SET 2026-11-02T12:30:00Z\n"
    assert calls == ["-c", "-h +20", "-c", "-h 12:30"]
    assert guest.lock.is_file()
    assert guest.lock_calls.read_text().splitlines() == ["9", "9", "9"]


def test_failed_boot_readback_does_not_mark_ready(guest):
    result, _ = guest("boot", fresh_boot=True, bus_fail="1", show="no shutdown scheduled")
    assert result.returncode != 0
    assert not guest.marker.exists()


def test_set_reads_back_utc_and_keeps_stdout_machine_readable(guest):
    result, calls = guest("set", "2026-11-02T12:30:45Z")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "DEADLINE_SET 2026-11-02T12:30:00Z\n"
    assert calls == ["-c", "-h 12:30"]


def test_next_day_uses_relative_minutes(guest):
    result, calls = guest("set", "2026-11-03T12:30:45Z", scheduled="1793709000000000")
    assert result.returncode == 0, result.stderr
    assert calls == ["-c", "-h +1470"]


@pytest.mark.parametrize("scheduled", ["1793622646000000", "1793622584000000", "1793622645000001"])
def test_readback_after_requested_or_over_60_seconds_early_fails(guest, scheduled):
    result, _ = guest("set", "2026-11-02T12:30:45Z", scheduled=scheduled)
    assert result.returncode != 0
    assert result.stdout.startswith("DEADLINE_MISMATCH ")
    assert "DEADLINE_SET" not in result.stdout


def test_shutdown_show_fallback(guest):
    result, calls = guest(
        "set",
        "2026-11-02T12:30:45Z",
        bus_fail="1",
        show="Shutdown scheduled for Mon 2026-11-02 12:30:00 UTC, use 'shutdown -c' to cancel.",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "DEADLINE_SET 2026-11-02T12:30:00Z\n"
    assert calls == ["-c", "-h 12:30", "--show"]


@pytest.mark.parametrize("mode", ["boot", "set"])
def test_shutdown_show_unsupported_fails_closed(guest, mode):
    args = (mode, "2026-11-02T12:30:45Z") if mode == "set" else (mode,)
    result, calls = guest(
        *args,
        bus_fail="1",
        show="shutdown: unrecognized option '--show'",
        show_exit="1",
        fresh_boot=mode == "boot",
    )
    assert result.returncode == 1
    assert calls[-1] == "--show"
    if mode == "boot":
        assert result.stdout == ""
        assert "cannot read back grace deadline" in result.stderr
        assert not guest.marker.exists()
    else:
        assert result.stdout == "DEADLINE_MISMATCH unknown 2026-11-02T12:30:45Z\n"


@pytest.mark.parametrize("mode", ["boot", "grace"])
def test_every_boot_replaces_deadline_with_grace(guest, mode):
    result, calls = guest(mode, scheduled="1793622000000000")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "DEADLINE_GRACE 2026-11-02T12:20:00Z\n"
    assert calls == ["-c", "-h +20"]


@pytest.mark.parametrize(
    "requested", ["yesterday", "2026-11-02T11:59:59Z", "2026-11-02T14:30:00+02:00"]
)
def test_bad_request_preserves_existing_guard(guest, requested):
    result, calls = guest("set", requested)
    assert result.returncode != 0
    assert calls == []


def test_unreadable_deadline_fails_closed(guest):
    result, _ = guest("set", "2026-11-02T12:30:45Z", bus_fail="1", show="no shutdown scheduled")
    assert result.returncode != 0
    assert "DEADLINE_SET" not in result.stdout


@pytest.mark.parametrize("fault", [None, "head", "hash", "layout"])
def test_setup_warms_twelve_tasks_with_pinned_metadata(tmp_path, monkeypatch, fault):
    """Exercise the saved warm-up offline, including the five new Hub mappings."""
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    revision = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"
    repo = "autogluon/fev_datasets"
    paths, siblings, configs, colab_tasks = {}, [], [], {}
    loaded = []

    def spec(i):
        return SimpleNamespace(name=f"task-{i}", dataset_config=f"config-{i}", to_task=lambda: i)

    specs = [spec(i) for i in range(12)]
    for i, task in enumerate(specs):
        filename = f"nested/{task.name}/train.parquet"
        path = tmp_path / f"{i}.parquet"
        path.write_bytes(f"fixture-{i}".encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        paths[filename] = path
        siblings.append(
            SimpleNamespace(
                rfilename=filename, lfs=SimpleNamespace(sha256=sha, size=path.stat().st_size)
            )
        )
        configs.append(
            dict(
                config_name=task.dataset_config,
                data_files=[dict(split="train", path=f"nested/{task.name}/*.parquet")],
            )
        )
        if i < 7:
            colab_tasks[task.name] = dict(
                files={filename: dict(sha256=sha, bytes=path.stat().st_size)}
            )
    if fault == "hash":
        paths[siblings[-1].rfilename].write_bytes(b"changed")
    if fault == "layout":
        configs[-1]["data_files"][0]["path"] = "missing/*.parquet"
    info = SimpleNamespace(
        sha=revision if fault != "head" else "moved",
        siblings=siblings,
        card_data=SimpleNamespace(to_dict=lambda: dict(configs=configs)),
    )
    api = SimpleNamespace(
        dataset_info=lambda *a, **k: info,
        list_repo_refs=lambda *a, **k: SimpleNamespace(
            branches=[SimpleNamespace(name="main", target_commit=revision)]
        ),
    )

    def download(repo_id, filename, **kwargs):
        assert repo_id == repo and kwargs == dict(repo_type="dataset", revision=revision)
        return paths[filename]

    def fingerprint(task):
        loaded.append(task)
        return f"fingerprint-{task}"

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api, hf_hub_download=download)
    )
    monkeypatch.setitem(
        sys.modules,
        "demand_forecast.bench.specs",
        SimpleNamespace(
            dataset_fingerprint=fingerprint,
            load_task_specs=lambda selector: specs[:7] if selector == "weekly" else specs[7:],
        ),
    )
    colab = tmp_path / "colab.json"
    colab.write_text(json.dumps(dict(repo=repo, revision=revision, tasks=colab_tasks)))
    manifest = tmp_path / "manifest.json"
    monkeypatch.setenv("COLAB_MANIFEST", str(colab))
    monkeypatch.setenv("FEV_DATASET_MANIFEST", str(manifest))
    source = (
        (EC2 / "pretrain-setup.sh")
        .read_text()
        .split("ops/pin-and-warm.py <<'PY'\n", 1)[1]
        .split("\nPY\n", 1)[0]
    )
    if fault:
        with pytest.raises(AssertionError):
            exec(compile(source, "pin-and-warm.py", "exec"), {})
        assert not manifest.exists()
    else:
        exec(compile(source, "pin-and-warm.py", "exec"), {})
        assert loaded == list(range(12))
        assert len(json.loads(manifest.read_text())["files"]) == 12
        # A later session compares fingerprints as well as hashes and sizes.
        before = manifest.read_bytes()
        specs[-1].to_task = lambda: "changed"
        with pytest.raises(AssertionError):
            exec(compile(source, "pin-and-warm.py", "exec"), {})
        assert manifest.read_bytes() == before
