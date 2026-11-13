"""The generated cloud guest job is pinned, restartable and shell-parseable."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.corpus_v4.make_corpus_job import render


def test_generated_job_syntax_and_contract(tmp_path):
    script = render("a" * 40, "gs://research/corpus-v4")
    path = tmp_path / "job.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    for text in (
        "RUN_DIR",
        "STATE_URI",
        '"$RUN_DIR/state"',
        "sleep 600",
        "fetch_native",
        "audit_fev_overlap",
        "audit.json",
        "stats.json",
        "git checkout --detach",
        "exit",
    ):
        assert text in script
    assert "a" * 40 in script and "gs://research/corpus-v4" in script
    assert "HF_TOKEN=" not in script


def test_environment_uses_system_packages_and_uv_fallback():
    script = render("a" * 40, "gs://research/corpus-v4")
    assert not re.search(r'python3 -m venv\s+"?\$', script)
    assert "if ! python3 -m venv" in script
    assert "--system-site-packages" in script
    assert "--no-python-downloads" in script
    assert "2.9.1" not in script and "cu129" not in script


@pytest.fixture
def environment_guest(tmp_path, monkeypatch):
    """Run the real job and embedded preflight; replace network/install commands."""
    guest = tmp_path / "guest"
    checkout = guest / f"corpus-repo-{'a' * 40}"
    (checkout / ".git").mkdir(parents=True)
    package = checkout / "demand_forecast"
    package.mkdir()
    (package / "__init__.py").touch()
    modules = tmp_path / "modules"
    modules.mkdir()
    for name in ("huggingface_hub", "datasets", "fev", "numpy", "pandas", "yaml"):
        (modules / f"{name}.py").touch()
    (modules / "pyarrow").mkdir()
    (modules / "pyarrow/__init__.py").touch()
    (modules / "pyarrow/parquet.py").touch()
    (modules / "torch.py").write_text(
        'import os\n__version__ = os.environ["IMAGE_TORCH_VERSION"]\n'
    )
    for name in ("scripts", "scripts/corpus_v4"):
        (checkout / name).mkdir(exist_ok=True)
        (checkout / name / "__init__.py").touch()
    for name in ("fetch_native", "audit_fev_overlap"):
        (checkout / f"scripts/corpus_v4/{name}.py").touch()
    metadata = modules / "demand_forecast-2.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: demand-forecast\nVersion: 2.0.0\n")
    (metadata / "direct_url.json").write_text(
        json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    # All Python here is stdlib-only, including execution of the generated heredocs.
    tool = (
        f"#!{sys.executable}\n"
        + r"""
import json, os, pathlib, shutil, sys
name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as log:
    log.write(json.dumps({"name": name, "args": args, "home": os.environ.get("HOME"),
                          "hf_home": os.environ.get("HF_HOME")}) + "\n")
def make_venv(target, clear):
    if clear:
        shutil.rmtree(target, ignore_errors=True)
    (target / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, target / "bin/python")
    (target / "bin/activate").write_text(
        'export PATH="' + str(target / "bin") + ':$PATH"\n')
if name == "git":
    if "rev-parse" in args:
        print(os.environ.get("CHECKOUT_HEAD", "a" * 40) if args[0] == "-C" else "a" * 40)
elif name == "gcloud":
    pass
elif name == "sleep":
    sys.exit(1)
elif name == "uv":
    assert args[:5] == ["venv", "--clear", "--seed", "--system-site-packages", "--python"]
    assert args[5] == str(pathlib.Path(os.environ["FAKE_BIN"]) / "python3")
    assert args[6] == "--no-python-downloads"
    target = pathlib.Path(args[-1])
    make_venv(target, True)
    if os.environ.get("UV_FAIL") == "1":
        (target / "interrupted").touch()
        sys.exit(19)
elif args[:2] == ["-m", "venv"]:
    target = pathlib.Path(args[-1])
    if os.environ.get("VENV_FAIL") == "1":
        (target / "bin").mkdir(parents=True, exist_ok=True)
        (target / "interrupted").touch()
        sys.exit(1)  # Model Ubuntu's missing ensurepip, including partial files.
    assert "--system-site-packages" in args
    make_venv(target, "--clear" in args)
elif args[:3] == ["-m", "pip", "install"]:
    if "--target" in args:
        target = pathlib.Path(args[args.index("--target") + 1]) / "bin"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, target / "uv")
    elif os.environ.get("REPLACE_TORCH") == "1":
        pathlib.Path(os.environ["PYTHONPATH"].split(os.pathsep)[0], "torch.py").write_text(
            '__version__ = "0.0"\n')
elif args == ["-m", "pip", "freeze"]:
    print("torch==" + os.environ["IMAGE_TORCH_VERSION"])
    print("# fake installed environment")
elif args[:2] == ["-m", "scripts.corpus_v4.fetch_native"]:
    sys.exit(int(os.environ.get("FETCH_EXIT", "0")))
elif args[:2] == ["-m", "scripts.corpus_v4.audit_fev_overlap"]:
    sys.exit(int(os.environ.get("AUDIT_EXIT", "0")))
else:
    os.execv(sys.executable, [sys.executable, *args])
"""
    )
    for name in ("python3", "git", "gcloud", "sleep"):
        path = binary / name
        path.write_text(tool)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    # Model an editable .pth link after site-packages, as well as the implicit cwd.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(modules), str(checkout))))
    monkeypatch.setenv("FAKE_BIN", str(binary))
    monkeypatch.setenv("RUN_DIR", str(guest))
    monkeypatch.setenv("STATE_URI", "gs://fake-state/job")
    monkeypatch.setenv("CALL_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("IMAGE_TORCH_VERSION", "2.9.1+cu129")
    monkeypatch.delenv("HOME", raising=False)
    path = tmp_path / "job.sh"
    path.write_text(render("a" * 40, "gs://fake-corpus/job"))

    def invoke():
        result = subprocess.run(["bash", str(path)], capture_output=True, text=True, timeout=15)
        calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        return result, calls

    return guest, modules, invoke


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("version", ["2.9.1+cu129", "2.10.0+image"])
def test_guest_builds_environment_before_fetch(environment_guest, monkeypatch, fallback, version):
    guest, _, invoke = environment_guest
    monkeypatch.setenv("VENV_FAIL", str(int(fallback)))
    monkeypatch.setenv("IMAGE_TORCH_VERSION", version)
    result, calls = invoke()
    assert result.returncode == 0, result.stderr
    assert any(call["name"] == "uv" for call in calls) == fallback
    install = next(
        call
        for call in calls
        if call["name"] == "python" and call["args"][:3] == ["-m", "pip", "install"]
    )
    assert ".[bench]" in install["args"]
    constraint = Path(install["args"][install["args"].index("--constraint") + 1])
    assert constraint.read_text() == f"torch=={version}\n"
    assert (guest / "state/environment.txt").read_text().startswith(f"torch=={version}\n")
    assert install["home"] == str(guest / "home")
    assert Path(install["home"]).is_dir()
    assert not Path(install["hf_home"]).is_relative_to(guest / "state")
    phases = [call["args"][:2] for call in calls if call["name"] == "python"]
    assert phases[-2:] == [
        ["-m", "scripts.corpus_v4.fetch_native"],
        ["-m", "scripts.corpus_v4.audit_fev_overlap"],
    ]
    assert json.loads((guest / "state/attempt-status.json").read_text())["complete"]


@pytest.mark.parametrize("fallback", [False, True])
def test_guest_restart_clears_half_built_environment(environment_guest, monkeypatch, fallback):
    guest, _, invoke = environment_guest
    monkeypatch.setenv("VENV_FAIL", "1")
    monkeypatch.setenv("UV_FAIL", "1")
    result, _ = invoke()
    assert result.returncode == 19, result.stderr
    assert (guest / "corpus-venv/interrupted").exists()
    monkeypatch.setenv("VENV_FAIL", str(int(fallback)))
    monkeypatch.delenv("UV_FAIL")
    result, _ = invoke()
    assert result.returncode == 0, result.stderr
    assert not (guest / "corpus-venv/interrupted").exists()


@pytest.mark.parametrize("broken", ["imports", "checkout", "torch"])
def test_failed_preflight_stops_before_download(environment_guest, monkeypatch, broken):
    guest, modules, invoke = environment_guest
    if broken == "imports":
        (modules / "pyarrow/parquet.py").unlink()
    elif broken == "checkout":
        (modules / "demand_forecast-2.0.0.dist-info/direct_url.json").write_text(
            json.dumps({"url": "file:///wrong-checkout", "dir_info": {"editable": True}})
        )
    else:
        monkeypatch.setenv("REPLACE_TORCH", "1")
    result, calls = invoke()
    assert result.returncode != 0
    assert not any(call["args"][:2] == ["-m", "scripts.corpus_v4.fetch_native"] for call in calls)
    status = json.loads((guest / "state/attempt-status.json").read_text())
    assert not status["complete"]
    assert status["exit_code"] == result.returncode


def test_preflight_accepts_editable_install_with_source_egg_info(environment_guest):
    """Source metadata must not hide the installed editable's direct_url.json."""
    guest, _, invoke = environment_guest
    checkout = guest / f"corpus-repo-{'a' * 40}"
    egg_info = checkout / "demand_forecast.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("Name: demand-forecast\nVersion: 2.0.0\n")
    # No installer/network needed: the child runs real importlib.metadata over
    # this source egg-info and the installed dist-info built by the fixture.
    result, calls = invoke()
    assert result.returncode == 0, result.stderr
    assert any(call["args"][:2] == ["-m", "scripts.corpus_v4.fetch_native"] for call in calls)


@pytest.mark.parametrize(
    "broken",
    ["non-editable", "other-editable", "missing", "stale-code", "relative-url", "remote-url"],
)
def test_preflight_rejects_wrong_install_despite_matching_cwd(environment_guest, broken):
    """A checkout on sys.path[0] cannot stand in for a correct installation."""
    guest, modules, invoke = environment_guest
    checkout = guest / f"corpus-repo-{'a' * 40}"
    direct_url = modules / "demand_forecast-2.0.0.dist-info/direct_url.json"
    if broken == "non-editable":
        direct_url.write_text(json.dumps({"url": checkout.as_uri(), "dir_info": {}}))
    elif broken == "other-editable":
        direct_url.write_text(json.dumps({"url": modules.as_uri(), "dir_info": {"editable": True}}))
    elif broken == "missing":
        direct_url.unlink()
    elif broken in {"relative-url", "remote-url"}:
        url = "file:" if broken == "relative-url" else "file://another-host" + checkout.as_posix()
        direct_url.write_text(json.dumps({"url": url, "dir_info": {"editable": True}}))
    else:
        stale = modules / "demand_forecast"
        stale.mkdir()
        (stale / "__init__.py").touch()
    result, calls = invoke()
    assert result.returncode != 0, result.stdout
    assert "pinned editable checkout" in result.stderr
    assert not any(call["args"][:2] == ["-m", "scripts.corpus_v4.fetch_native"] for call in calls)


@pytest.mark.parametrize("active_editable", [False, True])
def test_preflight_respects_installed_metadata_precedence(
    environment_guest, monkeypatch, active_editable
):
    """An inherited distribution cannot override the first installed copy."""
    guest, modules, invoke = environment_guest
    checkout = guest / f"corpus-repo-{'a' * 40}"
    inherited = modules / "system-site-packages"
    metadata = inherited / "demand_forecast-1.0.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: demand-forecast\nVersion: 1.0.0\n")
    for path, editable in (
        (modules / "demand_forecast-2.0.0.dist-info", active_editable),
        (metadata, not active_editable),
    ):
        (path / "direct_url.json").write_text(
            json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": editable}})
        )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(modules), str(inherited), str(checkout))))
    result, calls = invoke()
    if active_editable:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "pinned editable checkout" in result.stderr
        assert not any(
            call["args"][:2] == ["-m", "scripts.corpus_v4.fetch_native"] for call in calls
        )


@pytest.mark.parametrize("broken", ["fetch_native", "audit_fev_overlap", "head"])
def test_preflight_rejects_wrong_corpus_source_or_head(environment_guest, monkeypatch, broken):
    guest, modules, invoke = environment_guest
    checkout = guest / f"corpus-repo-{'a' * 40}"
    if broken == "head":
        monkeypatch.setenv("CHECKOUT_HEAD", "b" * 40)
        expected = "HEAD does not match REPO_SHA"
    else:
        stale = modules / f"{broken}.py"
        stale.touch()
        module = checkout / f"scripts/corpus_v4/{broken}.py"
        module.unlink()
        module.symlink_to(stale)
        expected = "corpus module is not from the pinned checkout"
    result, calls = invoke()
    assert result.returncode != 0
    assert expected in result.stderr
    assert not any(call["args"][:2] == ["-m", "scripts.corpus_v4.fetch_native"] for call in calls)


@pytest.mark.parametrize(
    "sha,prefix",
    [
        ("main", "gs://research/corpus"),
        ("a" * 40, "file:///tmp/corpus"),
        ("a" * 40, "gs://x/$(id)"),
    ],
)
def test_bad_job_parameters_refused(sha, prefix):
    with pytest.raises(ValueError):
        render(sha, prefix)


def test_failed_attempt_does_not_publish_restored_success(tmp_path, monkeypatch):
    guest = tmp_path / "guest"
    audit = guest / "state/audit/audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text('{"exit_code":0,"valid":true}')
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, source in {
        "git": "#!/bin/bash\nexit 7\n",
        "sleep": "#!/bin/bash\nexit 1\n",
        "gcloud": (
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
            'if [[ $2 == cp && $3 == */attempt-status.json ]]; then cp "$3" "$STATUS_COPY"; fi\nexit 0\n'
        ),
    }.items():
        path = binary / name
        path.write_text(source)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.setenv("RUN_DIR", str(guest))
    monkeypatch.setenv("STATE_URI", "gs://fake-state/job")
    monkeypatch.setenv("CALL_LOG", str(tmp_path / "calls"))
    monkeypatch.setenv("STATUS_COPY", str(tmp_path / "status.json"))
    path = tmp_path / "job.sh"
    path.write_text(render("a" * 40, "gs://fake-corpus/job"))
    result = subprocess.run(["bash", str(path)], capture_output=True, timeout=10)
    assert result.returncode == 7
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["exit_code"] == 7 and not status["complete"]
    assert status["repo_sha"] == "a" * 40
    calls = (tmp_path / "calls").read_text()
    assert not any(" cp " in f" {line} " and "/audit.json" in line for line in calls.splitlines())


@pytest.mark.parametrize("fetch_code,audit_code", [(2, 0), (2, 2), (0, 2)])
def test_guest_audits_completed_folders_after_fetch_failure(
    environment_guest, monkeypatch, fetch_code, audit_code
):
    guest, _, invoke = environment_guest
    monkeypatch.setenv("FETCH_EXIT", str(fetch_code))
    monkeypatch.setenv("AUDIT_EXIT", str(audit_code))
    result, calls = invoke()
    assert result.returncode != 0
    phases = [
        c["args"]
        for c in calls
        if c["args"][:2]
        in (["-m", "scripts.corpus_v4.fetch_native"], ["-m", "scripts.corpus_v4.audit_fev_overlap"])
    ]
    assert len(phases) == 2
    assert all("--memory-budget-gib" in args for args in phases)
    assert not json.loads((guest / "state/attempt-status.json").read_text())["complete"]
