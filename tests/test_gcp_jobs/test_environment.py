"""Execute generated jobs against a fake image without downloads or a GPU."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

PIN = "8e338e915d26c8960713889ef12ab1cf5c08d591"
SHA = "5591bb0" * 5 + "abcde"


@pytest.fixture
def machine(jobs, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    image = tmp_path / "image"
    (image / "torch").mkdir(parents=True)
    (image / "torch/__init__.py").write_text(
        "import os\n"
        "__version__ = os.environ.get('IMAGE_TORCH', '2.9.1+cu129')\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return os.environ.get('IMAGE_CUDA', '1') == '1'\n"
        "class version:\n"
        "    cuda = '12.9'\n"
    )
    source = tmp_path / "source"
    package = source / "src/simons"
    (package / "training").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "training/__init__.py").write_text("")
    (package / "training/spot_job.py").write_text(
        textwrap.dedent("""\
            if __name__ == "__main__":
                import json, os, pathlib, sys
                path = pathlib.Path(sys.argv[sys.argv.index("--environment-file") + 1])
                with open(os.environ["TRAIN_LOG"], "a") as handle:
                    handle.write(json.dumps({"environment": path.read_text()}) + "\\n")
            """)
    )
    metadata = source / "src/simons-0.1.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: simons\nVersion: 0.1.1\nRequires-Dist: torch>=2.5\n"
    )
    python = bin_dir / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
            import json, os, pathlib, shutil, sys
            args = sys.argv[1:]
            with open(os.environ["COMMAND_LOG"], "a") as handle:
                handle.write(json.dumps(args) + "\\n")
            if args[:2] == ["-m", "venv"] or args[:1] == ["--fake-create-env"]:
                if args[:2] == ["-m", "venv"] and os.environ.get("FAIL_VENV") == "1":
                    pathlib.Path(args[-1]).mkdir()
                    sys.exit("ensurepip is not available")
                target = pathlib.Path(args[-1]) / "bin"
                target.mkdir(parents=True, exist_ok=True)
                (target / "python").symlink_to(os.environ["FAKE_PYTHON"])
                sys.exit(0)
            if args[:3] == ["-m", "pip", "install"]:
                if "uv" in args:
                    target = pathlib.Path(args[args.index("--target") + 1]) / "bin"
                    target.mkdir(parents=True)
                    shutil.copy2(pathlib.Path(os.environ["FAKE_BIN"]) / "uv", target / "uv")
                else:
                    constraint = pathlib.Path(args[args.index("--constraint") + 1])
                    assert constraint.read_text().strip() == "torch==" + os.environ.get("IMAGE_TORCH", "2.9.1+cu129")
                    assert "--editable" in args
                    tree = pathlib.Path(args[-1])
                    pathlib.Path(os.environ["INSTALLED_TREE"]).write_text(str(tree))
                sys.exit(0)
            if args[:3] == ["-m", "pip", "freeze"]:
                print("simons==0.1.1\\ntorch==2.9.1+cu129")
                sys.exit(0)
            if args and args[0].endswith("/prepare.py"):
                sys.exit(0)  # Patch application is covered by the pinned-clone tests.
            installed = pathlib.Path(os.environ["INSTALLED_TREE"])
            if installed.exists():
                root = installed.read_text() if os.environ.get("PATCHED", "1") == "1" else os.environ["FAKE_SOURCE"]
                os.environ["PYTHONPATH"] = str(pathlib.Path(root) / "src") + os.pathsep + os.environ["PYTHONPATH"]
            os.execv(sys.executable, [sys.executable, *args])
            """)
    )
    (bin_dir / "uv").write_text(
        "#!/bin/bash\n"
        'printf "uv %s\\n" "$*" >> "$UV_LOG"\n'
        'exec "$FAKE_PYTHON" --fake-create-env "${@: -1}"\n'
    )
    (bin_dir / "git").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = clone ]; then mkdir -p "${@: -1}"; '
        'cp -R "$FAKE_SOURCE/." "${@: -1}"; fi\n'
    )
    (bin_dir / "nvidia-smi").write_text('#!/bin/bash\necho "580.105.08"\n')
    for executable in bin_dir.iterdir():
        executable.chmod(0o755)
    script = tmp_path / "job.sh"
    script.write_text(
        jobs.generate(
            size="25m",
            leg="stage1",
            seed=11,
            micro_batches=1,
            checkpoint_minutes=10,
            repo_sha=SHA,
            upstream_pin=PIN,
        )
    )
    run_dir = tmp_path / "run"
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(image),
        "RUN_DIR": str(run_dir),
        "STATE_URI": "gs://offline",
        "FAKE_BIN": str(bin_dir),
        "FAKE_PYTHON": str(python),
        "FAKE_SOURCE": str(source),
        "COMMAND_LOG": str(tmp_path / "commands.jsonl"),
        "UV_LOG": str(tmp_path / "uv.log"),
        "TRAIN_LOG": str(tmp_path / "training.jsonl"),
        "INSTALLED_TREE": str(tmp_path / "installed.txt"),
    }

    def run(**overrides):
        return subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env={**env, **overrides}
        )

    return run, run_dir


@pytest.mark.parametrize("fails", ["0", "1"], ids=["stdlib", "uv-fallback"])
def test_environment_creation_reuses_image_and_rebuilds_on_restart(machine, tmp_path, fails):
    run, run_dir = machine
    for _ in range(2):
        result = run(FAIL_VENV=fails)
        assert result.returncode == 0, result.stderr
    installs = sorted(run_dir.glob("simons-install-*"))
    assert len(installs) == 2
    commands = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
    venvs = [args for args in commands if args[:2] == ["-m", "venv"]]
    assert len(venvs) == 2
    assert all("--system-site-packages" in args for args in venvs)
    if fails == "1":
        uv = (tmp_path / "uv.log").read_text().splitlines()
        assert len(uv) == 2
        for line in uv:
            for flag in ("--system-site-packages", "--clear", "--seed", "--no-python-downloads"):
                assert flag in line
            assert "--python " + str(tmp_path / "bin/python3") in line
    else:
        assert not (tmp_path / "uv.log").exists()
    records = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert "torch==2.9.1+cu129" in record["environment"]
        line = next(
            line for line in record["environment"].splitlines() if line.startswith("# runtime ")
        )
        runtime = json.loads(line.removeprefix("# runtime "))
        assert runtime["torch"] == "2.9.1+cu129"
        assert runtime["cuda"] == "12.9"
        assert runtime["cuda_available"] is True
        assert runtime["nvidia_driver"] == ["580.105.08"]
        assert runtime["torch_path"] == str(tmp_path / "image/torch/__init__.py")
        assert runtime["simons_path"] in [str(path / "Simons/src/simons") for path in installs]
    for install in installs:
        assert (install / "torch-constraint.txt").read_text().strip() == "torch==2.9.1+cu129"


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"IMAGE_CUDA": "0"}, "CUDA is unavailable"),
        ({"IMAGE_TORCH": "2.4.0"}, "torch>=2.5"),
        ({"PATCHED": "0"}, "patched tree"),
    ],
    ids=["no-cuda", "old-torch", "wrong-simons"],
)
def test_environment_assertions_stop_before_training(machine, tmp_path, overrides, reason):
    run, _ = machine
    result = run(FAIL_VENV="1", **overrides)
    assert result.returncode != 0
    assert reason in result.stderr
    reasons = [line for line in result.stderr.splitlines() if "environment check failed:" in line]
    assert len(reasons) == 1
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "training.jsonl").exists()
