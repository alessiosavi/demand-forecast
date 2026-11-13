"""Optional source integration: SIMONS_UPSTREAM must name the pinned clean clone."""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def jobs():
    return module_at("make_simons_job", ROOT / "scripts/gcp/jobs/make_simons_job.py")


@pytest.fixture(scope="session")
def upstream(tmp_path_factory):
    source = os.environ.get("SIMONS_UPSTREAM")
    if not source or not (Path(source) / "src/simons").is_dir():
        pytest.skip("set SIMONS_UPSTREAM to a clean Simons clone at third_party/simons/PINNED")
    target = tmp_path_factory.mktemp("patched-simons")
    for name in ("src", "configs", ".git"):
        shutil.copytree(Path(source) / name, target / name)
    installer = module_at("prepare_simons", ROOT / "third_party/simons/prepare.py")
    installer.prepare(target, ROOT / "third_party/simons")
    sys.path.insert(0, str(target / "src"))
    yield target
    sys.path.remove(str(target / "src"))


@pytest.fixture
def audit_receipt():
    """Write the corpus release handshake for tiny local layout fixtures."""
    import json

    def write(corpus):
        (corpus / "attempt-status.json").write_text(
            json.dumps(dict(attempt_id="tiny", complete=True, exit_code=0))
        )
        (corpus / "audit.json").write_text(
            json.dumps(
                dict(
                    attempt_id="tiny",
                    valid=True,
                    exit_code=0,
                    folders=[dict(folder="dataset", status="allowed", verdict="clean")],
                )
            )
        )

    return write
