"""Embedded bundle identities cover content independently of gzip transport bytes."""

import base64
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["simons", "adapt", "score"])
def test_bundle_identity_ignores_compression(kind, monkeypatch, tmp_path):
    source = ROOT / f"scripts/gcp/jobs/make_{kind}_job.py"
    spec = importlib.util.spec_from_file_location(f"make_{kind}_job", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = dict(
        repo_sha="a" * 40,
        upstream_pin=(ROOT / "third_party/simons/PINNED").read_text().strip(),
    )
    if kind == "simons":
        options.update(size="25m", leg="stage1", seed=11, micro_batches=2, checkpoint_minutes=10)
    else:
        options.update(
            name="identity-probe",
            tasks_uri="gs://offline/tasks.yaml",
            tasks_sha256="b" * 64,
            checkpoint_uri="gs://offline/base",
        )
        if kind == "adapt":
            options["names"] = ["alpha"]

    compress = gzip.compress
    normalized = []
    payloads = []
    for level in (0, 1, 9):
        members = []

        def pack(content, *, mtime):
            packed = compress(content, compresslevel=level, mtime=mtime)
            members.append((content, packed))
            return packed

        monkeypatch.setattr(module.gzip, "compress", pack)
        script = module.generate(**options)
        marker = kind.upper() + "_BUNDLE"
        snippet = script.split(f"<<'{marker}'\n", 1)[1].split(f"\n{marker}\n", 1)[0]

        def unpack(code, name):
            target = tmp_path / f"level-{level}" / name
            target.mkdir(parents=True)
            result = subprocess.run(
                [sys.executable, "-", str(target)], input=code, capture_output=True, text=True
            )
            return result, target

        result, target = unpack(snippet, "valid")
        assert result.returncode == 0, result.stderr
        payloads.append(tuple(packed for _, packed in members))
        for index, (content, packed) in enumerate(members):
            expected = hashlib.sha256(content).hexdigest()
            assert expected in snippet
            if content.startswith(b"{"):
                files = json.loads(content)
                assert content == json.dumps(files, sort_keys=True).encode()
                directory = target / "bundle" if kind == "adapt" else target
                for name, text in files.items():
                    assert (directory / name).read_text() == text
                if kind == "adapt":
                    assert (
                        json.loads((target / "inputs.json").read_bytes())["patch_bundle_sha256"]
                        == expected
                    )
            else:
                assert content == module.GUEST.encode()
                assert (target / f"{kind}.py").read_bytes() == content
                assert json.loads((target / "inputs.json").read_bytes())["guest_sha256"] == expected
            if kind == "simons":
                assert f"--bundle-sha256 {expected}" in script

            encoded = base64.b64encode(packed).decode()
            # Keep valid gzip and JSON/Python, but change the verified content.
            changed = base64.b64encode(compress(content + b"\n", mtime=0)).decode()
            result, _ = unpack(snippet.replace(encoded, changed), f"tampered-{index}")
            assert result.returncode != 0 and "AssertionError" in result.stderr
            # A transport-byte digest must not be accepted by the unpacker.
            legacy = hashlib.sha256(packed).hexdigest()
            result, _ = unpack(snippet.replace(expected, legacy), f"legacy-{index}")
            assert result.returncode != 0 and "AssertionError" in result.stderr
            script = script.replace(encoded, "PAYLOAD")
        normalized.append(script)

    assert len(set(payloads)) == 3
    assert normalized[0] == normalized[1] == normalized[2]
