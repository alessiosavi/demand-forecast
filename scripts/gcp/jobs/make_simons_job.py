#!/usr/bin/env python3
"""Generate a self-contained Simons job for the RUN_DIR / STATE_URI guest contract."""

import argparse
import base64
import gzip
import hashlib
import json
import math
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "third_party/simons"
LEGS = ("stage1", "stage2", "stage3", "stage4", "stage4-long", "qualify", "export")
OVERRIDE_LEGS = {
    "covariate_weight": ("stage3", "stage4", "stage4-long"),
    "covariate_mix": ("stage3", "stage4", "stage4-long"),
    "long_weight": ("stage4", "stage4-long"),
}


def canonical_condition(name, overrides):
    """Encode the closed set of fork overrides independently of input spelling."""
    if name is None and not overrides:
        return None
    if name is None or not overrides:
        raise ValueError("condition and overrides require both or neither")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", name):
        raise ValueError("invalid condition name")
    values = {}
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or key not in OVERRIDE_LEGS or key in values:
            raise ValueError("unknown or duplicate override key")
        try:
            numbers = [float(x) for x in value.split(",")]
        except ValueError as error:
            raise ValueError("override must contain finite numbers") from error
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("override must contain finite numbers")
        if key == "covariate_mix":
            if (
                len(numbers) != 3
                or any(x < 0 or x > 1 for x in numbers)
                or abs(sum(numbers) - 1) > 1e-9
            ):
                raise ValueError("covariate_mix requires three shares summing to one")
        elif len(numbers) != 1 or not 0 < numbers[0] <= 1:
            raise ValueError("override weight must be in (0, 1]")
        values[key] = ",".join(repr(x) for x in numbers)
    return {"name": name, "overrides": dict(sorted(values.items()))}


def validate_fork_options(size, leg, recipe, condition, override, attempt, unsafe, corpus_uri):
    result = canonical_condition(condition, override)
    if result is not None or attempt is not None:
        if (
            size != "25m"
            or recipe != "full"
            or leg not in (*OVERRIDE_LEGS["covariate_weight"], "export")
        ):
            raise ValueError("fork flags require 25m full stage3, stage4, stage4-long or export")
    if attempt is not None:
        if not isinstance(attempt, str) or not attempt.strip():
            raise ValueError("provisional attempt id must be nonempty")
        if unsafe:
            raise ValueError("provisional and qualification corpus flags are mutually exclusive")
        if not corpus_uri:
            raise ValueError("provisional receipt requires a corpus URI")
    return result


ENVIRONMENT = r"""# Keep the image's CUDA-matched torch, including its local version suffix.
python3 - "$scratch/image-torch.json" "$scratch/torch-constraint.txt" <<'SIMONS_IMAGE'
import json, pathlib, sys
try:
    import torch
    image = {"version": torch.__version__, "path": str(pathlib.Path(torch.__file__).resolve())}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(image))
    pathlib.Path(sys.argv[2]).write_text("torch==" + image["version"] + "\n")
except Exception as error:
    sys.exit("Simons environment check failed: image torch: " + " ".join(str(error).splitlines()))
SIMONS_IMAGE
if ! python3 -m venv --system-site-packages "$scratch/venv"; then
    # Ubuntu's system Python may lack ensurepip. Bootstrap only a small tool in
    # scratch; uv seeds pip without ensurepip and clears the incomplete venv.
    python3 -m pip install --target "$scratch/uv-bootstrap" uv
    "$scratch/uv-bootstrap/bin/uv" venv --clear --seed --system-site-packages \
        --python "$(command -v python3)" --no-python-downloads "$scratch/venv"
fi
"$scratch/venv/bin/python" -m pip install --constraint "$scratch/torch-constraint.txt" \
    packaging --editable "$upstream"
"$scratch/venv/bin/python" -m pip freeze > "$scratch/environment.txt"
"$scratch/venv/bin/python" - "$upstream" "$scratch/image-torch.json" "$scratch/environment.txt" <<'SIMONS_ENVIRONMENT'
import json, pathlib, subprocess, sys
from importlib import metadata
try:
    import torch
    import simons
    from packaging.requirements import Requirement
    from simons.training import spot_job

    root = pathlib.Path(sys.argv[1]).resolve() / "src/simons"
    image = json.loads(pathlib.Path(sys.argv[2]).read_text())
    requirement = next(Requirement(r) for r in metadata.requires("simons")
                       if Requirement(r).name == "torch")
    if not requirement.specifier.contains(torch.__version__):
        raise RuntimeError("torch " + torch.__version__ + " does not satisfy " + str(requirement))
    if torch.__version__ != image["version"] or str(pathlib.Path(torch.__file__).resolve()) != image["path"]:
        raise RuntimeError("torch is not the image's CUDA-matched package")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if (pathlib.Path(simons.__file__).resolve() != root / "__init__.py"
            or pathlib.Path(spot_job.__file__).resolve() != root / "training/spot_job.py"):
        raise RuntimeError("installed simons is not the patched tree")
    drivers = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if not drivers:
        raise RuntimeError("NVIDIA driver version is unavailable")
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
               "torch_path": image["path"], "cuda": torch.version.cuda,
               "cuda_available": True, "nvidia_driver": drivers, "simons_path": str(root)}
    with pathlib.Path(sys.argv[3]).open("a") as handle:
        handle.write("# runtime " + json.dumps(runtime, sort_keys=True) + "\n")
except Exception as error:
    sys.exit("Simons environment check failed: " + " ".join(str(error).splitlines()))
SIMONS_ENVIRONMENT
"""


def generate(
    *,
    size,
    leg,
    seed,
    micro_batches,
    checkpoint_minutes,
    repo_sha,
    upstream_pin,
    corpus_uri=None,
    qualify_leg="stage1",
    name=None,
    warmup=20,
    max_micro_batches=128,
    qualify_step=None,
    recipe="full",
    screen_updates=None,
    unsafe_qualification_allow_unaudited_corpus=False,
    condition=None,
    override=None,
    provisional_unaudited_corpus_attempt=None,
):
    """Return executable shell text embedding the complete pinned source patch bundle."""
    pin = (BUNDLE / "PINNED").read_text().strip()
    if upstream_pin != pin:
        raise ValueError(f"upstream pin must equal {pin}")
    if not re.fullmatch(r"[0-9a-f]{40}", repo_sha):
        raise ValueError("repo SHA must be the full 40-character revision")
    if size not in ("25m", "12m", "60m") or leg not in LEGS or qualify_leg not in LEGS[:5]:
        raise ValueError("invalid size or leg")
    if (
        seed < 0
        or micro_batches < 1
        or max_micro_batches < micro_batches
        or warmup < 0
        or not math.isfinite(checkpoint_minutes)
        or checkpoint_minutes <= 0
    ):
        raise ValueError("invalid seed, micro-batches, warm-up or checkpoint minutes")
    if recipe not in ("full", "screen"):
        raise ValueError("invalid recipe")
    condition_record = validate_fork_options(
        size,
        leg,
        recipe,
        condition,
        override,
        provisional_unaudited_corpus_attempt,
        unsafe_qualification_allow_unaudited_corpus,
        corpus_uri,
    )
    if size == "60m" and recipe == "screen":
        raise ValueError("60M short schedule needs a registered protocol")
    if screen_updates is not None and recipe != "screen":
        raise ValueError("screen updates require the screen recipe")
    if recipe == "screen" and screen_updates is None:
        screen_updates = 25000
    if screen_updates is not None and (type(screen_updates) is not int or screen_updates < 100):
        raise ValueError("screen updates must be an integer >= 100")
    if unsafe_qualification_allow_unaudited_corpus and leg != "qualify":
        raise ValueError("unsafe corpus override is qualification-only")
    needs_corpus = (qualify_leg if leg == "qualify" else leg) in LEGS[2:5]
    if needs_corpus and not corpus_uri:
        raise ValueError("a corpus URI is required for legs 3 to 5")
    if corpus_uri and not re.fullmatch(r"gs://[A-Za-z0-9._-]+(?:/[^\r\n]*)?", corpus_uri):
        raise ValueError("corpus URI must be a Cloud Storage prefix")
    name = name or f"{size}-{qualify_leg}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(
            "qualification name must start with a letter or digit; "
            "then contain only letters, digits, underscore or hyphen"
        )
    paths = [BUNDLE / p for p in ("PINNED", "NOTICE", "prepare.py")]
    patches = sorted((BUNDLE / "patches").glob("*.patch"))
    if not patches:
        raise ValueError("missing upstream patch set")
    files = {str(p.relative_to(BUNDLE)): p.read_text() for p in [*paths, *patches]}
    content = json.dumps(files, sort_keys=True).encode()
    digest = hashlib.sha256(content).hexdigest()
    packed = gzip.compress(content, mtime=0)
    payload = base64.b64encode(packed).decode()
    arguments = [
        "--leg",
        leg,
        "--size",
        size,
        "--seed",
        str(seed),
        "--micro-batches",
        str(micro_batches),
        "--checkpoint-minutes",
        str(checkpoint_minutes),
        "--repo-sha",
        repo_sha,
        "--upstream-pin",
        pin,
        "--bundle-sha256",
        digest,
        "--qualify-leg",
        qualify_leg,
        "--name",
        name,
        "--warmup",
        str(warmup),
        "--max-micro-batches",
        str(max_micro_batches),
    ]
    if recipe == "screen":
        arguments += ["--recipe", recipe, "--screen-updates", str(screen_updates)]
    if unsafe_qualification_allow_unaudited_corpus:
        arguments += ["--unsafe-qualification-allow-unaudited-corpus"]
    if condition_record is not None:
        arguments += ["--condition", condition_record["name"]]
        for key, value in condition_record["overrides"].items():
            arguments += ["--override", f"{key}={value}"]
    if provisional_unaudited_corpus_attempt is not None:
        arguments += [
            "--provisional-unaudited-corpus-attempt",
            provisional_unaudited_corpus_attempt,
        ]
    if qualify_step is not None:
        arguments += ["--qualify-step", str(qualify_step)]
    if corpus_uri:
        arguments += ["--corpus-uri", corpus_uri]
    # A marker can upload before its artifact. Check sizes and checkpoint hashes
    # before taking the dependency-free fast path; otherwise recover normally.
    early_exit = ""
    if leg != "qualify":
        early_exit = rf"""if python3 - "$RUN_DIR/state/{leg}" {leg} <<'SIMONS_COMPLETED'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
try:
    marker = json.loads((root / "completed.json").read_text())
    identity_path = root / "identity.json"
    saved = json.loads(identity_path.read_text()) if identity_path.is_file() else marker.get("job_identity")
    if saved is None and sys.argv[2] == "export" and "recipe_name" in marker:
        saved = marker
    if saved is None:
        if {recipe!r} != "full":
            raise ValueError("missing screen identity")
    else:
        if (saved.get("recipe_name", "full"), saved.get("screen_updates")) != ({recipe!r}, {screen_updates!r}):
            raise ValueError("recipe identity changed")
        if saved.get("qualification_only"):
            raise ValueError("qualification output")
        for key, value in (("size", {size!r}), ("seed", {seed!r}), ("upstream_pin", {pin!r})):
            if key in saved and saved[key] != value:
                raise ValueError("experiment identity changed")
    # Compare all available identity records, including the marker when a
    # manifest is present. A refusal falls through to the guest's full checks.
    condition = {condition_record!r}
    attempt = {provisional_unaudited_corpus_attempt!r}
    uri = {corpus_uri!r}
    provisional = None if attempt is None else dict(provisional=True,
        warning="UNAUDITED_CORPUS_PROVISIONAL", attempt_id=attempt, corpus_uri=uri.rstrip("/"))
    records = [r for r in (saved, marker.get("job_identity")) if r is not None]
    if "condition" in marker or "corpus_release" in marker:
        records.append(dict(condition=marker.get("condition"), corpus_release=marker.get("corpus_release")))
    records.extend(marker.get("member_job_identities", []))
    records.extend(dict(condition=marker.get("condition"), corpus_release=receipt)
                   for receipt in marker.get("member_corpus_releases", []))
    if sys.argv[2] == "export":
        for leg in ("stage4", "stage4-long"):
            for name in ("identity.json", "completed.json"):
                path = root.parent / leg / name
                if path.is_file():
                    member = json.loads(path.read_text())
                    if name == "identity.json":
                        records.append(member)
                    else:
                        if member.get("job_identity") is not None:
                            records.append(member["job_identity"])
                        if "condition" in member or "corpus_release" in member:
                            records.append(dict(condition=member.get("condition"),
                                                corpus_release=member.get("corpus_release")))
    if not records and (condition is not None or provisional is not None):
        raise ValueError("missing fork identity")
    receipts = []
    for record in records:
        if record.get("condition") != condition:
            raise ValueError("condition identity changed")
        receipt = record.get("corpus_release")
        if provisional is not None or (receipt or {{}}).get("provisional"):
            if receipt != provisional:
                raise ValueError("provisional receipt identity changed")
        receipts.append(receipt)
    if receipts and condition is not None and any(r != receipts[0] for r in receipts):
        raise ValueError("condition receipt records disagree")
    audited = [r for r in receipts if r is not None]
    if audited and any(r != audited[0] for r in audited):
        raise ValueError("corpus receipt records disagree")
    if condition is not None or provisional is not None:
        # A digest authenticates bytes, not agreement with the embedded identity.
        # Let the patched guest read checkpoints before accepting fork completion.
        raise ValueError("fork completion requires embedded identity verification")
    if sys.argv[2] == "export":
        names = ["config.json", "model.safetensors"]
        valid = all((root / n).stat().st_size == marker["artifacts"][n] > 0 for n in names)
    else:
        name = marker["checkpoint"]
        valid = name == "checkpoint-final.pt"
        valid = valid and (root / name).stat().st_size == marker["bytes"] > 0
        if valid:
            footer = b"\nSIMONS-SHA256:"
            size = len(footer) + 64
            with (root / name).open("rb") as handle:
                handle.seek(-size, 2)
                trailer = handle.read()
                remaining = handle.seek(0, 2) - size
                handle.seek(0)
                digest = hashlib.sha256()
                while remaining:
                    block = handle.read(min(1 << 20, remaining))
                    if not block:
                        raise ValueError("truncated checkpoint")
                    digest.update(block)
                    remaining -= len(block)
            valid = trailer == footer + digest.hexdigest().encode()
            valid = valid and marker["sha256"] == digest.hexdigest()
except (OSError, ValueError, KeyError, TypeError):
    valid = False
sys.exit(0 if valid else 1)
SIMONS_COMPLETED
then
    exit 0
fi
"""
    fetch = ""
    if needs_corpus:
        fetch = f"""mkdir -p "$RUN_DIR/corpus"
# Remove stale release receipts before syncing a possibly incomplete prefix.
rm -f "$RUN_DIR/corpus/attempt-status.json" "$RUN_DIR/corpus/audit.json"
gcloud storage rsync --recursive {shlex.quote(corpus_uri)} "$RUN_DIR/corpus"
"""
    return f'''#!/usr/bin/env bash
# Simons {size} {leg}; repository {repo_sha}; upstream {pin}
set -euo pipefail
: "${{RUN_DIR:?guest must export RUN_DIR}}"
: "${{STATE_URI:?guest must export STATE_URI}}"
export RUN_DIR
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HOME="${{HOME:-$RUN_DIR/home}}"
export PYTORCH_KERNEL_CACHE_PATH="$RUN_DIR/torch-kernel-cache"
mkdir -p "$HOME" "$PYTORCH_KERNEL_CACHE_PATH"
case "$RUN_DIR" in /*) ;; *) echo "RUN_DIR must be absolute" >&2; exit 2 ;; esac
mkdir -p "$RUN_DIR/state"
{early_exit}scratch=$(mktemp -d "$RUN_DIR/simons-install-XXXXXX")
python3 - "$scratch/bundle" <<'SIMONS_BUNDLE'
import base64, gzip, hashlib, json, pathlib, sys
packed = base64.b64decode("{payload}")
content = gzip.decompress(packed)
assert hashlib.sha256(content).hexdigest() == "{digest}"
root = pathlib.Path(sys.argv[1])
for name, content in json.loads(content).items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
SIMONS_BUNDLE
upstream="$scratch/Simons"
git clone --filter=blob:none --no-checkout https://github.com/Abel-ai-lab/Simons.git "$upstream"
git -C "$upstream" sparse-checkout set --no-cone /src/ /configs/ /pyproject.toml /README.md /LICENSE /NOTICE
git -C "$upstream" checkout --detach {shlex.quote(pin)}
python3 "$scratch/bundle/prepare.py" "$upstream" "$scratch/bundle"
{ENVIRONMENT}{fetch}exec "$scratch/venv/bin/python" -m simons.training.spot_job \\
    --root "$upstream" --run-dir "$RUN_DIR" --corpus "$RUN_DIR/corpus" --environment-file "$scratch/environment.txt" \\
    {shlex.join(arguments)}
'''


def main():
    """Write one executable job without contacting any cloud service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=["25m", "12m", "60m"])
    parser.add_argument("--recipe", choices=["full", "screen"], default="full")
    parser.add_argument("--screen-updates", type=int)
    parser.add_argument("--unsafe-qualification-allow-unaudited-corpus", action="store_true")
    parser.add_argument("--condition")
    parser.add_argument("--override", action="append")
    parser.add_argument("--provisional-unaudited-corpus-attempt")
    parser.add_argument("--leg", required=True, choices=LEGS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--checkpoint-minutes", type=float, default=10)
    parser.add_argument("--corpus-uri")
    parser.add_argument("--repo-sha", required=True)
    parser.add_argument("--upstream-pin", required=True)
    parser.add_argument("--qualify-leg", choices=LEGS[:5], default="stage1")
    parser.add_argument("--name")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--qualify-step", type=int)
    parser.add_argument("--max-micro-batches", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        script = generate(**args)
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)
    output.chmod(0o755)


if __name__ == "__main__":
    main()
