#!/bin/bash
# Run as ubuntu in a reserved EC2 session using dispatch-job.sh (docs/ec2.md §2).
set -euo pipefail
: "${FEV_SHA:?full merged SHA}" "${INSTALL_CONSTRAINTS:?absolute filtered Colab lock}"
: "${HF_HOME:?absolute writable HF cache}"
export COLAB_LOCK=${COLAB_LOCK:-/home/ubuntu/fev-inputs/environment-lock.txt}
export COLAB_MANIFEST=${COLAB_MANIFEST:-/home/ubuntu/fev-inputs/colab-dataset-manifest.json}
export FEV_DATASET_MANIFEST=/home/ubuntu/fev/dataset-manifest.json
export FEV_LOCK_FILE=/home/ubuntu/fev/ec2-environment-lock.txt
export HF_HOME HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
[[ $FEV_SHA =~ ^[0-9a-f]{40}$ && $INSTALL_CONSTRAINTS == /* && $HF_HOME == /* ]]
[[ -f $COLAB_LOCK && -f $COLAB_MANIFEST && -f $INSTALL_CONSTRAINTS ]]
[[ ${HF_HUB_OFFLINE:-0} != 1 && ${HF_DATASETS_OFFLINE:-0} != 1 ]]
[[ -z ${FEV_DATASETS_PREFIX:-} && -z ${PYTHONPATH:-} ]]
unset LD_LIBRARY_PATH
sudo apt-get update
sudo apt-get install -y python3.12-venv git time jq rsync util-linux
mkdir -p /home/ubuntu/fev
cd /home/ubuntu/fev
if [[ ! -d code ]]; then
    git clone https://github.com/alessiosavi/demand-forecast code
    git -C code checkout --detach "$FEV_SHA"
fi
cd code
test "$(git rev-parse HEAD)" = "$FEV_SHA"
test -z "$(git status --porcelain)"
python3.12 - <<'PY'
import os
from pathlib import Path
source = Path(os.environ['COLAB_LOCK']).read_bytes()
expected = b''.join(line for line in source.splitlines(keepends=True) if b' @ ' not in line)
assert Path(os.environ['INSTALL_CONSTRAINTS']).read_bytes() == expected, 'constraints differ'
PY
python3.12 -m venv /home/ubuntu/fev/venv-pr
VENV=/home/ubuntu/fev/venv-pr
"$VENV/bin/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -c "$INSTALL_CONSTRAINTS" -e '.[bench]' huggingface_hub
mkdir -p /home/ubuntu/fev/ops "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
EVIDENCE=$(mktemp -d /home/ubuntu/fev/ops/setup.XXXXXXXX)
"$VENV/bin/python" -I -m pip freeze > "$EVIDENCE/pip-freeze.txt"
"$VENV/bin/python" -I scripts/ec2/closure_parity.py \
    --colab-lock "$COLAB_LOCK" --freeze "$EVIDENCE/pip-freeze.txt" --extras bench \
    --out "$EVIDENCE/closure-parity.txt"
"$VENV/bin/python" scripts/pip_check_closure.py --extras bench --report "$EVIDENCE/pip-check.txt"
sed '/^#/d; /^-e /d; /^$/d' "$EVIDENCE/pip-freeze.txt" | sort > "$EVIDENCE/actual-pins.txt"
if [[ ! -e $FEV_LOCK_FILE ]]; then
    cp "$EVIDENCE/actual-pins.txt" "$FEV_LOCK_FILE"
fi
sed '/^#/d; /^-e /d; /^$/d' "$FEV_LOCK_FILE" | sort > "$EVIDENCE/expected-pins.txt"
diff -u "$EVIDENCE/expected-pins.txt" "$EVIDENCE/actual-pins.txt" > "$EVIDENCE/lock.diff"
cat > /home/ubuntu/fev/ops/pin-and-warm.py <<'PY'
"""Same pin/byte/fingerprint gates as docs/ec2.md, extended to heldout tasks."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from demand_forecast.bench.specs import dataset_fingerprint, load_task_specs

repo = "autogluon/fev_datasets"
revision = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"
manifest_path = Path(os.environ["FEV_DATASET_MANIFEST"])
previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
colab = json.loads(Path(os.environ["COLAB_MANIFEST"]).read_text())
assert colab["repo"] == repo and colab["revision"] == revision
api = HfApi()


def verify_head():
    info = api.dataset_info(repo, files_metadata=True)
    main = next(b.target_commit for b in api.list_repo_refs(repo, repo_type="dataset").branches
                if b.name == "main")
    assert info.sha == main == revision, (info.sha, main, revision)
    return info


info = verify_head()
if previous is not None:
    assert previous["repo"] == repo and previous["revision"] == revision
weekly, heldout = load_task_specs("weekly"), load_task_specs("heldout")
assert len(weekly) == 7 and len(heldout) == 5
entries = []
for spec in weekly + heldout:
    if spec in weekly:
        colab_files = colab["tasks"][spec.name]["files"]
        assert len(colab_files) == 1, (spec.name, list(colab_files))
        ((rfilename, colab_file),) = colab_files.items()
        candidates = [s for s in info.siblings if s.rfilename == rfilename]
    else:
        # Held-out tasks were not in the seven-task Colab manifest. Resolve their
        # train parquet through the pinned Hub card, never a guessed path prefix.
        configs = info.card_data.to_dict()["configs"]
        matches = [c for c in configs if c["config_name"] == spec.dataset_config]
        assert len(matches) == 1, (spec.name, "ambiguous Hub config")
        splits = [f for f in matches[0]["data_files"] if f["split"] == "train"]
        assert len(splits) == 1, (spec.name, "ambiguous train split")
        patterns = splits[0]["path"]
        if isinstance(patterns, str):
            patterns = [patterns]
        candidates = [s for s in info.siblings if s.rfilename.endswith(".parquet")
                      and any(fnmatch.fnmatchcase(s.rfilename, p) for p in patterns)]
    assert len(candidates) == 1, (spec.name, [s.rfilename for s in candidates])
    sibling = candidates[0]
    assert sibling.rfilename.endswith(".parquet") and sibling.lfs is not None
    if spec in weekly:
        assert sibling.lfs.sha256 == colab_file["sha256"], (spec.name, "Colab sha256 differs")
        assert sibling.lfs.size == colab_file["bytes"], (spec.name, "Colab size differs")
    task = spec.to_task()
    fingerprint = dataset_fingerprint(task)  # sequential warm-up, never training
    path = Path(hf_hub_download(repo, sibling.rfilename, repo_type="dataset", revision=revision))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == sibling.lfs.sha256
    assert path.stat().st_size == sibling.lfs.size
    entry = dict(task=spec.name, filename=sibling.rfilename, path=str(path), sha256=digest,
                 size=path.stat().st_size, fingerprint=fingerprint)
    if previous is not None:
        old = next(x for x in previous["files"] if x["task"] == spec.name)
        assert all(entry[k] == old[k] for k in ("filename", "sha256", "size", "fingerprint"))
    entries.append(entry)
    print(json.dumps(entry), flush=True)
verify_head()
manifest_path.write_text(json.dumps(dict(repo=repo, revision=revision, files=entries), indent=2) + "\n")
PY
"$VENV/bin/python" /home/ubuntu/fev/ops/pin-and-warm.py > "$EVIDENCE/pins.log" 2>&1
cp "$FEV_DATASET_MANIFEST" "$EVIDENCE/dataset-manifest.json"
nvidia-smi -q > "$EVIDENCE/nvidia-smi.txt"
free -b > "$EVIDENCE/ram.txt"
df -B1 > "$EVIDENCE/disk.txt"
printf 'SETUP_VERIFIED %s\n' "$EVIDENCE"
