#!/bin/bash
# Linux/EC2 only. Invoke with an absolute venv; it supplies the executing CLI.
set -euo pipefail

usage() {
    echo "Usage: $0 --venv V --run-root ROOT --config C --tasks t1,t2 [--stop-after N]"
    echo "Required environment: FEV_LOCK_FILE, FEV_DATASET_MANIFEST (verified before launch)."
}
fail() { echo "Launch refused: $*" >&2; exit 2; }
venv='' root='' config='' tasks='' stop_after=''
while (($#)); do
    case "$1" in
        --venv|--run-root|--config|--tasks|--stop-after)
            (($# >= 2)) || fail "missing value for $1"
            case "$1" in
                --venv) venv=$2;;
                --run-root) root=$2;;
                --config) config=$2;;
                --tasks) tasks=$2;;
                --stop-after) stop_after=$2;;
            esac
            shift 2;;
        --help|-h) usage; exit 0;;
        *) fail "unknown argument $1";;
    esac
done
[[ -n $venv && -n $root && -n $config && -n $tasks ]] || { usage; exit 2; }
[[ -z $stop_after || $stop_after =~ ^[1-9][0-9]*$ ]] || fail "invalid --stop-after"
[[ $(uname -s) == Linux ]] || fail "this launcher requires Linux/EC2"
: "${FEV_LOCK_FILE:?set FEV_LOCK_FILE to the Colab constraints lock}"
: "${FEV_DATASET_MANIFEST:?set FEV_DATASET_MANIFEST to the verified dataset manifest}"
for command in setsid nohup flock nvidia-smi realpath; do command -v "$command" >/dev/null; done
[[ -x /usr/bin/time && -x $venv/bin/python && -x $venv/bin/demand-forecast ]] || fail "missing time or venv executables"
venv=$(realpath "$venv")
config=$(realpath "$config")
lock=$(realpath "$FEV_LOCK_FILE")
manifest=$(realpath "$FEV_DATASET_MANIFEST")
[[ -f $config && -f $lock && -f $manifest ]] || fail "missing config, lock or manifest"
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# -I prevents this checkout or PYTHONPATH from masking the fallback installation.
package_root=$(env -u LD_LIBRARY_PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$venv/bin/python" -I -c \
    'import demand_forecast; from pathlib import Path; print(Path(demand_forecast.__file__).resolve().parents[1])')
sha=$(git -C "$package_root" rev-parse HEAD)
git -C "$package_root" diff --quiet HEAD -- demand_forecast configs || fail "executing package/config has uncommitted changes"
mkdir -p "$root"
root=$(realpath "$root")
exec 9>"$root/.launch.lock"
flock -n 9 || fail "another launcher is preparing this root"
launch=$(date -u +%Y%m%dT%H%M%S)-$$
provenance="$root/provenance/$launch"
mkdir -p "$provenance"

# Finish and persist provenance before launching even the first baseline.
export FEV_EXECUTING_ROOT="$package_root" FEV_EXECUTING_SHA="$sha"
env -u LD_LIBRARY_PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$venv/bin/python" -I - "$provenance" "$config" "$tasks" <<'PY'
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
import demand_forecast
from demand_forecast.bench.specs import load_task_specs
from demand_forecast.config import Settings

out, config, selector = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
specs = load_task_specs(selector)
names = [s.name for s in specs]
if not names or len(set(names)) != len(names):
    raise SystemExit("empty or duplicate task selection")
if any(not n or n in (".", "..", "merged", "provenance") or Path(n).name != n for n in names):
    raise SystemExit("unsafe task name")
settings = Settings.from_yaml(config)
quick = settings.training.max_steps_cap is not None or settings.training.val_items_max is not None
(out / "model-name.txt").write_text("demand-forecast-v2-quick" if quick else "")
(out / "tasks.txt").write_text("\n".join(names) + "\n")
(out / "task-definitions.json").write_text(json.dumps([asdict(s) for s in specs], indent=2, default=str))
(out / "execution.json").write_text(json.dumps({
    "sha": os.environ["FEV_EXECUTING_SHA"], "interpreter": sys.executable,
    "package": demand_forecast.__file__, "python": platform.python_version(),
    "config": str(config), "threads": {"OMP_NUM_THREADS": 2, "MKL_NUM_THREADS": 2},
    "cache": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")},
}, indent=2))
source = Path(os.environ["FEV_EXECUTING_ROOT"])
for path in ("docs/protocol.md", "demand_forecast/bench/tasks/fev_bench_weekly.yaml",
             "demand_forecast/bench/tasks/profiles.yaml"):
    shutil.copy2(source / path, out / Path(path).name)
PY
cp "$config" "$provenance/config.yaml"
cp "$lock" "$provenance/environment-lock.txt"
cp "$manifest" "$provenance/dataset-manifest.json"
cp "$tools_dir/frozen_parallel.sh" "$tools_dir/sample_resources.py" "$provenance/"
printf '%s\n' "$sha" > "$provenance/commit.txt"
git -C "$package_root" status --porcelain > "$provenance/git-status.txt"
env -u LD_LIBRARY_PATH "$venv/bin/python" -I -m pip freeze > "$provenance/pip-freeze.txt"
# Editable paths/comments differ between the PR and fallback; package pins must not.
sed '/^#/d; /^-e /d; /^$/d' "$lock" | sort > "$provenance/expected-pins.txt"
sed '/^#/d; /^-e /d; /^$/d' "$provenance/pip-freeze.txt" | sort > "$provenance/actual-pins.txt"
diff -u "$provenance/expected-pins.txt" "$provenance/actual-pins.txt" > "$provenance/lock.diff" || fail "environment drift (see $provenance/lock.diff)"
nvidia-smi -q > "$provenance/nvidia-smi.txt"
mapfile -t task_names < "$provenance/tasks.txt"
model_name=$(cat "$provenance/model-name.txt")
task_locks=()

# Prepare all scripts and acquire all requested task locks before starting work.
for task in "${task_names[@]}"; do
    task_dir="$root/$task"
    mkdir -p "$task_dir/launches/$launch"
    exec {task_lock}>"$task_dir/active.lock"
    flock -n "$task_lock" || fail "$task already has an active process"
    task_locks+=("$task_lock")
    session_dir="$task_dir/launches/$launch"
    baseline=("$venv/bin/demand-forecast" fev baseline --tasks "$task" --out-dir "$task_dir")
    run=("$venv/bin/demand-forecast" fev run --tasks "$task" --config "$provenance/config.yaml"
         --policy first_window --out-dir "$task_dir" --resume)
    [[ -z $stop_after ]] || run+=(--stop-after "$stop_after")
    [[ -z $model_name ]] || run+=(--model-name "$model_name")
    {
        echo '#!/bin/bash'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$package_root"
        printf 'status_file=%q\n' "$session_dir/exit"
        cat <<'TRAP'
trap 'rc=$?; printf "%s\n" "$rc" > "$status_file.tmp"; mv "$status_file.tmp" "$status_file"' EXIT
TRAP
        printf '%q ' "${baseline[@]}"; printf '\n'
        printf '%q ' "${run[@]}"; printf '\n'
    } > "$session_dir/task.sh"
    cp "$session_dir/task.sh" "$provenance/$task.sh"
    command=(setsid nohup env -u LD_LIBRARY_PATH -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
             /usr/bin/time -v bash "$session_dir/task.sh")
    printf '%q ' "${command[@]}" >> "$provenance/commands.sh"
    printf '> %q 2>&1 < /dev/null &\n' "$session_dir/run.log" >> "$provenance/commands.sh"
done

for index in "${!task_names[@]}"; do
    task=${task_names[index]}
    task_dir="$root/$task"
    session_dir="$task_dir/launches/$launch"
    # Retain old logs/exits and publish aliases for the most recent session.
    for name in run.log exit; do
        [[ ! -e $task_dir/$name || -L $task_dir/$name ]] || fail "refusing to replace regular $task_dir/$name"
        ln -sfn "launches/$launch/$name" "$task_dir/$name"
    done
    (
        exec 9>&-
        for descriptor in "${task_locks[@]}"; do
            if [[ $descriptor != "${task_locks[index]}" ]]; then exec {descriptor}>&-; fi
        done
        exec setsid nohup env -u LD_LIBRARY_PATH -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
            /usr/bin/time -v bash "$session_dir/task.sh"
    ) > "$session_dir/run.log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$session_dir/pid"
    # Pollers own no task locks. Their failure is recorded separately from the CLI exit.
    (
        exec 9>&-
        for descriptor in "${task_locks[@]}"; do exec {descriptor}>&-; done
        exec setsid nohup env -u LD_LIBRARY_PATH "$venv/bin/python" -I "$provenance/sample_resources.py" \
            --session "$pid" --directory "$session_dir" --exit-file "$session_dir/exit"
    ) > "$session_dir/sampler.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$session_dir/sampler.pid"
    echo "$task pid=$pid log=$task_dir/run.log exit=$task_dir/exit sampled_maxima=$session_dir/resources.csv"
done
# The workers retain only their own lock; shell exit closes the parent's copies.
echo "Provenance: $provenance (executing $sha)"
