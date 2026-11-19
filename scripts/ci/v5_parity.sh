#!/usr/bin/env bash
set -euo pipefail

run_tests=false
case "${1:-}" in
    "") ;;
    --test) run_tests=true ;;
    *) printf 'Usage: %s [--test]\n' "$0" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    printf 'Usage: %s [--test]\n' "$0" >&2
    exit 2
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
work="$(mktemp -d "${V5_PARITY_TMPDIR:-${TMPDIR:-/tmp}}/v5-parity.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT
export TMPDIR="$work"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

extract_pin() {
    awk -F '==' -v name="$1" '
        $1 == name && NF == 2 { print; count++ }
        END { if (count != 1) exit 1 }
    ' "$repo/$2" || {
        printf 'Expected one %s pin in %s\n' "$1" "$2" >&2
        return 1
    }
}

make_env() {
    local name="$1" requirements="$2"
    local python="$work/$name/bin/python" pins="$work/$name-pins.txt"
    local package project="$repo" torch_pin
    {
        for package in numpy pandas scipy pyarrow safetensors datasets; do
            extract_pin "$package" "$requirements"
        done
        # Both environments install the same fev, so the stored task reloads; the windows
        # themselves are stored by the first environment and read back by the second.
        extract_pin fev requirements-yardsticks.txt
    } > "$pins"

    printf '\n%s pins from %s\n' "$name" "$requirements"
    cat "$pins"
    uv venv --python 3.12 "$work/$name"
    case "$(uname -s)" in
        Linux)
            torch_pin='torch==2.8.0+cpu'
            uv pip install --python "$python" --index https://download.pytorch.org/whl/cpu "$torch_pin"
            ;;
        Darwin)
            # macOS uses the PyPI wheel for CPU inference.
            torch_pin='torch==2.8.0'
            uv pip install --python "$python" "$torch_pin"
            ;;
        *)
            printf 'Unsupported platform: %s\n' "$(uname -s)" >&2
            return 1
            ;;
    esac
    if $run_tests && [ "$name" = training ]; then
        project="${repo}[dev]"
    fi
    # Resolve the project and its other pins from PyPI with torch already installed.
    uv pip install --python "$python" -r "$pins" -e "$project"
    printf '%s\n' "$torch_pin" >> "$pins"
    printf '\n%s installed versions\n' "$name"
    "$python" - "$pins" <<'PY' | tee "$work/$name-installed.txt"
from importlib.metadata import version
from pathlib import Path
import sys

mismatches = []
for pin in Path(sys.argv[1]).read_text().splitlines():
    name, expected = pin.split("==")
    installed = version(name)
    print(f"{name}=={installed}", flush=True)
    if installed != expected:
        mismatches.append(f"{pin} required, installed {name}=={installed}")
if mismatches:
    raise SystemExit("\n".join(mismatches))
PY
    "$python" -c 'import torch; assert torch.__version__.split("+")[0] == "2.8.0" and torch.version.cuda is None; print("torch", torch.__version__, "(CPU)")'
}

require_distinct_versions() {
    local package
    for package in numpy pandas datasets; do
        if [ "$(grep "^$package==" "$1")" = "$(grep "^$package==" "$2")" ]; then
            printf 'Training and scoring have the same %s version, so the gate would test one environment\n' "$package" >&2
            return 1
        fi
    done
}

make_env training requirements-v5.txt
make_env scoring requirements-yardsticks.txt
require_distinct_versions "$work/training-installed.txt" "$work/scoring-installed.txt"
"$work/training/bin/python" -m demand_forecast.models.v5.export selftest "$work/export"
"$work/scoring/bin/python" -m demand_forecast.models.v5.export check "$work/export"
"$work/training/bin/python" -m demand_forecast.bench.yardsticks.v5_adapter \
    selftest "$work/export" "$work/adapter"
"$work/scoring/bin/python" -m demand_forecast.bench.yardsticks.v5_adapter \
    check "$work/export" "$work/adapter"

if $run_tests; then
    cd "$repo"
    "$work/training/bin/python" -m pytest -q --no-cov \
        tests/test_models tests/test_data/test_episode_*.py \
        tests/test_pretrain_v5/test_t1.py tests/test_pretrain_v5/test_sampler.py \
        -k 'not worker and not loader_resume and not multiprocess and not file_limit'
fi
