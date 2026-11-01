#!/bin/bash
# Operator laptop / Linux. Bash 3.2 compatible. Use a supervised terminal.
set -euo pipefail

fail() { echo "Launch refused: $*" >&2; exit 2; }
[[ $# -gt 0 ]] || fail "usage: $0 provision|start|stop|terminate [--ledger PATH] [options]"
operation=$1
shift
case "$operation" in provision|start|stop|terminate) ;; *) fail "unknown command $operation";; esac
for variable in OPERATOR_IP VPC_ID SUBNET_ID AMI_ID; do
    [[ -n ${!variable:-} ]] || fail "missing environment variable $variable"
done
export AWS_DEFAULT_OUTPUT=json
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python=${PYTHON:-python3}
ledger=ops/cost-ledger.json
arguments=()
while (($#)); do
    case "$1" in
        --ledger)
            (($# >= 2)) || fail "missing value for --ledger"
            ledger=$2
            shift 2;;
        *) arguments+=("$1"); shift;;
    esac
done

# Guard owns key pair / security group setup, run-instances and all accounting.
# Installing the trap before launch also covers loss after the AWS reply but
# before its id is persisted. launch-trap discovers the id, stop-instances
# --force, waits for stopped, then cancels the transaction with evidence=trap.
# A failed stop retains the reservation. A verified stop schedule ends this gap.
child=''
recover_launch() {
    local code=$?
    trap - EXIT ERR INT TERM
    if [[ -n $child ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    if [[ $operation == provision && -f $ledger ]]; then
        "$python" "$tools_dir/cost_guard.py" --ledger "$ledger" launch-trap || true
    fi
    exit "$code"
}
trap recover_launch EXIT ERR
trap 'exit 130' INT
trap 'exit 143' TERM
# ${array[@]+...} also handles an empty array with Bash 3.2's nounset.
"$python" "$tools_dir/cost_guard.py" --ledger "$ledger" "$operation" ${arguments[@]+"${arguments[@]}"} &
child=$!
wait "$child"
child=''
trap - EXIT ERR INT TERM
