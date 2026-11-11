#!/bin/bash
set -u
set -eo pipefail
job=${1:?job directory} expected=${2:-0} timeout=${3:-43200}
[[ $expected =~ ^[0-9]+$ && $timeout =~ ^[1-9][0-9]*$ ]]
deadline=$((SECONDS + timeout))
while (( SECONDS < deadline )); do
    if [[ -s $job/exit ]]; then
        actual=$(cat "$job/exit")
        [[ $actual == "$expected" ]] || { echo "FAILED: $job exit=$actual expected=$expected" >&2; exit 1; }
        exit 0
    fi
    sleep 1
done
echo "UNKNOWN: $job has no exit status before timeout; next gate blocked" >&2
exit 124
