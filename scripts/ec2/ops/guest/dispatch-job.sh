#!/bin/bash
set -u
set -eo pipefail
label=${1:?label}
shift
[[ $label =~ ^[a-zA-Z0-9_-]+$ && $# -gt 0 ]]
log_root=${HOST_LOGS:-/home/ubuntu/fev/host-logs}
mkdir -p "$log_root"
job=$(mktemp -d "$log_root/$label.XXXXXXXX")
{
    printf '#!/bin/bash\nset -u\nset -eo pipefail\n'
    printf 'status_file=%q\n' "$job/exit"
    cat <<'BASH'
: "${status_file:?absolute status path set by dispatcher}"
publish_exit() {
    local rc=$?
    printf '%s\n' "$rc" > "$status_file.tmp"
    mv "$status_file.tmp" "$status_file"
    exit "$rc"
}
trap publish_exit EXIT
BASH
    printf '%q ' "$@"
    printf '\n'
} > "$job/command.sh"
setsid -f bash "$job/command.sh" < /dev/null > "$job/run.log" 2>&1
printf '%s\n' "$job"
