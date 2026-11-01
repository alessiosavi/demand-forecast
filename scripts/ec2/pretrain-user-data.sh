#!/bin/bash
# Cloud-init first boot: self-contained; the enabled unit covers every later boot.
set -euo pipefail
install -d /usr/local/sbin /etc/systemd/system
cat > /usr/local/sbin/arm-deadline.sh <<'ARM_DEADLINE'
#!/bin/bash
# The only guest shutdown caller. Ubuntu's GNU date and systemd are required.
set -euo pipefail
export TZ=UTC LC_ALL=C

fail() { printf '%s\n' "$*" >&2; exit 1; }

scheduled_usec() {
    local reply usec shown timestamp
    if reply=$(busctl get-property org.freedesktop.login1 /org/freedesktop/login1 \
        org.freedesktop.login1.Manager ScheduledShutdown 2>/dev/null); then
        if [[ $reply =~ ^\(st\)[[:space:]]+\"(poweroff|halt)\"[[:space:]]+([0-9]+)$ ]]; then
            usec=${BASH_REMATCH[2]}
            if (( usec > 0 )); then
                printf '%s\n' "$usec"
                return
            fi
        fi
    fi
    shown=$(shutdown --show) || return 1
    # systemd's --show includes a weekday, timestamp and timezone before the comma.
    timestamp=$(printf '%s\n' "$shown" | sed -n 's/^Shutdown scheduled for \([^,]*\),.*$/\1/p')
    [[ -n $timestamp ]] || return 1
    date -u -d "$timestamp" +%s%6N
}

utc_stamp() {
    local seconds fraction format
    seconds=$(($1 / 1000000))
    fraction=$(printf '%06d' "$(($1 % 1000000))")
    format=+%Y-%m-%dT%H:%M:%SZ
    [[ $fraction == 000000 ]] || format=+%Y-%m-%dT%H:%M:%S.%6NZ
    date -u -d "@$seconds.$fraction" "$format"
}

[[ $# -ge 1 ]] || fail 'usage: arm-deadline.sh boot|grace|set <UTC ISO-8601>'
exec 9>/run/lock/budget-deadline.lock
flock 9
case "$1" in
    boot|grace)
        [[ $# == 1 ]] || fail 'boot/grace takes no arguments'
        shutdown -c >&2
        shutdown -h +20 >&2
        scheduled=$(scheduled_usec) || fail 'cannot read back grace deadline'
        if [[ $1 == boot ]]; then
            touch /run/budget-deadline.booted
        fi
        printf 'DEADLINE_GRACE %s\n' "$(utc_stamp "$scheduled")"
        ;;
    set)
        [[ $# == 2 ]] || fail 'set requires a UTC ISO-8601 deadline'
        [[ $2 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(Z|\+00:00)$ ]] \
            || fail 'deadline must specify UTC (Z or +00:00)'
        requested=$(date -u -d "$2" +%s) || fail 'invalid deadline'
        now=$(date +%s)
        (( requested > now )) || fail 'deadline must be in the future'
        requested_utc=$(date -u -d "@$requested" +%Y-%m-%dT%H:%M:%SZ)
        if [[ ! -e /run/budget-deadline.booted ]]; then
            printf 'DEADLINE_NOT_READY %s\n' "$requested_utc"
            exit 3
        fi
        if [[ $(date -u -d "@$requested" +%F) == "$(date -u -d "@$now" +%F)" ]]; then
            when=$(date -u -d "@$requested" +%H:%M)
            # HH:MM in the current minute would schedule tomorrow. Preserve the guard.
            (( requested / 60 > now / 60 )) || fail 'deadline is inside the current minute'
        else
            when=+$(((requested - now) / 60))
        fi
        shutdown -c >&2
        shutdown -h "$when" >&2
        if ! scheduled=$(scheduled_usec); then
            printf 'DEADLINE_MISMATCH unknown %s\n' "$requested_utc"
            exit 1
        fi
        scheduled_utc=$(utc_stamp "$scheduled")
        if (( scheduled > requested * 1000000 || requested * 1000000 - scheduled > 60000000 )); then
            printf 'DEADLINE_MISMATCH %s %s\n' "$scheduled_utc" "$requested_utc"
            exit 1
        fi
        printf 'DEADLINE_SET %s\n' "$scheduled_utc"
        ;;
    *) fail 'usage: arm-deadline.sh boot|grace|set <UTC ISO-8601>' ;;
esac
ARM_DEADLINE
chmod 0755 /usr/local/sbin/arm-deadline.sh
cat > /etc/systemd/system/budget-deadline.service <<'BUDGET_DEADLINE'
[Unit]
Description=Pretraining twenty-minute boot budget grace
# Avoid the target's implicit After= ordering opposing our explicit After=.
DefaultDependencies=no
After=multi-user.target
Before=shutdown.target
Conflicts=shutdown.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/arm-deadline.sh boot
RemainAfterExit=yes
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=multi-user.target
BUDGET_DEADLINE
chmod 0644 /etc/systemd/system/budget-deadline.service
systemctl daemon-reload && systemctl enable --now budget-deadline.service
install -d -o ubuntu -g ubuntu /home/ubuntu/fev
install -d -o ubuntu -g ubuntu /home/ubuntu/fev/ops
