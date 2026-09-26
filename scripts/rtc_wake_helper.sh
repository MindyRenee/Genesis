#!/bin/bash
# rtc_wake_helper.sh — privileged helper for Genesis to arm the RTC
# wake alarm: the machine scheduling her own return from suspension.
#
# Usage:
#   rtc_wake_helper.sh set <epoch_seconds>
#   rtc_wake_helper.sh clear
#   rtc_wake_helper.sh test            (sudo probe — exits 0, no-op)
#
# This script is invoked via sudo (passwordless, restricted to this
# exact script by /etc/sudoers.d/genesis). All inputs are validated.

set -euo pipefail

# Find the RTC that actually exposes a wakealarm (usually rtc0).
find_wakealarm() {
    local i
    for i in 0 1 2 3 4 5 6 7; do
        local path="/sys/class/rtc/rtc${i}/wakealarm"
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    echo "error: no RTC wakealarm found" >&2
    exit 1
}

validate_epoch() {
    local e="$1"
    # Digits only, bounded length — anything else is rejected before
    # it reaches sysfs.
    if ! [[ "$e" =~ ^[0-9]{1,19}$ ]]; then
        echo "error: invalid epoch '$e'" >&2
        exit 1
    fi
    # Reject timestamps in the past — an alarm that can never fire
    # is a silent failure, which is worse than an explicit error.
    local now
    now="$(date -u +%s)"
    if [ "$e" -le "$now" ]; then
        echo "error: epoch $e is in the past (now $now)" >&2
        exit 1
    fi
}

cmd="${1:-}"
value="${2:-}"

case "$cmd" in
    test)
        # sudo-availability probe; deliberately a no-op.
        exit 0
        ;;
    set)
        validate_epoch "$value"
        wakealarm="$(find_wakealarm)"
        # The kernel requires disarming before arming a new value on
        # some rtc drivers — write 0 first.
        echo 0 > "$wakealarm"
        echo "$value" > "$wakealarm"
        # Read back so callers can verify the alarm landed.
        cat "$wakealarm"
        ;;
    clear)
        wakealarm="$(find_wakealarm)"
        echo 0 > "$wakealarm"
        cat "$wakealarm"
        ;;
    *)
        echo "usage: $0 {set <epoch_seconds>|clear|test}" >&2
        exit 1
        ;;
esac
