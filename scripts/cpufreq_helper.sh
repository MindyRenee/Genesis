#!/bin/bash
# cpufreq_helper.sh — privileged helper for Genesis's neurochemistry
# to control its own CPU frequency.
#
# Usage:
#   cpufreq_helper.sh set_governor <governor>
#   cpufreq_helper.sh set_min_freq <freq_khz>
#   cpufreq_helper.sh set_max_freq <freq_khz>
#   cpufreq_helper.sh set_freq <freq_khz>   (requires userspace governor)
#   cpufreq_helper.sh set_epp <profile>     (requires EPP support)
#   cpufreq_helper.sh set_boost <0|1>       (turbo gate; 1 = allow boost)
#   cpufreq_helper.sh test <anything>       (no-op probe for sudo detection)
#
# This script is invoked via sudo (passwordless, restricted to this
# exact script). It validates all inputs to prevent injection.

set -euo pipefail

# ─── Input validation ──────────────────────────────────────────

VALID_GOVERNORS="conservative ondemand userspace powersave performance schedutil"

# Valid EPP profiles (the actual available profiles are read from
# sysfs at runtime, but we validate against this superset to prevent
# arbitrary string injection).
VALID_EPP="performance balance_performance default balance_power power"

validate_governor() {
    local g="$1"
    for valid in $VALID_GOVERNORS; do
        if [ "$g" = "$valid" ]; then
            return 0
        fi
    done
    echo "error: invalid governor '$g'" >&2
    exit 1
}

validate_epp() {
    local e="$1"
    for valid in $VALID_EPP; do
        if [ "$e" = "$valid" ]; then
            return 0
        fi
    done
    echo "error: invalid EPP profile '$e'" >&2
    exit 1
}

validate_freq() {
    local f="$1"
    if ! [[ "$f" =~ ^[0-9]+$ ]]; then
        echo "error: invalid frequency '$f' (must be numeric kHz)" >&2
        exit 1
    fi
    if [ "$f" -lt 100000 ] || [ "$f" -gt 10000000 ]; then
        echo "error: frequency '$f' out of range (100000-10000000 kHz)" >&2
        exit 1
    fi
}

# ─── Main ──────────────────────────────────────────────────────

if [ $# -ne 2 ]; then
    echo "usage: $0 <command> <value>" >&2
    echo "commands: set_governor, set_min_freq, set_max_freq, set_freq, set_epp, set_boost" >&2
    exit 1
fi

CMD="$1"
VAL="$2"

case "$CMD" in
    test)
        # No-op probe used by sudo_available() to check whether the
        # sudoers rule permits running this helper. The sudoers rule
        # only allows this specific script, so probing `sudo -n true`
        # would fail even when sudo is correctly configured.
        echo "ok"
        exit 0
        ;;
    set_governor)
        validate_governor "$VAL"
        for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
            [ -f "$cpu" ] && echo "$VAL" > "$cpu"
        done
        ;;
    set_min_freq)
        validate_freq "$VAL"
        # Defense-in-depth: the kernel rejects min > max with
        # -EINVAL. The Rust side (derive_policy + apply_policy)
        # already ensures min <= max, but we check here too in case
        # a future caller bypasses apply_policy or the sysfs state
        # changes between the two writes. If the requested min is
        # above the current max, clamp it down to the current max
        # rather than failing — a too-low floor is harmless, a
        # failed write leaves the old (possibly wrong) value.
        for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_min_freq; do
            if [ -f "$cpu" ]; then
                max_file="${cpu%scaling_min_freq}scaling_max_freq"
                cur_max=0
                if [ -f "$max_file" ]; then
                    cur_max=$(cat "$max_file" 2>/dev/null || echo 0)
                fi
                if [ "$cur_max" -gt 0 ] && [ "$VAL" -gt "$cur_max" ]; then
                    echo "$cur_max" > "$cpu"
                else
                    echo "$VAL" > "$cpu"
                fi
            fi
        done
        ;;
    set_max_freq)
        validate_freq "$VAL"
        # Defense-in-depth: if the requested max is below the
        # current min, clamp it up to the current min rather than
        # failing. See set_min_freq comment for rationale.
        for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq; do
            if [ -f "$cpu" ]; then
                min_file="${cpu%scaling_max_freq}scaling_min_freq"
                cur_min=0
                if [ -f "$min_file" ]; then
                    cur_min=$(cat "$min_file" 2>/dev/null || echo 0)
                fi
                if [ "$cur_min" -gt 0 ] && [ "$VAL" -lt "$cur_min" ]; then
                    echo "$cur_min" > "$cpu"
                else
                    echo "$VAL" > "$cpu"
                fi
            fi
        done
        ;;
    set_freq)
        validate_freq "$VAL"
        for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_setspeed; do
            [ -f "$cpu" ] && echo "$VAL" > "$cpu"
        done
        ;;
    set_epp)
        validate_epp "$VAL"
        for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference; do
            [ -f "$cpu" ] && echo "$VAL" > "$cpu"
        done
        ;;
    set_boost)
        if [ "$VAL" != "0" ] && [ "$VAL" != "1" ]; then
            echo "error: invalid boost value '$VAL' (must be 0 or 1)" >&2
            exit 1
        fi
        # Turbo gate — the hardware permission to exceed the base
        # P-state. The generic per-cpu attribute, AMD's cpb alias, and
        # the global boost attribute all express the same permission.
        # They are aliases: a platform may expose only some of them, and
        # a redundant write to an alias that already matches can be
        # rejected with EINVAL — that is not a failure. Write every
        # attribute that exists and require at least one to succeed.
        wrote=0
        found=0
        for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/boost \
                 /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpb \
                 /sys/devices/system/cpu/cpufreq/boost; do
            [ -f "$f" ] || continue
            found=1
            if echo "$VAL" > "$f" 2>/dev/null; then
                wrote=1
            fi
        done
        if [ "$found" -eq 0 ]; then
            # No boost attribute on this hardware — nothing to gate.
            # Skip (success) so unsupported platforms don't fail the
            # whole body-control application over one N/A knob.
            echo "ok (no boost attribute on this hardware — skipped)"
            exit 0
        fi
        if [ "$wrote" -ne 1 ]; then
            echo "error: no boost attribute accepted the write" >&2
            exit 1
        fi
        ;;
    *)
        echo "error: unknown command '$CMD'" >&2
        exit 1
        ;;
esac

echo "ok"
