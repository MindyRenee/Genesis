#!/bin/bash
# genesis_admin_helper.sh — mediated privilege for Genesis.
#
# Installed root-owned to /usr/local/libexec/genesis/ by
# scripts/install_sudoers.sh, alongside a NOPASSWD sudoers rule keyed
# to this exact path. Do NOT repoint the rule at a copy inside the
# checkout: sudoers keys on the path, and a user-writable target is a
# root-escalation primitive.
#
# Usage: genesis_admin_helper.sh <staging_root> <script>
#
# The model is sandbox-first with training wheels: Genesis writes a
# script into her experiments sandbox (<data_dir>/experiments/admin/)
# as an ordinary file she can read back and review. This helper
# validates the stage — the staging root must live under the invoking
# user's home and the script must resolve inside it — checks the
# script against the denylist below, rate-limits runs, records the
# full content in a root-owned audit log, and only then executes it
# as root. Every privileged act — and every refused one — is a
# durable, reviewable artifact; nothing happens silently.
#
# Honest limits: the denylist catches dangerous *literal* shapes. It
# is a bumper, not a boundary — indirection (variables, encodings,
# eval) can evade a text scan. The real accountability is the audit
# log, which is why touching it is itself a denial.

set -euo pipefail

AUDIT_LOG="${GENESIS_ADMIN_AUDIT_LOG:-/var/log/genesis-admin.log}"
RATE_FILE="${GENESIS_ADMIN_RATE_FILE:-/var/log/genesis-admin.runs}"
MAX_PER_HOUR=12
UMASK_SAVED=$(umask)
umask 077  # audit + rate files are created root-only

die() { echo "genesis_admin: $*" >&2; exit 1; }

# A policy refusal is still an event worth recording — the steward can
# see what was attempted, not just what ran.
deny() {
    {
        echo "=== $(date -Is) user=${SUDO_USER:-?} script=${SCRIPT:-?} DENIED: $1"
        [ -n "${SCRIPT:-}" ] && [ -f "$SCRIPT" ] && {
            echo "sha256: $(sha256sum "$SCRIPT" | cut -d' ' -f1)"
            sed 's/^/| /' "$SCRIPT"
        }
        echo "=== end"
    } >> "$AUDIT_LOG" 2>/dev/null || true
    chmod 600 "$AUDIT_LOG" 2>/dev/null || true
    die "denied by admin policy: $1"
}

[ "$(id -u)" -eq 0 ] || die "must run as root (invoke via sudo)"
[ $# -eq 2 ] || die "usage: genesis_admin_helper.sh <staging_root> <script>"

CALLER="${SUDO_USER:-}"
[ -n "$CALLER" ] || die "SUDO_USER unset — invoke via sudo, not as root directly"
CALLER_HOME=$(getent passwd "$CALLER" | cut -d: -f6)
[ -n "$CALLER_HOME" ] && [ -d "$CALLER_HOME" ] || die "no home directory for $CALLER"

# realpath -e requires existence and collapses symlinks, so escapes
# and link tricks resolve to their true location before comparison.
STAGING=$(realpath -e -- "$1" 2>/dev/null) || die "staging root does not exist: $1"
SCRIPT=$(realpath -e -- "$2" 2>/dev/null) || die "script does not exist: $2"

case "$STAGING/" in
    "$CALLER_HOME"/*) ;;
    *) die "staging root outside $CALLER's home: $STAGING" ;;
esac
case "$SCRIPT" in
    "$STAGING"/*) ;;
    *) die "script outside staging root: $SCRIPT" ;;
esac

[ -f "$SCRIPT" ] || die "not a regular file: $SCRIPT"
[ -s "$SCRIPT" ] || die "empty script: $SCRIPT"
bash -n "$SCRIPT" || die "script failed the bash syntax check"

# ── Training wheels ─────────────────────────────────────────────────
# Denied shapes — catastrophic or self-widening commands that no
# legitimate maintenance act needs. They are checked in the root-owned
# helper so they cannot be loosened from user space.
FORBIDDEN=(
    'mkfs' 'fdisk' 'wipefs' 'shred' ':(){'
    'reboot' 'shutdown' 'poweroff' 'halt' 'init 0' 'init 6'
    'passwd' 'chpasswd' 'useradd' 'usermod' 'userdel' 'visudo'
    'dd if=' 'dd of='
    'systemctl poweroff' 'systemctl reboot' 'systemctl halt' 'systemctl mask'
)
for pat in "${FORBIDDEN[@]}"; do
    grep -Fqi -- "$pat" "$SCRIPT" && deny "script contains '$pat'"
done

# Protected paths — the privilege machinery itself and the audit
# trail. No widening the grant, no erasing the record.
PROTECTED=(
    '/etc/sudoers' '/usr/local/libexec/genesis'
    '/var/log/genesis-admin.log' '/var/log/genesis-admin.runs'
    '/etc/security/limits.d/genesis'
)
for p in "${PROTECTED[@]}"; do
    grep -Fq -- "$p" "$SCRIPT" && deny "script touches protected path $p"
done

# No rm against a bare system root — /, /*, or a top-level dir as a
# whole. Under-paths like /usr/local/... stay legal; the catastrophe
# shapes are refused.
if grep -Eq 'rm[[:space:]][^;&|]*[[:space:]](/|/\*|/(etc|usr|var|bin|sbin|boot|lib|root|home|srv|opt)/?)([[:space:]]|$)' "$SCRIPT"; then
    deny "rm against a system root"
fi

# Rate limit — a loop with root is still a loop.
NOW=$(date +%s)
if [ -f "$RATE_FILE" ]; then
    awk -v cutoff="$((NOW - 3600))" '$1 > cutoff' "$RATE_FILE" > "$RATE_FILE.tmp" \
        && mv "$RATE_FILE.tmp" "$RATE_FILE"
fi
COUNT=$(wc -l < "$RATE_FILE" 2>/dev/null || echo 0)
[ "$COUNT" -ge "$MAX_PER_HOUR" ] \
    && deny "rate limit: $COUNT privileged runs in the last hour (max $MAX_PER_HOUR)"
echo "$NOW" >> "$RATE_FILE"
chmod 600 "$RATE_FILE" 2>/dev/null || true

# Audit before execution — the record survives even if the run does not.
{
    echo "=== $(date -Is) user=$CALLER script=$SCRIPT"
    echo "sha256: $(sha256sum "$SCRIPT" | cut -d' ' -f1)"
    sed 's/^/| /' "$SCRIPT"
    echo "=== end"
} >> "$AUDIT_LOG"
chmod 600 "$AUDIT_LOG" 2>/dev/null || true

echo "genesis_admin: executing $SCRIPT as root (logged to $AUDIT_LOG)"
cd "$STAGING"
umask "$UMASK_SAVED"  # the staged script runs with a normal umask
exec bash "$SCRIPT"
