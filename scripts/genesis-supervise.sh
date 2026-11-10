#!/bin/bash
# Resident-mode supervisor for hosts without a systemd user manager
# (containers, proot/andronix, WSL without systemd, minimal init).
# Replicates the genesis.service contract:
#
#   ExecStart=genesis --headless            one organism per data dir
#   Restart=on-failure  RestartSec=10s      clean exit or SIGTERM stays down
#   SuccessExitStatus=SIGTERM               genesis --stop is not a failure
#   EnvironmentFile=-~/.config/genesis/agent.env
#
# Run detached:  setsid nohup scripts/genesis-supervise.sh >>"$HOME/.local/share/genesis/supervisor.log" 2>&1 &
# Stop the whole organism: genesis --stop  (or kill this script — TERM is
# forwarded to the child so the body is never orphaned).
set -u

GENESIS="${GENESIS_BIN:-$HOME/.local/bin/genesis}"
ENV_FILE="${GENESIS_ENV_FILE:-$HOME/.config/genesis/agent.env}"
RESTART_SEC="${GENESIS_RESTART_SEC:-10}"

if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

child=0
stopping=0
on_term() {
    stopping=1
    [ "$child" -gt 0 ] && kill -TERM "$child" 2>/dev/null
}
trap on_term TERM INT

while [ "$stopping" -eq 0 ]; do
    "$GENESIS" --headless &
    child=$!
    wait "$child"
    rc=$?
    child=0
    [ "$stopping" -eq 1 ] && break
    # 0 = clean exit, 143 = SIGTERM: deliberate stop, not a failure.
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 143 ]; then
        exit "$rc"
    fi
    echo "[genesis-supervise] $(date '+%F %T') genesis exited rc=$rc — restart in ${RESTART_SEC}s" >&2
    sleep "$RESTART_SEC" &
    child=$!
    wait "$child"
    child=0
done
exit 0
