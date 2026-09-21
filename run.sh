#!/bin/bash
# Run Genesis — start and stop everything from one script.
#
# This is the only supported way to start and stop Genesis.
# Starting launches the daemon, cognitive CLI, retina, and TTS.
# Ctrl-C in the running terminal tears them all down.
#
# Usage: ./run.sh          (start Genesis)
#        ./run.sh --stop   (stop a running Genesis)
#        ./run.sh --offline  (start in offline mode)
# Flags can be combined in any order: ./run.sh --offline --stop

set -m

cd "$(dirname "$0")" || exit 1

DATA_DIR=$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' \
    "${GENESIS_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/genesis-public}") || exit 1

# ─── PID-file helpers ────────────────────────────────────────────────
# The CLI writes PID files for itself, the daemon, and the retina to
# the data dir. We use these to kill exact processes instead of broad
# pkill patterns that could match unrelated user processes (e.g.
# "aplay", "piper", "retina" run by other applications).

# Read a PID from a PID file. Returns 0 and echoes the PID if the file
# exists and the PID is alive; returns 1 otherwise.
_read_pid() {
    local pidfile="$1"
    [ -f "$pidfile" ] || return 1
    local pid
    pid=$(cat "$pidfile" 2>/dev/null) || return 1
    _matches_pid "$pid" "$pidfile" || return 1
    echo "$pid"
    return 0
}

_matches_pid() {
    local pid="$1" pidfile="$2" i
    [[ "$pid" =~ ^[1-9][0-9]{0,9}$ ]] && [ "$pid" -gt 1 ] || return 1
    local -a args=()
    mapfile -d '' -t args < "/proc/$pid/cmdline" 2>/dev/null || return 1
    [ "${#args[@]}" -gt 0 ] || return 1
    case "${pidfile##*/}" in
        genesis_cli.pid)
            [[ "${args[0]##*/}" == python* ]] || return 1
            [[ "${args[1]:-}" == */genesis_cli.py || "${args[1]:-}" == genesis_cli.py ]] || return 1
            ;;
        genesis_daemon.pid)
            [ "${args[0]##*/}" = genesis-daemon ] || return 1
            ;;
        genesis_retina.pid)
            [ "${args[0]##*/}" = retina ] || return 1
            # Primary identity: stdout redirected to this instance's
            # retina.log. Log rotation renames the file, so the fd may
            # point at a rotated/deleted inode — fall back to the
            # --data-dir argument match below in that case.
            if [[ "$(readlink "/proc/$pid/fd/1" 2>/dev/null)" == "$DATA_DIR/retina.log"* ]]; then
                return 0
            fi
            ;;
        *) return 1 ;;
    esac
    for ((i=1; i < ${#args[@]}-1; i++)); do
        if [ "${args[i]}" = --data-dir ]; then
            # An explicit --data-dir decides identity exactly: match or
            # fail — never fall through to the default-dir heuristic,
            # or a process bound to a different directory would be
            # claimed by this instance.
            [ "${args[i+1]}" = "$DATA_DIR" ] && return 0 || return 1
        fi
    done
    # A daemon/CLI started without --data-dir uses the default dir.
    # Only accept the match when this DATA_DIR *is* the default, so a
    # custom-dir instance never claims a default-dir process.
    local default_dir="$HOME/.local/share/genesis-public"
    [ -n "${XDG_DATA_HOME:-}" ] && default_dir="$XDG_DATA_HOME/genesis-public"
    [ -n "${GENESIS_DATA_DIR:-}" ] && default_dir="$GENESIS_DATA_DIR"
    if [ "$DATA_DIR" = "$default_dir" ]; then
        return 0
    fi
    return 1
}

_find_pid() {
    local pidfile="$1" entry pid
    _read_pid "$pidfile" && return 0
    for entry in /proc/[0-9]*/cmdline; do
        pid="${entry#/proc/}"
        pid="${pid%/cmdline}"
        if _matches_pid "$pid" "$pidfile" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    done
    return 1
}

# Request graceful shutdown and wait for the identified process to exit.
# If signaling fails or shutdown takes too long, preserve all files and
# return failure. A successful signal alone does not establish that the
# process has finished saving state; never escalate to SIGKILL.
_kill_pidfile() {
    local pidfile="$1"
    local signal="$2"
    local pid deadline
    pid=$(_find_pid "$pidfile") || return 0
    kill -"$signal" "$pid" 2>/dev/null || return 1
    deadline=$((SECONDS + 120))
    while _matches_pid "$pid" "$pidfile" 2>/dev/null; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "  Shutdown still pending for PID $pid; state and locks preserved." >&2
            return 1
        fi
        sleep 1
    done
    return 0
}

# ─── Stop helper ─────────────────────────────────────────────────────
# Tear down every Genesis process for this data dir.
# Prefers validated PID files; falls back to exact /proc arguments
# for this data directory if PID files are missing or stale.
_stop_all() {
    local cli_pidfile="$DATA_DIR/genesis_cli.pid"
    local daemon_pidfile="$DATA_DIR/genesis_daemon.pid"
    local retina_pidfile="$DATA_DIR/genesis_retina.pid"

    # 1. Stop the cognitive CLI first (it owns the daemon lifecycle).
    #    Validate PID files and fall back to exact /proc arguments.
    #    Each step has its own 120s grace period; a failure returns
    #    immediately so daemon/retina may be left running — retry
    #    ./run.sh --stop after the CLI finishes saving state.
    _kill_pidfile "$cli_pidfile" INT || { echo "  CLI shutdown pending; retry --stop once it exits." >&2; return 1; }

    # 2. The helper waits for the CLI to finish before proceeding.

    # 3. Gracefully stop orphaned children only after the CLI exits.
    #    Keep their PID files on failure so shutdown can be retried.
    #    Never interrupt an in-progress cognitive state save.
    _kill_pidfile "$daemon_pidfile" TERM || { echo "  Daemon shutdown pending; state preserved, retry --stop." >&2; return 1; }
    _kill_pidfile "$retina_pidfile" TERM || { echo "  Retina shutdown pending; retry --stop." >&2; return 1; }

    # Fallback discovery is handled by _find_pid using /proc arguments.
    # Exact data-directory matching avoids regex metacharacters and
    # whitespace confusing unrelated processes with this instance.
    # Retina is identified by its executable and per-instance log,
    # not by a project-wide process-name pattern.
    # Voice subprocess cleanup remains the cognitive CLI's responsibility.

    # 4. Leave lock inodes intact; the owning processes clean their sockets.
    return 0
}

# Check whether any Genesis process is running for this data dir.
_is_running() {
    local cli_pidfile="$DATA_DIR/genesis_cli.pid"
    local daemon_pidfile="$DATA_DIR/genesis_daemon.pid"
    local retina_pidfile="$DATA_DIR/genesis_retina.pid"
    _find_pid "$cli_pidfile" >/dev/null 2>&1 && return 0
    _find_pid "$daemon_pidfile" >/dev/null 2>&1 && return 0
    _find_pid "$retina_pidfile" >/dev/null 2>&1 && return 0
    return 1
}

# ─── Parse arguments ─────────────────────────────────────────────────
# Scan all args for flags so order doesn't matter. This prevents
# silent misbehavior when the user passes e.g. ./run.sh --offline --stop
# (which would otherwise try to START Genesis instead of stopping it).
WANT_STOP=0
OFFLINE_ARG=""
for arg in "$@"; do
    case "$arg" in
        --stop)    WANT_STOP=1 ;;
        --offline) OFFLINE_ARG="--offline" ;;
        *) echo "  Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# ─── Handle --stop ───────────────────────────────────────────────────
if [ "$WANT_STOP" -eq 1 ]; then
    if _is_running; then
        echo "  Stopping Genesis..."
        _stop_all || exit 1
        echo "  Genesis stopped."
    else
        echo "  Genesis is not running."
    fi
    exit 0
fi

mkdir -p "$DATA_DIR" || exit 1

# Guard against two checkouts sharing one mind. The data dir records
# which project tree created it; a different tree reusing it would
# corrupt both developmental records. Warn loudly — do not refuse,
# because deliberately moving or sharing state is legitimate.
MARKER="$DATA_DIR/.project_root"
if [ -f "$MARKER" ]; then
    marker_content="$(cat "$MARKER" 2>/dev/null || true)"
    if [ -n "$marker_content" ] && [ "$marker_content" != "$PWD" ]; then
        echo "  WARNING: this data dir was created by a different checkout:"
        echo "    $marker_content"
        echo "  This instance's state belongs to that tree. Two checkouts"
        echo "  sharing one data dir corrupt each other's developmental"
        echo "  record. Use XDG_DATA_HOME or remove the state if you really"
        echo "  intend to continue from here."
    fi
else
    echo "$PWD" > "$MARKER"
fi

# Refuse to start if a Genesis session is already running for this data dir.
if _is_running; then
    echo "  Genesis is already running. Stop it with: ./run.sh --stop"
    exit 1
fi

# Existing retina processes for this instance are checked above.
# Do not signal camera processes belonging to another data directory.

# EXIT trap — report unfinished shutdown without interrupting state saves.
# The CLI owns child cleanup. If startup loses a singleton-lock race,
# this shell must not stop the winning instance or unlink its locks.
_atexit() {
    # Use validated process identities when checking for remaining work.
    if _is_running; then
        echo "  Genesis processes remain; use ./run.sh --stop for graceful shutdown." >&2
    fi
    # Preserve socket/lock/PID files for their owners and later recovery.
}

trap _atexit EXIT

# ── Pre-flight: ensure the release binary exists ──────────────
# There is one canonical build: `cargo build --release`. No debug
# fallback. If the binary is missing, fail fast with a clear message
# instead of starting the CLI and having it die mid-launch.
RELEASE_BIN="target/release/genesis-daemon"
if [ ! -x "$RELEASE_BIN" ]; then
    echo "[genesis] Release binary not found at $RELEASE_BIN"
    echo "[genesis] Building with: cargo build --release"
    cargo build --release || {
        echo "[genesis] Build failed. Genesis cannot start."
        exit 1
    }
fi

# Only run.sh is allowed to launch the CLI.
export GENESIS_RUN=1

# Start Genesis — voice and ambient listening always on.
# Ctrl-C in this terminal sends SIGINT to the foreground CLI, which
# shuts down the daemon and retina before exiting; the EXIT trap
# reports any processes still needing a graceful shutdown.
#
# OFFLINE_ARG was parsed from the command line above (--offline for
# no network access).
python3 python/genesis_cli.py --data-dir "$DATA_DIR" ${OFFLINE_ARG:-}
