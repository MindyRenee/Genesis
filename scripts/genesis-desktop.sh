#!/bin/bash
# Genesis's desktop — her own graphical session, nested as a window
# on the host display. Xephyr :N runs a real X server; openbox is the
# window manager inside it. Her headed browser and any GUI she opens
# land on this screen, never on the user's desktop. Her uinput hands
# reach it whenever the window has focus — click it once to aim.
#
# Idempotent: a live desktop is reused, a dead one is relaunched.
# The chosen display number is written to $DATA_DIR/desktop.display
# so tools can find her screen without searching.
#
# Usage: scripts/genesis-desktop.sh            (ensure + print DISPLAY)
#        scripts/genesis-desktop.sh --stop     (tear the desktop down)

set -u

DATA_DIR="${GENESIS_DATA_DIR:-$HOME/.local/share/genesis}"
MARKER="$DATA_DIR/desktop.display"
WIDTH="${GENESIS_DESKTOP_W:-1280}"
HEIGHT="${GENESIS_DESKTOP_H:-800}"

_die() { echo "genesis-desktop: $*" >&2; exit 1; }

if [ "${1:-}" = "--stop" ]; then
    if [ -f "$MARKER" ]; then
        D="$(cat "$MARKER")"
        pkill -f "Xephyr $D " 2>/dev/null
        rm -f "$MARKER"
        echo "  Desktop $D stopped."
    else
        echo "  No desktop running."
    fi
    exit 0
fi

command -v Xephyr >/dev/null || _die "Xephyr not found — install xserver-xephyr"
command -v openbox >/dev/null || _die "openbox not found — install openbox"
command -v xdpyinfo >/dev/null || _die "xdpyinfo not found — install x11-utils"

# Reuse a live desktop: marker exists AND the server answers.
if [ -f "$MARKER" ]; then
    D="$(cat "$MARKER")"
    if DISPLAY="$D" xdpyinfo >/dev/null 2>&1; then
        echo "$D"
        exit 0
    fi
    rm -f "$MARKER"
fi

# First free display number above :9 — host sessions live below.
N=10
while [ -e "/tmp/.X11-unix/X$N" ] || DISPLAY=":$N" xdpyinfo >/dev/null 2>&1; do
    N=$((N + 1))
    [ "$N" -lt 100 ] || _die "no free display number"
done
D=":$N"

mkdir -p "$DATA_DIR"
LOG="$DATA_DIR/desktop.log"

# The screen itself — a titled window on the host display so her
# session is watchable, -br for a plain root she can draw over.
Xephyr "$D" -title "Genesis" -screen "${WIDTH}x${HEIGHT}" -br \
    >>"$LOG" 2>&1 &
XEPHYR_PID=$!

# Wait for the server to answer (bounded).
for _ in $(seq 1 50); do
    DISPLAY="$D" xdpyinfo >/dev/null 2>&1 && break
    kill -0 "$XEPHYR_PID" 2>/dev/null || _die "Xephyr died — see $LOG"
    sleep 0.1
done
DISPLAY="$D" xdpyinfo >/dev/null 2>&1 || _die "Xephyr did not come up — see $LOG"

# The window manager — dbus so apps that want a session bus get one.
if command -v dbus-launch >/dev/null; then
    DISPLAY="$D" dbus-launch openbox >>"$LOG" 2>&1 &
else
    DISPLAY="$D" openbox >>"$LOG" 2>&1 &
fi
sleep 0.5

echo "$D" > "$MARKER"
echo "$D"
