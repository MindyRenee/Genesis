#!/data/data/com.termux/files/usr/bin/bash
# Run Genesis inside the Debian PRoot userspace while retaining access to
# Android capabilities exposed by native Termux:API commands.
set -euo pipefail

REPO_DIR="${GENESIS_REPO_DIR:-$HOME/Genesis}"
DISTRO="debian"

[ -d "$REPO_DIR/.git" ] || {
    echo "[genesis-android] Genesis checkout not found at $REPO_DIR" >&2
    echo "[genesis-android] Run android/termux-bootstrap.sh first." >&2
    exit 1
}

command -v proot-distro >/dev/null 2>&1 || {
    echo "[genesis-android] proot-distro is not installed." >&2
    exit 1
}

TERMUX_PREFIX="${TERMUX_PREFIX:-/data/data/com.termux/files/usr}"
for command_name in termux-camera-photo termux-microphone-record termux-tts-speak; do
    if [ ! -x "$TERMUX_PREFIX/bin/$command_name" ]; then
        echo "[genesis-android] Android capability unavailable: $command_name" >&2
        echo "[genesis-android] Install/enable the Termux Android API support for your Termux distribution." >&2
        exit 1
    fi
done

echo "[genesis-android] Starting Genesis in Debian PRoot..."
echo "[genesis-android] Android camera: Termux:API"
echo "[genesis-android] Android microphone: Termux:API + ffmpeg + Vosk"
echo "[genesis-android] Android speaker: Termux:API TTS"
echo

exec proot-distro login "$DISTRO" \
    --bind "$REPO_DIR:/workspace/Genesis" \
    -- /bin/bash -lc '
        set -e
        cd /workspace/Genesis
        [ -f .venv/bin/activate ] || {
            echo "[genesis-android] Python environment missing; rerun android/termux-bootstrap.sh." >&2
            exit 1
        }
        . .venv/bin/activate
        export TERMUX_PREFIX="/data/data/com.termux/files/usr"
        export GENESIS_RUN=1
        . "$HOME/.cargo/env" 2>/dev/null || true
        exec ./run.sh "$@"
    ' -- "$@"
