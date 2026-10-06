#!/data/data/com.termux/files/usr/bin/bash
# Genesis Android bootstrap for Termux + Debian PRoot.
# Installs the Linux userspace needed by Genesis and leaves Android device
# capabilities in the native Termux environment.
set -euo pipefail

REPO_DIR="${GENESIS_REPO_DIR:-$HOME/Genesis}"
DISTRO="debian"
BRANCH="${GENESIS_BRANCH:-multi-fiber-topology-refactor}"

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "[genesis-android] missing command: $1" >&2
        exit 1
    }
}

echo "[genesis-android] Updating Termux packages..."
pkg update -y
pkg install -y git proot-distro curl

need_cmd proot-distro
need_cmd git

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[genesis-android] Cloning Genesis..."
    git clone --branch "$BRANCH" --single-branch \
        https://github.com/MindyRenee/Genesis.git "$REPO_DIR"
else
    echo "[genesis-android] Updating existing Genesis checkout..."
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi

if ! proot-distro list 2>/dev/null | grep -Eq "(^|[[:space:]])$DISTRO([[:space:]]|$)"; then
    echo "[genesis-android] Installing Debian PRoot..."
    proot-distro install "$DISTRO"
fi

echo "[genesis-android] Installing Linux build/runtime dependencies..."
proot-distro login "$DISTRO" -- /bin/bash -lc '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y \
        build-essential \
        clang \
        libclang-dev \
        linux-libc-dev \
        pkg-config \
        curl \
        git \
        ca-certificates \
        python3 \
        python3-venv \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        portaudio19-dev
'

echo "[genesis-android] Installing Rust..."
proot-distro login "$DISTRO" -- /bin/bash -lc '
    set -e
    if ! command -v rustc >/dev/null 2>&1 || ! rustc --version | grep -q "1.8[5-9]\|1.[9-9][0-9]"; then
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi
    . "$HOME/.cargo/env"
    rustc --version
    cargo --version
'

echo "[genesis-android] Preparing Genesis Python environment..."
proot-distro login "$DISTRO" --bind "$REPO_DIR:/workspace/Genesis" -- /bin/bash -lc '
    set -e
    cd /workspace/Genesis
    if [ ! -d .venv ]; then
        python3 -m venv .venv
    fi
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r python/requirements.txt
    python -m pip install vosk==0.3.45
    mkdir -p "$HOME/.local/share/vosk-models"
    if [ ! -d "$HOME/.local/share/vosk-models/vosk-model-small-en-us-0.15" ]; then
        tmp="$(mktemp -d)"
        curl -fsSL https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -o "$tmp/model.zip"
        python -c "import zipfile; zipfile.ZipFile(\"$tmp/model.zip\").extractall(\"$HOME/.local/share/vosk-models\")"
        rm -rf "$tmp"
    fi
'

echo
echo "[genesis-android] Bootstrap complete."
echo "[genesis-android] Repo: $REPO_DIR"
echo "[genesis-android] Next: $REPO_DIR/android/termux-run.sh"
