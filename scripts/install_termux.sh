#!/data/data/com.termux/files/usr/bin/bash
# Genesis3 native Termux installer. No systemd, sudoers, or LLM.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

if [[ -z "${PREFIX:-}" || "$PREFIX" != *"/com.termux/files/usr" ]]; then
    printf 'Run this installer in native Termux, not desktop Linux or an isolated proot.\n' >&2
    exit 2
fi
command -v pkg >/dev/null 2>&1 || { echo "pkg is required" >&2; exit 2; }
command -v python >/dev/null 2>&1 || { echo "Python is required" >&2; exit 2; }

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Genesis requires Python 3.12 or newer; found " + sys.version.split()[0])
PY

# Avoid pkg upgrade: it can replace the phone's Python runtime and disrupt
# unrelated Termux work. Add repositories before installing native packages.
pkg install -y tur-repo x11-repo
pkg update
pkg install -y \
    git python python-pip python-setuptools rust clang make pkg-config \
    ffmpeg termux-api python-numpy python-pillow python-psutil \
    python-scipy python-cairo opencv-python

INSTALL_HOME="$HOME/.local/lib/genesis"
VENV="$INSTALL_HOME/venv"
mkdir -p "$INSTALL_HOME"
python -m venv --system-site-packages "$VENV"
VENV_PY="$VENV/bin/python"

# Termux's ABI-coupled scientific packages must come from pkg, not Linux wheels.
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install "nltk==3.10.3" "orjson==3.12.0" "websockets==15.0.1"
"$VENV_PY" -m pip install --no-build-isolation --no-deps -e .

PYTHONPATH="$ROOT/python:$ROOT" "$VENV_PY" - <<'PY'
import importlib
modules = ("numpy", "nltk", "scipy", "orjson", "psutil", "PIL", "cv2", "cairo", "websockets")
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("Genesis Termux dependency check failed:\n  " + "\n  ".join(missing))
print("Genesis Termux Python dependency check passed.")
PY

cargo build --release --bin genesis-daemon

cat <<'EOF'

Genesis3 Termux build completed. Runtime behavior still requires testing on this phone.

Run from this checkout:
  ./genesis

Optional hardware integrations are opt-in:
  export GENESIS_ANDROID_SENSORS=1
  export GENESIS_ANDROID_CAMERA=1
  export GENESIS_ANDROID_MICROPHONE=1

Install the Termux:API companion app and grant the requested permissions.
Camera and microphone capture may require Termux to remain in the foreground.
The installer does not enable these sensors automatically.
EOF
