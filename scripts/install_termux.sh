#!/data/data/com.termux/files/usr/bin/bash
# Install Genesis natively in Termux.
#
#   ./scripts/install_termux.sh
#
# Termux owns the native scientific/vision packages through pkg. Do not
# ask pip to replace those packages with desktop manylinux wheels or build
# NumPy/SciPy/OpenCV from source. The venv deliberately sees Termux's
# site-packages, and the project is installed with --no-deps.
#
# This installer does not use sudo, apt, systemd, or desktop Linux paths.
# It builds the Rust daemon and installs the Python entry point only.

set -euo pipefail

say() { printf '[genesis-termux] %s\n' "$*"; }
fail() { printf '[genesis-termux] ERROR: %s\n' "$*" >&2; exit 1; }

case "${PREFIX:-}" in
    /data/data/com.termux/files/usr|/data/user/0/com.termux/files/usr) ;;
    *) fail "This installer must run inside Termux (PREFIX is '${PREFIX:-unset}')." ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_HOME="${GENESIS_INSTALL_HOME:-$HOME/.local/lib/genesis}"
VENV="$INSTALL_HOME/venv"
USER_BIN="$HOME/.local/bin"

command -v pkg >/dev/null 2>&1 || fail "Termux pkg command not found."
command -v python >/dev/null 2>&1 || fail "Python is missing. Run: pkg install python"
command -v cargo >/dev/null 2>&1 || fail "Rust is missing. Run: pkg install rust"
command -v clang >/dev/null 2>&1 || fail "Clang is missing. Run: pkg install clang"
command -v pkg-config >/dev/null 2>&1 || fail "pkg-config is missing. Run: pkg install pkg-config"

say "Checking Termux Python and native runtime dependencies."
python - <<'PY'
import importlib
import sys

required = {
    "numpy": "numpy",
    "scipy": "scipy",
    "nltk": "nltk",
    "orjson": "orjson",
    "psutil": "psutil",
    "PIL": "Pillow",
    "cv2": "OpenCV",
    "cairo": "pycairo",
    "websockets": "websockets",
    "arc_agi": "arc-agi",
    "arcengine": "arcengine",
}
missing = []
for module, package in required.items():
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "")
        print(f"  [ok] {package}" + (f" {version}" if version else ""))
    except Exception as exc:
        missing.append((package, f"{type(exc).__name__}: {exc}"))
if missing:
    print("\nMissing or unloadable runtime dependencies:", file=sys.stderr)
    for package, error in missing:
        print(f"  [missing] {package}: {error}", file=sys.stderr)
    print(
        "\nInstall native packages with Termux pkg where available; "
        "install pure-Python packages with pip only when needed.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

say "Installing Termux build prerequisites is intentionally not automatic."
say "Building the Rust daemon."
(cd "$REPO" && cargo build --release --bin genesis-daemon)

mkdir -p "$INSTALL_HOME" "$USER_BIN"
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating shared-site-packages venv at $VENV."
    python -m venv --system-site-packages "$VENV"
fi

say "Installing Genesis from this checkout without replacing Termux packages."
"$VENV/bin/python" -m pip install --no-deps -e "$REPO"

[ -x "$REPO/target/release/genesis-daemon" ] || fail "Rust daemon build did not produce genesis-daemon."
cp "$REPO/target/release/genesis-daemon" "$VENV/bin/genesis-daemon"
ln -sf "$VENV/bin/genesis" "$USER_BIN/genesis"

say "Verifying the installed entry point and native imports."
"$VENV/bin/python" - <<'PY'
import importlib
import shutil
import sys

for module in ("numpy", "scipy", "nltk", "orjson", "psutil", "PIL", "cv2",
               "cairo", "websockets", "arc_agi", "arcengine"):
    importlib.import_module(module)

if shutil.which("cargo") is None:
    raise SystemExit("cargo is not on PATH")
print(f"Python {sys.version.split()[0]}: runtime imports passed")
PY

say "Installed. Launch with: $USER_BIN/genesis"
say "No systemd service was installed; start and stop Genesis from Termux."
