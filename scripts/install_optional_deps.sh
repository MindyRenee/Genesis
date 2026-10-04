#!/bin/bash
# Install and VERIFY Genesis's optional native capabilities.
#
# These are deliberately not in requirements.txt: the Python wheels are
# ordinary, but the things they bind to (PortAudio for the microphone,
# cairo for drawing, a TTS engine for the voice) are system libraries
# that may or may not be present. Installing the wheels alone produces
# a half-working install that fails late and confusingly — `sounddevice`
# in particular imports cleanly and only raises when it tries to
# dlopen libportaudio, which turns "no microphone" into a crash in the
# audio thread.
#
# So this script installs the wheels, installs the system libraries it
# can, and then *proves* each capability actually works. Anything it
# cannot fully verify is reported as unavailable rather than left to
# fail at runtime.
#
# Usage:  ./scripts/install_optional_deps.sh            # install + verify
#         ./scripts/install_optional_deps.sh --verify   # verify only
#
# Safe to re-run: every step is idempotent.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
PY=${PYTHON:-python3}
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

# Blank means unset, so this agrees with run.sh, config.py, and
# src/data_dir.rs instead of building a path out of whitespace.
if [ -n "${GENESIS_DATA_DIR:-}" ] && [ -z "${GENESIS_DATA_DIR//[[:space:]]/}" ]; then
    unset GENESIS_DATA_DIR
fi


# Where the vosk model lives. Must match
# genesis_conscious.infrastructure.config.vosk_model_dir(), which is the
# single source of truth; this shell resolves the same order run.sh does
# so the installer and the loader never disagree. An explicit
# GENESIS_VOSK_DIR wins outright.
if [ -n "${GENESIS_VOSK_DIR:-}" ]; then
    VOSK_DIR="$GENESIS_VOSK_DIR"
elif [ -n "${GENESIS_DATA_DIR:-}" ]; then
    VOSK_DIR="$GENESIS_DATA_DIR/vosk-models"
elif [ -f .genesis-data-dir ]; then
    VOSK_DIR="$(head -n1 .genesis-data-dir)/vosk-models"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
    VOSK_DIR="$XDG_DATA_HOME/vosk-models"
else
    VOSK_DIR="$HOME/.local/share/vosk-models"
fi
VOSK_NAME="vosk-model-small-en-us-0.15"
VOSK_URL="https://alphacephei.com/vosk/models/${VOSK_NAME}.zip"

# Report where the model would be installed and exit. Resolving the data
# dir is the part with a history of being wrong, so it must be
# inspectable without running an install; see
# python/tests/test_data_dir_conformance.py.
if [ "${1:-}" = "--data-dir" ]; then
    printf '%s\n' "$VOSK_DIR"
    exit 0
fi

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
head_() { printf '\n== %s ==\n' "$*"; }

# Run a command with sudo only if we can do so without a password
# prompt. Never blocks waiting for a password.
have_sudo() { sudo -n true 2>/dev/null; }

# ─────────────────────────────────────────────────────────────────
# 1. Python wheels
# ─────────────────────────────────────────────────────────────────
head_ "Python packages"
if [ "$VERIFY_ONLY" -eq 0 ]; then
    # --only-binary is deliberate. Every one of these publishes a
    # manylinux wheel; if pip tries to build from source it will fail
    # on a missing toolchain, and a source build of sounddevice is a
    # common way to end up with a half-installed tree.
    if "$PY" -m pip install --only-binary=:all: \
        vosk sounddevice SpeechRecognition piper-tts; then
        ok "wheels installed"
    else
        bad "wheel install failed (see above)"
    fi
else
    say "  (verify only — skipping install)"
fi

# ─────────────────────────────────────────────────────────────────
# 2. System libraries
# ─────────────────────────────────────────────────────────────────
head_ "System libraries"
install_pkgs() {
    local pkgs=("$@")
    if have_sudo; then
        if sudo -n apt-get install -y "${pkgs[@]}" 2>/dev/null; then
            return 0
        fi
    fi
    if command -v apt-get >/dev/null && have_sudo; then
        sudo -n apt-get update >/dev/null 2>&1
        sudo -n apt-get install -y "${pkgs[@]}" >/dev/null 2>&1
        return $?
    fi
    return 1
}

# PortAudio — microphone capture for sounddevice.
if "$PY" -c "import sounddevice as sd; sd.query_devices()" >/dev/null 2>&1; then
    ok "PortAudio present (sounddevice can open devices)"
elif [ "$VERIFY_ONLY" -eq 0 ]; then
    if install_pkgs libportaudio2 portaudio19-dev; then
        ok "installed PortAudio"
    else
        warn "PortAudio NOT installed — microphone unavailable."
        warn "  Needs: sudo apt-get install libportaudio2 portaudio19-dev"
    fi
fi

# cairo — canvas drawing.
if "$PY" -c "import cairo" >/dev/null 2>&1; then
    ok "pycairo present (drawing enabled)"
elif [ "$VERIFY_ONLY" -eq 0 ]; then
    # pycairo is source-only, so this also needs a C toolchain. Point
    # pip at the Zig shim if the host has no cc (see scripts/cargo-env.sh).
    if ! command -v cc >/dev/null && [ -x "${HOME}/tools/zig-x86_64-linux-0.16.0/zig" ]; then
        warn "no system C compiler; trying the Zig shim for pycairo"
        mkdir -p /tmp/genesis-cc-shim
        cat > /tmp/genesis-cc-shim/cc <<EOF
#!/bin/sh
exec ${HOME}/tools/zig-x86_64-linux-0.16.0/zig cc "\$@"
EOF
        chmod +x /tmp/genesis-cc-shim/cc
        PATH="/tmp/genesis-cc-shim:${PATH}" "$PY" -m pip install pycairo \
            && ok "pycairo built via Zig shim" \
            || warn "pycairo build failed — drawing unavailable"
    elif install_pkgs libcairo2-dev; then
        "$PY" -m pip install pycairo \
            && ok "pycairo installed" \
            || warn "pycairo build failed — drawing unavailable"
    else
        warn "libcairo2-dev NOT installed — drawing unavailable."
        warn "  Needs: sudo apt-get install libcairo2-dev  (then: pip install pycairo)"
    fi
fi

# ─────────────────────────────────────────────────────────────────
# 3. Vosk speech model
# ─────────────────────────────────────────────────────────────────
head_ "Vosk model"
if [ -d "${VOSK_DIR}/${VOSK_NAME}" ]; then
    ok "present at ${VOSK_DIR}/${VOSK_NAME}"
elif [ "$VERIFY_ONLY" -eq 0 ]; then
    say "  downloading ~40 MB from alphacephei.com ..."
    if "$PY" - "$VOSK_DIR" "$VOSK_URL" <<'PYEOF'
import io, sys, urllib.request, zipfile
dest, url = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url, timeout=300) as r:
    blob = r.read()
zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
PYEOF
    then
        ok "downloaded"
    else
        warn "download failed — offline speech recognition unavailable"
    fi
fi

# Piper: the code looks for the binary next to the voice models.
head_ "Piper (text to speech)"
VOICES_DIR="${PROJECT_ROOT}/python/voices"
if [ -x "${VOICES_DIR}/piper" ] || command -v piper >/dev/null; then
    ok "piper available"
elif [ "$VERIFY_ONLY" -eq 0 ]; then
    PIPER_BIN=$("$PY" -c "import shutil;print(shutil.which('piper') or '')" 2>/dev/null)
    if [ -n "$PIPER_BIN" ]; then
        mkdir -p "$VOICES_DIR"
        ln -sf "$PIPER_BIN" "${VOICES_DIR}/piper"
        ok "linked ${VOICES_DIR}/piper -> ${PIPER_BIN}"
    else
        warn "piper binary not found — voice unavailable"
    fi
fi
if ! ls "${VOICES_DIR}"/*.onnx >/dev/null 2>&1; then
    warn "no .onnx voice models in ${VOICES_DIR} — voice unavailable"
fi

# ─────────────────────────────────────────────────────────────────
# 4. Verify — prove each capability, don't assume it
# ─────────────────────────────────────────────────────────────────
head_ "Verification"
cd "${PROJECT_ROOT}/python" || exit 1
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")

rows = []

def check(name, fn):
    try:
        detail = fn()
        rows.append(("ok", name, detail or ""))
    except Exception as e:                                    # noqa: BLE001
        rows.append(("FAIL", name, f"{type(e).__name__}: {e}"))

def tts():
    from genesis_conscious.temporal_lobe.speech import _PIPER_BIN, _PIPER_MODEL
    import os
    if not (_PIPER_BIN.exists() and os.access(_PIPER_BIN, os.X_OK)):
        raise RuntimeError("piper binary missing/unexecutable")
    if not _PIPER_MODEL.exists():
        raise RuntimeError(f"voice model missing: {_PIPER_MODEL}")
    import subprocess, tempfile, wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out = f.name
    subprocess.run(
        [str(_PIPER_BIN), "-m", str(_PIPER_MODEL), "-f", out],
        input=b"verification", check=True, timeout=180,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with wave.open(out) as w:
        secs = w.getnframes() / w.getframerate()
    if secs <= 0:
        raise RuntimeError("synthesised 0 seconds of audio")
    return f"synthesised {secs:.2f}s @ {w.getframerate()}Hz"

def stt():
    from genesis_conscious.temporal_lobe.speech import (
        VOSK_MODEL_DIR, VOSK_MODEL_NAME,
    )
    p = VOSK_MODEL_DIR / VOSK_MODEL_NAME
    if not p.exists():
        raise RuntimeError(f"model missing at {p}")
    import vosk
    vosk.Model(str(p))
    return "model loads"

def mic():
    import sounddevice as sd
    devs = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    if not devs:
        raise RuntimeError("no input devices")
    return f"{len(devs)} input device(s)"

def drawing():
    import cairo
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
    return f"surface {s.get_width()}x{s.get_height()}"

def ions():
    from genesis_client.protocol import GET_ION_SUMMARY
    from genesis_conscious.neurochemical.electrochemistry import interpret_ions
    interpret_ions(None)
    return f"opcode {GET_ION_SUMMARY}, neutral ctx ok"

for n, f in (
    ("text to speech", tts),
    ("speech recognition", stt),
    ("microphone", mic),
    ("canvas drawing", drawing),
    ("ion layer", ions),
):
    check(n, f)

width = max(len(n) for _, n, _ in rows)
n_fail = 0
for status, name, detail in rows:
    mark = "ok  " if status == "ok" else "FAIL"
    if status != "ok":
        n_fail += 1
    print(f"  [{mark}] {name:<{width}}  {detail}")

print()
if n_fail:
    print(f"{n_fail} capability/capabilities unavailable — see [FAIL] above.")
    print("Genesis still runs; these degrade to text-only operation.")
else:
    print("All optional capabilities verified.")
PYEOF

head_ "Done"
say "  Genesis runs text-only when a capability is unavailable; that is"
say "  expected and not an error. Re-check any time with:"
say "    ./scripts/install_optional_deps.sh --verify"