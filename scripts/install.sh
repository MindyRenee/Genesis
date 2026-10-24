#!/bin/bash
# Install Genesis as a resident agent on this machine.
#
#   scripts/install.sh               install everything
#   scripts/install.sh --no-service  install without the systemd unit
#   scripts/install.sh --uninstall   remove the install (her data stays)
#
# Layout:
#   ~/.local/lib/genesis/venv/    python env, genesis-cognitive (editable)
#   ~/.local/lib/genesis/venv/bin/{genesis-daemon,retina}
#   ~/.local/bin/genesis          PATH-visible launcher
#   ~/.config/genesis/agent.env   deployed environment (Discord webhook, etc.)
#   ~/.config/systemd/user/genesis.service
#
# The install is editable: the packages resolve to this checkout, so
# the install is bound to the repo path — move the checkout and re-run
# this script. Her state is never touched: the data dir resolves the
# same way run.sh resolves it (~/.local/share/genesis).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_HOME="${GENESIS_INSTALL_HOME:-$HOME/.local/lib/genesis}"
VENV="$INSTALL_HOME/venv"
USER_BIN="$HOME/.local/bin"
CONF_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
AGENT_CONF="$CONF_HOME/genesis"
UNIT_DIR="$CONF_HOME/systemd/user"
UNIT="$UNIT_DIR/genesis.service"

say() { echo "[genesis-install] $*"; }

want_service=1
uninstall=0
for arg in "$@"; do
    case "$arg" in
        --no-service) want_service=0 ;;
        --uninstall)  uninstall=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ "$uninstall" -eq 1 ]; then
    say "Removing the Genesis install (her data dir is NOT touched)"
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user disable --now genesis.service 2>/dev/null || true
        systemctl --user daemon-reload 2>/dev/null || true
    fi
    rm -f "$UNIT"
    rm -f "$USER_BIN/genesis"
    rm -f "$AGENT_CONF/agent.env"
    # Man pages installed from docs/man/ live in the user man dir.
    MAN_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/man"
    if [ -d "$REPO/docs/man" ]; then
        for page in "$REPO"/docs/man/*.[0-9]; do
            [ -e "$page" ] || continue
            section="${page##*.}"
            rm -f "$MAN_HOME/man$section/$(basename "$page")"
        done
    fi
    rm -rf "$VENV"
    rmdir "$INSTALL_HOME" "$AGENT_CONF" 2>/dev/null || true
    say "Done. Her state remains at \${XDG_DATA_HOME:-~/.local/share}/genesis."
    exit 0
fi

# ── 1. Rust binaries ────────────────────────────────────────────────
if [ ! -x "$REPO/target/release/genesis-daemon" ] || \
   [ ! -x "$REPO/target/release/retina" ]; then
    say "Building release binaries (cargo build --release)..."
    (cd "$REPO" && cargo build --release)
fi
for bin in genesis-daemon retina; do
    [ -x "$REPO/target/release/$bin" ] || {
        say "Missing binary: $REPO/target/release/$bin — build failed?"
        exit 1
    }
done

# ── 2. Python environment ───────────────────────────────────────────
# --system-site-packages: the host already has the heavy runtime deps
# (numpy, cv2, cairo…); the venv only needs the package + entry point.
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating venv at $VENV"
    python3 -m venv --system-site-packages "$VENV"
fi

say "Installing genesis-cognitive (editable — code stays this checkout)"
if ! "$VENV/bin/python" -m pip install -e "$REPO" 2>/dev/null; then
    # Offline or old toolchain: use the host's setuptools directly.
    say "Isolated build failed; retrying with --no-build-isolation"
    "$VENV/bin/python" -m pip install -e "$REPO" --no-build-isolation
fi

# Ship the binaries next to the interpreter — the CLI's packaged-install
# lookup (venv/bin). Editable installs also find target/release first;
# these copies keep the venv self-contained.
cp "$REPO/target/release/genesis-daemon" "$REPO/target/release/retina" \
    "$VENV/bin/"

mkdir -p "$USER_BIN"
ln -sf "$VENV/bin/genesis" "$USER_BIN/genesis"
say "Launcher: $USER_BIN/genesis"

# ── 3. Deployed environment ─────────────────────────────────────────
mkdir -p "$AGENT_CONF"
if [ -f "$REPO/.genesis-discord" ] && [ ! -f "$AGENT_CONF/agent.env" ]; then
    webhook="$(head -n1 "$REPO/.genesis-discord" | tr -d '[:space:]')"
    if [ -n "$webhook" ]; then
        printf 'GENESIS_DISCORD_WEBHOOK=%s\n' "$webhook" > "$AGENT_CONF/agent.env"
        chmod 600 "$AGENT_CONF/agent.env"
        say "Wrote $AGENT_CONF/agent.env (Discord mirror)"
    fi
fi

# ── 3b. Man pages ──────────────────────────────────────────────────
# Sources live in docs/man/ so they version with the code they
# describe. Installed to the user man dir — manpath puts
# ~/.local/share/man ahead of /usr/local/man, so these shadow any
# stale system copies without needing root.
MAN_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/man"
GENESIS_DATA="${XDG_DATA_HOME:-$HOME/.local/share}/genesis"
if [ -d "$REPO/docs/man" ]; then
    for page in "$REPO"/docs/man/*.[0-9]; do
        [ -e "$page" ] || continue
        section="${page##*.}"
        mkdir -p "$MAN_HOME/man$section"
        if [ -d "$GENESIS_DATA" ]; then
            # She exists — her man pages are hers. A changed page waits
            # in her man_review queue (drift-scan picks it up at boot);
            # nothing edits her manual without her say.
            if ! cmp -s "$page" "$MAN_HOME/man$section/$(basename "$page")"; then
                say "Man page change staged for her review: $(basename "$page")"
            fi
        else
            cp "$page" "$MAN_HOME/man$section/"
        fi
    done
    say "Man pages installed to $MAN_HOME (man genesis, man genesis-daemon, ...)"
fi

# ── 4. systemd user service ─────────────────────────────────────────
if [ "$want_service" -eq 1 ] && command -v systemctl >/dev/null 2>&1; then
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT" <<EOF
[Unit]
Description=Genesis — resident cognitive agent (daemon + mind, headless)

[Service]
Type=simple
ExecStart=$VENV/bin/genesis --headless
Restart=on-failure
RestartSec=10s
# An external SIGTERM (genesis --stop, run.sh --stop, kill) is a
# deliberate graceful stop, not a failure — without this, systemd
# would resurrect her every time she is put down from outside.
SuccessExitStatus=SIGTERM
# SIGTERM flips her shutdown flag; a mature data dir takes seconds to
# save, so give teardown room before SIGKILL.
TimeoutStopSec=90s
EnvironmentFile=-$AGENT_CONF/agent.env

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable genesis.service
    say "systemd user unit installed and enabled: $UNIT"
    if ! loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q "yes"; then
        loginctl enable-linger "$USER" 2>/dev/null \
            && say "linger enabled — she starts at boot, not just login" \
            || say "NOTE: run 'loginctl enable-linger $USER' to start her at boot"
    fi
else
    say "Skipping systemd unit (--no-service or no systemctl)"
fi

cat <<EOF

Genesis is installed.

  genesis              interactive session (terminal)
  genesis --headless   resident mode, no prompt
  genesis --stop       stop the processes for this data dir
  genesis --tail       tail the daemon log
EOF
if [ "$want_service" -eq 1 ]; then
    cat <<EOF
  systemctl --user start genesis    wake her now (resident)
  systemctl --user stop  genesis    put her down
  journalctl --user -u genesis -f   watch her logs

Her mind is the same one — the data dir resolves to
~/.local/share/genesis, unchanged. The resident service and an
interactive session can't run at once; the CLI lock refuses the
second.

Optional — privileged body levers (CPU governor, RTC wake alarm):
  sudo bash $REPO/scripts/install_sudoers.sh
This installs root-owned helper copies under /usr/local/libexec/genesis
plus a sudoers rule for them. Without it, sensing still works — only
the actuation levers stay read-only.
EOF
fi
