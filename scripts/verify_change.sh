#!/usr/bin/env bash
# verify_change.sh — test gate for self-improvement changes.
#
# This script is the regression gate for Genesis's verified self-improvement
# loop (workstream D). Before any heuristic change is committed, this script
# must pass. It runs:
#
#   1. py_compile on all modified Python files
#   1b. Import check — executes each production module to catch
#       module-level failures that py_compile cannot detect (e.g.
#       an invalid re.compile at module scope)
#   2. The full Python test suite (pytest)
#   3. The Rust test suite (cargo test)
#
# If any step fails, the script exits with a non-zero code, and the
# self-improvement engine reverts the change.
#
# Usage:
#   scripts/verify_change.sh [project_root] [rust_dir] [arm]
#
# Arguments:
#   project_root — the repository root holding the three worlds
#                  (Self_World, Inner_World,
#                  Outer_World). Defaults to the directory
#                  containing this script's parent.
#   rust_dir      — path to the Rust crate root (default: project_root)
#   arm           — which half of the gate to run: "python" (py_compile,
#                   import check, pytest), "rust" (cargo test), or "all"
#                   (default). A Python-only edit cannot regress the crate
#                   and vice versa — running the irrelevant arm once
#                   reverted every proposal on a toolchain problem that
#                   had nothing to do with the change.
#
# The whole gate is serialized by a flock on $ROOT/.verify.lock: two
# concurrent runs fight over the same ports, sockets, and CPU, and a
# losing race once read as "tests failed" and reverted good work. A
# run that cannot take the lock within GENESIS_VERIFY_LOCK_WAIT_S
# (default 3600) exits 4 without touching the tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUST_DIR="${2:-$ROOT}"
ARM="${3:-all}"

case "$ARM" in
    python|rust|all) ;;
    *) echo "verify_change.sh: unknown arm '$ARM' (expected python|rust|all)" >&2; exit 2 ;;
esac

# Serialize the gate — one verification run at a time per tree.
exec 9>"$ROOT/.verify.lock"
if ! flock -w "${GENESIS_VERIFY_LOCK_WAIT_S:-3600}" 9; then
    echo "[verify] x could not take the verify lock — another run is still in progress" >&2
    exit 4
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${YELLOW}[verify]${NC} $*"; }
pass() { echo -e "${GREEN}[verify] ✓${NC} $*"; }
fail() { echo -e "${RED}[verify] ✗${NC} $*" >&2; }

# The three worlds are the package roots — every importable package
# lives inside one of them. A layout without world dirs (an installed
# tree, an older checkout) falls back to treating ROOT itself as the
# single package root.
WORLDS=()
for w in Self_World Inner_World Outer_World; do
    [ -d "$ROOT/$w" ] && WORLDS+=("$ROOT/$w")
done
if [ "${#WORLDS[@]}" -eq 0 ]; then
    WORLDS=("$ROOT")
fi

# PYTHONPATH for every Python invocation below: the worlds first so
# their packages resolve, then the root for loose root modules.
PYTHONPATH_JOINED="$(IFS=:; echo "${WORLDS[*]}"):$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$PYTHONPATH_JOINED"

# ─── Python arm (arm=python|all) ────────────────────────────────
if [ "$ARM" != "rust" ]; then

# The gate must run under an interpreter that can actually run the
# suite — a bare `python3` can be a system interpreter with no pytest
# and none of the project's deps, which fails every experiment on a
# false negative (exit 2, "No module named pytest"). Order: the
# caller's interpreter (GENESIS_PYTHON, the mind passes its own
# sys.executable), the checkout's venv, then PATH's python3 — and the
# first candidate that can import pytest wins.
PY=""
for cand in "${GENESIS_PYTHON:-}" "$ROOT/.venv/bin/python" python3; do
    [ -n "$cand" ] || continue
    if "$cand" -c 'import pytest' 2>/dev/null; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    fail "no Python with pytest found (tried GENESIS_PYTHON, $ROOT/.venv, python3)"
    exit 2
fi

# ─── 1. Python compile check ─────────────────────────────────────
log "Step 1: py_compile check"
# Compile all Python source — the world packages, genesis_client,
# the CLI, and the test suite. A syntax error in any of these can
# break Genesis at runtime even if the tests happen to pass.
#
# Run find inside each world dir: the '*/.*' exclusion is meant to
# skip hidden dirs inside the tree (.git, caches), but when a dir is
# absolute and itself lives under a dotfile parent — e.g.
# ~/.local/share/genesis/workspace/Self_World — every emitted path
# contains '/.' and the filter silently drops ALL files. py_compile
# then ran with no filenames and failed every experiment. The -r
# keeps an empty list from invoking py_compile bare, which exits
# non-zero with only a usage error.
PY_FILES=()
while IFS= read -r -d '' f; do
    PY_FILES+=("$f")
done < <(find "${WORLDS[@]}" -name '*.py' -not -path '*__pycache__*' -not -path '*/.*' -print0)
if [ "${#PY_FILES[@]}" -gt 0 ]; then
    if ! "$PY" -m py_compile "${PY_FILES[@]}"; then
        fail "py_compile failed"
        exit 1
    fi
fi
pass "py_compile OK"

# ─── 1b. Import check (executes module-level code) ───────────────
log "Step 1b: import check"
# py_compile only validates syntax — it does not execute the module.
# A file that raises at import time (e.g. an invalid re.compile at
# module scope, or a missing module-level name) passes py_compile but
# breaks every importer. Import each package module in a subprocess to
# catch this early, before the slow test gate. Optional dependencies
# (vosk, sounddevice) are lazily imported inside functions, so a
# blanket import here cannot fail on machines without them.
#
# The worlds are the package roots, so every package inside them is
# walked; genesis_cli is a loose module inside Self_World, not part
# of a package.
if ! "$PY" - "$ROOT" "${WORLDS[@]}" <<'PYEOF'
import importlib
import pathlib
import pkgutil
import sys
import traceback

root = pathlib.Path(sys.argv[1])
worlds = [pathlib.Path(p) for p in sys.argv[2:]]

failed = []
for world in worlds:
    for d in sorted(world.iterdir()):
        if not (d.is_dir() and (d / "__init__.py").exists()):
            continue
        pkg_name = d.name
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:
            traceback.print_exc()
            failed.append(pkg_name)
            continue
        for mod_info in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + "."):
            try:
                importlib.import_module(mod_info.name)
            except Exception:
                traceback.print_exc()
                failed.append(mod_info.name)
# The CLI is a loose module at a world's root, not part of a package.
for candidate in (root / "Self_World" / "genesis_cli.py", root / "genesis_cli.py"):
    if candidate.exists():
        try:
            importlib.import_module("genesis_cli")
        except Exception:
            traceback.print_exc()
            failed.append("genesis_cli")
        break
if failed:
    print(f"IMPORT FAILURES: {', '.join(failed)}")
    sys.exit(1)
PYEOF
then
    fail "import check failed — a module raises at import time"
    exit 1
fi
pass "import check OK"

# ─── 2. Python test suite ────────────────────────────────────────
log "Step 2: Python test suite (pytest)"
if ! (cd "$ROOT" && "$PY" -m pytest tests/python -q -o addopts=''); then
    fail "pytest failed"
    exit 2
fi
pass "pytest OK"

fi  # end Python arm

# ─── 3. Rust test suite (arm=rust|all) ───────────────────────────
if [ "$ARM" != "python" ]; then

log "Step 3: Rust test suite (cargo test)"

# The ambient PATH is not trustworthy here: under systemd it once
# resolved `cargo` to the system 1.75, which cannot parse
# `edition2024`, and every experiment was reverted on a toolchain
# error misreported as "tests failed". Prefer the rustup toolchain
# in ~/.cargo/bin, then whatever PATH offers.
CARGO="cargo"
if [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO="$HOME/.cargo/bin/cargo"
fi

# bindgen (v4l2-sys-mit) needs libclang — cargo-env.sh locates it via
# llvm-config and sets CC/LIBCLANG_PATH. Not fatal if absent: the
# cargo build either works without it or fails honestly below.
if [ -f "$SCRIPT_DIR/cargo-env.sh" ]; then
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/cargo-env.sh" || log "cargo-env.sh could not be sourced — continuing"
fi

# --release reuses the warm target/release artifacts and matches the
# binaries she actually runs; a cold dev-profile build can exceed the
# experiment timeout on its own.
if ! (cd "$RUST_DIR" && "$CARGO" test --release -q 2>&1); then
    fail "cargo test failed"
    exit 3
fi
pass "cargo test OK"

fi  # end Rust arm

# ─── All checks passed ────────────────────────────────────────────
pass "All verification checks passed"
exit 0
