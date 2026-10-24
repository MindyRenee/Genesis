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
#   scripts/verify_change.sh [python_dir] [rust_dir]
#
# Arguments:
#   python_dir   — path to the Python source (default: python)
#   rust_dir     — path to the Rust crate root (default: .)

set -euo pipefail

PYTHON_DIR="${1:-python}"
RUST_DIR="${2:-.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${YELLOW}[verify]${NC} $*"; }
pass() { echo -e "${GREEN}[verify] ✓${NC} $*"; }
fail() { echo -e "${RED}[verify] ✗${NC} $*" >&2; }

# ─── 1. Python compile check ─────────────────────────────────────
log "Step 1: py_compile check"
# Compile all Python source — the region packages, genesis_client,
# the CLI, and the test suite. A syntax error in any of these can
# break Genesis at runtime even if the tests happen to pass.
#
# Run find from inside PYTHON_DIR: the '*/.*' exclusion is meant to
# skip hidden dirs inside the tree (.git, caches), but when
# PYTHON_DIR is absolute and itself lives under a dotfile parent —
# e.g. ~/.local/share/genesis/workspace/python — every emitted path
# contains '/.' and the filter silently drops ALL files. py_compile
# then ran with no filenames and failed every experiment. The -r
# keeps an empty list from invoking py_compile bare, which exits
# non-zero with only a usage error.
if ! (cd "$PYTHON_DIR" && find . -name '*.py' -not -path '*__pycache__*' -not -path '*/.*' -print0 \
    | xargs -0 -r python3 -m py_compile); then
    fail "py_compile failed"
    exit 1
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
# The support packages (genesis_client, infrastructure, eval) live
# BESIDE python/ in the tree, so the tree root must be on PYTHONPATH —
# python/tests/conftest.py arranges the same thing for the pytest gate.
# Without it, every module that imports a sibling package fails here.
TREE_ROOT="$(cd "$(dirname "$PYTHON_DIR")" && pwd)"
if ! (
    cd "$PYTHON_DIR" && PYTHONPATH="$TREE_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PYEOF'
import importlib
import pkgutil
import sys
import traceback

failed = []
import pathlib
for pkg_name in sorted(
    d.name
    for d in pathlib.Path(".").iterdir()
    if d.is_dir() and (d / "__init__.py").exists()
):
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
# The CLI is a top-level module, not part of a package.
try:
    importlib.import_module("genesis_cli")
except Exception:
    traceback.print_exc()
    failed.append("genesis_cli")
if failed:
    print(f"IMPORT FAILURES: {', '.join(failed)}")
    sys.exit(1)
PYEOF
); then
    fail "import check failed — a module raises at import time"
    exit 1
fi
pass "import check OK"

# ─── 2. Python test suite ────────────────────────────────────────
log "Step 2: Python test suite (pytest)"
if ! (cd "$PYTHON_DIR" && python3 -m pytest tests/ -q -o addopts=''); then
    fail "pytest failed"
    exit 2
fi
pass "pytest OK"

# ─── 3. Rust test suite ──────────────────────────────────────────
log "Step 3: Rust test suite (cargo test)"
if ! (cd "$RUST_DIR" && cargo test -q 2>&1); then
    fail "cargo test failed"
    exit 3
fi
pass "cargo test OK"

# ─── All checks passed ────────────────────────────────────────────
pass "All verification checks passed"
exit 0
