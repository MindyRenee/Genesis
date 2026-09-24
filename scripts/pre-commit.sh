#!/bin/bash
# Pre-commit hook — prevents lint violations from entering the repo.
#
# Run manually:  bash scripts/pre-commit.sh
# Or install as a git hook if using git:
#   cp scripts/pre-commit.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
#
# Checks:
#   1. Rust: cargo clippy (warnings are errors)
#   2. Python: ruff check + pyflakes + mypy
#
# Staged files only would be ideal, but the substrate is small enough
# that full-repo checks run in under 5 seconds.

set -euo pipefail

# Resolve repo root. When run from scripts/, the parent dir is the root.
# When installed as .git/hooks/pre-commit, the parent is .git/ — ask git.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../Cargo.toml" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_ROOT="$(git rev-parse --show-toplevel)"
fi
cd "$REPO_ROOT"

# ─── Rust ──────────────────────────────────────────────────────
if command -v cargo >/dev/null 2>&1 && cargo metadata --no-deps --format-version 1 >/dev/null 2>&1; then
    clippy_out="$(cargo clippy --all-targets 2>&1)" || true
    if echo "$clippy_out" | grep -qE "^warning|^error"; then
        echo "❌ cargo clippy found issues. Fix them before committing."
        echo "$clippy_out" | grep -E "^warning|^error"
        exit 1
    fi
else
    echo "⚠ cargo not found — clippy check skipped."
fi

# ─── Python ─────────────────────────────────────────────────────
# These invocations mirror the toolchain documented in AGENTS.md.
# A missing tool warns loudly instead of silently passing — a skipped
# check is a false pass.
PY_DEPS="python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/"

if command -v ruff >/dev/null 2>&1; then
    if ! ruff_out="$(ruff check 2>&1)"; then
        echo "❌ ruff found issues. Run: ruff check --fix"
        echo "$ruff_out"
        exit 1
    fi
else
    echo "⚠ ruff not found — lint check skipped. Install: pip install -r python/requirements-dev.txt"
fi

if python3 -m pyflakes --version >/dev/null 2>&1; then
    if ! pyflakes_out="$(python3 -m pyflakes $PY_DEPS scripts/*.py 2>&1)"; then
        echo "❌ pyflakes found issues."
        echo "$pyflakes_out"
        exit 1
    fi
else
    echo "⚠ pyflakes not found — check skipped. Install: pip install -r python/requirements-dev.txt"
fi

if python3 -m mypy --version >/dev/null 2>&1; then
    if ! mypy_out="$(python3 -m mypy $PY_DEPS --ignore-missing-imports 2>&1)"; then
        echo "❌ mypy found type errors."
        echo "$mypy_out"
        exit 1
    fi
else
    echo "⚠ mypy not found — check skipped. Install: pip install -r python/requirements-dev.txt"
fi

echo "✓ All checks passed."
exit 0
