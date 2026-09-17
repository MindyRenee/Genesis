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

# Resolve repo root from script location (no git dependency).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── Rust ──────────────────────────────────────────────────────
if cargo metadata --no-deps --format-version 1 >/dev/null 2>&1; then
    clippy_out="$(cargo clippy --all-targets 2>&1)"
    if echo "$clippy_out" | grep -qE "^warning|^error"; then
        echo "❌ cargo clippy found issues. Fix them before committing."
        echo "$clippy_out" | grep -E "^warning|^error"
        exit 1
    fi
fi

# ─── Python ─────────────────────────────────────────────────────
# Scope covers production code, tests, evals, and scripts — ruff.toml
# at the repo root carries the same rule set for evals/scripts as
# python/pyproject.toml does for python/.
if command -v ruff >/dev/null 2>&1; then
    ruff_out="$(ruff check python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ evals/ scripts/ 2>&1)" || true
    if echo "$ruff_out" | grep -qE "^Found|error"; then
        echo "❌ ruff found issues. Run: ruff check --fix python/ evals/ scripts/"
        echo "$ruff_out" | grep -E "^Found|error"
        exit 1
    fi
fi

if command -v pyflakes >/dev/null 2>&1; then
    pyflakes_out="$(pyflakes python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ evals/*.py scripts/*.py 2>&1)" || true
    if [ -n "$pyflakes_out" ]; then
        echo "❌ pyflakes found issues."
        echo "$pyflakes_out"
        exit 1
    fi
fi

if command -v mypy >/dev/null 2>&1; then
    mypy_out="$(cd python && mypy genesis_cognitive/ genesis_client/ genesis_cli.py tests/ --ignore-missing-imports 2>&1)" || true
    if echo "$mypy_out" | grep -qE "^Found .* error"; then
        echo "❌ mypy found type errors."
        echo "$mypy_out" | grep -E "^Found|error"
        exit 1
    fi
fi

echo "✓ All checks passed."
exit 0
