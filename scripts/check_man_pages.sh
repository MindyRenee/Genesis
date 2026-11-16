#!/bin/bash
# Man-page audit — verifies every checkable claim docs/man makes
# against the repository, so the pages cannot silently drift the
# way they did before (stale paths, retired binaries, missing
# commands). Run manually or from scripts/pre-commit.sh.
#
# Checks, per page under docs/man/:
#   1. roff syntax         — groff -z must not warn
#   2. file paths          — every claimed source path must exist
#   3. GENESIS_* env vars  — each must appear in the source tree
#   4. binaries            — genesis-* executables must be declared
#   5. slash commands      — _SELF_COMMANDS keys must be documented
#
# Dynamic paths ($DATA_DIR, /dev/..., ~, URLs, version strings) are
# skipped — only repository-relative claims are verified.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MAN_DIR="docs/man"
fail=0

note() { echo "  $1"; fail=1; }

echo "man-page audit: $MAN_DIR"

# ─── 1. roff syntax ────────────────────────────────────────────
if command -v groff >/dev/null 2>&1; then
    for page in "$MAN_DIR"/*.[0-9]; do
        if ! groff -man -Tutf8 -z "$page" 2>/tmp/_man_groff_err; then
            note "groff error in $page: $(head -1 /tmp/_man_groff_err)"
        elif [ -s /tmp/_man_groff_err ]; then
            note "groff warning in $page: $(head -1 /tmp/_man_groff_err)"
        fi
    done
else
    echo "  (groff not installed — skipping syntax check)"
fi

# ─── 2. claimed file paths exist ───────────────────────────────
# Paths appear as \fIpath\fR, in .I blocks, or bare. Restrict to
# repo-looking roots to avoid devices, env paths, and URLs — and
# strip URLs entirely first so kaggle.com/docs/api never parses
# as a repo path.
for page in "$MAN_DIR"/*.[0-9]; do
    sed 's|https\?://[^ \t]*||g' "$page" \
        | grep -oE '(Self_World|Inner_World|Outer_World|scripts|tests|docs|src|tools|python)/[A-Za-z0-9_./-]+' \
        | sort -u | while read -r claimed; do
        claimed="${claimed%./}"; claimed="${claimed%.}"; claimed="${claimed%,}"
        [ -e "$claimed" ] || echo "MISSING_PATH|$page|$claimed"
    done > /tmp/_man_missing
    while IFS='|' read -r _ pg path; do
        note "$pg references missing path: $path"
    done < /tmp/_man_missing
done

# ─── 3. GENESIS_* env vars exist in source ─────────────────────
for page in "$MAN_DIR"/*.[0-9]; do
    grep -oE 'GENESIS_[A-Z_]+' "$page" | sort -u | while read -r var; do
        [ "$var" = "GENESIS_WAKE" ] || true
        grep -rq "$var" Self_World/ Outer_World/ Inner_World/ scripts/ Cargo.toml genesis 2>/dev/null \
            || echo "MISSING_ENV|$page|$var"
    done > /tmp/_man_env
    while IFS='|' read -r _ pg var; do
        note "$pg references env var with no source use: $var"
    done < /tmp/_man_env
done

# ─── 4. binaries are declared ──────────────────────────────────
for page in "$MAN_DIR"/*.[0-9]; do
    grep -oE '\bgenesis-(daemon|retina|relay)\b' "$page" | sort -u | while read -r bin; do
        grep -q "name = \"$bin\"" Cargo.toml \
            || echo "MISSING_BIN|$page|$bin"
    done > /tmp/_man_bin
    while IFS='|' read -r _ pg bin; do
        note "$pg references undeclared binary: $bin"
    done < /tmp/_man_bin
done

# ─── 5. slash commands are documented ──────────────────────────
# Authoritative list: _SELF_COMMANDS in Self_World/mind.py plus the
# CLI dispatch table in genesis_cli.py. Undocumented real commands
# are drift; documented-but-missing ones are stale docs.
CLI_PAGE="$MAN_DIR/genesis-cli.1"
if [ -f "$CLI_PAGE" ]; then
    python3 - "$CLI_PAGE" <<'PY' > /tmp/_man_cmds
import re, sys
page = open(sys.argv[1]).read()
# Documented = slash tokens on .B/.BR command-name lines only —
# prose and file paths can contain slashes legitimately.
documented = set()
for line in page.split('\n'):
    if re.match(r'^\.B[R]?\s+', line):
        documented.update(re.findall(r'"(/[a-z][a-z0-9-]*)"|(/[a-z][a-z0-9-]+)', line))
documented = {d for tup in documented for d in tup if d}

real = set()
# _SELF_COMMANDS keys — the self-invoked vocabulary.
src = open('Self_World/mind.py').read()
m = re.search(r'_SELF_COMMANDS[^=]*=\s*\{(.*?)\}', src, re.S)
if m:
    real.update(re.findall(r'["\'](/[a-z][a-z0-9-]*)["\']', m.group(1)))
cli = open('Self_World/genesis_cli.py').read()
# Dispatch-table keys: "/cmd": _cmd_x
real.update(re.findall(r'["\'](/[a-z][a-z0-9-]*)["\']\s*:\s*_cmd_', cli))
# Conditional literals — only on command-handling lines, so path
# literals like "/proc" and "/proc/{pid}/cmdline" stay excluded.
for line in cli.split('\n'):
    if re.search(r'\b(cmd|command|user_input)\b', line):
        real.update(re.findall(r'["\'](/[a-z][a-z0-9-]*)["\']', line))
# Aliases documented implicitly by their primary command line.
ALIASES = {"/q": "/quit", "/exit": "/quit"}
for a, primary in ALIASES.items():
    if primary in documented:
        documented.add(a)
for cmd in sorted(real - documented):
    print(f"UNDOCUMENTED|{cmd}")
for cmd in sorted(documented - real - {"/help"}):
    print(f"STALE|{cmd}")
PY
    while IFS='|' read -r kind cmd; do
        case "$kind" in
            UNDOCUMENTED) note "$CLI_PAGE: real command undocumented: $cmd" ;;
            STALE)        note "$CLI_PAGE: documents nonexistent command: $cmd" ;;
        esac
    done < /tmp/_man_cmds
fi

if [ "$fail" -eq 0 ]; then
    echo "man-page audit: clean"
else
    echo "man-page audit: drift found — fix or update docs/man/*" >&2
fi
exit "$fail"
