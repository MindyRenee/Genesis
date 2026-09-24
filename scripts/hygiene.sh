#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# hygiene.sh — Genesis project homeostasis system
#
# Mirrors Genesis's own architecture: interoception (measure) →
# regulation (act).  The project's physical size has a negative
# feedback loop instead of growing unboundedly.
#
# Every byte is classified as either:
#   SOURCE   — irreplaceable (code, configs, binary assets, docs)
#   ARTIFACT — disposable / regenerable (build output, caches, bytecode)
#
# Two locations are tracked:
#   PROJECT  — the source tree (cwd, where this script lives)
#   RUNTIME  — ~/.local/share/genesis/ (live state + runtime artifacts)
#
# Usage:
#   scripts/hygiene.sh                  # report (default — measure only)
#   scripts/hygiene.sh --report         # size breakdown: source vs. artifacts
#   scripts/hygiene.sh --check          # detect dead code + stale artifacts
#   scripts/hygiene.sh --clean          # remove all disposable artifacts
#   scripts/hygiene.sh --clean-cache    # remove only caches + bytecode
#   scripts/hygiene.sh --clean-debug    # remove only the debug build
#   scripts/hygiene.sh --clean-runtime  # remove orphaned runtime backups
#   scripts/hygiene.sh --help
#
# --clean-cache / --clean-debug / --clean-runtime are safe: they never
# touch source, configs, binary assets, live runtime state, or the
# release binary run.sh depends on. --clean (all) DOES remove
# target/release (regenerable via `cargo build --release`, and run.sh
# auto-rebuilds it) — use --clean-debug to preserve the release build.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Runtime data directory (where Genesis stores its live state + artifacts).
# Same resolution as run.sh: XDG_DATA_HOME (or ~/.local/share), with
# GENESIS_DATA_DIR kept as a manual override.
RUNTIME_DIR="${GENESIS_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/genesis}"

# ── Colors ────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'; CYAN=$'\033[0;36m'; DIM=$'\033[2m'
    BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; DIM=''; BOLD=''; NC=''
fi

# ── Helpers ───────────────────────────────────────────────────────────
bytes_to_human() {
    local b=$1
    if command -v bc >/dev/null 2>&1; then
        if   [ "$b" -ge 1073741824 ]; then printf "%.1fG" "$(echo "scale=1; $b/1073741824" | bc)"
        elif [ "$b" -ge 1048576 ];    then printf "%.1fM" "$(echo "scale=1; $b/1048576" | bc)"
        elif [ "$b" -ge 1024 ];       then printf "%.1fK" "$(echo "scale=1; $b/1024" | bc)"
        else                               printf "%dB" "$b"
        fi
    else
        # No bc — integer fallback, no decimals.
        if   [ "$b" -ge 1073741824 ]; then printf "%dG" "$(( b / 1073741824 ))"
        elif [ "$b" -ge 1048576 ];    then printf "%dM" "$(( b / 1048576 ))"
        elif [ "$b" -ge 1024 ];       then printf "%dK" "$(( b / 1024 ))"
        else                               printf "%dB" "$b"
        fi
    fi
}

dir_bytes() {
    # Total bytes of a directory (or 0 if it doesn't exist)
    [ -d "$1" ] && du -sb "$1" 2>/dev/null | cut -f1 || echo 0
}

print_bar() {
    # $1 = percentage (0-100), width = 30 chars
    local pct=$1 width=30
    [ "$pct" -lt 0 ] 2>/dev/null && pct=0
    [ "$pct" -gt 100 ] 2>/dev/null && pct=100
    local filled=$(( pct * width / 100 ))
    [ "$filled" -gt "$width" ] && filled=$width
    [ "$filled" -lt 0 ] && filled=0
    local empty=$(( width - filled ))
    printf "${DIM}[${GREEN}"
    [ "$filled" -gt 0 ] && printf '%0.s█' $(seq 1 "$filled" 2>/dev/null)
    printf "${DIM}"
    [ "$empty" -gt 0 ] && printf '%0.s·' $(seq 1 "$empty" 2>/dev/null)
    printf "]${NC}"
}

# ── Size model ────────────────────────────────────────────────────────
# The canonical list of disposable artifact directories.  This is the
# single source of truth — .gitignore mirrors it, this script enforces it.
ARTIFACT_DIRS=(
    "target/debug"
    "target/release"
    "target/doc"
    "target/tmp"
    ".mypy_cache"
    ".pytest_cache"
    ".ruff_cache"
)

# Source directories — never touched by --clean
SOURCE_DIRS=(
    "src"
    "python/genesis_cognitive"
    "python/genesis_client"
    "python/genesis_cli.py"
    "python/tests"
    "tests"
    "examples"
    "scripts"
    "docs"
)

# ── Report mode ───────────────────────────────────────────────────────
do_report() {
    echo ""
    echo "${BOLD}Genesis — Project Homeostasis Report${NC}"
    echo "${DIM}$(date)${NC}"
    echo ""

    # ── Artifact breakdown ──
    echo "${BOLD}Disposable artifacts (safe to delete)${NC}"
    echo "${DIM}─────────────────────────────────────────────────${NC}"

    local artifact_total=0
    for dir in "${ARTIFACT_DIRS[@]}"; do
        local b
        b=$(dir_bytes "$dir")
        artifact_total=$(( artifact_total + b ))
        if [ "$b" -gt 0 ]; then
            printf "  ${YELLOW}%-22s${NC} %8s\n" "$dir" "$(bytes_to_human "$b")"
        fi
    done

    # __pycache__ directories (scattered, counted collectively)
    local pycache_total=0
    while IFS= read -r d; do
        pycache_total=$(( pycache_total + $(du -sb "$d" 2>/dev/null | cut -f1) ))
    done < <(find . -name "__pycache__" -type d -not -path "*/target/*" 2>/dev/null)
    artifact_total=$(( artifact_total + pycache_total ))
    if [ "$pycache_total" -gt 0 ]; then
        printf "  ${YELLOW}%-22s${NC} %8s\n" "__pycache__ (all)" "$(bytes_to_human "$pycache_total")"
    fi

    # *.pyc orphans (bytecode with no source — genuinely dead)
    local orphan_bytes=0
    orphan_bytes=$(count_orphan_pyc_bytes)
    if [ "$orphan_bytes" -gt 0 ]; then
        printf "  ${RED}%-22s${NC} %8s  ${DIM}orphan .pyc (no source)${NC}\n" "dead bytecode" "$(bytes_to_human "$orphan_bytes")"
    fi

    printf "  ${DIM}─────────────────────────────────────────${NC}\n"
    printf "  ${BOLD}%-22s${NC} %8s\n" "ARTIFACT TOTAL" "$(bytes_to_human "$artifact_total")"
    echo ""

    # ── Source breakdown ──
    echo "${BOLD}Source (irreplaceable)${NC}"
    echo "${DIM}─────────────────────────────────────────────────${NC}"

    local source_total=0
    for item in "${SOURCE_DIRS[@]}"; do
        local b=0
        if [ -d "$item" ]; then
            b=$(dir_bytes "$item")
        elif [ -f "$item" ]; then
            b=$(stat -c%s "$item" 2>/dev/null || echo 0)
        fi
        source_total=$(( source_total + b ))
        if [ "$b" -gt 0 ]; then
            printf "  ${GREEN}%-22s${NC} %8s\n" "$item" "$(bytes_to_human "$b")"
        fi
    done

    # Top-level files (Cargo.toml, AGENTS.md, run.sh, etc.)
    local top_files=0
    for f in Cargo.toml Cargo.lock AGENTS.md README.md LICENSE run.sh ruff.toml .gitignore; do
        if [ -f "$f" ]; then
            local fb
            fb=$(stat -c%s "$f" 2>/dev/null || echo 0)
            top_files=$(( top_files + fb ))
        fi
    done
    source_total=$(( source_total + top_files ))
    printf "  ${GREEN}%-22s${NC} %8s  ${DIM}(config + docs)${NC}\n" "root files" "$(bytes_to_human "$top_files")"

    printf "  ${DIM}─────────────────────────────────────────${NC}\n"
    printf "  ${BOLD}%-22s${NC} %8s\n" "SOURCE TOTAL" "$(bytes_to_human "$source_total")"
    echo ""

    # ── Summary ──
    local grand_total=$(( artifact_total + source_total ))
    local artifact_pct=0 source_pct=0
    if [ "$grand_total" -gt 0 ]; then
        artifact_pct=$(( artifact_total * 100 / grand_total ))
        source_pct=$(( source_total * 100 / grand_total ))
    fi

    echo "${BOLD}Summary${NC}"
    echo "${DIM}─────────────────────────────────────────────────${NC}"
    printf "  Total project size:   %s\n" "$(bytes_to_human "$grand_total")"
    printf "  Artifacts (disposable): %s%%  " "$artifact_pct"
    print_bar "$artifact_pct"
    echo ""
    printf "  Source (irreplaceable):  %s%%  " "$source_pct"
    print_bar "$source_pct"
    echo ""
    echo ""

    # ── Recommendations ──
    if [ "$artifact_total" -gt 0 ]; then
        echo "${BOLD}Reclaimable space${NC}"
        echo "${DIM}─────────────────────────────────────────────────${NC}"

        local debug_b
        debug_b=$(dir_bytes "target/debug")
        if [ "$debug_b" -gt 1048576 ]; then
            echo "  ${CYAN}target/debug${NC} is $(bytes_to_human "$debug_b") — run.sh only uses release."
            echo "       Run: ${BOLD}scripts/hygiene.sh --clean-debug${NC}"
        fi

        local inc_b
        inc_b=$(dir_bytes "target/debug/incremental")
        if [ "$inc_b" -gt 104857600 ]; then
            echo "  ${CYAN}incremental cache${NC} is $(bytes_to_human "$inc_b") — stale compilation state."
            echo "       Run: ${BOLD}scripts/hygiene.sh --clean-debug${NC}"
        fi

        local cache_b=$(( pycache_total + $(dir_bytes ".mypy_cache") + $(dir_bytes ".pytest_cache") + $(dir_bytes ".ruff_cache") ))
        if [ "$cache_b" -gt 1048576 ]; then
            echo "  ${CYAN}caches + bytecode${NC} total $(bytes_to_human "$cache_b") — regenerable on next run."
            echo "       Run: ${BOLD}scripts/hygiene.sh --clean-cache${NC}"
        fi

        if [ "$orphan_bytes" -gt 0 ]; then
            echo "  ${RED}orphan .pyc files${NC} total $(bytes_to_human "$orphan_bytes") — bytecode with no source."
            echo "       Run: ${BOLD}scripts/hygiene.sh --clean-cache${NC}"
        fi

        echo ""
        echo "  Full cleanup: ${BOLD}scripts/hygiene.sh --clean${NC} (frees $(bytes_to_human "$artifact_total"))"
        echo ""
    fi

    # ── Runtime data directory ──
    if [ -d "$RUNTIME_DIR" ]; then
        echo "${BOLD}Runtime data ($RUNTIME_DIR)${NC}"
        echo "${DIM}─────────────────────────────────────────────────${NC}"

        local rt_total=0
        rt_total=$(dir_bytes "$RUNTIME_DIR")

        # Live state (irreplaceable)
        local rt_live=0
        for item in \
            cognitive_state.json \
            concept_archive.db concept_archive.db-shm concept_archive.db-wal \
            concept_vectors.npz \
            core_state.bin \
            embeddings.npz \
            holographic_graph.npz \
            inference_model.bin \
            journal.txt \
            ltm_store.bundles ltm_store.dat ltm_store.meta \
            stm_ring.bin \
            user_profile.json \
            vq_codebook.npz \
            bug_reports.jsonl \
            man_page_genus_cache.json \
            drawings/style_profile.json \
            visual_cortex \
            models \
            projects \
            visitor_messages \
            genesis_data \
            learner_cache \
            ; do
            local p="$RUNTIME_DIR/$item"
            local b=0
            if [ -e "$p" ]; then
                if [ -d "$p" ]; then
                    b=$(dir_bytes "$p")
                else
                    b=$(stat -c%s "$p" 2>/dev/null || echo 0)
                fi
            fi
            rt_live=$(( rt_live + b ))
        done
        # Ephemeral runtime endpoints — sockets, PID files, singleton
        # locks, logs (including rotations). Small, but counted so the
        # live total reconciles with du.
        for pattern in "*.sock" "*.pid" "*.lock" "retina.log*"; do
            for p in "$RUNTIME_DIR"/$pattern; do
                [ -e "$p" ] || continue
                b=$(stat -c%s "$p" 2>/dev/null || echo 0)
                rt_live=$(( rt_live + b ))
            done
        done

        # Drawings (creative output — live, but unbounded growth)
        local drawings_b=0 drawings_count=0
        if [ -d "$RUNTIME_DIR/drawings" ]; then
            drawings_b=$(dir_bytes "$RUNTIME_DIR/drawings")
            drawings_count=$(find "$RUNTIME_DIR/drawings" -name "drawing_*.webp" 2>/dev/null | wc -l)
        fi
        rt_live=$(( rt_live + drawings_b ))

        # Source cache (regenerable — fetched from online sources)
        local source_cache_b=0
        if [ -d "$RUNTIME_DIR/source_cache" ]; then
            source_cache_b=$(dir_bytes "$RUNTIME_DIR/source_cache")
        fi

        # Orphaned backups (stale — not created by any current code path)
        local backup_b=0
        local backup_files=""
        while IFS= read -r f; do
            local fb
            fb=$(stat -c%s "$f" 2>/dev/null || echo 0)
            backup_b=$(( backup_b + fb ))
            backup_files="${backup_files}${f}\n"
        done < <(find "$RUNTIME_DIR" -maxdepth 1 \( \
            -name "*.pre-restore-backup" \
            -o -name "*.pre-cleanup*" \
            -o -name "*.pre-proposal*" \
            -o -name "*.pre-prune-backup*" \
            -o -name "core_state.bin.bak.*" \
            -o -name "*.ltm_store.*.pre-restore-backup" \
            -o -name "ltm_store.*.pre-restore-backup" \
        \) 2>/dev/null)

        # Logs (rotated by CLI, but can grow between rotations)
        local logs_b=0
        for log in daemon.log retina.log; do
            if [ -f "$RUNTIME_DIR/$log" ]; then
                logs_b=$(( logs_b + $(stat -c%s "$RUNTIME_DIR/$log" 2>/dev/null || echo 0) ))
            fi
        done

        # Report
        printf "  ${GREEN}%-22s${NC} %8s  ${DIM}(live state)${NC}\n" "live state" "$(bytes_to_human "$rt_live")"
        if [ "$drawings_count" -gt 0 ]; then
            printf "  ${GREEN}%-22s${NC} %8s  ${DIM}(%d drawings, no rotation)${NC}\n" "  drawings" "$(bytes_to_human "$drawings_b")" "$drawings_count"
        fi
        if [ "$source_cache_b" -gt 0 ]; then
            printf "  ${YELLOW}%-22s${NC} %8s  ${DIM}(regenerable cache)${NC}\n" "source_cache" "$(bytes_to_human "$source_cache_b")"
        fi
        if [ "$backup_b" -gt 0 ]; then
            printf "  ${RED}%-22s${NC} %8s  ${DIM}(orphaned backups)${NC}\n" "stale backups" "$(bytes_to_human "$backup_b")"
        fi
        if [ "$logs_b" -gt 0 ]; then
            printf "  ${YELLOW}%-22s${NC} %8s  ${DIM}(rotated by CLI)${NC}\n" "logs" "$(bytes_to_human "$logs_b")"
        fi
        printf "  ${DIM}─────────────────────────────────────────${NC}\n"
        printf "  ${BOLD}%-22s${NC} %8s\n" "RUNTIME TOTAL" "$(bytes_to_human "$rt_total")"
        echo ""

        # Runtime recommendations
        if [ "$backup_b" -gt 0 ]; then
            echo "  ${RED}orphaned backups${NC} total $(bytes_to_human "$backup_b") — not created by any current code."
            echo "       Run: ${BOLD}scripts/hygiene.sh --clean-runtime${NC}"
        fi
        if [ "$drawings_count" -gt 200 ]; then
            echo "  ${YELLOW}drawings${NC} at $drawings_count files — no rotation in canvas.py, grows unboundedly."
        fi
        if [ "$source_cache_b" -gt 52428800 ]; then  # > 50M
            echo "  ${YELLOW}source_cache${NC} is $(bytes_to_human "$source_cache_b") — no eviction in source_registry.py."
        fi
        echo ""
    fi
}

# ── Orphan .pyc detection ─────────────────────────────────────────────
count_orphan_pyc_bytes() {
    local total=0
    while IFS= read -r pyc; do
        # Reconstruct the expected .py path:
        #   dir/__pycache__/mod.cpython-312.pyc  →  dir/mod.py
        local dir src
        dir=$(dirname "$pyc" | sed 's/__pycache__//')
        src="$dir/$(basename "$pyc" | sed 's/\.cpython-[0-9]*[-a-z0-9.]*\.pyc/.py/')"
        if [ ! -f "$src" ]; then
            total=$(( total + $(stat -c%s "$pyc" 2>/dev/null || echo 0) ))
        fi
    done < <(find . -name "*.pyc" -path "*__pycache__*" -not -path "*/target/*" 2>/dev/null)
    echo "$total"
}

list_orphan_pyc() {
    while IFS= read -r pyc; do
        local dir src
        dir=$(dirname "$pyc" | sed 's/__pycache__//')
        src="$dir/$(basename "$pyc" | sed 's/\.cpython-[0-9]*[-a-z0-9.]*\.pyc/.py/')"
        if [ ! -f "$src" ]; then
            echo "  $pyc"
        fi
    done < <(find . -name "*.pyc" -path "*__pycache__*" -not -path "*/target/*" 2>/dev/null)
}

# ── Check mode ────────────────────────────────────────────────────────
do_check() {
    echo ""
    echo "${BOLD}Genesis — Project Hygiene Check${NC}"
    echo "${DIM}$(date)${NC}"
    echo ""

    local issues=0

    # ── 1. Orphan .pyc files (dead bytecode) ──
    echo "${BOLD}[1] Dead bytecode (orphan .pyc)${NC}"
    local orphans
    orphans=$(list_orphan_pyc)
    if [ -n "$orphans" ]; then
        echo "$orphans" | while read -r f; do echo "  ${RED}✗${NC} $f"; done
        issues=$(( issues + 1 ))
    else
        echo "  ${GREEN}✓${NC} no orphan .pyc files"
    fi
    echo ""

    # ── 2. Rust dead code ──
    echo "${BOLD}[2] Rust dead code (cargo warnings)${NC}"
    local rust_warnings
    # Use cargo check (not build) — it detects the same dead_code warnings
    # without producing a full binary, keeping target/debug small.
    rust_warnings=$(cargo check --message-format=short 2>&1 | grep -E 'warning:.*never (used|read|constructed)|warning:.*is never used|warning:.*dead_code' || true)
    if [ -n "$rust_warnings" ]; then
        echo "$rust_warnings" | while read -r line; do echo "  ${YELLOW}⚠${NC} $line"; done
        issues=$(( issues + 1 ))
    else
        echo "  ${GREEN}✓${NC} no Rust dead_code warnings"
    fi
    echo ""

    # ── 3. Python unused imports / undefined names ──
    echo "${BOLD}[3] Python dead code (ruff + pyflakes)${NC}"
    local py_issues=""
    if command -v ruff >/dev/null 2>&1; then
        local ruff_out
        ruff_out=$(ruff check python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ scripts/ 2>&1 || true)
        if echo "$ruff_out" | grep -qE "F401|F811|F841|Found"; then
            py_issues="${py_issues}$(echo "$ruff_out" | grep -E "F401|F811|F841")"
        fi
    fi
    if command -v pyflakes >/dev/null 2>&1; then
        local pf_out
        pf_out=$(pyflakes python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ scripts/*.py 2>&1 || true)
        if [ -n "$pf_out" ]; then
            py_issues="${py_issues}${pf_out}"
        fi
    fi
    if [ -n "$py_issues" ]; then
        echo "$py_issues" | while read -r line; do echo "  ${YELLOW}⚠${NC} $line"; done
        issues=$(( issues + 1 ))
    else
        echo "  ${GREEN}✓${NC} no Python dead code detected"
    fi
    echo ""

    # ── 4. Oversized artifact directories ──
    echo "${BOLD}[4] Artifact size thresholds${NC}"
    local debug_b
    debug_b=$(dir_bytes "target/debug")
    if [ "$debug_b" -gt 536870912 ]; then  # > 512M
        echo "  ${YELLOW}⚠${NC} target/debug is $(bytes_to_human "$debug_b") (run.sh only needs release)"
        issues=$(( issues + 1 ))
    else
        echo "  ${GREEN}✓${NC} target/debug within bounds ($(bytes_to_human "$debug_b"))"
    fi

    local inc_b
    inc_b=$(dir_bytes "target/debug/incremental")
    if [ "$inc_b" -gt 268435456 ]; then  # > 256M
        echo "  ${YELLOW}⚠${NC} incremental cache is $(bytes_to_human "$inc_b") (stale entries accumulating)"
        issues=$(( issues + 1 ))
    else
        echo "  ${GREEN}✓${NC} incremental cache within bounds ($(bytes_to_human "$inc_b"))"
    fi

    local pycache_total=0
    while IFS= read -r d; do
        pycache_total=$(( pycache_total + $(du -sb "$d" 2>/dev/null | cut -f1) ))
    done < <(find . -name "__pycache__" -type d -not -path "*/target/*" 2>/dev/null)
    if [ "$pycache_total" -gt 20971520 ]; then  # > 20M
        echo "  ${YELLOW}⚠${NC} __pycache__ total is $(bytes_to_human "$pycache_total")"
        issues=$(( issues + 1 ))
    else
        echo "  ${GREEN}✓${NC} __pycache__ within bounds ($(bytes_to_human "$pycache_total"))"
    fi
    echo ""

    # ── 5. Runtime orphaned backups ──
    echo "${BOLD}[5] Runtime orphaned backups${NC}"
    if [ -d "$RUNTIME_DIR" ]; then
        local rt_backups
        rt_backups=$(find "$RUNTIME_DIR" -maxdepth 1 \( \
            -name "*.pre-restore-backup" \
            -o -name "*.pre-cleanup*" \
            -o -name "*.pre-proposal*" \
            -o -name "*.pre-prune-backup*" \
            -o -name "core_state.bin.bak.*" \
        \) 2>/dev/null)
        if [ -n "$rt_backups" ]; then
            echo "$rt_backups" | while read -r f; do
                local b
                b=$(stat -c%s "$f" 2>/dev/null || echo 0)
                echo "  ${RED}✗${NC} $(basename "$f") ($(bytes_to_human "$b"))"
            done
            issues=$(( issues + 1 ))
        else
            echo "  ${GREEN}✓${NC} no orphaned runtime backups"
        fi

        # Drawings growth check
        local drawings_count=0
        if [ -d "$RUNTIME_DIR/drawings" ]; then
            drawings_count=$(find "$RUNTIME_DIR/drawings" -name "drawing_*.webp" 2>/dev/null | wc -l)
        fi
        if [ "$drawings_count" -gt 200 ]; then
            echo "  ${YELLOW}⚠${NC} drawings: $drawings_count files (no rotation — unbounded growth)"
            issues=$(( issues + 1 ))
        else
            echo "  ${GREEN}✓${NC} drawings: $drawings_count files"
        fi
    else
        echo "  ${DIM}no runtime directory at $RUNTIME_DIR${NC}"
    fi
    echo ""

    # ── Summary ──
    if [ "$issues" -eq 0 ]; then
        echo "${GREEN}${BOLD}✓ All hygiene checks passed.${NC} Project is clean."
    else
        echo "${YELLOW}${BOLD}⚠ $issues hygiene issue(s) found.${NC} Run ${BOLD}scripts/hygiene.sh --clean${NC} to fix disposable artifacts."
    fi
    echo ""
}

# ── Clean mode ────────────────────────────────────────────────────────
clean_caches() {
    echo "${BOLD}Cleaning caches and bytecode...${NC}"
    local freed=0

    # __pycache__ directories
    while IFS= read -r d; do
        local b
        b=$(du -sb "$d" 2>/dev/null | cut -f1 || echo 0)
        freed=$(( freed + b ))
        rm -rf "$d"
        echo "  ${GREEN}✓${NC} removed $d ($(bytes_to_human "$b"))"
    done < <(find . -name "__pycache__" -type d -not -path "*/target/*" 2>/dev/null)

    # Tool caches
    for cache in .mypy_cache .pytest_cache .ruff_cache; do
        if [ -d "$cache" ]; then
            local b
            b=$(dir_bytes "$cache")
            freed=$(( freed + b ))
            rm -rf "$cache"
            echo "  ${GREEN}✓${NC} removed $cache ($(bytes_to_human "$b"))"
        fi
    done

    echo "  ${DIM}freed $(bytes_to_human "$freed")${NC}"
}

clean_debug() {
    echo "${BOLD}Cleaning debug build artifacts...${NC}"
    local freed=0

    if [ -d "target/debug" ]; then
        local b
        b=$(dir_bytes "target/debug")
        freed=$(( freed + b ))
        rm -rf "target/debug"
        echo "  ${GREEN}✓${NC} removed target/debug ($(bytes_to_human "$b"))"
    fi

    if [ -d "target/doc" ]; then
        local b
        b=$(dir_bytes "target/doc")
        freed=$(( freed + b ))
        rm -rf "target/doc"
        echo "  ${GREEN}✓${NC} removed target/doc ($(bytes_to_human "$b"))"
    fi

    if [ -d "target/tmp" ]; then
        local b
        b=$(dir_bytes "target/tmp")
        freed=$(( freed + b ))
        rm -rf "target/tmp"
        echo "  ${GREEN}✓${NC} removed target/tmp ($(bytes_to_human "$b"))"
    fi

    echo "  ${DIM}freed $(bytes_to_human "$freed")${NC}"
    echo "  ${DIM}release build preserved (run.sh needs it)${NC}"
}

clean_all() {
    clean_caches
    echo ""
    clean_debug
    echo ""
    clean_runtime

    # Release build — regenerable but needed by run.sh
    if [ -d "target/release" ]; then
        local b
        b=$(dir_bytes "target/release")
        echo ""
        echo "${BOLD}Cleaning release build...${NC}"
        rm -rf "target/release"
        echo "  ${GREEN}✓${NC} removed target/release ($(bytes_to_human "$b"))"
        echo "  ${DIM}regenerate with: cargo build --release${NC}"
    fi

    # If target/ is now empty, remove it
    if [ -d "target" ] && [ -z "$(ls -A target 2>/dev/null)" ]; then
        rmdir "target"
        echo "  ${GREEN}✓${NC} removed empty target/"
    fi
}

clean_runtime() {
    echo "${BOLD}Cleaning orphaned runtime backups...${NC}"
    local freed=0

    if [ ! -d "$RUNTIME_DIR" ]; then
        echo "  ${DIM}no runtime directory at $RUNTIME_DIR${NC}"
        return
    fi

    # Orphaned backup files — not created by any current code path.
    # These are stale leftovers from past restores, cleanups, and
    # prune operations.  The live state files (cognitive_state.json,
    # ltm_store.*, concept_archive.db, etc.) are NEVER touched.
    while IFS= read -r f; do
        local b
        b=$(stat -c%s "$f" 2>/dev/null || echo 0)
        freed=$(( freed + b ))
        rm -f "$f"
        echo "  ${GREEN}✓${NC} removed $(basename "$f") ($(bytes_to_human "$b"))"
    done < <(find "$RUNTIME_DIR" -maxdepth 1 \( \
        -name "*.pre-restore-backup" \
        -o -name "*.pre-cleanup*" \
        -o -name "*.pre-proposal*" \
        -o -name "*.pre-prune-backup*" \
        -o -name "core_state.bin.bak.*" \
    \) 2>/dev/null)

    if [ "$freed" -eq 0 ]; then
        echo "  ${GREEN}✓${NC} no orphaned backups found"
    else
        echo "  ${DIM}freed $(bytes_to_human "$freed")${NC}"
    fi
}

do_clean() {
    local mode=${1:-all}
    echo ""
    echo "${BOLD}Genesis — Project Cleanup${NC}"
    echo "${DIM}$(date)${NC}"
    echo ""

    case "$mode" in
        all)
            clean_all
            ;;
        cache)
            clean_caches
            ;;
        debug)
            clean_debug
            ;;
        runtime)
            clean_runtime
            ;;
    esac

    echo ""
    echo "${GREEN}${BOLD}✓ Cleanup complete.${NC}"
    echo ""
    echo "${DIM}After-report:${NC}"
    do_report
}

# ── Help ──────────────────────────────────────────────────────────────
do_help() {
    cat <<'EOF'
Genesis project homeostasis system.

Usage:
  scripts/hygiene.sh                  Report (default — measure only)
  scripts/hygiene.sh --report         Size breakdown: source vs. artifacts
  scripts/hygiene.sh --check          Detect dead code + stale artifacts
  scripts/hygiene.sh --clean          Remove ALL disposable artifacts
  scripts/hygiene.sh --clean-cache    Remove only caches + bytecode
  scripts/hygiene.sh --clean-debug    Remove only the debug build
  scripts/hygiene.sh --clean-runtime  Remove orphaned runtime backups
  scripts/hygiene.sh --help           This message

Two locations are tracked:
  PROJECT  — the source tree (build artifacts, caches, source code)
  RUNTIME  — ~/.local/share/genesis/ (live state + runtime artifacts)

The --clean variants are safe: they never touch source code, configs,
binary assets (python/voices/), live runtime state (cognitive_state,
ltm_store, concept_archive, drawings, etc.), or the release binary.

Every byte is classified:
  SOURCE   — irreplaceable (code, configs, assets, live state, docs)
  ARTIFACT — disposable / regenerable (build output, caches, bytecode,
             orphaned backups)
EOF
}

# ── Main ──────────────────────────────────────────────────────────────
case "${1:-}" in
    --report|"")   do_report ;;
    --check)       do_check ;;
    --clean)       do_clean all ;;
    --clean-cache) do_clean cache ;;
    --clean-debug) do_clean debug ;;
    --clean-runtime) do_clean runtime ;;
    --help|-h)     do_help ;;
    *)
        echo "Unknown option: $1"
        echo "Run: scripts/hygiene.sh --help"
        exit 1
        ;;
esac
