#!/usr/bin/env bash
# gen_docs.sh — regenerate her reference documentation from source.
#
# The two sides, documented the way each speaks:
#   Rust   — cargo doc renders the /// doc comments (the math side)
#            into target/doc/genesis/index.html
#   Python — pydoc renders the module docstrings (the words side)
#            into target/pydoc/<module>.html
#
# Both outputs land under target/, which is gitignored — they are
# generated views of the code, never frozen artifacts. Run this any
# time the tree changes:
#
#   scripts/gen_docs.sh
#
# Python modules that refuse to import are skipped with a note —
# pydoc documents by importing, and some modules are not importable
# out of context.

set -u
cd "$(dirname "$0")/.."

echo "==> Rust: cargo doc --no-deps --document-private-items"
cargo doc --no-deps --document-private-items 2>&1 | tail -3

PYD="target/pydoc"
mkdir -p "$PYD"
VENV_PY="${VENV_PY:-.venv/bin/python}"
[ -x "$VENV_PY" ] || VENV_PY=python3

export PYTHONPATH="Self_World:Inner_World:Outer_World:.${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Python: pydoc -> $PYD"
# Every .py under the three worlds becomes a dotted module name.
# __init__.py documents its package, __main__.py is skipped.
find Self_World Inner_World Outer_World -name '__pycache__' -prune -o \
    -name '*.py' -print \
    | sed -e 's/\.py$//' -e 's|/|.|g' -e 's/\.__init__$//' -e '/__main__$/d' \
    | sort -u \
    | xargs -P"$(nproc)" -I{} sh -c \
        'timeout 15 "'"$VENV_PY"'" -m pydoc -w "{}" 2>/dev/null \
            || echo "SKIP {}"' \
    | sort | uniq -c

# pydoc -w writes into cwd; sweep the rendered pages into place.
find . -maxdepth 1 -name '*.html' -newer "$0" -exec mv -f {} "$PYD/" \; 2>/dev/null

echo "==> done"
echo "    Rust:   target/doc/genesis/index.html"
echo "    Python: $PYD/ ($(ls "$PYD" | wc -l) modules)"
