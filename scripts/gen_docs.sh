#!/usr/bin/env bash
# gen_docs.sh — regenerate her reference documentation from source.
#
# The two sides, documented the way each speaks:
#   Rust   — cargo doc renders the /// doc comments (the math side)
#            into docs/rust/
#   Python — pydoc renders the module docstrings (the words side)
#            into docs/python/<module>.html
#
# Generated docs live at root docs/ so GitHub Pages can serve them;
# Self_World/docs is a symlink into the same tree, so the
# self-world path (self-knowledge) still resolves for her tools.
# Linux/ops man pages that generated API docs cannot cover stay
# under docs/man/.
#
# Python modules that refuse to import are skipped with a note —
# pydoc documents by importing, and some modules are not importable
# out of context.

set -u
cd "$(dirname "$0")/.."

DOCS="docs"
RUST_OUT="target/doc"
PY_OUT="$DOCS/python"

echo "==> Rust: cargo doc --no-deps --document-private-items"
cargo doc --no-deps --document-private-items 2>&1 | tail -3
rm -rf "$DOCS/rust"
mkdir -p "$DOCS"
cp -r "$RUST_OUT" "$DOCS/rust"

VENV_PY="${VENV_PY:-.venv/bin/python}"
[ -x "$VENV_PY" ] || VENV_PY=python3
VENV_PY="$(readlink -f "$VENV_PY")"
export PYTHONPATH="$PWD/Self_World:$PWD/Inner_World:$PWD/Outer_World:$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Python: pydoc -> $PY_OUT"
rm -rf "$PY_OUT"
mkdir -p "$PY_OUT"
# Every .py under the three worlds becomes a dotted module name.
# __init__.py documents its package, __main__.py is skipped.
cd "$PY_OUT"
find ../../Self_World ../../Inner_World ../../Outer_World \
    -name '__pycache__' -prune -o -name '*.py' -print \
    | sed -E -e 's,.*(Self_World|Inner_World|Outer_World)/,\1/,' \
          -e 's/\.py$//' -e 's|/|.|g' -e 's/\.__init__$//' -e '/__main__$/d' \
    | sort -u \
    | xargs -P"$(nproc)" -I{} sh -c \
        'timeout 15 "'"$VENV_PY"'" -m pydoc -w "{}" 2>/dev/null \
            || echo "SKIP {}"' \
    | sort | uniq -c
cd - >/dev/null

echo "==> done"
echo "    Rust:   $DOCS/rust/genesis/index.html"
echo "    Python: $PY_OUT/ ($(ls "$PY_OUT" | wc -l) modules)"
