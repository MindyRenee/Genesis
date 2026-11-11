"""Creation — scaffolding on the bench.

`scaffold` stamps out standard skeletons so new work starts from a
correct shape instead of a blank file: python module, test file,
rust binary, README, .gitignore, license. These are *scaffolds* —
structure, not content. What she writes inside them is her own.
"""

from __future__ import annotations

import pathlib
from typing import Any

from tools.framework import ToolResult

from infrastructure.text import is_blank

_SCAFFOLDS: dict[str, Any] = {
    "python_module": lambda name: (
        f"{name}.py",
        f'"""{name} — """\n\nfrom __future__ import annotations\n',
    ),
    "python_test": lambda name: (
        f"test_{name}.py",
        f'"""Tests for {name}."""\n\nfrom __future__ import annotations\n\n\n'
        f"def test_placeholder() -> None:\n    assert True\n",
    ),
    "rust_bin": lambda name: (
        f"{name}.rs",
        'fn main() {\n    println!("Hello");\n}\n',
    ),
    "readme": lambda name: (
        "README.md",
        f"# {name}\n\n",
    ),
    "gitignore": lambda name: (
        ".gitignore",
        "__pycache__/\n*.pyc\ntarget/\n.venv/\n",
    ),
}


def scaffold(
    kind: str,
    name: str = "main",
    dest: str = ".",
    project_root: str = ".",
) -> ToolResult:
    """Stamp out a standard file skeleton.

    ``kind``: python_module, python_test, rust_bin, readme, gitignore.
    ``name`` becomes the filename stem (readme/gitignore ignore it).
    Refuses to overwrite an existing file — scaffolding is a start,
    not a stomp.
    """
    maker = _SCAFFOLDS.get(kind)
    if maker is None:
        return ToolResult(
            success=False,
            error=f"unknown kind {kind!r}; known: {', '.join(sorted(_SCAFFOLDS))}",
        )
    if is_blank(name):
        name = "main"
    filename, content = maker(name)
    root = pathlib.Path(project_root).resolve()
    target = (root / pathlib.Path(dest) / filename).resolve()
    if target.exists():
        return ToolResult(success=False, error=f"already exists: {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(
            success=True,
            output=f"scaffolded {filename}",
            data={"path": str(target), "kind": kind},
        )
    except OSError as e:
        return ToolResult(success=False, error=str(e))


CREATION_TOOLS = (
    (
        "scaffold",
        "Stamp a standard skeleton: kind= python_module|python_test|"
        "rust_bin|readme|gitignore, name= stem, dest= directory. "
        "Refuses to overwrite — structure, never content.",
        scaffold,
    ),
)
