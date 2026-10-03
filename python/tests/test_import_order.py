"""Guard the load-bearing import order in cognition/engine.py.

``cognition/engine.py`` participates in a three-way import cycle::

    self -> perception -> language -> self

Each package's ``__init__`` eagerly imports its submodules, so whichever one
is entered first leaves the other two partially initialised. Concretely,
``language/generator.py`` does ``from ..self import SelfModel``; if
``genesis_cognitive.self`` is still executing its own ``__init__`` when that
line runs, ``SelfModel`` is not yet bound and the import raises.

The cycle is entered from ``infrastructure/narrative.py``, which
``cognition/engine.py`` imports. Placing that import *after* ``..self``,
``..perception`` and ``..language`` means those three are already bound
before the cycle is walked. Moving it earlier breaks the entire package:
every ``import genesis_cognitive`` fails.

Sorting tools (``ruff --fix``) will happily undo it, so this test pins the
order. ``ruff``'s I001 is ignored for this one file in pyproject.toml with a
pointer here.
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = _ROOT / "genesis_cognitive" / "cognition" / "engine.py"

# Only `language` must precede it. `language/generator.py` is the module that
# reads `SelfModel` out of the partially-initialised `self` package, so
# `language` is the edge that must already be bound. `perception` and `self`
# may follow `narrative`: by then `self` is mid-`__init__` either way, and the
# order that matters is the one `language` fixes.
MUST_PRECEDE_NARRATIVE = ("language",)


def _relative_import_order() -> dict[str, int]:
    """Map each ``..<pkg>`` import to the line it appears on."""
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    order: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module:
            order.setdefault(node.module.split(".")[0], node.lineno)
    return order


def test_narrative_imports_after_the_cycle() -> None:
    order = _relative_import_order()
    assert "infrastructure" in order, "expected a ..infrastructure import"
    narrative_line = order["infrastructure"]
    for package in MUST_PRECEDE_NARRATIVE:
        assert package in order, f"expected a ..{package} import"
        assert order[package] < narrative_line, (
            f"..{package} is imported on line {order[package]}, after "
            f"..infrastructure on line {narrative_line}. The cycle "
            f"(self -> perception -> language -> self) will be entered with "
            f"genesis_cognitive.self only partially initialised, and "
            f"language/generator.py's `from ..self import SelfModel` fails."
        )


def test_package_imports_cleanly() -> None:
    """The whole point: a fresh interpreter must import the package."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import genesis_cognitive"],
        capture_output=True,
        text=True,
        cwd=str(ENGINE.parents[2]),
    )
    assert result.returncode == 0, f"import genesis_cognitive failed:\n{result.stderr}"
