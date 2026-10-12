"""Guard the laziness that keeps the ``self`` import cycle harmless.

Several modules do ``from self import X`` at module level::

    frontal_lobe/generator.py     -> from self import SelfModel
    temporal_lobe/vocabulary.py   -> from self import PersonalityTraits
    temporal_lobe/memory_store.py -> from self import SelfModel
    infrastructure/narrative.py   -> from self import PersonalityTraits, SelfModel

Meanwhile ``self`` itself imports the rest of the tree
(``self.composer`` reaches ``frontal_lobe.graph_walk``,
``self.metacognitive_model``/``self.reflection`` reach ``frontal_lobe``).

If ``self/__init__.py`` ever *eagerly* imported its submodules, that
chain could run ``from self import SelfModel`` inside one of those
modules while ``self`` is only partially initialised — an ImportError
on ``import mind``. The package resolves attributes lazily (PEP 562,
``view_getattr``), so ``from self import X`` binds only the submodule
that defines X — and ``self.model``'s own imports (concepts,
genesis_client, infrastructure.config) never reach the language
modules, so the edge can never fire mid-import.

This test pins two things:

1. ``self/__init__.py`` stays lazy — it must not ``from .x import``
   a submodule at module level;
2. a fresh interpreter can ``import mind`` — the end-to-end check
   that the whole graph actually loads.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SELF_INIT = _ROOT / "frontal_lobe" / "self" / "__init__.py"


def test_self_package_stays_lazy() -> None:
    """``self/__init__`` must not eagerly import its submodules.

    The ``from self import X`` edges inside generator.py, vocabulary.py,
    memory_store.py and narrative.py are only safe while ``self`` binds
    submodules on demand. An eager ``from .model import SelfModel`` at
    the top of the package re-opens the self -> perception -> language
    -> self partial-initialisation cycle.
    """
    tree = ast.parse(SELF_INIT.read_text(encoding="utf-8"))
    eager = []
    for node in tree.body:  # top-level statements only
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is not None
        ):
            eager.append(node.module)
        if isinstance(node, ast.ImportFrom) and node.module == "_views":
            continue
    assert not eager, (
        f"self/__init__.py eagerly imports submodules: {eager}. "
        "That re-opens the partial-initialisation cycle — keep the "
        "package lazy via view_getattr."
    )


def test_package_imports_cleanly() -> None:
    """The whole point: a fresh interpreter must import the mind."""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{_ROOT.parent}{os.pathsep}{_ROOT}{os.pathsep}"
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-c", "import mind"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=env,
    )
    assert result.returncode == 0, f"import mind failed:\n{result.stderr}"
