"""Guard the import graph that keeps the ``self`` cycle harmless.

Several modules import ``self`` submodules directly::

    frontal_lobe/generator.py     -> from self.self_model import SelfModel
    temporal_lobe/vocabulary.py   -> from self.self_model import PersonalityTraits
    temporal_lobe/memory_store.py -> from self.self_model import SelfModel
    infrastructure/narrative.py   -> from self.self_model import PersonalityTraits, SelfModel

Meanwhile ``self`` itself reaches the rest of the tree
(``self.composer`` reaches ``frontal_lobe.graph_walk``,
``self.metacognitive_model``/``self.reflection`` reach ``frontal_lobe``).

If ``self/__init__.py`` ever *eagerly* imported its submodules, that
chain could run a ``self`` submodule import inside one of those
modules while ``self`` is only partially initialised — an ImportError
on ``import mind``. The package ``__init__`` stays a docstring-only
shell, so ``import self`` never pulls the cycle; each consumer binds
exactly the submodule it needs.

This test pins two things:

1. ``self/__init__.py`` stays free of eager submodule imports;
2. a fresh interpreter can ``import mind`` — the end-to-end check
   that the whole graph actually loads.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
SELF_INIT = _ROOT / "Self_World" / "self" / "__init__.py"


def test_self_package_stays_lazy() -> None:
    """``self/__init__`` must not eagerly import its submodules.

    The ``self.<module>`` imports inside generator.py, vocabulary.py,
    memory_store.py and narrative.py are only safe while ``self``'s
    ``__init__`` binds nothing. An eager ``from .self_model import
    SelfModel`` at the top of the package re-opens the
    self -> perception -> language -> self partial-initialisation
    cycle.
    """
    tree = ast.parse(SELF_INIT.read_text(encoding="utf-8"))
    eager = []
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None:
            eager.append(node.module)
    assert not eager, (
        f"self/__init__.py eagerly imports submodules: {eager}. "
        "That re-opens the partial-initialisation cycle — keep the "
        "package init free of submodule imports."
    )


def test_package_imports_cleanly() -> None:
    """The whole point: a fresh interpreter must import the mind."""
    import os

    env = dict(os.environ)
    worlds = os.pathsep.join(
        str(_ROOT / d) for d in ("Self_World", "Inner_World", "Outer_World", ".")
    )
    env["PYTHONPATH"] = worlds + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import mind"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=env,
    )
    assert result.returncode == 0, f"import mind failed:\n{result.stderr}"
