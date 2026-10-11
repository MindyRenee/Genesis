"""Install-layout-aware filesystem roots.

In a source checkout the packages live under ``python/`` inside the
repository root, and parent-relative lookups (``target/release/``
binaries, ``scripts/`` helpers, the tree Genesis reads to know her own
code) resolve against the checkout. In a packaged install the same
packages land flat in ``site-packages`` and those depths are wrong.

Resolve by content instead of by depth so both layouts work.
"""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """The directory holding the top-level packages.

    ``python/`` in a checkout; ``site-packages/`` in an installed
    wheel. Walk up while the parent is itself a package so the answer
    is correct however deep the calling module sits.
    """
    current = Path(__file__).resolve().parent
    while (current.parent / "__init__.py").exists():
        current = current.parent
    return current


def code_root() -> Path:
    """The root of the agent's own source tree.

    This is what modules mean when they ask for ``project_root`` — the
    directory containing ``python/``, ``src/``, ``run.sh``, and the
    helper scripts. Resolution order:

    1. ``GENESIS_ROOT`` — explicit override for unusual layouts.
    2. The parent of :func:`package_root`, when it looks like a source
       tree (``run.sh`` or ``src/`` present). Editable installs resolve
       ``__file__`` inside the checkout, so this covers both a real
       checkout and ``pip install -e``.
    3. :func:`package_root` itself — a non-editable install has no
       surrounding repository; the installed packages are the code.
    """
    env = os.environ.get("GENESIS_ROOT")
    if env:
        return Path(env).expanduser()
    pkg = package_root()
    parent = pkg.parent
    if (parent / "run.sh").exists() or (parent / "src").is_dir():
        return parent
    return pkg
