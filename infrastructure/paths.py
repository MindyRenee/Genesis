"""Install-layout-aware filesystem roots.

In a source checkout the top-level packages are split across two
containers: the support packages (``infrastructure``,
``genesis_client``, ``world``, ``eval``, ``mind``) sit beside the
repository root, and the brain-region tree lives under ``python/``.
The repo root is therefore the package root — the single directory
whose children, including ``python/``, hold every importable package.
In a packaged install the same packages land flat in ``site-packages``,
so one directory again holds them all.
"""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """The directory holding the top-level packages.

    In a checkout this is the repository root: its children are
    ``python/`` (the brain regions) and the support packages beside
    it. In an installed wheel it is ``site-packages``. The parent of
    ``infrastructure/`` is that container in both layouts.
    """
    return Path(__file__).resolve().parent.parent


def code_root() -> Path:
    """The root of the agent's own source tree.

    This is what modules mean when they ask for ``project_root`` — the
    directory containing ``python/``, ``src/``, ``genesis``, and the
    helper scripts. With the packages split, the checkout's repo root
    and the package root are the same directory, so the resolution is:

    1. ``GENESIS_ROOT`` — explicit override for unusual layouts.
    2. :func:`package_root` — in a checkout that is the repo root; in a
       non-editable install it is ``site-packages``, where the
       installed packages are the code.
    """
    env = os.environ.get("GENESIS_ROOT")
    if env:
        return Path(env).expanduser()
    return package_root()
