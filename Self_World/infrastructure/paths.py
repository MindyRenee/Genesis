"""Install-layout-aware filesystem roots.

Genesis's source is organized into three worlds — ``Self_World/``
(the self and organism), ``Inner_Non_Physical_World/`` (the non-physical
phenomena — field, waves, dreams, thoughts), and ``Outer_Physical_World/`` (the
environment) — each a package root holding the packages that belong
to it. In a source checkout all three sit under the repository
root; in a packaged install the same packages land flat in
``site-packages``, so one directory again holds them all.
"""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """The directory holding the top-level packages.

    In a checkout this is the repository root: its children are the
    three world directories. In an installed wheel it is
    ``site-packages``. ``infrastructure/`` sits inside
    ``Self_World/`` in a checkout and flat in ``site-packages`` —
    the parent of that container is the package root in both cases.
    """
    here = Path(__file__).resolve().parent.parent
    if here.name == "Self_World":
        return here.parent
    return here


def code_root() -> Path:
    """The root of the agent's own source tree.

    This is what modules mean when they ask for ``project_root`` — the
    directory containing the three worlds, ``scripts``, ``genesis``,
    and the helper files. With the worlds as package containers, the
    checkout's repo root and the package root are the same directory,
    so the resolution is:

    1. ``GENESIS_ROOT`` — explicit override for unusual layouts.
    2. :func:`package_root` — in a checkout that is the repo root; in a
       non-editable install it is ``site-packages``, where the
       installed packages are the code.
    """
    env = os.environ.get("GENESIS_ROOT")
    if env:
        return Path(env).expanduser()
    return package_root()
