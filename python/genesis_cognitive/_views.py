"""Internal helper for the brain-region view packages.

Each brain region package (``control/``, ``affect/``,
``action_selection/``, ``motor_learning/``, ``relay/``, ``autonomics/``,
and the re-export layers of ``association/``, ``auditory/``,
``vision/``) is a documented *view* over the top-level
cognitive modules. The modules themselves are multi-subsystem and stay
at the top level; the region package re-exports them so the anatomy
is a real connection layer:

    from genesis_cognitive.control import ExecutiveFunction

Re-exports resolve lazily through a PEP 562 module ``__getattr__``.
They must NOT be eager: several of the re-exported modules sit on an
import cycle through the subsystems (``language -> self -> perception ->
vision -> vision -> auditory -> language``), so importing
them at package-init time can hit a partially initialized package.
Lazy resolution makes every ``from genesis_cognitive.<subsystem> import X``
work once the package is initialized, regardless of entry order.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

__all__ = ["view_getattr"]


def view_getattr(exports: dict[str, str], package: str) -> Callable[[str], object]:
    """Build a module-level ``__getattr__`` for a brain-region package.

    Args:
        exports: Maps each public name to the module that provides it,
            expressed relative to ``genesis_cognitive`` (e.g.
            ``"emotion"`` or ``"memory"``).
        package: The calling subsystem package's ``__name__``, used in the
            AttributeError message for unknown names.
    """

    def __getattr__(name: str) -> object:
        target = exports.get(name)
        if target is None:
            raise AttributeError(
                f"module {package!r} has no attribute {name!r}"
            )
        module = importlib.import_module(f"genesis_cognitive.{target}")
        return getattr(module, name)

    return __getattr__
