"""Internal helper for the brain-region view packages.

Each brain region package (``frontal_lobe/``, ``temporal_lobe/``,
``parietal_lobe/``, ``occipital_lobe/``, ``limbic_system/``,
``basal_ganglia/``, ``cerebellum/``, ``thalamus/``, ``brainstem/``,
``hypothalamus/``, ``insula/``, ``basal_forebrain/``)
is the anatomical home of its cognitive modules. Modules that belong
to a region live inside the package; genuinely multi-subsystem modules
(memory, self, sleep, tools, ...) stay at the top level and
are claimed by the regions that use them. Either way the region package
re-exports them so the anatomy is a real connection layer:

    from frontal_lobe import ExecutiveFunction

Re-exports resolve lazily through a PEP 562 module ``__getattr__``.
They must NOT be eager: several of the re-exported modules sit on an
import cycle through the subsystems (``language -> self -> perception ->
vision -> vision -> auditory -> language``), so importing
them at package-init time can hit a partially initialized package.
Lazy resolution makes every ``from <subsystem> import X``
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
            expressed relative to ``the region packages`` (e.g.
            ``"emotion"`` or ``"memory"``).
        package: The calling subsystem package's ``__name__``, used in the
            AttributeError message for unknown names.
    """

    def __getattr__(name: str) -> object:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module = importlib.import_module(target)
        return getattr(module, name)

    return __getattr__
