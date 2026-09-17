"""Module sampler — statistical attribution of execution to brain parts.

Hardware telemetry can see which *process* is firing (per-subsystem
telemetry), but not which *subsystem* inside the cognitive process is
doing the work. This sampler closes that gap the way neuroscience
does: it doesn't declare which region is active, it *observes* where
execution lives.

A daemon thread samples ``sys._current_frames()`` at a fixed rate. A
thread is credited only when its *leaf* frame is executing Genesis
bytecode — under the GIL at most one thread runs at a time, so each
sample answers "which brain part is executing right now". Threads
parked in ``sleep``/``recv``/``join`` contribute nothing, exactly like
an idle region contributes nothing to an EEG trace. When the leaf is
inside ``genesis_client`` (an IPC call in flight), the sample is
credited to the deepest ``genesis_cognitive`` frame — the module that
initiated the call.

Because attribution follows the package layout — which already mirrors
her brain anatomy — it survives refactors, catches work nobody thought
to instrument (learner, inner life, speech, the sampler itself), and
can never drift from a hand-maintained stage table.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable

from genesis_client.protocol import (
    MODULE_ATTENTION,
    MODULE_DREAMING,
    MODULE_EMOTION,
    MODULE_INTENTION,
    MODULE_LANGUAGE,
    MODULE_MEMORY,
    MODULE_METACOGNITION,
    MODULE_MOTOR,
    MODULE_REASONING,
    MODULE_SENSORY,
)

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_CLIENT_DIR = os.path.join(os.path.dirname(_PKG_DIR), "genesis_client")

# Subsystem directory → manifest module. The brain-region packages
# (control, affect, …) are documented views over the
# top-level subsystems, so both the view name and the real
# implementation directory resolve here.
_DIR_MODULE = {
    "cognition": MODULE_REASONING,
    "reasoning": MODULE_REASONING,
    "memory": MODULE_MEMORY,
    "concepts": MODULE_MEMORY,
    "language": MODULE_LANGUAGE,
    "perception": MODULE_SENSORY,
    "vision": MODULE_SENSORY,
    "auditory": MODULE_SENSORY,
    "association": MODULE_SENSORY,
    "spatial": MODULE_SENSORY,
    "relay": MODULE_ATTENTION,  # relay/gate — attention switching
    "control": MODULE_INTENTION,  # executive / goal direction
    "action_selection": MODULE_INTENTION,  # action selection
    "motor_learning": MODULE_MOTOR,
    "affect": MODULE_EMOTION,
    "neurochemical": MODULE_EMOTION,
    "autonomics": MODULE_EMOTION,  # arousal / neurochemical regulation
    "sleep": MODULE_DREAMING,
    "learning": MODULE_INTENTION,  # self-directed acquisition
    "tools": MODULE_INTENTION,
    "self": MODULE_METACOGNITION,
}

# Files inside mind/ that attribute to something other than the
# default (metacognition — orchestration and self-monitoring).
_MIND_FILE_MODULE = {
    "volition.py": MODULE_INTENTION,
    "sleep.py": MODULE_DREAMING,
}

# Top-level module files → manifest module.
_FILE_MODULE = {
    "attention.py": MODULE_ATTENTION,
    "ambient.py": MODULE_SENSORY,
    "brain_waves.py": MODULE_METACOGNITION,
    "bug_reporter.py": MODULE_METACOGNITION,
    "canvas.py": MODULE_MOTOR,
    "emotional_regulator.py": MODULE_EMOTION,
    "emotion.py": MODULE_EMOTION,
    "executive.py": MODULE_INTENTION,
    "global_workspace.py": MODULE_ATTENTION,
    "growth_ledger.py": MODULE_METACOGNITION,
    "journal.py": MODULE_METACOGNITION,
    "module_sampler.py": MODULE_METACOGNITION,
    "narrative.py": MODULE_METACOGNITION,
    "notifications.py": MODULE_METACOGNITION,
    "persistence.py": MODULE_MEMORY,
    "speech.py": MODULE_MOTOR,
    "system_monitor.py": MODULE_METACOGNITION,
    "user_profile.py": MODULE_METACOGNITION,
    "volition.py": MODULE_INTENTION,
    "vq_codebook.py": MODULE_MEMORY,
    "wordnet_dictionary.py": MODULE_MEMORY,
}


def _module_of_filename(filename: str) -> int | None:
    """Resolve a code filename to a manifest module, or None."""
    rel = os.path.relpath(filename, _PKG_DIR)
    if rel.startswith(".."):
        return None
    parts = rel.split(os.sep)
    if len(parts) == 1:
        return _FILE_MODULE.get(parts[0])
    module = _DIR_MODULE.get(parts[0])
    if module is None and parts[0] == "mind":
        return _MIND_FILE_MODULE.get(parts[-1], MODULE_METACOGNITION)
    return module


def _module_for_frame(frame) -> int | None:
    """Attribute a thread's stack to a manifest module.

    Leaf in genesis_cognitive → that subsystem's module. Leaf inside
    genesis_client (IPC in flight) → the deepest genesis_cognitive
    frame, the module that initiated the call. Anything else → None
    (thread parked in stdlib — no Genesis code executing).
    """
    leaf_mod = _module_of_filename(frame.f_code.co_filename)
    if leaf_mod is not None:
        return leaf_mod
    filename = frame.f_code.co_filename
    if not os.path.relpath(filename, _CLIENT_DIR).startswith(".."):
        # Leaf is inside the IPC client — the caller's subsystem owns it.
        f = frame.f_back
        while f is not None:
            mod = _module_of_filename(f.f_code.co_filename)
            if mod is not None:
                return mod
            f = f.f_back
    return None


class ModuleSampler:
    """Samples running threads and credits execution time to modules.

    ``credit`` is called as ``credit(module_id, seconds)`` — in the
    Mind this is ``_credit_module``, accumulating into the per-module
    work counter that the heartbeat normalizes into ``cpu_share`` for
    UPDATE_MODULE_STATUS / GET_SUBSYSTEM_TELEMETRY.
    """

    SAMPLE_HZ = 25.0

    def __init__(self, credit: Callable[[int, float], None]) -> None:
        self._credit = credit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the sampling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="module-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampling thread to exit and join it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        interval = 1.0 / self.SAMPLE_HZ
        while not self._stop.wait(interval):
            try:
                self._sample(interval)
            except Exception as e:  # noqa: BLE001 — telemetry must never kill the loop
                logger.debug(f"module sample failed: {e}")

    def _sample(self, dt: float) -> None:
        """One sample tick: credit each executing thread's module dt."""
        for frame in sys._current_frames().values():
            module = _module_for_frame(frame)
            if module is not None:
                self._credit(module, dt)
