"""Module sampler — statistical attribution of execution to brain parts.

Hardware telemetry can see which *process* is firing (per-lobe
telemetry), but not which *subsystem* inside the conscious process is
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
credited to the deepest cognitive frame — the module that
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


def _package_root() -> str:
    """The ``python/`` source directory — the whole-brain tree.

    This module lives inside a brain-region subpackage, but it samples
    code across the whole package, so it needs the package root — not
    its own directory. Walking up while the parent is itself a package
    keeps this correct wherever the file ends up living.
    """
    current = os.path.dirname(os.path.abspath(__file__))
    while os.path.exists(os.path.join(os.path.dirname(current), "__init__.py")):
        current = os.path.dirname(current)
    return current


_PKG_DIR = _package_root()
_CLIENT_DIR = os.path.join(os.path.dirname(_PKG_DIR), "genesis_client")

# Subsystem directory → manifest module. The brain-region packages
# (frontal_lobe, limbic_system, …) are documented views over the
# top-level subsystems, so both the view name and the real
# implementation directory resolve here.
_DIR_MODULE = {
    "reasoning": MODULE_REASONING,
    "memory": MODULE_MEMORY,
    "concepts": MODULE_MEMORY,
    "perception": MODULE_SENSORY,
    "occipital_lobe": MODULE_SENSORY,
    "temporal_lobe": MODULE_SENSORY,
    "parietal_lobe": MODULE_SENSORY,
    "thalamus": MODULE_ATTENTION,  # relay/gate — attention switching
    "frontal_lobe": MODULE_INTENTION,  # executive / goal direction
    "basal_ganglia": MODULE_INTENTION,  # action selection
    "cerebellum": MODULE_MOTOR,
    "limbic_system": MODULE_EMOTION,
    "neurochemical": MODULE_EMOTION,
    "brainstem": MODULE_EMOTION,  # arousal / neurochemical regulation
    "sleep": MODULE_DREAMING,
    "tools": MODULE_INTENTION,
    "self": MODULE_METACOGNITION,
    "world": MODULE_REASONING,  # model of the world and other minds
    "eval": MODULE_METACOGNITION,  # measuring herself
}

# Files inside infrastructure/ that attribute to something other
# than the default (metacognition — support systems).
_INFRA_FILE_MODULE = {
    "persistence.py": MODULE_MEMORY,
    "vq_codebook.py": MODULE_MEMORY,
    "wordnet_dictionary.py": MODULE_MEMORY,
}

# Individual module files → manifest module. Checked before the
# directory table so a file that belongs to one brain part attributes
# to that part wherever it is filed — several of these now live
# inside a brain-region package rather than at the package root.
_FILE_MODULE = {
    "abstract_reasoning.py": MODULE_REASONING,
    "acquisition.py": MODULE_LANGUAGE,
    "answer_composer.py": MODULE_LANGUAGE,
    "attention.py": MODULE_ATTENTION,
    "base.py": MODULE_LANGUAGE,
    "brain_waves.py": MODULE_METACOGNITION,
    "canvas.py": MODULE_MOTOR,
    "cognitive_map.py": MODULE_MEMORY,
    "commitment.py": MODULE_INTENTION,
    "comprehension.py": MODULE_LANGUAGE,
    "concept_learner.py": MODULE_LANGUAGE,
    "conversation.py": MODULE_LANGUAGE,
    "dual.py": MODULE_MEMORY,
    "emotional_regulator.py": MODULE_EMOTION,
    "emotion.py": MODULE_EMOTION,
    "executive.py": MODULE_INTENTION,
    "figurative.py": MODULE_LANGUAGE,
    "flow.py": MODULE_LANGUAGE,
    "generator.py": MODULE_LANGUAGE,
    "global_workspace.py": MODULE_ATTENTION,
    "grammar.py": MODULE_LANGUAGE,
    "graph_walk.py": MODULE_LANGUAGE,
    "memory_store.py": MODULE_MEMORY,
    "meta_cognitive_router.py": MODULE_METACOGNITION,
    "mind.py": MODULE_METACOGNITION,
    "module_sampler.py": MODULE_METACOGNITION,
    "morphology.py": MODULE_LANGUAGE,
    "mtl_bridge.py": MODULE_MEMORY,
    "plasticity.py": MODULE_MEMORY,
    "pragmatics.py": MODULE_LANGUAGE,
    "prosody.py": MODULE_LANGUAGE,
    "question_composer.py": MODULE_LANGUAGE,
    "question_handler.py": MODULE_LANGUAGE,
    "response_styler.py": MODULE_LANGUAGE,
    "saliency.py": MODULE_ATTENTION,
    "self_inquiry.py": MODULE_METACOGNITION,
    "self_monitor.py": MODULE_LANGUAGE,
    "semantic_grounding.py": MODULE_LANGUAGE,
    "sleep.py": MODULE_DREAMING,
    "speech.py": MODULE_MOTOR,
    "statistical_learner.py": MODULE_LANGUAGE,
    "status.py": MODULE_METACOGNITION,
    "stdp.py": MODULE_MEMORY,
    "synapses.py": MODULE_MEMORY,
    "system_monitor.py": MODULE_METACOGNITION,
    "thought_composer.py": MODULE_LANGUAGE,
    "topic_resolver.py": MODULE_LANGUAGE,
    "voice.py": MODULE_LANGUAGE,
    "vocabulary.py": MODULE_LANGUAGE,
    "volition.py": MODULE_INTENTION,
}


def _module_of_filename(filename: str) -> int | None:
    """Resolve a code filename to a manifest module, or None."""
    rel = os.path.relpath(filename, _PKG_DIR)
    if rel.startswith(".."):
        return None
    parts = rel.split(os.sep)
    if len(parts) == 1:
        return _FILE_MODULE.get(parts[0])
    # A named module file keeps its own attribution even when it is
    # filed inside a brain-region package.
    per_file = _FILE_MODULE.get(parts[-1])
    if per_file is not None:
        return per_file
    # The immediate directory names the module's home — a subsystem
    # filed inside a brain-region package (frontal_lobe/reasoning/)
    # still attributes to its own function, not the host region's.
    module = _DIR_MODULE.get(parts[-2]) if len(parts) > 1 else None
    if module is None:
        module = _DIR_MODULE.get(parts[0])
    if module is None and parts[0] == "infrastructure":
        return _INFRA_FILE_MODULE.get(parts[-1], MODULE_METACOGNITION)
    return module


def _module_for_frame(frame) -> int | None:
    """Attribute a thread's stack to a manifest module.

    Leaf in the region tree → that subsystem's module. Leaf inside
    genesis_client (IPC in flight) → the deepest cognitive
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
    UPDATE_MODULE_STATUS / GET_LOBE_TELEMETRY.
    """

    SAMPLE_HZ = 25.0

    def __init__(self, credit: Callable[[int, float], None]) -> None:
        """Initialize the sampler with the callback receiving module-time credit."""
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
        """Sample running frames at a fixed rate until stopped."""
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
