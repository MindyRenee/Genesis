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
an idle region contributes nothing to an EEG trace: a C-level wait
pushes no Python frame, so a parked leaf sits on the same instruction
sample after sample, and a frame is credited only when it has moved
since the last tick. When the leaf is
inside ``genesis_client`` (an IPC call in flight), the sample is
credited to the deepest cognitive frame — the module that
initiated the call.

Because attribution follows the package layout — which already mirrors
her brain anatomy — it survives refactors, catches work nobody thought
to instrument (learner, inner life, speech, the sampler itself), and
can never drift from a hand-maintained stage table.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
from collections.abc import Callable

import genesis_client
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
from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread

logger = logging.getLogger(__name__)


def _package_root() -> str:
    """The directory holding the region packages — the whole-brain tree.

    This module lives inside a brain-region subpackage, so the
    region-tree root is the directory *above* its own package —
    ``python/`` in a checkout, the site root in an installed layout.
    Resolving through the imported package (rather than walking
    ``__file__`` parents for ``__init__.py``) makes it correct in
    both: a checkout wraps the regions in the ``python`` namespace
    package, an install does not — the filesystem walk used to stop
    at ``site-packages/brainstem`` and attribute nothing.
    """
    parts = __name__.split(".")
    # A ``python.``-rooted import sees the regions as subpackages —
    # anchor on ``python.<region>``; otherwise the first component IS
    # the region package.
    anchor = ".".join(parts[:2]) if parts[0] == "python" else parts[0]
    pkg = sys.modules.get(anchor)
    if pkg is not None:
        path = getattr(pkg, "__path__", None)
        pkg_dir = path[0] if path else os.path.dirname(pkg.__file__ or "")
        return os.path.dirname(pkg_dir)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_PKG_DIR = _package_root()
_CLIENT_DIR = genesis_client.__path__[0]


def _sibling_dir(package: str) -> str | None:
    """Real directory of a support package beside the region tree.

    The support packages (``infrastructure``, ``eval``) sit *next to*
    ``python/`` in a checkout and inside the site root when installed,
    so the region-tree relpath alone cannot find them — resolving by
    import works in either layout.
    """
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return os.path.abspath(spec.submodule_search_locations[0])
    if spec.origin:
        return os.path.dirname(os.path.abspath(spec.origin))
    return None


_INFRA_DIR = _sibling_dir("infrastructure")
_EVAL_DIR = _sibling_dir("eval")

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
    "hypothalamus": MODULE_EMOTION,  # homeostatic nuclei (VLPO, HPA, SCN)
    "insula": MODULE_EMOTION,  # interoceptive cortex — somatic markers
    "basal_forebrain": MODULE_EMOTION,  # ACh / adenosine source nuclei
    "sleep": MODULE_DREAMING,
    "tools": MODULE_INTENTION,
    "self": MODULE_METACOGNITION,
    "world": MODULE_REASONING,  # nested world-model packages
    "worlds": MODULE_REASONING,  # model of the world and other minds
    "eval": MODULE_METACOGNITION,  # measuring herself
}

# Files inside infrastructure/ that attribute to something other
# than the default (metacognition — support systems).
_INFRA_FILE_MODULE = {
    "persistence.py": MODULE_MEMORY,
    "vq_codebook.py": MODULE_MEMORY,
    "wordnet_dictionary.py": MODULE_MEMORY,
}

# Organism-level py-modules that live *beside* the region tree at the
# repository root (resolved by real path — in an installed layout they
# sit inside the site root and resolve by basename instead, so both
# ``_FILE_MODULE`` and this table name them).
_ROOT_FILE_MODULE: dict[str, int] = {}
for _root_name, _root_mod in (
    ("mind", MODULE_METACOGNITION),  # the assembled organ
    ("turn", MODULE_METACOGNITION),  # one turn across regions — orchestration
    ("genesis_cli", MODULE_METACOGNITION),  # launcher / rebirth shepherd
    ("commitment", MODULE_INTENTION),  # boundary machinery — volition's digitizer
    ("discord_feed", MODULE_SENSORY),  # the public channel — an ear
):
    try:
        _spec = importlib.util.find_spec(_root_name)
    except (ImportError, AttributeError, ValueError):
        _spec = None
    if _spec is not None and _spec.origin:
        _ROOT_FILE_MODULE[os.path.abspath(_spec.origin)] = _root_mod

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
    "discord_feed.py": MODULE_SENSORY,
    "dual.py": MODULE_MEMORY,
    "emotional_regulator.py": MODULE_EMOTION,
    "emotion.py": MODULE_EMOTION,
    "executive.py": MODULE_INTENTION,
    "figurative.py": MODULE_LANGUAGE,
    "flow.py": MODULE_LANGUAGE,
    "generator.py": MODULE_LANGUAGE,
    "genesis_cli.py": MODULE_METACOGNITION,
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
    "turn.py": MODULE_METACOGNITION,
    "voice.py": MODULE_LANGUAGE,
    "vocabulary.py": MODULE_LANGUAGE,
    "volition.py": MODULE_INTENTION,
}

# Files at the *root of the region tree* itself — not inside any
# brain-region package. Pinned separately because the basename also
# exists inside regions (``temporal_lobe/concepts/topology.py`` is the
# concept kind, not the organ's own topology map) and must keep its
# region attribution there.
_TOPLEVEL_FILE_MODULE = {
    "topology.py": MODULE_METACOGNITION,  # the organism's own wiring map
}


def _module_of_filename(filename: str) -> int | None:
    """Resolve a code filename to a manifest module, or None."""
    rel = os.path.relpath(filename, _PKG_DIR)
    if rel.startswith(".."):
        # Outside the region tree: the support packages and the
        # organism-level root modules, resolved by their own anchors.
        # They sit beside the region tree in a checkout and inside
        # the site root when installed, so they only reach this
        # branch in the checkout layout.
        if _INFRA_DIR and not os.path.relpath(filename, _INFRA_DIR).startswith(".."):
            return _INFRA_FILE_MODULE.get(
                os.path.basename(filename), MODULE_METACOGNITION
            )
        if _EVAL_DIR and not os.path.relpath(filename, _EVAL_DIR).startswith(".."):
            return MODULE_METACOGNITION
        return _ROOT_FILE_MODULE.get(os.path.abspath(filename))
    parts = rel.split(os.sep)
    if parts[0] == "python" and len(parts) > 1:
        # An install can keep the checkout's ``python`` namespace
        # package — site-packages/python/brainstem/x.py — so look one
        # level deeper for the region.
        parts = parts[1:]
    if len(parts) == 1:
        return _TOPLEVEL_FILE_MODULE.get(parts[0], _FILE_MODULE.get(parts[0]))
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
    filename = frame.f_code.co_filename
    # Pseudo-filenames ("<frozen importlib._bootstrap>", "<string>")
    # and empty names are not paths — relpath raises "no path
    # specified" on them. They can never attribute to a module.
    if not filename or filename.startswith("<"):
        return None
    leaf_mod = _module_of_filename(filename)
    if leaf_mod is not None:
        return leaf_mod
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
    work counter that the metronome normalizes into ``cpu_share`` for
    UPDATE_MODULE_STATUS / GET_LOBE_TELEMETRY.
    """

    SAMPLE_HZ = 25.0

    def __init__(self, credit: Callable[[int, float], None]) -> None:
        """Initialize the sampler with the callback receiving module-time credit."""
        self._credit = credit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # thread ident → (id of leaf frame, f_lasti) at the last
        # sample. An identical pair twice in a row means the leaf has
        # not advanced — the thread is parked in a C-level wait, not
        # executing.
        self._prev: dict[int, tuple[int, int]] = {}

    def start(self) -> None:
        """Start the sampling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = named_thread("module-sampler", self._run, comm="g-sampler")
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampling thread to exit and join it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        """Sample running frames until stopped, crediting real elapsed.

        The credited dt is the measured interval between samples, not
        the nominal period — ``Event.wait`` only promises to block *at
        least* that long, so crediting the nominal value systematically
        undercounts the wall time the sampled frames actually occupied.
        """
        interval = 1.0 / self.SAMPLE_HZ
        last = time.monotonic()
        while not self._stop.wait(interval):
            now = time.monotonic()
            try:
                self._sample(now - last)
            except Exception as e:  # noqa: BLE001 — telemetry must never kill the loop
                note_swallowed("brainstem.module_sampler._sample", e)
            last = now

    def _sample(self, dt: float) -> None:
        """One sample tick: credit each *executing* thread's module dt.

        A thread is credited only when its leaf frame advanced since
        the previous sample — a different frame or a different
        instruction offset. C-level waits (``time.sleep``, a lock
        acquire, a blocking ``recv``) push no Python frame, so a
        parked thread keeps the identical leaf ``(frame, f_lasti)``
        for the whole wait — skipping it is what makes the
        docstring's "parked contributes nothing" actually true. The
        same shape covers long C-level compute (a big numpy call),
        which errs toward undercount — honest, and far smaller than
        crediting a 10 s park as 10 s of work.
        """
        current: dict[int, tuple[int, int]] = {}
        for ident, frame in sys._current_frames().items():
            leaf = (id(frame), frame.f_lasti)
            current[ident] = leaf
            if self._prev.get(ident) == leaf:
                continue  # leaf unmoved — parked, not executing
            module = _module_for_frame(frame)
            if module is not None:
                self._credit(module, dt)
        self._prev = current
