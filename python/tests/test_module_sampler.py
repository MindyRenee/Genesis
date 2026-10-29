"""Tests for brainstem.module_sampler — the mind's EEG.

Verifies path→module attribution against real on-disk files — the
previous suite fabricated paths under the region root that never
existed (``infrastructure/`` inside ``python/``, a ``world/`` dir,
``mind.py`` beside the regions), so it asserted intent rather than
reality and half the attribution table went dead without a test
noticing. Frame-level attribution rules (leaf in the region tree,
IPC calls attributed to the initiating module) and the crediting
loop — including parked threads contributing nothing — are covered
the same way.
"""

import importlib.util
import os
import sys
import threading
import time
from collections.abc import Callable

from brainstem.module_sampler import (
    ModuleSampler,
    _module_for_frame,
    _module_of_filename,
)

from brainstem import module_sampler
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

PKG = module_sampler._PKG_DIR  # the region-tree root — python/ in this checkout


def _region(*parts: str) -> str:
    """A real file inside the region tree — must exist, else the test lies."""
    path = os.path.join(PKG, *parts)
    assert os.path.exists(path), f"attribution asserted on a file that is not there: {path}"
    return path


def _sibling(package: str, *parts: str) -> str:
    """A real file inside a support package beside the region tree."""
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.submodule_search_locations
    path = os.path.join(spec.submodule_search_locations[0], *parts)
    assert os.path.exists(path), f"attribution asserted on a file that is not there: {path}"
    return path


def _root_module(name: str) -> str:
    """Absolute path of an importable root module — must exist."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin and os.path.exists(spec.origin)
    return os.path.abspath(spec.origin)


# ─── _module_of_filename: directory → module ─────────────────


def test_region_directory_attribution():
    """Each region directory resolves to its manifest module."""
    cases = (
        (("memory", "memory.py"), MODULE_MEMORY),
        (("memory", "procedural.py"), MODULE_MEMORY),
        (("frontal_lobe", "autonomous.py"), MODULE_INTENTION),
        (("temporal_lobe", "morphology.py"), MODULE_LANGUAGE),
        (("occipital_lobe", "color.py"), MODULE_SENSORY),
        (("parietal_lobe", "agency.py"), MODULE_SENSORY),
        (("thalamus", "relay.py"), MODULE_ATTENTION),
        (("limbic_system", "amygdala.py"), MODULE_EMOTION),
        (("brainstem", "metronome.py"), MODULE_EMOTION),
        (("basal_ganglia", "basal_ganglia.py"), MODULE_INTENTION),
        (("cerebellum", "cerebellum.py"), MODULE_MOTOR),
        (("sleep", "inner_life.py"), MODULE_DREAMING),
        (("sleep", "compression.py"), MODULE_DREAMING),
        (("tools", "browser.py"), MODULE_INTENTION),
        (("self", "assessment.py"), MODULE_METACOGNITION),
        # The regions the audit found invisible: the homeostatic
        # nuclei, the interoceptive cortex, the forebrain modulator
        # sources — all felt substrate, so emotion.
        (("hypothalamus", "oxytocin.py"), MODULE_EMOTION),
        (("hypothalamus", "cortisol.py"), MODULE_EMOTION),
        (("insula", "somatic.py"), MODULE_EMOTION),
        (("basal_forebrain", "adenosine.py"), MODULE_EMOTION),
        (("basal_forebrain", "acetylcholine.py"), MODULE_EMOTION),
        # The world model — the table used to key a nonexistent
        # ``world/`` dir while the live package is ``worlds/``.
        (("worlds", "world.py"), MODULE_REASONING),
        (("worlds", "board.py"), MODULE_REASONING),
    )
    for parts, expected in cases:
        assert _module_of_filename(_region(*parts)) == expected, parts


def test_nested_subsystem_directories():
    """A subsystem filed inside a region keeps its own attribution —
    the immediate directory names the module's home."""
    assert _module_of_filename(_region("frontal_lobe", "reasoning", "logic.py")) == MODULE_REASONING
    assert (
        _module_of_filename(_region("occipital_lobe", "perception", "vision.py"))
        == MODULE_SENSORY
    )
    assert (
        _module_of_filename(_region("temporal_lobe", "concepts", "network.py"))
        == MODULE_MEMORY
    )


def test_region_root_topology():
    """python/topology.py — the organism's wiring map — attributes to
    metacognition, while the same basename inside a region keeps the
    region's own attribution."""
    assert _module_of_filename(_region("topology.py")) == MODULE_METACOGNITION
    assert (
        _module_of_filename(_region("temporal_lobe", "concepts", "topology.py"))
        == MODULE_MEMORY
    )


def test_file_pins_override_region():
    """A named module file keeps its own attribution wherever filed."""
    cases = (
        (("frontal_lobe", "answer_composer.py"), MODULE_LANGUAGE),
        (("frontal_lobe", "status.py"), MODULE_METACOGNITION),
        (("frontal_lobe", "volition.py"), MODULE_INTENTION),
        (("temporal_lobe", "speech.py"), MODULE_MOTOR),
        (("temporal_lobe", "stdp.py"), MODULE_MEMORY),
        (("temporal_lobe", "plasticity.py"), MODULE_MEMORY),
        (("temporal_lobe", "dual.py"), MODULE_MEMORY),
        (("temporal_lobe", "synapses.py"), MODULE_MEMORY),
        (("thalamus", "brain_waves.py"), MODULE_METACOGNITION),
        (("limbic_system", "emotional_regulator.py"), MODULE_EMOTION),
        (("limbic_system", "emotion.py"), MODULE_EMOTION),
        (("brainstem", "module_sampler.py"), MODULE_METACOGNITION),
        (("sleep", "sleep.py"), MODULE_DREAMING),
        (("basal_ganglia", "volition.py"), MODULE_INTENTION),
        (("parietal_lobe", "abstract_reasoning.py"), MODULE_REASONING),
        (("parietal_lobe", "attention.py"), MODULE_ATTENTION),
    )
    for parts, expected in cases:
        assert _module_of_filename(_region(*parts)) == expected, parts


def test_support_packages_attributed():
    """infrastructure/ and eval/ live beside python/ — the sampler
    resolves them by import, not by region relpath."""
    persistence = _sibling("infrastructure", "persistence.py")
    assert _module_of_filename(persistence) == MODULE_MEMORY
    codebook = _sibling("infrastructure", "vq_codebook.py")
    assert _module_of_filename(codebook) == MODULE_MEMORY
    # An infrastructure file without a pin is support machinery —
    # metacognition by default.
    config = _sibling("infrastructure", "config.py")
    assert _module_of_filename(config) == MODULE_METACOGNITION
    harness = _sibling("eval", "harness.py")
    assert _module_of_filename(harness) == MODULE_METACOGNITION


def test_root_modules_attributed():
    """Organism-level modules beside the region tree resolve to their
    own pins — the assembled organ, the turn orchestration, the CLI
    shepherd, the boundary machinery, the public ear."""
    assert _module_of_filename(_root_module("mind")) == MODULE_METACOGNITION
    assert _module_of_filename(_root_module("turn")) == MODULE_METACOGNITION
    assert _module_of_filename(_root_module("genesis_cli")) == MODULE_METACOGNITION
    assert _module_of_filename(_root_module("commitment")) == MODULE_INTENTION
    assert _module_of_filename(_root_module("discord_feed")) == MODULE_SENSORY


def test_foreign_files_unattributed():
    """Files outside the region tree resolve to None — stdlib, the
    test suite itself, and the doc-only _views namespace contribute
    no module share."""
    assert _module_of_filename("/usr/lib/python3.12/socket.py") is None
    assert _module_of_filename(f"{sys.executable}") is None
    # The IPC client is not a module — _module_for_frame walks back
    # from a client leaf to the cognitive caller instead.
    assert _module_of_filename(_sibling("genesis_client", "client.py")) is None
    # This test file lives under tests/ — not a brain part.
    assert _module_of_filename(os.path.abspath(__file__)) is None
    assert _module_of_filename(_region("_views", "__init__.py")) is None


# ─── _module_for_frame: leaf attribution ─────────────────────


def _leaf_in_package():
    """Call from inside this test file — the leaf is in tests/."""
    return _module_for_frame(sys._getframe())


def test_frame_outside_package_is_none():
    """A leaf frame outside the package attributes to nothing —
    parked/stdlib execution contributes no module share."""
    assert _leaf_in_package() is None


def test_pseudo_filenames_unattributed():
    """Frames with synthetic filenames (<string>, frozen bootstrap)
    can never attribute — relpath would raise on them."""
    frames = []
    exec(
        compile(
            "result.append(_module_for_frame(sys._getframe()))",
            "<string>",
            "exec",
        ),
        {"result": frames, "_module_for_frame": _module_for_frame, "sys": sys},
    )
    assert frames == [None]


def test_frame_inside_sampler_attributes():
    """A leaf frame inside module_sampler.py attributes to
    metacognition — the sampler honestly reports its own work."""
    frames = []
    code = compile(
        "result.append(_module_for_frame(sys._getframe()))",
        _region("brainstem", "module_sampler.py"),
        "exec",
    )
    exec(
        code,
        {"result": frames, "_module_for_frame": _module_for_frame, "sys": sys},
    )
    assert frames == [MODULE_METACOGNITION]


def test_ipc_leaf_attributes_to_cognitive_caller():
    """A leaf inside genesis_client (an IPC call in flight) walks back
    to the deepest cognitive frame — the module that initiated the
    call owns the wait."""
    frames = []
    # A function "in" metronome.py that makes the IPC call; the leaf
    # lands inside the client and must resolve to the caller's module.
    metronome_src = (
        "def ipc_caller():\n"
        "    code = compile(\n"
        "        'result.append(_module_for_frame(sys._getframe()))',\n"
        "        client_file, 'exec')\n"
        "    exec(code, ns)\n"
    )
    ns = {
        "result": frames,
        "_module_for_frame": _module_for_frame,
        "sys": sys,
        "client_file": _sibling("genesis_client", "client.py"),
    }
    ns["ns"] = ns
    exec(compile(metronome_src, _region("brainstem", "metronome.py"), "exec"), ns)
    ns["ipc_caller"]()
    assert frames == [MODULE_EMOTION]


# ─── ModuleSampler: crediting loop ───────────────────────────


def _credits() -> tuple[dict[int, float], Callable[[int, float], None]]:
    credited: dict[int, float] = {}
    lock = threading.Lock()

    def credit(module_id: int, seconds: float) -> None:
        with lock:
            credited[module_id] = credited.get(module_id, 0.0) + seconds

    return credited, credit


def test_parked_thread_contributes_nothing():
    """A thread blocked in a C-level wait keeps the same leaf
    (frame, f_lasti) sample after sample — it is credited on first
    observation, then never again while it stays parked.

    The wait is ``time.sleep``: a Python-level wait like
    ``Event.wait`` would push a stdlib frame as the leaf and
    attribute to nothing anyway, but a C-level call leaves the
    *calling* region frame as the leaf — the case the sampler's
    unmoved-leaf check exists for.
    """
    credited, credit = _credits()
    sampler = ModuleSampler(credit)
    stop_flag = [False]
    # The worker's frame carries a real region filename — a parked
    # wait inside volition must not bank intention work.
    worker_src = "time.sleep(1.5)\nfor _ in range(10 ** 8):\n    if stop_flag[0]: break\n"
    code = compile(worker_src, _region("basal_ganglia", "volition.py"), "exec")
    worker = threading.Thread(
        target=lambda: exec(code, {"time": time, "stop_flag": stop_flag}),
        daemon=True,
    )
    worker.start()
    try:
        # Let the worker reach the wait — its first observation
        # credits once, since a fresh leaf cannot be told parked yet.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            sampler._sample(0.04)
            if credited.get(MODULE_INTENTION):
                break
            time.sleep(0.01)
        parked_credit = credited.get(MODULE_INTENTION, 0.0)
        assert parked_credit > 0.0

        # Still asleep: further samples must add nothing.
        for _ in range(4):
            sampler._sample(0.04)
        assert credited.get(MODULE_INTENTION, 0.0) == parked_credit

        # The sleep expires, real work resumes — the leaf moves and
        # credit accrues again.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            sampler._sample(0.04)
            if credited.get(MODULE_INTENTION, 0.0) > parked_credit:
                break
            time.sleep(0.02)
        assert credited.get(MODULE_INTENTION, 0.0) > parked_credit
    finally:
        stop_flag[0] = True
        worker.join(timeout=5)


def test_dead_threads_drop_out_of_tracking():
    """A thread that exits leaves no stale tracking entry — the
    previous-sample map is rebuilt each tick."""
    sampler = ModuleSampler(lambda m, s: None)
    worker = threading.Thread(target=lambda: None, daemon=True)
    worker.start()
    worker.join(timeout=5)
    sampler._sample(0.04)
    assert worker.ident not in sampler._prev


def test_sampler_thread_lifecycle():
    """start() is idempotent and stop() joins the thread."""
    sampler = ModuleSampler(lambda m, s: None)
    sampler.start()
    first = sampler._thread
    sampler.start()  # idempotent — same thread
    assert sampler._thread is first
    assert first is not None and first.is_alive()
    sampler.stop()
    assert not first.is_alive()
    time.sleep(0)  # let the thread fully exit
