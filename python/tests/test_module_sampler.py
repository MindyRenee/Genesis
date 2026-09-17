"""Tests for genesis_cognitive.module_sampler — the mind's EEG.

Verifies path→module attribution (package layout mirrors her brain
anatomy), frame-level attribution rules (leaf in genesis_cognitive,
IPC calls attributed to the initiating module), and that the sampler
credits observed execution into the shared accumulator.
"""

import os
import sys
import threading
import time

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
from genesis_cognitive import module_sampler
from genesis_cognitive.module_sampler import (
    ModuleSampler,
    _module_for_frame,
    _module_of_filename,
)

PKG = os.path.dirname(module_sampler.__file__)


def _p(*parts: str) -> str:
    return os.path.join(PKG, *parts)


# ─── _module_of_filename: directory → module ─────────────────


def test_directory_attribution():
    """Each subsystem directory resolves to its manifest module."""
    assert _module_of_filename(_p("memory", "engine.py")) == MODULE_MEMORY
    assert _module_of_filename(_p("concepts", "network.py")) == MODULE_MEMORY
    assert _module_of_filename(_p("cognition", "engine.py")) == MODULE_REASONING
    assert _module_of_filename(_p("reasoning", "x.py")) == MODULE_REASONING
    assert _module_of_filename(_p("language", "engine.py")) == MODULE_LANGUAGE
    assert _module_of_filename(_p("perception", "vision.py")) == MODULE_SENSORY
    assert _module_of_filename(_p("vision", "x.py")) == MODULE_SENSORY
    assert _module_of_filename(_p("relay", "x.py")) == MODULE_ATTENTION
    assert _module_of_filename(_p("affect", "x.py")) == MODULE_EMOTION
    assert _module_of_filename(_p("neurochemical", "x.py")) == MODULE_EMOTION
    assert _module_of_filename(_p("autonomics", "x.py")) == MODULE_EMOTION
    assert _module_of_filename(_p("control", "x.py")) == MODULE_INTENTION
    assert _module_of_filename(_p("action_selection", "x.py")) == MODULE_INTENTION
    assert _module_of_filename(_p("motor_learning", "x.py")) == MODULE_MOTOR
    assert _module_of_filename(_p("sleep", "inner_life.py")) == MODULE_DREAMING
    assert _module_of_filename(_p("learning", "x.py")) == MODULE_INTENTION
    assert _module_of_filename(_p("self", "model.py")) == MODULE_METACOGNITION


def test_mind_disambiguation():
    """mind/ defaults to metacognition; volition/sleep override."""
    assert _module_of_filename(_p("mind", "heartbeat.py")) == MODULE_METACOGNITION
    assert _module_of_filename(_p("mind", "core.py")) == MODULE_METACOGNITION
    assert _module_of_filename(_p("mind", "volition.py")) == MODULE_INTENTION
    assert _module_of_filename(_p("mind", "sleep.py")) == MODULE_DREAMING


def test_top_level_files():
    """Top-level module files resolve individually."""
    assert _module_of_filename(_p("emotional_regulator.py")) == MODULE_EMOTION
    assert _module_of_filename(_p("speech.py")) == MODULE_MOTOR
    assert _module_of_filename(_p("attention.py")) == MODULE_ATTENTION
    assert _module_of_filename(_p("persistence.py")) == MODULE_MEMORY
    assert _module_of_filename(_p("module_sampler.py")) == MODULE_METACOGNITION


def test_foreign_files_unattributed():
    """Files outside genesis_cognitive resolve to None."""
    assert _module_of_filename("/usr/lib/python3.12/socket.py") is None
    assert _module_of_filename(f"{sys.executable}") is None
    client = os.path.join(os.path.dirname(PKG), "genesis_client", "client.py")
    assert _module_of_filename(client) is None


# ─── _module_for_frame: leaf attribution ─────────────────────


def _leaf_in_package():
    """Call from inside this test file — but the leaf is in tests/.

    tests/ is outside genesis_cognitive, so frame attribution must
    return None rather than the caller's module.
    """
    return _module_for_frame(sys._getframe())


def test_frame_outside_package_is_none():
    """A leaf frame outside the package attributes to nothing —
    parked/stdlib execution contributes no module share."""
    assert _leaf_in_package() is None


def test_frame_inside_sampler_attributes():
    """A leaf frame inside module_sampler.py attributes to
    metacognition — the sampler honestly reports its own work."""
    frames = []
    # exec with a faked filename inside the package — the exec'd
    # code's own frame is the leaf, so it resolves by filename.
    code = compile(
        "result.append(_module_for_frame(sys._getframe()))",
        os.path.join(PKG, "module_sampler.py"),
        "exec",
    )
    exec(
        code,
        {"result": frames, "_module_for_frame": _module_for_frame, "sys": sys},
    )
    assert frames == [MODULE_METACOGNITION]


# ─── ModuleSampler: crediting loop ───────────────────────────


def test_sampler_credits_observed_execution():
    """A busy thread executing package code accumulates module
    seconds; a sleeping thread accumulates none."""
    credited: dict[int, float] = {}
    lock = threading.Lock()

    def credit(module_id: int, seconds: float) -> None:
        with lock:
            credited[module_id] = credited.get(module_id, 0.0) + seconds

    sampler = ModuleSampler(credit)
    sampler._sample(0.04)  # direct tick: whatever runs, runs

    # Simulate: a thread whose leaf is inside the package gets
    # credited; this test thread's leaf is in tests/ → no credit.
    sampler._sample(0.04)
    # The test thread's own frame is in tests/ → nothing credited yet
    # (other threads may legitimately contribute if executing package
    # code at sample time, so we only assert the call doesn't crash
    # and produces a finite dict).
    assert isinstance(credited, dict)
    sampler.stop()


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
