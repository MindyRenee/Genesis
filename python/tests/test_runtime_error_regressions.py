"""The three recurring runtime errors, pinned as regressions.

All three were found by running her and reading the journal once the
swallow sink started carrying the exception. Each had been failing
silently for days or longer, and each disabled a real capability:

- the sleep-desync detector never ran at all (wrong object),
- one of the four inputs to her self-model was always empty,
- and "no camera" was tallied as a bug.

They are the argument for `swallow.py` recording the exception rather
than only a count: none of these was inferable from `site failed N
times`.
"""

from __future__ import annotations

from collections import deque

import pytest


def test_phase_info_has_no_emergent_phase() -> None:
    """The sleep-desync check read a field from the wrong object.

    `GenesisClient.get_phase()` returns `PhaseInfo`, which carries
    `phase` (an int code). `emergent_phase` belongs to `CoreState`. The
    check did `get_phase().emergent_phase`, which raised AttributeError —
    not caught by its `(OSError, ConnectionError, ValueError)` handler —
    so it escaped to the heartbeat on every wake. The "mind awake but
    daemon still asleep" fault it exists to record was never detected.
    """
    from genesis_client.types import PhaseInfo

    info = PhaseInfo(phase=5, arousal=0.2, valence=0.0)
    assert not hasattr(info, "emergent_phase"), (
        "if this ever gains the attribute, the sleep fix is masking a "
        "different field name — re-check _check_sleep_desync"
    )
    assert info.phase == 5


def test_sleep_desync_reads_the_phase_field(monkeypatch) -> None:
    """The reader must use the field `PhaseInfo` actually has."""
    import inspect

    from brainstem.sleep import SleepMixin

    src = inspect.getsource(SleepMixin._check_sleep_desync)
    assert "get_phase().phase" in src, (
        "_check_sleep_desync must read PhaseInfo.phase"
    )
    assert "get_phase().emergent_phase" not in src, (
        "emergent_phase is a CoreState field; reading it off PhaseInfo "
        "raises AttributeError and kills the desync check"
    )
    # And the handler must actually cover what it can now throw.
    assert "AttributeError" in src, (
        "the guard must catch AttributeError so a shape mismatch is "
        "reported as a swallowed error rather than escaping to the heartbeat"
    )


def test_curiosity_generate_questions_dereferences_its_emotion() -> None:
    """Passing None raises — so a None caller is a bug, not a style choice.

    `identity._synthesize_from_curiosity` passed `emotion=None` with a
    comment saying "handle gracefully". It does not: `generate_questions`
    reads the emotion, so every identity synthesis raised
    `AttributeError: 'NoneType' object has no attribute 'alertness'` and
    the curiosity source of her self-model was silently always empty.
    """
    from frontal_lobe.reasoning import Reasoner
    from limbic_system.curiosity import Curiosity
    from temporal_lobe.concepts import ConceptNetwork, RelationType

    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.1)
    curiosity = Curiosity(net, Reasoner(net))

    with pytest.raises(AttributeError):
        curiosity.generate_questions(emotion=None)  # type: ignore[arg-type]


def test_identity_curiosity_source_actually_produces_observations() -> None:
    """The source that used to always raise must now yield something."""
    from frontal_lobe.reasoning import Reasoner
    from frontal_lobe.self.identity import EmergentIdentity
    from limbic_system.curiosity import Curiosity
    from temporal_lobe.concepts import ConceptNetwork, RelationType

    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma", "delta"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.1)
    curiosity = Curiosity(net, Reasoner(net))
    # `questions_asked` counts retained question records; seed it so the
    # "deeply curious" branch is taken.
    curiosity._asked_questions = deque(
        [f"seeded-{i}" for i in range(120)], maxlen=200,
    )

    synth = EmergentIdentity.__new__(EmergentIdentity)
    source = synth._synthesize_from_curiosity(curiosity)

    assert source is not None
    # The signal is that it produced anything at all: previously the
    # emotion=None call raised before any observation could be recorded,
    # so this source was empty for the life of the process.
    assert source.observations, (
        "curiosity source produced no observations — generate_questions "
        "probably raised again"
    )


def test_curiosity_shared_state_is_thread_safe() -> None:
    """'deque mutated during iteration' crashed her inner-life cycle.

    The Curiosity engine is shared: the inner-life thread, the
    autonomous learner, the conversation path, and the workspace
    broadcast all reach it concurrently. Before the lock, one thread
    iterating `_concept_history` inside `_generate_cooccurrence_question`
    raced another thread's append in `generate_questions`, and CPython's
    deque mutation check raised RuntimeError mid-loop (journal,
    2026-10-16). `to_dict` had the same exposure against
    `_apply_habituation` (dict changed size during iteration) from the
    autosave thread.

    A large seeded history makes the combinations() scan the dominant
    cost of each call, so a racing append is all but guaranteed to land
    mid-iteration on an unfixed build.
    """
    import threading
    import types

    from frontal_lobe.reasoning import Reasoner
    from limbic_system.curiosity import Curiosity
    from limbic_system.emotion import EmotionalState
    from temporal_lobe.concepts import ConceptNetwork, RelationType

    net = ConceptNetwork()
    focus_names = ["apple", "river", "stone", "cloud", "bird"]
    for name in focus_names:
        net.add_concept(name, confidence=0.5)
    net.add_edge("apple", "river", RelationType.RELATED_TO, 0.1)
    curiosity = Curiosity(net, Reasoner(net))

    # Seed a wide co-occurrence history — letter-only names pass
    # is_world_concept, and ~31k pairs per window makes the deque
    # iteration the dominant cost, maximizing the race window.
    big_window = {
        f"seededconcept{chr(97 + i % 26)}{chr(97 + i // 26)}{chr(97 + i // 676)}"
        for i in range(250)
    }
    curiosity._concept_history.extend([set(big_window) for _ in range(4)])

    emotion = EmotionalState(
        label="curious",
        nuance="test",
        cognitive_style="test",
        alertness=0.5,
        valence=0.3,
        plasticity=0.9,
        creativity=0.5,
        caution=0.3,
        verbosity=1.0,
        formality=0.5,
        openness_to_engage=0.7,
    )

    errors: list[BaseException] = []

    def churn_questions() -> None:
        try:
            for _ in range(25):
                curiosity.generate_questions(
                    emotion, active_concepts=focus_names, max_questions=2
                )
        except Exception as e:  # noqa: BLE001 — collect, assert below
            errors.append(e)

    def churn_snapshot() -> None:
        try:
            for _ in range(25):
                curiosity.to_dict()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def churn_broadcast() -> None:
        item = types.SimpleNamespace(metadata={"topics": ["ocean", "whale"]})
        try:
            for _ in range(50):
                curiosity.receive_broadcast(item)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=churn_questions),
        threading.Thread(target=churn_questions),
        threading.Thread(target=churn_snapshot),
        threading.Thread(target=churn_broadcast),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert all(not t.is_alive() for t in threads), "churn threads wedged"
    assert not errors, (
        f"concurrent curiosity access raised {errors!r} — shared state "
        "must be serialized"
    )


def test_missing_camera_is_not_tallied_as_an_error() -> None:
    """A capability probe answering "absent" must not pollute the tally.

    `vision.is_available`, `recognition.is_available` and
    `object_recognition.is_available` each recorded a missing camera or
    model via `note_swallowed` — filing a documented, expected
    degradation beside genuine faults. 9, 8 and 5 occurrences
    respectively.
    """
    from occipital_lobe.perception.vision import Vision

    from genesis_client import swallow

    class _Vision:
        def __init__(self) -> None:
            self._available = None

        # The real body, with the frame source forced to fail the way an
        # absent camera does.
        is_available = Vision.is_available  # type: ignore[assignment,misc]

    import occipital_lobe.perception.vision as vision_mod

    def _no_frame(copy: bool = True):
        raise FileNotFoundError(2, "No such file or directory")

    original = vision_mod.latest_frame
    vision_mod.latest_frame = _no_frame
    site = "occipital_lobe.perception.vision.is_available"
    before = swallow.swallow_count(site)
    try:
        vis = _Vision()
        assert vis.is_available() is False  # type: ignore[misc]
    finally:
        vision_mod.latest_frame = original

    assert swallow.swallow_count(site) == before, (
        f"{site} tallied an expected 'no camera' as a swallowed error"
    )


def test_sleep_desync_runs_without_raising() -> None:
    """The check must survive a phase read — previously it always threw."""
    from genesis_client.types import PhaseInfo

    class _Client:
        def get_phase(self) -> PhaseInfo:
            return PhaseInfo(phase=5, arousal=0.2, valence=0.0)

    from brainstem.sleep import SleepMixin

    probe = SleepMixin.__new__(SleepMixin)
    probe._is_sleeping = False
    probe._wake_time = 0.0
    probe.client = _Client()  # type: ignore[attr-defined]

    # Must not raise AttributeError. The return value is not what is
    # under test; reaching a decision at all is.
    result = probe._check_sleep_desync()
    assert result in (True, False)
