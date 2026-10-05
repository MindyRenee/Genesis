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

    from genesis_conscious.mind.sleep import SleepMixin

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
    from genesis_conscious.concepts import ConceptNetwork, RelationType
    from genesis_conscious.learning.curiosity import CuriosityEngine
    from genesis_conscious.reasoning import ReasoningEngine

    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.1)
    curiosity = CuriosityEngine(net, ReasoningEngine(net))

    with pytest.raises(AttributeError):
        curiosity.generate_questions(emotion=None)  # type: ignore[arg-type]


def test_identity_curiosity_source_actually_produces_observations() -> None:
    """The source that used to always raise must now yield something."""
    from genesis_conscious.concepts import ConceptNetwork, RelationType
    from genesis_conscious.learning.curiosity import CuriosityEngine
    from genesis_conscious.reasoning import ReasoningEngine
    from genesis_conscious.self.identity import EmergentIdentity

    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma", "delta"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.1)
    curiosity = CuriosityEngine(net, ReasoningEngine(net))
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


def test_missing_camera_is_not_tallied_as_an_error() -> None:
    """A capability probe answering "absent" must not pollute the tally.

    `vision.is_available`, `recognition.is_available` and
    `object_recognition.is_available` each recorded a missing camera or
    model via `note_swallowed` — filing a documented, expected
    degradation beside genuine faults. 9, 8 and 5 occurrences
    respectively.
    """
    from genesis_client import swallow
    from genesis_conscious.perception.vision import Vision

    class _Vision:
        def __init__(self) -> None:
            self._available = None

        # The real body, with the frame source forced to fail the way an
        # absent camera does.
        is_available = Vision.is_available  # type: ignore[assignment,misc]

    import genesis_conscious.perception.vision as vision_mod

    def _no_frame(copy: bool = True):
        raise FileNotFoundError(2, "No such file or directory")

    original = vision_mod.latest_frame
    vision_mod.latest_frame = _no_frame
    site = "genesis_conscious.perception.vision.is_available"
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

    from genesis_conscious.mind.sleep import SleepMixin

    probe = SleepMixin.__new__(SleepMixin)
    probe._is_sleeping = False
    probe._wake_time = 0.0
    probe.client = _Client()  # type: ignore[attr-defined]

    # Must not raise AttributeError. The return value is not what is
    # under test; reaching a decision at all is.
    result = probe._check_sleep_desync()
    assert result in (True, False)
