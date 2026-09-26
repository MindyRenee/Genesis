"""Mind lifecycle — save/load, teaching mode, meditation pause."""

import logging
import os
import sys
import tempfile
import time

import pytest

from genesis_cognitive.concepts import RelationType
from genesis_cognitive.mind import Mind

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_mind_lifecycle.py
# ======================================================================

logger = logging.getLogger(__name__)


def test_mind_lifecycle_learner_persistence():
    """Learner state survives a full Mind save/restore cycle."""
    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")

        # ── Session 1: Create Mind, teach it, save ──
        mind1 = Mind(socket_path)
        # Don't call start() — it needs the daemon. We test
        # _save_state/_load_saved_state directly.
        learner = mind1.cognition.self_learner
        learner.learn_from_input("A dog is a mammal.")
        learner.learn_from_input("Cryptobiosis is defined as a state of suspended animation.")
        learner.learn_from_input("A spider has eight legs.")
        learner.run_inference_cycle(force=True)

        stats1 = learner.get_stats()
        assert stats1["conversation_facts"] > 0
        assert stats1["definitions_synthesized"] > 0

        # Save state
        mind1._save_state()

        # Verify the save file exists
        state_path = os.path.join(data_dir, "cognitive_state.json")
        assert os.path.exists(state_path)

        # ── Session 2: Create new Mind, load state, verify ──
        mind2 = Mind(socket_path)
        mind2._load_saved_state()

        learner2 = mind2.cognition.self_learner
        stats2 = learner2.get_stats()

        # Learner stats should match
        assert stats2 == stats1, f"Stats mismatch: {stats2} vs {stats1}"

        # Concept network should have the learned concepts
        net = mind2.cognition.network
        assert net.get_concept("dog") is not None
        assert net.get_concept("cryptobiosis") is not None
        assert net.get_concept("spider") is not None

        # Definition should be preserved
        c = net.get_concept("cryptobiosis")
        assert "definition" in c.properties
        assert "suspended animation" in c.properties["definition"].lower()

        # IS_A edge should be preserved
        edges = net.get_edges("dog", "out")
        is_a = [e for e in edges if e.relation == RelationType.IS_A]
        assert len(is_a) >= 1
        assert is_a[0].target == "mammal"

        # ── Session 2: Continue learning ──
        new_events = learner2.learn_from_input("A cat is a mammal.")
        assert len(new_events) > 0
        stats3 = learner2.get_stats()
        assert stats3["conversation_facts"] > stats2["conversation_facts"]


def test_failed_late_restore_does_not_commit_network(tmp_path):
    """A malformed later component must not partially publish a save."""
    import gzip
    import json

    from genesis_cognitive.persistence import load_state

    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()

    source = Mind(str(source_dir / "genesis.sock"))
    source.cognition.network.add_concept("restored_marker", "concept")
    source._save_state()

    target = Mind(str(target_dir / "genesis.sock"))
    target.cognition.network.add_concept("preexisting_marker", "concept")

    data = load_state(str(source_dir))
    assert data is not None
    data["plasticity"] = []
    (target_dir / "cognitive_state.json").write_bytes(
        gzip.compress(json.dumps(data).encode())
    )

    with pytest.raises(OSError, match="Cannot restore cognitive_state"):
        target._load_saved_state()
    assert target.cognition.network.get_concept("preexisting_marker") is not None
    assert target.cognition.network.get_concept("restored_marker") is None


def test_mind_lifecycle_backward_compat():
    """Old save files without learner state don't crash on load."""
    import json

    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")

        # Create a minimal old-format save file (no self_directed_learner key)
        state_path = os.path.join(data_dir, "cognitive_state.json")
        old_state = {
            "version": 2,
            "concept_network": {"concepts": [], "edges": []},
            "reflection": {
                "interaction_count": 0,
                "insights": [],
                "mood_history": [],
                "response_patterns": [],
            },
            "narrative": {"chapters": [], "personality_snapshots": []},
            "self_model": {"personality": {}, "self_knowledge": []},
        }
        with open(state_path, "w") as f:
            json.dump(old_state, f)

        # Loading should not crash
        mind = Mind(socket_path)
        mind._load_saved_state()

        # Learner should have default stats
        stats = mind.cognition.self_learner.get_stats()
        assert stats["total_inferences"] == 0
        assert stats["conversation_facts"] == 0


def test_mind_lifecycle_learner_log_persistence():
    """The learning log (event history) survives save/restore."""
    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")

        mind1 = Mind(socket_path)
        learner = mind1.cognition.self_learner
        learner.learn_from_input("A dog is a mammal.")
        learner.learn_from_input("A cat is a mammal.")
        learner.learn_from_input("Entropy means a measure of disorder.")

        log_before = list(learner._learning_log)
        assert len(log_before) >= 3

        mind1._save_state()

        mind2 = Mind(socket_path)
        mind2._load_saved_state()

        learner2 = mind2.cognition.self_learner
        log_after = list(learner2._learning_log)

        assert len(log_after) == len(log_before)
        for before, after in zip(log_before, log_after, strict=True):
            assert before.event_type == after.event_type
            assert before.description == after.description


# ─── Test runner ──────────────────────────────────────────────


def run_all():
    """Run all."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    logger.info(f"\n  Mind lifecycle tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)


# ======================================================================
# From tests/test_teaching_mode.py
# ======================================================================

logger = logging.getLogger(__name__)


class _MockClient:
    """Mock GenesisClient that records set_zone calls."""

    def __init__(self) -> None:
        """Initialize the mock client."""
        self.zones: list[int] = []

    def neuro_impulse(self, chem: int, magnitude: float, **kwargs) -> bool:
        """Neuro impulse."""
        return True

    def set_zone(self, zone: int, **kwargs) -> bool:
        """Set zone."""
        self.zones.append(zone)
        return True


def _make_mind(data_dir: str) -> Mind:
    """Create a Mind with a mock client for testing teaching mode."""
    socket_path = os.path.join(data_dir, "genesis.sock")
    mind = Mind(socket_path)
    mind.client = _MockClient()  # type: ignore[assignment]
    return mind


def test_enter_teaching_mode_sets_flag():
    """enter_teaching_mode() should set _is_teaching and the topic."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        assert not mind.is_teaching

        mind.enter_teaching_mode("neuroscience")

        assert mind.is_teaching
        assert mind.teaching_topic == "neuroscience"
        assert mind.cognition.teaching_mode
        assert mind.cognition.teaching_topic == "neuroscience"


def test_enter_teaching_mode_without_topic():
    """enter_teaching_mode() with no topic should still work."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind.enter_teaching_mode()
        assert mind.is_teaching
        assert mind.teaching_topic == ""
        assert mind.cognition.teaching_mode


def test_exit_teaching_mode_clears_flag():
    """exit_teaching_mode() should clear _is_teaching and the topic."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind.enter_teaching_mode("Python")
        assert mind.is_teaching

        mind.exit_teaching_mode()

        assert not mind.is_teaching
        assert mind.teaching_topic == ""
        assert not mind.cognition.teaching_mode
        assert mind.cognition.teaching_topic == ""


def test_exit_teaching_mode_when_not_teaching_is_noop():
    """exit_teaching_mode() when not teaching should be a no-op."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind.exit_teaching_mode()  # should not raise
        assert not mind.is_teaching


def test_teaching_mode_pauses_inner_life_and_learner():
    """enter_teaching_mode() should pause inner_life and the autonomous learner."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        assert not mind.inner_life._paused
        assert not mind.learner._paused

        mind.enter_teaching_mode("biology")

        assert mind.inner_life._paused, "inner_life should be paused during teaching"
        assert mind.learner._paused, "learner should be paused during teaching"


def test_resume_background_does_not_resume_during_teaching():
    """_resume_background() must not resume inner_life/learner if teaching."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind.enter_teaching_mode("physics")
        assert mind.inner_life._paused
        assert mind.learner._paused

        mind._resume_background()

        assert mind.is_teaching, "should still be teaching"
        assert mind.inner_life._paused, (
            "inner_life should remain paused — teaching takes precedence"
        )
        assert mind.learner._paused, (
            "learner should remain paused — teaching takes precedence"
        )


def test_exit_teaching_mode_resumes_background():
    """exit_teaching_mode() resumes learner; inner_life and volition
    resume via the quiet-window path (idle user → immediately)."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind.enter_teaching_mode("history")
        assert mind.inner_life._paused
        assert mind.learner._paused

        # No active conversation — the focus window has lapsed.
        mind._last_interaction_time -= mind.CONVERSATION_FOCUS_SECONDS + 1
        mind.exit_teaching_mode()

        assert not mind.is_teaching
        assert not mind.learner._paused, "learner should resume after teaching"
        assert not mind.inner_life._paused, "inner_life should resume after teaching"
        assert not mind._suppress_volition, "volition should release after teaching"


def test_exit_teaching_mode_holds_background_while_engaged():
    """Exiting teaching mid-conversation shouldn't unleash stray urges —
    inner life and volition wait for the user to go quiet."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind.enter_teaching_mode("history")
        mind._suppress_volition = True
        mind._last_interaction_time = time.time()  # user just spoke

        mind.exit_teaching_mode()

        assert not mind.is_teaching
        assert not mind.learner._paused, "learner still resumes on exit"
        assert mind.inner_life._paused, "inner_life waits for quiet"
        assert mind._suppress_volition, "volition stays held while engaged"

        # Once they go quiet, the deferred check releases both.
        mind._last_interaction_time -= mind.CONVERSATION_FOCUS_SECONDS + 1
        mind._resume_inner_life()
        assert not mind.inner_life._paused
        assert not mind._suppress_volition


def test_teaching_mode_updates_topic_when_already_teaching():
    """enter_teaching_mode() while already teaching should update the topic."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind.enter_teaching_mode("math")
        assert mind.teaching_topic == "math"

        mind.enter_teaching_mode("calculus")
        assert mind.teaching_topic == "calculus"
        assert mind.cognition.teaching_topic == "calculus"
        assert mind.is_teaching


def test_curiosity_filtering_in_teaching_mode():
    """In teaching mode, only lesson-related curiosity questions should be asked."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        cognition = mind.cognition

        # Add some concepts to the network
        net = cognition.network
        net.add_concept("neuron", origin="dictionary")
        net.add_concept("synapse", origin="dictionary")
        net.add_concept("pizza", origin="dictionary")
        # Add an edge so neuron and synapse are neighbors
        from genesis_cognitive.concepts import RelationType
        net.add_edge("neuron", "synapse", relation=RelationType.RELATED_TO)

        # Simulate teaching mode
        cognition.teaching_mode = True
        cognition.teaching_topic = "neuroscience"

        # Test the lesson topic helper
        from genesis_cognitive.perception import Intent, Perception, QuestionType
        perception = Perception(
            raw_text="Tell me about neurons",
            intent=Intent.STATEMENT,
            question_type=QuestionType.NONE,
            topics=["neuron"],
            sentiment=0.0,
            sentiment_label="neutral",
            emotion_word="",
            is_about_genesis=False,
            is_about_user=False,
            is_about_code=False,
            is_about_emotion=False,
            is_about_existence=False,
            entities={},
            key_phrases=[],
            word_count=4,
            confidence=0.8,
        )
        lesson_topics = cognition._teaching_lesson_topics(perception)
        assert "neuron" in lesson_topics
        assert "neuroscience" in lesson_topics

        # Test concept-relates-to-lesson helper
        assert cognition._concept_relates_to_lesson("neuron", "neuron")
        assert cognition._concept_relates_to_lesson("synapse", "neuron")
        assert not cognition._concept_relates_to_lesson("pizza", "neuroscience")


def test_pick_creation_topic_skips_internal_namespaces():
    """Internal ``:``-namespaced concepts are never project topics.

    Regression: ``_cat:cause:guarded_cortisol`` (a structural marker)
    was picked from the network, sanitized into the fused garbage
    name ``catcauseguarded_cortisol``, and leaked into user-facing
    output as a spoken word and a project directory. Every
    colon-namespaced ID is internal machinery — utterance seeds
    (``_cat:``/``_utt:``), code symbols (``python:``/``rust:``),
    task markers (``skill:``/``goal:``/``domain:``/``spatial:``).
    """
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        net = mind.cognition.network

        # Internal namespaces present in the live concept network.
        for internal in (
            "_cat:cause:guarded_cortisol",
            "_utt:hello",
            "python:genesis_cognitive.concepts",
            "rust:daemon::tick",
            "skill:sorter",
            "goal:explore",
            "domain:grid",
            "spatial:cell",
        ):
            net.add_concept(internal, confidence=0.9, origin="structural")
            mind.learner._curiosity_queue.append(internal)

        # One legitimate user-facing topic — high confidence.
        net.add_concept("ferret", confidence=0.9, origin="learned")

        for _ in range(20):
            topic = mind._pick_creation_topic()
            assert topic is not None
            assert ":" not in topic, (
                f"internal namespaced topic picked: {topic!r}"
            )
            # Never any of the internal symbols, from queue or network.
            assert not topic.startswith((
                "_cat:", "_utt:", "python:", "rust:", "man:",
                "wikipedia:", "wordnet:", "skill:", "goal:",
                "domain:", "spatial:", "var:", "type:",
            ))


def test_notification_thread_exits_within_join_budget():
    """The notification poller must exit well inside the 5s join budget.

    Regression: the loop slept a monolithic ``time.sleep(10.0)`` per
    cycle while shutdown joins with a 5s deadline — a Ctrl-C landing
    >5s into a sleep logged "notifications thread did not exit" and
    reported the shutdown as incomplete even though the state save
    succeeded. The loop now sleeps in 1s increments so it observes
    ``_running = False`` within ~1s.
    """
    import threading
    import time

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind(data_dir)
        mind._running = True
        thread = threading.Thread(
            target=mind._notification_loop,
            name="subcognitive-notifications",
            daemon=True,
        )
        thread.start()
        try:
            # Let it settle into its sleep cycle, then stop it.
            time.sleep(0.3)
            mind._running = False
            assert mind._join_shutdown_thread(thread, "notifications")
            assert not thread.is_alive()
        finally:
            mind._running = False
            thread.join(timeout=5.0)


# ======================================================================
# From tests/test_meditation_pause.py
# ======================================================================

logger = logging.getLogger(__name__)


class _MockClientMeditation:
    """Mock GenesisClient that records neuro_impulse and set_zone calls."""

    def __init__(self) -> None:
        """Initialize the mock client meditation."""
        self.impulses: list[tuple[int, float]] = []
        self.zones: list[int] = []

    def neuro_impulse(self, chem: int, magnitude: float, **kwargs) -> bool:
        """Neuro impulse."""
        self.impulses.append((chem, magnitude))
        return True

    def set_zone(self, zone: int, **kwargs) -> bool:
        """Set zone."""
        self.zones.append(zone)
        return True


def _make_mind_meditation(data_dir: str) -> Mind:
    """Create a Mind with a mock client for testing meditation."""
    socket_path = os.path.join(data_dir, "genesis.sock")
    mind = Mind(socket_path)
    mind.client = _MockClientMeditation()  # type: ignore[assignment]
    return mind


def test_meditate_pauses_inner_life_and_learner():
    """meditate() should pause both inner_life and the autonomous learner."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_meditation(data_dir)
        assert not mind.is_meditating
        assert not mind.inner_life._paused
        assert not mind.learner._paused

        mind.meditate()

        assert mind.is_meditating
        assert mind.inner_life._paused, "inner_life should be paused during meditation"
        assert mind.learner._paused, "learner should be paused during meditation"


def test_resume_background_does_not_resume_during_meditation():
    """_resume_background() must not resume inner_life/learner if meditating.

    This is the regression test for the bug where a delayed
    _resume_background() timer (from a prior conversation) fired during
    meditation and undid the pause.
    """
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_meditation(data_dir)
        mind.meditate()
        assert mind.inner_life._paused
        assert mind.learner._paused

        # Simulate the delayed timer firing during meditation
        mind._resume_background()

        assert mind.is_meditating, "should still be meditating"
        assert mind.inner_life._paused, (
            "inner_life should remain paused — meditation takes precedence"
        )
        assert mind.learner._paused, (
            "learner should remain paused — meditation takes precedence"
        )


def test_resume_background_does_not_resume_during_sleep():
    """_resume_background() must not resume inner_life/learner if sleeping."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_meditation(data_dir)
        mind.sleep(user_initiated=True)
        assert mind.is_sleeping

        # Pause inner_life/learner as a conversation would, then try to resume
        mind.inner_life.pause()
        mind.learner.pause()
        assert mind.inner_life._paused
        assert mind.learner._paused

        mind._resume_background()

        assert mind.is_sleeping, "should still be sleeping"
        assert mind.inner_life._paused, (
            "inner_life should remain paused — sleep takes precedence"
        )
        assert mind.learner._paused, (
            "learner should remain paused — sleep takes precedence"
        )


def test_resume_background_resumes_when_awake():
    """_resume_background() should resume normally when not meditating/sleeping."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_meditation(data_dir)
        # Simulate a conversation pause — set last interaction far
        # enough in the past that the conversation focus window has
        # elapsed and inner_life should resume immediately.
        mind._last_interaction_time = 0.0
        # Simulate a conversation pause
        mind.inner_life.pause()
        mind.learner.pause()
        assert mind.inner_life._paused
        assert mind.learner._paused

        mind._resume_background()

        assert not mind.is_meditating
        assert not mind.is_sleeping
        assert not mind.inner_life._paused, "inner_life should be resumed"
        assert not mind.learner._paused, "learner should be resumed"


def test_wake_from_meditation_resumes_background():
    """wake_from_meditation() should resume inner_life and learner."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_meditation(data_dir)
        mind.meditate()
        assert mind.inner_life._paused
        assert mind.learner._paused

        mind.wake_from_meditation()

        assert not mind.is_meditating
        assert not mind.inner_life._paused, "inner_life should resume after meditation"
        assert not mind.learner._paused, "learner should resume after meditation"


def test_restore_sleep_state_engages_sleep_mechanism():
    """Restoring a saved sleep state must engage the sleep mechanism.

    Regression test for the bug where a restart during sleep restored
    the ``_is_sleeping`` flag and the zone, but never paused the
    learners or put inner life into sleep mode — so it was flagged
    asleep yet behaved awake (still learning, generating waking
    thoughts). Re-issuing /sleep could not recover this because
    ``sleep()`` no-ops when ``_is_sleeping`` is already True.

    The fix: ``_start_engage_restored_sleep()`` calls
    ``_engage_sleep_mechanism()`` after the autonomous subsystems are
    started, pausing the learners so it actually sleeps.
    """
    with tempfile.TemporaryDirectory() as data_dir:
        # ── Session 1: put it to sleep and save ──
        mind1 = _make_mind_meditation(data_dir)
        mind1.sleep(user_initiated=True)
        assert mind1.is_sleeping
        assert mind1.learner._paused, "learner should be paused after sleep()"
        mind1._save_state()

        # ── Session 2: load the saved state (simulates restart) ──
        mind2 = _make_mind_meditation(data_dir)
        mind2._load_saved_state()

        # The restore sets the flag but does NOT engage the mechanism —
        # that's the job of _start_engage_restored_sleep(), called later
        # in the startup sequence after the autonomous subsystems start.
        assert mind2.is_sleeping, "sleep flag should be restored"
        assert not mind2.learner._paused, (
            "learner should NOT be paused yet — engagement happens in "
            "_start_engage_restored_sleep(), not _load_saved_state()"
        )

        # ── Engage the sleep mechanism (the fix) ──
        mind2._start_engage_restored_sleep()

        assert mind2.learner._paused, (
            "learner should be paused after _start_engage_restored_sleep()"
        )
        assert mind2.cognition.self_learner._paused, (
            "self-directed learner should be paused after restore engagement"
        )
        assert not mind2.inner_life._paused, (
            "inner_life should be running (in sleep mode) after restore "
            "engagement — it generates dreams, not waking thoughts"
        )


def test_engage_restored_sleep_noop_when_awake():
    """_start_engage_restored_sleep() must do nothing when not sleeping."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_meditation(data_dir)
        assert not mind.is_sleeping
        assert not mind.learner._paused

        mind._start_engage_restored_sleep()

        assert not mind.is_sleeping
        assert not mind.learner._paused, (
            "learner should not be paused when restoring an awake state"
        )
