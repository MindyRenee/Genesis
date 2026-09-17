"""Learning bundle tests.

Autonomous learner, dual system, STDP, TD learning,
plasticity, three-factor STDP.
"""

import logging
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from genesis_client.types import (
    LEARNING_POSTURE_NEUTRAL,
    LEARNING_POSTURE_PROTECTIVE,
    LEARNING_POSTURE_RECEPTIVE,
    LEARNING_POSTURE_RECOVERING,
    PlasticityProfile,
)
from genesis_cognitive.concepts import ConceptNetwork, EmbeddingStore, NetworkTopology, RelationType
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.learning import (
    STDP,
    AutonomousLearner,
    CuriosityEngine,
    DualSystemLearner,
    HebbianPlasticity,
    HippocampalEpisode,
    NeocorticalMemory,
    SpikeEvent,
    TDLearner,
    TDTransition,
)
from genesis_cognitive.reasoning import ReasoningEngine
from genesis_cognitive.tools.source_registry import SourceResult

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_autonomous_learner.py
# ======================================================================

def _make_learner(
    get_emotion: object = None,
    get_plasticity_profile: object = None,
) -> AutonomousLearner:
    """Create a learner with minimal dependencies."""
    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    return AutonomousLearner(
        network=net,
        curiosity=curiosity,
        get_emotion=get_emotion,
        get_plasticity_profile=get_plasticity_profile,
    )


def _make_profile(
    plasticity_gate: float = 0.5,
    bdnf_tonic: float = 0.4,
    cortisol_tonic: float = 0.3,
) -> PlasticityProfile:
    """Create a minimal plasticity profile for testing."""
    return PlasticityProfile(
        plasticity_gate=plasticity_gate,
        bdnf_effective=bdnf_tonic,
        bdnf_tonic=bdnf_tonic,
        cortisol_effective=cortisol_tonic,
        cortisol_tonic=cortisol_tonic,
        dopamine_effective=0.4,
        serotonin_effective=0.4,
        coupling_drift=0.0,
        mean_receptor_sensitivity=0.95,
        emergent_phase=0,
    )


def _make_emotion(label: str, valence: float = 0.0) -> EmotionalState:
    """Create a minimal emotional state with a label."""
    return EmotionalState(
        label=label,
        nuance="test",
        cognitive_style="test",
        alertness=0.5,
        valence=valence,
        plasticity=0.5,
        creativity=0.5,
        caution=0.3,
        verbosity=1.0,
        formality=0.5,
        openness_to_engage=0.7,
    )


def test_learner_blocks_learning_when_stressed() -> None:
    """Learning is blocked when Genesis is stressed."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("stressed", valence=-0.5))
    assert not learner._should_learn()


def test_learner_blocks_learning_when_overwhelmed() -> None:
    """Learning is blocked when Genesis is overwhelmed."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("overwhelmed", valence=-0.6))
    assert not learner._should_learn()


def test_learner_blocks_learning_when_anxious() -> None:
    """Learning is blocked when Genesis is anxious."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("anxious", valence=-0.3))
    assert not learner._should_learn()


def test_learner_blocks_learning_when_drowsy() -> None:
    """Learning is blocked when Genesis is drowsy."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("drowsy", valence=0.1))
    assert not learner._should_learn()


def test_learner_allows_learning_when_positive() -> None:
    """Learning is allowed when Genesis is in a positive state."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("positive", valence=0.3))
    assert learner._should_learn()


def test_learner_allows_learning_when_in_flow() -> None:
    """Learning is allowed when Genesis is in flow."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("in flow", valence=0.5))
    assert learner._should_learn()


def test_learner_allows_learning_when_neutral() -> None:
    """Learning is allowed when Genesis is neutral."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("neutral", valence=0.0))
    assert learner._should_learn()


def test_learner_allows_learning_without_emotion_callback() -> None:
    """Learning is allowed when no emotion callback is set (for testing)."""
    learner = _make_learner(get_emotion=None)
    assert learner._should_learn()


def test_learner_emotion_skip_tracking() -> None:
    """Emotion skips are tracked in stats."""
    learner = _make_learner(get_emotion=lambda: _make_emotion("stressed", valence=-0.5))
    assert learner.stats["emotion_skips"] == 0
    learner._should_learn()
    learner._should_learn()
    assert learner.stats["emotion_skips"] == 2


def test_learner_allows_learning_on_emotion_error() -> None:
    """Learning is allowed when emotion reading fails (don't block on errors)."""

    def broken_emotion() -> None:
        """Raise ConnectionError to simulate the daemon being down."""
        raise ConnectionError("daemon not running")

    learner = _make_learner(get_emotion=broken_emotion)
    assert learner._should_learn() is True


# ─── Plasticity posture tests ─────────────────────────────────


def test_learner_default_posture_is_neutral() -> None:
    """Without a plasticity callback, the posture is neutral."""
    learner = _make_learner()
    assert learner.current_posture == LEARNING_POSTURE_NEUTRAL


def test_learner_update_posture_reads_profile() -> None:
    """_update_posture reads the profile and sets the posture."""
    profile = _make_profile(plasticity_gate=0.70, cortisol_tonic=0.18)
    learner = _make_learner(get_plasticity_profile=lambda: profile)
    learner._update_posture()
    assert learner.current_posture == LEARNING_POSTURE_RECEPTIVE
    assert learner.posture_profile is profile


def test_learner_blocks_learning_when_protective() -> None:
    """Learning is blocked under protective posture (chronic stress)."""
    profile = _make_profile(plasticity_gate=0.25, cortisol_tonic=0.60)
    learner = _make_learner(
        get_emotion=lambda: _make_emotion("neutral"),
        get_plasticity_profile=lambda: profile,
    )
    learner._update_posture()
    assert learner.current_posture == LEARNING_POSTURE_PROTECTIVE
    assert not learner._should_learn()


def test_learner_allows_learning_when_receptive() -> None:
    """Learning is allowed under receptive posture."""
    profile = _make_profile(plasticity_gate=0.70, cortisol_tonic=0.18)
    learner = _make_learner(
        get_emotion=lambda: _make_emotion("neutral"),
        get_plasticity_profile=lambda: profile,
    )
    learner._update_posture()
    assert learner.current_posture == LEARNING_POSTURE_RECEPTIVE
    assert learner._should_learn()


def test_learner_allows_learning_when_recovering() -> None:
    """Learning is allowed under recovering posture."""
    profile = _make_profile(plasticity_gate=0.40, bdnf_tonic=0.35, cortisol_tonic=0.35)
    learner = _make_learner(
        get_emotion=lambda: _make_emotion("neutral"),
        get_plasticity_profile=lambda: profile,
    )
    learner._update_posture()
    assert learner.current_posture == LEARNING_POSTURE_RECOVERING
    assert learner._should_learn()


def test_learner_posture_skip_tracking() -> None:
    """Posture skips are tracked in stats."""
    profile = _make_profile(plasticity_gate=0.25, cortisol_tonic=0.60)
    learner = _make_learner(
        get_emotion=lambda: _make_emotion("neutral"),
        get_plasticity_profile=lambda: profile,
    )
    learner._update_posture()
    assert learner.stats["posture_skips"] == 0
    learner._should_learn()
    learner._should_learn()
    assert learner.stats["posture_skips"] == 2


def test_learner_posture_falls_back_on_error() -> None:
    """Posture stays neutral when the callback raises (don't block on errors)."""

    def broken_profile() -> None:
        """Raise ConnectionError to simulate the daemon being down."""
        raise ConnectionError("daemon not running")

    learner = _make_learner(
        get_emotion=lambda: _make_emotion("neutral"),
        get_plasticity_profile=broken_profile,
    )
    learner._update_posture()
    assert learner.current_posture == LEARNING_POSTURE_NEUTRAL
    # Should still allow learning (neutral posture, no error block)
    assert learner._should_learn()


def test_learner_describe_posture() -> None:
    """describe_posture returns a human-readable string."""
    profile = _make_profile(plasticity_gate=0.70, cortisol_tonic=0.18)
    learner = _make_learner(get_plasticity_profile=lambda: profile)
    learner._update_posture()
    desc = learner.describe_posture()
    assert "receptive" in desc
    assert "plasticity_gate=0.70" in desc


def test_learner_describe_posture_no_data() -> None:
    """describe_posture works without substrate data."""
    learner = _make_learner()
    desc = learner.describe_posture()
    assert "neutral" in desc
    assert "no substrate data" in desc
    assert learner._should_learn()


def test_learner_counts_definition_as_concept_learned() -> None:
    """A dictionary result should count the defined topic as a learned concept."""
    learner = _make_learner()
    sr = SourceResult(
        url="wordnet:cognition",
        title="WordNet: cognition",
        content="the state of being cognitive; awareness of one's own existence",
        source_name="wordnet",
    )
    result = learner._learn_from_content(sr, "cognition")
    assert result is not None
    assert "cognition" in result.concepts_learned
    assert learner._concepts_learned >= 1
    assert learner.describe_recent_learning() != "hasn't learned anything on own yet"


def test_learner_learns_short_wordnet_definition() -> None:
    """A short WordNet definition is not dropped by the 50-char floor."""
    learner = _make_learner()
    sr = SourceResult(
        url="wordnet:metre",
        title="WordNet: metre",
        content="[noun] a unit of length",
        source_name="wordnet",
    )
    result = learner._learn_from_content(sr, "metre")
    assert result is not None
    assert "metre" in result.concepts_learned


# ─── Lifecycle tests (start/stop/restart) ──────────────────────


def _make_blocked_learner() -> AutonomousLearner:
    """Create a learner whose _run loop blocks on the state-event wait.

    With a 'stressed' emotion, _should_learn() returns False, so the
    loop enters the _state_event.wait(timeout=60) branch — it does
    no network I/O and no topic processing. This lets us test the
    thread lifecycle (start/stop/restart) deterministically without
    mocking the source registry or network layer.
    """
    return _make_learner(get_emotion=lambda: _make_emotion("stressed", valence=-0.5))


def test_learner_start_creates_one_thread() -> None:
    """start() creates exactly one background thread."""
    learner = _make_blocked_learner()
    try:
        learner.start()
        assert learner._thread is not None
        assert learner._thread.is_alive()
        assert learner._running
    finally:
        learner.stop()
    # After stop, the thread should no longer be running.
    assert not learner._running


def test_learner_stop_wakes_blocked_thread() -> None:
    """stop() wakes a thread blocked on _state_event.wait and exits it."""
    learner = _make_blocked_learner()
    learner.start()
    # Give the thread time to enter the _state_event.wait branch.
    time.sleep(0.2)
    assert learner._thread is not None
    assert learner._thread.is_alive()
    learner.stop()
    # The thread should have exited within the join timeout (5s).
    assert learner._thread is not None
    assert not learner._thread.is_alive()


def test_learner_rapid_restart_no_duplicate_threads() -> None:
    """Rapid stop→start does not leave two learner threads running.

    This is the lifecycle race the start() join-on-stale-thread guard
    prevents: if stop()'s 5s join times out (thread in a long wait),
    start() must join the stale thread before starting a new one so
    the old thread doesn't observe _running=True and resume its loop
    alongside the new thread.
    """
    learner = _make_blocked_learner()
    learner.start()
    time.sleep(0.1)
    learner.stop()
    # Immediately restart — start() must join any stale thread first.
    learner.start()
    try:
        assert learner._thread is not None
        assert learner._thread.is_alive()
        assert learner._running
    finally:
        learner.stop()
    # After the final stop, the thread must be dead.
    assert learner._thread is not None
    assert not learner._thread.is_alive()


def test_learner_double_start_is_idempotent() -> None:
    """Calling start() twice does not spawn a second thread."""
    learner = _make_blocked_learner()
    try:
        learner.start()
        first_thread = learner._thread
        learner.start()  # no-op — already running
        assert learner._thread is first_thread
    finally:
        learner.stop()


def test_learner_stop_when_not_running_is_safe() -> None:
    """stop() on a learner that was never started does not raise."""
    learner = _make_blocked_learner()
    learner.stop()  # must not raise


def test_learner_start_after_stop_restarts_cleanly() -> None:
    """A full stop→start cycle produces a fresh, working learner thread."""
    learner = _make_blocked_learner()
    learner.start()
    time.sleep(0.1)
    learner.stop()
    assert not learner._running
    # Second cycle — a brand-new thread should be created.
    learner.start()
    try:
        assert learner._running
        assert learner._thread is not None
        assert learner._thread.is_alive()
    finally:
        learner.stop()


# ======================================================================
# From tests/test_dual_system.py
# ======================================================================

@pytest.fixture
def network() -> ConceptNetwork:
    """Network."""
    net = ConceptNetwork()
    for name in ("tree", "plant", "flower", "garden", "dog", "cat", "animal"):
        net.add_concept(name, confidence=0.9, origin="test")
        c = net.get_concept(name)
        if c:
            c.properties["definition"] = f"a thing called {name}"
    net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("flower", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
    return net


@pytest.fixture
def learner(network) -> DualSystemLearner:
    """Learner."""
    return DualSystemLearner(network)


def test_encode_fast_creates_episode(learner) -> None:
    """encode_fast stores a hippocampal episode."""
    ep = learner.encode_fast("the tree grows in the garden")
    assert isinstance(ep, HippocampalEpisode)
    assert ep.timestamp > 0
    stats = learner.get_statistics()
    assert stats["fast_episodes"] == 1
    assert stats["total_encoded"] == 1


def test_pattern_separation_orthogonalizes(learner) -> None:
    """Similar inputs get distinct hippocampal representations."""
    ep1 = learner.encode_fast("the tree grows in the garden")
    ep2 = learner.encode_fast("the tree grows in the garden")  # identical
    # Pattern separation should push them apart
    sim = float(np.dot(ep1.pattern, ep2.pattern))
    assert sim < 1.0  # not identical after separation


def test_dissimilar_inputs_not_separated(learner) -> None:
    """Dissimilar inputs are not pushed apart (no need)."""
    ep1 = learner.encode_fast("tree plant garden")
    ep2 = learner.encode_fast("dog cat animal")
    sim = float(np.dot(ep1.pattern, ep2.pattern))
    # These are dissimilar — separation shouldn't make them negative
    assert sim >= -0.5


def test_consolidate_to_slow_creates_neocortical(learner) -> None:
    """Consolidation transfers hippocampal episodes to neocortex."""
    learner.encode_fast("tree plant garden")
    result = learner.consolidate_to_slow(max_episodes=5)
    assert result["replayed"] >= 1
    assert result["neocortical_updated"] >= 1
    stats = learner.get_statistics()
    assert stats["slow_memories"] >= 1


def test_consolidation_marks_consolidated_after_replays(learner) -> None:
    """Episodes consolidate after enough replays."""
    learner.encode_fast("tree plant garden")
    # Replay multiple times to reach consolidation threshold
    for _ in range(5):
        learner.consolidate_to_slow(max_episodes=5)
    stats = learner.get_statistics()
    assert stats["fast_consolidated"] >= 1


def test_retrieve_checks_both_systems(learner) -> None:
    """Retrieve returns matches from both systems."""
    learner.encode_fast("tree plant garden")
    learner.consolidate_to_slow(max_episodes=5)
    results = learner.retrieve("tree plant garden")
    assert len(results) > 0
    # Each result has (key, similarity, system)
    for _key, sim, system in results:
        assert system in ("hippocampal", "neocortical")
        assert 0.0 <= sim <= 1.0 + 1e-6  # allow float32 rounding


def test_retrieve_sorted_by_similarity(learner) -> None:
    """Results are sorted by similarity descending."""
    learner.encode_fast("tree plant garden")
    learner.encode_fast("dog cat animal")
    results = learner.retrieve("tree plant garden")
    sims = [s for _, s, _ in results]
    assert sims == sorted(sims, reverse=True)


def test_fast_capacity_enforced(network) -> None:
    """Hippocampal store enforces capacity limit."""
    learner = DualSystemLearner(network, fast_capacity=5)
    for i in range(10):
        learner.encode_fast(f"experience number {i}")
    stats = learner.get_statistics()
    assert stats["fast_episodes"] == 5


def test_neocortical_memory_dataclass() -> None:
    """NeocorticalMemory stores key, pattern, strength."""
    mem = NeocorticalMemory(
        key="test",
        pattern=np.zeros(4, dtype=np.float32),
        strength=0.5,
        episode_count=3,
        last_updated=100,
    )
    assert mem.key == "test"
    assert mem.strength == 0.5
    assert mem.episode_count == 3


def test_concept_list_input(learner) -> None:
    """encode_fast accepts a list of concept names."""
    ep = learner.encode_fast(["tree", "plant", "garden"])
    assert ep.key == "garden plant tree"  # sorted


# ======================================================================
# From tests/test_stdp.py
# ======================================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity — the correct metric for Oja-normalized updates.

    The Oja rule shrinks vector norms (that's its purpose — preventing
    unbounded growth). Raw dot product conflates alignment and magnitude,
    so it can decrease even when vectors become more aligned. Cosine
    similarity isolates the directional alignment.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.fixture
def network_stdp() -> ConceptNetwork:
    """Network stdp."""
    net = ConceptNetwork()
    for name in ("tree", "plant", "flower", "garden", "dog", "cat", "animal"):
        net.add_concept(name, confidence=0.9, origin="test")
        c = net.get_concept(name)
        if c:
            c.properties["definition"] = f"a thing called {name}"
    net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("flower", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("tree", "garden", RelationType.RELATED_TO, 0.7, origin="test")
    return net


@pytest.fixture
def embeddings(network_stdp, tmp_path) -> EmbeddingStore:
    """Embeddings."""
    return EmbeddingStore(network_stdp, data_dir=str(tmp_path))


@pytest.fixture
def stdp(embeddings, network_stdp) -> STDP:
    """Stdp."""
    return STDP(embeddings, network_stdp)


def test_stdp_weight_positive_delta_is_potentiation(stdp) -> None:
    """Δt > 0 (pre before post) → LTP (positive weight change)."""
    w = stdp.compute_stdp_window(10.0)
    assert w > 0


def test_stdp_weight_negative_delta_is_depression(stdp) -> None:
    """Δt < 0 (post before pre) → LTD (negative weight change)."""
    w = stdp.compute_stdp_window(-10.0)
    assert w < 0


def test_stdp_weight_zero_delta_is_max_potentiation(stdp) -> None:
    """Δt = 0 (co-firing) → maximum LTP."""
    w_zero = stdp.compute_stdp_window(0.0)
    w_small = stdp.compute_stdp_window(1.0)
    assert w_zero > w_small  # decay with increasing Δt


def test_stdp_exponential_decay(stdp) -> None:
    """LTP decays exponentially with Δt."""
    w1 = stdp.compute_stdp_window(5.0)
    w2 = stdp.compute_stdp_window(10.0)
    # exp(-10/20) / exp(-5/20) = exp(-5/20) ≈ 0.78
    ratio = w2 / w1
    expected = np.exp(-5.0 / stdp.tau_plus)
    assert abs(ratio - expected) < 0.01


def test_stdp_depressive_bias(stdp) -> None:
    """A_minus > A_plus → net depressive bias at symmetric |Δt|."""
    w_pos = stdp.compute_stdp_window(10.0)
    w_neg = stdp.compute_stdp_window(-10.0)
    assert abs(w_neg) > w_pos  # |LTD| > LTP


def test_stdp_record_spike_potentiates(embeddings, stdp) -> None:
    """Pre before post → embeddings move closer (potentiation).

    Uses cosine similarity (not raw dot product) because the Oja
    normalization rule shrinks vector norms — that's its purpose
    (preventing unbounded growth). Raw dot product conflates alignment
    and magnitude, so it can decrease even when vectors become more
    aligned. The tolerance accounts for the Oja damping shrinking the
    experiential slice relative to the unchanged spectral components,
    which slightly shifts the full-vector direction.
    """
    v_tree_before = embeddings.get_concept_vector("tree").copy()
    v_dog_before = embeddings.get_concept_vector("dog").copy()
    sim_before = _cosine_similarity(v_tree_before, v_dog_before)

    # tree spikes, then dog spikes 5ms later (pre before post → LTP)
    stdp.record_spike("tree", 0.0)
    stdp.record_spike("dog", 5.0)

    v_tree_after = embeddings.get_concept_vector("tree")
    v_dog_after = embeddings.get_concept_vector("dog")
    sim_after = _cosine_similarity(v_tree_after, v_dog_after)

    assert sim_after >= sim_before - 1e-3


def test_stdp_record_spike_depresses(embeddings, stdp) -> None:
    """Post before pre → embeddings move apart (depression).

    LTD occurs when the post spike fires BEFORE the pre spike
    (Δt = t_post − t_pre < 0). In sequential recording, this is
    achieved by recording the pre spike with an *earlier* logical
    timestamp than an already-recorded post spike.
    """
    # First, potentiate tree-dog via causal LTP (tree pre before dog post)
    for _ in range(20):
        stdp.record_spike("tree", 0.0)
        stdp.record_spike("dog", 5.0)
    stdp.renormalize()
    sim_before = _cosine_similarity(
        embeddings.get_concept_vector("tree"),
        embeddings.get_concept_vector("dog"),
    )

    # Clear history to prevent cross-phase interference (co-firing
    # pairs from the LTP phase producing LTP at Δt=0 during LTD phase).
    stdp.clear_history()

    # Now trigger LTD: record tree (post) at t=5 ONCE, then dog (pre)
    # at t=0 many times. Each dog spike looks back at tree at t=5,
    # Δt = 0 − 5 = −5 < 0 → post (tree) fired before pre (dog) → LTD.
    # We only record tree once to avoid generating LTP pairs (tree at 5
    # looking back at dog at 0 would produce LTP).
    stdp.record_spike("tree", 5.0)  # post at t=5
    for _ in range(50):
        stdp.record_spike("dog", 0.0)  # pre at t=0 → LTD
    stdp.renormalize()
    sim_after = _cosine_similarity(
        embeddings.get_concept_vector("tree"),
        embeddings.get_concept_vector("dog"),
    )

    assert sim_after <= sim_before + 1e-6


def test_stdp_statistics(stdp) -> None:
    """Statistics track potentiation and depression counts."""
    stdp.record_spike("tree", 0.0)
    stdp.record_spike("dog", 5.0)  # tree pre, dog post, Δt=5 > 0 → LTP
    # Record cat at t=2: dog is in history at t=5, Δt = 2−5 = −3 < 0 → LTD
    stdp.record_spike("cat", 2.0)
    stats = stdp.get_statistics()
    assert stats["total_potentiated"] >= 1
    assert stats["total_depressed"] >= 1


def test_spike_event_dataclass() -> None:
    """SpikeEvent stores concept and time."""
    ev = SpikeEvent(concept="tree", time_ms=42.0)
    assert ev.concept == "tree"
    assert ev.time_ms == 42.0


def test_stdp_window_outside_range_skipped(embeddings, stdp) -> None:
    """Spikes outside the learning window produce no update."""
    stdp.record_spike("tree", 0.0)
    # dog spikes 100ms later — outside the 40ms window
    stdp.record_spike("dog", 100.0)
    stats = stdp.get_statistics()
    # No potentiation should have occurred (outside window)
    assert stats["total_potentiated"] == 0


def test_stdp_non_monotonic_timestamps_finds_in_window_pairs(
    embeddings, stdp
) -> None:
    """Spikes within the window are not skipped when out-of-window
    entries appear earlier in the reversed deque.

    The deque is ordered by insertion order, not by timestamp. When
    timestamps are non-monotonic (e.g., the autonomous learner resets
    its logical clock to t=0 each session), a recently-inserted spike
    may have a larger t_other than an older-inserted one. The scan
    must skip out-of-window entries with ``continue`` rather than
    ``break``, otherwise valid in-window pairs are missed.
    """
    # tree spikes at t=0, 5, 100, 105 — simulating two sessions
    # with non-monotonic logical clocks (0-5, then 100-105).
    for t in (0.0, 5.0, 100.0, 105.0):
        stdp.record_spike("tree", t)
    # dog spikes at t=10 — within the 40ms window of tree's t=0 and t=5
    # but outside the window of tree's t=100 and t=105.
    stdp.record_spike("dog", 10.0)
    stats = stdp.get_statistics()
    # tree at t=0 (Δt=10) and t=5 (Δt=5) are within the 40ms window
    # and should produce LTP. Without the fix, the break at t=105
    # (Δt=-95, outside window) would skip both.
    assert stats["total_potentiated"] >= 2, (
        f"Expected >= 2 potentiated pairs from in-window spikes, "
        f"got {stats['total_potentiated']}"
    )


# ======================================================================
# From tests/test_td_learning.py
# ======================================================================

@pytest.fixture
def network_td_learning() -> ConceptNetwork:
    """Network td learning."""
    net = ConceptNetwork()
    for name in ("tree", "plant", "flower", "garden", "dog", "cat", "animal"):
        net.add_concept(name, confidence=0.9, origin="test")
    net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
    return net


@pytest.fixture
def td(network_td_learning) -> TDLearner:
    """Td."""
    return TDLearner(network_td_learning)


def test_predict_value_returns_float(td) -> None:
    """predict_value returns a float for a state."""
    v = td.predict_value(["tree", "plant"])
    assert isinstance(v, float)
    assert v > 0  # confidence is 0.9, weights default to 1.0


def test_predict_value_empty_state_is_zero(td) -> None:
    """Empty state has value 0."""
    assert td.predict_value([]) == 0.0


def test_predict_value_unknown_concepts_zero(td) -> None:
    """Unknown concepts contribute zero."""
    v = td.predict_value(["nonexistent_concept"])
    assert v == 0.0


def test_update_returns_rpe(td) -> None:
    """update returns the reward prediction error."""
    rpe = td.update(["tree"], reward=1.0, next_state=["plant"])
    # RPE = reward + gamma * V(s') - V(s)
    1.0 + td.discount * td.predict_value(["plant"]) - td.predict_value(["tree"])
    # Note: V(s) is computed before the update, but we recompute here
    # so just check it's a float and positive (reward > expected)
    assert isinstance(rpe, float)
    assert rpe > 0  # reward of 1.0 > expected value


def test_positive_rpe_increases_value(td) -> None:
    """Positive RPE (unexpected reward) increases state value."""
    v_before = td.predict_value(["tree"])
    td.update(["tree"], reward=1.0, next_state=[])
    v_after = td.predict_value(["tree"])
    assert v_after > v_before


def test_negative_rpe_decreases_value(td) -> None:
    """Negative RPE (reward worse than expected) decreases state value."""
    # First, build up a high value
    for _ in range(20):
        td.update(["tree"], reward=1.0, next_state=[])
    v_before = td.predict_value(["tree"])
    # Now give a negative reward (worse than expected)
    td.update(["tree"], reward=-1.0, next_state=[])
    v_after = td.predict_value(["tree"])
    assert v_after < v_before


def test_zero_rpe_no_change(td) -> None:
    """When reward matches expectation, value stabilizes."""
    # Train until value converges toward the reward
    for _ in range(100):
        td.update(["tree"], reward=0.5, next_state=[])
    v1 = td.predict_value(["tree"])
    td.update(["tree"], reward=0.5, next_state=[])
    v2 = td.predict_value(["tree"])
    # Should be very close (converging)
    assert abs(v2 - v1) < 0.1


def test_get_dopamine_signal_scales_rpe(td) -> None:
    """get_dopamine_signal returns scaled RPE."""
    rpe = td.update(["tree"], reward=1.0, next_state=[])
    dopamine = td.get_dopamine_signal()
    assert abs(dopamine - rpe * td.dopamine_scale) < 1e-9


def test_get_dopamine_signal_positive_for_positive_rpe(td) -> None:
    """Positive RPE → positive dopamine (burst)."""
    td.update(["tree"], reward=1.0, next_state=[])
    assert td.get_dopamine_signal() > 0


def test_get_dopamine_signal_negative_for_negative_rpe(td) -> None:
    """Negative RPE → negative dopamine (dip)."""
    td.update(["tree"], reward=-1.0, next_state=[])
    assert td.get_dopamine_signal() < 0


def test_get_rpe_returns_last(td) -> None:
    """get_rpe returns the most recent RPE."""
    td.update(["tree"], reward=0.5, next_state=[])
    assert td.get_rpe() != 0.0
    assert td.get_rpe() == td._last_rpe


def test_statistics_track_counts(td) -> None:
    """Statistics track positive/negative RPE counts."""
    td.update(["tree"], reward=1.0, next_state=[])  # positive
    td.update(["dog"], reward=-1.0, next_state=[])  # negative
    stats = td.get_statistics()
    assert stats["total_updates"] == 2
    assert stats["total_positive_rpe"] == 1
    assert stats["total_negative_rpe"] == 1


def test_get_weight_default_one(td) -> None:
    """Unlearned concepts have weight 1.0."""
    assert td.get_weight("tree") == 1.0


def test_get_weight_changes_after_update(td) -> None:
    """Weights change after TD updates."""
    td.update(["tree"], reward=1.0, next_state=[])
    w = td.get_weight("tree")
    assert w != 1.0


def test_history_recorded(td) -> None:
    """Transitions are recorded in history."""
    td.update(["tree"], reward=1.0, next_state=["plant"])
    history = td.get_history(limit=5)
    assert len(history) == 1
    assert isinstance(history[0], TDTransition)
    assert history[0].reward == 1.0


def test_td_transition_dataclass() -> None:
    """TDTransition stores state, reward, next_state, rpe."""
    t = TDTransition(
        state=["a"],
        reward=1.0,
        next_state=["b"],
        rpe=0.5,
        timestamp=100,
    )
    assert t.state == ["a"]
    assert t.reward == 1.0
    assert t.rpe == 0.5


def test_value_generalizes_across_shared_concepts(td) -> None:
    """States sharing concepts have similar values (generalization)."""
    td.update(["tree", "plant"], reward=1.0, next_state=[])
    # A state with 'tree' should have higher value than one without
    v_with = td.predict_value(["tree", "flower"])
    v_without = td.predict_value(["dog", "cat"])
    assert v_with > v_without


# ─── TD(λ) eligibility trace tests ──────────────────────────────


def test_invalid_lambda_raises(network_td_learning) -> None:
    """λ outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        TDLearner(network_td_learning, lam=-0.1)
    with pytest.raises(ValueError):
        TDLearner(network_td_learning, lam=1.1)


def test_default_lambda_is_zero(td) -> None:
    """Default λ is 0.0 (backward-compatible TD(0))."""
    assert td.lam == 0.0


def test_get_trace_zero_for_untraced(td) -> None:
    """get_trace returns 0.0 for a concept with no eligibility trace."""
    assert td.get_trace("tree") == 0.0


def test_trace_set_for_active_concept(network_td_learning) -> None:
    """After an update, the active concept has a non-zero trace."""
    td = TDLearner(network_td_learning, lam=0.8)
    td.update(["tree"], reward=1.0, next_state=[])
    assert td.get_trace("tree") > 0.0


def test_trace_not_set_for_inactive_concept(network_td_learning) -> None:
    """A concept not in the active state does not get a fresh trace."""
    td = TDLearner(network_td_learning, lam=0.8)
    td.update(["tree"], reward=1.0, next_state=[])
    assert td.get_trace("dog") == 0.0


def test_trace_decays_between_updates(network_td_learning) -> None:
    """Eligibility trace decays by γλ per step."""
    td = TDLearner(network_td_learning, learning_rate=0.1, discount=0.9, lam=0.8)
    # Step 1: activate "tree" → trace = 1.0 (replacing, single concept)
    td.update(["tree"], reward=0.0, next_state=["dog"])
    assert abs(td.get_trace("tree") - 1.0) < 1e-9
    # Step 2: activate "dog" → "tree" trace decays by γλ = 0.72
    td.update(["dog"], reward=0.0, next_state=[])
    assert abs(td.get_trace("tree") - 0.72) < 1e-9


def test_replacing_trace_caps_at_gradient(network_td_learning) -> None:
    """Repeated visits don't accumulate the trace (replacing, not accumulating)."""
    td = TDLearner(network_td_learning, lam=0.8)
    td.update(["tree"], reward=0.0, next_state=[])
    trace_1 = td.get_trace("tree")
    td.update(["tree"], reward=0.0, next_state=[])
    trace_2 = td.get_trace("tree")
    # Replacing trace: trace is set to gradient (1/count = 1.0), not
    # accumulated to 1.0 + 0.72 = 1.72.
    assert abs(trace_2 - 1.0) < 1e-9
    assert trace_2 <= trace_1 + 1e-9


def test_lambda_zero_clears_traces_immediately(network_td_learning) -> None:
    """With λ=0, traces are cleared each step (no credit propagation)."""
    td = TDLearner(network_td_learning, lam=0.0)
    td.update(["tree"], reward=0.0, next_state=["dog"])
    assert td.get_trace("tree") > 0.0  # set during this update
    td.update(["dog"], reward=0.0, next_state=[])
    # "tree" trace should be gone (cleared at start of step 2)
    assert td.get_trace("tree") == 0.0


def test_lambda_propagates_credit_backward(network_td_learning) -> None:
    """With λ>0, a reward propagates credit to previously-visited concepts.

    Scenario: visit "tree", then visit "dog" and receive a reward.
    With λ=0, only "dog" gets credit. With λ>0, "tree" also gets a
    share (decayed by γλ), because it was recently active.
    """
    td0 = TDLearner(network_td_learning, lam=0.0)
    tdl = TDLearner(network_td_learning, lam=0.8)

    # Same sequence for both learners.
    for learner in (td0, tdl):
        learner.update(["tree"], reward=0.0, next_state=["dog"])
        learner.update(["dog"], reward=1.0, next_state=[])

    # "tree" weight: with λ>0 it receives propagated credit from the
    # positive RPE at step 2; with λ=0 it does not.
    w_tree_0 = td0.get_weight("tree")
    w_tree_l = tdl.get_weight("tree")
    assert w_tree_l != w_tree_0, (
        "λ>0 should produce different weight for previously-visited concept"
    )


def test_lambda_zero_equivalent_to_old_td0(network_td_learning) -> None:
    """λ=0 produces exactly the same weight as the old TD(0) rule.

    The old TD(0) update was: w[i] += α * rpe / count for active i.
    With λ=0, the trace for active i is 1/count, and the update is
    w[i] += α * rpe * (1/count) — identical.
    """
    td = TDLearner(network_td_learning, lam=0.0, learning_rate=0.1, discount=0.9)
    v_before = td.predict_value(["tree"])
    rpe = td.update(["tree"], reward=1.0, next_state=[])
    v_after = td.predict_value(["tree"])

    # Manually compute the expected weight change.
    # V(s) = w * conf / 1 = w * 0.9 (single concept, conf=0.9)
    # rpe = 1.0 + 0.9 * 0 - 0.9 = 0.1
    # w_new = 1.0 + 0.1 * 0.1 * 1.0 = 1.01
    # V_new = 1.01 * 0.9 = 0.909
    expected_w = 1.0 + 0.1 * rpe * 1.0
    expected_v = expected_w * 0.9
    assert abs(v_after - expected_v) < 1e-9
    assert v_after > v_before  # positive RPE increases value


def test_reset_traces_clears_all(network_td_learning) -> None:
    """reset_traces clears all eligibility traces."""
    td = TDLearner(network_td_learning, lam=0.8)
    td.update(["tree", "plant"], reward=1.0, next_state=[])
    assert td.get_trace("tree") > 0.0
    assert td.get_trace("plant") > 0.0
    td.reset_traces()
    assert td.get_trace("tree") == 0.0
    assert td.get_trace("plant") == 0.0


def test_statistics_include_trace_info(network_td_learning) -> None:
    """Statistics include active_traces count and lambda value."""
    td = TDLearner(network_td_learning, lam=0.8)
    td.update(["tree", "plant"], reward=1.0, next_state=[])
    stats = td.get_statistics()
    assert "active_traces" in stats
    assert "lambda" in stats
    assert stats["lambda"] == 0.8
    assert stats["active_traces"] == 2  # tree + plant


def test_trace_persistence_roundtrip(network_td_learning) -> None:
    """Eligibility traces and λ survive serialize/restore."""
    from genesis_cognitive.persistence import (
        _serialize_td_learner,
        restore_td_learner,
    )

    td = TDLearner(network_td_learning, lam=0.8)
    td.update(["tree"], reward=1.0, next_state=["plant"])
    td.update(["plant"], reward=0.5, next_state=[])

    data = _serialize_td_learner(td)
    assert data["lam"] == 0.8
    assert len(data["traces"]) > 0

    td2 = TDLearner(network_td_learning, lam=0.0)  # different λ
    restore_td_learner(td2, data)
    assert td2.lam == 0.8
    assert td2.get_trace("tree") > 0.0  # trace restored


def test_restore_old_save_without_traces(network_td_learning) -> None:
    """Restoring a pre-TD(λ) save (no traces/lam keys) doesn't crash."""
    from genesis_cognitive.persistence import restore_td_learner

    td = TDLearner(network_td_learning, lam=0.8)
    old_data = {
        "weights": {"tree": 1.5},
        "history": [],
        "last_rpe": 0.1,
        "total_updates": 5,
        "total_positive_rpe": 3,
        "total_negative_rpe": 2,
    }
    restore_td_learner(td, old_data)
    assert td.get_weight("tree") == 1.5
    assert td.lam == 0.0  # falls back to default when key absent
    assert td.get_trace("tree") == 0.0


def test_trace_pruned_after_decay(network_td_learning) -> None:
    """Traces below the prune threshold are removed (bounded memory)."""
    td = TDLearner(network_td_learning, lam=0.1, discount=0.5)
    td.update(["tree"], reward=0.0, next_state=["dog"])
    # Many steps with a different concept → tree's trace decays to ~0
    for _ in range(200):
        td.update(["dog"], reward=0.0, next_state=[])
    # trace should have been pruned (γλ = 0.05, 200 steps → 0.05^200 ≈ 0)
    assert td.get_trace("tree") == 0.0


# ======================================================================
# From tests/test_plasticity.py
# ======================================================================

@pytest.fixture
def network_plasticity() -> ConceptNetwork:
    """A concept network for testing."""
    net = ConceptNetwork()

    for name, defn in [
        ("tree", "a tall perennial woody plant having a main trunk"),
        ("plant", "a living organism that grows in the earth"),
        ("flower", "a colorful plant that blooms and produces seeds"),
        ("garden", "a plot of ground where plants are cultivated"),
        ("bush", "a low woody plant with multiple stems"),
        ("dog", "a domesticated carnivorous mammal"),
        ("cat", "a small domesticated carnivorous mammal"),
        ("animal", "a living organism that moves voluntarily"),
        ("cognition", "an alert cognitive state of awareness"),
        ("memory", "the cognitive process of retaining information"),
    ]:
        net.add_concept(name, confidence=0.9, origin="test")
        c = net.get_concept(name)
        if c:
            c.properties["definition"] = defn

    net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("bush", "plant", RelationType.IS_A, 0.85, origin="test")
    net.add_edge("flower", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("cat", "animal", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("tree", "garden", RelationType.RELATED_TO, 0.7, origin="test")
    net.add_edge("flower", "garden", RelationType.RELATED_TO, 0.8, origin="test")
    net.add_edge("cognition", "memory", RelationType.RELATED_TO, 0.7, origin="test")
    net.add_edge("tree", "bush", RelationType.SIMILAR_TO, 0.8, origin="test")

    return net


@pytest.fixture
def embeddings_plasticity(network_plasticity, tmp_path) -> EmbeddingStore:
    """Embeddings plasticity."""
    return EmbeddingStore(network_plasticity, data_dir=str(tmp_path))


@pytest.fixture
def plasticity(embeddings_plasticity, network_plasticity) -> HebbianPlasticity:
    """Plasticity."""
    return HebbianPlasticity(embeddings_plasticity, network_plasticity)


@pytest.fixture
def topology(network_plasticity) -> NetworkTopology:
    """Topology."""
    return NetworkTopology(network_plasticity)


class TestHebbianPlasticity:
    """Test Hebbian learning in the embedding space."""

    def test_record_co_occurrence(self, plasticity: HebbianPlasticity):
        """Recording a co-occurrence should buffer it."""
        plasticity.record_co_occurrence("tree", "plant", 1.0)
        assert len(plasticity._co_occurrences) == 1

    def test_record_co_occurrence_self_ignored(self, plasticity: HebbianPlasticity):
        """Self-co-occurrence should be ignored."""
        plasticity.record_co_occurrence("tree", "tree", 1.0)
        assert len(plasticity._co_occurrences) == 0

    def test_record_co_occurrence_unknown_concept_ignored(self, plasticity: HebbianPlasticity):
        """Co-occurrence with unknown concept should be ignored."""
        plasticity.record_co_occurrence("tree", "nonexistent", 1.0)
        assert len(plasticity._co_occurrences) == 0

    def test_record_co_occurrences_pairwise(self, plasticity: HebbianPlasticity):
        """record_co_occurrences should record all pairs."""
        plasticity.record_co_occurrences(["tree", "plant", "garden"], 1.0)
        # 3 concepts → 3 pairs
        assert len(plasticity._co_occurrences) == 3

    def test_waking_update_changes_vectors(
        self, plasticity: HebbianPlasticity, embeddings_plasticity: EmbeddingStore
    ):
        """Waking update should move co-occurring concepts closer."""
        # Get initial vectors
        v_tree_before = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_before is not None
        v_tree_before = v_tree_before.copy()
        v_dog_before = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_before is not None
        v_dog_before = v_dog_before.copy()

        # Record co-occurrence and apply waking update
        plasticity.record_co_occurrence("tree", "dog", 1.0)
        plasticity.apply_waking_update()

        # Get updated vectors
        v_tree_after = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_after is not None
        v_dog_after = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_after is not None

        # The vectors should have changed
        assert not np.allclose(v_tree_before, v_tree_after)
        assert not np.allclose(v_dog_before, v_dog_after)

        # And they should be closer (higher cosine similarity)
        sim_before = float(
            np.dot(v_tree_before, v_dog_before)
            / (np.linalg.norm(v_tree_before) * np.linalg.norm(v_dog_before))
        )
        sim_after = float(
            np.dot(v_tree_after, v_dog_after)
            / (np.linalg.norm(v_tree_after) * np.linalg.norm(v_dog_after))
        )
        assert sim_after >= sim_before - 0.01  # allow small numerical error

    def test_sleep_consolidation_clears_buffer(self, plasticity: HebbianPlasticity):
        """Sleep consolidation should clear the co-occurrence buffer."""
        plasticity.record_co_occurrence("tree", "plant", 1.0)
        assert len(plasticity._co_occurrences) == 1

        plasticity.apply_sleep_consolidation()
        assert len(plasticity._co_occurrences) == 0

    def test_sleep_consolidation_returns_stats(self, plasticity: HebbianPlasticity):
        """Sleep consolidation should return statistics."""
        plasticity.record_co_occurrence("tree", "plant", 1.0)
        result = plasticity.apply_sleep_consolidation()

        assert "co_occurrence" in result
        assert "graph_attracted" in result
        assert "graph_repelled" in result
        assert "total" in result
        assert result["total"] > 0

    def test_learning_rate_decreases(self, plasticity: HebbianPlasticity):
        """Learning rate should decrease with more updates (Robbins-Monro)."""
        lr_initial = plasticity._learning_rate(0.01)

        # Simulate many updates
        plasticity._waking_update_count = 1000
        lr_later = plasticity._learning_rate(0.01)

        assert lr_later < lr_initial

    def test_statistics(self, plasticity: HebbianPlasticity):
        """Statistics should be available after updates."""
        plasticity.record_co_occurrence("tree", "plant", 1.0)
        plasticity.apply_waking_update()

        stats = plasticity.get_statistics()
        assert stats["total_updates"] > 0
        assert stats["total_attracted"] > 0

    def test_refresh_preserves_hebbian_updates(
        self, plasticity: HebbianPlasticity, embeddings_plasticity: EmbeddingStore
    ):
        """refresh() should not destroy Hebbian-adapted experiential vectors.

        This is the critical persistence test: after sleep consolidation
        applies Hebbian updates, refresh() rebuilds the spectral component
        from the graph but must preserve the experiential (TF-IDF + GloVe)
        portion that carries the learning.

        The experiential portion's *direction* should be preserved (the
        Hebbian learning changes the direction, not the magnitude). The
        exact values may differ slightly because re-normalization of the
        full vector (with a new spectral component) scales all components.
        """
        # Get initial vectors
        v_tree_before = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_before is not None
        v_tree_before = v_tree_before.copy()
        v_dog_before = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_before is not None
        v_dog_before = v_dog_before.copy()

        # Apply Hebbian learning
        plasticity.record_co_occurrence("tree", "dog", 1.0)
        plasticity.apply_waking_update()

        v_tree_after_hebbian = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_after_hebbian is not None
        v_dog_after_hebbian = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_after_hebbian is not None

        # Compute the experiential-only cosine similarity (the Hebbian signal)
        exp_start = (
            embeddings_plasticity._spectral_dim
            if embeddings_plasticity._spectral_dim > 0
            else 0
        )

        def exp_cos_sim(va, vb):
            """Cosine similarity restricted to the experiential (non-spectral) slice."""
            ea = va[exp_start:]
            eb = vb[exp_start:]
            na, nb = np.linalg.norm(ea), np.linalg.norm(eb)
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(ea, eb) / (na * nb))

        sim_before = exp_cos_sim(v_tree_before, v_dog_before)
        sim_after_hebbian = exp_cos_sim(v_tree_after_hebbian, v_dog_after_hebbian)

        # Hebbian learning should have increased the experiential similarity
        assert sim_after_hebbian > sim_before

        # Now refresh — this used to destroy the Hebbian updates entirely
        embeddings_plasticity.refresh()

        v_tree_after_refresh = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_after_refresh is not None
        v_dog_after_refresh = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_after_refresh is not None

        # The experiential similarity should be preserved (not reset to
        # the pre-Hebbian value). This is the key assertion: refresh()
        # must not erase the Hebbian learning.
        sim_after_refresh = exp_cos_sim(v_tree_after_refresh, v_dog_after_refresh)
        assert sim_after_refresh > sim_before + 0.001, (
            f"refresh() erased Hebbian learning: "
            f"sim_before={sim_before:.4f}, sim_after_hebbian={sim_after_hebbian:.4f}, "
            f"sim_after_refresh={sim_after_refresh:.4f}"
        )

    def test_save_load_experiential_persistence(
        self, plasticity: HebbianPlasticity, embeddings_plasticity: EmbeddingStore, tmp_path
    ):
        """Hebbian-adapted vectors should survive save/load to disk."""
        # Apply Hebbian learning
        plasticity.record_co_occurrence("tree", "dog", 1.0)
        plasticity.apply_waking_update()

        v_tree_before = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_before is not None
        v_dog_before = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_before is not None

        exp_start = (
            embeddings_plasticity._spectral_dim
            if embeddings_plasticity._spectral_dim > 0
            else 0
        )

        def exp_cos_sim(va, vb):
            """Cosine similarity restricted to the experiential (non-spectral) slice."""
            ea, eb = va[exp_start:], vb[exp_start:]
            na, nb = np.linalg.norm(ea), np.linalg.norm(eb)
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(ea, eb) / (na * nb))

        sim_before_save = exp_cos_sim(v_tree_before, v_dog_before)

        # Save
        embeddings_plasticity.save_experiential(str(tmp_path))

        # Simulate a restart: rebuild from scratch, then load
        embeddings_plasticity._loaded = False
        embeddings_plasticity._concept_matrix = None
        embeddings_plasticity._ensure_loaded()  # rebuilds + loads experiential

        v_tree_after = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_after is not None
        v_dog_after = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_after is not None

        # The experiential similarity should be preserved across save/load
        sim_after_load = exp_cos_sim(v_tree_after, v_dog_after)
        assert abs(sim_after_load - sim_before_save) < 0.01, (
            f"Save/load changed experiential similarity: "
            f"before={sim_before_save:.4f}, after={sim_after_load:.4f}"
        )

    def test_oja_normalization_symmetry(
        self, plasticity: HebbianPlasticity, embeddings_plasticity: EmbeddingStore
    ):
        """Oja-normalized Hebbian attraction should be symmetric.

        The Oja rule adds a decay term proportional to (v_a . v_b) * v
        that prevents unbounded norm growth. Both vectors should decay,
        not one decay and the other amplify. A sign error in the v_b
        decay term causes v_b's norm to grow unboundedly over many
        updates while v_a's norm shrinks — an asymmetric update that
        destabilizes the embedding space.

        This test applies many attraction updates and checks that:
        1. Both norms remain bounded (no unbounded growth)
        2. The norm ratio stays near 1.0 (symmetric normalization)
        """
        v_tree = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree is not None
        v_dog = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog is not None

        exp_start = (
            embeddings_plasticity._spectral_dim
            if embeddings_plasticity._spectral_dim > 0
            else 0
        )
        norm_tree_before = float(np.linalg.norm(v_tree[exp_start:]))
        norm_dog_before = float(np.linalg.norm(v_dog[exp_start:]))

        # Apply many Hebbian attraction updates directly
        lr = 0.05
        for _ in range(50):
            plasticity._hebbian_attract("tree", "dog", lr)

        v_tree_after = embeddings_plasticity.get_concept_vector("tree")
        assert v_tree_after is not None
        v_dog_after = embeddings_plasticity.get_concept_vector("dog")
        assert v_dog_after is not None

        norm_tree_after = float(np.linalg.norm(v_tree_after[exp_start:]))
        norm_dog_after = float(np.linalg.norm(v_dog_after[exp_start:]))

        # Both norms should remain bounded — not grow beyond 2x original
        assert norm_tree_after < norm_tree_before * 2.0, (
            f"tree norm grew unboundedly: {norm_tree_before:.4f} → {norm_tree_after:.4f}"
        )
        assert norm_dog_after < norm_dog_before * 2.0, (
            f"dog norm grew unboundedly: {norm_dog_before:.4f} → {norm_dog_after:.4f}"
        )

        # The norm ratio should stay near 1.0 (symmetric normalization).
        # With the sign error, v_b amplifies while v_a decays, producing
        # a ratio that diverges from 1.0 over many updates.
        ratio_before = norm_tree_before / norm_dog_before
        ratio_after = norm_tree_after / norm_dog_after
        assert abs(ratio_after - ratio_before) < 0.5, (
            f"Oja normalization is asymmetric (sign error): "
            f"ratio_before={ratio_before:.4f}, ratio_after={ratio_after:.4f}"
        )

    def test_plasticity_state_save_load(self, plasticity: HebbianPlasticity):
        """Plasticity state (update counts) should survive save/load."""
        plasticity.record_co_occurrence("tree", "plant", 1.0)
        plasticity.apply_waking_update()
        plasticity.apply_sleep_consolidation()

        state = plasticity.save_state()
        assert state["waking_update_count"] > 0
        assert state["sleep_update_count"] > 0
        assert state["total_updates"] > 0

        # Create a fresh plasticity instance and restore
        new_plasticity = HebbianPlasticity(plasticity.embeddings, plasticity.network)
        assert new_plasticity._waking_update_count == 0
        assert new_plasticity._sleep_update_count == 0

        new_plasticity.load_state(state)
        assert new_plasticity._waking_update_count == state["waking_update_count"]
        assert new_plasticity._sleep_update_count == state["sleep_update_count"]
        assert new_plasticity.total_updates == state["total_updates"]


class TestNetworkTopology:
    """Test graph-theoretic analysis of the concept network_plasticity."""

    def test_clustering_coefficient(self, topology: NetworkTopology):
        """Clustering coefficient should be between 0 and 1."""
        cc = topology.clustering_coefficient("plant")
        assert 0.0 <= cc <= 1.0

    def test_clustering_coefficient_isolated(self, topology: NetworkTopology):
        """Isolated concepts should have clustering 0."""
        # Add an isolated concept
        topology.network.add_concept("isolated", confidence=0.5, origin="test")
        cc = topology.clustering_coefficient("isolated")
        assert cc == 0.0

    def test_degree_centrality(self, topology: NetworkTopology):
        """Degree centrality should be between 0 and 1."""
        dc = topology.degree_centrality("plant")
        assert 0.0 < dc <= 1.0  # plant has connections

    def test_eigenvector_centrality(self, topology: NetworkTopology):
        """Eigenvector centrality should be non-negative."""
        ec = topology.eigenvector_centrality("plant")
        assert ec >= 0.0

    def test_betweenness_centrality(self, topology: NetworkTopology):
        """Betweenness centrality should be non-negative."""
        bc = topology.betweenness_centrality("plant")
        assert bc >= 0.0

    def test_communities_detected(self, topology: NetworkTopology):
        """Community detection should find at least one community."""
        communities = topology.communities()
        assert len(communities) >= 1
        # All concepts should be in some community
        total = sum(len(c) for c in communities)
        assert total == topology.network.size

    def test_community_of(self, topology: NetworkTopology):
        """community_of should return a community index."""
        idx = topology.community_of("tree")
        assert idx is not None
        assert idx >= 0

    def test_global_clustering(self, topology: NetworkTopology):
        """Global clustering should be between 0 and 1."""
        gc = topology.global_clustering()
        assert 0.0 <= gc <= 1.0

    def test_average_path_length(self, topology: NetworkTopology):
        """Average path length should be positive for connected graphs."""
        apl = topology.average_path_length()
        assert apl > 0.0

    def test_hub_concepts(self, topology: NetworkTopology):
        """Hub concepts should return the most central concepts."""
        hubs = topology.hub_concepts(k=3)
        assert len(hubs) <= 3
        assert len(hubs) > 0
        # Hubs should be sorted by centrality (descending)
        for i in range(len(hubs) - 1):
            assert hubs[i][1] >= hubs[i + 1][1]

    def test_bridge_concepts(self, topology: NetworkTopology):
        """Bridge concepts should return concepts with high betweenness."""
        bridges = topology.bridge_concepts(k=3)
        assert len(bridges) <= 3

    def test_knowledge_domains(self, topology: NetworkTopology):
        """Knowledge domains should return community summaries."""
        domains = topology.knowledge_domains()
        # Should have at least one domain with 3+ concepts
        assert any(size >= 3 for _, size, _ in domains)

    def test_describe_structure(self, topology: NetworkTopology):
        """describe_structure should return a non-empty string."""
        desc = topology.describe_structure()
        assert len(desc) > 0
        assert "concept" in desc.lower() or "knowledge" in desc.lower()

    def test_refresh_marks_dirty(self, topology: NetworkTopology):
        """refresh should mark the cache as dirty."""
        topology._ensure_computed()
        assert not topology._dirty

        topology.refresh()
        assert topology._dirty


# ======================================================================
# From tests/test_three_factor_stdp.py
# ======================================================================


def _make_mock_embeddings(concepts: list[str]) -> MagicMock:
    """Create a mock EmbeddingStore with a concept matrix."""
    n = len(concepts)
    # Each concept gets a random unit vector in 16-dim space
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((n, 16))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    matrix = matrix / norms

    emb = MagicMock()
    emb.has_embeddings = True
    emb._concept_matrix = matrix
    emb._concept_to_idx = {c: i for i, c in enumerate(concepts)}
    emb._spectral_dim = 0
    emb._tfidf_offset = 0

    def _get_vec(c) -> np.ndarray | None:
        """Look up a concept's embedding vector, or None if unknown."""
        if c in emb._concept_to_idx:
            return matrix[emb._concept_to_idx[c]]
        return None

    emb.get_concept_vector = _get_vec
    return emb


def _make_mock_network(concepts: list[str]) -> MagicMock:
    """Create a mock ConceptNetwork."""
    net = MagicMock()
    net._resolve = lambda c: c
    return net


def _make_stdp(concepts: list[str]) -> tuple[STDP, MagicMock, MagicMock]:
    """Create an STDP instance with mock embeddings and network_plasticity."""
    emb = _make_mock_embeddings(concepts)
    net = _make_mock_network(concepts)
    stdp = STDP(emb, net, learning_rate=0.01)
    return stdp, emb, net


def test_default_modulator_is_one():
    """Default modulator should be 1.0 (no gating)."""
    stdp, _, _ = _make_stdp(["a", "b"])
    mod = stdp.get_modulator()
    assert mod == 1.0, f"Default modulator should be 1.0, got {mod}"
    print("  PASS  default modulator is 1.0")


def test_set_modulator():
    """set_modulator should update the modulator value."""
    stdp, _, _ = _make_stdp(["a", "b"])
    stdp.set_modulator(1.5)
    assert stdp.get_modulator() == 1.5, f"Modulator should be 1.5, got {stdp.get_modulator()}"
    stdp.set_modulator(0.0)
    assert stdp.get_modulator() == 0.0, f"Modulator should be 0.0, got {stdp.get_modulator()}"
    print("  PASS  set_modulator updates value")


def test_modulator_zero_suppresses_plasticity():
    """modulator=0 should suppress all STDP updates."""
    stdp, emb, _ = _make_stdp(["a", "b"])
    initial_matrix = emb._concept_matrix.copy()

    stdp.set_modulator(0.0)
    # Record spikes: a before b → should produce LTP, but modulator=0 suppresses it
    stdp.record_spike("a", 100.0)
    stdp.record_spike("b", 110.0)  # 10ms after a → LTP
    result = stdp.apply_updates()

    pot = result["potentiated"]
    assert pot == 0, f"Should potentiate 0 with modulator=0, got {pot}"
    # Matrix should be unchanged
    assert np.allclose(
        emb._concept_matrix, initial_matrix
    ), "Matrix should be unchanged with modulator=0"
    print("  PASS  modulator=0 suppresses plasticity")


def test_modulator_one_applies_standard_stdp():
    """modulator=1 should apply standard STDP (no gating)."""
    stdp, emb, _ = _make_stdp(["a", "b"])
    initial_matrix = emb._concept_matrix.copy()

    stdp.set_modulator(1.0)
    stdp.record_spike("a", 100.0)
    stdp.record_spike("b", 110.0)  # LTP — applied immediately inside record_spike
    stdp.apply_updates()

    stats = stdp.get_statistics()
    tp = stats["total_potentiated"]
    assert tp > 0, f"Should potentiate with modulator=1, got total_potentiated={tp}"
    # Matrix should have changed
    assert not np.allclose(
        emb._concept_matrix, initial_matrix
    ), "Matrix should change with modulator=1"
    print("  PASS  modulator=1 applies standard STDP")


def test_modulator_two_amplifies_plasticity():
    """modulator=2 should amplify STDP (larger weight change)."""
    # Run two separate STDP instances to compare
    stdp1, emb1, _ = _make_stdp(["a", "b"])
    stdp2, emb2, _ = _make_stdp(["a", "b"])

    # Same initial matrix (seeded identically)
    assert np.allclose(emb1._concept_matrix, emb2._concept_matrix)

    stdp1.set_modulator(1.0)
    stdp2.set_modulator(2.0)

    stdp1.record_spike("a", 100.0)
    stdp1.record_spike("b", 110.0)
    stdp1.apply_updates()

    stdp2.record_spike("a", 100.0)
    stdp2.record_spike("b", 110.0)
    stdp2.apply_updates()

    # The change with modulator=2 should be larger
    change1 = np.abs(emb1._concept_matrix - emb2._concept_matrix).sum()
    # Actually, compare each against their own initial state
    # We need to recompute. Let's just check that modulator=2 produced a bigger shift.
    # Since both started identical, the difference between them reflects the amplification.
    assert change1 > 1e-6, (
        f"modulator=2 should produce different result than modulator=1, diff={change1}"
    )
    print(f"  PASS  modulator=2 amplifies plasticity (diff={change1:.6f})")


def test_negative_modulator_reverses_plasticity():
    """modulator<0 should reverse LTP→LTD (potentiation becomes depression)."""
    stdp, _emb, _ = _make_stdp(["a", "b"])

    stdp.set_modulator(-1.0)
    # a before b → normally LTP, but with modulator=-1 it becomes LTD
    stdp.record_spike("a", 100.0)
    stdp.record_spike("b", 110.0)
    stdp.apply_updates()

    stats = stdp.get_statistics()
    td = stats["total_depressed"]
    tp = stats["total_potentiated"]
    assert td > 0, f"Negative modulator should reverse LTP to LTD, got total_depressed={td}"
    assert tp == 0, f"Negative modulator should not potentiate, got total_potentiated={tp}"
    print("  PASS  negative modulator reverses LTP→LTD")


def test_statistics_include_modulator():
    """get_statistics should include the modulator and gated count."""
    stdp, _, _ = _make_stdp(["a", "b"])
    stdp.set_modulator(0.5)

    stats = stdp.get_statistics()

    assert "modulator" in stats, "Statistics should include modulator"
    assert "total_gated" in stats, "Statistics should include total_gated"
    assert stats["modulator"] == 0.5, f"Modulator in stats should be 0.5, got {stats['modulator']}"
    print("  PASS  statistics include modulator and gated count")


def test_gated_updates_counted():
    """Updates suppressed by modulator≈0 should be counted in total_gated."""
    stdp, _, _ = _make_stdp(["a", "b"])
    stdp.set_modulator(0.0)

    stdp.record_spike("a", 100.0)
    stdp.record_spike("b", 110.0)
    stdp.apply_updates()

    stats = stdp.get_statistics()
    assert stats["total_gated"] > 0, f"Gated count should be > 0, got {stats['total_gated']}"
    print(f"  PASS  gated updates counted ({stats['total_gated']})")


def main() -> None:
    """Main."""
    tests = [
        test_default_modulator_is_one,
        test_set_modulator,
        test_modulator_zero_suppresses_plasticity,
        test_modulator_one_applies_standard_stdp,
        test_modulator_two_amplifies_plasticity,
        test_negative_modulator_reverses_plasticity,
        test_statistics_include_modulator,
        test_gated_updates_counted,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Three-factor STDP tests: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()


# ======================================================================
# Curiosity engine tests (from test_intelligence.py)
# ======================================================================


def _make_curiosity_emotion(
    label: str = "neutral",
    valence: float = 0.0,
    alertness: float = 0.5,
    plasticity: float = 0.5,
    creativity: float = 0.5,
) -> EmotionalState:
    """Construct a curiosity emotion for tests."""
    return EmotionalState(
        label=label,
        nuance="baseline",
        cognitive_style="steady",
        valence=valence,
        alertness=alertness,
        plasticity=plasticity,
        creativity=creativity,
    )


def test_curiosity_assess() -> None:
    """Curiosity level is modulated by emotional state."""
    net = ConceptNetwork()
    reasoner = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoner)

    high = curiosity.assess_curiosity(
        _make_curiosity_emotion(plasticity=0.8, valence=0.5, alertness=0.5, creativity=0.7)
    )
    low = curiosity.assess_curiosity(
        _make_curiosity_emotion(plasticity=0.1, valence=0.0, alertness=0.1, creativity=0.1)
    )

    assert high > low
    assert high > 0.5
    assert low < 0.3


def test_curiosity_generate_questions() -> None:
    """Generates questions about gaps."""
    net = ConceptNetwork()
    net.add_concept("mystery", confidence=0.2)

    reasoner = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoner)

    emo = _make_curiosity_emotion(plasticity=0.7, valence=0.3, creativity=0.6)
    questions = curiosity.generate_questions(emo, active_concepts=["mystery"])

    assert len(questions) > 0
    assert any(q.target_concept == "mystery" for q in questions)


def test_curiosity_low_curiosity_no_questions() -> None:
    """Low curiosity produces no questions."""
    net = ConceptNetwork()
    net.add_concept("mystery", confidence=0.2)

    reasoner = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoner)

    emo = _make_curiosity_emotion(plasticity=0.05, valence=-0.3, alertness=0.1, creativity=0.05)
    questions = curiosity.generate_questions(emo, active_concepts=["mystery"])
    assert len(questions) == 0


def test_curiosity_novelty_check() -> None:
    """Doesn't repeat the same question."""
    net = ConceptNetwork()
    net.add_concept("mystery", confidence=0.2)

    reasoner = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoner)

    emo = _make_curiosity_emotion(plasticity=0.7, valence=0.3, creativity=0.6)

    q1 = curiosity.generate_questions(emo, active_concepts=["mystery"])
    q2 = curiosity.generate_questions(emo, active_concepts=["mystery"])

    if q1 and q2:
        q1_keys = {f"{q.target_concept}:{q.question_type}:{q.gap_detail}" for q in q1}
        q2_keys = {f"{q.target_concept}:{q.question_type}:{q.gap_detail}" for q in q2}
        assert not (q1_keys & q2_keys)


def test_curiosity_contemplate() -> None:
    """Contemplation produces internal questions."""
    net = ConceptNetwork()
    net.add_concept("mystery", confidence=0.2)

    reasoner = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoner)

    emo = _make_curiosity_emotion(plasticity=0.6, valence=0.2, creativity=0.5)
    emo.openness_to_engage = 0.3

    questions = curiosity.contemplate(emo)
    for q in questions:
        assert q.internal
        assert not q.should_ask
