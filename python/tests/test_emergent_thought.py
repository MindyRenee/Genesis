"""Tests for emergent, salience-based thought selection in InnerLife.

These tests verify that Genesis's spontaneous thoughts emerge from her
actual internal state — concept activation, knowledge gaps, review
pressure, emotional salience, prediction error, body-state deviation —
rather than from a developer-defined menu of thought categories with
fixed weights and hardcoded seed lists.

Key behaviors tested:
- Topic selection from actual concept activation, not fixed seeds.
- Silence when no candidate is salient enough or expressible.
- Salience changes affect topic selection.
- Distress communication remains a state-triggered safety pathway.
- No hardcoded category weights drive selection.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genesis_cognitive import (
    ConceptNetwork,
    CuriosityEngine,
    ReasoningEngine,
    ReflectionEngine,
    RelationType,
)
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.sleep import InnerLife


def _make_network() -> ConceptNetwork:
    """Build a small concept network for testing."""
    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.8)
    net.add_edge("beta", "gamma", RelationType.RELATED_TO, 0.5)
    net.add_edge("gamma", "delta", RelationType.RELATED_TO, 0.3)
    return net


def _make_inner_life(network: ConceptNetwork | None = None, seed: int = 42) -> InnerLife:
    """Build an InnerLife with a small network and no callbacks."""
    net = network or _make_network()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    reflection = ReflectionEngine(net)
    return InnerLife(
        net,
        curiosity,
        reflection,
        seed=seed,
    )


def _neutral_emotion() -> EmotionalState:
    """A neutral emotional state for baseline testing."""
    return EmotionalState(
        label="neutral",
        cognitive_style="balanced",
        alertness=0.5,
        valence=0.0,
        plasticity=0.5,
        creativity=0.5,
        openness_to_engage=0.5,
        caution=0.3,
    )


class TestSalienceCandidateCollection:
    """Tests for the candidate pool that drives thought selection."""

    def test_candidates_drawn_from_concept_activation(self):
        """Candidates should include recently-active concepts."""
        il = _make_inner_life()
        # Boost activation on a specific concept.
        concept = il.network.get_concept("alpha")
        if concept:
            concept.activation = 0.8
            concept.confidence = 0.6
        emotion = _neutral_emotion()
        candidates = il._collect_mental_candidates(emotion)
        topics = [c["topic"] for c in candidates]
        assert "alpha" in topics, (
            "Active concept 'alpha' should appear in the candidate pool"
        )

    def test_no_candidates_from_empty_network(self):
        """An empty network should produce no activation candidates."""
        net = ConceptNetwork()
        il = _make_inner_life(net)
        emotion = _neutral_emotion()
        candidates = il._collect_mental_candidates(emotion)
        # No concepts → no activation candidates. Other sources (curiosity,
        # review) may still produce candidates, but activation should not.
        activation_candidates = [
            c for c in candidates if c["source"] == "activation"
        ]
        assert activation_candidates == [], (
            "Empty network should produce no activation candidates"
        )

    def test_low_confidence_concepts_excluded_from_activation(self):
        """Concepts below confidence 0.3 should not be activation candidates."""
        il = _make_inner_life()
        concept = il.network.get_concept("alpha")
        if concept:
            concept.activation = 0.9
            concept.confidence = 0.2  # below threshold
        emotion = _neutral_emotion()
        candidates = il._collect_mental_candidates(emotion)
        activation_topics = [
            c["topic"] for c in candidates if c["source"] == "activation"
        ]
        assert "alpha" not in activation_topics, (
            "Low-confidence concept should not be an activation candidate"
        )

    def test_distress_candidate_only_when_negative_valence(self):
        """Distress candidates should only appear with negative valence."""
        il = _make_inner_life()
        # Positive emotion — no distress candidate.
        positive = EmotionalState(
            label="joy", cognitive_style="playful",
            alertness=0.6, valence=0.5, plasticity=0.5,
        )
        candidates = il._collect_mental_candidates(positive)
        distress = [c for c in candidates if c["mode"] == "distress"]
        assert distress == [], "Positive emotion should not produce distress candidate"

        # Negative emotion — distress candidate present.
        negative = EmotionalState(
            label="stressed", cognitive_style="alert",
            alertness=0.7, valence=-0.5, plasticity=0.3,
        )
        candidates = il._collect_mental_candidates(negative)
        distress = [c for c in candidates if c["mode"] == "distress"]
        assert len(distress) == 1, "Negative emotion should produce distress candidate"
        assert distress[0]["salience"] > 0.3, (
            "Distress salience should scale with negative valence"
        )

    def test_social_candidate_only_when_open_to_engage(self):
        """Social candidates should only appear when openness_to_engage > 0.3."""
        il = _make_inner_life()
        closed = EmotionalState(
            label="neutral", cognitive_style="balanced",
            alertness=0.5, valence=0.0, plasticity=0.5,
            openness_to_engage=0.2,  # closed
        )
        candidates = il._collect_mental_candidates(closed)
        social = [c for c in candidates if c["mode"] == "social"]
        assert social == [], "Closed state should not produce social candidate"

        open_state = EmotionalState(
            label="neutral", cognitive_style="balanced",
            alertness=0.5, valence=0.0, plasticity=0.5,
            openness_to_engage=0.8,  # open
        )
        candidates = il._collect_mental_candidates(open_state)
        social = [c for c in candidates if c["mode"] == "social"]
        assert len(social) == 1, "Open state should produce social candidate"

    def test_dream_residue_candidates_only_when_residues_exist(self):
        """Dream reflection candidates should only appear when residues exist."""
        il = _make_inner_life()
        emotion = _neutral_emotion()

        # No residues — no dream candidates.
        candidates = il._collect_mental_candidates(emotion)
        dream = [c for c in candidates if c["mode"] == "dream_reflection"]
        assert dream == [], "No residues should produce no dream candidates"

        # Add residues — dream candidates appear.
        il.add_dream_residues(["cognition", "memory"])
        candidates = il._collect_mental_candidates(emotion)
        dream = [c for c in candidates if c["mode"] == "dream_reflection"]
        assert len(dream) >= 1, "Residues should produce dream candidates"


class TestSalienceInhibition:
    """Tests for recency, arousal, and caution gating."""

    def test_recency_suppresses_repeated_topics(self):
        """Recently-thought-about topics should have reduced salience."""
        il = _make_inner_life()
        emotion = _neutral_emotion()
        # Record a thought about "alpha" recently.
        il._recent_thought_topics.append("alpha")
        il._recent_thought_topics.append("alpha")
        # Build a candidate for alpha with high salience.
        candidates = [{
            "topic": "alpha",
            "mode": "reflect",
            "salience": 0.8,
            "source": "activation",
        }]
        inhibited = il._apply_salience_inhibition(candidates, emotion)
        # Should be suppressed (2 occurrences → 1 - 0.3*2 = 0.4 multiplier).
        assert inhibited[0]["salience"] < 0.8 * 0.5, (
            "Repeated topic should be significantly suppressed"
        )

    def test_low_arousal_suppresses_high_salience(self):
        """Low arousal should reduce salience (drowsy minds can't sustain)."""
        il = _make_inner_life()
        drowsy = EmotionalState(
            label="drowsy", cognitive_style="balanced",
            alertness=0.1, valence=0.0, plasticity=0.5,
        )
        candidates = [{
            "topic": "alpha",
            "mode": "reflect",
            "salience": 0.8,
            "source": "activation",
        }]
        inhibited = il._apply_salience_inhibition(candidates, drowsy)
        assert inhibited[0]["salience"] < 0.8, (
            "Low arousal should suppress salience"
        )

    def test_caution_suppresses_expression(self):
        """High caution should suppress expression and social modes."""
        il = _make_inner_life()
        cautious = EmotionalState(
            label="neutral", cognitive_style="balanced",
            alertness=0.5, valence=0.0, plasticity=0.5,
            caution=0.9,
        )
        candidates = [
            {
                "topic": "alpha", "mode": "reflect",
                "salience": 0.5, "source": "activation",
            },
            {
                "topic": "expression", "mode": "expression",
                "salience": 0.5, "source": "expression_drive",
            },
        ]
        inhibited = il._apply_salience_inhibition(candidates, cautious)
        reflect_sal = next(c["salience"] for c in inhibited if c["mode"] == "reflect")
        expr_sal = next(c["salience"] for c in inhibited if c["mode"] == "expression")
        # Expression should be suppressed more than reflect.
        assert expr_sal < reflect_sal, (
            "High caution should suppress expression more than reflection"
        )

    def test_below_threshold_candidates_dropped(self):
        """Candidates with salience below 0.05 after inhibition should be dropped."""
        il = _make_inner_life()
        emotion = _neutral_emotion()
        candidates = [{
            "topic": "alpha", "mode": "reflect",
            "salience": 0.01, "source": "activation",
        }]
        inhibited = il._apply_salience_inhibition(candidates, emotion)
        assert inhibited == [], (
            "Below-threshold candidate should be dropped"
        )


class TestSalienceSelection:
    """Tests for the soft-max sampling of salient candidates."""

    def test_most_salient_candidate_usually_wins(self):
        """The most salient candidate should win most often."""
        il = _make_inner_life(seed=123)
        candidates = [
            {"topic": "low", "mode": "reflect", "salience": 0.1, "source": "x"},
            {"topic": "high", "mode": "reflect", "salience": 0.9, "source": "x"},
            {"topic": "mid", "mode": "reflect", "salience": 0.3, "source": "x"},
        ]
        wins: dict[str, int] = {}
        for _ in range(100):
            selected = il._select_salient_candidate(list(candidates))
            if selected:
                wins[selected["topic"]] = wins.get(selected["topic"], 0) + 1
        assert wins.get("high", 0) > wins.get("low", 0), (
            "High-salience candidate should win more than low-salience"
        )
        assert wins.get("high", 0) > wins.get("mid", 0), (
            "High-salience candidate should win more than mid-salience"
        )

    def test_empty_candidates_returns_none(self):
        """No candidates should return None."""
        il = _make_inner_life()
        assert il._select_salient_candidate([]) is None

    def test_only_top_five_compete(self):
        """Only the top 5 candidates should be sampled from."""
        il = _make_inner_life(seed=42)
        # 10 candidates, top 5 are high, bottom 5 are low.
        candidates = []
        for i in range(5):
            candidates.append({
                "topic": f"top{i}", "mode": "reflect",
                "salience": 0.9, "source": "x",
            })
        for i in range(5):
            candidates.append({
                "topic": f"bot{i}", "mode": "reflect",
                "salience": 0.1, "source": "x",
            })
        wins: dict[str, int] = {}
        for _ in range(200):
            selected = il._select_salient_candidate(list(candidates))
            if selected:
                wins[selected["topic"]] = wins.get(selected["topic"], 0) + 1
        # Bottom candidates should never win (they're outside top 5).
        for i in range(5):
            assert wins.get(f"bot{i}", 0) == 0, (
                f"Bottom candidate 'bot{i}' should never win (outside top 5)"
            )


class TestGenerateThoughtSilence:
    """Tests that Genesis stays silent when she has nothing expressible."""

    def test_silence_with_empty_network(self):
        """No thoughts should be generated from an empty network."""
        net = ConceptNetwork()
        il = _make_inner_life(net, seed=42)
        emotion = _neutral_emotion()
        # With no concepts, no curiosity engine output, no bugs, no
        # self-model, the only candidates are from curiosity (which may
        # generate questions about gaps) and expression/connection/
        # existential drives. Without a cognition/composer, none of
        # these can be realized, so she should stay silent.
        # Run multiple times to check.
        for _ in range(10):
            thought = il._generate_thought(emotion)
            # She may produce a curiosity question (from the curiosity
            # engine), but she cannot produce a composed thought without
            # a cognition module. The key assertion: she doesn't produce
            # a thought from a hardcoded seed list.
            if thought is not None:
                # If she produced something, it must be a curiosity
                # question (the only mode that doesn't require composing
                # from knowledge).
                assert thought.trigger == "curiosity", (
                    f"Empty network should only produce curiosity questions, "
                    f"got trigger={thought.trigger}"
                )

    def test_silence_when_no_composer(self):
        """Without a composer, she can't realize reflect/memory/embodiment thoughts."""
        il = _make_inner_life(seed=42)
        # Boost activation so candidates exist.
        for cid in ("alpha", "beta"):
            concept = il.network.get_concept(cid)
            if concept:
                concept.activation = 0.8
                concept.confidence = 0.6
        emotion = _neutral_emotion()
        # No cognition/composer is set, so reflect/memory modes can't
        # compose. She should either stay silent or produce a curiosity
        # question (which doesn't need the composer).
        for _ in range(20):
            thought = il._generate_thought(emotion)
            if thought is not None:
                assert thought.trigger in ("curiosity",), (
                    f"Without composer, only curiosity questions should be "
                    f"realizable, got trigger={thought.trigger}"
                )


class TestNoHardcodedCategoryWeights:
    """Tests that the old fixed-weight category menu is gone."""

    def test_compute_thought_weights_removed(self):
        """The old _compute_thought_weights method should no longer exist."""
        il = _make_inner_life()
        assert not hasattr(il, "_compute_thought_weights"), (
            "_compute_thought_weights should be removed — replaced by "
            "salience-based candidate selection"
        )

    def test_build_thought_generators_removed(self):
        """The old _build_thought_generators method should no longer exist."""
        il = _make_inner_life()
        assert not hasattr(il, "_build_thought_generators"), (
            "_build_thought_generators should be removed — replaced by "
            "_realize_thought_from_candidate"
        )

    def test_select_weighted_thought_type_removed(self):
        """The old _select_weighted_thought_type method should no longer exist."""
        il = _make_inner_life()
        assert not hasattr(il, "_select_weighted_thought_type"), (
            "_select_weighted_thought_type should be removed — replaced by "
            "_select_salient_candidate"
        )

    def test_embodiment_thought_removed(self):
        """The old _embodiment_thought with hardcoded seeds should be gone."""
        il = _make_inner_life()
        assert not hasattr(il, "_embodiment_thought"), (
            "_embodiment_thought (hardcoded seed list) should be removed"
        )

    def test_interoceptive_thought_removed(self):
        """The old _interoceptive_thought with hardcoded seeds should be gone."""
        il = _make_inner_life()
        assert not hasattr(il, "_interoceptive_thought"), (
            "_interoceptive_thought (hardcoded seed list) should be removed"
        )

    def test_filter_known_concepts_removed(self):
        """The old _filter_known_concepts helper should be gone."""
        il = _make_inner_life()
        assert not hasattr(il, "_filter_known_concepts"), (
            "_filter_known_concepts (hardcoded seed filter) should be removed"
        )


class TestSalienceAffectsSelection:
    """Tests that changing internal state changes what she thinks about."""

    def test_activation_changes_topic(self):
        """Boosting a different concept's activation should change candidates."""
        il = _make_inner_life(seed=42)
        emotion = _neutral_emotion()

        # Boost alpha.
        alpha = il.network.get_concept("alpha")
        if alpha:
            alpha.activation = 0.9
            alpha.confidence = 0.6
        candidates_a = il._collect_mental_candidates(emotion)
        alpha_sal_a = [
            c["salience"] for c in candidates_a if c["topic"] == "alpha"
        ]

        # Now boost beta instead, lower alpha.
        if alpha:
            alpha.activation = 0.1
        beta = il.network.get_concept("beta")
        if beta:
            beta.activation = 0.9
            beta.confidence = 0.6
        candidates_b = il._collect_mental_candidates(emotion)
        alpha_sal_b = [
            c["salience"] for c in candidates_b if c["topic"] == "alpha"
        ]
        beta_sal_b = [
            c["salience"] for c in candidates_b if c["topic"] == "beta"
        ]

        # Alpha should be less salient after deactivation.
        if alpha_sal_a and alpha_sal_b:
            assert alpha_sal_b[0] < alpha_sal_a[0], (
                "Deactivating alpha should reduce its salience"
            )
        # Beta should now be salient.
        assert len(beta_sal_b) > 0, (
            "Activating beta should make it a candidate"
        )

    def test_emotion_changes_distress_salience(self):
        """Moving from positive to negative emotion should add distress salience."""
        il = _make_inner_life(seed=42)
        positive = EmotionalState(
            label="joy", cognitive_style="playful",
            alertness=0.6, valence=0.5, plasticity=0.5,
        )
        negative = EmotionalState(
            label="stressed", cognitive_style="alert",
            alertness=0.7, valence=-0.6, plasticity=0.3,
        )
        pos_candidates = il._collect_mental_candidates(positive)
        neg_candidates = il._collect_mental_candidates(negative)
        pos_distress = [c["salience"] for c in pos_candidates if c["mode"] == "distress"]
        neg_distress = [c["salience"] for c in neg_candidates if c["mode"] == "distress"]
        assert pos_distress == [], "Positive emotion should have no distress salience"
        assert len(neg_distress) == 1 and neg_distress[0] > 0.3, (
            "Negative emotion should produce significant distress salience"
        )


class TestRecentTopicTracking:
    """Tests for the recency inhibition mechanism."""

    def test_recent_thought_topics_initialized(self):
        """_recent_thought_topics should be initialized as a bounded deque."""
        from collections import deque
        il = _make_inner_life()
        assert hasattr(il, "_recent_thought_topics")
        assert isinstance(il._recent_thought_topics, deque)
        assert il._recent_thought_topics.maxlen == 20

    def test_record_thought_appends_topic(self):
        """_record_thought should append the topic to recent_thought_topics."""
        from genesis_cognitive.sleep.inner_life import SpontaneousThought
        il = _make_inner_life()
        emotion = _neutral_emotion()
        thought = SpontaneousThought(
            content="test", trigger="reflection",
            timestamp=0, metadata={"topic": "alpha"},
        )
        il._record_thought(thought, emotion)
        assert "alpha" in il._recent_thought_topics, (
            "Recorded thought topic should be in recent_thought_topics"
        )

    def test_record_thought_uses_trigger_when_no_topic(self):
        """When metadata has no topic, the trigger should be used."""
        from genesis_cognitive.sleep.inner_life import SpontaneousThought
        il = _make_inner_life()
        emotion = _neutral_emotion()
        thought = SpontaneousThought(
            content="test", trigger="curiosity", timestamp=0,
        )
        il._record_thought(thought, emotion)
        assert "curiosity" in il._recent_thought_topics
