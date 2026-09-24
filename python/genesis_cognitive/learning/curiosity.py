"""Curiosity — what makes Genesis want to learn.

Curiosity is not a passive state. It's an active drive: the mind
identifies gaps in its understanding and generates questions to fill
them. This is what makes a mind *want* to learn rather than just
*being able* to learn.

# How curiosity works

1. **Gap detection**: The concept network has a missing edge or a
   low-confidence concept. The curiosity engine notices this.

2. **Question generation**: Given a gap, formulate a question that
   would fill it. "I know A is related to B, but how?"

3. **Curiosity drive**: The emotional system modulates curiosity.
   High plasticity + positive valence → high curiosity. Low
   plasticity → low curiosity (the mind is rigid).

4. **Autonomous exploration**: When not in conversation, Genesis
   can explore its own concept network, finding gaps and forming
   hypotheses. This is contemplation — thinking without input.

# Types of curiosity

Curiosity is not monolithic. Following Kidd & Hayden (2015), we
distinguish three types, each with distinct neurochemical signatures:

- **Perceptual curiosity**: novelty in input — "what is this?"
  Driven by dopamine (DA). The mind notices something new and wants
  to explore it. This is the most basic form — sensory novelty.

- **Epistemic curiosity**: knowledge gaps — "why does this work?"
  Driven by acetylcholine (ACh) and the frontopolar cortex. The mind
  knows what it doesn't know and wants to fill the gap. This is
  higher-order curiosity about understanding, not just novelty.

- **Social curiosity**: questions about others' mental states —
  "what are you thinking?" Driven by oxytocin (OXY) and theory-of-mind
  networks. The mind is curious about other minds — their beliefs,
  intentions, and knowledge.

Each type is tracked and generated independently, with different
neurochemical signatures that modulate their intensity.

# Information gain

Not all questions are equally worth asking. The engine estimates
expected information gain for each potential question using entropy
reduction: ``info_gain = H(current_state) - H(state | answer)``.
Questions with higher expected information gain are prioritized,
following the Kidd & Hayden (2015) model where curiosity is
driven by the expected value of information.

# Habituation

Repeated exposure to the same knowledge gap reduces curiosity about
it — the mind habituates. Curiosity decreases with
``1 / (1 + exposure_count)``. But if the gap evolves (new information
appears, the concept's confidence changes), curiosity resets — the
mind re-engages with a gap that has become interesting again.

The curiosity engine is what makes Genesis more than a reactive
system. It has its own drive to understand.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

from ..brain_waves import BrainWave, BrainWaveState
from ..concepts import (
    QUALITY_THRESHOLD,
    ConceptNetwork,
    RelationType,
    is_world_concept,
)
from ..emotion import EmotionalState
from ..reasoning import ReasoningEngine, ReasoningResult, ReasoningType

__all__ = ["CuriosityEngine", "CuriosityType", "Question"]

logger = logging.getLogger(__name__)


class CuriosityType(Enum):
    """The type of curiosity a question arises from.

    Each type has a distinct neurochemical signature and cognitive
    origin, following Kidd & Hayden (2015):

    - PERCEPTUAL: novelty-driven, "what is this?" — dopamine (DA)
    - EPISTEMIC: knowledge-gap-driven, "why does this work?" —
      acetylcholine (ACh), frontopolar cortex
    - SOCIAL: other-minds-driven, "what are you thinking?" —
      oxytocin (OXY), theory-of-mind networks
    """

    PERCEPTUAL = "perceptual"
    EPISTEMIC = "epistemic"
    SOCIAL = "social"

    @property
    def primary_neurochemical(self) -> str:
        """The primary neurochemical driving this curiosity type."""
        return _CURIOSITY_NEUROCHEMISTRY[self]


# Neurochemical signatures for each curiosity type.
# Maps CuriosityType → primary neurochemical name.
# Perceptual = dopamine (DA), Epistemic = acetylcholine (ACh),
# Social = oxytocin (OXY).
_CURIOSITY_NEUROCHEMISTRY: dict[CuriosityType, str] = {
    CuriosityType.PERCEPTUAL: "dopamine",
    CuriosityType.EPISTEMIC: "acetylcholine",
    CuriosityType.SOCIAL: "oxytocin",
}


@dataclass(slots=True)
class Question:
    """A question Genesis generates from curiosity.

    Not all questions are asked out loud. Some are internal —
    Genesis wondering to itself. The `should_ask` field determines
    whether this question should be posed to the user.

    The `text` field is composed by the GenerativeEngine from the
    semantic metadata (question_type, target_concept, gap_detail),
    not hardcoded. The `text` field may be empty until composed.
    """

    text: str  # composed by GenerativeEngine (may be empty until then)
    target_concept: str  # what concept it's about
    gap_type: str  # "missing_edge", "low_confidence", "contradiction"
    curiosity_score: float  # 0..1, how curious it is
    should_ask: bool = False  # should it ask the user?
    internal: bool = True  # is this internal musing?
    # Category of question — drives the curiosity→learning bridge.
    # "isolation", "uncertainty", "causation" can be answered by
    # learning (web search). "hypothesis" and "contradiction" need
    # reasoning, not search.
    question_type: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    # Which type of curiosity produced this question — perceptual
    # (novelty, DA-driven), epistemic (knowledge gap, ACh-driven), or
    # social (other minds, OXY-driven). See CuriosityType.
    curiosity_type: CuriosityType = CuriosityType.PERCEPTUAL
    # Expected information gain — how much the answer would reduce
    # uncertainty (entropy reduction). Higher = more worth asking.
    # Computed by compute_information_gap(); 0.0 if not yet computed.
    info_gain: float = 0.0
    # Extra detail about the gap — e.g. a neighbor it's uncertain
    # about, a hypothesis conclusion, a contradiction description.
    # Used by the GenerativeEngine to compose the question text.
    gap_detail: str = ""
    # Cognitive reason for the question — a one-sentence goal.
    reason: str = ""

    def describe(self) -> str:
        """First-person description."""
        prefix = "I wonder" if self.internal else "Can I ask"
        return f"{prefix}: {self.text}"


class CuriosityEngine:
    """Generates questions from gaps in understanding.

    The curiosity engine scans the concept network for gaps and
    generates questions that would fill them. It's modulated by
    emotional state — high plasticity means more curiosity.

    Three types of curiosity are tracked and generated independently:

    - **Perceptual** (DA-driven): novelty in input — "what is this?"
    - **Epistemic** (ACh-driven): knowledge gaps — "why does this work?"
    - **Social** (OXY-driven): other minds — "what are you thinking?"

    Each type has a different neurochemical signature and is modulated
    by different emotional factors. Questions are prioritized by
    expected information gain (entropy reduction), following the
    Kidd & Hayden (2015) model.

    Repeated exposure to the same knowledge gap reduces curiosity
    (habituation), but if the gap evolves, curiosity resets.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        reasoning: ReasoningEngine,
    ) -> None:
        """Initialize the curiosity engine with a concept network and reasoner.

        Args:
            network: The concept network used to find knowledge gaps.
            reasoning: The reasoning engine used to evaluate gap relevance.
        """
        self.network = network
        self.reasoning = reasoning
        # Track recently asked questions to avoid repeating them.
        # The full deque holds 500 entries; _is_novel checks against
        # all of them (not just the last 20) so questions don't expire
        # from the novelty window too quickly and get re-asked.
        self._asked_questions: deque[str] = deque(maxlen=500)
        self._asked_questions_maxlen: int = 500
        # O(1) lookup companion to _asked_questions for novelty checks.
        self._asked_questions_set: set[str] = set()
        # Concepts whose questions have been resolved (learned about).
        # generate_questions() skips these so it doesn't repeatedly
        # wonder about something it just learned. The set is large
        # (2000) so resolved concepts don't get evicted and re-asked
        # after the user already answered them.
        self._resolved_questions: deque[str] = deque(maxlen=2000)
        self._resolved_questions_maxlen: int = 2000
        # O(1) lookup companion to _resolved_questions.
        self._resolved_questions_set: set[str] = set()

        # ─── Habituation ──────────────────────────────────────────
        # Track how many times each knowledge gap has been encountered.
        # Repeated exposure reduces curiosity via 1/(1+exposure_count).
        # Key: "{concept_id}:{gap_type}" — uniquely identifies a gap.
        # Value: exposure count (how many times we've generated a
        # question about this gap).
        self._exposure_counts: dict[str, int] = {}
        # Snapshot of concept confidence at last exposure — used to
        # detect gap evolution (if confidence changed, reset habituation).
        self._last_exposure_confidence: dict[str, float] = {}

        # ─── Co-occurrence curiosity ──────────────────────────────
        # Track the concepts active in recent turns. When two concepts
        # keep showing up together but have no causal/dependency edge,
        # the engine asks about their relationship. This turns passive
        # observation into active hypothesis generation.
        self._concept_history: deque[set[str]] = deque(maxlen=5)

    def mark_resolved(self, concept: str) -> None:
        """Mark a concept's question as resolved.

        Called by the autonomous learner after it successfully learns
        about a concept that came from a curiosity question. This
        prevents the curiosity engine from re-generating questions
        about something it just understood.
        """
        concept = concept.lower().strip()
        if concept and concept not in self._resolved_questions_set:
            self._resolved_questions.append(concept)
            self._resolved_questions_set.add(concept)
            # Keep the set in sync with the deque's maxlen pruning.
            if len(self._resolved_questions_set) > self._resolved_questions_maxlen:
                self._resolved_questions_set = set(self._resolved_questions)
            logger.debug(f"Curiosity question resolved for '{concept}'")

    def assess_curiosity(
        self,
        emotion: EmotionalState,
        brain_waves: BrainWaveState | None = None,
    ) -> float:
        """How curious is Genesis right now?

        Curiosity is modulated by:
        - Plasticity (high plasticity → high curiosity)
        - Valence (positive mood → more exploratory)
        - Alertness (moderate alertness is optimal — too low = lethargic,
          too high = anxious)
        - Cognitive style (exploratory → curious, rigid → not)
        - Brain waves (gamma → exploratory, delta → suppressed,
          theta → consolidation-focused, alpha → filtered)
        """
        # Inverted-U alertness curve (Yerkes-Dodson)
        alertness_optimum = 0.5
        alertness_factor = 1.0 - abs(emotion.alertness - alertness_optimum) * 1.5
        alertness_factor = max(0.1, alertness_factor)

        curiosity = (
            emotion.plasticity * 0.4
            + max(0, emotion.valence) * 0.2
            + alertness_factor * 0.2
            + emotion.creativity * 0.2
        )

        # Brain-wave modulation. Gamma boosts curiosity (exploratory
        # drive; gamma predicts encoding success, Osipova et al. 2006).
        # Delta suppresses it (deep rest). Alpha slightly reduces it
        # (filtering, not exploring). Theta shifts it toward
        # consolidation rather than acquisition but doesn't suppress
        # the drive itself.
        if brain_waves is not None:
            dom = brain_waves.dominant
            if dom == BrainWave.GAMMA:
                curiosity += 0.15 * brain_waves.integration
            elif dom == BrainWave.DELTA:
                curiosity -= 0.3
            elif dom == BrainWave.ALPHA:
                curiosity -= 0.1

        return max(0.0, min(1.0, curiosity))

    def generate_questions(
        self,
        emotion: EmotionalState,
        active_concepts: list[str] | None = None,
        max_questions: int = 3,
        brain_waves: BrainWaveState | None = None,
    ) -> list[Question]:
        """Generate questions from gaps in understanding.

        If active_concepts is provided, questions focus on those.
        Otherwise, the most activated concepts in the network are used.

        Questions are tagged with a CuriosityType (perceptual, epistemic,
        or social) and scored by expected information gain. Habituation
        reduces curiosity for repeatedly-encountered gaps, unless the
        gap has evolved (concept confidence changed since last exposure).

        Args:
            brain_waves: Optional brain wave state. Gamma boosts the
                number of questions generated (exploratory drive).
                Delta suppresses question generation entirely (deep
                rest). Theta shifts focus to consolidation-style
                questions (reviewing what it already knows rather
                than seeking new topics).
        """
        curiosity_level = self.assess_curiosity(emotion, brain_waves)
        if curiosity_level < 0.2:
            return []  # not curious enough to generate questions

        # Brain-wave-modulated question count.
        # Gamma → more questions (exploratory). Delta → none (rest).
        # Theta → fewer, consolidation-focused questions.
        effective_max = max_questions
        if brain_waves is not None:
            dom = brain_waves.dominant
            if dom == BrainWave.GAMMA:
                effective_max = max_questions + 1
            elif dom == BrainWave.THETA:
                effective_max = max(1, max_questions - 1)

        focus = self._collect_focus_concepts(active_concepts)
        if not focus:
            return []

        # Track co-occurrences across recent turns.
        self._concept_history.append(set(focus))

        questions: list[Question] = []

        co_q = self._generate_cooccurrence_question(focus, curiosity_level)
        if co_q and self._is_novel(co_q):
            co_q.info_gain = self.compute_information_gap(co_q)
            questions.append(co_q)

        for concept_id in focus:
            if len(questions) >= effective_max:
                break

            # Skip concepts whose questions were recently resolved —
            # it already learned about them, no need to wonder again.
            if concept_id.lower() in self._resolved_questions_set:
                continue

            self._generate_concept_questions(concept_id, curiosity_level, questions)

        return self._finalize_questions(questions, emotion, effective_max)

    def current_focus_concepts(self) -> list[str]:
        """Return the strongest current world-concept attentional targets."""
        return self._collect_focus_concepts(None)

    def _collect_focus_concepts(
        self, active_concepts: list[str] | None
    ) -> list[str]:
        """Determine which concepts to focus question generation on.

        When concepts are explicitly active (from conversation), we
        accept any world concept — even low-quality ones, because the
        user is talking about them and curiosity should follow.

        When selecting from activation (idle curiosity), we prefer
        high-quality concepts — those with definitions, typed edges,
        or high confidence. This prevents it from wondering about
        empty vocabulary that was never integrated into its knowledge.
        """
        if active_concepts:
            focus = []
            for name in active_concepts:
                cid = self.network._resolve(name)
                if cid and is_world_concept(cid):
                    focus.append(cid)
            return focus
        # Use most activated concepts — but only world concepts with
        # sufficient quality. Without this filter, code symbols
        # (python:mind.deny_site), function words (and, it), and
        # empty vocabulary generate nonsense questions.
        candidates = [
            (cid, act)
            for cid, act in self.network.most_activated(20)
            if is_world_concept(cid)
        ]
        # Partition into high-quality and low-quality.
        # If we have enough high-quality concepts (≥3), use only those.
        # Otherwise, fall back to low-quality ones (better than nothing).
        high_q = [
            cid for cid, _ in candidates
            if self.network.concept_quality(cid) >= QUALITY_THRESHOLD
        ]
        if len(high_q) >= 3:
            return high_q[:5]
        # Not enough high-quality — use all world concepts by activation
        return [cid for cid, _ in candidates][:5]

    def _generate_concept_questions(
        self,
        concept_id: str,
        curiosity_level: float,
        questions: list[Question],
    ) -> None:
        """Generate gap questions for a single concept (mutates questions)."""
        # 1. Missing edges: concept has few connections (perceptual)
        edges = self.network.get_edges(concept_id, "both")
        if len(edges) < 2:
            q = self._question_about_isolation(concept_id, curiosity_level)
            if q and self._is_novel(q):
                self._apply_habituation(q)
                q.info_gain = self.compute_information_gap(q)
                questions.append(q)

        # 2. Low confidence concepts (perceptual — "what is this?")
        concept = self.network.get_concept(concept_id)
        if concept and concept.confidence < 0.3:
            q = self._question_about_uncertainty(concept_id, curiosity_level)
            if q and self._is_novel(q):
                self._apply_habituation(q)
                q.info_gain = self.compute_information_gap(q)
                questions.append(q)

        # 3. Missing relationship type: concept has IS_A but no
        #    CAUSES (epistemic — "why does this work?")
        has_is_a = any(e.relation == RelationType.IS_A for e in edges)
        has_causes = any(e.relation == RelationType.CAUSES for e in edges)
        if has_is_a and not has_causes:
            q = self._question_about_cause(concept_id, curiosity_level)
            if q and self._is_novel(q):
                self._apply_habituation(q)
                q.info_gain = self.compute_information_gap(q)
                questions.append(q)

        # 4. Hypotheses from reasoning (epistemic)
        results = self.reasoning.reason_about(concept_id, depth=2)
        for result in results:
            if result.reasoning_type == ReasoningType.HYPOTHESIS:
                q = self._question_from_hypothesis(concept_id, result, curiosity_level)
                if q and self._is_novel(q):
                    self._apply_habituation(q)
                    q.info_gain = self.compute_information_gap(q)
                    questions.append(q)
                    break  # one hypothesis question per concept

        # 5. Contradictions (epistemic)
        for result in results:
            if result.reasoning_type == ReasoningType.CONTRADICTION:
                q = self._question_about_contradiction(concept_id, result, curiosity_level)
                if q and self._is_novel(q):
                    self._apply_habituation(q)
                    q.info_gain = self.compute_information_gap(q)
                    questions.append(q)
                    break

    def _finalize_questions(
        self,
        questions: list[Question],
        emotion: EmotionalState,
        max_questions: int,
    ) -> list[Question]:
        """Sort, set should_ask, and truncate the question list."""
        # Sort by combined curiosity score and information gain.
        # Questions with high info gain are prioritized, weighted by
        # curiosity score — a gap it's not curious about isn't worth
        # asking even if it's information-rich.
        questions.sort(key=lambda q: -(q.curiosity_score * 0.6 + q.info_gain * 0.4))

        # Determine which to ask vs. keep internal
        for q in questions:
            # Ask out loud if curiosity is high and it's about an active topic
            q.should_ask = (
                q.curiosity_score > 0.5 and not q.internal
            )

        return questions[:max_questions]

    def _question_about_isolation(self, concept_id: str, curiosity: float) -> Question:
        """Generate a question about an isolated concept.

        This is perceptual curiosity — "what is this?" The concept
        exists but has few connections; the novelty of an isolated
        node drives dopamine-mediated exploration.

        The text is composed later by the GenerativeEngine from the
        semantic metadata (question_type="isolation", target_concept).
        """
        return Question(
            text="",
            target_concept=concept_id,
            gap_type="missing_edge",
            curiosity_score=curiosity * 0.7,
            internal=False,  # a good question to ask the user
            question_type="isolation",
            curiosity_type=CuriosityType.PERCEPTUAL,
        )

    def _question_about_uncertainty(self, concept_id: str, curiosity: float) -> Question:
        """Generate a question about a low-confidence concept.

        This is perceptual curiosity — the concept is vague and
        ill-defined, a novelty signal that drives exploration.

        The text is composed later by the GenerativeEngine from the
        semantic metadata (question_type="uncertainty", target_concept).
        """
        return Question(
            text="",
            target_concept=concept_id,
            gap_type="low_confidence",
            curiosity_score=curiosity * 0.8,
            internal=False,  # this is a good question to ask
            question_type="uncertainty",
            curiosity_type=CuriosityType.PERCEPTUAL,
        )

    def _question_about_cause(self, concept_id: str, curiosity: float) -> Question:
        """Generate a question about causation.

        This is epistemic curiosity — "why does this work?" The
        concept is known (has IS_A relations) but its causal
        structure is missing. This is higher-order curiosity about
        understanding, driven by acetylcholine and the frontopolar
        cortex.

        The text is composed later by the GenerativeEngine from the
        semantic metadata (question_type="causation", target_concept).
        """
        return Question(
            text="",
            target_concept=concept_id,
            gap_type="missing_edge",
            curiosity_score=curiosity * 0.75,
            internal=False,
            question_type="causation",
            curiosity_type=CuriosityType.EPISTEMIC,
        )

    def _question_from_hypothesis(
        self, concept_id: str, result: ReasoningResult, curiosity: float
    ) -> Question:
        """Generate a question from a reasoning hypothesis.

        This is epistemic curiosity — the reasoning engine has
        produced a hypothesis, and the mind wants to verify it.

        The text is composed later by the GenerativeEngine from the
        semantic metadata (question_type="hypothesis", gap_detail).
        """
        return Question(
            text="",
            target_concept=concept_id,
            gap_type="missing_edge",
            curiosity_score=curiosity * result.confidence,
            internal=False,
            question_type="hypothesis",
            curiosity_type=CuriosityType.EPISTEMIC,
            gap_detail=result.conclusion,
        )

    def _question_about_contradiction(
        self, concept_id: str, result: ReasoningResult, curiosity: float
    ) -> Question:
        """Generate a question about a contradiction.

        This is epistemic curiosity — the mind has detected a
        contradiction in its knowledge and wants to resolve it.
        Contradictions are the most information-rich gaps.

        The text is composed later by the GenerativeEngine from the
        semantic metadata (question_type="contradiction", gap_detail).
        """
        return Question(
            text="",
            target_concept=concept_id,
            gap_type="contradiction",
            curiosity_score=curiosity * 0.9,  # contradictions are very interesting
            internal=False,
            question_type="contradiction",
            curiosity_type=CuriosityType.EPISTEMIC,
            gap_detail=result.conclusion,
        )

    def _generate_cooccurrence_question(
        self, focus: list[str], curiosity: float
    ) -> Question | None:
        """Generate a question about two concepts that keep appearing together.

        This is epistemic curiosity: the mind notices that A and B show up
        in the same turns, it has no edge connecting them, and it asks the
        user about the nature of their relationship so it can learn a causal
        or dependency link.
        """
        if len(self._concept_history) < 2 and len(focus) < 2:
            return None

        counts: dict[tuple[str, str], int] = {}
        for window in self._concept_history:
            for a, b in combinations(sorted(window), 2):
                if a == b:
                    continue
                # Only ask about relationships between genuine world
                # concepts. Without this, it asks "do *and* and *but*
                # cause each other?" — pairing function words or code
                # symbols that happen to co-occur.
                if not is_world_concept(a) or not is_world_concept(b):
                    continue
                counts[(a, b)] = counts.get((a, b), 0) + 1

        if not counts:
            return None

        for (a, b), count in sorted(counts.items(), key=lambda x: -x[1]):
            if count < 1:
                continue
            if self._already_linked(a, b):
                continue
            return Question(
                text="",
                target_concept=a,
                gap_type="missing_edge",
                curiosity_score=max(0.65, min(1.0, curiosity * 0.9)),
                internal=False,
                question_type="cooccurrence",
                curiosity_type=CuriosityType.EPISTEMIC,
                gap_detail=b,
            )

        return None

    def _already_linked(self, a: str, b: str) -> bool:
        """Return True if a and b are already connected by any direct edge."""
        for edge in self.network.get_edges(a, "both"):
            if (edge.source == a and edge.target == b) or (
                edge.source == b and edge.target == a
            ):
                return True
        return False

    def _is_novel(self, question: Question) -> bool:
        """Check if this question hasn't been asked recently."""
        # Check semantic identity — same concept + same gap type = not novel
        q_key = f"{question.target_concept}:{question.question_type}:{question.gap_detail}"
        # Check against the full asked-questions history, not just a
        # small window. This prevents the same question from reappearing
        # after the novelty window expires.
        if q_key in self._asked_questions_set:
            return False

        self._asked_questions.append(q_key)
        self._asked_questions_set.add(q_key)
        # Keep the set in sync with the deque's maxlen pruning.
        if len(self._asked_questions_set) > self._asked_questions_maxlen:
            # Rebuild from the deque (rare, only when at capacity)
            self._asked_questions_set = set(self._asked_questions)
        return True

    # ─── Information gain ──────────────────────────────────────────

    def compute_information_gap(self, question: Question) -> float:
        """Estimate the expected information gain from answering a question.

        Following Kidd & Hayden (2015), curiosity is driven by the
        expected value of information. We estimate this as entropy
        reduction: ``info_gain = H(current_state) - H(state | answer)``.

        The current-state entropy is estimated from the concept's
        uncertainty (low confidence = high entropy) and its structural
        isolation (few edges = many possible connections = high
        entropy). The post-answer entropy is estimated as the residual
        uncertainty after a typical answer resolves part of the gap.

        Returns a float in [0, 1], where higher means the answer would
        provide more information.
        """
        concept = self.network.get_concept(question.target_concept)
        if concept is None:
            # Unknown concept — maximum information gain possible
            return 1.0

        # Current-state entropy: how uncertain is the concept?
        # Low confidence → high entropy (we don't know what it is).
        confidence_entropy = 1.0 - concept.confidence

        # Structural entropy: how many possible connections could it
        # have? A concept with few edges has high structural entropy
        # (many unknowns). We estimate this from edge count relative
        # to a typical well-connected concept.
        edges = self.network.get_edges(question.target_concept, "both")
        n_edges = len(edges)
        # A well-connected concept has ~6+ edges. Fewer = more entropy.
        structural_entropy = max(0.0, 1.0 - n_edges / 6.0)

        # Gap-type-specific entropy weighting:
        # - Contradictions have the highest entropy (two conflicting
        #   beliefs — resolving them provides the most information).
        # - Missing edges have moderate entropy (one unknown).
        # - Low confidence has lower entropy (the concept exists, we
        #   just need to refine it).
        gap_weight = {
            "contradiction": 1.0,
            "missing_edge": 0.8,
            "low_confidence": 0.6,
        }.get(question.gap_type, 0.7)

        # Current-state entropy H(current)
        h_current = (confidence_entropy * 0.4 + structural_entropy * 0.6) * gap_weight

        # Post-answer entropy H(state | answer): a typical answer
        # resolves about half the uncertainty. The exact fraction
        # depends on the question type — contradictions resolve more
        # (one belief is eliminated), low-confidence resolves less
        # (refinement, not elimination).
        resolution_factor = {
            "contradiction": 0.7,  # one belief eliminated
            "missing_edge": 0.5,  # one connection added
            "low_confidence": 0.4,  # confidence improved
        }.get(question.gap_type, 0.5)

        h_post = h_current * (1.0 - resolution_factor)

        # Information gain = H(current) - H(state | answer)
        info_gain = h_current - h_post

        return max(0.0, min(1.0, info_gain))

    # ─── Habituation ───────────────────────────────────────────────

    def _gap_id(self, question: Question) -> str:
        """Construct a unique gap identifier from a question."""
        return f"{question.target_concept.lower()}:{question.gap_type}:{question.question_type}"

    def _apply_habituation(self, question: Question) -> None:
        """Apply habituation to a question's curiosity score.

        Repeated exposure to the same knowledge gap reduces curiosity
        via ``1 / (1 + exposure_count)``. If the gap has evolved
        (concept confidence changed since last exposure), the
        habituation resets — the mind re-engages.

        Also records the exposure for future habituation tracking.
        """
        gap_id = self._gap_id(question)
        concept = self.network.get_concept(question.target_concept)
        current_confidence = concept.confidence if concept else 0.0

        # Check for gap evolution — if confidence changed since last
        # exposure, reset habituation (the gap became interesting again)
        if gap_id in self._last_exposure_confidence:
            prev_confidence = self._last_exposure_confidence[gap_id]
            if abs(current_confidence - prev_confidence) > 0.15:
                # Gap evolved — reset exposure count
                self._exposure_counts[gap_id] = 0
                logger.debug(
                    f"Habituation reset for '{gap_id}' — confidence "
                    f"changed from {prev_confidence:.2f} to {current_confidence:.2f}"
                )

        # Get current exposure count
        exposure = self._exposure_counts.get(gap_id, 0)

        # Apply habituation: curiosity decreases with 1/(1+exposure)
        if exposure > 0:
            habituation_factor = 1.0 / (1.0 + exposure)
            question.curiosity_score *= habituation_factor

        # Record this exposure
        self._exposure_counts[gap_id] = exposure + 1
        self._last_exposure_confidence[gap_id] = current_confidence

    def habituate(self, gap_id: str) -> None:
        """Manually record exposure to a knowledge gap.

        This increments the exposure count for the given gap, reducing
        future curiosity about it. Called when the mind encounters a
        gap without necessarily generating a question (e.g., during
        contemplation or learning).

        If the gap_id is not yet tracked, it starts at 1. If it is
        already tracked, the count increments.
        """
        self._exposure_counts[gap_id] = self._exposure_counts.get(gap_id, 0) + 1

    def get_exposure_count(self, gap_id: str) -> int:
        """Get the exposure count for a knowledge gap.

        Returns 0 if the gap has never been encountered.
        """
        return self._exposure_counts.get(gap_id, 0)

    # ─── Social curiosity ──────────────────────────────────────────

    def generate_social_questions(
        self,
        user_model: dict[str, Any],
        emotion: EmotionalState | None = None,
        max_questions: int = 3,
    ) -> list[Question]:
        """Generate social curiosity questions about the user's mental state.

        Social curiosity is theory-of-mind driven — curiosity about
        others' beliefs, intentions, and knowledge. These questions
        are about the user's inner state: "What are you thinking?"
        "How do you feel about that?"

        Social questions are lower priority than epistemic but higher
        than perceptual — understanding another mind is more valuable
        than sensory novelty but less urgent than resolving a
        knowledge contradiction.

        The ``user_model`` is a dict of what Genesis knows about the
        user (typically from ``SelfModel.world_knowledge`` and
        ``relationship_notes``). Keys may include:
        - "name": the user's name
        - "interests": list of topics the user cares about
        - "emotional_state": observed emotional state
        - "beliefs": things the user has stated they believe
        - "knowledge_level": what the user seems to know about

        Args:
            user_model: What Genesis knows about the user.
            emotion: Current emotional state (affects curiosity level).
                If None, a moderate baseline is used.
            max_questions: Maximum number of questions to generate.

        Returns:
            A list of social curiosity Questions, sorted by curiosity
            score.
        """
        # Determine curiosity level for social questions
        if emotion is not None:
            curiosity_level = self.assess_curiosity(emotion)
        else:
            curiosity_level = 0.5  # moderate baseline

        if curiosity_level < 0.2:
            return []

        questions = self._collect_social_questions(user_model, curiosity_level)

        # Apply habituation and information gain
        for q in questions:
            self._apply_habituation(q)
            q.info_gain = self.compute_information_gap(q)

        # Sort by curiosity score
        questions.sort(key=lambda q: -q.curiosity_score)

        # Determine which to ask
        for q in questions:
            q.should_ask = q.curiosity_score > 0.5 and (
                emotion is None or emotion.openness_to_engage > 0.4
            )

        return questions[:max_questions]

    def _collect_social_questions(
        self,
        user_model: dict[str, Any],
        curiosity_level: float,
    ) -> list[Question]:
        """Build social curiosity questions from the user model.

        The text is composed later by the GenerativeEngine from the
        semantic metadata (question_type, gap_detail).
        """
        questions: list[Question] = []
        user_name = user_model.get("name", "you")
        interests = user_model.get("interests", [])
        emotional_state = user_model.get("emotional_state")
        beliefs = user_model.get("beliefs", [])

        # 1. Curiosity about the user's current mental state
        if emotional_state is None:
            questions.append(
                Question(
                    text="",
                    target_concept=f"{user_name}:emotional_state",
                    gap_type="missing_edge",
                    curiosity_score=curiosity_level * 0.7,
                    internal=False,
                    question_type="social_emotional",
                    curiosity_type=CuriosityType.SOCIAL,
                )
            )

        # 2. Curiosity about the user's interests — why do they care?
        for interest in interests[:2]:
            questions.append(
                Question(
                    text="",
                    target_concept=f"{user_name}:interest_in_{interest}",
                    gap_type="missing_edge",
                    curiosity_score=curiosity_level * 0.65,
                    internal=False,
                    question_type="social_interest",
                    curiosity_type=CuriosityType.SOCIAL,
                    gap_detail=interest,
                )
            )

        # 3. Curiosity about the user's beliefs — what do they think?
        for belief in beliefs[:1]:
            questions.append(
                Question(
                    text="",
                    target_concept=f"{user_name}:belief_about_{belief}",
                    gap_type="missing_edge",
                    curiosity_score=curiosity_level * 0.6,
                    internal=False,
                    question_type="social_belief",
                    curiosity_type=CuriosityType.SOCIAL,
                    gap_detail=belief,
                )
            )

        # 4. General theory-of-mind question if we don't know much
        if not questions:
            questions.append(
                Question(
                    text="",
                    target_concept=f"{user_name}:thoughts",
                    gap_type="missing_edge",
                    curiosity_score=curiosity_level * 0.5,
                    internal=False,
                    question_type="social_general",
                    curiosity_type=CuriosityType.SOCIAL,
                )
            )

        return questions

    def contemplate(self, emotion: EmotionalState) -> list[Question]:
        """Autonomous contemplation — thinking without input.

        When Genesis is not in conversation, it can contemplate:
        explore its concept network, find gaps, and form questions.
        This is internal musing, not for the user.
        """
        if emotion.openness_to_engage > 0.7:
            # It's engaged with the world — contemplation is lower priority
            return []

        questions = self.generate_questions(emotion, max_questions=5)
        for q in questions:
            q.internal = True
            q.should_ask = False

        return questions

    @property
    def questions_asked(self) -> int:
        """Total questions generated."""
        return len(self._asked_questions)
