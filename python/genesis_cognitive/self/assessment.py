"""Self-assessment engine — Genesis evaluating her own knowledge and answers.

This module gives Genesis metacognitive awareness: the ability to
evaluate whether she actually knows something, how confident she is,
and what she doesn't know. It's the foundation for self-improvement —
you can't learn what you're missing if you don't know you're missing it.

## What this provides

1. **Knowledge verification** — given a topic, check whether she
   actually has learned knowledge about it in the concept network.
   Not just "does the concept exist" but "does it have meaningful
   content" (definition, relationships, properties).

2. **Confidence calibration** — given a planned answer, estimate
   how confident she should be. This is based on:
   - How well-connected the relevant concepts are in the network
   - Whether the answer relies on concepts she actually knows about
   - Whether she's answering from learned knowledge or making things up

3. **Knowledge gap detection** — systematically identify what she
   doesn't know. A concept exists but has no definition. A concept
   has no relationships. A topic was asked about but she has no
   concept for it at all.

4. **Answer quality scoring** — after producing an answer, evaluate
   whether it was relevant, complete, and grounded in actual knowledge
   vs. being vague or fabricated.

5. **Capability awareness** — track what kinds of questions she can
   answer well and which ones she struggles with, so she can be
   honest about her limitations.

## How it connects to the rest of the system

- **SelfMonitor** (language/self_monitor.py) checks output for
  grammar and coherence. This module checks for *epistemic* quality
  — is the content actually grounded in knowledge?

- **ErrorMonitor** (reflection.py) detects prediction-outcome
  mismatches. This module provides the ground truth for those
  comparisons — was the answer actually correct?

- **CuriosityEngine** (curiosity.py) generates questions from gaps.
  This module identifies those gaps systematically.

- **ReflectionEngine** (reflection.py) produces insights. This
  module feeds it actionable insights about knowledge quality.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from ..concepts import ConceptNetwork

__all__ = [
    "AnswerAssessment",
    "CapabilityProfile",
    "KnowledgeAssessment",
    "SelfAssessmentEngine",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KnowledgeAssessment:
    """Assessment of Genesis's knowledge about a topic."""

    topic: str
    concept_exists: bool = False
    has_definition: bool = False
    has_relationships: bool = False
    has_properties: bool = False
    connection_count: int = 0
    confidence: float = 0.0  # 0 = knows nothing, 1 = knows well
    gaps: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerAssessment:
    """Assessment of a generated answer."""

    answer: str
    topics: list[str]
    grounded: bool = False  # is the answer based on real knowledge?
    confidence: float = 0.0  # how confident should she be?
    issues: list[str] = field(default_factory=list)
    knows_all_topics: bool = True
    missing_topics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CapabilityProfile:
    """Profile of what Genesis can and can't do well."""

    # Track success/failure rates by question type
    question_type_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    # Concepts she's been asked about but doesn't know
    known_gaps: set[str] = field(default_factory=set)
    # Concepts she's recently learned (gaps that were filled)
    recently_learned: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    # Topics she's confident about
    confident_topics: set[str] = field(default_factory=set)
    # Total assessments made
    total_assessments: int = 0


class SelfAssessmentEngine:
    """Metacognitive self-assessment — knowing what she knows and doesn't.

    This engine runs alongside the main cognition loop. Before answering,
    it assesses whether she actually has the knowledge to answer well.
    After answering, it scores the answer quality. Over time, it builds
    a profile of her capabilities.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize the assessment engine with a concept network and empty capability profile."""
        self.network = network
        self.profile = CapabilityProfile()
        self._answer_history: deque[AnswerAssessment] = deque(maxlen=200)
        self._topic_confidence: dict[str, float] = {}
        # Caps for unbounded collections. Topics come from arbitrary
        # user input, so known_gaps and _topic_confidence grow forever
        # in a long-running daemon without a bound.
        self._max_known_gaps = 1000
        self._max_topic_confidence = 2000

    # ═══════════════════════════════════════════════════════════════
    # 1. Knowledge verification
    # ═══════════════════════════════════════════════════════════════

    def _compute_knowledge_confidence(
        self,
        assessment: KnowledgeAssessment,
        definition: str,
        total_edges: int,
        topic: str,
    ) -> None:
        """Compute confidence from knowledge richness and track it."""
        confidence = 0.0
        if assessment.has_definition:
            confidence += 0.3
        if assessment.has_relationships:
            # More connections = more confidence, up to a cap
            confidence += min(0.4, total_edges * 0.05)
        if assessment.has_properties:
            confidence += 0.1
        # Definition quality — longer definitions suggest deeper knowledge
        if definition and len(definition) > 50:
            confidence += 0.1
        if definition and len(definition) > 100:
            confidence += 0.1

        assessment.confidence = min(1.0, confidence)

        # Track confidence over time
        self._topic_confidence[topic] = assessment.confidence
        # Bound the cache — evict the oldest entry (dicts are
        # insertion-ordered) when full. It's only a cache; eviction
        # just costs a re-assessment.
        if len(self._topic_confidence) > self._max_topic_confidence:
            self._topic_confidence.pop(next(iter(self._topic_confidence)))

        # If she knows it well, mark as confident topic
        if assessment.confidence > 0.6:
            self.profile.confident_topics.add(topic)
            # If it was a known gap, it's now filled
            if topic in self.profile.known_gaps:
                self.profile.known_gaps.discard(topic)
                self.profile.recently_learned.append(topic)
        elif assessment.confidence > 0.0 and topic in self.profile.known_gaps:
            # Concept now exists with some knowledge — it's no longer
            # a complete gap, even if confidence isn't high yet. This
            # prevents the metacognitive tracker from lagging behind
            # actual learning (e.g., listing "tardigrade" as unknown
            # right after learning about it).
            self.profile.known_gaps.discard(topic)
            self.profile.recently_learned.append(topic)

    def assess_knowledge(self, topic: str) -> KnowledgeAssessment:
        """Assess how well Genesis knows a topic.

        This is the core metacognitive function: given a topic, how
        much does she actually know about it? The assessment is based
        entirely on the concept network — no hardcoded knowledge.

        A concept is "well known" if it has:
        - A definition (learned content)
        - Multiple relationships (connected to other concepts)
        - Properties beyond defaults

        Returns a KnowledgeAssessment with confidence and identified gaps.
        """
        topic = topic.lower().strip()
        assessment = KnowledgeAssessment(topic=topic)

        concept = self.network.get_concept(topic)
        if concept is None:
            assessment.gaps.append(f"No concept for '{topic}' in network")
            assessment.confidence = 0.0
            self.remember_gap(topic)
            return assessment

        assessment.concept_exists = True

        # Check for definition
        definition = concept.properties.get("definition", "")
        if definition and len(definition) > 10:
            assessment.has_definition = True
        else:
            assessment.gaps.append(f"No definition for '{topic}'")

        # Check for relationships
        outgoing = self.network.get_edges(concept.id, "out")
        incoming = self.network.get_edges(concept.id, "in")
        total_edges = len(outgoing) + len(incoming)
        assessment.connection_count = total_edges

        if total_edges >= 2:
            assessment.has_relationships = True
        elif total_edges == 1:
            assessment.has_relationships = False
            assessment.gaps.append(f"'{topic}' has only one relationship")
        else:
            assessment.has_relationships = False
            assessment.gaps.append(f"No relationships for '{topic}'")

        # Check for properties beyond defaults
        non_default_props = {
            k: v
            for k, v in concept.properties.items()
            if k not in ("definition", "part_of_speech", "example") and v
        }
        if non_default_props:
            assessment.has_properties = True

        # Compute confidence from knowledge richness
        self._compute_knowledge_confidence(assessment, definition, total_edges, topic)

        return assessment

    # ═══════════════════════════════════════════════════════════════
    # 2. Answer confidence calibration
    # ═══════════════════════════════════════════════════════════════

    def _assess_answer_topics(self, assessment: AnswerAssessment) -> None:
        """Check each topic and compute base confidence from topic knowledge."""
        topic_confidences: list[float] = []
        for topic in assessment.topics:
            ka = self.assess_knowledge(topic)
            topic_confidences.append(ka.confidence)
            if ka.confidence < 0.2:
                assessment.missing_topics.append(topic)
                assessment.knows_all_topics = False

        # Base confidence on topic knowledge
        if topic_confidences:
            assessment.confidence = sum(topic_confidences) / len(topic_confidences)
        else:
            # No recognized topics — very low confidence
            assessment.confidence = 0.1
            assessment.issues.append("No recognized topics in the question")

    def _assess_answer_hedging(
        self, assessment: AnswerAssessment, answer_lower: str
    ) -> None:
        """Check for hedging language and adjust confidence."""
        hedge_words = {
            "maybe",
            "perhaps",
            "i think",
            "i believe",
            "possibly",
            "might be",
            "could be",
            "i'm not sure",
            "i guess",
            "probably",
            "likely",
        }
        hedge_count = sum(1 for w in hedge_words if w in answer_lower)
        if hedge_count > 0:
            # Hedging reduces confidence but also signals self-awareness.
            # Cap at 5 instances so confidence can't go below 0.5.
            hedge_penalty = min(hedge_count, 5) * 0.1
            assessment.confidence *= 1.0 - hedge_penalty
            assessment.issues.append(f"Answer contains hedging language ({hedge_count} instances)")

    def _check_ignorance(
        self, assessment: AnswerAssessment, answer_lower: str
    ) -> bool:
        """Check for 'I don't know' language."""
        ignorance_phrases = {
            "i don't know",
            "i don't know much",
            "new territory",
            "i could learn",
            "i'm not sure",
            "i don't have",
            "i haven't learned",
            "i don't understand",
        }
        is_ignorant = any(phrase in answer_lower for phrase in ignorance_phrases)
        if is_ignorant:
            assessment.confidence = min(assessment.confidence, 0.2)
            assessment.issues.append("Answer admits lack of knowledge")
            # This is actually good — she's being honest
            assessment.grounded = True  # honest about not knowing
        return is_ignorant

    def _check_grounding(
        self, assessment: AnswerAssessment, answer_lower: str, is_ignorant: bool
    ) -> None:
        """Check if answer is grounded in concept definitions."""
        if assessment.topics and not is_ignorant:
            grounded_words = 0
            for topic in assessment.topics:
                concept = self.network.get_concept(topic)
                if not concept:
                    continue
                definition = concept.properties.get("definition", "")
                if not definition:
                    continue
                # Check if answer shares meaningful words with definition
                def_words = set(definition.lower().split())
                ans_words = set(answer_lower.split())
                # Remove common words
                common = {
                    "the",
                    "a",
                    "an",
                    "is",
                    "are",
                    "of",
                    "and",
                    "to",
                    "in",
                    "that",
                    "it",
                    "for",
                    "with",
                    "as",
                    "by",
                    "on",
                    "at",
                    "from",
                    "or",
                    "but",
                    "not",
                    "this",
                    "which",
                    "be",
                    "has",
                    "have",
                    "had",
                    "was",
                    "were",
                    "been",
                    "being",
                    "its",
                    "their",
                }
                meaningful_def = def_words - common
                meaningful_ans = ans_words - common
                overlap = meaningful_def & meaningful_ans
                if overlap:
                    grounded_words += len(overlap)
            if grounded_words > 0:
                assessment.grounded = True
            else:
                # Secondary grounding check: does the answer mention
                # any concept that exists in the network? Even if the
                # answer doesn't share words with the *topic's* definition,
                # it may be grounded in other concepts. For example,
                # "Who created you?" has topic "created", but the answer
                # "Alice is my creator" mentions "alice" which exists.
                ans_words = set(answer_lower.split())
                found_concepts = 0
                for word in ans_words:
                    if len(word) < 3:
                        continue
                    concept = self.network.get_concept(word)
                    if concept and concept.properties.get("definition"):
                        found_concepts += 1
                if found_concepts >= 2:
                    assessment.grounded = True
                else:
                    # Answer doesn't share words with any definition — possibly fabricated
                    assessment.grounded = False
                    assessment.issues.append("Answer not grounded in learned definitions")

    def _check_answer_repetition(
        self, assessment: AnswerAssessment, answer: str
    ) -> None:
        """Check for repetition from previous answers."""
        if self._answer_history:
            recent = list(self._answer_history)[-5:]
            for prev in recent:
                if prev.answer == answer:
                    assessment.issues.append("Answer is identical to a recent answer")
                    assessment.confidence *= 0.7
                    break

    def assess_answer(
        self,
        answer: str,
        topics: list[str],
        user_input: str = "",
    ) -> AnswerAssessment:
        """Assess the quality and confidence of a planned answer.

        This runs BEFORE the answer is spoken (or right after). It
        checks whether the answer is grounded in actual knowledge or
        whether she's fabricating.

        Key signals:
        - Do all topics in the answer have concepts in the network?
        - Does the answer contain words from concept definitions?
        - Is the answer vague/hedging ("maybe", "perhaps", "I think")
          which suggests low confidence?
        - Does the answer contain "I don't know" language?
        """
        assessment = AnswerAssessment(
            answer=answer,
            topics=[t.lower().strip() for t in topics],
        )

        # Check each topic
        self._assess_answer_topics(assessment)

        # Check for hedging language (indicates uncertainty)
        answer_lower = answer.lower()
        self._assess_answer_hedging(assessment, answer_lower)

        # Check for "I don't know" language
        is_ignorant = self._check_ignorance(assessment, answer_lower)

        # Check if answer is grounded in concept definitions
        self._check_grounding(assessment, answer_lower, is_ignorant)

        # Check for repetition from previous answers
        self._check_answer_repetition(assessment, answer)

        # Track assessment
        self._answer_history.append(assessment)
        self.profile.total_assessments += 1

        return assessment

    # ═══════════════════════════════════════════════════════════════
    # 3. Knowledge gap detection
    # ═══════════════════════════════════════════════════════════════

    def detect_gaps(self, topics: list[str]) -> list[str]:
        """Detect knowledge gaps for a set of topics.

        Returns a list of gap descriptions — things she doesn't know
        that are relevant to the topics. These can be fed to the
        curiosity engine to generate learning questions.
        """
        gaps: list[str] = []
        for topic in topics:
            ka = self.assess_knowledge(topic)
            if not ka.concept_exists:
                gaps.append(f"don't know what '{topic}' is")
            elif not ka.has_definition:
                gaps.append(f"'{topic}' exists but no definition")
            elif not ka.has_relationships:
                gaps.append(
                    f"'{topic}' known but no relationships"
                )
            elif ka.confidence < 0.4:
                gaps.append(f"only a little known about '{topic}'")
        return gaps

    def find_weak_concepts(self, limit: int = 20) -> list[tuple[str, float]]:
        """Find concepts in the network that are weakly connected.

        These are concepts that exist but have poor knowledge —
        few relationships, no definition, etc. They're candidates
        for learning more about.

        Returns a list of (concept_name, confidence) sorted by
        lowest confidence first.
        """
        weak: list[tuple[str, float]] = []
        for cid, concept in list(self.network._concepts.items()):
            # Skip very new or system concepts
            if concept.origin in ("system", "conversation") and not concept.properties.get(
                "definition"
            ):
                continue
            ka = self.assess_knowledge(cid)
            if ka.confidence < 0.4:
                weak.append((cid, ka.confidence))
        weak.sort(key=lambda x: x[1])
        return weak[:limit]

    # ═══════════════════════════════════════════════════════════════
    # 4. Answer quality scoring (post-hoc)
    # ═══════════════════════════════════════════════════════════════

    def _score_relevance(self, answer_lower: str, user_lower: str) -> float:
        """Score relevance based on word overlap between answer and question."""
        user_words = set(user_lower.split()) - {
            "the",
            "a",
            "an",
            "is",
            "are",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "of",
            "and",
            "to",
            "in",
            "do",
            "does",
            "did",
            "you",
            "i",
            "me",
            "my",
            "your",
        }
        answer_words = set(answer_lower.split())
        if user_words:
            relevance = len(user_words & answer_words) / len(user_words)
            return relevance * 0.3
        return 0.0

    def score_answer_quality(
        self,
        answer: str,
        user_input: str,
        topics: list[str],
    ) -> float:
        """Score the quality of an answer after it's been given.

        Returns a score from 0 to 1:
        - 1.0: excellent — relevant, grounded, complete
        - 0.5: mediocre — partially relevant or partially grounded
        - 0.0: poor — irrelevant, fabricated, or admits ignorance

        This is used for self-improvement: low scores trigger
        reflection and learning.
        """
        score = 0.0
        answer_lower = answer.lower()
        user_lower = user_input.lower()

        # Relevance: does the answer share words with the question?
        score += self._score_relevance(answer_lower, user_lower)

        # Groundedness: is the answer based on real knowledge?
        assessment = self.assess_answer(answer, topics, user_input)
        if assessment.grounded:
            score += 0.3
        else:
            score += 0.1  # partial credit for trying

        # Completeness: does the answer address all topics?
        if assessment.knows_all_topics:
            score += 0.2
        else:
            # Partial credit for topics she does know
            known = len(assessment.topics) - len(assessment.missing_topics)
            if assessment.topics:
                score += (known / len(assessment.topics)) * 0.2

        # Honesty: admitting ignorance is better than fabricating
        ignorance_phrases = {
            "i don't know",
            "i don't know much",
            "new territory",
            "i could learn",
            "i'm not sure",
        }
        is_ignorant = any(phrase in answer_lower for phrase in ignorance_phrases)
        if is_ignorant and not assessment.grounded:
            # Honest about not knowing — better than making things up
            score += 0.1

        # Length appropriateness — not too short, not too long
        word_count = len(answer.split())
        if 5 <= word_count <= 100:
            score += 0.1
        elif word_count < 5:
            score += 0.03  # very short answers are rarely good
        # Very long answers don't get penalized — some questions need detail

        return min(1.0, score)

    # ═══════════════════════════════════════════════════════════════
    # 5. Capability awareness
    # ═══════════════════════════════════════════════════════════════

    def record_question_outcome(
        self,
        question_type: str,
        success: bool,
        topic: str = "",
    ) -> None:
        """Record the outcome of answering a question.

        This builds a profile of what Genesis can and can't do well.
        Called after each interaction to track performance over time.
        """
        if question_type not in self.profile.question_type_stats:
            self.profile.question_type_stats[question_type] = {"success": 0, "failure": 0}
        if success:
            self.profile.question_type_stats[question_type]["success"] += 1
            if topic:
                self.profile.confident_topics.add(topic.lower())
        else:
            self.profile.question_type_stats[question_type]["failure"] += 1
            if topic:
                self.remember_gap(topic.lower())

    def get_capability_summary(self) -> str:
        """Get a human-readable summary of her capabilities.

        This is what Genesis would say if asked "what do you know
        well?" or "what are you struggling with?"
        """
        parts: list[str] = []

        # Strengths
        if self.profile.confident_topics:
            confident = sorted(self.profile.confident_topics)[:10]
            parts.append(f"confident about: {', '.join(confident)}")

        # Weaknesses — don't list a gap if it's already a strength.
        if self.profile.known_gaps:
            gaps = [
                g
                for g in sorted(self.profile.known_gaps)
                if g not in set(confident)
            ][:10]
            if gaps:
                parts.append(f"still learning about: {', '.join(gaps)}")

        # Question type performance
        for qtype, stats in self.profile.question_type_stats.items():
            total = stats["success"] + stats["failure"]
            if total > 0:
                rate = stats["success"] / total
                if rate > 0.7:
                    parts.append(f"handles {qtype} questions well ({rate:.0%} success)")
                elif rate < 0.4:
                    parts.append(f"struggles with {qtype} questions ({rate:.0%} success)")

        # Recently learned
        if self.profile.recently_learned:
            recent = list(self.profile.recently_learned)[-5:]
            parts.append(f"recently learned about: {', '.join(recent)}")

        if not parts:
            return "still building self-awareness"

        return ". ".join(parts) + "."

    def get_confidence_for_topic(self, topic: str) -> float:
        """Get cached confidence for a topic, or assess it fresh."""
        topic = topic.lower().strip()
        if topic in self._topic_confidence:
            return self._topic_confidence[topic]
        ka = self.assess_knowledge(topic)
        return ka.confidence

    def should_hedge(self, topics: list[str]) -> bool:
        """Should Genesis hedge her answer given the topics?

        Returns True if she should express uncertainty — either
        because she doesn't know the topics well, or because she
        has a history of errors on similar topics.
        """
        for topic in topics:
            confidence = self.get_confidence_for_topic(topic)
            if confidence < 0.3:
                return True
        return False

    def get_learning_priorities(self) -> list[str]:
        """Get a prioritized list of things Genesis should learn about.

        This feeds the curiosity engine — the most important gaps
        are ones that:
        1. Have been asked about but couldn't answer
        2. Are weakly connected but exist in the network
        3. Are related to concepts she already knows well
        """
        priorities: list[str] = []

        # First priority: gaps she's been asked about
        priorities.extend(sorted(self.profile.known_gaps))

        # Second priority: weak concepts
        weak = self.find_weak_concepts(limit=20)
        for name, _ in weak:
            if name not in priorities:
                priorities.append(name)

        return priorities[:30]

    def invalidate_confidence(self, topics: list[str]) -> None:
        """Invalidate cached confidence for topics that were just learned.

        When new knowledge is added to the concept network, any
        previously cached confidence values for those topics are
        stale (they may have been 0.0 when the concept didn't exist).
        This clears the cache so the next assessment reflects the
        current network state.
        """
        for topic in topics:
            t = topic.lower().strip()
            self._topic_confidence.pop(t, None)

    def remember_gap(self, topic: str) -> None:
        """Record a topic she doesn't know, with a bound on the set.

        Topics come from arbitrary user input, so ``known_gaps`` would
        grow without bound in a long-running daemon. When the set is
        full, an arbitrary existing gap is evicted — the gaps that
        matter recur in conversation and are re-added.
        """
        topic = topic.lower().strip()
        if not topic:
            return
        if (
            topic not in self.profile.known_gaps
            and len(self.profile.known_gaps) >= self._max_known_gaps
        ):
            self.profile.known_gaps.pop()
        self.profile.known_gaps.add(topic)

    def reconcile_gaps(self, concepts: list[str]) -> None:
        """Reconcile known_gaps with the current network state.

        After learning new concepts, check whether any previously
        recorded gaps have been filled. This prevents the self-
        assessment from listing concepts as unknown that she has
        since learned about.
        """
        for concept_id in concepts:
            cid = concept_id.lower().strip()
            if cid in self.profile.known_gaps:
                concept = self.network.get_concept(cid)
                if concept is not None:
                    # Concept now exists — re-assess to clear the gap
                    self._topic_confidence.pop(cid, None)
                    self.assess_knowledge(cid)
