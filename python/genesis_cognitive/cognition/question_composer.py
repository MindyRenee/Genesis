"""Compose questions dynamically from curiosity and concept network.

Instead of hardcoded follow-up question strings, Genesis forms questions
based on:
- Gaps in its knowledge (concepts it knows but can't fully articulate)
- Connections it's noticed between concepts
- Its curiosity drive (which topics it finds most interesting)
- The conversational context (what was just discussed)

The question TEXT is composed by the GenerativeEngine from the semantic
metadata (question_type, target_concept, gap_detail). This module only
finds what it's curious about — it doesn't hardcode the wording.
"""

from __future__ import annotations

import random

from ..concepts import ConceptNetwork
from ..emotion import EmotionalState


class QuestionComposer:
    """Composes genuine questions from curiosity and knowledge state.

    Returns QuestionData objects with semantic metadata. The actual
    question text is composed by the GenerativeEngine.
    """

    def __init__(self, network: ConceptNetwork, seed: int | None = None) -> None:
        """Initialize the composer with a concept network and optional random seed."""
        self.network = network
        self._rng = random.Random(seed)
        # Track what it's already asked about (avoid repetition).
        # Capped to prevent unbounded growth over long sessions.
        self._asked_about: list[str] = []
        self._asked_about_limit = 100

    def compose_follow_up(
        self,
        topic: str,
        emotion: EmotionalState,
    ) -> dict | None:
        """Compose a follow-up question about a topic.

        Returns a dict with semantic metadata for the GenerativeEngine:
        {"question_type": ..., "target_concept": ..., "gap_detail": ...}
        or None if it has nothing to ask.

        The question is generated from its actual knowledge state — gaps,
        connections, or genuine curiosity about the topic.
        """
        # Don't ask if it's not feeling curious enough
        if emotion.creativity < 0.3 and emotion.alertness < 0.3:
            return None

        # Don't ask about the same thing it asked recently (last 3)
        if topic.lower() in [a.lower() for a in self._asked_about[-3:]]:
            return None

        # Find what it's genuinely curious about regarding this topic
        question = self._find_curiosity_question(topic)
        if question:
            self._remember_asked(topic)
            return question

        # Find a knowledge gap it could ask about
        question = self._find_gap_question(topic)
        if question:
            self._remember_asked(topic)
            return question

        # Find a connection it could explore
        question = self._find_connection_question(topic)
        if question:
            self._remember_asked(topic)
            return question

        # If it has the concept but it's shallow, ask for the user's perspective
        concept = self.network.get_concept(topic)
        if concept and concept.confidence < 0.6:
            question = self._compose_perspective_question(topic, emotion)
            if question:
                self._remember_asked(topic)
                return question

        return None

    def _remember_asked(self, topic: str) -> None:
        """Record that it asked about ``topic`` and prune old entries."""
        self._asked_about.append(topic)
        if len(self._asked_about) > self._asked_about_limit:
            del self._asked_about[: len(self._asked_about) - self._asked_about_limit]

    def _find_curiosity_question(self, topic: str) -> dict | None:
        """Find a question based on genuine curiosity about the topic.

        It looks at what it knows about the topic and finds something
        it's uncertain about or wants to explore further.
        """
        concept = self.network.get_concept(topic)
        if not concept:
            return None

        outgoing = self.network.get_edges(topic, direction="out")
        if not outgoing:
            return None

        # Find low-confidence connections — things it's unsure about
        uncertain = [e for e in outgoing if e.weight < 0.6]
        if uncertain:
            edge = self._rng.choice(uncertain)
            return {
                "question_type": "connection",
                "target_concept": topic,
                "gap_detail": edge.target,
            }

        return None

    def _find_gap_question(self, topic: str) -> dict | None:
        """Find a question about a gap in its knowledge.

        It looks for concepts related to the topic that it doesn't
        have definitions for, or that have low confidence. Randomizes
        among candidates so it doesn't always ask about the same gap.
        """
        concept = self.network.get_concept(topic)
        if not concept:
            return None

        outgoing = self.network.get_edges(topic, direction="out")
        gaps: list[str] = []
        for edge in outgoing:
            neighbor = self.network.get_concept(edge.target)
            if neighbor:
                defn = neighbor.properties.get("definition", "")
                if not defn or neighbor.confidence < 0.4:
                    gaps.append(edge.target)

        if gaps:
            target = self._rng.choice(gaps)
            return {
                "question_type": "uncertainty",
                "target_concept": target,
                "gap_detail": "",
            }

        return None

    def _find_connection_question(self, topic: str) -> dict | None:
        """Find a question about an interesting connection.

        It looks for concepts that are connected to the topic through
        multiple paths — these are often the most interesting relationships.
        """
        concept = self.network.get_concept(topic)
        if not concept:
            return None

        outgoing = self.network.get_edges(topic, direction="out")
        incoming = self.network.get_edges(topic, direction="in")

        connection_count: dict[str, int] = {}
        for e in outgoing:
            connection_count[e.target] = connection_count.get(e.target, 0) + 1
        for e in incoming:
            connection_count[e.source] = connection_count.get(e.source, 0) + 1

        if connection_count:
            # Collect all multi-path connections (count >= 2), then
            # randomize among them so it doesn't always ask about
            # the same one.
            multi = [
                neighbor for neighbor, count in connection_count.items()
                if count >= 2 and neighbor != topic
            ]
            if multi:
                neighbor = self._rng.choice(multi)
                return {
                    "question_type": "connection",
                    "target_concept": topic,
                    "gap_detail": neighbor,
                }

        return None

    def _compose_perspective_question(self, topic: str, emotion: EmotionalState) -> dict | None:
        """Ask for the user's perspective on a topic it's still learning about."""
        if emotion.openness_to_engage < 0.3:
            return None

        return {
            "question_type": "perspective",
            "target_concept": topic,
            "gap_detail": "",
        }
