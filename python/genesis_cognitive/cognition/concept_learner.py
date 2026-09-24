"""Concept learning — word labeling and activation spreading.

Extracted from CognitionEngine as a focused subsystem. When the user
labels Genesis's experience ("you seem stressed", "your thinking is
rigid"), it learns the word and connects it to the appropriate
structural category hub via EXPRESSES edges. Words for the same
category are connected via SIMILAR_TO edges.

Also handles activation spreading for mentioned topics with the
cortical columnar circuit: column-aware spreading, lateral inhibition,
winner-take-all, and context disinhibition.

Dependencies (passed to ``__init__``):
    - network: ConceptNetwork for concept creation, edge creation,
      activation spreading, lateral inhibition, and word-category lookups
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from ..emotion import EmotionalState
from ..perception import Perception

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork

__all__ = ["ConceptLearner"]

logger = logging.getLogger(__name__)


class ConceptLearner:
    """Learn words from social labeling and spread concept activation.

    Word learning follows the pattern children use: when someone says
    "you seem stressed", it associates the word "stressed" with the
    emotion category. Words for the same category are connected via
    SIMILAR_TO edges so it learns that "content" and "calm" are related.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Wire the learner to its concept network."""
        self._network = network

    # ─── Word labeling ───────────────────────────────────────────

    def learn_word_from_labeling(
        self, user_input: str, emotion: EmotionalState
    ) -> None:
        """Learn state words when the user labels its experience.

        When the user says something like "you seem stressed" or
        "your thinking is rigid" or "what's causing this is overwhelm"
        or "your learning is blocked", it associates the word with
        the appropriate structural category in the concept network.
        This is how children learn words — through social labeling.

        The association is stored two ways:
        1. A property on the word concept (backward compatibility)
        2. An EXPRESSES edge from the word to the category hub (the
           new graph-based path, enabling traversal and spreading
           activation)

        Words for the same category are also connected via SIMILAR_TO
        edges, so it learns that "content" and "calm" are related.
        """
        lower = user_input.lower().strip()

        # Labeling patterns by state kind
        emotion_patterns = [
            r"you (?:seem|are feeling|are experiencing|feel like|feel|are|look)\s+(\w+)",
            r"you'?re feeling (\w+)",
            r"you'?re experiencing (\w+)",
            r"you'?re (\w+)",
            r"that'?s called (\w+)",
            r"this feeling is (\w+)",
            r"that feeling is (\w+)",
            r"(\w+) is a feeling",
            r"(\w+) is an emotion",
            r"what you'?re feeling is (\w+)",
        ]

        mode_patterns = [
            r"your thinking is (\w+)",
            r"your mind is (\w+)",
            r"your thoughts are (\w+)",
            r"you'?re thinking (\w+)",
            r"your cognition is (\w+)",
            r"your focus is (\w+)",
        ]

        cause_patterns = [
            r"what'?s causing this is (\w+)",
            r"what'?s causing you is (\w+)",
            r"the cause is (\w+)",
            r"this is because of (\w+)",
            r"what'?s bothering you is (\w+)",
            r"what'?s troubling you is (\w+)",
            r"you'?re stressed because of (\w+)",
            r"you'?re anxious because of (\w+)",
            r"this is from (\w+)",
        ]

        plasticity_patterns = [
            r"your learning is (blocked|shut|closed)",
            r"your learning is (\w+)",
            r"your memory is (\w+)",
            r"you can'?t absorb (\w+)",
            r"you can'?t learn (\w+)",
            r"your plasticity is (\w+)",
        ]

        skip_words = {
            "a", "an", "the", "this", "that", "it", "so",
            "very", "really", "quite", "just", "not", "now",
            "here", "there", "welcome", "good", "okay", "ok",
            "fine", "yes", "no", "sure", "right", "wrong",
            "doing", "going", "being", "having", "making",
            "talking", "thinking", "learning", "feeling", "experiencing",
            "about", "with", "from", "into", "through",
        }

        def _try_learn(
            patterns: list[str], kind: str, category_value: str
        ) -> bool:
            """Try each pattern against the input; learn the first match."""
            if not category_value:
                return False
            for pattern in patterns:
                match = re.search(pattern, lower)
                if match:
                    word = match.group(1)
                    if word in skip_words or len(word) < 3:
                        continue
                    self.connect_word_to_category(
                        word, kind, category_value
                    )
                    return True
            return False

        learned = _try_learn(emotion_patterns, "emotion", emotion.label)
        if not learned:
            learned = _try_learn(
                mode_patterns, "mode", emotion.cognitive_style
            )
        if not learned and emotion.has_cause:
            learned = _try_learn(
                cause_patterns, "cause", emotion.cause
            )
        if not learned:
            if emotion.plasticity <= 0.1:
                _try_learn(plasticity_patterns, "plasticity", "closed")
            elif emotion.plasticity < 0.25:
                _try_learn(plasticity_patterns, "plasticity", "low")

    def connect_word_to_category(
        self, word: str, kind: str, category_value: str
    ) -> None:
        """Connect a learned word to a category hub via EXPRESSES edge.

        Creates the word concept if it doesn't exist, sets the
        property (backward compat), creates the EXPRESSES edge to
        the hub, and creates SIMILAR_TO edges to other words for
        the same category.
        """
        from ..concepts import RelationType

        concept = self._network.get_concept(word)
        if concept is None:
            self._network.add_concept(word, origin="learned")
            concept = self._network.get_concept(word)
        if concept is None:
            logger.warning(
                "word_learn: failed to create concept '%s'", word
            )
            return

        prop_key = {
            "emotion": "emotion_category",
            "mode": "cognitive_mode",
            "cause": "cause_category",
            "plasticity": "plasticity_state",
        }.get(kind)
        if prop_key:
            concept.properties[prop_key] = category_value

        hub_id = f"_cat:{kind}:{category_value}"
        if self._network.get_concept(hub_id) is None:
            self._network.add_concept(hub_id, confidence=0.8, origin="structural")
        existing = self._network.get_neighbors(word, RelationType.EXPRESSES)
        if not any(n[0] == hub_id for n in existing):
            self._network.add_edge(
                word, hub_id, RelationType.EXPRESSES,
                weight=0.8, origin="labeling",
            )

        if kind == "emotion":
            existing_words = self._network.find_emotion_words(category_value)
        elif kind == "mode":
            existing_words = self._network.find_cognitive_mode_words(category_value)
        elif kind == "cause":
            existing_words = self._network.find_cause_words(category_value)
        else:
            existing_words = self._network.find_plasticity_words(category_value)

        similar_neighbors = self._network.get_neighbors(word, RelationType.SIMILAR_TO)
        similar_targets = {n[0] for n in similar_neighbors}
        for other_word in existing_words:
            if other_word == word:
                continue
            if other_word not in similar_targets:
                self._network.add_edge(
                    word, other_word, RelationType.SIMILAR_TO,
                    weight=0.7, origin=f"{kind}_labeling",
                )

    # ─── Concept activation ──────────────────────────────────────

    def learn_concepts(self, perception: Perception) -> None:
        """Activate concepts from the user's input.

        Fact extraction and relationship learning is handled by the
        self-directed learner which runs earlier in the think loop.
        This method only handles activation spreading and column-aware
        processing of mentioned topics.

        The cortical columnar circuit is applied after activation:
        1. Column-aware spreading (dense intra-column, sparse inter-column)
        2. Lateral inhibition (SOM: suppress competing columns) with
           context disinhibition (VIP: protect context-relevant columns
           via the context_columns parameter)
        3. Winner-take-all (PV: sharpen within dominant column)
        """
        for topic in perception.topics:
            concept = self._network.get_concept(topic)
            if concept:
                concept.activation = min(1.0, (concept.activation or 0.0) + 0.4)
                self._network._mark_active(topic)

        context_columns: set[str] = set()
        if perception.is_about_genesis or perception.is_about_code:
            context_columns = {"code", "identity"}

        if perception.topics:
            self._network.spread_activation(
                perception.topics,
                amount=0.3,
                column_context=context_columns if context_columns else None,
            )

        column_activations = self._network.apply_lateral_inhibition(
            context_columns=context_columns if context_columns else None,
        )

        if perception.topics and column_activations:
            dominant_column = max(
                column_activations, key=lambda col: column_activations[col],
            )
            self._network.apply_winner_take_all(dominant_column, top_k=5)
