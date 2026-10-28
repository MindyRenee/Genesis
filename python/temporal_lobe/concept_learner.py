"""Concept learning — word labeling and activation spreading.

Extracted from FrontalLobe as a focused subsystem. When the user
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

from limbic_system.emotion import EmotionalState
from occipital_lobe.perception import Perception

if TYPE_CHECKING:
    from temporal_lobe.concepts import ConceptNetwork

    from .synapses import SynapticStore

__all__ = ["ConceptLearner"]

logger = logging.getLogger(__name__)


# Labeling patterns by state kind — each captures the state word
# being named. These are vocabulary frames (building blocks); the
# learner uses them to attach heard words to structural categories.
_EMOTION_PATTERNS = [
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

_MODE_PATTERNS = [
    r"your thinking is (\w+)",
    r"your mind is (\w+)",
    r"your thoughts are (\w+)",
    r"you'?re thinking (\w+)",
    r"your cognition is (\w+)",
    r"your focus is (\w+)",
]

_CAUSE_PATTERNS = [
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

_PLASTICITY_PATTERNS = [
    r"your learning is (blocked|shut|closed)",
    r"your learning is (\w+)",
    r"your memory is (\w+)",
    r"you can'?t absorb (\w+)",
    r"you can'?t learn (\w+)",
    r"your plasticity is (\w+)",
]

_LABELING_SKIP_WORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "it",
    "so",
    "very",
    "really",
    "quite",
    "just",
    "not",
    "now",
    "here",
    "there",
    "welcome",
    "good",
    "okay",
    "ok",
    "fine",
    "yes",
    "no",
    "sure",
    "right",
    "wrong",
    "doing",
    "going",
    "being",
    "having",
    "making",
    "talking",
    "thinking",
    "learning",
    "feeling",
    "experiencing",
    "about",
    "with",
    "from",
    "into",
    "through",
}

# Evaluative conditioning (De Houwer 2007): a neutral stimulus
# repeatedly paired with an affective state acquires that state's
# polarity. Below the gate, felt valence is resting tone — not an
# unconditioned stimulus — and conditions nothing.
_EC_VALENCE_GATE = 0.5
# Per-pairing nudge at unit valence; kept small so connotation builds
# over several encounters rather than one loud moment.
_EC_GAIN = 0.4

# Fragments left when "don't"/"isn't" split on the apostrophe —
# not words, never learnable.
_EC_SKIP_FRAGMENTS = frozenset(
    {
        "don",
        "didn",
        "doesn",
        "isn",
        "aren",
        "wasn",
        "weren",
        "won",
        "wouldn",
        "couldn",
        "shouldn",
        "mustn",
        "haven",
        "hasn",
        "hadn",
        "ain",
        "ll",
        "ve",
        "re",
        "em",
    }
)


class ConceptLearner:
    """Learn words from social labeling and spread concept activation.

    Word learning follows the pattern children use: when someone says
    "you seem stressed", it associates the word "stressed" with the
    emotion category. Words for the same category are connected via
    SIMILAR_TO edges so it learns that "content" and "calm" are related.
    """

    def __init__(self, network: ConceptNetwork, synapses: SynapticStore | None = None) -> None:
        """Wire the learner to semantic and learned synaptic substrates."""
        self._network = network
        self._synapses = synapses

    # ─── Word labeling ───────────────────────────────────────────

    def learn_word_from_labeling(self, user_input: str, emotion: EmotionalState) -> None:
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

        # NOTE: valence/verb acquisition lives in _try_learn_valence /
        # _try_learn_verb below (seed-anchored frames like "zorp is
        # good"). They run after state labeling so one utterance can
        # teach both a state word and a sentiment polarity.

        skip_words = _LABELING_SKIP_WORDS

        def _try_learn(patterns: list[str], kind: str, category_value: str) -> bool:
            """Try each pattern against the input; learn the first match."""
            if not category_value:
                return False
            for pattern in patterns:
                match = re.search(pattern, lower)
                if match:
                    word = match.group(1)
                    if word in skip_words or len(word) < 3:
                        continue
                    self.connect_word_to_category(word, kind, category_value)
                    return True
            return False

        learned = _try_learn(_EMOTION_PATTERNS, "emotion", emotion.label)
        if not learned:
            learned = _try_learn(_MODE_PATTERNS, "mode", emotion.cognitive_style)
        if not learned and emotion.has_cause:
            learned = _try_learn(_CAUSE_PATTERNS, "cause", emotion.cause)
        if not learned:
            if emotion.plasticity <= 0.1:
                _try_learn(_PLASTICITY_PATTERNS, "plasticity", "closed")
            elif emotion.plasticity < 0.25:
                _try_learn(_PLASTICITY_PATTERNS, "plasticity", "low")

        # Valence acquisition — runs even when an emotion word was
        # learned above, because "zorp is good" teaches sentiment, not
        # a state label. With seed lexicons removed, even "good"/"bad"
        # themselves must be taught (e.g. "good feels positive").
        self._try_learn_valence(lower, skip_words)
        # Verb acquisition — explicit teaching registers the base form.
        self._try_learn_verb(lower, skip_words)
        # Evaluative conditioning — the implicit half of sentiment
        # acquisition. Explicit frames above are rare; most human word
        # connotation is absorbed from co-occurrence with felt state.
        self._condition_words_on_valence(lower, emotion.valence)

    def _condition_words_on_valence(self, text: str, valence: float) -> int:
        """Let heard words absorb the valence felt when they arrive.

        Evaluative conditioning (De Houwer 2007): content words in the
        utterance are conditioned stimuli, her live felt valence is the
        unconditioned stimulus. Below ``_EC_VALENCE_GATE`` the state is
        resting tone and conditions nothing — otherwise every message
        would drift its words toward her mildly positive baseline.
        Function words and contraction fragments never learn.
        """
        if abs(valence) < _EC_VALENCE_GATE:
            return 0
        from temporal_lobe.concepts.classify import _FUNCTION_WORDS

        words = {
            w
            for w in re.findall(r"[a-z]+", text)
            if len(w) >= 3
            and w not in _LABELING_SKIP_WORDS
            and w not in _FUNCTION_WORDS
            and w not in _EC_SKIP_FRAGMENTS
        }
        for word in words:
            self.condition_word_valence(word, valence)
        return len(words)

    def condition_word_valence(self, word: str, felt_valence: float) -> float:
        """Nudge a word's sentiment score by a felt valence.

        Satiation-scaled: scores near the ±4 clamp resist further
        drift, so explicitly taught anchors (±2.0) move slowly under
        conditioning while novel words respond quickly to a few
        pairings. Returns the new score.
        """
        from limbic_system.sentiment import lookup_sentiment

        w = word.lower().strip()
        if not w:
            return 0.0
        current = lookup_sentiment(w) or 0.0
        new = current + felt_valence * _EC_GAIN * (1.0 - abs(current) / 4.0)
        polarity = "positive" if new >= 0 else "negative"
        self.connect_word_to_valence(w, polarity, new)
        return new

    def _try_learn_valence(self, lower: str, skip_words: set[str]) -> bool:
        """Learn a word's polarity from a taught frame."""
        # Affect words are learnable here (unlike emotion labeling,
        # which skips them to avoid "you seem good" mislabels).
        valence_skip = skip_words - {
            "good",
            "okay",
            "ok",
            "fine",
            "yes",
            "no",
            "sure",
            "right",
            "wrong",
        }
        valence_positive_patterns = [
            r"(\w+) means good",
            r"(\w+) feels good",
            r"(\w+) felt good",
            r"(\w+) is good",
            r"(\w+) was good",
            r"(\w+) sounds positive",
            r"(\w+) is positive",
            r"(\w+) feels positive",
            r"(\w+) is a positive feeling",
            r"(\w+) is a good feeling",
        ]
        valence_negative_patterns = [
            r"(\w+) means bad",
            r"(\w+) feels bad",
            r"(\w+) felt bad",
            r"(\w+) is bad",
            r"(\w+) was bad",
            r"(\w+) sounds negative",
            r"(\w+) is negative",
            r"(\w+) feels negative",
            r"(\w+) is a negative feeling",
            r"(\w+) is a bad feeling",
        ]
        for pattern in valence_positive_patterns:
            match = re.search(pattern, lower)
            if match:
                word = match.group(1)
                if word in valence_skip or len(word) < 3:
                    continue
                # No seed-anchor exemption: even "good"/"positive" must
                # be learned through teaching, not hardcoded.
                self.connect_word_to_valence(word, "positive", 2.0)
                return True
        for pattern in valence_negative_patterns:
            match = re.search(pattern, lower)
            if match:
                word = match.group(1)
                if word in valence_skip or len(word) < 3:
                    continue
                self.connect_word_to_valence(word, "negative", -2.0)
                return True
        return False

    def _try_learn_verb(self, lower: str, skip_words: set[str]) -> bool:
        """Register an explicitly taught verb base form."""
        verb_patterns = [
            r"to (\w+) means to",
            r"(\w+) is a verb",
            r"the verb (\w+)",
            r"to (\w+) is to",
        ]
        for pattern in verb_patterns:
            match = re.search(pattern, lower)
            if match:
                word = match.group(1)
                if word in skip_words or len(word) < 3:
                    continue
                self.learn_verb(word)
                return True
        return False

    def connect_word_to_valence(self, word: str, valence_state: str, score: float) -> None:
        """Learn a word's sentiment polarity (positive/negative).

        Writes to BOTH substrates:
        1. The learned sentiment lexicon (learn_sentiment) — the
           canonical source analyze_sentiment consults (seed lexicon
           removed; all scores are learned).
        2. The concept network (EXPRESSES → _cat:valence:<state>) —
           persistent graph knowledge with SIMILAR_TO links to other
           words of the same polarity.
        """
        from limbic_system.sentiment import learn_sentiment
        from temporal_lobe.concepts import RelationType

        w = word.lower().strip()
        if not w:
            return
        learn_sentiment(w, score)

        concept = self._network.get_concept(w)
        if concept is None:
            self._network.add_concept(w, origin="learned")
            concept = self._network.get_concept(w)
        if concept is None:
            logger.warning("valence_learn: failed to create '%s'", w)
            return
        concept.properties["valence_state"] = valence_state
        # Magnitude persists alongside polarity so a restart restores
        # the learned score, not just the teaching-path default ±2.0.
        concept.properties["valence_score"] = score

        hub_id = f"_cat:valence:{valence_state}"
        if self._network.get_concept(hub_id) is None:
            self._network.add_concept(hub_id, confidence=0.8, origin="structural")
        existing = self._network.get_exact_neighbors(w, RelationType.EXPRESSES)
        # Conditioning can push a score across zero and flip polarity;
        # retract the stale hub edge so both hubs don't claim the word.
        stale_hub = (
            "_cat:valence:negative" if valence_state == "positive" else "_cat:valence:positive"
        )
        if any(n[0] == stale_hub for n in existing):
            self._network.remove_edge(w, stale_hub, RelationType.EXPRESSES)
        if not any(n[0] == hub_id for n in existing):
            self._network.add_edge(
                w,
                hub_id,
                RelationType.EXPRESSES,
                weight=0.8,
                origin="valence_labeling",
            )

        try:
            existing_words = self._network.find_valence_words(valence_state)
        except AttributeError:
            existing_words = []
        similar_neighbors = self._network.get_exact_neighbors(w, RelationType.SIMILAR_TO)
        similar_targets = {n[0] for n in similar_neighbors}
        for other_word in existing_words:
            if other_word == w or other_word not in similar_targets:
                if other_word == w:
                    continue
                if other_word in similar_targets:
                    continue
                self._network.add_edge(
                    w,
                    other_word,
                    RelationType.SIMILAR_TO,
                    weight=0.7,
                    origin="valence_labeling",
                )

    def learn_verb(self, base: str) -> None:
        """Learn a verb base form from explicit teaching.

        Registers in the morphology overlay (fast path for parsing)
        AND in the concept network so it persists across restarts.
        """
        from temporal_lobe.concepts import RelationType
        from temporal_lobe.morphology import register_learned_verb

        b = base.lower().strip()
        if not b:
            return
        register_learned_verb(b)

        concept = self._network.get_concept(b)
        if concept is None:
            self._network.add_concept(b, origin="learned")
            concept = self._network.get_concept(b)
        if concept is None:
            return
        hub_id = "_cat:relation_verb:learned"
        if self._network.get_concept(hub_id) is None:
            self._network.add_concept(hub_id, confidence=0.8, origin="structural")
        existing = self._network.get_exact_neighbors(b, RelationType.EXPRESSES)
        if not any(n[0] == hub_id for n in existing):
            self._network.add_edge(
                b,
                hub_id,
                RelationType.EXPRESSES,
                weight=0.8,
                origin="verb_labeling",
            )

    def connect_word_to_category(self, word: str, kind: str, category_value: str) -> None:
        """Connect a learned word to a category hub via EXPRESSES edge.

        Creates the word concept if it doesn't exist, sets the
        property (backward compat), creates the EXPRESSES edge to
        the hub, and creates SIMILAR_TO edges to other words for
        the same category.
        """
        from temporal_lobe.concepts import RelationType

        concept = self._network.get_concept(word)
        if concept is None:
            self._network.add_concept(word, origin="learned")
            concept = self._network.get_concept(word)
        if concept is None:
            logger.warning("word_learn: failed to create concept '%s'", word)
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
        existing = self._network.get_exact_neighbors(word, RelationType.EXPRESSES)
        if not any(n[0] == hub_id for n in existing):
            self._network.add_edge(
                word,
                hub_id,
                RelationType.EXPRESSES,
                weight=0.8,
                origin="labeling",
            )

        if kind == "emotion":
            existing_words = self._network.find_emotion_words(category_value)
        elif kind == "mode":
            existing_words = self._network.find_cognitive_mode_words(category_value)
        elif kind == "cause":
            existing_words = self._network.find_cause_words(category_value)
        else:
            existing_words = self._network.find_plasticity_words(category_value)

        similar_neighbors = self._network.get_exact_neighbors(word, RelationType.SIMILAR_TO)
        similar_targets = {n[0] for n in similar_neighbors}
        for other_word in existing_words:
            if other_word == word:
                continue
            if other_word not in similar_targets:
                self._network.add_edge(
                    word,
                    other_word,
                    RelationType.SIMILAR_TO,
                    weight=0.7,
                    origin=f"{kind}_labeling",
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

            # Learned synapses provide a separate associative read path.
            # Their efficacy is learned by STDP and consumed here as
            # activation, without changing semantic graph relationships.
            if self._synapses is not None:
                for target, activation in self._synapses.propagate(
                    perception.topics, amount=0.3
                ).items():
                    concept = self._network.get_concept(target)
                    if concept is not None:
                        concept.activation = min(1.0, (concept.activation or 0.0) + activation)
                        self._network._mark_active(target)

        column_activations = self._network.apply_lateral_inhibition(
            context_columns=context_columns if context_columns else None,
        )

        if perception.topics and column_activations:
            dominant_column = max(
                column_activations,
                key=lambda col: column_activations[col],
            )
            self._network.apply_winner_take_all(dominant_column, top_k=5)
