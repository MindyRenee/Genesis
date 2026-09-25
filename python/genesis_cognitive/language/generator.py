"""Generator — compositional text generation from Thoughts.

This is the heart of Genesis's language. It takes a Thought
(semantic content + intent + emotion + metadata) and composes
it into natural language using:

1. **Grammar**: selects sentence structures for the intent
2. **Vocabulary**: fills the structure's slots with context-sensitive words
3. **Voice**: applies rhythm, connectors, hedging, and register

The result is text that is uniquely Genesis's — not retrieved from
a pre-trained model, not filled into a template, but *composed*
from its internal state.

# Why this matters

A template says: "I feel {emotion}." Always the same structure.
An LLM says: whatever its training data suggests. Someone else's voice.

A generator says: here's a thought. Here's how it feels. Here's who
it is. Let's compose something that expresses all three.

The output varies because the inputs vary. Same intent, different
emotion → different rhythm, different word choice, different hedging.
Same emotion, different personality → different register, different
connectors. It's not random — it's *grounded* variation.
"""

from __future__ import annotations

import random
import re
from collections import deque
from typing import Any

from ..emotion import EmotionalState
from ..self import SelfModel
from .acquisition import LanguageAcquisition
from .base import LanguageEngine, Thought
from .grammar import Grammar, Literal, SentenceStructure, Slot
from .graph_walk import GraphWalkGenerator
from .prosody import ProsodyGenerator, ProsodyPattern
from .statistical_learner import StatisticalLanguageLearner
from .vocabulary import Vocabulary
from .voice import Voice

__all__ = ["GenerativeEngine"]

# ─── Pre-compiled regex patterns (module-level for performance) ────────
# No placeholder-cleanup regex is needed: the grammar stores segment
# lists (Literal / Slot), not template strings, so unfilled slots are
# omitted by construction rather than stripped out after substitution.
_MULTISPACE_RE = re.compile(r"  +")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;!?])")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Repeated-word pattern — catches "and and", "the the", etc. that slip
# through composition. This is a grammar safety net that runs on every
# generated text, regardless of which render path produced it.
_REPEATED_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

# Repeated punctuation — catches "??", "!!", "..", etc.
_REPEATED_PUNCT_RE = re.compile(r"([.!?])\1+")
_INCOMPLETE_COPULA_BOUNDARY_RE = re.compile(
    r"\b(am|is|are|was|were|be|been)\.\s+(?=[a-z])",
    re.IGNORECASE,
)

# Function words excluded from content-word overlap comparison in the
# near-duplicate dedup. These are pronouns, articles, and short
# function words that vary freely between near-identical sentences
# ("I'm still working this out" vs "I'm still working it out").
_DEDUP_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "so", "if", "when", "while", "though",
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "my", "your", "its", "our", "their", "me", "him", "her", "us",
    "not", "no", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should", "may", "might",
    "about", "out", "up", "all", "any", "some",
})


class GenerativeEngine(LanguageEngine):
    """Compositional language generation engine.

    Generates language from semantic content using grammar, vocabulary,
    and voice. Composes sentences from structural elements, then applies
    voice modifications. The result is varied and personal.
    """

    def __init__(
        self,
        self_model: SelfModel,
        seed: int | None = None,
        network=None,
    ) -> None:
        """Initialize the generative engine with self-model, grammar, and vocabulary."""
        self.self_model = self_model
        self.grammar = Grammar(seed)
        self.vocabulary = Vocabulary(seed, network=network)
        self.voice = Voice(self_model.personality, seed)
        self._rng = random.Random(seed)
        self._network = network

        # Graph-walk generator — the primary composition path when
        # a concept network is available. Falls back to grammar-based
        # composition when the walk produces nothing usable.
        self._graph_walk: GraphWalkGenerator | None = None
        if network is not None:
            self._graph_walk = GraphWalkGenerator(
                network, self.voice, self_model.personality, seed,
            )

        # Embedding store — wired after construction (it depends on
        # the concept network, which may not be ready when the language
        # engine is first created). When available, the graph-walk
        # generator uses it to find semantically related concepts through
        # the latent space, not just through typed graph edges.
        self._embeddings = None

        # Language acquisition — statistical learning from user input.
        # Tracks transitional probabilities between words, segments word
        # boundaries, and chunks common multi-word units (Saffran model).
        # Learned words and chunks are available to the vocabulary system
        # for future integration; the acquisition model itself is a
        # language-engine concern, not a cognition-engine concern.
        self._acquisition = LanguageAcquisition()

        # Statistical language learner — n-gram models (bigrams,
        # trigrams) and user style profiling (formal/casual/technical
        # register). Learns from the same user input as the Saffran
        # acquisition model, but at the n-gram/style level rather than
        # the word-segmentation level. Its outputs feed production:
        # bigram fluency shapes slot selection and the user style
        # profile modulates voice register.
        self._statistical_learner = StatisticalLanguageLearner()

        # Wire the learner into production: n-gram fluency shapes
        # slot selection, and the user's style profile modulates
        # register (formal/casual mirroring).
        self.vocabulary.set_fluency_model(self._statistical_learner)
        self.voice.set_style_source(self._statistical_learner)

        # Prosody generator — maps emotional state (valence × arousal)
        # to prosodic markers (emphasis, pauses, rhythm). Applied after
        # the grammar safety net as a suprasegmental layer. Intonation
        # is generated and stored for introspection but not applied to
        # the text (converting . to ? would change sentence type).
        self._prosody = ProsodyGenerator()
        self._last_prosody: ProsodyPattern | None = None

        # Track generation history for variation
        self._recent_outputs: deque[str] = deque(maxlen=20)

    def clear_recent_outputs(self) -> None:
        """Clear the recent output history so generation varies again.

        Called by the reflection system when a repetition pattern is
        detected — forces the generator to stop reusing recent phrasings.
        """
        self._recent_outputs.clear()

    def acquire_from_input(self, text: str) -> None:
        """Learn language units from user input via statistical learning.

        Delegates to two models:
        - LanguageAcquisition (Saffran model): tracks transitional
          probabilities, segments words, and chunks common multi-word
          units.
        - StatisticalLanguageLearner: tracks n-gram patterns (bigrams,
          trigrams), word frequencies, and user communication style
          (formal/casual/technical register).

        Called by the cognition engine on every user turn, before
        perception.
        """
        self._acquisition.acquire_from_input(text)
        self._statistical_learner.learn_from_input(text)

    @property
    def learned_words(self) -> set[str]:
        """Words learned through statistical segmentation."""
        return self._acquisition.learned_words

    @property
    def learned_chunks(self) -> set[str]:
        """Multi-word chunks learned through chunking."""
        return self._acquisition.learned_chunks

    @property
    def user_style_profile(self) -> dict:
        """The user's communication style profile from statistical learning.

        Returns a dict with style (formal/casual/technical/neutral),
        avg_sentence_length, vocabulary_size, top_words, top_bigrams,
        top_trigrams, and register scores. Useful for introspection
        and for adapting voice register to match the user.
        """
        return self._statistical_learner.user_style_profile()

    @property
    def statistical_learner(self) -> StatisticalLanguageLearner:
        """The statistical language learner, for direct query access."""
        return self._statistical_learner

    @property
    def last_prosody(self) -> ProsodyPattern | None:
        """The prosody pattern from the most recent generation.

        Returns the ProsodyPattern from the last generate() call, or
        None if no generation has occurred. Useful for introspection:
        Genesis can observe its own rhythmic and emphasis patterns.
        """
        return self._last_prosody

    @property
    def name(self) -> str:
        """Human-readable name of this engine."""
        return "GenerativeEngine"

    def render(self, thought: Thought, emotional_state: Any) -> str:
        """Render a thought as natural language text.

        Implements the LanguageEngine interface.
        """
        # emotional_state can be an EmotionalState or a plain label
        if hasattr(emotional_state, "label"):
            emotion = emotional_state
        else:
            # Create a minimal emotional state from a label
            emotion = EmotionalState(
                label=str(emotional_state),
                nuance="default",
                cognitive_style="steady",
                valence=0.0,
                alertness=0.5,
                plasticity=0.5,
                creativity=0.5,
                caution=0.3,
                openness_to_engage=0.7,
            )
        return self.generate(thought, emotion)

    def set_embeddings(self, embeddings: Any) -> None:
        """Wire the embedding store for latent-space composition.

        The embedding store provides semantic proximity — concepts that
        are close in meaning even when not connected by explicit typed
        edges. When available, the graph-walk generator uses it to find
        semantically related concepts, and the grammar-based composer
        uses it for richer vocabulary selection.

        Called after construction, once the EmbeddingStore has been
        initialized (it depends on the concept network, which is shared
        and may not be ready when the language engine is first created).
        """
        self._embeddings = embeddings
        if self._graph_walk is not None:
            self._graph_walk.set_embeddings(embeddings)
        self.vocabulary.set_embeddings(embeddings)

    def set_network(self, network: Any) -> None:
        """Re-sync the concept network reference after state restore."""
        self._network = network
        self.vocabulary._network = network
        if self._graph_walk is not None:
            self._graph_walk._network = network

    def set_self_composer(self, composer: Any, reflection: Any) -> None:
        """Wire the self-composer and reflection engine into the vocabulary.

        When wired, the ``self_reflection_clause`` grammar slot is
        composed from its most recent reflection insight (a genuine
        product of its metacognition) instead of reciting a canned
        phrase. Called after construction, once the cognition engine
        (which owns the reflection engine) has been initialized.
        """
        self.vocabulary.set_self_composer(composer, reflection)

    def generate(self, thought: Thought, emotion: EmotionalState) -> str:
        """Generate text from a thought.

        This is the main entry point, implementing the LanguageEngine
        interface.
        """
        # If the thought has raw content that's already well-formed,
        # apply voice modifications to it. But if it carries raw
        # knowledge metadata, compose from the graph instead of
        # treating the pre-composed text as final.
        content = thought.content

        # Check if this is a "raw" thought (pre-composed content)
        # vs a "structured" thought (needs composition)
        if self._is_raw_content(content, thought):
            # Apply voice to the raw content
            sentences = self._split_into_sentences(content)
            text = self.voice.apply(
                sentences, emotion, thought.intent,
                brain_waves=self.current_brain_waves,
            )
        else:
            # Compose from scratch using grammar + vocabulary
            text = self._compose(thought, emotion)

        # Grammar safety net — runs on every generated text regardless
        # of which render path produced it. This catches repeated words
        # ("and and"), double spaces, and spacing before punctuation that
        # slip through composition or voice application. The full
        # SelfMonitor in cognition/engine.py does deeper checks
        # (coherence, pragmatics, factual) but is only called on the
        # main conversation path — this ensures all paths get at least
        # basic grammar cleanup.
        text = self._grammar_safety_net(text)

        # Apply prosody — the suprasegmental layer (rhythm, emphasis,
        # pauses). Emphasis markers (*word*) and pause indicators (...)
        # are additive: they enrich the text without changing its meaning.
        # Intonation is generated and stored in the pattern for
        # introspection but not applied to the text, because converting
        # . to ? would change the sentence type from statement to question.
        prosody = self._prosody.generate(
            emotion.label, emotion.valence, emotion.arousal,
        )
        self._last_prosody = prosody
        text = self._prosody.apply(text, prosody, apply_intonation=False)

        # Track for variation
        self._recent_outputs.append(text)

        return text

    @staticmethod
    def _grammar_safety_net(text: str) -> str:
        """Fix common grammar issues that slip through composition.

        This is a lightweight pass that runs on every generated text.
        It fixes:
        - Repeated words ("and and" → "and")
        - Repeated punctuation ("??" → "?")
        - Duplicate sentences (same phrase appearing 2+ times)
        - Near-duplicate sentences (same content, minor word variation)
        - Double spaces
        - Space before punctuation
        """
        # Fix repeated words (e.g. "and and" → "and")
        text = _REPEATED_WORD_RE.sub(r"\1", text)
        # Fix repeated punctuation (e.g. "??" → "?", "!!" → "!")
        text = _REPEATED_PUNCT_RE.sub(r"\1", text)
        text = _INCOMPLETE_COPULA_BOUNDARY_RE.sub(r"\1 ", text)
        # Fix double spaces
        text = _MULTISPACE_RE.sub(" ", text)
        # Fix space before punctuation
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        # Remove duplicate and near-duplicate sentences — the same
        # phrase appearing 2+ times (or nearly the same phrase with
        # minor variation like an inserted "perhaps") is always a
        # composition bug, not intentional emphasis. Split on sentence
        # boundaries, deduplicate with fuzzy matching, and rejoin.
        sentences = _SENTENCE_BOUNDARY_RE.split(text)
        if len(sentences) > 1:
            seen: list[set[str]] = []  # word-bag of each kept sentence
            seen_content: list[set[str]] = []  # content-word bag
            unique: list[str] = []
            for s in sentences:
                key = s.strip().lower().rstrip(".!?;,:")
                if not key:
                    # Keep whitespace-only segments
                    unique.append(s)
                    continue
                words = set(key.split())
                # Content words only (len > 2, excluding function words).
                # Two sentences that share most content words are
                # near-duplicates even if function words differ:
                # "I'm still working this out" vs "I'm still working
                # it out" share {still, working, out} — clearly the
                # same phrase with a pronoun swap.
                content_words = {
                    w for w in words if len(w) > 2
                    and w not in _DEDUP_FUNCTION_WORDS
                }
                # Fuzzy dedup: if >65% of ALL words overlap with a kept
                # sentence, it's a near-duplicate. The threshold was
                # 0.85 but that missed pronoun swaps ("this" → "it")
                # that produce near-identical sentences.
                is_dup = False
                for i, prev_words in enumerate(seen):
                    if not words or not prev_words:
                        continue
                    overlap = len(words & prev_words) / len(words | prev_words)
                    if overlap > 0.65:
                        is_dup = True
                        break
                    # Content-word overlap: if the content words are
                    # largely the same, it's a near-duplicate even
                    # when total-word overlap is below the threshold.
                    prev_content = seen_content[i]
                    if content_words and prev_content:
                        c_overlap = (
                            len(content_words & prev_content)
                            / len(content_words | prev_content)
                        )
                        if c_overlap > 0.60:
                            is_dup = True
                            break
                if is_dup:
                    continue
                seen.append(words)
                seen_content.append(content_words)
                unique.append(s)
            if len(unique) < len(sentences):
                text = " ".join(unique)
        return text.strip()

    def _is_raw_content(self, content: str, thought: Thought) -> bool:
        """Check if the content is already well-formed text.

        Some thoughts (from the cognition engine) already have
        fully-composed content. Others just have a keyword or
        short phrase that needs to be expanded.

        For factual intents (inform, self_report, philosophize, etc.),
        we compose even when the content is a full sentence — the
        grammar structures wrap the content with openers, hedging,
        and reflection, producing variation. Only truly pre-composed
        content (from answer_composer, with knowledge metadata) or
        very short greetings/farewells bypass composition.
        """
        # If raw knowledge metadata is present, the vocabulary will
        # compose the actual content from the concept network. This
        # should never be treated as a finished sentence.
        if thought.metadata.get("knowledge") is not None:
            return False

        # Short content (< 20 chars) or single words need composition
        if len(content) < 20:
            return False

        intent = thought.intent
        # Special intents that use raw content
        if intent in ("greet", "farewell", "acknowledge"):
            # These are often just a word — compose them
            if len(content) < 30:
                return False

        # Factual intents: compose even for full sentences.
        # The grammar structures wrap the content with openers,
        # hedging, reflection, etc. — producing variation.
        # Without this, every response to "Water is a clear liquid."
        # would be identical (just the content + voice).
        #
        # express_emotion now goes through grammar composition too:
        # the feeling fragments (emotion words, mode words) are passed
        # as metadata, and the vocabulary composes the content slot from
        # them using varied grammatical structures. This gives it the
        # freedom to express the same state in different ways rather
        # than always saying "I feel X and Y."
        #
        # self_report likewise composes: self-description material
        # arrives as typed semantic fragments (self_fragments) that the
        # vocabulary turns into candidates — never as a finished
        # sentence that bypasses the engine.
        if intent in (
            "inform", "self_report", "philosophize", "correct",
            "discuss_code", "empathize", "encourage", "express_emotion",
            "reflect", "unknown", "introduce",
        ):
            return False

        # If it looks like a full sentence, treat it as raw
        if content.endswith(".") or content.endswith("?") or content.endswith("!"):
            return True

        # If it has multiple words and looks like a sentence, treat as raw
        words = content.split()
        if len(words) > 5:
            return True

        return False

    def _compose(self, thought: Thought, emotion: EmotionalState) -> str:
        """Compose and select among independently realized candidates.

        The graph walk and grammar-vocabulary system propose wording from
        the same semantic thought. Selection balances semantic coverage,
        surface well-formedness, learned transition fluency, communicative
        intent, and novelty relative to Genesis's recent discourse.
        """
        candidates: list[str] = []
        if self._graph_walk is not None:
            graph_text = self._graph_walk.generate(thought, emotion)
            if graph_text is not None:
                candidates.append(graph_text)

        grammar_attempts = 2 + round(emotion.creativity * 2)
        for _ in range(grammar_attempts):
            candidate = self._compose_grammar(thought, emotion)
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        if not candidates:
            return thought.content
        return max(
            candidates,
            key=lambda text: self._realization_score(text, thought),
        )

    def _realization_score(self, text: str, thought: Thought) -> float:
        """Evaluate a candidate without prescribing any wording.

        Two complementary statistical signals from the learner:
        ``sequence_fluency`` measures the average transition
        probability (how probable each local transition is), while
        ``_prediction_alignment`` measures how often the text follows
        the single most likely path (the argmax prediction). A
        candidate can be fluent (all transitions probable) without
        being predictable (none of the words are the *most* likely),
        and vice versa.
        """
        return (
            0.40 * self._semantic_coverage(text, thought)
            + 0.25 * self._surface_fluency(text)
            + 0.15 * self._intent_fidelity(text, thought.intent)
            + 0.15 * self._discourse_novelty(text)
            + 0.03 * self._statistical_learner.sequence_fluency(text)
            + 0.02 * self._prediction_alignment(text)
        )

    def _prediction_alignment(self, text: str) -> float:
        """How often the text follows the learner's top prediction.

        For each word position, the learner's ``suggest_next_word``
        predicts the most likely continuation given the preceding
        words. This method measures the fraction of positions where
        the actual word matches that prediction. Positions where the
        learner has no data for the prefix are skipped — sparse
        experience cannot penalize the score.

        Returns 0.5 (neutral) when there is no evidence either way.
        """
        tokens = self._tokens(text)
        if len(tokens) < 2:
            return 0.5
        matches = 0
        scored = 0
        for i in range(1, len(tokens)):
            predicted = self._statistical_learner.suggest_next_word(tokens[:i])
            if predicted is not None:
                scored += 1
                if tokens[i] == predicted:
                    matches += 1
        if scored == 0:
            return 0.5
        return matches / scored

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Return normalized lexical tokens for realization comparison."""
        return [token.casefold() for token in _WORD_RE.findall(text)]

    def _semantic_coverage(self, text: str, thought: Thought) -> float:
        """Measure how much of the thought's lexical grounding survives."""
        sources = [*thought.topics]
        for key in ("target_concept", "topic"):
            value = thought.metadata.get(key)
            if isinstance(value, str):
                sources.append(value)
        if not sources:
            sources.append(thought.content)

        required = {
            token
            for source in sources
            for token in self._tokens(source)
            if len(token) > 2 and token not in _DEDUP_FUNCTION_WORDS
        }
        if not required:
            return 1.0
        realized = set(self._tokens(text))
        return len(required & realized) / len(required)

    def _surface_fluency(self, text: str) -> float:
        """Score general well-formedness and penalize local degeneration."""
        tokens = self._tokens(text)
        if not tokens:
            return 0.0

        score = 1.0
        if _REPEATED_WORD_RE.search(text) or _REPEATED_PUNCT_RE.search(text):
            score -= 0.25
        if _MULTISPACE_RE.search(text) or _SPACE_BEFORE_PUNCT_RE.search(text):
            score -= 0.15

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY_RE.split(text)
            if sentence.strip()
        ]
        malformed = 0
        incomplete_heads = {"a", "an", "the", "and", "but", "or", "to"}
        incomplete_predicates = {"am", "is", "are", "was", "were", "be", "been"}
        for sentence in sentences:
            sentence_tokens = self._tokens(sentence)
            first_letter = next((c for c in sentence if c.isalpha()), "")
            incomplete = bool(sentence_tokens) and (
                sentence_tokens[-1] in incomplete_heads
                or (
                    len(sentence_tokens) <= 3
                    and sentence_tokens[-1] in incomplete_predicates
                )
            )
            if (
                sentence[-1] not in ".!?"
                or not first_letter
                or first_letter.islower()
                or incomplete
            ):
                malformed += 1
        if sentences:
            score -= 0.3 * malformed / len(sentences)

        if len(tokens) >= 6:
            diversity = len(set(tokens)) / len(tokens)
            score -= max(0.0, 0.55 - diversity)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _intent_fidelity(text: str, intent: str) -> float:
        """Measure whether sentence force realizes the intended speech act."""
        stripped = text.rstrip()
        if not stripped:
            return 0.0
        if intent == "ask":
            return 1.0 if stripped.endswith("?") else 0.0
        declarative_intents = {
            "inform",
            "correct",
            "self_report",
            "reflect",
            "philosophize",
        }
        if intent in declarative_intents:
            return 0.4 if stripped.endswith("?") else 1.0
        return 1.0

    def _discourse_novelty(self, text: str) -> float:
        """Prefer realizations that do not repeat recent lexical sequences."""
        tokens = self._tokens(text)
        if not tokens or not self._recent_outputs:
            return 1.0

        units = self._comparison_units(tokens)
        greatest_overlap = 0.0
        for previous in self._recent_outputs:
            previous_units = self._comparison_units(self._tokens(previous))
            union = units | previous_units
            if union:
                greatest_overlap = max(
                    greatest_overlap,
                    len(units & previous_units) / len(union),
                )
        return 1.0 - greatest_overlap

    @staticmethod
    def _comparison_units(tokens: list[str]) -> set[tuple[str, ...]]:
        """Use trigrams for long text and lexical units for short text."""
        if len(tokens) < 3:
            return {(token,) for token in tokens}
        return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}

    def _compose_grammar(self, thought: Thought, emotion: EmotionalState) -> str:
        """Compose text using grammar + vocabulary (the fallback path)."""
        intent = thought.intent

        # Build context for slot filling
        context: dict[str, Any] = {
            "content": thought.content,
            "topics": thought.topics,
            "confidence": thought.confidence,
            "first_interaction": thought.metadata.get("first_interaction", False),
            # Self-description fragments — typed (kind, text) semantic
            # material from the self-composer (name, traits, predicates,
            # clauses, markers). The vocabulary composes the content
            # slot from them; the thought content itself stays a short
            # anchor (topic or name) rather than a finished sentence.
            "self_fragments": thought.metadata.get("self_fragments"),
            # Causal-answer qualification — structured relation data
            # (kind/subject/object/others) from the thought composer's
            # graph traversal. The vocabulary composes the connective
            # phrasing; the reasoning layer supplies only the facts.
            "qualification": thought.metadata.get("qualification"),
            # Its name — so the intro_clause slot can compose
            # "I'm <name>" from the self-model rather than a hardcoded
            # string. The name is a seed (building block), not a
            # hardcoded response.
            "name": self.self_model.name,
            # Question-specific metadata for the question_content slot
            "question_type": thought.metadata.get("question_type", ""),
            "target_concept": thought.metadata.get("target_concept", ""),
            "gap_detail": thought.metadata.get("gap_detail", ""),
            # Knowledge metadata — lets the vocabulary compose content
            # from the concept network instead of using pre-composed text
            "knowledge": thought.metadata.get("knowledge"),
            "definition": thought.metadata.get("definition"),
            "topic": thought.metadata.get("topic") or (
                thought.topics[0] if thought.topics else ""
            ),
            "reasoning": thought.metadata.get("reasoning"),
            # Episodic memory metadata — a cleaned, highly-relevant
            # episode from LTM that the vocabulary weaves into the
            # composed statement alongside concept-network knowledge.
            "memory": thought.metadata.get("memory"),
            # User belief metadata — lets the vocabulary compose a
            # response from what it knows about the user (preferences,
            # beliefs, etc.) without dumping raw profile data.
            "user_belief": thought.metadata.get("user_belief"),
            "user_verb": thought.metadata.get("user_verb"),
            # Relation answer metadata — lets the vocabulary compose a
            # response from the structured relation data (subject, verb,
            # objects, direction) that the question handler traversed,
            # instead of the handler building the words itself.
            "relation_answer": thought.metadata.get("relation_answer"),
            # Self-improvement metadata — lets the vocabulary compose a
            # response from its proposals, experiments, or growth data
            # instead of reciting the raw admin summary.
            "self_improvement_kind": thought.metadata.get("self_improvement_kind"),
            "self_improvement_data": thought.metadata.get("self_improvement_data"),
            # Feeling fragments — lets the vocabulary compose the content
            # slot from emotion/mode/cause/plasticity words (from the concept
            # network) using varied grammatical structures, instead of the
            # feeling_reporter pre-composing a fixed "I feel X and Y" frame.
            "feeling_fragments": thought.metadata.get("feeling_fragments"),
            # Reciprocal question — for express_emotion, a "how about you?"
            # follow-up composed from the vocabulary's seed list.
            "reciprocal_question": thought.metadata.get("reciprocal_question"),
            # Preference metadata — lets the vocabulary compose the
            # content slot as a predicate from structured data
            # (status/topic/value) instead of reciting a pre-composed
            # first-person sentence.
            "preference": thought.metadata.get("preference"),
            # Web search metadata — when it looked something up online,
            # the vocabulary composes its expression of what it found
            # from the summary, not by reciting the page verbatim.
            "web_summary": thought.metadata.get("web_summary"),
            "web_source": thought.metadata.get("web_source"),
            # Vision scene metadata — the retina feed's structured
            # percept (lighting, color, faces, objects, positions).
            # The vocabulary composes its report from the data; the
            # vision module itself never writes its words.
            "vision_scene": thought.metadata.get("vision_scene"),
            # Vision status — when perception itself failed or is
            # unavailable ("unavailable", "mid_update", "unprocessed",
            # "load_failed"), the vocabulary composes its report of
            # *that* instead of leaving a raw status word in the slot.
            "vision_status": thought.metadata.get("vision_status"),
            # Problem-solving metadata — the structured outcome of a
            # task the inner world just worked (family, solved, the
            # answer payload). The vocabulary composes the report
            # from the data; the practice layer never writes words.
            "problem_result": thought.metadata.get("problem_result"),
        }

        # Determine how many sentences to generate
        num_sentences = self._determine_length(intent, emotion, thought)

        # When the thought carries self-description fragments, the
        # vocabulary composes complete sentence candidates ("I care
        # about X", "Alice is a creator"). Copula frames ("I am
        # {content}") only fit when every candidate is guaranteed to
        # lead with "I'm" — i.e. when the fragments carry name/trait/
        # complement material. Otherwise they're excluded so the frame
        # can't produce "I am care about X".
        pool: list[SentenceStructure] | None = None
        if context.get("self_fragments"):
            pool = self._self_fragment_pool(
                self.grammar.get_structures(intent),
                context["self_fragments"],
            )

        # Select the first (content-bearing) sentence structure
        emotion_weight = emotion.creativity * 0.5 + emotion.openness_to_engage * 0.5
        first_structure = self.grammar.select_structure(
            intent, emotion_weight, pool=pool
        )

        # Generate the first sentence (carries the {content} slot)
        sentences: list[str] = []
        first_sentence = self._fill_structure(
            first_structure, context, emotion, thought
        )
        if first_sentence:
            sentences.append(first_sentence)

        # Generate follow-up sentences using content-free structures.
        # This prevents the content-echo defect: when every structure
        # fills the same {content} slot, multi-sentence output repeats
        # the content ("A dog is a mammal. A dog is a mammal."). The
        # follow-up structures express reflection, qualification, or
        # engagement WITHOUT repeating the content.
        for _ in range(num_sentences - 1):
            followup = self.grammar.select_followup(intent, emotion_weight)
            if followup is None:
                break  # no follow-up structures for this intent
            sentence = self._fill_structure(
                followup, context, emotion, thought
            )
            if sentence:
                sentences.append(sentence)

        # Drop content-free sentences (empty strings, bare punctuation
        # like "." left when every slot in the structure was empty).
        # If nothing has word content, fall back to the raw content —
        # a bare "." is never an acceptable utterance.
        sentences = [s for s in sentences if re.search(r"\w", s)]
        if not sentences:
            sentences = [thought.content]

        # Apply voice
        text = self.voice.apply(sentences, emotion, intent)

        return text

    def _fill_structure(
        self,
        structure: SentenceStructure,
        context: dict[str, Any],
        emotion: EmotionalState,
        thought: Thought,
    ) -> str:
        """Compose a sentence by walking the structure's segments.

        The structure is a segment list (Literal / Slot), not a template
        string. Each Literal is appended verbatim; each Slot is filled
        from the vocabulary. Unfilled slots (empty word) are omitted by
        construction — there is no placeholder string to strip.
        """
        personality = self.self_model.personality
        parts: list[str] = []
        prev_literal = ""
        for seg in structure.segments:
            if isinstance(seg, Literal):
                parts.append(seg.text)
                prev_literal = seg.text
            elif isinstance(seg, Slot):
                # Last emitted words — feeds n-gram fluency shaping so
                # slot fills prefer transitions it's observed. Two
                # words of context enable trigram shaping; one word
                # falls back to bigram.
                prev_word = ""
                prev_prev_word = ""
                if parts:
                    tail = parts[-1].split()
                    if tail:
                        prev_word = tail[-1]
                        if len(tail) >= 2:
                            prev_prev_word = tail[-2]
                word = self.vocabulary.fill_slot(
                    seg.name, context, emotion, personality,
                    prev_word=prev_word,
                    prev_prev_word=prev_prev_word,
                )
                # Prevent pronoun collision: when a frame literal
                # ends with "I am " or "I'm " and the slot content
                # starts with "I am"/"I'm"/"I ", strip the redundant
                # leading pronoun. Without this, "I'm {content}." with
                # content "I am Genesis..." produces "I'm I am Genesis..."
                if (
                    seg.name == "content"
                    and word
                    and prev_literal
                    and prev_literal.rstrip().endswith(("I am", "I'm"))
                    and word.startswith(("I am ", "I'm ", "I "))
                ):
                    word = self._strip_leading_pronoun(word)
                parts.append(word)

        text = "".join(parts)

        # Clean up: collapse multiple spaces (omitted slots can leave them)
        text = _MULTISPACE_RE.sub(" ", text)

        # Clean up: fix spacing around punctuation
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)

        # Add appropriate ending punctuation
        if structure.sentence_type.value == "interrogative":
            # Strip trailing "?" if the content already ends with one,
            # then add exactly one. This prevents "??" when the slot
            # content (e.g. question_content) already ends with "?".
            text = text.rstrip("?").rstrip() + "?"
        elif structure.sentence_type.value == "exclamatory":
            if not text.endswith("!"):
                text = text.rstrip(".").rstrip() + "!"
        else:
            if not text.endswith((".", "!", "?")):
                text += "."

        return text.strip()

    @staticmethod
    def _self_fragment_pool(
        structures: list[SentenceStructure],
        fragments: list,
    ) -> list[SentenceStructure]:
        """Exclude copula frames when fragment candidates aren't copula-safe.

        Copula frames ("I am {content}", "I'm {content}", "{opener} I am
        {content}") prepend "I am" to the content slot. That's safe when
        the fragments carry name/trait/complement material — every
        candidate then leads with "I'm", which the collision guard
        strips cleanly. When the fragments are pure predicates or
        clauses (no lead material), a candidate like "I care about X"
        or "Alice is a creator" under a copula frame produces "I am
        care about X" — so those frames are removed from the pool.
        """
        has_lead = any(
            isinstance(f, (tuple, list)) and len(f) == 2
            and f[0] in ("name", "trait", "comp")
            for f in fragments
        )
        if has_lead:
            return structures
        safe = [
            s for s in structures
            if not GenerativeEngine._is_copula_content_frame(s)
        ]
        return safe or structures

    @staticmethod
    def _is_copula_content_frame(structure: SentenceStructure) -> bool:
        """True when a literal ending in "I am"/"I'm" precedes {content}."""
        segs = structure.segments
        for i, seg in enumerate(segs[:-1]):
            nxt = segs[i + 1]
            if (
                isinstance(seg, Literal)
                and isinstance(nxt, Slot)
                and nxt.name == "content"
                and seg.text.rstrip().endswith(("I am", "I'm"))
            ):
                return True
        return False

    @staticmethod
    def _strip_leading_pronoun(word: str) -> str:
        """Strip a redundant leading first-person pronoun from content.

        When a grammar frame ("I am {content}", "I'm {content}") is
        applied to content that already starts with "I am"/"I'm"/"I ",
        the result is a pronoun collision ("I'm I am Genesis...").
        This strips the leading pronoun so the frame wraps the
        predicate, not the full first-person sentence.
        """
        for prefix in ("I am ", "I'm ", "I "):
            if word.startswith(prefix):
                return word[len(prefix):]
        return word

    def _determine_length(self, intent: str, emotion: EmotionalState, thought: Thought) -> int:
        """Determine how many sentences to generate.

        Based on intent, emotional state, and content complexity.
        """
        # Intent-based base length
        base: dict[str, int] = {
            "greet": 1,
            "farewell": 1,
            "acknowledge": 1,
            "ask": 1,
            "encourage": 1,
            "correct": 1,
            "inform": 2,
            "express_emotion": 2,
            "reflect": 2,
            "self_report": 2,
            "discuss_code": 2,
            "philosophize": 3,
            "introduce": 1,  # single multi-slot structure
            "unknown": 1,
        }

        # Social closures should stay exactly one sentence. Adding
        # extra clauses to a goodbye or hello sounds strange and can
        # produce unintended follow-up questions from the generator.
        if intent in {"greet", "farewell", "acknowledge"}:
            return 1

        # When self_fragments metadata is present, the vocabulary
        # already composes a complete multi-clause utterance from the
        # fragments. Generating multiple sentences from it produces
        # repetition. Keep it to one structure.
        if thought.metadata.get("self_fragments"):
            return 1

        # When reasoning metadata is present, the content is a composed
        # statement from _compose_reasoning_content. Generating multiple
        # sentences from the same reasoning produces repetition. Keep
        # it to one sentence.
        if thought.metadata.get("reasoning"):
            return 1

        n = base.get(intent, 2)

        # Emotional modulation — but never increase "ask" intent
        # (questions should be concise; 2 sentences duplicates content)
        if intent != "ask" and emotion.creativity > 0.6 and emotion.openness_to_engage > 0.6:
            n += 1  # more expressive when creative and engaged
        if emotion.caution > 0.6:
            n = max(1, n - 1)  # shorter when cautious
        if emotion.openness_to_engage < 0.3:
            n = max(1, n - 1)  # shorter when disengaged

        # Content-based: long content → more sentences
        if len(thought.content) > 100:
            n = max(n, 2)

        # If the thought carries raw knowledge metadata, the vocabulary
        # already composes a complete multi-clause sentence. Don't
        # duplicate it across multiple structures.
        if thought.metadata.get("knowledge") is not None and intent in ("reflect", "inform"):
            n = min(n, 1)

        # If the thought carries user belief metadata (preferences,
        # wants, etc.), there's only one fact to express — don't
        # duplicate it across multiple sentences.
        if thought.metadata.get("user_belief") is not None:
            n = min(n, 1)

        # Same for preference answers — one fact about its own
        # preference state, one sentence.
        if thought.metadata.get("preference") is not None:
            n = min(n, 1)

        return min(n, 4)  # cap at 4 sentences

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for voice processing."""
        # Split on sentence boundaries
        parts = _SENTENCE_BOUNDARY_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    # ─── LanguageEngine interface ──────────────────────────

    def greet(self, emotion: EmotionalState, first: bool = False) -> str:
        """Generate a greeting."""
        thought = Thought(
            content="greeting",
            intent="greet",
            emotion=emotion.label,
            confidence=0.9,
            metadata={"first_interaction": first},
        )
        return self.generate(thought, emotion)

    def farewell(self, emotion: EmotionalState) -> str:
        """Generate a farewell."""
        thought = Thought(
            content="farewell",
            intent="farewell",
            emotion=emotion.label,
            confidence=0.9,
        )
        return self.generate(thought, emotion)

    def acknowledge(self, emotion: EmotionalState, topic: str = "") -> str:
        """Generate an acknowledgment."""
        thought = Thought(
            content=topic or "acknowledged",
            intent="acknowledge",
            emotion=emotion.label,
            confidence=0.6,
            topics=[topic] if topic else [],
        )
        return self.generate(thought, emotion)

    def self_report(
        self,
        emotion: EmotionalState,
        self_fragments: list[tuple[str, str]] | None = None,
    ) -> str:
        """Generate a self-report.

        If ``self_fragments`` is supplied it provides the semantic
        inventory (typed (kind, text) fragments); otherwise the
        self-composer is invoked to select identity fragments from
        the concept network. The language engine composes the actual
        phrasing — no hardcoded self-description strings are recited.
        """
        fragments = self_fragments
        if fragments is None:
            # Select identity fragments from the self-model + concept
            # network rather than reciting a hardcoded self-description.
            from ..self.composer import SelfComposer

            composer = SelfComposer()
            fragments = composer.identity_fragments(
                self.self_model, self._network, emotion
            )
        thought = Thought(
            content=self.self_model.name or "identity",
            intent="self_report",
            emotion=emotion.label,
            confidence=0.8,
            self_reflection=True,
            metadata={"self_fragments": fragments, "field": "identity"},
        )
        return self.generate(thought, emotion)

    def philosophize(self, emotion: EmotionalState, topic: str = "") -> str:
        """Generate a philosophical reflection."""
        thought = Thought(
            content=topic or "existence",
            intent="philosophize",
            emotion=emotion.label,
            confidence=0.6,
            self_reflection=True,
        )
        return self.generate(thought, emotion)

    def express_emotion(self, emotion: EmotionalState) -> str:
        """Express an emotional state."""
        thought = Thought(
            content=emotion.label,
            intent="express_emotion",
            emotion=emotion.label,
            confidence=0.8,
        )
        return self.generate(thought, emotion)

    def vary(self, text: str, emotion: EmotionalState) -> str:
        """Apply variation to existing text."""
        sentences = self._split_into_sentences(text)
        return self.voice.apply(
            sentences, emotion, "inform",
            brain_waves=self.current_brain_waves,
        )

    def compose_question(
        self,
        question_data: dict,
        emotion: EmotionalState,
    ) -> str:
        """Compose a question from semantic metadata.

        Takes a dict with question_type, target_concept, gap_detail
        and composes a natural language question through the grammar
        + vocabulary + voice system. This is how all questions are
        generated — no hardcoded templates.
        """
        thought = Thought(
            content=question_data.get("gap_detail", ""),
            intent="ask",
            emotion=emotion.label,
            confidence=0.6,
            topics=(
                [question_data.get("target_concept", "")]
                if question_data.get("target_concept")
                else []
            ),
            metadata={
                "question_type": question_data.get("question_type", ""),
                "target_concept": question_data.get("target_concept", ""),
                "gap_detail": question_data.get("gap_detail", ""),
            },
        )
        # Override the context with question-specific fields
        text = self._compose(thought, emotion)
        # Defensive fallback: a question must never be empty or only
        # punctuation. If the grammar/voice pipeline produced nothing,
        # retry with a simpler structure so the user still sees something.
        if not re.search(r"\w", text):
            thought = Thought(
                content=question_data.get("target_concept", "something"),
                intent="ask",
                emotion=emotion.label,
                confidence=0.4,
                topics=([question_data.get("target_concept", "")]
                         if question_data.get("target_concept") else []),
            )
            text = self.generate(thought, emotion)
        return text
