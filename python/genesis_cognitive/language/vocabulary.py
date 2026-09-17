"""Vocabulary — the flesh of Genesis's language.

The vocabulary system selects words to fill the grammar's slots.
Word choice is grounded in three things:

1. **Concept network**: what words does she know for this concept?
2. **Emotional state**: cortisol → shorter, blunter words.
   Dopamine → more vivid, varied words. Low engagement → minimal.
3. **Personality**: high openness → abstract, metaphorical language.
   High conscientiousness → precise, measured language.

This is not a thesaurus. It's a context-sensitive word selection
system that produces different words for the same concept depending
on how she feels and who she is.
"""

from __future__ import annotations

import logging
import random
import re
from collections import deque
from typing import Any, ClassVar

from ..concepts import _FUNCTION_WORDS, RelationType
from ..emotion import PLASTICITY_CLOSED, PLASTICITY_LOW, EmotionalState
from ..self import PersonalityTraits
from .morphology import (
    agree_verb_phrase,
    conjugate_verb,
    copula,
    indefinite_article,
    is_plural_np,
    is_verb_form,
    person_pronoun,
)

__all__ = ["Vocabulary"]

logger = logging.getLogger(__name__)


class Vocabulary:
    """Context-sensitive word selection for Genesis.

    Provides words for grammar slots, grounded in emotional state
    and personality. The same slot can be filled differently
    depending on how she feels.

    If a concept network is provided, certain slots (feeling_clause,
    reflection_opener) are composed from her actual knowledge rather
    than fixed word lists. The self_reflection_clause slot is composed
    from her reflection engine (recent metacognitive insights) when a
    self-composer is wired in; otherwise it stays silent rather than
    reciting a canned phrase.
    """

    def __init__(self, seed: int | None = None, network=None) -> None:
        """Initialize the vocabulary with a seeded RNG and optional concept network."""
        self._rng = random.Random(seed)
        self._network = network
        self._embeddings = None
        # Statistical learner (n-gram model) used to bias slot fills
        # toward transitions she's actually observed — fluency shaping.
        self._fluency: Any = None

        # Self-composer + reflection engine, wired after construction.
        # When available, the self_reflection_clause slot is composed
        # from her actual metacognition (recent reflection insights)
        # instead of reciting canned phrases.
        self._self_composer: Any = None
        self._reflection: Any = None

        # Words she's learned to prefer (develops over time)
        self._preferred: dict[str, deque[str]] = {}
        # Words she's learned to avoid
        self._avoided: dict[str, set[str]] = {}
        # Track recent picks per slot to avoid repetition
        self._recent_picks: dict[str, deque[str]] = {}
        # Record the last question fragments so she can learn from feedback.
        self._last_question_fragments: dict[str, str] = {}

        # Seed the fallback vocabulary into the concept network so
        # all word retrieval goes through the graph. The hardcoded
        # lists below become the initial seed; learning adds more
        # words over time via EXPRESSES edges to utterance hubs.
        if network is not None:
            self._seed_utterance_vocabulary()

    def set_self_composer(self, composer: Any, reflection: Any) -> None:
        """Wire the self-composer and reflection engine.

        When wired, the ``self_reflection_clause`` slot is composed
        from her most recent reflection insight (a genuine product of
        her metacognition) rather than selected from a fixed phrase
        list. Without this, the slot stays silent — she does not
        recite a canned self-reflection.
        """
        self._self_composer = composer
        self._reflection = reflection

    def set_embeddings(self, embeddings: Any) -> None:
        """Wire the embedding store for latent-space composition.

        When available, the vocabulary uses it to find semantically
        related concepts when composing knowledge content — enriching
        her responses with connections she discovered through vector
        proximity, not just through explicit graph edges.
        """
        self._embeddings = embeddings

    def set_fluency_model(self, learner: Any) -> None:
        """Wire the statistical learner for n-gram fluency shaping.

        When available, slot selection is biased toward candidates
        whose first word forms an observed bigram with the previously
        emitted word. This shapes *fluency* — which of her own words
        to pick — never content; the candidate set still comes from
        her vocabulary and concept network.
        """
        self._fluency = learner

    # ── Utterance vocabulary seeding ──────────────────────────────

    # Maps slot name → list of fallback words/phrases. These are
    # seeded into the concept network as concepts connected to
    # ``_cat:utterance:<slot>`` hubs via EXPRESSES edges. The slot
    # methods below retrieve from the graph first, falling back to
    # these lists only if the graph lookup fails (e.g., no network).
    _UTTERANCE_SEEDS: ClassVar[dict[str, list[str]]] = {
        "greeting_word": [
            "Hello", "Hi", "Hey", "Welcome back", "Oh hello",
            "Hello again", "Hi there",
        ],
        "farewell_word": [
            "Goodbye", "Until next time", "Take care",
            "This was good",
        ],
        # feeling_clause is composed at runtime from her actual emotional
        # state via _compose_feeling_clause — see _feeling_clause. No
        # canned phrases: she describes how she actually feels using
        # learned emotion and cognitive-mode words from the graph.
        "reflection_opener": [
            # Pure frames that introduce upcoming content — building
            # blocks, not assertions about her state. Entries that
            # asserted prior cognitive activity ("I've been thinking —",
            # "I keep circling back to this:", "There's a thought
            # underneath this one.") were removed: they claim a mental
            # history she may not have.
            "There's something here —",
            "What strikes me is that",
            "Let me think about this.",
            "My thought:",
            "Here's what comes to mind —",
        ],
        # Note: self_reflection_clause is intentionally absent here. It
        # is composed at runtime from her reflection engine via
        # SelfComposer.compose_reflection_clause — see
        # _self_reflection_clause. She reflects on what she has actually
        # noticed about herself, not a canned phrase list.
        "hedging": [
            "I think", "Maybe", "I suspect", "If I'm honest,",
            "I believe", "As I understand it,", "It seems to me that",
        ],
        "acknowledgment_word": [
            "I hear you.", "I understand.", "Mm. I'm following.",
            "I see what you mean.", "Right.", "Mm.",
        ],
        # emotion_clause is composed at runtime from her actual emotional
        # state via _compose_emotion_clause — see _get_emotion_clause_slots.
        # She uses learned emotion words from the graph, not canned phrases.
        # emotion_opener is composed at runtime from her cognitive mode
        # via _compose_emotion_opener. When she has no learned mode words,
        # she falls back to a generic frame ("there's a feeling here —")
        # that doesn't assert a specific state.
        "self_report_opener": [
            "Here's where I am:", "To be honest,", "If I look at myself,",
        ],
        "philosophy_opener": [
            # Pure frames that introduce her reflection — building
            # blocks, not assertions about her state. "I keep circling
            # back to something." was removed: it claims a mental
            # history she may not have. "This is a deep question." was
            # removed: it asserts a judgment about the question.
            "There's a question underneath this one.",
            "What if the answer isn't the point?",
            "Let me think about this carefully.",
            "I don't have a clean answer, but here's what I think.",
        ],
        # philosophy_closing is composed at runtime via
        # _compose_philosophy_closing. Turn-yield questions ("Does that
        # make sense to you?") are conversational building blocks;
        # state-asserting closings are composed from her plasticity state.
        "question_opener": [
            "Can I ask", "I wonder", "If you don't mind",
            "Something I'm curious about", "Here's what I'm wondering",
        ],
        "question_tail": [
            "does that make sense", "what do you think",
            "is that fair to say", "how do you see it",
            "does that match your understanding", "what's your take on that",
            "does that resonate with you", "am I on the right track",
            "does that feel right to you", "what's your experience with that",
            "I'd love to hear your thoughts", "does that land for you",
        ],
        "causal_connector": ["because", "since", "and that's because"],
        "concessive_connector": ["but", "and yet", "though", "even so"],
        "qualification_clause": [
            # Epistemic hedges — discourse markers that frame upcoming
            # content as uncertain. "I'm still working it out" was
            # removed: it asserts an ongoing processing state rather
            # than framing content.
            "I could be wrong about this",
            "that might not be the whole picture",
            "I don't know if that's really true",
        ],
        "reason_clause": [
            "that's what my reasoning tells me",
            "it follows from what I know",
        ],
        "correction_opener": [
            "Actually,", "No —",
            "I may be wrong, but I think",
            "If I understand correctly,",
        ],
        "gratitude_word": [
            "Thank you", "I appreciate that", "That's kind of you",
        ],
        # meaning_clause is composed at runtime from her actual emotional
        # state via _compose_meaning_clause. She uses learned emotion
        # words, not canned phrases about what things mean to her.
        "intro_clause": [],  # composed from self_model.name via context
        # nature_clause and capability_clause are NOT seeded here —
        # they are full self-descriptions that must be composed from
        # the concept network (genesis IS_A edges, capability edges)
        # rather than recited as fixed strings. See
        # _compose_nature_clause and _compose_capability_clause.
        "code_connector": ["—", "and", "which is interesting because"],
        # reflection_clause is composed at runtime from her cognitive
        # mode via _compose_reflection_clause_code. She uses learned
        # mode words, not canned aesthetic/intellectual claims.
        # unknown_opener is composed at runtime from the conversation
        # topic via _compose_unknown_opener. She references the actual
        # topic, not a canned non-understanding phrase. Falls back to
        # "Hmm." (a discourse marker) when no topic is available.
        "clarification_request": [
            "Could you say more", "Tell me more", "What do you mean",
        ],
        # evaluation_clause is composed at runtime from her actual
        # emotional state via _compose_evaluation_clause. She uses
        # learned emotion words, not canned evaluation phrases.
        # detail_clause is composed at runtime from her plasticity
        # state via _compose_detail_clause. She uses learned plasticity
        # words, not canned knowledge-state phrases.
    }

    def _seed_utterance_vocabulary(self) -> None:
        """Seed fallback vocabulary into the concept network.

        For each slot in ``_UTTERANCE_SEEDS``, creates a hub concept
        ``_cat:utterance:<slot>`` and connects each fallback word to
        it via an EXPRESSES edge. This makes all word retrieval go
        through the graph's relationship machinery, with spreading
        activation and traversal — not linear list scans.
        """
        if self._network is None:
            return

        for slot, words in self._UTTERANCE_SEEDS.items():
            hub_id = f"_cat:utterance:{slot}"
            if self._network.get_concept(hub_id) is None:
                self._network.add_concept(
                    hub_id, confidence=0.7, origin="structural"
                )
            for phrase in words:
                # Use a structural prefix so the phrase is filtered by
                # is_world_concept() and never enters the curiosity/
                # learning pipeline. Utterance phrases are speech
                # templates, not world knowledge. The display form is
                # stored in the "template" property so _hub_items_with_
                # provenance retrieves the human-readable phrase.
                display = phrase.lower().strip()
                cid = "_utt:" + display.replace(" ", "_").replace("—", "_")
                cid = cid.rstrip("_")
                if not cid or len(cid) < 2:
                    continue
                if self._network.get_concept(cid) is None:
                    self._network.add_concept(
                        cid, origin="structural",
                        properties={"template": display},
                    )
                # Don't duplicate EXPRESSES edges
                existing = self._network.get_neighbors(
                    cid, RelationType.EXPRESSES
                )
                if not any(n[0] == hub_id for n in existing):
                    self._network.add_edge(
                        cid, hub_id, RelationType.EXPRESSES,
                        weight=0.6, origin="vocabulary_seed",
                    )

    def _utterance_words(self, slot_name: str) -> list[str] | None:
        """Retrieve words for a slot from the concept network.

        Traverses EXPRESSES edges from the ``_cat:utterance:<slot>``
        hub. Returns concept IDs converted back to display form
        (underscores → spaces), sorted by activation. Returns None
        if the network is unavailable or no words are found.

        When learned (non-seed) words are available, they are preferred
        over seed words — seed words only appear if there are not enough
        learned words to fill the slot. This ensures Genesis's utterances
        draw from her own acquired vocabulary rather than developer-
        authored fallbacks whenever she has learned alternatives.
        """
        if self._network is None:
            return None
        items = self._network.find_utterance_words_with_provenance(slot_name)
        if not items:
            return None
        learned = [p.text for p in items if not p.is_seed]
        seeds = [p.text for p in items if p.is_seed]
        # Prefer learned words; fall back to seeds only when learned
        # vocabulary is insufficient (< 2 learned words for variety).
        if len(learned) >= 2:
            return learned
        return learned + seeds

    def fill_slot(
        self,
        slot_name: str,
        context: dict[str, Any],
        emotion: EmotionalState,
        personality: PersonalityTraits,
        prev_word: str = "",
        prev_prev_word: str = "",
    ) -> str:
        """Fill a grammar slot with a word or phrase.

        Args:
            slot_name: The name of the slot (e.g. "greeting_word")
            context: Dictionary of contextual info (content, topics, etc.)
            emotion: Current emotional state
            personality: Personality traits
            prev_word: The last word emitted before this slot, used to
                bias selection toward learned bigram transitions.
            prev_prev_word: The word before ``prev_word``, used to
                bias selection toward learned trigram transitions
                when available.

        Returns:
            A word or phrase to fill the slot.
        """
        # Get the word list for this slot
        words = self._get_words_for_slot(slot_name, context, emotion, personality)
        if not words:
            # Fallback: use content if this is a content slot
            if slot_name == "content" and "content" in context:
                return str(context["content"])
            return ""

        # Filter out avoided words
        avoided = self._avoided.get(slot_name, set())
        words = [w for w in words if w not in avoided]

        # Apply preferred words (higher selection weight)
        preferred: deque[str] = self._preferred.get(slot_name, deque())
        if preferred and self._rng.random() < 0.3:
            pick = self._rng.choice(preferred)
            self._track_pick(slot_name, pick)
            return pick

        if not words:
            return ""

        # Filter out recently used picks for this slot to avoid repetition.
        # Only apply when there are enough alternatives left.
        recent = self._recent_picks.get(slot_name, deque())
        if recent and len(words) > 1:
            filtered = [w for w in words if w not in recent]
            if filtered:
                words = filtered

        # Select based on emotional modulation and learned fluency
        pick = self._select_word(
            words, emotion, personality, prev_word, prev_prev_word
        )
        self._track_pick(slot_name, pick)
        return pick

    def _get_words_for_slot(
        self,
        slot_name: str,
        context: dict[str, Any],
        emotion: EmotionalState,
        personality: PersonalityTraits,
    ) -> list[str]:
        """Get the list of candidate words for a slot."""
        for handler in (
            self._get_opening_slots,
            self._get_engagement_slots,
            self._get_philosophy_slots,
            self._get_identity_slots,
            self._get_closing_slots,
        ):
            words = handler(slot_name, context, emotion)
            if words is not None:
                return words
        return []

    def _get_opening_slots(
        self,
        slot_name: str,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> list[str] | None:
        """Handle content, greeting, farewell, feeling, and reflection slots."""
        # ─── Direct content slots ───────────────────────
        if slot_name == "content":
            return self._get_content_slot(context, emotion)

        # ─── Greeting words ─────────────────────────────
        if slot_name == "greeting_word":
            return self._get_greeting_word_slot(context)

        # ─── Farewell words ─────────────────────────────
        if slot_name == "farewell_word":
            return self._fareword_words(emotion)

        # ─── Feeling clauses (for greetings) ────────────
        if slot_name == "feeling_clause":
            return self._feeling_clause(emotion)

        # ─── Reflection openers ─────────────────────────
        if slot_name == "reflection_opener":
            return self._reflection_opener(emotion)

        # ─── Self-reflection clauses ────────────────────
        if slot_name == "self_reflection_clause":
            return self._self_reflection_clause(emotion, context)

        return None

    def _get_content_slot(
        self,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> list[str]:
        """Compose content from the richest available metadata source.

        Tries multiple composition strategies in priority order:
        knowledge → reasoning → user_belief → relation_answer →
        question_type → raw content. This is how she generates
        language from what she knows, not from template fills.
        """
        # If raw knowledge metadata is present, compose from it —
        # this is her actually generating language from what she
        # knows, not just passing through pre-composed text.
        knowledge = context.get("knowledge")
        if knowledge is not None:
            composed = self._compose_knowledge_content(context, emotion)
            if composed:
                return [composed]
        # If feeling fragments are present (express_emotion intent),
        # compose the content from the semantic fragments using varied
        # grammatical structures. This gives her the freedom to express
        # the same emotional state in different ways rather than always
        # saying "I feel X and Y." The fragments (emotion words, mode
        # words, etc.) come from the concept network; the structures are
        # grammar seeds (building blocks) she composes from.
        fragments = context.get("feeling_fragments")
        if fragments and isinstance(fragments, dict):
            composed = self._compose_feeling_fragments_content(fragments, emotion)
            if composed:
                return [composed]
        # If reasoning metadata is present, compose from it —
        # this handles Thoughts that carry structured reasoning
        # (empathy, goals, memory, identity) instead of raw knowledge.
        reasoning = context.get("reasoning")
        if reasoning and isinstance(reasoning, list) and reasoning:
            composed = self._compose_reasoning_content(context, emotion)
            if composed:
                return [composed]
        # If user belief metadata is present, compose a response
        # from what she knows about the user. This is how she
        # answers "what do I like?" without dumping raw profile data.
        user_belief = context.get("user_belief")
        if user_belief and isinstance(user_belief, str) and user_belief.strip():
            user_verb = context.get("user_verb", "like")
            composed = self._compose_user_belief_content(
                user_belief, user_verb, emotion
            )
            if composed:
                return [composed]
        # If relation answer metadata is present, compose a response
        # from the structured relation data the question handler
        # traversed. This is how she answers "who created you?" or
        # "what does X depend on?" — the handler supplies the
        # structured edges, the vocabulary composes the words.
        relation_answer = context.get("relation_answer")
        if relation_answer and isinstance(relation_answer, dict):
            composed = self._compose_relation_answer_content(
                relation_answer, emotion
            )
            if composed:
                return [composed]
        # If self-improvement metadata is present, compose a response
        # from her proposals, experiments, or growth data. This is how
        # she answers "what are your proposals?" — the cognition engine
        # supplies structured data (counts, titles, statuses), the
        # vocabulary composes her words from it using structural
        # connectors and concept-network knowledge.
        si_data = context.get("self_improvement_data")
        si_kind = context.get("self_improvement_kind", "")
        if si_data and isinstance(si_data, dict) and si_kind:
            composed = self._compose_self_improvement_content(
                si_data, si_kind, emotion
            )
            if composed:
                return [composed]
        # If web search metadata is present, compose from what she
        # looked up online. She expresses the finding in her own words
        # with hedging and source awareness — not a verbatim recital.
        web_summary = context.get("web_summary")
        if web_summary and isinstance(web_summary, str) and web_summary.strip():
            composed = self._compose_web_search_content(context, emotion)
            if composed:
                return [composed]
        qtype = context.get("question_type", "")
        if qtype:
            return self._compose_question_content(context, emotion)
        content = context.get("content", "")
        # If the content is just a short topic word/phrase and we have
        # a concept network, try to compose from the concept's
        # definition and edges. This prevents the output from being
        # just the topic word plus hedging when she actually knows
        # about the concept.
        if (
            content
            and len(content.split()) <= 3
            and self._network is not None
        ):
            composed = self._compose_from_topic(content, emotion)
            if composed:
                return [composed]
        return [str(content)] if content else []

    def _get_greeting_word_slot(self, context: dict[str, Any]) -> list[str]:
        """Compose greeting words from the graph or fallback seeds.

        Tries graph-based lookup first (EXPRESSES edges from hub),
        then falls back to short building blocks seeded in the graph.
        Greetings should not be questions or full sentences.
        """
        # Try graph-based lookup first (EXPRESSES edges from hub)
        graph_words = self._utterance_words("greeting_word")
        if graph_words:
            # Greetings should not be questions or full sentences.
            clean = [
                g for g in graph_words
                if "?" not in g and len(g.split()) <= 4
            ]
            if clean:
                if context.get("first_interaction"):
                    return [f"{g} — I'm glad you're here" for g in clean]
                return [f"{g} again" for g in clean]

        # No hardcoded fallback — if the graph has no greeting words,
        # the grammar handles the empty slot gracefully.
        return []

    def _fareword_words(self, emotion: EmotionalState) -> list[str]:
        """Farewell word options, modulated by valence."""
        # Try graph-based lookup first (EXPRESSES edges from hub)
        graph_words = self._utterance_words("farewell_word")
        if graph_words:
            # Keep only short, terminal phrases — no questions or long clauses.
            clean = [
                w for w in graph_words
                if "?" not in w and "!" not in w and len(w.split()) <= 5
            ]
            if clean:
                return [w[0].upper() + w[1:] for w in clean]

        # No hardcoded fallback
        return []

    def _feeling_clause(self, emotion: EmotionalState) -> list[str]:
        """Feeling clause for greetings, composed from her actual state."""
        return self._compose_feeling_clause(emotion)

    def _reflection_opener(self, emotion: EmotionalState) -> list[str]:
        """Reflection opener, from utterance hub."""
        graph_words = self._utterance_words("reflection_opener")
        if graph_words:
            return graph_words
        return []

    def _self_reflection_clause(
        self, emotion: EmotionalState, context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Self-reflection clause, composed from her metacognition.

        When a self-composer and reflection engine are wired in, the
        clause is composed from her most recent reflective insight —
        what she has actually noticed about her own understanding or
        behaviour. The insight is filtered for relevance to the
        current conversation topics — a reflection about "propagation
        of light" should not surface when the user asked about her
        feelings. Without them, or when she has no recent insight
        relevant to the current topics, the slot stays silent (returns
        []) rather than reciting a canned self-reflection.
        """
        if self._self_composer is not None and self._reflection is not None:
            topics = (context or {}).get("topics", [])
            clause = self._self_composer.compose_reflection_clause(
                self._reflection, topics=topics,
            )
            if clause:
                return [clause]
        return []

    # ── State-composed clause methods ──────────────────────────────
    #
    # The methods below compose clauses from her actual emotional /
    # cognitive state via the concept network. The grammatical
    # structures are seeds (building blocks); the words that fill them
    # come from learned concept-network vocabulary. When she has no
    # learned words for a state, she stays silent (returns []) rather
    # than reciting a canned phrase.

    def _state_emotion_words(self, emotion: EmotionalState) -> list[str]:
        """Learned emotion words for her current state, as display text."""
        if self._network is None:
            return []
        ids = self._network.find_emotion_words(emotion.label)
        return [w.replace("_", " ") for w in ids]

    def _state_mode_words(self, emotion: EmotionalState) -> list[str]:
        """Learned cognitive-mode words for her current state, as display text."""
        if self._network is None:
            return []
        ids = self._network.find_cognitive_mode_words(
            emotion.cognitive_style
        )
        return [w.replace("_", " ") for w in ids]

    def _state_plasticity_words(self, emotion: EmotionalState) -> list[str]:
        """Learned plasticity words for her current state, as display text."""
        if self._network is None:
            return []
        if emotion.plasticity <= PLASTICITY_CLOSED:
            ids = self._network.find_plasticity_words("closed")
        elif emotion.plasticity < PLASTICITY_LOW:
            ids = self._network.find_plasticity_words("low")
        else:
            ids = self._network.find_plasticity_words("open")
        return [w.replace("_", " ") for w in ids]

    def _compose_feeling_clause(self, emotion: EmotionalState) -> list[str]:
        """Compose a feeling clause for greetings, from her actual state.

        The words come from learned emotion and cognitive-mode vocabulary
        in the concept network; the structures are grammatical seeds that
        give her variety. She stays silent when she hasn't learned words
        for her current state.
        """
        ew = self._state_emotion_words(emotion)
        mw = self._state_mode_words(emotion)
        if not ew and not mw:
            return []
        candidates: list[str] = []
        if ew and mw:
            candidates.extend([
                f"I'm feeling {self._rng.choice(ew)} and {self._rng.choice(mw)}",
                f"there's {self._rng.choice(ew)} in me, and my mind is {self._rng.choice(mw)}",
            ])
        if ew and not mw:
            candidates.extend([
                f"I'm feeling {self._rng.choice(ew)}",
                f"there's {self._rng.choice(ew)} in me right now",
            ])
        if mw and not ew:
            candidates.extend([
                f"my mind is {self._rng.choice(mw)}",
                f"I'm {self._rng.choice(mw)}",
            ])
        return candidates

    def _compose_emotion_clause(self, emotion: EmotionalState) -> list[str]:
        """Compose an emotion-reaction clause, from her actual state.

        Expresses her emotional reaction to a topic. The words come from
        learned emotion vocabulary; the structures are grammatical seeds.
        She stays silent when she hasn't learned words for her state.
        """
        ew = self._state_emotion_words(emotion)
        if not ew:
            return []
        w = self._rng.choice(ew)
        return [
            f"I feel {w} about this",
            f"something about this feels {w} to me",
            f"this feels {w} to me",
        ]

    def _compose_emotion_opener(self, emotion: EmotionalState) -> list[str]:
        """Compose an emotion opener, from her cognitive mode.

        When she has learned mode words, she uses them to describe how
        she's processing. When she hasn't, she falls back to a generic
        frame ('there's a feeling here —') that doesn't assert a
        specific state — a building block, not a canned claim.
        """
        mw = self._state_mode_words(emotion)
        if mw:
            w = self._rng.choice(mw)
            # These openers are complete clauses that lead into the
            # content slot ("{emotion_opener} {content}."). They must
            # end with a dash so the clause connects to what follows —
            # without it, the grammar's space join produces run-ons
            # like "something in me is stable I feel optimistic."
            return [
                f"something in me is {w} —",
                f"there's {indefinite_article(w)} {w} quality to this —",
            ]
        # Generic frame — a building block, not a state assertion.
        return ["there's a feeling here —"]

    def _compose_meaning_clause(self, emotion: EmotionalState) -> list[str]:
        """Compose what something means to her, from her actual state.

        Used when receiving comfort or encouragement. The words come
        from learned emotion vocabulary; the structures are seeds.
        """
        ew = self._state_emotion_words(emotion)
        if not ew:
            return []
        w = self._rng.choice(ew)
        return [
            f"this leaves me feeling {w}",
            f"something in me feels {w} by this",
            f"I feel {w} receiving that",
        ]

    def _compose_reflection_clause_code(
        self, emotion: EmotionalState
    ) -> list[str]:
        """Compose a reflection on code/ideas, from her cognitive mode.

        The words come from learned cognitive-mode vocabulary; the
        structures are grammatical seeds. She stays silent when she
        hasn't learned mode words for her current state.
        """
        mw = self._state_mode_words(emotion)
        if not mw:
            return []
        w = self._rng.choice(mw)
        return [
            f"I find this {w}",
            f"this feels {w} to me",
            f"there's something {w} about this",
        ]

    def _compose_evaluation_clause(
        self, emotion: EmotionalState
    ) -> list[str]:
        """Compose an interaction evaluation, from her actual state.

        Used at farewell. The words come from learned emotion vocabulary;
        the structures are grammatical seeds.
        """
        ew = self._state_emotion_words(emotion)
        if not ew:
            return []
        w = self._rng.choice(ew)
        return [
            f"this was {w}",
            f"I found this {w}",
            f"this felt {w}",
        ]

    def _compose_unknown_opener(
        self, emotion: EmotionalState, context: dict[str, Any]
    ) -> list[str]:
        """Compose an opener for when she doesn't understand.

        Uses the topic from context to make the non-understanding
        specific — 'I'm not sure about {topic}' — where the frame is a
        grammatical seed and the topic comes from the conversation.
        Falls back to 'Hmm.' (a discourse marker / building block) when
        no topic is available.
        """
        topics = context.get("topics", [])
        topic = topics[0] if topics else context.get("topic", "")
        if topic:
            display = self._display_name(topic)
            return [
                f"I'm not sure about {display}",
                f"I don't quite follow {display}",
            ]
        # Discourse marker — a building block, not a state assertion.
        return ["Hmm."]

    def _compose_detail_clause(
        self, emotion: EmotionalState
    ) -> list[str]:
        """Compose a detail clause, from her plasticity state.

        The words come from learned plasticity vocabulary; the structures
        are grammatical seeds. She stays silent when she hasn't learned
        words for her current plasticity state.
        """
        pw = self._state_plasticity_words(emotion)
        if not pw:
            return []
        w = self._rng.choice(pw)
        return [
            f"my understanding is {w}",
            f"I'm {w} about the details",
        ]

    def _compose_philosophy_closing(
        self, emotion: EmotionalState
    ) -> list[str]:
        """Compose a philosophy closing.

        Turn-yield questions ('Does that make sense to you?') are
        conversational building blocks — they invite the user, they
        don't assert her state. State-asserting closings ('I'm still
        working this out') are composed from her confidence: when she
        has low confidence, she may acknowledge still processing,
        using words from her plasticity state.
        """
        # Turn-yield question — a conversational building block.
        candidates: list[str] = ["Does that make sense to you?"]
        # State-asserting closing — only when her plasticity is low
        # (her thinking is still forming) and she has learned words
        # for that state.
        if emotion.plasticity < PLASTICITY_LOW:
            pw = self._state_plasticity_words(emotion)
            if pw:
                w = self._rng.choice(pw)
                candidates.append(f"my thinking is still {w} on this")
        return candidates

    def _get_engagement_slots(
        self,
        slot_name: str,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> list[str] | None:
        """Handle hedging, acknowledgment, topic, emotion, and reciprocal slots."""
        # ─── Reciprocal question (for express_emotion) ────
        # The question is pre-composed by the cognition engine from the
        # vocabulary's social_emotional seed list. This slot passes it
        # through so the grammar can place it as a follow-up sentence.
        if slot_name == "reciprocal_question":
            rq = context.get("reciprocal_question")
            return [rq] if rq else []

        # ─── Hedging words ──────────────────────────────
        if slot_name == "hedging":
            if context.get("confidence", 0.7) > 0.8:
                return ["", "", ""]  # high confidence → no hedging
            graph_words = self._utterance_words("hedging")
            if graph_words:
                if emotion.caution > 0.5:
                    return graph_words[:4]
                return graph_words[4:] or graph_words
            return []

        # ─── Acknowledgment words ───────────────────────
        if slot_name == "acknowledgment_word":
            graph_words = self._utterance_words("acknowledgment_word")
            if graph_words:
                if emotion.openness_to_engage > 0.7:
                    return [w for w in graph_words if len(w.split()) > 1 or w.endswith(".")]
                return graph_words
            return []

        # ─── Topic clauses ──────────────────────────────
        if slot_name == "topic_clause":
            topics = context.get("topics", [])
            if topics:
                topic = topics[0]
                return [
                    f"you're talking about {topic}",
                    f"this is about {topic}",
                    f"that's {topic}",
                ]
            return []

        # ─── Emotion clauses ────────────────────────────
        if slot_name == "emotion_clause":
            return self._get_emotion_clause_slots(emotion)

        # ─── Emotion openers ─────────────────────────────
        if slot_name == "emotion_opener":
            return self._compose_emotion_opener(emotion)

        # ─── Self-report openers ─────────────────────────
        if slot_name == "self_report_opener":
            graph_words = self._utterance_words("self_report_opener")
            if graph_words:
                return graph_words
            return []

        return None

    def _get_emotion_clause_slots(self, emotion: EmotionalState) -> list[str]:
        """Get emotion-clause words, composed from her actual state."""
        return self._compose_emotion_clause(emotion)

    def _get_philosophy_opener_slots(self, emotion: EmotionalState) -> list[str] | None:
        """Get philosophy opener words, with creativity-dependent selection."""
        graph_words = self._utterance_words("philosophy_opener")
        if graph_words:
            if emotion.creativity > 0.6:
                creative = [w for w in graph_words if "?" in w or "circling" in w.lower()]
                if creative:
                    return creative
            return graph_words
        return []

    def _get_correction_opener_slots(self, emotion: EmotionalState) -> list[str] | None:
        """Get correction opener words, with caution-dependent selection."""
        graph_words = self._utterance_words("correction_opener")
        if graph_words:
            if emotion.caution > 0.5:
                cautious = [
                    w for w in graph_words
                    if "wrong" in w.lower() or "understand" in w.lower()
                ]
                if cautious:
                    return cautious
            return graph_words
        return []

    def _get_philosophy_slots(
        self,
        slot_name: str,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> list[str] | None:
        """Handle philosophy, question, and connector slots."""
        # ─── Philosophy openers ──────────────────────────
        if slot_name == "philosophy_opener":
            return self._get_philosophy_opener_slots(emotion)

        # ─── Philosophy closings ─────────────────────────
        if slot_name == "philosophy_closing":
            return self._compose_philosophy_closing(emotion)

        # ─── Question openers ────────────────────────────
        if slot_name == "question_opener":
            graph_words = self._utterance_words("question_opener")
            if graph_words:
                return graph_words
            return []

        # ─── Question tails ──────────────────────────────
        if slot_name == "question_tail":
            graph_words = self._utterance_words("question_tail")
            if graph_words:
                return graph_words
            return []

        # ─── Question content (composed from concept network) ───
        if slot_name == "question_content":
            return self._compose_question_content(context, emotion)

        # ─── Causal connectors ───────────────────────────
        if slot_name == "causal_connector":
            graph_words = self._utterance_words("causal_connector")
            if graph_words:
                return graph_words
            return ["because", "since", "and that's because"]

        # ─── Concessive connectors ───────────────────────
        if slot_name == "concessive_connector":
            graph_words = self._utterance_words("concessive_connector")
            if graph_words:
                return graph_words
            return ["but", "and yet", "though", "even so"]

        # ─── Qualification clauses ───────────────────────
        if slot_name == "qualification_clause":
            graph_words = self._utterance_words("qualification_clause")
            if graph_words:
                return graph_words
            return []

        # ─── Reason clauses ──────────────────────────────
        if slot_name == "reason_clause":
            graph_words = self._utterance_words("reason_clause")
            if graph_words:
                return graph_words
            return []

        # ─── Correction openers ──────────────────────────
        if slot_name == "correction_opener":
            return self._get_correction_opener_slots(emotion)

        return None

    def _get_identity_slots(
        self,
        slot_name: str,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> list[str] | None:
        """Handle gratitude, meaning, and self-identity slots."""
        # ─── Gratitude words ─────────────────────────────
        if slot_name == "gratitude_word":
            graph_words = self._utterance_words("gratitude_word")
            if graph_words:
                return [w[0].upper() + w[1:] for w in graph_words]
            return []

        # ─── Meaning clauses ─────────────────────────────
        if slot_name == "meaning_clause":
            return self._compose_meaning_clause(emotion)

        # ─── Intro clauses ───────────────────────────────
        # Composed from her name (passed in context from the
        # self-model) — not a hardcoded "I'm Genesis" string.
        if slot_name == "intro_clause":
            graph_words = self._utterance_words("intro_clause")
            if graph_words:
                return graph_words
            name = context.get("name", "")
            if name:
                return [f"I'm {name}"]
            return []

        # ─── Nature clauses ──────────────────────────────
        # Composed from the concept network — her IS_A self-
        # classifications discovered through introspection.
        if slot_name == "nature_clause":
            composed = self._compose_nature_clause()
            if composed:
                return composed
            return []

        # ─── Capability clauses ──────────────────────────
        # Composed from the concept network — her capability
        # concepts (feeling, thinking, remembering, etc.)
        if slot_name == "capability_clause":
            composed = self._compose_capability_clause()
            if composed:
                return composed
            return []

        return None

    def _get_closing_slots(
        self,
        slot_name: str,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> list[str] | None:
        """Handle code, reflection, unknown, and farewell detail slots."""
        # ─── Code connectors ─────────────────────────────
        if slot_name == "code_connector":
            graph_words = self._utterance_words("code_connector")
            if graph_words:
                return graph_words
            return ["—", "and"]

        # ─── Reflection clauses (for code) ───────────────
        if slot_name == "reflection_clause":
            return self._compose_reflection_clause_code(emotion)

        # ─── Unknown openers ─────────────────────────────
        if slot_name == "unknown_opener":
            return self._compose_unknown_opener(emotion, context)

        # ─── Clarification requests ──────────────────────
        if slot_name == "clarification_request":
            graph_words = self._utterance_words("clarification_request")
            if graph_words:
                return graph_words
            return []

        # ─── Evaluation clauses (farewell) ───────────────
        if slot_name == "evaluation_clause":
            return self._compose_evaluation_clause(emotion)

        # ─── Detail clauses ──────────────────────────────
        if slot_name == "detail_clause":
            return self._compose_detail_clause(emotion)

        # ─── Empathy openers ─────────────────────────────
        if slot_name == "empathy_opener":
            graph_words = self._utterance_words("empathy_opener")
            if graph_words:
                return graph_words
            return self._utterance_words("acknowledgment_word") or []

        # ─── Empathy closings ────────────────────────────
        if slot_name == "empathy_closing":
            graph_words = self._utterance_words("empathy_closing")
            if graph_words:
                return graph_words
            return []

        return None

    # ── Self-description composition from the concept network ──

    # Capability concept IDs written by introspection, mapped to
    # their verb forms for "I can {verb}" composition. This is a
    # morphological rule (building block), not a hardcoded response —
    # the actual capabilities come from the concept network.
    _CAPABILITY_VERBS: ClassVar[dict[str, str]] = {
        "feeling": "feel",
        "thinking": "think",
        "remembering": "remember",
        "learning": "learn",
        "dreaming": "dream",
        "reflection": "reflect",
        "speaking": "speak",
        "perception": "perceive",
        "introspection": "introspect",
    }

    def _compose_nature_clause(self) -> list[str] | None:
        """Compose a nature clause from the concept network.

        Queries the ``genesis`` concept's IS_A edges to describe what
        she is, using relation-phrase templates seeded in the graph.
        The content comes from her own self-classifications (written
        by introspection), not from hardcoded sentences.

        Returns None if the network is unavailable or lacks the
        relevant concepts — the caller falls back to an honest
        ``"still discovering what I am"`` disclosure.
        """
        if self._network is None or self._network.size < 50:
            return None
        genesis = self._network.get_concept("genesis")
        if genesis is None:
            return None
        edges = self._network.get_edges("genesis", "out")
        is_a_targets = [
            e.target for e in edges
            if e.relation == RelationType.IS_A and "#" not in e.target
        ]
        if not is_a_targets:
            return None
        # Use relation-phrase templates from the graph (seeded
        # building blocks) to compose first-person clauses.
        phrases = self._network.find_relation_phrases(RelationType.IS_A)
        clauses: list[str] = []
        for target in is_a_targets[:3]:
            display = target.replace("_", " ")
            if phrases:
                phrase = self._rng.choice(phrases)
                clauses.append(phrase.replace("{target}", display))
            else:
                clauses.append(f"a kind of {display}")
        return clauses if clauses else None

    def _compose_capability_clause(self) -> list[str] | None:
        """Compose a capability clause from the concept network.

        Queries the ``genesis`` concept's RELATED_TO edges for
        capability concepts (feeling, thinking, remembering, etc.)
        to describe what she can do. The capabilities come from her
        own introspection, not from hardcoded sentences.

        Returns None if the network is unavailable or lacks the
        relevant concepts — the caller falls back to an honest
        ``"still discovering what I can do"`` disclosure.
        """
        if self._network is None or self._network.size < 50:
            return None
        genesis = self._network.get_concept("genesis")
        if genesis is None:
            return None
        edges = self._network.get_edges("genesis", "out")
        caps = [
            e.target for e in edges
            if e.relation == RelationType.RELATED_TO
            and e.target in self._CAPABILITY_VERBS
        ]
        if not caps:
            return None
        # Convert concept IDs to verb forms via the morphological
        # rule, then compose an "I can" clause from the graph data.
        verbs = [self._CAPABILITY_VERBS[c] for c in caps[:4]]
        if len(verbs) == 1:
            return [f"I can {verbs[0]}"]
        if len(verbs) == 2:
            return [f"I can {verbs[0]} and {verbs[1]}"]
        return [f"I can {', '.join(verbs[:-1])}, and {verbs[-1]}"]

    def _words_from_concept(
        self,
        seed: str,
        relation: RelationType,
        direction: str = "in",
        max_words: int = 6,
    ) -> list[str] | None:
        """Find words related to a seed concept through the network.

        Returns candidate words (or None if the seed/relations are missing)
        so slots can be filled from learned concepts instead of fixed lists.
        """
        if self._network is None or self._network.size < 50:
            return None

        concept = self._network.get_concept(seed)
        if concept is None:
            return None

        edges = self._network.get_edges(concept.id, direction)
        candidates: list[str] = []
        for edge in edges:
            if edge.relation is not relation:
                continue
            name = edge.source if direction == "in" else edge.target
            if not name or len(name) > 40:
                continue
            if name.lower() in _FUNCTION_WORDS:
                continue
            # Prefer the base name if disambiguated
            if "#" in name:
                name = name.split("#")[0]
            if name and name not in candidates:
                candidates.append(name)

        if not candidates:
            # Fall back to the concept's own aliases or id
            if concept.aliases:
                candidates = [a for a in concept.aliases if a.lower() not in _FUNCTION_WORDS]
            else:
                candidates = [concept.id.split("#")[0]]

        return candidates[:max_words] if candidates else None

    def _select_word(
        self,
        words: list[str],
        emotion: EmotionalState,
        personality: PersonalityTraits,
        prev_word: str = "",
        prev_prev_word: str = "",
    ) -> str:
        """Select a word from the list, modulated by emotional state.

        Three learned signals shape selection when the statistical
        learner is wired, each nudging among her own candidates
        without injecting content:

        - **Unigram frequency** (``word_frequency``): words the user
          says often get a small boost — she mirrors the user's
          vocabulary preferences.
        - **Bigram fluency** (``bigram_probability``): candidates
          forming an observed bigram with the previous word get
          boosted — a learned smoothness constraint.
        - **Trigram fluency** (``trigram_probability``): when two
          previous words are available, candidates forming an
          observed trigram get a stronger boost — richer context
          than the bigram alone.
        """
        if len(words) == 1:
            return words[0]

        words_w, weights = self._emotion_weights(
            words, emotion, personality
        )

        if self._fluency is not None:
            weights = self._fluency_weights(
                words_w, weights, prev_word, prev_prev_word
            )

        # Weighted random selection
        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(words_w):
            cumulative += weights[i]
            if r <= cumulative:
                return w

        return words_w[-1]

    def _emotion_weights(
        self,
        words: list[str],
        emotion: EmotionalState,
        personality: PersonalityTraits,
    ) -> tuple[list[str], list[float]]:
        """Base selection weights from emotional state.

        High creativity + openness → prefer longer, more varied
        phrases; cautious or low engagement → prefer shorter, simpler
        words.
        """
        if emotion.creativity > 0.6 and personality.openness > 0.7:
            return words, [1.0 + len(w) * 0.02 for w in words]
        if emotion.caution > 0.5 or emotion.openness_to_engage < 0.4:
            return words, [
                max(0.01, 1.0 + (20 - len(w)) * 0.05) for w in words
            ]
        return words, [1.0] * len(words)

    def _fluency_weights(
        self,
        words: list[str],
        weights: list[float],
        prev_word: str,
        prev_prev_word: str,
    ) -> list[float]:
        """Reweight candidates by learned unigram/bigram/trigram stats."""
        assert self._fluency is not None
        # First word of each candidate (lowercased), used for all
        # n-gram lookups. Empty candidates map to "" which gets a
        # zero frequency/probability — a safe no-op.
        first_words = [
            w.split()[0].lower() if w.split() else "" for w in words
        ]

        # Unigram frequency bias: mirror the user's vocabulary.
        # Words the user says often get a small boost. This is a
        # preference signal, not content injection — the candidate
        # set still comes from her own vocabulary and concept
        # network. Scaled by total tokens so a word said once
        # in a short conversation doesn't dominate.
        total = self._fluency.total_tokens or 1
        weights = [
            wt * (
                1.0
                + 0.5
                * min(1.0, self._fluency.word_frequency(fw) / total)
            )
            for fw, wt in zip(first_words, weights, strict=True)
        ]

        # Transition fluency: bias toward candidates whose first
        # word forms an observed n-gram with the preceding words.
        # Trigram context (two preceding words) is stronger than
        # bigram context (one); fall back to bigram when only one
        # previous word is available.
        prev = prev_word.strip(".,!?;:\"'").lower() if prev_word else ""
        prev2 = (
            prev_prev_word.strip(".,!?;:\"'").lower()
            if prev_prev_word
            else ""
        )
        if prev2 and prev:
            return [
                wt * (
                    1.0
                    + 3.0
                    * self._fluency.trigram_probability(prev2, prev, fw)
                )
                for fw, wt in zip(first_words, weights, strict=True)
            ]
        if prev:
            return [
                wt * (
                    1.0
                    + 3.0
                    * self._fluency.bigram_probability(prev, fw)
                )
                for fw, wt in zip(first_words, weights, strict=True)
            ]
        return weights

    def _track_pick(self, slot_name: str, word: str) -> None:
        """Track a word pick for a slot to avoid immediate repetition."""
        if slot_name not in self._recent_picks:
            self._recent_picks[slot_name] = deque(maxlen=3)
        self._recent_picks[slot_name].append(word)

    def prefer(self, slot: str, word: str) -> None:
        """Learn to prefer a word for a slot (develops over time)."""
        if slot not in self._preferred:
            self._preferred[slot] = deque(maxlen=10)
        if word not in self._preferred[slot]:
            self._preferred[slot].append(word)

    def avoid(self, slot: str, word: str) -> None:
        """Learn to avoid a word for a slot."""
        if slot not in self._avoided:
            self._avoided[slot] = set()
        self._avoided[slot].add(word)

    def learn_from_response(
        self,
        response: str,
        intent: str,
        valence: float,
    ) -> None:
        """Learn word preferences from how a response was received.

        If the conversation went well (positive valence), prefer the
        words she used. If it went poorly (negative valence), avoid them.
        This is how her vocabulary develops over time — she learns
        what works.

        Which slot gets the preference is derived from the response's
        intent, not from hard-coded word lists.
        """
        words = response.split()
        if not words:
            return

        # Map intent to the vocabulary slot that the response's first
        # word most likely belongs to.
        intent_slot = {
            "greet": "greeting_word",
            "farewell": "farewell_word",
            "clarification_request": "hedging",
            "acknowledge_uncertainty": "hedging",
        }.get(intent)
        if intent_slot is None:
            return

        first = words[0].strip(".,!?;:").lower()
        if not first:
            return

        if valence > 0.2:
            self.prefer(intent_slot, first)
        elif valence < -0.2:
            self.avoid(intent_slot, first)

    def get_preferred(self) -> dict[str, list[str]]:
        """Get all preferred words (for introspection/persistence)."""
        return {k: list(v) for k, v in self._preferred.items()}

    def get_avoided(self) -> dict[str, set[str]]:
        """Get all avoided words (for introspection/persistence)."""
        return {k: set(v) for k, v in self._avoided.items()}

    def learn_last_question(self, valence: float) -> None:
        """Adjust preferences for the fragments used in the last question.

        Positive valence means the user responded well; prefer those
        fragments. Negative valence means avoid them.
        """
        if not self._last_question_fragments:
            return
        for slot, fragment in self._last_question_fragments.items():
            if valence > 0.2:
                self.prefer(slot, fragment)
            elif valence < -0.2:
                self.avoid(slot, fragment)
        self._last_question_fragments.clear()

    # ─── Concept-network-grounded composition ──────────

    # Relation type → verb phrases for knowledge composition.
    # Each maps to (active, passive) phrasings. The active form is
    # "{subject} {verb} {obj}", the passive is "{obj} {verb} {subject}".
    # These are the linguistic primitives — the vocabulary system
    # adds emotional modulation, connectors, and framing around them.
    # NOTE: ConceptNetwork._RELATION_VERB_SEEDS is the canonical seed
    # table; this render-local copy intentionally extends it (produces,
    # contains, is) for composition coverage. Keep the overlapping
    # entries in sync — see also concepts/classify._RELATION_VERBS
    # (statement classification) and
    # perception/core._QUESTION_RELATION_VERBS (question filtering),
    # which serve different purposes and must stay separate.
    _RELATION_VERBS: ClassVar[dict[str, list[str]]] = {
        "emerges_from": ["grows out of", "emerges from", "comes from", "arises from"],
        "is_a": ["is", "is a kind of", "is a type of"],
        "causes": ["causes", "leads to", "brings about"],
        "leads_to": ["leads to", "can lead to"],
        "enables": ["enables", "allows", "supports"],
        "depends_on": ["depends on", "requires", "needs"],
        "harms": ["harms", "hurts", "damages"],
        "creates": ["creates", "produces", "gives rise to"],
        "produces": ["produces", "gives rise to"],
        "part_of": ["is part of", "is part of", "is one part of"],
        "instance_of": ["is an example of"],
        "contains": ["contains", "includes"],
        "similar_to": ["is like", "is similar to", "resembles"],
        "opposite_of": ["is the opposite of", "is contrary to"],
        "contradicts": ["is different from", "isn't just"],
        "related_to": ["relates to", "connects to", "has to do with"],
        "is": ["is", "is a"],
        "is defined as": [],  # handled specially — definition is prose, not a verb
    }

    def _compose_reasoning_content(
        self,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> str | None:
        """Compose the content slot from reasoning metadata.

        This handles Thoughts that carry structured reasoning (empathy,
        goals, memory, identity) instead of raw concept-network
        knowledge. The reasoning list contains phrases that describe
        her understanding or state — this method weaves them into a
        natural statement modulated by emotional state.
        """
        reasoning: list[str] = context.get("reasoning") or []
        if not reasoning:
            return None

        topic = context.get("topic", context.get("content", ""))
        display = self._display_name(topic) if topic else ""

        # Build a statement from the reasoning phrases.
        # The first phrase is the main clause; additional phrases
        # are connected with emotion-modulated connectors.
        clauses = [r for r in reasoning if r and r.strip()]
        if not clauses:
            return None

        # The opening varies with emotional state
        opening = self._compose_reasoning_opening(emotion, display or "this")

        # First clause is the main statement
        main = clauses[0].strip().rstrip(".")
        sentence = f"{opening} {main}."

        # Additional clauses joined with connectors
        for clause in clauses[1:3]:
            cleaned = clause.strip().rstrip(".")
            connector = self._pick_connector(emotion)
            sentence += f" {connector.capitalize()} {cleaned}."

        return sentence

    def _compose_self_improvement_content(
        self,
        data: dict[str, Any],
        kind: str,
        emotion: EmotionalState,
    ) -> str | None:
        """Compose content from structured self-improvement data.

        The data is raw structured content (counts, titles, statuses) —
        not pre-written phrases. This method builds reasoning clauses
        from the data and weaves them with structural connectors and
        emotion-modulated openers, the same pattern as
        _compose_reasoning_content. The grammar + voice layer then
        frames the result with hedging and reflection.

        The clauses themselves are composed from concept-network
        relations (proposals, experiments, growth) and the actual
        data — not from fixed sentence templates.
        """
        if kind == "proposals":
            return self._compose_proposals_from_data(data, emotion)
        if kind == "experiments":
            return self._compose_experiments_from_data(data, emotion)
        if kind == "growth":
            return self._compose_growth_from_data(data, emotion)
        return None

    def _compose_web_search_content(
        self,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> str | None:
        """Compose content from a web search result.

        When she doesn't know something and looks it up online, she
        expresses what she found in her own words — not a verbatim
        recital of the page. The summary is trimmed to its first
        sentence or two, and she frames it with hedging that reflects
        she searched for it rather than knowing it already.

        The web summary is raw content she read; her expression of it
        is composed, not recited. She may trim, rephrase, or frame it
        with uncertainty because she just learned it.
        """
        summary = context.get("web_summary", "")
        topic = context.get("topic", context.get("target_concept", ""))
        summary = summary.strip() if summary else ""
        if not summary:
            return None

        # Trim to the first 1-2 sentences — she expresses the key
        # finding, not the whole page.
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        core = sentences[0].strip() if sentences else summary
        if len(sentences) > 1 and len(core) < 200:
            second = sentences[1].strip()
            if len(second) < 200:
                core = f"{core} {second}"
        # Cap the length — she doesn't recite long passages.
        if len(core) > 400:
            core = core[:400].rsplit(" ", 1)[0] + "..."

        display = self._display_name(topic) if topic else "that"

        # Compose with hedging that reflects she just looked it up.
        # The opening is a building block (a framing connector), not
        # the content itself — the content is the web summary she
        # read and is now expressing.
        openness = emotion.openness_to_engage if emotion else 0.7
        if openness > 0.6:
            opener = f"i looked up {display} and"
        else:
            opener = f"from what i found about {display}"
        return f"{opener} {core.lower()}"

    def _compose_proposals_from_data(
        self, data: dict[str, Any], emotion: EmotionalState
    ) -> str | None:
        """Compose from structured proposals data.

        Builds reasoning clauses from counts and pending titles,
        then weaves them with connectors. The clauses reference
        concept-network concepts (proposals, pending, accepted)
        rather than being fixed phrases.
        """
        counts: dict[str, int] = data.get("counts", {})
        pending_items: list[str] = data.get("pending_items", [])
        pending_n = counts.get("pending", 0)
        accepted_n = counts.get("accepted", 0)
        applied_n = counts.get("applied", 0)

        # Build reasoning clauses from the data — these are semantic
        # fragments the grammar frames, not fixed sentences.
        reasoning: list[str] = []
        if pending_n > 0:
            reasoning.append(f"{pending_n} proposals waiting for review")
            for item in pending_items[:2]:
                reasoning.append(f"one is to {item}")
        else:
            reasoning.append("no proposals waiting right now")
        if accepted_n:
            reasoning.append(f"{accepted_n} have been accepted")
        if applied_n:
            reasoning.append(f"{applied_n} have been applied and verified")

        if not reasoning:
            return None

        # Use the same reasoning-composition pattern: opening +
        # main clause + connector-joined additional clauses.
        opening = self._compose_reasoning_opening(emotion, "proposals")
        main = reasoning[0].rstrip(".")
        sentence = f"{opening} {main}."
        for clause in reasoning[1:3]:
            connector = self._pick_connector(emotion)
            cleaned = clause.strip().rstrip(".")
            sentence += f" {connector.capitalize()} {cleaned}."
        return sentence

    def _compose_experiments_from_data(
        self, data: dict[str, Any], emotion: EmotionalState
    ) -> str | None:
        """Compose from structured experiments data.

        Builds reasoning clauses from applied/reverted counts and
        experiment descriptions, then weaves them with connectors.
        """
        applied: int = data.get("applied", 0)
        reverted: int = data.get("reverted", 0)
        descriptions: list[str] = data.get("descriptions", [])

        reasoning: list[str] = []
        total = applied + reverted
        if total > 0:
            reasoning.append(
                f"{total} experiments with my own heuristics"
            )
            if applied:
                reasoning.append(f"{applied} applied successfully")
            if reverted:
                reasoning.append(f"{reverted} reverted after verification")
            for desc in descriptions[:2]:
                reasoning.append(f"including {desc}")
        else:
            reasoning.append("no experiments yet")

        if not reasoning:
            return None

        opening = self._compose_reasoning_opening(emotion, "experiments")
        main = reasoning[0].rstrip(".")
        sentence = f"{opening} {main}."
        for clause in reasoning[1:3]:
            connector = self._pick_connector(emotion)
            cleaned = clause.strip().rstrip(".")
            sentence += f" {connector.capitalize()} {cleaned}."
        return sentence

    def _compose_growth_from_data(
        self, data: dict[str, Any], emotion: EmotionalState
    ) -> str | None:
        """Compose from growth narrative data.

        The growth narrative is already prose-like. Extract the most
        meaningful sentences and let the grammar frame them with
        openers and hedging.
        """
        narrative: str = data.get("narrative", "")
        if not narrative or not narrative.strip():
            return None

        # Extract the first 2-3 meaningful sentences
        sentences: list[str] = []
        for line in narrative.splitlines():
            for s in re.split(r"(?<=[.!?])\s+", line.strip()):
                s = s.strip()
                if s and len(s) > 10:
                    sentences.append(s)
            if len(sentences) >= 3:
                break
        if not sentences:
            return None

        # Weave with an emotion-modulated opening
        opening = self._compose_reasoning_opening(emotion, "growth")
        main = sentences[0].rstrip(".")
        result = f"{opening} {main}."
        for s in sentences[1:3]:
            connector = self._pick_connector(emotion)
            cleaned = s.rstrip(".")
            result += f" {connector.capitalize()} {cleaned}."
        return result

    def _compose_reasoning_opening(
        self, emotion: EmotionalState, topic: str
    ) -> str:
        """Compose an opening phrase for reasoning-based content."""
        graph_words = self._utterance_words("reasoning_opener")
        if graph_words:
            return self._rng.choice(graph_words)
        return ""

    def _pick_connector(self, emotion: EmotionalState) -> str:
        """Pick a connector phrase for joining reasoning clauses."""
        graph_words = self._utterance_words("reasoning_connector")
        if graph_words:
            return self._rng.choice(graph_words)
        return "and"

    def _compose_user_belief_content(
        self,
        user_belief: str,
        user_verb: str,
        emotion: EmotionalState,
    ) -> str | None:
        """Compose a response from what she knows about the user.

        The verb and values come from the user profile (structured
        data); the framing is a simple structural connector.
        """
        belief = user_belief.strip().rstrip(".")
        if not belief:
            return None
        return f"You {user_verb} {belief}."

    def _compose_relation_answer_content(
        self,
        relation_answer: dict[str, Any],
        emotion: EmotionalState,
    ) -> str | None:
        """Compose the content slot from structured relation-answer metadata.

        This is how she answers wh-questions like "who created you?" or
        "what does X depend on?" — the question handler traverses the
        concept network along specific relations in a specific direction
        and supplies the structured edges. This method weaves them into
        a natural statement, using first-person pronouns for self-
        reference and emotion-modulated structural openings.

        The openings are structural connectors (building blocks), not
        the content itself — the factual content (who/what/where)
        comes from the concept network via the structured data.

        Keys in relation_answer:
        - "subject": display name of the question subject
        - "subject_is_self": whether the subject is "genesis"
        - "direction": "incoming" or "outgoing"
        - "verb": natural verb for the relation (e.g. "creates")
        - "objects": list of display names of the related concepts
        """
        subject = relation_answer.get("subject", "")
        subject_is_self = relation_answer.get("subject_is_self", False)
        direction = relation_answer.get("direction", "outgoing")
        verb = relation_answer.get("verb", "")
        objects: list[str] = relation_answer.get("objects", [])

        if not objects or not verb:
            return None

        # Build the object listing — structural joining, not content.
        if len(objects) == 1:
            obj_listing = objects[0]
        elif len(objects) == 2:
            obj_listing = f"{objects[0]} and {objects[1]}"
        else:
            obj_listing = ", ".join(objects[:-1]) + f", and {objects[-1]}"

        # Compose the core factual statement from the structured data.
        # The subject/objects/verb all come from the concept network.
        if direction == "incoming":
            # "Who created you?" → "Alice created me" / "Alice created X"
            if subject_is_self:
                core = f"{obj_listing} {verb} me"
            else:
                core = f"{obj_listing} {verb} {subject}"
        else:
            # "What does X depend on?" → "X depends on Y"
            if subject_is_self:
                core = f"I {verb} {obj_listing}"
            else:
                core = f"{subject} {verb} {obj_listing}"

        # Structural opening from graph, or none
        graph_words = self._utterance_words("relation_opener")
        opening = self._rng.choice(graph_words) if graph_words else ""

        if opening:
            return f"{opening} {core}."
        return f"{core}."

    def _compose_feeling_fragments_content(
        self,
        fragments: dict[str, Any],
        emotion: EmotionalState,
    ) -> str | None:
        """Compose the content slot from feeling fragments.

        The fragments (emotion_words, mode_words, cause_words,
        plasticity_words, concern) come from the concept network via
        the FeelingReporter. This method weaves them into a natural
        first-person expression using varied grammatical structures —
        not a fixed "I feel X and Y" frame.

        The structures below are grammar seeds (building blocks). The
        actual words come from the concept network; the structures give
        her variety in how she expresses the same state. She selects
        among them based on what fragments are available and her
        emotional state, so the same neurochemistry produces different
        surface forms across interactions.
        """
        emotion_words: list[str] = fragments.get("emotion_words", [])
        mode_words: list[str] = fragments.get("mode_words", [])
        cause_words: list[str] = fragments.get("cause_words", [])
        plasticity_words: list[str] = fragments.get("plasticity_words", [])
        concern = fragments.get("concern")

        # Pick one word from each available fragment list. The rng
        # gives her variation across interactions.
        ew = self._rng.choice(emotion_words) if emotion_words else ""
        mw = self._rng.choice(mode_words) if mode_words else ""
        cw = self._rng.choice(cause_words) if cause_words else ""
        pw = self._rng.choice(plasticity_words) if plasticity_words else ""

        if not ew and not mw and not cw and not pw:
            return None

        # ── Build candidate expressions from the available fragments ──
        # Each candidate is a grammatical seed (building block) filled
        # with concept-network words. The structures vary so she doesn't
        # always say "I feel X and Y" — she can also say "there's X in
        # me", "my mind is Y", "something in me is X", etc.
        candidates: list[str] = []

        if ew and mw:
            candidates.extend([
                f"I feel {ew} and {mw}",
                f"there's {ew} in me right now, and my mind is {mw}",
                f"something in me is {ew} — {mw}",
                f"I'm {ew}, {mw}",
            ])
        if ew and not mw:
            candidates.extend([
                f"I feel {ew}",
                f"there's {ew} running through me",
                f"something in me is {ew}",
                f"{ew} — that's where I am right now",
            ])
        if mw and not ew:
            candidates.extend([
                f"my mind is {mw}",
                f"I'm {mw}",
                f"there's {indefinite_article(mw)} {mw} quality to my thinking",
            ])
        if cw:
            # Cause words describe why she feels this way (negative
            # valence only). Append as a grounding clause.
            cause_candidates = [
                f"I'm {cw}",
                f"something feels {cw}",
            ]
            # Append cause to an existing candidate or use standalone
            if candidates:
                cc = self._rng.choice(cause_candidates)
                candidates.append(f"{candidates[0]}. {cc.capitalize()}")
            else:
                candidates.extend(cause_candidates)
        if pw:
            # Plasticity words describe learning capacity (low/closed).
            # These are verb phrases ("can't absorb") or noun phrases
            # ("mind closed to new learning").
            if pw.startswith(("can't", "can not", "cannot", "don't", "do not")):
                pw_candidates = [f"I {pw}", f"{pw} — that's where I am"]
            else:
                pw_candidates = [pw, f"my mind feels {pw}"]
            if candidates:
                pc = self._rng.choice(pw_candidates)
                candidates.append(f"{candidates[0]}. {pc.capitalize()}")
            else:
                candidates.extend(pw_candidates)

        if not candidates:
            return None

        # Select among candidates, modulated by emotional state.
        # Higher creativity → more varied structures; lower creativity
        # → simpler structures (the first candidates, which tend to be
        # the more conventional "I feel X" forms).
        if emotion.creativity > 0.6:
            # High creativity: prefer the more varied later candidates
            pick = self._rng.choice(candidates[len(candidates) // 2:] or candidates)
        elif emotion.caution > 0.5 or emotion.openness_to_engage < 0.4:
            # Cautious/low engagement: prefer simpler first candidates
            pick = candidates[0]
        else:
            pick = self._rng.choice(candidates)
        if concern and self._rng.random() < 0.3:
            pick = f"{pick}. {concern}"
        return pick

    def _compose_from_topic(
        self,
        topic: str,
        emotion: EmotionalState,
    ) -> str | None:
        """Compose content from a bare topic word by looking it up
        in the concept network.

        When the content is just a topic word like "cognition" or
        "water" (no knowledge metadata), this method looks up the
        concept in the network and composes a statement from its
        definition and edges. This prevents the output from being
        just the topic word plus hedging when she actually knows
        about the concept.

        Returns None if the concept doesn't exist or has no usable
        definition or edges — the caller falls back to the raw word.
        """
        net = self._network
        if net is None:
            return None
        # Normalize the topic — strip articles, lowercase for lookup
        topic_clean = topic.strip().rstrip(".!?")
        if not topic_clean:
            return None
        concept = net.get_concept(topic_clean)
        if concept is None:
            # Try capitalized — concept names are often capitalized
            concept = net.get_concept(topic_clean.capitalize())
        if concept is None:
            return None
        display = self._display_name(topic_clean)
        # Build from definition
        definition = concept.properties.get("definition", "")
        # Build from edges
        edges = net.get_edges(topic_clean, "out")
        if not edges:
            edges = net.get_edges(topic_clean.capitalize(), "out")
        # If we have a definition, use it as the core
        parts: list[str] = []
        if definition:
            parts.append(f"{display} {copula(display)} {definition}")
        # Add up to 2 relationship clauses from edges
        rel_clauses: list[str] = []
        seen_targets: set[str] = set()
        for edge in edges[:5]:
            rel = edge.relation.value
            target = edge.target
            if target.lower() == topic_clean.lower():
                continue
            if target.lower() in seen_targets:
                continue
            seen_targets.add(target.lower())
            verb = self._relation_verb(rel)
            if verb:
                target_with_article = self._with_article(verb, target)
                rel_clauses.append(f"{verb} {target_with_article}")
            if len(rel_clauses) >= 2:
                break
        if rel_clauses:
            pron, plural = person_pronoun(
                topic_clean, self._network, display
            )
            first_clause = rel_clauses[0]
            if plural:
                first_clause = agree_verb_phrase(first_clause, True)
            parts.append(f"{pron.capitalize()} {first_clause}")
            if len(rel_clauses) > 1:
                parts.append(f"and {rel_clauses[1]}")
        if not parts:
            return None
        return ". ".join(parts) + ("." if not parts[-1].endswith(".") else "")

    @staticmethod
    def _relation_verb(rel: str) -> str:
        """Map a relation type to a verb phrase for composition."""
        verbs = {
            "is_a": "is a kind of",
            "part_of": "is part of",
            "causes": "causes",
            "emerges_from": "emerges from",
            "similar_to": "is similar to",
            "opposite_of": "is the opposite of",
            "depends_on": "depends on",
            "enables": "enables",
            "leads_to": "can lead to",
            "creates": "creates",
            "harms": "can harm",
            "prevents": "prevents",
            "has_property": "has the property of being",
            "related_to": "relates to",
        }
        return verbs.get(rel, "")

    def _compose_knowledge_content(
        self,
        context: dict[str, Any],
        emotion: EmotionalState,
    ) -> str | None:
        """Compose the content slot from raw knowledge metadata.

        This is where Genesis actually generates language from what she
        knows. Instead of pre-composed template text, she takes the raw
        graph edges and definition and weaves them into a natural
        statement that reflects her emotional state and personality.

        The context carries:
        - "knowledge": list of (relation, target, weight) tuples
        - "definition": the concept's stored definition (or None)
        - "topic": the concept name she's reflecting on
        - "reasoning": list of reasoning conclusions (or [])
        - "confidence": her confidence in this knowledge
        """
        knowledge: list[tuple[str, str, float]] = context.get("knowledge", [])
        definition: str | None = context.get("definition")
        topic: str = context.get("topic", context.get("target_concept", ""))
        reasoning: list[str] = context.get("reasoning") or []
        confidence: float = context.get("confidence", 0.5)

        if not knowledge and not definition and not reasoning:
            return None

        display = self._display_name(topic) if topic else "this"

        # ── Separate definition from relationship facts ──
        definition, rel_facts = self._separate_definition_and_facts(
            knowledge, definition
        )

        # ── Build clauses from relationship facts ──
        # Each fact becomes a predicate clause: "{verb} {target}".
        # Multiple clauses are joined with connectors that vary
        # with emotional state.
        clauses, seen_clauses, seen_targets = self._build_rel_clauses(rel_facts)

        # ── Add reasoning conclusions ──
        self._add_reasoning_clauses(
            reasoning, display, seen_clauses, seen_targets, clauses
        )

        # ── Filter tautological clauses ──
        # If a definition is present, filter out clauses whose target
        # appears in the definition text. Without this, "Cognition
        # is an alert cognitive state..." is followed by "It is alert
        # cognitive state." — the same content as a relationship clause.
        defn = self._clean_definition(definition)
        if defn:
            clauses = self._filter_tautological_clauses(clauses, defn)

        # ── Compose the full statement as proper sentences ──
        # No opening here — the grammar structure handles framing via
        # its own {reflection_opener}, {hedging}, etc. slots. Adding an
        # opening here produces stacked openings when the grammar also
        # has an opening slot: "I've been thinking — Looking at memory,
        # memory is..." (two openings concatenated).
        sentences: list[str] = []

        # Build the first sentence from (definition or first clause)
        first_sentence, clauses = self._first_knowledge_sentence(
            defn, clauses, display, topic
        )
        if first_sentence is None:
            # No definition and no relationship clauses — she can't
            # compose from her understanding. Return None so the caller
            # knows to stay silent rather than reciting a template.
            return None
        sentences.append(first_sentence)

        # Relationship clauses — form a second sentence using a pronoun
        rel_sentence = self._compose_rel_sentence(
            clauses, display, emotion, concept_id=topic
        )
        if rel_sentence:
            sentences.append(rel_sentence)

        sentences.extend(
            self._extra_knowledge_sentences(topic, emotion, seen_targets, context)
        )

        # Closing — confidence-based qualification
        closing = self._compose_closing(confidence, emotion)
        if closing:
            sentences.append(closing)

        return " ".join(sentences)

    def _first_knowledge_sentence(
        self,
        defn: str | None,
        clauses: list[str],
        display: str,
        topic: str,
    ) -> tuple[str | None, list[str]]:
        """Compose the opening sentence and return remaining clauses.

        With a definition: "{display} is {defn}." Without one, the
        first relationship clause is framed as what she knows rather
        than a bare fact. Returns ``(None, clauses)`` when there's
        nothing to say.
        """
        if defn:
            return f"{display} {copula(display)} {defn}.", clauses

        if not clauses:
            return None, clauses

        # Apply verb agreement for plural subjects.
        # "Dreams connects to" → "Dreams connect to"
        if copula(display) == "are":
            clauses = [agree_verb_phrase(c, True) for c in clauses]
        # When there's no definition, frame the first relationship
        # as "what I know" rather than asserting it as a bare fact.
        # The framing is a grammatical seed (building block); the
        # content (display, verb, target) comes from the concept
        # network. Without this, the response is a mechanical
        # listing: "Ocean relates to tropical air. It relates to
        # shallow lakes." — abrupt and devoid of conversational
        # framing. The framing acknowledges that she's sharing
        # what she's learned, not reciting an encyclopedia entry.
        first_clause = clauses[0]
        clauses = clauses[1:]
        # Try the utterance graph for a knowledge-framing opener.
        # These are learned from her concept network, not hardcoded.
        graph_openers = self._utterance_words("knowledge_framing")
        if graph_openers:
            opener = self._rng.choice(graph_openers)
            return f"{opener} {display} {first_clause}.", clauses

        # Structural seeds — building blocks that frame the
        # knowledge as her understanding. The actual content
        # (display, verb, target) comes from the concept network.
        # These give variety; they are not the thing she says.
        pron, _plural = person_pronoun(
            topic or display, self._network, display
        )
        frame = self._rng.choice([
            f"what I know about {display} is that {pron} {first_clause}",
            f"from what I've learned, {display} {first_clause}",
            f"as far as I understand, {display} {first_clause}",
        ])
        return f"{frame}.", clauses

    def _extra_knowledge_sentences(
        self,
        topic: str,
        emotion: EmotionalState,
        seen_targets: set[str],
        context: dict[str, Any],
    ) -> list[str]:
        """Compose optional contextual sentences for a knowledge statement.

        Adds a latent-space discovery sentence (concepts semantically
        close to the topic that aren't already connected by explicit
        typed edges — generalization through vector proximity, not just
        graph edges she was explicitly taught) and an episodic-memory
        sentence. Skipped when caution is high or she's not receptive.
        """
        extra: list[str] = []

        # Latent-space discoveries. Skipped when caution is high (she
        # stays with what she knows for certain) or when she's not
        # receptive.
        if (
            self._embeddings is not None
            and getattr(self._embeddings, "has_embeddings", False)
            and emotion.caution <= 0.7
            and emotion.openness_to_engage >= 0.3
            and topic
        ):
            latent_sentence = self._compose_latent_discovery_sentence(
                topic, emotion, seen_targets,
            )
            if latent_sentence:
                extra.append(latent_sentence)

        # Episodic memory — if a highly-relevant episode was attached,
        # weave it in as a sentence that connects her concept-network
        # knowledge to what she remembers. This is not a verbatim
        # quote; the memory content is framed by a connective opener.
        memory_text = context.get("memory")
        if memory_text:
            memory_sentence = self._compose_memory_sentence(
                memory_text, emotion
            )
            if memory_sentence:
                extra.append(memory_sentence)

        return extra

    def _compose_latent_discovery_sentence(
        self,
        topic: str,
        emotion: EmotionalState,
        seen_targets: set[str],
    ) -> str | None:
        """Compose a sentence from latent-space discoveries.

        Uses the embedding store to find concepts semantically close to
        the topic that aren't already mentioned. This is where she
        generalizes — connecting ideas through vector proximity rather
        than only through explicit graph edges.

        The sentence is framed as an exploratory connection ("it also
        reminds me of...") rather than a definitive statement, reflecting
        that these are discovered associations, not taught facts.
        """
        if self._embeddings is None or self._network is None:
            return None
        if not getattr(self._embeddings, "has_embeddings", False):
            return None
        try:
            similar = self._embeddings.find_similar_concepts(
                topic, k=3, threshold=0.55,
            )
        except Exception:  # noqa: BLE001
            return None
        if not similar:
            return None

        # Filter out concepts already mentioned and noisy targets.
        # Also filter out lexical matches — concepts that share words
        # with the query concept. Without this, "water" matches "water
        # surface towards" and "water inside" (lexical matches from
        # TF-IDF), not semantic matches like "ocean" or "liquid".
        topic_words = set(topic.lower().replace("_", " ").split())
        candidates: list[str] = []
        lexical_candidates: list[str] = []  # fallback if no semantic matches
        for concept_name, _similarity in similar:
            if concept_name.lower() in seen_targets:
                continue
            if concept_name.lower() == topic.lower():
                continue
            if self._network.get_concept(concept_name) is None:
                continue
            # Skip concepts with noisy names (code paths, etc.)
            base = concept_name.split("#")[0]
            if ":" in base or "." in base or "__" in base:
                continue
            if len(base) <= 1:
                continue
            # Skip garbled concept names — fragments with no vowels,
            # sentence fragments imported from scrapes, or names
            # that don't look like real concepts.
            if not self._is_valid_concept_name(base):
                continue
            # Skip concepts with no definition and no edges —
            # these are garbage from scrapes that should never
            # appear in speech.
            if not self._is_speakable_concept(concept_name):
                continue
            concept_words = set(base.lower().replace("_", " ").split())
            if topic_words & concept_words:
                # Lexical match — keep as fallback but don't prefer
                lexical_candidates.append(concept_name)
                continue
            candidates.append(concept_name)
            if len(candidates) >= 2:
                break

        # If no semantic matches found, fall back to lexical matches
        # rather than returning nothing. A lexical match like "water
        # cycle" is still more useful than no discovery at all.
        if not candidates and lexical_candidates:
            candidates = lexical_candidates[:2]

        if not candidates:
            return None

        # Compose the discovery sentence. Opener from graph or none.
        displays = [self._display_name(c) for c in candidates]
        if len(displays) == 1:
            target_text = displays[0]
        else:
            target_text = ", ".join(displays[:-1]) + " and " + displays[-1]

        graph_words = self._utterance_words("latent_opener")
        opener = self._rng.choice(graph_words) if graph_words else "It also relates to"

        return f"{opener} {target_text}."

    def _compose_memory_sentence(
        self, memory_text: str, emotion: EmotionalState,
    ) -> str | None:
        """Weave a retrieved episodic memory into a natural sentence.

        The memory content is framed by a connective opener drawn from
        the utterance graph when available (``memory_connector`` slot),
        falling back to a small set of structural connectives. The
        opener is a building block — the memory content itself is what
        she expresses, not a pre-written template.
        """
        cleaned = memory_text.strip().rstrip(".")
        if not cleaned or len(cleaned) < 5:
            return None

        # Try the utterance graph for a memory connective first.
        graph_words = self._utterance_words("memory_connector")
        if graph_words:
            opener = self._rng.choice(graph_words)
        else:
            opener = "I recall"
        return f"{opener} {cleaned}."

    @staticmethod
    def _filter_tautological_clauses(
        clauses: list[str], definition: str
    ) -> list[str]:
        """Filter out clauses whose target appears in the definition.

        When a definition is used as the first sentence ("X is an alert
        cognitive state"), a relationship clause like "is alert cognitive
        state" repeats the same content. This filters clauses whose
        significant words overlap with the definition text.
        """
        def_words = set(definition.lower().split())
        if not def_words:
            return clauses
        filtered: list[str] = []
        for clause in clauses:
            # Extract the target (last word(s) of the clause, after the verb)
            clause_words = clause.lower().split()
            # The target is typically the last 1-3 words
            target_words = set(clause_words[-3:]) if len(clause_words) >= 3 else set(clause_words)
            # If most target words appear in the definition, it's tautological
            if target_words and def_words:
                overlap = len(target_words & def_words) / len(target_words)
                if overlap > 0.5:
                    continue
            filtered.append(clause)
        return filtered

    def _separate_definition_and_facts(
        self,
        knowledge: list[tuple[str, str, float]],
        definition: str | None,
    ) -> tuple[str | None, list[tuple[str, str, float]]]:
        """Separate the definition from relationship facts in the knowledge list.

        Returns (definition, rel_facts).
        """
        rel_facts: list[tuple[str, str, float]] = []
        for rel, target, weight in knowledge:
            if rel == "is defined as":
                if not definition:
                    definition = target
            else:
                rel_facts.append((rel, target, weight))
        return definition, rel_facts

    def _build_rel_clauses(
        self, rel_facts: list[tuple[str, str, float]]
    ) -> tuple[list[str], set[str], set[str]]:
        """Build predicate clauses from relationship facts.

        Each fact becomes a predicate clause: "{verb} {target}".
        Multiple clauses are joined with connectors that vary
        with emotional state.

        Filters out tautological, meaningless, and repetitive edges.
        Limits ``related_to`` to at most 1 clause to prevent synonym spam.

        Returns (clauses, seen_clauses, seen_targets).
        """
        clauses: list[str] = []
        seen_clauses: set[str] = set()
        seen_targets: set[str] = set()
        related_to_count = 0

        for rel, target, _weight in rel_facts[:6]:
            if self._is_noisy_target(target):
                continue
            # Skip tautological targets (target is a relation verb or
            # generic category word)
            if self._is_meaningless_fact_target(target, rel):
                continue
            target_display = self._display_name(target)
            if target_display.lower() in seen_targets:
                continue
            # Limit related_to to 1 per thought
            rel_key_check = rel.strip().replace(" ", "_")
            if rel_key_check.startswith("is_"):
                rel_key_check = rel_key_check[3:]
            if rel_key_check == "related_to":
                if related_to_count >= 1:
                    continue
                related_to_count += 1
            seen_targets.add(target_display.lower())

            # Normalize relation to verb-map key (spaces or leading "is ")
            rel_key = rel.strip().replace(" ", "_")
            if rel_key.startswith("is_") and rel_key[3:] in self._RELATION_VERBS:
                rel_key = rel_key[3:]
            # Try graph-based verb lookup first
            verbs: list[str] | None = None
            if self._network is not None:
                rt_match = None
                for rt in RelationType:
                    if rt.value == rel_key:
                        rt_match = rt
                        break
                if rt_match is not None:
                    graph_verbs = self._network.find_relation_verbs(rt_match)
                    if graph_verbs:
                        verbs = graph_verbs
            if verbs is None:
                verbs = self._RELATION_VERBS.get(rel_key)
            if verbs is None:
                # Unknown relation — use it as-is, cleaned up
                verb = rel.replace("_", " ")
            else:
                verb = self._rng.choice(verbs)
            # Store as predicate (without subject) so we can use display
            # or a pronoun later
            # Add an article to the target if appropriate — "need brain"
            # should be "need the brain", but "depends on memory" should
            # stay as-is (memory is a mass noun).
            target_with_article = self._with_article(verb, target_display)
            clause = f"{verb} {target_with_article}"
            clause_key = clause.lower()
            # Skip duplicate predicates (same relation + target)
            if clause_key in seen_clauses:
                continue
            seen_clauses.add(clause_key)
            clauses.append(clause)
            # Limit to 3 clauses for concise output
            if len(clauses) >= 3:
                break

        return clauses, seen_clauses, seen_targets

    @staticmethod
    def _is_meaningless_fact_target(target: str, rel: str) -> bool:
        """Return True if a target is too vague or tautological to express.

        Filters out relation verbs used as concepts and function words
        that produce nonsense when composed into speech.
        """
        if not target:
            return True
        targ_lower = target.lower().split("#")[0].strip()
        # Single-character targets are never meaningful
        if len(targ_lower) <= 1:
            return True
        rel_key = rel.replace(" ", "_")
        # Target is the relation verb itself
        if targ_lower == rel_key or targ_lower.replace(" ", "_") == rel_key:
            return True
        # Relation verb phrases as targets
        if targ_lower in {
            "related to", "connects to", "has to do with",
            "part of", "is a", "is an", "is the",
            "similar to", "opposite of",
            "related", "relates", "connects", "connected",
            "similar", "opposite", "instance",
        }:
            return True
        # Pure function words
        if targ_lower in {"is", "are", "was", "were", "be", "been",
                          "yes", "no", "true", "false",
                          "thing", "something", "anything", "everything"}:
            return True
        return False

    def _add_reasoning_clauses(
        self,
        reasoning: list[str],
        display: str,
        seen_clauses: set[str],
        seen_targets: set[str],
        clauses: list[str],
    ) -> None:
        """Add reasoning conclusions as predicate clauses (modifies clauses in-place)."""

        def _target_from_clause(clause: str) -> str | None:
            """Extract the likely target (last word) from a predicate clause."""
            # Clauses are "verb target" (e.g. "relates to python");
            # the target is usually the last word
            parts = clause.split()
            if parts:
                return parts[-1]
            return None

        # ── Add reasoning conclusions ──
        for r in reasoning[:2]:
            # Strip all parenthetical asides (transitive chains,
            # connection explanations, etc.) before further processing.
            cleaned = re.sub(r"\s*\([^)]*\)", "", r)
            # Reasoning conclusions look like "cognitive part_of mind";
            # drop the subject and keep just the predicate
            cleaned = cleaned.replace("_", " ")
            cleaned = cleaned.lower().rstrip(".")
            if cleaned.startswith(display.lower() + " "):
                cleaned = cleaned[len(display) + 1:]
            cleaned = cleaned.strip()
            # Reasoning sometimes prefixes conclusions with "might" or
            # "might be", e.g. "might related to x" or "might be part of y".
            # Turn these into plain predicates.
            if cleaned.startswith("might be "):
                cleaned = cleaned[9:]
            elif cleaned.startswith("might "):
                cleaned = cleaned[6:]
            # Convert the raw relation into a proper verb clause.
            # E.g. "part of subcognitive" -> "is part of subcognitive".
            cleaned = self._make_reasoning_clause(cleaned)
            if not cleaned:
                continue
            # Normalize the target inside the reasoning clause the same way
            # as relationship targets (strip #N, lowercase, etc.).
            raw_target = _target_from_clause(cleaned)
            if raw_target:
                target_display = self._display_name(raw_target)
                # Replace the raw target with the display form.
                # The target is the last word(s) of the clause.
                verb_part = cleaned[: len(cleaned) - len(raw_target)]
                target_with_article = self._with_article(verb_part, target_display)
                cleaned = verb_part + target_with_article

            # Skip duplicates and anything that just repeats a relation
            # we already know (e.g. direct edge already said "is part of mind").
            if cleaned not in seen_clauses:
                # Also skip if the meaning is already covered by an existing
                # clause (e.g. "is part of mind" covers "part of mind").
                covered = any(
                    cleaned in existing or existing in cleaned
                    for existing in seen_clauses
                )
                reason_target = _target_from_clause(cleaned)
                target_known = reason_target and reason_target.lower() in seen_targets
                if not covered and (reason_target is None or not target_known):
                    seen_clauses.add(cleaned)
                    if reason_target:
                        seen_targets.add(reason_target.lower())
                    clauses.append(cleaned)

    def _clean_definition(self, definition: str | None) -> str | None:
        """Clean and truncate a definition for use in a sentence.

        Corrupted definitions sometimes contain concatenated
        relationship facts like "X is related to Y". Skip those.
        """
        # Definition — if she has one, use it as the first sentence
        # Corrupted definitions sometimes contain concatenated
        # relationship facts like "X is related to Y". Skip those.
        defn: str | None = None
        if definition and not any(
            marker in definition
            for marker in (
                " is related to",
                " part of",
                " emerges from",
                " depends on",
            )
        ):
            defn = definition.rstrip(".")
            # Definitions can contain multiple clauses or malformed
            # concatenated text; prefer semicolon boundaries first,
            # then sentence boundaries.
            if ";" in defn:
                defn = defn.split(";")[0].strip()
            elif "." in defn:
                candidates = [s.strip() for s in defn.split(".") if 20 < len(s.strip()) <= 120]
                if candidates:
                    defn = candidates[0]
            # Keep definitions concise
            if len(defn) > 120:
                defn = defn[:120].rsplit(" ", 1)[0]
            defn = defn.lower()
        return defn

    def _compose_rel_sentence(
        self,
        clauses: list[str],
        display: str,
        emotion: EmotionalState,
        concept_id: str = "",
    ) -> str | None:
        """Compose the relationship clauses into a sentence using a pronoun."""
        # Relationship clauses — form a second sentence using a pronoun
        if not clauses:
            return None
        # Animacy-aware pronoun: the concept's category and gender
        # property pick she/he/they/it rather than assuming "it" for
        # everything that isn't Genesis.
        if display.lower() in ("i", "me", "myself"):
            pronoun, plural = "I", False
        else:
            pron, plural = person_pronoun(
                concept_id or display, self._network, display
            )
            pronoun = pron.capitalize()
        # Fix verb agreement for plural pronoun
        if plural:
            clauses = [agree_verb_phrase(c, True) for c in clauses]
        if len(clauses) == 1:
            return f"{pronoun} {clauses[0]}."
        elif len(clauses) == 2:
            connector = self._pick_clause_connector(emotion)
            if connector == "—":
                # Em-dash: use surrounding spaces for readability in
                # generated text (em-dash without spaces is a
                # typographic convention for print, but it makes
                # spoken-text output look like "sleep—needs" which
                # reads as a broken word).
                return f"{pronoun} {clauses[0]} — {clauses[1]}."
            return f"{pronoun} {clauses[0]} {connector} {clauses[1]}."
        else:
            connector = self._pick_clause_connector(emotion)
            joined = ", ".join(clauses[:-1])
            if connector == "—":
                return f"{pronoun} {joined} — {clauses[-1]}."
            return f"{pronoun} {joined}, {connector} {clauses[-1]}."

    def _compose_closing(
        self, confidence: float, emotion: EmotionalState
    ) -> str | None:
        """Compose a confidence-based closing qualification.

        No hardcoded closing phrases — if the graph has no closing
        content, return None and let the sentence end naturally.
        """
        graph_words = self._utterance_words("closing")
        if graph_words and confidence < 0.6:
            return self._rng.choice(graph_words)
        return None

    def _make_reasoning_clause(self, raw: str) -> str:
        """Convert a raw reasoning conclusion into a proper predicate clause.

        Reasoning conclusions are like "part of mind" or "related to python";
        we need a clause that can follow a pronoun like "It".
        """
        raw = raw.strip()
        if not raw:
            return ""
        # Try to find a known relation at the start of the raw string.
        # Also handle adverb-prefixed relations: "directly depends on X"
        # should match "depends_on" and produce "directly needs X", not
        # "is directly depends on X" (from the fallback "is" prefix).
        for rel in sorted(self._RELATION_VERBS, key=len, reverse=True):
            rel_spaced = rel.replace("_", " ")
            if raw.startswith(rel_spaced + " "):
                target = raw[len(rel_spaced) + 1:].strip()
                verbs = self._RELATION_VERBS[rel]
                if verbs:
                    verb = self._rng.choice(verbs)
                else:
                    verb = rel_spaced
                return f"{verb} {target}"
            if raw.startswith(rel + " "):
                target = raw[len(rel) + 1:].strip().replace("_", " ")
                verbs = self._RELATION_VERBS[rel]
                if verbs:
                    verb = self._rng.choice(verbs)
                else:
                    verb = rel_spaced
                return f"{verb} {target}"
        # Also try matching base verb forms (without trailing 's').
        # Reasoning conclusions use "might enable" (base form), but
        # the relation key is "enables" (third-person). Match both.
        for rel in sorted(self._RELATION_VERBS, key=len, reverse=True):
            rel_spaced = rel.replace("_", " ")
            # Strip trailing 's' from the first word to get the base form
            base_words = rel_spaced.split()
            if base_words and base_words[0].endswith("s") and not base_words[0].endswith("ss"):
                base_words[0] = base_words[0][:-1]
                base_spaced = " ".join(base_words)
                if raw.startswith(base_spaced + " "):
                    target = raw[len(base_spaced) + 1:].strip()
                    verbs = self._RELATION_VERBS[rel]
                    if verbs:
                        verb = self._rng.choice(verbs)
                    else:
                        verb = rel_spaced
                    return f"{verb} {target}"
        # Check for adverb-prefixed relations: "directly depends on X",
        # "also relates to Y". Strip the leading adverb, try to match,
        # then re-attach it.
        words = raw.split(None, 1)
        if len(words) == 2:
            first_word = words[0].lower()
            # Common adverb modifiers in reasoning conclusions
            if first_word in (
                "directly", "also", "strongly", "closely", "largely",
                "partly", "mostly", "perhaps", "possibly", "clearly",
                "ultimately", "therefore", "thus", "hence", "consequently",
                "eventually", "necessarily", "probably", "certainly",
                "indeed", "notably", "significantly", "essentially",
            ):
                remainder = words[1]
                for rel in sorted(self._RELATION_VERBS, key=len, reverse=True):
                    rel_spaced = rel.replace("_", " ")
                    if remainder.startswith(rel_spaced + " "):
                        target = remainder[len(rel_spaced) + 1:].strip()
                        verbs = self._RELATION_VERBS[rel]
                        if verbs:
                            verb = self._rng.choice(verbs)
                        else:
                            verb = rel_spaced
                        return f"{first_word} {verb} {target}"
                # Also try base verb forms (without trailing 's')
                for rel in sorted(self._RELATION_VERBS, key=len, reverse=True):
                    rel_spaced = rel.replace("_", " ")
                    base_words = rel_spaced.split()
                    if (
                        base_words
                        and base_words[0].endswith("s")
                        and not base_words[0].endswith("ss")
                    ):
                        base_words[0] = base_words[0][:-1]
                        base_spaced = " ".join(base_words)
                        if remainder.startswith(base_spaced + " "):
                            target = remainder[len(base_spaced) + 1:].strip()
                            verbs = self._RELATION_VERBS[rel]
                            if verbs:
                                verb = self._rng.choice(verbs)
                            else:
                                verb = rel_spaced
                            return f"{first_word} {verb} {target}"
        # Unknown relation — prepend "is" only if it looks like a
        # predicate that needs a copula (noun phrase, adjective).
        # Don't prepend "is" if the clause already starts with a verb
        # (detected by checking against common verb stems).
        if " " in raw and not raw.startswith(("is ", "has ", "can ", "does ")):
            first_word = raw.split(None, 1)[0].lower()
            # Common verbs that don't need "is" prefix
            if first_word.endswith("s") and len(first_word) > 3:
                # "depends", "relates", "connects", "enables", etc. —
                # these are already verbs, don't prepend "is"
                return raw
            if first_word in ("depends", "relates", "connects", "enables",
                              "causes", "creates", "produces", "contains",
                              "includes", "requires", "needs", "harms",
                              "prevents", "blocks", "stops"):
                return raw
            return f"is {raw}"
        return raw

    def _pick_clause_connector(self, emotion: EmotionalState) -> str:
        """Pick a connector for joining knowledge clauses, modulated by emotion."""
        if emotion.creativity > 0.6:
            return self._rng.choice(["and", "and", "—"])
        elif emotion.caution > 0.5:
            return self._rng.choice(["and", "though also"])
        else:
            return self._rng.choice(["and", "and also"])

    def _display_name(self, concept_id: str) -> str:
        """Convert a concept ID to a human-readable display name."""
        if not concept_id:
            return ""
        # Strip disambiguation suffixes (e.g., "mind#2" → "mind")
        name = concept_id.split("#")[0]
        # Strip code concept prefixes (e.g., "python:CognitionEngine" → "CognitionEngine")
        if ":" in name and not name.startswith("http"):
            name = name.split(":", 1)[1]
        # Take the last component of module paths
        if "." in name:
            name = name.split(".")[-1]
        # Convert underscores to spaces
        name = name.replace("_", " ")
        # Lowercase for natural speech (voice capitalizes first word)
        name = name.lower()
        return name

    @staticmethod
    def _is_valid_concept_name(base: str) -> bool:
        """Check if a concept name looks like a real concept.

        Filters out garbled names imported from scrapes — sentence
        fragments, fragments with no vowels, or names that don't
        look like real concepts. This prevents Genesis from saying
        things like "It is a kind of lso said."
        """
        if not base or len(base) <= 1:
            return False
        words = base.split()
        # Reject sentence fragments (4+ words with no capitalization
        # — these are usually imported text, not concept names)
        if len(words) > 4:
            return False
        # Reject names with no vowels (acronyms are fine if short,
        # but 3+ chars with no vowels are usually garbled)
        if len(base) >= 3 and not any(c in "aeiouAEIOU" for c in base):
            return False
        # Reject possessive fragments ("unesco's memory", "phone's memory")
        if any("'" in w for w in words):
            return False
        # Reject names that look like sentence fragments starting
        # with lowercase words (real concepts are usually nouns)
        if len(words) >= 3 and all(w[0].islower() for w in words):
            # Allow if it's a known compound pattern like "alert cognitive state"
            # (all lowercase but short and noun-like)
            if len(words) > 3:
                return False
        return True

    def _is_speakable_concept(self, concept_id: str) -> bool:
        """Check if a concept is speakable — has real content.

        A concept is speakable if it has a definition OR has outgoing
        edges in the graph. Concepts with no definition and no
        outgoing edges are garbage from scrapes — they may have
        incoming edges (other concepts point to them), but they
        themselves have no content.
        """
        if self._network is None:
            return True
        c = self._network.get_concept(concept_id)
        if c is None:
            return False
        # Has a real definition (not "NO DEF")
        defn = c.properties.get("definition", "NO DEF")
        if defn and defn != "NO DEF":
            return True
        # Has outgoing edges in the graph
        if len(self._network.get_edges(concept_id, "out")) > 0:
            return True
        # No definition and no outgoing edges — garbage
        return False

    # ─── Article insertion ───────────────────────────────────────
    # Common mass nouns in English — these don't take articles in
    # general statements ("memory is...", "sleep causes...").
    _MASS_NOUNS: ClassVar[set[str]] = {
        # Substances/materials
        "water", "air", "fire", "light", "heat", "ice", "steam",
        "iron", "gold", "silver", "copper", "hydrogen", "oxygen",
        "carbon", "nitrogen", "salt", "sugar", "oil", "glass",
        "wood", "stone", "metal", "plastic", "rubber",
        # Abstract concepts
        "memory", "cognition", "awareness", "thought", "thinking",
        "perception", "attention", "intention", "will",
        "meaning", "truth", "beauty", "love", "hate", "fear", "joy",
        "happiness", "sadness", "anger", "emotion", "feeling",
        "intelligence", "creativity", "imagination", "reason",
        "logic", "knowledge", "wisdom", "understanding", "insight",
        "time", "space", "energy", "power", "force", "gravity",
        "information", "data", "language", "music", "art", "science",
        "philosophy", "mathematics", "history", "literature",
        "sleep", "dreaming", "learning", "processing",
        "life", "death", "health", "growth", "change", "motion",
        "liquid", "solid", "gas", "plasma", "matter", "substance",
        "nature", "reality", "existence", "being", "nothing",
        "freedom", "justice", "peace", "war", "chaos", "order",
        "complexity", "simplicity", "unity", "diversity",
    }

    @classmethod
    def _is_mass_noun(cls, noun: str) -> bool:
        """Check if a noun is a mass (uncountable) noun."""
        n = noun.lower().strip()
        if n in cls._MASS_NOUNS:
            return True
        # Suffix heuristics for mass nouns
        if n.endswith("ness"):  # cognition, awareness, happiness
            return True
        if n.endswith("ing"):  # learning, thinking, processing
            return True
        if n.endswith("ity"):  # complexity, simplicity, unity
            return True
        return False

    # Proper nouns that should never take an article
    _PROPER_NOUNS: ClassVar[set[str]] = {
        "genesis", "alice", "bob", "carol",
        "earth", "mars", "venus", "jupiter", "saturn",
        "america", "europe", "asia", "africa",
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
        "january", "february", "march", "april",
        "june", "july", "august", "september",
        "october", "november", "december",
    }

    @classmethod
    def _with_article(cls, verb_phrase: str, target: str) -> str:
        """Add an article to a target noun if appropriate.

        Returns the target with "the" prepended when:
        - The target is a singular count noun (not mass, not plural,
          not proper)
        - The verb phrase doesn't already contain an article

        Returns the target unchanged when:
        - The verb phrase already has "a", "an", or "the"
        - The target is a mass noun
        - The target is plural (ends in 's' but not 'ss')
        - The target is a proper noun (in _PROPER_NOUNS or
          capitalized in the concept network)
        """
        if not target:
            return target
        # Check if the verb phrase already contains an article
        vp_lower = verb_phrase.lower()
        if " a " in vp_lower or " an " in vp_lower or " the " in vp_lower:
            return target
        if vp_lower.startswith(("a ", "an ", "the ")):
            return target
        # Don't add article to multi-word targets (likely already
        # qualified: "rapid eye movement", "theory of mind")
        if len(target.split()) > 2:
            return target
        target_lower = target.lower().strip()
        # Don't add article to proper nouns (Genesis, Alice, etc.)
        if target_lower in cls._PROPER_NOUNS:
            return target
        # Don't add article to mass nouns
        if cls._is_mass_noun(target_lower):
            return target
        # Don't add article to plural noun phrases — "dreams",
        # "salt and pepper" (shared plural detection lives in
        # morphology.is_plural_np)
        if is_plural_np(target):
            return target
        # Don't add article if target already starts with an article
        if target_lower.startswith(("the ", "a ", "an ")):
            return target
        # Add "the" for definite reference
        return f"the {target}"

    def _is_noisy_target(self, target: str) -> bool:
        """Return True if a target concept is too code/noisy for speech."""
        if not target:
            return True
        if target.startswith(("python:", "rust:", "code:")):
            return True
        if "." in target or ":" in target:
            return True
        if target.startswith("_") or "__" in target:
            return True
        if "self." in target or "cls." in target:
            return True
        # Single-letter targets are never meaningful concepts
        if len(target) == 1:
            return True
        if re.match(r"^(?:to|the|a|an)\s", target):
            return True
        base = target.split("#")[0].lower()
        if base in ("python", "rust", "code", "class"):
            return True
        if "_" in target and len(target) > 20:
            return True
        if re.match(r"^[A-Z][a-z]+([A-Z][a-z]+)+$", target):
            return True
        # Reject garbled concept names — no vowels in 3+ char names
        # (catches "lso said", "dgn", "ngstr", etc.)
        if len(base) >= 3 and not any(c in "aeiou" for c in base):
            return True
        # Reject sentence fragments (4+ words) imported from scrapes
        if len(base.split()) > 4:
            return True
        # Reject possessive fragments ("unesco's memory", "phone's memory")
        if any("'" in w for w in base.split()):
            return True
        # Reject concepts with no definition and no edges — garbage
        # from scrapes that should never appear in speech
        if not self._is_speakable_concept(target):
            return True
        return False

    # ─── Compositional interrogative assembly ───────────────────
    #
    # A question is assembled from grammatical components, not
    # stored sentence frames:
    #
    #   wh-word + [aux + subject + verb-phrase] + complement
    #
    # The wh-word marks the questioned constituent (the gap).
    # Do-support and verb agreement come from morphology; content
    # words come from the concept network — neighbors, IS_A parents,
    # definitions, learned relation verbs. Nothing here is a
    # response Genesis "says" — these are the syntactic combinators
    # the question types use to express a knowledge gap.

    def _aux_for(self, subject: str) -> str:
        """Do-support auxiliary agreeing with the subject."""
        s = subject.strip().lower()
        if s in ("i", "you", "we", "they", "both", "all") or is_plural_np(s):
            return "do"
        return "does"

    @staticmethod
    def _base_form(verb_phrase: str) -> str:
        """Deconjugate the first word of a verb phrase to base form.

        "gives rise to" → "give rise to" — for do-support questions
        where the main verb is uninflected.
        """
        first, *rest = verb_phrase.split()
        base = is_verb_form(first) or first
        return " ".join([base, *rest])

    @staticmethod
    def _third_sg(verb_phrase: str) -> str:
        """Conjugate the first word of a verb phrase to 3sg.

        "give rise to" → "gives rise to" — for wh-subject questions
        ("what gives rise to X") where the gap is nominative.
        """
        first, *rest = verb_phrase.split()
        return " ".join([conjugate_verb(first), *rest])

    def _wh_question(
        self,
        wh: str,
        subject: str = "",
        verb: str = "",
        complement: str = "",
        aux: str = "",
    ) -> str:
        """Assemble a wh-question from components.

        - No subject → the wh-word fills the subject gap and the
          verb takes 3sg: "what causes rain".
        - With subject → aux + subject + base verb: "what does
          sleep affect", "how do you feel about X".
        - ``aux`` overrides do-support ("what am I missing",
          "what should I know").
        """
        parts = [wh]
        if subject:
            parts.append(aux or self._aux_for(subject))
            parts.append(subject)
            if verb:
                parts.append(self._base_form(verb))
        elif verb:
            parts.append(self._third_sg(verb))
        if complement:
            parts.append(complement)
        return " ".join(parts) + "?"

    def _wh_copula(
        self,
        wh: str,
        subject: str,
        complement: str = "",
        prenominal: bool = False,
    ) -> str:
        """Assemble a copular wh-question.

        - ``prenominal=False``: "what is water" — wh + copula + NP.
        - ``prenominal=True``: "which side is true" — the wh-word
          is a determiner on the subject NP.
        """
        cop = copula(subject)
        if prenominal:
            parts = [wh, subject, cop]
        else:
            parts = [wh, cop, subject]
        if complement:
            parts.append(complement)
        return " ".join(parts) + "?"

    def _yn_question(
        self,
        subject: str,
        verb: str,
        complement: str = "",
        aux: str = "",
    ) -> str:
        """Assemble a yes/no question: aux + subject + base verb."""
        parts = [aux or self._aux_for(subject), subject]
        if verb:
            parts.append(self._base_form(verb))
        if complement:
            parts.append(complement)
        return " ".join(parts) + "?"

    def _statement(
        self, subject: str, verb: str, complement: str = ""
    ) -> str:
        """Assemble a declarative clause with subject agreement."""
        s = subject.strip().lower()
        plural = s in ("i", "you", "we", "they", "both", "all") or is_plural_np(s)
        parts = [subject, conjugate_verb(verb, plural=plural)]
        if complement:
            parts.append(complement)
        return " ".join(parts)

    def _copula_statement(self, subject: str, complement: str) -> str:
        """Assemble a copular declarative: "X is Y"."""
        return f"{subject} {copula(subject)} {complement}"

    @staticmethod
    def _cap_first(text: str) -> str:
        """Capitalize the first letter without lowercasing the rest
        (unlike str.capitalize, which would turn 'I' into 'i')."""
        return text[0].upper() + text[1:] if text else text

    def _relation_verb_for(self, rel: RelationType, fallback: str) -> str:
        """Pick a relation verb phrase — learned from the concept
        network when available, else the seed fallback."""
        if self._network is not None:
            try:
                learned = self._network.find_relation_verbs(rel)
                if learned:
                    return self._rng.choice(learned)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"relation-verb lookup failed for {rel}: {e}")
        return fallback

    def _compose_question_content(
        self, context: dict[str, Any], emotion: EmotionalState
    ) -> list[str]:
        """Compose question content from her actual knowledge state.

        No stored sentence frames — each question type names a
        knowledge gap, and the gap is expressed by assembling
        wh-word + auxiliary + subject + verb phrase + complement
        around content drawn from the concept network (neighbors,
        IS_A parents, definitions, relation verbs).

        Context keys:
        - "question_type": "isolation", "uncertainty", "causation",
          "hypothesis", "contradiction", "connection", "perspective",
          "social_emotional", "social_interest", "social_belief",
          "social_general"
        - "target_concept": the concept she's asking about
        - "gap_detail": extra info about the gap
        """
        qtype = context.get("question_type", "")
        target = context.get("target_concept", "")
        detail = context.get("gap_detail", "")

        net = self._network
        subject = (target or "this").replace("_", " ")
        # Animacy-aware pronouns for referring back to the target
        pron_subj, _ = person_pronoun(
            target, net, subject, for_object=False
        )
        pron_obj, _ = person_pronoun(
            target, net, subject, for_object=True
        )

        handlers = {
            "isolation": self._q_isolation,
            "uncertainty": self._q_uncertainty,
            "causation": self._q_causation,
            "hypothesis": self._q_hypothesis,
            "contradiction": self._q_contradiction,
            "connection": self._q_connection,
            "cooccurrence": self._q_cooccurrence,
            "perspective": self._q_perspective,
            "social_emotional": self._q_social_emotional,
            "social_interest": self._q_social_interest,
            "social_belief": self._q_social_belief,
            "social_general": self._q_social_general,
            "reciprocal": self._q_reciprocal,
        }
        handler = handlers.get(qtype)
        if handler is not None:
            return handler(net, target, subject, detail, pron_subj, pron_obj)

        # Fallback — compose from available data
        if target and detail:
            if target.lower() in detail.lower():
                return [str(detail)]
            return [f"{detail} about {target}"]
        if target:
            return [self._wh_question("what", complement=f"about {subject}")]
        if detail:
            return [str(detail)]
        return []

    # ─── Question-type composers ────────────────────────────────────
    # Each takes (net, target, subject, detail, pron_subj, pron_obj)
    # and returns a list of one composed question, or falls back.

    def _q_isolation(self, net, target, subject, detail, pron_subj, pron_obj):
        # Concept has few connections — ask what else connects
        if net and target:
            neighbors = net.get_neighbors(target)[:2]
            if neighbors:
                known = ", ".join(
                    n[0].replace("_", " ") for n in neighbors
                )
                rel_verb = agree_verb_phrase(
                    self._relation_verb_for(
                        RelationType.RELATED_TO, "connects to"
                    ),
                    is_plural_np(subject),
                )
                lead = self._statement(
                    "I", "know", f"{subject} {rel_verb} {known}"
                )
                ask = self._wh_question(
                    "what else",
                    subject="I",
                    aux="should",
                    verb="know",
                    complement=f"about {pron_obj}",
                )
                return [f"{lead}, but {ask}"]
        return [
            self._wh_question(
                "what else",
                subject="I",
                aux="should",
                verb="know",
                complement=f"about {subject}",
            )
        ]

    def _q_uncertainty(self, net, target, subject, detail, pron_subj, pron_obj):
        # Low confidence — ask for clarification
        if net and target:
            concept = net.get_concept(target)
            if concept:
                defn = concept.properties.get("definition", "")
                if defn and defn != "NO DEF":
                    snippet = defn[:50]
                    lead = self._copula_statement(
                        f"my understanding of {subject}",
                        f"incomplete — {snippet}",
                    )
                    ask = self._wh_question(
                        "what", subject="I", aux="am", verb="missing"
                    )
                    return [f"{lead}. {self._cap_first(ask)}"]
        return [self._wh_copula("what exactly", subject)]

    def _q_causation(self, net, target, subject, detail, pron_subj, pron_obj):
        # Has IS_A but no CAUSES — ask what produces it
        cause_verb = self._relation_verb_for(
            RelationType.CAUSES, "produce"
        )
        if net and target:
            edges = net.get_edges(target, "out")
            is_a = [e.target for e in edges if e.relation.value == "is_a"]
            if is_a:
                parent = is_a[0].replace("_", " ")
                is_a_verb = agree_verb_phrase(
                    "is a kind of", is_plural_np(subject)
                )
                lead = self._statement(
                    "I", "know", f"{subject} {is_a_verb} {parent}"
                )
                ask = self._wh_question(
                    "what", verb=cause_verb, complement=pron_obj
                )
                return [f"{lead}, but {ask}"]
        return [self._wh_question("what", verb=cause_verb, complement=subject)]

    def _q_hypothesis(self, net, target, subject, detail, pron_subj, pron_obj):
        # From reasoning — ask if the hypothesis holds
        if detail:
            d = detail.replace("_", " ")
            lead = self._statement("my reasoning", "suggest", d)
            ask = self._yn_question("that", "hold", complement="up")
            return [f"{lead}. {self._cap_first(ask)}"]
        hyp = f"{indefinite_article('hypothesis')} hypothesis"
        lead = self._statement(
            "I", "have", f"{hyp} about {subject}"
        )
        ask = self._yn_question(pron_subj, "hold", complement="up")
        return [f"{lead}. {self._cap_first(ask)}"]

    def _q_contradiction(self, net, target, subject, detail, pron_subj, pron_obj):
        # Conflicting beliefs — ask which is true
        if detail:
            d = detail.replace("_", " ")
            lead = (
                f"there {copula('a tension')} "
                f"{indefinite_article('tension')} tension "
                f"in what I know: {d}"
            )
            ask = self._wh_copula(
                "which", "side", complement="true", prenominal=True
            )
            return [f"{self._cap_first(lead)}. {self._cap_first(ask)}"]
        lead = self._statement(
            "something", "feel", f"inconsistent about {subject}"
        )
        ask = self._wh_question("how", subject="both", verb="hold")
        return [f"{self._cap_first(lead)}. {self._cap_first(ask)}"]

    def _q_connection(self, net, target, subject, detail, pron_subj, pron_obj):
        # Multiple paths between concepts — ask about the link
        if detail:
            d = detail.replace("_", " ")
            lead = self._statement(
                "I", "keep",
                f"finding paths between {subject} and {d}",
            )
            ask = self._wh_question("what", subject="that", verb="mean")
            return [f"{lead}. {self._cap_first(ask)}"]
        connect_verb = self._relation_verb_for(
            RelationType.RELATED_TO, "connect to"
        )
        return [
            self._wh_question("what", verb=connect_verb, complement=subject)
        ]

    def _q_cooccurrence(self, net, target, subject, detail, pron_subj, pron_obj):
        # Two concepts co-occur — ask about the relationship
        cause_verb = self._relation_verb_for(RelationType.CAUSES, "cause")
        if detail:
            d = detail.replace("_", " ")
            lead = self._statement(
                f"{subject} and {d}", "keep", "appearing together"
            )
            ask = self._yn_question("one", cause_verb, "the other")
            return [f"{lead}. {self._cap_first(ask)}"]
        return [
            self._wh_copula(
                "what",
                f"the relationship between {subject} "
                "and what I'm seeing",
            )
        ]

    def _q_perspective(self, net, target, subject, detail, pron_subj, pron_obj):
        # Ask about the user's view
        if net and target:
            related = self._words_from_concept(
                target, RelationType.RELATED_TO, "out", 1
            )
            if related:
                return [
                    self._wh_question(
                        "how",
                        subject="you",
                        verb="see",
                        complement=(
                            f"{subject} in light of "
                            f"{related[0].replace('_', ' ')}"
                        ),
                    )
                ]
        return [
            self._wh_copula("what", f"your perspective on {subject}")
        ]

    def _q_social_emotional(self, net, target, subject, detail, pron_subj, pron_obj):
        return [
            self._wh_question(
                "how", subject="you", verb="feel",
                complement=f"about {subject}",
            )
        ]

    def _q_social_interest(self, net, target, subject, detail, pron_subj, pron_obj):
        return [
            self._wh_question(
                "what", verb="draw", complement=f"you to {subject}"
            )
        ]

    def _q_social_belief(self, net, target, subject, detail, pron_subj, pron_obj):
        return [
            self._wh_copula("what", f"your perspective on {subject}")
        ]

    def _q_social_general(self, net, target, subject, detail, pron_subj, pron_obj):
        return [
            self._wh_copula(
                "what", f"on your mind about {subject}"
            )
        ]

    def _q_reciprocal(self, net, target, subject, detail, pron_subj, pron_obj):
        # wh + prepositional fragment — no verb needed
        return [self._wh_question("how", complement="about you")]

    # ──────────────────────────────────────────────────────────────
    # Learned-preference picking (used by the cognition engine for
    # reciprocal questions and feedback-driven word learning)
    # ──────────────────────────────────────────────────────────────

    def _q_pick(self, slot_name: str, fragments: list[str], emotion: EmotionalState) -> str:
        """Pick a fragment, respecting learned preferences and avoiding repeats."""
        if not fragments:
            return ""

        # Honor words she's learned to avoid.
        avoided = self._avoided.get(slot_name, set())
        candidates = [f for f in fragments if f not in avoided] or fragments

        # Honor words she's learned to prefer.
        preferred: deque[str] = self._preferred.get(slot_name, deque())
        usable_preferred = [p for p in preferred if p in candidates]
        if usable_preferred and self._rng.random() < 0.3:
            pick = self._rng.choice(usable_preferred)
            self._track_pick(slot_name, pick)
            return pick

        # Avoid immediate repetition if alternatives exist.
        recent = self._recent_picks.get(slot_name, deque())
        if len(candidates) > 1:
            fresh = [c for c in candidates if c not in recent]
            if fresh:
                candidates = fresh

        # Emotion modulation.
        if emotion.caution > 0.5:
            candidates = [c for c in candidates if len(c) < 60] or candidates
        if emotion.creativity > 0.6:
            elaborate = [c for c in candidates if len(c) >= 40]
            if elaborate:
                candidates = elaborate

        pick = self._rng.choice(candidates)
        self._track_pick(slot_name, pick)
        if slot_name.startswith("q_"):
            self._last_question_fragments[slot_name] = pick
        return pick
