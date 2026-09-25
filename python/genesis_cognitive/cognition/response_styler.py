"""Response styling — metacognitive tone adjustment and learning acknowledgment.

Extracted from CognitionEngine as a focused subsystem. Applies the
active metacognitive strategy (assertive, hedging, questioning, varied,
neutral) to a response, and weaves learning acknowledgments into the
response when the self-directed learner extracted new facts.

The modifications are conservative — they adjust tone, not content.
The underlying knowledge and reasoning remain unchanged. All phrasing
emerges from the language engine, never from hardcoded templates.

Dependencies (passed to ``__init__``):
    - language: LanguageEngine for composing ack phrases
    - composer: ThoughtComposer for composing about newly learned concepts
    - working_memory: WorkingMemory for current focus (anaphora resolution)
    - rng: random number generator for variation
    - response_style_getter: callable returning the current style string
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..emotion import EmotionalState
from ..language import LanguageEngine, Thought
from ..perception import Perception

if TYPE_CHECKING:
    from ..cognition.thought_composer import ThoughtComposer
    from ..memory import WorkingMemory

__all__ = ["ResponseStyler"]

# Common words that should not trigger learning acknowledgments.
# These are function words, pronouns, and everyday adjectives that
# appear in almost every sentence — acknowledging "learning" them
# makes the response incoherent and self-absorbed.
_TRIVIAL_WORDS: frozenset[str] = frozenset({
    "this", "that", "these", "those", "here", "there", "where",
    "when", "what", "which", "who", "whom", "whose", "why", "how",
    "with", "from", "into", "through", "about", "over", "under",
    "have", "been", "were", "they", "them", "their", "theirs",
    "some", "any", "all", "both", "each", "more", "most", "other",
    "same", "own", "too", "can", "did", "does", "doing", "make",
    "made", "take", "took", "get", "got", "come", "came", "like",
    "well", "even", "still", "now", "way", "thing", "things",
    "also", "just", "only", "very", "really", "quite", "such",
    "than", "then", "will", "would", "could", "should", "might",
    "good", "bad", "great", "nice", "fine", "okay", "yes", "sure",
    "wonderful", "amazing", "awesome", "cool", "happy",
    "glad", "pleased", "delighted", "thrilled", "excited",
    "so", "always", "never",
    "something", "anything", "everything", "nothing", "someone",
    "anyone", "everyone", "nobody", "somebody", "anybody",
    "today", "tomorrow", "yesterday", "later",
    "maybe", "perhaps", "probably", "definitely", "certainly",
    "talk", "talking", "said", "say", "saying", "tell", "telling",
    "know", "knew", "known", "knowing", "think", "thought",
    "thinking", "feel", "felt", "feeling", "want", "wanted",
    "need", "needed", "try", "tried", "trying", "let", "lets",
    "going", "gone", "went", "being", "having", "making",
    "lot", "bit", "kind", "sort", "type", "part", "side", "end",
    "first", "last", "next", "new", "old", "long", "short",
})

# Pronouns that can carry an anaphoric reference to the discourse
# focus. "you"/"your" are handled separately (they resolve to
# Genesis, not the focus).
_ANA_PRONOUN_RE = re.compile(
    r"\b(it|this|that|they|them|he|she)\b", re.IGNORECASE
)

_WORD_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _neighbor_words(text: str, start: int, end: int) -> tuple[str, str, str]:
    """Return (prev_word, next_word, word_after_next) around a span."""
    before = _WORD_TOKEN_RE.findall(text[:start])
    after = _WORD_TOKEN_RE.findall(text[end:])
    return (
        before[-1].lower() if before else "",
        after[0].lower() if after else "",
        after[1].lower() if len(after) > 1 else "",
    )


# Closed-class words for anaphora position checks. A pronoun whose
# neighbour is NOT in this set is adjacent to a content word — the
# signature of a relative clause ("mammal that likes"), a
# zero-relative ("the pet you want"), or a determiner ("that dog").
# Common transitive verbs are included so that demonstrative objects
# still resolve ("like that", "do this", "I saw that").
_ANA_FUNC_WORDS: frozenset[str] = frozenset({
    # determiners / quantifiers / wh-words
    "a", "an", "the", "some", "any", "all", "every", "each", "no",
    "none", "both", "few", "several", "many", "much", "more", "most",
    "other", "another", "such", "what", "whatever", "which",
    "whichever", "who", "whom", "whose", "where", "when", "why", "how",
    # pronouns (incl. contractions fragments like 's, n't, 're)
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "this", "that", "these", "those",
    "there", "here", "one", "ones", "itself", "themselves", "himself",
    "herself", "myself", "yourself", "ourselves",
    "'s", "'re", "'ve", "'ll", "'d", "'m", "n't",
    # auxiliaries / copula / modals
    "is", "are", "was", "were", "am", "be", "been", "being", "do",
    "does", "did", "done", "have", "has", "had", "having", "will",
    "would", "can", "could", "shall", "should", "may", "might", "must",
    "ought", "dare",
    # prepositions
    "of", "in", "on", "at", "to", "from", "by", "with", "about", "as",
    "into", "onto", "upon", "through", "throughout", "against",
    "between", "among", "under", "over", "after", "before", "without",
    "within", "around", "across", "behind", "beyond", "during",
    "inside", "outside", "toward", "towards", "near", "off", "out",
    "up", "down", "for", "since", "until", "till", "per", "via",
    "unlike", "despite", "except", "besides", "including",
    # conjunctions / discourse markers / adverbs
    "and", "or", "but", "so", "yet", "nor", "if", "then", "than",
    "because", "although", "though", "while", "whereas", "unless",
    "whether", "once", "also", "even", "just", "only", "really",
    "very", "quite", "rather", "too", "not", "still", "already",
    "again", "always", "never", "ever", "often", "sometimes",
    "usually", "now", "today", "yesterday", "tomorrow", "yes", "yeah",
    "ok", "okay", "well", "oh", "hey", "hi", "hello", "please",
    "thanks", "maybe", "perhaps", "probably", "actually", "indeed",
    # common transitive verbs that take demonstrative objects
    "like", "likes", "liked", "love", "loves", "loved", "hate",
    "hates", "hated", "want", "wants", "wanted", "need", "needs",
    "needed", "prefer", "prefers", "preferred", "watch", "watches",
    "watched", "enjoy", "enjoys", "enjoyed", "use", "uses", "used",
    "get", "gets", "got", "gotten", "take", "takes", "took", "taken",
    "make", "makes", "made", "give", "gives", "gave", "given", "keep",
    "keeps", "kept", "try", "tries", "tried", "explain", "explains",
    "explained", "describe", "describes", "described", "fix", "fixes",
    "fixed", "build", "builds", "built", "test", "tests", "tested",
    "choose", "chooses", "chose", "chosen", "pick", "picks", "picked",
    "bring", "brings", "brought", "change", "changes", "changed",
    "say", "says", "said", "tell", "tells", "told", "know", "knows",
    "knew", "known", "think", "thinks", "thought", "see", "sees",
    "saw", "seen", "hear", "hears", "heard", "feel", "feels", "felt",
    "mean", "means", "meant", "show", "shows", "showed", "shown",
    "find", "finds", "found", "remember", "remembers", "remembered",
    "forget", "forgets", "forgot", "notice", "notices", "noticed",
    "understand", "understands", "understood", "learn", "learns",
    "learned", "believe", "believes", "believed", "guess", "guesses",
    "hope", "hopes", "hoped", "wish", "wishes", "wished", "expect",
    "expects", "expected", "suppose", "supposes", "supposed",
    "prove", "proves", "proved", "realize", "realizes", "realized",
    "claim", "claims", "claimed", "imagine", "imagines", "imagined",
    "assume", "assumes", "assumed", "discover", "discovers",
    "discovered", "consider", "considers", "considered", "discuss",
    "discusses", "discussed", "mention", "mentions", "mentioned",
})

# Words after which "that" introduces a complement clause rather
# than referring — verbs of communication/cognition/perception plus
# copula and raising adjectives ("the problem is that…").
_THAT_COMPLEMENTIZER_PRECEDERS: frozenset[str] = frozenset({
    "say", "says", "said", "tell", "tells", "told", "think", "thinks",
    "thought", "know", "knows", "knew", "known", "believe", "believes",
    "believed", "mean", "means", "meant", "hear", "hears", "heard",
    "see", "sees", "saw", "seen", "feel", "feels", "felt", "guess",
    "guesses", "hope", "hopes", "hoped", "wish", "wishes", "wished",
    "expect", "expects", "expected", "suppose", "supposes", "supposed",
    "show", "shows", "showed", "shown", "prove", "proves", "proved",
    "learn", "learns", "learned", "realize", "realizes", "realized",
    "notice", "notices", "noticed", "remember", "remembers",
    "remembered", "forget", "forgets", "forgot", "understand",
    "understands", "understood", "claim", "claims", "claimed",
    "argue", "argues", "argued", "insist", "insists", "insisted",
    "admit", "admits", "admitted", "deny", "denies", "denied",
    "promise", "promises", "promised", "warn", "warns", "warned",
    "agree", "agrees", "agreed", "decide", "decides", "decided",
    "find", "found", "discover", "discovers", "discovered", "imagine",
    "imagines", "imagined", "assume", "assumes", "assumed", "is",
    "are", "was", "were", "seems", "seemed", "appears", "appeared",
    "looks", "looked", "sounds", "sounded", "remains", "remained",
    "becomes", "became", "sure", "certain", "glad", "sad", "happy",
    "afraid", "aware", "clear", "obvious", "likely", "unlikely",
    "possible", "impossible", "important",
})

# "it" expletive frames — weather/time/evaluative "it" refers to
# nothing and must not be resolved to the discourse focus.
_IT_EXPLETIVE_VERBS: frozenset[str] = frozenset({
    "seems", "seemed", "appears", "appeared", "looks", "looked",
    "sounds", "sounded", "feels", "rains", "rained", "snows",
    "snowed", "drizzles", "drizzled", "takes", "took", "depends",
    "depended", "helps", "helped", "matters", "mattered", "turns",
    "turned",
})
_IT_EXPLETIVE_PREDS: frozenset[str] = frozenset({
    "raining", "snowing", "drizzling", "hailing", "time", "late",
    "early", "dark", "light", "important", "possible", "impossible",
    "necessary", "hard", "difficult", "easy", "clear", "obvious",
    "true", "okay", "ok", "worth", "enough", "cold", "hot", "warm",
    "cool", "sunny", "cloudy", "windy", "foggy", "humid", "night",
    "day", "morning", "evening", "noon", "midnight", "fair",
})


class ResponseStyler:
    """Apply metacognitive tone adjustments and learning acknowledgments.

    All modifications are tone-level — the underlying knowledge and
    reasoning remain unchanged. Phrasing emerges from the language
    engine, never from hardcoded template strings.
    """

    def __init__(
        self,
        language: LanguageEngine,
        composer: ThoughtComposer,
        working_memory: WorkingMemory,
        rng: Any,
        response_style_getter: Callable[[], str],
        network: Any = None,
    ) -> None:
        """Wire the styler to its language engine, composer, and working memory."""
        self._language = language
        self._composer = composer
        self._working_memory = working_memory
        self._rng = rng
        self._response_style_getter = response_style_getter
        # Concept network for referent typing (animacy). Optional —
        # without it, anaphora falls back to the single current focus.
        self._network = network

    # ─── Response style application ──────────────────────────────

    def apply_response_style(self, response: str, thought: Thought) -> str:
        """Apply the active metacognitive strategy to the response.

        - **assertive**: trim hedging language, present knowledge directly
        - **hedging**: add uncertainty markers when confidence is low
        - **questioning**: no-op (questions are queued via /teach-questions)
        - **varied**: rephrase to break repetitive patterns
        - **neutral**: no modification
        """
        style = self._response_style_getter()
        if style == "neutral" or not response:
            return response

        lower = response.lower()

        if style == "assertive":
            return self._style_assertive(response, lower)
        elif style == "hedging":
            return self._style_hedging(response, lower, thought)
        elif style == "questioning":
            return self._style_questioning(response, lower, thought)
        elif style == "varied":
            return self._style_varied(response, lower)

        return response

    def _style_assertive(self, response: str, lower: str) -> str:
        """Trim hedging language to be direct when confident."""
        hedges = [
            "i think ",
            "i believe ",
            "perhaps ",
            "maybe ",
            "i'm not sure but ",
            "it seems like ",
        ]
        for hedge in hedges:
            if hedge in lower:
                idx = lower.index(hedge)
                response = response[:idx] + response[idx + len(hedge):]
                lower = response.lower()
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        return response

    def _style_hedging(self, response: str, lower: str, thought: Thought) -> str:
        """Hedging style — no longer appends uncertainty inline.

        Uncertainty is now expressed through two architecturally
        correct pathways, both of which compose words via the language
        engine rather than appending fixed sentences:

        1. **Voice layer** (``voice.py:_calibrate_hedging``): during
           render, high ``emotion.caution`` inserts hedges ("I think",
           "perhaps", "it seems to me") from its voice vocabulary.
        2. **Knowledge-gap hedge** (``cognition/engine.py``): when
           self-assessment detects it can't answer, a ``Thought`` with
           ``intent="unknown"`` and ``metadata={"knowledge_gap": ...}``
           is rendered by the language engine.

        This style is kept as a no-op to avoid breaking the response
        styler dispatch, but it no longer appends hardcoded sentences.
        """
        return response

    def _style_questioning(self, response: str, lower: str, thought: Thought) -> str:
        """Questioning style — no longer appends questions inline.

        All curiosity questions are now queued in the cognition engine
        and presented to the user via /teach-questions. This style is
        kept as a no-op to avoid breaking the response styler dispatch,
        but it no longer appends questions to the response.
        """
        return response

    def _style_varied(self, response: str, lower: str) -> str:
        """Force variation — rephrase the opening if it's repetitive."""
        openers = ["i ", "this ", "the ", "my ", "it "]
        if any(lower.startswith(op) for op in openers):
            varied_openers = [
                "So, ",
                "Well, ",
                "Actually, ",
                "Interestingly, ",
            ]
            opener = self._rng.choice(varied_openers)
            response = opener + response[0].lower() + response[1:]
        return response

    # ─── Anaphora resolution ─────────────────────────────────────

    def resolve_anaphora(self, user_input: str) -> str:
        """Resolve bare pronouns to the current focus of conversation.

        "You" and "your" resolve to "Genesis" — the user is talking
        to it — except inside zero-relative clauses where "you" is
        generic ("a pet you can keep"). "it", "they", "them", "he",
        "she" resolve to the current attentional focus.

        "this"/"that" resolve only as *demonstrative pronouns* in
        argument position — sentence-initial ("that is cool"), after
        a verb or preposition ("like that", "about that"), or final
        ("what is that?"). They are left untouched as determiners
        ("that dog"), and "that" is left untouched as a relative
        pronoun ("a mammal that likes X") or complementizer ("I
        think that X"). Rewriting those corrupts the sentence —
        "a ferret is a small mammal that likes to steal" would be
        learned verbatim as "…mammal <focus> likes to steal".

        "it" is skipped in expletive frames where it refers to
        nothing ("it seems", "it is raining", "it looks like …").
        """
        resolved = self._sub_matches(r"\byour\b", "Genesis's", user_input)
        resolved = self._sub_matches(r"\byou\b", "Genesis", resolved)

        target: re.Match[str] | None = None
        for m in _ANA_PRONOUN_RE.finditer(resolved):
            if self._is_anaphoric(resolved, m):
                target = m
                break
        if target is None:
            return resolved

        referent = self._pick_referent(target.group(0).lower())
        if not referent:
            return resolved

        return resolved[: target.start()] + referent + resolved[target.end() :]

    # Discourse participants that anaphoric it/that/this almost never
    # refer to — those roles already have dedicated pronouns (I/you).
    _ANA_NON_REFERENTS: frozenset[str] = frozenset({"genesis", "user", "self"})

    def _pick_referent(self, word: str) -> str | None:
        """Choose the discourse referent for an anaphoric pronoun.

        Walks the executive's focus history (newest first) and filters
        by pronoun type rather than blindly taking the single current
        focus:

        - ``he``/``she`` bind only to animate referents — binding them
          to an inanimate focus ("the wheel fell… he") corrupts input.
          When nothing animate is in history, the pronoun is left
          unresolved instead of forcing a wrong referent.
        - ``it``/``this``/``that`` skip the discourse participants
          (genesis/user/self) — those are already "I"/"you" — and take
          the most recent remaining entity.
        - ``they``/``them`` prefer an animate referent but accept any
          entity (English "they" is also plural-inanimate).
        """
        history = list(
            getattr(
                self._working_memory.central_executive, "focus_history", []
            )
        )
        focus = self._working_memory.central_executive.current_focus
        if not history and focus:
            history = [focus]
        if not history:
            top = self._working_memory.get_top_attention(1)
            if top:
                history = [top[0]]
        if not history:
            return None

        if word in ("he", "she"):
            for cand in history:
                if self._is_animate(cand):
                    return cand
            return None
        if word in ("they", "them"):
            for cand in history:
                if self._is_animate(cand):
                    return cand
            return history[0]
        # it / this / that — most recent entity that isn't a participant.
        for cand in history:
            if cand.lower() not in self._ANA_NON_REFERENTS:
                return cand
        return history[0]

    # Small person/creature set for the is_a fallback check — covers
    # referents whose category detection hasn't classified them yet.
    _ANIMATE_HEADS: frozenset[str] = frozenset({
        "person", "human", "people", "man", "woman", "friend", "user",
        "creator", "animal", "mammal", "creature", "bird", "fish",
        "insect", "cat", "dog", "pet",
    })

    def _is_animate(self, name: str) -> bool:
        """Whether a focus candidate refers to an animate entity.

        Uses the concept network when available: LIVING category, or an
        is_a edge into a LIVING/person-ish target. Unknown concepts are
        not animate.
        """
        if self._network is None:
            return False
        concept = self._network.get_concept(name)
        if concept is None:
            return False
        if getattr(concept.category, "value", concept.category) == "living":
            return True
        for edge in self._network.get_edges(concept.id, "out"):
            if getattr(edge.relation, "value", edge.relation) != "is_a":
                continue
            if edge.target in self._ANIMATE_HEADS:
                return True
            target = self._network.get_concept(edge.target)
            if target is not None and (
                getattr(target.category, "value", target.category) == "living"
            ):
                return True
        return False

    def _sub_matches(self, pattern: str, repl: str, text: str) -> str:
        """Replace every match that sits in anaphoric position."""
        out: list[str] = []
        last = 0
        for m in re.finditer(pattern, text):
            if not self._is_anaphoric(text, m):
                continue
            out.append(text[last : m.start()])
            out.append(repl)
            last = m.end()
        out.append(text[last:])
        return "".join(out)

    @staticmethod
    def _is_anaphoric(text: str, m: re.Match[str]) -> bool:
        """Whether a pronoun match occupies an anaphoric position.

        Decides from the neighbouring words: a pronoun between two
        content words is embedded in a clause ("mammal that likes",
        "the pet you want"), not a reference to the discourse focus.
        Function words are a closed class, so any neighbour outside
        _ANA_FUNC_WORDS is a content word.
        """
        word = m.group(0).lower()
        prev, nxt, nxt2 = _neighbor_words(text, m.start(), m.end())

        # "you"/"your": generic in zero-relative clauses — "the pet
        # you want", "a thing your friend said". The pronoun follows
        # a noun (content word) directly.
        if word in ("you", "your"):
            return not (prev and prev not in _ANA_FUNC_WORDS)

        # "it": skip expletive frames — "it seems", "it looks like",
        # "it is raining", "it was time".
        if word == "it":
            if nxt in _IT_EXPLETIVE_VERBS:
                return False
            if nxt in ("is", "was", "'s") and nxt2 in _IT_EXPLETIVE_PREDS:
                return False
            return True

        # they/them/he/she are unambiguously anaphoric.
        if word in ("they", "them", "he", "she"):
            return True

        # this/that — demonstrative only in argument position.
        if word in ("this", "that"):
            if nxt and nxt not in _ANA_FUNC_WORDS:
                # Followed by a content word — determiner ("that
                # dog"), relative pronoun ("mammal that likes"),
                # or complementizer ("think that ferrets…").
                return False
            if prev and prev not in _ANA_FUNC_WORDS:
                # Preceded by a content word — the head noun of a
                # relative clause ("a mammal that was…").
                return False
            if nxt in ("i", "you", "he", "she", "it", "we", "they") and nxt2:
                # A subject pronoun right after "that" starts a
                # complement clause ("told you that he left") — a
                # demonstrative can never take that position.
                return False
            if word == "that" and nxt and prev in _THAT_COMPLEMENTIZER_PRECEDERS:
                # "think that it…", "the thing is that it…" —
                # "that" introduces a clause, it doesn't refer.
                return False
            return True

        return True

    # ─── Learning acknowledgment ─────────────────────────────────

    def acknowledge_learning(
        self,
        response: str,
        learning_events: list,
        perception: Perception,
        emotion: EmotionalState,
    ) -> str:
        """Weave a brief learning acknowledgment into the response.

        The acknowledgment is composed from what it actually learned
        — the concept and its relationships — not from hardcoded
        templates.
        """
        if not learning_events:
            return response

        # Social exchanges (greetings, farewells, acknowledgments) are
        # not learning interactions. Appending knowledge content to
        # "Hey — I'm glad you're here" makes the response incoherent.
        # The learning acknowledgment is for when the user teaches
        # Genesis something new, not for social rituals.
        intent_value = (
            perception.intent.value
            if hasattr(perception.intent, "value")
            else str(perception.intent)
        )
        if intent_value in ("greeting", "farewell", "greeting_question"):
            return response

        conv_events = [
            e for e in learning_events if e.event_type == "conversation"
        ]
        if not conv_events:
            return response

        if len(response) < 15:
            return response

        lower = response.lower()
        if any(
            phrase in lower
            for phrase in (
                "i didn't know",
                "that's new to me",
                "thank you for teaching",
                "i've learned",
                "i just learned",
                "i didn't know about",
                "i've added",
            )
        ):
            return response

        primary_concept = ""
        for e in conv_events:
            for c in e.concepts_involved:
                if c and len(c) > 2 and c.lower() not in _TRIVIAL_WORDS:
                    primary_concept = c
                    break
            if primary_concept:
                break

        ack = ""
        if primary_concept:
            thought = self._composer.compose_about(primary_concept, emotion)
            if thought and thought.confidence > 0.2:
                content = thought.content
                content_lower = content.lower()
                prefix = f"{primary_concept} is "
                if content_lower.startswith(prefix):
                    understanding = content[len(prefix):]
                else:
                    understanding = content
                understanding = understanding.rstrip(".")
                ack = self._compose_learning_ack_phrase(
                    primary_concept, understanding, emotion
                )

        if not ack:
            fallback_content = primary_concept if primary_concept else "this"
            fallback_thought = Thought(
                content=fallback_content,
                intent="acknowledge",
                emotion=emotion.label,
                confidence=0.5,
                topics=[primary_concept] if primary_concept else [],
                metadata={"learning_ack": True, "new_concept": True},
            )
            ack = self._language.generate(fallback_thought, emotion)
            if not ack or not ack.strip():
                ack = fallback_content

        if ". " in response:
            parts = response.split(". ", 1)
            response = f"{parts[0]}. {ack} {parts[1]}"
        else:
            response = f"{ack} {response}"

        return response

    def _compose_learning_ack_phrase(
        self, concept: str, understanding: str, emotion: EmotionalState,
    ) -> str:
        """Compose a brief learning acknowledgment from understanding.

        Routes through the language engine so the phrasing emerges
        from its voice, not from hardcoded templates.
        """
        first_sentence = understanding.split(". ")[0]
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."

        ack_thought = Thought(
            content=first_sentence,
            intent="acknowledge",
            emotion=emotion.label,
            confidence=0.7,
            topics=[concept] if concept else [],
            metadata={"learning_ack": True},
        )
        rendered = self._language.generate(ack_thought, emotion)
        if not rendered or not rendered.strip():
            rendered = first_sentence
        return rendered
