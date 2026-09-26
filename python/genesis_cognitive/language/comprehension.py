"""Language comprehension — the receptive language system.

This is Genesis's Wernicke's area equivalent. While the generative
engine (generator.py) is the *productive* language system (Broca's
area — turning thoughts into speech), the comprehension engine is the
*receptive* system — turning speech into understanding.

The comprehension engine performs basic semantic parsing:

1. **Speech act identification**: Is this a question, statement,
   command, or exclamation? (Searle, 1969)
2. **Proposition extraction**: Who did what to whom? Parses sentences
   into subject-predicate-object triples.
3. **Reference resolution**: Pronouns ("he", "it", "it", "they") are
   resolved to their referents based on context (Hobbs, 1978; the
   centering theory of Grosz, Joshi & Weinstein, 1995).
4. **Key concept identification**: Which words carry the semantic
   weight of the sentence, and what roles do they play (agent,
   patient, instrument, location, etc.)?
5. **Comprehension representation**: A structured understanding of
   what was said, which downstream modules (pragmatics.py, figurative.py)
   can build on.

This module does the *basic* semantic parsing that pragmatics.py
(Gricean maxims) and figurative.py (metaphor/irony detection) build
on. It is the foundation layer of language understanding.

# Neural grounding

In human neuroscience, language comprehension is primarily associated
with Wernicke's area in the posterior superior temporal gyrus of the
left hemisphere (Wernicke, 1874). Damage to Wernicke's area produces
fluent but meaningless speech and impaired comprehension — the patient
speaks smoothly but says nothing sensible, and cannot understand what
others say. The comprehension engine models this receptive function.

The angular gyrus (Brodmann area 39) is involved in mapping between
visual word forms and semantic representations, and the temporal subsystem
stores the semantic network that words activate. Comprehension is not
a single process but a distributed one: phonological decoding (superior
temporal), semantic access (middle temporal), and thematic role
assignment (left temporal cortex) work together (Binder, 2015).

References:
- Wernicke, C. (1874). *Der aphasische Symptomencomplex*. Cohn.
- Searle, J. R. (1969). *Speech Acts*. Cambridge University Press.
- Hobbs, J. R. (1978). Resolving pronoun references. *Lingua*, 44,
  311–338.
- Grosz, B. J., Joshi, A. K., & Weinstein, S. (1995). Centering:
  A framework for modeling the local coherence of discourse.
  *Computational Linguistics*, 21(2), 203–225.
- Binder, J. R. (2015). The Wernicke area: Modern evidence and a new
  interpretation. *Neuroimage*, 115, 243–256.
- Kintsch, W. (1998). *Comprehension: A Paradigm for Cognition*.
  Cambridge University Press.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar

from .figurative import FigurativeLanguageProcessor, IronyDetection, Metaphor
from .morphology import (
    VERB_LEXICON,
    deconjugate_verb,
    is_participle,
    is_verb_form,
)

__all__ = [
    "ComprehensionEngine",
    "ComprehensionResult",
    "ConceptRole",
    "Proposition",
    "SemanticRole",
    "SpeechActType",
]

# ─── Pre-compiled regex patterns ──────────────────────────────────

# Split into sentences on . ! ? followed by whitespace or end
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Identify the main verb in a simple sentence — looks for common
# action verbs and copular verbs. This is a heuristic parser, not
# a full dependency parser.
_COPULAR_VERBS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "been",
        "being",
    }
)

# Common auxiliary verbs that modify the main verb
_AUX_VERBS = frozenset(
    {
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
    }
)

# Pronouns and their typical referent types
_SUBJECT_PRONOUNS = frozenset(
    {
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
    }
)
_OBJECT_PRONOUNS = frozenset(
    {
        "me",
        "you",
        "him",
        "her",
        "it",
        "us",
        "them",
    }
)
_POSSESSIVE_PRONOUNS = frozenset(
    {
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
    }
)

# Possessives that can serve as a predicate complement —
# "the ball is yours". (Determiner possessives like "my" are
# excluded; they open a noun phrase instead.)
_PREDICATE_POSSESSIVES = frozenset(
    {
        "mine",
        "yours",
        "his",
        "hers",
        "its",
        "ours",
        "theirs",
    }
)

# Question words
_WH_WORDS = frozenset({"what", "who", "where", "when", "why", "how", "which"})

# Command/imperative indicators — sentences that start with a base-form
# verb (no explicit subject) are likely commands.
_IMPERATIVE_STARTERS = frozenset(
    {
        "tell",
        "show",
        "give",
        "explain",
        "describe",
        "help",
        "stop",
        "start",
        "wait",
        "look",
        "listen",
        "remember",
        "forget",
        "think",
        "consider",
        "imagine",
        "suppose",
        "let",
        "try",
        "ask",
        "answer",
        "find",
        "search",
        "create",
        "build",
        "make",
        "run",
        "check",
        "verify",
        "test",
        "fix",
        "add",
        "remove",
        "delete",
        "update",
        "read",
        "write",
        "open",
        "close",
        "save",
        "load",
        "print",
        "compute",
        "calculate",
        "analyze",
        "compare",
        "sort",
        "count",
        # Manipulation imperatives — instructions about arranging
        # things ("place the block", "insert the peg"). "place" is
        # deliberately NOT in the verb lexicon ("that place is warm"
        # is a noun phrase) — but clause-initial position can only
        # be a verb.
        "place",
        "insert",
        "drop",
        "slide",
        "fill",
        "position",
        "arrange",
    }
)

# Common prepositions that introduce prepositional phrases
_PREPOSITIONS = frozenset(
    {
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "of",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "against",
    }
)

# Negation words
_NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nor",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "shouldn't",
        "wouldn't",
        "couldn't",
        "hasn't",
        "haven't",
        "hadn't",
    }
)

# Conjunctions that join clauses
_CONJUNCTIONS = frozenset(
    {
        "and",
        "but",
        "or",
        "so",
        "yet",
        "because",
        "although",
        "though",
        "while",
        "whereas",
        "if",
        "unless",
        "since",
        "as",
        "when",
    }
)

# A small set of very common verbs, used by the verb-detection
# heuristic in _looks_like_verb. Intentionally conservative — false
# negatives (missing a verb) are better than false positives
# (misidentifying a noun as a verb).
_COMMON_VERBS = frozenset(
    {
        "said",
        "told",
        "gave",
        "took",
        "made",
        "went",
        "came",
        "saw",
        "knew",
        "thought",
        "felt",
        "found",
        "asked",
        "answered",
        "showed",
        "sent",
        "brought",
        "bought",
        "built",
        "wrote",
        "read",
        "spoke",
        "heard",
        "ran",
        "sat",
        "stood",
        "fell",
        "rose",
        "grew",
        "became",
        "seemed",
        "appeared",
        "happened",
        "occurred",
        "existed",
        "liked",
        "loved",
        "hated",
        "wanted",
        "needed",
        "tried",
        "helped",
        "worked",
        "played",
        "lived",
        "died",
        "moved",
        "changed",
        "stayed",
        "began",
        "started",
        "stopped",
        "ended",
        "finished",
        "continued",
        "remained",
        "left",
        "returned",
        "arrived",
        "departed",
        "entered",
        "exited",
        "opened",
        "closed",
        "broke",
        "fixed",
        "lost",
        "kept",
        "dropped",
        "picked",
        "put",
        "set",
        "let",
        "get",
        "got",
        "have",
        "has",
        "had",
        "say",
        "tell",
        "know",
        "see",
        "think",
        "feel",
        "go",
        "come",
        "make",
        "take",
        "give",
        "find",
        "look",
        "seem",
        "show",
        "want",
        "use",
        "try",
        "help",
        "turn",
        "start",
        "live",
        "play",
        "run",
        "move",
        "believe",
        "hold",
        "bring",
        "happen",
        "write",
        "provide",
        "sit",
        "stand",
        "lose",
        "pay",
        "meet",
        "include",
        "continue",
        "learn",
        "change",
        "lead",
        "understand",
        "watch",
        "follow",
        "stop",
        "create",
        "speak",
        "spend",
        "grow",
        "open",
        "walk",
        "win",
        "offer",
        "remember",
        "love",
        "consider",
        "appear",
        "buy",
        "wait",
        "serve",
        "die",
        "send",
        "expect",
        "build",
        "stay",
        "fall",
        "cut",
        "reach",
        "remain",
        "suggest",
        "raise",
        "pass",
        "sell",
        "require",
        "report",
        "decide",
        "pull",
        "explain",
        "hope",
        "develop",
        "carry",
        "break",
        "receive",
        "agree",
        "support",
        "hit",
        "produce",
        "eat",
        "cover",
        "catch",
        "draw",
        "choose",
    }
)


class SpeechActType(Enum):
    """The type of speech act performed by an utterance.

    Following Searle (1969), speech acts are the actions performed
    by speaking. The comprehension engine identifies the illocutionary
    force of an utterance — what the speaker is *doing* by saying it.
    """

    STATEMENT = auto()  # asserting a fact or opinion
    QUESTION = auto()  # requesting information
    COMMAND = auto()  # requesting/demanding action
    EXCLAMATION = auto()  # expressing strong emotion
    GREETING = auto()  # social ritual (hello, goodbye)
    REQUEST = auto()  # polite request (can you...?)
    PROMISE = auto()  # committing to future action
    APOLOGY = auto()  # expressing regret

    @property
    def label(self) -> str:
        """Human-readable label for this speech act type."""
        labels = {
            SpeechActType.STATEMENT: "statement",
            SpeechActType.QUESTION: "question",
            SpeechActType.COMMAND: "command",
            SpeechActType.EXCLAMATION: "exclamation",
            SpeechActType.GREETING: "greeting",
            SpeechActType.REQUEST: "request",
            SpeechActType.PROMISE: "promise",
            SpeechActType.APOLOGY: "apology",
        }
        return labels.get(self, "unknown")


class SemanticRole(Enum):
    """Thematic roles in a proposition.

    Thematic roles (also called theta roles) describe the semantic
    relationship between a verb and its arguments. They answer the
    question "who did what to whom, with what, where, and when?"

    This is a simplified set based on the thematic role hierarchy
    from Jackendoff (1990) and Dowty (1991).
    """

    AGENT = auto()  # the entity performing the action
    PATIENT = auto()  # the entity affected by the action
    EXPERIENCER = auto()  # the entity experiencing a mental state
    THEME = auto()  # the entity in a state or being described
    INSTRUMENT = auto()  # the tool used to perform the action
    LOCATION = auto()  # where the action occurs
    TIME = auto()  # when the action occurs
    GOAL = auto()  # the destination of an action
    SOURCE = auto()  # the origin of an action
    BENEFICIARY = auto()  # the entity for whom the action is done
    ATTRIBUTE = auto()  # a property or characteristic being asserted

    @property
    def label(self) -> str:
        """Human-readable label for this semantic role."""
        labels = {
            SemanticRole.AGENT: "agent",
            SemanticRole.PATIENT: "patient",
            SemanticRole.EXPERIENCER: "experiencer",
            SemanticRole.THEME: "theme",
            SemanticRole.INSTRUMENT: "instrument",
            SemanticRole.LOCATION: "location",
            SemanticRole.TIME: "time",
            SemanticRole.GOAL: "goal",
            SemanticRole.SOURCE: "source",
            SemanticRole.BENEFICIARY: "beneficiary",
            SemanticRole.ATTRIBUTE: "attribute",
        }
        return labels.get(self, "unknown")


@dataclass(slots=True)
class Proposition:
    """A single proposition — a subject-predicate-object triple.

    A proposition is the smallest unit of meaning that can be true or
    false. "The cat sat on the mat" contains one proposition:
    (cat, sat, mat). "John gave Mary a book" contains one proposition
    with additional roles: (John, gave, book) with Mary as beneficiary.

    Attributes:
        subject: The entity performing or being described by the
            predicate (the "who").
        predicate: The action or relationship (the "did what" or
            "is what").
        object: The entity acted upon or related to (the "to whom"
            or "what"). May be empty for intransitive sentences.
        roles: Additional semantic roles mapped to their fillers.
            E.g., {SemanticRole.LOCATION: "the park"} for "in the park".
        negated: Whether this proposition is negated ("not", "never").
        tense: The tense of the predicate — "past", "present",
            "future", or "unknown".
        passive: Whether the clause is in passive voice ("water is
            needed by plants" — subject is the patient, the AGENT
            role holds the "by"-phrase).
        verb_found: False when no verb could be identified and the
            whole clause was stored as the subject with a stub "is"
            predicate — the signature of a failed parse. The
            self-monitor uses this for its inner-loop check.
        focus: For interrogative clauses, the questioned
            constituent — the wh-word ("what", "who", ...) or
            "whether" for yes/no questions. The focus is the gap the
            speaker wants filled, not an argument of the proposition.
    """

    subject: str = ""
    predicate: str = ""
    object: str = ""
    roles: dict[SemanticRole, str] = field(default_factory=dict)
    negated: bool = False
    tense: str = "unknown"
    passive: bool = False
    verb_found: bool = True
    focus: str = ""


@dataclass(slots=True)
class ConceptRole:
    """A key concept and its semantic role in the sentence.

    Attributes:
        concept: The word or phrase that is a key concept.
        role: The semantic role this concept plays in the sentence.
        salience: How important this concept is to the sentence's
            meaning (0–1). Subjects and main verbs have high salience;
            modifiers and function words have low salience.
    """

    concept: str
    role: SemanticRole
    salience: float = 0.5


@dataclass(slots=True)
class ComprehensionResult:
    """A structured understanding of what was said.

    This is the output of the comprehension engine — a complete
    representation of the meaning of an input text, ready for
    downstream processing by pragmatics.py, figurative.py, and the
    cognition engine.

    Attributes:
        raw_text: The original input text.
        speech_act: The identified speech act type.
        propositions: Extracted propositions (subject-predicate-object
            triples), one per clause.
        key_concepts: Key concepts and their semantic roles.
        resolved_references: A mapping from pronoun → resolved
            referent. Empty if no pronouns were found or if no
            context was available for resolution.
        is_negated: Whether the overall meaning is negated.
        is_question: Whether this is a question (convenience flag).
        is_command: Whether this is a command (convenience flag).
        focus: For questions, the questioned constituent — the
            wh-word ("what", "who", "where", ...) or "whether" for
            yes/no questions. Empty for statements.
        confidence: Overall confidence in the parse (0–1). Lower
            for complex or ambiguous sentences.
        clause_count: Number of clauses identified.
        metaphor: Detected metaphor, if any (right-hemisphere
            cross-domain mapping).
        irony: Detected irony, if any (sentiment-context contradiction).
        idiom: Detected idiom as (idiom, meaning) tuple, if any.
    """

    raw_text: str = ""
    speech_act: SpeechActType = SpeechActType.STATEMENT
    propositions: list[Proposition] = field(default_factory=list)
    # Grounded semantic bindings are populated by the cognition layer.\n    # Kept separate from propositions so linguistic parsing remains\n    # language-only.\n    grounded: list[Any] = field(default_factory=list)\n    key_concepts: list[ConceptRole] = field(default_factory=list)
    resolved_references: dict[str, str] = field(default_factory=dict)
    is_negated: bool = False
    is_question: bool = False
    is_command: bool = False
    focus: str = ""
    confidence: float = 0.5
    clause_count: int = 1
    metaphor: Metaphor | None = None
    irony: IronyDetection | None = None
    idiom: tuple[str, str] | None = None


class ComprehensionEngine:
    """Parse and understand input text — Genesis's receptive language.

    This is the Wernicke's area equivalent: it takes raw text and
    produces a structured comprehension of what was said. The
    comprehension includes speech act identification, proposition
    extraction, reference resolution, and key concept identification.

    The engine uses a heuristic parser — it doesn't require external
    NLP libraries. It handles common sentence structures (declarative,
    interrogative, imperative) and resolves pronouns using a simple
    recency-based strategy with context provided by the caller.

    Usage::

        engine = ComprehensionEngine()
        result = engine.comprehend(
            "It gave the book to him yesterday.",
            context={"recent_entities": ["Mary", "John"]},
        )
        # result.speech_act → STATEMENT
        # result.propositions[0].subject → "Mary" (resolved from "she")
        # result.propositions[0].object → "book"
        # result.resolved_references → {"she": "Mary", "him": "John"}
    """

    def __init__(self) -> None:
        """Initialize with an empty recent-entity list for pronoun resolution."""
        # Track recent entities for pronoun resolution across calls
        self._recent_entities: list[str] = []

        # Learned multi-word chunks (from the statistical language
        # acquisition system). Chunks like "trial and error" or
        # "salt and pepper" protect coordinator splitting — the
        # conjunction inside a learned chunk does not mark a clause
        # boundary.
        self._known_chunks: frozenset[str] = frozenset()

        # Figurative language processor — right-hemisphere contribution
        # to comprehension. Detects metaphors (cross-domain mappings),
        # irony (sentiment-context contradictions), and idioms
        # (non-compositional expressions). Runs after the basic
        # semantic parse to enrich the comprehension result.
        self._figurative = FigurativeLanguageProcessor()

    def comprehend(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> ComprehensionResult:
        """Comprehend input text into a structured representation.

        Args:
            text: The input text to comprehend.
            context: Optional context dict that may contain:
                - "recent_entities": list of recently mentioned
                  entities (for pronoun resolution)
                - "speaker": who is speaking (affects pronoun
                  resolution — "I" resolves to the speaker)
                - "known_facts": dict of known facts for consistency
                  checking

        Returns:
            A ComprehensionResult with the structured understanding.
        """
        if not text or not text.strip():
            return ComprehensionResult(raw_text=text or "", confidence=0.0)

        context = context or {}
        raw_text = text.strip()

        # Update recent entities from context
        ctx_entities = context.get("recent_entities", [])
        if ctx_entities:
            self._recent_entities = list(ctx_entities)

        # Split into clauses (sentences)
        clauses = self._split_clauses(raw_text)

        # Identify the speech act of the first clause (dominant)
        speech_act = self._identify_speech_act(raw_text)

        # Extract propositions from each clause
        propositions: list[Proposition] = []
        focus = ""
        for clause in clauses:
            # Shared-subject VP coordination ("I came and saw the
            # water") and gapped VP coordination ("put the star in
            # its hole and the moon in its slot") each expand into
            # one clause per conjunct, all getting the full
            # extraction pipeline.
            for gapped in self._expand_gapped_coordination(clause):
                for sub_clause in self._expand_vp_coordination(gapped):
                    prop = self._extract_proposition(sub_clause)
                    if prop.subject or prop.predicate:
                        propositions.append(prop)
                    # Record the questioned constituent — the gap the
                    # speaker wants filled, not an argument of the
                    # proposition.
                    if not focus and prop.focus:
                        focus = prop.focus

        # Restrictive relative clauses embedded in an object ("a
        # machine that has two parts") carry a second proposition —
        # the relativized clause is about the head noun, not part of
        # the NP string. Split it out so both facts are available.
        propositions = self._extract_relative_propositions(propositions)

        # Resolve pronoun references
        speaker = context.get("speaker", "user")
        resolved = self._resolve_references(propositions, speaker)

        # Identify key concepts and their roles
        key_concepts = self._identify_key_concepts(propositions, speech_act)

        # Determine negation
        is_negated = any(p.negated for p in propositions)

        # Confidence based on how much we could parse
        confidence = self._compute_confidence(propositions, clauses, speech_act)

        # Update recent entities for future calls
        self._update_recent_entities(propositions, resolved)

        # ── Figurative language detection (right-hemisphere) ──
        # After the basic semantic parse, run metaphor, irony, and
        # idiom detection. These enrich the comprehension result with
        # cross-domain mappings and non-literal meanings that the
        # left-hemisphere (proposition extraction) would miss.
        metaphor = self._figurative.detect_metaphor(raw_text)
        irony = self._figurative.detect_irony(raw_text)
        idiom = self._figurative.detect_idiom(raw_text)

        result = ComprehensionResult(
            raw_text=raw_text,
            speech_act=speech_act,
            propositions=propositions,
            key_concepts=key_concepts,
            resolved_references=resolved,
            is_negated=is_negated,
            is_question=speech_act in (SpeechActType.QUESTION, SpeechActType.REQUEST),
            is_command=speech_act == SpeechActType.COMMAND,
            focus=focus,
            confidence=confidence,
            clause_count=len(clauses),
            metaphor=metaphor,
            irony=irony,
            idiom=idiom,
        )

        return result

    # ─── Speech act identification ──────────────────────────────

    def _is_greeting(self, text_lower: str) -> bool:
        """Detect greeting/farewell phrases.

        Matches whole-word greetings and farewells, including when
        followed by punctuation or additional words.
        """
        greeting_words = {
            "hello",
            "hi",
            "hey",
            "greetings",
            "howdy",
            "good morning",
            "good afternoon",
            "good evening",
            "bye",
            "goodbye",
            "farewell",
            "see you",
        }
        for gw in greeting_words:
            if (
                text_lower == gw
                or text_lower.startswith(gw + " ")
                or text_lower.startswith(gw + "!")
            ):
                return True
        return False

    def _identify_speech_act(self, text: str) -> SpeechActType:
        """Identify the speech act type of the text.

        Uses punctuation, word patterns, and sentence structure to
        determine what the speaker is doing: asking, telling,
        commanding, etc.
        """
        text_lower = text.lower().strip()

        # Greeting detection
        if self._is_greeting(text_lower):
            return SpeechActType.GREETING

        # Exclamation detection (ends with ! and is short)
        if text.strip().endswith("!") and len(text.split()) <= 5:
            return SpeechActType.EXCLAMATION

        # Question detection (ends with ?)
        if text.strip().endswith("?"):
            # Check for request patterns
            request_patterns = {
                "can you",
                "could you",
                "would you",
                "will you",
                "please",
                "mind if",
            }
            for pattern in request_patterns:
                if pattern in text_lower:
                    return SpeechActType.REQUEST
            return SpeechActType.QUESTION

        # Command/imperative detection
        words = text_lower.split()
        if words:
            first_word = words[0].rstrip(",.!?;:")
            # Imperative: starts with a base-form verb, no explicit subject
            if first_word in _IMPERATIVE_STARTERS:
                return SpeechActType.COMMAND
            # "Let's" or "Let me" constructions
            if first_word == "let" and len(words) > 1:
                return SpeechActType.COMMAND

        # Apology detection
        apology_words = {
            "sorry",
            "apologize",
            "apologies",
            "my fault",
            "i regret",
            "pardon",
            "excuse me",
        }
        for aw in apology_words:
            if aw in text_lower:
                return SpeechActType.APOLOGY

        # Promise detection
        promise_patterns = {"i will", "i promise", "i'll", "i shall", "i guarantee", "i commit"}
        for pp in promise_patterns:
            if pp in text_lower:
                return SpeechActType.PROMISE

        # Default: statement
        return SpeechActType.STATEMENT

    # ─── Clause splitting ───────────────────────────────────────

    def set_known_chunks(self, chunks) -> None:
        """Provide learned multi-word chunks that protect splitting.

        The statistical acquisition system learns multi-word units
        like "trial and error" or "salt and pepper" from user input.
        A coordinator inside a learned chunk is part of the unit,
        not a clause boundary — this is how exposure teaches it
        that "salt and pepper" is one thing, not two clauses.
        """
        self._known_chunks = frozenset(
            c.lower().strip() for c in chunks if c and c.strip()
        )

    def _split_clauses(self, text: str) -> list[str]:
        """Split text into clauses (rough sentence segmentation).

        Splits on sentence-ending punctuation and on coordinating
        conjunctions *that join clauses*. A coordinator only marks
        a boundary when the right-hand side looks like a clause —
        it contains a finite verb and starts with a subject-like
        word. This protects:

        - NP coordination: "salt and pepper", "mind and body"
        - VP coordination sharing the subject: "I came and saw"
        - Learned chunks: "trial and error", "give and take"
        """
        sentences = _SENTENCE_SPLIT_RE.split(text)
        clauses: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            clauses.extend(self._split_coordinate(sentence))
        return clauses if clauses else [text.strip()]

    # Coordinators that can join clauses or smaller units
    _COORDINATOR_RE = re.compile(r"\s+(but|so|yet|and|or)\s+", re.IGNORECASE)

    def _split_coordinate(self, sentence: str) -> list[str]:
        """Split a sentence at coordinators that introduce clauses."""
        parts = [sentence]
        # Rescan until stable — splitting "A, and B, and C" needs
        # repeated passes since each boundary is checked separately.
        changed = True
        while changed:
            changed = False
            next_parts: list[str] = []
            for part in parts:
                split = self._split_at_coordinator(part)
                if len(split) > 1:
                    changed = True
                next_parts.extend(split)
            parts = next_parts
        return [p.strip() for p in parts if p.strip()]

    def _split_at_coordinator(self, part: str) -> list[str]:
        """Split at the first coordinator that introduces a clause.

        Returns [part] unchanged when every coordinator joins
        smaller units (NPs, VPs) rather than clauses.
        """
        for m in self._COORDINATOR_RE.finditer(part):
            left = part[: m.start()]
            right = part[m.end():]
            if not left.strip() or not right.strip():
                continue

            # Protect learned multi-word chunks: "trial and error"
            left_last = left.split()[-1].lower() if left.split() else ""
            right_first = right.split()[0].lower() if right.split() else ""
            if f"{left_last} {m.group(1).lower()} {right_first}" in self._known_chunks:
                continue

            # The left side must already contain a finite verb —
            # otherwise the coordinator joins noun phrases that
            # share the following verb ("the dog and the cat ran").
            if self._find_main_verb(left.split())[0] < 0:
                continue

            # The right side must contain a finite verb with a
            # subject-like word before it. "and saw" (VP
            # coordination sharing the subject) and "and pepper"
            # (NP coordination) both fail this test.
            if not self._right_is_clause(right):
                continue

            return [left, right]
        return [part]

    # Verb forms that agree with a plural (or compound) subject.
    # When a coordinator's right side starts "noun + one of these"
    # ("salt and pepper are..."), the compound — not the right noun
    # alone — is the likely subject, so don't split.
    _PLURAL_AGREEING = frozenset({
        "are", "were", "have", "do", "don't",
        "aren't", "weren't", "haven't",
    })

    # Adverbs that can open a shared-subject VP ("and quickly left")
    _SPLIT_ADVERBS = frozenset({
        "then", "also", "just", "never", "always", "often", "soon",
        "now", "later", "still", "even", "only", "again", "finally",
        "suddenly", "immediately", "quickly", "slowly", "simply",
        "really", "very", "too",
    })

    def _right_is_clause(self, right: str) -> bool:
        """Return True if the text after a coordinator is clause-like.

        A clause needs a finite verb AND a subject-like word before
        it (a pronoun, determiner, or noun). "and he saw" is a
        clause; "and saw it" (subject shared), "and quickly left"
        (adverb + shared-subject verb), and "and pepper" (no verb)
        are not.
        """
        words = right.split()
        if len(words) < 2:
            return False
        first = words[0].lower().rstrip(",.!?;:")
        # Verb-initial right side → shared-subject VP coordination
        if first in _COPULAR_VERBS or self._looks_like_verb(first):
            return False
        # Adverb + verb → still shared-subject VP coordination
        if len(words) >= 2:
            second = words[1].lower().rstrip(",.!?;:")
            if (
                first in self._SPLIT_ADVERBS or first.endswith("ly")
            ) and self._looks_like_verb(second):
                return False
            # NP-conjunction cue: "NOUN + plural-agreeing verb" means
            # the coordinator probably joined two noun phrases whose
            # compound is the real subject ("salt and pepper ARE on
            # the table" — "pepper" alone would take "is"). Pronouns
            # and capitalized names still open clauses ("and it
            # left", "and Mary is here").
            if (
                second in self._PLURAL_AGREEING
                and not words[0][0].isupper()
                and first not in _SUBJECT_PRONOUNS
                and first not in _OBJECT_PRONOUNS
            ):
                return False
        # Otherwise: a clause if any finite verb is present. A word
        # directly preceded by a pure determiner is a noun, not verb
        # evidence — "and a spring" is an NP ("spring" the noun),
        # while "and the results came" keeps "came" as its verb.
        for j, w in enumerate(words):
            wl = w.lower().rstrip(",.!?;:")
            if j > 0 and words[j - 1].lower().rstrip(",.!?;:") in self._PURE_DETERMINERS:
                continue
            if wl in _COPULAR_VERBS or wl in _AUX_VERBS or self._looks_like_verb(wl):
                return True
        return False

    # Coordinating conjunctions (mirrors _COORDINATOR_RE)
    _COORDINATORS = frozenset({"and", "or", "but", "yet", "so"})

    def _expand_vp_coordination(self, clause: str) -> list[str]:
        """Expand shared-subject VP coordination into clauses.

        "I came and saw the water" → ["I came", "I saw the water"]:
        when a coordinator's right side opens a verb phrase, the
        subject (and any auxiliaries) before the left verb are
        shared by both conjuncts, so each becomes a full clause.
        NP coordination ("cats and dogs") and learned chunks are
        left alone — their right side does not open a verb phrase.
        """
        words = clause.split()
        for i, w in enumerate(words):
            wl = w.lower().rstrip(",.!?;:")
            if wl not in self._COORDINATORS:
                continue
            left, right = words[:i], words[i + 1:]
            if not left or not right:
                continue
            # Protect learned multi-word chunks ("trial and error")
            chunk = (
                f"{left[-1].lower().rstrip(',.!?;:')} {wl} "
                f"{right[0].lower().rstrip(',.!?;:')}"
            )
            if chunk in self._known_chunks:
                continue
            if not self._opens_verb_phrase(right):
                continue
            vidx, _, _ = self._find_main_verb(left)
            if vidx < 0:
                continue
            prefix = left[:vidx]
            right_clause = " ".join([*prefix, *right])
            return [
                " ".join(left),
                *self._expand_vp_coordination(right_clause),
            ]
        return [clause]

    def _opens_verb_phrase(self, words: list[str]) -> bool:
        """True when the word list starts a verb phrase: a verb,
        optionally preceded by adverbs, auxiliaries, or negation
        ("saw it", "quickly left", "didn't see")."""
        for w in words:
            wl = w.lower().rstrip(",.!?;:")
            if wl in self._SPLIT_ADVERBS or (
                wl.endswith("ly") and len(wl) > 3
            ):
                continue
            if wl in _AUX_VERBS or wl in _NEGATION_WORDS:
                continue
            return wl in _COPULAR_VERBS or self._looks_like_verb(wl)
        return False

    def _expand_gapped_coordination(self, clause: str) -> list[str]:
        """Expand gapped VP coordination into full clauses.

        "put the star in its hole and the moon in its slot" is two
        instructions sharing the verb — the second conjunct elides
        it. The clause splitter and VP expander both rightly decline
        (the right side has no verb and does not open a VP), leaving
        the object parser to swallow the second object inside the
        first prepositional phrase. The signal that distinguishes a
        gap from NP coordination: the right side is "NP preposition
        ..." where that preposition also appears after the left
        clause's verb — "the moon IN its slot" repeats "star IN its
        hole". When it does, the left verb is re-copied over the
        right conjunct.
        """
        words = clause.split()
        for i, w in enumerate(words):
            wl = w.lower().rstrip(",.!?;:")
            if wl not in self._COORDINATORS:
                continue
            left, right = words[:i], words[i + 1:]
            if not left or not right:
                continue
            chunk = (
                f"{left[-1].lower().rstrip(',.!?;:')} {wl} "
                f"{right[0].lower().rstrip(',.!?;:')}"
            )
            if chunk in self._known_chunks:
                continue
            vidx, _, _ = self._find_main_verb(left)
            # Only subjectless (imperative) lefts are safe to copy:
            # with a subject present the right conjunct's NP is
            # ambiguous between shared-subject object ("she read a
            # book and a magazine on the porch") and new-subject
            # remnant ("the cat sat ... and the dog on the rug").
            if vidx != 0:
                continue
            # A right side that is clause-like or VP-initial belongs
            # to the other coordination passes.
            if self._opens_verb_phrase(right) or self._right_is_clause(
                " ".join(right)
            ):
                continue
            # The right side must be "NP + prepositional phrase":
            # a preposition inside, but not leading, the conjunct.
            rprep = next(
                (
                    j
                    for j, rw in enumerate(right)
                    if rw.lower().rstrip(",.!?;:") in _PREPOSITIONS
                ),
                None,
            )
            if rprep is None or rprep == 0:
                continue
            prep = right[rprep].lower().rstrip(",.!?;:")
            left_post_verb = {
                x.lower().rstrip(",.!?;:") for x in left[vidx + 1 :]
            }
            if prep not in left_post_verb:
                continue
            right_clause = " ".join([*left[: vidx + 1], *right])
            return [
                " ".join(left),
                *self._expand_gapped_coordination(right_clause),
            ]
        return [clause]

    # ─── Proposition extraction ─────────────────────────────────

    # Prepositions that cliticize onto specific verbs (phrasal verbs)
    # rather than opening a free prepositional phrase. "depend on X"
    # treats "on" as part of the predicate; "sit on X" does not —
    # "on" opens a LOCATION phrase. The mapping is by base verb.
    _PHRASAL_VERBS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "depend": (("on",),),
        "rely": (("on",),),
        "insist": (("on",),),
        "focus": (("on",),),
        "work": (("on",), ("out",)),
        "act": (("on",),),
        "emerge": (("from",),),
        "stem": (("from",),),
        "derive": (("from",),),
        "come": (("from",),),
        "differ": (("from",),),
        "suffer": (("from",),),
        "consist": (("of",),),
        "think": (("about",), ("of",)),
        "care": (("about",), ("for",)),
        "wonder": (("about",),),
        "ask": (("about",), ("for",)),
        "relate": (("to",),),
        "connect": (("to",), ("with",)),
        "lead": (("to",),),
        "belong": (("to",),),
        "point": (("to",), ("out",)),
        "respond": (("to",),),
        "amount": (("to",),),
        "contribute": (("to",),),
        "listen": (("to",),),
        "wait": (("for",),),
        "aim": (("to",), ("for",)),
        "account": (("for",),),
        "agree": (("with",), ("to",), ("on",)),
        "deal": (("with",),),
        "participate": (("in",),),
        "result": (("in",), ("from",),),
        "look": (("like",), ("for",), ("up",)),
        "give": (("rise", "to"), ("back",), ("away",), ("out",), ("up",)),
        "take": (("care", "of"), ("out",), ("back",), ("off",), ("up",), ("over",), ("away",)),
        "rise": (("to",),),
        # Separable phrasal verbs — the particle belongs to the verb,
        # not to a prepositional phrase ("picked up the milk" → object
        # is "the milk", not "up the milk"). Only particles immediately
        # following the verb are absorbed, so split orders ("put the
        # book down") are unaffected.
        "pick": (("up",),),
        "put": (("down",), ("on",), ("away",), ("back",), ("off",), ("out",), ("up",)),
        "bring": (("back",), ("in",), ("up",), ("out",)),
        "carry": (("out",),),
        "set": (("down",), ("aside",), ("up",), ("off",)),
        "throw": (("away",), ("out",),),
        "hand": (("in",), ("out",), ("over",)),
        "drop": (("off",),),
        "write": (("down",),),
        "clean": (("up",),),
        "fill": (("in",), ("up",), ("out",)),
        "turn": (("on",), ("off",), ("up",), ("down",)),
        "hang": (("up",),),
        "plug": (("in",),),
        "lock": (("up",),),
        "heat": (("up",),),
        "open": (("up",),),
        "shut": (("down",), ("off",)),
        "close": (("down",),),
        "cover": (("up",),),
        "mix": (("up",),),
        "tear": (("up",), ("down",)),
        "cut": (("off",),),
        "break": (("down",),),
        "blow": (("up",),),
        "burn": (("down",), ("up",)),
        "dry": (("off",),),
        "wake": (("up",),),
        "send": (("back",), ("off",)),
        "pay": (("back",),),
        "trade": (("in",),),
        "show": (("off",),),
        "check": (("out",),),
        "figure": (("out",),),
        "find": (("out",),),
        "rule": (("out",),),
        "sort": (("out",),),
        "try": (("out",),),
        "leave": (("out",),),
        "call": (("off",),),
        "eat": (("up",),),
        "drink": (("up",),),
        "fix": (("up",),),
        "calm": (("down",),),
        "cool": (("down",),),
        "hold": (("up",),),
        "back": (("up",),),
        "get": (("back",),),
    }

    # Determiners that open a noun phrase
    _DETERMINERS = frozenset({
        "the", "a", "an", "this", "that", "these", "those",
        "my", "your", "his", "her", "its", "our", "their",
        "some", "any", "each", "every", "no", "all", "both",
    })

    # Determiners that can ONLY be determiners — a word after one is
    # a noun, never a verb ("the work", "my plan"). Ambiguous
    # determiners like "this"/"that"/"some" double as pronouns
    # ("this works") and are excluded from this set.
    _PURE_DETERMINERS = frozenset({
        "the", "a", "an",
    })

    def _extract_proposition(self, clause: str) -> Proposition:
        """Extract a subject-predicate-object triple from a clause.

        Uses a heuristic parser that identifies the main verb, then
        splits the clause into subject (before verb) and object
        (after verb). Handles copular verbs (is/are/was/were),
        common auxiliary constructions, questions (wh-focus and
        yes/no), passive voice, and "there is/are" expletives.
        """
        is_question = clause.strip().endswith("?")

        # Clean the clause
        clause = clause.strip().rstrip(".!?")
        if not clause:
            return Proposition()

        words = clause.split()
        if not words:
            return Proposition()

        # Detect negation
        negated = False
        for w in words:
            if w.lower() in _NEGATION_WORDS:
                negated = True
                break

        # Detect tense
        tense = self._detect_tense(words)

        first = words[0].lower().rstrip(",.!?;:")

        # ── Interrogative and expletive clause forms ──
        # These have non-canonical word order: the questioned
        # constituent or expletive sits where the subject would be.
        if is_question and first in _WH_WORDS:
            return self._extract_wh_proposition(
                words, first, negated, tense
            )
        if is_question and first in _AUX_VERBS | _COPULAR_VERBS:
            return self._extract_yesno_proposition(
                words, first, negated, tense
            )
        if first == "there" and len(words) > 1 and words[1].lower() in _COPULAR_VERBS:
            return self._extract_there_proposition(words, negated, tense)

        # Find the main verb position
        verb_idx, verb, is_copular = self._find_main_verb(words)

        if verb_idx == -1:
            # No verb found — treat the whole thing as a subject/theme.
            # verb_found=False marks this as a failed parse so the
            # self-monitor's inner loop can catch malformed output.
            return Proposition(
                subject=clause,
                predicate="is",
                tense=tense,
                negated=negated,
                verb_found=False,
            )

        # Extract subject (words before the verb)
        subject = self._extract_subject(words, verb_idx)

        # Extract predicate (the verb itself), absorbing phrasal
        # particles that belong to it ("depends on", "gives rise to")
        predicate, consumed = self._absorb_particles(words, verb_idx, verb)

        # Extract object (words after the predicate)
        remaining = words[verb_idx + 1 + consumed :]

        passive = self._detect_passive(is_copular, remaining)

        obj, roles = self._parse_object_and_roles(
            remaining, is_copular, by_is_agent=passive
        )

        # For copular verbs, the "object" is an attribute — except
        # in passives, where the participle joins the predicate.
        if passive:
            predicate = f"{predicate} {obj}".strip()
            obj = ""
        elif is_copular and obj:
            roles[SemanticRole.ATTRIBUTE] = obj

        return Proposition(
            subject=subject,
            predicate=predicate,
            object=obj,
            roles=roles,
            negated=negated,
            tense=tense,
            passive=passive,
        )

    def _extract_subject(self, words: list[str], verb_idx: int) -> str:
        """Extract the subject from the words before the main verb."""
        # Filter out leading auxiliaries and negation contractions
        # from subject ("I don't believe" → subject="I", not "I don't")
        subject_words = [
            w for w in words[:verb_idx]
            if w.lower() not in _AUX_VERBS
            and w.lower() not in _NEGATION_WORDS
        ]
        # A word immediately before the verb that looks like an
        # adverb is a pre-verbal modifier, not part of the subject
        # ("I quickly left" → subject "I").
        while subject_words and (
            subject_words[-1].lower().rstrip(",.!?;:")
            in self._SPLIT_ADVERBS
            or self._looks_like_adverb(
                subject_words[-1].lower().rstrip(",.!?;:")
            )
        ):
            subject_words.pop()
        return " ".join(subject_words).strip() if subject_words else ""

    def _detect_passive(
        self, is_copular: bool, remaining: list[str]
    ) -> bool:
        """Detect passive voice: copula + past participle.

        "water is needed by plants" — the complement is a participle
        and a following "by" phrase names the agent, not a location.
        A negation or adverb can intervene ("is not needed", "was
        quickly forgotten") — skip them to reach the participle.
        """
        if not is_copular or not remaining:
            return False
        for k, cand in enumerate(remaining[:3]):
            cl = cand.lower().rstrip(",.!?;:")
            if (
                cl in _NEGATION_WORDS
                or cl in self._SPLIT_ADVERBS
                or self._looks_like_adverb(cl)
            ):
                continue
            # A relation word taking its own complement is locative,
            # not a participle — "the star IS LEFT OF the moon" is
            # position, not the passive of "leave".
            if (
                cl in self._POSTNOMINAL_REL
                and k + 1 < len(remaining)
                and remaining[k + 1].lower().rstrip(",.!?;:")
                in self._REL_COMPLEMENT_PREPS
            ):
                return False
            return is_participle(cl)
        return False

    def _extract_wh_proposition(
        self, words: list[str], wh: str, negated: bool, tense: str
    ) -> Proposition:
        """Extract a proposition from a wh-question.

        The wh-word marks the questioned constituent — a gap, not
        the subject. Three shapes:

        - "what is water" → wh + copula: the post-copular NP is the
          subject; the wh-word questions its attribute.
        - "who created genesis" → wh + verb: the wh-word fills the
          subject gap; the rest is predicate + object.
        - "what do you want" → wh + aux: the subject follows the
          auxiliary and the clause parses normally.
        """
        rest = words[1:]
        if not rest:
            return Proposition(predicate="is", focus=wh, verb_found=True)

        w1 = rest[0].lower().rstrip(",.!?;:")

        # wh + copula: "what is water" → subject="water"
        if w1 in _COPULAR_VERBS:
            subject = " ".join(rest[1:]).strip()
            return Proposition(
                subject=subject,
                predicate=w1,
                object="",
                negated=negated,
                tense=tense,
                focus=wh,
            )

        # wh + aux: "what do you want" → parse the declarative core
        # after the auxiliary as a normal clause.
        if w1 in _AUX_VERBS or w1 in _NEGATION_WORDS:
            prop = self._extract_proposition(" ".join(rest[1:]))
            prop.focus = wh
            prop.negated = prop.negated or negated
            if prop.tense == "unknown":
                prop.tense = tense
            return prop

        # wh + verb: "who created genesis" — the wh-word fills the
        # subject gap. The verb is at rest[0].
        if is_verb_form(w1):
            obj, roles = self._parse_object_and_roles(rest[1:], False)
            return Proposition(
                subject="",
                predicate=w1,
                object=obj,
                roles=roles,
                negated=negated,
                tense=tense,
                focus=wh,
            )

        # wh + bare NP: "what water" — treat the rest as subject.
        return Proposition(
            subject=" ".join(rest),
            predicate="is",
            negated=negated,
            tense=tense,
            focus=wh,
            verb_found=False,
        )

    def _extract_yesno_proposition(
        self, words: list[str], first: str, negated: bool, tense: str
    ) -> Proposition:
        """Extract a proposition from a yes/no question.

        "is water wet" → subject="water", attribute="wet".
        "do you like music" → parse the declarative core
        ("you like music") normally.
        """
        rest = words[1:]
        if not rest:
            return Proposition(predicate=first, focus="whether")

        # Copula-initial: the subject NP follows the copula, then the
        # complement. "is water wet" → subj="water", attr="wet".
        # The complement starts at the first complement-signal word —
        # a determiner ("is water a liquid"), an adjective/adverb
        # ("is water really wet"), a predicate possessive ("is the
        # ball yours"), a preposition ("is the cat on the mat"), or
        # a verb form ("is the water running"). With no signal, the
        # last word is the complement ("is water wet").
        if first in _COPULAR_VERBS:
            comp_start = -1
            for i in range(1, len(rest)):
                wl = rest[i].lower().rstrip(",.!?;:")
                if (
                    wl in self._DETERMINERS
                    or wl in _PREPOSITIONS
                    or wl in _NEGATION_WORDS
                    or wl in _PREDICATE_POSSESSIVES
                    or self._looks_like_adjective(wl)
                    or self._looks_like_adverb(wl)
                    or is_verb_form(wl)
                ):
                    comp_start = i
                    break
            if comp_start < 0:
                comp_start = len(rest) - 1
            subject = " ".join(rest[:comp_start])
            complement = " ".join(rest[comp_start:])
            roles: dict[SemanticRole, str] = {}
            if complement:
                roles[SemanticRole.ATTRIBUTE] = complement
            return Proposition(
                subject=subject,
                predicate=first,
                object=complement,
                roles=roles,
                negated=negated,
                tense=tense,
                focus="whether",
            )

        # Aux-initial: "do you like music" → declarative core is
        # the rest ("you like music") parsed as a statement.
        prop = self._extract_proposition(" ".join(rest))
        prop.focus = "whether"
        prop.negated = prop.negated or negated
        return prop

    def _extract_there_proposition(
        self, words: list[str], negated: bool, tense: str
    ) -> Proposition:
        """Extract a proposition from "there is/are X" expletives.

        "there" is a dummy subject — the real theme is the
        post-copular NP: "there is a cat on the mat" →
        subject="a cat", LOCATION="the mat".
        """
        copula = words[1].lower()
        remaining = words[2:]
        obj, roles = self._parse_object_and_roles(remaining, True)
        return Proposition(
            subject=obj,
            predicate=copula,
            object="",
            roles=roles,
            negated=negated,
            tense=tense,
        )

    def _absorb_particles(
        self, words: list[str], verb_idx: int, verb: str
    ) -> tuple[str, int]:
        """Absorb phrasal-verb particles into the predicate.

        "depends on memory" → predicate "depends on", object starts
        at "memory". "gives rise to X" → predicate "gives rise to".
        Returns (predicate, extra_words_consumed).
        """
        base = is_verb_form(verb) or verb.lower()
        patterns = self._PHRASAL_VERBS.get(base)
        if patterns is None:
            # The table is keyed by base verb; deconjugation reaches
            # bases that are missing from the verb lexicon ("picked"
            # → "pick" even when "pick" isn't a known verb).
            for candidate in deconjugate_verb(verb):
                patterns = self._PHRASAL_VERBS.get(candidate)
                if patterns is not None:
                    break
        if not patterns:
            return verb, 0

        following = [
            w.lower().rstrip(",.!?;:")
            for w in words[verb_idx + 1 : verb_idx + 4]
        ]
        for pattern in patterns:
            if tuple(following[: len(pattern)]) == pattern:
                parts = words[verb_idx : verb_idx + 1 + len(pattern)]
                return " ".join(parts), len(pattern)
        return verb, 0

    def _looks_like_adjective(self, word: str) -> bool:
        """Heuristic: adjective endings used to stop subject-NP growth
        in yes/no questions ("is water clear" — 'clear' is the
        attribute, not part of the subject)."""
        return word.endswith(
            ("able", "ible", "al", "ful", "ic", "ive", "less",
             "ous", "y", "ish", "ent", "ant")
        ) and len(word) > 3

    def _looks_like_adverb(self, word: str) -> bool:
        """Heuristic: -ly adverbs and degree words that open a
        complement ("is water really wet")."""
        if word.endswith("ly") and len(word) > 3:
            return True
        return word in {
            "very", "really", "quite", "so", "too", "rather",
            "somewhat", "fairly", "pretty", "truly",
        }

    def _find_main_verb(self, words: list[str]) -> tuple[int, str, bool]:
        """Find the main verb in a word list.

        Returns (index, verb, is_copular). Looks for the first
        non-auxiliary verb. Copular verbs (is/are/was/were) are
        identified separately because they take attributes rather
        than objects.

        Auxiliary handling: after an aux ("do", "will", "don't"),
        the *subject* can intervene in questions ("do YOU like
        music") — so words after the aux are skipped while they look
        like subject material (pronouns, determiners, nouns) until
        a verb form appears. This fixes the old bug where the word
        immediately after an aux was blindly taken as the verb
        ("do you like X" parsed "you" as the verb).
        """
        # Imperative detection: if the first word is a base-form verb,
        # the sentence is imperative and the first word IS the verb.
        # "Tell me about water" → verb="tell" at index 0. The gate is
        # the base-form lexicon plus the imperative starter set — not
        # _COMMON_VERBS, which is a mostly-inflected list that misses
        # ordinary base verbs ("sort the shapes" is a command).
        # Auxiliaries and copulas keep their own paths ("have fun"
        # reads as possession; "be quiet" keeps its copular frame).
        if words:
            first_lower = words[0].lower().rstrip(",.!?;:")
            if (
                first_lower not in _AUX_VERBS
                and first_lower not in _COPULAR_VERBS
                and (
                    first_lower in VERB_LEXICON
                    or first_lower in _IMPERATIVE_STARTERS
                )
            ):
                return 0, first_lower, False

        saw_aux = False
        post_aux_noun = -1
        lexical_aux = -1
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip(",.!?;:")

            # Pure negation words ("not", "never") are skipped — they
            # never take the verb slot.
            if word_lower in _NEGATION_WORDS and word_lower not in _AUX_VERBS:
                continue

            # Auxiliaries and negation contractions ("don't",
            # "doesn't", "won't", "can't") precede the main verb.
            if word_lower in _AUX_VERBS or word_lower in _NEGATION_WORDS:
                saw_aux = True
                # "have/has/had" and "do/does/did" double as lexical
                # verbs (possession, light verb). If no verb form
                # follows, the aux itself was the predicate —
                # "alpha has a lens", not "alpha has + lens(verb)".
                if word_lower in self._LEXICAL_AUX and lexical_aux < 0:
                    lexical_aux = i
                continue

            if word_lower in _COPULAR_VERBS:
                return i, word_lower, True

            if saw_aux:
                # After an auxiliary, subject material intervenes:
                # pronouns and determiners are part of the subject,
                # so keep scanning for a real verb form.
                if word_lower in _SUBJECT_PRONOUNS:
                    continue
                if word_lower in self._DETERMINERS:
                    # A determiner after a skipped content word opens
                    # a new NP — the skipped word was the verb
                    # ("will WATER the plants"). But a have/do aux
                    # claims the verb slot first: "has exactly TWO of
                    # these" is possession, not "exactly" the verb.
                    if post_aux_noun >= 0:
                        if lexical_aux >= 0:
                            return (
                                lexical_aux,
                                words[lexical_aux].lower().rstrip(",.!?;:"),
                                False,
                            )
                        return (
                            post_aux_noun,
                            words[post_aux_noun].lower().rstrip(",.!?;:"),
                            False,
                        )
                    continue
                if is_verb_form(word_lower) or self._looks_like_verb(word_lower):
                    return i, word_lower, False
                # Non-verb content word (a subject noun like "cats"
                # in "do cats sleep") — remember it; if no real verb
                # appears, it was the verb all along.
                if post_aux_noun < 0:
                    post_aux_noun = i
                continue

            # Check for common action verb patterns
            # We detect verbs by position: if we've seen some words
            # (the subject) and this word is not a pronoun, article,
            # or adjective, it's likely the verb.
            if i == 0:
                continue

            prev_lower = words[i - 1].lower().rstrip(",.!?;:")
            # If the previous word is a subject (noun/pronoun/article)
            # and this word is not a preposition or conjunction,
            # it might be the verb
            if prev_lower in _PREPOSITIONS or prev_lower in _CONJUNCTIONS:
                continue
            # A pure determiner or possessive means this word is a
            # noun — "the work", "my plan". Ambiguous determiners
            # ("this", "that", "some", "each") also function as
            # pronouns ("this works"), so they don't block.
            if (
                prev_lower in self._PURE_DETERMINERS
                or prev_lower in _POSSESSIVE_PRONOUNS
            ):
                continue
            if word_lower in _PREPOSITIONS or word_lower in _CONJUNCTIONS:
                continue
            # Heuristic: if this word is a known verb form
            if self._looks_like_verb(word_lower):
                # A verb-shaped word inside a determiner-headed NP
                # is the NP's head noun, not the predicate, when a
                # finite verb still follows — "a blue square block
                # IS on the table" parses block-as-noun + is, not
                # "a blue square" blocking the table.
                if self._inside_open_np(words, i):
                    continue
                return i, word_lower, False

        # Auxiliary with no verb form after it — either the aux was
        # itself the lexical verb (have/do-family: "alpha has a lens",
        # "she does yoga") or the skipped noun was the verb ("I will
        # water plants"). The lexical reading wins for have/do: a
        # non-verb-like complement under them is possession, not a
        # participle that lost its suffix.
        if lexical_aux >= 0:
            return (
                lexical_aux,
                words[lexical_aux].lower().rstrip(",.!?;:"),
                False,
            )
        if post_aux_noun >= 0:
            return (
                post_aux_noun,
                words[post_aux_noun].lower().rstrip(",.!?;:"),
                False,
            )
        return -1, "", False

    # Auxiliaries that are also lexical verbs — have (possession)
    # and do (light verb) can be the main verb when nothing
    # verb-like follows them.
    _LEXICAL_AUX = frozenset({"have", "has", "had", "do", "does", "did"})

    def _looks_like_verb(self, word: str) -> bool:
        """Heuristic: does this word look like a verb?

        Primary signal: deconjugation into a known base-form verb —
        "runs"→"run", "contains"→"contain", "thinks"→"think" are all
        detected, while plural nouns like "dogs"→"dog" are not
        (dog isn't a verb). Falls back to unambiguous verb endings
        (-ing, -ed, -ize, -ate, -ify) for verbs outside the lexicon.
        """
        if not word:
            return False

        # Known verb via morphological deconjugation
        if is_verb_form(word) is not None:
            return True

        # Unambiguous verb endings for words outside the lexicon.
        # "es" is deliberately absent: too many nouns end in -es
        # ("buses", "glasses") — those are handled by the lexicon
        # check above.
        verb_endings = ("ed", "ing", "ize", "ise", "ate", "ify")
        for ending in verb_endings:
            if word.endswith(ending) and len(word) > len(ending) + 1:
                return True

        # Check against the imperative/common verb set (all already
        # in VERB_LEXICON via is_verb_form, but kept for clarity)
        if word in _COMMON_VERBS:
            return True

        return False

    def _inside_open_np(self, words: list[str], i: int) -> bool:
        """Is position i inside a still-open determiner-headed NP?

        A determiner or possessive opens an NP that runs until a
        preposition, conjunction, or verb closes it. If a copula or
        auxiliary still follows i, the verb-shaped word at i is the
        NP's head noun — "a blue square block IS on the table" gets
        block-as-noun + is. A merely verb-SHAPED follower doesn't
        count — "the dog bit the man" keeps bit (the verb-shaped
        "man" is itself inside a determiner-headed NP), and "the
        strong men LIFT the box" keeps lift (nothing follows).
        """
        opened = False
        for j in range(i):
            prev = words[j].lower().rstrip(",.!?;:")
            if (
                prev in self._PURE_DETERMINERS
                or prev in _POSSESSIVE_PRONOUNS
            ):
                opened = True
            elif prev in _PREPOSITIONS or prev in _CONJUNCTIONS:
                opened = False
        if not opened:
            return False
        return any(
            w.lower().rstrip(",.!?;:") in _COPULAR_VERBS | _AUX_VERBS
            for w in words[i + 1 :]
        )

    def _detect_tense(self, words: list[str]) -> str:
        """Detect the tense of a clause.

        Returns "past", "present", "future", or "unknown".
        """
        words_lower = [w.lower().rstrip(",.!?;:") for w in words]

        # Future: will/would/shall + verb
        for w in words_lower:
            if w in ("will", "shall", "'ll"):
                return "future"
            if w == "would":
                return "future"  # conditional treated as future-like

        # Past: was/were/had/did or -ed ending
        for w in words_lower:
            if w in ("was", "were", "had", "did"):
                return "past"
        # Check for past tense verb ending
        for w in words_lower:
            if w.endswith("ed") and len(w) > 3 and w not in _NEGATION_WORDS:
                return "past"

        # Present: is/are/am/do/does or base form
        for w in words_lower:
            if w in ("is", "are", "am", "do", "does"):
                return "present"

        return "unknown"

    def _parse_object_and_roles(
        self,
        words: list[str],
        is_copular: bool,
        by_is_agent: bool = False,
    ) -> tuple[str, dict[SemanticRole, str]]:
        """Parse the object and additional semantic roles from words
        following the main verb.

        Identifies prepositional phrases (in the park → LOCATION),
        "to" phrases (to John → GOAL), "from" phrases (from London →
        SOURCE), "for" phrases (for Mary → BENEFICIARY), and "with"
        phrases (with a hammer → INSTRUMENT).

        Args:
            by_is_agent: True when the predicate is a passive
                participle — a "by"-phrase then names the AGENT
                ("needed by plants" → plants is the agent), not a
                location.
        """
        if not words:
            return "", {}

        roles: dict[SemanticRole, str] = {}
        obj_parts: list[str] = []
        i = 0

        # Collect the direct object (words until a preposition)
        while i < len(words):
            word_lower = words[i].lower().rstrip(",.!?;:")
            if word_lower in _PREPOSITIONS:
                # Copular complements absorb "of/as/like" phrases —
                # "is a kind of liquid", "is part of the brain" —
                # the phrase completes the attribute rather than
                # opening a role. Otherwise the phrase would be lost
                # ("of" isn't in the role map).
                if is_copular and word_lower in ("of", "as", "like"):
                    # Extend the object with the whole prepositional
                    # phrase.
                    j = i + 1
                    while j < len(words) and words[j].lower().rstrip(",.!?;:") not in _PREPOSITIONS:
                        j += 1
                    obj_parts.extend([words[i], *words[i + 1 : j]])
                    i = j
                    continue
                # Post-nominal relation phrases — "the star left of
                # the moon": a relation word ending the object takes
                # its own prepositional complement, which belongs to
                # the NP rather than opening a clause-level role.
                if (
                    obj_parts
                    and obj_parts[-1].lower().rstrip(",.!?;:")
                    in self._POSTNOMINAL_REL
                    and word_lower in self._REL_COMPLEMENT_PREPS
                ):
                    obj_parts.append(words[i])
                    i += 1
                    # The complement NP runs until the next
                    # preposition or the next relation word that
                    # carries its own complement ("... and the moon
                    # next to the sun").
                    while i < len(words):
                        nxt = words[i].lower().rstrip(",.!?;:")
                        if nxt in _PREPOSITIONS:
                            break
                        if (
                            nxt in self._POSTNOMINAL_REL
                            and i + 1 < len(words)
                            and words[i + 1].lower().rstrip(",.!?;:")
                            in self._REL_COMPLEMENT_PREPS
                        ):
                            break
                        obj_parts.append(words[i])
                        i += 1
                    continue
                break
            obj_parts.append(words[i])
            i += 1

        obj = " ".join(obj_parts).strip() if obj_parts else ""

        # Parse prepositional phrases into roles
        last_role: SemanticRole | None = None
        while i < len(words):
            prep = words[i].lower().rstrip(",.!?;:")
            i += 1
            phrase_parts: list[str] = []
            while i < len(words) and words[i].lower().rstrip(",.!?;:") not in _PREPOSITIONS:
                phrase_parts.append(words[i])
                i += 1
            phrase = " ".join(phrase_parts).strip() if phrase_parts else ""

            if not phrase:
                continue

            # Map preposition to semantic role
            if prep == "by" and by_is_agent:
                roles[SemanticRole.AGENT] = phrase
                last_role = SemanticRole.AGENT
            elif (
                prep == "at"
                and phrase.split()[:1]
                and phrase.split()[0].lower() in self._AT_ADVERBIALS
            ):
                # "at least / at most / at first" are adverbials, not
                # locations — they belong to the phrase they modify.
                if last_role is not None:
                    roles[last_role] = f"{roles[last_role]} {prep} {phrase}"
                else:
                    obj = f"{obj} {prep} {phrase}".strip()
            elif prep in (
                "in",
                "on",
                "at",
                "into",
                "above",
                "below",
                "under",
                "over",
                "between",
                "near",
                "by",
            ):
                roles[SemanticRole.LOCATION] = phrase
                last_role = SemanticRole.LOCATION
            elif prep == "to":
                roles[SemanticRole.GOAL] = phrase
                last_role = SemanticRole.GOAL
            elif prep == "from":
                roles[SemanticRole.SOURCE] = phrase
                last_role = SemanticRole.SOURCE
            elif prep == "for":
                roles[SemanticRole.BENEFICIARY] = phrase
                last_role = SemanticRole.BENEFICIARY
            elif prep == "with":
                roles[SemanticRole.INSTRUMENT] = phrase
                last_role = SemanticRole.INSTRUMENT
            elif prep == "about":
                roles[SemanticRole.THEME] = phrase
                last_role = SemanticRole.THEME
            elif prep in ("during", "before", "after"):
                roles[SemanticRole.TIME] = phrase
                last_role = SemanticRole.TIME
            elif last_role is not None:
                # An unmapped preposition ("of", "as", ...) extends
                # the open role phrase rather than vanishing —
                # "groups of 2, 5 and 4" keeps its complement.
                roles[last_role] = f"{roles[last_role]} {prep} {phrase}"
            else:
                # No role open yet — the phrase modifies the object
                # ("a cup of tea").
                obj = f"{obj} {prep} {phrase}".strip()

        return obj, roles

    # Relation words that take a prepositional complement inside an
    # NP — "left of", "right of", "next to", "far from". When one of
    # these ends the object, the following PP is its complement.
    _POSTNOMINAL_REL = frozenset(
        {"left", "right", "next", "far", "apart", "close", "opposite"}
    )
    _REL_COMPLEMENT_PREPS = frozenset({"of", "to", "from"})

    # "at + X" idioms that are adverbial modifiers, never locations:
    # "at least two", "at most five", "at first", "at once".
    _AT_ADVERBIALS = frozenset(
        {"least", "most", "first", "last", "best", "worst", "once", "all"}
    )

    # Relative pronouns that open a restrictive clause modifying the
    # preceding noun phrase — "the machine that has two parts".
    _RELATIVE_PRONOUNS = frozenset({"that", "which", "who"})

    def _extract_relative_propositions(
        self, propositions: list[Proposition]
    ) -> list[Proposition]:
        """Split restrictive relative clauses out of objects.

        "a machine that has two parts" arrives as one proposition
        whose object is the whole string; the "that"-clause is really
        a second proposition about the head noun ("machine has two
        parts"). Emitting both keeps the matrix fact and the embedded
        fact available to downstream reasoning instead of burying the
        clause inside an NP string.

        Only splits when the post-pronoun tail opens with verb
        material — "i know that song" (determiner use) and
        object-relatives like "the book that i read" stay whole.
        """
        out: list[Proposition] = []
        for prop in propositions:
            out.append(prop)
            if not prop.object:
                continue
            words = prop.object.split()
            for j, w in enumerate(words):
                wl = w.lower().rstrip(",.!?;:")
                if j == 0 or j + 1 >= len(words):
                    continue
                if wl in self._RELATIVE_PRONOUNS:
                    # "the machine that has two parts" — the tail
                    # must open with verb material; "that song"
                    # (determiner) and "the book that i read"
                    # (object-relative) stay whole.
                    tail_first = words[j + 1].lower().rstrip(",.!?;:")
                    if not (
                        tail_first in _AUX_VERBS
                        or tail_first in _COPULAR_VERBS
                        or tail_first in _NEGATION_WORDS
                        or self._looks_like_verb(tail_first)
                    ):
                        continue
                    rel_clause = f"{' '.join(words[:j])} {' '.join(words[j + 1:])}"
                elif wl.endswith("ing") and is_verb_form(wl):
                    # Reduced relative — "an object containing two
                    # parts" ≡ "an object that contains two parts".
                    rel_clause = " ".join(words)
                else:
                    continue
                rel = self._extract_proposition(rel_clause)
                if not rel.verb_found:
                    break
                old_obj = " ".join(words).strip()
                prop.object = " ".join(words[:j]).strip()
                # A copular proposition mirrors its object into
                # ATTRIBUTE — keep them consistent after trimming.
                if prop.roles.get(SemanticRole.ATTRIBUTE) == old_obj:
                    prop.roles[SemanticRole.ATTRIBUTE] = prop.object
                out.append(rel)
                break
        return out

    # ─── Reference resolution ───────────────────────────────────

    def _resolve_references(
        self,
        propositions: list[Proposition],
        speaker: str,
    ) -> dict[str, str]:
        """Resolve pronouns to their referents.

        Uses a recency-based strategy: pronouns refer to the most
        recently mentioned entity of the appropriate type. "I" and
        "me" resolve to the speaker. "You" resolves to Genesis (the
        listener). Third-person pronouns resolve to recent entities.

        This is a simplified version of Hobbs' (1978) pronoun
        resolution algorithm and centering theory (Grosz, Joshi &
        Weinstein, 1995).
        """
        resolved: dict[str, str] = {}

        # Collect all entity mentions in order
        entities: list[str] = []
        for prop in propositions:
            if prop.subject:
                entities.append(prop.subject)
            if prop.object:
                entities.append(prop.object)
            for role_fill in prop.roles.values():
                entities.append(role_fill)

        # Add recent entities from context
        entities.extend(self._recent_entities)

        for prop in propositions:
            subj_lower = prop.subject.lower()

            # First person pronouns → speaker
            if subj_lower == "i":
                resolved["i"] = speaker
                prop.subject = speaker
            elif subj_lower == "me":
                resolved["me"] = speaker
            elif subj_lower == "my":
                resolved["my"] = speaker

            # Second person → Genesis (the listener)
            elif subj_lower == "you":
                resolved["you"] = "Genesis"
                prop.subject = "Genesis"
            elif subj_lower == "your":
                resolved["your"] = "Genesis"

            # Third person pronouns → recent entity
            elif subj_lower in ("he", "she", "it", "they"):
                referent = self._find_referent(subj_lower, entities, prop)
                if referent:
                    resolved[subj_lower] = referent
                    prop.subject = referent

            # Object pronouns
            obj_lower = prop.object.lower()
            if obj_lower in ("him", "her", "it", "them"):
                referent = self._find_referent(obj_lower, entities, prop)
                if referent:
                    resolved[obj_lower] = referent
                    prop.object = referent
            elif obj_lower == "me":
                resolved["me"] = speaker
                prop.object = speaker
            elif obj_lower == "you":
                resolved["you"] = "Genesis"
                prop.object = "Genesis"

        return resolved

    def _find_referent(
        self,
        pronoun: str,
        entities: list[str],
        current_prop: Proposition,
    ) -> str | None:
        """Find the most likely referent for a third-person pronoun.

        "he" → most recent male-sounding entity
        "she" → most recent female-sounding entity
        "it" → most recent non-person entity
        "they" → most recent plural entity or group
        """
        # Object-form pronouns share the subject-form logic
        pronoun = {"him": "he", "her": "she", "them": "they"}.get(
            pronoun, pronoun
        )

        # Filter out the pronoun itself and the current subject.
        # Conjunct NPs ("salt and pepper", "mind and body") are not
        # referents for singular pronouns — only "they" fits them.
        candidates = [
            e
            for e in entities
            if e.lower() not in _SUBJECT_PRONOUNS
            and e.lower() not in _OBJECT_PRONOUNS
            and e.lower() not in _POSSESSIVE_PRONOUNS
            and e != current_prop.subject
            and (
                pronoun == "they"
                or (" and " not in e.lower() and " or " not in e.lower())
            )
        ]

        if not candidates:
            return None

        # Reverse to get most recent first
        candidates.reverse()

        # Female-sounding name endings (common in English)
        female_endings = ("a", "e", "ia", "ine", "elle", "ette", "ina", "y", "ah", "eth")

        if pronoun == "he":
            # Prefer male-sounding names (capitalized, non-female
            # ending, singular). Entities opening with "the" are
            # common NPs ("the song"), not names.
            for e in candidates:
                if (
                    e
                    and e[0].isupper()
                    and not e.lower().startswith("the ")
                    and not e.lower().endswith(female_endings)
                    and not (e.endswith("s") and not e.endswith("ss"))
                ):
                    return e
            # Fallback: first capitalized singular entity
            for e in candidates:
                if (
                    e
                    and e[0].isupper()
                    and not e.lower().startswith("the ")
                    and not (e.endswith("s") and not e.endswith("ss"))
                ):
                    return e
            # No proper noun found — don't resolve to a common noun
            return None

        elif pronoun == "she":
            # Prefer female-sounding names (capitalized, female
            # ending, singular)
            for e in candidates:
                if (
                    e
                    and e[0].isupper()
                    and not e.lower().startswith("the ")
                    and e.lower().endswith(female_endings)
                    and not (e.endswith("s") and not e.endswith("ss"))
                ):
                    return e
            # Fallback: first capitalized singular entity
            for e in candidates:
                if (
                    e
                    and e[0].isupper()
                    and not e.lower().startswith("the ")
                    and not (e.endswith("s") and not e.endswith("ss"))
                ):
                    return e
            # No proper noun found — don't resolve to a common noun
            return None

        elif pronoun == "it":
            # Look for non-person entities (lowercase or objects)
            for e in candidates:
                if e and not e[0].isupper():
                    return e
            # Fallback: first candidate if any
            return candidates[0] if candidates else None

        elif pronoun == "they":
            # Plural — return the most recent entity
            return candidates[0] if candidates else None

        return candidates[0] if candidates else None

    # ─── Key concept identification ────────────────────────────

    def _identify_key_concepts(
        self,
        propositions: list[Proposition],
        speech_act: SpeechActType,
    ) -> list[ConceptRole]:
        """Identify key concepts and their semantic roles.

        Key concepts are the content words that carry the semantic
        weight of the sentence: the subject (agent), the main verb
        (action), the object (patient), and any significant role
        fillers.
        """
        concepts: list[ConceptRole] = []

        for prop in propositions:
            # Subject → AGENT (high salience)
            if prop.subject:
                role = SemanticRole.AGENT
                if prop.predicate.lower() in _COPULAR_VERBS:
                    role = SemanticRole.THEME
                concepts.append(
                    ConceptRole(
                        concept=prop.subject,
                        role=role,
                        salience=0.9,
                    )
                )

            # Predicate → the action (high salience)
            if prop.predicate:
                concepts.append(
                    ConceptRole(
                        concept=prop.predicate,
                        role=SemanticRole.THEME,  # verb as the core relation
                        salience=0.85,
                    )
                )

            # Object → PATIENT (moderate salience)
            if prop.object:
                concepts.append(
                    ConceptRole(
                        concept=prop.object,
                        role=SemanticRole.PATIENT,
                        salience=0.7,
                    )
                )

            # Additional roles
            role_salience = {
                SemanticRole.LOCATION: 0.5,
                SemanticRole.TIME: 0.4,
                SemanticRole.GOAL: 0.6,
                SemanticRole.SOURCE: 0.6,
                SemanticRole.BENEFICIARY: 0.6,
                SemanticRole.INSTRUMENT: 0.5,
                SemanticRole.ATTRIBUTE: 0.65,
                SemanticRole.THEME: 0.5,
            }
            for role, filler in prop.roles.items():
                concepts.append(
                    ConceptRole(
                        concept=filler,
                        role=role,
                        salience=role_salience.get(role, 0.5),
                    )
                )

        return concepts

    # ─── Confidence estimation ──────────────────────────────────

    def _compute_confidence(
        self,
        propositions: list[Proposition],
        clauses: list[str],
        speech_act: SpeechActType,
    ) -> float:
        """Compute confidence in the parse.

        Higher confidence when we successfully extracted subjects and
        predicates. Lower for complex sentences with many clauses.
        """
        if not propositions:
            return 0.2

        # Start with base confidence
        conf = 0.5

        # Each successfully parsed proposition with both subject and
        # predicate increases confidence
        for prop in propositions:
            if prop.subject and prop.predicate:
                conf += 0.1
            elif prop.subject or prop.predicate:
                conf += 0.05

        # More clauses → lower confidence (harder to parse)
        if len(clauses) > 2:
            conf -= 0.1 * (len(clauses) - 2)

        # Clear speech acts are more confident
        if speech_act in (SpeechActType.QUESTION, SpeechActType.COMMAND, SpeechActType.GREETING):
            conf += 0.05

        return max(0.1, min(1.0, conf))

    # ─── Entity tracking ────────────────────────────────────────

    def _update_recent_entities(
        self,
        propositions: list[Proposition],
        resolved: dict[str, str],
    ) -> None:
        """Update the recent entity list for future pronoun resolution.

        Only stores actual entities — proper nouns and short noun
        phrases. Clause fragments from failed parses (where the
        whole clause became the subject) are filtered out to
        prevent contaminating future pronoun resolution.
        """
        for prop in propositions:
            # Check subject, object, and role fills for entities
            candidates = [prop.subject, prop.object]
            candidates.extend(prop.roles.values())
            for candidate in candidates:
                if not candidate:
                    continue
                cl = candidate.lower().rstrip(",.!?;:")
                # Skip pronouns
                if cl in _SUBJECT_PRONOUNS or cl in _OBJECT_PRONOUNS:
                    continue
                # Skip question words — they are not entities
                if cl in _WH_WORDS:
                    continue
                # Skip clause fragments: more than 4 words is almost
                # certainly a failed parse, not an entity.
                if len(candidate.split()) > 4:
                    continue
                # Skip if it contains a common verb — it's a clause
                # fragment, not an entity.
                words_in = {w.lower().rstrip(",.!?;:") for w in candidate.split()}
                if words_in & _COMMON_VERBS:
                    continue
                if words_in & _COPULAR_VERBS:
                    continue
                # Skip if it contains a negation word — "I don't" is
                # a clause fragment, not an entity.
                if words_in & _NEGATION_WORDS:
                    continue
                if candidate not in self._recent_entities:
                    self._recent_entities.append(candidate)

        # Keep only the most recent 10
        if len(self._recent_entities) > 10:
            self._recent_entities = self._recent_entities[-10:]

    # ─── Utility ────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the engine's state (recent entity tracking)."""
        self._recent_entities.clear()

    @property
    def figurative(self) -> FigurativeLanguageProcessor:
        """The figurative language processor (metaphor, irony, idiom detection)."""
        return self._figurative

    @property
    def recent_entities(self) -> list[str]:
        """Recently mentioned entities, for pronoun resolution."""
        return list(self._recent_entities)

    def set_recent_entities(self, entities: list[str]) -> None:
        """Replace the recent entity list (used by the cognition engine)."""
        self._recent_entities = list(entities)
