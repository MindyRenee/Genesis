"""Semantic memory — general knowledge extracted from experience.

Semantic memory is the declarative memory system that stores general
knowledge about the world — facts, concepts, and their relationships —
independent of the specific episodic context in which they were learned
(Tulving, 1972). Where episodic memory records *what happened to me*,
semantic memory records *what is the case*.

# Why this is separate from the concept network

The :class:`~genesis_cognitive.concepts.ConceptNetwork` already
stores concepts and their relationships, which is a form of semantic
knowledge. However, it lacks three things that a full semantic memory
system provides:

1. **Fact extraction** — parsing episodic memory text into discrete
   propositions ("X is Y", "X has property Z", "X relates to Y").
2. **Schema formation** — grouping related facts into structured
   schemas (e.g., learning about "brain" creates a schema with parts
   like "hippocampus", "amygdala" and functions like "memory",
   "emotion").
3. **Neocortical storage** — facts are stored in the concept network
   (the neocortical equivalent) independent of the hippocampal
   episodic memory store.

# Complementary learning systems

The hippocampus rapidly encodes episodic memories (specific events),
while the neocortex slowly integrates regularities across episodes into
semantic knowledge (McClelland et al., 1995). This complementary
learning-systems architecture avoids catastrophic interference: the
hippocampus can learn quickly without overwriting the slow, structured
neocortical knowledge base.

This module models the *neocortical* side of that division. Facts are
extracted from episodic memories (stored by the
:class:`~genesis_cognitive.memory_engine.MemoryEngine`) and consolidated
into the concept network, which acts as the neocortical semantic store.
Schemas abstract across multiple episodes: as more instances of a
category are encountered, the schema becomes more prototype-like
(Rosch, 1975).

References:
    - Tulving (1972): episodic vs semantic memory distinction
    - McClelland et al. (1995): complementary learning systems
    - Rosch (1975): prototype theory and schema abstraction
    - Rumelhart (1980): schemas as building blocks of cognition
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — import only for type checkers
    from ..concepts import ConceptNetwork, RelationType

__all__ = ["Fact", "Schema", "SemanticMemory"]


# ─── Pre-compiled regex patterns for fact extraction ──────────────────

# Split text into clauses/sentences.
_CLAUSE_SPLIT_RE = re.compile(r"[.;,\n]+")
# Word tokeniser.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Indefinite is_a: "X is a Y" / "X is an Y" / "X are a Y".
# High-confidence because the indefinite article signals a category.
_IS_A_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is|was)\s+(?:a|an)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Plural is_a: "X are Y" / "X were Y" / "X are a Y".
# Plural subjects often carry no article ("cats are mammals"). This is
# lower-confidence because the absence of an article removes the
# explicit category signal, but the plural verb form is still a strong
# grammatical cue for classification. Confidence is lowered at the
# extraction site so isolated mistakes (e.g. "they are here") are
# pruned unless reinforced.
_IS_A_PLURAL_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:are|were)\s+(?:a|an)?\s*"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Relational noun: "X is the Y of Z" / "X are the Y of Z".
# Captures both a category (X is a Y) and a relationship (X relates to Z).
# No hardcoded relation map: the "of" construction is too semantically
# varied, so we record the weaker `relates_to` and the category.
_IS_A_RELATIONAL_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is|are|was|were)\s+the\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*?)\s+of\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Copular identity: "X is Y" (no article). This is the most ambiguous
# copular form. Without an article we cannot tell property ("X is blue")
# from category ("X is human"). We therefore keep it conservative:
# multi-word objects are treated as a weak `relates_to` (noun phrase),
# single-word objects are skipped to avoid hardcoding adjectives.
# Excludes "X is a/an Y" and "X is the Y of Z" handled above.
_IS_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is|are|was|were)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Has-property: "X has Z", "X has a Z", "X have Z".
_HAS_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:has|have|had)\s+(?:a|an|the\s+)?"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Part-of: "X is part of Y", "X is a part of Y".
_PART_OF_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is|are)\s+(?:a\s+)?part\s+of\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Composition: "X is composed of Y", "X is made of Y", "X is made up of Y",
# "X consists of Y". These explicitly mark a whole→parts relationship.
# Components separated by "and" or "or" are split into separate part_of facts.
_COMPOSITION_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is|are|was|were)\s+"
    r"(?:composed|made|consisted|built|formed)\s+"
    r"(?:up\s+)?of\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Relates-to: "X relates to Y", "X is related to Y", "X connects to Y".
_RELATES_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:relates?|connects?|leads?|causes?|enables?)\s+to\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Action relation: "X causes Y", "X enables Y", "X creates Y".
# Note: these all map to "causes" in the fact relation, which
# _relation_to_edge then maps to CAUSES. For more specific relation
# types, see the typed patterns below.
_ACTION_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:causes|enables|creates|produces|generates)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Typed action relations — each maps to a specific RelationType.
# "X emerges from Y" / "X arises from Y" / "X stems from Y"
_EMERGES_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:emerges|arises|stems)\s+from\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# "X depends on Y" / "X relies on Y" / "X requires Y"
_DEPENDS_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:depends\s+on|relies\s+on|requires)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# "X is similar to Y" / "X resembles Y"
_SIMILAR_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is\s+similar\s+to|is\s+analogous\s+to|resembles)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# "X prevents Y" / "X blocks Y" / "X inhibits Y" / "X suppresses Y"
_PREVENTS_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:prevents|blocks|inhibits|suppresses)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# "X leads to Y" / "X results in Y"
_LEADS_TO_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:leads\s+to|results\s+in)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# "X harms Y" / "X impairs Y" / "X damages Y" / "X disrupts Y"
_HARMS_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:harms|impairs|damages|disrupts|degrades)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# "X is opposite of Y" / "X contrasts with Y"
_OPPOSITE_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is\s+(?:the\s+)?opposite\s+of|contrasts\s+with)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Named/called: "X named Y", "X called Y", "X is named Y", "X is called Y".
# Captures "I have a cat named Whiskers" → named(cat, Whiskers)
_NAMED_RE = re.compile(
    r"\b([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is|are|was|were)?\s*(?:named|called)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Possessive identity: "X's name is Y", "my name is Y".
# Captures "My name is Alice" → is_a(name, Alice) with subject "my"
_NAME_IS_RE = re.compile(
    r"\b(?:my|your|his|her|its|their)\s+name\s+(?:is|are|was|were)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Possessive property: "my favorite X is Y", "my X is Y".
# Captures "My favorite color is purple" → has_property(favorite color, purple)
_POSSESSIVE_IS_RE = re.compile(
    r"\b(?:my|your|his|her|its|their)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\s+"
    r"(?:is|are|was|were)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Lives-in: "X lives in Y", "we live in Y", "I live in Y".
# Captures "We live in Utah" → has_property(home, Utah)
_LIVES_IN_RE = re.compile(
    r"\b(?:we|i|they|he|she)\s+(?:live|lives|lived)\s+in\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)\b",
    re.IGNORECASE,
)
# Lives-with: "X lives with Y", "we live with Y".
# Captures "We live with Junior" → relates_to(family, Junior)
_LIVES_WITH_RE = re.compile(
    r"\b(?:we|i|they|he|she)\s+(?:live|lives|lived)\s+with\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)?)\b",
    re.IGNORECASE,
)
# Likes/loves: "X likes Y", "I like Y", "X loves Y".
# Captures "I like the ocean" → has_property(interests, ocean)
_LIKES_RE = re.compile(
    r"\b(?:i|we|they|he|she)\s+(?:like|likes|love|loves|enjoy|enjoys)\s+"
    r"([A-Za-z][\w]*(?:\s+[A-Za-z][\w]*(?:\s+[A-Za-z][\w]*)*)?)\b",
    re.IGNORECASE,
)



# ─── Fact and Schema dataclasses ──────────────────────────────────────


@dataclass(slots=True)
class Fact:
    """A single proposition extracted from experience.

    A fact is a typed relationship between a subject and an object
    (or a subject and a property). Facts are the atomic units of
    semantic memory — they are what get consolidated from episodic
    memory into the neocortical knowledge store.

    Attributes:
        subject: The concept the fact is about (e.g., "brain").
        relation: The typed relation (e.g., "is_a", "has_property",
            "part_of", "relates_to").
        object: The object of the relation (e.g., "organ", "hippocampus").
        confidence: How confident Genesis is in this fact [0..1].
            Repeated extraction from multiple episodes increases
            confidence (consolidation).
        source_count: How many distinct episodes this fact was
            extracted from. Higher counts → more abstract / reliable.
        extracted_at: When the fact was first extracted (ms).
        last_reinforced: When the fact was last reinforced (ms).
    """

    subject: str
    relation: str
    object: str
    confidence: float = 0.5
    source_count: int = 1
    extracted_at: int = 0
    last_reinforced: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        """A stable identity key for deduplication."""
        return (
            self.subject.lower(),
            self.relation.lower(),
            self.object.lower(),
        )

    def __str__(self) -> str:
        """Render the fact as a natural-language triple."""
        return f"{self.subject} {self.relation.replace('_', ' ')} {self.object}"


@dataclass(slots=True)
class Schema:
    """A structured grouping of facts about a concept.

    Schemas (Rumelhart, 1980) are knowledge structures that represent
    the generic concept of something — its parts, properties,
    functions, and relationships. As more instances of a category are
    encountered, the schema becomes more abstract and prototype-like
    (Rosch, 1975): features common to most instances are central,
    while idiosyncratic features fade.

    Attributes:
        concept: The concept this schema is about (e.g., "brain").
        parts: Component concepts that are part of this concept
            (e.g., "hippocampus", "amygdala").
        properties: Attributes of the concept (e.g., "soft", "gray").
        functions: What the concept does (e.g., "memory", "emotion").
        relations: Typed relations to other concepts.
        instances: How many distinct episodes contributed to this
            schema. More instances → more abstract / prototype-like.
        abstraction_level: How abstract the schema is [0..1]. Grows
            with the number of instances, asymptoting at 1.0.
        created_at: When the schema was first formed (ms).
        last_updated: When the schema was last updated (ms).
    """

    concept: str
    parts: set[str] = field(default_factory=set)
    properties: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)
    relations: dict[str, str] = field(default_factory=dict)
    instances: int = 0
    abstraction_level: float = 0.0
    created_at: int = 0
    last_updated: int = 0

    def describe(self) -> str:
        """Human-readable description of the schema."""
        lines = [f"Schema: {self.concept}"]
        if self.parts:
            lines.append(f"  parts: {', '.join(sorted(self.parts))}")
        if self.properties:
            lines.append(f"  properties: {', '.join(sorted(self.properties))}")
        if self.functions:
            lines.append(f"  functions: {', '.join(sorted(self.functions))}")
        if self.relations:
            rel_str = ", ".join(f"{k} {v}" for k, v in sorted(self.relations.items()))
            lines.append(f"  relations: {rel_str}")
        lines.append(f"  instances: {self.instances}, abstraction: {self.abstraction_level:.2f}")
        return "\n".join(lines)


# ─── Semantic memory system ───────────────────────────────────────────


class SemanticMemory:
    """Neocortical semantic memory — general knowledge from experience.

    Extracts facts from episodic memories, forms schemas by grouping
    related facts, and stores facts in the concept network (the
    neocortical equivalent) independent of the hippocampal episodic
    store.

    This implements the *neocortical* side of the complementary
    learning-systems architecture (McClelland et al., 1995): the
    hippocampus (``MemoryEngine``) rapidly encodes specific episodes,
    while this system slowly extracts and consolidates general
    knowledge into the concept network.

    Biological grounding:
        - Semantic memory is stored in the temporal neocortex,
          particularly the anterior temporal subsystem (the "semantic
          hub"; Patterson et al., 2007).
        - Schema formation reflects neocortical abstraction across
          hippocampal episodes during systems consolidation
          (McClelland et al., 1995; Kumaran et al., 2016).
        - Prototype abstraction: as category instances accumulate,
          the schema converges on the central tendency (Rosch, 1975).
    """

    def __init__(self, network: ConceptNetwork | None = None) -> None:
        """Initialize the semantic memory store with an optional concept network."""
        self._network = network
        # fact key → Fact
        self._facts: dict[tuple[str, str, str], Fact] = {}
        # concept name → Schema
        self._schemas: dict[str, Schema] = {}
        # Counters for introspection.
        self.facts_extracted: int = 0
        self.schemas_formed: int = 0
        self.consolidations: int = 0
        # Primed concepts from global workspace broadcasts.
        # Maps concept name → priming strength (0..1). Facts about
        # primed concepts get a retrieval boost, modeling semantic
        # priming from cognitive content (Meyer & Schvaneveldt, 1971).
        self._primed_concepts: dict[str, float] = {}

    # ── Fact extraction ──────────────────────────────────────────

    def extract_facts(self, episode_text: str) -> list[Fact]:
        """Extract facts (propositions) from episode text.

        Parses the text to identify propositions of the form:
        - "X is a Y" → is_a(X, Y)
        - "X is Y" → has_property(X, Y)  (when Y is not a category)
        - "X has Z" → has_property(X, Z)
        - "X is part of Y" → part_of(X, Y)
        - "X relates to Y" → relates_to(X, Y)
        - "X causes Y" → causes(X, Y)

        Extracted facts are stored and (if a concept network is
        attached) consolidated into the network as concepts and edges.

        Args:
            episode_text: The text of an episodic memory to extract
                facts from.

        Returns:
            The list of newly extracted (or reinforced) facts.
        """
        facts: list[Fact] = []
        now_ms = int(time.time() * 1000)
        seen_keys: set[tuple[str, str, str]] = set()

        # Normalise whitespace and split into clauses.
        text = re.sub(r"\s+", " ", episode_text).strip()
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]

        for clause in clauses:
            # Skip question clauses — they're not statements of fact.
            # Questions like "What is my name?" get mis-parsed as
            # "what is a name" which pollutes the concept network.
            if clause.rstrip().endswith("?"):
                continue
            # Skip clauses that start with a question word (but not
            # "I" or "We" which are statement subjects)
            first_word = clause.split()[0].lower() if clause.split() else ""
            if first_word in (
                "what",
                "who",
                "where",
                "when",
                "why",
                "how",
                "does",
                "did",
                "can",
                "could",
                "would",
                "should",
                "will",
            ):
                continue
            for fact in self._extract_from_clause(clause, now_ms):
                if fact.key in seen_keys:
                    continue
                seen_keys.add(fact.key)
                facts.append(fact)

        # Store / reinforce each extracted fact.
        for fact in facts:
            self._store_or_reinforce(fact, now_ms)

        self.facts_extracted += len(facts)
        return facts

    def _extract_from_clause(self, clause: str, now_ms: int) -> list[Fact]:
        """Extract facts from a single clause."""
        facts: list[Fact] = []

        # part_of, relates_to, action, is_a, has — basic structural facts.
        facts.extend(self._extract_basic_facts(clause, now_ms))

        # "X is Y" (no article) — needs existing_pairs from basic facts.
        existing_pairs = self._extract_is_facts(clause, now_ms, facts)

        # "my name is Y", "my X is Y", "X named Y" — possessive/named facts.
        self._extract_possessive_facts(clause, now_ms, facts, existing_pairs)

        # "we live in Y", "we live with Y", "I like Y" — lifestyle facts.
        facts.extend(self._extract_lifestyle_facts(clause, now_ms))

        return facts

    def _extract_basic_facts(self, clause: str, now_ms: int) -> list[Fact]:
        """Extract structural facts: part_of, relates_to, action, is_a, has,
        plus typed relations (emerges_from, depends_on, similar_to, etc.).
        """
        facts: list[Fact] = []

        self._extract_part_of_facts(clause, now_ms, facts)
        self._extract_relates_action_facts(clause, now_ms, facts)
        self._extract_typed_relation_facts(clause, now_ms, facts)
        self._extract_is_a_facts(clause, now_ms, facts)
        self._extract_has_facts(clause, now_ms, facts)

        return facts

    def _extract_part_of_facts(
        self, clause: str, now_ms: int, facts: list[Fact]
    ) -> None:
        """Extract part_of and composition facts from the clause."""
        # part_of (check before generic "is" to avoid mis-parsing).
        for m in _PART_OF_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="part_of",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # Composition: "X is composed/made/formed of Y [and Z]".
        # The whole is group 1; the component list is group 2. Each
        # component becomes a part_of fact. Conjunctions are split on
        # " and " and " or "; commas are already clause separators.
        for m in _COMPOSITION_RE.finditer(clause):
            whole = _strip_article(m.group(1).strip().lower())
            if not whole:
                continue
            raw_components = m.group(2).strip().lower()
            for sep in (" and ", " or "):
                raw_components = raw_components.replace(sep, "\n")
            for component in raw_components.split("\n"):
                component = _strip_article(component.strip().rstrip("."))
                if component and component != whole:
                    facts.append(
                        Fact(
                            subject=component,
                            relation="part_of",
                            object=whole,
                            confidence=0.6,
                            extracted_at=now_ms,
                            last_reinforced=now_ms,
                        )
                    )

    def _extract_relates_action_facts(
        self, clause: str, now_ms: int, facts: list[Fact]
    ) -> None:
        """Extract relates_to and action (causes) facts from the clause."""
        # relates_to / causes / enables via "X (relates|connects) to Y".
        for m in _RELATES_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="relates_to",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # action relations: "X causes Y", "X enables Y".
        for m in _ACTION_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="causes",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

    def _extract_typed_relation_facts(
        self, clause: str, now_ms: int, facts: list[Fact]
    ) -> None:
        """Extract typed relation facts (emerges_from, depends_on, etc.).

        These are checked before the generic is_a/has patterns so
        that "X emerges from Y" is not mis-parsed as "X is Y".
        """
        # ── Typed relations (specific relation types) ──
        # These are checked before the generic is_a/has patterns so
        # that "X emerges from Y" is not mis-parsed as "X is Y".

        # "X emerges from Y" / "X arises from Y" / "X stems from Y"
        for m in _EMERGES_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="emerges_from",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X depends on Y" / "X relies on Y" / "X requires Y"
        for m in _DEPENDS_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="depends_on",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X is similar to Y" / "X resembles Y"
        for m in _SIMILAR_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="similar_to",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X prevents Y" / "X blocks Y" / "X inhibits Y"
        for m in _PREVENTS_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="prevents",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X leads to Y" / "X results in Y"
        for m in _LEADS_TO_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="leads_to",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X harms Y" / "X impairs Y" / "X damages Y"
        for m in _HARMS_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="harms",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X is opposite of Y" / "X contrasts with Y"
        for m in _OPPOSITE_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="opposite_of",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

    def _extract_is_a_facts(
        self, clause: str, now_ms: int, facts: list[Fact]
    ) -> None:
        """Extract is_a facts (indefinite, plural, relational) from the clause."""
        # "X is a/an Y" / "X was a/an Y" → is_a (indefinite category).
        for m in _IS_A_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="is_a",
                        object=obj,
                        confidence=0.6,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X are Y" / "X are a/an Y" / "X were Y" → is_a (plural).
        # Lower confidence: no article means the category is implied,
        # not explicitly marked. This lets correct cases ("cats are
        # mammals") reinforce while most mistakes drop out in pruning.
        for m in _IS_A_PLURAL_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="is_a",
                        object=obj,
                        confidence=0.35,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "X is the Y of Z" → is_a(X, Y) + relates_to(X, Z).
        # The "of" construction is too semantically open to hardcode a
        # specific typed relation, so we record the category and a weak
        # relation. `relates_to` captures the connection without claiming
        # a direction like creates/owns/depends_on that would need world
        # knowledge.
        for m in _IS_A_RELATIONAL_RE.finditer(clause):
            subj, role, target = (
                _strip_article(m.group(1).strip().lower()),
                _strip_article(m.group(2).strip().lower()),
                _strip_article(m.group(3).strip().lower()),
            )
            if subj and role:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="is_a",
                        object=role,
                        confidence=0.6,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )
            if subj and target:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="relates_to",
                        object=target,
                        confidence=0.5,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

    def _extract_has_facts(
        self, clause: str, now_ms: int, facts: list[Fact]
    ) -> None:
        """Extract has_property facts from the clause."""
        # "X has Y" → has_property.
        for m in _HAS_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if subj and obj:
                facts.append(
                    Fact(
                        subject=subj,
                        relation="has_property",
                        object=obj,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

    def _extract_is_facts(
        self, clause: str, now_ms: int, facts: list[Fact]
    ) -> set[tuple[str, str]]:
        """Extract bare 'X is Y' facts, returning the existing-pairs set."""
        # Bare "X is Y" is the most ambiguous copular form. Without an
        # article we cannot distinguish property ("the sky is blue") from
        # category ("Alice is human"). We therefore apply conservative
        # grammar-only rules:
        #   - skip single-word objects (most adjectives)
        #   - skip objects that begin with a determiner or possessive
        #     (those are handled by is_a or are specific references)
        #   - skip possessive subjects (e.g. "my name is ...")
        #   - record the rest as a weak `relates_to` instead of the
        #     semantically stronger `has_property`, so we don't claim a
        #     property/category distinction we cannot justify.
        existing_pairs = {(f.subject.lower(), f.object.lower()) for f in facts}
        for m in _IS_RE.finditer(clause):
            raw_subj = m.group(1).strip()
            raw_obj = m.group(2).strip()
            subj, obj = _normalise_pair(raw_subj, raw_obj)
            if not subj or not obj:
                continue
            if obj.startswith("part of"):
                continue
            if (subj.lower(), obj.lower()) in existing_pairs:
                continue

            # Skip possessive subjects like "my name is ...".
            subj_first = raw_subj.split()[0].lower() if raw_subj.split() else ""
            if subj_first in ("my", "your", "his", "her", "its", "their"):
                continue

            # Skip objects that begin with a determiner or possessive
            # (e.g. "the creator", "my friend"). Definite references are
            # either relational nouns handled above or too specific to
            # extract without world knowledge.
            obj_first = raw_obj.split()[0].lower() if raw_obj.split() else ""
            if obj_first in ("the", "a", "an", "my", "your", "his", "her", "its", "their"):
                continue

            # Skip single-word objects; they are most often adjectives
            # ("blue", "happy"), and without a lexicon we cannot tell.
            if len(obj.split()) < 2:
                continue

            # Skip objects that still contain "of", "and", or "or". These
            # are usually complex phrases better handled by the
            # composition/relational patterns (e.g. "composed of ...",
            # "the creator of ..."). Without world knowledge, a bare
            # `relates_to` edge for the entire phrase is noise.
            if any(token in obj for token in (" of ", " and ", " or ")):
                continue

            # Multi-word, determiner-free object: a weak relation.
            # This does not hardcode categories or properties.
            facts.append(
                Fact(
                    subject=subj,
                    relation="relates_to",
                    object=obj,
                    confidence=0.35,
                    extracted_at=now_ms,
                    last_reinforced=now_ms,
                )
            )
            existing_pairs.add((subj.lower(), obj.lower()))
        return existing_pairs

    def _extract_possessive_facts(
        self,
        clause: str,
        now_ms: int,
        facts: list[Fact],
        existing_pairs: set[tuple[str, str]],
    ) -> None:
        """Extract name, possessive, and named facts in place."""
        # "my name is Y" and other possessive-identity statements are
        # *pronoun* statements about the speaker's identity. They should
        # not become generic facts in the concept network (e.g. "name
        # is_a Y" creates the backwards definition "Y is a name").
        # The user's actual name is tracked by the memory system /
        # cognition layer (MemoryEngine.learn_user_fact, _recognize_bonded_user).
        # We intentionally do not store it as a semantic fact here.

        # "my X is Y" → has_property(X, Y). Captures "my favorite
        # color is purple", "my cat is Whiskers", etc. Only if not
        # already captured by the generic "is" pattern.
        for m in _POSSESSIVE_IS_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if not subj or not obj:
                continue
            if (subj.lower(), obj.lower()) in existing_pairs:
                continue
            # Skip the "name is Y" case — handled by _NAME_IS_RE above.
            if subj.lower() == "name":
                continue
            facts.append(
                Fact(
                    subject=subj,
                    relation="has_property",
                    object=obj,
                    extracted_at=now_ms,
                    last_reinforced=now_ms,
                )
            )
            existing_pairs.add((subj.lower(), obj.lower()))

        # "X named Y", "X called Y" → has_property(X, Y). Captures
        # "a cat named Whiskers", "my cat called Whiskers".
        for m in _NAMED_RE.finditer(clause):
            subj, obj = _normalise_pair(m.group(1), m.group(2))
            if not subj or not obj:
                continue
            if (subj.lower(), obj.lower()) in existing_pairs:
                continue
            facts.append(
                Fact(
                    subject=subj,
                    relation="has_property",
                    object=obj,
                    extracted_at=now_ms,
                    last_reinforced=now_ms,
                )
            )
            existing_pairs.add((subj.lower(), obj.lower()))

    def _extract_lifestyle_facts(self, clause: str, now_ms: int) -> list[Fact]:
        """Extract lives-in, lives-with, and likes facts."""
        facts: list[Fact] = []

        # "we live in Y" → has_property(home, Y).
        for m in _LIVES_IN_RE.finditer(clause):
            place = _strip_article(m.group(1).strip().lower())
            if place:
                facts.append(
                    Fact(
                        subject="home",
                        relation="has_property",
                        object=place,
                        extracted_at=now_ms,
                        last_reinforced=now_ms,
                    )
                )

        # "we live with Y" → relates_to(family, Y). Only takes the
        # first name from a list ("Junior, Carol, and the dogs").
        for m in _LIVES_WITH_RE.finditer(clause):
            companion = _strip_article(m.group(1).strip().lower())
            # Split on "and" / "," to get individual names
            for name in re.split(r"\s*,\s*|\s+and\s+", companion):
                name = name.strip()
                if name and len(name) > 1:
                    facts.append(
                        Fact(
                            subject="family",
                            relation="relates_to",
                            object=name,
                            extracted_at=now_ms,
                            last_reinforced=now_ms,
                        )
                    )

        # "I like Y" / "I love Y" → has_property(interests, Y).
        for m in _LIKES_RE.finditer(clause):
            interest = _strip_article(m.group(1).strip().lower())
            # Split on "and" / "," to get individual interests
            for item in re.split(r"\s*,\s*|\s+and\s+", interest):
                item = item.strip()
                if item and len(item) > 1:
                    facts.append(
                        Fact(
                            subject="interests",
                            relation="has_property",
                            object=item,
                            extracted_at=now_ms,
                            last_reinforced=now_ms,
                        )
                    )

        return facts

    # ── Storage / consolidation ──────────────────────────────────

    def _store_or_reinforce(self, fact: Fact, now_ms: int) -> None:
        """Store a new fact or reinforce an existing one."""
        existing = self._facts.get(fact.key)
        if existing is None:
            self._facts[fact.key] = fact
        else:
            # Reinforce: increase confidence and source count.
            existing.source_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.15)
            existing.last_reinforced = now_ms

        # Consolidate into the concept network (neocortical storage).
        if self._network is not None:
            self._consolidate_to_network(fact)

    def store_fact(self, fact: Fact) -> None:
        """Explicitly store a fact in semantic memory.

        Useful for facts learned through definition rather than
        extraction from episodes.
        """
        now_ms = int(time.time() * 1000)
        if fact.extracted_at == 0:
            fact.extracted_at = now_ms
        if fact.last_reinforced == 0:
            fact.last_reinforced = now_ms
        self._store_or_reinforce(fact, now_ms)
        self.facts_extracted += 1

    def _consolidate_to_network(self, fact: Fact) -> None:
        """Consolidate a fact into the concept network.

        This is the neocortical storage step: facts become concepts and
        edges in the concept network, independent of the hippocampal
        episodic store (McClelland et al., 1995).
        """
        network = self._network
        if network is None:
            return
        # Ensure both concepts exist in the network.
        network.add_concept(fact.subject, confidence=fact.confidence)
        network.add_concept(fact.object, confidence=fact.confidence)
        # Map the fact relation to a RelationType edge. If the relation
        # string has no direct mapping, fall back to RELATED_TO rather
        # than dropping the fact entirely. This ensures that unknown or
        # newly added fact relations still become graph edges.
        from ..concepts import RelationType

        rel_type = _relation_to_edge(fact.relation)
        if rel_type is None:
            rel_type = RelationType.RELATED_TO
        network.add_edge(
            fact.subject,
            fact.object,
            rel_type,
            weight=min(1.0, 0.4 + fact.confidence * 0.4),
            origin="semantic",
        )
        self.consolidations += 1

    # ── Schema formation ─────────────────────────────────────────

    def form_schemas(self, facts: list[Fact] | None = None) -> list[Schema]:
        """Form or update schemas from facts.

        For each concept that appears in the facts, builds a schema:

        - **parts**: concepts that are *part of* this concept — i.e.
          facts ``part_of(Y, X)`` mean Y is a part of X, so X's schema
          lists Y. Also ``has_property(X, Y)`` where Y denotes a
          component contributes a part.
        - **properties**: ``has_property(X, Y)``.
        - **functions**: ``causes(X, Y)`` / ``enables(X, Y)`` — what X
          does.
        - **relations**: ``is_a(X, Y)`` and ``relates_to(X, Y)``.

        As more instances (episodes) contribute to a schema, its
        abstraction level rises asymptotically toward 1.0 — modelling
        prototype abstraction (Rosch, 1975).

        Args:
            facts: The facts to form schemas from. If ``None``, all
                stored facts are used.

        Returns:
            The list of schemas formed or updated.
        """
        source_facts = facts if facts is not None else list(self._facts.values())
        now_ms = int(time.time() * 1000)
        schemas: list[Schema] = []

        concepts, as_subject, part_of_by_object = self._collect_schema_indices(source_facts)

        for concept_lower, display in concepts.items():
            schema = self._update_schema_for_concept(
                concept_lower, display, as_subject, part_of_by_object, now_ms
            )
            schemas.append(schema)

        return schemas

    def _collect_schema_indices(
        self, source_facts: list[Fact]
    ) -> tuple[dict[str, str], dict[str, list[Fact]], dict[str, list[Fact]]]:
        """Collect concept indices for schema formation."""
        # Collect every concept that appears (as subject or object).
        concepts: dict[str, str] = {}  # lower → display name
        for f in source_facts:
            concepts.setdefault(f.subject.lower(), f.subject)
            concepts.setdefault(f.object.lower(), f.object)

        # Facts where a concept is the subject, keyed by concept.
        as_subject: dict[str, list[Fact]] = {}
        # part_of facts keyed by the OBJECT concept (the whole): the
        # subject is a part of the object.
        part_of_by_object: dict[str, list[Fact]] = {}
        for f in source_facts:
            as_subject.setdefault(f.subject.lower(), []).append(f)
            if f.relation.lower() == "part_of":
                part_of_by_object.setdefault(f.object.lower(), []).append(f)

        return concepts, as_subject, part_of_by_object

    def _update_schema_for_concept(
        self,
        concept_lower: str,
        display: str,
        as_subject: dict[str, list[Fact]],
        part_of_by_object: dict[str, list[Fact]],
        now_ms: int,
    ) -> Schema:
        """Build or update a schema for a single concept."""
        schema = self._schemas.get(concept_lower)
        if schema is None:
            schema = Schema(
                concept=display,
                created_at=now_ms,
                last_updated=now_ms,
            )
            self._schemas[concept_lower] = schema

        # Parts: things that are part of this concept.
        for f in part_of_by_object.get(concept_lower, []):
            schema.parts.add(f.subject)
        # Subject-level facts.
        for f in as_subject.get(concept_lower, []):
            rel = f.relation.lower()
            obj = f.object
            if rel in ("has_property", "is_a"):
                if rel == "is_a":
                    schema.relations["is_a"] = obj
                else:
                    schema.properties.add(obj)
            elif rel in ("causes", "enables"):
                schema.functions.add(obj)
            elif rel == "relates_to":
                schema.relations["relates_to"] = obj

        # Track instances: number of distinct source episodes is
        # approximated by the sum of source counts of the facts
        # involving this concept.
        involved = as_subject.get(concept_lower, []) + part_of_by_object.get(concept_lower, [])
        schema.instances = sum(f.source_count for f in involved)
        # Abstraction level: asymptotes at 1.0 as instances grow
        # (prototype convergence, Rosch 1975).
        schema.abstraction_level = 1.0 - 1.0 / (1.0 + schema.instances * 0.5)
        schema.last_updated = now_ms
        self.schemas_formed += 1
        return schema

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve_facts(self, query: str) -> list[Fact]:
        """Retrieve facts relevant to a query.

        Matches facts whose subject, object, or relation contains any
        of the query terms. Results are ranked by confidence (facts
        reinforced from multiple episodes rank higher), with a boost
        for facts about concepts primed by global workspace broadcasts.

        Args:
            query: A free-text query (e.g., "brain hippocampus").

        Returns:
            Matching facts, highest confidence first.
        """
        terms = {w.lower() for w in _WORD_RE.findall(query) if w.lower()}
        if not terms:
            return []
        results: list[Fact] = []
        for fact in self._facts.values():
            blob = f"{fact.subject} {fact.relation} {fact.object}".lower()
            if any(t in blob for t in terms):
                results.append(fact)
        # Boost confidence for facts about primed concepts (semantic
        # priming from cognitive content).
        if self._primed_concepts:
            def _rank_key(f: Fact) -> float:
                """Sort key: confidence boosted by priming for relevant concepts."""
                priming = max(
                    self._primed_concepts.get(f.subject.lower(), 0.0),
                    self._primed_concepts.get(f.object.lower(), 0.0),
                )
                return f.confidence + priming * 0.3
            results.sort(key=_rank_key, reverse=True)
        else:
            results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def get_schema(self, concept_name: str) -> Schema | None:
        """Retrieve the schema for a concept, if one exists."""
        return self._schemas.get(concept_name.lower())

    def get_all_facts(self) -> list[Fact]:
        """Return all stored facts."""
        return list(self._facts.values())

    def get_all_schemas(self) -> list[Schema]:
        """Return all formed schemas."""
        return list(self._schemas.values())

    # ── Introspection ────────────────────────────────────────────

    @property
    def fact_count(self) -> int:
        """Number of distinct facts stored."""
        return len(self._facts)

    @property
    def schema_count(self) -> int:
        """Number of schemas formed."""
        return len(self._schemas)

    # ── Global workspace integration ──────────────────────────────

    def receive_broadcast(self, item: Any) -> None:
        """Receive a broadcast from the global workspace.

        This is **semantic priming** — when cognitive content ignites
        the workspace, related semantic knowledge is pre-activated for
        retrieval. In the brain, cognitive access to a concept primes
        associated semantic representations, speeding and biasing
        subsequent retrieval (Meyer & Schvaneveldt, 1971; Neely, 1977).

        The broadcast's topics are stored as primed concepts with
        strength proportional to the broadcast activation. When
        ``retrieve_facts`` is next called, facts about primed concepts
        get a confidence boost — they rise in the ranking.

        Priming decays: each new broadcast replaces the priming set,
        so only currently-cognitive content primes retrieval. This
        models the transient nature of semantic priming.

        Args:
            item: A :class:`WorkspaceItem` carrying the broadcast content.
        """
        topics = item.metadata.get("topics") if item.metadata else None
        if not topics:
            return
        strength = max(0.0, min(1.0, item.activation))
        for topic in topics:
            if topic and isinstance(topic, str):
                self._primed_concepts[topic.lower()] = strength


# ─── Helpers ──────────────────────────────────────────────────────────


def _normalise_pair(a: str, b: str) -> tuple[str, str]:
    """Normalise and validate a (subject, object) extraction pair.

    Strips leading articles ("the", "a", "an") from both sides so that
    "the brain" → "brain" and "an organ" → "organ". Returns empty
    strings if either side is empty.
    """
    subj = _strip_article(a.strip().lower())
    obj = _strip_article(b.strip().lower())
    if not subj or not obj:
        return "", ""
    if subj == obj:
        return "", ""
    return subj, obj


def _strip_article(phrase: str) -> str:
    """Strip a single leading article from a phrase."""
    for article in ("the ", "a ", "an "):
        if phrase.startswith(article):
            return phrase[len(article) :].strip()
    return phrase


def _relation_to_edge(relation: str) -> RelationType | None:
    """Map a fact relation string to a ConceptNetwork RelationType."""
    # Imported lazily to avoid a circular import at module load.
    from ..concepts import RelationType as _RT

    mapping = {
        "is_a": _RT.IS_A,
        "part_of": _RT.PART_OF,
        "causes": _RT.CAUSES,
        "enables": _RT.ENABLES,
        "relates_to": _RT.RELATED_TO,
        "has_property": _RT.HAS_PROPERTY,
        "emerges_from": _RT.EMERGES_FROM,
        "depends_on": _RT.DEPENDS_ON,
        "similar_to": _RT.SIMILAR_TO,
        "opposite_of": _RT.OPPOSITE_OF,
        "prevents": _RT.PREVENTS,
        "leads_to": _RT.LEADS_TO,
        "creates": _RT.CREATES,
        "harms": _RT.HARMS,
        "contradicts": _RT.CONTRADICTS,
    }
    return mapping.get(relation.lower())
