"""Ground language propositions in Genesis's persistent concept network.

This module deliberately sits between linguistic parsing and cognition.
It does not create concepts or mutate the network: comprehension names
are resolved against the existing semantic substrate, while unresolved
words remain unresolved for the learning system to handle.

The boundary is:

    Proposition (language) -> GroundedProposition (cognition-ready)

A predicate remains lexical here. Mapping verbs to RelationType is a
separate decision because many verbs are context-sensitive and should
not be collapsed into a graph relation prematurely.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .comprehension import Proposition, SemanticRole

if TYPE_CHECKING:
    from ..concepts.network import ConceptNetwork


# Leading determiners that carry no content and are never part of a
# stored concept name.
_DETERMINER_RE = re.compile(
    r"^(?:the|a|an|this|that|these|those|some|any|each|every|"
    r"my|your|his|her|its|our|their)\s+",
    re.IGNORECASE,
)
# Possessive / plural endings: "alice's" -> "alice", "dogs" handled by
# the caller's own normalization.
_POSSESSIVE_RE = re.compile(r"['’]s$", re.IGNORECASE)


def _strip_determiner(text: str) -> str:
    """Drop a leading determiner, if present."""
    return _DETERMINER_RE.sub("", text, count=1)


def _strip_possessive(text: str) -> str:
    """Drop a possessive clitic, if present."""
    return _POSSESSIVE_RE.sub("", text)


@dataclass(frozen=True, slots=True)
class GroundedArgument:
    """One linguistic argument and its best existing concept binding."""

    surface: str
    concept_id: str | None
    confidence: float


@dataclass(frozen=True, slots=True)
class GroundedProposition:
    """A proposition whose arguments are bound to concept IDs when possible."""

    source: Proposition
    subject: GroundedArgument
    predicate: str
    object: GroundedArgument
    roles: dict[SemanticRole, GroundedArgument] = field(default_factory=dict)

    @property
    def grounding_confidence(self) -> float:
        """Conservative confidence for the complete proposition grounding."""
        values = [self.subject.confidence, self.object.confidence]
        values.extend(arg.confidence for arg in self.roles.values())
        return min(values) if values else 0.0

    @property
    def fully_grounded(self) -> bool:
        """Whether every non-empty argument has a concept binding."""
        args = [self.subject, self.object, *self.roles.values()]
        return all(not arg.surface or arg.concept_id is not None for arg in args)


class SemanticGrounder:
    """Resolve comprehension arguments against an existing ConceptNetwork.

    Grounding is read-only. Unknown language does not become a fabricated
    concept merely because it was spoken. Learning remains responsible
    for deciding when and how an unknown concept should be added.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        self.network = network

    def ground_argument(
        self, surface: str, *, context: str = ""
    ) -> GroundedArgument:
        """Resolve a surface phrase through the network, using context for polysemy."""
        text = surface.strip()
        if not text:
            return GroundedArgument("", None, 0.0)

        # Grounding is read-only: an unknown surface must not become a
        # fabricated concept. The determiner fallback below respects
        # that — it can only ever *find* a concept that already exists,
        # never add one.
        for candidate in self._lookup_candidates(text):
            if context:
                concept_id = self.network.resolve_in_context(candidate, context)
                concept = (
                    self.network.get_concept(concept_id)
                    if concept_id is not None
                    else None
                )
            else:
                concept = self.network.get_concept(candidate)
            if concept is None:
                continue
            concept_id = getattr(concept, "id", None)
            if not isinstance(concept_id, str) or not concept_id:
                continue
            confidence = float(getattr(concept, "confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            # `surface` is reported as given, not as the stripped form
            # that happened to match — the caller's text is unchanged.
            return GroundedArgument(text, concept_id, confidence)

        return GroundedArgument(text, None, 0.0)

    @staticmethod
    def _lookup_candidates(text: str) -> Iterator[str]:
        """Surface forms to try, most faithful first.

        Comprehension emits noun phrases with determiners and
        possessives attached — "The dog", "Alice's book" — while
        concepts are stored as bare names. Looking the surface up
        verbatim therefore failed on exactly the sentences most likely
        to be grounded, and silently: the object of the same sentence
        ("water") resolved while the subject ("The dog") did not.

        Yields the surface as given, then determiner-stripped, then
        possessive-stripped, deduplicated, preserving order.
        """
        seen: set[str] = set()
        for form in (text, _strip_determiner(text), _strip_possessive(text)):
            key = form.strip()
            if key and key.casefold() not in seen:
                seen.add(key.casefold())
                yield key

    def ground(
        self, proposition: Proposition, *, context: str = ""
    ) -> GroundedProposition:
        """Ground one proposition without changing linguistic information."""
        roles = {
            role: self.ground_argument(value, context=context)
            for role, value in proposition.roles.items()
            if value
        }
        return GroundedProposition(
            source=proposition,
            subject=self.ground_argument(proposition.subject, context=context),
            predicate=proposition.predicate,
            object=self.ground_argument(proposition.object, context=context),
            roles=roles,
        )

    def ground_all(
        self, propositions: list[Proposition], *, context: str = ""
    ) -> list[GroundedProposition]:
        """Ground a complete comprehension result's propositions."""
        return [self.ground(proposition, context=context) for proposition in propositions]


__all__ = [
    "GroundedArgument",
    "GroundedProposition",
    "SemanticGrounder",
]
