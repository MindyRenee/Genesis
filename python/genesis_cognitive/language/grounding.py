"""Ground language propositions in Genesis's persistent concept network.

This module deliberately sits between linguistic parsing and cognition.
It does not create concepts or mutate the network: comprehension names
are resolved against the existing semantic substrate, while unresolved
words remain unresolved for the learning system to handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .comprehension import Proposition, SemanticRole

if TYPE_CHECKING:
    from ..concepts.network import ConceptNetwork


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
    """Resolve comprehension arguments against an existing ConceptNetwork."""

    _LEADING_DETERMINERS = frozenset({"the", "a", "an"})

    def __init__(self, network: ConceptNetwork) -> None:
        self.network = network

    def ground_argument(
        self, surface: str, *, context: str = ""
    ) -> GroundedArgument:
        """Resolve a surface phrase through the network, using context for polysemy."""
        text = surface.strip()
        if not text:
            return GroundedArgument("", None, 0.0)

        candidates = [text]
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in self._LEADING_DETERMINERS:
            candidates.append(parts[1])

        for candidate in candidates:
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
            return GroundedArgument(text, concept_id, confidence)

        return GroundedArgument(text, None, 0.0)

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
