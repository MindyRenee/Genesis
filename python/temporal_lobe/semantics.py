"""Semantic synthesis — the meaning of the whole, composed from the
meaning of its parts.

Frege's principle of compositionality: the sense of a complex
expression is determined by the senses of its parts and the mode of
their combination. Comprehension supplies the constituents — a
``Proposition``'s subject, predicate, roles, and embeddings; this
module composes them into the sense they collectively express.

The anterior temporal lobe is the semantic hub of this kind — the
site where distributed feature representations converge into
single coherent concepts (Patterson, Nestor & Rogers 2007). What
holds for concepts holds for sentences: synthesis is convergence.

Two Fregean distinctions are load-bearing:

- **Sense and reference diverge at embeddings.** In "John believes
  that Mary is in the kitchen", the embedded clause does not
  contribute its truth-value to the whole — John can be wrong. It
  contributes its *sense*: the content believed. ``synthesize``
  composes complement clauses into ``complementation`` senses —
  the meaning the matrix predicate operates on is the embedded
  sense itself, not its reference. Checking that sense against the
  world is a different act (``parietal_lobe.correspondence``), and
  it must not be collapsed into composition.

- **The analytic is truth by convention** (Lewis's other
  "Convention", Carnap's meaning postulates): some truths the
  semantic rules alone settle — "bachelors are unmarried" holds in
  every world consistent with the postulates, no observation
  needed. ``analytic`` tests a formula against a postulate set;
  what they cannot settle is synthetic — left to the world, which
  is exactly where correspondence takes over.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from frontal_lobe.reasoning.logic import Formula, entails
from temporal_lobe.comprehension import Proposition, SemanticRole

# Composition modes — the ways constituents may combine into a
# sense, after Frege (saturated/unsaturated) and the parser's own
# structural output.
PREDICATION = "predication"  # subject saturates the predicate's first place
ROLE_FILLING = "role_filling"  # arguments saturate role slots
COMPLEMENTATION = "complementation"  # a clause's sense fills an argument slot
ATTRIBUTION = "attribution"  # an object/attribute is ascribed to the subject


@dataclass(frozen=True)
class Sense:
    """The composed meaning of an expression — an intension.

    ``mode`` records which composition rule produced the node;
    ``head`` the unsaturated core (the predicate or functor);
    ``arguments`` the saturating parts keyed by the slot they fill
    (``subject``/``object``/semantic-role names); ``embedded`` the
    senses of complement clauses — contributed as senses, never as
    references; ``negated`` the outer truth-condition's sign.
    """

    mode: str
    head: str
    arguments: tuple[tuple[str, str], ...] = ()
    embedded: tuple[Sense, ...] = ()
    negated: bool = False

    def key(self) -> tuple:
        """Canonical identity — two parses compose the same sense iff
        they express the same meaning."""
        return (
            self.mode,
            self.head,
            tuple(sorted(self.arguments)),
            tuple(e.key() for e in self.embedded),
            self.negated,
        )


def synthesize(prop: Proposition) -> Sense:
    """Compose the sense of a proposition — recursively, embedding
    first (complement clauses contribute their sense, obliquely),
    then saturating the matrix predicate's open places."""
    embedded = tuple(synthesize(e) for e in (prop.embedded,) if e is not None)
    mode = COMPLEMENTATION if embedded else _mode_of(prop)

    arguments: list[tuple[str, str]] = []
    # Passive voice: the grammatical subject is the patient — the
    # agent, when present, fills the agent slot instead.
    if prop.passive:
        if prop.subject:
            arguments.append(("patient", prop.subject))
        agent = prop.roles.get(SemanticRole.AGENT)
        if agent:
            arguments.append(("agent", agent))
    elif prop.subject:
        arguments.append(("subject", prop.subject))
    if prop.object:
        arguments.append(("object", prop.object))
    for role, value in sorted(prop.roles.items(), key=lambda kv: kv[0].name):
        if prop.passive and role is SemanticRole.AGENT:
            continue
        arguments.append((role.name.lower(), value))

    return Sense(
        mode=mode,
        head=(prop.predicate or "").strip(),
        arguments=tuple(arguments),
        embedded=embedded,
        negated=prop.negated,
    )


def _mode_of(prop: Proposition) -> str:
    """Which composition rule this proposition's shape selects."""
    if prop.roles:
        return ROLE_FILLING
    if prop.object:
        return ATTRIBUTION
    return PREDICATION


def analytic(
    f: Formula,
    postulates: Iterable[Formula],
) -> bool:
    """Truth by convention — f is analytic under the meaning
    postulates iff they entail it: the semantic rules alone settle
    the question, no world required. What they leave open is
    synthetic — settled, if at all, by what is the case."""
    return entails(postulates, f)


def analytic_status(
    f: Formula,
    postulates: Iterable[Formula],
) -> str:
    """The three-way Carnapian verdict: the postulates settle f
    (analytic), settle ¬f (contradictory — analytically false), or
    stay silent (synthetic)."""
    if entails(postulates, f):
        return "analytic"
    from frontal_lobe.reasoning.logic import not_

    if entails(postulates, not_(f)):
        return "contradictory"
    return "synthetic"


__all__ = [
    "ATTRIBUTION",
    "COMPLEMENTATION",
    "PREDICATION",
    "ROLE_FILLING",
    "Sense",
    "analytic",
    "analytic_status",
    "synthesize",
]
