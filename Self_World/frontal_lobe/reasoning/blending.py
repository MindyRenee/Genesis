"""Conceptual blending — innovation through integration (Fauconnier
& Turner, The Way We Think, 2002).

Conceptual blending is how genuinely new concepts arise: not by
accumulation but by *integration* of separate mental spaces into a
blended space whose structure belongs to neither input. The
machinery, formalized:

- **Mental spaces** are small structured packets — elements with
  properties and relations, built for local purposes.
- A **cross-space mapping** links counterpart elements — the same
  partial structure seen from two domains.
- The **generic space** is what the inputs share: the counterpart
  pairs and their common structure.
- **Selective projection** brings chosen elements and relations
  from each input into the blend — not everything, only what the
  blend's purpose needs.
- The blend develops **emergent structure** through three routes:
  *composition* (fused elements gain relations the inputs could not
  host together), *completion* (frame knowledge fills in the rest
  of a partially matched pattern), and *elaboration* (running the
  blend unfolds consequences present in neither input).

Emergence is what makes this innovation rather than analogy: the
blend contains relational structure that exists in neither input —
a new concept, with its own inferences, that could not have been
read off either space alone. The canonical case is the Debate with
Kant: a modern philosopher "argues with" Kant — the blend runs two
people separated by two centuries in one room, and *argues* is
emergent structure that exists in neither input space.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

#: Elements with less than this shared-property overlap are not
#: counterparts — the mapping refuses structure the inputs don't share.
_MIN_OVERLAP = 0.0


@dataclass(frozen=True)
class Space:
    """A mental space — a packet of elements, their properties, and
    the relations among them, built for local purposes."""

    name: str
    elements: Mapping[str, frozenset[str]]  # element -> properties
    relations: frozenset[tuple[str, str, str]] = frozenset()  # (rel, a, b)

    def elements_relating(self, rel: str, x: str) -> frozenset[str]:
        return frozenset(b for r, a, b in self.relations if r == rel and a == x)


@dataclass(frozen=True)
class Blend:
    """The blended space — selective projection plus what emerged."""

    elements: Mapping[str, frozenset[str]]  # blended element -> properties
    relations: frozenset[tuple[str, str, str]]
    mapping: Mapping[str, str]  # input-A element -> input-B element
    generic: frozenset[tuple[str, str]]  # counterpart pairs
    emergent_relations: frozenset[tuple[str, str, str]]  # in neither input
    emergent_properties: Mapping[str, frozenset[str]]  # blend elt -> new props

    @property
    def novel(self) -> bool:
        """Whether the blend holds structure neither input had —
        the innovation criterion."""
        return bool(self.emergent_relations or self.emergent_properties)


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def align(a: Space, b: Space) -> dict[str, str]:
    """The cross-space counterpart map — greedy best-match on shared
    properties and relational role, the way structure-mapping aligns
    domains. Each B-element pairs with at most one A-element."""
    mapping: dict[str, str] = {}
    taken: set[str] = set()

    def role_sig(space: Space, x: str) -> frozenset[str]:
        return frozenset(r for r, s, _ in space.relations if s == x) | frozenset(
            r for r, _, s in space.relations if s == x
        )

    for ea, pa in a.elements.items():
        best, best_score = None, _MIN_OVERLAP
        for eb, pb in b.elements.items():
            if eb in taken:
                continue
            score = 0.5 * _similarity(pa, pb) + 0.5 * _similarity(role_sig(a, ea), role_sig(b, eb))
            if score > best_score:
                best, best_score = eb, score
        if best is not None:
            mapping[ea] = best
            taken.add(best)
    return mapping


def blend(
    a: Space,
    b: Space,
    mapping: Mapping[str, str] | None = None,
    *,
    name: str = "blend",
) -> Blend:
    """Integrate two mental spaces into a blend.

    With ``mapping`` None the counterparts are aligned by shared
    structure. Fused pairs carry the union of their properties;
    unmatched elements project through on their own. Every relation
    from either input is re-instantiated on the blended elements it
    mentions — and a relation that instantiates across the spaces
    (one relatum fused, or the relation transplanted into structure
    the source lacked) is **emergent**: structure the blend holds
    that no input did.
    """
    mapping = dict(mapping if mapping is not None else align(a, b))
    generic = frozenset(mapping.items())

    # Blended element names: fused pairs take "a|b", projected take
    # their own name. Elements from B not in the mapping keep theirs.
    fused: dict[str, str] = {}  # source element -> blend element
    elements: dict[str, frozenset[str]] = {}
    for ea, eb in mapping.items():
        be = f"{ea}|{eb}"
        fused[ea] = fused[eb] = be
        elements[be] = frozenset(a.elements.get(ea, ())) | frozenset(b.elements.get(eb, ()))
    for e, props in a.elements.items():
        if e not in fused:
            fused[e] = e
            elements[e] = frozenset(props)
    for e, props in b.elements.items():
        if e not in fused:
            fused[e] = e
            elements[e] = frozenset(props)

    # Re-instantiate each input's relations on blended elements, and
    # track provenance: which input(s) each blend element is made of.
    # A relation is emergent when it binds material from the space
    # that did not supply the relation — the blend asserts, of the
    # other input's element, a structure that input never had.
    a_material: set[str] = set()  # blend elements containing A-material
    b_material: set[str] = set()
    for ea, eb in mapping.items():
        a_material.add(fused[ea])
        b_material.add(fused[eb])
    for e in a.elements:
        if e not in mapping:
            a_material.add(e)
    for e in b.elements:
        if e not in mapping.values() and e not in mapping:
            b_material.add(e)

    relations: set[tuple[str, str, str]] = set()
    emergent_rels: set[tuple[str, str, str]] = set()
    for space, other in ((a, b_material), (b, a_material)):
        for r, x, y in space.relations:
            inst = (r, fused.get(x, x), fused.get(y, y))
            relations.add(inst)
            if inst[1] in other or inst[2] in other:
                emergent_rels.add(inst)

    # Emergent properties: on a fused element, properties contributed
    # by the *other* side — structure the element gains in the blend
    # that its source lacked.
    emergent_props: dict[str, frozenset[str]] = {}
    for ea, eb in mapping.items():
        be = fused[ea]
        new_for_a = frozenset(b.elements.get(eb, ())) - frozenset(a.elements.get(ea, ()))
        new_for_b = frozenset(a.elements.get(ea, ())) - frozenset(b.elements.get(eb, ()))
        gained = new_for_a | new_for_b
        if gained:
            emergent_props[be] = gained

    return Blend(
        elements=elements,
        relations=frozenset(relations),
        mapping=mapping,
        generic=generic,
        emergent_relations=frozenset(emergent_rels),
        emergent_properties=emergent_props,
    )


__all__ = ["Blend", "Space", "align", "blend"]
