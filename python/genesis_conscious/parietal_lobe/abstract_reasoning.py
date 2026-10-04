"""Abstract reasoning — relational abstraction in the parietal cortex.

════════════════════════════════════════════════════════════════════════
WHAT THIS IS
════════════════════════════════════════════════════════════════════════

Abstract reasoning is the capacity to reason about *structure*
independent of *content*. "The atom is like the solar system" does
not require knowing what atoms or solar systems are made of — only
that both instantiate the same relational skeleton: a central body,
orbiting bodies, an attractive force. That skeleton is an
abstraction: surface features stripped away, relational structure
retained.

This module implements that capacity over the concept network. It is
distinct from the neighboring systems it composes with:

    - ReasoningEngine (reasoning/engine.py) traverses *concrete*
      edges: given concept A, what can be concluded about A.
      Content-bound inference.

    - AnalogyEngine (reasoning/analogy.py) maps between *two*
      concrete structures: given S and C, align them and project
      inferences. Pairwise transfer.

    - AbstractReasoner (this file) produces the *content-free*
      objects both of those depend on: relational skeletons
      (schemas), complexity measures, least-general generalizations
      over multiple instances, and n-ary relational systems. It is
      upstream of inference — it answers "what is the structure
      here?" before anyone asks "what follows from it?".

════════════════════════════════════════════════════════════════════════
ANATOMY
════════════════════════════════════════════════════════════════════════

Abstract/relational reasoning is frontoparietal, with the parietal
contribution dominant for the representational core:

    - The posterior parietal cortex, especially the intraparietal
      sulcus (IPS), encodes relational structure and scales its
      activity with *relational complexity* — the number of entities
      that must be bound simultaneously (Halford, Wilson & Phillips,
      1998; Vendetti & Bunge, 2014). Rostrolateral PFC contributes
      the second-order comparisons; the parietal cortex holds the
      bound representation itself.

    - Raven's Progressive Matrices — the standard psychometric for
      abstract reasoning — recruit IPS and superior parietal lobule
      most strongly, and difficulty tracks relational complexity
      (Christoff et al., 2001; Carpenter, Just & Shell, 1990). The
      spatial solver (spatial/solver.py) is Genesis's RPM-style
      machinery for visual matrices; this module is the same
      capacity over her symbolic knowledge — the dorsal stream's
      structural parsing applied to the concept network.

    - Multisensory integration in the superior parietal lobule is
      what makes content-free structure possible: representations
      arrive already partially abstracted from their modality of
      origin.

So the parietal lobe is where abstractions live; the frontal lobe
(reasoning/, executive.py) is where they get *used* for deduction,
planning, and problem-solving.

════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

1. Relational skeleton (abstraction proper)
-------------------------------------------

For concept c, take its incident edges E(c) = {(s, r, t)} and replace
every endpoint with a variable, holding c's own position fixed:

    skeleton(c) = { (?c, r_i, ?x_i) }  ∪  { (?x_j, r_j, ?c) }

Two concepts with isomorphic skeletons share structure regardless of
what they are about. The skeleton is the abstraction.

2. Relational complexity (Halford et al., 1998)
-----------------------------------------------

A unary relation binds 1 entity, binary binds 2, ternary 3,
quaternary 4. Processing load — and parietal activation — rises with
the largest relation that must be represented *jointly*. We compute
the complexity of a concept's local relational system as the number
of distinct entities bound into the largest connected relational
component, capped at QUATERNARY (humans cannot exceed it without
chunking; we report ``saturated=True`` when chunking would be
required).

3. Anti-unification (least general generalization)
--------------------------------------------------

Given several concrete instances, induction is the dual of
unification: find the *most specific* pattern that covers all of
them (Plotkin, 1970). For each relation type shared across the
instance skeletons, align the atoms; where aligned endpoints agree,
keep the constant; where they differ, introduce a shared variable.
The result is the least general schema — the abstraction of the
instance set — with ``coverage`` measuring how much of the concrete
structure survived generalization.

4. Structural isomorphism
-------------------------

A schema matches a candidate concept when the candidate's skeleton
contains the schema's (relation, direction) pattern — surface
identity ignored. Matching across cortical columns is the
abstraction criterion: a schema induced in one domain that matches
concepts in another column is a genuine cross-domain abstraction,
not a similarity echo.

5. Relational binding (n-ary systems)
-------------------------------------

Reasoning about a *set* of concepts requires binding them into one
representation: every explicit edge among the set, plus shared
mediators (concepts adjacent to ≥2 members). The bound system is an
n-ary relation over the members — the object frontoparietal circuits
actually reason over, and the input complexity is measured on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from ..concepts import ConceptNetwork, RelationType

__all__ = [
    "AbstractReasoner",
    "RelationalComplexity",
    "RelationalSchema",
    "RelationalSystem",
    "SchemaAtom",
]


class RelationalComplexity(Enum):
    """Halford relational complexity levels.

    The number of entities that must be bound simultaneously into a
    single relational representation. Human working capacity tops out
    at quaternary; beyond that, systems must be chunked.
    """

    UNARY = 1  # a single attribute of one entity
    BINARY = 2  # a relation between two entities
    TERNARY = 3  # three entities bound jointly (e.g. "X gives Y to Z")
    QUATERNARY = 4  # four entities; the human ceiling

    @classmethod
    def from_entity_count(cls, n: int) -> RelationalComplexity:
        """Map a bound-entity count to a Halford complexity level."""
        return {
            0: cls.UNARY,
            1: cls.UNARY,
            2: cls.BINARY,
            3: cls.TERNARY,
        }.get(n, cls.QUATERNARY)


@dataclass(frozen=True, slots=True)
class SchemaAtom:
    """One atom of a relational schema: (subject, relation, object).

    ``source`` and ``target`` are either concrete concept ids or
    variables (prefixed ``?``). A fully-variable atom is pure
    structure — "something stands in relation R to something else".
    """

    source: str
    relation: RelationType
    target: str

    @property
    def is_abstract(self) -> bool:
        """Return whether either side of the atom is an open variable."""
        return self.source.startswith("?") or self.target.startswith("?")

    def describe(self) -> str:
        """Render the atom as a compact source-relation-target clause."""
        return f"{self.source} --{self.relation.value}--> {self.target}"


@dataclass(slots=True)
class RelationalSchema:
    """A content-light relational pattern abstracted from the graph.

    ``atoms`` are the schema's relational clauses. ``variables`` are
    the open slots. ``instances`` records which concrete concepts the
    schema was abstracted from. ``coverage`` (0..1) is the fraction of
    instance structure the schema accounts for — low coverage means
    the instances were structurally diverse and the common core is
    thin.
    """

    atoms: list[SchemaAtom]
    variables: set[str] = field(default_factory=set)
    instances: list[str] = field(default_factory=list)
    coverage: float = 1.0

    @property
    def complexity(self) -> RelationalComplexity:
        """Return the Halford complexity implied by the schema's bound entities."""
        bound = len({a.source for a in self.atoms} | {a.target for a in self.atoms})
        return RelationalComplexity.from_entity_count(bound)

    @property
    def relations(self) -> set[RelationType]:
        """Return the relation types used by the schema."""
        return {a.relation for a in self.atoms}

    def describe(self) -> str:
        """Render the schema's clauses and coverage."""
        if not self.atoms:
            return "empty schema"
        clauses = "; ".join(a.describe() for a in self.atoms)
        return f"[{clauses}] (coverage {self.coverage:.2f})"


@dataclass(slots=True)
class RelationalSystem:
    """An n-ary relational system binding a set of concepts.

    ``members`` are the bound concepts; ``atoms`` are the concrete
    edges among them plus edges through shared mediators; ``mediators``
    are the bridging concepts. ``complexity`` is the Halford level of
    the whole bound representation and ``saturated`` marks systems
    that exceed quaternary capacity (chunking required).
    """

    members: list[str]
    atoms: list[SchemaAtom]
    mediators: list[str]
    complexity: RelationalComplexity
    saturated: bool
    coherence: float  # 0..1, how densely the members interconnect

    def describe(self) -> str:
        """Render the bound members, relation count, complexity, and coherence."""
        if not self.atoms:
            return f"{self.members}: no relational structure binding them"
        return (
            f"{self.members}: {len(self.atoms)} relations "
            f"({self.complexity.name.lower()}-complex"
            f"{', saturated' if self.saturated else ''}, "
            f"coherence {self.coherence:.2f})"
        )


class AbstractReasoner:
    """Relational abstraction over the concept network.

    Produces the content-free structural objects — skeletons,
    schemas, complexity measures, bound systems — that the frontal
    reasoning machinery consumes. All operations are pure graph
    computation: no surface string matching, no templates.
    """

    # Relations that carry semantic structure. Code-graph relations
    # (CALLS, DEFINES) and vocabulary links (EXPRESSES) are structural
    # bookkeeping, not conceptual structure — abstracting them would
    # produce schemas about her implementation, not her knowledge.
    _STRUCTURAL: ClassVar[set[RelationType]] = {
        r for r in RelationType
        if r
        not in {
            RelationType.CALLS,
            RelationType.DEFINES,
            RelationType.EXPRESSES,
            RelationType.BRIDGES,
        }
    }

    def __init__(self, network: ConceptNetwork) -> None:
        """Create a relational-abstraction reasoner over a concept network."""
        self.network = network

    # ── Abstraction ──────────────────────────────────────────────

    def skeleton(self, concept: str, anchor: str = "?self") -> RelationalSchema | None:
        """Abstract a concept into its relational skeleton.

        Every incident structural edge becomes a schema atom with the
        neighbor replaced by a variable; the concept's own position
        stays fixed as ``anchor``. Two concepts with equal skeleton
        relations play the same structural role no matter what they
        are about.
        """
        edges = self.network.get_edges(concept, direction="both")
        edges = [e for e in edges if e.relation in self._STRUCTURAL]
        if not edges:
            return None

        cid = self.network.get_concept(concept)
        self_id = cid.id if cid else concept
        atoms: list[SchemaAtom] = []
        variables: set[str] = set()
        for i, e in enumerate(edges):
            if e.source == self_id:
                var = f"?x{i}"
                variables.add(var)
                atoms.append(SchemaAtom(anchor, e.relation, var))
            else:
                var = f"?x{i}"
                variables.add(var)
                atoms.append(SchemaAtom(var, e.relation, anchor))

        return RelationalSchema(
            atoms=atoms, variables=variables, instances=[self_id]
        )

    def induce(self, instances: list[str]) -> RelationalSchema | None:
        """Least-general generalization over a set of instances.

        Anti-unifies the instances' skeletons: relations present in
        *every* instance survive; endpoints that disagree across
        instances generalize to shared variables; endpoints that agree
        stay constant only when they refer to the same external
        concept. The result is the abstract rule the instances
        jointly instantiate — e.g. inducing over {fire, friction,
        growth} when each ``causes`` something and ``depends_on``
        something yields ``?x causes ?y; ?x depends_on ?z``.
        """
        skeletons: list[tuple[str, RelationalSchema]] = []
        for name in instances:
            s = self.skeleton(name)
            if s is not None:
                skeletons.append((name, s))
        if not skeletons:
            return None

        # Relations every instance exhibits (the invariant backbone).
        shared = set.intersection(*[s.relations for _, s in skeletons])
        if not shared:
            return None

        atoms: list[SchemaAtom] = []
        variables: set[str] = set()
        covered = 0
        total = sum(len(s.atoms) for _, s in skeletons)

        var_i = 0
        for rel in sorted(shared, key=lambda r: r.value):
            # Collect, per instance, the endpoint sets for this
            # relation in each direction relative to the anchor.
            out_sets: list[set[str]] = []
            in_sets: list[set[str]] = []
            for _, s in skeletons:
                out_sets.append(
                    {a.target for a in s.atoms if a.relation == rel and a.source == "?self"}
                )
                in_sets.append(
                    {a.source for a in s.atoms if a.relation == rel and a.target == "?self"}
                )

            # Outgoing invariant: every instance has self --rel--> something.
            if all(out_sets):
                common = set.intersection(*out_sets)
                if common:
                    # Same concrete target in all instances — a true
                    # shared endpoint, not a variable.
                    for t in sorted(common):
                        atoms.append(SchemaAtom("?self", rel, t))
                else:
                    v = f"?y{var_i}"
                    var_i += 1
                    variables.add(v)
                    atoms.append(SchemaAtom("?self", rel, v))
                covered += sum(len(o) for o in out_sets)

            # Incoming invariant: every instance has something --rel--> self.
            if all(in_sets):
                common = set.intersection(*in_sets)
                if common:
                    for t in sorted(common):
                        atoms.append(SchemaAtom(t, rel, "?self"))
                else:
                    v = f"?y{var_i}"
                    var_i += 1
                    variables.add(v)
                    atoms.append(SchemaAtom(v, rel, "?self"))
                covered += sum(len(i) for i in in_sets)

        if not atoms:
            return None
        coverage = covered / total if total else 0.0
        return RelationalSchema(
            atoms=atoms,
            variables=variables | {"?self"},
            instances=[name for name, _ in skeletons],
            coverage=coverage,
        )

    # ── Complexity ───────────────────────────────────────────────

    def relational_complexity(self, concept: str) -> tuple[RelationalComplexity, bool]:
        """Halford complexity of a concept's bound relational system.

        Counts the distinct entities (self + structural neighbors)
        that must be jointly represented to hold the concept's
        relational structure. Returns (level, saturated) where
        saturated means the true arity exceeds quaternary — the
        system must be chunked before frontal reasoning can hold it.
        """
        neighbors = {
            n for n, rel, _w in self.network.get_neighbors(concept)
            if rel in self._STRUCTURAL
        }
        bound = 1 + len(neighbors)  # self plus each bound entity
        saturated = bound > RelationalComplexity.QUATERNARY.value
        return RelationalComplexity.from_entity_count(min(bound, 4)), saturated

    # ── Isomorphism ──────────────────────────────────────────────

    def find_isomorphs(
        self,
        schema: RelationalSchema,
        *,
        exclude_instances: bool = True,
        column: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Find concepts whose skeletons instantiate a schema.

        Structure-only matching: a candidate matches atom
        ``(?a, R, ?b)`` when it has any incident edge with relation R
        in the same direction relative to the anchor — regardless of
        what the other endpoint is. Returns (concept_id, score) where
        score is the fraction of schema atoms satisfied, sorted
        descending. Full matches first.

        ``column`` restricts candidates to one cortical column —
        passing a column different from the schema's origin column is
        the cross-domain test: the abstraction is real only if it
        re-instantiates under different surface content.
        """
        candidates = (
            self.network.get_column_concepts(column)
            if column is not None
            else self.network.concept_ids
        )
        excluded = set(schema.instances) if exclude_instances else set()

        results: list[tuple[str, float]] = []
        for cid in candidates:
            if cid in excluded:
                continue
            edges = self.network.get_edges(cid, direction="both")
            out_rels = {e.relation for e in edges if e.source == cid}
            in_rels = {e.relation for e in edges if e.target == cid}
            matched = 0
            for atom in schema.atoms:
                if atom.source == "?self" and atom.relation in out_rels:
                    matched += 1
                elif atom.target == "?self" and atom.relation in in_rels:
                    matched += 1
            if matched:
                results.append((cid, matched / len(schema.atoms)))

        results.sort(key=lambda kv: (-kv[1], kv[0]))
        return results[:limit]

    # ── Binding ──────────────────────────────────────────────────

    def bind(self, concepts: list[str]) -> RelationalSystem:
        """Bind a set of concepts into one n-ary relational system.

        Collects every explicit structural edge among the members
        plus edges through shared mediators (non-member concepts
        adjacent to ≥2 members — the bridges that make the set a
        *system* rather than a list). This is the bound representation
        the frontoparietal network actually reasons over.
        """
        members = sorted(set(concepts))
        member_set = set(members)
        atoms: list[SchemaAtom] = []
        mediators: set[str] = set()

        # Direct member-to-member relations.
        for m in members:
            for e in self.network.get_edges(m, direction="out"):
                if e.relation not in self._STRUCTURAL:
                    continue
                if e.target in member_set:
                    atoms.append(SchemaAtom(e.source, e.relation, e.target))
                else:
                    # Potential mediator: check whether it touches
                    # another member.
                    other_neighbors = {
                        n for n, rel, _w in self.network.get_neighbors(e.target)
                        if rel in self._STRUCTURAL
                    }
                    if other_neighbors & (member_set - {m}):
                        mediators.add(e.target)
                        atoms.append(SchemaAtom(e.source, e.relation, e.target))

        # Incoming mediator edges (mediator --rel--> member).
        for m in members:
            for e in self.network.get_edges(m, direction="in"):
                if e.relation not in self._STRUCTURAL or e.source in member_set:
                    continue
                neighbors = {
                    n for n, rel, _w in self.network.get_neighbors(e.source)
                    if rel in self._STRUCTURAL
                }
                if neighbors & (member_set - {m}):
                    mediators.add(e.source)
                    atoms.append(SchemaAtom(e.source, e.relation, e.target))

        bound = len(member_set) + len(mediators)
        saturated = bound > RelationalComplexity.QUATERNARY.value
        complexity = RelationalComplexity.from_entity_count(min(bound, 4))

        # Coherence: realized member-to-member connections over
        # possible directed pairs.
        possible = len(member_set) * (len(member_set) - 1)
        direct = sum(1 for a in atoms if a.source in member_set and a.target in member_set)
        coherence = direct / possible if possible else 0.0

        return RelationalSystem(
            members=members,
            atoms=atoms,
            mediators=sorted(mediators),
            complexity=complexity,
            saturated=saturated,
            coherence=coherence,
        )
