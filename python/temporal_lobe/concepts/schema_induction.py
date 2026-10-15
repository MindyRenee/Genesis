"""Schema induction — reification of repeated relational structure.

The edge proposer finds pairs that *look* alike (embedding proximity).
The analogy engine finds pairs that *work* alike (relational profile).
This module does the third thing: when many concepts stand in the same
relation to the same endpoint, it *reifies the shared role into a
concept of its own*.

Given observed edges

    caffeine -CAUSES-> alertness
    exercise -CAUSES-> alertness
    morning  -CAUSES-> alertness

the inducer creates a schema hub ``_gen:causes:to:alertness`` —
"the things that cause alertness" — writes each member under it
(``caffeine -IS_A-> _gen:causes:to:alertness``), and asserts the
compressed claim ``_gen:causes:to:alertness -CAUSES-> alertness``.

K earned edges become one concept, K membership edges, and one
schema-level edge. The schema node is born ``origin="reified"`` —
a concept the system constructed rather than was told — and it is
eligible input to later induction passes: ``_gen:`` hubs appearing as
members or endpoints of structural edges produce schemas over
schemas, which is how middle layers of the taxonomy form instead of
everything hanging one step below a top category.

Two directions are induced:

- ``to``   — slot-fillers over a shared target:
  ``{S_i -R-> T}`` → ``_gen:<R>:to:<T>`` ("things that stand in R to T").
- ``from`` — roles over a shared source:
  ``{S -R-> T_i}`` → ``_gen:<R>:from:<S>`` ("things S stands in R to").

Eligibility is deliberately narrow. Only typed, asymmetric structural
relations induce (CAUSES, ENABLES, PREVENTS, ...): bare association
(RELATED_TO, SIMILAR_TO) carries no class structure, taxonomic
relations (IS_A, INSTANCE_OF) already *are* the class, and EXPRESSES
is already the ``_cat:`` hub mechanism — reifying it would just
rebuild existing hubs. Endpoints and members must be world concepts
or internal category/schema hubs (``_cat:``, ``_gen:``); code symbols,
utterance seeds, and task markers are excluded.

Consistency: writing is idempotent across passes. Existing reified
edges are refreshed to the computed weight via ``set_edge_weight``
rather than re-asserted (``add_edge``'s +0.2 reinforcement bump would
saturate them within a few nights). Membership that would close an
IS_A cycle is skipped.

Runs during sleep alongside the edge proposer and analogy engine via
``discover_and_accept()`` — hippocampal consolidation does not just
re-sort associations, it abstracts over them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .classify import is_world_concept
from .types import RelationType

if TYPE_CHECKING:
    from .network import ConceptNetwork

__all__ = ["ReifiedSchema", "SchemaInducer"]

logger = logging.getLogger(__name__)

# Origin tag for everything induction writes. Deliberately not in
# DERIVABLE_ORIGINS — reified schemas are earned compressions, not
# geometry materialized as edges.
_ORIGIN_REIFIED = "reified"

_SCHEMA_PREFIX = "_gen:"

# Relations that induce schema classes. The shared endpoint is the
# *object* of the relation, so the class is "things bearing R to it".
_SCHEMA_RELATIONS: frozenset[RelationType] = frozenset({
    RelationType.CAUSES,
    RelationType.ENABLES,
    RelationType.PREVENTS,
    RelationType.HARMS,
    RelationType.LEADS_TO,
    RelationType.EMERGES_FROM,
    RelationType.CREATES,
    RelationType.DEPENDS_ON,
    RelationType.HAS_PROPERTY,
    RelationType.CONTRADICTS,
    RelationType.OPPOSITE_OF,
    RelationType.GOAL_DIRECTED,
    RelationType.PART_OF,
})

# Namespaces eligible as members/endpoints beyond plain world concepts:
# the self-model categories and previously reified schemas. Everything
# else internal (code, utterances, task markers) is excluded.
_INTERNAL_ELIGIBLE_PREFIXES: tuple[str, ...] = ("_cat:", _SCHEMA_PREFIX)

# Hard cap on endpoint id length — long fragments produce unusable
# hub names and are usually misparses.
_MAX_ENDPOINT_LEN = 48

# Hub edge weights: the schema-level claim is as strong as the mean
# member evidence plus a small bonus per member beyond the minimum.
_MEMBER_WEIGHT_CAP = 0.9
_SCHEMA_WEIGHT_CAP = 0.9
_SUPPORT_BONUS = 0.02

# Mean member weight below which a bucket is noise, not structure.
# Compressing weak-evidence clusters crystallizes misparses instead
# of abstracting earned knowledge.
_MIN_MEAN_WEIGHT = 0.35


def _eligible_node(concept_id: str) -> bool:
    """Whether a concept may participate in a schema as member or endpoint."""
    if concept_id.startswith(_INTERNAL_ELIGIBLE_PREFIXES):
        return True
    return is_world_concept(concept_id)


@dataclass(slots=True)
class ReifiedSchema:
    """One reified schema hub and its membership."""

    hub_id: str  # the new (or refreshed) schema concept
    relation: RelationType  # the shared relation
    direction: str  # "to" (shared target) or "from" (shared source)
    endpoint: str  # the shared endpoint concept
    members: list[str]  # member concepts folded under the hub
    member_weights: dict[str, float]  # member -> its shared-edge weight
    weight: float  # schema-edge weight (mean member evidence + support)
    new: bool  # True if the hub concept was created this pass
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def describe(self) -> str:
        """Human-readable summary for logs and the journal."""
        rel = self.relation.value.replace("_", " ")
        if self.direction == "to":
            shared = f"members share '{rel} -> {self.endpoint}'"
        else:
            shared = f"members are '{rel}' targets of {self.endpoint}"
        preview = ", ".join(self.members[:4])
        if len(self.members) > 4:
            preview += f", … ({len(self.members)} members)"
        return f"schema {self.hub_id}: {shared}: {preview}"


class SchemaInducer:
    """Induces schema concepts from repeated relational structure.

    Stateless across calls — reads the network's live edge fold, writes
    schema hubs, member IS_A edges, and schema-level edges. Pass history
    is retained for introspection.

    Usage::

        inducer = SchemaInducer(network)
        schemas = inducer.discover_schemas()
        written = inducer.discover_and_accept()
    """

    def __init__(
        self,
        network: ConceptNetwork,
        min_members: int = 3,
        max_members_per_hub: int = 40,
        max_hubs_per_pass: int = 25,
        min_mean_weight: float = _MIN_MEAN_WEIGHT,
    ) -> None:
        self.network = network
        self.min_members = min_members
        self.max_members_per_hub = max_members_per_hub
        self.max_hubs_per_pass = max_hubs_per_pass
        self.min_mean_weight = min_mean_weight
        self._schemas: list[ReifiedSchema] = []

    @property
    def schemas(self) -> list[ReifiedSchema]:
        """All schemas reified in recorded history (bounded, copy)."""
        return list(self._schemas)

    @property
    def schema_count(self) -> int:
        """Number of schemas reified in recorded history."""
        return len(self._schemas)

    # ─── Discovery ───────────────────────────────────────────────

    def discover_schemas(self) -> list[ReifiedSchema]:
        """Scan the edge fold for repeated relational structure.

        Returns candidate schemas ranked by support (member count ×
        mean member weight). Does not modify the network — call
        ``accept_schemas`` or ``discover_and_accept`` to write.
        """
        to_buckets: dict[tuple[RelationType, str], dict[str, float]] = {}
        from_buckets: dict[tuple[RelationType, str], dict[str, float]] = {}
        live = self.network._concepts

        for edge in self.network._edges:
            rel = edge.relation
            if rel not in _SCHEMA_RELATIONS:
                continue
            src, tgt = edge.source, edge.target
            if src == tgt:
                continue
            # Working-memory only: never pull archived concepts back as
            # a side effect of a consolidation pass (same discipline as
            # EdgeProposer).
            if src not in live or tgt not in live:
                continue
            if not _eligible_node(src) or not _eligible_node(tgt):
                continue
            to_buckets.setdefault((rel, tgt), {})[src] = edge.weight
            from_buckets.setdefault((rel, src), {})[tgt] = edge.weight

        candidates: list[ReifiedSchema] = []
        for (rel, endpoint), members in to_buckets.items():
            if len(members) < self.min_members or len(endpoint) > _MAX_ENDPOINT_LEN:
                continue
            hub = f"{_SCHEMA_PREFIX}{rel.value}:to:{endpoint}"
            prop = self._proposal(hub, rel, "to", endpoint, members)
            if prop is not None:
                candidates.append(prop)
        for (rel, endpoint), members in from_buckets.items():
            if len(members) < self.min_members or len(endpoint) > _MAX_ENDPOINT_LEN:
                continue
            hub = f"{_SCHEMA_PREFIX}{rel.value}:from:{endpoint}"
            prop = self._proposal(hub, rel, "from", endpoint, members)
            if prop is not None:
                candidates.append(prop)

        # Rank by support; deterministic tiebreak on hub id.
        candidates.sort(key=lambda s: (-len(s.members) * s.weight, s.hub_id))
        return candidates[: self.max_hubs_per_pass]

    def _proposal(
        self,
        hub: str,
        rel: RelationType,
        direction: str,
        endpoint: str,
        members: dict[str, float],
    ) -> ReifiedSchema | None:
        ranked = sorted(members.items(), key=lambda kv: -kv[1])
        kept = ranked[: self.max_members_per_hub]
        mean_w = sum(w for _, w in kept) / len(kept)
        if mean_w < self.min_mean_weight:
            return None
        weight = min(
            _SCHEMA_WEIGHT_CAP,
            mean_w + _SUPPORT_BONUS * (len(kept) - self.min_members),
        )
        return ReifiedSchema(
            hub_id=hub,
            relation=rel,
            direction=direction,
            endpoint=endpoint,
            members=[m for m, _ in kept],
            member_weights=dict(kept),
            weight=max(0.0, weight),
            new=hub not in self.network._concepts,
        )

    # ─── Acceptance ──────────────────────────────────────────────

    def accept_schemas(self, schemas: list[ReifiedSchema] | None) -> int:
        """Write reified schemas into the concept network.

        Returns the number of *new* edges written (refreshed weights
        on existing reified edges do not count).
        """
        if schemas is None:
            schemas = self.discover_schemas()

        added = 0
        for schema in schemas:
            # Create the hub concept (no-op merge if it already exists).
            if schema.new:
                self.network.add_concept(
                    schema.hub_id,
                    confidence=min(1.0, schema.weight),
                    origin=_ORIGIN_REIFIED,
                    properties={
                        "definition": schema.describe(),
                        "reified": True,
                        "member_count": len(schema.members),
                    },
                )
            if schema.hub_id not in self.network._concepts:
                continue

            # The compressed claim: the schema itself bears the shared
            # relation to/from the endpoint.
            if schema.direction == "to":
                added += self._upsert(
                    schema.hub_id, schema.endpoint, schema.relation, schema.weight
                )
            else:
                added += self._upsert(
                    schema.endpoint, schema.hub_id, schema.relation, schema.weight
                )

            # Membership edges. Skip members whose addition would close
            # an IS_A cycle (hub already a descendant of the member).
            ancestors = self._isa_ancestors(schema.hub_id)
            for member in schema.members:
                if member == schema.hub_id or member in ancestors:
                    continue
                w = min(_MEMBER_WEIGHT_CAP, schema.member_weights.get(member, 0.4))
                added += self._upsert(
                    member, schema.hub_id, RelationType.IS_A, w
                )

            self._schemas.append(schema)

        if self._schemas:
            logger.info(
                f"Schema induction: {len(schemas)} hubs, {added} new edges "
                f"({self.schema_count} schemas in history)"
            )
        if len(self._schemas) > 500:
            del self._schemas[: len(self._schemas) - 500]
        return added

    def discover_and_accept(self) -> int:
        """One-shot: discover schemas and write them into the network.

        Convenience method for the sleep consolidation phase.
        """
        return self.accept_schemas(self.discover_schemas())

    # ─── Internals ───────────────────────────────────────────────

    def _upsert(
        self, src: str, tgt: str, rel: RelationType, weight: float
    ) -> int:
        """Write an edge if new, else refresh a reified edge's weight.

        Returns 1 when a new edge was written, 0 otherwise.
        """
        index = self.network._edge_key_index
        existing = index.get((src, tgt, rel))
        if existing is None:
            return 1 if self.network.add_edge(
                src, tgt, rel, weight, origin=_ORIGIN_REIFIED
            ) is not None else 0
        # Refresh only edges this pass owns; leave earned edges alone.
        if existing.origin == _ORIGIN_REIFIED and weight > existing.weight:
            self.network.set_edge_weight(src, tgt, rel, weight)
        return 0

    def _isa_ancestors(self, concept_id: str) -> set[str]:
        """All transitive IS_A/INSTANCE_OF parents of *concept_id*."""
        seen: set[str] = set()
        stack = [concept_id]
        while stack:
            cid = stack.pop()
            for edge in self.network._edge_index.get(cid, ()):
                if edge.relation not in (RelationType.IS_A, RelationType.INSTANCE_OF):
                    continue
                if edge.target not in seen:
                    seen.add(edge.target)
                    stack.append(edge.target)
        return seen
