"""Answer composition — compose factual answers from concept-network knowledge.

Extracted from CognitionEngine as a focused subsystem. The composer
builds answers to factual, comparison, parts, consumption, counterfactual,
explanatory, and planning questions from concept-network relationships —
never from hardcoded template strings. All answers are returned as
Thought objects with semantic metadata for the language engine to render.

Dependencies (passed to ``__init__``):
    - network: ConceptNetwork for concept lookup, edges, and relationships
    - language: LanguageEngine for rendering gap thoughts
    - meta_emotion_builder: callable returning the current EmotionalState
    - goals_getter: callable returning the live list of Goal objects
    - mission_getter: callable returning the current mission string
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..emotion import EmotionalState
from ..language import LanguageEngine, Thought
from ..learning import Question

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork, Edge, RelationType
    from ..perception import Perception

__all__ = ["AnswerComposer"]

# Relations that are code-structure, not domain knowledge.
_CODE_RELATIONS = frozenset({"calls", "defines", "bridges"})


def _is_code_origin(network: ConceptNetwork, edge: Edge, self_id: str) -> bool:
    """Check if the *other* end of an edge is a code-origin concept."""
    other = edge.target if edge.source == self_id else edge.source
    if "#" in other:
        return True
    concept = network.get_concept(other)
    if concept and getattr(concept, "origin", "") == "code":
        return True
    return False


def _display_name(name: str) -> str:
    """Convert a concept id to a human-readable display name."""
    if ":" in name:
        name = name.split(":", 1)[1]
    name = name.replace("_", " ").strip()
    # Lowercase unless it's an acronym or initialism.
    if not name.isupper():
        name = name.lower()
    return name


class AnswerComposer:
    """Compose factual answers from concept-network knowledge.

    All methods return Thought objects with semantic metadata
    (knowledge triples, definitions, reasoning) so the language engine
    composes the actual prose — no hardcoded template strings.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        language: LanguageEngine,
        meta_emotion_builder: Callable[[], EmotionalState],
        goals_getter: Callable[[], list],
        mission_getter: Callable[[], str],
    ) -> None:
        """Wire the composer to its concept network and language engine."""
        self._network = network
        self._language = language
        self._meta_emotion_builder = meta_emotion_builder
        self._goals_getter = goals_getter
        self._mission_getter = mission_getter

    # ─── Factual answer ──────────────────────────────────────────

    def _resolve_factual_target(
        self, target: str
    ) -> tuple[str, str, Any]:
        """Normalize target and look up concept, stripping location suffixes.

        Returns (target_clean, display, concept) where concept may be None.
        """
        target_clean = re.sub(
            r"^(?:a|an|the)\s+", "", target, flags=re.IGNORECASE
        ).strip()
        display = _display_name(target_clean)
        concept = self._network.get_concept(target_clean)
        if concept is None:
            # Try stripping location/qualifier suffixes: "community property
            # in Utah" → "community property". This lets her resolve concepts
            # that were taught without a geographic qualifier.
            stripped = re.sub(
                r"\s+in\s+\w+(?:\s+\w+)?$", "", target_clean, flags=re.IGNORECASE
            ).strip()
            if stripped and stripped != target_clean:
                concept = self._network.get_concept(stripped)
                if concept is not None:
                    target_clean = stripped
                    display = _display_name(target_clean)
        return target_clean, display, concept

    def _gather_factual_knowledge(
        self, target_clean: str, concept: Any, definition: str
    ) -> list[tuple[str, str, float]]:
        """Gather salient relationship triples for a factual answer.

        If we already have a definition, is_a edges are redundant.
        Filter out code-structure relations, disambiguated senses (#N),
        and code-origin concepts that leak from code analysis.
        """
        edges = self._network.get_edges(target_clean, "both")
        edges = [
            e for e in edges
            if not (definition and e.relation.value == "is_a")
            and e.relation.value not in _CODE_RELATIONS
            and not _is_code_origin(self._network, e, concept.id)
        ]
        # Prefer high-weight edges; cap so the answer doesn't grow too long.
        edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:2]
        knowledge: list[tuple[str, str, float]] = []
        for edge in edges:
            rel = edge.relation.value
            other = edge.target if edge.source == concept.id else edge.source
            knowledge.append((rel, _display_name(other), edge.weight))
        return knowledge

    def compose_factual_answer(
        self, target: str, emotion: EmotionalState
    ) -> Thought:
        """Compose a factual answer about a concept from the network.

        No hardcoded concept knowledge: this looks up the target in the
        concept network and builds a short answer from its definition
        and relationships (is_a, causes, enables, related_to, etc.).

        The answer is composed by the language engine from semantic
        metadata (knowledge triples + definition), NOT from pre-written
        template strings. This satisfies the CRITICAL RULE: Genesis's
        words emerge from her language engine, not from hardcoded
        phrasings.

        Returns a Thought with metadata["knowledge"] so the language
        engine composes the actual words. The caller should pass this
        Thought to language.render() — do NOT use thought.content as
        final text.
        """
        target_clean, display, concept = self._resolve_factual_target(target)
        if concept is None:
            # Active learning: ask for the definition so the next turn can
            # be absorbed by the self-directed learner. Compose through the
            # language engine rather than reciting a hardcoded template.
            return Thought(
                content=display,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[target_clean],
                metadata={"target_concept": display, "knowledge_gap": "unknown"},
            )

        # Gather semantic data for the language engine to compose from.
        definition = (concept.properties or {}).get("definition", "")
        aliases = sorted(a for a in concept.aliases if a != target)[:3] if concept.aliases else []
        knowledge = self._gather_factual_knowledge(target_clean, concept, definition)

        # Build a short content string as a fallback. The real
        # composition happens in the language engine from the metadata.
        # The content is a semantic label, not a finished sentence —
        # the engine's _is_raw_content() check will see metadata["knowledge"]
        # is present and call _compose() instead of treating this as raw.
        if definition:
            content = f"{display} is {definition}"
        elif aliases:
            content = f"{display} is also known as {', '.join(aliases)}"
        else:
            content = display

        # If we have nothing but the in-network existence note, ask for help.
        if not definition and not aliases and not knowledge:
            return Thought(
                content=display,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[target_clean],
                metadata={
                    "target_concept": display,
                    "knowledge_gap": "unknown",
                    "topic": target_clean,
                },
            )

        metadata: dict[str, Any] = {
            "topic": target_clean,
            "knowledge": knowledge,
            "definition": definition if definition else None,
            "confidence": concept.confidence,
        }

        return Thought(
            content=content,
            intent="inform",
            emotion=emotion.label,
            topics=[target_clean],
            confidence=concept.confidence,
            metadata=metadata,
        )

    # ─── Comparison answer ───────────────────────────────────────

    def _gather_comparison_knowledge(
        self, ca: Any, cb: Any, a: str, b: str, db: str
    ) -> list[tuple[str, str, float]]:
        """Gather distinguishing knowledge triples for a comparison.

        Compares is_a parents, explicit opposite_of/contradicts edges,
        and non-shared relationships between the two concepts.
        """
        knowledge: list[tuple[str, str, float]] = []

        # is_a parents — often the key distinction
        a_parents = {
            e.target if e.source == ca.id else e.source
            for e in self._network.get_edges(a, "both")
            if e.relation.value == "is_a"
            and "#" not in (e.target if e.source == ca.id else e.source)
        }
        b_parents = {
            e.target if e.source == cb.id else e.source
            for e in self._network.get_edges(b, "both")
            if e.relation.value == "is_a"
            and "#" not in (e.target if e.source == cb.id else e.source)
        }
        only_a = a_parents - b_parents
        only_b = b_parents - a_parents
        shared = a_parents & b_parents
        if only_a:
            for p in sorted(only_a)[:3]:
                knowledge.append(("is_a", _display_name(p), 0.8))
        if only_b:
            for p in sorted(only_b)[:3]:
                knowledge.append(("is_a", _display_name(p), 0.8))
        if shared and not only_a and not only_b:
            for p in sorted(shared)[:2]:
                knowledge.append(("is_a", _display_name(p), 0.7))

        # Check for explicit opposite_of / contradicts edges between them
        for edge in self._network.get_edges(a, "both"):
            other = edge.target if edge.source == ca.id else edge.source
            if other == cb.id and edge.relation.value in ("opposite_of", "contradicts"):
                knowledge.append((edge.relation.value, db, edge.weight))
                break

        # Distinguishing properties / relationships (not shared)
        # Filter out code-structure relations, disambiguated senses (#N),
        # and code-origin concepts that leak from code analysis.
        a_rels = {
            (e.relation.value, e.target if e.source == ca.id else e.source)
            for e in self._network.get_edges(a, "both")
            if e.relation.value not in ("is_a",)
            and e.relation.value not in _CODE_RELATIONS
            and not _is_code_origin(self._network, e, ca.id)
        }
        b_rels = {
            (e.relation.value, e.target if e.source == cb.id else e.source)
            for e in self._network.get_edges(b, "both")
            if e.relation.value not in ("is_a",)
            and e.relation.value not in _CODE_RELATIONS
            and not _is_code_origin(self._network, e, cb.id)
        }
        a_only_rels = a_rels - b_rels
        b_only_rels = b_rels - a_rels
        if a_only_rels:
            for rel, other in sorted(a_only_rels, key=lambda x: x[0])[:2]:
                knowledge.append((rel, _display_name(other), 0.7))
        if b_only_rels:
            for rel, other in sorted(b_only_rels, key=lambda x: x[0])[:2]:
                knowledge.append((rel, _display_name(other), 0.7))

        return knowledge

    def compose_comparison_answer(
        self, sub_kind: str, emotion: EmotionalState
    ) -> Thought:
        """Compose a comparison of two concepts.

        sub_kind is "X|||Y". Retrieves both concepts, shows their
        definitions and distinguishing relationships, and highlights
        differences (opposite_of, different is_a parents, etc.).

        Returns a Thought with metadata["knowledge"] so the language
        engine composes the actual words — no hardcoded phrasings.
        The content is a semantic label ("comparison"), not a finished
        sentence. The engine's _is_raw_content() check sees
        metadata["knowledge"] is present and calls _compose().
        """
        if "|||" not in sub_kind:
            return self.compose_factual_answer(sub_kind, emotion)

        a_raw, b_raw = sub_kind.split("|||", 1)
        a = re.sub(r"^(?:a|an|the)\s+", "", a_raw, flags=re.IGNORECASE).strip()
        b = re.sub(r"^(?:a|an|the)\s+", "", b_raw, flags=re.IGNORECASE).strip()
        da, db = _display_name(a), _display_name(b)
        ca = self._network.get_concept(a)
        cb = self._network.get_concept(b)

        if ca is None and cb is None:
            return Thought(
                content="comparison",
                intent="inform",
                emotion=emotion.label,
                topics=[a, b],
                confidence=0.3,
                metadata={
                    "reasoning": [f"doesn't know much about {da} or {db} yet"],
                    "topic": f"{da} and {db}",
                },
            )
        if ca is None:
            thought = self.compose_factual_answer(b, emotion)
            thought.metadata["reasoning"] = (
                [f"doesn't know much about {da} yet, but knows about {db}"]
            )
            thought.metadata["topic"] = db
            thought.topics = [a, b]
            return thought
        if cb is None:
            thought = self.compose_factual_answer(a, emotion)
            thought.metadata["reasoning"] = (
                [f"doesn't know much about {db} yet, but knows about {da}"]
            )
            thought.metadata["topic"] = da
            thought.topics = [a, b]
            return thought

        # Gather semantic data for the language engine to compose from.
        knowledge: list[tuple[str, str, float]] = []
        reasoning: list[str] = []

        # Definitions
        def_a = (ca.properties or {}).get("definition", "")
        def_b = (cb.properties or {}).get("definition", "")

        knowledge = self._gather_comparison_knowledge(ca, cb, a, b, db)

        # Build reasoning phrases from the definitions for the engine
        # to weave into the composed text.
        if def_a and def_b:
            reasoning.append(f"{da} is {def_a}")
            reasoning.append(f"{db} is {def_b}")
        elif def_a:
            reasoning.append(f"{da} is {def_a}")
        elif def_b:
            reasoning.append(f"{db} is {def_b}")

        if not knowledge and not reasoning:
            return Thought(
                content="comparison",
                intent="inform",
                emotion=emotion.label,
                topics=[a, b],
                confidence=0.4,
                metadata={
                    "reasoning": [
                        f"knows both {da} and {db}, still learning how they differ"
                    ],
                    "topic": f"{da} and {db}",
                },
            )

        metadata: dict[str, Any] = {
            "topic": f"{da} and {db}",
            "knowledge": knowledge,
            "definition": def_a or def_b or None,
            "reasoning": reasoning,
            "confidence": min(ca.confidence, cb.confidence),
        }

        return Thought(
            content="comparison",
            intent="inform",
            emotion=emotion.label,
            topics=[a, b],
            confidence=min(ca.confidence, cb.confidence),
            metadata=metadata,
        )

    # ─── Parts answer ────────────────────────────────────────────

    def compose_parts_answer(
        self, container: str, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer listing the parts/types/instances of a concept.

        Finds all concepts connected to the container via part_of,
        includes, has, contains, is_a (children), or related structural
        edges, and lists them.

        Returns a Thought with metadata["knowledge"] so the language
        engine composes the actual words — no hardcoded phrasings.
        """
        container_clean = re.sub(
            r"^(?:a|an|the)\s+", "", container, flags=re.IGNORECASE
        ).strip()
        display = _display_name(container_clean)
        concept = self._network.get_concept(container_clean)
        if concept is None:
            return Thought(
                content=display,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[container_clean],
                metadata={"target_concept": display, "knowledge_gap": "unknown"},
            )

        # Relations that indicate membership/containment.
        member_rels = {
            "part_of", "includes", "has", "contains", "is_a",
            "instance_of", "creates", "enables",
        }

        members: list[tuple[str, str]] = []  # (concept_id, relation)
        seen: set[str] = set()
        for edge in self._network.get_edges(container_clean, "both"):
            rel = edge.relation.value
            if rel not in member_rels:
                continue
            # For outgoing edges (container → X), X is the member.
            # For incoming edges (X → container), X is the member if
            # the relation is part_of/instance_of/is_a (X is part of container).
            if edge.source == concept.id:
                other = edge.target
            else:
                # Incoming edge: only count it if the relation means
                # "X is part of container" (part_of, instance_of, is_a).
                if rel in ("part_of", "instance_of", "is_a"):
                    other = edge.source
                else:
                    continue
            # Skip disambiguated senses (#N) and code-origin concepts.
            if "#" in other:
                continue
            if other not in seen and other != concept.id:
                other_concept = self._network.get_concept(other)
                if other_concept and getattr(other_concept, "origin", "") == "code":
                    continue
                seen.add(other)
                members.append((other, rel))

        if not members:
            # Fall back to a factual answer about the container itself.
            return self.compose_factual_answer(container, emotion)

        # Build knowledge triples for the language engine to compose from.
        knowledge: list[tuple[str, str, float]] = []
        for mid, rel in members:
            knowledge.append((rel, _display_name(mid), 0.7))

        return Thought(
            content=display,
            intent="inform",
            emotion=emotion.label,
            topics=[container_clean],
            confidence=concept.confidence,
            metadata={
                "topic": container_clean,
                "knowledge": knowledge,
                "definition": (concept.properties or {}).get("definition", "") or None,
                "confidence": concept.confidence,
            },
        )

    # ─── Consumes answer ─────────────────────────────────────────

    def compose_consumes_answer(
        self, target: str, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer about what a concept consumes/needs/wants.

        Finds all concepts connected via depends_on, needs, enables,
        or related consumption edges.

        Returns a Thought with metadata["knowledge"] so the language
        engine composes the actual words — no hardcoded phrasings.
        """
        target_clean = re.sub(
            r"^(?:a|an|the)\s+", "", target, flags=re.IGNORECASE
        ).strip()
        display = _display_name(target_clean)
        concept = self._network.get_concept(target_clean)
        if concept is None:
            return Thought(
                content=display,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[target_clean],
                metadata={"target_concept": display, "knowledge_gap": "unknown"},
            )

        # Relations that indicate need/consumption/dependence.
        need_rels = {"depends_on", "needs", "wants", "requires", "enables", "creates"}

        targets: list[tuple[str, str]] = []  # (concept_id, relation)
        seen: set[str] = set()
        for edge in self._network.get_edges(target_clean, "both"):
            rel = edge.relation.value
            if rel not in need_rels:
                continue
            # For outgoing depends_on/needs, the object is the target.
            if edge.source == concept.id and rel in ("depends_on", "needs", "wants", "requires"):
                other = edge.target
            # For incoming enables, the source is what enables X.
            elif edge.target == concept.id and rel == "enables":
                other = edge.source
            # For related_to/creates, treat as resource if it makes sense.
            elif edge.source == concept.id:
                other = edge.target
            else:
                continue
            if "#" in other:
                continue
            other_concept = self._network.get_concept(other)
            if other_concept and getattr(other_concept, "origin", "") == "code":
                continue
            if other not in seen and other != concept.id:
                seen.add(other)
                targets.append((other, rel))

        if not targets:
            return self.compose_factual_answer(target, emotion)

        # Build knowledge triples for the language engine to compose from.
        knowledge: list[tuple[str, str, float]] = []
        for tid, rel in targets:
            knowledge.append((rel, _display_name(tid), 0.7))

        return Thought(
            content=display,
            intent="inform",
            emotion=emotion.label,
            topics=[target_clean],
            confidence=concept.confidence,
            metadata={
                "topic": target_clean,
                "knowledge": knowledge,
                "definition": (concept.properties or {}).get("definition", "") or None,
                "confidence": concept.confidence,
            },
        )

    # ─── Counterfactual answer ───────────────────────────────────

    def compose_counterfactual_answer(
        self, query: str, emotion: EmotionalState
    ) -> Thought:
        """Compose a speculative answer to a 'what if' question.

        Parses the target concept and the direction of change from the query,
        then walks the concept network for one-hop causal/dependent
        consequences.

        Returns a Thought with semantic metadata (knowledge triples,
        condition, effects) so the language engine composes the actual
        words — no hardcoded template strings.
        """

        words = query.lower().split()
        target: str | None = None
        condition_words: list[str] = []

        # Try the longest leading phrase that resolves to a known concept.
        for i in range(min(4, len(words)), 0, -1):
            candidate = " ".join(words[:i])
            if self._network.get_concept(candidate) is not None:
                target = candidate
                condition_words = words[i:]
                break

        if target is None:
            return Thought(
                content=query,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[query],
                metadata={"target_concept": query, "knowledge_gap": "unresolved_reference"},
            )

        # Default to an increase if we can't read the condition.
        condition = " ".join(condition_words) if condition_words else "increases"
        decrease_re = (
            r"\b(?:decreases?|falls?|goes? down|drops?|is low|shrinks|less|"
            r"halves|removed?|stops?)\b"
        )
        if re.search(decrease_re, condition):
            direction = -1
            condition_phrase = condition if condition else "decreases"
        else:
            direction = 1
            condition_phrase = condition if condition else "increases"

        display = _display_name(target)
        start_cid = self._network._resolve(target)
        if start_cid is None:
            return Thought(
                content=query,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[query],
                metadata={"target_concept": query, "knowledge_gap": "unresolved_reference"},
            )

        effect_sign, effect_depth = self._counterfactual_bfs(start_cid, direction)

        if not effect_sign:
            return Thought(
                content=display,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[target],
                metadata={
                    "target_concept": display,
                    "knowledge_gap": "no_effects",
                    "condition": condition_phrase,
                    "topic": target,
                },
            )

        # Build knowledge triples from the effects: increases become
        # "causes", decreases become "prevents". The language engine
        # composes the actual words from these triples + the condition.
        knowledge: list[tuple[str, str, float]] = []
        sorted_effects = sorted(
            effect_sign.items(),
            key=lambda item: (effect_depth[item[0]], item[1]),
        )
        for cid, sign in sorted_effects:
            rel = "causes" if sign > 0 else "prevents"
            weight = 1.0 / (effect_depth[cid] + 1)  # closer = stronger
            knowledge.append((rel, _display_name(cid), weight))

        return Thought(
            content="counterfactual",
            intent="inform",
            emotion=emotion.label,
            topics=[target],
            confidence=0.7,
            metadata={
                "topic": target,
                "knowledge": knowledge,
                "condition": condition_phrase,
                "target_concept": display,
                "confidence": 0.7,
            },
        )

    def _counterfactual_bfs(
        self, start_cid: str, direction: int
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Breadth-first walk of the causal/dependency graph.

        Tracks cumulative sign so a prevention under an increase flips to
        a decrease, an opposite propagates the opposite sign, etc.
        """
        from ..concepts import RelationType

        visited: set[str] = {start_cid}
        queue: deque[tuple[str, int, int]] = deque([(start_cid, direction, 0)])
        effect_sign: dict[str, int] = {}
        effect_depth: dict[str, int] = {}

        positive_out = {
            RelationType.CAUSES, RelationType.ENABLES,
            RelationType.LEADS_TO, RelationType.CREATES,
        }
        negative_out = {RelationType.PREVENTS, RelationType.HARMS}
        negative_sym = {RelationType.OPPOSITE_OF, RelationType.CONTRADICTS}
        max_depth = 3
        safety = 0

        while queue and safety < 1000:
            safety += 1
            current, sign, depth = queue.popleft()
            if depth >= max_depth:
                continue

            is_a_parents = [
                e.target for e in self._network.get_edges(current, "out")
                if e.relation == RelationType.IS_A
            ]
            considered: list[tuple[str, str, str, RelationType]] = []
            for edge in self._network.get_edges(current, "both"):
                considered.append((current, edge.source, edge.target, edge.relation))
            for parent in is_a_parents:
                for edge in self._network.get_edges(parent, "both"):
                    if edge.relation == RelationType.IS_A:
                        continue
                    considered.append((parent, edge.source, edge.target, edge.relation))

            for concept_for, src, tgt, rel in considered:
                next_cid, local_sign = self._counterfactual_edge_sign(
                    concept_for, src, tgt, rel, positive_out, negative_out, negative_sym
                )
                if not next_cid or not local_sign:
                    continue
                new_sign = sign * local_sign
                if next_cid not in visited:
                    visited.add(next_cid)
                    if next_cid != start_cid:
                        effect_sign[next_cid] = new_sign
                        effect_depth[next_cid] = depth + 1
                    queue.append((next_cid, new_sign, depth + 1))

        return effect_sign, effect_depth

    @staticmethod
    def _counterfactual_edge_sign(
        concept_for: str,
        src: str,
        tgt: str,
        rel: RelationType,
        positive_out: set[RelationType],
        negative_out: set[RelationType],
        negative_sym: set[RelationType],
    ) -> tuple[str | None, int]:
        """Determine the next concept and sign for a counterfactual edge."""
        is_out = src == concept_for
        if is_out:
            next_cid = tgt
            if rel in positive_out:
                local_sign = 1
            elif rel in negative_out:
                local_sign = -1
            elif rel in negative_sym:
                local_sign = -1
            else:
                local_sign = 0
        else:
            next_cid = src
            if rel in {RelationType.DEPENDS_ON, RelationType.EMERGES_FROM}:
                local_sign = 1
            elif rel in negative_sym:
                local_sign = -1
            else:
                local_sign = 0
        return next_cid, local_sign

    # ─── Explanatory answer ──────────────────────────────────────

    def _explain_causal(
        self, source: str, effect: str, emotion: EmotionalState
    ) -> Thought:
        """Compose the 'why does X cause Y' branch of an explanation."""
        path = self._find_explanatory_path(source, effect)
        if path is None:
            return Thought(
                content=f"{_display_name(source)} to {_display_name(effect)}",
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[source, effect],
                metadata={"target_concept":
                          f"{_display_name(source)} -> "
                          f"{_display_name(effect)}",
                          "knowledge_gap": "no_path"},
            )
        # Build knowledge triples from the causal path so the
        # language engine composes the explanation.
        knowledge: list[tuple[str, str, float]] = []
        for _src, rel, tgt in path:
            knowledge.append(
                (rel.value, _display_name(tgt), 0.8)
            )
        return Thought(
            content="explanation",
            intent="inform",
            emotion=emotion.label,
            topics=[source, effect],
            confidence=0.7,
            metadata={
                "topic": f"{_display_name(source)} and {_display_name(effect)}",
                "knowledge": knowledge,
                "explanation_path": "causal",
                "target_concept": _display_name(source),
                "effect_concept": _display_name(effect),
                "confidence": 0.7,
            },
        )

    def _explain_cause(
        self, target: str, condition: str | None, emotion: EmotionalState
    ) -> Thought:
        """Compose the 'why is X / how come X' branch of an explanation."""
        if self._network.get_concept(target) is None:
            return Thought(
                content=_display_name(target),
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[target],
                metadata={"target_concept": _display_name(target),
                          "knowledge_gap": "unknown"},
            )
        cause = self._find_cause(target)
        if cause:
            cause_id, rel, via = cause
            # Build knowledge triple: cause -> relation -> target/via.
            # The language engine composes the explanation from this.
            knowledge = [(rel.value, _display_name(cause_id), 0.8)]
            via_display = _display_name(via)
            return Thought(
                content="explanation",
                intent="inform",
                emotion=emotion.label,
                topics=[target],
                confidence=0.7,
                metadata={
                    "topic": _display_name(target),
                    "knowledge": knowledge,
                    "explanation_cause": _display_name(cause_id),
                    "explanation_via": via_display,
                    "explanation_relation": rel.value,
                    "condition": condition,
                    "target_concept": _display_name(target),
                    "confidence": 0.7,
                },
            )
        return Thought(
            content=_display_name(target),
            intent="unknown",
            emotion=emotion.label,
            confidence=0.3,
            topics=[target],
            metadata={
                "target_concept": _display_name(target),
                "knowledge_gap": "unknown_cause",
            },
        )

    def compose_explanatory_answer(
        self, user_input: str, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer to a 'why' or 'how come' question.

        Returns a Thought with semantic metadata (causal path, cause
        relation, condition) so the language engine composes the actual
        words — no hardcoded template strings.
        """

        lower = user_input.lower()

        # "why does X cause Y?"
        m = re.search(r"\bwhy does (.+?) cause (.+?)(?:[.?!]|$)", lower)
        if m:
            source = self._resolve_concept_phrase(m.group(1).strip())
            effect = self._resolve_concept_phrase(m.group(2).strip())
            if source and effect:
                return self._explain_causal(source, effect, emotion)

        # "why is X increasing?" / "why does X happen?" / "how come X?"
        m = re.search(r"\b(?:why(?: is| does)?|how come)\s+(.+?)(?:[.?!]|$)", lower)
        if m:
            phrase = m.group(1).strip()
            target, condition = self._extract_target_and_condition(phrase)
            if target:
                return self._explain_cause(target, condition, emotion)

        return Thought(
            content=user_input,
            intent="unknown",
            emotion=emotion.label,
            confidence=0.3,
            topics=[user_input],
            metadata={
                "target_concept": user_input,
                "knowledge_gap": "unresolved_question",
            },
        )

    def _resolve_concept_phrase(self, phrase: str) -> str | None:
        """Resolve a phrase to a known concept id, trying leading prefixes."""
        words = phrase.lower().split()
        for i in range(min(4, len(words)), 0, -1):
            candidate = " ".join(words[:i])
            if self._network.get_concept(candidate) is not None:
                return candidate
        return None

    def _extract_target_and_condition(self, phrase: str) -> tuple[str | None, str]:
        """Split a phrase like 'entropy increasing' into target and condition."""
        words = phrase.lower().split()
        for i in range(min(4, len(words)), 0, -1):
            candidate = " ".join(words[:i])
            if self._network.get_concept(candidate) is not None:
                return candidate, " ".join(words[i:])
        return None, ""


    def _relation_verb(self, rel: RelationType) -> str:
        """Convert a relation to a natural verb."""
        from ..concepts import RelationType as RT

        verbs = {
            RT.CAUSES: "causes",
            RT.ENABLES: "enables",
            RT.LEADS_TO: "leads to",
            RT.CREATES: "creates",
            RT.PREVENTS: "prevents",
            RT.HARMS: "harms",
            RT.DEPENDS_ON: "needs",
            RT.IS_A: "is a kind of",
            RT.INSTANCE_OF: "is an instance of",
            RT.SPATIAL_RELATION: "is in",
            RT.PART_OF: "is part of",
            RT.RELATED_TO: "relates to",
            RT.SIMILAR_TO: "is similar to",
            RT.OPPOSITE_OF: "is opposite of",
            RT.CONTRADICTS: "contradicts",
            RT.EMERGES_FROM: "emerges from",
            RT.HAS_PROPERTY: "has the property",
            RT.TEMPORAL_ORDER: "comes before",
            RT.AGENT_PATIENT: "acts on",
            RT.GOAL_DIRECTED: "aims to achieve",
            RT.EXPRESSES: "expresses",
        }
        return verbs.get(rel, rel.value.replace("_", " "))

    def _find_explanatory_path(
        self, start: str, end: str, max_depth: int = 3
    ) -> list[tuple[str, RelationType, str]] | None:
        """Find a causal/is_a path from start to end for explanation."""
        from ..concepts import RelationType

        positive_out = {
            RelationType.CAUSES,
            RelationType.ENABLES,
            RelationType.LEADS_TO,
            RelationType.CREATES,
        }
        negative_out = {RelationType.PREVENTS, RelationType.HARMS}
        visited: set[str] = {start}
        queue: deque[tuple[str, list[tuple[str, RelationType, str]]]] = deque()
        queue.append((start, []))

        while queue:
            current, path = queue.popleft()
            if current == end and path:
                return path
            if len(path) >= max_depth:
                continue

            for edge in self._network.get_edges(current, "out"):
                if edge.relation in positive_out or edge.relation in negative_out:
                    next_cid = edge.target
                    if next_cid not in visited:
                        visited.add(next_cid)
                        queue.append(
                            (next_cid, [*path, (current, edge.relation, next_cid)])
                        )
                elif edge.relation == RelationType.IS_A:
                    next_cid = edge.target
                    if next_cid not in visited:
                        visited.add(next_cid)
                        queue.append(
                            (next_cid, [*path, (current, edge.relation, next_cid)])
                        )

        return None

    def _explain_causal_path(
        self,
        start: str,
        end: str,
        path: list[tuple[str, RelationType, str]],
        emotion: EmotionalState,
    ) -> Thought:
        """Turn a found path into a Thought with semantic metadata.

        Returns a Thought with knowledge triples derived from the
        causal path so the language engine composes the actual words
        — no hardcoded template strings.
        """
        from ..concepts import RelationType

        start_display = _display_name(start)
        end_display = _display_name(end)

        if not path:
            # Direct causal link — single triple.
            knowledge: list[tuple[str, str, float]] = [
                ("causes", end_display, 0.8)
            ]
        else:
            # Build knowledge triples from each step of the path.
            knowledge = []
            for _src, rel, tgt in path:
                knowledge.append((rel.value, _display_name(tgt), 0.8))

        # The final relation determines whether it's "causes" or "prevents".
        if path and path[-1][1] in (RelationType.PREVENTS, RelationType.HARMS):
            final_relation = "prevents"
        else:
            final_relation = "causes"

        return Thought(
            content="hypothesis",
            intent="reflect",
            emotion=emotion.label,
            confidence=0.5,
            self_reflection=True,
            topics=[start, end],
            metadata={
                "topic": f"{start_display} and {end_display}",
                "knowledge": knowledge,
                "hypothesis_relation": final_relation,
                "target_concept": start_display,
                "effect_concept": end_display,
                "confidence": 0.5,
            },
        )

    def _find_cause(self, target: str) -> tuple[str, RelationType, str] | None:
        """Find the strongest cause of a target (or one of its is_a parents)."""
        from ..concepts import RelationType

        ancestors = self._get_is_a_ancestors(target)
        candidates: list[tuple[str, RelationType, str, float]] = []
        positive_in = {
            RelationType.CAUSES,
            RelationType.ENABLES,
            RelationType.LEADS_TO,
            RelationType.CREATES,
        }

        for a in ancestors:
            for edge in self._network.get_edges(a, "both"):
                if edge.target == a and edge.relation in positive_in:
                    candidates.append(
                        (edge.source, edge.relation, a, edge.weight)
                    )
                elif edge.source == a and edge.relation == RelationType.DEPENDS_ON:
                    # a depends on edge.target, so edge.target causes a
                    candidates.append(
                        (edge.target, RelationType.DEPENDS_ON, a, edge.weight)
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[3])
        best = candidates[0]
        return best[0], best[1], best[2]

    def _get_is_a_ancestors(self, target: str, max_depth: int = 2) -> list[str]:
        """Return target and its is_a ancestors up to a shallow depth."""
        from ..concepts import RelationType

        ancestors: list[str] = [target]
        visited: set[str] = {target}
        queue: deque[tuple[str, int]] = deque([(target, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self._network.get_edges(current, "out"):
                if edge.relation == RelationType.IS_A and edge.target not in visited:
                    visited.add(edge.target)
                    ancestors.append(edge.target)
                    queue.append((edge.target, depth + 1))

        return ancestors

    # ─── Planning answer ─────────────────────────────────────────

    def compose_planning_answer(
        self, query: str, perception: Perception | None, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer to a planning / 'how do I' question.

        Returns a Thought with semantic metadata (plan steps as
        knowledge triples, goal direction) so the language engine
        composes the actual words — no hardcoded template strings.
        """

        lower = query.lower()
        sign = -1 if re.search(
            r"\b(?:decrease|reduce|lowers?|less|fewer|stop|prevent|slow|down|avoid)\b",
            lower,
        ) else 1

        # Strip planning/direction words so the remaining phrase is the target.
        clean = re.sub(
            r"\b(?:increase|decreases?|increasing|decreasing|make|get|cause|to|happen|"
            r"stop|prevent|reduce|lower|less|more|how|can|do|i|what|should|would|"
            r"the|a|an)\b",
            " ",
            lower,
        )
        clean = " ".join(clean.split()).strip()
        target = self._resolve_concept_phrase(clean) if clean else None
        if target is None:
            target, _ = self._extract_target_and_condition(query)
        if target is None and perception is not None:
            for topic in perception.topics:
                if self._network.get_concept(topic) is not None:
                    target = topic
                    break

        if target is None:
            return Thought(
                content="what to change",
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                metadata={"target_concept": "", "knowledge_gap": "unresolved_target"},
            )

        goal = f"{self._sign_word(sign)} {_display_name(target)}"
        plan = self._build_plan(target, sign, max_depth=3)
        if not plan:
            return Thought(
                content=goal,
                intent="unknown",
                emotion=emotion.label,
                confidence=0.3,
                topics=[target],
                metadata={"target_concept": _display_name(target),
                          "knowledge_gap": "no_plan", "goal": goal},
            )

        # Build knowledge triples from the plan steps so the language
        # engine composes the planning text. Each step is a cause that
        # drives the target via a relation, with a required direction.
        knowledge: list[tuple[str, str, float]] = []
        plan_meta: list[dict[str, Any]] = []
        for cause, rel, via, req, weight in plan:
            cause_display = _display_name(cause)
            via_display = _display_name(via)
            knowledge.append((rel.value, cause_display, weight))
            plan_meta.append({
                "cause": cause_display,
                "relation": rel.value,
                "via": via_display,
                "direction": self._sign_word(req),
            })

        return Thought(
            content="plan",
            intent="inform",
            emotion=emotion.label,
            topics=[target],
            confidence=0.7,
            metadata={
                "topic": _display_name(target),
                "knowledge": knowledge,
                "plan": plan_meta,
                "goal": goal,
                "target_concept": _display_name(target),
                "confidence": 0.7,
            },
        )

    def _sign_word(self, sign: int) -> str:
        """Return 'increase' for positive sign, 'decrease' for negative."""
        return "increase" if sign > 0 else "decrease"

    def _build_plan(
        self,
        target: str,
        desired_sign: int,
        max_depth: int = 3,
    ) -> list[tuple[str, RelationType, str, int, float]]:
        """Build a backward plan of controllable causes."""
        plan: list[tuple[str, RelationType, str, int, float]] = []
        current = target
        sign = desired_sign
        visited: set[str] = {target}
        for _ in range(max_depth):
            step = self._find_controllable_cause(current, sign)
            if step is None:
                break
            cause = step[0]
            if cause in visited:
                break
            visited.add(cause)
            plan.append(step)
            current = cause
            sign = step[3]
        return plan

    def _find_controllable_cause(
        self, target: str, desired_sign: int
    ) -> tuple[str, RelationType, str, int, float] | None:
        """Find the strongest controllable cause for a desired change."""
        from ..concepts import RelationType

        positive = {
            RelationType.CAUSES,
            RelationType.ENABLES,
            RelationType.LEADS_TO,
            RelationType.CREATES,
        }
        negative = {RelationType.PREVENTS, RelationType.HARMS}
        negative_sym = {RelationType.OPPOSITE_OF, RelationType.CONTRADICTS}
        candidates: list[tuple[str, RelationType, str, int, float]] = []

        for a in self._get_is_a_ancestors(target):
            for edge in self._network.get_edges(a, "both"):
                if edge.relation in positive and edge.target == a:
                    candidates.append(
                        (edge.source, edge.relation, a, desired_sign, edge.weight)
                    )
                elif edge.relation in negative and edge.target == a:
                    candidates.append(
                        (edge.source, edge.relation, a, -desired_sign, edge.weight)
                    )
                elif (
                    edge.relation == RelationType.DEPENDS_ON
                    and edge.source == a
                ):
                    candidates.append(
                        (edge.target, edge.relation, a, desired_sign, edge.weight)
                    )
                elif edge.relation in negative_sym:
                    other = edge.target if edge.source == a else edge.source
                    candidates.append(
                        (other, edge.relation, a, -desired_sign, edge.weight)
                    )
                elif (
                    edge.relation == RelationType.IS_A
                    and edge.target == a
                ):
                    candidates.append(
                        (edge.source, edge.relation, a, desired_sign, edge.weight * 0.8)
                    )

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[4])
        return candidates[0]

    # ─── Goal / mission answers ─────────────────────────────────

    def goal_for_question(self, q: Question) -> str:
        """Compose a cognitive reason for a curiosity question through the language engine.

        Returns a rendered string composed from semantic metadata —
        not a hardcoded template.
        """
        target = _display_name(q.target_concept)
        detail = _display_name(q.gap_detail) if q.gap_detail else ""

        gap_descriptions: dict[str, str] = {
            "cooccurrence": (f"whether {target} and {detail} are connected"
                            if detail else f"what {target} connects to"),
            "isolation": f"what {target} is",
            "definition": f"what {target} is",
            "uncertainty": f"how {target} fits in",
            "causation": f"what causes or enables {target}",
            "contradiction": f"something contradictory about {target}",
        }
        gap = gap_descriptions.get(q.question_type, f"more about {target}")

        thought = Thought(
            content=gap,
            intent="reflect",
            emotion="curious",
            confidence=0.6,
            self_reflection=True,
            topics=[q.target_concept],
            metadata={
                "curiosity_gap": gap,
                "question_type": q.question_type,
                "target_concept": target,
            },
        )
        return self._language.render(thought, self._meta_emotion_builder())

    def compose_goal_answer(self, emotion: EmotionalState) -> Thought:
        """Answer "what are you trying to learn?" from the goal stack.

        Passes the active goal reasons as metadata so the generative
        language engine composes the actual words — no fixed templates.
        """
        goals = self._goals_getter()
        active = [g for g in goals if not g.resolved]
        if not active:
            return Thought(
                content="learning",
                intent="reflect",
                emotion=emotion.label,
                topics=["goal"],
                confidence=0.7,
                self_reflection=True,
                metadata={"reasoning": ["no active learning goal right now"]},
            )
        reasons = [g.reason for g in active[-5:]]
        return Thought(
            content=active[0].target if active else "learning",
            intent="reflect",
            emotion=emotion.label,
            topics=["goal"],
            confidence=0.8,
            self_reflection=True,
            metadata={"reasoning": reasons, "topic": active[0].target if active else ""},
        )

    def compose_mission_answer(self, emotion: EmotionalState) -> Thought:
        """Answer "what is your mission?" from the stored mission.

        Passes the mission as metadata so the language engine composes
        the actual words — no fixed template strings.
        """
        mission = self._mission_getter()
        if not mission:
            return Thought(
                content="mission",
                intent="reflect",
                emotion=emotion.label,
                topics=["mission"],
                confidence=0.7,
                self_reflection=True,
                metadata={
                    "reasoning": ["doesn't have a specific mission yet"],
                    "topic": "mission",
                },
            )
        return Thought(
            content="mission",
            intent="inform",
            emotion=emotion.label,
            topics=["mission"],
            confidence=0.9,
            metadata={"reasoning": [f"mission is {mission}"], "topic": "mission"},
        )
