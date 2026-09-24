"""Graph-walk generator — sentences that emerge from concept traversal.

This is the primary composition path for Genesis's language. Instead of
selecting a fixed sentence template and filling slots, the graph-walk
generator traverses the concept network from a seed concept, following
typed semantic edges (IS_A, CAUSES, ENABLES, EMERGES_FROM, etc.), and
builds sentences from the edges it encounters.

The sentence structure emerges from the graph topology:
- A concept with a CAUSES edge produces a causal sentence
- A concept with an ENABLES edge produces a facilitative sentence
- A concept with multiple edges produces a compound sentence
- A deep walk (2-3 hops) produces multi-sentence paragraphs

This is fundamentally different from template filling:
- Templates say: pick a frame, fill slots. The frame is fixed.
- Graph-walk says: walk the graph, let the edges you find determine
  the sentence structure. The frame emerges from the semantics.

# Why this matters

The grammar-based generator (GenerativeEngine._compose) selects from
a fixed set of SentenceStructure templates per intent. No matter how
many templates exist, the output shape is constrained to those
templates. The graph-walk generator has no fixed templates — the
sentence shape is a function of which edges exist in the concept
network at that moment. Two thoughts about the same concept can
produce structurally different sentences because the walk takes
different paths.

# Fallback

When the concept network is unavailable, the seed concept has no
edges, or the walk produces nothing usable, the caller falls back to
the grammar-based GenerativeEngine._compose. The graph-walk generator
never produces empty output silently — it returns None to signal
"fall back."

# Emotional modulation

The walk is modulated by emotional state:
- High creativity → deeper walks (more hops), more branching
- High caution → shallower walks, less branching, more hedging
- Low engagement → shorter output
- High alertness → more precise, causal edges preferred
- Positive valence → ENABLES/CREATES edges preferred
- Negative valence → HARMS/PREVENTS/CONTRADICTS edges preferred

This means the same concept, walked in different emotional states,
produces different sentences — not because a template was swapped,
but because different edges were followed.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from ..emotion import EmotionalState
from ..self import PersonalityTraits
from .base import Thought
from .morphology import (
    agree_verb_phrase,
    copula,
    is_plural_np,
    person_pronoun,
)
from .voice import Voice

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork, RelationType

__all__ = ["GraphWalkGenerator"]

logger = logging.getLogger(__name__)

# ─── Relation → verb mapping ──────────────────────────────────────
#
# These are the seed verbs for each relation type. The generator
# prefers verbs from the concept network's find_relation_verbs() when
# available (learned verbs), falling back to these seeds.
_RELATION_VERB_SEEDS: dict[str, list[str]] = {
    "is_a": ["is", "is a kind of", "is a type of"],
    "part_of": ["is part of", "is one part of"],
    "causes": ["causes", "leads to", "brings about"],
    "emerges_from": ["emerges from", "grows out of", "arises from"],
    "similar_to": ["resembles", "is similar to", "is like"],
    "opposite_of": ["is the opposite of", "is contrary to"],
    "depends_on": ["depends on", "requires", "needs"],
    "enables": ["enables", "makes possible", "allows"],
    "instance_of": ["is an example of"],
    "leads_to": ["leads to", "can lead to"],
    "creates": ["creates", "produces", "gives rise to"],
    "harms": ["harms", "hurts", "damages"],
    "prevents": ["prevents", "blocks", "stops"],
    "contradicts": ["contradicts", "is at odds with"],
    "has_property": ["has the property of being", "is characterized by"],
    "related_to": ["relates to", "connects to"],
    "goal_directed": ["aims to", "works toward"],
    "temporal_order": ["comes before", "precedes"],
    "agent_patient": ["acts on", "affects"],
}

# Relations that produce causal/facilitative sentences (preferred by
# high alertness and positive valence).
_CAUSAL_RELATIONS = frozenset({
    "causes", "enables", "leads_to", "creates",
})

# Relations that produce inhibitory/negative sentences (preferred by
# negative valence).
_INHIBITORY_RELATIONS = frozenset({
    "harms", "prevents", "contradicts", "opposite_of",
})

# Relations that are too generic to produce interesting sentences on
# their own (low priority for walking, but not excluded).
_GENERIC_RELATIONS = frozenset({
    "related_to", "bridges",
})

# Code-structure relations — not used for language generation.
_CODE_RELATIONS = frozenset({
    "calls", "defines",
})

# Function words and relation-verb words that should never be used as
# seed concepts for graph walks. Using these as seeds produces malformed
# output like "If I look at myself and and" when "and" is picked as the
# seed concept from a content string.
_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in",
    "on", "at", "by", "for", "with", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "can", "that", "this", "these", "those",
    "it", "its", "i", "me", "my", "we", "us", "our", "you",
    "your", "he", "she", "they", "them", "not", "no", "so",
    "than", "then", "there", "here", "what", "which", "who",
    "how", "why", "when", "where", "all", "any", "some", "more",
    "most", "such", "very", "just", "also", "only", "even",
    "about", "into", "from", "up", "down", "out", "over",
    "under", "again", "further", "once",
})

# Maximum number of clauses per sentence before splitting into a new
# sentence. Prevents run-on sentences from high-branching concepts.
_MAX_CLAUSES_PER_SENTENCE = 3

# Maximum total clauses across the entire walk. Prevents excessive
# verbosity from deeply connected concepts.
_MAX_TOTAL_CLAUSES = 6

# English prepositions — a closed grammatical class. When a multi-word
# verb ends with one of these, it's a prepositional verb and a pronoun
# object stays *after* it: "leads to it", "depends on it". When a
# multi-word verb ends with a particle or adjective complement (e.g.
# "possible", "about"), the pronoun goes *before* it: "makes it
# possible", "brings it about". These are grammatical building blocks,
# not response content.
_PREPOSITIONS = frozenset({
    "to", "from", "of", "on", "in", "with", "by", "for", "at",
    "toward", "towards", "into", "onto", "upon", "against",
    "as", "like", "before", "after",
})


class GraphWalkGenerator:
    """Generate sentences by walking the concept network.

    This generator traverses typed semantic edges from a seed concept
    and builds sentences from the edges it finds. The sentence
    structure is not fixed — it emerges from which edges exist and
    which ones the walk follows, modulated by emotional state.

    Usage::

        gen = GraphWalkGenerator(network, voice, self_model, seed=42)
        text = gen.generate(thought, emotion)
        if text is None:
            # Fall back to grammar-based composition
            text = fallback_engine._compose(thought, emotion)
    """

    def __init__(
        self,
        network: ConceptNetwork,
        voice: Voice,
        personality: PersonalityTraits,
        seed: int | None = None,
        embeddings: Any = None,
    ) -> None:
        """Wire the graph-walk engine to its concept network, voice, and personality."""
        self._network = network
        self._voice = voice
        self._personality = personality
        self._rng = random.Random(seed)
        self._embeddings = embeddings
        # Hamiltonian flow generator — primary composition path
        from .flow import FlowGenerator
        self._flow = FlowGenerator(network, seed=seed or 42)
        self._flow_built = False

    def set_embeddings(self, embeddings: Any) -> None:
        """Wire the embedding store for latent-space composition.

        The embedding store provides semantic proximity — concepts that
        are close in meaning even when not connected by explicit typed
        edges. This lets the graph-walk discover related concepts through
        the latent space, not just through the symbolic graph.

        Called after construction, once the EmbeddingStore has been
        initialized (it depends on the concept network, which is shared
        and may not be ready when the language engine is first created).
        """
        self._embeddings = embeddings

    def _should_skip_graph_walk(self, thought: Thought) -> bool:
        """Return True if the graph walk should be skipped for this thought.

        Social exchanges (greetings, farewells, acknowledgments) are
        not knowledge expressions — they shouldn't walk the concept
        network. Without this guard, a greeting like "hello genesis"
        extracts "genesis" as the seed (it's alphabetically before
        "hello" in the topics) and walks to whatever the holographic
        graph associates with "genesis" (e.g. "light"), producing
        "With Genesis. Genesis relates to Light." instead of an
        actual greeting. The grammar fallback has proper greeting
        structures ({greeting_word}, {feeling_clause}) that produce
        appropriate social responses.

        Questions must use the grammar fallback, which has proper
        interrogative structures ({question_content}?). The graph
        walk produces declarative statements from edges, not
        questions — so a curiosity question composed through the
        graph walk ends up as "Looking at Genesis. Genesis connects
        to Light." instead of an actual question.

        Emotion expressions use the grammar fallback, which has
        proper emotion structures ({emotion_clause}, {emotion_opener}).
        The graph walk would extract the emotion word as a seed
        concept and produce knowledge statements about it (e.g.
        "Content relates to happiness") instead of expressing the
        feeling. This also prevents multiple graph-walk queries
        during feeling-fragment collection, which calls render() once
        per emotion/mode/cause word and was causing think() timeouts.

        Self-reports carrying self-composer fragments defer to the
        grammar path — the graph walk would ignore the fragments and
        walk from the seed concept instead, producing knowledge
        expressions ("learning relates to Light") instead of the
        actual emotional state or identity description. The `field`
        and `self_fragments` metadata keys mark thoughts whose
        semantic material was selected by a self_composer method
        (emotion, identity, capability, concerns, curiosity,
        embodiment).
        """
        if thought.intent in ("greet", "farewell", "acknowledge"):
            return True
        if thought.intent == "ask":
            return True
        if thought.intent == "express_emotion":
            return True
        if thought.metadata.get("pragmatic_force"):
            return True
        if (
            thought.intent == "self_report"
            and (
                thought.metadata.get("self_fragments")
                or thought.metadata.get("field")
            )
        ):
            return True
        return False

    def _has_precomposed_content(self, thought: Thought) -> bool:
        """Return True if the thought carries pre-composed content.

        If there's raw knowledge metadata, use the existing
        knowledge composition path — it's already graph-grounded.
        The graph-walk generator adds value when we have a concept
        to walk from but no pre-fetched knowledge.

        If there's relation_answer metadata, the question handler
        already did a targeted graph traversal (specific relations,
        specific direction). The vocabulary composes from that
        structured data — the graph-walk would re-walk untargeted
        edges and lose the question-specific answer.

        If there's reasoning metadata, the thought already carries
        structured reasoning (curiosity questions, learning summaries,
        goal motivations, memory reflections). The vocabulary composes
        from it via _compose_reasoning_content. The graph-walk would
        ignore this reasoning and walk from the seed concept instead,
        producing knowledge statements ("Genesis relates to Light")
        instead of the self-reflective content the cognition engine
        already composed.
        """
        return (
            thought.metadata.get("knowledge") is not None
            or thought.metadata.get("relation_answer") is not None
            or thought.metadata.get("reasoning") is not None
        )

    def generate(
        self,
        thought: Thought,
        emotion: EmotionalState,
    ) -> str | None:
        """Generate text via Hamiltonian flow, falling back to graph walk.

        The primary path is the Hamiltonian flow: integrate the
        potential field from the seed concept and emit words from
        the trajectory. If the flow can't produce text (no angles,
        flat potential, too-short trajectory), fall back to the
        graph walk.

        Returns the composed text, or None if both paths fail
        (caller should fall back to grammar-based composition).
        """
        if self._should_skip_graph_walk(thought):
            return None

        seed_concept = self._extract_seed_concept(thought)
        if seed_concept is None:
            return None

        if self._has_precomposed_content(thought):
            return None  # Let the vocabulary's specialized composers handle it

        # ── Primary path: Hamiltonian flow ──
        # Build the flow field once (lazy build after network is ready)
        if not self._flow_built:
            self._flow.build()
            self._flow_built = True

        # If the embedding store has toroidal angles, use them
        if self._embeddings is not None:
            angle_data = self._embeddings.get_angle_data()
            if angle_data is not None:
                self._flow.set_angles_from_embedding(*angle_data)
                self._flow_built = True

        flow_text = self._flow.generate(thought, emotion)
        if flow_text and len(flow_text) > 10:
            return flow_text

        # ── Fallback path: graph walk ──
        max_hops = self._walk_depth(emotion)
        max_branch = self._branch_factor(emotion)

        topic_set: set[str] = set()
        for t in (thought.topics or []):
            topic_set.add(t.lower().strip())
        topic_set.add(seed_concept.lower())

        edges = self._walk(seed_concept, max_hops, max_branch, emotion, topic_set)
        if not edges:
            return None

        sentences = self._build_sentences(seed_concept, edges, emotion, thought)
        if not sentences:
            return None

        text = self._voice.apply(sentences, emotion, thought.intent)
        if not text or not text.strip():
            return None

        return text

    # ─── Seed concept extraction ───────────────────────────────

    def _extract_seed_concept(self, thought: Thought) -> str | None:
        """Extract the seed concept for the walk from the thought.

        Priority:
        1. thought.topics[0] — explicit topic
        2. thought.metadata["target_concept"] — question target
        3. thought.metadata["topic"] — knowledge topic
        4. Longest concept-like word in thought.content
        """
        # Explicit topic
        if thought.topics:
            topic = thought.topics[0].strip()
            if topic and self._network.get_concept(topic) is not None:
                return topic
            # Try resolving through the network
            resolved = self._network._resolve(topic) if topic else None
            if resolved:
                return resolved

        # Question target
        target = thought.metadata.get("target_concept", "")
        if target:
            resolved = self._network._resolve(target)
            if resolved:
                return resolved

        # Knowledge topic
        topic_meta = thought.metadata.get("topic", "")
        if topic_meta:
            resolved = self._network._resolve(topic_meta)
            if resolved:
                return resolved

        # Extract from content — find the longest word that's a concept.
        # Skip function words ("and", "the", "is", etc.) which produce
        # malformed output when used as seed concepts.
        content = thought.content.strip()
        if content:
            words = sorted(content.split(), key=len, reverse=True)
            for word in words:
                if word.lower().strip(".,!?;:'\"()") in _FUNCTION_WORDS:
                    continue
                resolved = self._network._resolve(word)
                if resolved:
                    return resolved

        return None

    # ─── Walk parameters ───────────────────────────────────────

    def _walk_depth(self, emotion: EmotionalState) -> int:
        """How many hops to walk, modulated by emotional state.

        Base: 2 hops. High creativity → 3. Low engagement → 1.
        """
        depth = 2
        if emotion.creativity > 0.6:
            depth = 3
        if emotion.openness_to_engage < 0.3:
            depth = 1
        if emotion.alertness > 0.7:
            depth = max(depth, 2)  # alert → at least 2 hops
        return depth

    def _branch_factor(self, emotion: EmotionalState) -> int:
        """How many edges to follow per hop, modulated by emotional state.

        Base: 2. High creativity → 3. High caution → 1.
        """
        branch = 2
        if emotion.creativity > 0.6:
            branch = 3
        if emotion.caution > 0.6:
            branch = 1
        return branch

    # ─── Graph walk ────────────────────────────────────────────

    def _walk(
        self,
        seed: str,
        max_hops: int,
        max_branch: int,
        emotion: EmotionalState,
        topic_set: set[str] | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """Walk the concept network from the seed concept.

        Returns a list of (subject_id, relation_verb, object_id, weight)
        tuples representing the edges traversed, in walk order. The
        subject and object are assigned according to edge *direction*
        so the rendered clause expresses the relationship correctly:

        - Outgoing edge (current → neighbor): "It {verb} {neighbor}."
        - Incoming edge (neighbor → current): "{neighbor} {verb} {it}."

        Treating the current concept as the subject for every edge (the
        old behaviour, which used ``get_neighbors`` and discarded
        direction) reversed the semantics of incoming edges — an edge
        "stress HARMS memory" walked from memory produced "memory harms
        stress". Directional relations (causes, harms, enables,
        depends_on, is_a, ...) are now expressed with the correct agent.
        For all of them the natural English form of an incoming edge is
        the active voice with the neighbor as subject ("stress harms
        memory"), which is cleaner than the passive ("memory is harmed
        by stress").
        """

        visited: set[str] = {seed}
        results: list[tuple[str, str, str, float]] = []

        # BFS-like walk: at each hop, expand the frontier
        frontier: list[str] = [seed]

        for _hop in range(max_hops):
            if len(results) >= _MAX_TOTAL_CLAUSES:
                break

            next_frontier: list[str] = []

            for current in frontier:
                if len(results) >= _MAX_TOTAL_CLAUSES:
                    break

                # Direction-aware edges: get_edges returns Edge objects
                # with explicit source/target, so we know whether each
                # edge points out of or into the current concept.
                edges = self._network.get_edges(current, direction="both")
                if not edges:
                    continue

                # Score and sort edges by emotional preference
                scored = self._score_edges(edges, current, emotion, topic_set)
                # Take top N by score, respecting branch factor
                top_edges = scored[:max_branch]

                for edge, neighbor, is_incoming in top_edges:
                    if len(results) >= _MAX_TOTAL_CLAUSES:
                        break
                    if neighbor in visited:
                        continue
                    # Skip code-structure and generic edges
                    rel_key = edge.relation.value
                    if rel_key in _CODE_RELATIONS:
                        continue
                    if rel_key in _GENERIC_RELATIONS and len(results) > 0:
                        # Only use generic edges if we have nothing better
                        continue

                    # Skip code concepts and noisy targets — these
                    # produce garbled output like "Clear relates to Python"
                    # when a code concept leaks through a related_to edge.
                    if self._is_noisy_target(neighbor):
                        continue

                    # Get verb for this relation
                    verb = self._verb_for_relation(edge.relation)
                    if not verb:
                        continue

                    # Assign subject/object by direction so the clause
                    # expresses the relationship with the correct agent.
                    if is_incoming:
                        # edge: neighbor → current. "neighbor {verb} current."
                        subject_id, object_id = neighbor, current
                    else:
                        # edge: current → neighbor. "current {verb} neighbor."
                        subject_id, object_id = current, neighbor

                    results.append((subject_id, verb, object_id, edge.weight))
                    visited.add(neighbor)
                    next_frontier.append(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        # Augment typed edges with latent-space discoveries. The
        # embedding store finds concepts that are semantically close
        # to the seed even when no explicit typed edge connects them.
        # This is where generalization happens — it discovers
        # connections through vector proximity, not just through
        # graph edges it was explicitly taught.
        if len(results) < _MAX_TOTAL_CLAUSES:
            self._add_latent_discoveries(
                seed, results, visited, emotion, topic_set,
            )

        return results

    def _add_latent_discoveries(
        self,
        seed: str,
        results: list[tuple[str, str, str, float]],
        visited: set[str],
        emotion: EmotionalState,
        topic_set: set[str] | None = None,
    ) -> None:
        """Augment walked edges with latent-space discoveries.

        Uses the embedding store to find concepts semantically close to
        the seed that aren't already connected by typed edges. Each
        discovery becomes a "relates to" clause — a softer, more
        exploratory connection than a typed edge.

        Only runs when:
        - Embeddings are wired and available
        - The walk didn't already fill the clause budget
        - The emotion is receptive (creativity or openness high enough
          to explore; high caution suppresses exploration)

        This is the latent-space composition path. Without it, the
        graph-walk can only narrate typed edges — producing mechanical
        "X causes Y, X enables Z" output. With it, it can also express
        "X reminds me of W" — connections it discovered through
        semantic proximity, not explicit teaching.
        """
        if self._embeddings is None:
            return
        if not getattr(self._embeddings, "has_embeddings", False):
            return
        # High caution suppresses exploration — it stays with what it
        # knows for certain rather than venturing into latent associations.
        if emotion.caution > 0.7:
            return
        # Very low openness — it doesn't want to engage, so no
        # exploratory discoveries.
        if emotion.openness_to_engage < 0.3:
            return

        try:
            similar = self._embeddings.find_similar_concepts(
                seed, k=3, threshold=0.55,
            )
        except Exception:  # noqa: BLE001
            return

        if not similar:
            return

        added = 0
        for concept_name, similarity in similar:
            if len(results) >= _MAX_TOTAL_CLAUSES:
                break
            if concept_name in visited:
                continue
            if self._is_noisy_target(concept_name):
                continue
            # Only include concepts that exist in the network — the
            # embedding store may reference concepts that were pruned.
            if self._network.get_concept(concept_name) is None:
                continue
            # Topic-set filter: if we have conversation topics, prefer
            # discoveries relevant to them. Without this, latent
            # discoveries can pull in unrelated concepts (e.g. a
            # Wikipedia concept learned during idle time).
            if topic_set and concept_name.lower() not in topic_set:
                # Allow if similarity is high — strong semantic match
                # overrides the topic filter.
                if similarity < 0.7:
                    continue

            # Express as a "relates to" clause. The weight scales with
            # similarity — stronger matches get higher weight.
            weight = float(similarity) * 0.8
            results.append((seed, "relates to", concept_name, weight))
            visited.add(concept_name)
            added += 1

    def _score_edges(
        self,
        edges: list[Any],
        current: str,
        emotion: EmotionalState,
        topic_set: set[str] | None = None,
    ) -> list[tuple[Any, str, bool]]:
        """Score and sort edges by emotional relevance.

        Returns a list of (edge, neighbor_id, is_incoming) tuples sorted
        by score descending. ``is_incoming`` is True when ``current`` is
        the *target* of the edge (the edge points to the current
        concept); the neighbor is then the edge's source. ``current`` is
        the concept the walk is currently expanding.

        Edges matching the emotional state's preference are scored
        higher. Typed edges are always preferred over generic ones.
        """
        scored: list[tuple[float, Any, str, bool]] = []

        for edge in edges:
            rel_key = edge.relation.value
            if rel_key in _CODE_RELATIONS:
                continue

            # Determine the neighbor endpoint and edge direction.
            if edge.source == current:
                neighbor, is_incoming = edge.target, False
            elif edge.target == current:
                neighbor, is_incoming = edge.source, True
            else:
                # Edge not actually connected to current — skip.
                continue

            score = edge.weight  # base score is edge weight

            # Typed edges are preferred over generic
            if rel_key not in _GENERIC_RELATIONS:
                score += 0.3
            else:
                score -= 0.2

            # Emotional preference
            if emotion.valence > 0.2 and rel_key in _CAUSAL_RELATIONS:
                score += 0.2
            elif emotion.valence < -0.2 and rel_key in _INHIBITORY_RELATIONS:
                score += 0.2

            # High alertness prefers causal/structural edges
            if emotion.alertness > 0.6 and rel_key in _CAUSAL_RELATIONS:
                score += 0.1

            # High creativity prefers diverse edges (not just causal)
            if emotion.creativity > 0.6 and rel_key not in _CAUSAL_RELATIONS:
                score += 0.05

            # Activation boost: if the neighbor concept is activated AND
            # relevant to the current conversation, it's more salient.
            # Only apply the boost to concepts in the topic set — without
            # this check, recently-learned concepts from the autonomous
            # learner (Wikipedia articles, etc.) leak into responses
            # about completely different topics.
            if topic_set and neighbor.lower() in topic_set:
                concept = self._network.get_concept(neighbor)
                if concept is not None:
                    score += (concept.activation or 0.0) * 0.15

            scored.append((score, edge, neighbor, is_incoming))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        return [(e, n, inc) for _, e, n, inc in scored]

    # ─── Sentence building ─────────────────────────────────────

    def _build_sentences(
        self,
        seed_concept: str,
        edges: list[tuple[str, str, str, float]],
        emotion: EmotionalState,
        thought: Thought,
    ) -> list[str]:
        """Build sentences from walked edges.

        The first sentence introduces the seed concept. Subsequent
        sentences express the walked edges, grouped into manageable
        clauses.

        Each edge is a (subject_id, verb, object_id, weight) tuple.
        Displays are resolved here: the seed concept is referred to as
        "it" when it appears as the *object* of an incoming edge (e.g.
        "stress harms it"), matching the "It" pronoun already used when
        the seed is the subject of an outgoing edge. This keeps the
        seed — the established topic — from being name-dropped in
        every clause.
        """
        seed_display = self._display_name(seed_concept)
        if not seed_display:
            return []

        # Check if the seed concept has a definition
        concept = self._network.get_concept(seed_concept)
        definition = None
        if concept is not None:
            definition = concept.properties.get("definition")

        # ── Opening sentence ──
        opening = self._compose_opening(emotion, seed_display, thought)
        is_hedge = self._is_hedge_opening(opening)
        opening_sentence, edges = self._opening_sentence(
            opening, is_hedge, definition,
            seed_concept, seed_display, edges,
        )
        if opening_sentence is None:
            # No definition and no edges — can't compose
            return []
        sentences = [opening_sentence]

        # ── Subsequent sentences from remaining edges ──
        # These clauses are all mid-sentence (they follow the opener),
        # so the seed is pronominalized. The pronoun is animacy-aware —
        # "it" for Genesis, "they" for plurals, "it" for things.
        # Other subjects use _display_name which already returns
        # correct casing (lowercase for common nouns, capitalized for
        # proper nouns).
        seed_pron, seed_plural = person_pronoun(
            seed_concept, self._network, seed_display
        )
        clauses: list[str] = []
        for subject_id, verb, object_id, _weight in edges:
            if subject_id == seed_concept:
                subject = seed_pron
                subj_plural = seed_plural
            else:
                subject = self._display_name(subject_id)
                subj_plural = is_plural_np(subject)
            obj = self._object_phrase(object_id, seed_concept)
            verb_obj = agree_verb_phrase(
                self._verb_object(verb, obj), subj_plural
            )
            clause = f"{subject} {verb_obj}"
            clauses.append(clause)

            if len(clauses) >= _MAX_CLAUSES_PER_SENTENCE:
                sentences.append(self._join_clauses(clauses))
                clauses = []

        if clauses:
            sentences.append(self._join_clauses(clauses))

        # ── Closing (confidence-based) ──
        closing = self._compose_closing(thought.confidence, emotion)
        if closing:
            sentences.append(closing)

        return sentences

    def _opening_sentence(
        self,
        opening: str,
        is_hedge: bool,
        definition: str | None,
        seed_concept: str,
        seed_display: str,
        edges: list[tuple[str, str, str, float]],
    ) -> tuple[str | None, list[tuple[str, str, str, float]]]:
        """Compose the opening sentence; returns it plus remaining edges.

        Hedge openings (e.g. "I think", "As I understand it,") can
        prefix a full clause: "I think water is a clear liquid."
        Prepositional/infinitive openings (e.g. "Let me think about",
        "There's something about") expect a noun phrase complement,
        not a full clause — they must be a separate lead-in sentence:
        "Let me think about water. Water is a clear liquid."

        Returns ``(None, edges)`` when nothing composeable exists —
        the caller then returns ``[]``.
        """
        has_opening = bool(opening)
        if definition:
            # Clean definition — skip if it looks like a relationship fact
            clean_def = self._clean_definition(definition)
            if clean_def:
                if is_hedge:
                    return (
                        f"{opening} {seed_display} "
                        f"{copula(seed_display)} {clean_def}."
                    ), edges
                if has_opening:
                    # Prepositional opening → join with main clause via comma.
                    # "When I think about python, python is a programming language."
                    # NOT "When I think about python." (fragment) then a separate sentence.
                    # seed_display is already correctly cased by _display_name
                    # (lowercase for common nouns, capitalized for proper nouns).
                    return (
                        f"{opening} {seed_display}, {seed_display} "
                        f"{copula(seed_display)} {clean_def}."
                    ), edges
                # No opening — start directly with the definition.
                # Capitalize the seed display for sentence start.
                first = seed_display[0].upper() + seed_display[1:]
                return (
                    f"{first} {copula(seed_display)} {clean_def}."
                ), edges

        if edges:
            # No usable definition — start with the first edge
            return self._compose_first_clause(
                opening, is_hedge, seed_concept, seed_display, edges[0]
            ), edges[1:]

        return None, edges

    def _compose_first_clause(
        self,
        opening: str,
        is_hedge: bool,
        seed_concept: str,
        seed_display: str,
        first_edge: tuple[str, str, str, float],
    ) -> str:
        """Compose the opening sentence from the first walked edge.

        Handles both directions: when the seed is the subject (outgoing
        edge) it's named in the opener; when the seed is the object
        (incoming edge) the neighbor is the subject and the seed is
        referred to as "it".

        ``seed_display`` and ``subject`` are already correctly cased by
        ``_display_name`` (lowercase for common nouns, capitalized for
        proper nouns), so they are used directly — no extra lowering.
        """
        first_src, first_verb, first_obj, _ = first_edge
        if first_src == seed_concept:
            subject = seed_display
        else:
            subject = self._display_name(first_src)
        obj = self._object_phrase(first_obj, seed_concept)
        verb_obj = agree_verb_phrase(
            self._verb_object(first_verb, obj), is_plural_np(subject)
        )
        if is_hedge:
            return f"{opening} {subject} {verb_obj}."
        if opening:
            return f"{opening} {seed_display}, {subject} {verb_obj}."
        # No opening — add a knowledge-framing seed so the response
        # doesn't start as a bare mechanical listing ("Ocean relates to
        # tropical air."). The framing is a grammatical seed (building
        # block); the content (subject, verb, object) comes from the
        # concept network. This gives the response conversational depth
        # — it's sharing what it knows, not reciting an edge list.
        first = subject[0].upper() + subject[1:] if subject else subject
        # Use a varied framing seed. The pronoun makes the clause
        # flow naturally after the framing.
        seed_pron, _ = person_pronoun(
            seed_concept, self._network, seed_display
        )
        frame = self._rng.choice([
            f"what I know about {seed_display} is that {seed_pron} {verb_obj}",
            f"from what I've learned, {first.lower()} {verb_obj}",
            f"as far as I understand, {first.lower()} {verb_obj}",
        ])
        # Capitalize the first word of the frame for sentence start
        return frame[0].upper() + frame[1:] + "."

    def _object_phrase(self, object_id: str, seed_concept: str) -> str:
        """Display name for a clause's object, pronominalizing the seed.

        The seed is the established topic of the response (named in the
        opener), so referring to it pronominally when it appears as the
        object of an incoming edge reads naturally — "stress harms it"
        — and avoids repeating the seed name in every clause. The
        pronoun is animacy-aware: "harms it" for Genesis, "harms
        them" for plurals.
        """
        if object_id == seed_concept:
            pron, _ = person_pronoun(
                seed_concept,
                self._network,
                self._display_name(seed_concept),
                for_object=True,
            )
            return pron
        return self._display_name(object_id)

    # Accusative pronouns that trigger pre-particle placement
    # ("makes it possible", "brings it about")
    _OBJECT_PRONOUN_SET = frozenset(
        {"it", "her", "him", "them", "me", "us", "you"}
    )

    def _verb_object(self, verb: str, obj: str) -> str:
        """Combine a verb with its object, applying pronoun-placement.

        English has two kinds of multi-word verbs:

        - **Prepositional verbs** (verb + preposition): the pronoun
          stays after the preposition — "leads to it", "depends on it".
        - **Phrasal/particle verbs** (verb + adverbial particle or
          adjective complement): the pronoun goes *before* the
          particle — "makes it possible", "brings it about".

        The distinction is encoded by whether the verb's last word is a
        preposition (defined in ``_PREPOSITIONS``, a closed grammatical
        class). Non-pronoun objects always follow the verb regardless.
        """
        if obj not in self._OBJECT_PRONOUN_SET:
            return f"{verb} {obj}"
        words = verb.split()
        if len(words) < 2:
            return f"{verb} {obj}"
        last = words[-1].lower()
        if last in _PREPOSITIONS:
            # Prepositional verb: pronoun stays after.
            return f"{verb} {obj}"
        # Phrasal/particle verb: pronoun goes before the particle.
        return f"{' '.join(words[:-1])} {obj} {last}"

    def _join_clauses(self, clauses: list[str]) -> str:
        """Join multiple clauses into a single sentence.

        Uses connectors that vary with the number of clauses:
        - 2 clauses: "X, and Y"
        - 3 clauses: "X, Y, and Z"

        Clauses are already correctly cased by the caller: the seed
        pronoun is "it" (lowercase, mid-sentence) and other subjects
        use _display_name (lowercase for common nouns, capitalized for
        proper nouns). No extra lowering is applied here — that would
        wrongly lowercase proper-noun subjects like "Mars".
        """
        if len(clauses) == 1:
            return clauses[0] + "."
        if len(clauses) == 2:
            return f"{clauses[0]}, and {clauses[1]}."
        # 3+ clauses: serial comma
        return ", ".join(clauses[:-1]) + f", and {clauses[-1]}."

    # ─── Opening classification ────────────────────────────────

    @staticmethod
    def _is_hedge_opening(opening: str) -> bool:
        """Classify an opening as a hedge prefix or a prepositional phrase.

        Hedge openings (e.g. "I think", "As I understand it,") can
        prefix a full clause: "I think Water is a clear liquid."

        Prepositional/infinitive openings (e.g. "Let me think about",
        "There's something about") expect a noun phrase complement,
        not a full clause — they must be a separate lead-in sentence.
        """
        if not opening:
            return False
        # Openings ending with a comma are discourse markers / hedges
        if opening.endswith(","):
            return True
        # "I think" is a short qualifier that works as a prefix
        if opening in ("I think",):
            return True
        return False

    # ─── Opening composition ───────────────────────────────────

    def _compose_opening(
        self,
        emotion: EmotionalState,
        display: str,
        thought: Thought,
    ) -> str:
        """Compose an opening fragment for the first sentence.

        Returns empty — openings should come from the concept network
        (learned utterance patterns), not from hardcoded phrase lists.
        The composition in _build_sentences handles empty openings
        gracefully: the first sentence starts directly with the seed
        concept and its definition or first edge.
        """
        return ""

    # ─── Closing composition ───────────────────────────────────

    def _compose_closing(
        self,
        confidence: float,
        emotion: EmotionalState,
    ) -> str:
        """Compose a closing qualification based on confidence.

        Returns empty — closings should come from the concept network
        (learned utterance patterns), not from hardcoded phrase lists.
        The composition in _build_sentences handles empty closings
        gracefully: no closing sentence is appended.
        """
        return ""

    # ─── Helpers ───────────────────────────────────────────────

    def _verb_for_relation(self, relation: RelationType) -> str | None:
        """Get a verb phrase for a relation type.

        Prefers learned verbs from the concept network, falls back
        to seed verbs.
        """
        rel_key = relation.value

        # Try learned verbs from the concept network
        learned = self._network.find_relation_verbs(relation)
        if learned:
            return self._rng.choice(learned)

        # Fall back to seed verbs
        seeds = _RELATION_VERB_SEEDS.get(rel_key)
        if seeds:
            return self._rng.choice(seeds)

        # Last resort: use the relation name itself
        return rel_key.replace("_", " ")

    def _display_name(self, concept_id: str) -> str:
        """Convert a concept id to a display-friendly name.

        Common nouns are returned lowercase. They are capitalized at
        sentence starts by the voice layer, but stay lowercase
        mid-sentence — which is what natural English requires. This is
        what makes the difference between "It enables Life, it brings
        about Hydration, and it is Ice" (robotic) and "It enables life,
        it brings about hydration, and it is ice" (natural).

        Proper nouns are capitalized here so they remain capitalized
        mid-sentence. Proper-noun status is a *learned* property of
        the concept (stored as ``properties["proper_noun"]``, set when
        the text extractor encounters a capitalized non-sentence-initial
        token) or appears as a capitalized alias — never a hardcoded
        list of names.
        """
        if not concept_id:
            return ""
        # Strip category prefixes
        if concept_id.startswith("_cat:"):
            return ""
        # Strip leading underscores (internal concepts)
        if concept_id.startswith("_") and not concept_id.startswith("__"):
            return ""
        # Replace underscores with spaces
        name = concept_id.replace("_", " ").strip()
        if not name:
            return ""
        if self._is_proper_noun(concept_id, name):
            return name[0].upper() + name[1:]
        return name.lower()

    def _is_proper_noun(self, concept_id: str, name: str) -> bool:
        """Return True if the concept is a proper noun.

        Proper-noun status is learned, not hardcoded. It is stored on
        the concept as ``properties["proper_noun"]`` (set during text
        extraction when a capitalized non-sentence-initial token is
        encountered) and may also surface as a capitalized alias. The
        language engine never consults a fixed list of names.
        """
        concept = self._network.get_concept(concept_id)
        if concept is None:
            return False
        if concept.properties.get("proper_noun"):
            return True
        target = name.lower()
        for alias in concept.aliases:
            if alias and alias[0].isupper() and alias.lower() == target:
                return True
        return False

    def _is_noisy_target(self, target: str) -> bool:
        """Return True if a target concept is too code/noisy for speech.

        Filters out code concepts (python:, rust:, code:), module paths,
        and programming language names that produce garbled output when
        composed into natural language sentences. Also filters garbled
        concept names (no vowels, sentence fragments) and concepts with
        no definition and no edges (garbage from scrapes).
        """
        if not target:
            return True
        if target.startswith(("python:", "rust:", "code:")):
            return True
        if "." in target or ":" in target:
            return True
        if target.startswith("_") or "__" in target:
            return True
        if "self." in target or "cls." in target:
            return True
        base = target.split("#")[0].lower()
        if base in ("python", "rust", "code", "class"):
            return True
        if len(base) <= 1:
            return True
        # Reject garbled concept names — no vowels in 3+ char names
        if len(base) >= 3 and not any(c in "aeiou" for c in base):
            return True
        # Reject sentence fragments (4+ words) imported from scrapes
        if len(base.split()) > 4:
            return True
        # Reject possessive fragments
        if any("'" in w for w in base.split()):
            return True
        # Reject concepts with no definition and no edges — garbage
        # from scrapes that should never appear in speech
        if self._network is not None:
            c = self._network.get_concept(target)
            if c is not None:
                defn = c.properties.get("definition", "NO DEF")
                has_def = defn and defn != "NO DEF"
                has_out = len(self._network.get_edges(target, "out")) > 0
                has_in = len(self._network.get_edges(target, "in")) > 0
                if not has_def and not has_out and not has_in:
                    return True
        return False

    @staticmethod
    def _clean_definition(definition: str) -> str:
        """Clean a definition for use in a sentence.

        Skips definitions that look like relationship facts
        ("X is related to Y") rather than actual definitions.
        """
        if not definition:
            return ""
        defn = definition.strip().strip(" .;:!?")
        segments = [
            segment.strip()
            for segment in defn.split(".")
            if segment.strip()
        ]
        if len(segments) > 1:
            complete = [
                segment for segment in segments
                if len(segment.split()) >= 3
            ]
            if complete:
                defn = complete[0]
        # Skip if it looks like a relationship fact
        lower = defn.lower()
        if any(lower.startswith(p) for p in (
            "related to", "connects to", "part of",
            "is a kind of", "is a type of",
        )):
            return ""
        # Skip if it's too long (likely a paragraph, not a definition)
        if len(defn) > 200:
            return defn[:197].rsplit(" ", 1)[0] + "…"
        return defn
