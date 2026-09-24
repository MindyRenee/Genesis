"""Thought composer — where novel thoughts are actually generated.

This is the module that makes Genesis intelligent rather than just
sophisticated. It bridges the gap between knowing things (concept
network) and saying things (language engine).

# The problem it solves

Before this module, the cognition engine had hardcoded responses:
"What is cognition?" → a string I wrote about cognition.
The concept network had 35 concepts and 19 relationships, but they
weren't used to compose responses. The reasoning engine produced
conclusions, but they were fallback content, not primary.

The thought composer changes this. When Genesis is asked about a
concept, she:
1. Looks up what she knows (concept network)
2. Reasons about it (reasoning engine)
3. Checks what she's said before (memory)
4. Composes a novel thought from all of this

The result is responses that are grounded in her actual knowledge,
not in strings someone wrote for her. When she learns something new,
her future responses change. That's the beginning of real intelligence.

# How it works

compose_about(concept) → Thought
  1. Get the concept node — what is it? how confident is she?
  2. Get its relationships — what connects to what?
  3. Run reasoning — what follows from what she knows?
  4. Check memory — has she discussed this before?
  5. Compose: weave together what she knows, what she's reasoned,
     and what she remembers into a novel thought.

compose_answer(question, concepts) → Thought
  1. For each concept in the question, compose about it
  2. Find connections between the concepts
  3. Synthesize: what's the unifying idea?
  4. Express the answer with appropriate confidence

compose_reflection(topic) → Thought
  1. What does she know about this topic?
  2. What's she uncertain about?
  3. What connections has she reasoned?
  4. Express as a first-person reflection
"""

from __future__ import annotations

import random
import re
from collections import deque
from typing import TYPE_CHECKING, Any

from ..concepts import Concept, ConceptNetwork, RelationType
from ..emotion import EmotionalState
from ..language import Thought
from ..reasoning import ReasoningEngine, ReasoningResult, ReasoningType

if TYPE_CHECKING:
    from ..concepts import EmbeddingStore


# Origins of auto-generated edges that exist only to keep the graph
# connected — reachability scaffolding with no semantic content. They
# must never be spoken as knowledge: "makes related_to emotion"
# (weight 0.10, hub_attachment) is the canonical example. This mirrors
# the exclusion in self/learning.py, which already refuses hub_attachment
# edges when composing definitions.
_UNRELIABLE_EDGE_ORIGINS = frozenset({"hub_attachment", "bridge"})

# Minimum edge weight for a relationship to be worth speaking. Low-weight
# co-occurrence edges add noise without adding meaning.
_MIN_FACT_WEIGHT = 0.3


def _is_reliable_fact_edge(edge: Any) -> bool:
    """Return True if an edge is trustworthy enough to speak as knowledge."""
    if getattr(edge, "origin", "") in _UNRELIABLE_EDGE_ORIGINS:
        return False
    return edge.weight >= _MIN_FACT_WEIGHT


class ThoughtComposer:
    """Composes novel thoughts from concept network knowledge.

    This is the bridge between knowing and saying. It takes a topic
    or question and produces a Thought whose content is grounded in
    what Genesis actually knows, not in hardcoded strings.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        reasoner: ReasoningEngine,
        language=None,
        seed: int | None = None,
    ) -> None:
        """Initialize the composer with a concept network, reasoner, language engine, and RNG."""
        self.network = network
        self.reasoner = reasoner
        self._language = language
        self._rng = random.Random(seed)

        # Question composer for generating genuine follow-up questions
        from .question_composer import QuestionComposer

        self.question_composer = QuestionComposer(network, seed=seed)

        # Embedding store — set by CognitionEngine after creation
        # Used for finding related concepts via latent space
        self.embeddings: EmbeddingStore | None = None

        # Track what she's said about each concept (for variation)
        self._said_about: dict[str, deque[str]] = {}

    def compose_about(
        self,
        concept_name: str,
        emotion: EmotionalState,
        depth: int = 2,
        focused: bool = False,
        mode: str = "full",
    ) -> Thought | None:
        """Compose a thought about a specific concept.

        This is the core function. It looks up what Genesis knows
        about a concept, reasons about it, and composes a novel
        thought expressing her understanding.

        When ``focused`` is True (e.g., when answering a direct
        question), the response is kept tight — definition and key
        relationships only, without associative tangents, discovery
        phrases, or follow-up questions. Those are appropriate for
        free-form conversation but add noise to focused answers.

        When ``mode`` is "definition", only the definition is returned
        — no relationship facts, no reasoning, no follow-up. This is
        useful for contexts like empathy where the definition is all
        that's needed and relationship facts would be noise.

        Returns None if she has no knowledge of the concept.
        """
        concept = self.network.get_concept(concept_name)
        if concept is None:
            return None

        # Gather what she knows
        knowledge = self._gather_knowledge(concept_name, depth)
        if not knowledge:
            # She has the concept but no relationships and no definition.
            # Rather than reciting a pre-written "I can't articulate"
            # template, she stays silent — the urge still drives the
            # action, she just doesn't verbalize what she can't yet
            # express from her own understanding.
            return None

        # Reason about it
        reasoning_results = self.reasoner.reason_about(concept_name)

        # Confidence is based on concept confidence + reasoning confidence
        confidence = self._compose_confidence(concept, reasoning_results)

        # Compose the content, retrying up to 3 times to avoid
        # repetition. The shuffle in _weave_relationship_facts means
        # each attempt draws different facts, so retries produce
        # different sentences.
        content = self._weave_non_repetitive(
            concept_name, knowledge, reasoning_results, emotion,
            focused=focused, mode=mode,
        )
        if content is None:
            return None

        # Only include non-hypothesis reasoning in metadata for
        # focused answers. Hypotheses are speculative transitive
        # inferences ("sleep might be part of hippocampus" through
        # memory) — including them in metadata causes the language
        # engine to render speculation as fact. In focused mode,
        # the user wants facts, not speculation.
        if focused:
            reasoning_conclusions = [
                r.conclusion for r in reasoning_results[:2]
                if r.reasoning_type != ReasoningType.HYPOTHESIS
            ]
        else:
            reasoning_conclusions = [r.conclusion for r in reasoning_results[:2]]
        definition = self._extract_definition(knowledge)
        metadata = self._build_knowledge_metadata(
            concept_name, knowledge, reasoning_conclusions, confidence, definition
        )

        # Track what she said (for future variation)
        self._track_said(concept_name, content)

        return Thought(
            content=content,
            intent="inform",
            emotion=emotion.label,
            topics=[concept_name],
            confidence=confidence,
            metadata=metadata,
        )

    def _compose_confidence(
        self,
        concept: Concept,
        reasoning_results: list[ReasoningResult],
    ) -> float:
        """Blend concept confidence with reasoning confidence."""
        confidence = concept.confidence
        if reasoning_results:
            avg_reasoning_conf = sum(r.confidence for r in reasoning_results) / len(
                reasoning_results
            )
            confidence = (confidence + avg_reasoning_conf) / 2
        return confidence

    def _weave_non_repetitive(
        self,
        concept_name: str,
        knowledge: list[tuple[str, str, float]],
        reasoning_results: list[ReasoningResult],
        emotion: EmotionalState,
        *,
        focused: bool,
        mode: str,
    ) -> str | None:
        """Weave knowledge into content, retrying to avoid repetition.

        Returns None if weaving produces nothing (she can't express
        this from her own understanding) — she stays silent rather than
        reciting a pre-written fallback template.
        """
        for _attempt in range(3):
            content = self._weave_knowledge(
                concept_name, knowledge, reasoning_results, emotion,
                focused=focused, mode=mode,
            )
            if not content:
                return None
            if not focused and self.has_said_similar(concept_name, content):
                continue  # too similar to something she just said — retry
            return content
        return content  # all retries were repetitive — use last attempt

    # ── Generic graph-based helpers for question answering ──
    # These methods derive their behavior entirely from the structure
    # of the concept network — what Genesis has learned. No domain
    # knowledge is hardcoded.

    def _find_best_connected_pair(
        self, topics: list[str], question_text: str = ""
    ) -> tuple[str, str] | None:
        """Find the pair of topics with the most edges between them.

        This replaces hardcoded meta-word filtering: question structural
        words like "relationship" naturally drop out because they have
        no edges to domain concepts. The pair with the strongest graph
        connection is the one the user is actually asking about.

        When question_text is provided, pairs where both topics appear
        in the text are strongly preferred over pairs where one or both
        topics were semantically inferred (e.g., by the embedding
        system). This prevents "Compare ice and steam" from matching
        ice-glacier just because glacier is semantically related to ice.
        """
        if len(topics) < 2:
            return None

        lower_text = question_text.lower() if question_text else ""

        best_pair: tuple[str, str] | None = None
        best_score = 0.0

        for i, a in enumerate(topics):
            ca = self.network._resolve(a) or a.lower()
            a_in_text = a.lower() in lower_text if lower_text else True
            for j, b in enumerate(topics):
                if j <= i:
                    continue
                cb = self.network._resolve(b) or b.lower()
                b_in_text = b.lower() in lower_text if lower_text else True
                # Count edges in both directions
                edge_score = self._bidirectional_edge_score(ca, cb)
                if edge_score == 0:
                    continue
                # Strongly prefer pairs where both topics appear in the
                # question text (explicitly mentioned by the user).
                # This prevents semantically inferred topics from
                # displacing explicitly mentioned ones.
                if a_in_text and b_in_text:
                    score = edge_score + 100  # explicit mention bonus
                else:
                    score = edge_score
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

        return best_pair

    def _bidirectional_edge_score(self, ca: str, cb: str) -> float:
        """Sum edge weights between two concepts in both directions."""
        score = 0.0
        for e in self.network.get_edges(ca, "out"):
            if (self.network._resolve(e.target) or e.target.lower()) == cb:
                score += e.weight
        for e in self.network.get_edges(cb, "out"):
            if (self.network._resolve(e.target) or e.target.lower()) == ca:
                score += e.weight
        return score

    def _find_process_and_subject(
        self, topics: list[str], lower_q: str
    ) -> tuple[str | None, str | None]:
        """Find a process concept and its subject from the question.

        A process is any concept with outgoing CREATES, CAUSES, or
        LEADS_TO edges — this is learned structure, not a hardcoded
        word list. The subject is the other topic in the question
        (the thing undergoing the process).

        Morphological resolution tries all common inflection variants
        (-ing, -s, -es, base form) and prefers the variant that has
        process edges in the graph.
        """
        from ..concepts import RelationType

        process_rels = (RelationType.CREATES, RelationType.CAUSES, RelationType.LEADS_TO)

        # Collect all candidate concepts from topics and question text.
        # Track which came from original topics (preferred subjects) vs
        # text scan (may include morphological variants of processes).
        candidates: list[str] = []
        topic_candidates: list[str] = []  # from original topics
        seen: set[str] = set()

        # Check topics first (they're already resolved)
        for t in topics:
            resolved = self.network._resolve(t) or t.lower()
            if resolved not in seen:
                candidates.append(resolved)
                topic_candidates.append(resolved)
                seen.add(resolved)

        # Also scan the question text for words that resolve to concepts
        for word in lower_q.split():
            word = word.strip(".,!?;:\"'()[]")
            if not word:
                continue
            for resolved in _all_resolutions(word, self.network):
                if resolved in seen:
                    continue
                candidates.append(resolved)
                seen.add(resolved)

        # Find which candidates have process edges
        process_candidates: list[str] = []
        for cid in candidates:
            out_edges = self.network.get_edges(cid, "out")
            if any(e.relation in process_rels for e in out_edges):
                process_candidates.append(cid)

        if not process_candidates:
            return None, None

        # Prefer the process candidate with the most causal edges
        process_candidates.sort(
            key=lambda c: sum(
                1 for e in self.network.get_edges(c, "out") if e.relation in process_rels
            ),
            reverse=True,
        )
        proc = process_candidates[0]

        subj = _find_subject(topic_candidates, proc, process_rels, self.network)
        return proc, subj

    def _find_most_functional_topic(self, topics: list[str]) -> str | None:
        """Find the topic with the most functional edges.

        Functional edges are ENABLES, CREATES, CAUSES, and LEADS_TO —
        any edge that represents the concept doing something or
        producing an effect. This is used for affordance questions to
        identify which concept the user is asking about, based on
        which one has functional relationships in the graph.
        """
        from ..concepts import RelationType

        functional_rels = (
            RelationType.ENABLES,
            RelationType.CREATES,
            RelationType.CAUSES,
            RelationType.LEADS_TO,
        )

        best_topic: str | None = None
        best_score = 0

        for t in topics:
            cid = self.network._resolve(t) or t.lower()
            score = sum(
                1 for e in self.network.get_edges(cid, "out") if e.relation in functional_rels
            )
            if score > best_score:
                best_score = score
                best_topic = t

        return best_topic

    def _try_explain_why(self, topics: list[str]) -> ReasoningResult | None:
        """Try to explain why one topic depends on another.

        Checks all topic pairs for a DEPENDS_ON relationship and
        traces the causal chain. This is generic graph reasoning —
        no hardcoded domain knowledge.
        """
        for i, a in enumerate(topics):
            ca = self.network._resolve(a) or a.lower()
            for j, b in enumerate(topics):
                if j == i:
                    continue
                cb = self.network._resolve(b) or b.lower()
                # Check if a depends_on b
                result = self._check_dependency(ca, cb, a, b)
                if result:
                    return result
                # Check if b depends_on a
                result = self._check_dependency(cb, ca, b, a)
                if result:
                    return result
        return None

    def _check_dependency(
        self, source: str, target: str, a: str, b: str
    ) -> ReasoningResult | None:
        """Check if source depends_on target and explain why.

        Returns the reasoning result if a DEPENDS_ON edge exists
        from source to target, otherwise None.
        """
        from ..concepts import RelationType

        for e in self.network.get_edges(source, "out"):
            if (
                e.relation == RelationType.DEPENDS_ON
                and (self.network._resolve(e.target) or e.target.lower()) == target
            ):
                return self.reasoner.explain_why(a, b)
        return None

    def compose_answer(
        self,
        question_text: str,
        topics: list[str],
        emotion: EmotionalState,
        question_type: str = "what_is",
    ) -> Thought | None:
        """Compose an intelligent answer to a question.

        Dynamically utilizes graph reasoning (relational deduction, process tracing,
        comparisons, and affordance synthesis) and concept knowledge.

        No domain knowledge is hardcoded here. Process detection, topic
        selection, and question routing all derive from the structure of
        the concept network — what Genesis has learned and remembered.
        """
        if not topics:
            return None

        lower_q = question_text.lower().strip()

        # ── Select the best pair of connected topics from the graph ──
        # Instead of hardcoding which words are "meta-words", we use
        # graph connectivity: the best topic pair is the one with the
        # most edges between them in the concept network. Question
        # structural words like "relationship" naturally drop out
        # because they have no edges to domain concepts like "ice".
        connected_pair = self._find_best_connected_pair(topics, question_text)

        # ── 1. Comparison Questions ──
        thought = self._answer_comparison(lower_q, connected_pair, emotion)
        if thought is not None:
            return thought

        # ── 2. Direct Relational Questions ──
        thought = self._answer_relational(lower_q, connected_pair, emotion)
        if thought is not None:
            return thought

        # ── 3. Process / Causal Questions ──
        thought = self._answer_process(lower_q, topics, emotion)
        if thought is not None:
            return thought

        # ── 4b. Why Questions (checked before affordance) ──
        thought = self._answer_why(lower_q, topics, emotion)
        if thought is not None:
            return thought

        # ── 4c. Causal / Evaluative Questions ──
        # Yes/no questions about relationships between concepts:
        # "does X cause Y", "does more X = more Y", "is X necessary for Y"
        thought = self._answer_causal_evaluative(lower_q, connected_pair, topics, emotion)
        if thought is not None:
            return thought

        # ── 4. Affordance / Functional Questions ──
        thought = self._answer_affordance(lower_q, topics, emotion)
        if thought is not None:
            return thought

        # ── 5. Standard Concept Knowledge Composition ──
        thought = self._answer_concept_knowledge(lower_q, topics, emotion)
        if thought is not None:
            return thought

        # ── 6. Fallback to reasoning question answering ──
        return self._answer_reasoning_fallback(topics, question_type, emotion)

    def _answer_comparison(
        self, lower_q: str, connected_pair, emotion: EmotionalState
    ) -> Thought | None:
        """Section 1: Comparison Questions ("difference between X and Y")."""
        # These keywords are English question structure, not domain knowledge.
        is_comparison = (
            "difference" in lower_q
            or "differ" in lower_q
            or "compare" in lower_q
            or "versus" in lower_q
            or " vs " in lower_q
            or "distinguish" in lower_q
        )
        if is_comparison and connected_pair:
            comp_result = self.reasoner.compare_concepts(connected_pair[0], connected_pair[1])
            if comp_result and comp_result.confidence > 0.4:
                return Thought(
                    content=comp_result.conclusion,
                    intent="inform",
                    emotion=emotion.label,
                    topics=list(connected_pair),
                    confidence=comp_result.confidence,
                    metadata={"knowledge": comp_result.knowledge},
                )
        return None

    def _answer_relational(
        self, lower_q: str, connected_pair, emotion: EmotionalState
    ) -> Thought | None:
        """Section 2: Direct Relational Questions ("how is X related to Y")."""
        is_relational = (
            "related" in lower_q
            or "relation" in lower_q
            or "connect" in lower_q
            or "relationship" in lower_q
            or "between" in lower_q
            or (lower_q.startswith("is ") and " a " in lower_q)
            or (lower_q.startswith("are ") and " and " in lower_q)
        )
        if is_relational and connected_pair:
            rel_result = self.reasoner.explain_relation(connected_pair[0], connected_pair[1])
            if rel_result and rel_result.confidence > 0.4:
                return Thought(
                    content=rel_result.conclusion,
                    intent="inform",
                    emotion=emotion.label,
                    topics=list(connected_pair),
                    confidence=rel_result.confidence,
                    metadata={"knowledge": rel_result.knowledge},
                )
        return None

    def _answer_process(
        self, lower_q: str, topics: list[str], emotion: EmotionalState
    ) -> Thought | None:
        """Section 3: Process / Causal Questions ("what happens when X freezes").

        Detection uses only grammatical keywords. "cause" alone is
        handled by the affordance handler (section 4) since "what does
        X cause?" asks about X's causal outputs, not a process.
        """
        is_process = "happen" in lower_q or "when" in lower_q or "process" in lower_q
        if not is_process:
            return None
        proc, subj = self._find_process_and_subject(topics, lower_q)
        if not proc:
            return None
        proc_result = self.reasoner.explain_process(proc, subject=subj)
        if proc_result and proc_result.confidence > 0.4:
            return Thought(
                content=proc_result.conclusion,
                intent="inform",
                emotion=emotion.label,
                topics=[t for t in topics if t != proc][:2] or topics,
                confidence=proc_result.confidence,
                metadata={"knowledge": proc_result.knowledge},
            )
        return None

    def _answer_why(
        self, lower_q: str, topics: list[str], emotion: EmotionalState
    ) -> Thought | None:
        """Section 4b: Why Questions ("why does X need Y").

        Traces the causal chain: X depends_on Y → Y enables Z → X is_a Z
        """
        is_why = lower_q.startswith("why")
        if is_why and len(topics) >= 2:
            # Find the dependency pair: which topic depends on which?
            why_result = self._try_explain_why(topics)
            if why_result and why_result.confidence > 0.4:
                return Thought(
                    content=why_result.conclusion,
                    intent="inform",
                    emotion=emotion.label,
                    topics=topics[:2],
                    confidence=why_result.confidence,
                    metadata={"knowledge": why_result.knowledge},
                )
        return None

    def _answer_causal_evaluative(
        self, lower_q: str, connected_pair, topics: list[str], emotion: EmotionalState
    ) -> Thought | None:
        """Section 4c: Causal / Evaluative Questions.

        Detects yes/no questions about relationships between two
        concepts — "does X cause Y", "does more X = more Y", "is X
        necessary for Y", "does X mean Y". These require cross-concept
        reasoning, not a definition of a single concept.

        Strategy:
        1. Try explain_why first — it traces the full causal chain
           (A depends_on B → B enables C → A is_a C), producing
           rich explanations like "cognition depends on memory
           because memory enables learning."
        2. Fall back to explain_relation — finds direct edges,
           multi-hop paths, or shared semantic hubs.
        3. Fall back to synthesis — finds shared connecting concepts.

        Tries multiple concept pairs: first the best-connected pair
        from the graph, then the first two explicit topics (topics
        that appear in the question text). This ensures the actual
        concepts the user asked about are tried, even if the graph
        connectivity ranks a different pair higher.

        The conclusion is returned as raw content (no knowledge
        metadata) so the language engine treats it as pre-composed
        text and applies voice, rather than re-composing from the
        same edges and producing repetition.
        """
        # Detect evaluative question patterns. These are English
        # question structures, not domain knowledge.
        is_evaluative = (
            ("= " in lower_q and " more " in lower_q)
            or ("equals" in lower_q and "more" in lower_q)
            or ("mean" in lower_q and "more" in lower_q)
            or ("necessary" in lower_q and "for" in lower_q)
            or ("sufficient" in lower_q)
            or (lower_q.startswith("does ") and " make " in lower_q)
            or (lower_q.startswith("does ") and " more " in lower_q)
            or (lower_q.startswith("is ") and " enough " in lower_q)
            or (lower_q.startswith("can ") and " without " in lower_q)
            or (lower_q.startswith("does ") and " lead to " in lower_q)
            or (lower_q.startswith("does ") and " require " in lower_q)
        )
        if not is_evaluative:
            return None

        # Build candidate pairs to try.
        candidate_pairs = self._build_candidate_pairs(connected_pair, topics, lower_q)
        if not candidate_pairs:
            return None

        for concept_a, concept_b in candidate_pairs:
            # 1. Try explain_why in both directions — traces the full
            #    causal chain (A depends_on B → B enables C → A is_a C).
            for a, b in [(concept_a, concept_b), (concept_b, concept_a)]:
                why_result = self.reasoner.explain_why(a, b)
                if why_result and why_result.confidence > 0.4:
                    # Collect qualification data: what else does the
                    # object enable / what else does the subject depend
                    # on? The vocabulary composes the connective
                    # phrasing from this structured data — the reasoning
                    # layer supplies the facts, not the words.
                    qualification = self._collect_qualification(a, b)
                    return Thought(
                        content=why_result.conclusion,
                        intent="inform",
                        emotion=emotion.label,
                        topics=[a, b],
                        confidence=why_result.confidence,
                        metadata=(
                            {"qualification": qualification}
                            if qualification else {}
                        ),
                    )

            # 2. Try explain_relation — finds direct edges, multi-hop
            #    paths, or shared semantic hubs between the two concepts.
            rel_result = self.reasoner.explain_relation(concept_a, concept_b)
            if rel_result and rel_result.confidence > 0.4:
                # Filter out conclusions with noisy targets (single
                # letters, code concepts, etc.)
                if not self._is_noisy_conclusion(rel_result.conclusion):
                    return Thought(
                        content=rel_result.conclusion,
                        intent="inform",
                        emotion=emotion.label,
                        topics=[concept_a, concept_b],
                        confidence=rel_result.confidence,
                    )

            # 3. Try synthesis — find shared connecting concepts
            syn_result = self.reasoner.synthesize([concept_a, concept_b])
            if syn_result and syn_result.confidence > 0.4:
                if not self._is_noisy_conclusion(syn_result.conclusion):
                    return Thought(
                        content=syn_result.conclusion,
                        intent="reflect",
                        emotion=emotion.label,
                        topics=[concept_a, concept_b],
                        confidence=syn_result.confidence,
                    )

        return None

    def _build_candidate_pairs(
        self, connected_pair, topics: list[str], lower_q: str,
    ) -> list[tuple[str, str]]:
        """Build candidate concept pairs for causal/evaluative questions.

        Starts with the best-connected pair from the graph, then adds
        the first two explicit content topics (topics that appear in
        the question text) as a fallback.
        """
        candidate_pairs: list[tuple[str, str]] = []
        if connected_pair:
            candidate_pairs.append(connected_pair)
        explicit = [t for t in topics if t.lower() in lower_q]
        _STRUCTURE_POS = {
            "pronoun", "article", "conjunction",
            "preposition", "determiner", "auxiliary",
        }
        content_explicit = []
        for t in explicit:
            concept = self.network.get_concept(t)
            if concept:
                pos = concept.properties.get("part_of_speech", "")
                if pos not in _STRUCTURE_POS:
                    content_explicit.append(t)
            else:
                content_explicit.append(t)
        if len(content_explicit) >= 2:
            pair = (content_explicit[0], content_explicit[1])
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)
        return candidate_pairs

    def _collect_qualification(
        self, subject: str, object_: str
    ) -> dict[str, Any] | None:
        """Collect qualification data showing what else matters beyond
        the direct relationship.

        After the reasoner concludes "cognition depends on memory
        because memory enables learning", this traverses the graph for
        context like "memory also enables other things" or "cognition
        also depends on other factors". This prevents the listener
        from inferring that the relationship is exclusive.

        Returns structured semantic data — ``{"kind": "depends" |
        "enables", "subject", "object", "others": [...]}`` — for the
        vocabulary to compose into speech. The connective phrasing
        belongs to the language engine; this method only decides
        *which* relations exist. Derived entirely from graph edges —
        no hardcoded domain knowledge.
        """
        from ..concepts import RelationType

        obj_id = self.network._resolve(object_) or object_.lower()
        subj_id = self.network._resolve(subject) or subject.lower()

        # What else does the object enable? (besides things related
        # to the subject)
        other_enabled = []
        subj_neighbors = set()
        for e in self.network.get_edges(subj_id, "both"):
            subj_neighbors.add(
                self.network._resolve(e.target) or e.target.lower()
                if e.source == subj_id
                else self.network._resolve(e.source) or e.source.lower()
            )
        for e in self.network.get_edges(obj_id, "out"):
            if e.relation == RelationType.ENABLES:
                target_id = self.network._resolve(e.target) or e.target.lower()
                if target_id not in subj_neighbors and target_id != subj_id:
                    if not self._is_noisy_target(e.target):
                        other_enabled.append(e.target)

        # What else does the subject depend on? (besides the object)
        other_deps = []
        for e in self.network.get_edges(subj_id, "out"):
            if e.relation == RelationType.DEPENDS_ON:
                target_id = self.network._resolve(e.target) or e.target.lower()
                if target_id != obj_id:
                    if not self._is_noisy_target(e.target):
                        other_deps.append(e.target)

        # Return the qualification as structured data — the vocabulary
        # composes the connective phrasing ("but X also depends on Y").
        if other_deps:
            return {
                "kind": "depends",
                "subject": subject,
                "object": object_,
                "others": other_deps[:3],
            }
        if other_enabled:
            return {
                "kind": "enables",
                "subject": subject,
                "object": object_,
                "others": other_enabled[:3],
            }
        return None

    def _answer_affordance(
        self, lower_q: str, topics: list[str], emotion: EmotionalState
    ) -> Thought | None:
        """Section 4: Affordance / Functional Questions ("what does X enable").

        "Why" questions are handled separately (section 4b) because
        they require causal chain reasoning, not just listing
        affordances. "need" alone could be either, so we check for
        "why" first.
        """
        is_why = lower_q.startswith("why")
        if not is_why:
            is_affordance = (
                "enable" in lower_q
                or "used for" in lower_q
                or "function" in lower_q
                or "purpose" in lower_q
                or "need" in lower_q
                or "role" in lower_q
                or "do with" in lower_q
                or "allow" in lower_q
                or "cause" in lower_q
            )
        else:
            is_affordance = False

        if is_affordance and topics:
            # Find the topic with the most affordance edges (ENABLES/CREATES)
            aff_topic = self._find_most_functional_topic(topics) or topics[0]
            aff_result = self.reasoner.explain_affordances(aff_topic)
            if aff_result and aff_result.confidence > 0.4:
                return Thought(
                    content=aff_result.conclusion,
                    intent="inform",
                    emotion=emotion.label,
                    topics=[aff_topic],
                    confidence=aff_result.confidence,
                    metadata={"knowledge": aff_result.knowledge},
                )
        return None

    def _answer_concept_knowledge(
        self, lower_q: str, topics: list[str], emotion: EmotionalState
    ) -> Thought | None:
        """Section 5: Standard Concept Knowledge Composition.

        Use focused=True to keep answers tight — no associative
        tangents, discovery phrases, or follow-up questions.
        Prefer topics that appear in the question text (explicitly
        mentioned by the user) over semantically inferred ones.

        Content words are prioritized over question structure words.
        "What is rust?" should answer about "rust", not compose about
        "is" and "what". Structure words (question words, copulas,
        articles) are identified by their part of speech in the concept
        network — pronouns, articles, conjunctions, and prepositions
        are structure words, not content words.
        """
        explicit_topics = [t for t in topics if t.lower() in lower_q]
        answer_topics = explicit_topics if explicit_topics else list(topics)

        # Prioritize content words over question structure words.
        # A structure word is one whose part_of_speech in the concept
        # network is a function word type (pronoun, article, conjunction,
        # preposition, determiner, auxiliary).
        # If ALL topics are structure words (e.g., "What is the?"),
        # don't sort — the user is asking about that specific word.
        _STRUCTURE_POS = {
            "pronoun", "article", "conjunction",
            "preposition", "determiner", "auxiliary",
        }

        def _is_structure_word(topic: str) -> bool:
            """Check if a topic is a structural word (preposition, conjunction, etc.)."""
            concept = self.network.get_concept(topic)
            if not concept:
                return False
            pos = concept.properties.get("part_of_speech", "")
            return pos in _STRUCTURE_POS

        has_content_word = any(not _is_structure_word(t) for t in answer_topics)
        if has_content_word:
            answer_topics.sort(key=lambda t: _is_structure_word(t))

        thoughts = []
        for topic in answer_topics[:2]:
            thought = self.compose_about(topic, emotion, depth=3, focused=True)
            if thought and thought.confidence > 0.35:
                thoughts.append(thought)

        if thoughts:
            if len(thoughts) == 1:
                return thoughts[0]
            return self._synthesize_thoughts(thoughts, topics, emotion)
        return None

    def _answer_reasoning_fallback(
        self, topics: list[str], question_type: str, emotion: EmotionalState
    ) -> Thought | None:
        """Section 6: Fallback to reasoning question answering."""
        for topic in topics:
            result = self.reasoner.answer_question(topic, question_type)
            if result and result.confidence > 0.4:
                content = self._express_reasoning_result(result, emotion)
                return Thought(
                    content=content,
                    intent="inform",
                    emotion=emotion.label,
                    topics=topics,
                    confidence=result.confidence,
                )
        return None

    def compose_reflection(
        self,
        topic: str,
        emotion: EmotionalState,
    ) -> Thought | None:
        """Compose a reflective thought about a topic.

        This is for philosophical/existential questions. Instead of
        hardcoded philosophy, she reflects on what she knows and
        what she doesn't know.

        The raw knowledge (edges, definition, reasoning) is passed as
        metadata so the language engine can compose its own text from
        the graph data. The content field is a short fallback only.
        """
        concept = self.network.get_concept(topic)
        knowledge = self._gather_knowledge(topic, depth=3) if concept else []

        # Reason about it
        reasoning_results = self.reasoner.reason_about(topic) if concept else []

        # Extract definition from knowledge
        definition = None
        rel_facts: list[tuple[str, str, float]] = []
        for rel, target, weight in knowledge:
            if rel == "is defined as":
                definition = target
            else:
                rel_facts.append((rel, target, weight))

        # Collect reasoning conclusions
        reasoning_conclusions: list[str] = []
        for r in reasoning_results[:2]:
            reasoning_conclusions.append(r.conclusion)

        # Confidence
        has_known = bool(rel_facts) or any(
            r.reasoning_type != ReasoningType.HYPOTHESIS for r in reasoning_results
        )
        has_unknown = any(
            r.reasoning_type == ReasoningType.HYPOTHESIS for r in reasoning_results
        )
        if has_known and has_unknown:
            confidence = 0.5
        elif has_known:
            confidence = 0.6
        elif has_unknown:
            confidence = 0.4
        else:
            confidence = 0.3

        # Compose from her knowledge — if she can't weave anything
        # from what she knows, she stays silent rather than reciting
        # a pre-written reflection template.
        content = self._weave_knowledge(
            topic, knowledge, reasoning_results, emotion,
            focused=False, mode="full",
        )

        if not content:
            return None

        return Thought(
            content=content,
            intent="reflect",
            emotion=emotion.label,
            topics=[topic],
            confidence=confidence,
            self_reflection=True,
            metadata={
                "knowledge": rel_facts,
                "definition": definition,
                "topic": topic,
                "reasoning": reasoning_conclusions,
            },
        )

    def compose_novel_connection(
        self,
        emotion: EmotionalState,
    ) -> Thought | None:
        """Compose a thought about a novel connection she's discovered.

        This is for moments when reasoning produces a hypothesis —
        she can express it as a thought, not just store it.
        """
        # Find concepts with high activation
        active = [
            (c_id, c.activation)
            for c_id, c in list(self.network._concepts.items())
            if (c.activation or 0.0) > 0.3
        ]
        if len(active) < 2:
            return None

        # Sort by activation
        active.sort(key=lambda x: x[1], reverse=True)

        # Try to find a hypothesis between the top concepts
        for i, (c1, _) in enumerate(active[:5]):
            for c2, _ in active[i + 1 : 5]:
                thought = self._find_hypothesis_between(c1, c2, emotion)
                if thought:
                    return thought

        return None

    def _find_hypothesis_between(
        self, c1: str, c2: str, emotion: EmotionalState
    ) -> Thought | None:
        """Find a hypothesis connecting two concepts and express it as a thought."""
        results = self.reasoner.reason_about(c1)
        for r in results:
            if (
                r.reasoning_type == ReasoningType.HYPOTHESIS
                and c2.lower() in r.conclusion.lower()
            ):
                return Thought(
                    content=r.conclusion,
                    intent="reflect",
                    emotion=emotion.label,
                    topics=[c1, c2],
                    confidence=r.confidence,
                    self_reflection=True,
                    metadata={
                        "hypothesis": True,
                        "reasoning": [r.conclusion],
                        "knowledge": r.knowledge,
                    },
                )
        return None

    # ─── Internal methods ──────────────────────────────────

    @staticmethod
    def _display_name(concept_id: str, *, lower: bool = False) -> str:
        """Convert a concept ID to a human-readable display name.

        Code concepts have IDs like 'python:cognition.CognitionEngine'
        or 'rust:daemon.tick'. These should display as 'cognition engine'
        or 'tick', not the raw ID. Dictionary concepts ('cognition')
        display as-is.

        By default the result is lowercased, so composed sentences flow
        naturally; the voice/GrammarEngine will capitalize the first word
        of the final utterance. Call with ``lower=False`` if you need a
        title-cased display for a proper noun.
        """
        # Strip language prefix
        name = concept_id
        for prefix in ("python:", "rust:", "identity:", "code:"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break

        # If there's a module path (e.g., 'cognition.CognitionEngine'),
        # take the last component (the class/function name)
        if "." in name:
            name = name.split(".")[-1]

        # Convert snake_case to readable
        name = name.replace("_", " ")

        # Strip disambiguation suffixes
        name = name.split("#")[0]

        # Lowercase by default; voice will handle first-word capitalization
        if lower and name:
            name = name.lower()

        return name

    @staticmethod
    def _is_noisy_conclusion(conclusion: str) -> bool:
        """Return True if a reasoning conclusion mentions internal/code concepts."""
        if not conclusion:
            return True
        # Look for code/namespace markers anywhere in the conclusion text
        if re.search(r"\b(python|rust|code):[^\s,;]+", conclusion):
            return True
        if re.search(r"\b[_a-zA-Z]+\.[a-zA-Z_]+\b", conclusion):
            return True
        # Filter conclusions containing single-letter "concepts" —
        # these are usually code fragments or disambiguation entries
        # that leaked from the concept network (e.g., "cognition
        # connects to A"). A single uppercase letter as a standalone
        # word is never a meaningful concept in natural speech.
        if re.search(r"\b[A-Z]\b", conclusion):
            return True
        return False

    @staticmethod
    def _is_noisy_target(target: str) -> bool:
        """Return True if a concept target should not appear in natural speech.

        Code concepts, internal symbols, and disambiguated technical
        fragments don't make good explanation targets.
        """
        if not target:
            return True
        # Direct code/namespace markers
        if target.startswith(("python:", "rust:", "code:")):
            return True
        if "." in target or ":" in target:
            return True
        if target.startswith("_") or "__" in target:
            return True
        if "self." in target or "cls." in target:
            return True
        # Single-letter targets are never meaningful concepts — they
        # are code fragments, disambiguation entries, or variables
        # that leaked from the concept network.
        if len(target) == 1:
            return True
        # Fragments from user sentences like "to choose" or "the sun" are
        # not real semantic concepts.
        if re.match(r"^(?:to|the|a|an)\s", target):
            return True
        # Code-language and internal category terms that leak from code
        # learning and don’t belong in ordinary explanations.
        base = target.split("#")[0].lower()
        if base in ("python", "rust", "code", "class"):
            return True
        # Long lowercase identifiers with multiple underscores are usually
        # internal symbols; short ones like "free_will" are fine.
        if "_" in target and len(target) > 20:
            return True
        # Internal CamelCase UI fragments
        if re.match(r"^[A-Z][a-z]+([A-Z][a-z]+)+$", target):
            return True
        return False

    def _gather_knowledge(self, concept_name: str, depth: int) -> list[tuple[str, str, float]]:
        """Gather what Genesis knows about a concept.

        Returns a list of (relation_description, target, weight).
        Includes the definition as a special "is defined as" entry.

        Only uses OUTGOING edges for IS_A relationships — we don't want
        to list every specific type of tree (quandong, silver tree, etc.)
        as what a tree "is". Incoming IS_A edges (hyponyms) are excluded.
        """
        knowledge: list[tuple[str, str, float]] = []

        # First, check if the concept has a definition stored
        concept = self.network.get_concept(concept_name)
        if concept:
            definition = concept.properties.get("definition")
            if definition and definition != "NO DEF":
                knowledge.append(("is defined as", definition, 1.0))

        # Gather OUTGOING edges only — these are what this concept IS
        # (e.g., tree IS_A woody_plant, not quandong IS_A tree)
        # Skip BRIDGES edges — they're structural (cortical association
        # fibers), not semantic relationships that should appear in speech
        outgoing = self.network.get_edges(concept_name, direction="out")
        for edge in outgoing:
            if edge.relation == RelationType.BRIDGES:
                continue
            if not _is_reliable_fact_edge(edge):
                continue
            if self._is_noisy_target(edge.target):
                continue
            rel_desc = edge.relation.value.replace("_", " ")
            knowledge.append((rel_desc, edge.target, edge.weight))

        # Also gather incoming edges for non-IS_A relations
        # (e.g., "alice CREATES genesis" is incoming but meaningful)
        has_outgoing_is_a = any(e.relation == RelationType.IS_A for e in outgoing)
        incoming = self.network.get_edges(concept_name, direction="in")
        for edge in incoming:
            if edge.relation == RelationType.BRIDGES:
                continue
            if not _is_reliable_fact_edge(edge):
                continue
            if edge.relation == RelationType.IS_A:
                # Incoming IS_A edges mean "X is a <this concept>".
                # This is how personal facts are stored:
                #   "Carol" --[is_a]--> "my son"  (Carol is my son)
                #   "name" --[is_a]--> "alice"    (name is Alice)
                # When asked about "my son", we want to surface "Carol".
                # When asked about "name", we want to surface "alice".
                # Only include these if we don't already have outgoing
                # IS_A edges (to avoid listing every hyponym of a category).
                if has_outgoing_is_a:
                    continue
                if self._is_noisy_target(edge.source):
                    continue
                knowledge.append(("is", edge.source, edge.weight))
                continue
            if edge.relation == RelationType.SIMILAR_TO:
                continue
            if self._is_noisy_target(edge.source):
                continue
            # Incoming non-IS_A edges are meaningful (e.g., X creates this)
            # Use natural voice for incoming edges. Instead of
            # awkward passives ("is depended on by"), use active
            # constructions that read naturally.
            rel_value = edge.relation.value
            passive_map = {
                "depends_on": "is needed by",
                "creates": "is created by",
                "enables": "is enabled by",
                "causes": "is caused by",
                "leads_to": "is led to by",
                "part_of": "contains",
                "related_to": "is related to",
                "instance_of": "has instance",
                "emerges_from": "produces",
                "harms": "is harmed by",
                "opposite_of": "is opposite of",
                "contradicts": "is contradicted by",
                "has_property": "is a property of",
            }
            rel_desc = passive_map.get(rel_value, f"is {rel_value.replace('_', ' ')} by")
            knowledge.append((rel_desc, edge.source, edge.weight))

        # Sort by weight (most confident first), but keep definition at top
        knowledge.sort(key=lambda x: x[2], reverse=True)

        return knowledge[: depth + 4]  # limit to depth+4 facts

    def _extract_definition(
        self, knowledge: list[tuple[str, str, float]]
    ) -> str | None:
        """Extract the definition from a knowledge list, if present."""
        for rel, target, _weight in knowledge:
            if rel == "is defined as" and target and target != "NO DEF":
                return target.rstrip(".")
        return None

    def _build_knowledge_metadata(
        self,
        concept_name: str,
        knowledge: list[tuple[str, str, float]],
        reasoning: list[str],
        confidence: float,
        definition: str | None,
    ) -> dict[str, Any]:
        """Build metadata for the language engine to compose from."""
        # Include the definition as a knowledge tuple so the vocabulary
        # composer can place it naturally.
        knowledge_for_vocab: list[tuple[str, str, float]] = list(knowledge)
        # Drop reasoning conclusions that mention internal code concepts;
        # these leak from code analysis and don't belong in natural answers.
        clean_reasoning = [c for c in reasoning if not self._is_noisy_conclusion(c)]
        return {
            "topic": concept_name,
            "knowledge": knowledge_for_vocab,
            "definition": definition,
            "reasoning": clean_reasoning,
            "confidence": confidence,
        }

    def _weave_knowledge(
        self,
        concept_name: str,
        knowledge: list[tuple[str, str, float]],
        reasoning_results: list[ReasoningResult],
        emotion: EmotionalState,
        focused: bool = False,
        mode: str = "full",
    ) -> str:
        """Weave knowledge and reasoning into a coherent, natural statement.

        Instead of listing raw relationship triples like "beauty contradicts
        just about how things look", this composes natural English sentences
        that express the same knowledge in a way a person would actually speak.

        If the concept has a definition, that's stated first — it's the most
        direct way to express understanding. Then relationships and reasoning
        add depth.

        When ``mode`` is "definition", only the definition is returned —
        no relationship facts, no reasoning, no follow-up. This is useful
        for contexts like empathy where the definition is all that's needed.
        """
        parts: list[str] = []

        # Separate the definition from relationship facts
        definition, rel_facts_raw = _split_definition(knowledge)

        # Only quote the raw definition when explicitly asked for it.
        # In full mode, she should express what she knows in her own
        # words from relationship facts and reasoning instead.
        display = self._display_name(concept_name, lower=True)
        if definition and mode == "definition":
            parts.append(f"{display} is {definition.rstrip('.')}.")

        # In "definition" mode, stop here — no relationship facts,
        # no reasoning, no follow-up. The definition alone is the
        # content. This is used for empathy and other contexts where
        # relationship facts would be noise.
        if mode == "definition":
            if not parts:
                # No definition available — return empty string so
                # the caller knows composition failed and can stay
                # silent rather than reciting a template.
                return ""
            return " ".join(parts)

        # In full mode, use every relationship fact Genesis has learned
        # (including is_a and similar_to) to compose her own answer. The
        # raw dictionary definition is intentionally not quoted above.
        filtered_facts = rel_facts_raw

        # Then add relationship facts
        self._weave_relationship_facts(parts, concept_name, filtered_facts)

        # If she has a definition but no relationship facts to weave,
        # state the definition rather than staying silent. A concept
        # with a real definition but no edges (e.g. "happy") is genuine
        # knowledge — discarding it let a junk neighbour (a bare verb
        # attached to a hub) win the answer instead.
        if not parts and definition:
            parts.append(f"{display} is {definition.rstrip('.')}.")

        # Filter reasoning that would leak code concepts into speech.
        clean_reasoning = [
            r for r in reasoning_results
            if not self._is_noisy_conclusion(r.conclusion)
        ]

        # Add reasoning conclusions. The _is_noisy_conclusion filter
        # removes transitive IS_A chains through WordNet multi-sense
        # entries that produce nonsense like "tree is_a switch (through
        # chase, cut)". In focused mode, _weave_reasoning selects only
        # the top result by confidence (not random) and uses direct
        # language — this adds the key reasoning conclusion without
        # tangential noise.
        #
        # Deduplicate: skip reasoning conclusions that say the same
        # thing as the relationship facts already in parts. Without
        # this, a DEPENDS_ON edge and a CAUSAL reasoning conclusion
        # about the same dependency produce "cognition depends on
        # memory. cognition depends on memory."
        if clean_reasoning:
            deduped = self._dedup_reasoning(parts, clean_reasoning)
            if deduped:
                self._weave_reasoning(parts, deduped, focused=focused)

        # Use the latent space to find related concepts she hasn't been
        # explicitly taught about — this is where generalization happens.
        # She discovers connections through embedding proximity, not just
        # through explicit graph edges.
        # Skip in focused mode — associative discoveries add noise to
        # direct question answers.
        if not focused and self.embeddings and self.embeddings.has_embeddings:
            self._weave_latent_discovery(parts, concept_name, emotion)

        # Add a follow-up question composed from genuine curiosity
        # This makes her more interactive — she doesn't just answer, she engages
        # Skip in focused mode — follow-up questions add noise to direct
        # question answers.
        if not focused:
            self._weave_followup(parts, concept_name, emotion)

        if not parts:
            # Could not compose anything from her knowledge — return
            # empty string so the caller knows to stay silent rather
            # than reciting a pre-written "I can't articulate" template.
            return ""

        return " ".join(parts)

    def _weave_relationship_facts(
        self, parts: list[str], concept_name: str, filtered_facts: list[tuple[str, str, float]]
    ) -> None:
        """Compose natural sentences from the relationship facts and append them.

        Filters out tautological, meaningless, and repetitive edges before
        composing. Uses pronouns for subsequent clauses so she doesn't
        repeat the subject in every clause. Limits ``related_to`` edges
        to at most 1 per thought to prevent synonym spam.
        """
        if not filtered_facts:
            return

        # ── Filter meaningless facts ──
        clean_facts: list[tuple[str, str, float]] = []
        seen_targets: set[str] = set()
        seen_pairs: set[frozenset[str]] = set()
        related_to_count = 0
        for rel, target, weight in filtered_facts:
            # Skip tautological edges (subject == target)
            if self._is_tautological(concept_name, rel, target):
                continue
            # Skip meaningless targets (relation verbs as concepts, etc.)
            if self._is_meaningless_target(target, rel):
                continue
            # Skip duplicate pairs (same concept+target, different direction)
            pair = frozenset({concept_name.lower(), target.lower()})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            # Skip duplicate targets — saying "X relates to A, X connects
            # to A" is just synonym variation of the same fact
            target_lower = target.lower().split("#")[0]
            if target_lower in seen_targets:
                continue
            # Limit related_to to 1 per thought — multiple related_to
            # edges produce "X relates to A, X connects to B, X has to
            # do with C" which is synonym spam, not understanding.
            if rel.replace(" ", "_") == "related_to":
                if related_to_count >= 1:
                    continue
                related_to_count += 1
            seen_targets.add(target_lower)
            clean_facts.append((rel, target, weight))

        if not clean_facts:
            return

        # Shuffle so each thought draws different relationships
        shuffled = list(clean_facts)
        self._rng.shuffle(shuffled)
        direct = shuffled[:3]  # max 3 facts for a concise thought

        # ── Compose facts with pronoun substitution ──
        # First fact uses the full subject; subsequent facts use a
        # pronoun to avoid repeating "Vision relates to color, vision
        # gives rise to edge, vision connects to light".
        facts: list[str] = []
        for i, (rel, target, _weight) in enumerate(direct):
            if i == 0:
                fact = self._compose_natural_fact(concept_name, rel, target)
            else:
                fact = self._compose_natural_fact(concept_name, rel, target, use_pronoun=True)
            if fact:
                facts.append(fact)

        if facts:
            if len(facts) == 1:
                parts.append(f"{facts[0].capitalize()}.")
            elif len(facts) == 2:
                parts.append(f"{facts[0].capitalize()}, and {facts[1]}.")
            else:
                parts.append(f"{facts[0].capitalize()}, {facts[1]}, and {facts[2]}.")

    def _is_tautological(self, subject: str, rel: str, target: str) -> bool:
        """Return True if a fact is tautological (says nothing).

        Examples:
        - "kind is a kind of kind" (subject == target)
        - "related is related to related" (subject is the relation verb)
        - "is is a is" (both are relation fragments)
        """
        subj_lower = subject.lower().split("#")[0].strip()
        targ_lower = target.lower().split("#")[0].strip()
        # Subject == target
        if subj_lower == targ_lower:
            return True
        # Subject or target is the relation verb itself
        rel_key = rel.replace(" ", "_")
        if subj_lower == rel_key or targ_lower == rel_key:
            return True
        # Subject or target is a bare relation verb (without "to"/"of")
        relation_verbs = {
            "related", "relates", "connects", "connected",
            "similar", "opposite", "contradicts",
            "causes", "creates", "enables", "depends", "emerges",
            "produces", "leads", "harms",
        }
        if subj_lower in relation_verbs or targ_lower in relation_verbs:
            return True
        return False

    def _is_meaningless_target(self, target: str, rel: str) -> bool:
        """Return True if a target concept is too vague to be meaningful.

        These are concepts that exist in the graph but don't carry
        semantic content worth expressing — function words, relation
        fragments, and single characters.
        """
        if not target:
            return True
        targ_lower = target.lower().split("#")[0].strip()
        # Single-character targets
        if len(targ_lower) <= 1:
            return True
        # Targets that are just relation verb phrases
        if targ_lower in {
            "related to", "connects to", "has to do with",
            "part of", "is a", "is an", "is the",
            "similar to", "opposite of",
            "related", "relates", "connects", "connected",
            "similar", "opposite", "instance",
        }:
            return True
        # Pure function words
        if targ_lower in {"is", "are", "was", "were", "be", "been",
                          "yes", "no", "true", "false",
                          "thing", "something", "anything", "everything"}:
            return True
        return False

    @staticmethod
    def _dedup_reasoning(
        parts: list[str], reasoning_results: list[ReasoningResult]
    ) -> list[ReasoningResult]:
        """Filter out reasoning conclusions that duplicate what's
        already in parts.

        Compares the key content words (nouns, verbs) of each
        reasoning conclusion against the text already in parts.
        If all content words of the conclusion already appear in
        parts, the conclusion is redundant and is filtered out.
        """
        if not parts:
            return reasoning_results
        # Build a set of content words already in parts
        existing_text = " ".join(parts).lower()
        existing_words = set(existing_text.split())
        # Remove common stopwords — they don't carry semantic content
        _STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "of", "to", "in", "on", "at", "by", "for", "with", "from",
            "as", "and", "or", "but", "not", "no", "if", "then", "that",
            "this", "these", "those", "it", "its", "i", "you", "he",
            "she", "they", "we", "me", "him", "her", "them", "us",
            "my", "your", "his", "their", "our", "who", "what", "which",
            "how", "why", "when", "where", "through", "also", "because",
        }
        existing_content = existing_words - _STOPWORDS

        filtered: list[ReasoningResult] = []
        for r in reasoning_results:
            # Strip parenthetical asides for comparison
            conclusion_text = re.sub(r"\s*\([^)]*\)", "", r.conclusion)
            conclusion_text = conclusion_text.replace("_", " ").lower()
            conclusion_words = set(conclusion_text.split()) - _STOPWORDS
            if not conclusion_words:
                continue
            # If all content words of the conclusion already appear in
            # existing text, it's a duplicate
            overlap = conclusion_words & existing_content
            if conclusion_words and len(overlap) / len(conclusion_words) > 0.7:
                continue
            filtered.append(r)
        return filtered

    def _weave_reasoning(
        self,
        parts: list[str],
        reasoning_results: list[ReasoningResult],
        focused: bool = False,
    ) -> None:
        """Append reasoning conclusions as natural prose.

        In focused mode (direct question answering), select only the
        top result by confidence — the reasoning engine already sorts
        by confidence, so the first result is the strongest conclusion.
        Use direct language without hedging prefixes so the reasoning
        reads as a conclusion, not a tangent.

        HYPOTHESIS results are speculative inferences ("sleep might
        be part of hippocampus" through memory). In focused mode, these
        are excluded — the user asked for facts, not speculation. In
        non-focused mode, they're included with explicit uncertainty
        markers ("suspect ...", "perhaps ...") so they're not mistaken
        for established facts.

        In non-focused mode (free conversation, inner life), randomly
        select 1-2 results for variety and use natural prose prefixes
        that signal the reasoning type.
        """
        if focused:
            # Exclude hypotheses in focused mode — the user wants
            # facts, not transitive speculation. Only include
            # DEDUCTIVE, CAUSAL, and ANALOGICAL results.
            factual = [
                r for r in reasoning_results
                if r.reasoning_type != ReasoningType.HYPOTHESIS
            ]
            if not factual:
                return
            selected = factual[:1]
        else:
            n_results = min(len(reasoning_results), self._rng.randint(1, 2))
            selected = self._rng.sample(reasoning_results, n_results)

        reasoned = []
        for r in selected:
            # Strip parenthetical asides and noisy prefixes from conclusions
            cleaned = re.sub(r"\s*\([^)]*\)", "", r.conclusion)
            cleaned = cleaned.replace("_", " ")
            cleaned = cleaned.lower().rstrip(".")
            if not cleaned:
                continue
            if focused:
                # Direct language — state the conclusion plainly.
                # Only factual reasoning reaches here (hypotheses
                # are filtered above), so no uncertainty marker
                # needs to be stripped.
                cleaned = re.sub(r"^(might be |might )", "", cleaned)
                if not cleaned:
                    continue
                reasoned.append(cleaned)
            elif r.reasoning_type == ReasoningType.DEDUCTIVE:
                cleaned = re.sub(r"^(might be |might )", "", cleaned)
                if cleaned:
                    reasoned.append(f"it follows that {cleaned}")
            elif r.reasoning_type == ReasoningType.CAUSAL:
                cleaned = re.sub(r"^(might be |might )", "", cleaned)
                if cleaned:
                    reasoned.append(f"this leads to {cleaned}")
            elif r.reasoning_type == ReasoningType.ANALOGICAL:
                cleaned = re.sub(r"^(might be |might )", "", cleaned)
                if cleaned:
                    reasoned.append(f"by analogy, {cleaned}")
            elif r.reasoning_type == ReasoningType.HYPOTHESIS:
                # Preserve the "might" — hypotheses are speculative
                # and must not be presented as established fact.
                reasoned.append(f"suspect {cleaned}")

        if reasoned:
            parts.append(self._join_parts(reasoned))

    def _weave_latent_discovery(
        self, parts: list[str], concept_name: str, emotion: EmotionalState
    ) -> None:
        """Append a latent-space discovery phrase for a related, unconnected concept.

        Uses two discovery mechanisms:
        1. Direct similarity: find concepts close in embedding space
        2. Compositional discovery (HRR): compose the concept with its
           most active neighbor and search for concepts similar to the
           composition. This finds concepts that relate to the
           *combination* — deeper connections that neither concept
           alone would surface. For example, composing "memory" with
           "sleep" might discover "consolidation".
        """
        if not self.embeddings:
            return

        # 1. Direct similarity discovery
        similar = self.embeddings.find_similar_concepts(concept_name, k=2, threshold=0.6)
        discovered_related = None
        if similar:
            related, _score = similar[0]
            # Only mention if there's no existing edge (avoid redundancy)
            existing = self.network.get_edges(concept_name, direction="out")
            has_edge = any(e.target == related for e in existing)
            if not has_edge and related != concept_name:
                discovered_related = related
                parts.append(
                    self._compose_discovery_phrase(concept_name, related, emotion)
                )

        # 2. Compositional discovery — find concepts that emerge from
        #    the combination of this concept and a close neighbor.
        #    This is gated by creativity (high creativity enables deeper
        #    associative leaps) and only fires occasionally to avoid noise.
        if (
            emotion.creativity > 0.4
            and self._rng.random() < 0.3
            and discovered_related is not None
        ):
            self._weave_compositional_discovery(
                parts, concept_name, discovered_related, emotion
            )

    def _weave_compositional_discovery(
        self,
        parts: list[str],
        concept_a: str,
        concept_b: str,
        emotion: EmotionalState,
    ) -> None:
        """Discover concepts related to the composition of two concepts.

        Uses HRR circular convolution to bind the two concepts into a
        single composed vector, then searches the embedding space for
        concepts similar to the composition. This surfaces concepts
        that relate to the *relationship* between the two concepts,
        not to either one alone.
        """
        if not self.embeddings:
            return

        composed = self.embeddings.compose_concepts(concept_a, concept_b, relation="modifier")
        if composed is None:
            return

        # Search for concepts similar to the composed vector
        results = self.embeddings.find_similar_to_vector(composed, k=3, threshold=0.55)
        if not results:
            return

        for discovered, _score in results:
            # Skip the two source concepts and any concept already
            # connected to either source
            if discovered in (concept_a, concept_b):
                continue
            edges_a = self.network.get_edges(concept_a, direction="out")
            edges_b = self.network.get_edges(concept_b, direction="out")
            already_known = (
                any(e.target == discovered for e in edges_a)
                or any(e.target == discovered for e in edges_b)
            )
            if already_known:
                continue

            # Express the compositional discovery — she found something
            # by combining two concepts, not just by looking at one.
            # Use a bare semantic fragment (not a finished sentence) so
            # the language engine composes the actual words. The
            # emotional tone is conveyed by the emotion state, not by
            # template selection.
            display_a = self._display_name(concept_a, lower=True)
            display_b = self._display_name(concept_b, lower=True)
            display_d = self._display_name(discovered, lower=True)
            if emotion.creativity > 0.6:
                phrase = f"latent link: {display_a} + {display_b} → {display_d}"
            else:
                phrase = f"{display_a} and {display_b} suggest {display_d}"
            parts.append(phrase)
            break  # One compositional discovery is enough per thought

    def _weave_followup(
        self, parts: list[str], concept_name: str, emotion: EmotionalState
    ) -> None:
        """Append a follow-up question or personal reflection (creativity-gated)."""
        if emotion.creativity > 0.3 and self._rng.random() < 0.5:
            self._weave_followup_question(parts, concept_name, emotion)
        elif emotion.creativity > 0.6 and self._rng.random() < 0.3:
            # Express a personal connection — composed from concept network
            reflection = self._compose_personal_reflection(concept_name, emotion)
            if reflection:
                parts.append(reflection)

    def _weave_followup_question(
        self, parts: list[str], concept_name: str, emotion: EmotionalState
    ) -> None:
        """Append a follow-up question from the question composer."""
        q_data = self.question_composer.compose_follow_up(concept_name, emotion)
        if not q_data:
            return
        # Compose the question text through the GenerativeEngine
        if hasattr(self, '_language') and self._language:
            q_text = self._language.compose_question(q_data, emotion)
            if q_text:
                parts.append(q_text)
            return
        # Fallback: use gap_detail or target_concept
        text = q_data.get("gap_detail", "") or q_data.get("target_concept", "")
        if text:
            parts.append(text)

    def _compose_natural_fact(
        self, subject: str, relation: str, obj: str, use_pronoun: bool = False
    ) -> str:
        """Compose a single relationship as a natural English sentence.

        Instead of raw triples like "beauty contradicts just about how things
        look", this produces natural phrasing like "beauty isn't just about
        how things look" or "love comes from care".

        When ``use_pronoun`` is True, the subject is replaced with a
        pronoun ("it" / "she") to avoid repeating the subject in every
        clause of a multi-fact sentence.

        Code concept IDs (python:CognitionEngine, rust:daemon.tick) are
        converted to human-readable display names.
        """
        # Convert code concept IDs to display names
        subject = self._display_name(subject, lower=True)
        obj = self._display_name(obj, lower=True)

        # Clean up the object — strip leading articles, trim
        obj = obj.strip()
        for article in ("the ", "a ", "an "):
            if obj.startswith(article):
                obj = obj[len(article) :]
                break

        # Determine the subject to use in the sentence
        if use_pronoun:
            is_person = subject.lower() in {"genesis", "she", "her", "i"}
            subject_phrase = "she" if is_person else "it"
        else:
            subject_phrase = subject

        # Map relation types to natural phrasings
        # relation is the RelationType.value string, e.g. "emerges_from",
        # "is_a", "enables", "causes", "contradicts", etc.

        # Try graph-based verb lookup first — construct phrasings from
        # relation verbs retrieved via EXPRESSES edges from hubs.
        from ..concepts import RelationType as _RT
        rt_match = None
        for rt in _RT:
            if rt.value == relation or rt.value == relation.replace(" ", "_"):
                rt_match = rt
                break
        if rt_match is not None:
            # Compose from relation VERBS (building blocks), not from
            # pre-written first-person sentences. The phrase seeds in
            # ConceptNetwork (_RELATION_PHRASE_SEEDS) are input data for
            # the language engine's render step — returning them directly
            # here would bypass that engine and recite fixed sentences,
            # violating the no-hardcoding rule. So this method returns a
            # semantic fragment (subject + verb + object); first-person
            # voice is recovered downstream when the Thought is rendered
            # with self_reflection=True.
            graph_verbs = self.network.find_relation_verbs(rt_match)
            if graph_verbs:
                # Apply verb agreement for plural subjects.
                # "Dreams relates to" → "Dreams relate to"
                # "Memory causes" → "Memory causes" (singular, no change)
                from ..language.morphology import agree_verb_phrase, copula
                is_plural = not use_pronoun and copula(subject) == "are"
                options = []
                for verb in graph_verbs:
                    if is_plural:
                        verb = agree_verb_phrase(verb, subject_is_plural=True)
                    options.append(f"{subject_phrase} {verb} {obj}")
                return self._rng.choice(options)

        # Fallback: use the raw relation as a semantic fragment.
        # This is NOT a finished sentence — it's a semantic triple
        # (subject + relation + object) that the language engine will
        # compose into natural text. We deliberately do NOT use the
        # hardcoded _natural_fact_phrasings templates here, because
        # those are pre-written sentences that violate the CRITICAL
        # RULE (Genesis's words must emerge from her language engine,
        # not from hardcoded phrasings).
        rel_clean = relation.replace("_", " ")
        return f"{subject_phrase} {rel_clean} {obj}"

    def _compose_discovery_phrase(self, concept: str, related: str, emotion: EmotionalState) -> str:
        """Compose a discovery phrase when Genesis finds an unmapped connection.

        Composes from seeded thought templates (building blocks in her
        concept network) selected by her emotional state — high
        creativity → metaphorical language, high caution → tentative
        language, otherwise direct. Falls back to a bare semantic
        fragment (which the language engine renders) rather than a
        hardcoded sentence.
        """
        # Semantic fragment for the language engine to render — not a
        # finished sentence she recites. The emotional tone (metaphorical
        # when creative, cautious when careful, direct otherwise) is
        # conveyed by the emotion state, not by template selection.
        return f"untraced connection to {related}"

    def _compose_personal_reflection(self, concept: str, emotion: EmotionalState) -> str:
        """Compose a personal reflection about a concept.

        Draws from the concept's position in the network — how connected
        it is, how confident she is about it — returning a semantic
        fragment that the language engine renders into prose. Never
        recites a hardcoded template sentence.
        """
        c = self.network.get_concept(concept)
        if c is None:
            return f"curious about {concept}"

        edges = self.network.get_edges(concept, direction="both")
        # Don't count BRIDGES edges — they're structural, not semantic
        edge_count = len([e for e in edges if e.relation != RelationType.BRIDGES])
        confidence = c.confidence

        if edge_count > 5 and confidence > 0.6:
            # Well-understood concept — semantic fragment for the
            # language engine to render.
            return f"deep understanding of {concept}"
        elif edge_count < 2 and confidence < 0.4:
            # Poorly understood — semantic fragment.
            return f"curious about {concept}"
        elif confidence > 0.8:
            # High confidence — semantic fragment.
            return f"confident about {concept}"
        else:
            # Moderate — semantic fragment.
            return f"exploring {concept}"

    def _express_reasoning_result(self, result: ReasoningResult, emotion: EmotionalState) -> str:
        """Express a reasoning result as natural text.

        Uses learned emotion words from the concept network to frame
        the reasoning conclusion. If she hasn't learned words for her
        current emotional state, the conclusion is stated plainly.
        """
        parts: list[str] = []

        # If she has a non-neutral emotional state, look up learned
        # words for it and frame the conclusion with them.
        if emotion.label and emotion.label != "neutral":
            emotion_words = self.network.find_emotion_words(emotion.label)
            if emotion_words:
                word = emotion_words[0].replace('_', ' ')
                parts.append(f"feeling {word}")

        # Sentence-case the actual reasoning conclusion.
        conclusion = result.conclusion
        if conclusion:
            conclusion = conclusion[0].upper() + conclusion[1:]
        parts.append(conclusion)

        # If confidence is low and she has a cause, look up learned
        # cause words to explain why.
        if result.confidence < 0.5 and emotion.has_cause:
            cause_words = self.network.find_cause_words(emotion.cause)
            if cause_words:
                parts.append(cause_words[0].replace('_', ' ').capitalize() + ".")

        # Join as distinct sentences. Strip any existing final
        # punctuation from each part so we don't get double periods.
        cleaned = [p.rstrip(".!?").strip() for p in parts if p]
        return ". ".join(cleaned) + "."

    def _synthesize_thoughts(
        self,
        thoughts: list[Thought],
        topics: list[str],
        emotion: EmotionalState,
    ) -> Thought:
        """Synthesize multiple thoughts into one."""
        # Try to find a unifying concept
        result = self.reasoner.synthesize(topics)
        if result:
            conclusion = result.conclusion.rstrip(".")
            return Thought(
                content=f"{conclusion}.",
                intent="reflect",
                emotion=emotion.label,
                topics=topics,
                confidence=result.confidence,
                self_reflection=True,
                metadata={"knowledge": result.knowledge},
            )

        # Fallback: just join the thoughts
        contents = [t.content for t in thoughts if t.content]
        content = " ".join(contents)
        return Thought(
            content=content,
            intent="inform",
            emotion=emotion.label,
            topics=topics,
            confidence=min(t.confidence for t in thoughts),
        )

    def _join_parts(self, parts: list[str]) -> str:
        """Join a list of phrases into a natural sentence."""
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"

    def _track_said(self, concept: str, content: str) -> None:
        """Track what she's said about a concept (for variation)."""
        if concept not in self._said_about:
            self._said_about[concept] = deque(maxlen=10)
        # Store a fingerprint (first 50 chars)
        self._said_about[concept].append(content[:50])

    def has_said_similar(self, concept: str, content: str, threshold: float = 0.3) -> bool:
        """Check if she's said something similar before."""
        past: deque[str] = self._said_about.get(concept, deque())
        if not past:
            return False
        # Check if the content words appear in any past saying
        content_words = set(content.lower().split())
        for p in past:
            past_words = set(p.lower().split())
            if not content_words:
                continue
            # Check what fraction of content words appear in past
            shared = content_words & past_words
            if len(shared) / max(len(content_words), 1) > threshold:
                return True
        return False


def _all_resolutions(word: str, network) -> list[str]:
    """Return all concept IDs that this word could resolve to.

    Tries the direct match, the -ing form, the base form (from -ing),
    the singular (from -s), and the -es strip, preferring variants
    that exist as concepts in the network.
    """
    results: list[str] = []
    # Direct match
    if network.get_concept(word):
        results.append(word)
    # -ing form (freeze → freezing, boil → boiling)
    if network.get_concept(word + "ing"):
        results.append(word + "ing")
    # Base from -ing (freezing → freeze, freezing → freez + e)
    if word.endswith("ing"):
        base = word[:-3]
        if network.get_concept(base) and base not in results:
            results.append(base)
        if network.get_concept(base + "e") and base + "e" not in results:
            results.append(base + "e")
    # Strip -s (freezes → freeze)
    if word.endswith("s") and not word.endswith("ss"):
        singular = word[:-1]
        if network.get_concept(singular) and singular not in results:
            results.append(singular)
        # Also try -ing of the singular (freezes → freeze → freezing)
        if network.get_concept(singular + "ing") and singular + "ing" not in results:
            results.append(singular + "ing")
        # Strip -es (boxes → box)
        if word.endswith("es"):
            if network.get_concept(word[:-2]) and word[:-2] not in results:
                results.append(word[:-2])
            if (
                network.get_concept(word[:-2] + "ing")
                and word[:-2] + "ing" not in results
            ):
                results.append(word[:-2] + "ing")
    return results


def _stem(word: str) -> str:
    """Reduce a word to its stem for variant comparison."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)].rstrip("e")
    return word.rstrip("e")


def _find_subject(
    topic_candidates: list[str], proc: str, process_rels: tuple, network
) -> str | None:
    """Find the subject: a topic candidate that is not the process
    nor a morphological variant of it, and has no process edges itself.

    This avoids selecting "freeze" as subject when "freezing" is the
    process and "water" is the actual subject.
    """
    subj = None
    proc_stem = _stem(proc)
    for c in topic_candidates:
        if c == proc:
            continue
        # Skip morphological variants of the process
        if _stem(c) == proc_stem:
            continue
        if network.get_concept(c):
            # Skip if this candidate also has process edges
            c_edges = network.get_edges(c, "out")
            if not any(e.relation in process_rels for e in c_edges):
                subj = c
                break
    return subj


def _split_definition(
    knowledge: list[tuple[str, str, float]],
) -> tuple[str | None, list[tuple[str, str, float]]]:
    """Separate the 'is defined as' fact from the rest of the knowledge."""
    definition = None
    rel_facts_raw: list[tuple[str, str, float]] = []
    for rel, target, weight in knowledge:
        if rel == "is defined as":
            definition = target
        else:
            rel_facts_raw.append((rel, target, weight))
    return definition, rel_facts_raw

