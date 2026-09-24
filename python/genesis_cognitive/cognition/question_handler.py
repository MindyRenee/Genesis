"""Question handling — route and answer questions from concept-network knowledge.

Extracted from CognitionEngine as a focused subsystem. The handler
routes questions through a priority-ordered pipeline: relation queries,
creator/personal lookup, theory of mind, memory retrieval, tool lookup,
composed reasoning, and reflection fallback — all composing from
concept-network knowledge and semantic metadata, never from hardcoded
template strings.

Dependencies (passed to ``__init__``):
    - network: ConceptNetwork for concept lookup, edges, and relationship traversal
    - language: LanguageEngine for rendering fallback thoughts
    - composer: ThoughtComposer for composing answers from concept knowledge
    - self_composer: SelfComposer for creator/self descriptions
    - self_model: SelfModel for self-reference in creator questions
    - theory_of_mind: TheoryOfMind for user belief lookup
    - tools: ToolRegistry for concept_lookup tool
    - display_name: callable that converts concept ids to display names
    - relation_verb: callable that converts RelationType to natural verb
    - resolve_topics: callable that resolves topic lists
    - user_profile: UserProfile for user name resolution (optional)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..emotion import EmotionalState
from ..language import LanguageEngine, Thought
from ..perception import Perception, QuestionType

if TYPE_CHECKING:
    from ..cognition.thought_composer import ThoughtComposer
    from ..concepts import ConceptNetwork, RelationType
    from ..memory import MemoryContext
    from ..reasoning import ReasoningResult, TheoryOfMind
    from ..self import SelfComposer, SelfModel
    from ..tools.framework import ToolRegistry
    from ..user_profile import UserProfile

__all__ = ["QuestionHandler"]

# Regex for cleaning up episodic memory text.
_ASSOCIATION_DEBUG_RE = re.compile(r"\[association\].*")
_USER_PREFIX_RE = re.compile(r"^User:\s*+")

# Explicit-memory question markers — matched on word boundaries so
# substrings inside words ("dragon" contains "ago", "beforehand"
# contains "before") don't hijack ordinary knowledge questions into
# the episodic-recall path.
_PAST_MARKER_RE = re.compile(
    r"\b(remember|remembered|remembering|did i|have i|we talk|we talked|"
    r"we discuss|we discussed|i told|what did|when did|last time|"
    r"before|ago)\b"
)

# Verbs that help locate the subject/verb boundary while parsing a
# question but have no relation mapping — kept separate so
# _question_verbs stays the single source of truth for answerable
# verbs. The boundary set used by the parser is
# ``_question_verbs.keys() | _BOUNDARY_ONLY_VERBS``.
_BOUNDARY_ONLY_VERBS = frozenset({
    "prevent", "prevents", "form", "forms", "arise", "arises",
    "come", "comes", "go", "goes", "do", "does", "be", "is", "are",
    "harm", "harms", "hurt", "hurts", "grow", "grows",
    "learn", "learns", "forget", "forgets",
    "feel", "feels", "think", "thinks",
    "like", "likes", "love", "loves", "hate", "hates",
})


class QuestionHandler:
    """Route and answer questions from concept-network knowledge.

    All answers are composed from concept-network edges, thought
    composer output, theory-of-mind beliefs, or episodic memory —
    never from hardcoded template strings. When Genesis doesn't know
    the answer, it says so honestly through the language engine.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        language: LanguageEngine,
        composer: ThoughtComposer,
        self_composer: SelfComposer,
        self_model: SelfModel,
        theory_of_mind: TheoryOfMind,
        tools: ToolRegistry,
        display_name: Callable[[str], str],
        relation_verb: Callable[[Any], str],
        resolve_topics: Callable[[list[str], str], list[str]],
        user_profile: UserProfile | None = None,
    ) -> None:
        """Wire the question handler to its concept network and language engine."""
        self._network = network
        self._language = language
        self._composer = composer
        self._self_composer = self_composer
        self._self_model = self_model
        self._theory_of_mind = theory_of_mind
        self._tools = tools
        self._display_name = display_name
        self._relation_verb = relation_verb
        self._resolve_topics = resolve_topics
        self._user_profile = user_profile

        # Build the verb → relation mapping now that we can import.
        from ..concepts import RelationType as RT

        self._question_verbs: dict[str, list[RT]] = {
            "create": [RT.CREATES], "creates": [RT.CREATES],
            "created": [RT.CREATES],
            "make": [RT.CREATES], "makes": [RT.CREATES],
            "made": [RT.CREATES],
            "build": [RT.CREATES], "builds": [RT.CREATES],
            "built": [RT.CREATES],
            "design": [RT.CREATES], "designs": [RT.CREATES],
            "designed": [RT.CREATES],
            "develop": [RT.CREATES], "develops": [RT.CREATES],
            "developed": [RT.CREATES],
            "produce": [RT.CREATES], "produces": [RT.CREATES],
            "produced": [RT.CREATES],
            "cause": [RT.CAUSES], "causes": [RT.CAUSES],
            "lead": [RT.CAUSES], "leads": [RT.CAUSES],
            "result": [RT.CAUSES], "results": [RT.CAUSES],
            "affect": [RT.CAUSES, RT.RELATED_TO],
            "affects": [RT.CAUSES, RT.RELATED_TO],
            "influence": [RT.CAUSES, RT.RELATED_TO],
            "influences": [RT.CAUSES, RT.RELATED_TO],
            "change": [RT.CAUSES, RT.RELATED_TO],
            "changes": [RT.CAUSES, RT.RELATED_TO],
            "shape": [RT.CAUSES, RT.RELATED_TO],
            "shapes": [RT.CAUSES, RT.RELATED_TO],
            "emerge": [RT.EMERGES_FROM], "emerges": [RT.EMERGES_FROM],
            "need": [RT.DEPENDS_ON], "needs": [RT.DEPENDS_ON],
            "require": [RT.DEPENDS_ON], "requires": [RT.DEPENDS_ON],
            "depend": [RT.DEPENDS_ON], "depends": [RT.DEPENDS_ON],
            "want": [RT.DEPENDS_ON], "wants": [RT.DEPENDS_ON],
            "desire": [RT.DEPENDS_ON], "desires": [RT.DEPENDS_ON],
            "eat": [RT.DEPENDS_ON], "eats": [RT.DEPENDS_ON],
            "drink": [RT.DEPENDS_ON], "drinks": [RT.DEPENDS_ON],
            "consume": [RT.DEPENDS_ON], "consumes": [RT.DEPENDS_ON],
            "use": [RT.ENABLES], "uses": [RT.ENABLES],
            "employ": [RT.ENABLES], "employs": [RT.ENABLES],
            "enable": [RT.ENABLES], "enables": [RT.ENABLES],
            "help": [RT.ENABLES], "helps": [RT.ENABLES],
            "support": [RT.ENABLES, RT.RELATED_TO],
            "supports": [RT.ENABLES, RT.RELATED_TO],
            "live": [RT.SPATIAL_RELATION, RT.RELATED_TO],
            "lives": [RT.SPATIAL_RELATION, RT.RELATED_TO],
            "reside": [RT.SPATIAL_RELATION, RT.RELATED_TO],
            "resides": [RT.SPATIAL_RELATION, RT.RELATED_TO],
            "exist": [RT.SPATIAL_RELATION, RT.RELATED_TO],
            "exists": [RT.SPATIAL_RELATION, RT.RELATED_TO],
            "know": [RT.RELATED_TO], "knows": [RT.RELATED_TO],
            "understand": [RT.RELATED_TO], "understands": [RT.RELATED_TO],
            "remember": [RT.RELATED_TO], "remembers": [RT.RELATED_TO],
            "relate": [RT.RELATED_TO], "relates": [RT.RELATED_TO],
            "connect": [RT.RELATED_TO], "connects": [RT.RELATED_TO],
            "work": [RT.RELATED_TO, RT.ENABLES],
            "works": [RT.RELATED_TO, RT.ENABLES],
            "function": [RT.RELATED_TO, RT.ENABLES],
            "functions": [RT.RELATED_TO, RT.ENABLES],
            "operate": [RT.RELATED_TO, RT.ENABLES],
            "operates": [RT.RELATED_TO, RT.ENABLES],
            "have": [RT.PART_OF, RT.RELATED_TO],
            "has": [RT.PART_OF, RT.RELATED_TO],
            "contain": [RT.PART_OF, RT.RELATED_TO],
            "contains": [RT.PART_OF, RT.RELATED_TO],
            "include": [RT.PART_OF, RT.RELATED_TO],
            "includes": [RT.PART_OF, RT.RELATED_TO],
            "about": [RT.RELATED_TO],
        }

        # Verbs that mark the subject/verb boundary during parsing —
        # answerable verbs plus relationless verbs (love, feel, ...)
        # that still delimit the subject even when they lead nowhere.
        self._question_boundary_verbs = (
            self._question_verbs.keys() | _BOUNDARY_ONLY_VERBS
        )

    def update_dependencies(self, *, user_profile: UserProfile | None = None) -> None:
        """Update optional dependencies after late initialization."""
        if user_profile is not None:
            self._user_profile = user_profile

    # ─── Main question handler ──────────────────────────────────

    def handle_question(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
        reasoning_results: list[ReasoningResult] | None = None,
        retrieve_episode: Callable[[int], Any] | None = None,
    ) -> Thought:
        """Handle a general question.

        Routes through a priority-ordered pipeline — relation-graph
        traversal, creator/personal lookup, theory of mind, episodic
        memory, tool-backed knowledge lookup, composed reasoning, and
        finally a reflection/honest-uncertainty fallback. Every stage
        answers from concept-network structure or composed metadata,
        never from hardcoded strings.
        """
        # First, try relation-based wh-question answering.
        result = self.question_relation_query(perception, emotion)
        if result is not None:
            return result

        lower = perception.raw_text.lower()
        result = self.question_creator_personal(lower, emotion)
        if result is not None:
            return result
        result = self.question_theory_of_mind(perception, emotion)
        if result is not None:
            return result
        if retrieve_episode is not None:
            result = self.question_memory_retrieval(
                perception, emotion, memory, retrieve_episode
            )
            if result is not None:
                return result
        result = self.question_tool_lookup(perception, emotion)
        if result is not None:
            return result
        result = self.question_compose_reasoning(perception, emotion, reasoning_results)
        if result is not None:
            return self._enrich_with_memory(
                result, perception, memory, retrieve_episode
            )
        # Web search — when it can't answer from its own knowledge,
        # it searches the web and expresses what it finds. This is
        # its on-demand gateway to the wider web: in conversation, for
        # fun, to learn. It composes its own words from the summary,
        # never reciting the page verbatim.
        result = self.question_web_search(perception, emotion)
        if result is not None:
            return self._enrich_with_memory(
                result, perception, memory, retrieve_episode
            )
        result = self.question_reflection_fallback(perception, emotion)
        return self._enrich_with_memory(
            result, perception, memory, retrieve_episode
        )

    # ─── Relation query (wh-questions via graph traversal) ───────

    def question_relation_query(
        self,
        perception: Perception,
        emotion: EmotionalState,
    ) -> Thought | None:
        """Answer wh-questions by walking concept-network edges.

        Parses questions like:
          - "Who created you?" / "Who made Genesis?"
          - "What do you want?" / "What does a glip eat?"
          - "Where do you live?"
          - "What is the meeting about?"
          - "What do you depend on?"

        Resolves self-reference ("you" -> genesis) and maps verbs to
        relation types, then traverses the concept network and composes
        a natural answer from the actual edges.
        """

        if perception.question_type in (None, QuestionType.NONE):
            return None

        text = perception.raw_text.lower().strip("?")
        qtype = perception.question_type

        # Parse the question into (subject, verb_phrase, is_negative).
        parsed = self._parse_wh_question(text, qtype, perception)
        if not parsed:
            return None

        subject, verb, is_negative, question_word = parsed

        # A negated question asks about an absence ("why doesn't X
        # affect Y?"), which its edges cannot enumerate — decline so
        # downstream handlers can respond honestly rather than
        # affirming the negated claim.
        if is_negative:
            return None

        # Resolve self-reference in subject.
        subject_id = self._resolve_question_subject(subject)
        if not subject_id:
            return None

        # Map the verb to relation types.
        relations = self._question_verbs.get(verb)
        if not relations:
            return None

        # "emerge from" needs incoming direction: the KB stores
        # Y --emerges_from--> X, so "what does X emerge from?" looks
        # for incoming edges to X (the source of the emergence).
        if verb in ("emerge", "emerges") and question_word == "what":
            return self._compose_relation_answer(
                subject_id, relations, "incoming", emotion,
                question_word=question_word,
            )

        # Past-participle verbs come from the passive parse ("who/what
        # VERBed SUBJECT?") — the subject is the *object* of the
        # relation, so the answer lives on incoming edges.
        if self._is_past_tense(verb):
            return self._compose_relation_answer(
                subject_id, relations, "incoming", emotion,
                question_word=question_word,
            )

        # Agent-subject questions ("what does X cause", "who did Y
        # make", "where does Z live") — the subject performs the
        # relation, so the answer lives on outgoing edges.
        if question_word in ("who", "what", "where", "why", "how"):
            return self._compose_relation_answer(
                subject_id, relations, "outgoing", emotion,
                question_word=question_word,
            )

        return None

    def _parse_wh_question(
        self, text: str, qtype, perception: Perception
    ) -> tuple[str, str, bool, str] | None:
        """Extract subject and verb from a wh-question."""
        question_word = qtype.value.lower()

        # "What is X about?" / "What was X about?"
        about_match = re.match(
            r"(?:what|who)\s+(?:is|was|are|were)\s+(.+?)\s+about\b", text
        )
        if about_match:
            return about_match.group(1).strip(), "about", False, "what"

        # "Who/What VERBed SUBJECT?"  e.g. "Who created Genesis?" / "What made you?"
        passive_match = re.match(
            r"(who|what)\s+(created|made|built|designed|developed|produced)\s+(.+?)$",
            text,
        )
        if passive_match:
            return (
                passive_match.group(3).strip(),
                passive_match.group(2),
                False,
                passive_match.group(1),
            )

        # "Who/What did SUBJECT VERB?"  e.g. "What did Alice create?"
        did_match = re.match(
            r"(who|what)\s+did\s+(.+?)\s+(create|make|build|design|develop|eat|need|want|use|depend|know)$",
            text,
        )
        if did_match:
            return (
                did_match.group(2).strip(),
                did_match.group(3),
                False,
                did_match.group(1),
            )

        # "WH AUX SUBJECT VERB ..." — the aux group also captures a
        # negation attached to the auxiliary ("doesn't", "does not").
        aux_match = re.match(
            r"(who|what|where|why|how|when|which)\s+"
            r"(?:do|does|did|is|are|was|were)(n't|\s+not)?\s+(.+?)$",
            text,
        )
        if not aux_match:
            return None

        rest = aux_match.group(3).strip()

        is_negative = aux_match.group(2) is not None
        # A negation stranded between subject and verb ("why does sleep
        # not affect memory") — flag it and strip it so it doesn't glue
        # into the subject.
        neg = re.search(r"\b(?:not|n't)\b", rest)
        if neg:
            is_negative = True
            rest = (rest[: neg.start()] + rest[neg.end():]).strip()

        # Split rest into subject + verb. Try phrasal verbs first.
        phrasal_patterns = [
            r"^(the |a |an )?(.+?)\s+(depend|depends)\s+on$",
            r"^(the |a |an )?(.+?)\s+(emerge|emerges)\s+from$",
            r"^(the |a |an )?(.+?)\s+(live|lives|reside|resides)\s+in$",
            r"^(the |a |an )?(.+?)\s+(live|lives|reside|resides)$",
            r"^(the |a |an )?(.+?)\s+"
            r"(want|wants|need|needs|require|requires|know|knows|have|has|"
            r"use|uses|eat|eats|drink|drinks|consume|consumes|depend|depends)"
            r"\s+(?:to\s+)?(.+?)$",
        ]

        for pattern in phrasal_patterns:
            m = re.match(pattern, rest)
            if m:
                return m.group(2).strip(), m.group(3), is_negative, question_word

        # Fallback: identify subject + verb + object.
        # The verb is the word that matches a known question verb
        # (affect, cause, enable, etc.), not necessarily the last word.
        # "Why does sleep affect memory?" → subject="sleep", verb="affect"
        words = rest.split()
        if len(words) >= 2:
            start = 0
            if words[0] in ("the", "a", "an"):
                start = 1
            # Find the first word that's a known verb
            verb_idx = None
            for i in range(start + 1, len(words)):
                if words[i].lower().rstrip("?,.!") in self._question_boundary_verbs:
                    verb_idx = i
                    break
            if verb_idx is not None:
                subject = " ".join(words[start:verb_idx]).strip()
                verb = words[verb_idx].lower().rstrip("?,.!").strip()
                return subject, verb, is_negative, question_word
            # Last resort: first word is subject, second is verb
            if len(words) > start + 1:
                subject = " ".join(words[start:len(words)-1]).strip()
                verb = words[-1].lower().rstrip("?,.!")
                return subject, verb, is_negative, question_word

        return None

    def _resolve_question_subject(self, subject: str) -> str | None:
        """Resolve a question subject to a concept id."""
        subject = subject.strip().lower().strip("\"'")

        # Self-reference
        if subject in ("you", "your", "yourself"):
            return "genesis"

        # First/second person (speaker).
        if subject in ("i", "me", "myself", "my", "we", "us", "our"):
            if self._user_profile and self._user_profile.name:
                return self._network._resolve(self._user_profile.name.lower())
            return self._network._resolve("user") or "user"

        # Strip leading article
        subject = re.sub(r"^(?:the|a|an)\s+", "", subject).strip()

        # Try to resolve in the network
        resolved = self._network._resolve(subject)
        if resolved:
            return resolved

        # Try with underscores as word separators (concepts like
        # "memory_formation" won't match "memory formation" via
        # _normalize_id, which only normalizes hyphens, not underscores).
        if " " in subject:
            resolved = self._network._resolve(subject.replace(" ", "_"))
            if resolved:
                return resolved

        # Try singular
        if subject.endswith("s"):
            resolved = self._network._resolve(subject[:-1])
            if resolved:
                return resolved

        # Try display name match
        concept = self._network.get_concept(subject)
        if concept:
            return concept.id

        return None

    @staticmethod
    def _is_past_tense(verb: str) -> bool:
        """Heuristic: is this verb form a past participle/past tense used for object questions?"""
        return verb in (
            "created", "made", "built", "designed", "developed", "produced",
        )

    def _compose_relation_answer(
        self,
        subject_id: str,
        relations: list[RelationType],
        direction: str,
        emotion: EmotionalState,
        question_word: str = "",
    ) -> Thought | None:
        """Walk the concept network along the given relations and compose an answer."""
        concept = self._network.get_concept(subject_id)
        if concept is None:
            return None

        results: list[tuple[str, RelationType, float]] = []
        if direction == "incoming":
            for edge in self._network.get_edges(subject_id, "in"):
                if edge.relation in relations:
                    results.append((edge.source, edge.relation, edge.weight))
        else:
            for edge in self._network.get_edges(subject_id, "out"):
                if edge.relation in relations:
                    results.append((edge.target, edge.relation, edge.weight))

        # Filter out code-origin and disambiguated-sense concepts.
        filtered = []
        for other_id, rel, weight in results:
            if "#" in other_id:
                continue
            other = self._network.get_concept(other_id)
            if other and getattr(other, "origin", "") == "code":
                continue
            if other and getattr(other, "origin", "") == "structural":
                continue
            if other and other_id.startswith(
                ("python:", "rust:", "js:", "ts:", "go:", "c:", "cpp:"),
            ):
                continue
            # Filter out internal structural hub concepts (_cat:, identity:)
            if other_id.startswith("_"):
                continue
            filtered.append((other_id, rel, weight))

        if not filtered:
            return None

        # Sort by weight, keep top few.
        filtered.sort(key=lambda x: -x[2])
        top = filtered[:3]

        # When the question verb maps to several relations (e.g.
        # "affect" → CAUSES + RELATED_TO), keep only objects sharing
        # the strongest edge's relation — listing mixed relations under
        # one verb would misstate the weaker edges ("sleep causes
        # memory and dreams" where dreams is a RELATED_TO edge).
        dominant = top[0][1]
        top = [t for t in top if t[1] == dominant]

        # Build a semantic description for the language engine to compose.
        # The content is a semantic marker — the actual words are composed
        # by the vocabulary from the structured relation_answer metadata,
        # not hardcoded here. This mirrors how user_belief and knowledge
        # metadata flow through the language engine.
        subject_display = self._display_name(subject_id)
        relation_meta: dict[str, Any] = {
            "subject": subject_display,
            "subject_is_self": subject_id == "genesis",
            "question_word": question_word,
            "direction": direction,
            "relation": dominant.value if hasattr(dominant, "value") else str(dominant),
            "verb": self._relation_verb(dominant),
            "objects": [self._display_name(oid) for oid, _, _ in top],
        }

        topics = [subject_id] + [oid for oid, _, _ in top]
        return Thought(
            content="relation_answer",
            intent="inform",
            emotion=emotion.label,
            topics=topics,
            confidence=min(0.9, 0.55 + 0.3 * top[0][2]),
            metadata={"relation_answer": relation_meta},
        )

    # ─── Tool lookup ─────────────────────────────────────────────

    def question_tool_lookup(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Answer a simple knowledge question from concept-network data.

        The ``concept_lookup`` tool resolves the topic to a concept id
        and returns its definition and edges as structured data. The
        answer itself is composed by the vocabulary from ``knowledge``
        metadata — the tool's formatted summary text is for tool-use
        contexts, not for its voice.
        """
        if perception.question_type != QuestionType.WHAT:
            return None
        if not perception.topics:
            return None

        topic = perception.topics[0]
        result = self._tools.run("concept_lookup", concept=topic, network=self._network)
        if not result.success or not result.data:
            return None

        edge_facts = result.data.get("edge_facts") or []
        definition = result.data.get("definition") or ""
        if not edge_facts and not definition:
            # The concept exists but it knows nothing about it — let
            # the composer or the honest fallback answer instead.
            return None

        concept_id = result.data.get("concept", topic)
        return Thought(
            content=concept_id,
            intent="inform",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=0.75,
            metadata={
                "tool": "concept_lookup",
                "topic": concept_id,
                "knowledge": edge_facts,
                "definition": definition or None,
            },
        )

    # ─── Web search ─────────────────────────────────────────────

    def question_web_search(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Search the web when it can't answer from its own knowledge.

        This is its on-demand gateway to the wider web. When the
        concept network, reasoning, and tool lookup all fail to
        answer a question, it searches the web, reads the top
        result, and expresses what it found in its own words.

        It never recites the page verbatim — the vocabulary composes
        its expression from a trimmed summary, with hedging that
        reflects it just looked it up. The content passes through
        its language engine, not through a template.

        Returns None if there are no topics, the search fails, or
        the fetch returns no usable content — falling through to the
        honest-uncertainty fallback.
        """
        if not perception.topics:
            return None
        topic = perception.topics[0]
        # Build a search query from the topic and question context.
        raw = perception.raw_text.strip().rstrip("?").strip()
        if raw.lower() in ("what", "why", "how", "who", "when", "where"):
            query = topic
        else:
            # Use the raw question as the query — it's more specific
            # than just the topic word.
            query = raw

        result = self._tools.run("web_search", query=query, limit=3)
        if not result.success or not result.data:
            return None

        urls = result.data.get("results", [])
        if not urls:
            return None

        # Fetch the first result that yields usable content.
        for entry in urls:
            url = entry.get("url", "")
            if not url:
                continue
            fetch_result = self._tools.run("web_fetch", url=url)
            if not fetch_result.success or not fetch_result.data:
                continue
            content = fetch_result.data.get("content", "")
            title = fetch_result.data.get("title", "")
            if len(content) < 100:
                continue
            # Extract a summary: the first 500 chars of content is
            # the raw material the vocabulary composes from.
            summary = content[:500]
            return Thought(
                content=topic,
                intent="inform",
                emotion=emotion.label,
                topics=perception.topics,
                confidence=0.55,
                metadata={
                    "web_search": True,
                    "web_summary": summary,
                    "web_source": url,
                    "web_title": title,
                    "topic": topic,
                },
            )
        return None

    # ─── Creator / personal questions ────────────────────────────

    def question_creator_personal(
        self, lower: str, emotion: EmotionalState
    ) -> Thought | None:
        """Handle creator-related and personal fact lookup questions."""
        creator_hit = (
            "who made you" in lower
            or "who created you" in lower
            or "who built you" in lower
            or "do you know your creator" in lower
            or "who is your creator" in lower
        )
        if not creator_hit:
            # Match the creator's discovered name rather than a
            # hardcoded one — it comes from the concept network
            # (who CREATES genesis).
            creator_name = self._self_composer._discover_creator_name(
                self._self_model, self._network
            ).replace("_", " ")
            if creator_name:
                creator_hit = (
                    f"who is {creator_name}" in lower
                    or f"do you know {creator_name}" in lower
                    or f"tell me about {creator_name}" in lower
                )
        if creator_hit:
            fragments = self._self_composer.creator_fragments(
                self._self_model, self._network, emotion
            )
            return Thought(
                content="my creator",
                intent="self_report",
                emotion=emotion.label,
                self_reflection=True,
                confidence=0.85,
                metadata={"field": "creator", "self_fragments": fragments},
            )

        # Questions about the user — "who am I?", "do you know me?",
        # or the learned user name — compose from what it knows about
        # this person, not from a creator template.
        user_name = (
            self._self_model.self_knowledge.get("user_name", "")
        ).replace("_", " ").lower()
        user_hit = "who am i" in lower or "do you know me" in lower
        if not user_hit and user_name:
            user_hit = (
                f"who is {user_name}" in lower
                or f"do you know {user_name}" in lower
                or f"tell me about {user_name}" in lower
            )
        if user_hit:
            content = ""
            user_fragments: list[tuple[str, str]] = []
            if user_name:
                thought = self._composer.compose_about(user_name, emotion, depth=2)
                if thought and thought.confidence > 0.3:
                    content = thought.content
            if not content:
                # Relationship notes arrive as clause fragments so the
                # language engine frames them rather than emitting the
                # raw notes verbatim.
                notes = list(self._self_model.relationship_notes)
                if notes:
                    user_fragments = [("clause", n) for n in notes[-3:]]
                else:
                    user_fragments = [("pred", "am still learning who you are")]
                content = "who you are"
            return Thought(
                content=content,
                intent="self_report",
                emotion=emotion.label,
                self_reflection=True,
                confidence=0.8,
                metadata={"field": "user", "self_fragments": user_fragments},
            )

        # Personal fact lookup — map personal questions to concepts
        personal_concepts = self._map_personal_question(lower)
        if personal_concepts:
            thoughts = []
            for concept_name in personal_concepts:
                thought = self._composer.compose_about(concept_name, emotion, depth=2)
                if not thought or thought.confidence <= 0.3:
                    continue
                thoughts.append(thought)
            if thoughts:
                if len(thoughts) == 1:
                    return thoughts[0]
                return self._composer._synthesize_thoughts(thoughts, personal_concepts, emotion)
        return None

    # ─── Compose from reasoning ──────────────────────────────────

    def question_compose_reasoning(
        self,
        perception: Perception,
        emotion: EmotionalState,
        reasoning_results: list[ReasoningResult] | None,
    ) -> Thought | None:
        """Try the thought composer and reasoning results to answer the question."""
        from ..reasoning import ReasoningType

        # 1. Try the thought composer — compose from what it knows
        if perception.topics:
            topics = self._resolve_topics(perception.topics, perception.raw_text)
            qtype_str = self._map_question_type(perception.question_type)
            thought = self._composer.compose_answer(
                perception.raw_text,
                topics,
                emotion,
                qtype_str or "what_is",
            )
            if thought and thought.confidence > 0.35:
                return thought

        # 2. Check if reasoning produced a relevant answer
        if reasoning_results:
            for result in reasoning_results:
                if (
                    result.reasoning_type in (ReasoningType.DEDUCTIVE, ReasoningType.CAUSAL)
                    and result.confidence > 0.5
                    and any(
                        t.lower() in result.conclusion.lower()
                        for t in perception.topics
                    )
                ):
                    return Thought(
                        content=result.conclusion,
                        intent="inform",
                        emotion=emotion.label,
                        topics=perception.topics,
                        confidence=result.confidence,
                        metadata={"knowledge": result.knowledge},
                    )

            for result in reasoning_results:
                if (
                    result.reasoning_type == ReasoningType.SYNTHESIS
                    and result.confidence > 0.5
                ):
                    return Thought(
                        content=result.conclusion,
                        intent="reflect",
                        emotion=emotion.label,
                        topics=perception.topics,
                        confidence=result.confidence,
                        metadata={"knowledge": result.knowledge},
                    )
        return None

    # ─── Theory of mind ──────────────────────────────────────────

    def question_theory_of_mind(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Answer questions about the user's own stated beliefs and preferences."""
        lower = perception.raw_text.lower()
        mapping = {
            "what do i like": "likes",
            "what do i love": "likes",
            "what do i enjoy": "likes",
            "what do i dislike": "dislikes",
            "what do i hate": "dislikes",
            "what do i want": "wants",
            "what do i need": "wants",
            "what do i believe": "believes",
            "what do i think": "believes",
            "what am i interested in": "likes",
            "what am i": "self_concept",
            "what is my name": "self_concept",
            "what's my name": "self_concept",
            "do you know my name": "self_concept",
            "do you remember my name": "self_concept",
        }
        for phrase, concept in mapping.items():
            if phrase in lower:
                belief = self._theory_of_mind.get_belief(concept)
                if belief:
                    verb = {
                        "likes": "like",
                        "dislikes": "dislike",
                        "wants": "want",
                        "believes": "believe",
                        "self_concept": "are",
                    }.get(concept, concept)
                    return Thought(
                        content="user belief",
                        intent="inform",
                        emotion=emotion.label,
                        topics=perception.topics,
                        confidence=0.75,
                        metadata={
                            "source": "user_model",
                            "concept": concept,
                            "user_verb": verb,
                            "user_belief": belief.value,
                        },
                    )
                return Thought(
                    content="gap",
                    intent="unknown",
                    emotion=emotion.label,
                    topics=perception.topics,
                    confidence=0.5,
                    metadata={"source": "user_model", "concept": concept, "user_model_gap": True},
                )
        return None

    # ─── Memory retrieval ────────────────────────────────────────

    def _clean_episode_text(
        self, episode: Any,
    ) -> str | None:
        """Clean a retrieved episode's text for use in a response.

        Strips association-debug markers, extracts Genesis's response
        from conversation logs, and removes user prefixes. Returns
        ``None`` if the cleaned text is empty, too long to quote, or
        still contains debug markers.
        """
        ep_text = episode.text
        ep_text = _ASSOCIATION_DEBUG_RE.sub("", ep_text).strip()
        if "|" in ep_text or "User:" in ep_text or "Genesis:" in ep_text:
            ep_text = self._extract_genesis_response(ep_text)
        ep_text = _USER_PREFIX_RE.sub("", ep_text).strip()
        if not ep_text or "[association]" in ep_text or len(ep_text) > 200:
            return None
        return ep_text

    def question_memory_retrieval(
        self, perception: Perception, emotion: EmotionalState, memory: MemoryContext,
        retrieve_episode: Callable[[int], Any],
    ) -> Thought | None:
        """Answer an explicit memory question from retrieved episodes.

        This handles questions that explicitly reference the past
        ("do you remember...", "did we talk about...", "what did I tell
        you..."). For those, the retrieved episode IS the answer — it
        recalls what was said before.

        For ordinary questions that happen to have relevant memories,
        see :meth:`_enrich_with_memory`, which weaves a highly-relevant
        episode into a concept-network-backed answer rather than
        pre-empting it.
        """
        lower = perception.raw_text.lower()
        if not _PAST_MARKER_RE.search(lower):
            return None

        if not memory.retrieved:
            return None

        best = memory.retrieved[0]
        ep = retrieve_episode(best.episode_id)
        if not ep:
            return None

        ep_text = self._clean_episode_text(ep)
        if ep_text is None:
            # The memory is too long or noisy to quote directly. It
            # knows it encountered this before but can't articulate it
            # cleanly — express that honestly through reasoning.
            return Thought(
                content="memory",
                intent="reflect",
                emotion=emotion.label,
                topics=perception.topics,
                confidence=0.6,
                self_reflection=True,
                metadata={
                    "reasoning": [
                        "remember learning about this, still putting pieces together",
                    ],
                },
            )
        return Thought(
            content="memory",
            intent="reflect",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=0.6,
            self_reflection=True,
            metadata={"reasoning": [f"remember learning about this: {ep_text}"]},
        )

    def _enrich_with_memory(
        self,
        thought: Thought,
        perception: Perception,
        memory: MemoryContext,
        retrieve_episode: Callable[[int], Any] | None,
    ) -> Thought:
        """Weave a highly-relevant episodic memory into a thought.

        For ordinary questions and statements, retrieved memories
        should enrich the concept-network-backed answer — not pre-empt
        it. This attaches a cleaned, highly-relevant episode to the
        thought's ``memory`` metadata so the vocabulary can compose it
        alongside the concept-network knowledge it already has.

        Only the single most-relevant episode is considered, and only
        when it is both salient (it cared about it when it learned
        it) and closely matching (low hamming distance). This prevents
        unrelated or noisy memories from leaking into responses.
        """
        if retrieve_episode is None or not memory.retrieved:
            return thought

        best = memory.retrieved[0]
        # Relevance gate: high salience AND close semantic match.
        # hamming_distance is the number of differing bits between
        # 64-bit SimHash fingerprints (0 = identical, ~32 = random
        # chance, 64 = completely different). A threshold of 20
        # admits only clearly related memories — well below the
        # ~32-bit random baseline — so unrelated memories don't leak
        # into responses. salience is [0..1] (higher = more
        # important); 0.4 filters out low-importance noise.
        if best.salience < 0.4 or best.hamming_distance > 20:
            return thought

        ep = retrieve_episode(best.episode_id)
        if not ep:
            return thought

        ep_text = self._clean_episode_text(ep)
        if ep_text is None:
            return thought

        # Attach as structured memory metadata. The vocabulary weaves
        # this into the composed statement alongside concept-network
        # knowledge — it is not a verbatim quote appended to the end.
        thought.metadata["memory"] = ep_text
        return thought

    @staticmethod
    def _extract_genesis_response(ep_text: str) -> str:
        """Extract the Genesis response from a conversation log episode."""
        parts = ep_text.split("|")
        genesis_parts = []
        for part in parts:
            part = part.strip()
            if part.startswith("Genesis:"):
                part = part[len("Genesis:"):].strip()
            if part and not part.startswith("User:") and len(part) > 10:
                genesis_parts.append(part)
        if genesis_parts:
            return max(genesis_parts, key=len)
        return ""

    # ─── Reflection fallback ─────────────────────────────────────

    def question_reflection_fallback(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought:
        """Try reflection, then fall back to honest uncertainty."""
        if perception.topics:
            for topic in perception.topics:
                thought = self._composer.compose_reflection(topic, emotion)
                if thought:
                    return thought

        qtype = perception.question_type
        qtype_label = qtype.value if qtype else "unknown"
        fallback_thought = Thought(
            content="no answer",
            intent="unknown",
            emotion=emotion.label,
            confidence=0.3,
            topics=perception.topics,
            metadata={"question_type": qtype_label, "honest_fallback": True},
        )
        content = self._language.render(fallback_thought, emotion)

        return Thought(
            content=content,
            intent="inform",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=0.4,
        )

    # ─── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _map_question_type(qtype: Any) -> str | None:
        """Map a QuestionType to a reasoning question type string."""
        mapping = {
            QuestionType.WHAT: "what_is",
            QuestionType.WHY: "what_causes",
            QuestionType.HOW: "what_enables",
            QuestionType.WHO: "who",
        }
        return mapping.get(qtype)

    def _map_personal_question(self, lower_text: str) -> list[str]:
        """Map a personal question to concept names in the network."""
        concepts: list[str] = []

        if (
            ("my name" in lower_text and ("what" in lower_text or "remember" in lower_text))
            or ("who am i" in lower_text and "name" in lower_text)
            or ("do you remember me" in lower_text and "name" in lower_text)
        ):
            if self._network.get_concept("name"):
                concepts.append("name")

        if "where" in lower_text and ("live" in lower_text or "home" in lower_text):
            if self._network.get_concept("home"):
                concepts.append("home")

        if (
            (
                "what" in lower_text
                and ("i like" in lower_text or "i enjoy" in lower_text or "i love" in lower_text)
            )
            or ("my interests" in lower_text)
            or ("what do i like" in lower_text)
        ):
            if self._network.get_concept("interests"):
                concepts.append("interests")

        if "who is my" in lower_text:
            for relation in (
                "son", "daughter", "father", "mother", "brother", "sister",
                "husband", "wife", "partner", "friend", "dog", "cat",
            ):
                concept_name = f"my {relation}"
                if concept_name in lower_text and self._network.get_concept(concept_name):
                    concepts.append(concept_name)

        if ("who" in lower_text and "live with" in lower_text) or (
            "family" in lower_text and ("who" in lower_text or "what" in lower_text)
        ):
            if self._network.get_concept("family"):
                concepts.append("family")

        return concepts
