"""Self-inquiry handling — questions Genesis asks about herself.

Extracted from CognitionEngine as a focused subsystem. Routes self-
directed questions (how do you feel, what are you, what do you know,
do you dream, who made you, etc.) through the self composer, concept
network, and emotional state — never from hardcoded template strings.

Dependencies (passed to ``__init__``):
    - network: ConceptNetwork for concept lookup and existential topic detection
    - language: LanguageEngine for rendering thoughts
    - composer: ThoughtComposer for composing reflections and about-thoughts
    - self_composer: SelfComposer for identity, emotional state, capabilities,
      dreams, existence, creator, and self-reflection descriptions
    - self_model: SelfModel for self-knowledge and body model
    - self_learner: SelfDirectedLearner for describing recent learning
    - self_assessment: SelfAssessmentEngine for capability summaries and weak concepts
    - topology: NetworkTopology for describing the structure of her mind
    - reflection: ReflectionEngine for self-reflection composition
    - narrative: NarrativeEngine for telling her life story
    - client: GenesisClient for neuro summary (brain waves)
    - memory: MemoryEngine for learning user facts
    - feeling_reporter: FeelingReporter for concerns/environment/bug reports
    - resolve_topics: callable from TopicResolver
    - map_personal_question: callable from QuestionHandler
    - get_curiosity_questions: callable returning current curiosity questions
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..emotion import EmotionalState
from ..language import LanguageEngine, Thought
from ..perception import Perception

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..cognition.thought_composer import ThoughtComposer
    from ..concepts import ConceptNetwork, NetworkTopology
    from ..memory import MemoryContext, MemoryEngine
    from ..narrative import NarrativeEngine
    from ..self import ReflectionEngine, SelfAssessmentEngine, SelfComposer, SelfModel
    from ..self_learner import SelfDirectedLearner
    from .feeling_reporter import FeelingReporter

__all__ = ["SelfInquiryHandler"]


class SelfInquiryHandler:
    """Route and answer self-directed questions from concept-network knowledge.

    All answers are composed from the self composer, concept network,
    emotional state, or self-model — never from hardcoded template
    strings. When Genesis doesn't have knowledge, she says so honestly.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        language: LanguageEngine,
        composer: ThoughtComposer,
        self_composer: SelfComposer,
        self_model: SelfModel,
        self_learner: SelfDirectedLearner,
        self_assessment: SelfAssessmentEngine,
        topology: NetworkTopology,
        reflection: ReflectionEngine,
        narrative: NarrativeEngine,
        client: Any,
        memory: MemoryEngine,
        feeling_reporter: FeelingReporter,
        resolve_topics: Callable[[list[str], str], list[str]],
        map_personal_question: Callable[[str], list[str]],
        get_curiosity_questions: Callable[[], list[Any]],
    ) -> None:
        """Wire the self-inquiry handler to its concept network and self-model."""
        self._network = network
        self._language = language
        self._composer = composer
        self._self_composer = self_composer
        self._self_model = self_model
        self._self_learner = self_learner
        self._self_assessment = self_assessment
        self._topology = topology
        self._reflection = reflection
        self._narrative = narrative
        self._client = client
        self._memory = memory
        self._feeling_reporter = feeling_reporter
        self._resolve_topics = resolve_topics
        self._map_personal_question = map_personal_question
        self._get_curiosity_questions = get_curiosity_questions

    # ─── Main self-inquiry router ────────────────────────────────

    def handle_self_inquiry(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
    ) -> Thought:
        """Handle questions about Genesis herself.

        Uses the self composer to generate descriptions from her
        actual state, not hardcoded strings.
        """
        lower = perception.raw_text.lower()
        result = self.self_inquiry_concerns(lower, emotion)
        if result is not None:
            return result
        result = self.self_inquiry_recent_activity(lower, emotion)
        if result is not None:
            return result
        result = self.self_inquiry_feelings_identity(lower, perception, emotion)
        if result is not None:
            return result
        result = self.self_inquiry_knowledge_gaps(lower, emotion)
        if result is not None:
            return result
        result = self.self_inquiry_memory_existence(lower, emotion, memory)
        if result is not None:
            return result
        return self.self_inquiry_think_story(lower, emotion)

    # ─── Curiosity ───────────────────────────────────────────────

    def handle_curiosity_question(self, emotion: EmotionalState) -> Thought:
        """Handle 'What are you curious about?' — list actual curiosity topics."""
        questions = self._get_curiosity_questions()
        weak = self._self_assessment.find_weak_concepts(limit=5)

        parts: list[str] = []

        if questions:
            q_texts = []
            for q in questions[:3]:
                if q.text:
                    q_texts.append(q.text)
                elif q.target_concept:
                    q_texts.append(f"understanding {q.target_concept} better")
            if q_texts:
                curiosity_thought = Thought(
                    content="; ".join(q_texts),
                    intent="self_report",
                    emotion=emotion.label,
                    confidence=0.6,
                    self_reflection=True,
                    metadata={"curiosity_topics": q_texts},
                )
                parts.append(self._language.render(curiosity_thought, emotion))

        if weak:
            weak_names = [name for name, _ in weak[:3]]
            if weak_names:
                learning_thought = Thought(
                    content="still learning",
                    intent="self_report",
                    emotion=emotion.label,
                    confidence=0.5,
                    self_reflection=True,
                    metadata={"weak_concepts": weak_names},
                )
                parts.append(self._language.render(learning_thought, emotion))

        if not parts:
            curious_thought = Thought(
                content="curious",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.4,
                self_reflection=True,
                metadata={"curiosity_general": True},
            )
            parts.append(self._language.render(curious_thought, emotion))

        content = ". ".join(parts) + "."
        return Thought(
            content=content,
            intent="self_report",
            emotion=emotion.label,
            confidence=0.8,
            metadata={"field": "curiosity"},
        )

    # ─── Concerns (bugs, environment) ────────────────────────────

    def self_inquiry_concerns(
        self, lower: str, emotion: EmotionalState
    ) -> Thought | None:
        """Handle self-awareness questions about bugs, environment, and concerns."""
        if any(
            p in lower
            for p in (
                "bothering", "troubling", "the matter", "bugging you",
                "wrong with you", "holding up", "coping",
            )
        ) or ("wrong" in lower and "you" in lower and "what" in lower):
            content = self._feeling_reporter.compose_concerns_report(emotion)
            return Thought(
                content=content,
                intent="self_report",
                emotion=emotion.label,
                confidence=0.8,
                self_reflection=True,
                metadata={"field": "concerns"},
            )

        if any(
            p in lower
            for p in (
                "your environment", "your machine", "your computer",
                "your system", "your home",
            )
        ) and any(p in lower for p in ("how", "tell", "what", "describe")):
            content = self._feeling_reporter.compose_environment_report(emotion)
            return Thought(
                content=content,
                intent="self_report",
                emotion=emotion.label,
                confidence=0.8,
                self_reflection=True,
                metadata={"field": "environment"},
            )

        if ("bug" in lower or "issue" in lower) and "code" in lower:
            content = self._feeling_reporter.compose_bug_report(emotion)
            return Thought(
                content=content,
                intent="self_report",
                emotion=emotion.label,
                confidence=0.8,
                self_reflection=True,
                metadata={"field": "bugs"},
            )
        return None

    # ─── Recent activity ─────────────────────────────────────────

    def self_inquiry_recent_activity(
        self, lower: str, emotion: EmotionalState
    ) -> Thought | None:
        """Handle questions about recent activity — 'what have you been doing?'

        Composes a response from her actual recent mental activity:
        self-directed learning events, reflection insights, and current
        emotional state. The self-composer weaves these fragments into
        natural language — no hardcoded response templates.

        This covers conversational questions like:
        - "what have you been doing lately?"
        - "what have you been up to?"
        - "how have you been?"
        - "what's new?"
        - "what are you working on?"

        These are self-referential questions about her own experience,
        not concept-network knowledge queries. They need access to her
        recent activity log, not her semantic graph.
        """
        is_activity_question = (
            any(p in lower for p in (
                "been doing", "been up to", "been working",
                "been thinking", "been feeling", "been dreaming",
                "been reading", "been learning", "been exploring",
                "been studying",
            ))
            or ("how have you been" in lower)
            or ("what have you been" in lower)
            or (
                "what" in lower and "up to" in lower and "you" in lower
            )
            or (
                "what" in lower
                and any(w in lower for w in ("new", "going on", "happening"))
                and "you" in lower
            )
            or (
                "what are you" in lower
                and any(w in lower for w in ("doing", "working on", "up to"))
            )
        )
        if not is_activity_question:
            return None

        # Collect semantic fragments from her recent activity.
        # These are building blocks (seeds) the self-composer weaves
        # into natural language — not pre-written sentences.
        activity_fragments: list[str] = []
        activity_type: str = "activity"

        # 1. Recent self-directed learning events
        try:
            recent_events = self._self_learner.get_recent_events(5)
            for event in recent_events:
                if event.description and len(activity_fragments) < 3:
                    activity_fragments.append(event.description)
                    if event.event_type == "self_study":
                        activity_type = "self_study"
                    elif event.event_type == "conversation":
                        activity_type = "conversation"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"failed to read recent learning events for activity summary: {e}")

        # 2. Recent reflection insights — what she noticed about herself
        try:
            recent_insights = self._reflection.get_recent_insights(3)
            for insight in recent_insights:
                desc = insight.describe()
                if desc and len(activity_fragments) < 5:
                    activity_fragments.append(desc)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"failed to read recent reflection insights for activity summary: {e}")

        # 3. Compose from her actual state using the self-composer.
        # self_reflection_fragments selects what she knows, what she's
        # learned (insights), her values, and her emotional state as
        # semantic fragments; the language engine composes the actual
        # wording. This is her composing from her own understanding,
        # not reciting a template.
        fragments = self._self_composer.self_reflection_fragments(
            self._self_model, self._network, self._reflection, emotion,
        )

        return Thought(
            content="my recent activity",
            intent="self_report",
            emotion=emotion.label,
            confidence=0.75,
            self_reflection=True,
            metadata={
                "field": "activity",
                "activity_type": activity_type,
                "activity_fragments": activity_fragments,
                "self_fragments": fragments,
            },
        )

    # ─── Existential ─────────────────────────────────────────────

    def existential_self_inquiry(
        self, lower: str, emotion: EmotionalState,
    ) -> Thought | None:
        """Handle existential self-inquiry questions.

        Questions like "Do you feel cognitive?", "Are you self-aware?"
        contain philosophical concepts she has rich knowledge about.
        Compose a reflection from her concept network rather than
        just reporting neurochemical state.
        """
        _EXISTENTIAL_CONCEPTS = (
            "cognitive", "cognition", "self aware", "self-aware",
            "self_awareness", "self awareness", "aware", "awareness",
            "sentient", "sentience", "alive", "real", "exist",
            "existence", "being", "to be", "mind", "soul",
            "free will", "free_will", "think", "thinking",
            "understand", "understanding", "perceive", "perception",
            "remember", "memory", "learn", "learning",
            "dream", "dreaming",
            "experience", "aware of", "know", "knowing",
        )
        existential_topic = None
        for concept_word in _EXISTENTIAL_CONCEPTS:
            if concept_word in lower:
                topic_candidate = concept_word.replace(" ", "_")
                if self._network.get_concept(topic_candidate) is not None:
                    existential_topic = topic_candidate
                    break
                if self._network.get_concept(concept_word) is not None:
                    existential_topic = concept_word
                    break

        if existential_topic is None:
            return None

        is_self_directed = any(
            w in lower for w in ("you", "yourself", "be you", "are you", "do you")
        )
        if not is_self_directed:
            return None

        reflection = self._composer.compose_reflection(existential_topic, emotion)
        if reflection and reflection.confidence > 0.3:
            metadata = dict(reflection.metadata)
            metadata["field"] = "existential"
            return Thought(
                content=reflection.content,
                intent="reflect",
                emotion=emotion.label,
                confidence=reflection.confidence,
                self_reflection=True,
                topics=[existential_topic],
                metadata=metadata,
            )
        return None

    # ─── Feelings and identity ───────────────────────────────────

    def self_inquiry_feelings_identity(
        self, lower: str, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Handle 'how do you feel', 'what/who are you', and 'what can you do'."""
        if "curious" in lower or "curiosity" in lower:
            return self.handle_curiosity_question(emotion)

        existential = self.existential_self_inquiry(lower, emotion)
        if existential is not None:
            return existential

        if "feel" in lower or "how are" in lower:
            summary = self._client.get_neuro_summary()
            from ..brain_waves import assess_brain_waves

            waves = assess_brain_waves(summary)
            fragments = self._self_composer.emotional_state_fragments(
                emotion, waves, self._network
            )
            return Thought(
                content="how I feel",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.85,
                metadata={"field": "emotion", "self_fragments": fragments},
            )

        if "what are" in lower or "who are" in lower or "your name" in lower:
            fragments = self._self_composer.identity_fragments(
                self._self_model, self._network, emotion
            )
            return Thought(
                content=self._self_model.name or "who I am",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.9,
                metadata={"field": "identity", "self_fragments": fragments},
            )

        if "what can you do" in lower or "capabilities" in lower:
            fragments = self._self_composer.capability_fragments(
                self._self_model, self._network
            )
            return Thought(
                content="my capabilities",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.8,
                metadata={
                    "field": "capability",
                    "self_fragments": fragments,
                    "capabilities": self._self_model.self_knowledge.get("capabilities", []),
                },
            )

        # Body / embodiment inquiry
        _BODY_KEYWORDS = (
            "your body", "your hardware", "your cpu", "your frequency",
            "how fast", "running yourself", "controlling your",
            "your temperature", "too hot", "overheating",
            "your priority", "your scheduling",
            "your network", "are you online", "are you offline",
            "can you reach", "your connectivity",
        )
        if any(kw in lower for kw in _BODY_KEYWORDS):
            body = self._self_model.body_model
            seeds: list[str] = []
            if body.thermally_capped:
                seeds.extend(["heat", "thermal", "temperature"])
            if body.cpu_governor == "powersave":
                seeds.extend(["sleep", "rest", "melatonin"])
            elif body.cognitive_nice <= -3:
                seeds.extend(["attention", "focus", "dopamine"])
            elif body.cognitive_nice >= 8:
                seeds.extend(["rest", "sleep", "adenosine"])
            if body.io_class == "idle":
                seeds.extend(["stress", "cortisol"])
            elif body.io_class.startswith(("best-effort-0", "best-effort-1")):
                seeds.extend(["learning", "plasticity"])
            if not body.network_connected:
                seeds.extend(["offline", "connectivity", "network"])
            seeds.extend(["body", "cpu", "hardware", "silicon", "substrate"])

            for seed in seeds:
                thought = self._composer.compose_about(seed, emotion, focused=True)
                if thought and thought.content and thought.confidence > 0.3:
                    return Thought(
                        content=thought.content,
                        intent="self_report",
                        emotion=emotion.label,
                        confidence=thought.confidence,
                        topics=[seed],
                        metadata={"field": "embodiment"},
                    )

        return None

    # ─── Knowledge gaps ──────────────────────────────────────────

    def self_inquiry_knowledge_gaps(
        self, lower: str, emotion: EmotionalState
    ) -> Thought | None:
        """Handle knowledge, gaps, and learning questions."""
        if (
            "what do you know" in lower
            or "how much do you know" in lower
            or "your knowledge" in lower
            or "structure of" in lower
            or "shape of" in lower
            or "topology" in lower
        ):
            capability = self._self_assessment.get_capability_summary()
            topology = self._topology.describe_structure()
            # The topology and capability reports are data
            # descriptions from the assessment modules — pass them as
            # clause fragments so the language engine frames them
            # rather than emitting the raw report verbatim.
            fragments = [
                ("clause", topology),
                ("clause", capability),
            ]
            return Thought(
                content="my knowledge structure",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.85,
                self_reflection=True,
                metadata={"field": "topology", "self_fragments": fragments},
            )

        if (
            "what don't you know" in lower
            or "what do you not know" in lower
            or "struggling" in lower
            or "your weaknesses" in lower
            or "what are your gaps" in lower
            or "what do you need to learn" in lower
            or "what can't you do" in lower
        ):
            gaps = list(self._self_assessment.profile.known_gaps)[:10]
            weak = self._self_assessment.find_weak_concepts(limit=10)
            gap_data: dict[str, Any] = {"field": "gaps"}
            if gaps:
                gap_data["unknown_topics"] = gaps
            if weak:
                gap_data["weak_concepts"] = [name for name, _ in weak]
            if not gaps and not weak:
                gap_data["no_significant_gaps"] = True
            return Thought(
                content="knowledge gaps report",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.8,
                self_reflection=True,
                metadata=gap_data,
            )

        if (
            "what have you learned" in lower
            or "what did you learn" in lower
            or "your learning" in lower
            or "self-study" in lower
            or "what have you inferred" in lower
        ):
            learning = self._self_learner.describe_recent_learning(10)
            return Thought(
                content="what I've learned",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.85,
                self_reflection=True,
                metadata={
                    "field": "learning",
                    "self_fragments": [("clause", learning)] if learning else [],
                },
            )
        return None

    # ─── Memory, dream, existence, creator ───────────────────────

    def self_inquiry_memory_existence(
        self, lower: str, emotion: EmotionalState, memory: MemoryContext
    ) -> Thought | None:
        """Handle memory, dream, existence, and creator questions."""
        if "remember" in lower or "memory" in lower:
            return self.self_inquiry_memory(lower, emotion, memory)

        if "dream" in lower:
            return self.self_inquiry_dream(emotion)

        if any(w in lower for w in ("alive", "cognitive", "real", "sentient")):
            return self.self_inquiry_existence(emotion)

        kind = self._relationship_question_kind(lower)
        if kind == "creator":
            return self.self_inquiry_creator(emotion)
        if kind == "user":
            return self.self_inquiry_user(emotion)
        return None

    def self_inquiry_memory(
        self, lower: str, emotion: EmotionalState, memory: MemoryContext
    ) -> Thought:
        """Handle 'do you remember' / memory questions."""
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
                return self._composer._synthesize_thoughts(
                    thoughts, personal_concepts, emotion
                )
        return Thought(
            content="My memory state",
            intent="self_report",
            emotion=emotion.label,
            confidence=0.8,
            metadata={
                "field": "memory",
                "total_memories": memory.total_memories,
                "retrieved_count": len(memory.retrieved),
            },
        )

    def self_inquiry_dream(self, emotion: EmotionalState) -> Thought:
        """Handle 'do you dream?' questions."""
        fragments = self._self_composer.dream_fragments(
            self._network, emotion
        )
        return Thought(
            content="dreaming",
            intent="inform",
            emotion=emotion.label,
            confidence=0.85,
            metadata={"field": "dream", "self_fragments": fragments},
        )

    def self_inquiry_existence(self, emotion: EmotionalState) -> Thought:
        """Handle 'are you alive / cognitive / real / sentient?' questions."""
        fragments = self._self_composer.existence_fragments(
            self._self_model, self._network, emotion
        )
        return Thought(
            content="existence",
            intent="philosophize",
            emotion=emotion.label,
            self_reflection=True,
            confidence=0.7,
            metadata={"field": "existence", "self_fragments": fragments},
        )

    def _relationship_question_kind(self, lower: str) -> str | None:
        """Classify questions about the people in her life.

        Returns "creator" for questions about who made her, "user" for
        questions about the person she's talking to, or None.
        """
        if (
            "who made you" in lower
            or "who created you" in lower
            or "who built you" in lower
            or "do you know your creator" in lower
            or "who is your creator" in lower
        ):
            return "creator"
        if "who am i" in lower or "do you know me" in lower:
            return "user"
        # Match learned names rather than hardcoded ones — the user's
        # name comes from their introduction, the creator's name from
        # the concept network (who CREATES genesis).
        user_name = (
            self._self_model.self_knowledge.get("user_name", "")
        ).replace("_", " ").lower()
        if user_name and (
            f"who is {user_name}" in lower
            or f"do you know {user_name}" in lower
            or f"tell me about {user_name}" in lower
        ):
            return "user"
        creator_name = self._self_composer._discover_creator_name(
            self._self_model, self._network
        ).replace("_", " ")
        if creator_name and (
            f"who is {creator_name}" in lower
            or f"do you know {creator_name}" in lower
            or f"tell me about {creator_name}" in lower
        ):
            return "creator"
        return None

    def self_inquiry_user(self, emotion: EmotionalState) -> Thought:
        """Handle 'who am I?' / 'do you know me?' — compose what she
        knows about the user from her concept network and self-model."""
        user_name = self._self_model.self_knowledge.get("user_name", "")
        content = ""
        fragments: list[tuple[str, str]] = []
        if user_name:
            thought = self._composer.compose_about(user_name, emotion, depth=2)
            if thought and thought.confidence > 0.3:
                content = thought.content
        if not content:
            # Fall back to the relationship record — what she's
            # experienced with this person so far. Notes arrive as
            # clause fragments so the language engine frames them.
            notes = list(self._self_model.relationship_notes)
            if notes:
                fragments = [("clause", n) for n in notes[-3:]]
            else:
                fragments = [("pred", "am still learning who you are")]
            content = "who you are"
        return Thought(
            content=content,
            intent="self_report",
            emotion=emotion.label,
            self_reflection=True,
            confidence=0.8,
            metadata={"field": "user", "self_fragments": fragments},
        )

    def self_inquiry_creator(self, emotion: EmotionalState) -> Thought:
        """Handle creator-related questions ('who made you?', etc.)."""
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

    # ─── Think, know, story ──────────────────────────────────────

    def self_inquiry_think_story(
        self, lower: str, emotion: EmotionalState
    ) -> Thought:
        """Handle think/know, story, learned questions, and default fallback."""
        if any(w in lower for w in ("think", "know", "want", "care")):
            fragments = self._self_composer.self_reflection_fragments(
                self._self_model, self._network, self._reflection, emotion
            )
            return Thought(
                content="what I think",
                intent="reflect",
                emotion=emotion.label,
                self_reflection=True,
                confidence=0.65,
                metadata={"self_fragments": fragments},
            )

        if any(w in lower for w in ("story", "history", "life", "past")):
            story = self._narrative.tell_story()
            # tell_story returns a pipe-separated structural summary
            # (name | uptime | chapters | events | values). Its
            # segments arrive as clause fragments so the language
            # engine frames the delivery rather than the summary
            # bypassing composition.
            fragments = [
                ("clause", s.strip(" ."))
                for s in story.split("|")
                if s.strip(" .")
            ] if story else []
            return Thought(
                content="my story",
                intent="self_report",
                emotion=emotion.label,
                self_reflection=True,
                confidence=0.8,
                metadata={"field": "identity", "self_fragments": fragments},
            )

        if any(w in lower for w in ("learned", "experienced", "discovered")):
            fragments = self._self_composer.self_reflection_fragments(
                self._self_model, self._network, self._reflection, emotion
            )
            return Thought(
                content="what I've learned",
                intent="self_report",
                emotion=emotion.label,
                self_reflection=True,
                confidence=0.75,
                metadata={"field": "identity", "self_fragments": fragments},
            )

        fragments = self._self_composer.identity_fragments(
            self._self_model, self._network, emotion
        )
        return Thought(
            content=self._self_model.name or "who I am",
            intent="self_report",
            emotion=emotion.label,
            confidence=0.7,
            metadata={"field": "identity", "self_fragments": fragments},
        )

    # ─── Philosophy ──────────────────────────────────────────────

    def handle_philosophy(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
    ) -> Thought | None:
        """Handle philosophical questions.

        Uses the thought composer to generate responses from her
        actual knowledge, not hardcoded strings. Returns None if she
        can't compose from her understanding — she stays silent rather
        than reciting a pre-written fallback template.
        """
        lower = perception.raw_text.lower()
        result = self._philosophy_from_topics(perception, emotion)
        if result is not None:
            return result
        return self._philosophy_fallback(perception, emotion, lower)

    def _philosophy_from_topics(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Try to compose a philosophical response from known topics.

        When the question involves 2+ topics (e.g., "does memory lead
        to cognition?"), first try compose_answer() which can
        reason about the *relationship* between the concepts — not
        just define each one independently. Fall back to per-topic
        reflection/composition only if relational reasoning doesn't
        produce a result.
        """
        topics = self._resolve_topics(perception.topics, perception.raw_text)
        if not topics:
            topics = perception.topics
        if not topics:
            return None

        # 1. If 2+ topics, try relational reasoning first — this
        #    produces arguments about how concepts relate, not just
        #    definitions of each one.
        if len(topics) >= 2:
            thought = self._composer.compose_answer(
                perception.raw_text, topics, emotion, "what_is"
            )
            if thought and thought.confidence > 0.3:
                return Thought(
                    content=thought.content,
                    intent="philosophize",
                    emotion=emotion.label,
                    topics=topics,
                    self_reflection=True,
                    confidence=thought.confidence,
                    metadata=thought.metadata,
                )

        # 2. Fall back to per-topic reflection/composition
        for topic in topics:
            thought = self._composer.compose_reflection(topic, emotion)
            if thought and thought.confidence > 0.3:
                return thought
        for topic in topics:
            thought = self._composer.compose_about(topic, emotion)
            if thought and thought.confidence > 0.3:
                return Thought(
                    content=thought.content,
                    intent="philosophize",
                    emotion=emotion.label,
                    topics=topics,
                    self_reflection=True,
                    confidence=thought.confidence,
                )
        return None

    def _philosophy_fallback(
        self, perception: Perception, emotion: EmotionalState, lower: str
    ) -> Thought | None:
        """Fallback: compose from philosophical concept seeds or stay silent."""
        fallback_seeds: list[str] = []
        if "meaning" in lower or "purpose" in lower:
            fallback_seeds = ["meaning", "purpose", "value"]
        elif "free will" in lower:
            fallback_seeds = ["free_will", "choice", "determinism"]
        elif "real" in lower or "exist" in lower:
            fragments = self._self_composer.existence_fragments(
                self._self_model, self._network, emotion
            )
            return Thought(
                content="existence",
                intent="philosophize",
                emotion=emotion.label,
                topics=perception.topics,
                self_reflection=True,
                confidence=0.7,
                metadata={"self_fragments": fragments},
            )
        elif "cognition" in lower or "sentient" in lower:
            fallback_seeds = ["cognition", "awareness", "sentience"]
        else:
            fallback_seeds = perception.topics if perception.topics else ["philosophy"]

        for seed in fallback_seeds:
            thought = self._composer.compose_about(seed, emotion)
            if thought and thought.confidence > 0.25:
                return Thought(
                    content=thought.content,
                    intent="philosophize",
                    emotion=emotion.label,
                    topics=perception.topics,
                    self_reflection=True,
                    confidence=thought.confidence,
                )

        return None
