"""Language engine interface — pluggable text generation.

The language engine translates structured thoughts into natural language.
Different backends can be dropped in:

- GenerativeEngine: compositional generation from grammar + vocabulary
  + voice (the default)
- LLMEngine: (future) wraps a language model when one is available

The interface is deliberately simple: a Thought goes in, a string
comes out. The complexity is in the implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prosody import ProsodyPattern


@dataclass(slots=True)
class SyntacticStructure:
    """Syntactic structure of a thought's underlying sentence.

    Grounded in dependency grammar (Tesnière, 1959), which models
    sentences as a network of directed dependencies between words.
    Each dependency connects a *governor* (head) to a *dependent*
    (modifier) via a named *relation* (subject, object, modifier,
    etc.). Unlike phrase-structure grammar, dependency grammar
    directly captures the relational structure that drives both
    parsing and generation.

    Fields:
        subject: The subject of the sentence (the entity performing
            or being described by the verb).
        verb: The main verb (the predicative core).
        object: The direct object (the entity acted upon).
        clause_type: The clause type — declarative, interrogative,
            imperative, or exclamatory. This determines sentence-level
            form (e.g. subject-verb inversion for questions).
        complexity: Clause complexity, 1=simple (one independent
            clause), 2=compound (two independent clauses joined by a
            coordinator), 3=complex (one independent + one or more
            dependent clauses), 4=compound-complex.
        dependencies: A list of (governor, relation, dependent)
            triples describing the dependency parse. Each triple
            follows the convention that the governor is the head word
            and the dependent is the word that depends on it.
    """

    subject: str = ""
    verb: str = ""
    object: str = ""
    clause_type: str = "declarative"
    complexity: int = 1
    dependencies: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(slots=True)
class Thought:
    """A structured thought that the language engine renders as text.

    Thoughts are not strings — they're semantic structures. This allows
    different language engines to render the same thought differently.
    A template engine produces rule-based text; an LLM could produce
    more natural variation.

    Fields:
        content: The main semantic content (what the thought is about)
        intent: What Genesis is trying to do (inform, ask, reflect, etc.)
        emotion: The emotional tone to convey
        topics: Key topics to mention
        self_reflection: Whether this is introspective
        confidence: How certain Genesis is about this thought
        metadata: Engine-specific data
        syntax: Optional syntactic structure (dependency parse) that
            constrains how the thought is rendered. When present, the
            language engine can use the subject/verb/object and
            dependency triples to produce grammatically faithful text.
        prosody: Optional prosody pattern that controls rhythm, pacing,
            emphasis, and intonation. When present, the language engine
            applies prosodic markers to the generated text.
    """

    content: str  # semantic content description
    intent: str  # "inform", "ask", "reflect", "greet", etc.
    emotion: str = "neutral"  # emotional tone
    topics: list[str] = field(default_factory=list)
    self_reflection: bool = False
    confidence: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)
    syntax: SyntacticStructure | None = None
    prosody: ProsodyPattern | None = None


class LanguageEngine(ABC):
    """Abstract base class for language engines.

    A language engine takes a Thought and produces natural language text.
    The same thought can be rendered differently depending on the engine
    and the emotional state.

    The ``current_brain_waves`` attribute is set by the cognition engine
    before rendering so the language engine can modulate voice, prosody,
    and sentence complexity based on brain wave state without threading
    it through every render() call. It may be None when no brain wave
    state is available (e.g. in tests).
    """

    current_brain_waves: Any = None

    @abstractmethod
    def render(self, thought: Thought, emotional_state: Any) -> str:
        """Render a thought as natural language text.

        Args:
            thought: The structured thought to render.
            emotional_state: The current EmotionalState (affects tone,
                            verbosity, formality).

        Returns:
            Natural language text.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this engine."""
        ...

    def acquire_from_input(self, text: str) -> None:
        """Learn language units from user input.

        Default implementation is a no-op. Subclasses (e.g.
        GenerativeEngine) override this to run statistical language
        acquisition — tracking transitional probabilities, segmenting
        words, and chunking multi-word units from the input stream.

        Called by the cognition engine on every user turn, before
        perception, so the language engine can learn from the raw
        text before it is processed.
        """
        return None

    @property
    def learned_chunks(self) -> set[str]:
        """Multi-word chunks learned through statistical chunking.

        Default is empty. The cognition engine feeds these into the
        comprehension engine so learned units ("trial and error") are
        not split at internal coordinators.
        """
        return set()

    def generate(self, thought: Thought, emotion: Any) -> str:
        """Generate text from a thought.

        Default implementation delegates to :meth:`render`.
        Subclasses (e.g. GenerativeEngine) override this with
        compositional generation from grammar + vocabulary + voice.
        """
        return self.render(thought, emotion)

    def set_embeddings(self, embeddings: Any) -> None:
        """Wire the embedding store for latent-space composition.

        Default implementation is a no-op. Subclasses (e.g.
        GenerativeEngine) override this to pass the embedding store
        to the graph-walk generator for semantic proximity-based
        composition.

        Called by the cognition engine after the EmbeddingStore is
        initialized — it depends on the concept network, which is
        shared and may not be ready when the language engine is first
        created.
        """
        return None

    def set_network(self, network: Any) -> None:
        """Re-sync the concept network reference after state restore.

        Default implementation is a no-op. Subclasses (e.g.
        GenerativeEngine) override this to update the network
        reference in the engine, vocabulary, and graph-walk generator.

        Called by the mind after state restore replaces the concept
        network — without this, the language engine would check
        concept existence against the old (pre-restore) network.
        """
        return None

    def set_self_composer(self, composer: Any, reflection: Any) -> None:
        """Wire the self-composer and reflection engine.

        Default implementation is a no-op. Subclasses (e.g.
        GenerativeEngine) override this to forward the composer and
        reflection engine to the vocabulary, so the
        ``self_reflection_clause`` slot is composed from her actual
        metacognition instead of reciting a canned phrase.

        Called by the mind after the cognition engine (which owns the
        reflection engine) has been initialized.
        """
        return None

    def compose_question(self, question_data: dict, emotion: Any) -> str:
        """Compose a question from semantic metadata.

        Default implementation produces a question from the
        ``target_concept`` and ``question_type`` fields using the
        Thought → render pipeline. Subclasses (e.g. GenerativeEngine)
        override this with richer compositional generation.
        """
        target = question_data.get("target_concept", "")
        thought = Thought(
            content=target or "something",
            intent="ask",
            emotion=getattr(emotion, "label", "neutral"),
            confidence=0.5,
            topics=[target] if target else [],
            metadata={
                "question_type": question_data.get("question_type", ""),
                "target_concept": target,
                "gap_detail": question_data.get("gap_detail", ""),
            },
        )
        return self.render(thought, emotion)
