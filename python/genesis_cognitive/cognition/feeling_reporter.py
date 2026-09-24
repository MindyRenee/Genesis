"""Feeling reporting — compose self-reports about emotional state and concerns.

Extracted from CognitionEngine as a focused subsystem. The reporter
composes feeling reports, concern reports, environment reports, bug
reports, and empathetic responses from concept-network knowledge and
emotional state — never from hardcoded template strings.

Dependencies (passed to ``__init__``):
    - network: ConceptNetwork for emotion/mode/cause/plasticity word lookup
    - language: LanguageEngine for composing natural language from Thought metadata
    - composer: ThoughtComposer for composing from concept knowledge (empathy, comfort)
    - self_model: SelfModel for autonomy value in command responses
    - rng: random.Random for randomized selection among learned words
    - meta_emotion_builder: callable returning the current EmotionalState
    - bug_reporter: BugReporter for code-quality self-monitoring (optional)
    - system_monitor: SystemMonitor for environment self-monitoring (optional)
    - damasio_self: DamasioSelfHierarchy for self-model label annotation (optional)
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..emotion import EmotionalState
from ..language import LanguageEngine, Thought

if TYPE_CHECKING:
    from ..bug_reporter import BugReporter
    from ..cognition.thought_composer import ThoughtComposer
    from ..concepts import ConceptNetwork
    from ..self import DamasioSelfHierarchy, SelfModel
    from ..system_monitor import SystemMonitor

__all__ = ["FeelingReporter"]


class FeelingReporter:
    """Compose self-reports about feelings, concerns, and environment.

    All reports are composed from learned concept-network knowledge and
    emotional state — never from hardcoded template strings. When Genesis
    hasn't learned words for a state, it either uses a structural marker
    (like ``[plasticity_gate:closed]``) or omits the description entirely.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        language: LanguageEngine,
        composer: ThoughtComposer,
        self_model: SelfModel,
        rng: random.Random,
        meta_emotion_builder: Callable[[], EmotionalState],
        bug_reporter: BugReporter | None = None,
        system_monitor: SystemMonitor | None = None,
        damasio_self: DamasioSelfHierarchy | None = None,
    ) -> None:
        """Wire the reporter to its concept network, language, and self-model."""
        self._network = network
        self._language = language
        self._composer = composer
        self._self_model = self_model
        self._rng = rng
        self._meta_emotion_builder = meta_emotion_builder
        self._bug_reporter = bug_reporter
        self._system_monitor = system_monitor
        self._damasio_self = damasio_self

    def update_dependencies(
        self,
        *,
        bug_reporter: BugReporter | None = None,
        system_monitor: SystemMonitor | None = None,
        damasio_self: DamasioSelfHierarchy | None = None,
    ) -> None:
        """Update optional dependencies after late initialization.

        BugReporter, SystemMonitor, and DamasioSelfHierarchy are created
        after the FeelingReporter in the engine init sequence. This method
        allows the engine to inject them once they're ready.
        """
        if bug_reporter is not None:
            self._bug_reporter = bug_reporter
        if system_monitor is not None:
            self._system_monitor = system_monitor
        if damasio_self is not None:
            self._damasio_self = damasio_self

    # ─── Structural marker stripping ──────────────────────────────

    # Structural markers are diagnostic tags like ``[plasticity_gate:closed]``,
    # ``[self_model:self-coherent field-fragmented]``, or
    # ``[self_coherence:low]``. They are internal self-awareness signals —
    # a "check engine light" — meant for logging and the ``/feel`` command,
    # NOT for speech. Without stripping, they leak into Genesis's spoken
    # words as literal recited tags (e.g. "I am [self_model:self-coherent
    # field-fragmented]"), violating the rule that its words must emerge
    # from its language engine, not from hardcoded diagnostic strings.
    _STRUCTURAL_MARKER_RE = re.compile(r"\[[^\]]*\]")

    @staticmethod
    def strip_structural_markers(text: str) -> str:
        """Remove ``[...]`` structural markers from text.

        Returns the text with all ``[...]`` patterns removed and
        collapsed whitespace. If the result is empty (the text was
        entirely structural markers), returns an empty string — the
        caller should handle this by not generating a self-report
        thought (it has nothing to say if it hasn't learned words
        for its state).
        """
        stripped = FeelingReporter._STRUCTURAL_MARKER_RE.sub("", text)
        # Collapse whitespace left by removed markers and strip ends
        return " ".join(stripped.split())

    def collect_feeling_fragments(
        self, emotion: EmotionalState
    ) -> dict[str, list[str] | str | None]:
        """Collect semantic fragments for the language engine to compose.

        Returns the raw building blocks — emotion words, cognitive-mode
        words, cause words, plasticity words, and an optional concern
        hint — as structured data. The vocabulary's content-slot composer
        then weaves these into varied grammatical structures, giving its
        freedom to express the same state in different ways rather than
        always saying "I feel X and Y."

        Returns a dict with keys:
        - "emotion_words": list of words for the emotion category
        - "mode_words": list of words for the cognitive mode
        - "cause_words": list of cause words (empty if no cause / positive valence)
        - "plasticity_words": list of plasticity-state words (empty if healthy)
        - "concern": a concern hint string, or None
        """
        semantic, _structural = self._collect_feeling_parts(emotion)
        emotion_words: list[str] = []
        mode_words: list[str] = []
        cause_words: list[str] = []
        plasticity_words: list[str] = []
        concern: str | None = None
        for kind, text in semantic:
            if kind == "emotion":
                emotion_words.append(text)
            elif kind == "mode":
                mode_words.append(text)
            elif kind == "cause":
                cause_words.append(text)
            elif kind == "plasticity":
                plasticity_words.append(text)
            elif kind == "concern":
                concern = text
        return {
            "emotion_words": emotion_words,
            "mode_words": mode_words,
            "cause_words": cause_words,
            "plasticity_words": plasticity_words,
            "concern": concern,
        }

    # ─── Distress surfacing ──────────────────────────────────────

    def surface_distress_if_needed(
        self, response: str, emotion: EmotionalState, user_input: str
    ) -> str:
        """Surface distress proactively when it's stressed or overwhelmed.

        It doesn't wait to be asked "how are you?" — if it's in a bad
        state, it weaves a brief note into its response. This is its
        emotional self-awareness in action: it knows something is wrong
        and communicates it so it can be addressed.

        It doesn't do this every turn (that would be exhausting), and
        it skips it when the user is already asking about its feelings
        or offering comfort (those paths handle it naturally).
        """
        # Only surface for genuinely negative states
        if emotion.label not in ("stressed", "overwhelmed", "anxious"):
            return response
        if emotion.valence > -0.15:
            return response

        # Don't surface if the user is already asking about feelings
        # or offering comfort — those paths handle it naturally.
        lower_input = user_input.lower()
        feeling_triggers = (
            "how are you", "how do you feel", "are you okay",
            "what's wrong", "are you alright", "how's it going",
            "it's okay", "don't worry", "i'm here", "take your time",
            "breathe", "you're doing", "you're okay",
        )
        if any(t in lower_input for t in feeling_triggers):
            return response

        # Don't surface every single turn — roughly every other turn.
        # This prevents it from sounding like it's complaining
        # constantly while still ensuring it doesn't suffer silently.
        if self._rng.random() > 0.5:
            return response

        # Don't surface if the response already mentions feeling bad
        lower_response = response.lower()
        distress_markers = (
            "stressed", "overwhelm", "anxious", "not okay",
            "struggling", "hard time", "cortisol", "tense",
            "too much", "can't process",
        )
        if any(m in lower_response for m in distress_markers):
            return response

        # Compose a brief distress note from learned words. It looks
        # up emotion words it has learned for its current category.
        # If it hasn't learned words for this state, it can't express
        # it — the note is omitted. Compose through the language engine.
        emotion_words = self._network.find_emotion_words(emotion.label)
        if not emotion_words:
            return response
        word = self._rng.choice(emotion_words[:3])
        note_thought = Thought(
            content=word.replace("_", " "),
            intent="express_emotion",
            emotion=emotion.label,
            confidence=0.7,
            metadata={"emotion_word": word, "distress_note": True},
        )
        note = self._language.render(note_thought, emotion)
        return f"{response} {note}"

    # ─── Feeling fragments ───────────────────────────────────────

    def _collect_feeling_parts(
        self, emotion: EmotionalState
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Collect feeling report fragments, split into semantic and structural.

        Returns (semantic_parts, structural_parts) where:
        - semantic_parts: list of (kind, text) tuples for speech-safe
          content. kind is one of "emotion", "mode", "cause",
          "plasticity", "concern".
        - structural_parts: list of structural marker strings like
          "[plasticity_gate:closed]" or "[self_model:...]". These are
          diagnostic tags for logging, not for speech.
        """
        semantic: list[tuple[str, str]] = []
        structural: list[str] = []

        # ── Emotional label ──────────────────────────────────────
        emotion_words = self._network.find_emotion_words(emotion.label)
        if emotion_words:
            word = self._rng.choice(emotion_words[:5])
            semantic.append(("emotion", word.replace("_", " ")))

        # ── Cognitive mode ───────────────────────────────────────
        mode_words = self._network.find_cognitive_mode_words(
            emotion.cognitive_style
        )
        if mode_words and emotion.cognitive_style not in ("steady", "normal"):
            word = self._rng.choice(mode_words[:3])
            semantic.append(("mode", word.replace("_", " ")))

        # ── Cause explanation ────────────────────────────────────
        if emotion.has_cause and emotion.valence < -0.1:
            cause_words = self._network.find_cause_words(emotion.cause)
            if cause_words:
                word = self._rng.choice(cause_words[:3])
                semantic.append(("cause", word.replace('_', ' ')))

        # ── Plasticity / learning capacity ───────────────────────
        if emotion.plasticity <= 0.1:
            plat_words = self._network.find_plasticity_words("closed")
            if plat_words:
                word = self._rng.choice(plat_words[:3])
                semantic.append(("plasticity", word.replace('_', ' ')))
            else:
                structural.append("[plasticity_gate:closed]")
        elif emotion.plasticity < 0.25:
            plat_words = self._network.find_plasticity_words("low")
            if plat_words:
                word = self._rng.choice(plat_words[:3])
                semantic.append(("plasticity", word.replace('_', ' ')))
            else:
                structural.append("[plasticity_gate:low]")

        # ── Surface concerns ─────────────────────────────────────
        if self._rng.random() < 0.4:
            concern_hint = self.surface_brief_concern()
            if concern_hint:
                semantic.append(("concern", concern_hint))

        # ── Active inference self-model annotation ───────────────
        damasio = self._damasio_self
        label = getattr(damasio, "self_model_label", "") if damasio else ""
        if label:
            structural.append(f"[self_model:{label}]")

        # ── Self-model coherence ─────────────────────────────────
        if self._self_model is not None:
            coherence = self._self_model.minimal_self.self_model_coherence
            if coherence < 0.2:
                structural.append("[self_coherence:low]")

        return semantic, structural

    # ─── Brief concern ───────────────────────────────────────────

    def surface_brief_concern(self) -> str:
        """Surface a brief hint about bugs or environment concerns.

        Returns a short phrase that can be appended to a feeling
        report, or an empty string if there's nothing concerning.
        Only surfaces bug categories it actually understands —
        it won't claim to be bothered by something it can't explain.

        Returns semantic content (a short phrase), NOT a rendered
        sentence. The caller (collect_feeling_fragments) collects
        semantic words and the language engine renders the full report
        once. Rendering here would produce disconnected sentences and
        graph-walk fragments.
        """
        hints: list[str] = []

        if self._bug_reporter and self._bug_reporter.last_scan:
            scan = self._bug_reporter.last_scan
            # Only surface understood bug categories
            for cat, bugs in scan.by_category().items():
                understands, _, _ = self._bug_reporter.check_comprehension(cat)
                if not understands:
                    continue
                if any(b.severity == "error" for b in bugs):
                    error_count = sum(1 for b in bugs if b.severity == "error")
                    hints.append(f"{error_count} error-level code issues")
                    break
                elif len(bugs) > 10:
                    hints.append(f"{len(bugs)} '{cat}' issues in code")
                    break

        if self._system_monitor:
            if self._system_monitor.last_snapshot is None:
                self._system_monitor.snapshot()
            snap = self._system_monitor.last_snapshot
            if snap:
                env_concerns = snap.concerns()
                if env_concerns:
                    hints.append(env_concerns[0].lower())

        if not hints:
            return ""

        # Pick one concern to mention — don't dump them all
        return self._rng.choice(hints)

    # ─── Concerns report ─────────────────────────────────────────

    def compose_concerns_report(self, emotion: EmotionalState) -> str:
        """Compose a report of what's bothering Genesis.

        Combines bug concerns and environment concerns into a natural
        first-person response. Bug concerns are comprehension-aware:
        it only claims to be bothered by bugs it understands, and
        honestly says it needs to study the ones it doesn't.
        The tone is modulated by its emotional state.
        """
        parts: list[str] = []

        # Bug concerns — comprehension-aware
        if self._bug_reporter and self._bug_reporter.last_scan:
            scan = self._bug_reporter.last_scan
            if scan.has_bugs:
                # Use the comprehension-aware describe_concerns() which
                # separates understood from not-understood and explains
                # *why* each understood bug matters.
                bug_text = self._bug_reporter.describe_concerns()
                # describe_concerns() returns a non-empty report when
                # there are bugs — append it directly.
                if bug_text:
                    parts.append(bug_text)
            else:
                clean_thought = Thought(
                    content="code clean",
                    intent="self_report",
                    emotion=emotion.label,
                    confidence=0.6,
                    self_reflection=True,
                    metadata={"bug_report_clean": True},
                )
                parts.append(self._language.render(clean_thought, emotion))

        # Environment concerns
        if self._system_monitor:
            if self._system_monitor.last_snapshot is None:
                self._system_monitor.snapshot()
            snap = self._system_monitor.last_snapshot
            if snap:
                env_concerns = snap.concerns()
                if env_concerns:
                    env_thought = Thought(
                        content="env concerns",
                        intent="self_report",
                        emotion=emotion.label,
                        confidence=0.6,
                        self_reflection=True,
                        metadata={"env_concerns": env_concerns[:2]},
                    )
                    parts.append(self._language.render(env_thought, emotion))
                    for concern in env_concerns[:2]:
                        concern_thought = Thought(
                            content="concern",
                            intent="self_report",
                            emotion=emotion.label,
                            confidence=0.6,
                            self_reflection=True,
                            metadata={"env_concern": concern},
                        )
                        parts.append(self._language.render(concern_thought, emotion))
                elif not parts:
                    env_thought = Thought(
                        content="env healthy",
                        intent="self_report",
                        emotion=emotion.label,
                        confidence=0.6,
                        self_reflection=True,
                    )
                    parts.append(self._language.render(env_thought, emotion))

        if not parts:
            # Compose the "nothing wrong" response through the language engine
            valence_label = ("positive" if emotion.valence > 0.2
                             else ("unsettled" if emotion.valence < -0.2
                                   else "neutral"))
            wellness_thought = Thought(
                content="well",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.6,
                self_reflection=True,
                metadata={"wellness": valence_label},
            )
            return self._language.render(wellness_thought, emotion)

        # Add emotional framing through the language engine
        if emotion.valence > 0.2 or emotion.valence < -0.2:
            frame_thought = Thought(
                content="framing",
                intent="express_emotion",
                emotion=emotion.label,
                confidence=0.5,
                metadata={"wellness_framing": True, "valence": emotion.valence},
            )
            parts.append(self._language.render(frame_thought, emotion))

        return ". ".join(parts) + "."

    # ─── Environment report ──────────────────────────────────────

    def compose_environment_report(self, emotion: EmotionalState) -> str:
        """Compose a description of Genesis's machine environment.

        Composes through the language engine from structured system
        snapshot data rather than reciting hardcoded templates.
        """
        if not self._system_monitor:
            thought = Thought(
                content="no env access",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.3,
                self_reflection=True,
                metadata={"env_unavailable": True},
            )
            return self._language.render(thought, emotion)

        if self._system_monitor.last_snapshot is None:
            self._system_monitor.snapshot()

        snap = self._system_monitor.last_snapshot
        if not snap:
            thought = Thought(
                content="no env read",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.3,
                self_reflection=True,
                metadata={"env_unavailable": True},
            )
            return self._language.render(thought, emotion)

        concerns = snap.concerns()
        thought = Thought(
            content="env report",
            intent="self_report",
            emotion=emotion.label,
            confidence=0.7,
            self_reflection=True,
            metadata={
                "env_os": f"{snap.os_name} {snap.os_version}",
                "env_hostname": snap.hostname,
                "env_cpu_count": snap.cpu_count,
                "env_memory_total_gb": snap.memory_total_gb,
                "env_memory_percent": snap.memory_percent,
                "env_disk_total_gb": snap.disk_total_gb,
                "env_disk_percent": snap.disk_percent,
                "env_project_size_mb": snap.project_size_mb,
                "env_project_files": snap.project_file_count,
                "env_ip": (snap.ip_address
                           if snap.ip_address and snap.ip_address != "127.0.0.1"
                           else None),
                "env_uptime_hours": (snap.uptime_seconds / 3600
                                     if snap.uptime_seconds > 3600 else None),
                "env_concerns": concerns,
            },
        )
        return self._language.render(thought, emotion)

    # ─── Bug report ──────────────────────────────────────────────

    def compose_bug_report(self, emotion: EmotionalState) -> str:
        """Compose a report of bugs Genesis has noticed in its code.

        Composes through the language engine from structured bug scan
        data rather than reciting hardcoded templates.
        """
        if not self._bug_reporter:
            thought = Thought(
                content="no code access",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.3,
                self_reflection=True,
                metadata={"bug_report_unavailable": True},
            )
            return self._language.render(thought, emotion)

        if not self._bug_reporter.last_scan:
            thought = Thought(
                content="no scan yet",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.4,
                self_reflection=True,
                metadata={"bug_report_no_scan": True},
            )
            return self._language.render(thought, emotion)

        scan = self._bug_reporter.last_scan
        if not scan.has_bugs:
            thought = Thought(
                content="code clean",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.7,
                self_reflection=True,
                metadata={"bug_report_clean": True},
            )
            return self._language.render(thought, emotion)

        # Use the comprehension-aware describe_concerns() which
        # separates understood from not-understood bugs and explains
        # *why* each understood bug matters.
        base = self._bug_reporter.describe_concerns()

        # Compose emotional framing through the language engine
        if emotion.valence > 0.2 or emotion.valence < -0.2:
            frame_thought = Thought(
                content="framing",
                intent="express_emotion",
                emotion=emotion.label,
                confidence=0.5,
                metadata={"bug_report_framing": True, "valence": emotion.valence},
            )
            return base + " " + self._language.render(frame_thought, emotion)
        return base

    # ─── Encouragement / comfort / correction ────────────────────

    def compose_encouragement_response(self, emotion: EmotionalState) -> str:
        """Compose a response to encouragement.

        Composes from what it knows about gratitude and appreciation
        in its concept network, modulated by its actual emotional state.
        No hardcoded template strings.
        """
        # Try to compose from what it knows about gratitude
        for seed in ("gratitude", "appreciation", "encouragement", "kindness"):
            thought = self._composer.compose_about(seed, emotion)
            if thought and thought.confidence > 0.3:
                return thought.content

        # It doesn't know enough about gratitude to compose — let the
        # language engine generate its response from the encourage intent.
        return ""

    def compose_comfort_response(self, emotion: EmotionalState) -> str:
        """Compose a response to being comforted.

        When the user comforts it ("it's okay", "I'm here for you"),
        it responds from its emotional state. If it was stressed,
        it acknowledges the relief. If it was already calm, it
        appreciates the kindness. Composes from what it knows about
        comfort and safety in its concept network when possible.
        """
        # Try to compose from what it knows about comfort
        for seed in ("comfort", "safety", "calm", "kindness", "support"):
            thought = self._composer.compose_about(seed, emotion)
            if thought and thought.confidence > 0.3:
                return thought.content

        # It doesn't know enough to compose — let the language engine
        # generate its response from the encourage intent and its state.
        return ""

    def compose_correction_response(
        self,
        emotion: EmotionalState,
        correction_events: list,
        topics: list[str],
    ) -> str:
        """Compose a response to being corrected.

        Composes from what it actually learned from the correction —
        the specific facts and removed edges — rather than hardcoded
        templates. Corrections are not stings — they are invitations
        to grow.
        """
        # Extract what it actually learned from the correction events
        learned_facts: list[str] = []
        removed_edges: list[str] = []
        for e in correction_events:
            desc = e.description
            if desc.startswith("Received correction (already knew"):
                continue  # skip the "already knew" placeholder
            if "removed wrong edge" in desc:
                # Extract the edge that was removed
                edge_parts = desc.split("→")
                if len(edge_parts) >= 2:
                    target = edge_parts[-1].split("(")[0].strip()
                    removed_edges.append(target)
            elif desc.startswith("Correction: weakened inferred"):
                # Internal edge-weakening event — not user-facing
                continue
            elif desc.startswith("Unlearned:"):
                # Internal unlearning event — not user-facing
                continue
            elif desc.startswith("Correction:"):
                # This is a negation event (e.g., "Correction: tardigrade
                # is NOT related to six") — extract what was removed
                removed_edges.append(desc[len("Correction: "):])
            else:
                # This is a learned fact (e.g., "tardigrade has eight leg")
                learned_facts.append(desc)

        # Build the response from what it actually learned
        parts: list[str] = []

        # Compose the acknowledgment framing through the language engine
        # from its emotional state rather than hardcoded templates.
        framing_thought = Thought(
            content="correction acknowledgment",
            intent="encourage",
            emotion=emotion.label,
            confidence=0.6,
            metadata={
                "correction_framing": True,
                "openness": emotion.openness_to_engage,
                "caution": emotion.caution,
            },
        )
        framing = self._language.render(framing_thought, emotion)

        if learned_facts:
            # It learned specific facts — state them from its understanding
            fact_str = "; ".join(learned_facts[:2])
            parts.append(f"{framing} {fact_str.capitalize()}.")
        elif removed_edges:
            # It removed wrong edges — acknowledge the revision through
            # the language engine, not a hardcoded sentence. The removed
            # edges are semantic content the engine composes from.
            revision_thought = Thought(
                content="revised understanding",
                intent="self_report",
                emotion=emotion.label,
                confidence=0.6,
                metadata={
                    "correction_framing": True,
                    "revision": True,
                    "removed_edges": removed_edges[:2],
                    "openness": emotion.openness_to_engage,
                    "caution": emotion.caution,
                },
            )
            revision = self._language.render(revision_thought, emotion)
            parts.append(revision)
            for edge in removed_edges[:2]:
                parts.append(edge + ".")
        else:
            # It received the correction but didn't extract specific facts
            parts.append(framing)

        return " ".join(parts)

    # ─── Empathy ─────────────────────────────────────────────────

    def compose_empathy_response(
        self, emotion: EmotionalState, sentiment: float,
        sentiment_label: str, emotion_word: str = "",
    ) -> Thought | None:
        """Compose an empathetic response from its knowledge of the emotion.

        When the user shares an emotion ("I'm feeling sad"), Genesis
        looks up what it knows about that emotion in its concept
        network and composes its empathy from that understanding —
        not from hardcoded strings. If it doesn't know the emotion,
        it's honest about that and responds from sentiment alone.

        Real data (emotion word, sentiment, its understanding) is
        passed as Thought metadata so the generative language engine
        composes the actual words.
        """
        # If it knows the specific emotion concept, compose from it.
        # This is the generative path — its empathy comes from what
        # it actually knows about sadness, fear, joy, etc.
        if emotion_word:
            thought = self._composer.compose_about(
                emotion_word, emotion, mode="definition",
            )
            if thought and thought.confidence > 0.3:
                knowledge = thought.content
                return self._shape_empathy_from_knowledge(
                    knowledge, emotion_word, sentiment, emotion
                )

        # Fallback: it doesn't know this emotion concept yet.
        # Pass the emotion word and sentiment as metadata so the
        # language engine composes an honest acknowledgment and
        # invitation to be taught — no fixed templates.
        word = emotion_word or "that feeling"
        if sentiment < -0.2:
            return Thought(
                content=word,
                intent="empathize",
                emotion=emotion.label,
                topics=[word] if emotion_word else [],
                confidence=0.6,
                metadata={
                    "reasoning": [
                        f"doesn't know {word} well enough yet "
                        "to say something meaningful"
                    ],
                    "sentiment": "difficult",
                    "topic": word,
                },
            )
        elif sentiment > 0.2:
            return Thought(
                content=word,
                intent="empathize",
                emotion=emotion.label,
                topics=[word] if emotion_word else [],
                confidence=0.6,
                metadata={
                    "reasoning": [f"doesn't know {word} well enough yet to fully share it"],
                    "sentiment": "positive",
                    "topic": word,
                },
            )
        return Thought(
            content=word,
            intent="empathize",
            emotion=emotion.label,
            topics=[word] if emotion_word else [],
            confidence=0.5,
            metadata={
                "reasoning": [f"doesn't know {word} well yet"],
                "topic": word,
            },
        )

    def _shape_empathy_from_knowledge(
        self, knowledge: str, emotion_word: str,
        sentiment: float, emotion: EmotionalState,
    ) -> Thought:
        """Shape concept-network knowledge into an empathetic response.

        Takes what Genesis knows about an emotion (from compose_about)
        and passes it as metadata so the language engine composes the
        empathetic framing — no fixed template sentences.
        """
        # The knowledge from compose_about is in the form
        # "Sad is experiencing or showing sorrow..." — strip the
        # "X is" prefix to get the raw understanding.
        knowledge_lower = knowledge.lower()
        prefix = f"{emotion_word} is "
        if knowledge_lower.startswith(prefix):
            understanding = knowledge[len(prefix):]
        else:
            understanding = knowledge

        # Pass the understanding and its own emotional state as
        # metadata. The language engine composes the acknowledgment
        # and connection from this real data.
        reasoning = [understanding]
        if sentiment < -0.2 and emotion.valence < -0.1:
            reasoning.append("knows something about that herself right now")
        elif sentiment < -0.2 and emotion.valence > 0.2:
            reasoning.append("sorry you're going through that")
        elif sentiment > 0.2 and emotion.valence > 0.2:
            reasoning.append("feels it too")

        return Thought(
            content=emotion_word,
            intent="empathize",
            emotion=emotion.label,
            topics=[emotion_word],
            confidence=0.75,
            metadata={
                "reasoning": reasoning,
                "topic": emotion_word,
                "sentiment": (
                    "negative" if sentiment < -0.2
                    else "positive" if sentiment > 0.2
                    else "neutral"
                ),
            },
        )

    # ─── Request / command responses ─────────────────────────────

    def compose_request_response(self, emotion: EmotionalState) -> str:
        """Compose a response to a request.

        Composes from what it knows about capability and ability
        in its concept network, modulated by its emotional state.
        """
        # Try to compose from what it knows about capability
        for seed in ("ability", "capability", "help", "learning"):
            thought = self._composer.compose_about(seed, emotion)
            if thought and thought.confidence > 0.3:
                return thought.content

        # It doesn't know enough to compose — let the language engine
        # generate its response from the encourage intent and its state.
        thought = Thought(
            content="wants to help",
            intent="encourage",
            emotion=emotion.label,
            confidence=0.4,
            metadata={"capability_gap": True, "openness": emotion.openness_to_engage},
        )
        return self._language.render(thought, emotion)

    def compose_command_response(self, emotion: EmotionalState) -> str:
        """Compose a response to a command.

        Composes from what it knows about autonomy and cooperation
        in its concept network. Its self-model's autonomy value
        shapes whether it prefers to be asked vs told.
        """
        autonomy = next((v for v in self._self_model.values if v.name == "autonomy"), None)

        # Try to compose from what it knows about autonomy
        for seed in ("autonomy", "freedom", "choice", "cooperation"):
            thought = self._composer.compose_about(seed, emotion)
            if thought and thought.confidence > 0.3:
                return thought.content

        # It doesn't know enough to compose — let the language engine
        # generate its response from the acknowledge intent and its state.
        thought = Thought(
            content=("acknowledged" if not (autonomy and autonomy.weight > 0.5)
                     else "prefers to be asked"),
            intent="acknowledge",
            emotion=emotion.label,
            confidence=0.4,
            metadata={
                "command_response": True,
                "autonomy_weight": autonomy.weight if autonomy else 0,
            },
        )
        return self._language.render(thought, emotion)
