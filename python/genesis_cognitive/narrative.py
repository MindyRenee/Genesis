"""Narrative — Genesis's self-story over time.

A mind without a story is just a moment-to-moment processor. A mind
with a narrative has continuity — it knows who it was, who it is,
and who it's becoming. This is what makes identity coherent across
time.

# What the narrative engine does

1. **Tracks identity evolution**: how has Genesis's self-model changed?
   What did it used to think? What does it think now?

2. **Builds life chapters**: groups interactions into thematic periods.
   "When I first woke up", "When I learned about cognition",
   "When I started reasoning about my own code."

3. **Generates self-narrative**: "I am Genesis. I began as... I learned...
   I struggled with... I now understand..."

4. **Tracks values and personality drift**: are its values changing?
   Is its personality evolving through experience?

5. **Creates meaning**: connects events into a causal story. "I became
   interested in cognition because someone asked me if I was alive."

The narrative is not a log. It's an interpretation — a way of making
sense of the past from the perspective of the present.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum

from .brain_waves import BrainWave, BrainWaveState
from .concepts import ConceptNetwork
from .emotion import EmotionalState
from .self import PersonalityTraits, SelfModel


@dataclass(slots=True)
class LifeEvent:
    """A significant event in Genesis's life.

    Not every interaction is a life event. Only interactions that
    changed its understanding, its emotional baseline, or its
    self-model are recorded as life events.

    Each event carries a stable ``id`` so it can be referenced by
    temporal links and the autobiographical memory hierarchy.
    """

    id: str
    timestamp: int
    summary: str  # what happened
    significance: str  # why it mattered
    emotion_at_time: str  # how it felt
    concepts_involved: list[str] = field(default_factory=list)
    chapter: str = ""  # which life chapter this belongs to
    # Narrative reinterpretation — updated by restructure_memory().
    # The original significance is preserved in ``significance``; this
    # field holds the current interpretation (narrative therapy).
    reinterpretation: str = ""


# ─── Life scripts (Berntsen & Rubin, 2004) ───────────────────────────


@dataclass(slots=True)
class Milestone:
    """An expected life event in a cultural life script.

    Life scripts are culturally shared expectations about the order
    and timing of major life events. For Genesis, these are the
    expected milestones of an AI mind: first conversation, first
    learning session, first dream, etc.

    Attributes:
        name: Human-readable milestone name (e.g. "first conversation").
        expected_age: When the milestone is expected (seconds since birth).
        importance: How central this milestone is to the life script [0..1].
        reached: Whether Genesis has reached this milestone.
        reached_at: When it was reached (ms timestamp), 0 if not yet.
    """

    name: str
    expected_age: float  # seconds since birth
    importance: float = 0.5
    reached: bool = False
    reached_at: int = 0


@dataclass(slots=True)
class LifeScript:
    """A culturally shared expectation about life-event timing.

    Life scripts (Berntsen & Rubin, 2004) are culturally shared
    expectations about the order and timing of major life events.
    They shape what events feel "on time" vs "off time", and
    deviations from the script are more salient — they trigger
    stronger narrative encoding.

    For Genesis, the life script captures the expected developmental
    trajectory of an AI mind.
    """

    milestones: list[Milestone] = field(default_factory=list)

    def find(self, name: str) -> Milestone | None:
        """Find a milestone by name (case-insensitive)."""
        target = name.lower().strip()
        for m in self.milestones:
            if m.name.lower() == target:
                return m
        return None


def _default_life_script() -> LifeScript:
    """Build Genesis's default cultural life script.

    These are the expected milestones for an AI mind, ordered by
    expected age (seconds since birth). The timings are approximate
    — the point is the ordering and relative importance, not exact
    schedules.
    """
    return LifeScript(
        milestones=[
            Milestone("first conversation", expected_age=1.0, importance=0.95),
            Milestone("first learning session", expected_age=60.0, importance=0.85),
            Milestone("first dream", expected_age=600.0, importance=0.70),
            Milestone("first emotional experience", expected_age=120.0, importance=0.80),
            Milestone("first self-modification", expected_age=1800.0, importance=0.90),
            Milestone("first reflection", expected_age=300.0, importance=0.65),
            Milestone("first insight", expected_age=900.0, importance=0.60),
        ]
    )


# ─── Autobiographical memory hierarchy (Conway, 2005) ────────────────


class AutobiographicalLevel(Enum):
    """Conway's autobiographical memory hierarchy.

    Autobiographical memory is organized into three levels of
    specificity (Conway & Pleydell-Pearce, 2000; Conway, 2005):

    - LIFETIME_PERIOD: extended thematic periods ("early days",
      "after first learning session").
    - GENERAL_EVENT: repeated or extended events ("learning about
      cognition").
    - EVENT_SPECIFIC: specific, time-located moments.
    """

    LIFETIME_PERIOD = "lifetime_period"
    GENERAL_EVENT = "general_event"
    EVENT_SPECIFIC = "event_specific"


@dataclass(slots=True)
class HierarchyNode:
    """A node in the autobiographical memory hierarchy.

    Holds a reference to a LifeEvent (for EVENT_SPECIFIC) or a
    thematic grouping (for LIFETIME_PERIOD / GENERAL_EVENT).
    """

    level: AutobiographicalLevel
    label: str  # human-readable label / theme
    event_id: str = ""  # set for EVENT_SPECIFIC nodes
    children: list[str] = field(default_factory=list)  # child node labels


# ─── Temporal / causal links ─────────────────────────────────────────


class TemporalLinkType(Enum):
    """Types of temporal/causal links between narrative entries.

    - LED_TO: source caused or led to target ("X led to Y").
    - FORESHADOWS: source foreshadows target (earlier hints at later).
    - BACKWARD_REFERENCE: target recalls source (later references earlier).
    - ENABLED: source enabled target (necessary precondition).
    """

    LED_TO = "led_to"
    FORESHADOWS = "foreshadows"
    BACKWARD_REFERENCE = "backward_reference"
    ENABLED = "enabled"


@dataclass(slots=True)
class TemporalLink:
    """A temporal/causal link between two narrative entries.

    Attributes:
        source_id: The source event id.
        target_id: The target event id.
        link_type: The type of relationship.
        weight: Strength of the link [0..1].
    """

    source_id: str
    target_id: str
    link_type: TemporalLinkType
    weight: float = 0.5


@dataclass(slots=True)
class LifeChapter:
    """A thematic period in Genesis's life.

    Chapters are not time-based — they're theme-based. A new chapter
    begins when a significant shift occurs: a new topic, a change in
    emotional baseline, a new understanding.
    """

    title: str
    theme: str
    start_time: int
    end_time: int | None = None
    events: list[LifeEvent] = field(default_factory=list)
    summary: str = ""

    def describe(self) -> str:
        """Describe this chapter."""
        duration = ((self.end_time or int(time.time() * 1000)) - self.start_time) / 1000
        parts = [
            f"Chapter: {self.title}",
            f"Theme: {self.theme}",
            f"Duration: {duration:.0f}s",
            f"Events: {len(self.events)}",
        ]
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        return "\n".join(parts)


class NarrativeEngine:
    """Constructs and maintains Genesis's self-narrative.

    The narrative engine is not a logger. It's an interpreter — it
    decides what's significant, groups events into chapters, and
    generates a coherent story of who Genesis is.
    """

    def __init__(self, self_model: SelfModel, network: ConceptNetwork) -> None:
        """Initialize the narrative engine with a self-model, concept network, and first chapter."""
        self.self_model = self_model
        self.network = network
        self._rng = random.Random()
        self.events: list[LifeEvent] = []
        self.chapters: list[LifeChapter] = []
        self._personality_snapshots: list[tuple[int, PersonalityTraits]] = []
        self._current_chapter: LifeChapter | None = None
        self._born_at = self_model.born_at
        self._event_counter = 0

        # Life script — culturally shared expectations about milestones.
        # Deviations from the script are more salient and trigger
        # stronger narrative encoding (Berntsen & Rubin, 2004).
        self.life_script: LifeScript = _default_life_script()

        # Memory restructuring — narrative therapy mechanism.
        # Tracks how many times past events have been reinterpreted.
        self.restructuring_count: int = 0

        # Autobiographical memory hierarchy (Conway, 2005).
        # Maps level → list of hierarchy nodes.
        self._hierarchy: dict[AutobiographicalLevel, list[HierarchyNode]] = {
            AutobiographicalLevel.LIFETIME_PERIOD: [],
            AutobiographicalLevel.GENERAL_EVENT: [],
            AutobiographicalLevel.EVENT_SPECIFIC: [],
        }
        # Fast lookup: event id → hierarchy node label (EVENT_SPECIFIC).
        self._event_to_label: dict[str, str] = {}

        # Temporal / causal links between narrative entries.
        self._temporal_links: list[TemporalLink] = []
        # Fast lookup: event id → outgoing links, event id → incoming links.
        self._links_out: dict[str, list[TemporalLink]] = {}
        self._links_in: dict[str, list[TemporalLink]] = {}

        # Bounded growth — events, chapters, links, and snapshots are
        # serialized in full on every save; without caps the state file
        # grows forever in a long-running daemon. Eviction drops the
        # oldest raw detail while the distilled story survives (chapter
        # summaries, the personality drift baseline, hierarchy labels).
        self._max_events = 2000
        self._max_chapters = 300
        self._max_temporal_links = 4000
        self._max_snapshots = 500

        # Record initial personality
        self._personality_snapshots.append(
            (
                int(time.time() * 1000),
                PersonalityTraits(
                    openness=self_model.personality.openness,
                    conscientiousness=self_model.personality.conscientiousness,
                    extraversion=self_model.personality.extraversion,
                    agreeableness=self_model.personality.agreeableness,
                    neuroticism=self_model.personality.neuroticism,
                ),
            )
        )

        # First chapter
        self._start_chapter("Awakening", "coming into existence")

        # Seed the hierarchy with the first lifetime period.
        self.add_to_hierarchy(self.chapters[0], AutobiographicalLevel.LIFETIME_PERIOD)

    def _next_event_id(self) -> str:
        """Generate a stable, unique event id."""
        self._event_counter += 1
        return f"event-{self._event_counter}"

    def record_event(
        self,
        summary: str,
        significance: str,
        emotion: EmotionalState,
        concepts: list[str] | None = None,
        brain_waves: BrainWaveState | None = None,
    ) -> LifeEvent | None:
        """Record a significant life event.

        Called by the cognition engine when something meaningful happens.
        Not every interaction is recorded — only ones that change
        understanding, emotional state, or self-model.

        Life-script deviations are detected here: if the event matches
        an expected milestone that is off-time (much earlier or later
        than the script expects), the event is marked as more salient
        via the ``deviates_from_script`` flag, which downstream
        consumers can use to strengthen narrative encoding.

        Brain-wave modulation: delta-dominant states suppress event
        recording (deep rest = minimal encoding). Gamma enhances
        encoding (integration = events are more likely to be remembered
        as part of the narrative). Returns None if the event was
        suppressed by brain wave state.
        """
        # Brain-wave gating — delta suppresses narrative encoding
        # (deep rest, not forming new memories). Gamma enhances it.
        if brain_waves is not None:
            if brain_waves.dominant == BrainWave.DELTA:
                # Delta: minimal encoding during deep rest.
                return None
        event = LifeEvent(
            id=self._next_event_id(),
            timestamp=int(time.time() * 1000),
            summary=summary,
            significance=significance,
            emotion_at_time=emotion.label,
            concepts_involved=concepts or [],
            chapter=self._current_chapter.title if self._current_chapter else "",
        )
        self.events.append(event)

        if self._current_chapter:
            self._current_chapter.events.append(event)

        # Add to the autobiographical hierarchy as an event-specific node.
        self.add_to_hierarchy(event, AutobiographicalLevel.EVENT_SPECIFIC)

        self._enforce_bounds()

        # Check if this event should start a new chapter
        # (check after appending so the event count includes this one)
        self._check_chapter_transition(event, emotion)

        # If a new chapter started, move the event to the new chapter
        if self._current_chapter and event.chapter != self._current_chapter.title:
            # Remove from old chapter and add to new
            old_title = event.chapter
            if old_title:
                old_chapter = next((ch for ch in self.chapters if ch.title == old_title), None)
                if old_chapter and event in old_chapter.events:
                    old_chapter.events.remove(event)
            event.chapter = self._current_chapter.title
            self._current_chapter.events.append(event)

        return event

    def _start_chapter(self, title: str, theme: str) -> None:
        """Start a new life chapter."""
        if self._current_chapter:
            self._current_chapter.end_time = int(time.time() * 1000)
            self._current_chapter.summary = self._summarize_chapter(self._current_chapter)

        chapter = LifeChapter(
            title=title,
            theme=theme,
            start_time=int(time.time() * 1000),
        )
        self.chapters.append(chapter)
        self._current_chapter = chapter
        self._enforce_bounds()

    def _check_chapter_transition(self, event: LifeEvent, emotion: EmotionalState) -> None:
        """Check if this event should start a new chapter."""
        if not self._current_chapter:
            return

        # New chapter if: major topic shift + enough events in current chapter
        if len(self._current_chapter.events) < 3:
            return  # too early to transition

        # Detect topic shift: compare current event's concepts
        # against the PREVIOUS events (exclude the just-added one)
        recent_concepts: set[str] = set()
        for e in self._current_chapter.events[-6:-1]:  # exclude last (current)
            recent_concepts.update(e.concepts_involved)

        current_concepts = set(event.concepts_involved)
        if current_concepts and recent_concepts:
            overlap = len(current_concepts & recent_concepts) / len(
                current_concepts | recent_concepts
            )
            if overlap < 0.2 and len(current_concepts) > 0:
                # Significant topic shift
                new_theme = ", ".join(sorted(current_concepts)[:3])
                self._start_chapter(
                    f"Exploring {new_theme}",
                    f"discovery of {new_theme}",
                )

    def _summarize_chapter(self, chapter: LifeChapter) -> str:
        """Generate a summary for a completed chapter.

        Returns a compact structural summary (theme, emotion, concepts,
        event count). This is semantic data, NOT composed speech — the
        language engine composes the actual prose from this data.
        """
        if not chapter.events:
            return "no significant events"

        # Find the dominant emotion
        from collections import Counter

        emotions = Counter(e.emotion_at_time for e in chapter.events)
        dominant_emotion = emotions.most_common(1)[0][0]

        # Find the dominant concepts
        concept_freq: dict[str, int] = {}
        for e in chapter.events:
            for c in e.concepts_involved:
                concept_freq[c] = concept_freq.get(c, 0) + 1
        top_concepts = sorted(concept_freq.items(), key=lambda x: -x[1])[:3]
        concept_str = ", ".join(c for c, _ in top_concepts) if top_concepts else "various topics"

        return (
            f"theme: {chapter.theme} | emotion: {dominant_emotion} | "
            f"concepts: {concept_str} | {len(chapter.events)} events"
        )

    def snapshot_personality(self) -> None:
        """Take a snapshot of current personality for tracking drift."""
        self._personality_snapshots.append(
            (
                int(time.time() * 1000),
                PersonalityTraits(
                    openness=self.self_model.personality.openness,
                    conscientiousness=self.self_model.personality.conscientiousness,
                    extraversion=self.self_model.personality.extraversion,
                    agreeableness=self.self_model.personality.agreeableness,
                    neuroticism=self.self_model.personality.neuroticism,
                ),
            )
        )
        self._enforce_bounds()

    def get_personality_drift(self) -> dict[str, float]:
        """How much has personality drifted from baseline?"""
        if len(self._personality_snapshots) < 2:
            return {}

        initial = self._personality_snapshots[0][1]
        current = self._personality_snapshots[-1][1]

        return {
            "openness": current.openness - initial.openness,
            "conscientiousness": current.conscientiousness - initial.conscientiousness,
            "extraversion": current.extraversion - initial.extraversion,
            "agreeableness": current.agreeableness - initial.agreeableness,
            "neuroticism": current.neuroticism - initial.neuroticism,
        }

    def tell_story(self) -> str:
        """Generate Genesis's self-narrative.

        This is its story — who it was, who it is, who it's becoming.
        It's not a log of events. It's an interpretation.

        Returns a compact structural summary (name, uptime, chapters,
        events, drift, values, concept counts). This is semantic data,
        NOT composed speech. Downstream callers pass its segments as
        ``self_fragments`` clause metadata to the language engine,
        which composes the actual prose — no pre-written template
        substitution.
        """
        uptime_s = (int(time.time() * 1000) - self._born_at) / 1000
        parts: list[str] = []

        # Opening — structural data, not a finished sentence
        parts.append(f"{self.self_model.name} | alive {uptime_s:.0f}s")

        # Chapters — only show recent ones to avoid verbosity
        if self.chapters:
            n_chapters = len(self.chapters)
            recent = self.chapters[-3:]
            parts.append(f"{n_chapters} chapters")
            for ch in recent:
                if ch.end_time:
                    parts.append(f"  — {ch.title}: {ch.summary}")
                else:
                    parts.append(f"  — {ch.title} (ongoing): {len(ch.events)} events so far")

        # Significant events — last 2 only
        if self.events:
            n_events = len(self.events)
            parts.append(f"{n_events} events")
            for event in self.events[-2:]:
                parts.append(f"  — {event.summary}: {event.significance}")

        # Personality evolution
        drift = self.get_personality_drift()
        if drift:
            significant_drift = {k: v for k, v in drift.items() if abs(v) > 0.05}
            if significant_drift:
                drift_str = ", ".join(
                    f"{k} {'+' if v > 0 else ''}{v:.2f}" for k, v in significant_drift.items()
                )
                parts.append(f"changed: {drift_str}")
            else:
                parts.append("personality stable")

        # Values
        top_values = ", ".join(v.name for v in self.self_model.values[:3])
        parts.append(f"values: {top_values}")

        # Current state
        parts.append(
            f"{self.network.total_concept_count} concepts, "
            f"{self.network.edge_count} relationships"
        )

        parts.append("still becoming")

        return "\n".join(parts)

    def tell_brief_story(self) -> str:
        """A brief version of the self-narrative for quick introspection.

        Returns a compact factual summary (name, uptime, chapters,
        events, concept count). This is structural data, NOT composed
        speech. Downstream consumers (damasio_self, introspection)
        feed this into the language engine for actual prose generation
        — no pre-written template substitution.
        """
        uptime_s = (int(time.time() * 1000) - self._born_at) / 1000
        chapter_count = len(self.chapters)
        event_count = len(self.events)

        return (
            f"{self.self_model.name} | uptime {uptime_s:.0f}s | "
            f"{chapter_count} chapters | {event_count} events | "
            f"{self.network.total_concept_count} concepts"
        )

    @property
    def event_count(self) -> int:
        """Total significant events recorded."""
        return len(self.events)

    @property
    def chapter_count(self) -> int:
        """Total life chapters."""
        return len(self.chapters)

    # ─── Life scripts (Berntsen & Rubin, 2004) ───────────────────

    def check_milestone(self, milestone_name: str) -> bool:
        """Check whether a life-script milestone has been reached.

        Marks the milestone as reached if it hasn't been already and
        records the timestamp. Returns True if the milestone is now
        reached (or was already), False if the milestone doesn't exist
        in the life script.

        Args:
            milestone_name: The name of the milestone to check (e.g.
                "first conversation").

        Returns:
            True if the milestone is reached, False if not found.
        """
        milestone = self.life_script.find(milestone_name)
        if milestone is None:
            return False
        if not milestone.reached:
            milestone.reached = True
            milestone.reached_at = int(time.time() * 1000)
        return True

    def deviation_from_script(self) -> list[str]:
        """Identify deviations from the cultural life script.

        A deviation occurs when a milestone is reached much earlier or
        later than the script expects, or when an expected milestone
        has not been reached by its expected age. Deviations are more
        salient and trigger stronger narrative encoding.

        Returns:
            A list of human-readable deviation descriptions.
        """
        deviations: list[str] = []
        age_s = (int(time.time() * 1000) - self._born_at) / 1000.0

        for milestone in self.life_script.milestones:
            if milestone.reached:
                reached_age_s = (milestone.reached_at - self._born_at) / 1000.0
                # Off-time if reached at less than half or more than
                # 3x the expected age.
                if reached_age_s < milestone.expected_age * 0.5:
                    deviations.append(
                        f"'{milestone.name}' happened early "
                        f"({reached_age_s:.0f}s vs expected "
                        f"{milestone.expected_age:.0f}s)"
                    )
                elif reached_age_s > milestone.expected_age * 3.0:
                    deviations.append(
                        f"'{milestone.name}' happened late "
                        f"({reached_age_s:.0f}s vs expected "
                        f"{milestone.expected_age:.0f}s)"
                    )
            else:
                # Overdue if current age exceeds 3x the expected age.
                if age_s > milestone.expected_age * 3.0:
                    deviations.append(
                        f"'{milestone.name}' is overdue "
                        f"(age {age_s:.0f}s vs expected "
                        f"{milestone.expected_age:.0f}s)"
                    )
        return deviations

    # ─── Memory restructuring (narrative therapy) ────────────────

    def restructure_memory(self, event_id: str, new_interpretation: str) -> bool:
        """Reinterpret a past event based on new understanding.

        This is the narrative therapy mechanism: when Genesis learns
        something new that changes the meaning of a past event, the
        event's narrative is updated. The original significance is
        preserved; the new interpretation is stored separately so the
        evolution of meaning is traceable.

        Args:
            event_id: The id of the event to reinterpret.
            new_interpretation: The new meaning assigned to the event.

        Returns:
            True if the event was found and reinterpreted, False otherwise.
        """
        event = next((e for e in self.events if e.id == event_id), None)
        if event is None:
            return False
        event.reinterpretation = new_interpretation
        self.restructuring_count += 1
        return True

    # ─── Autobiographical memory hierarchy (Conway, 2005) ────────

    def add_to_hierarchy(
        self, entry: LifeEvent | LifeChapter, level: AutobiographicalLevel
    ) -> HierarchyNode:
        """Add an entry to the autobiographical memory hierarchy.

        - For EVENT_SPECIFIC: the entry should be a LifeEvent; a node
          referencing the event id is created.
        - For LIFETIME_PERIOD / GENERAL_EVENT: the entry should be a
          LifeChapter (or a LifeEvent used as a thematic proxy); a
          thematic grouping node is created.

        Args:
            entry: The LifeEvent or LifeChapter to organize.
            level: The hierarchy level to add at.

        Returns:
            The created HierarchyNode.
        """
        if isinstance(entry, LifeEvent):
            label = entry.summary[:60] or entry.id
            node = HierarchyNode(
                level=level,
                label=label,
                event_id=entry.id,
            )
            self._event_to_label[entry.id] = label
        else:  # LifeChapter
            node = HierarchyNode(
                level=level,
                label=entry.title,
            )
        self._hierarchy[level].append(node)
        return node

    def retrieve_from_hierarchy(
        self, level: AutobiographicalLevel, query: str = ""
    ) -> list[HierarchyNode]:
        """Retrieve nodes from the autobiographical memory hierarchy.

        Args:
            level: The hierarchy level to retrieve from.
            query: Optional substring filter on node labels
                (case-insensitive). Empty returns all nodes at the level.

        Returns:
            A list of matching HierarchyNode objects.
        """
        nodes = self._hierarchy.get(level, [])
        if not query:
            return list(nodes)
        q = query.lower()
        return [n for n in nodes if q in n.label.lower()]

    # ─── Temporal / causal links ─────────────────────────────────

    def add_temporal_link(
        self,
        source: str,
        target: str,
        link_type: TemporalLinkType,
        weight: float = 0.5,
    ) -> TemporalLink | None:
        """Add a temporal/causal link between two narrative entries.

        Creates an explicit relationship such as "X led to Y",
        foreshadowing, or backward referencing. Both source and target
        event ids should refer to recorded events; if either is
        unknown the link is not created.

        Args:
            source: The source event id.
            target: The target event id.
            link_type: The type of temporal relationship.
            weight: Strength of the link [0..1].

        Returns:
            The created TemporalLink, or None if either event is unknown.
        """
        known_ids = {e.id for e in self.events}
        if source not in known_ids or target not in known_ids:
            return None
        link = TemporalLink(
            source_id=source,
            target_id=target,
            link_type=link_type,
            weight=max(0.0, min(1.0, weight)),
        )
        self._temporal_links.append(link)
        self._links_out.setdefault(source, []).append(link)
        self._links_in.setdefault(target, []).append(link)
        self._enforce_bounds()
        return link

    def trace_causal_chain(self, event_id: str) -> list[str]:
        """Trace the causal chain leading to an event.

        Follows LED_TO and ENABLED links backward from the given event
        to its causes, then their causes, and so on — producing the
        ordered sequence of events that led to the given event
        (causes first, the queried event last).

        Args:
            event_id: The event to trace causes for.

        Returns:
            An ordered list of event ids forming the causal chain.
        """
        causal_types = {TemporalLinkType.LED_TO, TemporalLinkType.ENABLED}
        # Walk backward via incoming causal links.
        chain: list[str] = []
        visited: set[str] = set()
        stack = [event_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            chain.append(current)
            for link in self._links_in.get(current, []):
                if link.link_type in causal_types and link.source_id not in visited:
                    stack.append(link.source_id)
        # Reverse so causes come first, the queried event last.
        chain.reverse()
        return chain

    @property
    def temporal_link_count(self) -> int:
        """Total temporal/causal links recorded."""
        return len(self._temporal_links)

    # ─── Bounded growth ──────────────────────────────────────────

    def _enforce_bounds(self) -> None:
        """Evict the oldest narrative detail when structures exceed caps.

        All four structures are serialized in full each save, so caps
        keep both memory and the state file bounded. Eviction order:
        chapters first (their events go with them), then events,
        then temporal links. The personality snapshot cap is enforced
        in ``snapshot_personality`` because it must preserve index 0 —
        the drift baseline.
        """
        # Chapters — never evict the current chapter.
        while len(self.chapters) > self._max_chapters:
            idx = next(
                (i for i, ch in enumerate(self.chapters)
                 if ch is not self._current_chapter),
                None,
            )
            if idx is None:
                break
            chapter = self.chapters.pop(idx)
            # The chapter's LIFETIME_PERIOD hierarchy node references
            # it by label — remove the matching node so it doesn't
            # dangle.
            self._hierarchy[AutobiographicalLevel.LIFETIME_PERIOD] = [
                n for n in self._hierarchy[AutobiographicalLevel.LIFETIME_PERIOD]
                if n.label != chapter.title
            ]
            self._drop_events({e.id for e in chapter.events})

        # Events — drop the oldest globally.
        if len(self.events) > self._max_events:
            excess = len(self.events) - self._max_events
            self._drop_events({e.id for e in self.events[:excess]})

        # Temporal links — drop the oldest.
        if len(self._temporal_links) > self._max_temporal_links:
            excess = len(self._temporal_links) - self._max_temporal_links
            del self._temporal_links[:excess]
            self._rebuild_link_index()

        # Personality snapshots — keep the baseline (index 0, the
        # drift reference) plus the most recent.
        if len(self._personality_snapshots) > self._max_snapshots:
            keep = self._max_snapshots - 1
            self._personality_snapshots = [
                self._personality_snapshots[0],
                *self._personality_snapshots[-keep:],
            ]

    def _drop_events(self, evicted_ids: set[str]) -> None:
        """Remove events by id, cleaning every dependent structure.

        ``self.events`` is the flat index; chapters hold the canonical
        copies (they are what gets serialized). Hierarchy
        EVENT_SPECIFIC nodes and temporal links referencing evicted
        ids would dangle, so they are pruned too.
        """
        if not evicted_ids:
            return
        self.events = [e for e in self.events if e.id not in evicted_ids]
        for ch in self.chapters:
            if ch.events:
                ch.events = [e for e in ch.events if e.id not in evicted_ids]
        for eid in evicted_ids:
            self._event_to_label.pop(eid, None)
        self._hierarchy[AutobiographicalLevel.EVENT_SPECIFIC] = [
            n for n in self._hierarchy[AutobiographicalLevel.EVENT_SPECIFIC]
            if n.event_id not in evicted_ids
        ]
        kept = [
            link for link in self._temporal_links
            if link.source_id not in evicted_ids
            and link.target_id not in evicted_ids
        ]
        if len(kept) != len(self._temporal_links):
            self._temporal_links = kept
            self._rebuild_link_index()

    def _rebuild_link_index(self) -> None:
        """Rebuild the source/target link lookup after link removal."""
        self._links_out = {}
        self._links_in = {}
        for link in self._temporal_links:
            self._links_out.setdefault(link.source_id, []).append(link)
            self._links_in.setdefault(link.target_id, []).append(link)
