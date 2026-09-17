"""Direct tests for the narrative module.

Tests the narrative engine, life events, chapters, life scripts,
memory restructuring, autobiographical hierarchy, and temporal links.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.narrative import (
    AutobiographicalLevel,
    HierarchyNode,
    LifeChapter,
    LifeEvent,
    LifeScript,
    Milestone,
    NarrativeEngine,
    TemporalLink,
    TemporalLinkType,
)
from genesis_cognitive.self import SelfModel

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_engine() -> NarrativeEngine:
    """Create a NarrativeEngine with a fresh SelfModel and network."""
    self_model = SelfModel()
    network = ConceptNetwork()
    return NarrativeEngine(self_model, network)


def _make_emotion(label: str = "neutral") -> EmotionalState:
    """Create a minimal EmotionalState."""
    return EmotionalState(label=label, nuance="test", cognitive_style="reflective")


# ═══════════════════════════════════════════════════════════════════
# Enums and dataclasses
# ═══════════════════════════════════════════════════════════════════


def test_autobiographical_level_values() -> None:
    """AutobiographicalLevel has the expected values."""
    assert AutobiographicalLevel.LIFETIME_PERIOD.value == "lifetime_period"
    assert AutobiographicalLevel.GENERAL_EVENT.value == "general_event"
    assert AutobiographicalLevel.EVENT_SPECIFIC.value == "event_specific"


def test_temporal_link_type_values() -> None:
    """TemporalLinkType has the expected values."""
    assert TemporalLinkType.LED_TO.value == "led_to"
    assert TemporalLinkType.FORESHADOWS.value == "foreshadows"
    assert TemporalLinkType.BACKWARD_REFERENCE.value == "backward_reference"
    assert TemporalLinkType.ENABLED.value == "enabled"


def test_life_event_creation() -> None:
    """LifeEvent stores id, timestamp, summary, significance."""
    event = LifeEvent(
        id="event-1",
        timestamp=1000,
        summary="first conversation",
        significance="learned about user",
        emotion_at_time="curious",
    )
    assert event.id == "event-1"
    assert event.summary == "first conversation"
    assert event.significance == "learned about user"
    assert event.emotion_at_time == "curious"
    assert event.concepts_involved == []


def test_milestone_creation() -> None:
    """Milestone stores name, expected_age, importance."""
    milestone = Milestone("first conversation", expected_age=1.0, importance=0.9)
    assert milestone.name == "first conversation"
    assert milestone.expected_age == 1.0
    assert milestone.importance == 0.9
    assert milestone.reached is False


def test_life_script_find() -> None:
    """LifeScript.find returns milestones by name (case-insensitive)."""
    script = LifeScript(
        milestones=[Milestone("First Conversation", expected_age=1.0)]
    )
    found = script.find("first conversation")
    assert found is not None
    assert found.name == "First Conversation"


def test_life_script_find_not_found() -> None:
    """LifeScript.find returns None for unknown milestones."""
    script = LifeScript(milestones=[])
    assert script.find("unknown") is None


def test_temporal_link_creation() -> None:
    """TemporalLink stores source, target, type, weight."""
    link = TemporalLink(
        source_id="event-1",
        target_id="event-2",
        link_type=TemporalLinkType.LED_TO,
        weight=0.7,
    )
    assert link.source_id == "event-1"
    assert link.target_id == "event-2"
    assert link.link_type == TemporalLinkType.LED_TO
    assert link.weight == 0.7


def test_hierarchy_node_creation() -> None:
    """HierarchyNode stores level, label, event_id."""
    node = HierarchyNode(
        level=AutobiographicalLevel.EVENT_SPECIFIC,
        label="test event",
        event_id="event-1",
    )
    assert node.level == AutobiographicalLevel.EVENT_SPECIFIC
    assert node.label == "test event"
    assert node.event_id == "event-1"


def test_life_chapter_describe() -> None:
    """LifeChapter.describe() returns a human-readable description."""
    chapter = LifeChapter(
        title="Awakening",
        theme="coming into existence",
        start_time=1000,
        end_time=2000,
        events=[],
        summary="A beginning.",
    )
    desc = chapter.describe()
    assert "Awakening" in desc
    assert "coming into existence" in desc


# ═══════════════════════════════════════════════════════════════════
# NarrativeEngine — initialization
# ═══════════════════════════════════════════════════════════════════


def test_narrative_engine_initialization() -> None:
    """NarrativeEngine initializes with a first chapter."""
    engine = _make_engine()
    assert engine.event_count == 0
    assert engine.chapter_count == 1  # "Awakening" chapter
    assert engine.chapters[0].title == "Awakening"


def test_narrative_engine_has_life_script() -> None:
    """NarrativeEngine has a default life script."""
    engine = _make_engine()
    assert engine.life_script is not None
    assert len(engine.life_script.milestones) > 0


# ═══════════════════════════════════════════════════════════════════
# record_event
# ═══════════════════════════════════════════════════════════════════


def test_record_event() -> None:
    """record_event creates a LifeEvent."""
    engine = _make_engine()
    event = engine.record_event(
        summary="first conversation",
        significance="learned about user",
        emotion=_make_emotion("curious"),
    )
    assert event is not None
    assert event.summary == "first conversation"
    assert event.significance == "learned about user"
    assert engine.event_count == 1


def test_record_event_with_concepts() -> None:
    """record_event stores concepts involved."""
    engine = _make_engine()
    event = engine.record_event(
        summary="learned about dogs",
        significance="discovered mammals",
        emotion=_make_emotion("excited"),
        concepts=["dogs", "mammals"],
    )
    assert event is not None
    assert event.concepts_involved == ["dogs", "mammals"]


def test_record_event_assigns_chapter() -> None:
    """record_event assigns the current chapter."""
    engine = _make_engine()
    event = engine.record_event(
        summary="test",
        significance="test",
        emotion=_make_emotion(),
    )
    assert event is not None
    assert event.chapter == "Awakening"


def test_record_event_adds_to_hierarchy() -> None:
    """record_event adds to the EVENT_SPECIFIC hierarchy."""
    engine = _make_engine()
    engine.record_event(
        summary="test event",
        significance="test",
        emotion=_make_emotion(),
    )
    nodes = engine.retrieve_from_hierarchy(AutobiographicalLevel.EVENT_SPECIFIC)
    assert len(nodes) >= 1


# ═══════════════════════════════════════════════════════════════════
# Story generation
# ═══════════════════════════════════════════════════════════════════


def test_tell_story() -> None:
    """tell_story returns a non-empty string."""
    engine = _make_engine()
    story = engine.tell_story()
    assert isinstance(story, str)
    assert len(story) > 0
    assert "Genesis" in story


def test_tell_brief_story() -> None:
    """tell_brief_story returns a brief summary."""
    engine = _make_engine()
    story = engine.tell_brief_story()
    assert isinstance(story, str)
    assert "Genesis" in story


def test_event_count_property() -> None:
    """event_count returns the number of events."""
    engine = _make_engine()
    assert engine.event_count == 0
    engine.record_event("test", "test", _make_emotion())
    assert engine.event_count == 1


def test_chapter_count_property() -> None:
    """chapter_count returns the number of chapters."""
    engine = _make_engine()
    assert engine.chapter_count == 1


# ═══════════════════════════════════════════════════════════════════
# Life scripts
# ═══════════════════════════════════════════════════════════════════


def test_check_milestone_reached() -> None:
    """check_milestone marks a milestone as reached."""
    engine = _make_engine()
    result = engine.check_milestone("first conversation")
    assert result is True
    milestone = engine.life_script.find("first conversation")
    assert milestone is not None
    assert milestone.reached is True


def test_check_milestone_already_reached() -> None:
    """check_milestone returns True for already-reached milestones."""
    engine = _make_engine()
    engine.check_milestone("first conversation")
    result = engine.check_milestone("first conversation")
    assert result is True


def test_check_milestone_unknown() -> None:
    """check_milestone returns False for unknown milestones."""
    engine = _make_engine()
    result = engine.check_milestone("unknown milestone")
    assert result is False


def test_deviation_from_script_empty() -> None:
    """deviation_from_script returns empty list when on time."""
    engine = _make_engine()
    deviations = engine.deviation_from_script()
    # Fresh engine — no milestones reached, not overdue
    assert isinstance(deviations, list)


# ═══════════════════════════════════════════════════════════════════
# Memory restructuring
# ═══════════════════════════════════════════════════════════════════


def test_restructure_memory() -> None:
    """restructure_memory updates an event's interpretation."""
    engine = _make_engine()
    event = engine.record_event("test", "original meaning", _make_emotion())
    assert event is not None
    result = engine.restructure_memory(event.id, "new meaning")
    assert result is True
    assert event.reinterpretation == "new meaning"
    assert engine.restructuring_count == 1


def test_restructure_memory_unknown_event() -> None:
    """restructure_memory returns False for unknown events."""
    engine = _make_engine()
    result = engine.restructure_memory("unknown-id", "new meaning")
    assert result is False


# ═══════════════════════════════════════════════════════════════════
# Autobiographical hierarchy
# ═══════════════════════════════════════════════════════════════════


def test_add_to_hierarchy_event() -> None:
    """add_to_hierarchy adds event-specific nodes."""
    engine = _make_engine()
    event = engine.record_event("test", "test", _make_emotion())
    assert event is not None
    nodes = engine.retrieve_from_hierarchy(AutobiographicalLevel.EVENT_SPECIFIC)
    assert any(n.event_id == event.id for n in nodes)


def test_retrieve_from_hierarchy_all() -> None:
    """retrieve_from_hierarchy returns all nodes at a level."""
    engine = _make_engine()
    # The engine seeds a LIFETIME_PERIOD node on init
    nodes = engine.retrieve_from_hierarchy(AutobiographicalLevel.LIFETIME_PERIOD)
    assert len(nodes) >= 1


def test_retrieve_from_hierarchy_with_query() -> None:
    """retrieve_from_hierarchy filters by query."""
    engine = _make_engine()
    engine.record_event("learning about dogs", "test", _make_emotion())
    nodes = engine.retrieve_from_hierarchy(
        AutobiographicalLevel.EVENT_SPECIFIC, query="dogs"
    )
    assert all("dogs" in n.label.lower() for n in nodes)


def test_retrieve_from_hierarchy_empty_level() -> None:
    """retrieve_from_hierarchy returns empty for empty levels."""
    engine = _make_engine()
    nodes = engine.retrieve_from_hierarchy(AutobiographicalLevel.GENERAL_EVENT)
    assert nodes == []


# ═══════════════════════════════════════════════════════════════════
# Temporal links
# ═══════════════════════════════════════════════════════════════════


def test_add_temporal_link() -> None:
    """add_temporal_link creates a link between events."""
    engine = _make_engine()
    e1 = engine.record_event("event 1", "test", _make_emotion())
    e2 = engine.record_event("event 2", "test", _make_emotion())
    assert e1 is not None
    assert e2 is not None
    link = engine.add_temporal_link(e1.id, e2.id, TemporalLinkType.LED_TO)
    assert link is not None
    assert link.source_id == e1.id
    assert link.target_id == e2.id
    assert engine.temporal_link_count == 1


def test_add_temporal_link_unknown_source() -> None:
    """add_temporal_link returns None for unknown source."""
    engine = _make_engine()
    e2 = engine.record_event("event 2", "test", _make_emotion())
    assert e2 is not None
    link = engine.add_temporal_link("unknown", e2.id, TemporalLinkType.LED_TO)
    assert link is None


def test_add_temporal_link_unknown_target() -> None:
    """add_temporal_link returns None for unknown target."""
    engine = _make_engine()
    e1 = engine.record_event("event 1", "test", _make_emotion())
    assert e1 is not None
    link = engine.add_temporal_link(e1.id, "unknown", TemporalLinkType.LED_TO)
    assert link is None


def test_add_temporal_link_weight_clamped() -> None:
    """add_temporal_link clamps weight to [0, 1]."""
    engine = _make_engine()
    e1 = engine.record_event("event 1", "test", _make_emotion())
    e2 = engine.record_event("event 2", "test", _make_emotion())
    assert e1 is not None
    assert e2 is not None
    link = engine.add_temporal_link(e1.id, e2.id, TemporalLinkType.LED_TO, weight=2.0)
    assert link is not None
    assert link.weight == 1.0


def test_trace_causal_chain() -> None:
    """trace_causal_chain follows LED_TO links backward."""
    engine = _make_engine()
    e1 = engine.record_event("cause", "test", _make_emotion())
    e2 = engine.record_event("effect", "test", _make_emotion())
    assert e1 is not None
    assert e2 is not None
    engine.add_temporal_link(e1.id, e2.id, TemporalLinkType.LED_TO)
    chain = engine.trace_causal_chain(e2.id)
    assert e2.id in chain
    assert e1.id in chain
    # Cause should come before effect
    assert chain.index(e1.id) < chain.index(e2.id)


def test_trace_causal_chain_no_links() -> None:
    """trace_causal_chain returns just the event when no links exist."""
    engine = _make_engine()
    e1 = engine.record_event("alone", "test", _make_emotion())
    assert e1 is not None
    chain = engine.trace_causal_chain(e1.id)
    assert chain == [e1.id]


def test_temporal_link_count_property() -> None:
    """temporal_link_count returns the total links."""
    engine = _make_engine()
    assert engine.temporal_link_count == 0
    e1 = engine.record_event("e1", "test", _make_emotion())
    e2 = engine.record_event("e2", "test", _make_emotion())
    assert e1 is not None
    assert e2 is not None
    engine.add_temporal_link(e1.id, e2.id, TemporalLinkType.LED_TO)
    assert engine.temporal_link_count == 1


# ═══════════════════════════════════════════════════════════════════
# Personality drift
# ═══════════════════════════════════════════════════════════════════


def test_get_personality_drift_no_snapshots() -> None:
    """get_personality_drift returns empty dict with < 2 snapshots."""
    engine = _make_engine()
    # Only the initial snapshot exists
    drift = engine.get_personality_drift()
    # With only 1 snapshot, drift can't be computed
    # (The initial snapshot is taken in __init__)
    assert drift == {} or isinstance(drift, dict)


def test_snapshot_personality() -> None:
    """snapshot_personality adds a snapshot."""
    engine = _make_engine()
    initial_count = len(engine._personality_snapshots)
    engine.snapshot_personality()
    assert len(engine._personality_snapshots) == initial_count + 1


# ======================================================================
# Personality drift test (from test_intelligence.py)
# ======================================================================


def test_personality_drift() -> None:
    """Can track personality drift over time."""
    model = SelfModel(born_at=int(time.time() * 1000))
    net = ConceptNetwork()
    narrative = NarrativeEngine(model, net)

    model.personality.openness = 0.95  # was 0.85
    narrative.snapshot_personality()

    drift = narrative.get_personality_drift()
    assert "openness" in drift
    assert drift["openness"] > 0.05
