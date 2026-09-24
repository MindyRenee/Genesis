"""Tests for the external world — events, presences, and OuterWorld."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genesis_client.protocol import CHEM_OXYTOCIN
from genesis_cognitive import (
    ConceptNetwork,
    CuriosityEngine,
    ReasoningEngine,
    ReflectionEngine,
)
from genesis_cognitive.sleep import InnerLife
from genesis_cognitive.world import (
    Beta,
    EventKind,
    EventStream,
    ExternalEvent,
    OuterWorld,
    Presence,
    PresenceBelief,
    PresenceKind,
)
from genesis_cognitive.world.belief import BID_WINDOW
from genesis_cognitive.world.events import ground_topics
from genesis_cognitive.world.presence import (
    AMBIENT_PRESENCE_TIMEOUT,
    USER_PRESENCE_TIMEOUT,
)


def _make_world(**kwargs) -> OuterWorld:
    """Construct an OuterWorld with no wiring unless given."""
    return OuterWorld(**kwargs)


def _make_inner_life() -> InnerLife:
    """Build a minimal InnerLife for social-drive tests."""
    net = ConceptNetwork()
    net.add_concept("alpha", confidence=0.5)
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    reflection = ReflectionEngine(net)
    return InnerLife(net, curiosity, reflection, seed=42)


# ─── Events ─────────────────────────────────────────────────


def test_event_inbound_classification():
    """Inbound kinds are the world acting on her; outbound are her acts."""
    inbound = ExternalEvent(
        kind=EventKind.USER_SPEECH, source="user", content="hello"
    )
    outbound = ExternalEvent(
        kind=EventKind.UTTERANCE, source="self", content="hi"
    )
    assert inbound.inbound
    assert not outbound.inbound


def test_event_describe_direction():
    """describe() marks direction without being her voice."""
    e = ExternalEvent(
        kind=EventKind.USER_SPEECH, source="user", content="hi there"
    )
    assert "user" in e.describe() and "hi there" in e.describe()
    u = ExternalEvent(
        kind=EventKind.UTTERANCE, source="self", content="hello"
    )
    assert "she said" in u.describe()


def test_event_dict_roundtrip():
    """to_dict/from_dict preserves the event."""
    e = ExternalEvent(
        kind=EventKind.PERCEPTION,
        source="auditory",
        content="a bang",
        salience=0.7,
        addressed=False,
        topics=["sound"],
        timestamp=1234.5,
    )
    restored = ExternalEvent.from_dict(e.to_dict())
    assert restored is not None
    assert restored.kind == EventKind.PERCEPTION
    assert restored.content == "a bang"
    assert restored.salience == 0.7
    assert restored.topics == ["sound"]
    assert restored.timestamp == 1234.5


def test_event_from_dict_rejects_malformed():
    """Malformed payloads deserialize to None, not exceptions."""
    assert ExternalEvent.from_dict({"kind": "bogus_kind"}) is None
    assert ExternalEvent.from_dict({}) is None
    assert ExternalEvent.from_dict({"kind": EventKind.USER_SPEECH.value}) is None


def test_ground_topics_uses_known_concepts():
    """Only words she has concepts for become topics."""
    known = {"memory", "dreams"}
    topics = ground_topics(
        "memory and dreams are wonderful flibbertigibbet",
        lambda w: w in known,
    )
    assert "memory" in topics
    assert "dreams" in topics
    assert "flibbertigibbet" not in topics
    # Short words are grammatical noise, not topics
    assert "are" not in topics


def test_event_stream_bounded_and_ordered():
    """The stream keeps recent events, oldest first, bounded."""
    stream = EventStream(max_size=5)
    for i in range(8):
        stream.append(
            ExternalEvent(kind=EventKind.UTTERANCE, source="self", content=str(i))
        )
    assert len(stream) == 5
    recent = stream.recent(5)
    assert [e.content for e in recent] == ["3", "4", "5", "6", "7"]


# ─── Presence ───────────────────────────────────────────────


def test_presence_interaction_grows_familiarity():
    """Familiarity grows asymptotically with addressed exchanges."""
    p = Presence(presence_id="user", kind=PresenceKind.USER)
    for _ in range(10):
        p.interacted(sentiment=0.2, topics=["memory"], now=time.time())
    assert 0.0 < p.familiarity < 1.0
    first = p.familiarity
    p.interacted(sentiment=0.2, now=time.time())
    assert p.familiarity > first
    assert p.interactions == 11
    assert p.present


def test_presence_bond_follows_sentiment():
    """Positive sentiment builds bond; negative erodes it."""
    p = Presence(presence_id="user", kind=PresenceKind.USER)
    for _ in range(20):
        p.interacted(sentiment=0.5, now=time.time())
    assert p.bond > 0.0
    bond_peak = p.bond
    for _ in range(10):
        p.interacted(sentiment=-0.8, now=time.time())
    assert p.bond < bond_peak
    assert p.bond >= 0.0


def test_presence_topics_dedup_and_recency():
    """Repeated topics move to most-recent without duplicating."""
    p = Presence(presence_id="user", kind=PresenceKind.USER)
    p.interacted(topics=["memory", "dreams"], now=time.time())
    p.interacted(topics=["memory"], now=time.time())
    topics = list(p.topics)
    assert topics.count("memory") == 1
    assert topics[-1] == "memory"


def test_presence_display_name_and_timeout():
    """Unnamed presences display by kind; user gets a longer leash."""
    p = Presence(presence_id="user", kind=PresenceKind.USER)
    assert p.display_name == "the user"
    assert p.timeout == USER_PRESENCE_TIMEOUT
    v = Presence(presence_id="voices", kind=PresenceKind.VOICE)
    assert v.timeout == AMBIENT_PRESENCE_TIMEOUT
    p.set_name("Mindy")
    assert p.display_name == "Mindy"


def test_presence_dict_roundtrip_restores_absent():
    """A serialized presence restores its history but never 'present'."""
    p = Presence(presence_id="user", kind=PresenceKind.USER)
    p.set_name("Mindy")
    p.interacted(sentiment=0.4, topics=["memory"], now=time.time())
    assert p.present
    data = p.to_dict()
    restored = Presence.from_dict(data)
    assert restored is not None
    assert restored.name == "Mindy"
    assert restored.interactions == 1
    assert restored.familiarity == p.familiarity
    assert list(restored.topics) == ["memory"]
    # Presence is re-established by fresh activity, not by the save.
    assert not restored.present


def test_presence_from_dict_rejects_malformed():
    """Malformed presence payloads deserialize to None."""
    assert Presence.from_dict({}) is None
    assert Presence.from_dict({"presence_id": "x", "kind": "bogus"}) is None


# ─── OuterWorld inbound ─────────────────────────────────────


def test_hear_user_records_event_and_presence():
    """Addressed speech creates the user presence and an event."""
    world = _make_world()
    event = world.hear_user("hello genesis")
    assert event.kind == EventKind.USER_SPEECH
    assert event.addressed
    presence = world.presence("user")
    assert presence is not None
    assert presence.present
    assert presence.interactions == 1
    assert world.engaged_presence() is presence


def test_hear_user_applies_social_impulse():
    """Addressed speech emits the social-contact oxytocin impulse."""
    impulses = []
    world = _make_world(neuro_impulse=lambda c, a: impulses.append((c, a)))
    world.hear_user("hello")
    assert any(c == CHEM_OXYTOCIN and a > 0 for c, a in impulses)


def test_hear_user_resets_isolation_and_bids():
    """Addressed contact clears social pressure and outreach bids."""
    world = _make_world()
    world._last_addressed_time = time.time() - 4000.0
    world.note_outreach()
    world.note_outreach()
    assert world.unanswered_bids == 2
    assert world.social_isolation() > 0.0
    world.hear_user("hi")
    assert world.unanswered_bids == 0
    assert world.social_isolation() < 0.05


def test_hear_overheard_creates_ambient_presence():
    """Overheard speech lands on the shared ambient presence."""
    world = _make_world()
    event = world.hear_overheard("someone talking nearby")
    assert event is not None
    assert event.kind == EventKind.OVERHEARD_SPEECH
    assert not event.addressed
    presence = world.presence("voices")
    assert presence is not None
    assert presence.present
    # Overheard speech is not addressed engagement — the user
    # presence must not be created or credited.
    assert world.presence("user") is None


def test_inbound_gated_during_sleep():
    """Overheard speech and percepts are gated while she sleeps."""
    world = _make_world(is_sleeping=lambda: True)
    assert world.hear_overheard("noise") is None
    assert world.perceive("a sound") is None
    # Addressed speech is not gated — respond() wakes her first.
    assert world.hear_user("genesis") is not None


def test_perceive_records_percept():
    """Non-speech percepts are recorded with their salience."""
    world = _make_world()
    event = world.perceive("a loud bang", salience=0.8, source="auditory")
    assert event is not None
    assert event.kind == EventKind.PERCEPTION
    assert event.salience == 0.8


def test_outbound_events_recorded():
    """Her utterances and acts join the same stream."""
    world = _make_world()
    u = world.she_said("something she composed")
    a = world.she_acted("looked around", detail="a report")
    assert u.kind == EventKind.UTTERANCE and not u.inbound
    assert a.kind == EventKind.ACTION and not a.inbound
    kinds = [e.kind for e in world.recent_events(2)]
    assert EventKind.UTTERANCE in kinds and EventKind.ACTION in kinds


def test_event_listener_receives_events():
    """on_external_event fires for every recorded event."""
    world = _make_world()
    seen = []
    world.on_external_event = seen.append
    world.hear_user("hi")
    world.she_said("hello")
    assert len(seen) >= 2
    assert any(e.kind == EventKind.USER_SPEECH for e in seen)
    assert any(e.kind == EventKind.UTTERANCE for e in seen)


def test_listener_failure_does_not_break_world():
    """A throwing listener must never break event recording."""
    world = _make_world()
    def bad(_e):
        raise RuntimeError("listener exploded")
    world.on_external_event = bad
    event = world.hear_user("hi")  # must not raise
    assert event.kind == EventKind.USER_SPEECH


# ─── Presence lifecycle ─────────────────────────────────────


def test_mark_presence_emits_arrival_once():
    """First mark emits an ARRIVAL; re-marks while present do not."""
    world = _make_world()
    events = []
    world.on_external_event = events.append
    world.mark_presence("face:mindy", PresenceKind.FACE, name="Mindy")
    world.mark_presence("face:mindy", PresenceKind.FACE, name="Mindy")
    arrivals = [e for e in events if e.kind == EventKind.ARRIVAL]
    assert len(arrivals) == 1
    assert world.presence("face:mindy").name == "Mindy"


def test_tick_departures_for_silent_presences():
    """A presence silent past its timeout leaves — a DEPARTURE event."""
    world = _make_world()
    world.hear_overheard("a voice")
    presence = world.presence("voices")
    assert presence is not None
    # Force the presence stale and tick.
    presence.last_seen = time.time() - (AMBIENT_PRESENCE_TIMEOUT + 60)
    events = []
    world.on_external_event = events.append
    world.tick(now=time.time() + 10.0)
    assert not presence.present
    assert any(e.kind == EventKind.DEPARTURE for e in events)


def test_tick_rate_limited():
    """tick() runs its decay logic at most every few seconds."""
    world = _make_world()
    now = time.time()
    world.tick(now=now)  # first tick establishes the interval
    world.hear_overheard("a voice")
    presence = world.presence("voices")
    presence.last_seen = now - (AMBIENT_PRESENCE_TIMEOUT + 60)
    world.tick(now=now + 1.0)  # inside _TICK_INTERVAL — no departure
    assert presence.present
    world.tick(now=now + 10.0)  # past the interval — departs
    assert not presence.present


# ─── Social isolation ───────────────────────────────────────


def test_social_isolation_rises_over_time():
    """Isolation grows toward 1.0 over the isolation period."""
    world = _make_world()
    world._last_addressed_time = time.time() - 900.0
    iso = world.social_isolation()
    assert 0.4 < iso < 0.6


def test_company_dampens_isolation():
    """Someone present-but-silent counts socially, just less."""
    world = _make_world()
    world._last_addressed_time = time.time() - 900.0
    alone = world.social_isolation()
    world.hear_overheard("a voice nearby")
    with_company = world.social_isolation()
    assert with_company < alone


def test_engaged_and_last_seen_presence():
    """engaged_presence picks the present, last_seen the most recent."""
    world = _make_world()
    world.hear_overheard("voice")
    world.hear_user("hi")
    engaged = world.engaged_presence()
    assert engaged is not None and engaged.presence_id == "user"
    assert world.last_seen_presence() is not None


def test_note_outreach_counts_bids():
    """Outreach bids accumulate until someone engages her."""
    world = _make_world()
    assert world.unanswered_bids == 0
    world.note_outreach()
    world.note_outreach()
    assert world.unanswered_bids == 2
    world.hear_user("hello")
    assert world.unanswered_bids == 0


# ─── Persistence ────────────────────────────────────────────


def test_world_dict_roundtrip():
    """The world serializes and restores presences, events, clocks."""
    world = _make_world()
    world.hear_user("hello genesis")
    world.she_said("hi there")
    world.note_outreach()
    data = world.to_dict()

    restored = _make_world()
    restored.restore_from_dict(data)
    user = restored.presence("user")
    assert user is not None
    assert user.interactions == 1
    assert not user.present  # presences restore absent
    assert restored.unanswered_bids == 1
    kinds = [e.kind for e in restored.recent_events(10)]
    assert EventKind.USER_SPEECH in kinds
    assert EventKind.UTTERANCE in kinds


def test_world_restore_rejects_non_dict():
    """A malformed world_state is an error, not a silent reset."""
    import pytest

    world = _make_world()
    with pytest.raises(ValueError):
        world.restore_from_dict("not a dict")


def test_world_restore_skips_malformed_entries():
    """Bad presence/event entries are skipped, good ones restore."""
    world = _make_world()
    world.restore_from_dict({
        "presences": {
            "user": {"presence_id": "user", "kind": "user", "interactions": 3},
            "broken": {"kind": "nonsense"},
            "notadict": 42,
        },
        "events": [{"kind": "bogus"}, "junk"],
        "last_addressed_time": 1000.0,
        "unanswered_bids": 2,
    })
    user = world.presence("user")
    assert user is not None and user.interactions == 3
    assert world.presence("broken") is None
    assert world.unanswered_bids == 2


# ─── Inner-life coupling ────────────────────────────────────


def test_feed_social_drive_bounded():
    """External social pressure feeds the inner drive, bounded."""
    il = _make_inner_life()
    il.feed_social_drive(0.5)
    assert il._social_drive == 0.5
    il.feed_social_drive(10.0)
    assert il._social_drive == 2.0  # capped
    il.feed_social_drive(-5.0)
    assert il._social_drive == 0.0  # floored


# ─── Beliefs: posteriors ────────────────────────────────────


def test_beta_prior_is_uniform_ignorance():
    """A fresh Beta is maximally uncertain — mean 0.5, no evidence."""
    b = Beta()
    assert b.mean == 0.5
    assert b.evidence == 0.0
    assert b.variance > 0.08  # wide posterior


def test_beta_observe_shifts_and_concentrates():
    """Observations move the mean and shrink the variance."""
    b = Beta()
    wide = b.variance
    for _ in range(8):
        b.observe(True)
    assert b.mean > 0.8
    assert b.evidence == 8.0
    assert b.variance < wide


def test_beta_dict_roundtrip_and_malformed():
    """Posteriors serialize; malformed payloads yield fresh priors."""
    b = Beta(alpha=5.0, beta=2.0)
    restored = Beta.from_dict(b.to_dict())
    assert restored.alpha == 5.0 and restored.beta == 2.0
    assert Beta.from_dict({"a": -1, "b": 2}).evidence == 0.0
    assert Beta.from_dict("junk").evidence == 0.0
    assert Beta.from_dict(None).evidence == 0.0


# ─── Beliefs: bids and inferred state ───────────────────────


def test_belief_answered_bid_builds_responsiveness():
    """An answered bid credits responsiveness and its topics."""
    belief = PresenceBelief()
    now = time.time()
    belief.note_bid(now, ["music"])
    assert belief.resolve_bid(True, now + 30.0)
    assert belief.responsiveness.mean > 0.5
    assert belief.responsiveness.evidence == 1.0
    assert belief.topic_receptivity["music"].mean > 0.5
    assert belief.pending_bid is None


def test_belief_unanswered_bid_erodes_responsiveness():
    """Unanswered bids push the posterior toward 'they don't answer'."""
    belief = PresenceBelief()
    now = time.time()
    for i in range(4):
        belief.note_bid(now + i * 100, ["music"])
        belief.resolve_bid(False, now + i * 100 + 60)
    assert belief.responsiveness.mean < 0.2
    assert belief.unresponsive()
    assert belief.topic_receptivity["music"].mean < 0.2


def test_belief_new_bid_unanswers_stale_pending():
    """Bidding again before an answer resolves the old bid unanswered."""
    belief = PresenceBelief()
    now = time.time()
    belief.note_bid(now, [])
    belief.note_bid(now + 10.0, [])
    assert belief.responsiveness.evidence == 1.0
    assert belief.responsiveness.mean < 0.5


def test_belief_late_answer_counts_unanswered():
    """Engagement past BID_WINDOW is not evidence they respond to her."""
    belief = PresenceBelief()
    now = time.time()
    belief.note_bid(now, [])
    assert belief.resolve_bid(True, now + BID_WINDOW + 60.0)
    assert belief.responsiveness.mean < 0.5


def test_belief_resolve_bid_without_pending_is_noop():
    """resolve_bid returns False when no bid is open."""
    belief = PresenceBelief()
    assert not belief.resolve_bid(True, time.time())
    assert belief.responsiveness.evidence == 0.0


def test_belief_unresponsive_needs_evidence():
    """A stranger is never written off — evidence threshold required."""
    belief = PresenceBelief()
    now = time.time()
    for i in range(2):  # below _UNRESPONSIVE_MIN_BIDS
        belief.note_bid(now + i * 100, [])
        belief.resolve_bid(False, now + i * 100 + 60)
    assert not belief.unresponsive()


def test_belief_attention_decays():
    """Addressed activity spikes attention; silence lets it fade."""
    belief = PresenceBelief()
    now = time.time()
    belief.note_activity(now, weight=1.0)
    assert belief.attention_now(now) == 1.0
    faded = belief.attention_now(now + 1200.0)  # one tau
    assert 0.3 < faded < 0.45


def test_belief_rhythm_learns_active_hours():
    """The rhythm is trusted only after real observations."""
    from datetime import datetime

    belief = PresenceBelief()
    at_9am = datetime(2026, 1, 15, 9, 0).timestamp()
    at_3am = datetime(2026, 1, 15, 3, 0).timestamp()
    # No evidence yet — she doesn't assume dead hours.
    assert belief.likely_awake(at_3am)
    for _ in range(15):
        belief.note_activity(at_9am, weight=1.0)
    assert belief.activity_evidence >= 15.0
    assert belief.expected_activity(9) > 0.5
    # Now the histogram says 3am is a dead hour.
    assert not belief.likely_awake(at_3am)
    assert belief.likely_awake(at_9am)


def test_belief_sample_topic_prefers_receptive():
    """Thompson sampling favors proven topics but keeps exploring."""
    import random

    random.seed(7)
    belief = PresenceBelief()
    belief.topic_receptivity["proven"] = Beta(alpha=8.0, beta=2.0)
    # An uncertain topic — sparse evidence, wide posterior — still
    # gets a fair draw against the proven one.
    belief.topic_receptivity["unknown"] = Beta(alpha=1.0, beta=1.0)
    picks = [belief.sample_topic() for _ in range(200)]
    assert picks.count("proven") > picks.count("unknown")
    assert "unknown" in picks  # exploration still happens


def test_belief_describe_shows_only_evidence():
    """describe() renders what she actually has evidence for."""
    belief = PresenceBelief()
    now = time.time()
    assert belief.describe(now) == ""
    belief.note_bid(now, [])
    belief.resolve_bid(True, now + 5.0)
    belief.note_activity(now, weight=1.0)
    text = belief.describe(now)
    assert "replies" in text and "attentive" in text


def test_belief_bid_window_defaults_until_evidence():
    """A fresh presence gets the default window — no learned patience."""
    belief = PresenceBelief()
    assert belief.bid_window() == BID_WINDOW
    # Even a few answers aren't enough to trust the fit.
    now = time.time()
    for i in range(4):
        belief.note_bid(now + i * 100, [])
        belief.resolve_bid(True, now + i * 100 + 10.0)
    assert belief.bid_window() == BID_WINDOW


def test_belief_bid_window_learns_reply_pace():
    """Enough answered bids teach her this presence's reply pace."""
    belief = PresenceBelief()
    now = time.time()
    # A slow replier — ~200s typical → window should land well above
    # the 60s floor, near their p95 (~460s with the sigma floor).
    for i in range(6):
        belief.note_bid(now + i * 1000, [])
        belief.resolve_bid(True, now + i * 1000 + 200.0)
    window = belief.bid_window()
    assert 300.0 < window < 1200.0
    # A fast replier — ~10s typical → window clamps at the 60s floor.
    belief2 = PresenceBelief()
    for i in range(6):
        belief2.note_bid(now + i * 1000, [])
        belief2.resolve_bid(True, now + i * 1000 + 10.0)
    assert belief2.bid_window() == 60.0


def test_belief_learned_window_marks_late_answers():
    """Past her learned window, an answer counts as unanswered."""
    belief = PresenceBelief()
    now = time.time()
    for i in range(6):  # fast replier → 60s window
        belief.note_bid(now + i * 1000, [])
        belief.resolve_bid(True, now + i * 1000 + 10.0)
    # A 90s-late answer exceeds the learned 60s window.
    belief.note_bid(now, [])
    assert belief.resolve_bid(True, now + 90.0)
    # Six timely answers + this late one: alpha = 1+6, beta = 1+1.
    assert belief.responsiveness.alpha == 7.0
    assert belief.responsiveness.beta == 2.0


def test_belief_latency_persists():
    """The reply-latency fit rides persistence like the posteriors."""
    belief = PresenceBelief()
    now = time.time()
    for i in range(6):
        belief.note_bid(now + i * 1000, [])
        belief.resolve_bid(True, now + i * 1000 + 200.0)
    restored = PresenceBelief.from_dict(belief.to_dict())
    assert restored.reply_latency.n == 6
    assert restored.bid_window() == belief.bid_window()


def test_belief_dict_roundtrip_drops_pending_bid():
    """Beliefs persist posteriors; pending bids are transient."""
    belief = PresenceBelief()
    now = time.time()
    belief.note_bid(now, ["music"])
    belief.resolve_bid(False, now + 10.0)
    belief.note_bid(now + 20.0, ["dreams"])  # left open
    belief.activity_hours[9] += 5.0
    restored = PresenceBelief.from_dict(belief.to_dict())
    assert restored.responsiveness.evidence == 1.0
    assert restored.topic_receptivity["music"].mean < 0.5
    assert restored.activity_hours[9] == belief.activity_hours[9]
    assert restored.pending_bid is None
    # Malformed belief payloads downgrade to priors, not errors.
    assert PresenceBelief.from_dict("junk").responsiveness.evidence == 0.0
    assert PresenceBelief.from_dict(
        {"responsiveness": {"a": "x"}}
    ).responsiveness.evidence == 0.0


# ─── Beliefs: world integration ─────────────────────────────


def test_world_outreach_attributes_bid_to_engaged():
    """note_outreach opens a bid on who she'd expect an answer from."""
    world = _make_world()
    world.hear_user("hi")
    world.note_outreach(topics=["music"])
    user = world.presence("user")
    assert user.belief.pending_bid is not None
    assert user.belief.pending_bid.topics == ["music"]


def test_world_answered_bid_builds_belief():
    """A reply after her bid teaches the presence's responsiveness."""
    impulses = []
    world = _make_world(neuro_impulse=lambda c, a: impulses.append((c, a)))
    world.hear_user("hi")
    world.note_outreach(topics=["music"])
    world.hear_user("yes, tell me more")
    belief = world.presence("user").belief
    assert belief.responsiveness.evidence == 1.0
    assert belief.responsiveness.mean > 0.5
    assert belief.topic_receptivity["music"].mean > 0.5
    # Being answered carries its own small social reward.
    assert sum(a for c, a in impulses if c == CHEM_OXYTOCIN) > 0.05


def test_world_bid_expires_on_tick():
    """A bid past its window resolves unanswered on tick."""
    world = _make_world()
    world.hear_user("hi")
    world.note_outreach(topics=["music"])
    user = world.presence("user")
    user.belief.pending_bid.at -= BID_WINDOW + 60.0
    world.tick(now=time.time())
    belief = user.belief
    assert belief.pending_bid is None
    assert belief.responsiveness.evidence == 1.0
    assert belief.responsiveness.mean < 0.5


def test_world_departure_unanswers_bid():
    """A presence that leaves without answering didn't answer."""
    world = _make_world()
    world.hear_overheard("a voice")
    world.note_outreach(topics=["music"])
    voices = world.presence("voices")
    voices.last_seen = time.time() - (AMBIENT_PRESENCE_TIMEOUT + 60)
    world.tick(now=time.time())
    assert voices.belief.responsiveness.mean < 0.5


def test_world_beliefs_persist_across_restart():
    """Inferred state rides the same persistence as the world."""
    world = _make_world()
    world.hear_user("hi")
    world.note_outreach(topics=["music"])
    world.hear_user("of course")
    data = world.to_dict()
    restored = _make_world()
    restored.restore_from_dict(data)
    belief = restored.presence("user").belief
    assert belief.responsiveness.evidence == 1.0
    assert belief.responsiveness.mean > 0.5
    assert belief.topic_receptivity["music"].mean > 0.5


def test_summarize_includes_beliefs():
    """/world shows what she believes about who's there."""
    world = _make_world()
    world.hear_user("hi")
    world.note_outreach(topics=["music"])
    world.hear_user("yes")
    text = world.summarize()
    assert "replies" in text
