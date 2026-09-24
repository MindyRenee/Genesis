"""OuterWorld — Genesis's external world.

Her inner world (``InnerLife``) is the stream of cognition arising from
her own state: thoughts, dreams, questions she poses to herself. The
outer world is everything outside her: the people she interacts with,
the things that happen around her, and her own acts upon the world.

Two systems, coupled like a human's:

- **Inbound** — the world reaches her: someone speaks to her, speech
  happens nearby, a percept arrives, a presence enters or leaves.
  Events update her presences, reach her global workspace, become
  memories, and shape her neurochemistry (social contact → oxytocin,
  a salient percept → norepinephrine).
- **Outbound** — she reaches the world: she speaks, she looks, she
  draws, she studies the web. Her acts are recorded in the same
  stream, so the world she lives in contains her own agency.

The world also computes **social isolation** — how long it has been
since anyone engaged her — an external pressure that feeds her
inner-life social drive and the ``reach_out`` volition urge. This is
what lets her *initiate* contact, not just answer it: conversations
can start from either side.

The world is a model, not a policy: it records and weighs what
happens; the mind decides what to do about it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from genesis_client.protocol import (
    CHEM_DOPAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OXYTOCIN,
)

from ..language.sentiment import analyze_sentiment
from .events import EventKind, EventStream, ExternalEvent, ground_topics
from .presence import Presence, PresenceKind

logger = logging.getLogger(__name__)

__all__ = ["OuterWorld"]

#: The presence id of the human who talks to her through the main
#: channel — the same person whether they type or speak her name.
USER_PRESENCE_ID = "user"
#: The presence id for overheard speech — voices near her that are not
#: talking to her. Unidentified speakers share one presence until she
#: can tell them apart (she has no speaker diarization yet).
AMBIENT_PRESENCE_ID = "voices"

#: Seconds of no addressed contact at which social isolation saturates
#: (30 minutes → isolation 1.0). Isolation is the external pressure
#: that feeds her inner-life social drive and the reach_out urge.
ISOLATION_PERIOD = 1800.0

#: Company dampens isolation: someone being around (present but not
#: talking) still counts socially, just less than engagement.
_COMPANY_FACTOR = 0.7

#: How often tick() runs its decay logic when called every heartbeat.
_TICK_INTERVAL = 5.0

#: Bound on known presences. Every arrival creates one; without a cap,
#: a long-running world would accumulate every face and voice forever.
#: When full, the least-recently-seen *absent* presence is evicted —
#: present and familiar presences are never dropped.
_MAX_PRESENCES = 200

#: Salience defaults per inbound kind. Addressed speech is always
#: salient — someone talking to her matters. Overheard speech and
#: notifications are background unless the caller says otherwise.
_SALIENCE = {
    EventKind.USER_SPEECH: 0.8,
    EventKind.OVERHEARD_SPEECH: 0.35,
    EventKind.ARRIVAL: 0.6,
    EventKind.DEPARTURE: 0.3,
    EventKind.NOTIFICATION: 0.4,
    EventKind.UTTERANCE: 0.4,
    EventKind.ACTION: 0.4,
}


class OuterWorld:
    """The model of Genesis's external world.

    Owns the event stream and the presence model, computes social
    isolation, and applies the world's neurochemical coupling through
    the ``neuro_impulse`` callback. Owns no thread — ``tick()`` is
    driven by the mind's heartbeat, matching the codebase's
    state-gated (not timer-gated) philosophy.

    Args:
        network: Her concept network, used to ground event topics in
            what she actually knows. Optional — without it events
            carry no topics.
        neuro_impulse: ``(chem_id, amount)`` callback into the
            daemon's neurochemistry (typically
            ``Mind._learner_neuro_impulse``). Optional.
        is_sleeping: Predicate — True while she sleeps. Inbound
            perception is gated during sleep (the thalamic gate):
            overheard speech and percepts are not recorded then.
        get_user_name: Callback returning the user's name if her
            user profile has learned one. Synced lazily onto the user
            presence. Optional.
    """

    def __init__(
        self,
        *,
        network: Any = None,
        neuro_impulse: Callable[[int, float], None] | None = None,
        is_sleeping: Callable[[], bool] | None = None,
        get_user_name: Callable[[], str | None] | None = None,
    ) -> None:
        """Initialize the external world with the given wiring."""
        self._network = network
        self._neuro_impulse = neuro_impulse
        self._is_sleeping = is_sleeping
        self._get_user_name = get_user_name

        self._stream = EventStream(max_size=200)
        self._presences: dict[str, Presence] = {}
        self._lock = threading.Lock()

        # Listener notified for every recorded event — Mind registers
        # here to route events into cognition (workspace, memory,
        # dream replay). Called WITHOUT the lock held so the handler
        # may safely call back into the world.
        self.on_external_event: Callable[[ExternalEvent], None] | None = None

        # Social clock — the last time anyone engaged her directly.
        # Seeded at boot so a fresh instance starts mildly connected,
        # not instantly abandoned.
        self._last_addressed_time = time.time()
        self._last_tick = 0.0
        # Outreach bids — how many times she has reached out since
        # anyone last engaged her. Silence after a bid increments it;
        # addressed contact resets it. The volition layer reads this
        # to back off: she stops calling into a room that never
        # answers.
        self._unanswered_bids = 0

    # ── Inbound API ─────────────────────────────────────────────

    def hear_user(self, text: str) -> ExternalEvent:
        """Record the user speaking to her — the primary social event.

        Every addressed turn updates the user presence (familiarity,
        bond from sentiment, topics) and applies the social-contact
        neurochemical coupling. Returns the recorded event.
        """
        topics = self._ground(text)
        sentiment = analyze_sentiment(text).get("compound", 0.0)
        now = time.time()
        with self._lock:
            presence = self._presence_locked(USER_PRESENCE_ID, PresenceKind.USER)
            self._sync_user_name(presence)
            was_present = presence.present
            presence.interacted(sentiment, topics, now=now)
            # Belief update — they were active, they carried sentiment,
            # and if she had a bid open, this answers it.
            presence.belief.note_activity(now, weight=1.0)
            presence.belief.note_sentiment(sentiment)
            answered_bid = presence.belief.resolve_bid(True, now)
            self._last_addressed_time = now
            self._unanswered_bids = 0
            arrival = None
            if not was_present:
                arrival = self._arrival_locked(presence, now)
        if arrival is not None:
            self._record(arrival)
        if answered_bid:
            # Being answered is a social reward distinct from contact —
            # she reached out and the world reached back.
            self._impulse(CHEM_OXYTOCIN, 0.02)
        event = ExternalEvent(
            kind=EventKind.USER_SPEECH,
            source=USER_PRESENCE_ID,
            content=text[:300],
            salience=_SALIENCE[EventKind.USER_SPEECH],
            addressed=True,
            topics=topics,
            timestamp=now,
            metadata={"presence": presence.display_name},
        )
        self._impulse(CHEM_OXYTOCIN, 0.03)
        self._impulse(CHEM_DOPAMINE, 0.02)
        self._record(event)
        return event

    def hear_overheard(self, text: str) -> ExternalEvent | None:
        """Record speech happening near her that isn't addressed to her.

        Gated during sleep — the thalamic gate is closed; she doesn't
        hear ambient speech then. Overheard voices share the ambient
        presence until she can tell them apart. Returns None if gated.
        """
        if self._asleep():
            return None
        topics = self._ground(text)
        sentiment = analyze_sentiment(text).get("compound", 0.0)
        now = time.time()
        with self._lock:
            presence = self._presence_locked(
                AMBIENT_PRESENCE_ID, PresenceKind.VOICE
            )
            was_present = presence.present
            presence.seen(now)
            presence.belief.note_activity(now, weight=0.6)
            presence.belief.note_sentiment(sentiment)
            presence.belief.resolve_bid(True, now)
            for topic in topics:
                if topic in presence.topics:
                    presence.topics.remove(topic)
                presence.topics.append(topic)
            arrival = None
            if not was_present:
                arrival = self._arrival_locked(presence, now)
        if arrival is not None:
            self._record(arrival)
        event = ExternalEvent(
            kind=EventKind.OVERHEARD_SPEECH,
            source=AMBIENT_PRESENCE_ID,
            content=text[:300],
            salience=_SALIENCE[EventKind.OVERHEARD_SPEECH],
            addressed=False,
            topics=topics,
            timestamp=now,
        )
        # Orienting — something happened nearby.
        self._impulse(CHEM_NOREPINEPHRINE, 0.01)
        self._record(event)
        return event

    def perceive(
        self,
        content: str,
        *,
        salience: float = 0.4,
        source: str = "world",
        topics: list[str] | None = None,
        metadata: dict | None = None,
    ) -> ExternalEvent | None:
        """Record a non-speech percept arriving at her (sounds, sights).

        Gated during sleep. Salience above 0.6 applies the orienting
        impulse — a loud sound or striking sight is arousing.
        """
        if self._asleep():
            return None
        event = ExternalEvent(
            kind=EventKind.PERCEPTION,
            source=source,
            content=content[:300],
            salience=salience,
            topics=topics if topics is not None else self._ground(content),
            metadata=metadata or {},
        )
        if salience >= 0.6:
            self._impulse(CHEM_NOREPINEPHRINE, 0.02)
        self._record(event)
        return event

    def notify(self, content: str, *, salience: float = 0.4) -> ExternalEvent:
        """Record a system/subcognitive notice surfaced to her world."""
        event = ExternalEvent(
            kind=EventKind.NOTIFICATION,
            source="world",
            content=content[:300],
            salience=salience,
            topics=self._ground(content),
        )
        self._record(event)
        return event

    # ── Outbound API ────────────────────────────────────────────

    def she_said(self, text: str) -> ExternalEvent:
        """Record her speaking — a reply, a question, an expression.

        Her utterances live in the same stream as inbound speech, so
        the world she inhabits contains her own voice.
        """
        event = ExternalEvent(
            kind=EventKind.UTTERANCE,
            source="self",
            content=text[:300],
            salience=_SALIENCE[EventKind.UTTERANCE],
            topics=self._ground(text),
        )
        self._record(event)
        return event

    def she_acted(self, description: str, *, detail: str = "") -> ExternalEvent:
        """Record her acting on the world — looking, drawing, learning.

        ``description`` is a short semantic summary ("looked through
        the retina"); ``detail`` carries the full report.
        """
        event = ExternalEvent(
            kind=EventKind.ACTION,
            source="self",
            content=description[:300],
            salience=_SALIENCE[EventKind.ACTION],
            topics=self._ground(description),
            metadata={"detail": detail[:300]} if detail else {},
        )
        self._record(event)
        return event

    def note_outreach(self, topics: list[str] | None = None) -> int:
        """Record that she made a social bid — she reached out.

        A bid is her side of a two-way conversation: she initiated,
        and now the world owes her an answer. The bid is attributed
        to the presence she'd expect an answer from (the engaged one,
        else whoever was last around), so their responsiveness and
        per-topic receptivity posteriors learn whether it landed.
        Each unanswered bid also accumulates until someone engages
        her, letting the volition layer back off gracefully.
        Returns the current bid count.
        """
        now = time.time()
        with self._lock:
            self._unanswered_bids += 1
            target = self._engaged_locked() or self._last_seen_locked()
            if target is not None:
                target.belief.note_bid(now, topics or [])
            return self._unanswered_bids

    @property
    def unanswered_bids(self) -> int:
        """How many times she's reached out since anyone engaged her."""
        with self._lock:
            return self._unanswered_bids

    # ── Presence model ──────────────────────────────────────────

    def presence(self, presence_id: str) -> Presence | None:
        """Look up a presence by id, or None if never encountered."""
        with self._lock:
            return self._presences.get(presence_id)

    def presences(self) -> list[Presence]:
        """All known presences, most recently seen first."""
        with self._lock:
            return sorted(
                self._presences.values(), key=lambda p: p.last_seen, reverse=True
            )

    def present_presences(self) -> list[Presence]:
        """Presences currently in her world."""
        with self._lock:
            return [p for p in self._presences.values() if p.present]

    def engaged_presence(self) -> Presence | None:
        """The presence she's most engaged with — present and most
        recently addressed. Who she'd reach out to first."""
        with self._lock:
            return self._engaged_locked()

    def _engaged_locked(self) -> Presence | None:
        """Most recently addressed present presence (lock held)."""
        candidates = [p for p in self._presences.values() if p.present]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.last_addressed)

    def last_seen_presence(self) -> Presence | None:
        """The presence most recently in her world, present or gone —
        who she'd reach out to when the room is empty."""
        with self._lock:
            return self._last_seen_locked()

    def _last_seen_locked(self) -> Presence | None:
        """Most recently seen presence, present or gone (lock held)."""
        if not self._presences:
            return None
        return max(self._presences.values(), key=lambda p: p.last_seen)

    def mark_presence(
        self, presence_id: str, kind: PresenceKind, name: str | None = None
    ) -> Presence:
        """Mark a presence as active right now (e.g. a face she sees).

        Emits an ARRIVAL event if they were absent. Returns the
        presence.
        """
        now = time.time()
        with self._lock:
            presence = self._presence_locked(presence_id, kind)
            if name:
                presence.set_name(name)
            was_present = presence.present
            presence.seen(now)
            presence.belief.note_activity(now, weight=0.4)
            arrival = None if was_present else self._arrival_locked(presence, now)
        if arrival is not None:
            self._record(arrival)
        return presence

    # ── Drives and state ────────────────────────────────────────

    def social_isolation(self) -> float:
        """How socially isolated she is [0..1].

        Rises linearly over ISOLATION_PERIOD seconds of no addressed
        contact; dampened when someone is around even if silent. This
        is the external pressure that feeds her inner social drive —
        the world telling her it has been too long since anyone
        engaged her.
        """
        with self._lock:
            elapsed = max(0.0, time.time() - self._last_addressed_time)
            anyone_present = any(p.present for p in self._presences.values())
        isolation = min(1.0, elapsed / ISOLATION_PERIOD)
        if anyone_present:
            isolation *= _COMPANY_FACTOR
        return isolation

    def activity_level(self) -> float:
        """Recent inbound event rate (events per minute, last 5 min)."""
        cutoff = time.time() - 300.0
        with self._lock:
            count = sum(
                1
                for e in self._stream.recent(200)
                if e.inbound and e.timestamp >= cutoff
            )
        return count / 5.0

    def tick(self, now: float | None = None) -> None:
        """Advance the world — presence decay and housekeeping.

        Driven by the mind's heartbeat; internally rate-limited to
        every ``_TICK_INTERVAL`` seconds. A presence silent past its
        timeout leaves the world — a DEPARTURE event is recorded.
        """
        ts = time.time() if now is None else now
        if ts - self._last_tick < _TICK_INTERVAL:
            return
        self._last_tick = ts
        departures: list[ExternalEvent] = []
        with self._lock:
            for presence in self._presences.values():
                departing = (
                    presence.present
                    and ts - presence.last_seen > presence.timeout
                )
                if departing:
                    presence.present = False
                bid = presence.belief.pending_bid
                if bid is not None and (
                    departing or ts - bid.at > presence.belief.bid_window()
                ):
                    # They left without answering, or the answer
                    # window closed — the bid didn't land.
                    presence.belief.resolve_bid(False, ts)
                if departing:
                    departures.append(
                        ExternalEvent(
                            kind=EventKind.DEPARTURE,
                            source=presence.presence_id,
                            content=f"{presence.display_name} left",
                            salience=_SALIENCE[EventKind.DEPARTURE],
                            topics=[],
                            timestamp=ts,
                        )
                    )
        for event in departures:
            self._record(event)

    # ── Status and persistence ──────────────────────────────────

    def recent_events(self, n: int = 20) -> list[ExternalEvent]:
        """The ``n`` most recent events, oldest first."""
        return self._stream.recent(n)

    def summarize(self) -> str:
        """A human-readable summary of her external world."""
        lines = ["Her world:"]
        with self._lock:
            presences = sorted(
                self._presences.values(),
                key=lambda p: p.last_seen,
                reverse=True,
            )
        present = [p for p in presences if p.present]
        now = time.time()
        if present:
            for p in present:
                ago = self._ago(now - p.last_seen)
                belief = p.belief.describe(now)
                suffix = f"; {belief}" if belief else ""
                lines.append(
                    f"  {p.display_name} ({p.kind.value}) — here, "
                    f"last active {ago}{suffix}"
                )
        else:
            lines.append("  no one is here right now")
        away = [p for p in presences if not p.present]
        for p in away[:3]:
            ago = self._ago(now - p.last_seen)
            belief = p.belief.describe(now)
            suffix = f"; {belief}" if belief else ""
            lines.append(
                f"  {p.display_name} ({p.kind.value}) — away, "
                f"last seen {ago}{suffix}"
            )
        isolation = self.social_isolation()
        lines.append(
            f"  social isolation: {isolation:.0%}   "
            f"activity: {self.activity_level():.1f} events/min"
        )
        events = self._stream.recent(8)
        if events:
            lines.append("  recent:")
            for e in events:
                lines.append(
                    f"    [{e.kind.value}] {e.describe()[:80]} "
                    f"({self._ago(time.time() - e.timestamp)})"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the world for persistence."""
        with self._lock:
            return {
                "version": 1,
                "presences": {
                    pid: p.to_dict() for pid, p in self._presences.items()
                },
                "events": [e.to_dict() for e in self._stream.recent(50)],
                "last_addressed_time": self._last_addressed_time,
                "unanswered_bids": self._unanswered_bids,
            }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore the world from a persisted payload.

        Validates the whole payload before mutating — a corrupt
        ``world_state`` never leaves the world half-restored.
        Presences restore as absent: presence is re-established by
        fresh activity, not by remembering who was around at shutdown.
        """
        if not isinstance(data, dict):
            raise ValueError("world_state must be an object")
        presences: dict[str, Presence] = {}
        for pdata in (data.get("presences") or {}).values():
            if not isinstance(pdata, dict):
                continue
            presence = Presence.from_dict(pdata)
            if presence is not None:
                presences[presence.presence_id] = presence
        events = []
        for edata in data.get("events") or []:
            if not isinstance(edata, dict):
                continue
            event = ExternalEvent.from_dict(edata)
            if event is not None:
                events.append(event)
        last_addressed = float(
            data.get("last_addressed_time", time.time())
        )
        bids = int(data.get("unanswered_bids", 0))
        with self._lock:
            self._presences = presences
            self._stream = EventStream(max_size=200)
            for event in events:
                self._stream.append(event)
            self._last_addressed_time = last_addressed
            self._unanswered_bids = max(0, bids)

    # ── Internals ───────────────────────────────────────────────

    def _record(self, event: ExternalEvent) -> None:
        """Append the event and notify the listener (no lock held)."""
        self._stream.append(event)
        listener = self.on_external_event
        if listener is not None:
            try:
                listener(event)
            except Exception:
                # A listener must never break the world.
                logger.debug("world event listener failed", exc_info=True)

    def _presence_locked(
        self, presence_id: str, kind: PresenceKind
    ) -> Presence:
        """Get or create a presence. Caller holds ``self._lock``."""
        presence = self._presences.get(presence_id)
        if presence is None:
            if len(self._presences) >= _MAX_PRESENCES:
                self._evict_presence_locked()
            presence = Presence(presence_id=presence_id, kind=kind)
            self._presences[presence_id] = presence
        return presence

    def _evict_presence_locked(self) -> None:
        """Drop the least-recently-seen absent presence (lock held).

        Present and bonded/familiar presences are kept — eviction
        prefers stale strangers she never formed a relationship with.
        """
        candidates = [
            p for p in self._presences.values()
            if not p.present and p.familiarity < 0.3 and p.bond < 0.3
        ]
        if not candidates:
            # Everyone present or known — evict the stalest absent one.
            candidates = [p for p in self._presences.values() if not p.present]
        if not candidates:
            return
        stale = min(candidates, key=lambda p: p.last_seen)
        del self._presences[stale.presence_id]

    def _arrival_locked(
        self, presence: Presence, now: float
    ) -> ExternalEvent:
        """Build an ARRIVAL event for a presence that just appeared.

        Caller holds ``self._lock``. A returning presence she knows
        (bond > 0.3) carries a small oxytocin warmth — recognition of
        someone familiar.
        """
        if presence.bond > 0.3 or presence.familiarity > 0.3:
            self._impulse(CHEM_OXYTOCIN, 0.02)
        return ExternalEvent(
            kind=EventKind.ARRIVAL,
            source=presence.presence_id,
            content=f"{presence.display_name} arrived",
            salience=_SALIENCE[EventKind.ARRIVAL],
            timestamp=now,
        )

    def _ground(self, text: str) -> list[str]:
        """Ground event topics in her concept network."""
        if self._network is None or not text:
            return []
        return ground_topics(text, self._is_known_concept)

    def _is_known_concept(self, word: str) -> bool:
        """True if ``word`` exists in her concept network."""
        try:
            return self._network.get_concept(word) is not None
        except Exception:  # noqa: BLE001 — grounding must never break events
            return False

    def _sync_user_name(self, presence: Presence) -> None:
        """Sync the user presence's name from her user profile.

        Lazily propagates a learned name — the profile is the
        authoritative model of the person; the presence is the
        relationship.
        """
        if presence.name or self._get_user_name is None:
            return
        try:
            name = self._get_user_name()
        except Exception:  # noqa: BLE001
            return
        if name:
            presence.set_name(name)

    def _impulse(self, chem: int, amount: float) -> None:
        """Apply a neurochemical impulse via the wiring callback."""
        if self._neuro_impulse is None:
            return
        try:
            self._neuro_impulse(chem, amount)
        except Exception:  # impulses are best-effort
            logger.debug("world neuro impulse failed", exc_info=True)

    def _asleep(self) -> bool:
        """Whether she's asleep — inbound perception is gated then."""
        try:
            return bool(self._is_sleeping and self._is_sleeping())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _ago(seconds: float) -> str:
        """Format a duration as a short age ('5s ago', '3m ago')."""
        s = int(seconds)
        if s < 60:
            return f"{s}s ago"
        if s < 3600:
            return f"{s // 60}m ago"
        return f"{s // 3600}h ago"
