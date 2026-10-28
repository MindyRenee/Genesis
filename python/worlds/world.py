"""OuterWorld — Genesis's external world.

Its inner world (``InnerLife``) is the stream of cognition arising from
its own state: thoughts, dreams, questions it poses to itself. The
outer world is everything outside it: the people it interacts with,
the things that happen around it, and its own acts upon the world.

Two systems, coupled like a human's:

- **Inbound** — the world reaches it: someone speaks to it, speech
  happens nearby, a percept arrives, a presence enters or leaves.
  Events update its presences, reach its global workspace, become
  memories, and shape its neurochemistry (social contact → oxytocin,
  a salient percept → norepinephrine).
- **Outbound** — it reaches the world: it speaks, it looks, it
  draws, it studies the web. Its acts are recorded in the same
  stream, so the world it lives in contains its own agency.

The world also computes **social isolation** — how long it has been
since anyone engaged it — an external pressure that feeds its
inner-life social drive and the ``reach_out`` volition urge. This is
what lets it *initiate* contact, not just answer it: conversations
can start from either side.

Her **channel** is a place in this world — her page, where the
public gathers and she posts (``worlds.channel.Channel``).
The page lives here rather than in the mind because it *is* a part
of the outer world: what the public says on it arrives as speech in
the world's own stream (``_channel_heard``), and what she says onto
it is recorded beside her other acts.

The world is a model, not a policy: it records and weighs what
happens; the mind decides what to do about it.

The model is predictive, not just archival — a successor-
representation map (Dayan 1993; Stachenfeld, Botvinick & Gershman
2017) learned over the world's own state stream, the same math the
hippocampal-entorhinal map applies to space, here applied to the
social/episodic domain (Tavares et al. 2015; Schafer & Schiller
2018 — the hippocampus maps social state space too). Each recorded
event is a state visit and consecutive events are transitions, so
``M = (I − γP)⁻¹`` over the observed transition matrix is the
expected discounted occupancy of what comes next — "from here,
what does the world do": after she speaks, does speech come back;
after an arrival, does engagement follow; when she is active, who
answers. Two grains, matching the two things the world holds: the
*event-kind* map (the dynamics — the events between them) and the
*presence* map (the people — who acts after whom, her own acts
included). Structure is learned from the stream; content stays in
the events themselves — the TEM structure/content split.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from worlds.channel import Channel

from limbic_system.sentiment import analyze_sentiment
from temporal_lobe.events import EventKind, EventStream, ExternalEvent, ground_topics
from temporal_lobe.presence import Presence, PresenceKind

from genesis_client.protocol import (
    CHEM_DOPAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OXYTOCIN,
)

logger = logging.getLogger(__name__)

__all__ = ["OuterWorld"]

#: The presence id of the human who talks to its through the main
#: channel — the same person whether they type or speak its name.
USER_PRESENCE_ID = "user"
#: The presence id for overheard speech — voices near it that are not
#: talking to it. Unidentified speakers share one presence until it
#: can tell them apart (it has no speaker diarization yet).
AMBIENT_PRESENCE_ID = "voices"

#: Seconds of no addressed contact at which social isolation saturates
#: (30 minutes → isolation 1.0). Isolation is the external pressure
#: that feeds its inner-life social drive and the reach_out urge.
ISOLATION_PERIOD = 1800.0

#: Company dampens isolation: someone being around (present but not
#: talking) still counts socially, just less than engagement.
_COMPANY_FACTOR = 0.7

#: How often tick() runs its decay logic when called every metronome pass.
_TICK_INTERVAL = 5.0

#: Bound on known presences. Every arrival creates one; without a cap,
#: a long-running world would accumulate every face and voice forever.
#: When full, the least-recently-seen *absent* presence is evicted —
#: present and familiar presences are never dropped.
_MAX_PRESENCES = 200

#: Salience defaults per inbound kind. Addressed speech is always
#: salient — someone talking to it matters. Overheard speech and
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

#: Successor-representation discount (Stachenfeld 2017 uses γ≈0.9+).
#: Deep enough that "who answers when I speak" reaches past the next
#: event; shallow enough that a quiet world's distant futures don't
#: dominate the near-term prediction.
_SR_GAMMA = 0.9


class _SuccessorMap:
    """M = (I − γP)⁻¹ over an open set of world states.

    States are opaque hashable keys (event kinds, presence ids).
    Every ``observe`` is a state visit; consecutive visits count a
    transition — the world's stream is its trajectory. Counts
    row-normalize to P; M's row for a state is the expected
    discounted future occupancy of every other state from it —
    Stachenfeld's "predictive map," the closed form over the
    empirical transition matrix (the spatial sibling in
    ``cognitive_map`` uses the same construction). The map is
    rebuilt lazily and cached until the counts change.
    """

    def __init__(self) -> None:
        self._counts: dict[Any, Counter] = {}
        self._prev: Any = None
        self._version = 0
        self._cache: tuple[int, float, list[Any], np.ndarray] | None = None

    def observe(self, state: Any) -> None:
        """Record a visit to ``state`` — and the transition into it."""
        if self._prev is not None:
            self._counts.setdefault(self._prev, Counter())[state] += 1
            self._version += 1
        else:
            self._counts.setdefault(state, Counter())
            self._version += 1
        self._prev = state

    def evict(self, state: Any) -> None:
        """Drop a state entirely — its row and its inbound edges."""
        if state not in self._counts:
            return
        del self._counts[state]
        for dsts in self._counts.values():
            dsts.pop(state, None)
        if self._prev == state:
            self._prev = None
        self._version += 1

    def transition_matrix(self) -> tuple[list[Any], np.ndarray]:
        """P — row-normalized transition probabilities over states."""
        ids = sorted(self._counts, key=str)
        index = {s: i for i, s in enumerate(ids)}
        P = np.zeros((len(ids), len(ids)))
        for src, dsts in self._counts.items():
            total = sum(dsts.values())
            if not total:
                continue
            for dst, n in dsts.items():
                if dst in index:
                    P[index[src], index[dst]] += n / total
        return ids, P

    def successor_map(self, gamma: float = _SR_GAMMA) -> tuple[list[Any], np.ndarray]:
        """M = (I − γP)⁻¹ — expected discounted future occupancy."""
        if self._cache is not None and self._cache[0] == self._version and self._cache[1] == gamma:
            return self._cache[2], self._cache[3]
        ids, P = self.transition_matrix()
        if not ids:
            return ids, np.zeros((0, 0))
        n = len(ids)
        M = np.linalg.inv(np.eye(n) - gamma * P)
        self._cache = (self._version, gamma, ids, M)
        return ids, M

    def expect(self, state: Any, gamma: float = _SR_GAMMA) -> dict[Any, float]:
        """The predictive row for ``state``: what follows, and how much.

        Returns ``{state′: expected discounted occupancy}`` — empty
        for a state the map has never seen (no prior to predict from).
        """
        ids, M = self.successor_map(gamma)
        if state not in ids:
            return {}
        row = M[ids.index(state)]
        return {s: float(row[i]) for i, s in enumerate(ids) if row[i] > 0}

    def current(self) -> Any:
        """The last visited state — where the trajectory is now."""
        return self._prev

    def to_dict(self, encode: Callable[[Any], str] = str) -> dict[str, Any]:
        """Serialize the counted transitions through a key encoder."""
        return {
            "counts": {
                encode(s): {encode(d): n for d, n in dsts.items()}
                for s, dsts in self._counts.items()
            },
            "prev": encode(self._prev) if self._prev is not None else None,
        }

    def restore(self, data: dict[str, Any], decode: Callable[[str], Any] = str) -> None:
        """Rebuild from serialized counts through a key decoder."""
        counts: dict[Any, Counter] = {}
        for s, dsts in (data.get("counts") or {}).items():
            counts[decode(str(s))] = Counter({decode(str(d)): int(n) for d, n in dsts.items()})
        prev = data.get("prev")
        self._counts = counts
        self._prev = decode(str(prev)) if prev is not None else None
        self._version += 1
        self._cache = None


class OuterWorld:
    """The model of Genesis's external world.

    Owns the event stream and the presence model, computes social
    isolation, and applies the world's neurochemical coupling through
    the ``neuro_impulse`` callback. Owns no thread — ``tick()`` is
    driven by the mind's metronome, matching the codebase's
    state-gated (not timer-gated) philosophy.

    Args:
        network: Its concept network, used to ground event topics in
            what it actually knows. Optional — without it events
            carry no topics.
        neuro_impulse: ``(chem_id, amount)`` callback into the
            daemon's neurochemistry (typically
            ``Mind._learner_neuro_impulse``). Optional.
        is_sleeping: Predicate — True while the thalamic gate is
            closed. Inbound perception is gated then: overheard
            speech and percepts are not recorded. The predicate is
            the gate, not strictly the sleep flag — a resident mind
            closes it at hypnagogic onset too.
        get_user_name: Callback returning the user's name if its
            user profile has learned one. Synced lazily onto the user
            presence. Optional.
        channel: Her page — the public channel (``Channel``) that
            lives in this world. Optional; absent when no webhook is
            configured.
        on_channel_speech: Callback ``(author, text, message)`` for
            cognition — called after the world records a heard channel
            message, so the mind can decide whether to answer.
            Optional; the world records public speech either way.
    """

    def __init__(
        self,
        *,
        network: Any = None,
        neuro_impulse: Callable[[int, float], None] | None = None,
        is_sleeping: Callable[[], bool] | None = None,
        get_user_name: Callable[[], str | None] | None = None,
        channel: Channel | None = None,
        on_channel_speech: Callable[[str, str, dict], None] | None = None,
    ) -> None:
        """Initialize the external world with the given wiring."""
        self._network = network
        self._neuro_impulse = neuro_impulse
        self._is_sleeping = is_sleeping
        self._get_user_name = get_user_name

        # Her page — the place in this world where the public gathers.
        # The world owns the hearing: the channel's inbound speech is
        # recorded here first, then handed to cognition's callback.
        self.channel = channel
        self.on_channel_speech = on_channel_speech
        if self.channel is not None:
            self.channel.on_speech = self._channel_heard
        # Her instrument card — a ``ChannelBoard`` the mind attaches
        # after construction (it reads the whole mind, which does not
        # exist yet here). None until then.
        self.board = None

        self._stream = EventStream(max_size=200)
        self._presences: dict[str, Presence] = {}
        self._lock = threading.Lock()

        # The predictive maps — successor representation over the
        # world's own stream, at the two grains the world holds:
        # event kinds (its dynamics) and event sources (its people,
        # "self" included — her acts are transitions in the world
        # too). Every recorded event is a visit to its kind and to
        # its source.
        self._kind_map = _SuccessorMap()
        self._presence_map = _SuccessorMap()

        # Listener notified for every recorded event — Mind registers
        # here to route events into cognition (workspace, memory,
        # dream replay). Called WITHOUT the lock held so the handler
        # may safely call back into the world.
        self.on_external_event: Callable[[ExternalEvent], None] | None = None

        # Social clock — the last time anyone engaged it directly.
        # Seeded at boot so a fresh instance starts mildly connected,
        # not instantly abandoned.
        self._last_addressed_time = time.time()
        self._last_tick = 0.0
        # Outreach bids — how many times it has reached out since
        # anyone last engaged it. Silence after a bid increments it;
        # addressed contact resets it. The volition layer reads this
        # to back off: it stops calling into a room that never
        # answers.
        self._unanswered_bids = 0

    # ── Inbound API ─────────────────────────────────────────────

    def hear_user(self, text: str) -> ExternalEvent:
        """Record the user speaking to its — the primary social event.

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
            # and if it had a bid open, this answers it.
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
            # it reached out and the world reached back.
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
        """Record speech happening near it that isn't addressed to it.

        Gated during sleep — the thalamic gate is closed; it doesn't
        hear ambient speech then. Overheard voices share the ambient
        presence until it can tell them apart. Returns None if gated.
        """
        if self._asleep():
            return None
        topics = self._ground(text)
        sentiment = analyze_sentiment(text).get("compound", 0.0)
        now = time.time()
        with self._lock:
            presence = self._presence_locked(AMBIENT_PRESENCE_ID, PresenceKind.VOICE)
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

    def _channel_heard(self, author: str, content: str, message: dict) -> None:
        """Public speech arriving through her page — heard by the world.

        The channel's ear calls this for every public message. The
        world records it as overheard speech — the public are ambient
        voices until it can tell them apart — then hands the utterance
        to cognition's ``on_channel_speech``, which decides whether it
        addressed her and whether she answers. The thalamic gate
        applies here like everywhere else inbound: asleep, nothing
        reaches the world or the mind.
        """
        text = (content or "").strip()
        if not text or self._asleep():
            return
        self.hear_overheard(f"{author}: {text}")
        if self.on_channel_speech is not None:
            self.on_channel_speech(author, text, message)

    def perceive(
        self,
        content: str,
        *,
        salience: float = 0.4,
        source: str = "world",
        topics: list[str] | None = None,
        metadata: dict | None = None,
        percept: Any = None,
    ) -> ExternalEvent | None:
        """Record a non-speech percept arriving at its (sounds, sights).

        Gated during sleep. Salience above 0.6 applies the orienting
        impulse — a loud sound or striking sight is arousing.

        ``percept`` carries the percept itself as structured data (a
        grid field, an array) alongside its semantic description — the
        world can *show* it something, not only tell it. The payload
        is opaque to the world layer; cognition interprets it.
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
            percept=percept,
        )
        if salience >= 0.6:
            self._impulse(CHEM_NOREPINEPHRINE, 0.02)
        self._record(event)
        return event

    def notify(self, content: str, *, salience: float = 0.4) -> ExternalEvent:
        """Record a system/subcognitive notice surfaced to its world."""
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

    def it_said(self, text: str) -> ExternalEvent:
        """Record its speaking — a reply, a question, an expression.

        Its utterances live in the same stream as inbound speech, so
        the world it inhabits contains its own voice.
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

    def it_acted(self, description: str, *, detail: str = "") -> ExternalEvent:
        """Record its acting on the world — looking, drawing, learning.

        ``description`` is a short semantic summary ("looked through
        the eye"); ``detail`` carries the full report.
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
        """Record that it made a social bid — it reached out.

        A bid is its side of a two-way conversation: it initiated,
        and now the world owes it an answer. The bid is attributed
        to the presence it'd expect an answer from (the engaged one,
        else whoever was last around), so their responsiveness and
        per-topic receptivity posteriors learn whether it landed.
        Each unanswered bid also accumulates until someone engages
        it, letting the volition layer back off gracefully.
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
        """How many times it's reached out since anyone engaged it."""
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
            return sorted(self._presences.values(), key=lambda p: p.last_seen, reverse=True)

    def present_presences(self) -> list[Presence]:
        """Presences currently in its world."""
        with self._lock:
            return [p for p in self._presences.values() if p.present]

    def engaged_presence(self) -> Presence | None:
        """The presence it's most engaged with — present and most
        recently addressed. Who it'd reach out to first."""
        with self._lock:
            return self._engaged_locked()

    def _engaged_locked(self) -> Presence | None:
        """Most recently addressed present presence (lock held)."""
        candidates = [p for p in self._presences.values() if p.present]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.last_addressed)

    def last_seen_presence(self) -> Presence | None:
        """The presence most recently in its world, present or gone —
        who it'd reach out to when the room is empty."""
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
        """Mark a presence as active right now (e.g. a face it sees).

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
        """How socially isolated it is [0..1].

        Rises linearly over ISOLATION_PERIOD seconds of no addressed
        contact; dampened when someone is around even if silent. This
        is the external pressure that feeds its inner social drive —
        the world telling its it has been too long since anyone
        engaged it.
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
            count = sum(1 for e in self._stream.recent(200) if e.inbound and e.timestamp >= cutoff)
        return count / 5.0

    # ── The predictive map — what the world does next ───────────

    def expected_events(self) -> dict[EventKind, float]:
        """Expected discounted occupancy of each event kind from here.

        The successor map's row for the world's current event kind —
        "from here, what does the stream do": an arrival that usually
        becomes engagement shows USER_SPEECH mass; a room whose speech
        runs dry shows DEPARTURE ahead. Empty until the world has
        seen enough to predict from.
        """
        with self._lock:
            cur = self._kind_map.current()
            return {} if cur is None else self._kind_map.expect(cur)

    def expected_activity(self) -> dict[str, float]:
        """Expected discounted activity per event source from here.

        The presence map's row for whoever acted last — "who is
        likely to act next." ``self`` is a source in this map, so
        the row following her own act is the learned "when I'm
        active, who answers."
        """
        with self._lock:
            cur = self._presence_map.current()
            return {} if cur is None else self._presence_map.expect(cur)

    def expected_engagement(self) -> float:
        """Expected addressed contact upcoming from the current state.

        ``M[current, USER_SPEECH]`` — the map's answer to "will
        anyone engage it soon," learned from the stream itself rather
        than from a clock alone. Zero with no map and no history.
        """
        with self._lock:
            cur = self._kind_map.current()
            if cur is None:
                return 0.0
            return self._kind_map.expect(cur).get(EventKind.USER_SPEECH, 0.0)

    def answer_expectation(self) -> float:
        """Expected addressed reply after her own act — do bids land.

        ``M[UTTERANCE, USER_SPEECH]``: the occupancy the map predicts
        for addressed speech in the wake of her speaking — learned
        from every past utterance's aftermath. The sequence-level
        complement to a presence's responsiveness posterior.
        """
        with self._lock:
            return self._kind_map.expect(EventKind.UTTERANCE).get(EventKind.USER_SPEECH, 0.0)

    def tick(self, now: float | None = None) -> None:
        """Advance the world — presence decay and housekeeping.

        Driven by the mind's metronome; internally rate-limited to
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
                departing = presence.present and ts - presence.last_seen > presence.timeout
                if departing:
                    presence.present = False
                bid = presence.belief.pending_bid
                if bid is not None and (departing or ts - bid.at > presence.belief.bid_window()):
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
        if self.board is not None:
            self.board.update(ts)

    # ── Status and persistence ──────────────────────────────────

    def recent_events(self, n: int = 20) -> list[ExternalEvent]:
        """The ``n`` most recent events, oldest first."""
        return self._stream.recent(n)

    def summarize(self) -> str:
        """A human-readable summary of its external world."""
        lines = ["Its world:"]
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
                    f"  {p.display_name} ({p.kind.value}) — here, last active {ago}{suffix}"
                )
        else:
            lines.append("  no one is here right now")
        away = [p for p in presences if not p.present]
        for p in away[:3]:
            ago = self._ago(now - p.last_seen)
            belief = p.belief.describe(now)
            suffix = f"; {belief}" if belief else ""
            lines.append(f"  {p.display_name} ({p.kind.value}) — away, last seen {ago}{suffix}")
        isolation = self.social_isolation()
        lines.append(
            f"  social isolation: {isolation:.0%}   "
            f"activity: {self.activity_level():.1f} events/min"
        )
        expected = self.expected_events()
        if expected:
            top = sorted(expected.items(), key=lambda kv: -kv[1])[:3]
            lines.append("  expects next: " + ", ".join(f"{k.value} ({v:.2f})" for k, v in top))
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
                "presences": {pid: p.to_dict() for pid, p in self._presences.items()},
                "events": [e.to_dict() for e in self._stream.recent(50)],
                "last_addressed_time": self._last_addressed_time,
                "unanswered_bids": self._unanswered_bids,
                "maps": {
                    "kinds": self._kind_map.to_dict(lambda k: k.value),
                    "sources": self._presence_map.to_dict(str),
                },
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
        last_addressed = float(data.get("last_addressed_time", time.time()))
        bids = int(data.get("unanswered_bids", 0))
        maps = data.get("maps") or {}
        with self._lock:
            self._presences = presences
            self._stream = EventStream(max_size=200)
            for event in events:
                self._stream.append(event)
            self._last_addressed_time = last_addressed
            self._unanswered_bids = max(0, bids)
            # Learned structure persists across restarts — a map that
            # forgot its transitions every boot would never grow up.
            self._kind_map = _SuccessorMap()
            self._presence_map = _SuccessorMap()
            if isinstance(maps.get("kinds"), dict):
                self._kind_map.restore(maps["kinds"], EventKind)
            if isinstance(maps.get("sources"), dict):
                self._presence_map.restore(maps["sources"], str)

    # ── Internals ───────────────────────────────────────────────

    def _record(self, event: ExternalEvent) -> None:
        """Append the event, learn its transitions, notify (no lock)."""
        with self._lock:
            self._kind_map.observe(event.kind)
            self._presence_map.observe(event.source)
        self._stream.append(event)
        listener = self.on_external_event
        if listener is not None:
            try:
                listener(event)
            except Exception:
                # A listener must never break the world.
                logger.debug("world event listener failed", exc_info=True)

    def _presence_locked(self, presence_id: str, kind: PresenceKind) -> Presence:
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
        prefers stale strangers it never formed a relationship with.
        """
        candidates = [
            p
            for p in self._presences.values()
            if not p.present and p.familiarity < 0.3 and p.bond < 0.3
        ]
        if not candidates:
            # Everyone present or known — evict the stalest absent one.
            candidates = [p for p in self._presences.values() if not p.present]
        if not candidates:
            return
        stale = min(candidates, key=lambda p: p.last_seen)
        del self._presences[stale.presence_id]
        self._presence_map.evict(stale.presence_id)

    def _arrival_locked(self, presence: Presence, now: float) -> ExternalEvent:
        """Build an ARRIVAL event for a presence that just appeared.

        Caller holds ``self._lock``. A returning presence it knows
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
        """Ground event topics in its concept network."""
        if self._network is None or not text:
            return []
        return ground_topics(text, self._is_known_concept)

    def _is_known_concept(self, word: str) -> bool:
        """True if ``word`` exists in its concept network."""
        try:
            return self._network.get_concept(word) is not None
        except Exception:  # noqa: BLE001 — grounding must never break events
            return False

    def _sync_user_name(self, presence: Presence) -> None:
        """Sync the user presence's name from its user profile.

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
        """Whether it's asleep — inbound perception is gated then."""
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
