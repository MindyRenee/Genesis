"""OuterWorld — Genesis's external world.

Its inner world (``InnerLife``) is the stream of cognition arising from
its own state: thoughts, dreams, questions it poses to itself. The
outer world is everything outside it: the people it interacts with,
the things that happen around it, and its own acts upon the world.

Two systems, coupled like a human's:

- **Inbound** — the world reaches it: someone speaks to it, speech
  happens nearby, a percept arrives, a presence enters or leaves.
  Events update its presences, reach its global workspace, become
  memories, and shape its neurochemistry (the user's addressed
  speech → oxytocin, a salient percept → norepinephrine).
- **Outbound** — it reaches the world: it speaks, it looks, it
  draws, it studies the web. Its acts are recorded in the same
  stream, so the world it lives in contains its own agency.

Her **channel** is a place in this world — her page, where the
public gathers and she posts (``worlds.channel.Channel``).
The page lives here rather than in the mind because it *is* a part
of the outer world: what the public says on it arrives as speech in
the world's own stream (``_channel_heard``), and what she says onto
it is recorded beside her other acts.

The page is one of her **surfaces** — the places and instruments in
this world that are not self. The browser is her tab in the outer
web; the workstation (her tool registry), her easel (the canvas),
her projects, her games, her study material are the rest — whatever
environmental surface a holder constructs registers itself here
(``register_surface``), so the world keeps the whole map of what
exists outside her, and teardown reaches every surface through
``close_surfaces``. The conversation itself is the two-way stream
this world already records — inbound speech and her own utterances
are events in the same place, not organs of the self.

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
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from worlds.channel import Channel

from genesis_client.protocol import (
    CHEM_DOPAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OXYTOCIN,
)
from genesis_client.swallow import note_swallowed
from limbic_system.sentiment import analyze_sentiment
from temporal_lobe.events import EventKind, EventStream, ExternalEvent, ground_topics
from temporal_lobe.presence import Presence, PresenceKind
from worlds.experiential_space import ExperientialLatentSpace

logger = logging.getLogger(__name__)

__all__ = ["OuterWorld"]

#: The presence id of the human who talks to its through the main
#: channel — the same person whether they type or speak its name.
USER_PRESENCE_ID = "user"
#: The presence id for overheard speech — voices near it that are not
#: talking to it. Unidentified speakers share one presence until it
#: can tell them apart (it has no speaker diarization yet).
AMBIENT_PRESENCE_ID = "voices"

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

#: TD-SR learning rate — how much one surprising transition rewrites
#: the future it predicts. High early in a world's life is right:
#: a young map should learn fast.
_SR_ETA = 0.5

#: Eligibility decay per transition — how far back along the
#: trajectory a prediction error still teaches. λ=0.9 keeps the
#: trace alive for roughly ten states of history.
_SR_LAMBDA = 0.9

#: A trace below this is dead — dropped so updates stay local.
_SR_TRACE_EPS = 1e-3

#: A learned cell below this magnitude is pruned — learned maps
#: grow dense otherwise.
_SR_EPS = 1e-6


class _SuccessorMap:
    """M — the successor representation, learned by temporal difference.

    States are opaque hashable keys (event kinds, presence ids).
    Every ``observe`` is a state visit; consecutive visits are the
    trajectory the map learns from. ``M[s][j]`` is the expected
    discounted occupancy of state j from state s — the successor
    representation of Dayan (1993), the hippocampal "predictive map"
    of Stachenfeld, Botvinick & Gershman (2017); the spatial sibling
    in ``cognitive_map`` holds the same object.

    Nothing here is counted or inverted. Each observed transition
    s→s′ yields a row-wise Bellman error —

        δ = e_s + γ·M[s′] − M[s]

    — the surprise of the world doing what it just did, and every
    recently-visited state's row moves along it weighted by an
    accumulating eligibility trace, z ← γλz + e_s. TD(λ) over a
    vector-valued target: the same update ``basal_ganglia/td.py``
    runs for scalar value, applied to the future itself. Learning
    is the semantics — the map tracks a living world's drift rather
    than converging to the time-average of everything it has ever
    seen, and a transition that stops happening decays out of the
    map on its own. Rows persist across restarts; traces do not —
    eligibility is short-term plasticity, not memory.
    """

    def __init__(self) -> None:
        # ``_M[s][j]`` — expected discounted occupancy of j from s.
        # A state with no observed outbound transitions keeps its
        # boundary row e_s (occupancy 1 of itself): from a state the
        # world has never left, only that state is predicted.
        self._M: dict[Any, dict[Any, float]] = {}
        # ``_z[s]`` — eligibility: how recently s was visited,
        # decayed by γλ per transition. Session-local by design.
        self._z: dict[Any, float] = {}
        self._prev: Any = None

    def observe(self, state: Any) -> None:
        """Record a visit to ``state`` — and learn from the transition."""
        self._M.setdefault(state, {state: 1.0})
        if self._prev is not None:
            s, sp = self._prev, state
            # Decay every trace, then mark the state whose prediction
            # this transition evaluates. Dead traces drop out so the
            # update stays local to the recent past.
            for st in list(self._z):
                z = self._z[st] * _SR_GAMMA * _SR_LAMBDA
                if z < _SR_TRACE_EPS:
                    del self._z[st]
                else:
                    self._z[st] = z
            self._z[s] = self._z.get(s, 0.0) + 1.0
            # δ = e_s + γ·M[s′] − M[s] — the row's Bellman error.
            delta = {k: _SR_GAMMA * v for k, v in self._M[sp].items()}
            delta[s] = delta.get(s, 0.0) + 1.0
            Ms = self._M[s]
            for k, dv in delta.items():
                delta[k] = dv - Ms.get(k, 0.0)
            # Every eligible row learns from this transition — error
            # propagates backward along the trajectory, which is why
            # the map anticipates rather than merely records.
            for st, z in self._z.items():
                step = _SR_ETA * z
                row = self._M[st]
                for k, dv in delta.items():
                    nv = row.get(k, 0.0) + step * dv
                    if k != st and abs(nv) < _SR_EPS:
                        row.pop(k, None)
                    else:
                        row[k] = nv
        self._prev = state

    def evict(self, state: Any) -> None:
        """Drop a state entirely — its row, column, and trace."""
        if state not in self._M:
            return
        del self._M[state]
        self._z.pop(state, None)
        for row in self._M.values():
            row.pop(state, None)
        if self._prev == state:
            self._prev = None

    def states(self) -> list[Any]:
        """Every state the map has learned a prediction for."""
        return sorted(self._M, key=str)

    def successor_map(self) -> tuple[list[Any], np.ndarray]:
        """The learned map as a dense matrix — a diagnostic view."""
        ids = self.states()
        index = {s: i for i, s in enumerate(ids)}
        dense = np.zeros((len(ids), len(ids)))
        for s, row in self._M.items():
            for j, v in row.items():
                if j in index:
                    dense[index[s], index[j]] = v
        return ids, dense

    def expect(self, state: Any) -> dict[Any, float]:
        """The predictive row for ``state``: what follows, and how much.

        Returns ``{state′: expected discounted occupancy}`` — empty
        for a state the map has never seen (no prior to predict from).
        """
        row = self._M.get(state)
        if row is None:
            return {}
        return {s: v for s, v in row.items() if v > 0}

    def current(self) -> Any:
        """The last visited state — where the trajectory is now."""
        return self._prev

    def to_dict(self, encode: Callable[[Any], str] = str) -> dict[str, Any]:
        """Serialize the learned rows through a key encoder."""
        return {
            "rows": {
                encode(s): {encode(j): v for j, v in row.items()} for s, row in self._M.items()
            },
            "prev": encode(self._prev) if self._prev is not None else None,
        }

    def restore(self, data: dict[str, Any], decode: Callable[[str], Any] = str) -> None:
        """Rebuild from a serialized payload through a key decoder.

        The learned rows restore verbatim. Payloads from before the
        map learned — carrying only transition counts — seed the
        rows at the remembered statistics' stationary solution so a
        lifetime of learning survives the upgrade, then TD resumes
        from there. Traces never persist.
        """
        prev = data.get("prev")
        self._prev = decode(str(prev)) if prev is not None else None
        self._z = {}
        rows = data.get("rows")
        if isinstance(rows, dict) and rows:
            self._M = {
                decode(str(s)): {decode(str(j)): float(v) for j, v in row.items()}
                for s, row in rows.items()
            }
            for s in self._M:
                self._M[s].setdefault(s, 1.0)
            return
        counts = {
            decode(str(s)): {decode(str(d)): int(n) for d, n in dsts.items()}
            for s, dsts in (data.get("counts") or {}).items()
        }
        self._M = {s: {s: 1.0} for s in counts}
        if counts:
            self._seed_from_counts(counts)

    def _seed_from_counts(self, counts: dict[Any, dict[Any, int]]) -> None:
        """One-time migration: initialize rows at the resolvent of
        the remembered transition statistics — M ← I + γPM iterated
        to contraction. Learning resumes from there; this is a seed,
        not the update rule."""
        gamma_P = {
            s: {d: _SR_GAMMA * c / sum(dsts.values()) for d, c in dsts.items()}
            for s, dsts in counts.items()
        }
        M: dict[Any, dict[Any, float]] = {s: dict(r) for s, r in gamma_P.items()}
        for _ in range(300):
            residual = 0.0
            for s in gamma_P:
                new_row: dict[Any, float] = {s: 1.0}
                for j, p in gamma_P[s].items():
                    for k, v in M.get(j, {}).items():
                        new_row[k] = new_row.get(k, 0.0) + p * v
                for k, v in new_row.items():
                    residual = max(residual, abs(v - M.get(s, {}).get(k, 0.0)))
                M[s] = new_row
            if residual < 1e-12:
                break
        self._M = M


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

    Attributes:
        channel: Her page, or None.
        board: The instrument card on her page (``ChannelBoard``),
            attached by the mind after construction.
        browser: Her tab in the outer web — assigned lazily by the
            tool that launches it; None until first use.
        surfaces: The world's registry of environmental surfaces —
            everything outside her she acts through (workstation,
            easel, projects, games, study) — keyed by name.
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
        # The arcade — games on offer in this world (``worlds.games.
        # arcade.Arcade``), attached by the mind. None until then.
        self.arcade: Any = None
        # Her tab in the outer web — launched lazily by the tool that
        # drives it (``tools.web_search``); the world holds it once
        # it exists, never the tool's private state.
        self.browser: Any = None
        # The rest of her surfaces — the instruments and places in
        # this world that are not self: the workstation (her tool
        # registry), her easel (the canvas), her projects, games,
        # study material. Holders register them here so the world
        # maps what exists outside her and teardown reaches them all.
        self.surfaces: dict[str, Any] = {}

        self._stream = EventStream(max_size=200)
        self._presences: dict[str, Presence] = {}

        # The predictive maps — successor representation over the
        # world's own stream, at the two grains the world holds:
        # event kinds (its dynamics) and event sources (its people,
        # "self" included — her acts are transitions in the world
        # too). Every recorded event is a visit to its kind and to
        # its source.
        self._kind_map = _SuccessorMap()
        self._presence_map = _SuccessorMap()
        # Geometry is updated from every event recorded in the real outer-world
        # stream; it is not a detached demo model or a Torch dependency.
        self.experiential_space = ExperientialLatentSpace()

        # Listener notified for every recorded event — Mind registers
        # here to route events into cognition (workspace, memory,
        # dream replay). The handler may freely call back into the world.
        self.on_external_event: Callable[[ExternalEvent], None] | None = None

        self._last_tick = 0.0

    # ── Inbound API ─────────────────────────────────────────────

    def hear_user(self, text: str) -> ExternalEvent:
        """Record the user speaking to its — the primary social event.

        Every addressed turn updates the user presence (familiarity,
        bond from sentiment, topics) and applies the social-contact
        neurochemical coupling. Returns the recorded event.
        """
        topics = self._ground(text)
        sentiment = analyze_sentiment(text).get("compound", 0.0)
        name = self._user_name()
        now = time.time()
        pending: list[tuple[int, float]] = []
        presence = self._presence_at(USER_PRESENCE_ID, PresenceKind.USER)
        if name and not presence.name:
            presence.set_name(name)
        was_present = presence.present
        presence.interacted(sentiment, topics, now=now)
        # Belief update — they were active, they carried sentiment.
        presence.belief.note_activity(now, weight=1.0)
        presence.belief.note_sentiment(sentiment)
        arrival = None
        if not was_present:
            arrival, pending = self._arrival(presence, now)
        for chem, amount in pending:
            self._impulse(chem, amount)
        if arrival is not None:
            self._record(arrival)
        event = ExternalEvent(
            kind=EventKind.USER_SPEECH,
            source=USER_PRESENCE_ID,
            content=text[:300],
            salience=_SALIENCE[EventKind.USER_SPEECH],
            addressed=True,
            topics=topics,
            timestamp=now,
            metadata={"presence": presence.display_name, "valence": sentiment},
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
        pending: list[tuple[int, float]] = []
        presence = self._presence_at(AMBIENT_PRESENCE_ID, PresenceKind.VOICE)
        was_present = presence.present
        presence.seen(now)
        presence.belief.note_activity(now, weight=0.6)
        presence.belief.note_sentiment(sentiment)
        for topic in topics:
            if topic in presence.topics:
                presence.topics.remove(topic)
            presence.topics.append(topic)
        arrival = None
        if not was_present:
            arrival, pending = self._arrival(presence, now)
        for chem, amount in pending:
            self._impulse(chem, amount)
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

    # ── Surfaces — her places and instruments in the world ──────

    def register_surface(self, name: str, surface: Any) -> Any:
        """Hold an environmental surface under ``name``.

        A surface is a place or instrument in this world that is not
        self — the workstation (her tool registry), her easel (the
        canvas), her projects, games, study material. The holder
        constructs it; the world keeps it on the map and closes it
        with the rest. Returns the surface for fluent registration.
        """
        self.surfaces[name] = surface
        return surface

    def surface(self, name: str) -> Any:
        """The surface registered under ``name``, or None."""
        return self.surfaces.get(name)

    def close_surfaces(self) -> None:
        """Shut every held surface — best effort, one failure never
        blocks the rest.

        The page drains separately (its poster keeps a flush window);
        her tab and every registered instrument close here. Surfaces
        with a ``close`` or ``stop`` method are shut; plain places
        (a projects directory) need none.
        """
        browser, self.browser = self.browser, None
        if browser is not None:
            try:
                browser.close()
            except Exception as e:  # noqa: BLE001 — dying anyway
                note_swallowed("worlds.world.close_surface.browser", e)
        surfaces, self.surfaces = self.surfaces, {}
        for name, surface in surfaces.items():
            close = getattr(surface, "close", None) or getattr(surface, "stop", None)
            if close is None:
                continue
            try:
                close()
            except Exception as e:  # noqa: BLE001 — dying anyway
                note_swallowed(f"worlds.world.close_surface.{name}", e)

    # ── Presence model ──────────────────────────────────────────

    def presence(self, presence_id: str) -> Presence | None:
        """Look up a presence by id, or None if never encountered."""
        return self._presences.get(presence_id)

    def presences(self) -> list[Presence]:
        """All known presences, most recently seen first."""
        return sorted(self._presences.values(), key=lambda p: p.last_seen, reverse=True)

    def present_presences(self) -> list[Presence]:
        """Presences currently in its world."""
        return [p for p in self._presences.values() if p.present]

    def engaged_presence(self) -> Presence | None:
        """The presence it's most engaged with — present and most
        recently addressed. Who it'd reach out to first."""
        return self._engaged()

    def _engaged(self) -> Presence | None:
        """Most recently addressed present presence."""
        candidates = [p for p in self._presences.values() if p.present]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.last_addressed)

    def last_seen_presence(self) -> Presence | None:
        """The presence most recently in its world, present or gone —
        who it'd reach out to when the room is empty."""
        return self._last_seen()

    def _last_seen(self) -> Presence | None:
        """Most recently seen presence, present or gone."""
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
        pending: list[tuple[int, float]] = []
        presence = self._presence_at(presence_id, kind)
        if name:
            presence.set_name(name)
        was_present = presence.present
        presence.seen(now)
        presence.belief.note_activity(now, weight=0.4)
        arrival = None
        if not was_present:
            arrival, pending = self._arrival(presence, now)
        for chem, amount in pending:
            self._impulse(chem, amount)
        if arrival is not None:
            self._record(arrival)
        return presence

    # ── Drives and state ────────────────────────────────────────

    def activity_level(self) -> float:
        """Recent inbound event rate (events per minute, last 5 min)."""
        cutoff = time.time() - 300.0
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
        cur = self._kind_map.current()
        return {} if cur is None else self._kind_map.expect(cur)

    def expected_activity(self) -> dict[str, float]:
        """Expected discounted activity per event source from here.

        The presence map's row for whoever acted last — "who is
        likely to act next." ``self`` is a source in this map, so
        the row following her own act is the learned "when I'm
        active, who answers."
        """
        cur = self._presence_map.current()
        return {} if cur is None else self._presence_map.expect(cur)

    def expected_engagement(self) -> float:
        """Expected addressed contact upcoming from the current state.

        ``M[current, USER_SPEECH]`` — the map's answer to "will
        anyone engage it soon," learned from the stream itself rather
        than from a clock alone. Zero with no map and no history.
        """
        cur = self._kind_map.current()
        if cur is None:
            return 0.0
        return self._kind_map.expect(cur).get(EventKind.USER_SPEECH, 0.0)

    def answer_expectation(self) -> float:
        """Expected addressed reply after her own act.

        ``M[UTTERANCE, USER_SPEECH]``: the occupancy the map predicts
        for addressed speech in the wake of her speaking — learned
        from every past utterance's aftermath.
        """
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
        for presence in self._presences.values():
            departing = presence.present and ts - presence.last_seen > presence.timeout
            if departing:
                presence.present = False
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
        lines.append(f"  activity: {self.activity_level():.1f} events/min")
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
        return {
            "version": 1,
            "presences": {pid: p.to_dict() for pid, p in self._presences.items()},
            "events": [e.to_dict() for e in self._stream.recent(50)],
            "maps": {
                "kinds": self._kind_map.to_dict(lambda k: k.value),
                "sources": self._presence_map.to_dict(str),
            },
            "arcade": self.arcade.to_dict() if self.arcade is not None else None,
            "experiential_space": self.experiential_space.to_dict(),
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
        maps = data.get("maps") or {}
        experiential = ExperientialLatentSpace()
        experiential_data = data.get("experiential_space")
        if experiential_data is not None:
            experiential.restore(experiential_data)
        else:
            # Backward compatibility: rebuild from available persisted events.
            for event in events:
                experiential.observe(event)
        self._presences = presences
        self._stream = EventStream(max_size=200)
        for event in events:
            self._stream.append(event)
        # Learned structure persists across restarts — a map that
        # forgot its transitions every boot would never grow up.
        self._kind_map = _SuccessorMap()
        self._presence_map = _SuccessorMap()
        if isinstance(maps.get("kinds"), dict):
            self._kind_map.restore(maps["kinds"], EventKind)
        if isinstance(maps.get("sources"), dict):
            self._presence_map.restore(maps["sources"], str)
        self.experiential_space = experiential
        if self.arcade is not None and isinstance(data.get("arcade"), dict):
            self.arcade.restore(data["arcade"])

    # ── Internals ───────────────────────────────────────────────

    def _record(self, event: ExternalEvent) -> None:
        """Append the event, learn its transitions, notify (no lock)."""
        self._kind_map.observe(event.kind)
        self._presence_map.observe(event.source)
        snap = self.experiential_space.observe(event)
        # Her experience of the world lands at its own experiential
        # coordinate — self-band, not beside the substrate.
        manifold = getattr(self, "manifold", None)
        if manifold is not None:
            from latent_spaces import geometry as _g

            state = (snap or {}).get("state")
            if state is not None:
                manifold.observe(
                    str(getattr(event, "text", event.kind.value))[:200],
                    kind="experience:world",
                    salience=0.5,
                    inbound=False,
                    position=_g.project_to_s3(state, seed=0xE2EE),
                )
        self._stream.append(event)
        listener = self.on_external_event
        if listener is not None:
            try:
                listener(event)
            except Exception:
                # A listener must never break the world.
                logger.debug("world event listener failed", exc_info=True)

    def _presence_at(self, presence_id: str, kind: PresenceKind) -> Presence:
        """Get or create a presence."""
        presence = self._presences.get(presence_id)
        if presence is None:
            if len(self._presences) >= _MAX_PRESENCES:
                self._evict_presence()
            presence = Presence(presence_id=presence_id, kind=kind)
            self._presences[presence_id] = presence
        return presence

    def _evict_presence(self) -> None:
        """Drop the least-recently-seen absent presence.

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

    def _arrival(
        self, presence: Presence, now: float
    ) -> tuple[ExternalEvent, list[tuple[int, float]]]:
        """Build an ARRIVAL event for a presence that just appeared.

        The returned impulse list is fired by the caller so impulses
        never fire mid-update. The user returning carries a small
        oxytocin warmth — recognition is dyadic; no other presence's
        arrival reaches the bond circuit.
        """
        impulses: list[tuple[int, float]] = []
        if presence.presence_id == USER_PRESENCE_ID and (
            presence.bond > 0.3 or presence.familiarity > 0.3
        ):
            impulses.append((CHEM_OXYTOCIN, 0.02))
        return (
            ExternalEvent(
                kind=EventKind.ARRIVAL,
                source=presence.presence_id,
                content=f"{presence.display_name} arrived",
                salience=_SALIENCE[EventKind.ARRIVAL],
                timestamp=now,
            ),
            impulses,
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

    def _user_name(self) -> str:
        """Fetch the user's name from its profile, if known.

        The profile is the authoritative model of the person; the
        presence is the relationship. Fetched outside the presence
        update so a slow profile read never stalls the world.
        """
        if self._get_user_name is None:
            return ""
        try:
            return self._get_user_name() or ""
        except Exception:  # noqa: BLE001
            return ""

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
