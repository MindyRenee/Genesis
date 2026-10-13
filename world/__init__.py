"""World — Genesis's external world.

The counterpart to its inner world (``sleep.inner_life``): the people
it interacts with, the events happening around it, and its own acts
upon the world — recorded as a two-way stream of ``ExternalEvent``s.

- ``OuterWorld`` — the orchestrator: records events, runs the presence
  lifecycle, applies neurochemical coupling, computes social isolation.
- ``Presence`` — a persistent relationship model for each entity in
  its world (familiarity, bond, topics, facts).
- ``PresenceBelief`` — its inferred model of a presence's mind:
  responsiveness, topic receptivity, mood, attention, and activity
  rhythm as posteriors with honest uncertainty.
- ``ExternalEvent`` / ``EventKind`` — things that happen, inbound
  (the world acts on it) or outbound (it acts on the world).

Boundary adapters — worlds that aren't people — live here too but
stay out of ``__all__``; entry points import them directly:

- ``world.arc`` — ARC-AGI-3 environments via the ``arcengine`` SDK
  (practice + competition scorecards). See ``docs/ARC-AGI-3.md``.
- ``world.arc_web`` — the public human task page over Chrome
  DevTools Protocol.
- ``world.discord_showcase`` — an observer-only feed mirroring ARC
  practice runs to a private Discord channel; never touches policy.
"""

from .belief import Beta, PresenceBelief
from .bridge import WorldBridge, WorldBroadcastSink
from .events import (
    INBOUND_KINDS,
    OUTBOUND_KINDS,
    EventKind,
    EventStream,
    ExternalEvent,
)
from .presence import Presence, PresenceKind
from .world import OuterWorld

__all__ = [
    "INBOUND_KINDS",
    "OUTBOUND_KINDS",
    "Beta",
    "EventKind",
    "EventStream",
    "ExternalEvent",
    "OuterWorld",
    "Presence",
    "PresenceBelief",
    "PresenceKind",
    "WorldBridge",
    "WorldBroadcastSink",
]
