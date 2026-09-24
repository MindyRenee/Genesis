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
"""

from .belief import Beta, PresenceBelief
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
]
