"""World — Genesis's external world.

The counterpart to her inner world (``sleep.inner_life``): the people
she interacts with, the events happening around her, and her own acts
upon the world — recorded as a two-way stream of ``ExternalEvent``s.

- ``OuterWorld`` — the orchestrator: records events, runs the presence
  lifecycle, applies neurochemical coupling, computes social isolation.
- ``Presence`` — a persistent relationship model for each entity in
  her world (familiarity, bond, topics, facts).
- ``PresenceBelief`` — her inferred model of a presence's mind:
  responsiveness, topic receptivity, mood, attention, and activity
  rhythm as posteriors with honest uncertainty.
- ``ExternalEvent`` / ``EventKind`` — things that happen, inbound
  (the world acts on her) or outbound (she acts on the world).
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
