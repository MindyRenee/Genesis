"""Presences — who is in Genesis's external world.

A presence is an entity outside her that she can interact with: the
user at the terminal, a voice in the room, a face she recognizes.
Presences are her *social world* — each carries a persistent
relationship model:

- **familiarity** — how well she knows them (grows with interaction,
  asymptotically — a stranger becomes known quickly, then deepens slowly)
- **bond** — how the relationship feels (grows slowly from
  positive-valence interaction, erodes with negative)
- **topics / facts** — what they've talked about and what she's learned
  about them, grounded in her concept network

Presences come and go. One that speaks becomes *present*; one that
falls silent past a timeout *leaves* — the world notices both.

This is distinct from ``user_profile`` (her deep model of the primary
human) — the presence model covers *everyone* out there, not just the
person at the keyboard, and tracks the relationship itself rather than
the person's attributes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .belief import PresenceBelief

#: How much a single addressed interaction grows familiarity. The
#: update is asymptotic (``f += (1-f) * rate``), so early exchanges
#: teach her a lot and later ones add nuance.
_FAMILIARITY_RATE = 0.08

#: Bond moves slower than familiarity — trust accumulates over many
#: interactions, not one. Positive sentiment grows it; negative erodes
#: it at the same slow rate.
_BOND_RATE = 0.04

#: Sentiment above/below these bounds counts as positive/negative
#: interaction for the bond update.
_BOND_POSITIVE = 0.1
_BOND_NEGATIVE = -0.2

#: Seconds before a silent presence is considered to have left. The
#: user at the terminal gets a longer leash — they may be reading,
#: thinking, or away at the keyboard but still "around".
USER_PRESENCE_TIMEOUT = 1800.0
AMBIENT_PRESENCE_TIMEOUT = 900.0


class PresenceKind(Enum):
    """What sort of entity a presence is."""

    USER = "user"        # the human who talks to her through the main channel
    VOICE = "voice"      # an unidentified speaker in the room
    FACE = "face"        # someone she recognizes visually
    AMBIENT = "ambient"  # background speech not directed at her


@dataclass
class Presence:
    """A persistent model of one entity in her external world.

    Fields:
        presence_id: Stable identifier ("user", "voice", "face:alice").
        kind: What sort of presence this is.
        name: Learned display name, or None if she doesn't know it.
        familiarity: How well she knows them [0..1], asymptotic growth.
        bond: Relationship valence [0..1], slow accumulation.
        interactions: Count of addressed exchanges.
        present: Whether they're currently in her world.
        first_seen / last_seen: When she first and most recently
            encountered them (epoch seconds).
        last_addressed: When they last spoke *to* her.
        topics: Recent topics they've engaged about (bounded).
        facts: Things learned about them (bounded).
        sentiment_history: Recent (timestamp, valence) pairs (bounded).
        belief: Her inferred model of this presence's mind —
            responsiveness, topic receptivity, mood, attention, and
            activity rhythm, each with honest uncertainty.
    """

    presence_id: str
    kind: PresenceKind
    name: str | None = None
    familiarity: float = 0.0
    bond: float = 0.0
    interactions: int = 0
    present: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_addressed: float = 0.0
    topics: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    facts: dict[str, str] = field(default_factory=dict)
    sentiment_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=50)
    )
    belief: PresenceBelief = field(default_factory=PresenceBelief)

    @property
    def display_name(self) -> str:
        """What she calls them — their name if known, else their kind."""
        if self.name:
            return self.name
        return {
            PresenceKind.USER: "the user",
            PresenceKind.VOICE: "a voice",
            PresenceKind.FACE: "someone",
            PresenceKind.AMBIENT: "voices nearby",
        }[self.kind]

    @property
    def timeout(self) -> float:
        """Seconds of silence before this presence is considered gone."""
        if self.kind == PresenceKind.USER:
            return USER_PRESENCE_TIMEOUT
        return AMBIENT_PRESENCE_TIMEOUT

    def seen(self, now: float | None = None) -> None:
        """Mark that this presence is active right now."""
        ts = time.time() if now is None else now
        self.last_seen = ts
        self.present = True

    def interacted(
        self,
        sentiment: float = 0.0,
        topics: list[str] | None = None,
        now: float | None = None,
    ) -> None:
        """Record an addressed exchange with this presence.

        Familiarity grows asymptotically; bond drifts with sentiment.
        Called once per addressed turn, not per word.
        """
        ts = time.time() if now is None else now
        self.seen(ts)
        self.last_addressed = ts
        self.interactions += 1
        self.familiarity += (1.0 - self.familiarity) * _FAMILIARITY_RATE
        if sentiment > _BOND_POSITIVE:
            self.bond += (1.0 - self.bond) * _BOND_RATE
        elif sentiment < _BOND_NEGATIVE:
            self.bond = max(0.0, self.bond + self.bond * _BOND_RATE * -1.0)
        self.sentiment_history.append((ts, sentiment))
        for topic in topics or ():
            if topic in self.topics:
                self.topics.remove(topic)
            self.topics.append(topic)

    def set_name(self, name: str) -> None:
        """Learn this presence's name."""
        name = name.strip()
        if name:
            self.name = name

    def learn_fact(self, key: str, value: str) -> None:
        """Record something learned about this presence (bounded)."""
        key, value = key.strip(), value.strip()
        if not key or not value:
            return
        if len(self.facts) >= 50 and key not in self.facts:
            # Evict the oldest-looking entry to stay bounded.
            oldest = next(iter(self.facts))
            del self.facts[oldest]
        self.facts[key] = value

    def recent_sentiment(self, n: int = 10) -> float:
        """Mean sentiment over the last ``n`` interactions, or 0."""
        if not self.sentiment_history:
            return 0.0
        tail = list(self.sentiment_history)[-n:]
        return sum(v for _, v in tail) / len(tail)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "presence_id": self.presence_id,
            "kind": self.kind.value,
            "name": self.name,
            "familiarity": self.familiarity,
            "bond": self.bond,
            "interactions": self.interactions,
            "present": self.present,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "last_addressed": self.last_addressed,
            "topics": list(self.topics),
            "facts": dict(self.facts),
            "sentiment_history": [list(p) for p in self.sentiment_history],
            "belief": self.belief.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Presence | None:
        """Deserialize; returns None for malformed payloads."""
        try:
            presence = cls(
                presence_id=str(data["presence_id"]),
                kind=PresenceKind(data.get("kind", "user")),
                name=data.get("name"),
                familiarity=float(data.get("familiarity", 0.0)),
                bond=float(data.get("bond", 0.0)),
                interactions=int(data.get("interactions", 0)),
                # Presences never restore as present — presence is
                # re-established by fresh activity after boot, not by
                # remembering that someone was around before shutdown.
                present=False,
                first_seen=float(data.get("first_seen", time.time())),
                last_seen=float(data.get("last_seen", time.time())),
                last_addressed=float(data.get("last_addressed", 0.0)),
                topics=deque(data.get("topics", []), maxlen=20),
                facts=dict(data.get("facts", {})),
                sentiment_history=deque(
                    (tuple(p) for p in data.get("sentiment_history", [])),
                    maxlen=50,
                ),
                # A malformed belief downgrades to priors — losing an
                # inference is recoverable, losing the presence isn't.
                belief=PresenceBelief.from_dict(data.get("belief")),
            )
            # Older saves lack the attention anchor — use last_seen so
            # restored attention decays from when they were last active.
            if presence.belief._attention_at <= 0.0 and presence.belief.attention > 0.0:
                presence.belief._attention_at = presence.last_seen
            return presence
        except (KeyError, TypeError, ValueError):
            return None
