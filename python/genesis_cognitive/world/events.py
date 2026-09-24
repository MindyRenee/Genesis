"""External events — the stream of things happening in Genesis's world.

The external world is modeled as a bounded stream of ``ExternalEvent``s
flowing in both directions:

- **Inbound** — the world acts on her: someone speaks to her, someone
  speaks near her, a percept arrives, a presence enters or leaves.
- **Outbound** — she acts on the world: she says something, she looks,
  she draws, she studies the web.

This is the counterpart to the inner life's thought stream: thoughts
arise inside her; events arrive from outside her. Both feed the same
cognition — the global workspace, memory, and neurochemistry.

Events carry ``topics`` — words in the event that exist as concepts in
her network. Topics ground the event in what she actually knows, so a
presence's interests and the workspace's topic coherence reflect her
real concept graph rather than raw text.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

# Words this short are usually grammatical noise, not topics.
_MIN_TOPIC_WORD_LEN = 4

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")


class EventKind(Enum):
    """The kind of an external world event.

    Inbound kinds describe the world acting on Genesis; outbound kinds
    describe Genesis acting on the world.
    """

    # ── Inbound ──
    USER_SPEECH = "user_speech"          # addressed to her (typed or voiced)
    OVERHEARD_SPEECH = "overheard"       # speech near her, not addressed
    PERCEPTION = "perception"            # a percept arrived (sound, sight)
    ARRIVAL = "arrival"                  # a presence entered her world
    DEPARTURE = "departure"              # a presence left her world
    NOTIFICATION = "notification"        # a system/subcognitive notice

    # ── Outbound ──
    UTTERANCE = "utterance"              # she spoke (reply, question, expression)
    ACTION = "action"                    # she acted (looked, drew, learned)


#: Kinds produced by the world, received by her.
INBOUND_KINDS = frozenset({
    EventKind.USER_SPEECH,
    EventKind.OVERHEARD_SPEECH,
    EventKind.PERCEPTION,
    EventKind.ARRIVAL,
    EventKind.DEPARTURE,
    EventKind.NOTIFICATION,
})

#: Kinds produced by her, received by the world.
OUTBOUND_KINDS = frozenset({
    EventKind.UTTERANCE,
    EventKind.ACTION,
})


@dataclass(slots=True)
class ExternalEvent:
    """One thing that happened in Genesis's external world.

    Fields:
        kind: What sort of event this is.
        source: Who produced it — a presence id for inbound events,
            ``"self"`` for her own outbound acts.
        content: Semantic description of what happened. For speech this
            is the utterance text; for actions a short semantic summary.
        salience: How much this event matters [0..1]. Salient events
            are broadcast to the workspace and stored as memories.
        addressed: Whether the event was directed at her specifically.
        topics: Concept names from her network that appear in the
            event — grounds the event in her knowledge.
        timestamp: When the event happened (seconds since epoch).
        metadata: Kind-specific extras (e.g. the presence's name, her
            full report for an action).
    """

    kind: EventKind
    source: str
    content: str
    salience: float = 0.4
    addressed: bool = False
    topics: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def inbound(self) -> bool:
        """Whether this event flows from the world toward her."""
        return self.kind in INBOUND_KINDS

    def describe(self) -> str:
        """A short semantic description, used for memory and broadcast.

        This is internal representation, not her voice — it is what the
        event *is*, not what she says about it.
        """
        if self.kind == EventKind.USER_SPEECH:
            return f"{self.source} said: {self.content}"
        if self.kind == EventKind.OVERHEARD_SPEECH:
            return f"overheard: {self.content}"
        if self.kind == EventKind.UTTERANCE:
            return f"she said: {self.content}"
        if self.kind == EventKind.ACTION:
            return f"she acted: {self.content}"
        if self.kind == EventKind.PERCEPTION:
            return f"she perceived: {self.content}"
        return self.content

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "kind": self.kind.value,
            "source": self.source,
            "content": self.content,
            "salience": self.salience,
            "addressed": self.addressed,
            "topics": list(self.topics),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExternalEvent | None:
        """Deserialize; returns None for malformed or unknown kinds."""
        try:
            return cls(
                kind=EventKind(data["kind"]),
                source=str(data["source"]),
                content=str(data["content"]),
                salience=float(data.get("salience", 0.4)),
                addressed=bool(data.get("addressed", False)),
                topics=list(data.get("topics", [])),
                timestamp=float(data.get("timestamp", time.time())),
            )
        except (KeyError, TypeError, ValueError):
            return None


def ground_topics(
    text: str,
    is_known: Callable[[str], bool],
    *,
    max_topics: int = 5,
) -> list[str]:
    """Extract the words in ``text`` that exist in her concept network.

    Grounding ties the event to what she actually knows: a topic is a
    word she has a concept for, not just any token. The longest matches
    are preferred so "memory consolidation" outranks "memory".

    Args:
        text: The event text to scan.
        is_known: Predicate — True if a lowercased word is a concept
            she has (e.g. ``network.get_concept`` is not None).
        max_topics: Maximum topics to return.
    """
    seen: set[str] = set()
    topics: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) < _MIN_TOPIC_WORD_LEN or token in seen:
            continue
        seen.add(token)
        try:
            if is_known(token):
                topics.append(token)
                if len(topics) >= max_topics:
                    break
        except Exception:  # noqa: BLE001 — a lookup must never break events
            continue
    return topics


class EventStream:
    """A bounded, thread-safe stream of external events.

    The stream is the world's working memory: recent events stay
    available for attention, status, and persistence; older events fall
    off the end (their consolidation into episodic memory is the memory
    engine's job, not the stream's).
    """

    def __init__(self, max_size: int = 200) -> None:
        """Create a bounded stream holding at most ``max_size`` events."""
        self._events: deque[ExternalEvent] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, event: ExternalEvent) -> None:
        """Record an event (thread-safe)."""
        with self._lock:
            self._events.append(event)

    def recent(self, n: int = 20) -> list[ExternalEvent]:
        """Return the ``n`` most recent events, oldest first."""
        with self._lock:
            events = list(self._events)
        return events[-n:] if len(events) > n else events

    def __len__(self) -> int:
        """Number of events currently in the stream."""
        with self._lock:
            return len(self._events)
