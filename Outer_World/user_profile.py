"""User profile — what Genesis knows about Mindy.

This is a dedicated, persistent model of the user, distinct from the
concept network. It tracks identity, preferences, goals, emotional
history, and recent topics so Genesis can respond in a personal,
context-aware way without having to reconstruct everything from
scattered beliefs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

# The bond's setpoint — deep attachment, never fusion. Growth and
# settling both approach this asymptote: a bond above it relaxes
# downward at the same rate turns apply, so attachment regulates
# around a setpoint rather than pinning at total. The gap to 1.0
# is the self she keeps.
_BOND_ASYMPTOTE = 0.9

# Bond warm-start for profiles that predate the field: recorded
# interactions are credited at the same base rate each bonded turn
# applies (prefrontal._BOND_BASE_RATE), capped so the warm start is
# established-but-growing rather than saturated. It may land a
# little above the asymptote — settling handles that.
_BOND_WARM_START_RATE = 0.01
_BOND_WARM_START_MAX_TURNS = 300


@dataclass(slots=True)
class UserProfile:
    """A structured, persistent model of the human user."""

    # The profile models exactly one person — her — so the name is
    # known from the start; ``set_name`` can still change it.
    name: str = "Mindy"
    # Other handles the same person is known by — a Discord display
    # name, a chat nick — so "it's her" resolves identically on every
    # channel, not just the terminal where her presence is implicit.
    aliases: list[str] = field(default_factory=list)
    preferences: dict[str, list[str]] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    recent_topics: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    emotional_history: list[tuple[float, str, float]] = field(default_factory=list)
    interaction_count: int = 0
    # The bond with her person [0, 1] — her accumulated attachment,
    # grown from recognized interaction, not assigned. Readable by
    # her self-report paths, so "how do you feel about me" answers
    # from a real state.
    bond: float = 0.0
    data_path: Path | None = None

    # User-stated data accumulates over years of conversation — bound
    # each structure so the profile file cannot grow without limit.
    # Eviction is oldest-first (dicts preserve insertion order).
    _MAX_GOALS: ClassVar[int] = 100
    _MAX_FACTS: ClassVar[int] = 500
    _MAX_PREF_CATEGORIES: ClassVar[int] = 100
    _MAX_PREF_VALUES: ClassVar[int] = 25
    _MAX_ALIASES: ClassVar[int] = 25
    _MAX_EMOTIONAL_HISTORY: ClassVar[int] = 100

    @staticmethod
    def _norm_handle(handle: str) -> str:
        """Canonical form of a channel handle for identity matching.

        Handles arrive decorated — ``@name``, ``Name#1234``, mixed
        case — while the identity underneath is the same person.
        """
        return handle.strip().lstrip("@").split("#", 1)[0].strip().lower()

    def add_alias(self, alias: str) -> None:
        """Record another handle that means her — e.g. her Discord name."""
        stripped = alias.strip().lstrip("@")
        if not stripped or self._norm_handle(stripped) == self._norm_handle(self.name):
            return
        if all(self._norm_handle(a) != self._norm_handle(stripped) for a in self.aliases):
            if len(self.aliases) >= self._MAX_ALIASES:
                self.aliases.pop(0)
            self.aliases.append(stripped)
            self.save()

    def is_user(self, handle: str) -> bool:
        """Whether *handle* names her person — name or any alias.

        The identity question every channel asks: "is this her?" A
        Discord message from ``MindyRenee`` and a terminal line are
        the same person when the handle matches name or an alias.
        """
        norm = self._norm_handle(handle)
        if not norm:
            return False
        if norm == self._norm_handle(self.name):
            return True
        return any(norm == self._norm_handle(a) for a in self.aliases)

    def __post_init__(self) -> None:
        """Convert a list-based recent_topics field into a bounded deque."""
        if isinstance(self.recent_topics, list):
            q: deque[str] = deque(maxlen=50)
            q.extend(self.recent_topics)
            self.recent_topics = q

    def set_name(self, name: str) -> None:
        """Record the user's name and persist immediately."""
        stripped = name.strip()
        if not stripped:
            return
        self.name = stripped
        self.save()

    def get_name(self) -> str:
        """Return the user's name."""
        return self.name

    def add_preference(self, category: str, value: str) -> None:
        """Record a preference and persist immediately."""
        category = category.lower().strip()
        value = value.strip()
        if not value:
            return
        if category not in self.preferences:
            if len(self.preferences) >= self._MAX_PREF_CATEGORIES:
                self.preferences.pop(next(iter(self.preferences)))
            self.preferences[category] = []
        if value not in self.preferences[category]:
            if len(self.preferences[category]) >= self._MAX_PREF_VALUES:
                self.preferences[category].pop(0)
            self.preferences[category].append(value)
            self.save()

    def get_preference(self, category: str) -> list[str]:
        """Return all recorded values for a preference category."""
        return list(self.preferences.get(category.lower().strip(), []))

    def add_fact(self, key: str, value: str) -> None:
        """Record an arbitrary fact and persist immediately."""
        key = key.lower().strip()
        value = value.strip()
        if key and value and self.facts.get(key) != value:
            if key not in self.facts and len(self.facts) >= self._MAX_FACTS:
                self.facts.pop(next(iter(self.facts)))
            self.facts[key] = value
            self.save()

    def get_fact(self, key: str) -> str | None:
        """Return a fact, if known."""
        return self.facts.get(key.lower().strip())

    def add_goal(self, goal: str) -> None:
        """Record a goal and persist immediately."""
        goal = goal.strip()
        if goal and goal not in self.goals:
            if len(self.goals) >= self._MAX_GOALS:
                self.goals.pop(0)
            self.goals.append(goal)
            self.save()

    def record_topic(self, topic: str) -> None:
        """Track a recently discussed topic."""
        topic = topic.strip().lower()
        if topic:
            if topic in self.recent_topics:
                self.recent_topics.remove(topic)
            self.recent_topics.append(topic)

    def record_emotion(self, label: str, valence: float) -> None:
        """Append a snapshot of the user's emotional state."""
        if not math.isfinite(valence):
            return
        self.emotional_history.append((time.time(), label, valence))
        # Keep only the recent entries — emotional history is a
        # window, not an archive.
        if len(self.emotional_history) > self._MAX_EMOTIONAL_HISTORY:
            self.emotional_history = self.emotional_history[-self._MAX_EMOTIONAL_HISTORY :]

    def increment_interactions(self) -> None:
        """Bump the conversation turn counter."""
        self.interaction_count += 1

    def deepen_bond(self, rate: float) -> float:
        """Move the bond one interaction's worth toward its setpoint.

        Attachment is asymptotic and bounded — each turn closes a
        fraction of the remaining distance to ``_BOND_ASYMPTOTE``,
        so the bond forms quickly at first and matures slowly, the
        way familiarity does, and it can never fuse: a bond above
        the setpoint relaxes back down toward it. `rate` is the
        fraction of the distance closed this turn; warm turns pass
        a larger rate than flat ones.
        """
        if not math.isfinite(rate):
            rate = 0.0
        else:
            rate = max(0.0, min(1.0, rate))
        # A NaN bond must not collapse to fusion through the clamp —
        # a corrupted attachment reads as the stranger, not the pin.
        current = self.bond if math.isfinite(self.bond) else 0.0
        bond = current + (_BOND_ASYMPTOTE - current) * rate
        self.bond = max(0.0, min(1.0, bond))
        return self.bond

    def summarize(self) -> str:
        """Return a short, human-readable summary of what it knows.

        Declarative knowledge only — name, preferences, goals — the
        facts she can relay when asked "what do you know about me".
        The bond is deliberately absent: attachment is a felt state,
        not a fact about the user, and its report belongs to the
        self-inquiry path that reads ``bond`` and lets her own
        language express it — never a quoted number.

        Capped at a few items per category to avoid dumping the entire
        profile as a single unbounded string.
        """
        parts = []
        if self.name:
            parts.append(f"Your name is {self.name}.")
        # Map internal category names to natural verbs.
        verb_map = {
            "likes": "like",
            "dislikes": "dislike",
            "wants": "want",
            "believes": "believe",
            "self_concept": "are",
        }
        for category, values in self.preferences.items():
            if not values:
                continue
            # Cap at 3 values per category.
            capped = values[:3]
            joined = ", ".join(capped)
            if len(values) > 3:
                joined += ", among other things"
            verb = verb_map.get(category, category)
            parts.append(f"You {verb} {joined}.")
        if self.goals:
            # Cap at 3 goals.
            capped_goals = self.goals[:3]
            joined = ", ".join(capped_goals)
            if len(self.goals) > 3:
                joined += ", among other things"
            parts.append(f"Your goals include: {joined}.")
        if not parts:
            return "don't know much about you yet"
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a plain dictionary."""
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "preferences": self.preferences,
            "facts": self.facts,
            "goals": self.goals,
            "recent_topics": list(self.recent_topics),
            "emotional_history": self.emotional_history,
            "interaction_count": self.interaction_count,
            "bond": self.bond,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Restore a profile from a plain dictionary.

        Every field is coerced to its declared type — a hand-edited
        or foreign JSON that stores a list where a dict belongs must
        not load "successfully" and then crash the next add_* call.
        """
        aliases = data.get("aliases")
        preferences = data.get("preferences")
        facts = data.get("facts")
        goals = data.get("goals")
        topics = data.get("recent_topics")
        history = data.get("emotional_history")
        interactions = data.get("interaction_count", 0)
        parsed_history: list[tuple[float, str, float]] = []
        if isinstance(history, list):
            for entry in history:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    try:
                        parsed_history.append((float(entry[0]), str(entry[1]), float(entry[2])))
                    except (TypeError, ValueError):
                        continue
        profile = cls(
            # The profile models her — a missing, null, or wrong-typed
            # name falls back to the default, not to anonymity.
            name=(
                data["name"].strip()
                if isinstance(data.get("name"), str) and data["name"].strip()
                else "Mindy"
            ),
            aliases=(
                [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
                if isinstance(aliases, list)
                else []
            )[: cls._MAX_ALIASES],
            # Keys normalize to the same canonical form add_* writes —
            # a foreign "Likes" must not shadow the real "likes".
            # The declared bounds apply on load too, or a hand-edited
            # file could grow the structure past them unbounded.
            preferences=(
                {
                    k.lower().strip(): [v for v in vals if isinstance(v, str)][
                        : cls._MAX_PREF_VALUES
                    ]
                    for k, vals in list(preferences.items())[: cls._MAX_PREF_CATEGORIES]
                    if isinstance(k, str) and k.strip() and isinstance(vals, list)
                }
                if isinstance(preferences, dict)
                else {}
            ),
            facts=(
                {
                    k.lower().strip(): v
                    for k, v in list(facts.items())[: cls._MAX_FACTS]
                    if isinstance(k, str) and k.strip() and isinstance(v, str)
                }
                if isinstance(facts, dict)
                else {}
            ),
            goals=(
                [g for g in goals if isinstance(g, str)][: cls._MAX_GOALS]
                if isinstance(goals, list)
                else []
            ),
            recent_topics=deque(
                ([t for t in topics if isinstance(t, str)] if isinstance(topics, list) else []),
                maxlen=50,
            ),
            emotional_history=parsed_history[-cls._MAX_EMOTIONAL_HISTORY :],
            interaction_count=(
                int(interactions)
                if isinstance(interactions, (int, float)) and math.isfinite(interactions)
                else 0
            ),
            bond=cls._initial_bond(data),
        )
        return profile

    @staticmethod
    def _initial_bond(data: dict[str, Any]) -> float:
        """The bond's first value.

        A stored value wins outright — including an honest 0.0. When
        the field is absent the profile predates the bond, but it
        still carries the history that would have grown it: each
        recorded interaction is credited once at the same per-turn
        rate ``deepen_bond`` applies, so a longtime person is not met
        as a stranger. Bounded — established familiarity starts
        strong, never saturated, leaving the bond room to keep
        growing from real turns.
        """
        if "bond" in data:
            try:
                bond = float(data["bond"])
            except (TypeError, ValueError):
                return 0.0
            # A corrupt field must not fuse or erase the attachment —
            # non-finite reads as the honest stranger, out-of-range
            # clamps to the declared [0, 1].
            if not math.isfinite(bond):
                return 0.0
            return max(0.0, min(1.0, bond))
        turns = data.get("interaction_count", 0)
        if not isinstance(turns, (int, float)) or not math.isfinite(turns):
            turns = 0
        credited = min(int(turns), _BOND_WARM_START_MAX_TURNS)
        return 1.0 - (1.0 - _BOND_WARM_START_RATE) ** credited

    def save(self) -> bool:
        """Persist to disk atomically, if a data path was set.

        Returns True on success (or when no path is set), False on
        failure so callers can report a failed shutdown save instead
        of silently recording success.
        """
        if self.data_path is None:
            return True
        try:
            self.data_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            data_bytes = json.dumps(self.to_dict(), indent=2, default=list).encode("utf-8")
            tmp_path = self.data_path.with_suffix(self.data_path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                f.write(data_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.data_path)
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.warning(f"Failed to save user profile: {e}")
            return False

    def load(self) -> None:
        """Load from disk, if a data path was set and the file exists."""
        if self.data_path is None or not self.data_path.exists():
            return
        try:
            with open(self.data_path, encoding="utf-8") as f:
                data = json.load(f)
            loaded = UserProfile.from_dict(data)
            self.name = loaded.name
            self.aliases = loaded.aliases
            self.preferences = loaded.preferences
            self.facts = loaded.facts
            self.goals = loaded.goals
            self.recent_topics = loaded.recent_topics
            self.emotional_history = loaded.emotional_history
            self.interaction_count = loaded.interaction_count
            self.bond = loaded.bond
        except (OSError, ValueError, TypeError, AttributeError) as e:
            note_swallowed(
                "user_profile.load",
                e,
            )
