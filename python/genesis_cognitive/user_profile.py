"""User profile — what Genesis knows about the human.

This is a dedicated, persistent model of the user, distinct from the
concept network. It tracks identity, preferences, goals, emotional
history, and recent topics so Genesis can respond in a personal,
context-aware way without having to reconstruct everything from
scattered beliefs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class UserProfile:
    """A structured, persistent model of the human user."""

    name: str | None = None
    preferences: dict[str, list[str]] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    recent_topics: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    emotional_history: list[tuple[float, str, float]] = field(default_factory=list)
    interaction_count: int = 0
    data_path: Path | None = None

    # User-stated data accumulates over years of conversation — bound
    # each structure so the profile file cannot grow without limit.
    # Eviction is oldest-first (dicts preserve insertion order).
    _MAX_GOALS: ClassVar[int] = 100
    _MAX_FACTS: ClassVar[int] = 500
    _MAX_PREF_CATEGORIES: ClassVar[int] = 100
    _MAX_PREF_VALUES: ClassVar[int] = 25

    def __post_init__(self) -> None:
        """Convert a list-based recent_topics field into a bounded deque."""
        if isinstance(self.recent_topics, list):
            q: deque[str] = deque(maxlen=50)
            q.extend(self.recent_topics)
            self.recent_topics = q

    def set_name(self, name: str) -> None:
        """Record the user's name and persist immediately."""
        self.name = name.strip()
        self.save()

    def get_name(self) -> str | None:
        """Return the user's name, if known."""
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
        self.emotional_history.append((time.time(), label, valence))
        # Keep only the last 100 entries to avoid unbounded growth.
        if len(self.emotional_history) > 100:
            self.emotional_history = self.emotional_history[-100:]

    def increment_interactions(self) -> None:
        """Bump the conversation turn counter."""
        self.interaction_count += 1

    def summarize(self) -> str:
        """Return a short, human-readable summary of what she knows.

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
            "preferences": self.preferences,
            "facts": self.facts,
            "goals": self.goals,
            "recent_topics": list(self.recent_topics),
            "emotional_history": self.emotional_history,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Restore a profile from a plain dictionary."""
        profile = cls(
            name=data.get("name"),
            preferences=data.get("preferences", {}),
            facts=data.get("facts", {}),
            goals=data.get("goals", []),
            recent_topics=deque(data.get("recent_topics", []), maxlen=50),
            emotional_history=data.get("emotional_history", []),
            interaction_count=data.get("interaction_count", 0),
        )
        return profile

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
            self.preferences = loaded.preferences
            self.facts = loaded.facts
            self.goals = loaded.goals
            self.recent_topics = loaded.recent_topics
            self.emotional_history = loaded.emotional_history
            self.interaction_count = loaded.interaction_count
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to load user profile: {e}")
