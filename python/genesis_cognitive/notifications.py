"""Subcognitive→cognitive notification queue.

The Rust subcognitive daemon and the Python cognitive layer
communicate via request-response IPC — there is no push mechanism.
This module implements a *pull-based notification queue* that
detects significant events from the subcognitive (phase changes,
dream insights) during the existing polling cycle and surfaces
them to the cognitive layer.

The notification IDs match the protocol constants defined in
``genesis_client.protocol`` (NOTIFY_PHASE_CHANGED=100,
NOTIFY_DREAM_INSIGHT=102, etc.) — these are placeholders for a
future push mechanism. This queue implements the Python-side
detection logic that would eventually be replaced by real push
notifications from the Rust daemon.

# Why this matters

Without notifications, the cognitive layer can only discover
subcognitive events by accident — e.g., stumbling on a dream
insight when retrieving memories. With this queue, phase changes
(waking from sleep, entering flow) and dream insights (novel
connections discovered during sleep) are proactively surfaced,
strengthening the bridge between the two layers.

References:
    - The audit identified that the Rust subcognitive and Python
      cognitive layers are "more parallel than integrated, with a
      thin bridge between them." This module thickens that bridge.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Source module IDs (matching the Rust daemon's manifest).
_DREAM_MODULE = 9


@dataclass(slots=True)
class Notification:
    """A notification from the subcognitive to the cognitive layer.

    Attributes:
        kind: The notification type — "phase_changed",
            "dream_insight", or "episode_consolidated".
        message: Human-readable description of the event.
        timestamp: When the notification was detected (ms).
        data: Optional payload (e.g., the new phase name, or the
            dream insight episode ID).
    """

    kind: str
    message: str
    timestamp: int
    data: dict[str, object] = field(default_factory=dict)


class NotificationQueue:
    """A pull-based notification queue for subcognitive→cognitive events.

    The cognitive layer polls the subcognitive via
    ``get_neuro_summary()`` and ``get_recent_episodes()``. This queue
    wraps those polls to detect *changes* and *new events*, enqueuing
    notifications that the cognitive layer can drain at its leisure.

    The queue is bounded — old notifications are dropped if the
    cognitive layer doesn't drain them fast enough (the subcognitive
    doesn't wait).

    Thread-safe. The polling thread enqueues; the cognitive thread
    drains.
    """

    def __init__(
        self,
        get_neuro_summary: Callable,
        get_recent_episodes: Callable,
        *,
        max_size: int = 64,
    ) -> None:
        """Initialize the notification queue.

        Args:
            get_neuro_summary: Callback to poll neurochemical summary
                (typically ``client.get_neuro_summary``).
            get_recent_episodes: Callback to poll recent LTM episodes
                (typically ``client.get_recent_episodes``).
            max_size: Maximum number of notifications to buffer.
        """
        self._get_neuro_summary = get_neuro_summary
        self._get_recent_episodes = get_recent_episodes
        self._queue: deque[Notification] = deque(maxlen=max_size)
        self._lock = threading.Lock()

        # State tracking for change detection.
        self._last_phase: str = ""
        self._last_dream_episode_id: int = 0
        self._dream_initialized = False
        self._initialized = False

    def poll(self) -> int:
        """Poll the subcognitive for new events and enqueue notifications.

        This should be called periodically (e.g., every 5 seconds from
        the InnerLife polling loop). It detects:
        - Phase changes (e.g., alert→drowsy, sleeping→awake)
        - New dream insights (episodes with source_module=9)

        Returns the number of new notifications enqueued.
        """
        enqueued = 0
        now_ms = int(time.time() * 1000)

        # ── Phase change detection ──
        try:
            summary = self._get_neuro_summary()
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"notification poll failed: {e}")
            return 0

        if summary is not None:
            current_phase = getattr(summary, "phase_name", "") or ""
            if not self._initialized:
                # First poll — just record the state, don't notify.
                self._last_phase = current_phase
                self._initialized = True
            elif current_phase != self._last_phase:
                # Phase changed — enqueue notification.
                self._enqueue(Notification(
                    kind="phase_changed",
                    message=f"Phase shifted from '{self._last_phase}' to '{current_phase}'",
                    timestamp=now_ms,
                    data={
                        "old_phase": self._last_phase,
                        "new_phase": current_phase,
                    },
                ))
                enqueued += 1
                self._last_phase = current_phase

        # ── Dream insight detection ──
        try:
            recent = self._get_recent_episodes(limit=10, source_module=_DREAM_MODULE)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"dream insight poll failed: {e}")
            return enqueued

        # On the first successful poll, set the high-water mark to the
        # most recent dream episode (if any) without flooding the queue
        # with old insights.
        if not self._dream_initialized:
            if recent:
                self._last_dream_episode_id = max(
                    ep.episode_id for ep in recent
                )
            self._dream_initialized = True
            return enqueued

        if recent:
            # Find dream insights newer than the last seen one.
            new_insights = [
                ep for ep in recent
                if ep.episode_id > self._last_dream_episode_id
            ]
            if new_insights:
                # Sort by episode_id to update the high-water mark correctly.
                new_insights.sort(key=lambda e: e.episode_id)
                for ep in new_insights:
                    self._enqueue(Notification(
                        kind="dream_insight",
                        message=f"dream insight: {ep.text[:120]}",
                        timestamp=now_ms,
                        data={
                            "episode_id": ep.episode_id,
                            "text": ep.text,
                        },
                    ))
                    enqueued += 1
                self._last_dream_episode_id = new_insights[-1].episode_id

        return enqueued

    def drain(self) -> list[Notification]:
        """Drain and return all pending notifications.

        The cognitive layer calls this to process accumulated
        notifications (e.g., surfacing them in conversation).

        Returns:
            A list of notifications, oldest first. May be empty.
        """
        with self._lock:
            notifications = list(self._queue)
            self._queue.clear()
        return notifications

    def peek(self) -> int:
        """Return the number of pending notifications without draining."""
        with self._lock:
            return len(self._queue)

    def _enqueue(self, notification: Notification) -> None:
        """Add a notification to the queue (thread-safe)."""
        with self._lock:
            self._queue.append(notification)

    def reset(self) -> None:
        """Reset state tracking (e.g., after a daemon restart).

        Clears the queue and resets the phase/episode high-water marks
        so the next poll doesn't generate a flood of notifications.
        """
        with self._lock:
            self._queue.clear()
            self._last_phase = ""
            self._last_dream_episode_id = 0
            self._dream_initialized = False
            self._initialized = False
