#!/usr/bin/env python3
"""Unit tests for subcognitive→cognitive notification queue.

Tests that the notification queue correctly detects:
- Phase changes (e.g., alert→sleeping, sleeping→awake)
- New dream insights (episodes with source_module=9)

Run with: python3 tests/test_notifications.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from genesis_client.types import NeuroSummary, RecentEpisode
from genesis_cognitive.notifications import Notification, NotificationQueue


def _make_summary(phase: int = 0) -> NeuroSummary:
    """Create a NeuroSummary with the given phase."""
    return NeuroSummary(
        arousal=0.5, valence=0.5, global_tone=0.5,
        plasticity_gate=0.5, encoding_weight=0.5,
        consolidation_weight=0.5, retrieval_weight=0.5,
        phase=phase,
    )


def _make_dream_episode(eid: int, text: str = "A novel connection") -> RecentEpisode:
    """Create a RecentEpisode from the dream module (source_module=9)."""
    return RecentEpisode(
        episode_id=eid, timestamp=1000, salience=0.7,
        event_type=2, source_module=9, text=text,
    )


def _make_queue(
    summaries=None,
    episodes=None,
) -> NotificationQueue:
    """Create a NotificationQueue with mock callbacks."""
    summaries = summaries or [_make_summary(phase=0)]
    episodes = episodes or []
    summary_iter = iter(summaries)
    episode_iter = iter(episodes)

    def get_summary() -> NeuroSummary | None:
        """Return the next queued NeuroSummary (or the last one if exhausted)."""
        try:
            return next(summary_iter)
        except StopIteration:
            return summaries[-1] if summaries else None

    def get_episodes(limit=20, source_module=None) -> list[RecentEpisode]:
        """Return the next queued episode batch (or an empty list if exhausted)."""
        try:
            return next(episode_iter)
        except StopIteration:
            return []

    return NotificationQueue(
        get_neuro_summary=get_summary,
        get_recent_episodes=get_episodes,
    )


def test_first_poll_does_not_notify():
    """The first poll should NOT generate a phase change notification."""
    queue = _make_queue(summaries=[_make_summary(phase=0)])

    enqueued = queue.poll()

    assert enqueued == 0, f"First poll should not notify, got {enqueued}"
    assert queue.peek() == 0, "Queue should be empty after first poll"
    print("  PASS  first poll does not notify")


def test_phase_change_detected():
    """A phase change between polls should generate a notification."""
    queue = _make_queue(summaries=[
        _make_summary(phase=0),  # first poll (active)
        _make_summary(phase=4),  # second poll (drowsy)
    ])

    queue.poll()  # first poll — no notification
    enqueued = queue.poll()  # second poll — phase changed

    assert enqueued >= 1, f"Should detect phase change, got {enqueued}"
    notifications = queue.drain()
    assert len(notifications) == 1
    assert notifications[0].kind == "phase_changed"
    assert notifications[0].data["old_phase"] == "active"
    assert notifications[0].data["new_phase"] == "drowsy"
    print("  PASS  phase change detected")


def test_no_phase_change_no_notification():
    """No phase change should produce no notification."""
    queue = _make_queue(summaries=[
        _make_summary(phase=0),
        _make_summary(phase=0),  # same phase
    ])

    queue.poll()
    enqueued = queue.poll()

    assert enqueued == 0, f"Should not notify for same phase, got {enqueued}"
    print("  PASS  no phase change → no notification")


def test_dream_insight_detected():
    """New dream episodes should generate notifications."""
    queue = _make_queue(
        summaries=[_make_summary(phase=0), _make_summary(phase=0)],
        episodes=[
            [],  # first poll — no dreams
            [_make_dream_episode(eid=5, text="Insight about memory")],  # second poll
        ],
    )

    queue.poll()  # first poll — no dreams
    enqueued = queue.poll()  # second poll — new dream

    assert enqueued >= 1, f"Should detect dream insight, got {enqueued}"
    notifications = queue.drain()
    assert len(notifications) == 1
    assert notifications[0].kind == "dream_insight"
    assert notifications[0].data["episode_id"] == 5
    print("  PASS  dream insight detected")


def test_dream_insight_high_water_mark():
    """Already-seen dream episodes should not generate new notifications."""
    queue = _make_queue(
        summaries=[_make_summary(phase=0)] * 3,
        episodes=[
            [],
            [_make_dream_episode(eid=5)],
            [_make_dream_episode(eid=5)],  # same episode — should not notify
        ],
    )

    queue.poll()  # no dreams
    queue.poll()  # dream eid=5 → notification
    enqueued = queue.poll()  # same dream → no notification

    assert enqueued == 0, f"Should not re-notify for same dream, got {enqueued}"
    print("  PASS  dream insight high-water mark works")


def test_multiple_new_dreams():
    """Multiple new dream episodes should each generate a notification."""
    queue = _make_queue(
        summaries=[_make_summary(phase=0)] * 2,
        episodes=[
            [],
            [_make_dream_episode(eid=5), _make_dream_episode(eid=6), _make_dream_episode(eid=7)],
        ],
    )

    queue.poll()
    enqueued = queue.poll()

    assert enqueued == 3, f"Should notify for 3 new dreams, got {enqueued}"
    notifications = queue.drain()
    ids = [n.data["episode_id"] for n in notifications]
    assert ids == [5, 6, 7], f"Expected ids [5,6,7], got {ids}"
    print("  PASS  multiple new dreams each notify")


def test_drain_clears_queue():
    """drain() should empty the queue."""
    queue = _make_queue(summaries=[
        _make_summary(phase=0),
        _make_summary(phase=4),
    ])

    queue.poll()
    queue.poll()
    assert queue.peek() > 0, "Queue should have notifications"

    notifications = queue.drain()
    assert len(notifications) > 0
    assert queue.peek() == 0, "Queue should be empty after drain"
    print("  PASS  drain clears queue")


def test_reset_clears_state():
    """reset() should clear the queue and reset state tracking."""
    queue = _make_queue(summaries=[
        _make_summary(phase=0),
        _make_summary(phase=4),
    ])

    queue.poll()
    queue.poll()
    assert queue.peek() > 0

    queue.reset()

    assert queue.peek() == 0, "Queue should be empty after reset"
    # After reset, the next poll should NOT notify (it re-initializes)
    enqueued = queue.poll()
    assert enqueued == 0, f"First poll after reset should not notify, got {enqueued}"
    print("  PASS  reset clears state")


def test_connection_error_handled_gracefully():
    """Connection errors during poll should not crash."""
    def failing_summary() -> None:
        """Raise a ConnectionError to simulate a daemon outage."""
        raise ConnectionError("daemon unreachable")

    queue = NotificationQueue(
        get_neuro_summary=failing_summary,
        get_recent_episodes=lambda **kw: [],
    )

    enqueued = queue.poll()

    assert enqueued == 0, f"Should return 0 on connection error, got {enqueued}"
    print("  PASS  connection error handled gracefully")


def test_notification_data_structure():
    """Notification should have the correct data structure."""
    notif = Notification(
        kind="phase_changed",
        message="Phase shifted",
        timestamp=12345,
        data={"old_phase": "alert", "new_phase": "sleeping"},
    )

    assert notif.kind == "phase_changed"
    assert notif.message == "Phase shifted"
    assert notif.timestamp == 12345
    assert notif.data["old_phase"] == "alert"
    assert notif.data["new_phase"] == "sleeping"
    print("  PASS  notification data structure correct")


def test_queue_bounded():
    """Queue should be bounded — old notifications dropped when full."""
    queue = NotificationQueue(
        get_neuro_summary=lambda: _make_summary(phase=0),
        get_recent_episodes=lambda **kw: [],
        max_size=3,
    )
    # Manually enqueue 5 notifications
    for i in range(5):
        queue._enqueue(Notification(
            kind="test", message=f"msg {i}", timestamp=i,
        ))

    notifications = queue.drain()
    # Only the last 3 should remain (deque maxlen drops oldest)
    assert len(notifications) == 3, f"Expected 3 (bounded), got {len(notifications)}"
    assert notifications[0].message == "msg 2"  # oldest surviving
    print("  PASS  queue is bounded")


def main() -> None:
    """Main."""
    tests = [
        test_first_poll_does_not_notify,
        test_phase_change_detected,
        test_no_phase_change_no_notification,
        test_dream_insight_detected,
        test_dream_insight_high_water_mark,
        test_multiple_new_dreams,
        test_drain_clears_queue,
        test_reset_clears_state,
        test_connection_error_handled_gracefully,
        test_notification_data_structure,
        test_queue_bounded,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Notification queue tests: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
