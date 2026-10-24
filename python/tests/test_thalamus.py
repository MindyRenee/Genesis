"""Thalamus — region-to-region messages on the module manifest.

The channel is how regions talk: a post lands on the source module's
``task_id`` field (what the heartbeat writes and telemetry returns),
fires local subscribers synchronously, and is delivered back to
subscribers when the same task_id arrives over the wire.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from thalamus.relay import (
    KIND_FELT,
    KIND_HEARD,
    KIND_SPOKE,
    Thalamus,
    pack,
    unpack,
)

MOD_A = 6  # sensory
MOD_B = 4  # language


def _mod(module_id: int, task_id: int) -> SimpleNamespace:
    """A telemetry module record as GET_SUBSYSTEM_TELEMETRY yields it."""
    return SimpleNamespace(module_id=module_id, task_id=task_id)


# ─── pack / unpack ────────────────────────────────────────────────


def test_pack_unpack_round_trip() -> None:
    for kind in (KIND_HEARD, KIND_SPOKE, KIND_FELT):
        assert unpack(pack(kind, 7)) == kind


def test_unpack_zero_is_no_message() -> None:
    assert unpack(0) == 0


# ─── synchronous local dispatch ───────────────────────────────────


def test_post_fires_subscribers_synchronously() -> None:
    ch = Thalamus()
    heard: list[tuple[int, object]] = []
    ch.register(MOD_A, lambda kind, payload: heard.append((kind, payload)))
    task_id = ch.post(MOD_A, KIND_HEARD, ("turn-1", "hello"))
    assert heard == [(KIND_HEARD, ("turn-1", "hello"))]
    assert unpack(task_id) == KIND_HEARD


def test_post_only_reaches_projected_module() -> None:
    """A post on MOD_A does not fire MOD_B's subscribers."""
    ch = Thalamus()
    heard_b: list[object] = []
    ch.register(MOD_B, lambda kind, payload: heard_b.append(payload))
    ch.post(MOD_A, KIND_HEARD, "x")
    assert heard_b == []


def test_multiple_subscribers_all_hear() -> None:
    ch = Thalamus()
    a, b = [], []
    ch.register(MOD_A, lambda k, p: a.append(p))
    ch.register(MOD_A, lambda k, p: b.append(p))
    ch.post(MOD_A, KIND_HEARD, "x")
    assert a == b == ["x"]


def test_unregister_removes_projection() -> None:
    ch = Thalamus()
    heard: list[object] = []
    h = ch.register(MOD_A, lambda k, p: heard.append(p))
    ch.unregister(MOD_A, h)
    ch.post(MOD_A, KIND_HEARD, "x")
    assert heard == []


# ─── once ─────────────────────────────────────────────────────────


def test_once_drops_after_match() -> None:
    ch = Thalamus()
    heard: list[object] = []

    def _take(_kind: int, payload: object) -> bool:
        heard.append(payload)
        return True

    ch.once(MOD_A, _take)
    ch.post(MOD_A, KIND_HEARD, "first")
    ch.post(MOD_A, KIND_HEARD, "second")
    assert heard == ["first"]


def test_once_survives_non_matching_messages() -> None:
    """A once-handler that returns False stays projected."""
    ch = Thalamus()
    heard: list[object] = []

    def _take(_kind: int, payload: object) -> bool:
        heard.append(payload)
        return payload == "wanted"

    ch.once(MOD_A, _take)
    ch.post(MOD_A, KIND_HEARD, "other")
    ch.post(MOD_A, KIND_HEARD, "wanted")
    ch.post(MOD_A, KIND_HEARD, "after")
    assert heard == ["other", "wanted"]


# ─── telemetry publication ────────────────────────────────────────


def test_take_returns_pending_task_id() -> None:
    ch = Thalamus()
    task_id = ch.post(MOD_A, KIND_SPOKE, "payload")
    assert ch.take(MOD_A) == task_id


def test_take_none_when_silent() -> None:
    ch = Thalamus()
    assert ch.take(MOD_A) is None


def test_take_stays_until_replaced() -> None:
    """A reader sampling the manifest between heartbeats still sees it."""
    ch = Thalamus()
    first = ch.post(MOD_A, KIND_SPOKE, "one")
    assert ch.take(MOD_A) == first
    assert ch.take(MOD_A) == first  # still there
    second = ch.post(MOD_A, KIND_SPOKE, "two")
    assert ch.take(MOD_A) == second
    assert second != first


def test_same_kind_posts_have_distinct_task_ids() -> None:
    """Two identical messages register as a change on the wire."""
    ch = Thalamus()
    ids = {ch.post(MOD_A, KIND_HEARD, "same") for _ in range(4)}
    assert len(ids) == 4


# ─── deliver: wire-side messages ──────────────────────────────────


def test_deliver_ignores_own_echo() -> None:
    """A locally posted message must not replay through deliver."""
    ch = Thalamus()
    heard: list[object] = []
    ch.register(MOD_A, lambda k, p: heard.append(p))
    task_id = ch.post(MOD_A, KIND_HEARD, "local")
    assert len(heard) == 1
    ch.deliver([_mod(MOD_A, task_id)])
    assert len(heard) == 1


def test_deliver_fires_for_external_task_ids() -> None:
    """A task_id this process did not post is delivered once."""
    ch = Thalamus()
    heard: list[tuple[int, object]] = []
    ch.register(MOD_A, lambda k, p: heard.append((k, p)))
    foreign = pack(KIND_FELT, 4242)
    ch.deliver([_mod(MOD_A, foreign)])
    assert heard == [(KIND_FELT, None)]  # wire carries no payload
    ch.deliver([_mod(MOD_A, foreign)])  # same task_id again
    assert len(heard) == 1


def test_deliver_skips_zero_task_id() -> None:
    ch = Thalamus()
    heard: list[object] = []
    ch.register(MOD_A, lambda k, p: heard.append(p))
    ch.deliver([_mod(MOD_A, 0)])
    assert heard == []


def test_deliver_recovers_after_new_post() -> None:
    """A newer foreign task_id on the same module is delivered."""
    ch = Thalamus()
    heard: list[object] = []
    ch.register(MOD_A, lambda k, p: heard.append(p))
    ch.deliver([_mod(MOD_A, pack(KIND_FELT, 1))])
    ch.deliver([_mod(MOD_A, pack(KIND_HEARD, 2))])
    assert heard == [None, None]


# ─── thread safety ────────────────────────────────────────────────


def test_concurrent_posts_are_safe() -> None:
    ch = Thalamus()
    heard: list[object] = []
    ch.register(MOD_A, lambda k, p: heard.append(p))

    def poster(i: int) -> None:
        for j in range(50):
            ch.post(MOD_A, KIND_HEARD, (i, j))

    threads = [threading.Thread(target=poster, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(heard) == 200


def test_once_under_contention_fires_once() -> None:
    ch = Thalamus()
    fired: list[object] = []

    def _take(_kind: int, payload: object) -> bool:
        fired.append(payload)
        return True

    ch.once(MOD_A, _take)
    threads = [threading.Thread(target=ch.post, args=(MOD_A, KIND_SPOKE, i)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Exactly one post won the race to consume the one-shot listener;
    # whichever it was, the listener fired for exactly one message.
    assert len(fired) == 1
