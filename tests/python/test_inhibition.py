"""Claims about the inhibitory substrate.

``Self_World/inhibition.py`` provides the organism's coordination
primitives — the mechanisms an OS would call locks, named after the
inhibitory circuits that perform the same job in neural tissue. Each
test below pins one claim: exclusion, re-entry, bounded channel
capacity, broadcast gating, relay release, refractory deduplication,
and the contention telemetry that makes inhibition measurable.
"""

from __future__ import annotations

import threading
import time

from inhibition import (
    CompetitiveSelection,
    InhibitoryGate,
    NeuromodulatoryGate,
    RecurrentInhibition,
    RefractorySet,
    ThalamicRelay,
    gate_telemetry,
    inhibitory_tone,
)


def test_gate_excludes_concurrent_access() -> None:
    gate = InhibitoryGate()
    gate.acquire()
    done = []
    t = threading.Thread(target=lambda: done.append(gate.acquire(blocking=False)))
    t.start()
    t.join()
    assert done == [False]
    gate.release()


def test_gate_serializes_a_critical_section() -> None:
    gate = InhibitoryGate()
    shared = []

    def append(i):
        with gate:
            shared.append(i)

    threads = [threading.Thread(target=append, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(shared) == list(range(8))


def test_recurrent_inhibition_reenters_for_owner_only() -> None:
    gate = RecurrentInhibition()
    with gate:
        with gate:
            pass
    entered = []
    gate.acquire()

    def foreign():
        entered.append(gate.acquire(blocking=False))

    t = threading.Thread(target=foreign)
    t.start()
    t.join()
    assert entered == [False]
    gate.release()


def test_recurrent_gate_backs_a_condition() -> None:
    gate = RecurrentInhibition()
    cond = threading.Condition(gate)
    woke = []

    def waiter():
        with cond:
            cond.wait(2.0)
            woke.append(True)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    with cond:
        cond.notify()
    t.join()
    assert woke == [True]


def test_competitive_selection_bounds_concurrent_channels() -> None:
    sel = CompetitiveSelection(2)
    assert sel.acquire(blocking=False)
    assert sel.acquire(blocking=False)
    assert not sel.acquire(blocking=False)
    assert sel.occupancy() == 1.0
    sel.release()
    assert sel.occupancy() == 0.5
    assert sel.acquire(blocking=False)


def test_neuromodulatory_gate_broadcasts() -> None:
    gate = NeuromodulatoryGate()
    heard = []

    def listener():
        heard.append(gate.wait(2.0))

    t = threading.Thread(target=listener)
    t.start()
    time.sleep(0.05)
    gate.set()
    t.join()
    assert heard == [True]
    gate.clear()
    assert not gate.is_set()


def test_neuromodulatory_gate_pulses_a_wired_sink() -> None:
    import inhibition

    pulses = []
    inhibition.set_pulse_sink(pulses.append)
    try:
        NeuromodulatoryGate().set()
    finally:
        inhibition.set_pulse_sink(None)
    assert len(pulses) == 1


def test_thalamic_relay_releases_waiters() -> None:
    relay = ThalamicRelay()
    released = []

    def relay_cell():
        with relay:
            relay.wait(2.0)
            released.append(True)

    t = threading.Thread(target=relay_cell)
    t.start()
    time.sleep(0.05)
    assert relay.waiting() == 1
    with relay:
        relay.notify()
    t.join()
    assert released == [True]
    assert relay.waiting() == 0


def test_refractory_set_fires_each_name_once() -> None:
    rs = RefractorySet()
    assert rs.try_fire("improve")
    assert not rs.try_fire("improve")
    assert rs.try_fire("speech")
    rs.release("improve")
    assert rs.try_fire("improve")


def test_refractory_set_dedup_is_atomic_under_race() -> None:
    rs = RefractorySet()
    fired = []

    def try_fire():
        if rs.try_fire("probe"):
            fired.append(True)

    threads = [threading.Thread(target=try_fire) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fired == [True]


def test_contention_is_measured() -> None:
    gate = InhibitoryGate()

    def hold():
        with gate:
            time.sleep(0.2)

    t = threading.Thread(target=hold)
    t.start()
    time.sleep(0.02)
    with gate:
        pass
    t.join()
    stats = gate.stats()
    assert stats["acquisitions"] == 2.0
    assert stats["contended"] >= 1.0
    assert stats["wait_seconds"] > 0.0


def test_inhibitory_tone_is_bounded() -> None:
    assert 0.0 <= inhibitory_tone() < 1.0
    assert isinstance(gate_telemetry(), list)
