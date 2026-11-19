"""Pulse — a charge flag. Signals without gates.

A neuron does not lock to hear its input; it integrates charge and
fires. This is the smallest signaling primitive that still lets a
sleeping thread wake early: a mutable flag plus a short-sleep wait.
No mutex, no condition variable — the GIL makes the flag read atomic;
a set lands within ~50 ms of the sleeper noticing.

Same surface as Event/Condition minus the gate:

- set/clear/is_set: latching flag semantics (Event).
- wait: returns when the flag is set, or when a notify edge lands.
- notify/notify_all: a generation bump, not a latch — wakes only
  threads already asleep; it does not leak into the next wait.
- wait_for: polls a predicate, waking early on notify edges.
"""

from __future__ import annotations

import time
from collections.abc import Callable

_POLL_S = 0.05


class Pulse:
    __slots__ = ("seq", "value")

    def __init__(self) -> None:
        self.value = False
        self.seq = 0

    def set(self) -> None:
        self.value = True

    def clear(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value

    def notify(self, n: int = 1) -> None:
        self.seq += 1

    def notify_all(self) -> None:
        self.seq += 1

    def wait(self, timeout: float | None = None) -> bool:
        """Wait.

        Args:
            timeout: The timeout in seconds.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        seq = self.seq
        while not self.value:
            if self.seq != seq:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_S)
        return True

    def wait_for(self, predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        """Wait for.

        Args:
            predicate: The predicate.
            timeout: The timeout in seconds.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        seq = self.seq
        while not predicate():
            if self.seq != seq:
                seq = self.seq
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return predicate()
            time.sleep(_POLL_S)
        return True
