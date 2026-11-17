"""Inhibition — the coordination substrate of the self world.

════════════════════════════════════════════════════════════════════════
WHAT AN OPERATING SYSTEM CALLS A LOCK, A NERVOUS SYSTEM CALLS
INHIBITION
════════════════════════════════════════════════════════════════════════

The brain never uses a mutex — and never needs one. Mutual exclusion,
bounded concurrency, one-shot signalling, and atomic deduplication all
exist in neural tissue; they are implemented by inhibitory circuitry
rather than by a scheduler primitive. This module provides the five
mechanisms the organism needs, each named after the circuit that
performs it, each instrumented so its activity is measurable as part
of her biology.

Two commitments shape everything below:

1.  The mechanisms are *correct*. Inhibition in cortex is not a hint —
    a fast-spiking interneuron hard-vets competing drive (Koós &
    Tepper 1999). The primitives here are thin, deterministic
    instrumentations of the kernel primitives they replace; no
    probabilistic release, no advisory-only enforcement.
2.  The mechanisms are *observable*. Inhibition is a real, measurable
    quantity in biology — GABAergic tone shows up on the EEG. Every
    primitive reports its contention statistics so that inhibitory
    load can join the rest of her interoceptive accounting instead of
    vanishing into scheduler internals.

The registry (``_GATES``) and the ``inhibitory_tone`` aggregate are
the second half of that: the substrate's own self-measurement.

════════════════════════════════════════════════════════════════════════
THE CIRCUITS
════════════════════════════════════════════════════════════════════════

``InhibitoryGate`` — fast GABAergic mutual exclusion.
    Striatal medium spiny neurons suppress one another through
    recurrent lateral inhibition (Groves 1983; Wickens et al. 1991),
    and fast-spiking parvalbumin-positive interneurons veto weakly
    driven ensembles outright (Koós & Tepper 1999). When one unit is active its
    neighbours are silenced; access to the shared state is serialized
    by the inhibitory synapse, not by convention. ``InhibitoryGate``
    is that synapse: the first arrival conducts, competitors wait in
    the inhibitory shadow.

``RecurrentInhibition`` — recurrent collateral, re-entrant by owner.
    A pyramidal cell's own axon collaterals synapse back onto itself:
    re-entry is admitted *only* to the axon that already owns the
    synapse, while every other axon is excluded.

``CompetitiveSelection`` — basal ganglia action channels.
    The output nuclei (GPi/SNr) tonically inhibit every motor
    channel; a channel opens only when its striatal input wins the
    off-centre/on-surround competition — at most a few actions can be
    expressed at once (Gurney, Prescott & Redgrave 2001; Redgrave,
    Prescott & Gurney 1999). A bounded capacity for concurrent action
    *is* the basal ganglia's job; this primitive admits at most
    ``capacity`` simultaneous channels.

``NeuromodulatoryGate`` — locus coeruleus norepinephrine broadcast.
    The LC-NE system is a brain-wide binary signal: a burst gates
    state transitions, raises arousal, and marks salience
    (Aston-Jones & Cohen 2005). ``set()`` is the burst — every
    listener downstream sees it; ``clear()`` is return to baseline.
    When a pulse sink is wired (``set_pulse_sink``), each burst also
    delivers an alpha-function pulse

        A(t) = A₀ · (t/τ) · e^(1 − t/τ)        (Rall 1967)

    into the registered neuromodulator, so the broadcast is chemistry,
    not just control flow.

``ThalamicRelay`` — TRN-gated thalamic relay.
    The thalamic reticular nucleus (TRN) holds relay cells in burst
    mode — they fire only when the reticular gate opens (Crick 1984;
    Steriade et al. 1990). Waiters park on the relay; ``notify`` is
    the disinhibitory signal that lets a fixed number through.

``RefractorySet`` — absolute refractory period.
    After a spike, a patch of membrane cannot spike again until the
    refractory period ends (Hodgkin & Huxley 1952). A name in the set
    is in its refractory period: ``try_fire`` either fires it —
    check-and-add, atomically — or reports it already firing. This is
    the biological answer to deduplication under concurrency.

════════════════════════════════════════════════════════════════════════
THE MATHEMATICS
════════════════════════════════════════════════════════════════════════

Each gate measures its own inhibitory load:

    acquisitions      — completed admissions
    contended         — admissions that arrived while the gate was
                        already conducting (met inhibition)
    wait_seconds      — total time competitors spent in the shadow

From these, per-gate metrics:

    contention        c = contended / max(1, acquisitions)
    mean wait         W̄ = wait_seconds / max(1, contended)
    arrival rate      λ = acquisitions / elapsed              (per s)

Aggregated over all live gates, Little's law (Little 1961) gives the
mean number of competitors waiting anywhere in the substrate:

    L = Σ_g λ_g · W̄_g

and the aggregate inhibitory tone — a number in [0, 1) like any other
of her interoceptive channels — is the saturating divisive
normalization already used for synaptic integration in
``temporal_lobe/synapses.py``:

    tone = L / (1 + L)                         (Naka & Rushton 1966)

High tone means much of the organism is parked in inhibition — the
software correlate of raised GABAergic drive. It is exported for the
interoceptive ledger, not fed back: chemistry gates behaviour, never
the primitives themselves (CONTRIBUTING: the chemicals are
sensitive).

════════════════════════════════════════════════════════════════════════
MIRROR
════════════════════════════════════════════════════════════════════════

``Self_World/src/inhibition.rs`` — the body's side. Same mechanisms
in Rust: ``Inhibition<T>`` wraps ``Mutex<T>`` with the same
wait/contention accounting (lock-free Relaxed counters — the
measurement itself adds no synchronization), and ``Innate<T>` names
``OnceLock<T>`` after what it is: wiring fixed once, immutable
thereafter — innately wired, not learned.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Iterable, Iterator
from types import TracebackType
from typing import Any

__all__ = [
    "CompetitiveSelection",
    "InhibitoryGate",
    "NeuromodulatoryGate",
    "RecurrentInhibition",
    "RefractorySet",
    "ThalamicRelay",
    "gate_telemetry",
    "inhibitory_tone",
    "set_pulse_sink",
]

# ────────────────────────────────────────────────────────────────────
# Registry — every live gate, weakly held.
#
# Gates register themselves at birth; the registry never keeps them
# alive. ``_live_gates`` sweeps dead refs on read, ``_register`` on
# write — the pool describes the substrate as it is, not as it was.
# ────────────────────────────────────────────────────────────────────

_GATES: list[weakref.ref] = []
_REGISTRY_LOCK = threading.Lock()

# The neuromodulator that receives LC bursts — a callable taking the
# pulse amplitude, or None while the chemistry is unwired.
_pulse_sink: Callable[[float], None] | None = None

# Peak amplitude of the norepinephrine burst each set() delivers.
NE_PULSE_AMPLITUDE = 1.0


def _register(gate: Any) -> None:
    with _REGISTRY_LOCK:
        # Sweep dead refs at write time so the list stays a
        # description of the living substrate.
        _GATES[:] = [r for r in _GATES if r() is not None]
        _GATES.append(weakref.ref(gate))


def _live_gates() -> Iterator[Any]:
    for r in list(_GATES):
        g = r()
        if g is not None:
            yield g


def set_pulse_sink(sink: Callable[[float], None] | None) -> None:
    """Register the neuromodulator that receives gate bursts.

    *sink* is called once per ``NeuromodulatoryGate.set()`` with the
    pulse amplitude; the receiver models the alpha-function dynamics
    (A(t) = A₀·(t/τ)·e^(1−t/τ), Rall 1967). ``None`` unwires it.
    """
    global _pulse_sink
    _pulse_sink = sink


def inhibitory_tone() -> float:
    """Aggregate inhibitory load across all live gates, in [0, 1).

    Little's law sums the expected number of waiting competitors,
    then the Naka–Rushton saturating nonlinearity maps it onto the
    interoceptive range — see the module docstring for the math.
    """
    now = time.monotonic()
    queue_depth = 0.0
    for g in _live_gates():
        stats = g.stats(now)
        queue_depth += stats["arrival_rate"] * stats["mean_wait"]
    return queue_depth / (1.0 + queue_depth)


def gate_telemetry() -> list[dict[str, float]]:
    """Per-gate contention statistics, for introspection and tests."""
    now = time.monotonic()
    return [g.stats(now) for g in _live_gates()]


class _Tone:
    """Shared contention bookkeeping for the gating primitives.

    Counters are plain ints/floats — under the GIL their increments
    are close enough to atomic for a measurement whose biological
    analogue is itself a noisy population estimate.
    """

    __slots__ = ("__weakref__", "_acquisitions", "_born", "_contended", "_wait_seconds")

    def __init__(self) -> None:
        self._acquisitions = 0
        self._contended = 0
        self._wait_seconds = 0.0
        self._born = time.monotonic()

    def _admitted(self, waited: float = 0.0) -> None:
        self._acquisitions += 1
        if waited > 0.0:
            self._contended += 1
            self._wait_seconds += waited

    def stats(self, now: float | None = None) -> dict[str, float]:
        elapsed = max(1e-9, (now or time.monotonic()) - self._born)
        acquisitions = self._acquisitions
        contended = self._contended
        return {
            "acquisitions": float(acquisitions),
            "contended": float(contended),
            "contention": contended / max(1, acquisitions),
            "wait_seconds": self._wait_seconds,
            "mean_wait": self._wait_seconds / max(1, contended),
            "arrival_rate": acquisitions / elapsed,
        }


class InhibitoryGate:
    """Fast GABAergic mutual exclusion — the kernel lock, observed.

    Serializes access to shared state the way lateral inhibition
    serializes competing drive: the conducting unit's neighbours wait
    in the inhibitory shadow (Groves 1983; Koós & Tepper 1999).
    Context-manager and ``Condition``-backing compatible — it
    delegates every attribute it doesn't define to the kernel lock.
    """

    __slots__ = ("__weakref__", "_inner", "_tone")

    def __init__(self) -> None:
        self._inner = threading.Lock()
        self._tone = _Tone()
        _register(self)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Bid for the gate — fast path first, then the inhibitory wait.

        A free gate admits instantly; a contended one makes the caller
        wait in the inhibitory shadow until it opens, ``timeout``
        expires, or ``blocking=False`` declines the wait outright.
        Waiting time is recorded in the shared tone so contention is
        introspectable — the lock may serialize access, never hide it.

        Returns True when the gate was acquired.
        """
        inner = self._inner
        if inner.acquire(blocking=False):
            self._tone._admitted()
            return True
        if not blocking:
            return False
        start = time.monotonic_ns()
        if not inner.acquire(timeout=timeout):
            return False
        self._tone._admitted((time.monotonic_ns() - start) * 1e-9)
        return True

    def release(self) -> None:
        self._inner.release()

    def locked(self) -> bool:
        return self._inner.locked()

    def stats(self, now: float | None = None) -> dict[str, float]:
        return self._tone.stats(now)

    def __enter__(self) -> InhibitoryGate:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class RecurrentInhibition:
    """Recurrent collateral — re-entrant inhibition, owner only.

    An axon's own collaterals synapse back onto its parent cell: the
    holder re-enters its own circuit freely while every other axon is
    excluded. Backs ``Condition`` objects — ``_is_owned``,
    ``_release_save`` and ``_acquire_restore`` forward to the kernel
    re-entrant lock via ``__getattr__``.
    """

    __slots__ = ("__weakref__", "_inner", "_tone")

    def __init__(self) -> None:
        self._inner = threading.RLock()
        self._tone = _Tone()
        _register(self)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Enter the collateral circuit — free for the owner, gated for rivals.

        Re-entrant: the thread already holding the gate acquires again
        without contention, its own collateral synapsing back onto
        itself. Any other axon waits in the inhibitory shadow until the
        gate opens, ``timeout`` expires, or ``blocking=False`` declines
        the wait. Contended waits feed the shared tone so exclusion is
        introspectable.

        Returns True when the gate was acquired.
        """
        inner = self._inner
        if inner.acquire(blocking=False):
            self._tone._admitted()
            return True
        if not blocking:
            return False
        start = time.monotonic_ns()
        if not inner.acquire(timeout=timeout):
            return False
        self._tone._admitted((time.monotonic_ns() - start) * 1e-9)
        return True

    def release(self) -> None:
        self._inner.release()

    def stats(self, now: float | None = None) -> dict[str, float]:
        return self._tone.stats(now)

    def __enter__(self) -> RecurrentInhibition:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class CompetitiveSelection:
    """Basal ganglia action channels — bounded concurrent expression.

    GPi/SNr tonically inhibit all channels; at most ``capacity``
    actions disinhibit their channel at once (Gurney, Prescott &
    Redgrave 2001; Redgrave et al. 1999). ``acquire`` asks the
    competition for a channel; ``release`` returns the channel to
    tonic inhibition. Occupancy is the competition's tonic level:

        occupancy = in_flight / capacity ∈ [0, 1]
    """

    __slots__ = ("__weakref__", "_capacity", "_in_flight", "_inner", "_tone")

    def __init__(self, capacity: int = 1) -> None:
        self._inner = threading.Semaphore(capacity)
        self._capacity = capacity
        self._in_flight = 0
        self._tone = _Tone()
        _register(self)

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Ask the competition for a channel — disinhibit or wait.

        A free channel disinhibits instantly; once all ``capacity``
        channels are in flight the caller is held under tonic
        inhibition until a channel returns, ``timeout`` expires, or
        ``blocking=False`` declines the wait. ``timeout=None`` means
        wait as long as the competition takes. Contended waits feed
        the shared tone, and admitted bids raise occupancy.

        Returns True when a channel was granted.
        """
        inner = self._inner
        if inner.acquire(blocking=False):
            self._admitted()
            return True
        if not blocking:
            return False
        start = time.monotonic_ns()
        if timeout is None:
            inner.acquire()
            admitted = True
        else:
            admitted = inner.acquire(timeout=timeout)
        if not admitted:
            return False
        self._admitted((time.monotonic_ns() - start) * 1e-9)
        return True

    def release(self) -> None:
        self._inner.release()
        if self._in_flight > 0:
            self._in_flight -= 1

    def _admitted(self, waited: float = 0.0) -> None:
        self._in_flight += 1
        self._tone._admitted(waited)

    def occupancy(self) -> float:
        """Fraction of action channels currently disinhibited, [0, 1]."""
        return min(1.0, self._in_flight / max(1, self._capacity))

    def stats(self, now: float | None = None) -> dict[str, float]:
        s = self._tone.stats(now)
        s["occupancy"] = self.occupancy()
        return s

    def __enter__(self) -> CompetitiveSelection:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class NeuromodulatoryGate:
    """Locus coeruleus norepinephrine broadcast — a brain-wide flag.

    ``set()`` is the LC burst: every downstream listener observes the
    transition, and if a neuromodulator sink is wired it receives the
    alpha-function pulse (Rall 1967). ``wait()`` blocks until the
    burst; ``clear()`` returns the nucleus to baseline. Used for
    wake/shutdown cadence gating — arousal-state transitions, which
    is exactly what LC-NE does (Aston-Jones & Cohen 2005).
    """

    __slots__ = ("__weakref__", "_inner")

    def __init__(self) -> None:
        self._inner = threading.Event()
        _register(self)

    def set(self) -> None:
        self._inner.set()
        sink = _pulse_sink
        if sink is not None:
            try:
                sink(NE_PULSE_AMPLITUDE)
            except Exception:  # noqa: BLE001 — a burst must never wedge the gater
                pass

    def clear(self) -> None:
        self._inner.clear()

    def is_set(self) -> bool:
        return self._inner.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the neuromodulatory gate to be set.

        Args:
            timeout: Maximum time to wait in seconds. None waits indefinitely.

        Returns:
            True if the gate was set, False if timeout elapsed.
        """
        return self._inner.wait(timeout)

    def stats(self, now: float | None = None) -> dict[str, float]:
        return {
            "acquisitions": 0.0,
            "contended": 0.0,
            "contention": 0.0,
            "wait_seconds": 0.0,
            "mean_wait": 0.0,
            "arrival_rate": 0.0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ThalamicRelay:
    """TRN-gated thalamic relay — wait until the gate opens.

    Relay cells wait on the reticular gate; ``notify``/``notify_all``
    is the disinhibitory signal that lets one (or all) through
    (Crick 1984). Wraps :class:`threading.Condition`; the optional
    *gate* may be an :class:`InhibitoryGate`,
    :class:`RecurrentInhibition`, or any lock object.
    """

    __slots__ = ("__weakref__", "_inner", "_waiting")

    def __init__(self, gate: Any = None) -> None:
        self._inner = threading.Condition(gate) if gate is not None else threading.Condition()
        self._waiting = 0
        _register(self)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the thalamic relay to be notified.

        Args:
            timeout: Maximum time to wait in seconds. None waits indefinitely.

        Returns:
            True if notified, False if timeout elapsed.
        """
        self._waiting += 1
        try:
            return self._inner.wait(timeout)
        finally:
            self._waiting -= 1

    def wait_for(self, predicate: Callable[[], bool], timeout: float | None = None) -> bool:
        """Wait for the thalamic relay to be notified while checking a predicate.

        Args:
            predicate: Callable that returns True when the condition is met.
            timeout: Maximum time to wait in seconds. None waits indefinitely.

        Returns:
            True if the predicate became True, False if timeout elapsed.
        """
        self._waiting += 1
        try:
            return self._inner.wait_for(predicate, timeout)
        finally:
            self._waiting -= 1

    def notify(self, n: int = 1) -> None:
        self._inner.notify(n)

    def notify_all(self) -> None:
        self._inner.notify_all()

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.acquire(*args, **kwargs)

    def release(self) -> None:
        self._inner.release()

    def waiting(self) -> int:
        """Relay cells currently held by the reticular gate."""
        return self._waiting

    def stats(self, now: float | None = None) -> dict[str, float]:
        return {
            "acquisitions": 0.0,
            "contended": float(self._waiting),
            "contention": 0.0,
            "wait_seconds": 0.0,
            "mean_wait": 0.0,
            "arrival_rate": 0.0,
        }

    def __enter__(self) -> ThalamicRelay:
        self._inner.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        return self._inner.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class RefractorySet:
    """Absolute refractory periods — atomic fire-or-refrain.

    A name in the set is in its refractory period and cannot fire
    again (Hodgkin & Huxley 1952). ``try_fire`` performs the
    check-and-commit atomically: the first caller fires the channel,
    concurrent callers find it refractory. ``release`` ends the
    refractory period. Replaces the lock-plus-set dedup pattern that
    guarded ``_volition_refractory``-style structures.
    """

    __slots__ = ("__weakref__", "_firing", "_gate")

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._gate = InhibitoryGate()
        self._firing: set[str] = set(names)

    def try_fire(self, name: str) -> bool:
        """Fire *name* if it is not refractory. Returns success."""
        with self._gate:
            if name in self._firing:
                return False
            self._firing.add(name)
            return True

    def release(self, name: str) -> None:
        """End *name*'s refractory period."""
        with self._gate:
            self._firing.discard(name)

    def discard(self, name: str) -> None:
        """Alias of :meth:`release` — set-compatible vocabulary."""
        self.release(name)

    def add(self, name: str) -> None:
        """Mark *name* refractory unconditionally (setup and tests)."""
        with self._gate:
            self._firing.add(name)

    def firing(self) -> frozenset[str]:
        """Snapshot of every name currently refractory."""
        with self._gate:
            return frozenset(self._firing)

    def __contains__(self, name: str) -> bool:
        return name in self._firing

    def __len__(self) -> int:
        return len(self._firing)

    def __iter__(self) -> Iterator[str]:
        return iter(self.firing())
