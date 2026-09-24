"""Commitment boundaries — continuous signal → discrete state.

Genesis's internal state is continuous (graded activations, sigmoid
dose-responses, floating-point neurochemistry), but her transitions
are discrete: she is asleep or awake, announced-drowsy or not. A
commitment boundary is the digitizer that sits between the two — the
functional analog of an all-or-none spike threshold or a logic-level
band, generalized for a scalar signal:

- **enter / release** (hysteresis deadband): the signal must rise to
  ``enter`` to commit and fall to ``release`` to un-commit. Inside the
  band the state holds, so flicker near the boundary cannot oscillate
  the output — the same noise margin that separates 0.4V from 2.4V.
- **confirm_s** (sustained crossing): the signal must hold at or above
  ``enter`` *continuously* for this long before the boundary commits.
  This is the time-domain analog of the venus flytrap's two-trigger
  rule — a lone transient spike cannot commit, only sustained drive
  can. A single dip below ``enter`` restarts the confirmation window.
- **refractory_s**: once committed, the boundary cannot release for
  this long regardless of the signal — a minimum dwell that prevents
  a fresh transition from un-committing on the first dip.

The boundary is deliberately **content-blind**: it sees a magnitude
and a clock, never what the signal means. It gates *whether and when*
a transition happens, never *which* transition. That is what makes it
physics rather than a script — the same discipline as seeds vs.
hardcoded responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CommitmentBoundary"]


@dataclass(slots=True)
class CommitmentBoundary:
    """Latch a continuous scalar signal into a binary committed state.

    Feed the signal with :meth:`update` on every observation; read
    :attr:`committed` for the latched state and :attr:`just_committed`
    / :attr:`just_released` for the transition edges.

    Attributes:
        enter: Signal level at or above which commitment can begin.
        release: Signal level at or below which a committed boundary
            releases. Must be ≤ ``enter``; the gap is the deadband.
        confirm_s: Seconds the signal must hold ≥ ``enter``
            continuously before committing. 0 = commit on first
            crossing (legacy single-trigger behavior).
        refractory_s: Minimum seconds the boundary stays committed
            once latched, regardless of the signal.
        just_committed: True only on the update that latched.
        just_released: True only on the update that released.
    """

    enter: float
    release: float
    confirm_s: float = 0.0
    refractory_s: float = 0.0
    just_committed: bool = field(default=False, init=False)
    just_released: bool = field(default=False, init=False)
    _committed: bool = field(default=False, init=False)
    _crossed_at: float | None = field(default=None, init=False)
    _committed_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.release > self.enter:
            raise ValueError(
                f"release ({self.release}) must be ≤ enter ({self.enter}) — "
                "a deadband cannot be negative"
            )
        if self.confirm_s < 0.0 or self.refractory_s < 0.0:
            raise ValueError("confirm_s and refractory_s must be non-negative")

    @property
    def committed(self) -> bool:
        """The latched state — True while the boundary is committed."""
        return self._committed

    def update(self, signal: float, now: float) -> bool:
        """Feed one observation; return the committed state.

        ``now`` is the observation time (seconds, any monotonic clock).
        Edge flags :attr:`just_committed` / :attr:`just_released` are
        set for exactly the update on which the state flips.
        """
        self.just_committed = False
        self.just_released = False

        if not self._committed:
            if signal >= self.enter:
                if self._crossed_at is None:
                    self._crossed_at = now
                if now - self._crossed_at >= self.confirm_s:
                    self._committed = True
                    self._committed_at = now
                    self._crossed_at = None
                    self.just_committed = True
            else:
                # Below enter — the confirmation window restarts.
                # Sustained means sustained.
                self._crossed_at = None
            return self._committed

        if (
            signal <= self.release
            and now - self._committed_at >= self.refractory_s
        ):
            self._committed = False
            self.just_released = True
        return self._committed

    def reset(self) -> None:
        """Return to the uncommitted state, clearing all timers."""
        self._committed = False
        self._crossed_at = None
        self._committed_at = 0.0
        self.just_committed = False
        self.just_released = False
