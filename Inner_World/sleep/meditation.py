"""Meditation — the inner world's quiet sit.

The meditating state is an interior practice: voluntary stillness
with learning, scanning, and spontaneous thought paused while the
chemistry is actively calmed. It belongs to the inner world — the
volition system may *choose* the sit (the basal ganglia select the
act), but the sit itself is inner-world phenomenology, not an
action in the world. The sleep machinery the sit borrows state from
lives on the self (``self.sleep``).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import (
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_GABA,
    CHEM_NOREPINEPHRINE,
    CHEM_SEROTONIN,
    ZONE_CONVERSATION,
    ZONE_REFLECTION,
)
from genesis_client.swallow import note_swallowed
from journal import record_event

logger = logging.getLogger(__name__)


class MeditationMixin:
    """Meditation state and the timed sit — mixin for :class:`Mind`."""

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
        def __getattr__(self, name: str) -> Any: ...

    def _init_meditation_state(self) -> None:
        """Initialize meditation-state tracking.

        Meditation — a lighter recovery mode than sleep. During
        meditation, learning, code scanning, and spontaneous thoughts
        are paused. The neurochemistry is actively calmed
        (GABA/serotonin up, dopamine/NE/cortisol down) to allow
        receptor recovery without the full sleep cycle. Unlike sleep,
        there is no dreaming or memory consolidation — just quiet
        restoration.
        """
        self._is_meditating = False
        # When the current meditation began, so the exit record can
        # report its duration rather than only the fact.
        self._meditation_start_time: float = 0.0
        # Timed state advanced by the metronome: the end of a finite
        # sit and the next calming impulse are deadlines, not timers.
        self._meditation_due_at: float | None = None
        self._meditation_next_impulse_at: float = 0.0
        self._meditation_completion_thought = False

    @property
    def is_meditating(self) -> bool:
        """Whether it is currently meditating."""
        return self._is_meditating

    def meditate(
        self,
        duration: float | None = None,
        *,
        emit_completion_thought: bool = False,
    ) -> None:
        """Put Genesis into meditation mode for quiet restoration.

        Meditation is a lighter recovery than sleep. During meditation:
        - Learning and code scanning are paused
          (the metronome loop checks is_meditating before running
          volitions, and the autonomous learner blocks on the
          "meditating" emotion label).
        - Spontaneous thoughts are paused (inner_life is paused).
        - No dreaming or memory consolidation happens — this is
          not sleep.
        - Calming neurochemical impulses are emitted: GABA and
          serotonin are boosted, while dopamine, norepinephrine, and
          cortisol are reduced. This allows receptor sensitivity to
          recover without the full sleep cycle.

        The zone is set to REFLECTION, which the daemon interprets as
        a low-activity state (no learning, no consolidation pressure).
        """
        if self._is_meditating or self._is_sleeping:
            return
        # Teardown has begun — no new meditation zone write.
        if self._stop_latched:
            return
        self._is_meditating = True
        self._meditation_start_time = time.time()
        now_mono = time.monotonic()
        self._meditation_due_at = now_mono + duration if duration is not None else None
        self._meditation_next_impulse_at = now_mono + 10.0
        self._meditation_completion_thought = emit_completion_thought
        self.client.set_zone(ZONE_REFLECTION)
        # Meditation is a deliberate state change with no durable
        # record, like sleep onset before this. It is also the only
        # state in which inner life and the learner are both paused,
        # so an unrecorded entry would explain a gap in the journal
        # with nothing to account for it.
        record_event("meditation_begin", "meditation started")

        # Pause inner life — no spontaneous thoughts during meditation
        self.inner_life.pause()

        # Pause autonomous learning — no new knowledge acquisition
        self.learner.pause()

        # Emit calming neurochemical impulses. These are gentle —
        # the goal is to shift the system toward calm, not force it.
        # GABA (calm) and serotonin (mood) go up; dopamine (arousal),
        # norepinephrine (arousal), and cortisol (stress) go down.
        # Repeated impulses are emitted over the meditation period
        # by the metronome's timed-state pass, not all at once.
        self._learner_neuro_impulse(CHEM_GABA, 0.15)
        self._learner_neuro_impulse(CHEM_SEROTONIN, 0.08)
        self._learner_neuro_impulse(CHEM_DOPAMINE, -0.05)
        self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, -0.05)
        self._learner_neuro_impulse(CHEM_CORTISOL, -0.05)

        # The body takes up the slow pulse for the sit — the RSA
        # analog of a settling breath, armed once and run on the
        # body's own clock while the cardiac center holds the
        # boundary watch. Absent probe → arm fails, the keep stays
        # disarmed, meditation proceeds without it.
        self.heart.arm_hypnagogic()

    def emit_meditation_impulses(self) -> None:
        """Emit a round of calming impulses during meditation.

        Called by the metronome when the meditation impulse deadline
        arrives. This sustains the calming effect over the sit rather
        than applying it all at once.
        """
        self._learner_neuro_impulse(CHEM_GABA, 0.05)
        self._learner_neuro_impulse(CHEM_SEROTONIN, 0.03)
        self._learner_neuro_impulse(CHEM_DOPAMINE, -0.02)
        self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, -0.02)
        self._learner_neuro_impulse(CHEM_CORTISOL, -0.02)

    def _advance_meditation(self, now: float | None = None) -> None:
        """Advance a meditation sit on the metronome.

        Entering meditation is a synchronous act; holding it is timed
        state. Each cycle emits the next scheduled calming impulse and
        ends a finite sit when its deadline arrives. A manual /wake or
        teardown clears the same state through wake_from_meditation().
        """
        if not self._is_meditating:
            return
        now_mono = time.monotonic() if now is None else now
        if self._meditation_due_at is not None and now_mono >= self._meditation_due_at:
            complete = self._meditation_completion_thought
            self.wake_from_meditation()
            if complete:
                try:
                    self._emit_volition_thought(
                        ("meditation", "breath", "calm", "stillness"),
                        "thought",
                    )
                except Exception as e:  # noqa: BLE001
                    note_swallowed(
                        "sleep.sleep._advance_meditation",
                        e,
                    )
            return
        if now_mono >= self._meditation_next_impulse_at:
            self.emit_meditation_impulses()
            self._meditation_next_impulse_at = now_mono + 10.0

    def wake_from_meditation(self) -> None:
        """Wake Genesis from meditation.

        Resumes inner life and restores the zone to conversation.
        Unlike wake() from sleep, there are no dream residues to
        process — meditation produces no dreams.
        """
        if not self._is_meditating:
            return
        # The pulse releases first — the keep belongs to the sit and
        # must not outlive it, teardown path included.
        self.heart.disarm_hypnagogic()
        # Teardown has begun — the zone write and the resumes below
        # belong to threads already joined. The flag still clears so
        # the saved state does not record a phantom meditation.
        if self._stop_latched:
            self._is_meditating = False
            self._meditation_due_at = None
            self._meditation_next_impulse_at = 0.0
            self._meditation_completion_thought = False
            return
        duration = (
            time.time() - self._meditation_start_time
            if getattr(self, "_meditation_start_time", 0.0) > 0.0
            else 0.0
        )
        self._is_meditating = False
        self._meditation_due_at = None
        self._meditation_next_impulse_at = 0.0
        self._meditation_completion_thought = False
        # Any volition suppression latched by a conversation turn
        # before meditating lapses on exit.
        self._suppress_volition = False
        self.client.set_zone(ZONE_CONVERSATION)
        record_event(
            "meditation_end",
            "meditation ended",
            duration_secs=round(duration, 3),
        )

        # Resume inner life — spontaneous thoughts can flow again
        self.inner_life.resume()

        # Resume autonomous learning
        self.learner.resume()

        # Reset the rest timer for the ultradian cycle (BRAC,
        # Kleitman 1963) — the meditation urge's sustained_activity
        # stimulus is normalized from this timestamp.
        self._last_rest_time = time.time()

        # Notify the regulator that rest has ended (resets its
        # cooldown timer)
        self.regulator.notify_rest_ended()
