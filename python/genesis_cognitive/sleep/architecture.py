"""Sleep architecture — the structural machinery of sleep.

Contains the standalone subsystems that implement sleep architecture:
the ultradian cycle tracker, synaptic homeostasis downscaler, and
hippocampal replay. These are pure subsystems with no dependency on
InnerLife — they operate on the concept network and sleep stage
enum directly, and are tested independently.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from ..brain_waves import SleepStage
from ..concepts import ConceptNetwork, RelationType

__all__ = [
    "REPLAY_COMPRESSION",
    "HippocampalReplay",
    "ReplayItem",
    "SleepCycleTracker",
    "SynapticDownscaler",
]

# ─── 90-minute ultradian sleep cycle ──────────────────────────────
#
# Stage progression within each ~90-minute cycle:
#   N1 → N2 → N3 → N2 → REM
# The first cycle has more N3 (deep sleep); later cycles have more REM.
# (Carskadon & Dement, 2011; Diekelmann & Born, 2010)

# The ordered stages within a single ultradian cycle.
_CYCLE_STAGES: tuple[SleepStage, ...] = (
    SleepStage.N1,
    SleepStage.N2,
    SleepStage.N3,
    SleepStage.N2,
    SleepStage.REM,
)


@dataclass(slots=True)
class SleepCycleTracker:
    """Tracks the 90-minute ultradian NREM-REM sleep cycle progression.

    Human sleep is organised into 4-6 cycles per night, each lasting
    roughly 90 minutes. Within each cycle the sleep architecture
    follows a stereotyped descent and ascent:

        N1 → N2 → N3 → N2 → REM

    The first half of the night is dominated by slow-wave sleep (N3),
    while REM duration grows across successive cycles — the first
    cycle's REM may be only ~10 minutes, increasing to ~30 minutes in
    later cycles. Conversely, N3 is longest in the first cycle
    (~20-40 min) and shortens thereafter (Carskadon & Dement, 2011;
    Diekelmann & Born, 2010).

    The tracker advances in subjective time via :meth:`advance`. Stage
    durations vary by cycle number to reproduce the well-known
    redistribution of N3 and REM across the night.

    In nap mode (``nap=True``), the cycle is limited to light sleep
    (N1→N2) only — no N3 deep consolidation (no synaptic downscaling,
    no hippocampal replay) and no REM (no emotional processing). This
    models a short rest: light consolidation via N2 sleep spindles
    (hippocampal→neocortical transfer) without the heavy systems
    consolidation of N3 or the emotional processing of REM. She wakes
    after the N2 period ends rather than continuing into deep sleep.
    """

    cycle_number: int = 1  # 1-based; increments after each REM
    current_stage: SleepStage = SleepStage.N1
    _stage_slot: int = 0  # 0-based index into _CYCLE_STAGES
    time_in_stage: float = 0.0  # subjective seconds in the current stage
    time_in_cycle: float = 0.0  # subjective seconds in the current cycle
    cycles_completed: int = 0  # full N1→REM cycles finished
    nap: bool = False  # light sleep only (N1→N2, no N3/REM)

    # ── Stage duration model ──────────────────────────────────────

    @staticmethod
    def _stage_durations(cycle: int) -> dict[SleepStage, float]:
        """Return target durations (seconds) for each stage in a cycle.

        Durations vary with cycle number to model the well-documented
        shift from deep-sleep-dominant early cycles to REM-dominant
        later cycles:

        - **N1**: ~5 min, roughly constant across cycles.
        - **N2**: ~15 min in the first cycle, increasing slightly in
          later cycles (N2 grows as N3 shrinks).
        - **N3**: ~40 min in cycle 1, decreasing toward ~10 min by
          cycle 5+ (deep sleep front-loaded).
        - **REM**: ~10 min in cycle 1, increasing toward ~30 min by
          cycle 5+ (REM back-loaded).

        The second N2 (the ascent back toward REM) is shorter than the
        first N2, matching polysomnographic observations.
        """
        # N1 — brief transition, ~5 min, stable.
        n1 = 300.0
        # N2 (descent) — grows slightly in later cycles.
        n2_descent = 900.0 + min(cycle - 1, 4) * 60.0  # 15 → 19 min
        # N3 — front-loaded: longest early, shortest late.
        n3 = max(600.0, 2400.0 - (cycle - 1) * 450.0)  # 40 → 10 min
        # N2 (ascent) — shorter than the descent N2.
        # Computed in _stage_duration() for slot 3, not stored here
        # because the dict can only hold one N2 value.
        # REM — back-loaded: shortest early, longest late.
        rem = 600.0 + min(cycle - 1, 4) * 360.0  # 10 → 30 min
        return {
            SleepStage.N1: n1,
            SleepStage.N2: n2_descent,  # used for the first N2 slot
            SleepStage.N3: n3,
            # The second N2 slot uses n2_ascent; we disambiguate by
            # position in the cycle (handled in advance()).
            SleepStage.REM: rem,
        }

    def _stage_duration(self, stage: SleepStage, slot: int) -> float:
        """Duration for *stage* at cycle position *slot* (0-based index
        into ``_CYCLE_STAGES``).

        The two N2 slots (index 1 and 3) have different durations: the
        descent N2 is longer, the ascent N2 shorter.
        """
        durations = self._stage_durations(self.cycle_number)
        if stage is SleepStage.N2 and slot == 3:
            # Ascent N2 — shorter.
            n2_descent = durations[SleepStage.N2]
            return max(300.0, n2_descent * 0.4)
        return durations[stage]

    # ── Progression ───────────────────────────────────────────────

    @property
    def _nap_complete(self) -> bool:
        """Whether a nap cycle has finished (N1→N2 done, no N3/REM)."""
        return self.nap and self._stage_slot >= 2

    def advance(self, dt: float) -> SleepStage:
        """Advance the cycle by *dt* subjective seconds.

        Progresses through the current stage and transitions to the
        next stage (or starts a new cycle after REM) when the stage
        duration is reached. Returns the (possibly new) current stage.

        In nap mode, the cycle stops after N2 (slot 1) — it does not
        descend into N3 or REM. The caller detects the end of the nap
        via :attr:`_nap_complete` and can trigger a wake.

        Args:
            dt: Elapsed subjective time in seconds (must be ≥ 0).
        """
        if dt <= 0.0:
            return self.current_stage

        self.time_in_stage += dt
        self.time_in_cycle += dt

        # Consume elapsed time across as many stages as needed. A
        # single large dt may span multiple stage transitions; the
        # surplus time carries forward into the next stage rather
        # than being discarded.
        while True:
            slot = self._stage_slot
            duration = self._stage_duration(self.current_stage, slot)

            if self.time_in_stage < duration:
                break

            # In nap mode, stop after N2 — don't descend into N3.
            # The nap is light sleep only (N1→N2). The caller detects
            # the end via _nap_complete and triggers a wake.
            if self.nap and slot + 1 >= 2:
                # Clamp to the end of N2 — don't carry surplus into N3.
                self.time_in_stage = duration
                self._stage_slot = 2  # mark nap as complete
                break

            # Transition to the next stage in the cycle.
            if slot + 1 < len(_CYCLE_STAGES):
                self._stage_slot = slot + 1
                self.current_stage = _CYCLE_STAGES[slot + 1]
                self.time_in_stage -= duration
            else:
                # REM finished — a full cycle is complete.
                self.cycles_completed += 1
                self.cycle_number += 1
                self._stage_slot = 0
                self.current_stage = SleepStage.N1
                self.time_in_stage -= duration
                # The surplus carries into the new cycle's time.
                self.time_in_cycle = self.time_in_stage
                # If the surplus is large enough to span into the new
                # cycle, keep consuming; otherwise we're done.
                if self.time_in_stage < self._stage_duration(
                    self.current_stage, 0
                ):
                    break

        return self.current_stage

    @property
    def cycle_progress(self) -> float:
        """Fraction of the current cycle elapsed, in [0, 1)."""
        total = sum(self._stage_duration(s, i) for i, s in enumerate(_CYCLE_STAGES))
        if total <= 0.0:
            return 0.0
        return min(0.9999, self.time_in_cycle / total)

    @property
    def stage_progress(self) -> float:
        """Fraction of the current stage elapsed, in [0, 1)."""
        duration = self._stage_duration(self.current_stage, self._stage_slot)
        if duration <= 0.0:
            return 0.0
        return min(0.9999, self.time_in_stage / duration)

    def reset(self) -> None:
        """Reset the tracker to the start of cycle 1 (N1)."""
        self.cycle_number = 1
        self.current_stage = SleepStage.N1
        self._stage_slot = 0
        self.time_in_stage = 0.0
        self.time_in_cycle = 0.0

    def describe(self) -> str:
        """Human-readable description of the cycle state."""
        return (
            f"[sleep-cycle] cycle {self.cycle_number}, "
            f"stage {self.current_stage.value.upper()} "
            f"({self.stage_progress:.0%} of stage, "
            f"{self.cycle_progress:.0%} of cycle, "
            f"{self.cycles_completed} completed)"
        )


# ─── Synaptic Homeostasis Hypothesis (SHY) ────────────────────────
#
# The Synaptic Homeostasis Hypothesis (Tononi & Cirelli, 2003, 2006,
# 2014) proposes that wakefulness potentiates synapses (a net increase
# in synaptic strength from learning and neural activity) and that the
# primary function of slow-wave sleep is to renormalise — downscale —
# synaptic strengths back toward a baseline. This prevents synaptic
# saturation, restores the capacity for learning, and reduces the
# metabolic cost of sustained strong connections.


@dataclass(slots=True)
class SynapticDownscaler:
    """Synaptic Homeostasis Hypothesis (SHY) downscaling mechanism.

    Models the accumulation of synaptic load during wakefulness and its
    renormalisation during N3 slow-wave sleep (Tononi & Cirelli, 2003,
    2006, 2014):

    - **Wakefulness** potentiates synapses — every learning event,
      concept activation, and neurochemical impulse increases a
      ``synaptic_load`` accumulator. This is the "price" the brain
      pays for being awake and learning.
    - **N3 (slow-wave sleep)** downscales all synaptic strengths
      (concept-connection weights) by a small amount per tick,
      proportional to the accumulated load. The downscaling is
      *homeostatic*: stronger connections are reduced more (the
      reduction is proportional to the current weight), which prevents
      the strongest synapses from dominating and restores signal-to-
      noise for the next waking period.
    - After downscaling, the synaptic load decreases, completing the
      homeostatic loop: wake potentiates → sleep renormalises.

    The total downscaling applied and a bounded history of load values
    are tracked for introspection.

    Thread safety: ``accumulate_synaptic_load`` may be called from
    external threads (waking cognition) while ``downscale_during_n3``
    runs in the background sleep thread. A lock guards the shared
    ``synaptic_load`` field.
    """

    synaptic_load: float = 0.0  # accumulated during wakefulness [0, ∞)
    total_downscaling: float = 0.0  # cumulative weight removed during N3
    # Bounded history of synaptic load snapshots for introspection.
    load_history: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def accumulate_synaptic_load(self, amount: float) -> None:
        """Accumulate synaptic load during wakefulness.

        Wakefulness potentates synapses through learning, concept
        activation, and impulse activity. Each such event increases
        the synaptic load — the "sleep pressure" that N3 will later
        renormalise (Tononi & Cirelli, 2014).

        Args:
            amount: The load increment (≥ 0). Typical small per-event
                values are ~0.01-0.05.
        """
        if amount > 0.0:
            with self._lock:
                self.synaptic_load += amount

    def downscale_during_n3(self, network: ConceptNetwork, dt: float) -> float:
        """Downscale synaptic strengths during N3 slow-wave sleep.

        Implements the SHY renormalisation: every connection weight in
        the concept network is reduced by an amount proportional to
        (a) the accumulated synaptic load and (b) the current weight
        itself — so stronger synapses are downscaled more, preventing
        the strongest connections from saturating and dominating
        (Tononi & Cirelli, 2006, 2014).

        After downscaling, the synaptic load is reduced proportionally,
        modelling the homeostatic resolution of sleep pressure.

        Args:
            network: The concept network whose edge weights to
                renormalise.
            dt: Elapsed subjective time in seconds.

        Returns:
            The total weight reduction applied this call.
        """
        with self._lock:
            if dt <= 0.0 or self.synaptic_load <= 0.0:
                return 0.0

            # Downscaling rate per second, gated by synaptic load.
            # A higher load → faster renormalisation, but bounded so a
            # single tick never strips the network.
            rate = min(0.05, 0.01 * self.synaptic_load)

            total_removed = 0.0
            for edge in network.edges:
                if edge.weight <= 0.0:
                    continue
                # Proportional downscaling: stronger synapses lose more.
                # This is the core of SHY — renormalisation that preserves
                # relative ordering while reducing absolute strengths.
                reduction = rate * dt * edge.weight
                edge.weight = max(0.0, edge.weight - reduction)
                total_removed += reduction

            # Resolve synaptic load: downscaling consumes sleep pressure.
            # The load decreases by the same rate, completing the
            # homeostatic loop (wake potentiates → sleep renormalises).
            self.synaptic_load = max(0.0, self.synaptic_load - rate * dt)
            self.total_downscaling += total_removed

            # Record a load snapshot for history.
            self.load_history.append(self.synaptic_load)
        return total_removed

    def get_synaptic_load(self) -> float:
        """Return the current accumulated synaptic load."""
        with self._lock:
            return self.synaptic_load

    def describe(self) -> str:
        """Human-readable description of the downscaling state."""
        return (
            f"[shy] synaptic load {self.synaptic_load:.3f}, "
            f"total downscaling {self.total_downscaling:.3f}"
        )


# ─── Hippocampal replay during sleep ──────────────────────────────
#
# During N3 slow-wave sleep, hippocampal sharp wave-ripples drive the
# compressed replay of recent experiences (~20x faster than real-time).
# Replay strengthens the connections between co-active concepts and
# transfers information from hippocampal to neocortical representation
# (systems consolidation).
# (Wilson & McNaughton, 1994; Stickgold, 2005; Diekelmann & Born, 2010)

# Replay compression factor — a 10-second experience replays in ~0.5 s.
REPLAY_COMPRESSION = 20.0


@dataclass(slots=True)
class ReplayItem:
    """A queued experience sequence awaiting hippocampal replay.

    Each item captures an ordered sequence of concept IDs that were
    co-active during a recent waking experience, together with a
    salience weight that gates how strongly the replay strengthens
    the corresponding connections.
    """

    sequence: list[str]  # ordered concept IDs in the experience
    salience: float  # 0..1, how important this experience was
    replay_count: int = 0  # how many times this has been replayed
    queued_at: float = 0.0  # wall-clock time when queued


@dataclass(slots=True)
class HippocampalReplay:
    """Compressed sequence replay during N3 slow-wave sleep.

    During slow-wave sleep, hippocampal sharp wave-ripples (SWRs) drive
    the replay of recent experiences in compressed time — roughly 20x
    faster than the original experience (Wilson & McNaughton, 1994;
    Skaggs & McNaughton, 1996). Replay serves two consolidation
    functions (Stickgold, 2005; Diekelmann & Born, 2010):

    1. **Connection strengthening**: co-active concepts in the replayed
       sequence have their connections strengthened, reinforcing the
       associative structure of the experience.
    2. **Systems consolidation**: replay transfers the hippocampal
       (episodic) representation toward a neocortical (semantic)
       representation — modelled here by boosting concept confidence.

    The replay queue is populated during wakefulness via
    :meth:`queue_for_replay` and drained during N3 via
    :meth:`replay_during_n3`. Each SWR event triggers a replay burst.
    """

    # Queue of experience sequences awaiting replay (most recent last).
    _queue: deque[ReplayItem] = field(default_factory=lambda: deque(maxlen=50))
    # Total number of replay bursts performed.
    _replay_count: int = 0
    # Map of sequence-signature → replay count, for tracking which
    # experiences have been replayed and how many times. Bounded to
    # prevent unbounded growth over long runs — least-replayed entries
    # are evicted when the cap is reached.
    _replay_history: dict[str, int] = field(default_factory=dict)
    _REPLAY_HISTORY_MAX = 200

    def queue_for_replay(self, experience_sequence: list[str], salience: float) -> None:
        """Queue a recent experience for replay during N3 sleep.

        Called during wakefulness when a salient experience occurs
        (e.g. a learning event, a conversation, an insight). The
        sequence is an ordered list of concept IDs that were co-active
        during the experience. Higher salience experiences are replayed
        more times before being dropped from the queue.

        Args:
            experience_sequence: Ordered concept IDs in the experience.
            salience: Importance of the experience in [0, 1].
        """
        if not experience_sequence:
            return
        item = ReplayItem(
            sequence=list(experience_sequence),
            salience=max(0.0, min(1.0, salience)),
            queued_at=time.time(),
        )
        self._queue.append(item)

    def replay_during_n3(self, network: ConceptNetwork, dt: float) -> list[ReplayItem]:
        """Replay queued experience sequences during N3 sleep.

        Each call processes as many replay bursts as fit within *dt*
        subjective seconds, given the compressed replay timescale
        (``REPLAY_COMPRESSION``× faster than real-time). Each burst
        corresponds to one sharp wave-ripple event and replays a single
        queued sequence:

        - Strengthens the connections between consecutive (and nearby)
          co-active concepts in the sequence.
        - Boosts the confidence of the replayed concepts (systems
          consolidation — hippocampal→neocortical transfer).

        Sequences with higher salience survive more replays before
        being dropped; low-salience sequences are replayed once and
        discarded.

        Args:
            network: The concept network to strengthen during replay.
            dt: Elapsed subjective time in seconds.

        Returns:
            A list of the ``ReplayItem``s that were replayed this call.
        """


        if dt <= 0.0 or not self._queue:
            return []

        # Compressed replay budget: each second of N3 affords
        # REPLAY_COMPRESSION seconds of replayed experience. We model
        # each replay burst as taking ~0.5 s of compressed time, so the
        # number of bursts available is dt * REPLAY_COMPRESSION / 0.5.
        burst_duration_compressed = 0.5  # seconds of replayed experience
        max_bursts = max(0, int(dt * REPLAY_COMPRESSION / burst_duration_compressed))

        replayed: list[ReplayItem] = []
        for _ in range(max_bursts):
            if not self._queue:
                break
            item = self._queue[0]  # peek; don't pop yet
            self._replay_one(network, item, RelationType.RELATED_TO)
            item.replay_count += 1
            self._replay_count += 1
            replayed.append(item)

            # Signature for history tracking.
            sig = "→".join(item.sequence)
            self._replay_history[sig] = self._replay_history.get(sig, 0) + 1
            # Evict least-replayed entries if history grows too large.
            if len(self._replay_history) > self._REPLAY_HISTORY_MAX:
                sorted_items = sorted(
                    self._replay_history.items(), key=lambda kv: kv[1]
                )
                excess = len(self._replay_history) - self._REPLAY_HISTORY_MAX
                for key, _ in sorted_items[:excess]:
                    del self._replay_history[key]

            # High-salience items survive more replays; low-salience
            # items are dropped after one. The survival threshold scales
            # with salience: salience 1.0 → ~5 replays, salience 0.2 →
            # ~1 replay.
            survival = int(1 + item.salience * 4)
            if item.replay_count >= survival:
                self._queue.popleft()

        return replayed

    def _replay_one(
        self,
        network: ConceptNetwork,
        item: ReplayItem,
        relation_type: RelationType,
    ) -> None:
        """Replay a single experience sequence once.

        Strengthens connections between consecutive and near-by
        co-active concepts (Hebbian co-activation) and boosts concept
        confidence (systems consolidation).
        """
        seq = item.sequence
        n = len(seq)
        if n == 0:
            return

        # Strengthen connections between consecutive concepts (the
        # temporal order of the experience) and near-neighbours
        # (co-activation within a short window).
        for i in range(n):
            # Boost confidence of each replayed concept — systems
            # consolidation (hippocampal→neocortical transfer).
            concept = network.get_concept(seq[i])
            if concept is not None and concept.confidence < 1.0:
                concept.confidence = min(
                    1.0,
                    concept.confidence + 0.02 * item.salience,
                )

            # Strengthen edges to the next 1-2 concepts in the sequence.
            for j in range(i + 1, min(i + 3, n)):
                # Weight scales with salience and decays with temporal
                # distance in the sequence.
                distance = j - i
                weight = 0.15 * item.salience / distance
                network.add_edge(
                    seq[i],
                    seq[j],
                    relation_type,
                    weight,
                    origin="hippocampal-replay",
                )

    def get_replay_count(self) -> int:
        """Return the total number of replay bursts performed."""
        return self._replay_count

    @property
    def queue_size(self) -> int:
        """Number of experience sequences awaiting replay."""
        return len(self._queue)

    @property
    def replay_history(self) -> dict[str, int]:
        """Map of sequence-signature → replay count."""
        return dict(self._replay_history)

    def describe(self) -> str:
        """Human-readable description of the replay state."""
        return f"[replay] {self._replay_count} bursts, {len(self._queue)} queued"
