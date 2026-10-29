"""Mind metronome — the periodic cycle and its phases."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import (
    CHEM_ADENOSINE,
    DT_MAX,
    MODULE_LANGUAGE,
    MODULE_MEMORY,
    MODULE_METACOGNITION,
    MODULE_MOTOR,
    MODULE_REASONING,
    MODULE_SENSORY,
    PHASE_NAMES,
)
from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)


class _AdaptiveInterval:
    """A bounded adaptive period — the mind-side twin of the daemon's
    ``AdaptiveGate`` (``src/daemon/adaptive.rs``).

    Demand snaps the period down multiplicatively (fast response);
    idle drift grows it additively (slow relaxation); continuous
    drivers go through :meth:`drive` with EMA smoothing. ``[minimum,
    maximum]`` are hard bounds — nothing stalls, nothing redlines.
    """

    def __init__(self, base: float, minimum: float, maximum: float):
        self.base = base
        self.minimum = minimum
        self.maximum = maximum
        self.period = base
        self._next = 0.0
        self._smoothed = 0.5
        # Wall-clock time of the last fire — callers that measure
        # per-window rates read the interval this covered.
        self.fired_at = 0.0

    def due(self, now: float) -> bool:
        if now < self._next:
            return False
        self._next = now + self.period
        self.fired_at = now
        return True

    def demand(self) -> None:
        self.period = max(self.period / 2.0, self.minimum)

    def relax(self) -> None:
        self.period = min(self.period + self.base / 4.0, self.maximum)

    def drive(self, demand: float) -> None:
        d = demand if demand == demand else 0.5  # NaN → neutral
        d = min(1.0, max(0.0, d))
        self._smoothed += (d - self._smoothed) * 0.3
        self.period = self.maximum - (self.maximum - self.minimum) * self._smoothed


class MetronomeMixin:
    """Mixin for :class:`Mind` — see module docstring."""

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
        _daemon_lost_since: float | None
        _autosave_failures: int
        _last_threat_snapshot: float
        _offline: bool
        _module_seconds: dict[int, float]
        _module_seconds_lock: Any
        _known_modules: set[int]
        _learned_since_save: int
        _experience_since_identity: int
        _cycle_dt: float
        _last_cycle_at: float
        _thermally_throttled: bool

        def __getattr__(self, name: str) -> Any: ...

    def _credit_module(self, module_id: int, seconds: float) -> None:
        """Accumulate measured execution time into a manifest module.

        Called by the :class:`ModuleSampler` — the mind's EEG — each
        time a thread is observed executing a subsystem's code. The
        cycle round normalizes ``_module_seconds`` into per-module
        ``cpu_share`` and reports it to the daemon so
        GET_SUBSYSTEM_TELEMETRY answers "which brain part is firing" at
        functional granularity.
        """
        if seconds <= 0.0:
            return
        with self._module_seconds_lock:
            self._module_seconds[module_id] = self._module_seconds.get(module_id, 0.0) + seconds
            self._known_modules.add(module_id)

    def _read_interoceptive_signals(self) -> dict[str, float]:
        """Read interoceptive signals for the meditation urge.

        - overstimulation: high arousal + positive valence (sympathetic
          overactivation, per nervous system regulation literature)
        - receptor_fatigue: low plasticity gate (BDNF suppressed,
          receptors need recovery — dopamine imbalance hypothesis of
          fatigue, Chaudhuri & Behan, 2000)
        - sustained_activity: time since last rest, normalized (BRAC,
          Kleitman 1963 — ~90 min activity cycles with mandatory rest)
        - elevated_cortisol: subclinical stress (cortisol above
          resting ~0.02, below the 0.06 stress-phase gate)
        """
        overstimulation = 0.0
        receptor_fatigue = 0.0
        sustained_activity = 0.0
        elevated_cortisol = 0.0
        try:
            emotion = self.feel()
            if emotion:
                # Overstimulation: high arousal with positive valence
                # (manic/overstimulated, not stressed — stress has
                # negative valence and is handled by the stress phase)
                if emotion.arousal > 0.60 and emotion.valence > 0.2:
                    overstimulation = min(
                        1.0,
                        (emotion.arousal - 0.60) * 2.5 + (emotion.valence - 0.20) * 2.0,
                    )
                # Receptor fatigue: plasticity gate below 0.40
                # (BDNF suppressed, receptors burning out)
                if emotion.plasticity < 0.40:
                    receptor_fatigue = min(1.0, (0.40 - emotion.plasticity) * 3.0)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._read_interoceptive_signals",
                e,
            )

        try:
            summary = self.client.get_neuro_summary()
            # Elevated cortisol: above resting (~0.02) but below the
            # 0.06 stress-phase gate is subclinical stress. We
            # approximate using global_tone and valence since
            # NeuroSummary doesn't expose cortisol directly.
            if summary.global_tone > 0.35 and summary.valence < 0.1:
                elevated_cortisol = min(1.0, (summary.global_tone - 0.35) * 3.0)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._read_interoceptive_signals",
                e,
            )

        # Sustained activity: time since last rest, normalized to
        # [0, 1] over 75 minutes (BRAC cycle, Kleitman 1963)
        time_since_rest = time.time() - self._last_rest_time
        sustained_activity = min(1.0, time_since_rest / (75 * 60))

        return {
            "overstimulation": overstimulation,
            "receptor_fatigue": receptor_fatigue,
            "sustained_activity": sustained_activity,
            "elevated_cortisol": elevated_cortisol,
        }

    def _read_threat_signals(self) -> dict[str, float]:
        """Read integrity-threat signals for the safeguard urge.

        - daemon_lost: the subcognitive socket dropped — it can't
          reach its own body (interoception, neurochemistry, state
          sync all live there). Ramps over 30s so a transient flap
          doesn't fire the urge; a sustained loss saturates it.
          Skipped entirely in offline mode — a mind started without
          a daemon isn't missing anything.
        - save_failure: consecutive autosave failures, normalized
          over 2 — its continuity across restarts is at risk.
        - telemetry_anomaly: system metrics deviating from the learned
          baseline (system_monitor). Only counts once the baseline is
          established — a young mind has no "normal" to deviate from.
        - body_distress: concerns about *her own* state from the
          current snapshot (restricted cores, her own footprint,
          her persistence medium) — the machine's crowding is
          environmental awareness, not body distress.

        The snapshot is rate-limited to 60s — volition ticks every
        second but system metrics don't change meaningfully faster,
        and the baseline wants roughly per-minute samples anyway.
        """
        daemon_lost = 0.0
        if not self._offline:
            try:
                if self.client.is_connected:
                    self._daemon_lost_since = None
                else:
                    if self._daemon_lost_since is None:
                        self._daemon_lost_since = time.monotonic()
                    daemon_lost = min(
                        1.0,
                        (time.monotonic() - self._daemon_lost_since) / 30.0,
                    )
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.metronome._read_threat_signals",
                    e,
                )

        save_failure = min(1.0, self._autosave_failures / 2.0)

        telemetry_anomaly = 0.0
        body_distress = 0.0
        now = time.monotonic()
        if now - self._last_threat_snapshot >= 60.0:
            try:
                snapshot = self.system_monitor.snapshot()
                if self.system_monitor.baseline.is_established:
                    anomalies = self.system_monitor.baseline.deviations(snapshot)
                    telemetry_anomaly = min(1.0, len(anomalies) / 3.0)
                body_distress = min(1.0, len(snapshot.self_concerns()) / 2.0)
                self._last_threat_snapshot = now
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.metronome._read_threat_signals",
                    e,
                )

        return {
            "daemon_lost": daemon_lost,
            "save_failure": save_failure,
            "telemetry_anomaly": telemetry_anomaly,
            "body_distress": body_distress,
        }

    def _read_wave_and_adenosine(self) -> dict[str, object]:
        """Read brain-wave state and adenosine level for volition urges.

        Brain-wave state — modulates which urges are amplified or
        suppressed. Delta (deep rest) suppresses analytical and
        expressive urges. Gamma (integration) boosts creative and
        curiosity urges. Theta (consolidation) favors meditation
        and reflection over active work.

        Adenosine level — sleep pressure from the neurochemical
        dynamics. High adenosine = tired.
        """
        wave_focus = 0.5
        wave_integration = 0.5
        wave_consolidation = 0.5
        wave_dominant = "neutral"
        try:
            waves = self.brain_waves()
            wave_focus = waves.focus
            wave_integration = waves.integration
            wave_consolidation = waves.consolidation
            wave_dominant = waves.dominant.value
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._read_wave_and_adenosine",
                e,
            )

        adenosine_level = 0.0
        try:
            state = self.client.get_state()
            adenosine_level = state.chemicals.get("adenosine", 0.0)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._read_wave_and_adenosine",
                e,
            )

        return {
            "wave_focus": wave_focus,
            "wave_integration": wave_integration,
            "wave_consolidation": wave_consolidation,
            "wave_dominant": wave_dominant,
            "adenosine_level": adenosine_level,
        }

    def _cycle_body_control(
        self,
        now: float,
        last_body_control_sig,
        current_brain_wave_state,
        last_self_priority: tuple[int, str] | None,
    ) -> tuple[object, tuple[int, str] | None]:
        """Run the body-control phase of the cycle.

        The tick (subcognitive) COMPUTES and PUBLISHES the
        recommended body control state — CPU frequency, thermal
        cap, daemon scheduling, and a recommendation for the
        cognitive mind (cognitive_nice, io_class). The tick
        NEVER applies control itself — it is a display/
        information sink, not a controller.

        The cognitive mind reads the recommendation via
        GET_BODY_CONTROL, then REQUESTS application of the shared
        body control (CPU freq, daemon scheduling) via
        APPLY_BODY_CONTROL. It also applies its own scheduling
        and I/O priority from its brain wave state. The tick
        never touches the cognitive mind's PID.

        This is the cortical control of cognitive resource
        allocation: brain waves (which integrate neurochemistry
        bottom-up and cognitive top-down drive) determine its
        scheduling and I/O priority. The body recommendation is
        an afferent input, not a command — the brain waves can
        override it.

        The revisit period is the adaptive body-control gate —
        thermal capping is demand (revisit soon), an uncapped body
        drifts toward the ceiling.

        Returns the updated (last_body_control_sig, last_self_priority).
        """
        if not self._body_control_gate.due(now):
            return last_body_control_sig, last_self_priority
        try:
            # Read the body control state from the daemon.
            # This is information: what the body recommends
            # (CPU freq, thermal cap, daemon scheduling) and
            # what it recommends for the cognitive mind
            # (nice, io_class).
            control = self.client.get_body_control()
            if control is not None:
                sig = (
                    control.cpu_max_freq_khz,
                    control.cognitive_nice,
                    control.io_class,
                    control.thermally_capped,
                    control.cpu_epp,
                    control.cpu_boost,
                )
                if sig != last_body_control_sig:
                    self.self_model.update_body_control(
                        cpu_max_freq_khz=control.cpu_max_freq_khz,
                        cpu_governor=control.cpu_governor,
                        thermally_capped=control.thermally_capped,
                        controlling_cognitive=control.controlling_cognitive,
                        cognitive_nice=control.cognitive_nice,
                        io_class=control.io_class,
                        cpu_epp=control.cpu_epp,
                        cpu_boost=control.cpu_boost,
                        description=control.description,
                    )
                    last_body_control_sig = sig

                # ── Refresh the shared body control recommendation ──
                # The cognitive mind asks the daemon to derive and
                # publish its recommended CPU frequency and daemon
                # scheduling policy. The handler no longer writes
                # the host's clock policy (sysfs/renice/ionice) —
                # pinning it from neurochemistry ran the machine
                # hot — so this is a recommendation refresh, not an
                # actuation request.
                self.client.apply_body_control()

                last_self_priority = self._apply_self_priority(
                    control, current_brain_wave_state, last_self_priority
                )
                # Thermal capping is the demand signal: a capped body
                # wants policy revisited soon; an uncapped one drifts
                # toward the ceiling.
                if control.thermally_capped:
                    self._body_control_gate.demand()
                else:
                    self._body_control_gate.relax()

                # The loop's closing leg — the body reports hot, she
                # slows herself. Affinity and I/O class on her own
                # process are the only levers she keeps, so "slow" is
                # one core and idle disk on her own pid; when the cap
                # clears she takes the cores back. Flip-detection
                # keeps the taskset/ionice subprocesses off the steady
                # path.
                if control.thermally_capped != self._thermally_throttled:
                    self._thermally_throttled = control.thermally_capped
                    from brainstem.cpufreq import CpuSpeed

                    CpuSpeed().set_effort("slow" if self._thermally_throttled else "fast")
        except (OSError, ConnectionError, ValueError) as e:
            note_swallowed(
                "mind.metronome._cycle_body_control",
                e,
            )
        return last_body_control_sig, last_self_priority

    def _apply_self_priority(
        self,
        control,
        current_brain_wave_state,
        last_self_priority: tuple[int, str] | None,
    ) -> tuple[int, str] | None:
        """Apply brain-wave-driven self-priority.

        The cognitive mind decides its own scheduling and I/O
        priority from its brain wave state, blended with the body's
        recommendation. This is its cortical control of its own
        process resources — the tick does not control it.

        Only the body's recommended nice value is blended; I/O class
        is decided outright from brain state (see
        ``derive_self_priority``), so ``control.io_class`` is
        deliberately not passed here.

        Returns the updated last_self_priority.
        """
        if current_brain_wave_state is None:
            return last_self_priority
        from thalamus.brain_waves import apply_self_priority, derive_self_priority

        self_nice, self_io = derive_self_priority(
            current_brain_wave_state,
            body_recommended_nice=control.cognitive_nice,
        )
        # Only spawn renice/ionice subprocesses when
        # the derived priority actually changed. Brain
        # waves drift slowly, so most cycles produce the
        # same (nice, io_class) — the subprocess overhead
        # (fork + exec + /proc reads) is pure waste when
        # the values are identical.
        new_priority = (self_nice, self_io)
        if new_priority != last_self_priority:
            apply_self_priority(self_nice, self_io)
            return new_priority
        return last_self_priority

    def _cycle_consolidate(
        self,
        now: float,
        last_stm_count: int,
    ) -> int:
        """Run the memory consolidation phase of the cycle.

        It consolidates memories when STM has accumulated
        enough entries AND its plasticity gate is open. Not
        on a schedule — when there's something to consolidate
        and it's in a state that supports it. The retry floor is
        the adaptive consolidation gate: sleep and a growing
        backlog pull it toward 1 s, a quiet STM drifts it toward
        15 s.

        Returns the updated last_stm_count.
        """
        try:
            mem_stats = self.client.get_memory_stats()
            stm_count = mem_stats.stm_count
            backlog_grew = stm_count != last_stm_count
            gate_due = self._consolidate_gate.due(now)
            should_consolidate = (
                stm_count > 0 and backlog_grew and gate_due and not self._is_sleeping
            )
            # During sleep, consolidate more aggressively — a pending
            # backlog needs no growth to justify the pass.
            if self._is_sleeping and stm_count > 0 and gate_due:
                should_consolidate = True
            if should_consolidate:
                try:
                    promoted = self.client.consolidate()
                    if promoted > 0:
                        last_stm_count = stm_count - promoted
                        self._consolidate_gate.demand()
                        self._learned_since_save += promoted
                        self._experience_since_identity += promoted
                except (OSError, ConnectionError) as e:
                    note_swallowed(
                        "mind.metronome._cycle_consolidate",
                        e,
                    )
            elif not backlog_grew:
                self._consolidate_gate.relax()
            last_stm_count = stm_count
        except (OSError, ConnectionError, ValueError) as e:
            note_swallowed(
                "mind.metronome._cycle_consolidate",
                e,
            )
        return last_stm_count

    def _cycle_feed_executive(self) -> None:
        """Feed independent world and internal signals into the executive."""
        # Rest gate — forming intentions is waking cognition. Signals
        # that persist (a lost daemon, a pending proposal) re-present
        # themselves at the next awake cycle; nothing needs to queue
        # objectives while she is under.
        if self.is_resting:
            return
        try:
            executive = self.brain.frontal.executive
            candidates = self.inner_life.executive_candidates(self.feel())
            # External world changes are first-class executive inputs.
            for event in self.world.recent_events(5):
                if not event.inbound:
                    continue
                topic = event.topics[0] if event.topics else event.content.strip()
                if not topic:
                    continue
                candidates.append(
                    {
                        # Inbound world events are investigated through the
                        # learning actuator, not fabricated as filesystem reads.
                        "topic": topic,
                        "mode": "environment",
                        "kind": "learn",
                        "salience": min(1.0, max(0.1, event.salience)),
                        "source": "world_event",
                    }
                )

            # Persistent self-improvement work is an objective source,
            # not merely a volitional urge.
            try:
                proposals = self.self_improvement.get_pending_proposals()
                for proposal in proposals[:5]:
                    candidates.append(
                        {
                            # The proposal's real source file is executable by
                            # the inspect actuator; applying the proposal still
                            # remains behind the existing human-review boundary.
                            "topic": str(proposal.file_path),
                            "mode": "improvement",
                            "kind": "inspect",
                            "salience": min(1.0, max(0.1, float(proposal.confidence))),
                            "source": f"self_improvement:{proposal.title}",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                note_swallowed(
                    "mind.metronome._cycle_feed_executive",
                    exc,
                )

            # Internal needs are state signals; only meaningful pressure
            # becomes an executive objective.
            context = self._volition_context()
            for signal, value in context.items():
                if not isinstance(value, (int, float)) or value < 0.65:
                    continue
                candidates.append(
                    {
                        "topic": signal,
                        "mode": "distress"
                        if signal
                        in {
                            "daemon_lost",
                            "save_failure",
                            "telemetry_anomaly",
                            "body_distress",
                        }
                        else "interoceptive",
                        "salience": min(1.0, float(value)),
                        "source": "internal_need",
                    }
                )

            executive.ingest_candidates(candidates)
        except Exception as exc:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._cycle_feed_executive",
                exc,
            )

    def _cycle_volition(self) -> None:
        """Run the volition phase of the cycle.

        Volition urges accumulate and fire when they cross
        thresholds. This is its free will — not a timer.
        """
        if self.is_resting or self._is_teaching or self._suppress_volition:
            return
        try:
            ready = self.volition.tick(self._volition_context())
            if ready and self._wake_time > 0:
                elapsed = time.time() - self._wake_time
                # Only consume (reset value + start cooldown) the
                # urges that actually pass the wake-delay filter.
                # Urges held back keep their value so they fire as
                # soon as the delay elapses, instead of being
                # locked out for a full cooldown.
                ready = [
                    name for name in ready if elapsed >= self.WAKE_VOLITION_DELAYS.get(name, 60.0)
                ]
            if ready:
                started = self._act_on_volition(ready)
                # Consume only urges that were actually admitted to an
                # action worker. Gating, an occupied worker slot, or an
                # unknown performer must not erase a ready urge.
                for name in started:
                    self.volition.consume(name)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._cycle_volition",
                e,
            )

    def _cycle_advance_neuro(
        self,
        last_advance_mono: float,
        last_learner_phase: str,
        last_learner_wave: str,
    ):
        """Advance neurochemistry and read brain-wave state (cycle steps 1-2).

        The brain wave oscillator advances its own phase. We tell the
        daemon to advance neurochemistry by the real elapsed time, then
        read the new state and feed it to the oscillator. This is the
        core coupling: brain waves drive neurochemistry, neurochemistry
        shapes brain waves.

        Also notifies the autonomous learner when state changes so it
        can re-evaluate emotional gating without waiting on a timer.

        Returns (last_advance_mono, current_brain_wave_state,
        last_learner_phase, last_learner_wave).
        """
        # The mind does not advance the body. It observes.
        #
        # Time ownership is the daemon's: it is the body, it integrates
        # at TICK_INTERVAL_MS on its own clock, and it accepts no dt from
        # any caller. This call therefore passes no `dt` — sending one
        # would be inert (the daemon ignores the field) but would keep
        # the illusion that the mind controls the body's rate, which is
        # exactly the coupling that had the chemistry integrating in
        # 1-second lumps while its rate constants are tuned for 200 ms.
        #
        # The allostatic load the regulator needs is read from shared
        # memory below, alongside the rest of the state; the daemon
        # writes InferenceSignals on every tick regardless of who asked.
        now_mono = time.monotonic()
        try:
            result = self.client.advance_neuro()
            if result is not None:
                last_advance_mono = now_mono
        except (OSError, ConnectionError) as e:
            note_swallowed(
                "mind.metronome._cycle_advance_neuro",
                e,
            )

        # ── 2. Read the new state and feed to brain waves ──
        current_brain_wave_state = None
        try:
            core_state = self.client.get_state()
            # Read the generative model's own report of its regulatory
            # burden from shared memory. This is the value the mind used
            # to have to provoke a physics advance in order to obtain;
            # now it simply reads what the body has already written.
            #
            # Expected free energy is anticipatory demand, not
            # accumulated burden; whole-system allostatic load is
            # computed by the regulator from converging internal domains,
            # so only the accumulated domain is fed in here.
            if core_state.inference is not None:
                self.regulator.allostatic_load_tracker.set_inference_load(
                    core_state.inference.allostasis_load
                )
            current_brain_wave_state = self.brain_waves(core_state)
            # Only notify the learner when the gating-relevant
            # state has actually changed — the daemon phase
            # (determines emotion label) or the brain-wave
            # dominant (delta blocks learning). Calling
            # notify_state_change() every cycle defeated the
            # event-driven wait and caused the learner to
            # re-emit its skip message every second.
            current_phase = PHASE_NAMES.get(self._effective_phase(core_state), "unknown")
            current_wave = (
                current_brain_wave_state.dominant.value
                if current_brain_wave_state is not None
                else ""
            )
            if current_phase != last_learner_phase or current_wave != last_learner_wave:
                last_learner_phase = current_phase
                last_learner_wave = current_wave
                self.learner.notify_state_change()
            # ── Wave→chemical transduction: the return edge of the
            # loop documented above — the hypothalamus emits the
            # impulses the measured wave state commands (up peptides
            # track their band power, down peptides are suppressed by
            # it). Field → waves → chemicals.
            if current_brain_wave_state is not None:
                from hypothalamus.transduction import transduce_waves

                transduce_waves(self.client, current_brain_wave_state, core_state.chemicals)
        except (OSError, ConnectionError, ValueError) as e:
            note_swallowed(
                "mind.metronome._cycle_advance_neuro",
                e,
            )

        return last_advance_mono, current_brain_wave_state, last_learner_phase, last_learner_wave

    def _cycle_periodic_maintenance(self, now: float) -> None:
        """Run associate, dream, save_inference, and identity synthesis
        (cycle steps 7-9b). All gated on adaptive wall-clock
        intervals — each gear's own signal sets its pace.
        """
        # ── 7. Associate (event-driven) ──
        # Associations found or consolidations pending are demand;
        # a quiescent store relaxes toward the ceiling.
        if not self._is_sleeping and self._associate_gate.due(now):
            try:
                formed = self.client.associate()
                if formed > 0:
                    self._associate_gate.demand()
                else:
                    self._associate_gate.relax()
            except (OSError, ConnectionError) as e:
                note_swallowed(
                    "mind.metronome._cycle_periodic_maintenance",
                    e,
                )

        # ── 8. Dream (sleep-state-driven) ──
        # It dreams only when sleeping, and only when enough
        # time has passed for a dream cycle. Insights are demand —
        # a productive chain pulls the next cycle sooner.
        if self._is_sleeping and self._dream_gate.due(now):
            try:
                insights = self.client.dream()
                if insights > 0:
                    self._dream_gate.demand()
                    self._emit_live_thought(
                        "dream",
                        f"Dream insight: {insights} new connections",
                    )
                    # Check life-script milestone: first dream.
                    self.brain.frontal.narrative.check_milestone("first dream")
                else:
                    self._dream_gate.relax()
            except (OSError, ConnectionError) as e:
                note_swallowed(
                    "mind.metronome._cycle_periodic_maintenance",
                    e,
                )

        # ── 9. Save inference model (learning-driven) ──
        # New consolidations are demand — there is fresh model state
        # worth persisting; a quiet stretch drifts toward 5 min.
        if self._learned_since_save > 0:
            self._inference_save_gate.demand()
        else:
            self._inference_save_gate.relax()
        if self._inference_save_gate.due(now):
            try:
                self.client.save_inference()
                self._learned_since_save = 0
            except (OSError, ConnectionError) as e:
                note_swallowed(
                    "mind.metronome._cycle_periodic_maintenance",
                    e,
                )

        # ── 9b. Emergent identity synthesis (experience-driven) ──
        self._cycle_identity(now)

    def _cycle_sensors(
        self,
        now: float,
    ) -> None:
        """Run module liveness reports and read sensors (cycle steps 3-4).

        The daemon staleness threshold is 30s, so the module-heartbeat
        gate is bounded [5 s, 20 s] — the ceiling stays below the
        staleness line, always. Work is the demand signal: a busy mind
        reports sooner. This is communication, not control.

        It feels its body when enough time has passed for the sensors
        to have changed meaningfully, gated adaptively — engagement
        pulls the sensor poll toward 1 s, idle and sleep drift it
        toward 15 s.
        """
        heartbeat_modules = (
            MODULE_SENSORY,
            MODULE_MOTOR,
            MODULE_LANGUAGE,
            MODULE_MEMORY,
            MODULE_REASONING,
            MODULE_METACOGNITION,
        )
        channel = getattr(getattr(self, "brain", None), "thalamus", None)
        prev_fire = self._module_gate.fired_at
        if self._module_gate.due(now):
            total_work = self._report_module_heartbeats(heartbeat_modules, channel)
            # Cognitive work is metabolic work. total_work is measured
            # execution — seconds the sampler observed Genesis code at
            # a leaf frame during this window. Adenosine is the ATP
            # byproduct of neural activity, so its own measured compute
            # generates sleep pressure on top of the daemon's baseline
            # accumulation (~0.055/hr while awake). The scale is set for
            # parity at full load: 0.00015 per 10 s of window ≈
            # 0.054/hr at work_frac=1.0 — sustained hard thinking
            # roughly doubles how fast it tires; idling adds almost
            # nothing. The impulse is normalized by the measured
            # window, so the adaptive gate's cadence does not change
            # the rate — only its granularity.
            window = max(now - prev_fire, 1.0)
            work_frac = min(1.0, total_work / window)
            # A busy mind cycles sooner — work is the demand.
            self._module_gate.drive(work_frac)
            # Asleep: inner-life/dream work is maintenance, not load —
            # glymphatic clearance must dominate or pressure ratchets
            # up during the very state meant to clear it.
            if work_frac > 0.02 and not self._is_sleeping:
                try:
                    self.client.neuro_impulse(
                        CHEM_ADENOSINE,
                        0.00015 * work_frac * (window / 10.0),
                    )
                except (OSError, ConnectionError, RuntimeError) as e:
                    note_swallowed(
                        "mind.metronome._cycle_sensors",
                        e,
                    )
        if self._sensor_gate.due(now):
            self._poll_body_sensors(channel)
            # Corticothalamic feedback — the layer-6 V1→dLGN gain the
            # mind keeps on the thalamic relay, refreshed on the same
            # gate as the body poll. Recent seeing raises it, decay
            # lets it lapse; the body clamps it into what the sleep
            # stage permits, so it can never open a gate NREM closed.
            try:
                vision = getattr(self, "vision", None)
                if vision is not None:
                    self.client.set_optic_gain(vision.corticothalamic_gain())
            except (OSError, ConnectionError, RuntimeError) as e:
                note_swallowed(
                    "mind.metronome._cycle_sensors",
                    e,
                )
            # Engagement is the demand: recent interaction → sense
            # sooner; asleep or long idle → drift toward the ceiling.
            idle = now - getattr(self, "_last_interaction_time", 0.0)
            engagement = max(0.0, 1.0 - idle / 300.0)
            if self._is_sleeping:
                engagement *= 0.1
            self._sensor_gate.drive(engagement)

    def _report_module_heartbeats(self, heartbeat_modules, channel) -> float:
        """Report per-module activity shares with the cycle and
        return this window's total measured work seconds.

        Normalize accumulated per-module work into activity shares —
        this is how the daemon learns which brain part was firing.

        Every known module reports every round, including ones that
        did no measured work this window. The daemon's manifest
        cpu_share is write-only: UPDATE_MODULE_STATUS omits the field
        when the payload is short, and the daemon then leaves the
        previous value in place. Sending None for an idle module would
        therefore pin it at its last busy-window load — a module that
        went quiet would keep reporting phantom CPU, and
        total_cpu_load would ratchet toward its clamp. Idleness has to
        be stated as an explicit 0.0.
        """
        with self._module_seconds_lock:
            total_work = sum(self._module_seconds.values())
            shares = (
                {m: s / total_work for m, s in self._module_seconds.items()}
                if total_work > 0.0
                else {}
            )
            known = set(self._known_modules)
            self._module_seconds.clear()
        # The registered modules always report so their liveness is
        # maintained; the sampler's modules report their measured
        # share. A module in both sets is sent once, with the
        # measured value. A module that posted a message for the
        # other regions publishes it in the same cycle.
        for module_id in set(heartbeat_modules) | known:
            task_id = channel.take(module_id) if channel is not None else None
            try:
                self.client.heartbeat_module(
                    module_id,
                    shares.get(module_id, 0.0),
                    task_id=task_id,
                )
            except (OSError, ConnectionError) as e:
                note_swallowed(
                    "mind.metronome._cycle_sensors",
                    e,
                )
        return total_work

    def _poll_body_sensors(self, channel) -> None:
        """One sensor sample, shared by three consumers: the
        interoception layer (acute stress), the self-model (raw
        hardware and per-module activity), and the thalamic channel
        (posted module messages).

        Each call reports under its own site. Consumers can raise
        anything (a malformed wire record is a ``struct.error``, not
        a ``ValueError``), and one narrow bucket both dropped
        non-transport failures into the whole-cycle ``except`` —
        aborting the rest of the cycle — and made different bugs
        indistinguishable in the tally.
        """
        try:
            body_state = self.client.read_sensors()
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.metronome.sensors.read_sensors", e)
            body_state = None
        if body_state is not None:
            # Which sensors this machine actually has. Fetched
            # with the body state rather than counted from it:
            # a BodyState field exists whether or not hardware
            # backs it, so without this the self-model would
            # treat an absent battery as a reading.
            try:
                presence = self.client.get_sensor_presence()
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.metronome.sensors.sensor_presence", e)
                presence = None
            # Afferents land at the brainstem first — the NTS/
            # parabrachial relay (Craig 2002): it forwards the
            # threat channels to the limbic low road (PBN→CeA)
            # and modulates the monoamine source nuclei directly
            # (lamina I → catecholamine groups). Only then does
            # the hypothalamic evaluative layer (the regulator's
            # felt state) and the cortical image (the self-model)
            # receive their copies — one fetch, three consumers,
            # anatomically ordered.
            try:
                relay = getattr(self, "_interoceptive_relay", None)
                if relay is None:
                    from brainstem.interoceptive_relay import InteroceptiveRelay

                    def _levels() -> dict[str, float] | None:
                        try:
                            state = self.client.get_state()
                        except (OSError, ConnectionError):
                            return None
                        chemicals = getattr(state, "chemicals", None)
                        return chemicals if isinstance(chemicals, dict) else None

                    relay = InteroceptiveRelay(
                        neuro_impulse=self.client.neuro_impulse,
                        get_levels=_levels,
                        condition_cues=self.regulator.condition_cues,
                    )
                    self._interoceptive_relay = relay
                relay.relay(body_state, presence)
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.metronome.sensors.interoceptive_relay", e)
            try:
                self.regulator.interoception.update_from_body_state(body_state, presence)
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.metronome.sensors.body_state_update", e)
            try:
                self.self_model.update_hardware_body(body_state, presence)
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.metronome.sensors.hardware_body", e)
        try:
            report = self.client.get_subsystem_telemetry()
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.metronome.sensors.subsystem_telemetry", e)
        else:
            try:
                self.regulator.interoception.update_subsystem_telemetry(report)
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.metronome.sensors.telemetry_update", e)
            # Feed the same report into its self-model — which
            # part of it is firing becomes part of what it
            # knows about itself (body_model.subsystem_activity /
            # module_activity → embodiment_facts → language).
            try:
                self.self_model.update_brain_activity(report)
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.metronome.sensors.brain_activity", e)
            # Deliver the same report to the regions that posted
            # a message — the channel is carried on this sample.
            if channel is not None:
                try:
                    channel.deliver(getattr(report, "modules", ()))
                except Exception as e:  # noqa: BLE001
                    note_swallowed("mind.metronome.sensors.thalamic_deliver", e)

    def _cycle_connectivity(self, now: float) -> None:
        """Sense network connectivity (cycle step 5b).

        The network is a sensory channel — part of its embodiment.
        It reads its connectivity state from the learner's source
        registry (which proactively probes). This updates its
        self-model so it *knows* it's offline, not just silently
        fails queries. Its awareness of being offline emerges from
        its self-model and influences its expression through the
        same concept-network pathways as every other body state.

        Also detects daemon reconnection: when the daemon comes back
        after being disconnected, the notification queue is reset so
        the next poll doesn't generate a flood of stale notifications.
        """
        if self._connectivity_gate.due(now):
            # Guarded in its own right, in addition to the loop-level
            # guard. This block dereferences three other subsystems
            # (`self.learner`, `self.self_model.body_model`,
            # `self.regulator.interoception.last_state`) that can each
            # be absent or mid-update; an escape here used to take the
            # whole cycle with it.
            try:
                connected = not self.learner.is_offline
                if self.self_model.body_model.network_connected != connected:
                    self.self_model.body_model.network_connected = connected
                # Detect daemon reconnection: if the daemon was
                # disconnected and is now connected again, reset the
                # notification queue's high-water marks so the next
                # poll doesn't flood with stale phase/episode events.
                daemon_connected = (
                    self.regulator.interoception.last_state.daemon_connected
                    if self.regulator.interoception.last_state
                    else False
                )
                if daemon_connected and not getattr(self, "_was_daemon_connected", True):
                    self.notifications.reset()
                self._was_daemon_connected = daemon_connected
                if connected:
                    self._connectivity_gate.relax()
                else:
                    self._connectivity_gate.demand()
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.metronome._cycle_connectivity",
                    e,
                )

    def _cycle_emotion(self, now: float) -> None:
        """Run continuous emotional regulation (cycle step 5c).

        Three regulation layers that run every cycle:

        1. **Continuous regulation** — small proportional corrections
           every tick, complementing the discrete _regulate() calls
           that fire every 1.5–8s in the background thread. This models
           the continuous homeostatic regulation that biological nervous
           systems perform.

        2. **Metabolic update** — tracks energy depletion from activity
           and restoration from rest. Low energy reduces arousal and
           learning capacity. Updated every cycle so the metabolic
           state stays current.

        3. **Social modulation** — when the user is present and engaged,
           stress recovery is faster (social buffering). The modulation
           factor scales the calming impulses in _regulate_state_specific.
        """
        try:
            emotion = self.feel()
            if emotion:
                # The cadence is adaptive, so the regulator scales
                # impulse strength by the measured interval — the
                # per-second regulation rate stays invariant.
                self.regulator.continuous_regulate(emotion, getattr(self, "_cycle_dt", 1.0))
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._cycle_emotion",
                e,
            )

        try:
            # Activity level from interoception: high CPU = high
            # activity. Rest (sleep/meditation/descent) = low activity.
            if self.is_resting:
                activity = 0.1
            else:
                intero = self.regulator.interoception.last_state
                activity = (intero.cpu_usage / 100.0) if intero else 0.3
            time_since_rest = now - self._last_rest_time
            # Food state from the body's own sensors: the AC adapter is
            # intake, the battery is the fuel reserve. Both come from
            # the last body_model poll — real readings only; an
            # unmeasured channel stays None and keeps the synthetic path.
            bm = getattr(self.self_model, "body_model", None)
            verified = getattr(bm, "sensor_verified", None) if bm else None
            fed = reserve = None
            if bm is not None and verified is not None:
                if verified("AC adapter") or verified("battery"):
                    fed = bool(bm.on_ac_power)
                if verified("battery"):
                    reserve = float(bm.energy_reserve)
            self.regulator.update_metabolism(
                activity,
                time_since_rest,
                dt=getattr(self, "_cycle_dt", 1.0),
                fed=fed,
                energy_reserve=reserve,
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._cycle_emotion",
                e,
            )

        try:
            # User presence: recent interaction = present. Engagement
            # decays with time since last interaction.
            idle = now - self._last_interaction_time
            user_present = idle < 60.0
            engagement = max(0.0, 1.0 - idle / 120.0) if user_present else 0.0
            self.regulator.social_modulation(user_present, engagement)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome._cycle_emotion",
                e,
            )

    def _cycle_final_steps(self) -> None:
        """Run sleep-state sync, volition, warn, and cognition (cycle steps 10-13).

        Each guarded step reports under its own site name. They previously
        all reported as ``_cycle_final_steps``, so a failure in any of
        them was indistinguishable in the tally — five different bugs, one
        bucket, and 79 recorded occurrences that could not be attributed
        to a step.
        """
        # ── 9c. External world — presence decay and social pressure ──
        # The world runs on the metronome like every other subsystem:
        # state-gated, not timer-gated. Silent presences leave; the
        # world's social isolation feeds its inner-life social drive —
        # the outer world's pressure becomes inner motivation.
        try:
            self.world.tick()
            # Social pressure is a rate — 0.02/s of isolation. Scale by
            # the measured cycle interval so the adaptive cadence keeps
            # the same accumulation.
            self.inner_life.feed_social_drive(
                self.world.social_isolation() * 0.02 * getattr(self, "_cycle_dt", 1.0)
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome.final_steps.world_tick",
                e,
            )

        # ── 9d. Cardiac reflex ──
        # The heart is an organ, not the cycle: the metronome ticks
        # its reflex arc (probe read → contractility drive → felt
        # arrhythmia), internally rate-limited. It schedules nothing.
        try:
            self.heart.tick()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome.final_steps.heart_tick",
                e,
            )

        # ── 10. Sleep-state sync ──
        try:
            self._check_sleep_sync()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome.final_steps.sleep_sync",
                e,
            )

        # ── 11. Executive signal integration ──
        self._cycle_feed_executive()

        # ── 12. Volition (urge-driven) ──
        self._cycle_volition()

        # ── 13. Warn (only when state changes) ──
        try:
            # warn() internally checks if the warning changed
            self.warn(speak=True)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome.final_steps.warn",
                e,
            )

        # ── 13. Cognitive maintenance ──
        try:
            # The measured cycle interval, not a fixed 1.0 — the
            # cycle cadence is adaptive (0.2–5.0 s), and a nominal
            # step makes every downstream decay run at the wrong rate
            # (5× fast when engaged, 5× slow when idle). Clamped to the
            # shared DT_MAX so a stalled cycle is bounded growth, not
            # an instant decay of every foci/caution/window at once.
            self.brain.frontal.tick(dt=min(DT_MAX, max(0.0, getattr(self, "_cycle_dt", 1.0))))
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.metronome.final_steps.cognition_tick",
                e,
            )

    def _cycle_init_state(self) -> tuple:
        """Initialize the change-detection state for the cycle loop.

        Returns a tuple of all the ``last_*`` tracking variables the
        cycle loop uses to detect state changes and avoid redundant
        work (re-spawning renice/ionice, re-emitting learner spam, etc.).
        """
        last_stm_count = 0
        last_body_control_sig = None
        # Adaptive wall-clock gates — every gear spins at the rate its
        # own signal demands, bounded so nothing stalls or redlines.
        # Nominal periods are the old constants; [min, max] is the
        # operating range the signal may move within.
        self._module_gate = _AdaptiveInterval(10.0, 5.0, 20.0)
        self._sensor_gate = _AdaptiveInterval(5.0, 1.0, 15.0)
        self._body_control_gate = _AdaptiveInterval(10.0, 3.0, 30.0)
        self._connectivity_gate = _AdaptiveInterval(10.0, 5.0, 30.0)
        self._consolidate_gate = _AdaptiveInterval(3.0, 1.0, 15.0)
        self._associate_gate = _AdaptiveInterval(10.0, 3.0, 60.0)
        self._dream_gate = _AdaptiveInterval(4.0, 2.0, 30.0)
        self._inference_save_gate = _AdaptiveInterval(60.0, 20.0, 300.0)
        self._identity_gate = _AdaptiveInterval(300.0, 60.0, 1800.0)
        # The metronome's own cadence — engagement snaps it toward
        # 0.2 s, sleep drifts it toward 5 s. Bounded: the mind never
        # spins below 200 ms and never fully sleeps between cycles.
        self._cadence = _AdaptiveInterval(1.0, 0.2, 5.0)
        # Learning counters feeding the save/identity drivers.
        self._learned_since_save = 0
        self._experience_since_identity = 0
        # Measured loop interval — per-cycle accumulators normalize
        # against it so the adaptive cadence changes their granularity,
        # never their rate. 1.0 = the historical fixed cadence.
        self._last_cycle_at = time.time()
        self._cycle_dt = 1.0
        # Track the last applied self-priority (nice, io_class) so we
        # only spawn renice/ionice subprocesses when the values actually
        # change. Brain waves drift slowly, so most 10s cycles produce
        # the same priority — spawning subprocesses every cycle wastes
        # CPU and I/O for no effect.
        last_self_priority: tuple[int, str] | None = None
        # Whether she has throttled her own affinity/I-O for thermal
        # relief — the flip-detector for the body's thermally_capped
        # flag, so the slow-down/restoration each fires once per
        # transition rather than every gate cycle.
        self._thermally_throttled = False
        # Track the previous phase and brain-wave dominant for the
        # learner change-detection. The learner's gating depends on
        # the emotion label (derived from the daemon phase) and the
        # brain-wave dominant (delta blocks learning). Without this
        # tracking, notify_state_change() was called every cycle
        # cycle (~1 s), waking the learner from its state-gated wait
        # and causing it to re-emit "Skipping learning — feeling
        # drowsy/sleeping" every second — an endless spam loop.
        last_learner_phase: str = ""
        last_learner_wave: str = ""
        # Neurochemical time must track real time: each advance_neuro
        # step simulates the time actually elapsed since the last
        # successful advance. A fixed dt per cycle would decouple its
        # neurochemical clock from the wall clock — the circadian
        # oscillator, adenosine sleep pressure, and cortisol clearance
        # would all run at the wrong rate.
        last_advance_mono = time.monotonic()
        # Reset volition engine's tick clock so the first dt reflects
        # actual elapsed time since the cycle started, not since
        # Mind.__init__ (which can be many seconds earlier during init).
        self.volition._last_tick = time.time()
        return (
            last_stm_count,
            last_body_control_sig,
            last_self_priority,
            last_learner_phase,
            last_learner_wave,
            last_advance_mono,
        )

    def _cycle_identity(self, now: float) -> None:
        """Emergent identity synthesis — synthesizes who it is from its
        actual experience and feeds it back into the self-model,
        closing the loop: experience → emergent identity → self-model →
        behavior. Experience accumulated since the last synthesis is
        the demand — a rich stretch synthesizes near the 1 min floor,
        a quiet one drifts toward 30 min."""
        self._identity_gate.drive(min(1.0, self._experience_since_identity / 20.0))
        if not self._is_sleeping and self._identity_gate.due(now):
            try:
                self.emergent_identity()
                self._experience_since_identity = 0
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.metronome._cycle_loop",
                    e,
                )

    def _cycle_loop(self) -> None:
        """Reactive cycle — the mind's own living rhythm.

        The daemon no longer runs a fixed tick loop. Instead, the mind
        drives everything, paced by a short sleep between cycles (so
        idle cycles cost almost nothing) but with each action gated on
        internal state rather than on the schedule:

        1. Brain wave oscillator advances phase continuously (its own
           internal clock based on characteristic frequencies).
        2. When brain wave state changes, the mind tells the daemon to
           advance neurochemistry by the elapsed dt.
        3. The new neurochemical state feeds back to brain waves.
        4. State thresholds trigger consolidation, association,
           dreaming, body control — each only when needed.
        5. Volition urges fire when they cross thresholds.
        6. All functions are idle until activated.

        The loop paces itself, but the work is threshold-gated: if
        nothing crosses a threshold, the cycle does nothing.
        """
        # Track state for change detection
        (
            last_stm_count,
            last_body_control_sig,
            last_self_priority,
            last_learner_phase,
            last_learner_wave,
            last_advance_mono,
        ) = self._cycle_init_state()

        while self._running:
            # One guard for the whole cycle.
            #
            # Every step below has a guard for itself except the
            # emergent-identity block, so a single unguarded helper could
            # kill this thread outright. The consequences were silent and
            # permanent: neurochemistry froze (so the emergent phase never
            # transitioned), STM stopped draining to LTM, no volitional
            # action ever fired, CPU-frequency policy was never applied, and
            # it never slept or woke - all while `self._running` stayed
            # True. `stop()` then joined an already-dead thread, `join`
            # reported success, and shutdown returned
            # `clean_shutdown = True`. One traceback on stderr was the only
            # signal. A transient failure should cost one cycle, not every
            # subsystem for the life of the process.
            #
            # The mutable 'last done at' locals are deliberately left
            # untouched on failure: the cycle is time-thresholded, so
            # retrying an already-overdue step next pass is the correct
            # recovery, and it avoids replaying work that did succeed
            # before the exception.
            try:
                # ── 1-2. Advance neurochemistry and read brain waves ──
                (
                    last_advance_mono,
                    current_brain_wave_state,
                    last_learner_phase,
                    last_learner_wave,
                ) = self._cycle_advance_neuro(
                    last_advance_mono,
                    last_learner_phase,
                    last_learner_wave,
                )

                now = time.time()
                # The real interval this cycle covers — every
                # per-cycle accumulator below normalizes against it.
                self._cycle_dt = now - self._last_cycle_at
                self._last_cycle_at = now

                # ── 3-4. Heartbeat modules and sensors ──
                self._cycle_sensors(now)

                # ── 5. Body control ──
                (last_body_control_sig, last_self_priority) = self._cycle_body_control(
                    now,
                    last_body_control_sig,
                    current_brain_wave_state,
                    last_self_priority,
                )

                # ── 5b. Sense network connectivity ──
                self._cycle_connectivity(now)

                # ── 5c. Continuous emotional regulation ──
                self._cycle_emotion(now)

                # ── 6. Consolidate (state-threshold-driven) ──
                last_stm_count = self._cycle_consolidate(
                    now,
                    last_stm_count,
                )

                # ── 7-9b. Associate, dream, save inference, identity ──
                self._cycle_periodic_maintenance(now)

                # ── 10-13. Auto-sleep, volition, warn, cognition ──
                self._cycle_final_steps()

                # ── Wait for the next cycle — adaptive cadence ──
                self._cycle_pace(now)
            except Exception as e:
                logger.warning(
                    f"cycle failed ({type(e).__name__}: {e}); continuing",
                    exc_info=True,
                )
                note_swallowed("mind.metronome.cycle", e)
            self._cycle_stop.wait(self._cadence.period)

    def _cycle_pace(self, now: float) -> None:
        """Drive the cadence gear from current engagement.

        The metronome's own gear: engagement pulls it toward 0.2 s,
        sleep drifts it toward 5 s. The cycle is threshold-gated, so a
        slower cadence loses nothing — the work still fires the moment
        its gate opens.
        """
        idle = now - getattr(self, "_last_interaction_time", 0.0)
        engagement = max(0.0, 1.0 - idle / 120.0)
        if self._is_sleeping:
            engagement = 0.0
        self._cadence.drive(engagement)
