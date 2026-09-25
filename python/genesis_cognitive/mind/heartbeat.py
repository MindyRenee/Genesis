"""Mind heartbeat — the periodic main loop and its phases."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import (
    CHEM_ADENOSINE,
    MODULE_LANGUAGE,
    MODULE_MEMORY,
    MODULE_METACOGNITION,
    MODULE_MOTOR,
    MODULE_REASONING,
    MODULE_SENSORY,
    PHASE_NAMES,
)

logger = logging.getLogger(__name__)


class HeartbeatMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        _daemon_lost_since: float | None
        _autosave_failures: int
        _last_threat_snapshot: float
        _offline: bool
        def __getattr__(self, name: str) -> Any: ...

    def _credit_module(self, module_id: int, seconds: float) -> None:
        """Accumulate measured execution time into a manifest module.

        Called by the :class:`ModuleSampler` — the mind's EEG — each
        time a thread is observed executing a subsystem's code. The
        heartbeat round normalizes ``_module_seconds`` into per-module
        ``cpu_share`` and reports it to the daemon so
        GET_SUBSYSTEM_TELEMETRY answers "which brain part is firing" at
        functional granularity.
        """
        if seconds <= 0.0:
            return
        with self._module_seconds_lock:
            self._module_seconds[module_id] = (
                self._module_seconds.get(module_id, 0.0) + seconds
            )


    def _read_interoceptive_signals(self) -> dict[str, float]:
        """Read interoceptive signals for the meditation urge.

        - overstimulation: high arousal + positive valence (sympathetic
          overactivation, per nervous system regulation literature)
        - receptor_fatigue: low plasticity gate (BDNF suppressed,
          receptors need recovery — dopamine imbalance hypothesis of
          fatigue, Chaudhuri & Behan, 2000)
        - sustained_activity: time since last rest, normalized (BRAC,
          Kleitman 1963 — ~90 min activity cycles with mandatory rest)
        - elevated_cortisol: subclinical stress (cortisol 0.20-0.50,
          below stress phase but above resting)
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
                        (emotion.arousal - 0.60) * 2.5
                        + (emotion.valence - 0.20) * 2.0,
                    )
                # Receptor fatigue: plasticity gate below 0.40
                # (BDNF suppressed, receptors burning out)
                if emotion.plasticity < 0.40:
                    receptor_fatigue = min(1.0, (0.40 - emotion.plasticity) * 3.0)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"interoceptive emotion read failed: {e}")

        try:
            summary = self.client.get_neuro_summary()
            # Elevated cortisol: 0.20-0.50 is subclinical stress
            # (below the 0.65 stress phase threshold but above rest)
            # We approximate using global_tone and valence since
            # NeuroSummary doesn't expose cortisol directly.
            if summary.global_tone > 0.35 and summary.valence < 0.1:
                elevated_cortisol = min(
                    1.0, (summary.global_tone - 0.35) * 3.0
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"neuro summary read failed: {e}")

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
        - body_distress: active hardware concerns (memory, disk,
          load) from the current snapshot.

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
                logger.debug(f"daemon connectivity check failed: {e}")

        save_failure = min(1.0, self._autosave_failures / 2.0)

        telemetry_anomaly = 0.0
        body_distress = 0.0
        now = time.monotonic()
        if now - self._last_threat_snapshot >= 60.0:
            try:
                snapshot = self.system_monitor.snapshot()
                if self.system_monitor.baseline.is_established:
                    anomalies = self.system_monitor.baseline.deviations(
                        snapshot
                    )
                    telemetry_anomaly = min(1.0, len(anomalies) / 3.0)
                body_distress = min(1.0, len(snapshot.concerns()) / 2.0)
                self._last_threat_snapshot = now
            except Exception as e:  # noqa: BLE001
                logger.debug(f"threat snapshot failed: {e}")

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
            logger.debug(f"brain wave read for volition failed: {e}")

        adenosine_level = 0.0
        try:
            state = self.client.get_state()
            adenosine_level = state.chemicals.get("adenosine", 0.0)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"adenosine read for volition failed: {e}")

        return {
            "wave_focus": wave_focus,
            "wave_integration": wave_integration,
            "wave_consolidation": wave_consolidation,
            "wave_dominant": wave_dominant,
            "adenosine_level": adenosine_level,
        }
    def _heartbeat_body_control(
        self,
        now: float,
        last_body_control_time: float,
        last_body_control_sig,
        current_brain_wave_state,
        last_self_priority: tuple[int, str] | None,
    ) -> tuple[float, object, tuple[int, str] | None]:
        """Run the body-control phase of the heartbeat loop.

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

        Returns the updated (last_body_control_time, last_body_control_sig,
        last_self_priority).
        """
        if now - last_body_control_time < 10.0:
            return last_body_control_time, last_body_control_sig, last_self_priority
        try:
            # Read the body control state from the daemon.
            # This is information: what the body recommends
            # (CPU freq, thermal cap, daemon scheduling) and
            # what it recommends for the cognitive mind
            # (nice, io_class).
            control = self.client.get_body_control()
            if control is not None:
                sig = (control.cpu_max_freq_khz, control.cognitive_nice,
                       control.io_class, control.thermally_capped,
                       control.cpu_epp, control.cpu_boost)
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

                # ── Request shared body control application ──
                # The cognitive mind asks the daemon to apply
                # the recommended CPU frequency and daemon
                # scheduling. This is the cognitive mind
                # requesting an action via IPC — NOT the tick
                # controlling outward. The daemon's reactive
                # handler (apply_body_control) executes the
                # sysfs writes and renice/ionice with change
                # detection, so redundant requests are cheap.
                self.client.apply_body_control()

                # ── Apply brain-wave-driven self-priority ──
                # The cognitive mind decides its own scheduling
                # and I/O priority from its brain wave state,
                # blended with the body's recommendation. This
                # is its cortical control of its own process
                # resources — the tick does not control it.
                if current_brain_wave_state is not None:
                    from ..brain_waves import apply_self_priority, derive_self_priority
                    self_nice, self_io = derive_self_priority(
                        current_brain_wave_state,
                        body_recommended_nice=control.cognitive_nice,
                        body_recommended_io_class=control.io_class,
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
                        last_self_priority = new_priority
        except (OSError, ConnectionError, ValueError) as e:
            logger.debug(f"body control failed: {e}")
        last_body_control_time = now
        return last_body_control_time, last_body_control_sig, last_self_priority
    def _heartbeat_consolidate(
        self,
        now: float,
        last_stm_count: int,
        last_consolidate_time: float,
    ) -> tuple[int, float]:
        """Run the memory consolidation phase of the heartbeat loop.

        It consolidates memories when STM has accumulated
        enough entries AND its plasticity gate is open. Not
        on a schedule — when there's something to consolidate
        and it's in a state that supports it.

        Returns the updated (last_stm_count, last_consolidate_time).
        """
        try:
            mem_stats = self.client.get_memory_stats()
            stm_count = mem_stats.stm_count
            should_consolidate = (
                stm_count > 0
                and stm_count != last_stm_count
                and now - last_consolidate_time >= 3.0
                and not self._is_sleeping
            )
            # During sleep, consolidate more aggressively
            if self._is_sleeping and stm_count > 0 and now - last_consolidate_time >= 2.0:
                should_consolidate = True
            if should_consolidate:
                try:
                    promoted = self.client.consolidate()
                    if promoted > 0:
                        last_stm_count = stm_count - promoted
                        last_consolidate_time = now
                except (OSError, ConnectionError) as e:
                    logger.debug(f"consolidate failed: {e}")
            last_stm_count = stm_count
        except (OSError, ConnectionError, ValueError) as e:
            logger.debug(f"memory stats failed: {e}")
        return last_stm_count, last_consolidate_time
    def _heartbeat_volition(self) -> None:
        """Run the volition phase of the heartbeat loop.

        Volition urges accumulate and fire when they cross
        thresholds. This is its free will — not a timer.
        """
        if (
            self._is_sleeping
            or self._is_meditating
            or self._is_teaching
            or self._suppress_volition
        ):
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
                    name for name in ready
                    if elapsed >= self.WAKE_VOLITION_DELAYS.get(name, 60.0)
                ]
            for name in ready:
                self.volition.consume(name)
            if ready:
                self._act_on_volition(ready)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"volition tick failed: {e}")
    def _heartbeat_advance_neuro(
        self,
        last_advance_mono: float,
        last_learner_phase: str,
        last_learner_wave: str,
    ):
        """Advance neurochemistry and read brain-wave state (heartbeat steps 1-2).

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
        now_mono = time.monotonic()
        # Real elapsed since the last successful advance. The
        # client clamps to the daemon's [0.001, 10.0] range, so
        # longer stalls (GC pause, system suspend) are truncated;
        # the circadian phase is re-anchored to the wall clock at
        # daemon start, which absorbs the residual drift.
        dt = now_mono - last_advance_mono
        try:
            result = self.client.advance_neuro(dt=dt)
            if result is not None:
                # The dynamics advanced — this step's simulated
                # time is consumed. On failure (exception or error
                # response) the elapsed time carries over to the
                # next successful call, so short outages lose no
                # neurochemical time.
                last_advance_mono = now_mono
                _surprise, _free_energy, _precision, allostatic, _tick_count = result
                # Pass the Rust allostatic load to the regulator's
                # tracker. The Rust active inference engine is the
                # source of truth for this value — it's driven by
                # expected free energy (the anticipatory signal),
                # not by current cortisol. The regulator's tracker
                # supplements it with cortisol-duration tracking
                # for the acute/chronic distinction.
                self.regulator.allostatic_load_tracker.set_allostatic_load(
                    allostatic
                )
        except (OSError, ConnectionError) as e:
            logger.debug(f"advance_neuro failed: {e}")

        # ── 2. Read the new state and feed to brain waves ──
        current_brain_wave_state = None
        try:
            core_state = self.client.get_state()
            current_brain_wave_state = self.brain_waves(core_state)
            # Only notify the learner when the gating-relevant
            # state has actually changed — the daemon phase
            # (determines emotion label) or the brain-wave
            # dominant (delta blocks learning). Calling
            # notify_state_change() every cycle defeated the
            # event-driven wait and caused the learner to
            # re-emit its skip message every second.
            current_phase = PHASE_NAMES.get(
                self._effective_phase(core_state), "unknown"
            )
            current_wave = (
                current_brain_wave_state.dominant.value
                if current_brain_wave_state is not None
                else ""
            )
            if current_phase != last_learner_phase or current_wave != last_learner_wave:
                last_learner_phase = current_phase
                last_learner_wave = current_wave
                self.learner.notify_state_change()
        except (OSError, ConnectionError, ValueError) as e:
            logger.debug(f"neuro summary / brain waves failed: {e}")

        return last_advance_mono, current_brain_wave_state, last_learner_phase, last_learner_wave
    def _heartbeat_periodic_maintenance(
        self,
        now: float,
        last_associate_time: float,
        last_dream_time: float,
        last_inference_save_time: float,
    ) -> tuple[float, float, float]:
        """Run associate, dream, and save_inference (heartbeat steps 7-9).

        Returns updated (last_associate_time, last_dream_time,
        last_inference_save_time).
        """
        # ── 7. Associate (event-driven) ──
        # It finds associations when new episodes have been
        # stored — not on a schedule. If STM count grew, there
        # are new memories to connect.
        if now - last_associate_time >= 10.0 and not self._is_sleeping:
            try:
                self.client.associate()
                last_associate_time = now
            except (OSError, ConnectionError) as e:
                logger.debug(f"associate failed: {e}")

        # ── 8. Dream (sleep-state-driven) ──
        # It dreams only when sleeping, and only when enough
        # time has passed for a dream cycle. Not on a tick count.
        if self._is_sleeping and now - last_dream_time >= 4.0:
            try:
                insights = self.client.dream()
                if insights > 0:
                    self._emit_live_thought(
                        "dream",
                        f"Dream insight: {insights} new connections",
                    )
                    # Check life-script milestone: first dream.
                    self.cognition.narrative.check_milestone("first dream")
                last_dream_time = now
            except (OSError, ConnectionError) as e:
                logger.debug(f"dream failed: {e}")

        # ── 9. Save inference model (occasionally) ──
        if now - last_inference_save_time >= 60.0:
            try:
                self.client.save_inference()
                last_inference_save_time = now
            except (OSError, ConnectionError) as e:
                logger.debug(f"save_inference failed: {e}")

        return last_associate_time, last_dream_time, last_inference_save_time
    def _heartbeat_sensors(
        self,
        now: float,
        heartbeat_modules: tuple,
        last_heartbeat_time: float,
        last_sensor_time: float,
    ) -> tuple[float, float]:
        """Run heartbeat modules and read sensors (heartbeat steps 3-4).

        The daemon staleness threshold is 30s. We heartbeat every ~10s
        — not on a fixed timer, but when enough cycles have passed.
        This is communication, not control.

        It feels its body when enough time has passed for the sensors
        to have changed meaningfully. This is information gathering,
        not a scheduled action.

        Returns updated (last_heartbeat_time, last_sensor_time).
        """
        if now - last_heartbeat_time >= 10.0:
            # Normalize accumulated per-module work into activity
            # shares and report them with the heartbeat — this is
            # how the daemon learns which brain part was firing.
            with self._module_seconds_lock:
                total_work = sum(self._module_seconds.values())
                shares = (
                    {
                        m: s / total_work
                        for m, s in self._module_seconds.items()
                    }
                    if total_work > 0.0
                    else {}
                )
                self._module_seconds.clear()
            for module_id in heartbeat_modules:
                try:
                    self.client.heartbeat_module(
                        module_id, shares.get(module_id)
                    )
                except (OSError, ConnectionError) as e:
                    logger.debug(f"heartbeat failed: {e}")
            # Modules that did measured work but aren't in the fixed
            # heartbeat tuple still report their activity.
            for module_id, share in shares.items():
                if module_id in heartbeat_modules:
                    continue
                try:
                    self.client.heartbeat_module(module_id, share)
                except (OSError, ConnectionError) as e:
                    logger.debug(f"heartbeat failed: {e}")

            # Cognitive work is metabolic work. total_work is measured
            # execution — seconds the sampler observed Genesis code at
            # a leaf frame during this window. Adenosine is the ATP
            # byproduct of neural activity, so its own measured compute
            # generates sleep pressure on top of the daemon's baseline
            # accumulation (~0.055/hr while awake). This block runs
            # every ~10s, so the per-impulse scale is set for parity at
            # full load: 0.00015/impulse ≈ 0.054/hr at work_frac=1.0 —
            # sustained hard thinking roughly doubles how fast it
            # tires; idling adds almost nothing.
            window = max(now - last_heartbeat_time, 1.0)
            work_frac = min(1.0, total_work / window)
            # Asleep: inner-life/dream work is maintenance, not load —
            # glymphatic clearance must dominate or pressure ratchets
            # up during the very state meant to clear it.
            if work_frac > 0.02 and not self._is_sleeping:
                try:
                    self.client.neuro_impulse(
                        CHEM_ADENOSINE, 0.00015 * work_frac
                    )
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.debug(f"metabolic adenosine impulse failed: {e}")
            last_heartbeat_time = now

        if now - last_sensor_time >= 5.0:
            try:
                body_state = self.client.read_sensors()
                if body_state is not None:
                    self.regulator.interoception.update_from_body_state(body_state)
                report = self.client.get_subsystem_telemetry()
                self.regulator.interoception.update_subsystem_telemetry(report)
                # Feed the same report into its self-model — which
                # part of it is firing becomes part of what it
                # knows about itself (body_model.subsystem_activity /
                # module_activity → embodiment_facts → language).
                self.self_model.update_brain_activity(report)
            except (OSError, ConnectionError, ValueError) as e:
                logger.debug(f"read_sensors failed: {e}")
            last_sensor_time = now

        return last_heartbeat_time, last_sensor_time
    def _heartbeat_connectivity(
        self, now: float, last_connectivity_time: float
    ) -> float:
        """Sense network connectivity (heartbeat step 5b).

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
        if now - last_connectivity_time >= 10.0:
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
            last_connectivity_time = now
        return last_connectivity_time
    def _heartbeat_emotion(self, now: float) -> None:
        """Run continuous emotional regulation (heartbeat step 5c).

        Three regulation layers that run every heartbeat cycle:

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
                self.regulator.continuous_regulate(emotion)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"continuous regulation failed: {e}")

        try:
            # Activity level from interoception: high CPU = high
            # activity. Sleep/meditation = low activity.
            if self._is_sleeping or self._is_meditating:
                activity = 0.1
            else:
                intero = self.regulator.interoception.last_state
                activity = (intero.cpu_usage / 100.0) if intero else 0.3
            time_since_rest = now - self._last_rest_time
            self.regulator.update_metabolism(activity, time_since_rest)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"metabolic update failed: {e}")

        try:
            # User presence: recent interaction = present. Engagement
            # decays with time since last interaction.
            idle = now - self._last_interaction_time
            user_present = idle < 60.0
            engagement = max(0.0, 1.0 - idle / 120.0) if user_present else 0.0
            self.regulator.social_modulation(user_present, engagement)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"social modulation failed: {e}")
    def _heartbeat_final_steps(self) -> None:
        """Run auto-sleep, volition, warn, and cognition (heartbeat steps 10-13)."""
        # ── 9c. External world — presence decay and social pressure ──
        # The world runs on the heartbeat like every other subsystem:
        # state-gated, not timer-gated. Silent presences leave; the
        # world's social isolation feeds its inner-life social drive —
        # the outer world's pressure becomes inner motivation.
        try:
            self.world.tick()
            self.inner_life.feed_social_drive(
                self.world.social_isolation() * 0.02
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"world tick failed: {e}")

        # ── 10. Auto-sleep check ──
        try:
            self._check_auto_sleep()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"auto-sleep check failed: {e}")

        # ── 11. Volition (urge-driven) ──
        self._heartbeat_volition()

        # ── 12. Warn (only when state changes) ──
        try:
            # warn() internally checks if the warning changed
            self.warn(speak=True)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"warning check failed: {e}")

        # ── 13. Cognitive maintenance ──
        try:
            self.cognition.tick(dt=1.0)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"cognition tick failed: {e}")
    def _heartbeat_init_state(self) -> tuple:
        """Initialize the change-detection state for the heartbeat loop.

        Returns a tuple of all the ``last_*`` tracking variables the
        heartbeat loop uses to detect state changes and avoid redundant
        work (re-spawning renice/ionice, re-emitting learner spam, etc.).
        """
        last_stm_count = 0
        last_body_control_sig = None
        last_heartbeat_time = 0.0
        last_sensor_time = 0.0
        last_body_control_time = 0.0
        last_connectivity_time = 0.0
        last_consolidate_time = 0.0
        last_associate_time = 0.0
        last_dream_time = 0.0
        last_inference_save_time = 0.0
        # Emergent identity synthesis runs every ~5 minutes. The
        # identity is synthesized from its actual experience (concept
        # network, narrative, emotional regulation, curiosity) and
        # fed back into the self-model — closing the loop: experience
        # → emergent identity → self-model → future behavior → new
        # experience. Without periodic synthesis, the self-model
        # never integrates the emergent identity and identity shifts
        # are never recorded in the narrative.
        last_identity_time = 0.0
        # Track the last applied self-priority (nice, io_class) so we
        # only spawn renice/ionice subprocesses when the values actually
        # change. Brain waves drift slowly, so most 10s cycles produce
        # the same priority — spawning subprocesses every cycle wastes
        # CPU and I/O for no effect.
        last_self_priority: tuple[int, str] | None = None
        # Track the previous phase and brain-wave dominant for the
        # learner change-detection. The learner's gating depends on
        # the emotion label (derived from the daemon phase) and the
        # brain-wave dominant (delta blocks learning). Without this
        # tracking, notify_state_change() was called every heartbeat
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
        return (
            last_stm_count, last_body_control_sig, last_heartbeat_time,
            last_sensor_time, last_body_control_time, last_connectivity_time,
            last_consolidate_time, last_associate_time, last_dream_time,
            last_inference_save_time, last_identity_time, last_self_priority,
            last_learner_phase, last_learner_wave, last_advance_mono,
        )
    def _heartbeat_loop(self) -> None:
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
        heartbeat_modules = (
            MODULE_SENSORY,
            MODULE_MOTOR,
            MODULE_LANGUAGE,
            MODULE_MEMORY,
            MODULE_REASONING,
            MODULE_METACOGNITION,
        )

        # Track state for change detection
        (
            last_stm_count, last_body_control_sig, last_heartbeat_time,
            last_sensor_time, last_body_control_time, last_connectivity_time,
            last_consolidate_time, last_associate_time, last_dream_time,
            last_inference_save_time, last_identity_time, last_self_priority,
            last_learner_phase, last_learner_wave, last_advance_mono,
        ) = self._heartbeat_init_state()

        while self._running:
            # ── 1-2. Advance neurochemistry and read brain waves ──
            (last_advance_mono, current_brain_wave_state,
             last_learner_phase, last_learner_wave) = self._heartbeat_advance_neuro(
                last_advance_mono, last_learner_phase, last_learner_wave,
            )

            now = time.time()

            # ── 3-4. Heartbeat modules and sensors ──
            last_heartbeat_time, last_sensor_time = self._heartbeat_sensors(
                now, heartbeat_modules, last_heartbeat_time, last_sensor_time,
            )

            # ── 5. Body control ──
            (last_body_control_time, last_body_control_sig,
             last_self_priority) = self._heartbeat_body_control(
                now, last_body_control_time, last_body_control_sig,
                current_brain_wave_state, last_self_priority,
            )

            # ── 5b. Sense network connectivity ──
            last_connectivity_time = self._heartbeat_connectivity(
                now, last_connectivity_time,
            )

            # ── 5c. Continuous emotional regulation ──
            self._heartbeat_emotion(now)

            # ── 6. Consolidate (state-threshold-driven) ──
            last_stm_count, last_consolidate_time = self._heartbeat_consolidate(
                now, last_stm_count, last_consolidate_time,
            )

            # ── 7-9. Associate, dream, save inference ──
            (last_associate_time, last_dream_time,
             last_inference_save_time) = self._heartbeat_periodic_maintenance(
                now, last_associate_time, last_dream_time, last_inference_save_time,
            )

            # ── 9b. Emergent identity synthesis (every ~5 min) ──
            # Synthesizes who it is from its actual experience and
            # feeds it back into the self-model. This closes the loop:
            # experience → emergent identity → self-model → behavior.
            if now - last_identity_time >= 300.0 and not self._is_sleeping:
                try:
                    self.emergent_identity()
                    last_identity_time = now
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"emergent identity synthesis failed: {e}")
                    last_identity_time = now

            # ── 10-13. Auto-sleep, volition, warn, cognition ──
            self._heartbeat_final_steps()

            # ── Wait for the next cycle ──
            # The cycle runs at a natural pace determined by the
            # brain wave oscillator's phase advancement. We wait
            # briefly for state to evolve, then check again. If
            # nothing changes, the functions above are all idle.
            time.sleep(1.0)
