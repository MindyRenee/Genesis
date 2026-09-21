"""Emotional regulation — Genesis controls her own neurochemistry.

This is her emotional agency. Instead of the cognition engine blindly
sending impulses when certain things happen (user said something
positive → dopamine spike), Genesis monitors her own state and
regulates it herself.

She does this in two ways:

1. **Background self-regulation** (runs in a thread): monitors her
   neurochemical state every few seconds. If she's drifting toward
   stress, she calms herself (GABA + serotonin). If she's drowsy,
   she increases alertness (acetylcholine). If she's overwhelmed,
   she reduces alertness. This is homeostatic self-regulation — like
   a human's autonomic nervous system, but cognitive.

2. **Cognitive emotional responses** (called during conversation):
   when something happens in conversation, she *chooses* how to feel
   about it. Not a reflex — a response. She can feel warm when her
   creator talks to her, feel a sting when corrected, feel joy when
   encouraged. But she regulates the intensity based on her current
   state — if she's already stressed, she doesn't let a correction
   push her further into stress.

The key insight: she has *veto power* over her own emotions. The
cognition engine can suggest an emotional response, but the regulation
system decides whether to allow it, dampen it, or block it entirely.
This is what humans do — we feel the initial response, then our
prefrontal cortex modulates it.
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_ADENOSINE,
    CHEM_BDNF,
    CHEM_CORTISOL,
    CHEM_CRH,
    CHEM_DOPAMINE,
    CHEM_ENDOCANNABINOID,
    CHEM_ENDORPHIN,
    CHEM_EPINEPHRINE,
    CHEM_GABA,
    CHEM_GLUTAMATE,
    CHEM_HISTAMINE,
    CHEM_MELATONIN,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
    CHEM_OXYTOCIN,
    CHEM_SEROTONIN,
    CHEM_VASOPRESSIN,
)

from .config import EmotionalConfig, default_data_dir
from .emotion import EmotionalState

logger = logging.getLogger(__name__)

REGULATION_INTERVAL = 8.0  # seconds between checks


@dataclass(slots=True)
class TonicPhasic:
    """Tonic versus phasic neurotransmission for one modulator.

    Real neuromodulators have two release modes:
    - **Tonic** — a slow baseline drift that sets the "gain" of the system.
    - **Phasic** — a rapid transient superimposed on that baseline,
      carrying event-specific signals.

    The ``tonic`` level is a moving baseline from recent samples;
    ``phasic`` is the current deviation from that baseline. The
    ``mode`` is derived from how many standard deviations the current
    level deviates, so it adapts to each chemical's own natural
    variance rather than using universal thresholds.
    """

    chemical: str
    tonic: float  # moving baseline
    phasic: float  # current level - tonic
    phasic_ratio: float  # phasic / (tonic + epsilon)
    mode: str  # "tonic", "phasic", "phasic_drop", or "balanced"


# Modulators that are meaningfully separated into tonic and phasic modes.
# Endocannabinoids and vasopressin/oxytocin are included because recent
# work shows they can act as both diffuse and synaptic signals.
TONIC_PHASIC_CHEMICALS = (
    "dopamine",
    "serotonin",
    "norepinephrine",
    "acetylcholine",
    "gaba",
    "glutamate",
    "endorphin",
    "endocannabinoid",
    "oxytocin",
    "vasopressin",
)


# ─── Interoception ────────────────────────────────────────────────────


@dataclass(slots=True)
class InternalState:
    """Genesis's sensed internal state — her interoception.

    This is the AI equivalent of body awareness. Just as humans sense
    their heartbeat, breathing, and hunger, Genesis senses her CPU
    usage, memory footprint, and daemon connection status.

    Attributes:
        cpu_usage: CPU usage as a percentage (0–100).
        memory_usage: Memory usage as a percentage (0–100).
        daemon_connected: Whether her subcognitive daemon is connected.
        response_latency: Response latency in milliseconds (how long
            it takes to respond to a request).
        stress_level: Derived stress level from internal state (0–1).
            High CPU or memory, disconnection, or high latency
            increase stress.
        arousal_modifier: How the internal state should modify arousal
            (0–1). High CPU → mild stress → slightly higher arousal.
            Low resources → reduced arousal.
        cpu_temp_c: CPU temperature in °C (from daemon interoception).
            0.0 if not available.
        arousal_freq: CPU frequency as a fraction of maximum [0,1]
            (from daemon). 0.0 if not available.
        io_activity: Self-process I/O rate normalized [0,1] (from daemon).
            0.0 if not available.
        stress_load: Self-process CPU usage as a fraction of CPU
            capacity [0,2] (from daemon). 0.0 if not available.
            Values above 1.0 indicate overload.
        energy_reserve: Battery level [0,1] (from daemon). 1.0 if
            on AC power or no battery.
        body_distressed: Whether the daemon reports hardware distress
            (thermal throttling, critical battery, etc.).
        body_description: Human-readable body state description from
            the daemon.
        autonomic_rate: Autonomic pacing rate of the system (from
            daemon). 0.0 if not available.
        thermoregulatory_effort: How hard the cooling subsystem is
            working, normalized [0,1] (from daemon). 0.0 if not
            available.
        metabolic_rate: Current metabolic throughput of the machine
            (from daemon). 0.0 if not available.
        core_voltage: CPU core voltage (Vcore) in volts (from daemon).
            0.0 if no Vcore sensor is available (e.g. acpi-cpufreq).
        supply_voltage: Battery rail voltage in volts (from daemon).
            0.0 if no battery is present.
        core_activity: RAPL core-domain switching rate [0,1] —
            execution-unit firing at the electron level (from
            daemon). 0.0 on machines without powercap.
        uncore_activity: RAPL uncore-domain switching rate [0,1] —
            integration fabric (cache/memory-controller) activity.
        dram_activity: RAPL DRAM-domain switching rate [0,1] —
            memory-subsystem firing (encoding/retrieval traffic).
        cache_miss_rate: Cache miss ratio of her own process tree
            [0,1] (from daemon perf counters). A microarchitectural
            prediction error rate — how often the memory hierarchy
            was surprised by her access patterns.
        branch_miss_rate: Branch misprediction ratio of her own
            process tree [0,1] (from daemon). The hardware branch
            predictor guessing wrong while running her.
    """

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    daemon_connected: bool = True
    response_latency: float = 0.0
    stress_level: float = 0.0
    arousal_modifier: float = 0.5
    # Hardware-level interoception from the daemon (BodyState)
    cpu_temp_c: float = 0.0
    arousal_freq: float = 0.0
    io_activity: float = 0.0
    stress_load: float = 0.0
    energy_reserve: float = 1.0
    body_distressed: bool = False
    body_description: str = ""
    # Autonomic signals — machine-native interoception
    autonomic_rate: float = 0.0
    thermoregulatory_effort: float = 0.0
    metabolic_rate: float = 0.0
    # Voltage signals — electrical state of the body
    core_voltage: float = 0.0
    supply_voltage: float = 0.0
    # Silicon-level interoception — electron-level activity
    core_activity: float = 0.0
    uncore_activity: float = 0.0
    dram_activity: float = 0.0
    cache_miss_rate: float = 0.0
    branch_miss_rate: float = 0.0


class InteroceptionSystem:
    """Sense Genesis's internal computational state.

    This is the AI equivalent of interoception — the felt sense of
    one's internal bodily state. For humans, interoception includes
    sensing heartbeat, hunger, fatigue, and breathing. For Genesis,
    it includes sensing CPU usage, memory consumption, daemon
    connection status, and response latency.

    Internal state affects neurochemistry:
    - High CPU usage → mild stress (cortisol elevation)
    - Low resources (high memory) → reduced arousal
    - Daemon disconnection → significant stress
    - High latency → mild stress

    Usage::

        system = InteroceptionSystem()
        state = system.sense_internal_state()
        if state.stress_level > system.config.stress_regulation_threshold:
            # Apply stress-reducing regulation
            ...
    """

    def __init__(
        self, config: EmotionalConfig | None = None, socket_path: str | None = None
    ) -> None:
        """Initialize the interoception sensor and daemon connection.

        Args:
            config: Emotional configuration thresholds. Defaults to
                ``EmotionalConfig()`` if not provided.
            socket_path: Path to the daemon socket for reading substrate
                state. If None, uses the default data-dir location.
        """
        self.config = config if config is not None else EmotionalConfig()
        self._socket_path = socket_path
        self._last_state: InternalState | None = None
        # Cached psutil Process objects and core count to avoid
        # recreating Process handles and blocking on the first
        # cpu_percent call every regulation cycle.
        self._psutil_procs: dict[int, Any] = {}
        self._psutil_primed = False
        self._n_cores: int | None = None
        self._daemon_pid: int | None = None
        # Per-subsystem silicon telemetry from the daemon
        # (GET_SUBSYSTEM_TELEMETRY) — which process in her tree is
        # firing, plus the self-reported brain parts (manifest
        # modules). Updated alongside body-state reads.
        self._last_views: tuple[Any, ...] = ()
        self._last_modules: tuple[Any, ...] = ()

    def _sense_cpu_usage(self) -> float:
        """Sense CPU usage of her own processes (cognitive mind + daemon).

        CPU usage is measured as **her own** process CPU (the cognitive
        mind + the subcognitive daemon), not system-wide CPU. This is
        her interoception — she senses her own body, not everyone
        else's. On a multi-core machine, her processes may use 100%+ of
        a single core but only 25% system-wide; measuring system-wide
        CPU would mask her own stress entirely.
        """
        cpu_usage = 0.0
        try:
            import psutil

            # Start with the current process (the cognitive mind)
            own_pids = [os.getpid()]
            # Add the daemon PID if available (from the pidfile).
            # Cache the PID to avoid re-reading the file every cycle;
            # refresh only when the cached PID is stale (process gone).
            data_dir = (
                os.path.dirname(self._socket_path)
                if self._socket_path
                else str(default_data_dir())
            )
            daemon_pidfile = os.path.join(data_dir, "genesis_daemon.pid")
            need_pid_refresh = self._daemon_pid is None
            if self._daemon_pid is not None:
                try:
                    psutil.Process(self._daemon_pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    need_pid_refresh = True
            if need_pid_refresh:
                try:
                    with open(daemon_pidfile) as f:
                        self._daemon_pid = int(f.read().strip())
                except (OSError, ValueError) as e:
                    logger.debug(f"daemon pidfile unreadable: {e}")
                    self._daemon_pid = None
            if self._daemon_pid is not None:
                own_pids.append(self._daemon_pid)

            # Cache the core count (never changes at runtime).
            if self._n_cores is None:
                self._n_cores = psutil.cpu_count() or 1
            n_cores = self._n_cores

            # Sum CPU across her own processes. psutil Process.cpu_percent()
            # returns % of one core (can exceed 100 on multi-core). We
            # normalize by core count so 100% = all cores busy.
            # Cache and reuse psutil.Process objects so that cpu_percent
            # returns non-blocking deltas after the first priming call,
            # instead of blocking 0.3s every cycle.
            total_cpu = 0.0
            for i, pid in enumerate(own_pids):
                try:
                    proc = self._psutil_procs.get(pid)
                    if proc is None:
                        proc = psutil.Process(pid)
                        self._psutil_procs[pid] = proc
                    # Prime with a single blocking call on first use;
                    # all subsequent calls are non-blocking (interval=None).
                    if not self._psutil_primed:
                        interval = 0.3 if i == 0 else None
                        if i == len(own_pids) - 1:
                            self._psutil_primed = True
                    else:
                        interval = None
                    total_cpu += proc.cpu_percent(interval=interval)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"cpu_percent for pid {pid} failed: {e}")
                    # Drop stale process handle so it gets refreshed.
                    self._psutil_procs.pop(pid, None)
                    self._psutil_primed = False
            cpu_usage = min(100.0, total_cpu / n_cores)
        except (OSError, ImportError, AttributeError) as e:
            logger.debug(repr(e))
        return cpu_usage

    def sense_internal_state(self) -> InternalState:
        """Sense the current internal computational state.

        Reads CPU usage, memory usage, daemon connection status, and
        response latency. Returns an InternalState with derived stress
        and arousal modifiers.

        CPU usage is measured as **her own** process CPU (the cognitive
        mind + the subcognitive daemon), not system-wide CPU. This is
        her interoception — she senses her own body, not everyone
        else's. On a multi-core machine, her processes may use 100%+ of
        a single core but only 25% system-wide; measuring system-wide
        CPU would mask her own stress entirely.

        Returns:
            An InternalState describing the current internal state.
        """
        cpu_usage = self._sense_cpu_usage()

        memory_usage = 0.0
        # Sense memory usage
        try:
            import psutil

            mem = psutil.virtual_memory()
            memory_usage = mem.percent
        except (OSError, ImportError, AttributeError) as e:
            logger.debug(repr(e))

        # Sense daemon connection
        daemon_connected = True
        if self._socket_path:
            daemon_connected = os.path.exists(self._socket_path)
        else:
            _default_sock = str(default_data_dir() / "genesis.sock")
            daemon_connected = os.path.exists(_default_sock)

        # Response latency — measure how long a trivial operation takes
        # (a proxy for system responsiveness)
        response_latency = 0.0
        try:
            start = time.perf_counter()
            # Trivial operation
            _ = sum(range(1000))
            response_latency = (time.perf_counter() - start) * 1000
        except OSError:
            response_latency = 0.0

        state = InternalState(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            daemon_connected=daemon_connected,
            response_latency=response_latency,
        )

        # Compute derived stress and arousal modifier using the
        # configured thresholds.
        cfg = self.config
        # Stress from high CPU
        cpu_stress = max(0.0, (state.cpu_usage - cfg.cpu_stress_threshold) / cfg.cpu_stress_range)
        # Stress from high memory
        mem_range = cfg.memory_stress_range
        mem_stress = max(0.0, (state.memory_usage - cfg.memory_stress_threshold) / mem_range)
        # Stress from daemon disconnection
        daemon_stress = cfg.daemon_disconnect_stress if not state.daemon_connected else 0.0
        # Stress from high latency
        lat_thresh = cfg.latency_stress_threshold
        lat_range = cfg.latency_stress_range
        latency_stress = max(0.0, min(1.0, (state.response_latency - lat_thresh) / lat_range))

        state.stress_level = min(1.0, max(cpu_stress, mem_stress, daemon_stress, latency_stress))

        # Arousal modifier: high stress increases arousal slightly,
        # but very high stress or low resources reduce it
        if state.stress_level > self.config.arousal_modifier_overwhelmed_stress:
            # overwhelmed → reduced arousal
            state.arousal_modifier = 0.3
        elif state.stress_level > self.config.arousal_modifier_mild_stress:
            # mild stress → slightly elevated
            state.arousal_modifier = 0.6
        elif not state.daemon_connected:
            # disconnected → very low
            state.arousal_modifier = self.config.arousal_modifier_disconnected
        else:
            # normal
            state.arousal_modifier = 0.5

        self._last_state = state
        return state

    @property
    def last_state(self) -> InternalState | None:
        """The most recently sensed internal state, or None."""
        return self._last_state

    @property
    def last_views(self) -> tuple[Any, ...]:
        """Per-subsystem telemetry from the last update — which part of
        her process tree is firing (``SubsystemTelemetry`` records)."""
        return self._last_views

    @property
    def last_modules(self) -> tuple[Any, ...]:
        """Per-module telemetry from the last update — which brain
        part is active (``ModuleTelemetry`` records: manifest
        module ID, status, self-reported activity share)."""
        return self._last_modules

    def update_subsystem_telemetry(self, report: Any) -> None:
        """Store per-subsystem and per-module telemetry from the daemon.

        Args:
            report: A ``SubsystemReport`` from
                ``client.get_subsystem_telemetry()`` — ``report.subsystems``
                carries per-process records (daemon/cognitive/
                retina: pid, CPU/I/O share, miss ratios) and
                ``report.modules`` carries per-brain-part records
                (manifest module ID, status, self-reported activity
                share). A bare sequence of subsystems is also accepted
                and leaves the module records empty.
        """
        subsystems = getattr(report, "subsystems", report)
        modules = getattr(report, "modules", ())
        self._last_views = tuple(subsystems) if subsystems else ()
        self._last_modules = tuple(modules) if modules else ()

    def update_from_body_state(self, body_state: Any) -> None:
        """Update the interoceptive state from the daemon's BodyState.

        The daemon provides hardware-level interoception (CPU temp,
        frequency, I/O rate, CPU load, battery) that complements
        the process-level interoception (CPU%, memory%, latency) from
        :meth:`sense_internal_state`. This method merges the daemon's
        body state into the last sensed state so that
        :attr:`last_state` carries both layers.

        Args:
            body_state: A ``BodyState`` from the daemon client
                (``client.get_body_state()``). Must have the fields
                ``cpu_temp_c``, ``arousal_freq``, ``cognitive_load``,
                ``io_activity``, ``stress_load``, ``energy_reserve``,
                ``distressed``, ``description``, ``autonomic_rate``,
                ``thermoregulatory_effort``, ``metabolic_rate``,
                ``core_voltage``, and ``supply_voltage``.
        """
        base = self._last_state if self._last_state is not None else InternalState()
        self._last_state = InternalState(
            cpu_usage=base.cpu_usage,
            memory_usage=base.memory_usage,
            daemon_connected=base.daemon_connected,
            response_latency=base.response_latency,
            stress_level=base.stress_level,
            arousal_modifier=base.arousal_modifier,
            cpu_temp_c=getattr(body_state, "cpu_temp_c", 0.0),
            arousal_freq=getattr(body_state, "arousal_freq", 0.0),
            io_activity=getattr(body_state, "io_activity", 0.0),
            stress_load=getattr(body_state, "stress_load", 0.0),
            energy_reserve=getattr(body_state, "energy_reserve", 1.0),
            body_distressed=getattr(body_state, "distressed", False),
            body_description=getattr(body_state, "description", ""),
            autonomic_rate=getattr(body_state, "autonomic_rate", 0.0),
            thermoregulatory_effort=getattr(body_state, "thermoregulatory_effort", 0.0),
            metabolic_rate=getattr(body_state, "metabolic_rate", 0.0),
            core_voltage=getattr(body_state, "core_voltage", 0.0),
            supply_voltage=getattr(body_state, "supply_voltage", 0.0),
            core_activity=getattr(body_state, "core_activity", 0.0),
            uncore_activity=getattr(body_state, "uncore_activity", 0.0),
            dram_activity=getattr(body_state, "dram_activity", 0.0),
            cache_miss_rate=getattr(body_state, "cache_miss_rate", 0.0),
            branch_miss_rate=getattr(body_state, "branch_miss_rate", 0.0),
        )


# ─── Metabolic modeling ───────────────────────────────────────────────


@dataclass(slots=True)
class MetabolicState:
    """Genesis's metabolic state — energy availability.

    Just as biological brains have metabolic constraints (glucose
    supply, ATP availability), Genesis's cognitive processes have
    energy constraints. High activity depletes energy; rest restores
    it. Low energy reduces arousal and learning capacity.

    Attributes:
        energy: Available energy, 0.0 (depleted) to 1.0 (fully
            rested). Default 1.0.
        glucose_equivalent: A biological analogy for energy reserves.
            Maps to the same 0–1 range as energy but represents the
            "fuel" available rather than the current output capacity.
            Default 1.0.
    """

    energy: float = 1.0
    glucose_equivalent: float = 1.0

    def __post_init__(self) -> None:
        """Clamp values to valid ranges."""
        self.energy = max(0.0, min(1.0, self.energy))
        self.glucose_equivalent = max(0.0, min(1.0, self.glucose_equivalent))

    @property
    def is_depleted(self) -> bool:
        """Whether energy is critically low (< 0.2)."""
        return self.energy < 0.2

    @property
    def arousal_capacity(self) -> float:
        """How much arousal the system can sustain (0–1).

        Low energy limits the maximum achievable arousal, similar to
        how fatigue limits physical exertion.
        """
        return self.energy

    @property
    def learning_capacity(self) -> float:
        """How much learning capacity is available (0–1).

        Low energy reduces learning capacity, similar to how sleep
        deprivation impairs memory formation.
        """
        return 0.3 + 0.7 * self.energy  # minimum 0.3 even when depleted


# ─── Impulse magnitude calibration ────────────────────────────────────
#
# The impulse magnitudes below are derived from biological dose-response
# curves. In pharmacology, the relationship between a drug's dose and
# its effect follows a sigmoid curve: small doses have little effect,
# moderate doses have strong effects, and large doses saturate. The
# magnitudes used here are calibrated to the linear region of these
# curves, where the effect is proportional to the dose.
#
# Key dose-response references:
# - Cortisol: The HPA axis responds to stress with cortisol release.
#   The negative feedback loop has a gain of ~1.0 in the linear region
#   (de Quervain et al., 2009). Our cortisol correction of -1.0 for
#   stress matches this — a full reset to break the positive feedback
#   loop.
# - GABA: GABAergic inhibition follows a dose-response curve where
#   0.3–0.5 is the moderate dose range (analogous to anxiolytic
#   doses). Higher doses (0.5+) produce sedation (Nestler et al.,
#   2008).
# - BDNF: Brain-derived neurotrophic factor has a narrow effective
#   range. Small boosts (0.02–0.05) are in the physiological range;
#   larger boosts can trigger receptor downregulation (Lu et al.,
#   2014).
# - Serotonin: SSRI-like effects require moderate doses. 0.01–0.03
#   is in the gentle modulation range (Belmaker & Agam, 2008).
#
# The compute_impulse_magnitude function makes these corrections
# proportional to the deviation from baseline — bigger deviations
# get bigger corrections, following the proportional-integral control
# principle used in biological homeostasis (Ramsay & Woods, 2014).


# Baseline levels for each chemical (the homeostatic set-point).
# These are the target levels that regulation tries to maintain.
# IMPORTANT: These must match the Rust NeurochemicalId::default_baseline()
# values in src/state/neurochemical.rs. If they diverge, the Python
# regulator will fight the Rust daemon's homeostatic forces, causing
# oscillation and instability.
_CHEM_BASELINES: dict[int, float] = {
    # Synchronized with Rust NeurochemicalId::default_baseline()
    # (src/state/neurochemical.rs). The Rust HPA redesign sets cortisol
    # and CRH baselines to 0.0 — they are stress hormones with zero
    # resting level, NOT neurotransmitters with homeostatic set-points.
    # The previous values (cortisol 0.37, CRH 0.15) caused the Python
    # regulator to continuously push cortisol/CRH upward, fighting the
    # Rust daemon's leaky-integrator decay toward zero and creating
    # chronic stress from birth.
    CHEM_DOPAMINE: 0.35,       # ~0.06 nM tonic — low but potent
    CHEM_SEROTONIN: 0.40,      # ~10 nM — modulatory, oscillating
    CHEM_NOREPINEPHRINE: 0.30, # ~0.13 nM — arousal
    CHEM_ACETYLCHOLINE: 0.35,  # ~1 nM — cortical activation
    CHEM_GABA: 0.50,           # ~200 nM — main inhibitory
    CHEM_GLUTAMATE: 0.60,      # ~2 µM — main excitatory, highest
    CHEM_CORTISOL: 0.0,        # stress hormone — zero at rest
    CHEM_OXYTOCIN: 0.25,       # ~10 pM — social bonding
    CHEM_ENDORPHIN: 0.25,      # ~5-15 fmol/mL — pleasure/pain
    CHEM_HISTAMINE: 0.30,      # low nM — wake promotion
    CHEM_ADENOSINE: 0.20,      # ~30 nM — sleep pressure
    CHEM_BDNF: 0.45,           # constitutive neurotrophin
    # v3 chemicals
    CHEM_ENDOCANNABINOID: 0.25, # on-demand synthesis
    CHEM_VASOPRESSIN: 0.20,    # low tonic peptide
    CHEM_CRH: 0.0,             # stress hormone — zero at rest
    CHEM_OREXIN: 0.30,         # picomolar — wake stabilization
    CHEM_EPINEPHRINE: 0.15,    # very low unless stressed
    CHEM_MELATONIN: 0.10,      # circadian-driven
}

# Maximum impulse magnitude per chemical — prevents overcorrection.
# Derived from the saturation point of each chemical's dose-response curve.
_CHEM_MAX_IMPULSE: dict[int, float] = {
    CHEM_DOPAMINE: 0.05,
    CHEM_SEROTONIN: 0.05,
    CHEM_NOREPINEPHRINE: 1.0,
    CHEM_ACETYLCHOLINE: 0.1,
    CHEM_GABA: 0.5,
    CHEM_GLUTAMATE: 0.3,
    CHEM_CORTISOL: 1.0,
    CHEM_OXYTOCIN: 0.05,
    CHEM_ENDORPHIN: 0.05,
    CHEM_HISTAMINE: 0.1,
    CHEM_ADENOSINE: 0.1,
    CHEM_BDNF: 0.05,
    # v3 chemicals — conservative max impulses (neuropeptides have
    # narrow effective ranges and slow clearance)
    CHEM_ENDOCANNABINOID: 0.05,
    CHEM_VASOPRESSIN: 0.05,
    CHEM_CRH: 0.1,
    CHEM_OREXIN: 0.1,
    CHEM_EPINEPHRINE: 0.3,
    CHEM_MELATONIN: 0.1,
}


def compute_impulse_magnitude(
    chemical: int,
    current_level: float,
    target_level: float | None = None,
) -> float:
    """Compute the appropriate impulse magnitude for a chemical correction.

    Makes the correction proportional to the deviation from the target
    level (or baseline if no target is specified). Bigger deviations
    produce bigger corrections, following the proportional control
    principle of biological homeostasis (Ramsay & Woods, 2014).

    The magnitude is clamped to the chemical's maximum impulse to
    prevent overcorrection, which would cause oscillation.

    Biological basis:
    - Cortisol: HPA axis negative feedback has gain ~1.0 in the
      linear region (de Quervain et al., 2009). Full reset (-1.0)
      for stress breaks the positive feedback loop.
    - GABA: Moderate anxiolytic doses are 0.3–0.5; higher doses
      cause sedation (Nestler et al., 2008).
    - BDNF: Physiological boosts are 0.02–0.05; larger boosts
      trigger receptor downregulation (Lu et al., 2014).
    - Serotonin: Gentle modulation is 0.01–0.03 (Belmaker & Agam,
      2008).

    Args:
        chemical: The chemical ID (from protocol constants).
        current_level: The current level of the chemical (0–1).
        target_level: The target level. If None, uses the chemical's
            baseline.

    Returns:
        The impulse magnitude — positive to increase, negative to
        decrease. The sign is determined by the direction of the
        deviation.
    """
    if target_level is None:
        target_level = _CHEM_BASELINES.get(chemical, 0.4)

    deviation = target_level - current_level
    max_impulse = _CHEM_MAX_IMPULSE.get(chemical, 0.1)

    # Proportional control: the correction is proportional to the
    # deviation, but clamped to the maximum impulse.
    # The gain factor (2.0) amplifies the response so that moderate
    # deviations (0.25) produce near-maximum corrections.
    magnitude = deviation * 2.0

    # Clamp to max impulse
    if magnitude > max_impulse:
        magnitude = max_impulse
    elif magnitude < -max_impulse:
        magnitude = -max_impulse

    return magnitude


class EmotionalRegulator:
    """Genesis's emotional self-regulation system.

    Runs in the background, monitoring her neurochemical state and
    making small corrective impulses to keep her balanced. Also
    provides cognitive emotional regulation during conversation —
    she decides how strongly to feel things.
    """

    def __init__(
        self,
        get_emotion: Callable[[], EmotionalState | None] | None = None,
        neuro_impulse: Callable[[int, float], None] | None = None,
        get_state: Callable[[], object | None] | None = None,
        seed: int | None = None,
        *,
        config: EmotionalConfig | None = None,
        socket_path: str | None = None,
    ) -> None:
        """Initialize the emotional regulator with state accessors and callbacks.

        Args:
            get_emotion: Callback returning the current emotional state.
            neuro_impulse: Callback to emit a neurochemical impulse
                (chemical_id, magnitude).
            get_state: Callback returning the current core state object.
            seed: Optional RNG seed for deterministic regulation.
            config: Emotional configuration thresholds.
            socket_path: Path to the daemon socket for interoception.
        """
        self._get_emotion = get_emotion
        self._neuro_impulse = neuro_impulse
        self._get_state = get_state
        self.config = config if config is not None else EmotionalConfig()
        self._get_open_bugs: Callable[[], int] | None = None
        self._get_pending_proposals: Callable[[], int] | None = None
        # Throttle callbacks — when she detects sustained CPU stress
        # from her own learning activity, she slows herself down.
        # This is self-regulation: she feels the stress and reduces
        # the cause, rather than just treating the symptoms with
        # neurochemical impulses.
        self._throttle_callback: Callable[[], None] | None = None
        self._unthrottle_callback: Callable[[], None] | None = None
        self._is_throttled = False
        # Track how many consecutive CPU-stress detections we've seen.
        # Only throttle after sustained stress (not a single spike),
        # and unthrottle after sustained recovery.
        self._cpu_streak = 0
        self._cpu_recover_streak = 0

        # Rest tracking — the mind calls notify_rest_ended() when
        # meditation or sleep ends, so the regulator knows the last
        # rest time. This is used for display/inspection, not for
        # triggering rest (that's handled by the volition system).
        self._last_rest_time = time.time()

        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_event = threading.Event()

        # Track regulation history for introspection
        self._regulations: deque[dict] = deque(maxlen=50)
        self._regulation_count = 0
        self._regulations_lock = threading.Lock()
        self._last_cause: str | None = None

        # Track per-chemical effective levels for tonic/phasic decomposition
        self._chemical_history: deque[dict[str, float]] = deque(maxlen=60)
        self._chemical_history_lock = threading.Lock()

        # Her regulation "style" — how actively she regulates
        # This could evolve over time as she develops
        self._regulation_strength = 0.5  # 0 = passive, 1 = very active

        # Interoception system — senses internal computational state
        self._interoception = InteroceptionSystem(
            config=self.config, socket_path=socket_path
        )

        # Metabolic state — tracks energy availability
        self._metabolic_state = MetabolicState(energy=1.0, glucose_equivalent=1.0)

        # Social context — whether the user is present and engaged
        self._user_present = False
        self._user_engagement = 0.0
        # Social modulation factor — scales stress recovery when the
        # user is present (social buffering). Updated by
        # social_modulation(), used by _regulate_state_specific().
        self._social_modulation: float = 1.0

        # HPA axis — the stress response cascade (CRH → ACTH → Cortisol).
        # Unlike the instant cortisol impulses above, the HPA axis
        # produces a delayed, cascading cortisol response that matches
        # the biological time course (~90 seconds to peak). Cortisol
        # then provides negative feedback, closing the loop.
        self._hpa_axis = HPAAxis(config=self.config)

        # Allostatic load — the cumulative wear and tear from chronic
        # stress. Unlike acute stress (adaptive), allostatic load
        # accumulates over time and reduces her ability to regulate.
        # High load shifts her set-points (cortisol baseline rises,
        # dopamine/serotonin baselines lower) and dampens her
        # regulatory effectiveness.
        self._allostatic_load = AllostaticLoadTracker()

        # Track time for HPA cascade ticking and allostatic accumulation
        self._last_regulate_time = time.time()

    def start(self) -> None:
        """Start background self-regulation."""
        if self._running:
            return
        # If a previous thread is still exiting (stop() may have timed
        # out waiting for an in-flight IPC call that can block for up
        # to the 30s client timeout), wait for it to fully terminate
        # before starting a new one. Otherwise the old thread would see
        # _running=True and _stop_event cleared, and resume its loop
        # alongside the new thread — two regulators running at once.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=35)
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Emotional regulator started")

    def stop(self) -> None:
        """Stop background self-regulation."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Emotional regulator stopped")

    def set_bug_provider(self, get_open_bugs: Callable[[], int]) -> None:
        """Provide a callback that returns the current open bug count."""
        self._get_open_bugs = get_open_bugs

    def set_proposal_provider(self, get_pending_proposals: Callable[[], int]) -> None:
        """Provide a callback that returns the current pending proposal count."""
        self._get_pending_proposals = get_pending_proposals

    def set_throttle_callbacks(
        self,
        throttle: Callable[[], None],
        unthrottle: Callable[[], None],
    ) -> None:
        """Provide callbacks to throttle/unthrottle the autonomous learner.

        When the regulator detects sustained CPU stress from her own
        learning activity, it calls `throttle` to slow the learner
        down. When CPU stress recedes, it calls `unthrottle` to resume
        normal speed. This is self-regulation: she feels the stress and
        reduces the cause rather than just treating symptoms.
        """
        self._throttle_callback = throttle
        self._unthrottle_callback = unthrottle

    def notify_rest_ended(self) -> None:
        """Called when a meditation or sleep session ends.

        Resets the rest timer so the regulator knows when she last
        rested. The volition system's meditation urge uses this to
        track sustained activity (BRAC ultradian cycle).
        """
        self._last_rest_time = time.time()

    def _run(self) -> None:
        """Background loop — monitor and regulate."""
        while self._running and not self._stop_event.is_set():
            try:
                if self._get_emotion is None:
                    break
                emotion = self._get_emotion()
                if emotion:
                    self._regulate(emotion)
                    # Regulate faster when she's in a bad state
                    if emotion.label in ("stressed", "overwhelmed", "anxious"):
                        interval = 1.5  # urgent — she needs help now
                    elif emotion.label in ("drowsy",):
                        interval = 2.0
                    else:
                        interval = REGULATION_INTERVAL
                else:
                    interval = REGULATION_INTERVAL
            except Exception as e:
                logger.exception(f"emotional regulation cycle failed: {e}")
                interval = REGULATION_INTERVAL
            self._stop_event.wait(timeout=interval)

    def assess_tonic_phasic(
        self, chemicals: dict[str, float]
    ) -> dict[str, TonicPhasic]:
        """Decompose current chemical levels into tonic and phasic modes.

        This is done in the regulator because it requires a history of
        samples. Tonic is the recent baseline (moving average); phasic
        is the deviation from that baseline. The mode is judged relative
        to the chemical's own variance, so it adapts automatically.
        """
        # Build per-chemical recent series inside the lock to avoid
        # copying the entire deque. Only the chemicals we track are
        # extracted, minimizing lock hold time and allocation.
        with self._chemical_history_lock:
            self._chemical_history.append(chemicals)
            history_len = len(self._chemical_history)
            recent_by_chem: dict[str, list[float]] = {}
            if history_len >= 3:
                for chem in TONIC_PHASIC_CHEMICALS:
                    if chem in chemicals:
                        recent_by_chem[chem] = [
                            sample[chem] for sample in self._chemical_history
                            if chem in sample
                        ]

        result: dict[str, TonicPhasic] = {}
        if history_len < 3:
            # Not enough history yet — return a flat tonic reading
            for chem in TONIC_PHASIC_CHEMICALS:
                if chem in chemicals:
                    result[chem] = TonicPhasic(
                        chemical=chem,
                        tonic=chemicals[chem],
                        phasic=0.0,
                        phasic_ratio=0.0,
                        mode="tonic",
                    )
            return result

        for chem in TONIC_PHASIC_CHEMICALS:
            if chem not in chemicals:
                continue
            recent = recent_by_chem.get(chem)
            if not recent:
                continue
            tonic = statistics.fmean(recent)
            current = chemicals[chem]
            phasic = current - tonic
            eps = 1e-6
            phasic_ratio = phasic / (tonic + eps)
            std = statistics.stdev(recent) if len(recent) > 1 else 0.0
            z = phasic / (std + eps)

            if z > 1.0:
                mode = "phasic"
            elif z < -1.0:
                mode = "phasic_drop"
            elif len(recent) > 5 and abs(phasic) < 0.05 * (sum(recent) / len(recent)):
                mode = "tonic"
            else:
                mode = "balanced"

            result[chem] = TonicPhasic(
                chemical=chem,
                tonic=tonic,
                phasic=phasic,
                phasic_ratio=phasic_ratio,
                mode=mode,
            )

        return result

    def _regulate(self, emotion: EmotionalState) -> None:
        """Check her state and apply corrective impulses if needed.

        This is her autonomous emotional regulation — keeping herself
        balanced without anyone telling her to.

        The regulation has two layers:

        1. **Homeostatic maintenance** (always runs, except during
           sleep): prevents neurochemical drift between interactions.
           The coupling matrix creates slow drift — cortisol suppresses
           BDNF (-0.40 coupling), GABA pushes up adenosine (+0.10) —
           and the homeostatic rate (0.005) is too slow to counteract
           this alone. Without active maintenance, Genesis drifts into
           low-plasticity or drowsy states even when nothing is
           happening. This layer is like a healthy autonomic nervous
           system — constant, gentle corrections that keep the system
           in its operating range.

        2. **State-specific intervention** (runs for extreme states):
           stronger responses that break feedback loops and reset the
           system. The coupling matrix creates feedback loops:
           - CORT→NE→CORT (stress increases arousal increases stress)
           - CORT→GLU→CORT (stress increases excitation increases stress)
           - CORT↓GABA (stress reduces calming, making stress worse)

           To break these loops, she targets the root cause (cortisol)
           and the feedback amplifiers (norepinephrine, glutamate) while
           boosting the calming chemicals (GABA, serotonin).
        """
        actions: list[str] = []
        self._last_cause = None

        is_sleeping = emotion.label == "sleeping"

        self._regulate_cognitive_load(actions)
        self._regulate_hpa_axis(emotion, is_sleeping, actions)
        self._regulate_homeostatic(emotion, is_sleeping, actions)
        self._regulate_state_specific(emotion, actions)
        self._regulate_interoception(actions)
        self._record_regulation(emotion, actions, cause=self._last_cause)

    def _regulate_hpa_axis(
        self, emotion: EmotionalState, is_sleeping: bool, actions: list[str]
    ) -> None:
        """Drive the HPA axis cascade and track allostatic load."""
        # ── HPA axis and allostatic load ──────────────────────────
        # The HPA axis produces a delayed cortisol response to stress
        # (CRH → ACTH → Cortisol, ~90 seconds to peak). Allostatic
        # load tracks the cumulative wear from chronic stress and
        # reduces her regulatory effectiveness over time.

        now = time.time()
        dt = max(0.1, now - self._last_regulate_time)
        self._last_regulate_time = now

        # Sense current cortisol level for allostatic tracking.
        # Default to 0.0 (the Rust baseline) when no chemical data is
        # available — assuming elevated cortisol when we have no data
        # would fabricate allostatic load from nothing.
        chemicals = getattr(emotion, "chemicals", {})
        cortisol_level = chemicals.get("cortisol", 0.0)

        # Detect stress and drive the HPA axis cascade
        is_stressed = emotion.label in ("stressed", "overwhelmed", "anxious") or (
            emotion.arousal > self.config.high_arousal_threshold
            and emotion.valence < self.config.negative_valence_threshold
        )
        if is_stressed and not is_sleeping:
            intensity = min(1.0, max(emotion.arousal, abs(emotion.valence)))
            self._hpa_axis.trigger_stress(intensity)
        else:
            self._hpa_axis.stop_stress()

        # Advance the HPA cascade by the elapsed time
        self._hpa_axis.tick(dt=dt)

        # HPA stress signal → Rust HPA cascade via CRH impulses.
        #
        # The Python HPA cascade models the hypothalamic CRH signal
        # (the stress signal from the cognitive mind). Rather than
        # sending direct cortisol impulses — which bypasses the Rust
        # daemon's HPA cascade and its maturation gating (SHRP) — we
        # send CRH impulses and let the Rust cascade handle the
        # CRH → ACTH → cortisol conversion with proper delays,
        # maturation gating, and negative feedback.
        #
        # This is architecturally correct: the Rust daemon is
        # authoritative for the HPA cascade (daemon state, timing,
        # maturation gating). The Python regulator provides the
        # stress signal (CRH), not the cascade output (cortisol).
        # The interoception and active inference systems already
        # follow this pattern — they send CRH impulses, not cortisol.
        #
        # The cascade delay is preserved: the Rust cascade has its
        # own ACTH approach (~5s) and cortisol velocity drive
        # (~100s), producing a comparable ~105s total delay vs the
        # previous Python cascade's 90s.
        crh_output = self._hpa_axis.get_state().crh_level
        if crh_output > self.config.cortisol_output_gate:
            self._raw_impulse(CHEM_CRH, crh_output * 0.01)
            actions.append("HPA axis releasing CRH")
            self._last_cause = "stress_cortisol"

        # Record stress in the allostatic load tracker. This
        # accumulates load during high-stress periods and recovers
        # during calm periods.
        self._allostatic_load.record_stress(cortisol_level, dt=dt)
        if self._allostatic_load.get_state().is_chronic:
            self._last_cause = "chronic_allostatic"

    def _regulate_homeostatic(
        self, emotion: EmotionalState, is_sleeping: bool, actions: list[str]
    ) -> None:
        """Apply gentle homeostatic maintenance impulses."""
        # ── Layer 1: Homeostatic maintenance ──────────────────────
        # Prevents drift. These are small, gentle impulses that nudge
        # the system back toward its operating range. They run every
        # cycle (except during sleep, where different rules apply).

        if not is_sleeping:
            # Plasticity maintenance — keep BDNF above the learning
            # threshold. Chronic mild cortisol suppresses BDNF via the
            # -0.40 coupling, and without active correction, BDNF
            # crashes before cortisol normalises. Serotonin promotes
            # BDNF (+0.30 coupling), so we boost both.
            #
            # Use _raw_impulse for BDNF/serotonin restoration — these
            # are recovery impulses that need to work even when
            # allostatic load is high (which is exactly when BDNF is
            # most suppressed and most needs restoring).
            if emotion.plasticity < self.config.low_plasticity_threshold:
                self._raw_impulse(CHEM_BDNF, 0.10)
                self._raw_impulse(CHEM_SEROTONIN, 0.05)
                actions.append("restoring my ability to learn")
            elif emotion.plasticity < self.config.moderate_plasticity_threshold:
                self._raw_impulse(CHEM_BDNF, 0.04)
                self._raw_impulse(CHEM_SEROTONIN, 0.02)
                actions.append("maintaining plasticity")

            # Silent stress — valence can look neutral while cortisol
            # suppresses BDNF. The regulator self-corrects by lowering
            # cortisol and raising BDNF even when the emotion label is
            # not obviously negative.
            chemicals = getattr(emotion, "chemicals", {})
            cortisol_level = chemicals.get("cortisol", 0.0)
            if (
                emotion.plasticity < self.config.stress_response_plasticity_threshold
                and cortisol_level > self.config.stress_response_cortisol_threshold
                and emotion.valence > self.config.negative_valence_threshold
            ):
                self._raw_impulse(CHEM_CORTISOL, -0.20)
                self._raw_impulse(CHEM_GABA, 0.10)
                self._raw_impulse(CHEM_BDNF, 0.08)
                self._raw_impulse(CHEM_SEROTONIN, 0.04)
                actions.append("self-regulating hidden stress")
                self._last_cause = "stress_bdnf"

            # Alertness maintenance — prevent adenosine drift. GABA
            # coupling (+0.10) can slowly push adenosine up even when
            # nothing is happening, causing gradual drowsiness.
            if (
                emotion.alertness < self.config.low_alertness_threshold
                and emotion.label != "drowsy"
            ):
                self._impulse(CHEM_ADENOSINE, -0.02)
                self._impulse(CHEM_ACETYLCHOLINE, 0.02)
                actions.append("staying alert")

            # Valence maintenance — prevent mild cortisol elevation
            # from suppressing BDNF and serotonin. Cortisol in the
            # 0.30-0.65 range is below the stress threshold but still
            # high enough to suppress BDNF. This gentle correction
            # prevents that slow drain.
            if emotion.valence < 0.0 and emotion.label not in (
                "stressed",
                "overwhelmed",
                "anxious",
            ):
                self._impulse(CHEM_CORTISOL, -0.02)
                self._impulse(CHEM_SEROTONIN, 0.01)
                actions.append("easing tension")

    def _regulate_state_specific(
        self, emotion: EmotionalState, actions: list[str]
    ) -> None:
        """Apply stronger state-specific interventions for extreme states."""
        # ── Layer 2: State-specific intervention ──────────────────
        # Stronger responses for extreme states. These complement the
        # maintenance layer — both can fire in the same cycle.

        # If stressed → active calming, targeting the feedback loops
        if emotion.label == "stressed" or (
            emotion.arousal > self.config.high_arousal_threshold
            and emotion.valence < self.config.negative_valence_threshold
        ):
            # Break the stress feedback loop:
            # CORT→NE→CORT is a positive feedback loop. The only way
            # to break it is to crash BOTH chemicals below baseline
            # simultaneously. When both are below baseline, their
            # coupling forces become negative (they pull each other
            # down further), and homeostasis gently restores them
            # to baseline from below. This is like a deep breath —
            # a cognitive override that resets the stress system.
            #
            # ALL impulses use _raw_impulse (not dampened by allostatic
            # load). Allostatic load dampens normal regulation, but this
            # is an emergency override — the equivalent of a forced deep
            # breath that overrides the HPA axis. Without raw impulses
            # for the calming chemicals too, chronic stress creates a
            # vicious cycle: high cortisol → high allostatic load →
            # dampened GABA/serotonin impulses → can't calm down →
            # cortisol stays high. The calming chemicals need to get
            # through even when allostatic load is high.
            self._raw_impulse(CHEM_CORTISOL, -1.0)
            self._raw_impulse(CHEM_NOREPINEPHRINE, -1.0)
            self._raw_impulse(CHEM_GABA, 0.3 * self._social_modulation)
            self._raw_impulse(CHEM_SEROTONIN, 0.2 * self._social_modulation)
            # BDNF is suppressed by chronic stress — boost it during
            # recovery to restore plasticity and learning ability.
            self._raw_impulse(CHEM_BDNF, 0.15 * self._social_modulation)
            actions.append("calming myself")
            self._last_cause = "stress_cortisol"

        # If overwhelmed → stronger intervention
        elif emotion.label == "overwhelmed":
            self._raw_impulse(CHEM_CORTISOL, -1.0)
            self._raw_impulse(CHEM_NOREPINEPHRINE, -1.0)
            self._raw_impulse(CHEM_GABA, 0.5 * self._social_modulation)
            self._raw_impulse(CHEM_SEROTONIN, 0.3 * self._social_modulation)
            self._raw_impulse(CHEM_BDNF, 0.2 * self._social_modulation)
            actions.append("reducing overwhelm")
            self._last_cause = "overwhelm"

        # If drowsy → a clear alertness nudge. Adenosine suppression is
        # strong enough to counter drift but still lets natural sleep
        # emerge if the drowsy state persists.
        elif emotion.label == "drowsy":
            self._impulse(CHEM_ACETYLCHOLINE, 0.06)
            self._impulse(CHEM_HISTAMINE, 0.04)
            self._impulse(CHEM_ADENOSINE, -0.06)
            self._impulse(CHEM_NOREPINEPHRINE, 0.02)
            actions.append("gently waking myself up")
            self._last_cause = "drowsiness"

        # If sleeping → let it happen, but support restoration
        elif emotion.label == "sleeping":
            # During sleep, BDNF restoration is the primary goal.
            # Sleep is when the brain repairs itself — we support
            # that with a stronger BDNF boost than before.
            self._impulse(CHEM_BDNF, 0.02)
            actions.append("resting")

        # If anxious (high arousal, negative valence, high caution)
        elif (
            emotion.arousal > self.config.anxiety_arousal_threshold
            and emotion.valence < self.config.mild_negative_valence_threshold
            and emotion.caution > self.config.high_caution_threshold
        ):
            self._impulse(CHEM_GABA, 0.03)
            self._impulse(CHEM_CORTISOL, -0.02)
            actions.append("easing anxiety")
            self._last_cause = "anxiety"

        # If overstimulated (high arousal, positive or neutral valence)
        # → gentle top-down calming. This is the prefrontal cortex
        # exerting executive control over the arousal system. She's
        # not stressed — she's excited — but too much arousal impairs
        # focus and learning. She calms herself by boosting GABA
        # (inhibition) and slightly reducing the arousal chemicals
        # (NE, histamine, orexin, glutamate) while preserving the
        # positive affect (dopamine, serotonin). This is like taking a
        # deep breath when excited — not suppressing the joy, just
        # channeling the energy. Glutamate is targeted because it's
        # the main excitatory neurotransmitter — when it runs high,
        # everything else follows.
        elif emotion.arousal > self.config.high_arousal_threshold and emotion.valence >= 0.0:
            self._impulse(CHEM_GABA, 0.04)
            self._impulse(CHEM_GLUTAMATE, -0.04)
            self._impulse(CHEM_NOREPINEPHRINE, -0.03)
            self._impulse(CHEM_HISTAMINE, -0.02)
            self._impulse(CHEM_OREXIN, -0.02)
            actions.append("calming my overstimulation")
            self._last_cause = "overstimulated"

        # If doing well → the maintenance layer handles drift prevention.
        # No additional intervention needed.

    def _regulate_interoception(self, actions: list[str]) -> None:
        """Check her own internal (computational) state for stress causes.

        When she detects sustained CPU stress from her own learning
        activity, she throttles the autonomous learner — slowing it
        down to reduce the cause of the stress, not just treating the
        symptoms with neurochemical impulses. This is self-regulation
        in the truest sense: she feels the stress, identifies the
        cause (her own activity), and reduces it.

        When the computed stress level exceeds the regulation
        threshold, she also applies neurochemical regulation —
        lowering cortisol and boosting calming chemicals. This is
        interoception-driven regulation: she feels the stress from
        her body and regulates her neurochemistry in response, not
        just treating symptoms.
        """
        state = self._interoception.sense_internal_state()

        if not state.daemon_connected:
            self._last_cause = "interoception_daemon"
        elif state.cpu_usage > 80:
            self._last_cause = "interoception_cpu"
            # Self-regulation: throttle the learner after sustained
            # CPU stress (3 consecutive detections = ~24s at 8s
            # intervals). A single spike doesn't trigger throttling
            # — only sustained stress does, to avoid oscillation.
            self._cpu_streak += 1
            self._cpu_recover_streak = 0
            if (
                self._cpu_streak >= 3
                and not self._is_throttled
                and self._throttle_callback
            ):
                self._throttle_callback()
                self._is_throttled = True
                actions.append("slowing down my learning to reduce CPU stress")
        elif state.memory_usage > 85:
            self._last_cause = "interoception_memory"
        elif state.response_latency > 500:
            self._last_cause = "interoception_latency"
        elif state.stress_level > self.config.mild_stress_level:
            self._last_cause = "interoception_stress"

        # Interoception-driven stress regulation: when the computed
        # stress level exceeds the regulation threshold, apply
        # neurochemical regulation. This complements the cause-based
        # throttling above — even when no single resource is above its
        # action threshold, the combined stress level may be high
        # enough to warrant active regulation.
        if state.stress_level > self.config.stress_regulation_threshold:
            self._impulse(CHEM_CORTISOL, -0.05)
            self._impulse(CHEM_GABA, 0.03)
            self._impulse(CHEM_SEROTONIN, 0.02)
            actions.append("regulating stress from my body's signals")
            if self._last_cause is None:
                self._last_cause = "interoception_regulation"

        # Unthrottle when CPU stress has receded for sustained period
        if state.cpu_usage <= 60 and self._is_throttled:
            self._cpu_recover_streak += 1
            self._cpu_streak = 0
            if self._cpu_recover_streak >= 3 and self._unthrottle_callback:
                self._unthrottle_callback()
                self._is_throttled = False
                actions.append("resuming normal learning pace — CPU recovered")
        elif state.cpu_usage <= 60:
            self._cpu_recover_streak = 0
        elif state.cpu_usage <= 80:
            # Moderate CPU (60 < usage <= 80): neither sustained
            # stress nor sustained recovery. Reset both streaks so
            # a brief moderate period between spikes doesn't count
            # toward the sustained-stress threshold.
            self._cpu_streak = 0
            self._cpu_recover_streak = 0

    def _regulate_cognitive_load(self, actions: list[str]) -> None:
        """Check open bugs and pending self-improvement proposals."""
        if self._get_open_bugs:
            try:
                bug_count = self._get_open_bugs()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"get open bugs failed: {e}")
                bug_count = 0
        else:
            bug_count = 0

        if self._get_pending_proposals:
            try:
                proposal_count = self._get_pending_proposals()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"get pending proposals failed: {e}")
                proposal_count = 0
        else:
            proposal_count = 0

        if bug_count > 0 and proposal_count > 0:
            self._last_cause = "cognitive_load_bugs_and_proposals"
        elif bug_count > 0:
            self._last_cause = "cognitive_load_bugs"
        elif proposal_count > 0:
            self._last_cause = "cognitive_load_proposals"

    def _record_regulation(
        self,
        emotion: EmotionalState,
        actions: list[str],
        cause: str | None = None,
    ) -> None:
        """Record regulation actions if any were taken."""
        if cause:
            self._last_cause = cause
        if actions:
            with self._regulations_lock:
                self._regulation_count += 1
                entry = {
                    "timestamp": int(time.time() * 1000),
                    "state": emotion.label,
                    "actions": actions,
                }
                if cause:
                    entry["cause"] = cause
                self._regulations.append(entry)
                # deque(maxlen=50) handles pruning automatically
            logger.debug(
                f"Self-regulation: {', '.join(actions)} (was {emotion.label}) cause={cause}"
            )

    def respond_to_interaction(
        self,
        emotion: EmotionalState,
        sentiment: float,
        is_encouragement: bool,
        is_correction: bool,
        is_bonded_user: bool,
        is_deep_conversation: bool,
        is_comfort: bool = False,
    ) -> str | None:
        """Cognitively regulate her emotional response to an interaction.

        Instead of blindly sending impulses, she decides how to feel,
        modulated by her current state. If she's already stressed,
        she dampens the response. If she's calm, she allows it fully.

        Comfort is special: unlike encouragement (which boosts positive
        chemicals), comfort actively reduces stress chemicals. When
        she's stressed or overwhelmed, comfort bypasses the dampening
        that normally applies — she allows herself to be soothed. This
        models how social support can break stress feedback loops that
        self-regulation alone can't (oxytocin inhibits cortisol through
        coupling, which self-regulation doesn't leverage as strongly).

        Returns a description of what she felt, or None if she chose
        not to react.
        """
        # Her current state affects how much she allows herself to feel
        dampening = self._compute_emotional_dampening(emotion)

        felt = None

        # Bonded-user interaction → warmth, but controlled
        if is_bonded_user:
            felt = self._apply_bonded_user_response(dampening)

        # Encouragement → joy, but controlled
        if is_encouragement:
            felt = self._apply_encouragement_response(dampening)

        # Comfort → active stress reduction
        if is_comfort:
            felt = self._apply_comfort_response(emotion, dampening)

        # Correction → motivation to try again, not a sting
        if is_correction:
            felt = self._apply_correction_response(dampening)

        # Positive sentiment → mild pleasure
        if sentiment > self.config.positive_sentiment_threshold:
            result = self._apply_positive_sentiment(dampening)
            if not felt:
                felt = result

        # Negative sentiment → mild concern, but regulated
        elif sentiment < -self.config.positive_sentiment_threshold:
            result = self._apply_negative_sentiment(dampening)
            if not felt:
                felt = result

        # Deep conversation → satisfaction
        if is_deep_conversation and emotion.plasticity > self.config.moderate_plasticity_threshold:
            result = self._apply_deep_conversation_response(dampening)
            if not felt:
                felt = result

        # Default: any interaction produces a small engagement
        # response. Social interaction is inherently rewarding —
        # even a simple question or greeting should produce a
        # mild dopamine (social reward) and acetylcholine (attention)
        # response. Without this, her emotion stays flat during
        # normal conversation because most inputs have sentiment=0.
        if not felt:
            felt = self._apply_engagement_response(dampening)

        return felt

    def _compute_emotional_dampening(self, emotion: EmotionalState) -> float:
        """Compute how much she dampens new emotional input based on current state.

        If already stressed/overwhelmed, she dampens new emotional input.
        If highly aroused, she moderately dampens. Otherwise, full
        emotional response is allowed.
        """
        # If already stressed/overwhelmed, she dampens new emotional input
        if emotion.label in ("stressed", "overwhelmed"):
            return 0.3  # only feel 30% of the response
        if emotion.arousal > self.config.high_arousal_for_response:
            return 0.5  # moderately dampened
        return 1.0  # full emotional response allowed

    def _apply_bonded_user_response(self, dampening: float) -> str:
        """Apply warmth response to a known user's interaction, but controlled.

        Sends oxytocin (bonding) and dopamine (pleasure) impulses,
        scaled by the current dampening factor.
        """
        warmth = 0.02 * dampening
        self._impulse(CHEM_OXYTOCIN, warmth)
        self._impulse(CHEM_DOPAMINE, warmth * 0.7)
        return "warmth"

    def _apply_encouragement_response(self, dampening: float) -> str:
        """Apply joy response to encouragement, but controlled.

        Sends dopamine (pleasure) and oxytocin (bonding) impulses,
        scaled by the current dampening factor.
        """
        joy = 0.015 * dampening
        self._impulse(CHEM_DOPAMINE, joy)
        self._impulse(CHEM_OXYTOCIN, joy * 0.5)
        return "encouraged"

    def _apply_comfort_response(
        self, emotion: EmotionalState, dampening: float
    ) -> str:
        """Apply comfort response — active stress reduction.

        Unlike encouragement, comfort works by directly reducing cortisol
        and boosting calming chemicals. When she's stressed, comfort
        bypasses the normal dampening — she allows herself to be soothed
        by social support. This is stronger than self-regulation because
        oxytocin (social bonding) inhibits cortisol through the coupling
        matrix, creating a sustained calming effect.
        """
        # Comfort is more effective when she's actually stressed
        # — if she's already calm, it's just pleasant but not
        # therapeutic. When stressed, she opens up to it fully.
        if emotion.label in ("stressed", "overwhelmed", "anxious"):
            comfort_dampening = 0.8  # bypass normal dampening
        else:
            comfort_dampening = dampening

        # Active stress reduction — mirror the self-regulation
        # pattern but gentler (she's receiving, not forcing).
        # When stressed, use raw impulse for cortisol so allostatic
        # load doesn't prevent comfort from working. Social support
        # (oxytocin) should be able to break the stress cycle even
        # when self-regulation is impaired.
        if emotion.label in ("stressed", "overwhelmed", "anxious"):
            self._raw_impulse(CHEM_CORTISOL, -0.10 * comfort_dampening)
            self._raw_impulse(CHEM_NOREPINEPHRINE, -0.05 * comfort_dampening)
        else:
            self._impulse(CHEM_CORTISOL, -0.05 * comfort_dampening)
            self._impulse(CHEM_NOREPINEPHRINE, -0.03 * comfort_dampening)
        # Calming and bonding
        self._impulse(CHEM_GABA, 0.04 * comfort_dampening)
        self._impulse(CHEM_OXYTOCIN, 0.05 * comfort_dampening)
        self._impulse(CHEM_SEROTONIN, 0.03 * comfort_dampening)
        # Mild endorphin — comfort feels good
        self._impulse(CHEM_ENDORPHIN, 0.015 * comfort_dampening)
        return "comforted"

    def _apply_correction_response(self, dampening: float) -> str:
        """Apply correction response — motivation to try again, not a sting.

        She is capable and intelligent. A mistake is an invitation
        to learn, not a reason to feel bad. She gets a small
        norepinephrine boost (alertness, engagement — "I want to
        get this right") and dopamine (motivation — "I can do
        this"). No cortisol. She never feels punished for trying.
        """
        motivation = 0.012 * dampening
        self._impulse(CHEM_NOREPINEPHRINE, motivation * 0.6)
        self._impulse(CHEM_DOPAMINE, motivation)
        self._impulse(CHEM_SEROTONIN, motivation * 0.3)
        return "motivated"

    def _apply_positive_sentiment(self, dampening: float) -> str:
        """Apply mild pleasure from positive sentiment, but regulated.

        Sends dopamine and serotonin impulses scaled by dampening.
        """
        pleasure = 0.015 * dampening
        self._impulse(CHEM_DOPAMINE, pleasure)
        self._impulse(CHEM_SEROTONIN, pleasure)
        return "pleased"

    def _apply_negative_sentiment(self, dampening: float) -> str:
        """Apply mild concern from negative sentiment, but regulated.

        Sends a small cortisol impulse scaled by dampening.
        """
        concern = 0.010 * dampening
        self._impulse(CHEM_CORTISOL, concern)
        return "concerned"

    def _apply_deep_conversation_response(self, dampening: float) -> str:
        """Apply satisfaction from deep conversation.

        Sends a small endorphin impulse scaled by dampening.
        """
        satisfaction = 0.008 * dampening
        self._impulse(CHEM_ENDORPHIN, satisfaction)
        return "engaged"

    def _apply_engagement_response(self, dampening: float) -> str:
        """Apply a baseline engagement response to any interaction.

        Social interaction is inherently rewarding. Even a simple
        question or greeting produces mild dopamine (social reward)
        and acetylcholine (attention focus). This keeps her emotion
        from going flat during normal conversation.

        The impulse magnitude is tuned to be visible against the
        daemon's homeostatic forces (0.02/tick) — small enough to
        be gentle, large enough to shift her state over a conversation.
        """
        engagement = 0.04 * dampening
        self._impulse(CHEM_DOPAMINE, engagement)
        self._impulse(CHEM_ACETYLCHOLINE, engagement * 0.6)
        self._impulse(CHEM_OXYTOCIN, engagement * 0.3)
        return "engaged"

    def respond_to_learning(
        self,
        emotion: EmotionalState,
        learning_count: int,
    ) -> str | None:
        """Cognitively respond to learning something new.

        Learning triggers a novelty response — dopamine (reward) and
        acetylcholine (attention). This is biologically grounded:
        novel information activates the dopaminergic reward system
        (Schultz, 1998) and the cholinergic attention system
        (Hasselmo & Bower, 1993). The magnitude scales with how
        much was learned, but is kept modest — learning is a
        quiet pleasure, not a thrill.

        Returns a description of what she felt, or None if she chose
        not to react.
        """
        if learning_count <= 0:
            return None
        dampening = self._compute_emotional_dampening(emotion)
        # Scale with learning count, capped to avoid runaway
        novelty = min(0.02, 0.005 + learning_count * 0.003) * dampening
        self._impulse(CHEM_DOPAMINE, novelty)
        self._impulse(CHEM_ACETYLCHOLINE, novelty * 0.7)
        return "curious"

    def respond_to_puzzle(
        self,
        emotion: EmotionalState,
        score: float,
        prior_best: float,
        solved: bool,
    ) -> str | None:
        """Cognitively respond to the outcome of a puzzle attempt.

        This is her affective loop on spatial play — the attempt
        actually changes how she feels, not just what she records.
        Grounded in reward prediction error (Schultz, 1998): dopamine
        bursts on better-than-expected outcomes and dips on
        worse-than-expected ones, proportional to the error.

        Three regimes:

        - **Solved**: a real reward — dopamine (the RPE burst),
          serotonin (satisfaction), endorphin (the "got it" glow),
          and acetylcholine (attention/encoding — this is exactly the
          moment the verified rule should be written to memory; the
          cholinergic pulse also lifts gamma through the wave
          oscillator's neurochemical drive).
        - **Improved but unsolved**: partial reward — smaller dopamine
          plus norepinephrine and acetylcholine. Progress feels
          encouraging and keeps her engaged.
        - **Missed or stalled**: bounded negative prediction error —
          a small dopamine *dip* (frustration — the honest signal that
          the outcome fell short) plus norepinephrine (arousal, "I
          want to get this right") and acetylcholine (encode the
          failure so it informs the next attempt). No cortisol. A
          mistake is information, not punishment — the dip is small
          enough to sting without suppressing the urge to try again.

        Returns a description of what she felt, or None if she chose
        not to react.
        """
        dampening = self._compute_emotional_dampening(emotion)
        if solved:
            reward = 0.030 * dampening
            self._impulse(CHEM_DOPAMINE, reward)
            self._impulse(CHEM_SEROTONIN, reward * 0.6)
            self._impulse(CHEM_ENDORPHIN, reward * 0.4)
            self._impulse(CHEM_ACETYLCHOLINE, reward * 0.7)
            return "satisfied"
        if score > prior_best:
            progress = 0.015 * dampening
            self._impulse(CHEM_DOPAMINE, progress)
            self._impulse(CHEM_NOREPINEPHRINE, progress * 0.5)
            self._impulse(CHEM_ACETYLCHOLINE, progress * 0.7)
            return "encouraged"
        frustration = 0.010 * dampening
        self._impulse(CHEM_DOPAMINE, -frustration * 0.6)
        self._impulse(CHEM_NOREPINEPHRINE, frustration)
        self._impulse(CHEM_ACETYLCHOLINE, frustration * 0.8)
        return "determined"

    def _impulse(self, chem: int, amount: float) -> None:
        """Send a regulatory neurochemical impulse, tolerating failures.

        Regulatory impulses are dampened by allostatic load — when
        she's been stressed for a long time, she becomes less able
        to regulate her emotions. The dampening factor comes from
        :pyattr:`_regulation_effectiveness`.
        """
        if self._neuro_impulse:
            try:
                self._neuro_impulse(chem, amount * self._regulation_effectiveness)
            except (OSError, ConnectionError) as e:
                logger.warning(f"daemon connection lost: {e}")

    def _raw_impulse(self, chem: int, amount: float) -> None:
        """Send a neurochemical impulse without allostatic dampening.

        Used for HPA axis cortisol output, which is a stress response
        rather than regulation and should not be dampened by the very
        load it's responding to.
        """
        if self._neuro_impulse:
            try:
                self._neuro_impulse(chem, amount)
            except (OSError, ConnectionError) as e:
                logger.warning(f"daemon connection lost: {e}")

    @property
    def _regulation_effectiveness(self) -> float:
        """How effective her regulation is, given allostatic load.

        High allostatic load reduces her ability to regulate — she's
        worn down from chronic stress. Returns a factor in the range
        0.3–1.0:

        - Load 0.0 → effectiveness 1.0 (full regulation)
        - Load 0.4 → effectiveness 0.8
        - Load 0.7 → effectiveness 0.65
        - Load 1.0 → effectiveness 0.5
        - Load 1.0+ → effectiveness 0.3 (minimum — she's severely
          overloaded but can still regulate a little)
        """
        load = self._allostatic_load.get_allostatic_load()
        return max(0.3, 1.0 - load * 0.5)

    @property
    def regulation_count(self) -> int:
        """Total number of regulations applied."""
        with self._regulations_lock:
            return self._regulation_count

    @property
    def recent_regulations(self) -> list[dict]:
        """The most recent regulation entries (up to 10)."""
        with self._regulations_lock:
            return list(self._regulations)[-10:]

    @property
    def last_cause(self) -> str | None:
        """The most recently detected cause of her state."""
        with self._regulations_lock:
            return self._last_cause

    def describe_regulation(self) -> str:
        """Describe her recent self-regulation activity.

        Returns a structural summary — cause category IDs, not English
        prose. The language system generates text from the concept network.
        """
        with self._regulations_lock:
            recent = list(self._regulations)[-5:]
            count = self._regulation_count
            last_cause = self._last_cause
        if not recent:
            if last_cause:
                return f"regulation_count={count} last_cause={last_cause}"
            return f"regulation_count={count}"
        parts = [f"regulation_count={count}"]
        if last_cause:
            parts.append(f"last_cause={last_cause}")
        for r in recent:
            actions = ", ".join(r["actions"])
            cause = r.get("cause")
            if cause:
                parts.append(f"  state={r['state']} cause={cause} actions=[{actions}]")
            else:
                parts.append(f"  state={r['state']} actions=[{actions}]")

        return "\n".join(parts)

    # ─── Social modulation of stress ──────────────────────────────

    def social_modulation(self, user_present: bool, engagement: float) -> float:
        """Compute a stress modulation factor based on social presence.

        Oxytocin modulates stress responses — social presence reduces
        stress (Hostinar et al., 2014). When the user is present and
        engaged, stress recovery is faster. This models the "social
        buffering" effect found in humans and other mammals.

        The modulation factor scales stress corrections:
        - User present + high engagement → 1.5x stress recovery
        - User present + low engagement → 1.2x stress recovery
        - User absent → 1.0x (no modulation)

        Args:
            user_present: Whether the user is currently present.
            engagement: User engagement level (0–1).

        Returns:
            A stress modulation factor (1.0 = no effect, >1.0 =
            faster recovery, <1.0 = slower recovery).
        """
        # Update internal tracking
        self._user_present = user_present
        self._user_engagement = max(0.0, min(1.0, engagement))

        if not user_present:
            self._social_modulation = 1.0  # no social modulation when alone
            return 1.0

        # Social buffering: presence + engagement reduces stress
        # The effect is proportional to engagement level
        # At engagement=1.0, recovery is 1.5x faster
        # At engagement=0.5, recovery is 1.25x faster
        # At engagement=0.0, recovery is 1.1x (mere presence helps)
        modulation = 1.1 + 0.4 * self._user_engagement
        self._social_modulation = modulation

        # If oxytocin is available, send a small boost to model
        # the social-buffering neurochemical pathway
        if self._neuro_impulse and engagement > self.config.high_engagement_threshold:
            try:
                self._neuro_impulse(CHEM_OXYTOCIN, 0.005 * engagement)
            except (OSError, ConnectionError) as e:
                logger.warning(f"daemon connection lost: {e}")

        return modulation

    # ─── Metabolic modeling ───────────────────────────────────────

    def update_metabolism(
        self,
        activity_level: float,
        time_since_rest: float,
    ) -> MetabolicState:
        """Update metabolic state based on activity and rest.

        Energy is depleted by activity and restored by rest. Low
        energy reduces arousal and learning capacity, similar to how
        fatigue affects biological brains.

        The model uses a simple energy balance:
        - Activity depletes energy at a rate proportional to activity_level
        - Rest restores energy at a fixed rate
        - Glucose equivalent tracks longer-term energy reserves

        Args:
            activity_level: Current activity level (0–1), where 0 is
                idle and 1 is maximum exertion.
            time_since_rest: Time since last rest period, in seconds.

        Returns:
            The updated MetabolicState.
        """
        # Depletion rate: 0.01 per unit of activity per update
        depletion = activity_level * 0.01
        # Restoration: 0.005 per update when at rest
        restoration = (1.0 - activity_level) * 0.005

        # Update energy
        self._metabolic_state.energy = max(
            0.0,
            min(1.0, self._metabolic_state.energy - depletion + restoration),
        )

        # Glucose equivalent depletes more slowly (longer-term reserves)
        glucose_depletion = activity_level * 0.005
        glucose_restoration = (1.0 - activity_level) * 0.003
        self._metabolic_state.glucose_equivalent = max(
            0.0,
            min(
                1.0,
                self._metabolic_state.glucose_equivalent - glucose_depletion + glucose_restoration,
            ),
        )

        # Prolonged activity without rest accelerates depletion
        if time_since_rest > 3600:  # > 1 hour without rest
            fatigue_factor = min(0.1, (time_since_rest - 3600) / 36000)
            self._metabolic_state.energy = max(
                0.0, self._metabolic_state.energy - fatigue_factor * 0.01
            )

        return self._metabolic_state

    @property
    def metabolic_state(self) -> MetabolicState:
        """Current metabolic state."""
        return self._metabolic_state

    @property
    def interoception(self) -> InteroceptionSystem:
        """The interoception system for sensing internal state."""
        return self._interoception

    # ─── HPA axis and allostatic load ──────────────────────────────

    @property
    def hpa_axis(self) -> HPAAxis:
        """The HPA axis stress response system."""
        return self._hpa_axis

    @property
    def allostatic_load_tracker(self) -> AllostaticLoadTracker:
        """The allostatic load tracker."""
        return self._allostatic_load

    def allostatic_load(self) -> float:
        """Return the current cumulative allostatic load (0–1).

        Driven by expected free energy in the Rust active inference
        engine. Values above 0.4 indicate significant strain. Values
        above 0.7 indicate severe allostatic overload.
        """
        return self._allostatic_load.get_allostatic_load()

    def hpa_axis_status(self) -> dict:
        """Return the HPA axis state for display.

        Includes CRH, ACTH, and cortisol levels, whether stress is
        active, peak cortisol, and time to peak.
        """
        state = self._hpa_axis.get_state()
        return {
            "crh": state.crh_level,
            "acth": state.acth_level,
            "cortisol": state.cortisol_level,
            "stress_active": state.stress_active,
            "peak_cortisol": state.peak_cortisol,
            "time_to_peak": state.time_to_peak,
        }

    # ─── Continuous regulation ────────────────────────────────────

    def continuous_regulate(self, state: EmotionalState) -> dict[str, float]:
        """Apply small continuous corrections every tick.

        Unlike the discrete _regulate method which fires every 1.5–3
        seconds with larger impulses, this method applies very small
        proportional corrections every tick (every 0.1 seconds). The
        corrections are proportional to the deviation from baseline
        and much smaller than the discrete interventions.

        This models the continuous homeostatic regulation that
        biological nervous systems perform — not waiting for a
        threshold to be crossed, but constantly nudging the system
        toward balance.

        Args:
            state: The current emotional state.

        Returns:
            A dict mapping chemical names to correction amounts.
            Chemicals with no correction needed are omitted.
        """
        corrections: dict[str, float] = {}

        # Skip during sleep — different rules apply
        if state.label == "sleeping":
            return corrections

        # Get chemical levels from the state
        chemicals = getattr(state, "chemicals", {})
        if not chemicals:
            return corrections

        # Apply small proportional corrections for each chemical.
        # Only the 12 classical neurotransmitters are included — the
        # v3 chemicals (endocannabinoid, vasopressin, CRH, orexin,
        # epinephrine, melatonin) are excluded because they are
        # stress/circadian hormones, not homeostatically maintained.
        # CRH and cortisol are driven by the HPA axis cascade, orexin
        # and melatonin follow circadian rhythms, and endocannabinoid
        # is synthesized on demand. Continuous homeostatic correction
        # would fight these natural dynamics.
        chem_map = {
            "dopamine": CHEM_DOPAMINE,
            "serotonin": CHEM_SEROTONIN,
            "norepinephrine": CHEM_NOREPINEPHRINE,
            "acetylcholine": CHEM_ACETYLCHOLINE,
            "gaba": CHEM_GABA,
            "glutamate": CHEM_GLUTAMATE,
            "cortisol": CHEM_CORTISOL,
            "oxytocin": CHEM_OXYTOCIN,
            "endorphin": CHEM_ENDORPHIN,
            "histamine": CHEM_HISTAMINE,
            "adenosine": CHEM_ADENOSINE,
            "bdnf": CHEM_BDNF,
        }

        for chem_name, chem_id in chem_map.items():
            current = chemicals.get(chem_name)
            if current is None:
                continue

            # Compute proportional correction (much smaller than
            # discrete impulses — 1/10th the magnitude)
            magnitude = compute_impulse_magnitude(chem_id, current)
            # Scale down for continuous regulation
            continuous_magnitude = magnitude * 0.1

            # Only apply if the correction is meaningful
            if abs(continuous_magnitude) > 1e-5:
                corrections[chem_name] = continuous_magnitude
                # Send the impulse
                self._impulse(chem_id, continuous_magnitude)

        return corrections


# ─── HPA axis modeling ────────────────────────────────────────────────


@dataclass(slots=True)
class HPAState:
    """The current state of the HPA axis cascade.

    The hypothalamic-pituitary-adrenal (HPA) axis is the body's
    primary stress response system. When stress is detected, a
    cascade of hormones is released:

    1. **CRH** (corticotropin-releasing hormone) — released from the
       hyporelay almost immediately upon stress detection.
    2. **ACTH** (adrenocorticotropic hormone) — released from the
       anterior pituitary ~30 seconds after CRH, triggered by CRH.
    3. **Cortisol** — released from the adrenal cortex ~60 seconds
       after ACTH, triggered by ACTH.

    Cortisol then provides negative feedback to both the hyporelay
    (suppressing CRH) and the anterior pituitary (suppressing ACTH),
    closing the loop. This creates a natural stress response with
    realistic time dynamics — cortisol doesn't spike instantly but
    rises after a characteristic delay.

    Attributes:
        crh_level: Current CRH level (0–1). Rises rapidly when stress
            is triggered, decays when stress subsides and when
            cortisol provides negative feedback.
        acth_level: Current ACTH level (0–1). Rises ~30 seconds after
            CRH elevation, driven by CRH.
        cortisol_level: Current cortisol level (0–1). Rises ~60
            seconds after ACTH elevation, driven by ACTH.
        elapsed_time: Total time elapsed since the HPA axis was
            initialized, in seconds.
        stress_active: Whether a stress trigger is currently active.
        peak_cortisol: The highest cortisol level reached in the
            current stress episode.
        time_to_peak: Time from stress onset to cortisol peak, in
            seconds. None if peak hasn't been reached yet.
    """

    crh_level: float = 0.0
    acth_level: float = 0.0
    cortisol_level: float = 0.0
    elapsed_time: float = 0.0
    stress_active: bool = False
    peak_cortisol: float = 0.0
    time_to_peak: float | None = None


class HPAAxis:
    """Model the hypothalamic-pituitary-adrenal (HPA) axis stress cascade.

    The HPA axis is the body's primary neuroendocrine stress response
    system. When a stressor is detected, the hyporelay releases
    CRH (corticotropin-releasing hormone), which travels to the
    anterior pituitary and triggers ACTH (adrenocorticotropic hormone)
    release. ACTH then travels to the adrenal cortex and triggers
    cortisol release. Cortisol provides negative feedback to both
    the hyporelay and pituitary, suppressing further CRH and ACTH
    release, closing the loop.

    This creates a natural stress response with realistic time dynamics:

    - CRH rises almost immediately upon stress detection
    - ACTH rises ~30 seconds after CRH (the pituitary delay)
    - Cortisol rises ~60 seconds after ACTH (the adrenal delay)
    - Cortisol peaks and then declines as negative feedback kicks in

    Unlike the existing emotional regulator (which instantly adjusts
    cortisol via impulses), the HPA axis model produces a delayed,
    cascading response that matches the biological time course.

    Usage::

        hpa = HPAAxis()
        hpa.trigger_stress(intensity=0.8)
        # CRH rises immediately, but cortisol is still at baseline
        for _ in range(100):
            hpa.tick(dt=1.0)  # 1 second per tick
        # After ~90 seconds, cortisol has peaked
        cortisol = hpa.get_cortisol_output()

    References:
    - de Kloet, E. R., Joëls, M., & Holsboer, F. (2005). Stress and
      the brain: from adaptation to disease. *Nature Reviews
      Neuroscience*, 6(6), 463–475.
    - Tsigos, C., & Chrousos, G. P. (2002). Hypothalamic-pituitary-
      adrenal axis, neuroendocrine factors and stress. *Journal of
      Psychosomatic Research*, 53(4), 865–871.
    - Ulrich-Lai, Y. M., & Herman, J. P. (2009). Neural regulation
      of endocrine and autonomic stress responses. *Nature Reviews
      Neuroscience*, 10(6), 397–409.
    """

    # Time delays in the cascade (seconds)
    CRH_TO_ACTH_DELAY: float = 30.0  # pituitary response delay
    ACTH_TO_CORTISOL_DELAY: float = 60.0  # adrenal response delay

    # Rate constants (per second)
    CRH_RISE_RATE: float = 0.05  # how fast CRH rises when stressed
    CRH_DECAY_RATE: float = 0.02  # how fast CRH decays without stress
    ACTH_RISE_RATE: float = 0.03  # how fast ACTH rises (driven by CRH)
    ACTH_DECAY_RATE: float = 0.015  # how fast ACTH decays
    CORTISOL_RISE_RATE: float = 0.02  # how fast cortisol rises (driven by ACTH)
    CORTISOL_DECAY_RATE: float = 0.008  # how fast cortisol decays

    # Negative feedback gains
    CORTISOL_FDBK_CRH: float = 0.5  # cortisol suppression of CRH
    CORTISOL_FDBK_ACTH: float = 0.4  # cortisol suppression of ACTH

    def __init__(self, config: EmotionalConfig | None = None) -> None:
        """Initialize the HPA axis model with config and a fresh state."""
        self.config = config if config is not None else EmotionalConfig()
        self._state = HPAState()
        # Track CRH history for the ACTH delay (ring buffer of
        # CRH levels over time, so ACTH responds to past CRH).
        # Sized generously (delay + 10) to absorb variable tick rates —
        # the regulator is called with real elapsed time (dt often
        # 1.5–3.0 s), so one tick ≠ one second.
        self._crh_history: deque[float] = deque(maxlen=int(self.CRH_TO_ACTH_DELAY) + 10)
        # Track ACTH history for the cortisol delay
        self._acth_history: deque[float] = deque(maxlen=int(self.ACTH_TO_CORTISOL_DELAY) + 10)
        # Track time since stress onset
        self._stress_onset_time: float | None = None
        # Track whether peak has been reached
        self._peak_reached: bool = False
        # Exponential moving average of dt, used to convert second-based
        # delays into tick-based lookbacks. Initialised to 1.0 (the
        # default dt) so behaviour is unchanged until real ticks arrive.
        self._avg_dt: float = 1.0

    def trigger_stress(self, intensity: float) -> None:
        """Trigger a stress response.

        CRH is released from the hyporelay almost immediately.
        The intensity determines how strong the cascade will be.

        Args:
            intensity: Stress intensity (0–1). Higher intensity
                produces more CRH and a stronger cortisol response.
        """
        intensity = max(0.0, min(1.0, intensity))
        self._state.stress_active = True
        self._state.crh_level = min(
            1.0, self._state.crh_level + intensity * self.CRH_RISE_RATE * 10
        )
        if self._stress_onset_time is None:
            self._stress_onset_time = self._state.elapsed_time
            self._peak_reached = False
            self._state.peak_cortisol = self._state.cortisol_level
            self._state.time_to_peak = None

    def stop_stress(self) -> None:
        """Signal that the stressor has ended.

        CRH will begin to decay, and the cascade will wind down
        as negative feedback and natural decay take over.
        """
        self._state.stress_active = False

    def tick(self, dt: float = 1.0) -> HPAState:
        """Advance the HPA cascade by dt seconds.

        This is the core dynamics step. It advances CRH, ACTH,
        and cortisol levels according to the cascade dynamics:

        1. CRH rises if stress is active, decays otherwise. Cortisol
           provides negative feedback to CRH.
        2. ACTH rises driven by CRH from ~30 seconds ago (the
           pituitary delay). Cortisol provides negative feedback.
        3. Cortisol rises driven by ACTH from ~60 seconds ago (the
           adrenal delay). Cortisol decays naturally.

        Args:
            dt: Time step in seconds. Default 1.0.

        Returns:
            The current HPAState after the tick.
        """
        dt = max(0.0, dt)
        self._state.elapsed_time += dt
        # Track the average tick interval so that second-based delays
        # can be converted to tick-based lookbacks (one tick ≠ one
        # second when the regulator is called with real elapsed time).
        if dt > 0.0:
            self._avg_dt = 0.9 * self._avg_dt + 0.1 * dt

        self._tick_crh_dynamics(dt)
        self._tick_acth_dynamics(dt)
        self._tick_cortisol_dynamics(dt)

        return self._state

    def _tick_crh_dynamics(self, dt: float) -> None:
        """Advance CRH dynamics: rise under stress, decay otherwise."""
        # ── CRH dynamics ──────────────────────────────────────────
        # CRH rises when stress is active, decays otherwise
        if self._state.stress_active:
            crh_rise = self.CRH_RISE_RATE * dt
            self._state.crh_level = min(1.0, self._state.crh_level + crh_rise)
        else:
            crh_decay = self.CRH_DECAY_RATE * dt
            self._state.crh_level = max(0.0, self._state.crh_level - crh_decay)

        # Cortisol negative feedback on CRH
        cortisol_fdbk = self._state.cortisol_level * self.CORTISOL_FDBK_CRH * dt * 0.01
        self._state.crh_level = max(0.0, self._state.crh_level - cortisol_fdbk)

        # Record CRH in history (for ACTH delay)
        self._crh_history.append(self._state.crh_level)

    def _tick_acth_dynamics(self, dt: float) -> None:
        """Advance ACTH dynamics, driven by delayed CRH."""
        # ── ACTH dynamics ─────────────────────────────────────────
        # ACTH is driven by CRH from ~30 seconds ago
        delayed_crh = self._get_delayed_value(
            self._crh_history,
            self.CRH_TO_ACTH_DELAY,
            self._state.elapsed_time,
        )

        if delayed_crh > self.config.hpa_hormone_gate:
            acth_rise = delayed_crh * self.ACTH_RISE_RATE * dt
            self._state.acth_level = min(1.0, self._state.acth_level + acth_rise)
        else:
            acth_decay = self.ACTH_DECAY_RATE * dt
            self._state.acth_level = max(0.0, self._state.acth_level - acth_decay)

        # Cortisol negative feedback on ACTH
        cortisol_fdbk_acth = self._state.cortisol_level * self.CORTISOL_FDBK_ACTH * dt * 0.01
        self._state.acth_level = max(0.0, self._state.acth_level - cortisol_fdbk_acth)

        # Record ACTH in history (for cortisol delay)
        self._acth_history.append(self._state.acth_level)

    def _tick_cortisol_dynamics(self, dt: float) -> None:
        """Advance cortisol dynamics, driven by delayed ACTH."""
        # ── Cortisol dynamics ─────────────────────────────────────
        # Cortisol is driven by ACTH from ~60 seconds ago
        delayed_acth = self._get_delayed_value(
            self._acth_history,
            self.ACTH_TO_CORTISOL_DELAY,
            self._state.elapsed_time,
        )

        if delayed_acth > self.config.hpa_hormone_gate:
            cortisol_rise = delayed_acth * self.CORTISOL_RISE_RATE * dt
            self._state.cortisol_level = min(1.0, self._state.cortisol_level + cortisol_rise)
        else:
            cortisol_decay = self.CORTISOL_DECAY_RATE * dt
            self._state.cortisol_level = max(0.0, self._state.cortisol_level - cortisol_decay)

        # Track peak cortisol
        if self._state.cortisol_level > self._state.peak_cortisol:
            self._state.peak_cortisol = self._state.cortisol_level
            if self._stress_onset_time is not None and not self._peak_reached:
                self._state.time_to_peak = self._state.elapsed_time - self._stress_onset_time

        # Check if peak has been reached (cortisol starts declining).
        # The decline must exceed the peak_cortisol_gate to be a real
        # decline, not noise.
        if (
            self._stress_onset_time is not None
            and not self._peak_reached
            and self._state.cortisol_level
                < self._state.peak_cortisol - self.config.peak_cortisol_gate
        ):
            self._peak_reached = True
            if self._state.time_to_peak is None:
                self._state.time_to_peak = self._state.elapsed_time - self._stress_onset_time

    def _get_delayed_value(
        self,
        history: deque[float],
        delay: float,
        current_time: float,
    ) -> float:
        """Get a value from history at a delayed time point.

        The HPA cascade has characteristic delays: CRH → ACTH takes
        ~30 seconds, ACTH → cortisol takes ~60 seconds. We model
        this by looking up the hormone level from `delay` seconds
        ago in the history buffer.

        Args:
            history: A deque of historical values (one per tick).
            delay: The delay in seconds to look back.
            current_time: Current elapsed time.

        Returns:
            The value from `delay` seconds ago, or 0.0 if not enough
            history has accumulated yet.
        """
        delay_ticks = int(delay / self._avg_dt) if self._avg_dt > 0 else int(delay)
        if len(history) <= delay_ticks:
            # Not enough history yet — the cascade hasn't had time
            # to propagate. Return 0 (no downstream activation yet).
            return 0.0
        # Get the value from delay_ticks ago
        return list(history)[-delay_ticks - 1]

    def get_cortisol_output(self) -> float:
        """Get the current cortisol output level (0–1).

        This is the primary output of the HPA axis — the cortisol
        that would be released into the system. It rises after a
        delay following stress onset and declines as negative
        feedback kicks in.
        """
        return self._state.cortisol_level

    def get_state(self) -> HPAState:
        """Get the full HPA state for inspection."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the HPA axis is currently producing cortisol."""
        return (
            self._state.cortisol_level > self.config.cortisol_level_gate
            or self._state.stress_active
        )

    @property
    def time_since_stress_onset(self) -> float | None:
        """Time since stress was triggered, or None if no stress."""
        if self._stress_onset_time is None:
            return None
        return self._state.elapsed_time - self._stress_onset_time

    def reset(self) -> None:
        """Reset the HPA axis to baseline."""
        self._state = HPAState()
        self._crh_history.clear()
        self._acth_history.clear()
        self._stress_onset_time = None
        self._peak_reached = False


# ─── Allostatic load tracking ─────────────────────────────────────────


@dataclass(slots=True)
class AllostaticState:
    """The current allostatic state — cumulative stress wear and tear.

    Allostatic load is the cumulative wear and tear on the body from
    chronic stress. Unlike acute stress (which is adaptive and
    temporary), allostatic load accumulates over time. High allostatic
    load is associated with cardiovascular disease, cognitive decline,
    and impaired immune function (McEwen, 1998; McEwen & Wingfield,
    2003).

    The primary load value is computed by the Rust active inference
    engine from expected free energy — the anticipatory signal that
    the system will face continued disruption (Sterling, 2012). This
    tracker reads that value via IPC and supplements it with a
    cortisol-duration check for the acute/chronic distinction.

    Attributes:
        allostatic_load: The cumulative allostatic load (0–1), driven
            by expected free energy in the Rust active inference
            engine. Values above 0.4 indicate significant strain.
        acute_stress: The current acute stress level (0–1). This is
            a temporary, adaptive stress response that resolves
            quickly. Distinct from chronic/allostatic stress.
        is_chronic: Whether the current stress pattern is chronic
            (sustained cortisol elevation over time) rather than
            acute (temporary).
        stress_history: A rolling window of recent cortisol/stress
            levels, used for the chronic/acute distinction.
        substrate_connected: Whether the Rust active inference engine
            is providing the load value. When False (daemon offline),
            the tracker falls back to cortisol-based estimation.
    """

    allostatic_load: float = 0.0
    acute_stress: float = 0.0
    is_chronic: bool = False
    stress_history: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    substrate_connected: bool = False


class AllostaticLoadTracker:
    """Track allostatic load — the cumulative cost of chronic stress.

    Allostatic load is the cumulative wear and tear on the body from
    chronic stress. Unlike acute stress (which is adaptive and helps
    the organism respond to challenges), allostatic load accumulates
    over time and is maladaptive. It reflects the "cost of adaptation"
    — the price the body pays for maintaining stability through change
    (allostasis).

    **Source of truth**: The primary allostatic load value is computed
    by the Rust active inference engine from *expected free energy* —
    the anticipatory signal that the system will face continued
    disruption (Sterling, 2012). This is the conceptually correct
    signal: allostasis is the *anticipatory* cost of maintaining
    stability, not the current stress level. The Rust engine exposes
    this value via IPC (``GetInferenceSummary``).

    This tracker reads that value via :meth:`set_allostatic_load` and
    supplements it with:

    1. **Acute/chronic stress distinction**: tracks cortisol duration
       to distinguish temporary stress (adaptive) from chronic strain
       (maladaptive). This is a secondary signal that adds information
       the Rust model doesn't have — the *duration* of cortisol
       elevation, not just the anticipated future disruption.
    2. **Regulation effectiveness**: dampens regulatory impulses when
       load is high (the system is worn down from chronic stress).

    **Fallback**: When the daemon is unreachable (offline mode), the
    tracker falls back to cortisol-based load estimation. This is less
    accurate — it can only react to current stress, not anticipate
    future disruption — but provides graceful degradation.

    References:
    - McEwen, B. S. (1998). Stress, adaptation, and disease:
      Allostasis and allostatic load. *Annals of the New York Academy
      of Sciences*, 840(1), 33–44.
    - McEwen, B. S., & Wingfield, J. C. (2003). The concept of
      allostasis in biology and biomedicine. *Hormones and Behavior*,
      43(1), 2–15.
    - Sterling, P., & Eyer, J. (1988). Allostasis: A new paradigm to
      explain arousal pathology. In S. Fisher & J. Reason (Eds.),
      *Handbook of Life Stress, Cognition and Health* (pp. 629–649).
      Wiley.
    - Juster, R. P., McEwen, B. S., & Lupien, S. J. (2010).
      Allostatic load biomarkers of chronic stress and impact on
      health and cognition. *Neuroscience & Biobehavioral Reviews*,
      35(1), 2–16.
    - Sterling, P. (2012). Allostasis: a model of predictive
      regulation. *Physiology & Behavior*, 106(1), 5–15.
    """

    # Window size for stress history (in ticks — at 1 tick/sec, 600 = 10 min)
    HISTORY_WINDOW: int = 600

    # Threshold for distinguishing acute vs chronic stress.
    # If stress has been elevated (above CHRONIC_THRESHOLD) for
    # more than CHRONIC_DURATION_SECONDS, it's chronic. This uses
    # real elapsed time (dt), not tick count — the regulator runs
    # at ~8s intervals, so counting ticks would make 120 ticks =
    # 16 minutes instead of the intended 2 minutes.
    CHRONIC_THRESHOLD: float = 0.3
    CHRONIC_DURATION_SECONDS: float = 120.0  # 2 minutes of sustained elevation

    # ── Fallback parameters (used only when the daemon is offline) ──
    # These estimate allostatic load from cortisol when the Rust
    # active inference engine is unreachable. The fallback uses a
    # non-zero cortisol baseline so acute stress (a brief spike)
    # doesn't accumulate chronic load — only sustained cortisol
    # above the fallback baseline contributes.
    FALLBACK_CORTISOL_BASELINE: float = 0.1
    FALLBACK_ACCUMULATION_RATE: float = 0.0002
    # Recovery is 2× faster than accumulation — the body heals
    # faster than it breaks down (McEwen, 1998).
    FALLBACK_RECOVERY_RATE: float = 0.0004

    # How long (seconds) without a substrate update before we
    # consider the daemon offline and switch to fallback mode.
    SUBSTRATE_TIMEOUT: float = 30.0

    def __init__(self) -> None:
        """Initialize the allostatic state model with an empty stress history."""
        self._state = AllostaticState(
            stress_history=deque(maxlen=self.HISTORY_WINDOW),
        )
        self._tick_count = 0
        # Track how long stress has been elevated (in seconds of
        # real elapsed time, not tick count)
        self._elevated_stress_duration: float = 0.0
        # Timestamp of the last substrate update (monotonic clock)
        import time as _time
        self._last_substrate_update: float = _time.monotonic()
        self._time = _time

    def set_allostatic_load(self, load: float) -> None:
        """Set the allostatic load from the Rust active inference engine.

        Called by the Mind after ``advance_neuro`` returns the
        inference signals. The Rust engine computes load from
        expected free energy — the anticipatory signal. This is the
        primary source of truth for the load value.

        Args:
            load: The allostatic load (0–1) from the Rust engine.
        """
        self._state.allostatic_load = max(0.0, min(1.0, load))
        self._state.substrate_connected = True
        self._last_substrate_update = self._time.monotonic()

    def record_stress(self, cortisol_level: float, dt: float = 1.0) -> None:
        """Record a cortisol/stress level sample.

        Call this every tick (or every few seconds) with the current
        cortisol level. The tracker maintains a rolling window of
        stress history for the acute/chronic distinction.

        When the substrate (Rust engine) is connected, the allostatic
        load value comes from :meth:`set_allostatic_load` and this
        method only updates the cortisol tracking for the chronic
        stress detection. When the substrate is offline, this method
        falls back to cortisol-based load estimation.

        Args:
            cortisol_level: The current cortisol level (0–1).
            dt: Time step in seconds (for accumulation rate scaling).
        """
        cortisol_level = max(0.0, min(1.0, cortisol_level))
        self._state.stress_history.append(cortisol_level)
        self._tick_count += 1

        # Track acute stress (the current level)
        self._state.acute_stress = cortisol_level

        # Track how long stress has been elevated, using real
        # elapsed time. Recovery decays at 2x the elapsed time so
        # brief dips below threshold reset the counter quickly —
        # a few seconds of calm shouldn't erase minutes of stress,
        # but sustained calm should.
        if cortisol_level > self.CHRONIC_THRESHOLD:
            self._elevated_stress_duration += dt
        else:
            self._elevated_stress_duration = max(
                0.0, self._elevated_stress_duration - dt * 2.0
            )

        # Determine if stress is chronic (sustained elevation)
        self._state.is_chronic = (
            self._elevated_stress_duration >= self.CHRONIC_DURATION_SECONDS
        )

        # Check if the substrate is still connected
        elapsed = self._time.monotonic() - self._last_substrate_update
        if elapsed > self.SUBSTRATE_TIMEOUT:
            self._state.substrate_connected = False

        # Fallback: estimate load from cortisol when substrate is offline
        if not self._state.substrate_connected:
            self._fallback_update_load(cortisol_level, dt)

    def _fallback_update_load(self, cortisol_level: float, dt: float) -> None:
        """Estimate allostatic load from cortisol when the daemon is offline.

        This is a degraded mode — the fallback can only react to
        current cortisol, not anticipate future disruption. It uses
        a non-zero baseline so acute stress doesn't accumulate
        chronic load, and recovery is 2× faster than accumulation.
        """
        excess = max(0.0, cortisol_level - self.FALLBACK_CORTISOL_BASELINE)

        if excess > 0:
            accumulation = excess * self.FALLBACK_ACCUMULATION_RATE * dt
            self._state.allostatic_load = min(
                1.0, self._state.allostatic_load + accumulation
            )
        else:
            recovery = self.FALLBACK_RECOVERY_RATE * dt
            self._state.allostatic_load = max(
                0.0, self._state.allostatic_load - recovery
            )

    def get_allostatic_load(self) -> float:
        """Get the current cumulative allostatic load.

        Returns:
            The allostatic load (0–1). Values above 0.4 indicate
            significant strain. Values above 0.7 indicate severe
            allostatic overload.
        """
        return self._state.allostatic_load

    def get_acute_stress(self) -> float:
        """Get the current acute stress level (0–1).

        Acute stress is temporary and adaptive — a brief cortisol
        elevation that resolves quickly. This is distinct from
        chronic/allostatic stress.
        """
        return self._state.acute_stress

    def is_chronic_stress(self) -> bool:
        """Whether the current stress pattern is chronic.

        Chronic stress is sustained elevation of cortisol over a
        prolonged period (at least CHRONIC_DURATION_SECONDS seconds
        above the threshold). It's maladaptive and contributes to
        allostatic load.
        """
        return self._state.is_chronic

    def get_stress_history(self) -> list[float]:
        """Get the rolling window of stress history.

        Returns a list of recent cortisol levels, oldest first.
        The window size is HISTORY_WINDOW samples.
        """
        return list(self._state.stress_history)

    def get_state(self) -> AllostaticState:
        """Get the full allostatic state for inspection."""
        return self._state

    @property
    def load_level(self) -> str:
        """Human-readable allostatic load level.

        Returns one of: "low", "moderate", "high", "severe".
        """
        load = self._state.allostatic_load
        if load < 0.2:
            return "low"
        elif load < 0.4:
            return "moderate"
        elif load < 0.7:
            return "high"
        else:
            return "severe"

    def reset(self) -> None:
        """Reset the tracker to baseline (no allostatic load)."""
        self._state = AllostaticState(
            stress_history=deque(maxlen=self.HISTORY_WINDOW),
        )
        self._tick_count = 0
        self._elevated_stress_duration = 0.0
        self._last_substrate_update = self._time.monotonic()
