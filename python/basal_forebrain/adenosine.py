"""Adenosine — sleep pressure; ATP metabolism byproduct.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Throughout the brain — adenosine is a byproduct of ATP
    metabolism. It accumulates in the extracellular space during
    wakefulness as neurons consume ATP.

Projections:
    Volume transmission throughout the brain. Acts on A1 and A2A
    receptors to promote sleep.

Receptors:
    A1 — Gi-coupled, inhibitory. Widely distributed. Adenosine
        binding to A1 inhibits excitatory neurotransmission.
    A2A — Gs-coupled, excitatory on sleep-promoting neurons.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.20

~30 nM at rest. Sleep pressure. Adenosine accumulates during
wakefulness and is cleared by the glymphatic system during sleep.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.3

The somnogenic receptor is A1, which resists downregulation — A1Rs
are not readily phosphorylated and internalize slowly (half-life of
several hours), where A2A/A2B downregulate in under an hour
(Klaasse et al., 2008). The slow multiplier keeps effective adenosine
from collapsing under chronic sleep pressure.


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT ADENOSINE (matrix row 10)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    Adenosine's interactions with arousal systems are handled by the sleep pressure mechanism
    (level accumulation/clearance), NOT by the coupling matrix.
    Only GABA (promotes sleep) and cortisol (stress-related adenosine release) have direct coupling.
    Orexin mildly inhibits adenosine (wake promotion).

    GABA -> ADN: +0.10  (weak volume transmission.)
    CORT -> ADN: +0.10  (stress-related adenosine release.)
    OX   -> ADN: -0.10  (wake promotion — orexin mildly inhibits adenosine.)
    DA   -> ADN: 0.00   (no direct coupling)
    SRT  -> ADN: 0.00   (no direct coupling)
    NE   -> ADN: 0.00   (no direct coupling)
    ACh  -> ADN: 0.00   (no direct coupling)
    GLU  -> ADN: 0.00   (no direct coupling)
    OXY  -> ADN: 0.00   (no direct coupling)
    END  -> ADN: 0.00   (no direct coupling)
    HIST -> ADN: 0.00   (no direct coupling)
    BDNF -> ADN: 0.00   (no direct coupling)
    ECB  -> ADN: 0.00   (no direct coupling)
    VP   -> ADN: 0.00   (no direct coupling)
    CRH  -> ADN: 0.00   (no direct coupling)
    EPI  -> ADN: 0.00   (no direct coupling)
    MEL  -> ADN: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW ADENOSINE AFFECTS OTHERS (matrix column 10)
════════════════════════════════════════════════════════════════════════

    ADN -> GABA:  +0.10  (sleep pressure promotes inhibition.)
    ADN -> OX :  -0.10  (sleep pressure inhibits orexin (the sleep switch).)
    ADN -> DA :  0.00   (no direct coupling)
    ADN -> SRT:  0.00   (no direct coupling)
    ADN -> NE :  0.00   (no direct coupling)
    ADN -> ACh:  0.00   (no direct coupling)
    ADN -> GLU:  0.00   (no direct coupling)
    ADN -> CORT:  0.00   (no direct coupling)
    ADN -> OXY:  0.00   (no direct coupling)
    ADN -> END:  0.00   (no direct coupling)
    ADN -> HIST:  0.00   (no direct coupling)
    ADN -> BDNF:  0.00   (no direct coupling)
    ADN -> ECB:  0.00   (no direct coupling)
    ADN -> VP :  0.00   (no direct coupling)
    ADN -> CRH:  0.00   (no direct coupling)
    ADN -> EPI:  0.00   (no direct coupling)
    ADN -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Sleep pressure mechanism (Process S):
    Adenosine has its own dedicated sleep-pressure mechanism that
    directly manages its level. The homeostatic force is SKIPPED
    for adenosine — it would fight the accumulation/clearance.

    During WAKEFULNESS (and Drowsy):
        Adenosine accumulates as a saturating exponential toward
        an upper asymptote (Process S, Daan et al., 1984):

        S(t+dt) = U - exp(-r' * dt_hours) * (U - S(t))

        where:
            U = 0.95 (upper asymptote)
            r' = adenosine_accumulation_rate * activity_factor * drowsy_factor
            activity_factor = 1 + adenosine_activity_coupling * arousal
            drowsy_factor = 0.5 if Drowsy else 1.0
            dt_hours = dt_scale * DT / 3600

        Biological default: r' = 0.055/h (τ = 18.2 hours), derived
        from EEG slow-wave activity buildup during wakefulness.

    During SLEEP (NREM/REM — the network's stage, or the mind's
    Sleeping zone):
        The glymphatic system clears adenosine toward its lower
        asymptote exponentially (Daan, Beersma & Borbély, 1984):

        S(t+dt) = L + exp(-dt_hours / chi_s) * (S(t) - L)

        where:
            L = 0.20 (lower asymptote, the resting baseline)
            chi_s = 4.2 h — the clearance time constant fitted from
                    the exponential decline of EEG slow-wave activity
                    across baseline and post-deprivation sleep

        The earlier linear decrement is retired: it made the two
        directions asymmetric and could not reach the cholinergic
        rebound the REM stage needs (REM occupancy measured 0%).
        Exponential clearance crosses 0.90 → 0.65 in ~1.9 h, inside
        the first NREM bouts.

    The main loop SKIPS adenosine during sleep (continue) so the
    homeostatic/coupling forces don't crash the level to the
    resting baseline (0.20) in seconds, overriding the slow
    (hours-timescale) glymphatic clearance.

Sleep onset is network-owned:
    There is no adenosine exhaustion override. Sleep onset and the
    NREM/REM stages are decided by the sourced sleep-wake network
    (ModelDB 244485) in the Rust substrate — LC firing below 4 Hz is
    sleep onset — and injected into the phase. Adenosine is the
    homeostatic pressure *signal* the mind reads for voluntary sleep
    and drowsiness, not the body's sleep gate.

No baseline adaptation:
    Adenosine has its own dedicated sleep-pressure mechanism that
    directly manages its level. The general baseline adaptation
    would fight with that mechanism, so it is skipped.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force:
    homeostatic_force = 0.0  (skipped — dedicated mechanism)

Sleep pressure (wakefulness — saturating exponential):
    S(t+dt) = U - exp(-r' * dt_hours) * (U - S(t))
    where:
        U = 0.95
        r' = accumulation_rate * (1 + activity_coupling * arousal) * drowsy_factor
        drowsy_factor = 0.5 if Drowsy else 1.0
        dt_hours = dt_scale * DT / 3600

Glymphatic clearance (sleep):
    S(t+dt) = L + exp(-dt_hours / chi_s) * (S(t) - L)
    where:
        L = 0.20, chi_s = 4.2 h

Effective level, coupling force, vesicular pool, tonic/phasic,
receptor adaptation: same general formulas as dopamine.py.


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)

Adenosine's circadian behavior is driven by the sleep pressure
mechanism (accumulation during wakefulness, clearance during
sleep), not by baseline modulation.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Adenosine -> delta/theta (sleep pressure, drowsiness)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation (sleep promoter):
    sleep_promoters = GABA*0.25 + ADN*0.65 + MEL*0.35
    (adenosine weight = 0.65 — the primary homeostatic sleep drive,
    weighted more than GABA because adenosine is the primary
    homeostatic sleep drive)

Sleep stages (NREM/REM):
    Owned by the Rust sleep-wake network, not this layer. The
    chemistry-side adenosine/histamine sleep gates are retired;
    the network's LC firing below 4 Hz is sleep onset.

    The adenosine signal still blends effective (receptor-mediated)
    and raw (metabolic) levels for the *waking* classifier:
    adn = adn_eff * 0.5 + adn_raw * 0.5

    so high metabolic pressure gates Drowsy even when receptor
    sensitivity is at floor.

Drowsy phase:
    adn > 0.50 (enter) / 0.40 (stay) + arousal < 0.30 (enter) / 0.46 (stay)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.1
(Conservative — adenosine is the primary sleep driver)

One-way rule:
    Process S only moves down through sleep. `apply_impulse_capped`
    ignores negative adenosine impulses: external events may add
    metabolic load, but clearance belongs to the dedicated mechanism
    (glymphatic during sleep, offline rest). Fighting drowsiness is
    done on the alerting transmitters (ACh, histamine, NE), never by
    editing the homeostat — a periodic drain cannot distinguish
    coupling drift from genuine wake-time accumulation, and at any
    plausible cadence it outruns the ~0.055/h accumulation rate by
    orders of magnitude, pinning pressure at the floor (every NREM
    episode staged N1).

Python modules that send adenosine impulses:
    - brainstem/heartbeat.py: metabolic work load (positive)

Python modules that read adenosine:
    - emotion.py: maps to EmotionalState (drowsiness)
    - emotional_regulator.py: homeostatic regulation
    - sleep/ (Python): sleep pressure, sleep timing

Typical impulse scenarios:
    - Sustained compute: +0.00015/work-second (heartbeat)
    - Rest policy: +0.03 (active_inference — skipped during sleep)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_ADENOSINE

__all__ = ["CHEM_ADENOSINE", "Adenosine"]


@dataclass(frozen=True, slots=True)
class Adenosine:
    """Adenosine — sleep pressure; ATP metabolism byproduct.

    All values synchronized with src/state/neurochemical.rs and
    src/state/basal_forebrain.rs (the A1 site-of-action mechanism —
    math on the Rust side, words here).
    """

    index: int = CHEM_ADENOSINE
    name: str = "adenosine"
    baseline: float = 0.20
    receptor_adaptation_multiplier: float = 0.3
    receptor_subtypes: tuple[str, str] = ("A1", "A2A")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.1
    origin: tuple[str, ...] = ("throughout brain (ATP byproduct)",)
    projections: tuple[str, ...] = ("throughout brain (volume transmission)",)
    receptors: tuple[str, ...] = ("A1", "A2A")
    brain_waves: tuple[str, ...] = ("delta", "theta")
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "sleep pressure, Process S, glymphatic clearance"
    accumulation_rate: float = 0.055  # per hour (biological default, tau = 18.2 h)
    clearance_tau_hours: float = 4.2  # chi_s, Daan/Beersma/Borbély 1984
    activity_coupling: float = 0.5
    upper_asymptote: float = 0.95
    lower_asymptote: float = 0.20
