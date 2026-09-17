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

    multiplier = 1.0

Adenosine receptors adapt at the baseline reference rate.


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT ADENOSINE (matrix row 10)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    Adenosine's interactions with arousal systems are handled by the sleep pressure mechanism
    (level accumulation/clearance), NOT by the coupling matrix.
    Only GABA (promotes sleep) and cortisol (stress-related adenosine release) have direct coupling.
    Orexin mildly inhibits adenosine (wake promotion).

    GABA -> ADN: +0.05  (weak volume transmission.)
    CORT -> ADN: +0.05  (stress-related adenosine release.)
    OX   -> ADN: -0.05  (wake promotion — orexin mildly inhibits adenosine.)
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

    ADN -> GABA:  +0.05  (sleep pressure promotes inhibition.)
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

    During SLEEP (NREM/REM):
        The glymphatic system directly clears adenosine from the
        extracellular space:

        level -= adenosine_clearance_rate * dt_scale

        Biological default: 0.000002/tick at 10Hz → ~7.6 hours to
        clear from 0.75 back to 0.20, matching the human ~8h sleep
        period (Porkka-Heiskanen et al., 1997).

    The main loop SKIPS adenosine during sleep (continue) so the
    homeostatic/coupling forces don't crash the level to the
    resting baseline (0.20) in seconds, overriding the slow
    (hours-timescale) glymphatic clearance.

Exhaustion override:
    When raw adenosine (metabolic sleep pressure) is extreme
    (>0.90), force sleep regardless of histamine or orexin. This
    is the "passing out from exhaustion" mechanism — in humans,
    extreme sleep deprivation eventually overcomes all arousal
    systems (Borbély & Achermann, 1999).

    if adn_raw > 0.90:
        return NREM (or REM if already in REM with ACh > 0.40, NE < 0.35)

    This prevents the fatal feedback trap where receptor burnout
    suppresses effective adenosine so much that sleep can never be
    entered, trapping the system awake without recovery.

No baseline adaptation:
    Adenosine has its own dedicated sleep-pressure mechanism that
    directly modifies its baseline. The general baseline adaptation
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
    level -= clearance_rate * dt_scale

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

Sleep phase:
    adn > sleep_adn (0.75 enter / 0.65 stay, modulated by MEL and OX)
    + hist < sleep_hist (0.25 enter / 0.38 stay)

    The adenosine signal uses a 50/50 blend of effective (receptor-
    mediated) and raw (metabolic) levels:
    adn = adn_eff * 0.5 + adn_raw * 0.5

    This ensures that extreme sleep pressure (raw adenosine > 0.90)
    can still trigger sleep even when receptor sensitivity is at
    floor (0.50), preventing the feedback trap where receptor
    burnout makes sleep impossible.

Drowsy phase:
    adn > 0.50 (enter) / 0.40 (stay) + arousal < 0.30 (enter) / 0.46 (stay)

Exhaustion override:
    if adn_raw > 0.90: force NREM (or REM if already in REM)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.1
(Conservative — adenosine is the primary sleep driver)

Python modules that send adenosine impulses:
    - emotional_regulator.py: wakefulness (negative), drowsiness

Python modules that read adenosine:
    - emotion.py: maps to EmotionalState (drowsiness)
    - emotional_regulator.py: homeostatic regulation
    - sleep/ (Python): sleep pressure, sleep timing

Typical impulse scenarios:
    - Wakefulness: -0.06 (emotional_regulator)
    - Drowsiness: +0.06 (emotional_regulator, negative for wake)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_ADENOSINE

__all__ = ["CHEM_ADENOSINE", "Adenosine"]


@dataclass(frozen=True, slots=True)
class Adenosine:
    """Adenosine — sleep pressure; ATP metabolism byproduct.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_ADENOSINE
    name: str = "adenosine"
    baseline: float = 0.20
    receptor_adaptation_multiplier: float = 1.0
    receptor_subtypes: tuple[str, str] = ("A1", "A2A")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.1
    origin: tuple[str, ...] = ("throughout brain (ATP byproduct)",)
    projections: tuple[str, ...] = ("throughout brain (volume transmission)",)
    receptors: tuple[str, ...] = ("A1", "A2A")
    brain_waves: tuple[str, ...] = ("delta", "theta")
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "sleep pressure, Process S, glymphatic clearance"
    accumulation_rate: float = 0.055  # per hour (biological default)
    clearance_rate: float = 0.000002  # per tick at 10Hz (biological default)
    activity_coupling: float = 0.5
    upper_asymptote: float = 0.95
    exhaustion_threshold: float = 0.90
