"""Thalamus — the relay, gate, and rhythm generator of the brain.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The relay is the brain's central relay. Almost every channel of
information reaching the cortex passes through it — sensory input,
motor feedback, arousal signals. It is not a passive switchboard:
through the thalamic reticular nucleus (TRN) it actively *gates*
what reaches cognition, and through thalamocortical loops it
*generates* the brain's dominant rhythms — sleep spindles, slow
waves, and the alpha rhythm of the idling cortex.

Key structures:
    - Relay nuclei (LGN, MGN, VPL, pulvinar) — thalamocortical (TC)
      projection neurons carrying driver signals to cortex.
    - Thalamic reticular nucleus (TRN) — a thin GABAergic shell
      surrounding the relay. Every thalamocortical and
      corticothalamic axon gives collaterals to TRN; TRN's only
      output is inhibition back onto TC relay cells.
    - Higher-order nuclei (pulvinar, mediodorsal) — relay
      cortico-cortical communication rather than sensory input.

Key functions:
    - Relay (first-order: driver input -> cortex; higher-order:
      cortico-thalamo-cortical loops — Sherman & Guillery)
    - Attentional gating (TRN selects which channels reach cortex)
    - Rhythm generation (spindles, slow oscillations, alpha)
    - Cognitive broadcast (thalamocortical ignition — the workspace
      relay)
    - Sleep-state implementation (the TC membrane voltage is set by
      autonomics neuromodulators; below threshold the loop spindles)


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: TRN ATTENTIONAL GATING
════════════════════════════════════════════════════════════════════════

Crick's "searchlight hypothesis" (1984) proposed the TRN as the
substrate of selective attention: because TRN neurons are inhibitory
and topographically organized, they can selectively suppress
thalamic relay channels — an inhibitory spotlight.

The circuit motif:

    cortex  ──(glutamate, collaterals)──>  TRN
    TC cell ──(glutamate, collaterals)──>  TRN
    TRN     ──(GABA)──────────────────>  TC cell

Both forward (TC->TRN) and feedback (cortex->TRN) excitation
converge on the same inhibitory neurons, so gating is competitive:
channels that drive TRN hardest suppress their neighbors.

Computational findings (Halassa lab; Schmitt et al., Nature 2017;
bioRxiv 2020.09.16.300749):
    - Top-down (prefrontal) input onto TRN regulates thalamic GAIN
      more effectively than direct input onto TC cells — attention
      works by biasing the gate, not the relay.
    - TC cells lack recurrent excitation, which makes top-down and
      bottom-up signals separable in the response geometry — the
      gate can tell "cortex asked for this" from "the world sent
      this."
    - Amygdala -> TRN projection (John et al., PLOS Comp. Biol.
      2016, "Emotional Gatekeeper"): affective salience gets a
      private line to the attention gate — emotionally charged
      input can seize or suppress channels.

Rate-model form (linear-threshold mesoscale model, Nozari et al.,
arXiv:2201.00850): each subnetwork's activity obeys

    τ dx/dt = -x + [ W x + u ]+      (linear-threshold dynamics)

and selective inhibition/recruitment is provably achievable by
feedback+feedforward control through the thalamic node — the
relay is a control point, not a wire.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: SLEEP SPINDLES AND K-COMPLEXES
════════════════════════════════════════════════════════════════════════

When the autonomics withdraws arousal drive (see ``autonomics/``), TC
membrane voltage hyperpolarizes and the thalamic circuit flips into
oscillation:

    RE (reticular) burst
        -> GABA_A/GABA_B IPSPs in TC relay cells
        -> hyperpolarization DEINACTIVATES the low-threshold
           T-type Ca2+ current I_T
        -> post-inhibitory rebound: TC cells fire a burst
        -> TC burst re-excites RE
        -> repeat at 12-15 Hz, amplitude waxing then waning

That is a sleep spindle — the signature of N2 sleep, generated
entirely by the RE<->TC loop (Destexhe, Contreras & Steriade).
Conductance-based models implement it with Hodgkin-Huxley kinetics:
I_RE = I_Na + I_K + I_KL + I_T; the T-current is the pacemaker.

K-complexes: Mak-McClure et al. (PLOS Comp. Biol. 2015) showed in a
conductance-based thalamocortical model that KCs are cortical
downstates produced by *disruption* of thalamic spindling — a focal
prefrontal depolarization recruits RE, depolarization inactivates
I_T, spindling collapses, TC drive to cortex drops abruptly, and
cortex falls silent. The same mechanism is evoked by sensory stimuli
during sleep — which is why a sound in the night can produce a
K-complex rather than waking.

Slow oscillation / spindle coupling: cross-frequency models (PLOS
Comp. Biol. 2022; Frontiers 2022) show cortical DOWN states open a
"window of opportunity" in which the thalamic node spindles, and
spindles can trigger DOWN->UP transitions — the mechanism that
nests spindles inside slow waves during memory consolidation.


════════════════════════════════════════════════════════════════════════
RELAY MODES: DRIVER AND MODULATOR
════════════════════════════════════════════════════════════════════════

Sherman & Guillery's classification:
    - First-order relays carry *driver* input from the periphery:
      retina -> LGN -> V1; cochlea -> MGN -> A1. The relay
      transmits a copy.
    - Higher-order relays (pulvinar, MD) carry *modulator* loops
      between cortical areas: cortex -> relay -> cortex. Much of
      "cortical" communication is actually transthalamic.

Cognitive broadcast: in Global Neuronal Workspace terms, ignition
of a representation involves widespread thalamocortical
recruitment — the higher-order nuclei are the physical fan-out of
the workspace broadcast.


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

    - Brainstem (``autonomics/``): the ARAS sets TC membrane voltage
      and thereby the relay/oscillate mode switch. PGO waves are
      generated in the pons and RELAYED through the relay (LGN)
      to occipital cortex.
    - Occipital subsystem: LGN -> V1 is the canonical first-order relay.
    - Parietal subsystem: attention.py's top-down biasing is the cortical
      command; the TRN is its thalamic effector.
    - Frontal subsystem: the workspace broadcast is prefrontal-driven but
      physically relayed through higher-order thalamic nuclei.
    - Temporal subsystem: MGN -> A1 auditory relay; mediodorsal nucleus
      serves memory (Papez circuit).


════════════════════════════════════════════════════════════════════════
MODULE ORGANIZATION
════════════════════════════════════════════════════════════════════════

The relay's functions are implemented by multi-subsystem modules that
stay at the top level / in the sleep package and are re-exported
here (the ``association`` pattern):

    from genesis_cognitive.relay import BrainWave, SleepSpindle

Re-exported modules:

    (top-level) brain_waves.py — the thalamocortical rhythm layer

    BrainWave, BrainWaveState, assess_brain_waves
        The EEG bands — delta/theta/alpha/beta/gamma are
        thalamocortical loop resonances (spindles are sigma band;
        slow waves are delta; alpha is the idling relay).
    GammaSynchrony, compute_gamma_synchrony
        Gamma binding — thalamocortical coherence during attended
        processing.
    ThetaGammaCoupling, compute_theta_gamma_coupling
        Cross-frequency coupling — memory encoding rhythm.
    SleepStageSignature
        The spectral fingerprint used to classify the sleep stages
        the autonomics switches between.

    (sleep package — the thalamic-generated events)

    SleepSpindle
        The RE<->TC rebound loop event — N2 signature, memory
        consolidation window.
    KComplex
        The cortically-induced spindling disruption — the isolated
        downstate that protects sleep.

    (top-level) global_workspace.py — the broadcast relay

    GlobalWorkspace, WorkspaceItem, WorkspaceModule
        Dehaene's workspace — the higher-order thalamic nuclei are
        the physical fan-out of ignition. Also re-exported by
        control (the PFC is the primary broadcaster; the
        relay is the wire).

Referenced but not re-exported:

    (top-level) attention.py
        The parietal-prefrontal attention system. The TRN gating it
        exercises is documented here; the module itself is
        re-exported by association.
"""

from __future__ import annotations

from .._views import view_getattr

# The relay's functions are implemented by multi-subsystem modules that
# stay at the top level / in the sleep package and are lazily
# re-exported here so the anatomy is a real connection layer, not
# just documentation.
_EXPORTS: dict[str, str] = {
    # The thalamocortical rhythm layer — brain_waves.py tracks the
    # EEG bands generated by thalamocortical loops
    "BrainWave": "brain_waves",
    "BrainWaveState": "brain_waves",
    "GammaSynchrony": "brain_waves",
    "SleepStageSignature": "brain_waves",
    "ThetaGammaCoupling": "brain_waves",
    "assess_brain_waves": "brain_waves",
    "compute_gamma_synchrony": "brain_waves",
    "compute_theta_gamma_coupling": "brain_waves",
    # The thalamic-generated sleep events — RE<->TC loop products
    "SleepSpindle": "sleep",
    "KComplex": "sleep",
    # The broadcast relay — higher-order thalamic nuclei are the
    # physical fan-out of workspace ignition (also re-exported by
    # control: the PFC is the primary broadcaster)
    "GlobalWorkspace": "global_workspace",
    "WorkspaceItem": "global_workspace",
    "WorkspaceModule": "global_workspace",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
