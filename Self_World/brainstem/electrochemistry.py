"""Electrochemical interpretation for conscious cognition.

The Rust substrate owns the ion dynamics: concentrations, reversal
potentials, conductances, membrane voltage, transporters, ATP availability,
and physiological sanitization. This module performs the narrower conscious
task of interpreting the resulting :class:`IonSummary` as finite cognitive
signals — how much attention, global broadcast, memory consolidation, and
synaptic learning the current electrochemical state can support.

Nothing here recomputes electrochemistry. The derived gains stay finite for
malformed input, but valid substrate values are not range-limited here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from genesis_client.types import IonSummary

__all__ = ["IonCognitiveContext", "interpret_ions"]


@dataclass(frozen=True, slots=True)
class IonCognitiveContext:
    """Finite interpretation of the substrate's ion state.

    Raw normalized fields are the substrate's own derived readouts. Gain
    fields are cognitive modulation factors centered near 1.0 at physiological
    rest: 1.0 is neutral, <1 suppresses, >1 amplifies.
    """

    membrane_potential_mv: float
    nak_pump_rate: float
    kcl_cotransporter_flux: float
    ncx_flux: float
    net_membrane_current: float

    atp_availability: float
    calcium_signal: float
    chloride_efficacy: float
    excitability: float
    gradient_integrity: float
    energy_load: float

    metabolic_strain: float
    physiological_balance: float
    attention_gain: float
    workspace_gain: float
    memory_consolidation_gain: float
    synaptic_plasticity_gain: float

    @classmethod
    def neutral(cls) -> IonCognitiveContext:
        """The behavior-preserving context used when no summary is available."""
        return cls(
            membrane_potential_mv=-75.0,
            nak_pump_rate=0.018,
            kcl_cotransporter_flux=0.010,
            ncx_flux=0.0002,
            net_membrane_current=0.0,
            atp_availability=0.9,
            calcium_signal=0.0,
            chloride_efficacy=0.5,
            excitability=0.0,
            gradient_integrity=1.0,
            energy_load=0.25,
            metabolic_strain=0.0,
            physiological_balance=0.875,
            attention_gain=1.0,
            workspace_gain=1.0,
            memory_consolidation_gain=1.0,
            synaptic_plasticity_gain=1.0,
        )

    @property
    def needs_physiological_recovery(self) -> bool:
        """Whether ion state is strained enough to favor protective learning."""
        return (
            self.metabolic_strain > 0.65
            or self.physiological_balance < 0.45
            or self.gradient_integrity < 0.50
        )


def interpret_ions(summary: IonSummary | None) -> IonCognitiveContext:
    """Interpret an ``IonSummary`` as finite cognitive modulation.

    The summary's normalized fields are sanitized again defensively because
    this boundary may be reached through older clients, test doubles, or a
    malformed IPC response. No electrochemical equations live here — the Rust
    ``IonState`` remains the source of physical truth.
    """
    if not isinstance(summary, IonSummary):
        return IonCognitiveContext.neutral()

    membrane = _finite_or(summary.membrane_potential_mv, -75.0)
    nak = _finite_or(summary.nak_pump_rate, 0.018)
    kcl = _finite_or(summary.kcl_cotransporter_flux, 0.010)
    ncx = _finite_or(summary.ncx_flux, 0.0002)
    net_current = _finite_or(summary.net_membrane_current, 0.0)

    atp = _finite_or(summary.atp_availability, 0.9)
    calcium = _finite_or(summary.calcium_signal, 0.0)
    chloride = _finite_or(summary.chloride_efficacy, 0.5)
    excitability = _finite_or(summary.excitability, 0.0)
    integrity = _finite_or(summary.gradient_integrity, 1.0)
    load = _finite_or(summary.energy_load, 0.25)

    # Resting values are centered rather than absolute: load=0.25 and
    # atp=0.9 are physiological baseline, not strain. This keeps the
    # interpretation neutral when the daemon reports a healthy rest state.
    metabolic_strain = _finite_or(
        0.70 * ((load - 0.25) / 0.75) + 0.30 * ((0.9 - atp) / 0.9),
        0.0,
    )
    physiological_balance = _finite_or(
        0.45 * integrity + 0.25 * atp + 0.20 * (1.0 - load) + 0.10 * chloride,
        0.875,
    )

    attention_gain = _finite_or(
        1.0
        + 0.20 * excitability
        + 0.12 * (chloride - 0.5)
        - 0.20 * metabolic_strain
        - 0.10 * (1.0 - integrity),
        1.0,
    )
    workspace_gain = _finite_or(
        1.0
        + 0.20 * excitability
        + 0.15 * (physiological_balance - 0.875)
        - 0.10 * metabolic_strain,
        1.0,
    )
    memory_consolidation_gain = _finite_or(
        1.0 + 0.15 * (atp - 0.9) + 0.20 * (integrity - 1.0) - 0.15 * (load - 0.25),
        1.0,
    )
    synaptic_plasticity_gain = _finite_or(
        1.0 + 0.45 * calcium + 0.20 * (atp - 0.9) + 0.20 * (integrity - 1.0) - 0.20 * (load - 0.25),
        1.0,
    )

    return IonCognitiveContext(
        membrane_potential_mv=membrane,
        nak_pump_rate=nak,
        kcl_cotransporter_flux=kcl,
        ncx_flux=ncx,
        net_membrane_current=net_current,
        atp_availability=atp,
        calcium_signal=calcium,
        chloride_efficacy=chloride,
        excitability=excitability,
        gradient_integrity=integrity,
        energy_load=load,
        metabolic_strain=metabolic_strain,
        physiological_balance=physiological_balance,
        attention_gain=attention_gain,
        workspace_gain=workspace_gain,
        memory_consolidation_gain=memory_consolidation_gain,
        synaptic_plasticity_gain=synaptic_plasticity_gain,
    )


def _finite_or(value: object, default: float) -> float:
    """Preserve a finite scalar and replace malformed/NaN/±∞ values."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default
