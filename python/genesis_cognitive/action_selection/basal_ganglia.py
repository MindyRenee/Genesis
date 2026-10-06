"""Reduced-order basal-ganglia action selection.

This is a computational selection circuit, not an anatomical simulation.
It follows center-surround/disinhibition logic: option-specific striatal
drive competes against diffuse STN-like control at a tonically inhibitory
output stage. Conflict-dependent hyperdirect input raises the effective
selection threshold.

The dopamine input is a normalized modulatory signal, not a claim that a
TD prediction error is identical to measured dopamine concentration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ActionBid", "SelectionResult", "BasalGangliaSelector"]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True, slots=True)
class ActionBid:
    """One candidate action entering the selection circuit."""
    action: str
    salience: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Inspectable state of one basal-ganglia selection computation."""
    selected: str | None
    salience: dict[str, float]
    d1_drive: dict[str, float]
    d2_drive: dict[str, float]
    stn_control: float
    hyperdirect_brake: float
    output_inhibition: dict[str, float]
    disinhibition: dict[str, float]
    conflict: float
    effective_threshold: float

    @property
    def selected_disinhibition(self) -> float:
        if self.selected is None:
            return 0.0
        return self.disinhibition[self.selected]


class BasalGangliaSelector:
    """Reduced computational model of basal-ganglia action selection.

    Reduced equations:

        D1_i = clip(s_i * (1 + k * RPE), 0, max_drive)
        D2_i = clip(s_i * (1 - k * RPE), 0, max_drive)
        STN  = control_gain * mean(D2_i)
        GPi_i = tonic_output + STN + hyperdirect - D1_i
        disinhibition_i = tonic_output - GPi_i

    The action with greatest disinhibition is selected only when it clears
    the conflict-adjusted threshold.

    This is deliberately a reduced-order model. It preserves the
    computational roles of focused striatal inhibition, diffuse STN
    excitation, tonic inhibitory output, and conflict-dependent braking
    without pretending that scalar equations reproduce the full,
    heterogeneous basal ganglia.

    Positive RPE increases D1 gain and decreases D2 gain; negative RPE has
    the opposite effect. The caller supplies an appropriately scaled
    modulatory signal.
    """

    def __init__(
        self,
        *,
        tonic_output: float = 1.0,
        control_gain: float = 0.5,
        dopamine_gain: float = 0.5,
        hyperdirect_gain: float = 0.8,
        base_threshold: float = 0.15,
        max_drive: float = 2.0,
    ) -> None:
        if not all(math.isfinite(v) for v in (
            tonic_output, control_gain, dopamine_gain,
            hyperdirect_gain, base_threshold, max_drive,
        )):
            raise ValueError("selector parameters must be finite")
        if tonic_output <= 0 or control_gain < 0 or dopamine_gain < 0:
            raise ValueError("invalid selector gain")
        if hyperdirect_gain < 0 or base_threshold < 0 or max_drive <= 0:
            raise ValueError("invalid selector threshold/gain")
        self.tonic_output = tonic_output
        self.control_gain = control_gain
        self.dopamine_gain = dopamine_gain
        self.hyperdirect_gain = hyperdirect_gain
        self.base_threshold = base_threshold
        self.max_drive = max_drive

    def select(
        self,
        bids: list[ActionBid] | tuple[ActionBid, ...],
        *,
        dopamine_rpe: float = 0.0,
        conflict: float | None = None,
    ) -> SelectionResult:
        """Run one selection computation."""
        if not bids:
            return SelectionResult(
                selected=None, salience={}, d1_drive={}, d2_drive={},
                stn_control=0.0, hyperdirect_brake=0.0,
                output_inhibition={}, disinhibition={}, conflict=0.0,
                effective_threshold=self.base_threshold,
            )
        if not math.isfinite(dopamine_rpe):
            raise ValueError("dopamine_rpe must be finite")

        salience: dict[str, float] = {}
        for bid in bids:
            if not bid.action:
                raise ValueError("action names must be non-empty")
            if not math.isfinite(bid.salience):
                raise ValueError("action salience must be finite")
            if bid.action in salience:
                raise ValueError(f"duplicate action: {bid.action}")
            salience[bid.action] = _clip(bid.salience, 0.0, 1.0)

        rpe = _clip(dopamine_rpe, -1.0, 1.0)
        d1_gain = 1.0 + self.dopamine_gain * rpe
        d2_gain = 1.0 - self.dopamine_gain * rpe
        d1_drive = {
            action: _clip(value * d1_gain, 0.0, self.max_drive)
            for action, value in salience.items()
        }
        d2_drive = {
            action: _clip(value * d2_gain, 0.0, self.max_drive)
            for action, value in salience.items()
        }

        stn_control = self.control_gain * (
            sum(d2_drive.values()) / len(d2_drive)
        )

        ranked = sorted(salience.values(), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else 0.0
        derived_conflict = 0.0 if top <= 0 else _clip(second / top, 0.0, 1.0)
        effective_conflict = (
            derived_conflict
            if conflict is None
            else _clip(conflict, 0.0, 1.0)
        )
        hyperdirect_brake = self.hyperdirect_gain * effective_conflict
        effective_threshold = self.base_threshold + hyperdirect_brake

        output_inhibition = {
            action: (
                self.tonic_output + stn_control
                + hyperdirect_brake - d1_drive[action]
            )
            for action in salience
        }
        disinhibition = {
            action: max(0.0, self.tonic_output - output_inhibition[action])
            for action in salience
        }

        winner = max(
            disinhibition,
            key=lambda action: (disinhibition[action], salience[action]),
        )
        selected = (
            winner
            if disinhibition[winner] >= effective_threshold
            else None
        )

        return SelectionResult(
            selected=selected,
            salience=salience,
            d1_drive=d1_drive,
            d2_drive=d2_drive,
            stn_control=stn_control,
            hyperdirect_brake=hyperdirect_brake,
            output_inhibition=output_inhibition,
            disinhibition=disinhibition,
            conflict=effective_conflict,
            effective_threshold=effective_threshold,
        )
