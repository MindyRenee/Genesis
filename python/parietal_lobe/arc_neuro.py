"""ARC neuro loop — her EM field, brain waves, and chemicals while she plays.

Reuses Genesis's own organs in the order her body runs them
(see ``hypothalamus.transduction``): EM field -> brain waves -> chemicals.

How she uses it during an ARC game:

1. ``field`` (optional) is the EM channel: a callable
   ``field(oscillator) -> bool`` that holds measured field levels on her
   ``thalamus.brain_waves.BrainWaveOscillator`` (e.g. a wrapper around
   ``brainstem.field_probe``). Absent reads as absent, never as zero.
2. Game events release chemicals: dopamine follows the TD reward
   prediction error, norepinephrine a never-seen board, GABA a move
   that changed nothing. Each relaxes to its baseline,
   ``C <- C0 + (C - C0) * exp(-1/tau) + impulse``.
3. ``oscillator.assess`` turns the chemicals into band powers.
4. ``hypothalamus.transduction.wave_impulses`` lets the bands release
   their peptides: the waves talking back to the chemicals.
5. The wave state's ``focus`` feeds back into play (the policy weights
   learned value by ``0.5 + focus``).

Pure numbers in, numbers out: no language is involved.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hypothalamus.transduction import _PEPTIDES, wave_impulses
from thalamus.brain_waves import BrainWaveOscillator, BrainWaveState

from genesis_client import NeuroSummary

BASELINES = {"dopamine": 0.5, "norepinephrine": 0.4, "gaba": 0.5, "acetylcholine": 0.5}
TAU_BEATS = 8.0
DOPAMINE_GAIN = 0.25
NOREPINEPHRINE_GAIN = 0.15
GABA_GAIN = 0.1


def _clip(value: float) -> float:
    """Clamp a level to [0, 1]."""
    return max(0.0, min(1.0, value))


@dataclass
class ArcNeuroLoop:
    """One closed EM-field -> brain-wave -> chemical loop for one player."""

    field_source: Callable[[BrainWaveOscillator], bool] | None = None
    seed: int = 0
    chemicals: dict[str, float] = field(default_factory=dict)
    state: BrainWaveState | None = None
    field_live: bool = False
    _oscillator: BrainWaveOscillator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Start every chemical at baseline and build her oscillator."""
        self._oscillator = BrainWaveOscillator(rng=random.Random(self.seed))
        levels = dict(BASELINES)
        levels.update({pep.name: pep.baseline for pep in _PEPTIDES})
        levels.update(self.chemicals)
        self.chemicals = levels
        self._baseline = dict(levels)
        self._peptide = {pep.index: pep.name for pep in _PEPTIDES}

    @property
    def focus(self) -> float:
        """Her current attentional focus from the wave state (0.5 before any beat)."""
        return self.state.focus if self.state is not None else 0.5

    def beat(self, *, dopamine_rpe: float, novel: bool, changed: bool) -> BrainWaveState:
        """Run one loop cycle after a move and return the new wave state."""
        decay = math.exp(-1.0 / TAU_BEATS)
        for name, level in self.chemicals.items():
            base = self._baseline[name]
            self.chemicals[name] = base + (level - base) * decay
        chem = self.chemicals
        chem["dopamine"] = _clip(chem["dopamine"] + DOPAMINE_GAIN * dopamine_rpe)
        if novel:
            chem["norepinephrine"] = _clip(chem["norepinephrine"] + NOREPINEPHRINE_GAIN)
        if not changed:
            chem["gaba"] = _clip(chem["gaba"] + GABA_GAIN)
        self.field_live = bool(self.field_source and self.field_source(self._oscillator))
        summary = NeuroSummary(
            arousal=chem["norepinephrine"],
            valence=2.0 * chem["dopamine"] - 1.0,
            global_tone=1.0 - chem["gaba"],
            plasticity_gate=chem["dopamine"],
            encoding_weight=chem["acetylcholine"],
            consolidation_weight=0.5,
            retrieval_weight=0.5,
            phase=0,
            dopamine_tonic=chem["dopamine"],
            dopamine_phasic=dopamine_rpe,
        )
        self.state = self._oscillator.assess(summary, chemicals=chem)
        for index, impulse in wave_impulses(self.state, chem):
            peptide = self._peptide.get(index)
            if peptide is not None:
                chem[peptide] = _clip(chem[peptide] + impulse)
        return self.state

    def telemetry(self) -> dict[str, Any]:
        """Field, wave, and chemical readings for decision reasoning."""
        state = self.state
        return {
            "em_field": "live" if self.field_live else "absent",
            "dominant_wave": state.dominant.name.lower() if state else None,
            "focus": round(self.focus, 6),
            "chemicals": {
                name: round(self.chemicals[name], 6)
                for name in ("dopamine", "norepinephrine", "gaba", "orexin", "cortisol")
            },
        }
