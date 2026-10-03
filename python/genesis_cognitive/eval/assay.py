"""Per-chemical causal assay: what each chemical actually does.

Protocol (live isolated daemon only, per the harness):

1. Settle to rest once.
2. For each of the 18 chemicals, in protocol ID order:
   a. record pre-impulse state,
   b. apply a uniform suprathreshold impulse (default +0.30),
   c. advance 1 tick  -> post_fast (direct + fast coupled effects),
   d. advance 9 ticks -> post_slow (coupled reverberations),
   e. wash out (default 20 ticks) before the next chemical.

The uniform +0.30 dose is deliberately suprathreshold for the
Hill-function coupling gate (EC50 ~ baseline) so every chemical's
downstream influence is measurable and comparable. It is a causal
probe, not a physiological dose — potent modulators such as dopamine
(documented max impulse 0.05) are intentionally overdosed to map
their full influence. Washout returns the system to rest between
probes; residual cross-contamination is bounded by the decay curves
(impulse residuals after 30 ticks are ~0.01).

Known confound: the HPA cascade (CRH → ACTH → cortisol) is
maturation-gated (stress hyporesponsive period,
``src/state/neurochemical.rs``). Chemicals run in protocol ID order,
so late-sequence rows (CRH at 14) see a more mature HPA than early
rows — a fresh daemon correctly shows almost no CRH → cortisol
response. Interpret CRH/cortisol rows with instance age in mind.

Absorbing chemicals: adenosine (sleep-pressure sink) and melatonin
(circadian-driven) clear impulses within one tick by design; their
fast self-response is near zero. This is documented dynamics, not a
failed probe.

Each :class:`ChemicalEffect` records directional deltas for arousal,
valence, plasticity, memory gating, phase transitions, the chemical's
own response, and the top coupled responders among the other 17.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from genesis_client import GenesisClient
from genesis_client.protocol import CHEM_NAMES

ASSAY_IMPULSE = 0.30
ASSAY_SETTLE_TICKS = 60
ASSAY_WASHOUT_TICKS = 20
ASSAY_DT = 1.0
FAST_TICKS = 1
SLOW_EXTRA_TICKS = 9
TOP_COUPLED = 3


@dataclass
class ChemicalEffect:
    """Measured causal effect of one chemical's impulse."""

    chem_id: int = 0
    name: str = ""
    impulse: float = ASSAY_IMPULSE
    # Own response
    pre_self: float = 0.0
    post_fast_self: float = 0.0
    post_slow_self: float = 0.0
    # Arousal / valence deltas
    d_arousal_fast: float = 0.0
    d_arousal_slow: float = 0.0
    d_valence_fast: float = 0.0
    d_valence_slow: float = 0.0
    # Plasticity + memory gating deltas (slow point)
    d_plasticity_slow: float = 0.0
    d_encoding_slow: float = 0.0
    d_consolidation_slow: float = 0.0
    d_retrieval_slow: float = 0.0
    # Phase transitions
    phase_pre: str = ""
    phase_fast: str = ""
    phase_slow: str = ""
    # Top coupled responders at the slow point: [(name, delta)]
    top_coupled: list[list[Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChemicalEffect:
        """Deserialize; unknown keys are ignored."""
        known = set(cls.__dataclass_fields__)
        eff = cls()
        for k, v in data.items():
            if k in known:
                setattr(eff, k, v)
        return eff


@dataclass
class AssayResult:
    """Full 18-chemical assay matrix. JSON-serializable."""

    format_version: int = 1
    captured_at_unix: float = 0.0
    impulse: float = ASSAY_IMPULSE
    settle_ticks: int = ASSAY_SETTLE_TICKS
    washout_ticks: int = ASSAY_WASHOUT_TICKS
    dt: float = ASSAY_DT
    effects: list[ChemicalEffect] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "format_version": self.format_version,
            "captured_at_unix": self.captured_at_unix,
            "impulse": self.impulse,
            "settle_ticks": self.settle_ticks,
            "washout_ticks": self.washout_ticks,
            "dt": self.dt,
            "effects": [e.to_dict() for e in self.effects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssayResult:
        """Deserialize."""
        return cls(
            format_version=int(data.get("format_version", 1)),
            captured_at_unix=float(data.get("captured_at_unix", 0.0)),
            impulse=float(data.get("impulse", ASSAY_IMPULSE)),
            settle_ticks=int(data.get("settle_ticks", ASSAY_SETTLE_TICKS)),
            washout_ticks=int(data.get("washout_ticks", ASSAY_WASHOUT_TICKS)),
            dt=float(data.get("dt", ASSAY_DT)),
            effects=[ChemicalEffect.from_dict(e) for e in data.get("effects", [])],
        )

    def by_name(self, name: str) -> ChemicalEffect | None:
        """Look up one chemical's effect by name."""
        for e in self.effects:
            if e.name == name:
                return e
        return None

    def summary_lines(self) -> list[str]:
        """Human-readable effect table, one line per chemical."""
        lines = [
            f"{'chemical':<16} {'d_arousal':>9} {'d_valence':>9} "
            f"{'d_plast':>8} {'phase':<22} top_coupled"
        ]
        for e in self.effects:
            transition = (
                e.phase_pre if e.phase_pre == e.phase_slow
                else f"{e.phase_pre}->{e.phase_slow}"
            )
            coupled = ",".join(
                f"{name}{delta:+.2f}" for name, delta in e.top_coupled
            )
            lines.append(
                f"{e.name:<16} {e.d_arousal_slow:+9.3f} {e.d_valence_slow:+9.3f} "
                f"{e.d_plasticity_slow:+8.3f} {transition:<22} {coupled}"
            )
        return lines


def _is_finite_state(state: Any) -> bool:
    values = [
        state.arousal, state.valence, state.plasticity_gate,
        *state.chemicals.values(),
    ]
    return all(math.isfinite(v) for v in values)


def run_chemical_assay(
    client: GenesisClient,
    *,
    impulse: float = ASSAY_IMPULSE,
    settle_ticks: int = ASSAY_SETTLE_TICKS,
    washout_ticks: int = ASSAY_WASHOUT_TICKS,
    dt: float = ASSAY_DT,
) -> AssayResult:
    """Run the full 18-chemical impulse-response assay.

    Raises ValueError if any post-impulse state is non-finite (that
    would indicate dynamics corruption, not a chemical effect).
    """
    from .harness import settle

    result = AssayResult(
        captured_at_unix=time.time(),
        impulse=impulse,
        settle_ticks=settle_ticks,
        washout_ticks=washout_ticks,
        dt=dt,
    )
    settle(client, ticks=settle_ticks, dt=dt)

    for chem_id in sorted(CHEM_NAMES):
        name = CHEM_NAMES[chem_id]
        pre = client.get_state()

        # Drift control. The system is not quiescent between samples:
        # adenosine accumulates sleep pressure continuously while
        # awake, and the whole coupled system relaxes after startup.
        # Without a same-length, impulse-free control, that background
        # motion is indistinguishable from the impulse's own effect,
        # and gets reported as the chemical's response to the dose.
        #
        # Adenosine is the clearest case: its "response" to a 0.30
        # impulse measured +0.197, which is ordinary sleep-pressure
        # accumulation, not a response to anything. Subtracting the
        # control is what makes this a *causal* assay, as the module
        # describes itself.
        control_pre = client.get_state()
        for _ in range(FAST_TICKS):
            client.advance_physics(dt)
        control_fast = client.get_state()
        for _ in range(SLOW_EXTRA_TICKS):
            client.advance_physics(dt)
        control_slow = client.get_state()

        client.neuro_impulse(chem_id, impulse)
        for _ in range(FAST_TICKS):
            client.advance_physics(dt)
        fast = client.get_state()
        for _ in range(SLOW_EXTRA_TICKS):
            client.advance_physics(dt)
        slow = client.get_state()

        for snap, tag in ((fast, "fast"), (slow, "slow")):
            if not _is_finite_state(snap):
                raise ValueError(
                    f"non-finite state after {name} impulse ({tag} point)"
                )

        # Background motion over the measurement window, per chemical.
        fast_drift = {
            n: control_fast.chemicals[n] - control_pre.chemicals[n]
            for n in CHEM_NAMES.values()
        }
        slow_drift = {
            n: control_slow.chemicals[n] - control_pre.chemicals[n]
            for n in CHEM_NAMES.values()
        }
        d_self_fast = (
            fast.chemicals[name] - pre.chemicals[name] - fast_drift[name]
        )
        d_self_slow = (
            slow.chemicals[name] - pre.chemicals[name] - slow_drift[name]
        )

        coupled: list[tuple[str, float]] = []
        for other_id, other_name in CHEM_NAMES.items():
            if other_id == chem_id:
                continue
            coupled.append(
                (
                    other_name,
                    slow.chemicals[other_name]
                    - pre.chemicals[other_name]
                    - slow_drift[other_name],
                )
            )
        coupled.sort(key=lambda kv: abs(kv[1]), reverse=True)

        result.effects.append(
            ChemicalEffect(
                chem_id=chem_id,
                name=name,
                impulse=impulse,
                pre_self=pre.chemicals[name],
                post_fast_self=pre.chemicals[name] + d_self_fast,
                post_slow_self=pre.chemicals[name] + d_self_slow,
                d_arousal_fast=fast.arousal - pre.arousal,
                d_arousal_slow=slow.arousal - pre.arousal,
                d_valence_fast=fast.valence - pre.valence,
                d_valence_slow=slow.valence - pre.valence,
                d_plasticity_slow=slow.plasticity_gate - pre.plasticity_gate,
                d_encoding_slow=slow.encoding_weight - pre.encoding_weight,
                d_consolidation_slow=slow.consolidation_weight - pre.consolidation_weight,
                d_retrieval_slow=slow.retrieval_weight - pre.retrieval_weight,
                phase_pre=pre.phase_name,
                phase_fast=fast.phase_name,
                phase_slow=slow.phase_name,
                top_coupled=[[n, d] for n, d in coupled[:TOP_COUPLED]],
            )
        )
        settle(client, ticks=washout_ticks, dt=dt)

    return result
