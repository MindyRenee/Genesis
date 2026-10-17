"""Amygdala — the organ. A container for conditioned value.

This is the structure only: the fields a signed-value associative
learner must hold, with no behavior attached. The math that animates
it — Rescorla-Wagner acquisition, Pearce-Hall associability,
context-keyed ITC extinction — is documented here so whoever wires
the functions later knows what each field is FOR:

    δa = λa − net_a       net_a = Va[cs] − w_e · Ea[ctx][cs]
    δr = λr − net_r       net_r = Vr[cs] − w_e · Er[ctx][cs]
    λa = max(0, −λ),  λr = max(0, λ)    (signed US splits into channels)

    δa > 0 → Va[cs] += α·β·δa            aversive acquisition (BLA)
    δa < 0 → Ea[ctx][cs] += η·(−δa)      safety learning (ITC/vmPFC)
    δr > 0 → Vr[cs] += α·β·δr            appetitive acquisition (BLA)
    δr < 0 → Er[ctx][cs] += η·(−δr)      reward omission → extinction

    α[cs] ← γ·|δ| + (1−γ)·α[cs]          associability (Pearce-Hall 1980)

Design invariants the fields encode:

- **Opponent channels.** Va and Vr are separate populations, not one
  signed dial — appetitive and aversive learning have independent
  dynamics and independent extinction traces.
- **Extinction is new inhibitory learning.** E lives apart from V and
  is keyed by context; V is never decremented. Renewal and spontaneous
  recovery fall out of that structure.
- **Associability.** α tracks recent surprise — cues that mispredicted
  learn fastest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Chem channel indices (Rust canonical ordering) ────────────────
_CHEM_DOPAMINE = 0
_CHEM_NOREPINEPHRINE = 2
_CHEM_ACETYLCHOLINE = 3
_CHEM_CORTISOL = 6

# ── Learning constants ────────────────────────────────────────────
# β: base learning rate — fast enough for single-trial conditioning
# with strong US (fear learning is often one-shot), slow enough that
# weak noise doesn't condition.
_BETA = 0.35
# γ: Pearce-Hall decay — associability tracks recent surprise.
_GAMMA = 0.55
# η: extinction (ITC) learning rate — slower than acquisition, as in
# biology: safety is learned cautiously.
_ETA = 0.25
# w_e: extinction trace weight in the net readout.
_EXTINCTION_WEIGHT = 0.85
# α floor: a never-surprising cue still learns, slowly.
_ALPHA_MIN = 0.15
# α ceiling: keeps learning bounded under repeated shocks.
_ALPHA_MAX = 0.95
# Impulse floor: below this aversive net the low road stays silent —
# weak conditioned cues bias encoding but don't startle.
_IMPULSE_THRESHOLD = 0.25
# Capacity bound: the cue store is a conditioned-value table, not an
# unbounded log — past the cap, the cue whose recent surprise is
# lowest (least associable) is the one let go.
_MAX_CUES = 4096


@dataclass(slots=True)
class Amygdala:
    """The organ: BLA value channels, CeA associability, ITC traces.

    Basolateral complex — conditioned value, opponent channels:

        v_aversive:   cue → learned aversive strength Va[cs]  [0,1]
        v_appetitive: cue → learned appetitive strength Vr[cs] [0,1]

    Central nucleus — associability (Pearce-Hall), the attention gate:

        associability: cue → learning-rate multiplier α[cs]  [0,1]

    Intercalated cells / vmPFC — context-keyed safety learning.
    Extinction is a separate inhibitory trace, never unlearning:

        e_aversive:   context → cue → inhibitory strength Ea[ctx][cs]
        e_appetitive: context → cue → inhibitory strength Er[ctx][cs]

    Counters — introspection counters only; they gate nothing.
    """

    v_aversive: dict[str, float] = field(default_factory=dict)
    v_appetitive: dict[str, float] = field(default_factory=dict)
    associability: dict[str, float] = field(default_factory=dict)
    e_aversive: dict[str, dict[str, float]] = field(default_factory=dict)
    e_appetitive: dict[str, dict[str, float]] = field(default_factory=dict)
    conditioning_events: int = 0
    extinction_events: int = 0
    low_road_fires: int = 0

    # ── Conditioned readouts ─────────────────────────────────────

    def net_aversive(self, cue: str, ctx: str) -> float:
        """net_a = Va[cs] − w_e · Ea[ctx][cs]."""
        return self.v_aversive.get(cue, 0.0) - _EXTINCTION_WEIGHT * self.e_aversive.get(
            ctx, {}
        ).get(cue, 0.0)

    def net_appetitive(self, cue: str, ctx: str) -> float:
        """net_r = Vr[cs] − w_e · Er[ctx][cs]."""
        return self.v_appetitive.get(cue, 0.0) - _EXTINCTION_WEIGHT * self.e_appetitive.get(
            ctx, {}
        ).get(cue, 0.0)

    def low_road(self, cue: str, ctx: str) -> float:
        """Fast pre-cortical aversive output (LeDoux's low road).

        Returns the aversive net when it crosses the impulse floor,
        else 0.0 — a sub-threshold conditioned cue biases encoding but
        does not startle.
        """
        net = self.net_aversive(cue, ctx)
        if net < _IMPULSE_THRESHOLD:
            return 0.0
        self.low_road_fires += 1
        return net

    # ── Learning ─────────────────────────────────────────────────

    def condition(self, cue: str, us: float, ctx: str) -> None:
        """Rescorla-Wagner acquisition + Pearce-Hall associability.

        ``us`` is the signed unconditioned-stimulus value for this
        pairing (negative = aversive, positive = appetitive); the
        channel split (λa = max(0, −λ), λr = max(0, λ)) is applied
        inside. Negative prediction errors write *extinction* — a new
        context-keyed inhibitory trace — and never decrement V, so
        renewal and spontaneous recovery fall out of the structure.
        """
        if not cue or us == 0.0:
            return
        lam_a, lam_r = max(0.0, -us), max(0.0, us)
        delta_a = lam_a - self.net_aversive(cue, ctx)
        delta_r = lam_r - self.net_appetitive(cue, ctx)
        alpha = self.associability.get(cue, _ALPHA_MIN)
        if delta_a > 0.0:
            self.v_aversive[cue] = min(
                1.0, self.v_aversive.get(cue, 0.0) + alpha * _BETA * delta_a
            )
            self.conditioning_events += 1
        elif delta_a < 0.0:
            ea = self.e_aversive.setdefault(ctx, {})
            ea[cue] = min(1.0, ea.get(cue, 0.0) + _ETA * (-delta_a))
            self.extinction_events += 1
        if delta_r > 0.0:
            self.v_appetitive[cue] = min(
                1.0, self.v_appetitive.get(cue, 0.0) + alpha * _BETA * delta_r
            )
            self.conditioning_events += 1
        elif delta_r < 0.0:
            er = self.e_appetitive.setdefault(ctx, {})
            er[cue] = min(1.0, er.get(cue, 0.0) + _ETA * (-delta_r))
            self.extinction_events += 1
        # Pearce-Hall: associability tracks recent surprise — the
        # larger of the two channel errors is what this cue failed
        # to predict.
        surprise = max(abs(delta_a), abs(delta_r))
        self.associability[cue] = min(
            _ALPHA_MAX,
            max(_ALPHA_MIN, _GAMMA * surprise + (1.0 - _GAMMA) * alpha),
        )
        self._evict_weakest()

    # ── Housekeeping ─────────────────────────────────────────────

    def _evict_weakest(self) -> None:
        """Bound the cue store — drop the least associable cue."""
        if len(self.associability) <= _MAX_CUES:
            return
        victim = min(self.associability, key=self.associability.get)  # type: ignore[arg-type]
        self.associability.pop(victim, None)
        self.v_aversive.pop(victim, None)
        self.v_appetitive.pop(victim, None)
        for ctx_table in self.e_aversive.values():
            ctx_table.pop(victim, None)
        for ctx_table in self.e_appetitive.values():
            ctx_table.pop(victim, None)

    def to_dict(self) -> dict:
        return {
            "v_aversive": self.v_aversive,
            "v_appetitive": self.v_appetitive,
            "associability": self.associability,
            "e_aversive": self.e_aversive,
            "e_appetitive": self.e_appetitive,
            "conditioning_events": self.conditioning_events,
            "extinction_events": self.extinction_events,
            "low_road_fires": self.low_road_fires,
        }

    def restore(self, data: dict) -> None:
        self.v_aversive = dict(data.get("v_aversive") or {})
        self.v_appetitive = dict(data.get("v_appetitive") or {})
        self.associability = dict(data.get("associability") or {})
        self.e_aversive = {
            k: dict(v) for k, v in (data.get("e_aversive") or {}).items()
        }
        self.e_appetitive = {
            k: dict(v) for k, v in (data.get("e_appetitive") or {}).items()
        }
        self.conditioning_events = int(data.get("conditioning_events") or 0)
        self.extinction_events = int(data.get("extinction_events") or 0)
        self.low_road_fires = int(data.get("low_road_fires") or 0)


__all__ = ["Amygdala"]
