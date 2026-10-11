"""Hypothesis layer — candidate rules and goals, verified by the world.

``TaskCompetence`` learns what transitions *usually* happen
(``EffectModel``) and consolidates verified episodes into skills.
This module holds the layer in between: named, falsifiable beliefs
about a task — "touching a red thing collects it", "the level ends
when nothing is left to take" — that stand or fall on observed
transitions, and that a policy can act on *before* they are proven
(testing an undecided hypothesis is itself worth an action).

Two families share one machinery:

- **Mechanics** — ``context ∧ action ⇒ effect``. Abduced from
  transitions the adapter marks noteworthy (a contact, a spawn, a
  recolor). Every later transition matching the context and action
  is a trial: the effect holds (support) or doesn't (violation).
- **Goals** — conditions inferred from terminal states. A win's
  final state is abductive evidence: the variables that held — above
  all, the variables the last transition changed — are candidate
  success conditions. A goal hypothesis is *falsified* when its
  conditions are fully satisfied and the episode keeps going, and
  *supported* each time they hold at a win. ``avoid`` hypotheses are
  the mirror image, abduced from losses.

The vocabulary is ``GoalCondition`` over flattened state mappings —
the same language ``TaskCompetence.verify`` already speaks — plus
three transition-only keys the ledger synthesizes for mechanics:
``<key>.after`` (the post-value), ``<key>.delta`` (numeric change),
``<key>.changed`` (did it move at all). Effects can therefore say
"decrements the count" without naming a bound value.

Hypotheses live on the ``TaskSchema``: a new environment that
schema-matches inherits its hypotheses as *priors to re-verify*,
which is where cross-task generalization actually comes from.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .competence import GoalCondition, Scalar, State, _as_float, _flatten_state, _hash

__all__ = [
    "HYPOTHESIS_CAP",
    "Hypothesis",
    "HypothesisLedger",
    "transition_view",
]

# How much evidence a hypothesis needs before it may stand as
# verified, and how decisively violations may kill it. The thresholds
# are deliberately low — hypotheses gate *attention and bias*, not
# actions outright, so a wrong verification is a mistaken lean, not
# a committed plan.
_VERIFY_SUPPORT = 3
_VERIFY_CONFIDENCE = 0.75
_REFUTE_VIOLATIONS = 2
_REFUTE_CONFIDENCE = 0.4

#: A schema carries at most this many live hypotheses — abduction
#: could otherwise mint a rule per noise fluctuation. Weakest
#: (refuted-then-least-confident) are evicted first.
HYPOTHESIS_CAP = 24

#: Weight of ``avoid`` evidence against a state's goal score.
_AVOID_WEIGHT = 0.6


def transition_view(before: State, after: State) -> dict[str, Scalar]:
    """Flatten a transition into the keys mechanic hypotheses test.

    ``key`` carries the before-value, ``key.after`` the post-value,
    ``key.delta`` the numeric change (when both ends are numeric),
    and ``key.changed`` whether the variable moved at all — so an
    effect condition like ``GoalCondition("objects.count.delta",
    "lt", 0)`` reads as "the action decremented the count".
    """
    b = _flatten_state(before)
    a = _flatten_state(after)
    view: dict[str, Scalar] = dict(b)
    for key in b.keys() | a.keys():
        bv, av = b.get(key), a.get(key)
        view[f"{key}.after"] = av
        view[f"{key}.changed"] = bv != av
        bn, an = _as_float(bv), _as_float(av)
        if bn is not None and an is not None:
            view[f"{key}.delta"] = an - bn
    return view


@dataclass(slots=True)
class Hypothesis:
    """One falsifiable belief about a task.

    ``kind`` is ``"mechanic"`` (context+action ⇒ effect),
    ``"goal"`` (conditions the winning state satisfies), or
    ``"avoid"`` (conditions the losing state satisfied). ``status``
    is derived from the evidence counters — a hypothesis is never
    declared verified by fiat, only by trials.
    """

    hypothesis_id: str
    kind: str
    # Preconditions over flat state — empty means "always on".
    context: tuple[GoalCondition, ...] = ()
    # The action a mechanic hypothesis governs; None for goal/avoid.
    action: str | None = None
    # Mechanic: conditions over the transition view (see
    # ``transition_view``). Goal/avoid: conditions over flat state.
    effect: tuple[GoalCondition, ...] = ()
    support: int = 0
    violations: int = 0
    note: str = ""

    @property
    def confidence(self) -> float:
        return (self.support + 0.5) / (self.support + self.violations + 1.0)

    @property
    def status(self) -> str:
        if self.violations >= _REFUTE_VIOLATIONS and self.confidence < _REFUTE_CONFIDENCE:
            return "refuted"
        if self.support >= _VERIFY_SUPPORT and self.confidence >= _VERIFY_CONFIDENCE:
            return "verified"
        return "proposed"

    def applies(self, state: Mapping[str, Scalar]) -> bool:
        """Every precondition satisfied by this flat state?"""
        return all(c.satisfied(state) for c in self.context)

    def holds(self, state: Mapping[str, Scalar]) -> bool:
        """Every effect condition satisfied by this state/view?"""
        return bool(self.effect) and all(c.satisfied(state) for c in self.effect)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind,
            "context": [c.to_dict() for c in self.context],
            "action": self.action,
            "effect": [c.to_dict() for c in self.effect],
            "support": self.support,
            "violations": self.violations,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Hypothesis:
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            kind=str(data["kind"]),
            context=tuple(
                GoalCondition.from_dict(c) for c in data.get("context", [])
            ),
            action=(None if data.get("action") is None else str(data["action"])),
            effect=tuple(
                GoalCondition.from_dict(c) for c in data.get("effect", [])
            ),
            support=int(data.get("support", 0)),
            violations=int(data.get("violations", 0)),
            note=str(data.get("note", "")),
        )


@dataclass(slots=True)
class HypothesisLedger:
    """The live hypothesis set for one task schema.

    Pure bookkeeping: abduction and verification are driven by
    :class:`TaskCompetence`, which owns the observation stream.
    """

    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)

    # ── Mutation ──────────────────────────────────────────────

    def add(self, hypothesis: Hypothesis) -> Hypothesis:
        """Merge or insert a hypothesis; refuted stays down-weighted."""
        existing = self.hypotheses.get(hypothesis.hypothesis_id)
        if existing is not None:
            return existing
        if len(self.hypotheses) >= HYPOTHESIS_CAP:
            # Evict the weakest: refuted first, then least confident.
            victim = min(
                self.hypotheses.values(),
                key=lambda h: (h.status != "refuted", h.confidence),
            )
            if victim.confidence >= hypothesis.confidence:
                return victim
            del self.hypotheses[victim.hypothesis_id]
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        return hypothesis

    # ── Abduction ─────────────────────────────────────────────

    def abduce_mechanics(
        self,
        action: str,
        before: State,
        after: State,
        *,
        context_keys: Iterable[str] = (),
        max_new: int = 3,
    ) -> list[Hypothesis]:
        """Propose rules for what just changed.

        Each changed state variable seeds one hypothesis per offered
        context fact: ``context ∧ action ⇒ key moved``. Context keys
        are caller-chosen salient facts (``"touching.color:5"``); an
        empty context list yields the unconditional form, which
        accumulates violations wherever the rule is really
        conditional — falsification is the point.
        """
        view = transition_view(before, after)
        changed = [
            key
            for key, v in view.items()
            if key.endswith(".changed") and v
        ]
        new: list[Hypothesis] = []
        for changed_key in changed:
            key = changed_key[: -len(".changed")]
            after_v = view[f"{key}.after"]
            delta = view.get(f"{key}.delta")
            # Prefer the weakest sufficient claim: "the value changed
            # by delta" when numeric, else "it became this".
            if delta is not None and delta != 0:
                effect = (GoalCondition(f"{key}.delta", "eq", delta),)
            else:
                effect = (GoalCondition(f"{key}.after", "eq", after_v),)
            contexts: list[tuple[GoalCondition, ...]] = [()]
            flat = _flatten_state(before)
            for ck in context_keys:
                if ck in flat and flat[ck] is not None:
                    contexts.append((GoalCondition(ck, "eq", flat[ck]),))
            for ctx in contexts:
                h = Hypothesis(
                    hypothesis_id=_hash(
                        {
                            "kind": "mechanic",
                            "action": str(action),
                            "context": [c.descriptor() for c in ctx],
                            "effect": [c.descriptor() for c in effect],
                        }
                    ),
                    kind="mechanic",
                    context=ctx,
                    action=str(action),
                    effect=effect,
                    support=1,
                    note=f"abduced from {action} on {key}",
                )
                merged = self.add(h)
                if merged is h:
                    new.append(h)
                else:
                    merged.support += 1
                if len(new) >= max_new:
                    return new
        return new

    def abduce_terminal(
        self, won: bool, final_state: State, *, changed_last: Iterable[str] = ()
    ) -> list[Hypothesis]:
        """Propose goal (win) or avoid (loss) conditions from an ending.

        On a win, every variable of the final state is a candidate —
        but variables the last transition changed are weighted first,
        because the change that ended the episode is usually the
        condition that satisfied the goal. On a loss the same shape
        becomes an ``avoid`` hypothesis: "states like this killed me".
        """
        flat = _flatten_state(final_state)
        kind = "goal" if won else "avoid"
        keys = [k for k in changed_last if k in flat]
        keys += [k for k in flat if k not in keys]
        new: list[Hypothesis] = []
        for key in keys[:6]:
            value = flat[key]
            if value is None:
                continue
            effect = (GoalCondition(key, "eq", value),)
            h = Hypothesis(
                hypothesis_id=_hash(
                    {"kind": kind, "effect": [c.descriptor() for c in effect]}
                ),
                kind=kind,
                effect=effect,
                support=1 if won else 0,
                note=f"abduced from {'win' if won else 'loss'} state",
            )
            merged = self.add(h)
            if merged is h:
                new.append(h)
            elif won:
                merged.support += 1
        return new

    # ── Verification ──────────────────────────────────────────

    def observe_transition(
        self, action: str, before: State, after: State
    ) -> int:
        """Score every applicable mechanic hypothesis on this transition.

        Returns how many hypotheses were put on trial — the caller
        can read "nothing matched" as "my rule coverage here is thin".
        """
        view = transition_view(before, after)
        flat = _flatten_state(before)
        tested = 0
        for h in self.hypotheses.values():
            if h.kind != "mechanic" or h.action != str(action):
                continue
            if not h.applies(flat):
                continue
            tested += 1
            if h.holds(view):
                h.support += 1
            else:
                h.violations += 1
        return tested

    def observe_continuation(self, state: State) -> None:
        """Non-terminal evidence: a satisfied goal that play moved past
        was never the goal. Call with the *before* state of a recorded
        transition — the state is proven non-terminal by the fact that
        a transition out of it exists."""
        flat = _flatten_state(state)
        for h in self.hypotheses.values():
            if h.kind != "goal" or h.status == "refuted":
                continue
            if h.holds(flat):
                h.violations += 1

    # ── Read side ─────────────────────────────────────────────

    def best_goal(self) -> Hypothesis | None:
        """Strongest standing goal hypothesis, verified first."""
        goals = [
            h
            for h in self.hypotheses.values()
            if h.kind == "goal" and h.status != "refuted"
        ]
        if not goals:
            return None
        return max(
            goals,
            key=lambda h: (h.status == "verified", h.confidence, h.support),
        )

    def goal_score(self, state: State) -> float:
        """Satisfaction of the best goal hypothesis, minus avoid weight."""
        flat = _flatten_state(state)
        goal = self.best_goal()
        score = 0.0
        if goal is not None:
            total = sum(c.weight for c in goal.effect) or 1.0
            score = (
                sum(c.score(flat) * c.weight for c in goal.effect) / total
            ) * goal.confidence
        for h in self.hypotheses.values():
            if h.kind == "avoid" and h.status != "refuted" and h.holds(flat):
                score -= _AVOID_WEIGHT * h.confidence
        return score

    def epistemic_bonus(self, action: str, state: State) -> float:
        """How much trying ``action`` here would decide something.

        A proposed mechanic hypothesis whose context holds and whose
        action is offered is an experiment waiting to be run — worth
        more while its evidence is thin and its confidence undecided.
        """
        flat = _flatten_state(state)
        bonus = 0.0
        for h in self.hypotheses.values():
            if h.kind != "mechanic" or h.status != "proposed":
                continue
            if h.action != str(action) or not h.applies(flat):
                continue
            evidence = h.support + h.violations
            undecided = 1.0 - abs(h.confidence - 0.5) * 2.0
            bonus += 0.15 * undecided / (1.0 + 0.5 * evidence)
        return min(0.5, bonus)

    def to_dict(self) -> dict[str, Any]:
        return {"hypotheses": [h.to_dict() for h in self.hypotheses.values()]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HypothesisLedger:
        ledger = cls()
        for item in data.get("hypotheses", []):
            h = Hypothesis.from_dict(item)
            ledger.hypotheses[h.hypothesis_id] = h
        return ledger
