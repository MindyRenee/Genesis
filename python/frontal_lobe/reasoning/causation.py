"""Causation — the analytic counterfactual theory (Lewis 1973, 2000).

Lewis's analysis: *c* causes *e* iff there is a chain of
counterfactual dependencies from c to e — causation is the ancestral
(transitive closure) of counterfactual dependence. Dependence: had c
not occurred, e would not have occurred — evaluated at the closest
¬c-worlds.

The evaluation needs more than belief revision: a Ramsey-style
minimal mutilation of a flat belief set leaves *e* standing when its
support is merely removed — logic alone cannot retract a conclusion
whose ground is cut. What the counterfactual needs is a mechanism:
a **structural causal model** (Pearl's equations are the machine
version of Lewis's similarity ordering — the closest antecedent
world keeps the mechanisms and perturbs only the cause). Each
variable carries an equation over its parents; intervening
do(c = v) cuts c's equation and fixes its value, then the mechanism
recomputes downstream. Counterfactual dependence is read off the
surgically altered model directly.

What this implements:

- ``depends(c, e)`` — e obtains actually and fails in the do(¬c)
  world. Both must actually occur: causation relates events, and an
  event that never happened causes nothing.
- ``causes(c, e)`` — the ancestral: a chain c → d₁ → … → e of
  dependencies, so causation reaches where direct dependence is cut.
  This is Lewis's answer to early preemption: the preempted backup
  leaves dependence broken, but the actual process stands as a chain
  of dependencies through its intermediates.

What it does not decide: *trumping* (two perfectly adequate
simultaneous causes, neither needed) and *late preemption* with no
intermediate to carry the chain are the cases where the
counterfactual theory needs Lewis's later refinement — causation as
*influence* ("Causation as Influence", 2000) — graded ranges of
whether/when/how alterations. The dependence graph exposes that
boundary honestly rather than papering over it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from frontal_lobe.reasoning.logic import Formula, Model

#: Fixpoint bound for equation evaluation — an acyclic model
#: converges within its own depth; a cyclic one exceeds this and
#: is reported rather than silently truncated.
_MAX_PASSES = 64


@dataclass(frozen=True)
class StructuralModel:
    """A structural causal model — variables and the mechanisms
    (formulas over parents) that produce them.

    ``equations`` maps each endogenous variable to the mechanism
    determining it; ``exogenous`` holds the background values no
    mechanism explains. Values are Boolean — an event either occurs
    or does not, which is the granularity Lewis's analysis assumes.
    """

    equations: Mapping[str, Formula]
    exogenous: Mapping[str, bool] | None = None

    def variables(self) -> frozenset[str]:
        out: set[str] = set(self.equations) | set(self.exogenous or ())
        for f in self.equations.values():
            out |= set(f.atoms())
        return frozenset(out)

    def evaluate(self, evidence: Mapping[str, bool] | None = None) -> Model:
        """Solve the model — propagate exogenous evidence through the
        mechanisms by fixpoint. Acyclic models converge within their
        depth; a model that fails to converge is cyclic, and cycles
        are reported rather than silently halved."""
        values = dict(self.exogenous or {})
        values.update(evidence or {})
        for _ in range(_MAX_PASSES):
            changed = False
            for var, f in self.equations.items():
                v = f.evaluate(values)
                if values.get(var) != v:
                    values[var] = v
                    changed = True
            if not changed:
                return values
        raise ValueError("equations did not converge — the model is cyclic")

    def intervene(self, var: str, value: bool) -> tuple[StructuralModel, Model]:
        """do(var = value) — cut the variable's mechanism and hold it
        fixed. Surgery, not conditioning: downstream variables keep
        their own mechanisms; only the intervened variable's incoming
        arrows are severed — the closest-world semantics Lewis's
        similarity ordering describes."""
        eqs = {k: f for k, f in self.equations.items() if k != var}
        return StructuralModel(eqs, self.exogenous), {var: value}

    def depends(
        self,
        c: str,
        e: str,
        actual: Model | None = None,
    ) -> bool:
        """Counterfactual dependence — e obtains actually, and at the
        closest world where c does not occur, e does not either."""
        actual = self.evaluate() if actual is None else actual
        if not actual.get(c, False) or not actual.get(e, False):
            return False
        model, fixed = self.intervene(c, False)
        return not model.evaluate(fixed).get(e, False)

    def causes(
        self,
        c: str,
        e: str,
        events: Iterable[str] | None = None,
        actual: Model | None = None,
    ) -> bool:
        """Causation as the ancestral of dependence — BFS over
        intermediate events; c causes e iff some chain of
        dependencies runs from c to e. This is Lewis's handling of
        early preemption: the backup process breaks direct
        dependence, but the actual mechanism remains a chain."""
        actual = self.evaluate() if actual is None else actual
        chain = self.causal_chain(c, e, events, actual)
        return chain is not None

    def causal_chain(
        self,
        c: str,
        e: str,
        events: Iterable[str] | None = None,
        actual: Model | None = None,
    ) -> tuple[str, ...] | None:
        """The witnessing chain for causation, or None — a path
        through the dependence graph from c to e (shortest first)."""
        actual = self.evaluate() if actual is None else actual
        if not actual.get(c, False) or not actual.get(e, False):
            return None
        if c == e:
            return (c,)
        pool = [v for v in (events or self.variables()) if v != c]
        # Dependence graph among the event pool, computed once —
        # an edge x → y iff y counterfactually depends on x.
        edges: dict[str, set[str]] = {v: set() for v in [c, *pool]}
        for x in [c, *pool]:
            for y in pool:
                if x == y:
                    continue
                if self.depends(x, y, actual):
                    edges[x].add(y)
        # BFS for a path c → … → e
        frontier: list[tuple[str, ...]] = [(c,)]
        seen = {c}
        while frontier:
            path = frontier.pop(0)
            for nxt in edges[path[-1]]:
                if nxt == e:
                    return (*path, e)
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((*path, nxt))
        return None


def depends(
    model: StructuralModel,
    c: str,
    e: str,
    actual: Model | None = None,
) -> bool:
    """Functional shorthand for ``model.depends``."""
    return model.depends(c, e, actual)


def causes(
    model: StructuralModel,
    c: str,
    e: str,
    events: Iterable[str] | None = None,
    actual: Model | None = None,
) -> bool:
    """Functional shorthand for ``model.causes``."""
    return model.causes(c, e, events, actual)


__all__ = ["StructuralModel", "causes", "depends"]
