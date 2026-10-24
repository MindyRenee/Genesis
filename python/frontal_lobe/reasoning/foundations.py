"""Foundations — first principles and the structure of justification.

Aristotle's Posterior Analytics: demonstrative knowledge (episteme)
proceeds from premises that are true, primary, immediate, better
known than and prior to the conclusion — archai, first principles,
which cannot themselves be demonstrated (or nothing could be: the
regress would be infinite). What grasps them is not demonstration
but nous — direct apprehension.

Foundationalism formalizes the picture for an arbitrary belief set:
a belief is **basic** (held as a first principle, needing no
derivation) or **derived** (deduced from the basics), and the whole
edifice is *grounded* iff every non-basic belief is entailed by the
basics. Agrippa's trilemma is the design pressure this answers:
justification can only terminate at indemonstrable foundations,
loop circularly, or regress without end — and of those, only the
first stands.

What this module computes over ``logic``'s model theory:

- ``proves`` / ``demonstrate`` — deductive closure and its warrant;
- ``support`` — the minimal axiom subset entailing a claim
  (abduction over the axioms: which first principles carry it);
- ``indispensable`` — which axioms a proof cannot do without
  (drop one and the proof collapses — epistemic load-bearing);
- ``audit`` — the Agrippan check over a whole belief set: which
  beliefs are basic, which derived, which ungrounded (reachable by
  no chain from the basics — the infinite-regress symptom), and
  which trapped in circular justification (each citing the next).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from frontal_lobe.reasoning.logic import (
    BeliefSet,
    Deduction,
    Formula,
    entails,
    explain,
    verify,
)


@dataclass(frozen=True)
class Foundations:
    """A foundationalist belief system: indemonstrable first
    principles plus the deductive closure they ground.

    ``axioms`` are the archai — asserted, not derived. Everything
    else this system knows is exactly what the axioms entail.
    """

    axioms: BeliefSet = frozenset()

    # ── Demonstration ───────────────────────────────────────────

    def proves(self, q: Formula) -> bool:
        """axioms ⊨ q — q is derived knowledge: episteme."""
        return entails(self.axioms, q)

    def demonstrate(self, q: Formula) -> Deduction:
        """Attempt the deduction; on failure return the countermodel —
        the possibility where the principles hold and q does not."""
        return verify(self.axioms, q)

    def support(self, q: Formula) -> frozenset[Formula] | None:
        """The minimal axiom subset that entails q — which first
        principles the claim rests on. None when the axioms cannot
        reach it at all."""
        return explain((), q, self.axioms)

    def indispensable(self, q: Formula) -> frozenset[Formula]:
        """The axioms every proof of q needs: a is indispensable iff
        the whole set proves q but the set minus a does not — the
        load-bearing members of the foundation."""
        if not self.proves(q):
            return frozenset()
        return frozenset(a for a in self.axioms if not entails(self.axioms - {a}, q))

    # ── Membership ──────────────────────────────────────────────

    def is_basic(self, f: Formula) -> bool:
        return f in self.axioms

    def is_derived(self, f: Formula) -> bool:
        return f not in self.axioms and self.proves(f)


@dataclass(frozen=True)
class Audit:
    """The Agrippan verdict on a belief set."""

    basic: frozenset[Formula]  # held as first principles
    derived: frozenset[Formula]  # reached by deduction from the basics
    ungrounded: frozenset[Formula]  # no chain back to any basic — the regress
    circular: frozenset[frozenset[Formula]]  # mutual-justification cycles

    @property
    def grounded(self) -> bool:
        """Whether every belief bottoms out at a first principle —
        the whole edifice stands on archai."""
        return not self.ungrounded and not self.circular


def audit(
    foundations: Foundations,
    beliefs: Iterable[Formula],
) -> Audit:
    """Classify each belief by its justificatory status.

    - *basic*: an axiom — indemonstrable by design.
    - *derived*: entailed by the axioms.
    - *circular*: not derived from the axioms, but justified by a
      cycle of other beliefs (b₁ supports b₂ supports b₁) — Agrippa's
      second horn. Justification within the set is found by minimal
      abduction over the *other* beliefs: a belief's warrant is the
      smallest set of its fellows entailing it.
    - *ungrounded*: everything else — a belief with no route back to
      a first principle, the infinite-regress symptom.
    """
    axioms = foundations.axioms
    beliefs = frozenset(beliefs)
    basic = beliefs & axioms
    derived = frozenset(b for b in beliefs - basic if entails(axioms, b))
    rest = beliefs - basic - derived

    # Justification among the non-derived beliefs: b is grounded by
    # others if some subset of the rest entails it. Cycles in that
    # relation are circular justification; nodes unreachable from
    # any axiom-touching node are ungrounded.
    support_of: dict[Formula, frozenset[Formula]] = {}
    for b in rest:
        # The warrant within the set: the minimal subset of the OTHER
        # beliefs that alone entails b — explain's kb stays empty so
        # the support is what does the justifying, not the assumed set.
        others = beliefs - {b}
        support_of[b] = explain((), b, others) or frozenset()

    # Walk the justification graph; a cycle is any node that reaches
    # itself through support edges.
    circular_nodes: set[Formula] = set()
    for b in rest:
        frontier = list(support_of[b])
        seen = set(frontier)
        while frontier:
            x = frontier.pop()
            if x == b:
                circular_nodes.add(b)
                break
            for y in support_of.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    frontier.append(y)

    circular = frozenset(frozenset({b} | _cycle_members(b, support_of)) for b in circular_nodes)
    ungrounded = rest - circular_nodes
    return Audit(basic, derived, ungrounded, circular)


def _reach(b: Formula, support_of: Mapping[Formula, frozenset[Formula]]) -> set[Formula]:
    """Everything b's justification chain reaches."""
    out: set[Formula] = set()
    frontier = list(support_of.get(b, ()))
    while frontier:
        x = frontier.pop()
        if x in out:
            continue
        out.add(x)
        frontier.extend(support_of.get(x, ()))
    return out


def _cycle_members(b: Formula, support_of: Mapping[Formula, frozenset[Formula]]) -> set[Formula]:
    """The circle's true members — reachable from b and reaching b
    (the strongly connected component through it)."""
    forward = _reach(b, support_of)
    return {x for x in forward if b in _reach(x, support_of)}


__all__ = ["Audit", "Foundations", "audit"]
