"""Convention — Lewis's analysis of convention as a self-sustaining
regularity (Convention: A Philosophical Study, 1969).

Lewis's question: what separates a convention — driving on the
right, calling this thing "Tuesday" — from a natural law or a mere
coincidence of habit? His answer is game-theoretic. A regularity R
in a population P is a convention iff:

1. **Conformity** — almost everyone in P conforms to R;
2. **Mutual belief** — almost everyone believes almost everyone
   else conforms to R;
3. **Conditional preference** — each has a good and decisive reason
   to conform given that the others do (conformity is what each
   wants, conditional on the rest conforming — R is a coordination
   equilibrium: no one prefers to deviate alone);
4. **Arbitrariness** — there is an alternative regularity R′ such
   that, had the others conformed to R′, each would prefer to
   conform to R′ too. A convention could have been otherwise: left
   would have served as well as right. A regularity with no viable
   alternative is not conventional — it is just the thing to do;
5. **Common knowledge** — these conditions hold and are known to
   hold, to every iterate of "knows that" (implemented at the first
   order: each believes the regularity obtains and that the others
   so believe; deeper orders are the same check iterated over
   nested belief).

The discriminations the conditions make:

- *Habit is not convention*: a regularity everyone follows for
  independent reasons (each would do it whatever the others do)
  fails conditional preference — there is nothing to coordinate on.
- *Convention is not law*: a regularity forced by the situation
  (one viable option) fails arbitrariness — nothing else could have
  been conventional.
- *Dead convention*: a regularity still performed but no longer
  believed to be performed fails mutual belief — the convention has
  lapsed even while the motions continue.

Lewis's other use of "convention" — truth by convention, the
analytic as true by linguistic rule — lives where meaning is
settled: ``temporal_lobe.semantics.analytic``. Coordination is the
sense analyzed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

#: Lewis's "almost everyone" — the conformity and belief floors.
_ALMOST_ALL = 0.9


@dataclass(frozen=True)
class Agent:
    """One member of the population, as the analysis needs them:

    - ``action``: the regularity they actually follow;
    - ``expects``: the regularity they believe the others follow;
    - ``conform_payoff`` / ``deviate_payoff``: preference while the
      group regularity holds — conditional preference asks whether
      conforming beats deviating *given* the others conform;
    - ``alt_payoffs``: what each alternative regularity would pay
      this agent if it were the one others followed — arbitrariness
      asks whether some alternative could have sustained the same
      conditional conformity.
    """

    name: str
    action: str
    expects: str
    conform_payoff: float
    deviate_payoff: float
    alt_payoffs: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ConventionReport:
    """One clause of Lewis's analysis per field — a convention is the
    conjunction of all of them."""

    regularity: str
    conformity: float  # fraction conforming
    mutual_belief: float  # fraction believing the others conform
    conditional_preference: bool  # every agent prefers conforming over deviating
    arbitrary: bool  # some alternative could equally have been the convention
    threshold: float

    @property
    def is_convention(self) -> bool:
        return (
            self.conformity >= self.threshold
            and self.mutual_belief >= self.threshold
            and self.conditional_preference
            and self.arbitrary
        )

    @property
    def failure(self) -> str:
        """Which clause a non-convention falls on — the diagnosis,
        not just the verdict."""
        if self.conformity < self.threshold:
            return "the regularity is not actually followed"
        if self.mutual_belief < self.threshold:
            return "a dead convention — followed, but no longer believed"
        if not self.conditional_preference:
            return "mere habit — the conformity does not depend on the others'"
        if not self.arbitrary:
            return "no alternative could have been conventional — this is forced"
        return "conventional"


def is_convention(
    agents: Iterable[Agent],
    regularity: str,
    alternatives: Iterable[str] = (),
    *,
    threshold: float = _ALMOST_ALL,
) -> ConventionReport:
    """Evaluate a putative convention against Lewis's clauses.

    ``agents`` is the population; ``regularity`` the regularity under
    test (matched against each agent's ``action``); ``alternatives``
    the candidate other-conventions — arbitrariness holds when at
    least one alternative would pay every agent under conditional
    conformity (they would have conformed to it had the others).
    """
    pop = tuple(agents)
    if not pop:
        return ConventionReport(regularity, 0.0, 0.0, False, False, threshold)

    conformity = sum(a.action == regularity for a in pop) / len(pop)
    mutual_belief = sum(a.expects == regularity for a in pop) / len(pop)
    conditional_preference = all(a.conform_payoff > a.deviate_payoff for a in pop)
    arbitrary = any(
        all(a.alt_payoffs.get(alt, float("-inf")) > a.deviate_payoff for a in pop)
        for alt in alternatives
    )
    return ConventionReport(
        regularity,
        conformity,
        mutual_belief,
        conditional_preference,
        arbitrary,
        threshold,
    )


__all__ = ["Agent", "ConventionReport", "is_convention"]
