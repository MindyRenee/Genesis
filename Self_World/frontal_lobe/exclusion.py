"""Exclusion — reasoning by elimination, the first logic.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

The earliest logical organ humans grow. Preverbal infants — twelve to
nineteen months, before a single word — run the disjunctive syllogism:
watch two objects (snake, ball) disappear behind an occluder, watch a
cup scoop one up, and when the occluder drops to reveal the *scooped*
object still sitting there, they stare and their pupils dilate
(Cesana-Arlotti et al., Science 2018). The syllogism is

    P ∨ Q ; ¬P ; ∴ Q

—the process of elimination. Halberda's accompanying Perspective calls
it the earliest known foundation of human logical reasoning, and the
same oculomotor signature appears in monolingual and bilingual infants
alike, so it does not depend on language (Tégla´s et al.; Current
Biology 2023 review).

Two operators, and only two, carry the inference:

- **Disjunction** circumscribes the option space — "it is A or it is
  B" — and *yokes* the alternatives: they are jointly exhaustive, so
  the falsity of one transmits warrant to the rest. This is what
  separates genuine exclusion reasoning from mere avoidance — an
  organism that can only negate would skip the empty cup but gain no
  confidence about the other one; the disjunctive representation
  forces the confidence transfer (Bohn, Call & Völter's formulation
  of the Bermúdez/Burge point).

- **Negation** prunes: evidence that an option is false removes it,
  and the residue's mass rises by the Boolean conditioning below.

Developmentally it precedes everything else logical: single unary
predicates consolidate ~12 months (Halford's relational-complexity
ladder), explicit binary relations ~18 months, role-governed
relational structure ~3 years (Holyoak & Lu's Late System, the
frontoparietal machinery that this file's flat option space is the
precursor of — dlpfc.py's compositional grammar is its descendant).

In adults the exclusion network is left inferior frontal + dlPFC
maintaining the alternatives, and the violation signal is a phasic
**locus coeruleus** burst — the infant's pupil dilation is the
peripheral signature of LC-NE firing on violated expectation. Here
the space lives in prefrontal working memory (an organ on
``FrontalLobe``, per-problem contents like the dlpfc induction
state), and ``observe()`` reports surprisal so the caller can drive
the noradrenergic impulse through the daemon — the same impulse
channel the workspace uses for ignition orienting.


════════════════════════════════════════════════════════════════════════
MATH
════════════════════════════════════════════════════════════════════════

**Exclusive space** (the disjunctive syllogism's semantics: exactly
one option is true). Options carry prior weights, normalized over the
live set:

    P(h) = w(h) / Σ_{h' live} w(h')

**Elimination** removes an option and renormalizes — the warrant
transfer that is the entire content of "∴ Q":

    P(h_j | ¬h_i) = P(h_j) / (1 − P(h_i))

Eliminating the second-to-last option resolves the space; eliminating
the last *falsifies the disjunction itself* — the world was not in
the space the mind circumscribed.

**Observation.** Revealing the outcome o scores surprisal against the
standing distribution:

    S(o) = −log₂ P(o)          bits

S = 0 when the residue was already certain (full prediction, no LC
burst); S → ∞ when o was excluded or never entertained — an
impossible outcome, i.e. the disjunction was false. ``inf`` is
returned honestly; the caller caps it for impulse magnitude.

**Non-exclusive space** (``exclusive=False``) models the contrast
case: options are independent plausibilities ∈ [0,1]; elimination
removes mass without transfer — the avoidance strategy, kept so the
difference between negation-only and true disjunctive inference is
representable rather than asserted.


════════════════════════════════════════════════════════════════════════
THE FORMAL SEMANTICS THIS ORGAN RUNS
════════════════════════════════════════════════════════════════════════

The option space is **alternative / inquisitive semantics** made
mechanical (Hamblin 1973; Ciardelli, Groenendijk & Roelofsen 2018):
a proposition is a *set of alternative possibilities* — sets of
possible worlds — and an utterance's meaning is its context-change
potential, a map from information states to information states. A
disjunction's proposition has two or more alternatives, so it is
simultaneously **informative** (at least one possibility obtains)
and **inquisitive** (it requests that the discourse establish one
of them). ``entertain`` declares an alternative, ``eliminate`` is
the update intersection ``s ∩ ¬h_i``, ``resolved`` is the state in
which an alternative is established, ``falsified`` is the empty
state — and ``observe``'s surprisal is the information the outcome
actually carried against the standing possibilities, in bits.

Opposites pair throughout: entertain ↔ eliminate, inform ↔
inquire, resolved ↔ falsified, expected ↔ impossible, and the
whole organ's exclusive "or" ↔ its non-exclusive contrast.


════════════════════════════════════════════════════════════════════════
CENTERING — WHERE THE PRIORS COME FROM
════════════════════════════════════════════════════════════════════════

When the space holds referent candidates, the priors are not free:
centering theory (Grosz, Joshi & Weinstein 1995) orders each
utterance's forward-looking centers Cf by discourse salience —
grammatical role (subject > object > oblique) and the first-mention
advantage — and the backward-looking center Cb of the next
utterance is *the highest-ranked element of Cf(U_{n-1}) realized in
it*. That ranking is a partial order, not a measure, so callers map
rank → prior monotonically (a geometric ρ^rank encodes the order
without inventing metric content the theory does not supply);
``observe()`` then scores how surprising the realized referent was
against the salience ordering — the prediction Cp makes and the
surprise a violated Cb carries.

Contents are working memory: ``reset()`` opens a new ambiguity.

References:
- Cesana-Arlotti, N., et al. (2018). Precursors of logical reasoning
  in preverbal human infants. Science, 359, 1263–1266.
- Halberda, J. (2018). Logic in babies. Science, 359, 1214–1215.
- Mody, S., & Carey, S. (2016). The emergence of reasoning by the
  disjunctive syllogism in early childhood. Cognition, 154, 40–48.
- Holyoak, K. J., & Lu, H. (2021). Emergence of relational reasoning.
  Current Opinion in Behavioral Sciences, 37, 78–84.
- Ciardelli, I., Groenendijk, J., & Roelofsen, F. (2018).
  Inquisitive Semantics. Oxford University Press.
- Grosz, B. J., Joshi, A. K., & Weinstein, S. (1995). Centering: A
  framework for modeling the local coherence of discourse.
  Computational Linguistics, 21(2), 203–225.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "Elimination",
    "Observation",
    "OptionSpace",
]


@dataclass(frozen=True)
class Elimination:
    """The result of pruning options.

    ``eliminated`` — the options removed this call (missing options
    are skipped, not errors). ``transferred`` — mass each survivor
    gained from the removal; empty in a non-exclusive space, since
    avoidance transfers nothing. ``resolved`` — the single surviving
    option when the space collapsed to a point, else ``None``.
    ``falsified`` — the disjunction itself failed: every option is
    gone.
    """

    eliminated: tuple[Hashable, ...]
    transferred: Mapping[Hashable, float]
    resolved: Hashable | None
    falsified: bool


@dataclass(frozen=True)
class Observation:
    """The result of revealing an outcome.

    ``surprisal`` — −log₂ P(outcome) under the standing space, in
    bits; ``inf`` when the outcome was excluded or never entertained.
    ``impossible`` — that zero-probability case: the disjunction was
    wrong, not merely unlikely. ``resolved`` — the observed option
    (reality collapses the space to what was seen). ``falsified`` —
    the space no longer stands.
    """

    outcome: Hashable
    surprisal: float
    impossible: bool
    resolved: Hashable | None
    falsified: bool


class OptionSpace:
    """A live hypothesis space held in working memory.

    Options are mutually exclusive alternatives (``exclusive=True``,
    the disjunctive syllogism: exactly one is true) or independent
    plausibilities (``exclusive=False``, the avoidance contrast).

    ``entertain`` opens alternatives, ``eliminate`` prunes them with
    warrant transfer, ``observe`` reveals the outcome and scores the
    surprisal, and ``resolved``/``falsified`` report the two terminal
    states — the syllogism concluded, or the disjunction itself
    refuted.

    ``on_surprise(surprisal, impossible)``, if given, fires on every
    observation — the LC drive; magnitude policy (thresholds, caps,
    alertness and sleep gating) belongs to the caller.
    """

    def __init__(
        self,
        options: Iterable[Hashable] = (),
        priors: Mapping[Hashable, float] | None = None,
        *,
        exclusive: bool = True,
        on_surprise: Callable[[float, bool], None] | None = None,
    ) -> None:
        self.exclusive = exclusive
        self._on_surprise = on_surprise
        self._live: dict[Hashable, float] = {}
        self._falsified = False
        for o in options:
            self.entertain(o, (priors or {}).get(o, 1.0))

    # ── The space ──────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the contents — a new ambiguity, new options."""
        self._live.clear()
        self._falsified = False

    def entertain(self, option: Hashable, prior: float = 1.0) -> None:
        """Open an alternative with a prior weight (exclusive) or
        standalone plausibility ∈ [0,1] (non-exclusive)."""
        if prior < 0 or (not self.exclusive and prior > 1):
            raise ValueError(f"prior out of range: {prior}")
        self._live[option] = float(prior)
        self._falsified = False

    def __len__(self) -> int:
        return len(self._live)

    def __contains__(self, option: Hashable) -> bool:
        return option in self._live

    @property
    def resolved(self) -> Hashable | None:
        """The surviving option, when exactly one remains."""
        return next(iter(self._live)) if len(self._live) == 1 else None

    @property
    def falsified(self) -> bool:
        """The disjunction itself failed — nothing left to believe."""
        return self._falsified

    def posterior(self) -> dict[Hashable, float]:
        """The standing distribution.

        Exclusive: normalized masses summing to 1 over the live set.
        Non-exclusive: each option's independent plausibility.
        """
        if not self.exclusive:
            return dict(self._live)
        total = sum(self._live.values())
        if total <= 0:
            return dict.fromkeys(self._live, 0.0)
        return {o: w / total for o, w in self._live.items()}

    # ── Negation ───────────────────────────────────────────────

    def eliminate(self, *options: Hashable) -> Elimination:
        """Prune options as false — the syllogism's ¬P step.

        In an exclusive space the freed mass is redistributed over
        the survivors (warrant transfer — confidence in the residue
        *rises*; that rise is the difference between inference and
        avoidance). In a non-exclusive space elimination only
        removes.
        """
        hit = tuple(o for o in options if o in self._live)
        if not hit:
            return Elimination((), {}, self.resolved, self._falsified)

        before = self.posterior() if self.exclusive else {}
        freed = sum(before.get(o, 0.0) for o in hit)
        for o in hit:
            del self._live[o]

        transferred: dict[Hashable, float] = {}
        if self.exclusive and freed < 1.0:
            after = self.posterior()
            transferred = {o: after[o] - before[o] for o in after}
        if not self._live:
            self._falsified = True
        return Elimination(hit, transferred, self.resolved, self._falsified)

    # ── The moment of truth ────────────────────────────────────

    def observe(self, outcome: Hashable) -> Observation:
        """Reveal what the world actually held — score the surprisal.

        The LC phasic burst is proportional to −log₂ P(outcome):
        nothing when the residue was certain, maximal when the
        outcome was excluded or never entertained — in which case the
        disjunction itself is falsified and reality still wins (the
        space collapses to what was *seen*).
        """
        p = self.posterior().get(outcome, 0.0)
        impossible = p <= 0.0
        surprisal = math.inf if impossible else -math.log2(p)
        if impossible:
            self._falsified = True
        self._live = {outcome: 1.0}
        obs = Observation(outcome, surprisal, impossible, outcome, self._falsified)
        if self._on_surprise is not None:
            self._on_surprise(surprisal, impossible)
        return obs
