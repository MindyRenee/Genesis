"""Dorsolateral prefrontal cortex — rule induction from examples.

Mid-DLPFC is the brain's hypothesis generator: it is active during
rule *search* and *acquisition* and disengages once the rule is
learned, when control passes to temporal and motor systems
(Crescentini et al., 2011). What it computes during acquisition is a
posterior over compositional rules — the rational analysis of
rule-based concept learning (Goodman, Tenenbaum, Feldman &
Griffiths, 2008):

    P(h | E) ∝ P(h | G) · P(E | h)

# Hypothesis space — a DNF grammar

Rules are formulae in disjunctive normal form over signed feature
literals: each formula is a definition, a disjunction of conjunctive
"senses" —

    ∀x  ℓ(x) ⇔ (f1 ∧ ¬f3) ∨ (f2 ∧ f4)

Grammar productions (D → C | D∨C ; C → P | C∧P ; P → ±fᵢ) give
every canonical formula exactly one derivation.

# Prior — production reuse, Dirichlet-marginalized

Production probabilities are integrated over a symmetric
Dirichlet(α=1), so a formula's prior is the Dirichlet–multinomial
marginal of its derivation counts (Goodman et al., eq. 4):

    P(h) = ∏_X [ Γ(k_X) / Γ(k_X + n_X) · ∏_j n_{X,j}! ]

for non-terminal X with k_X productions used n_X = Σⱼ n_{X,j} times.
Short derivations win, and reusing a literal across senses is
cheaper than introducing a new one — a Bayesian Occam bias over
syntax rather than extension.

# Likelihood — outliers, not exact match

Labels need not all be explainable: any example may be an outlier,
so the likelihood falls off exponentially with Q(h,E), the number
of examples the formula misclassifies (Goodman et al., eq. 5):

    P(E | h) ∝ exp(−b · Q(h, E))

The b→∞ limit is exact-match verification; finite b tolerates a
mislabeled example rather than abandoning the rule.

# Inference

The hypothesis space is enumerated up to a complexity bound — the
posterior is exact within it, the way working memory is finite.
Predictions are posterior-weighted votes, P(ℓ(x)) = Σ_h P(h|E)·h(x).
Confidence is the posterior mass on the MAP formula; when it crosses
the acquisition threshold the rule is treated as learned — the
mid-DLPFC disengagement signal of Crescentini et al.

References:
- Goodman, N. D., Tenenbaum, J. B., Feldman, J., & Griffiths, T. L.
  (2008). A rational analysis of rule-based concept learning.
  Cognitive Science, 32(1), 108–154.
- Crescentini, C., Seyed-Allaei, S., De Pisapia, N., Jovicich, J.,
  Amati, D., & Shallice, T. (2011). Mechanisms of rule acquisition
  and rule following in inductive reasoning. Journal of
  Neuroscience, 31(21), 7763–7774.
- Wallis, J. D., Anderson, K. C., & Miller, E. K. (2001). Single
  neurons in prefrontal cortex encode abstract rules. Nature,
  411, 953–956.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

__all__ = [
    "DorsolateralPFC",
    "InducedRule",
    "InductionResult",
    "Literal",
]

_MAX_EXAMPLES = 512
_MAX_HYPOTHESES = 250_000


@dataclass(frozen=True)
class Literal:
    """One signed predicate: a feature that is present or absent."""

    feature: str
    present: bool = True

    def holds(self, features: Collection[str]) -> bool:
        return (self.feature in features) == self.present

    def describe(self) -> str:
        return self.feature if self.present else f"¬{self.feature}"


@dataclass(frozen=True)
class InducedRule:
    """A DNF formula: a disjunction of conjunctive senses.

    ``senses`` is canonically ordered so syntactically identical
    formulae are the same object — the DNF grammar gives each
    formula a single derivation.
    """

    senses: tuple[frozenset[Literal], ...]

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                (frozenset(s) for s in self.senses),
                key=lambda s: sorted(lit.describe() for lit in s),
            )
        )
        object.__setattr__(self, "senses", ordered)

    def holds(self, features: Collection[str]) -> bool:
        """The formula's extension: any sense fully satisfied."""
        return any(
            all(lit.holds(features) for lit in sense) for sense in self.senses
        )

    @property
    def n_senses(self) -> int:
        return len(self.senses)

    @property
    def n_literals(self) -> int:
        return sum(len(s) for s in self.senses)

    def describe(self) -> str:
        """Formal notation — the rule itself, not prose about it."""
        parts = []
        for sense in self.senses:
            lits = " ∧ ".join(sorted(lit.describe() for lit in sense))
            parts.append(f"({lits})" if len(sense) > 1 else lits)
        return " ∨ ".join(parts)


@dataclass
class InductionResult:
    """The induced posterior over rules.

    ``confidence`` is the mass on the MAP rule; ``acquired`` is the
    mid-DLPFC disengagement signal — the rule has been learned and
    hypothesis search may stop.
    """

    posterior: list[tuple[InducedRule, float]]
    map_rule: InducedRule | None
    confidence: float
    entropy: float
    acquired: bool
    enumerated: int


class DorsolateralPFC:
    """Bayesian rule induction over a bounded DNF hypothesis space.

    ``observe(features, label)`` adds a labeled example; ``induce()``
    recomputes the posterior lazily; ``predict(features)`` returns the
    posterior-weighted probability that an unlabeled item belongs.

    The hypothesis space is all satisfiable DNF formulae of at most
    ``max_senses`` senses of at most ``max_literals`` literals over the
    vocabulary of features seen so far (both polarities). When the space
    would exceed ``_MAX_HYPOTHESES``, complexity bounds are lowered in
    the order that truncates the lowest-prior tail first.
    """

    def __init__(
        self,
        *,
        exception_cost: float = 4.0,
        max_senses: int = 3,
        max_literals: int = 3,
        acquire_threshold: float = 0.9,
    ) -> None:
        self.exception_cost = exception_cost
        self.max_senses = max_senses
        self.max_literals = max_literals
        self.acquire_threshold = acquire_threshold
        # Unkeyed evidence and keyed corrections: a keyed example
        # replaces the earlier label for that item rather than
        # double-counting it (feedback revealing a true label is new
        # evidence — a correction, not a second vote).
        self._examples: list[tuple[frozenset[str], bool]] = []
        self._keyed: dict[Any, tuple[frozenset[str], bool]] = {}
        self._vocab: set[str] = set()
        self._cache: (
            tuple[list[tuple[InducedRule, float]], InductionResult] | None
        ) = None

    # ── Evidence ───────────────────────────────────────────────

    @property
    def has_examples(self) -> bool:
        return bool(self._examples or self._keyed)

    @property
    def example_count(self) -> int:
        return len(self._examples) + len(self._keyed)

    def _evidence(self) -> Iterable[tuple[frozenset[str], bool]]:
        yield from self._examples
        yield from self._keyed.values()

    def observe(
        self,
        features: Iterable[str],
        label: bool,
        *,
        key: Any = None,
    ) -> None:
        """Add a labeled example and invalidate the cached posterior."""
        feats = frozenset(features)
        if key is not None:
            self._keyed[key] = (feats, bool(label))
        else:
            self._examples.append((feats, bool(label)))
            if len(self._examples) > _MAX_EXAMPLES:
                self._examples = self._examples[-_MAX_EXAMPLES:]
        if not feats <= self._vocab:
            self._vocab |= feats
        self._cache = None

    def reset(self) -> None:
        """Drop all evidence — a fresh induction episode."""
        self._examples.clear()
        self._keyed.clear()
        self._vocab.clear()
        self._cache = None

    # ── Hypothesis space ───────────────────────────────────────

    def _senses(self, max_literals: int) -> list[frozenset[Literal]]:
        """All contradiction-free conjunctions over the vocabulary."""
        vocab = sorted(self._vocab)
        out: list[frozenset[Literal]] = []
        for k in range(1, max_literals + 1):
            for feats in combinations(vocab, k):
                for signs in product((True, False), repeat=k):
                    out.append(
                        frozenset(
                            Literal(f, s)
                            for f, s in zip(feats, signs, strict=True)
                        )
                    )
        return out

    def _space(self) -> tuple[list[frozenset[Literal]], list[InducedRule]]:
        """Enumerate the bounded DNF space.

        Bounds are lowered until the space fits _MAX_HYPOTHESES —
        senses first, then literals, so the truncated tail is the
        deepest (lowest-prior) part of the space.
        """
        for s_lim in range(self.max_senses, 0, -1):
            for l_lim in range(self.max_literals, 0, -1):
                senses = self._senses(l_lim)
                if sum(
                    math.comb(len(senses), m) for m in range(1, s_lim + 1)
                ) <= _MAX_HYPOTHESES:
                    rules = [
                        InducedRule(combo)
                        for m in range(1, s_lim + 1)
                        for combo in combinations(senses, m)
                    ]
                    return senses, rules
        raise AssertionError("unreachable: minimal space always fits")

    # ── Prior ──────────────────────────────────────────────────

    def _log_prior(self, rule: InducedRule) -> float:
        """Dirichlet–multinomial marginal of the derivation counts.

        For each non-terminal X with k_X productions: with a
        symmetric Dirichlet(α=1) over production probabilities, the
        marginal of a derivation using counts (n_1..n_k) is

            Γ(k_X)/Γ(k_X + n_X) · ∏_j n_j!

        D uses counts (m−1 extends, 1 stop); C uses (N−m, m); P uses
        the per-literal counts across all senses.
        """
        m = rule.n_senses
        n = rule.n_literals
        logp = _dm_logp(2, (m - 1, 1)) + _dm_logp(2, (n - m, m))
        counts: dict[Literal, int] = {}
        for sense in rule.senses:
            for lit in sense:
                counts[lit] = counts.get(lit, 0) + 1
        logp += _dm_logp(2 * len(self._vocab), counts.values())
        return logp

    # ── Posterior ──────────────────────────────────────────────

    def _distribution(self) -> list[tuple[InducedRule, float]]:
        """P(h|E) over the bounded space, normalized."""
        if self._cache is not None:
            return self._cache[0]
        senses, rules = self._space()
        examples = list(self._evidence())
        label_mask = 0
        for i, (_, label) in enumerate(examples):
            if label:
                label_mask |= 1 << i
        sense_masks = {
            sense: sum(
                (1 << i)
                for i, (feats, _) in enumerate(examples)
                if all(lit.holds(feats) for lit in sense)
            )
            for sense in senses
        }
        log_weights = []
        for rule in rules:
            mask = 0
            for sense in rule.senses:
                mask |= sense_masks[sense]
            mismatches = (mask ^ label_mask).bit_count()
            log_weights.append(
                self._log_prior(rule) - self.exception_cost * mismatches
            )
        top = max(log_weights) if log_weights else 0.0
        weights = [math.exp(w - top) for w in log_weights]
        total = sum(weights) or 1.0
        dist = sorted(
            zip(rules, (w / total for w in weights), strict=True),
            key=lambda rw: -rw[1],
        )
        self._cache = (dist, self._result(dist))
        return dist

    def _result(self, dist: list[tuple[InducedRule, float]]) -> InductionResult:
        entropy = -sum(p * math.log(p) for _, p in dist if p > 0)
        top_rule, top_p = dist[0] if dist else (None, 0.0)
        return InductionResult(
            posterior=dist[:20],
            map_rule=top_rule,
            confidence=top_p,
            entropy=entropy,
            acquired=top_p >= self.acquire_threshold,
            enumerated=len(dist),
        )

    def induce(self) -> InductionResult:
        """The posterior summary: MAP rule, confidence, entropy."""
        self._distribution()
        assert self._cache is not None
        return self._cache[1]

    def posterior(self) -> list[tuple[InducedRule, float]]:
        """The full normalized distribution over the bounded space."""
        return self._distribution()

    def predict(self, features: Iterable[str]) -> float:
        """Posterior-predictive P(ℓ(x) = +) — a weighted vote."""
        dist = self._distribution()
        if not dist:
            return 0.5
        feats = frozenset(features)
        return min(
            1.0,
            sum(p for rule, p in dist if rule.holds(feats)),
        )

    def best_rule(self) -> InducedRule | None:
        """The MAP rule — the current hypothesis."""
        return self.induce().map_rule

    # ── Persistence ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": [
                [sorted(feats), label]
                for feats, label in self._evidence()
            ],
            "exception_cost": self.exception_cost,
            "max_senses": self.max_senses,
            "max_literals": self.max_literals,
            "acquire_threshold": self.acquire_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DorsolateralPFC:
        dlpfc = cls(
            exception_cost=float(data.get("exception_cost", 4.0)),
            max_senses=int(data.get("max_senses", 3)),
            max_literals=int(data.get("max_literals", 3)),
            acquire_threshold=float(data.get("acquire_threshold", 0.9)),
        )
        for feats, label in data.get("examples", []):
            dlpfc.observe(feats, bool(label))
        return dlpfc


def _dm_logp(k: int, counts: Collection[int]) -> float:
    """Log Dirichlet–multinomial marginal, symmetric α=1.

    Γ(k)/Γ(k+n) · ∏ n_j! in log space.
    """
    n = sum(counts)
    return (
        math.lgamma(k)
        - math.lgamma(k + n)
        + sum(math.lgamma(c + 1) for c in counts)
    )
