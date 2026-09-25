"""Classification — "does this fit the rule?" / "what's the rule?"

The eighth task family wired into ``TaskCompetence``: rule-membership
tasks — the invented-word games ("a tessel is a machine with exactly
two of..."), odd-one-out, "which of these belong." ``RuleGame`` is
honest: per-guess oracle feedback, wrong guesses reveal the true
label, and stated-example assertions let the normalizer reject a
predicate that contradicts what was claimed. ``RuleAgent`` evaluates
stated predicates directly and induces unstated rules from
feature→label evidence — attribute presence and count are the
perceptible features, so "exactly two of" is learnable as a count
rule.
"""

from .agent import ClassificationResult, RuleAgent
from .world import (
    GuessTarget,
    Item,
    RuleGame,
    eval_predicate,
    valid_predicate,
)

__all__ = [
    "ClassificationResult",
    "GuessTarget",
    "Item",
    "RuleAgent",
    "RuleGame",
    "eval_predicate",
    "valid_predicate",
]
