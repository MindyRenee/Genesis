"""Classification — "does this fit the rule?" / "what's the rule?"

The rule-membership family: things carry attribute sets; a rule
decides which belong. Two honest modes:

- ``apply`` — the rule is *stated* (a predicate over attributes).
  The agent reads it and labels each thing; the world's oracle
  checks. Stated-example assertions in the spec let the normalizer
  reject a predicate that contradicts what was actually claimed —
  so a miscompiled rule fails validation instead of silently
  "verifying" against itself.
- ``induce`` — no rule is stated. Labeled examples are shown;
  unlabeled items must be classified. The oracle knows the hidden
  labels (the answer key lives in the spec, never visible to the
  agent) — the agent has to learn which features predict membership
  from feedback alone.

Per-guess feedback is honest: wrong guesses reveal the true label,
cost a wasted move, and are how the induce-mode agent learns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_PREDICATE_OPS = (
    "all_of",
    "any_of",
    "none_of",
    "exactly",
    "at_least",
    "at_most",
)


def eval_predicate(predicate: dict, attrs: frozenset[str]) -> bool:
    """Evaluate a rule predicate against an attribute set."""
    op = predicate.get("op")
    of = {str(a) for a in predicate.get("of", [])}
    n = len(attrs & of)
    if op == "all_of":
        return n == len(of)
    if op == "any_of":
        return n >= 1
    if op == "none_of":
        return n == 0
    try:
        count = int(predicate.get("count", 0))
    except (TypeError, ValueError):
        return False
    if op == "exactly":
        return n == count
    if op == "at_least":
        return n >= count
    if op == "at_most":
        return n <= count
    return False


def valid_predicate(predicate: Any) -> bool:
    """Defensive spec check — closed op vocabulary, sane fields."""
    if not isinstance(predicate, dict):
        return False
    if predicate.get("op") not in _PREDICATE_OPS:
        return False
    of = predicate.get("of")
    if (
        not isinstance(of, list)
        or not of
        or len(of) > 12
        or not all(isinstance(a, str) and a.strip() for a in of)
    ):
        return False
    if predicate["op"] in ("exactly", "at_least", "at_most"):
        raw_count: Any = predicate.get("count")
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            return False
        if not 0 <= count <= len(of):
            return False
    return True


@dataclass(frozen=True)
class Item:
    """A thing to classify: a name and the attributes it has."""

    item_id: int
    name: str
    attrs: tuple[str, ...]
    label: bool | None = None  # hidden truth for induce-mode items


@dataclass
class GuessTarget:
    """One unlabeled item with its perceptible features."""

    name: str
    attrs: tuple[str, ...]
    n_attrs: int


class RuleGame:
    """The worksheet: label every item correctly."""

    def __init__(
        self,
        predicate: dict | None,
        items: list[Item],
        examples: list[Item] | None = None,
    ) -> None:
        self.predicate = predicate  # stated rule (apply) or None
        self.items = list(items)
        self.examples = list(examples or [])
        # The oracle's answer key — never exposed to the agent.
        self._truth: dict[str, bool] = {}
        for it in self.items:
            if it.label is not None:
                self._truth[it.name] = it.label
            elif predicate is not None:
                self._truth[it.name] = eval_predicate(
                    predicate, frozenset(it.attrs)
                )
        self._pending: dict[str, Item] = {
            it.name: it for it in self.items if it.name in self._truth
        }
        self._labeled: dict[str, bool] = {}

    # ── Queries ───────────────────────────────────────────────

    def mismatches(self) -> int:
        """Items not yet correctly labeled — zero means solved."""
        return len(self._pending)

    def complete(self) -> bool:
        return not self._pending

    def candidates(self) -> list[GuessTarget]:
        """Unlabeled items with perceptible features only."""
        return [
            GuessTarget(it.name, it.attrs, len(it.attrs))
            for it in self._pending.values()
        ]

    def known_examples(self) -> list[tuple[Item, bool]]:
        """The labeled examples — visible supervision, like a
        worksheet's worked row."""
        return [(e, bool(e.label)) for e in self.examples]

    def labels(self) -> dict[str, bool]:
        """Correctly-labeled items so far."""
        return dict(self._labeled)

    def state(self) -> dict[str, Any]:
        return {
            "items.unlabeled": self.mismatches(),
            "items.labeled": len(self._labeled),
        }

    def goal_state(self) -> dict[str, Any]:
        return {"items.unlabeled": self.mismatches()}

    # ── Actions ───────────────────────────────────────────────

    def guess(self, name: str, label: bool) -> dict[str, Any] | None:
        """Classify an item. Honest feedback: correct locks it in,
        wrong costs a move and reveals the true label — supervision,
        not silence."""
        item = self._pending.get(name)
        if item is None:
            return None
        truth = self._truth[name]
        correct = bool(label) == truth
        if correct:
            del self._pending[name]
            self._labeled[name] = truth
        return {"correct": correct, "truth": truth, "item": name}
