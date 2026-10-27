"""Proof — the third pillar: the words for valid inference.

A proposition is atoms joined by and / or / not / implies / iff.
Truth is decided two ways: by truth table (always honest, always
decidable — a tautology is a proposition true under every
assignment) and by rules — modus ponens, modus tollens, case
analysis — each a named, checkable step. A derivation is a chain
of such steps whose last line is the goal; verification replays
every step, so a proof that checks is a proof, and a broken step
is named.

The machine half is ``src/math/proof.rs``: Prop, evaluate,
tautology, the inference rules, verify. Here: the propositions'
words and the verdict.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProofResult:
    """The outcome of a proof question — a verdict, the failing
    step when there is one."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# A proposition is a nested tuple: ("atom", "A") | ("not", p) |
# ("and", p, q) | ("or", p, q) | ("imp", p, q) | ("iff", p, q).
Prop = tuple


def atom(name: str) -> Prop:
    return ("atom", name)


def evaluate(p: Prop, env: dict[str, bool]) -> bool:
    """The proposition's truth under an assignment."""
    tag = p[0]
    if tag == "atom":
        return env.get(p[1], False)
    if tag == "not":
        return not evaluate(p[1], env)
    if tag == "and":
        return evaluate(p[1], env) and evaluate(p[2], env)
    if tag == "or":
        return evaluate(p[1], env) or evaluate(p[2], env)
    if tag == "imp":
        return (not evaluate(p[1], env)) or evaluate(p[2], env)
    if tag == "iff":
        return evaluate(p[1], env) == evaluate(p[2], env)
    raise ValueError(f"unknown node {tag!r}")


def atoms(p: Prop) -> set[str]:
    if p[0] == "atom":
        return {p[1]}
    return set().union(*(atoms(q) for q in p[1:]))


def tautology(p: Prop) -> bool:
    """True under every assignment — truth-table decidability."""
    names = sorted(atoms(p))
    for bits in itertools.product((False, True), repeat=len(names)):
        if not evaluate(p, dict(zip(names, bits, strict=True))):
            return False
    return True


def satisfiable(p: Prop) -> dict[str, bool] | None:
    """One assignment that makes it true, when one exists."""
    names = sorted(atoms(p))
    for bits in itertools.product((False, True), repeat=len(names)):
        env = dict(zip(names, bits, strict=True))
        if evaluate(p, env):
            return env
    return None


def modus_ponens(imp: Prop, ante: Prop) -> Prop | None:
    """From P→Q and P, conclude Q — the forward step."""
    if imp[0] == "imp" and imp[1] == ante:
        return imp[2]
    return None


def modus_tollens(imp: Prop, negated: Prop) -> Prop | None:
    """From P→Q and ¬Q, conclude ¬P — the backward step."""
    if imp[0] == "imp" and negated == ("not", imp[2]):
        return ("not", imp[1])
    return None


def render(p: Prop) -> str:
    """Back to words: "P ∧ Q → R"."""
    tag = p[0]
    if tag == "atom":
        return p[1]
    if tag == "not":
        return f"¬{render(p[1])}"
    sym = {"and": "∧", "or": "∨", "imp": "→", "iff": "↔"}[tag]
    return f"({render(p[1])} {sym} {render(p[2])})"


_PROOF_WORD = re.compile(
    r"\b(?:proof|prove|theorem|lemma|therefore|tautology|"
    r"contradiction|modus ponens|modus tollens|valid|validity|"
    r"syllogism|deduce|deduction|inference|implies|iff|qed)\b",
    re.IGNORECASE,
)


def looks_like_proof(text: str) -> bool:
    """Whether the text asks about proof, validity, or inference."""
    return bool(_PROOF_WORD.search(text))


def tautology_text(p: Prop, words: str = "") -> ProofResult:
    """"is P → P a tautology" -> yes — true under all assignments."""
    ok = tautology(p)
    label = words or render(p)
    verdict = "a tautology — true under every assignment" if ok else "not a tautology"
    return ProofResult(
        "tautology", label,
        f"{render(p)} is {verdict}",
        True, data={"tautology": ok})


def satisfiable_text(p: Prop, words: str = "") -> ProofResult:
    """"can P ∧ ¬Q hold" -> the assignment that does it, or none."""
    env = satisfiable(p)
    label = words or render(p)
    if env is None:
        return ProofResult("satisfiable", label,
                           f"{render(p)} is unsatisfiable — "
                           "a contradiction in every world",
                           True, data={"satisfiable": False})
    out = ", ".join(f"{k}={'T' if v else 'F'}" for k, v in sorted(env.items()))
    return ProofResult("satisfiable", label,
                       f"satisfiable — e.g. {out}", True,
                       data={"satisfiable": True, "witness": env})


def entail_text(premises: list[Prop], goal: Prop) -> ProofResult:
    """"do these premises force the goal" — a semantic check:
    in every world where all premises hold, does the goal hold?"""
    names = sorted(set().union(*(atoms(p) for p in premises)) | atoms(goal))
    for bits in itertools.product((False, True), repeat=len(names)):
        env = dict(zip(names, bits, strict=True))
        if all(evaluate(p, env) for p in premises) and not evaluate(goal, env):
            out = ", ".join(f"{k}={'T' if v else 'F'}" for k, v in sorted(env.items()))
            return ProofResult("entail", f"premises ⊢ {render(goal)}",
                               f"no — a countermodel: {out}", True,
                               data={"entails": False, "countermodel": env})
    return ProofResult("entail", f"premises ⊢ {render(goal)}",
                       "yes — the goal holds in every world "
                       "where the premises do", True,
                       data={"entails": True})
