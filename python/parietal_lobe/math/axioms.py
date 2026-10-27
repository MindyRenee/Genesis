"""Axioms — the first pillar: the words for the ladder of laws.

Closure, associativity, identity, inverses, commutativity — each
axiom is a checkable claim, and their stacking is the ladder from
bare magma to abelian group. The faculty can take a small
operation table — a Cayley square — and answer "what is this
structure": where it climbs, which axiom first fails, and the
witness pair that fails it. Naming what structure a thing *is*
is the faculty's whole job.

The machine half is ``src/math/axioms.rs``: table checkers —
closed, associative, identity, inverses, commutative — and the
ladder magma→semigroup→monoid→group→abelian. Here: the axiom
words and the verdict with its witness.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AxiomResult:
    """The outcome of asking which axioms a structure satisfies."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# A binary operation on {0..n-1} is its table: op[a][b] = a·b.
Table = list[list[int]]


def check_closed(op: Table) -> tuple[bool, tuple[int, int] | None]:
    """Every product lands back in the carrier."""
    n = len(op)
    if not n:
        return False, None
    for a in range(n):
        for b in range(n):
            if b >= len(op[a]) or not isinstance(op[a][b], int) or not 0 <= op[a][b] < n:
                return False, (a, b)
        if len(op[a]) != n:
            return False, (a, n)
    return True, None


def check_associative(op: Table) -> tuple[bool, tuple[int, int, int] | None]:
    """(a·b)·c = a·(b·c) — checked over all triples."""
    if not check_closed(op)[0]:
        return False, None
    n = len(op)
    for a, b, c in itertools.product(range(n), repeat=3):
        if op[op[a][b]][c] != op[a][op[b][c]]:
            return False, (a, b, c)
    return True, None


def check_identity(op: Table) -> tuple[bool, int | None]:
    """Some e with e·a = a·e = a for all a — return it when found."""
    n = len(op)
    for e in range(n):
        if all(op[e][a] == a and op[a][e] == a for a in range(n)):
            return True, e
    return False, None


def check_inverses(op: Table, e: int) -> tuple[bool, int | None]:
    """Every a has an a' with a·a' = a'·a = e — the failing element."""
    n = len(op)
    for a in range(n):
        if not any(op[a][b] == e and op[b][a] == e for b in range(n)):
            return False, a
    return True, None


def check_commutative(op: Table) -> tuple[bool, tuple[int, int] | None]:
    """a·b = b·a over all pairs."""
    n = len(op)
    for a in range(n):
        for b in range(n):
            if op[a][b] != op[b][a]:
                return False, (a, b)
    return True, None


def classify(op: Table) -> dict[str, Any]:
    """Climb the ladder honestly — report each rung and the first
    failed axiom with its witness."""
    out: dict[str, Any] = {"n": len(op)}
    ok, w = check_closed(op)
    out["closed"] = ok
    if not ok:
        out["structure"] = "not even a magma"
        out["witness"] = w
        return out
    ok, assoc_witness = check_associative(op)
    out["associative"] = ok
    if not ok:
        out["structure"] = "magma"
        out["witness"] = assoc_witness
        return out
    ok, e = check_identity(op)
    out["identity"] = e
    if not ok:
        out["structure"] = "semigroup"
        return out
    assert e is not None
    ok, a = check_inverses(op, e)
    if not ok:
        out["structure"] = "monoid"
        out["witness"] = a
        return out
    out["inverses"] = True
    ok, w = check_commutative(op)
    out["commutative"] = ok
    if not ok:
        out["structure"] = "group"
        out["witness"] = w
        return out
    out["structure"] = "abelian group"
    return out


def cyclic_table(n: int) -> Table:
    """Z_n's addition table — the canonical small group."""
    return [[(a + b) % n for b in range(n)] for a in range(n)]


def klein_table() -> Table:
    """Klein's four-group — the other order-4 group."""
    return [[a ^ b for b in range(4)] for a in range(4)]


_AX_WORD = re.compile(
    r"\b(?:axiom|axioms|closed|closure|associative|identity|"
    r"inverse|commutative|abelian|magma|semigroup|monoid|"
    r"group|ring|field|cayley table|operation table)\b",
    re.IGNORECASE,
)


def looks_like_axioms(text: str) -> bool:
    """Whether the text asks about algebraic structure — axioms,
    tables, the ladder's rungs."""
    return bool(_AX_WORD.search(text))


def classify_text(table: list[list[int]]) -> AxiomResult:
    """"what is this structure" over a Cayley table — the rung it
    reaches and the axiom it fails, witnessed."""
    if not table or any(len(r) != len(table) for r in table):
        return AxiomResult("classify", str(table), "", False,
                           "the table must be square")
    if not check_closed(table)[0]:
        return AxiomResult("classify", str(table), "", False,
                           "table entries must be elements of the carrier")
    info = classify(table)
    out = info["structure"]
    if "witness" in info:
        out += f" — witnessed by {info['witness']}"
    return AxiomResult("classify", str(table), out, True, data=info)
