"""Combinatorics — the words for counting without listing.

"how many ways to choose 3 from 10" is C(10,3) = 120; "how many
orders of 5" is 5! = 120; "how many derangements of 5" counts the
permutations that leave nothing home. The faculty answers the
count and names which counting principle produced it — the
multiplication rule, the binomial coefficient, inclusion-exclusion
inside the derangement.

The machine half is ``src/math/combinatorics.rs``: factorial,
choose, multinomial, derangements, Stirling, stars-and-bars.
Here: the counting question and the named principle.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CountResult:
    """The outcome of a counting question — the number and the
    principle that produced it."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def choose(n: int, k: int) -> int | None:
    """C(n,k) — subsets of size k. None when out of range."""
    if k < 0 or k > n or n < 0:
        return None
    return math.comb(n, k)


def permute(n: int, k: int) -> int | None:
    """P(n,k) — ordered k-arrangements of n things."""
    if k < 0 or k > n or n < 0:
        return None
    return math.perm(n, k)


def multinomial(counts: list[int]) -> int:
    """n! / ∏ k_i! — arrangements of a multiset."""
    n = sum(counts)
    out = math.factorial(n)
    for k in counts:
        out //= math.factorial(k)
    return out


def derangements(n: int) -> int:
    """!n — permutations with no fixed point, by recurrence."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return 1
    a, b = 1, 0
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    return b


def stirling2(n: int, k: int) -> int:
    """S(n,k) — partitions of n labeled things into k nonempty
    unlabeled blocks, by the table recurrence."""
    if k == 0:
        return 1 if n == 0 else 0
    prev = [0] * (k + 1)
    prev[0] = 1
    for i in range(1, n + 1):
        cur = [0] * (k + 1)
        for j in range(1, min(i, k) + 1):
            cur[j] = j * prev[j] + prev[j - 1]
        prev = cur
    return prev[k]


def stars_bars(n: int, k: int) -> int:
    """Ways to put n identical things into k bins — C(n+k-1, k-1)."""
    if k < 1 or n < 0:
        return 0
    return math.comb(n + k - 1, k - 1)


_COUNT_WORD = re.compile(
    r"\b(?:how many ways|choose|chooses|choosing|pick|arrange|"
    r"arrangements|permutations?|combinations?|committee|"
    r"derangements?|factorial|binomial|counting|number of ways|"
    r"how many different|partitions?)\b",
    re.IGNORECASE,
)


def looks_like_counting(text: str) -> bool:
    """Whether the text asks a counting question — ways, orders,
    committees, arrangements."""
    return bool(_COUNT_WORD.search(text))


def choose_text(n: int, k: int) -> CountResult:
    """"choose 3 from 10" -> 120 — the binomial coefficient."""
    c = choose(n, k)
    if c is None:
        return CountResult("choose", f"C({n},{k})", "", False,
                           "k must satisfy 0 ≤ k ≤ n")
    return CountResult("choose", f"C({n},{k})",
                       f"{c} — the binomial coefficient: "
                       f"{n}!/({k}!·{n-k}!), subsets not orders",
                       True, data={"count": c})


def arrange_text(n: int, k: int | None = None) -> CountResult:
    """"arrange 5" -> 5! = 120; "arrange 3 of 10" -> P(10,3)."""
    if k is None or k == n:
        if n < 0:
            return CountResult("arrange", str(n), "", False, "n must be ≥ 0")
        c = math.factorial(n)
        return CountResult("arrange", f"{n}!",
                           f"{c} — {n}! orders for {n} distinct things",
                           True, data={"count": c})
    count = permute(n, k)
    if count is None:
        return CountResult("arrange", f"P({n},{k})", "", False,
                           "k must satisfy 0 ≤ k ≤ n")
    return CountResult("arrange", f"P({n},{k})",
                       f"{count} — ordered arrangements: "
                       f"{n}!/{n-k}!, orders not subsets",
                       True, data={"count": count})


def derange_text(n: int) -> CountResult:
    """"derangements of 5" -> 44 — nothing stays home."""
    if n < 0:
        return CountResult("derange", str(n), "", False, "n must be ≥ 0")
    c = derangements(n)
    return CountResult("derange", f"!{n}",
                       f"{c} — permutations of {n} with no fixed point",
                       True, data={"count": c})
