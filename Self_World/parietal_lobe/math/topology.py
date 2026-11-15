"""Topology — the words for shape without distance.

The faculty speaks of what bends without tearing: open and closed
sets, connectedness, the Euler characteristic that counts vertices
minus edges plus faces and knows a sphere from a torus. Finite
topological spaces are small enough to hold whole — a space is its
points and the opens it admits.

The machine half is ``src/math/topology.rs``: finite spaces as
bitmasks, open/closed tests, closure/interior, Euler
characteristic, simple invariants. Here: the shape's words.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TopoResult:
    """The outcome of a topological question."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# A finite topological space: n points, opens as frozensets of
# point-indices. The axioms are checkable — this is a machine.

FiniteSpace = frozenset[frozenset[int]]


def is_topology(n: int, opens: FiniteSpace) -> bool:
    """The three axioms: ∅ and X open; unions open; finite
    intersections open."""
    if n < 0:
        return False
    universe = frozenset(range(n))
    if frozenset() not in opens or universe not in opens or any(not o <= universe for o in opens):
        return False
    for a, b in itertools.combinations_with_replacement(list(opens), 2):
        if (a | b) not in opens or (a & b) not in opens:
            return False
    return True


def closure(n: int, opens: FiniteSpace, s: frozenset[int]) -> frozenset[int]:
    """The smallest closed set containing s — closed sets are the
    complements of opens."""
    universe = frozenset(range(n))
    closed = [universe - o for o in opens]
    containing = [c for c in closed if s <= c]
    return frozenset.intersection(*containing) if containing else universe


def interior(n: int, opens: FiniteSpace, s: frozenset[int]) -> frozenset[int]:
    """The largest open set inside s."""
    contained = [o for o in opens if o <= s]
    return frozenset.union(*contained) if contained else frozenset()


def euler_characteristic(faces: list[int]) -> int:
    """χ = Σ (−1)^i f_i — the alternating count that knows the
    surface: V−E+F for a polyhedron, so a cube reads 2 (a sphere)."""
    return sum((1 if i % 2 == 0 else -1) * f for i, f in enumerate(faces))


def euler_surface(v: int, e: int, f: int) -> TopoResult:
    """ "V-E+F for 8, 12, 6" -> 2 — a sphere's signature."""
    chi = v - e + f
    name = {
        2: "sphere-like (χ=2)",
        0: "torus-like (χ=0)",
        -2: "two-holed (χ=−2)",
        1: "disk-like (χ=1)",
    }.get(chi, f"χ={chi}")
    return TopoResult(
        "euler",
        f"V={v}, E={e}, F={f}",
        f"χ = {v} − {e} + {f} = {chi} — {name}",
        True,
        data={"chi": chi},
    )


_TOPO_WORD = re.compile(
    r"\b(?:topology|topological|open set|closed set|closure|"
    r"interior|boundary|connected|compact|hausdorff|manifold|"
    r"homeomorph|euler characteristic|genus|torus|sphere|"
    r"homotopy|fundamental group|knot)\b",
    re.IGNORECASE,
)


def looks_like_topology(text: str) -> bool:
    """Whether the text asks about opens, connectedness, surfaces —
    shape without distance."""
    return bool(_TOPO_WORD.search(text))


def describe_space_text(n: int, opens: list[list[int]]) -> TopoResult:
    """Is this collection of subsets a topology on n points?"""
    fs = frozenset(frozenset(o) for o in opens)
    ok = is_topology(n, fs)
    out = f"{len(fs)} opens on {n} points — " + (
        "a topology: ∅, X, unions, and intersections all hold"
        if ok
        else "not a topology — an axiom fails"
    )
    return TopoResult("topology", str(opens), out, True, data={"is_topology": ok})
