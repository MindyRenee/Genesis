"""Discrete mathematics — the words for relations, sets, graphs.

Relations arrive as pairs — "is {(1,1),(2,2)} reflexive over
{1,2,3}" — and leave as property verdicts: reflexive, symmetric,
transitive, antisymmetric, or which ones fail and why. Sets are
finite and whole; graphs are edge lists with connectivity and
paths. Everything here is decidable — the finite is the machine's
home ground.

The machine half is ``src/math/discrete.rs``: Relation,
properties, equivalence check, reachability, graph walks. Here:
the relation's words and the honest verdict.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiscreteResult:
    """The outcome of a discrete-structure question."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


Relation = frozenset[tuple[int, int]]


def parse_pairs(text: str) -> Relation:
    """Read "{(1,2),(2,3)}" into a relation."""
    try:
        lit = ast.literal_eval(text.strip())
        if not isinstance(lit, (set, frozenset, list, tuple)):
            raise ValueError("relation must be a collection of pairs")
        pairs: set[tuple[int, int]] = set()
        for x in lit:
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                raise ValueError("each relation entry must be a pair")
            pairs.add((int(x[0]), int(x[1])))
    except (SyntaxError, ValueError, TypeError) as e:
        raise ValueError(f"not a set of pairs: {text!r}") from e
    return frozenset(pairs)


def is_reflexive(rel: Relation, universe: set[int]) -> bool:
    return all((x, x) in rel for x in universe)


def is_symmetric(rel: Relation) -> bool:
    return all((b, a) in rel for a, b in rel)


def is_antisymmetric(rel: Relation) -> bool:
    return all(a == b or (b, a) not in rel for a, b in rel)


def is_transitive(rel: Relation) -> bool:
    return all((a, c) in rel
               for a, b in rel for b2, c in rel if b == b2)


def is_equivalence(rel: Relation, universe: set[int]) -> bool:
    return (is_reflexive(rel, universe) and is_symmetric(rel)
            and is_transitive(rel))


def reachable(edges: list[tuple[int, int]], src: int) -> set[int]:
    """The set of nodes a walk from src can reach — BFS."""
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = {src}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def connected(edges: list[tuple[int, int]], nodes: set[int]) -> bool:
    """One component — every node reachable from the first."""
    if not nodes:
        return True
    return reachable(edges, next(iter(nodes))) >= nodes


_REL_WORD = re.compile(
    r"\b(?:relation|reflexive|symmetric|antisymmetric|transitive|"
    r"equivalence|partial order|poset|function|injective|"
    r"surjective|bijective|graph|connected|path|cycle|tree|"
    r"bipartite|degree)\b",
    re.IGNORECASE,
)


def looks_like_discrete(text: str) -> bool:
    """Whether the text asks about relations, sets, or graphs."""
    return bool(_REL_WORD.search(text))


def relation_text(text: str, universe_text: str | None = None) -> DiscreteResult:
    """"is {(1,1),(2,2),(1,2)} reflexive on {1,2}" — each property
    named, the failures given witnesses."""
    try:
        rel = parse_pairs(text)
        universe = ({int(x) for x in ast.literal_eval(universe_text)}
                    if universe_text else {v for p in rel for v in p})
        if any(a not in universe or b not in universe for a, b in rel):
            raise ValueError("relation contains elements outside the universe")
    except (ValueError, SyntaxError, TypeError) as e:
        return DiscreteResult("relation", text, "", False, str(e))
    props = {
        "reflexive": is_reflexive(rel, universe),
        "symmetric": is_symmetric(rel),
        "antisymmetric": is_antisymmetric(rel),
        "transitive": is_transitive(rel),
    }
    holds = [k for k, v in props.items() if v]
    fails = [k for k, v in props.items() if not v]
    out = (", ".join(holds) if holds else "none")
    if fails:
        out += f" — not {', not '.join(fails)}"
    if props["reflexive"] and props["symmetric"] and props["transitive"]:
        out += " — an equivalence relation"
    elif props["reflexive"] and props["antisymmetric"] and props["transitive"]:
        out += " — a partial order"
    return DiscreteResult("relation", text, out, True,
                          data={**props, "equivalence": is_equivalence(rel, universe)})


def connected_text(text: str) -> DiscreteResult:
    """"is the graph {(1,2),(2,3)} connected" -> yes."""
    try:
        edges = [(int(a), int(b)) for a, b in parse_pairs(text)]
    except ValueError as e:
        return DiscreteResult("connected", text, "", False, str(e))
    nodes = {v for e in edges for v in e}
    ok = connected(edges, nodes)
    return DiscreteResult("connected", text,
                          f"{'connected' if ok else 'disconnected'} — "
                          f"{len(nodes)} nodes, {len(edges)} edges",
                          True, data={"connected": ok})
