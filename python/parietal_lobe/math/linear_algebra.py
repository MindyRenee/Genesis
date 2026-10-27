"""Linear algebra — the words for spaces and maps.

The faculty hears "the determinant of [[1,2],[3,4]]", "solve the
system", "the rank of that matrix". Matrices arrive as bracketed
rows and leave as answers in the same notation. The computation
is exact where the input is exact — Fraction arithmetic through
elimination, so [[1,2],[3,4]] has determinant -2, not -2.0000001.

The machine half is ``src/math/linear_algebra.rs``: the dense
Matrix, products, echelon, det, solve, eigen. Here: the bracketed
words and the structured reply.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class LinearResult:
    """The outcome of a linear-algebra question."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


Matrix = list[list[Fraction]]


def parse_matrix(text: str) -> Matrix:
    """Parse "[[1, 2], [3, 4]]" (or ((1,2),(3,4))) into Fractions."""
    try:
        lit = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"not a matrix literal: {text!r}") from e
    if not isinstance(lit, (list, tuple)) or not lit:
        raise ValueError("empty matrix")
    if any(not isinstance(r, (list, tuple)) for r in lit):
        raise ValueError("every row must be a list")
    rows = [list(r) for r in lit]
    width = len(rows[0])
    if width == 0 or any(len(r) != width for r in rows):
        raise ValueError("ragged rows")
    return [[Fraction(str(x)) for x in r] for r in rows]


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def render_matrix(m: Matrix) -> str:
    return "[" + ", ".join("[" + ", ".join(_fmt(x) for x in r) + "]" for r in m) + "]"


def _echelon(m: Matrix) -> tuple[Matrix, list[int], int]:
    """Forward elimination — returns (echelon, pivot_cols, sign)."""
    m = [row[:] for row in m]
    rows, cols = len(m), len(m[0])
    pivots: list[int] = []
    sign = 1
    r = 0
    for c in range(cols):
        pivot = max(range(r, rows), key=lambda i: abs(m[i][c]), default=None)
        if pivot is None or r >= rows or m[pivot][c] == 0:
            continue
        if pivot != r:
            m[pivot], m[r] = m[r], m[pivot]
            sign = -sign
        pv = m[r][c]
        for i in range(r + 1, rows):
            f = m[i][c] / pv
            if f:
                m[i] = [m[i][j] - f * m[r][j] for j in range(cols)]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return m, pivots, sign


def matmul(a: Matrix, b: Matrix) -> Matrix | None:
    if len(a[0]) != len(b):
        return None
    return [
        [sum((a[i][k] * b[k][j] for k in range(len(b))), Fraction(0))
         for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def determinant(m: Matrix) -> Fraction | None:
    n = len(m)
    if any(len(r) != n for r in m):
        return None
    ech, pivots, sign = _echelon(m)
    if len(pivots) < n:
        return Fraction(0)
    d = Fraction(sign)
    for i in range(n):
        d *= ech[i][i]
    return d


def rank(m: Matrix) -> int:
    return len(_echelon(m)[1])


def solve(a: Matrix, b: list[Fraction]) -> list[Fraction] | None:
    """Solve Ax = b by Gauss–Jordan — exact fractions."""
    n = len(a)
    if any(len(r) != n for r in a) or len(b) != n:
        return None
    aug = [[*a[i][:], b[i]] for i in range(n)]
    for c in range(n):
        pivot = max(range(c, n), key=lambda i: abs(aug[i][c]), default=None)
        if pivot is None or aug[pivot][c] == 0:
            return None
        aug[c], aug[pivot] = aug[pivot], aug[c]
        pv = aug[c][c]
        aug[c] = [x / pv for x in aug[c]]
        for r in range(n):
            if r != c and aug[r][c]:
                f = aug[r][c]
                aug[r] = [aug[r][j] - f * aug[c][j] for j in range(n + 1)]
    return [aug[i][n] for i in range(n)]


_MATRIX_OP = re.compile(
    r"\b(?:determinant|det|rank|transpose|inverse|matrix|matrices|"
    r"eigenvalue|eigenvector|solve the system|linear system)\b",
    re.IGNORECASE,
)


def looks_like_linear_algebra(text: str) -> bool:
    """Whether the text asks a matrix question.

    True for "determinant of [[1,2],[3,4]]", "rank of that
    matrix", "solve the system" — the vocabulary of maps.
    """
    return bool(_MATRIX_OP.search(text) or "[[" in text)


def determinant_text(text: str) -> LinearResult:
    """"determinant of [[1,2],[3,4]]" -> -2."""
    try:
        m = parse_matrix(text)
        d = determinant(m)
    except ValueError as e:
        return LinearResult("det", text, "", False, str(e))
    if d is None:
        return LinearResult("det", text, "", False, "determinant needs a square matrix")
    return LinearResult("det", text, _fmt(d), True, data={"det": str(d)})


def rank_text(text: str) -> LinearResult:
    """"rank of [[1,2],[2,4]]" -> 1."""
    try:
        m = parse_matrix(text)
    except ValueError as e:
        return LinearResult("rank", text, "", False, str(e))
    return LinearResult("rank", text, str(rank(m)), True)


def matmul_text(a_text: str, b_text: str) -> LinearResult:
    """"[[1,0],[0,1]] times [[1,2],[3,4]]" — the product."""
    try:
        a, b = parse_matrix(a_text), parse_matrix(b_text)
    except ValueError as e:
        return LinearResult("matmul", f"{a_text} * {b_text}", "", False, str(e))
    out = matmul(a, b)
    if out is None:
        return LinearResult("matmul", f"{a_text} * {b_text}", "", False,
                            "inner dimensions must agree")
    return LinearResult("matmul", f"{a_text} * {b_text}", render_matrix(out), True)


def solve_text(a_text: str, b_text: str) -> LinearResult:
    """"solve [[2,1],[1,-1]] x = [5, 1]" -> x = (2, 1)."""
    try:
        a = parse_matrix(a_text)
        lit = ast.literal_eval(b_text.strip())
        b = [Fraction(str(x)) for x in lit]
    except (ValueError, SyntaxError) as e:
        return LinearResult("solve", f"{a_text} x = {b_text}", "", False, str(e))
    x = solve(a, b)
    if x is None:
        return LinearResult("solve", f"{a_text} x = {b_text}", "", False,
                            "the system has no unique solution")
    out = "(" + ", ".join(_fmt(v) for v in x) + ")"
    return LinearResult("solve", f"{a_text} x = {b_text}", f"x = {out}", True)
