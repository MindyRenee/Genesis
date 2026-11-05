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
        [sum((a[i][k] * b[k][j] for k in range(len(b))), Fraction(0)) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def matrix_add(a: Matrix, b: Matrix, sign: int = 1) -> Matrix | None:
    """Add or subtract matrices with identical dimensions."""
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b, strict=True)):
        return None
    return [
        [x + sign * y for x, y in zip(row_a, row_b, strict=True)]
        for row_a, row_b in zip(a, b, strict=True)
    ]


def matrix_scale(m: Matrix, scalar: Fraction) -> Matrix:
    """Multiply every matrix entry by an exact scalar."""
    return [[scalar * value for value in row] for row in m]


def vector_cross(a: list[Fraction], b: list[Fraction]) -> list[Fraction] | None:
    """Return the three-dimensional cross product."""
    if len(a) != 3 or len(b) != 3:
        return None
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
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


def transpose(m: Matrix) -> Matrix:
    """Transpose a rectangular matrix."""
    return [list(column) for column in zip(*m, strict=True)]


def trace(m: Matrix) -> Fraction | None:
    """Return the exact trace of a square matrix."""
    if any(len(row) != len(m) for row in m):
        return None
    return sum((m[i][i] for i in range(len(m))), Fraction(0))


def inverse(m: Matrix) -> Matrix | None:
    """Invert a nonsingular square matrix by exact Gauss–Jordan elimination."""
    n = len(m)
    if any(len(row) != n for row in m):
        return None
    augmented = [
        [*row[:], *(Fraction(int(i == j)) for j in range(n))]
        for i, row in enumerate(m)
    ]
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][j] - factor * augmented[column][j]
                    for j in range(2 * n)
                ]
    return [row[n:] for row in augmented]


def vector_dot(a: list[Fraction], b: list[Fraction]) -> Fraction | None:
    """Return the exact dot product when vector dimensions agree."""
    if len(a) != len(b):
        return None
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def vector_add(
    a: list[Fraction], b: list[Fraction], sign: int = 1
) -> list[Fraction] | None:
    """Add vectors, or subtract when sign is -1."""
    if len(a) != len(b):
        return None
    return [x + sign * y for x, y in zip(a, b, strict=True)]


def _parse_vector(text: str) -> list[Fraction]:
    """Parse a one-dimensional vector literal into exact fractions."""
    literal = ast.literal_eval(text.strip())
    if not isinstance(literal, (list, tuple)) or not literal:
        raise ValueError("a non-empty vector literal is required")
    if any(isinstance(value, (list, tuple, dict)) for value in literal):
        raise ValueError("a vector must be one-dimensional")
    return [Fraction(str(value)) for value in literal]


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
    r"solve the system|linear system|trace|"
    r"dot product|inner product|cross product|vector)\b",
    re.IGNORECASE,
)


def looks_like_linear_algebra(text: str) -> bool:
    """Whether the text asks a matrix question.

    True for "determinant of [[1,2],[3,4]]", "rank of that
    matrix", "solve the system" — the vocabulary of maps.
    """
    return bool(_MATRIX_OP.search(text) or "[[" in text)


def transpose_text(text: str) -> LinearResult:
    """Transpose a matrix literal."""
    try:
        matrix = parse_matrix(text)
    except ValueError as error:
        return LinearResult("transpose", text, "", False, str(error))
    return LinearResult("transpose", text, render_matrix(transpose(matrix)), True)


def inverse_text(text: str) -> LinearResult:
    """Invert a nonsingular square matrix using exact fractions."""
    try:
        matrix = parse_matrix(text)
    except ValueError as error:
        return LinearResult("inverse", text, "", False, str(error))
    result = inverse(matrix)
    if result is None:
        return LinearResult(
            "inverse", text, "", False, "inverse requires a nonsingular square matrix"
        )
    return LinearResult("inverse", text, render_matrix(result), True)


def trace_text(text: str) -> LinearResult:
    """Compute the trace of a square matrix."""
    try:
        matrix = parse_matrix(text)
    except ValueError as error:
        return LinearResult("trace", text, "", False, str(error))
    result = trace(matrix)
    if result is None:
        return LinearResult("trace", text, "", False, "trace requires a square matrix")
    return LinearResult("trace", text, _fmt(result), True)


def vector_dot_text(a_text: str, b_text: str) -> LinearResult:
    """Compute the dot product of two explicit vector literals."""
    try:
        a, b = _parse_vector(a_text), _parse_vector(b_text)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return LinearResult("dot", f"{a_text} · {b_text}", "", False, str(error))
    result = vector_dot(a, b)
    if result is None:
        return LinearResult(
            "dot", f"{a_text} · {b_text}", "", False, "vector dimensions must agree"
        )
    return LinearResult("dot", f"{a_text} · {b_text}", _fmt(result), True)


def vector_add_text(a_text: str, b_text: str, subtract: bool = False) -> LinearResult:
    """Add or subtract two explicit vector literals."""
    try:
        a, b = _parse_vector(a_text), _parse_vector(b_text)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return LinearResult("vector", f"{a_text}, {b_text}", "", False, str(error))
    result = vector_add(a, b, -1 if subtract else 1)
    if result is None:
        return LinearResult(
            "vector", f"{a_text}, {b_text}", "", False, "vector dimensions must agree"
        )
    rendered = "(" + ", ".join(_fmt(value) for value in result) + ")"
    return LinearResult("vector", f"{a_text}, {b_text}", rendered, True)


def matrix_add_text(a_text: str, b_text: str, subtract: bool = False) -> LinearResult:
    """Add or subtract matrices with matching dimensions."""
    try:
        a, b = parse_matrix(a_text), parse_matrix(b_text)
    except ValueError as error:
        return LinearResult("matrix", f"{a_text}, {b_text}", "", False, str(error))
    result = matrix_add(a, b, -1 if subtract else 1)
    if result is None:
        return LinearResult(
            "matrix", f"{a_text}, {b_text}", "", False, "matrix dimensions must agree"
        )
    return LinearResult("matrix", f"{a_text}, {b_text}", render_matrix(result), True)


def matrix_scale_text(matrix_text: str, scalar_text: str) -> LinearResult:
    """Multiply a matrix by an exact rational scalar."""
    try:
        matrix = parse_matrix(matrix_text)
        scalar = Fraction(scalar_text.strip())
    except (ValueError, ZeroDivisionError) as error:
        return LinearResult("scale", f"{scalar_text} * {matrix_text}", "", False, str(error))
    result = matrix_scale(matrix, scalar)
    return LinearResult("scale", f"{scalar_text} * {matrix_text}", render_matrix(result), True)


def vector_cross_text(a_text: str, b_text: str) -> LinearResult:
    """Compute the exact cross product of two 3D vectors."""
    try:
        a, b = _parse_vector(a_text), _parse_vector(b_text)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return LinearResult("cross", f"{a_text} × {b_text}", "", False, str(error))
    result = vector_cross(a, b)
    if result is None:
        return LinearResult(
            "cross",
            f"{a_text} × {b_text}",
            "",
            False,
            "cross product requires two 3D vectors",
        )
    rendered = "(" + ", ".join(_fmt(value) for value in result) + ")"
    return LinearResult("cross", f"{a_text} × {b_text}", rendered, True)


def determinant_text(text: str) -> LinearResult:
    """ "determinant of [[1,2],[3,4]]" -> -2."""
    try:
        m = parse_matrix(text)
        d = determinant(m)
    except ValueError as e:
        return LinearResult("det", text, "", False, str(e))
    if d is None:
        return LinearResult("det", text, "", False, "determinant needs a square matrix")
    return LinearResult("det", text, _fmt(d), True, data={"det": str(d)})


def rank_text(text: str) -> LinearResult:
    """ "rank of [[1,2],[2,4]]" -> 1."""
    try:
        m = parse_matrix(text)
    except ValueError as e:
        return LinearResult("rank", text, "", False, str(e))
    return LinearResult("rank", text, str(rank(m)), True)


def matmul_text(a_text: str, b_text: str) -> LinearResult:
    """ "[[1,0],[0,1]] times [[1,2],[3,4]]" — the product."""
    try:
        a, b = parse_matrix(a_text), parse_matrix(b_text)
    except ValueError as e:
        return LinearResult("matmul", f"{a_text} * {b_text}", "", False, str(e))
    out = matmul(a, b)
    if out is None:
        return LinearResult(
            "matmul", f"{a_text} * {b_text}", "", False, "inner dimensions must agree"
        )
    return LinearResult("matmul", f"{a_text} * {b_text}", render_matrix(out), True)


def solve_text(a_text: str, b_text: str) -> LinearResult:
    """ "solve [[2,1],[1,-1]] x = [5, 1]" -> x = (2, 1)."""
    try:
        a = parse_matrix(a_text)
        lit = ast.literal_eval(b_text.strip())
        b = [Fraction(str(x)) for x in lit]
    except (ValueError, SyntaxError) as e:
        return LinearResult("solve", f"{a_text} x = {b_text}", "", False, str(e))
    x = solve(a, b)
    if x is None:
        return LinearResult(
            "solve", f"{a_text} x = {b_text}", "", False, "the system has no unique solution"
        )
    out = "(" + ", ".join(_fmt(v) for v in x) + ")"
    return LinearResult("solve", f"{a_text} x = {b_text}", f"x = {out}", True)
