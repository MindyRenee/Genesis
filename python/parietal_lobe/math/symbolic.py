"""Symbolic mathematics — the parietal faculty's formal layer.

Where ``arithmetic.py`` computes exact numbers and
``number_sense.py`` estimates magnitudes, this module manipulates
structure: algebraic simplification, calculus, equation solving,
matrix operations, and exact-equality checking. It is the seed of a
proof substrate — the part of mathematical thought that does not
guess.

Verification is separated from generation on purpose. Cognition
proposes expressions; ``check_equality`` alone returns the verdict,
and "undecided" is reported honestly rather than rounded into a yes.
That mirrors the provenance discipline used elsewhere in the
architecture — a claim earns "verified" only through the oracle,
never by assertion.

Backed by SymPy. The library is imported lazily so this module loads
in environments where the CAS wheel is absent; callers then get a
clean failure rather than a crash at import.
"""

from __future__ import annotations

import itertools
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolicResult:
    """The outcome of a symbolic operation."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """The oracle's answer on a claimed equality.

    ``status`` is one of "verified" (the oracle proved the sides
    equal), "refuted" (proved unequal, with a numeric ``witness``
    when one could be produced), or "undecided" (the oracle could
    not tell — the honest third answer).
    """

    status: str
    lhs: str
    rhs: str
    detail: str
    witness: dict[str, Any] | None = None


def _sympy() -> Any:
    """Import sympy lazily; return None when the wheel is absent."""
    try:
        import sympy
    except ImportError:
        return None
    return sympy


def _parse(text: str, sp: Any) -> Any:
    """Parse natural math text ("2x^2 + x") into a sympy expression."""
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    return parse_expr(
        text.strip(),
        transformations=(
            *standard_transformations,
            convert_xor,
            implicit_multiplication_application,
        ),
    )


_MATRIX_OPS = ("det", "eigenvals", "trace", "inv", "charpoly")

_OPERATIONS = (
    "simplify",
    "expand",
    "factor",
    "cancel",
    "apart",
    "together",
    "trigsimp",
    "diff",
    "integrate",
    "solve",
    "limit",
    "series",
    "evalf",
    *_MATRIX_OPS,
)


def compute(
    operation: str,
    expression: str,
    variable: str = "x",
    point: str = "0",
    order: int = 0,
    lower: str = "",
    upper: str = "",
) -> SymbolicResult:
    """Run one symbolic operation on an expression string.

    Operations: simplify, expand, factor, cancel, apart, together,
    trigsimp, diff (order = nth derivative), integrate (indefinite,
    or definite when ``lower``/``upper`` are both given), solve
    (an equation "lhs = rhs" or an expression set to zero), limit
    (at ``point``), series (about ``point``, ``order`` terms),
    evalf (20-digit value), and the matrix ops det, eigenvals,
    trace, inv, charpoly on a "[[a, b], [c, d]]" literal.
    """
    op = operation.strip().lower()
    if op in ("differentiate", "derivative"):
        op = "diff"
    if op not in _OPERATIONS:
        return SymbolicResult(
            operation=op,
            input=expression,
            output="",
            ok=False,
            error=f"unknown operation '{operation}'; known: {', '.join(_OPERATIONS)}",
        )
    sp = _sympy()
    if sp is None:
        return SymbolicResult(
            operation=op,
            input=expression,
            output="",
            ok=False,
            error="sympy is not installed in this environment",
        )
    try:
        var = sp.Symbol(variable)
        if op in _MATRIX_OPS:
            matrix = sp.Matrix(_parse(expression, sp))
            result = _matrix_op(matrix, op, var, sp)
            return SymbolicResult(
                operation=op,
                input=expression,
                output=sp.sstr(result),
                ok=True,
                data={"result": sp.sstr(result)},
            )
        if op == "solve":
            result = sp.solve(_as_equation(expression, sp), var)
        else:
            expr = _parse(expression, sp)
            result = _expr_op(expr, op, var, point, order, lower, upper, sp)
        out = sp.sstr(result)
        data: dict[str, Any] = {"result": out}
        if op == "solve":
            data["solutions"] = [sp.sstr(s) for s in result]
        return SymbolicResult(operation=op, input=expression, output=out, ok=True, data=data)
    except Exception as e:  # noqa: BLE001 — sympy raises many exception types
        return SymbolicResult(
            operation=op,
            input=expression,
            output="",
            ok=False,
            error=f"{type(e).__name__}: {e}",
        )


def _expr_op(
    expr: Any,
    op: str,
    var: Any,
    point: str,
    order: int,
    lower: str,
    upper: str,
    sp: Any,
) -> Any:
    """Dispatch a single-expression operation on a parsed expression."""
    if op == "diff":
        return sp.diff(expr, var) if order <= 1 else sp.diff(expr, var, order)
    if op == "integrate":
        if lower or upper:
            if not (lower and upper):
                raise ValueError("definite integration needs both lower and upper")
            return sp.integrate(expr, (var, _parse(lower, sp), _parse(upper, sp)))
        return sp.integrate(expr, var)
    if op == "limit":
        return sp.limit(expr, var, _parse(point, sp))
    if op == "series":
        return sp.series(expr, var, _parse(point, sp), n=order if order > 0 else 6)
    if op == "evalf":
        return expr.evalf(20)
    if op == "apart":
        return sp.apart(expr, var)
    return {
        "simplify": sp.simplify,
        "expand": sp.expand,
        "factor": sp.factor,
        "cancel": sp.cancel,
        "together": sp.together,
        "trigsimp": sp.trigsimp,
    }[op](expr)


def _matrix_op(matrix: Any, op: str, var: Any, sp: Any) -> Any:
    """Dispatch a matrix operation on a parsed Matrix."""
    if op == "det":
        return matrix.det()
    if op == "eigenvals":
        return matrix.eigenvals()
    if op == "trace":
        return matrix.trace()
    if op == "inv":
        return matrix.inv()
    return matrix.charpoly(var).as_expr()


def _as_equation(text: str, sp: Any) -> Any:
    """Turn "lhs = rhs" into an Eq; a bare expression means "= 0"."""
    if "=" not in text:
        return _parse(text, sp)
    lhs, rhs = text.split("=", 1)
    return sp.Eq(_parse(lhs, sp), _parse(rhs, sp))


def check_equality(lhs: str, rhs: str) -> Verdict:
    """Decide whether two expressions are exactly equal.

    The oracle is sympy's ``equals`` (simplification plus numeric
    evaluation) backed by a ``simplify(lhs - rhs)`` second opinion.
    A ``refuted`` verdict tries to carry a witness: a substitution
    where the two sides numerically differ. Raises ``ValueError``
    when a side cannot be parsed.
    """
    sp = _sympy()
    if sp is None:
        return Verdict("undecided", lhs, rhs, "sympy is not installed")
    try:
        left = _parse(lhs, sp)
        right = _parse(rhs, sp)
    except Exception as e:
        raise ValueError(f"could not parse: {e}") from e
    try:
        equal = left.equals(right)
    except Exception as e:  # noqa: BLE001
        return Verdict("undecided", lhs, rhs, f"oracle failed: {e}")
    if equal is True:
        return Verdict("verified", lhs, rhs, "sympy.equals proved the sides equal")
    if equal is False:
        return Verdict(
            "refuted",
            lhs,
            rhs,
            "sympy.equals proved the sides unequal",
            witness=_find_witness(left, right, sp),
        )
    diff = sp.simplify(left - right)
    if diff.is_zero is True:
        return Verdict("verified", lhs, rhs, "simplify(lhs - rhs) reduced to 0")
    if diff.is_zero is False:
        return Verdict(
            "refuted",
            lhs,
            rhs,
            f"simplify(lhs - rhs) is nonzero: {sp.sstr(diff)}",
            witness=_find_witness(left, right, sp),
        )
    return Verdict(
        "undecided",
        lhs,
        rhs,
        "oracle could not decide — this claim stays unverified",
    )


def _find_witness(left: Any, right: Any, sp: Any) -> dict[str, Any] | None:
    """Find a substitution where lhs and rhs numerically differ."""
    syms = sorted((left - right).free_symbols, key=lambda s: s.name)
    candidates = (
        sp.Integer(2),
        sp.Integer(3),
        sp.Integer(5),
        sp.Rational(1, 2),
        sp.Integer(-1),
        sp.Integer(7),
    )
    pools = [candidates[:3]] * min(len(syms), 3)
    for combo in itertools.product(*pools):
        subs = dict(zip(syms[: len(combo)], combo, strict=False))
        try:
            gap = complex((left - right).subs(subs).evalf(30))
        except Exception:  # noqa: BLE001 — some substitutions are singular
            continue
        if abs(gap) > 1e-6:
            return {
                "substitution": {s.name: sp.sstr(v) for s, v in subs.items()},
                "difference": sp.sstr(sp.Float(gap.real)),
            }
    return None


def split_equality(text: str) -> tuple[str, str] | None:
    """Split "lhs = rhs" into its sides, or None if not equational.

    Only a bare single ``=`` splits — ``<=``, ``>=``, ``!=``, ``==``
    are not equations and return None, which keeps non-equational
    claims out of the oracle's reach rather than misjudging them.
    """
    parts = re.split(r"(?<![<>=!])=(?!=)", text, maxsplit=1)
    if len(parts) != 2:
        return None
    lhs, rhs = parts[0].strip(), parts[1].strip()
    return (lhs, rhs) if lhs and rhs else None


# Relations the oracle can judge beyond bare equality.
_RELATION_OPS = ("<=", ">=", "<", ">", "=")
# A bounded-universal claim: "lhs = rhs for n <= 100" — the variable
# is an integer point checked exhaustively inside the bound.
_BOUNDED_RE = re.compile(
    r"^(?:for\s+(?:all\s+)?|∀\s*)([a-zA-Z])\s*(?:<=|≤)\s*(\d+)\s*[:,]\s*(.+)$"
    r"|^(.+?)\s+for\s+(?:all\s+)?([a-zA-Z])\s*(?:<=|≤)\s*(\d+)\s*$"
)


def split_claim(text: str) -> dict[str, Any] | None:
    """Split a research claim into judgeable form, or None.

    Two claim shapes are understood:

    * ``lhs op rhs`` — an equality or inequality (``=``, ``<=``,
      ``>=``, ``<``, ``>``) judged symbolically.
    * ``lhs = rhs for n <= K`` / ``for n <= K: lhs = rhs`` — a
      bounded universal over integers, checked exhaustively.

    ``!=`` and ``==`` are deliberately not relations the oracle
    judges; unparseable or non-relational text returns None.
    """
    m = _BOUNDED_RE.match(text.strip())
    if m is not None:
        if m.group(1) is not None:
            # "for n <= K: lhs = rhs" — groups (var, bound, body)
            var, bound, body = m.group(1), int(m.group(2)), m.group(3)
        else:
            # "lhs = rhs for n <= K" — groups (body, var, bound)
            body, var, bound = m.group(4), m.group(5), int(m.group(6))
        sides = split_equality(body)
        if sides is None:
            return None
        return {"lhs": sides[0], "op": "=", "rhs": sides[1], "var": var, "lo": 1, "hi": bound}
    parts = re.split(
        r"(<=|>=|<|>|(?<![<>=!])=(?!=))",
        text.strip(),
        maxsplit=1,
    )
    if len(parts) != 3:
        return None
    lhs, op, rhs = (p.strip() for p in parts)
    if not lhs or not rhs:
        return None
    return {"lhs": lhs, "op": op, "rhs": rhs, "var": None, "lo": 1, "hi": 100}


def check_inequality(lhs: str, rhs: str, op: str) -> Verdict:
    """Judge ``lhs op rhs`` symbolically for ``op`` in <, <=, >, >=.

    The oracle asks sympy whether ``simplify(lhs - rhs)`` carries
    the required sign; a ``refuted`` verdict tries to carry a
    witness — a substitution where the sign fails. Raises
    ``ValueError`` when a side cannot be parsed.
    """
    sp = _sympy()
    if sp is None:
        return Verdict("undecided", lhs, rhs, "sympy is not installed")
    try:
        left = _parse(lhs, sp)
        right = _parse(rhs, sp)
    except Exception as e:
        raise ValueError(f"could not parse: {e}") from e
    # Normalize to "difference must be nonneg/positive". Bare symbols
    # parse as complex; inequalities are real-domain claims, so if the
    # sign is undecidable retry with the symbols declared real —
    # x**2 >= 0 is provable over the reals, not over the complexes.
    diff = sp.simplify((left - right) if op in (">", ">=") else (right - left))
    strict = op in (">", "<")
    try:
        holds = diff.is_positive if strict else diff.is_nonnegative
    except Exception as e:  # noqa: BLE001
        return Verdict("undecided", lhs, rhs, f"oracle failed: {e}")
    if holds is None and diff.free_symbols:
        reals = {s: sp.Symbol(s.name, real=True) for s in diff.free_symbols}
        try:
            real_diff = sp.simplify(diff.subs(reals))
            holds = real_diff.is_positive if strict else real_diff.is_nonnegative
        except Exception as e:  # noqa: BLE001 — undecided stands
            logger.debug(f"real-domain simplification failed: {e}")
    if holds is True:
        return Verdict(
            "verified",
            lhs,
            rhs,
            f"sympy proved lhs {op} rhs (difference has the sign)",
        )
    if holds is False:
        return Verdict(
            "refuted",
            lhs,
            rhs,
            f"sympy proved lhs {op} rhs does not hold",
            witness=_find_sign_witness(left, right, sp, op),
        )
    return Verdict(
        "undecided",
        lhs,
        rhs,
        "oracle could not decide — this claim stays unverified",
    )


def check_bounded_equality(
    lhs: str,
    rhs: str,
    var: str = "n",
    lo: int = 1,
    hi: int = 100,
) -> Verdict:
    """Check ``lhs = rhs`` at every integer ``var`` in ``[lo, hi]``.

    One failing point is a real refutation — the witness is the
    failing integer. Every point passing is strong evidence but
    *not* a proof: the verdict stays ``undecided`` with the checked
    range in its detail, so bounded support can never masquerade
    as a terminal verdict. Raises ``ValueError`` on unparseable
    input; ``hi`` is capped at 10 000 to bound the work.
    """
    sp = _sympy()
    if sp is None:
        return Verdict("undecided", lhs, rhs, "sympy is not installed")
    try:
        left = _parse(lhs, sp)
        right = _parse(rhs, sp)
    except Exception as e:
        raise ValueError(f"could not parse: {e}") from e
    hi = min(hi, 10_000)
    symbol = sp.Symbol(var)
    for value in range(lo, hi + 1):
        try:
            gap = complex((left - right).subs(symbol, value).evalf(30))
        except Exception:  # noqa: BLE001 — singular point is a witness too
            return Verdict(
                "refuted",
                lhs,
                rhs,
                f"undefined at {var}={value}",
                witness={"substitution": {var: str(value)}, "difference": "undefined"},
            )
        if abs(gap) > 1e-9:
            return Verdict(
                "refuted",
                lhs,
                rhs,
                f"sides differ at {var}={value}",
                witness={
                    "substitution": {var: str(value)},
                    "difference": sp.sstr(sp.Float(gap.real)),
                },
            )
    return Verdict(
        "undecided",
        lhs,
        rhs,
        f"holds at every integer {var} in [{lo},{hi}] — bounded evidence, not a proof",
    )


def _find_sign_witness(
    left: Any,
    right: Any,
    sp: Any,
    op: str,
) -> dict[str, Any] | None:
    """Find a substitution where ``lhs op rhs`` fails."""
    syms = sorted((left - right).free_symbols, key=lambda s: s.name)
    candidates = (
        sp.Integer(-1),
        sp.Integer(0),
        sp.Rational(1, 2),
        sp.Integer(2),
        sp.Integer(-3),
        sp.Integer(5),
        sp.Integer(-7),
    )
    pools = [candidates] * min(len(syms), 2)
    for combo in itertools.product(*pools):
        subs = dict(zip(syms[: len(combo)], combo, strict=False))
        try:
            gap = float((left - right).subs(subs).evalf(30))
        except Exception:  # noqa: BLE001 — singular point may be the witness
            return {
                "substitution": {s.name: sp.sstr(v) for s, v in subs.items()},
                "difference": "undefined",
            }
        fails = (
            gap <= 0
            if op in (">",)
            else gap < 0
            if op in (">=",)
            else gap >= 0
            if op in ("<",)
            else gap > 0
        )
        if fails:
            return {
                "substitution": {s.name: sp.sstr(v) for s, v in subs.items()},
                "difference": str(gap),
            }
    return None


def mutate_identity(
    lhs: str,
    rhs: str,
    rng: random.Random | None = None,
    variable: str = "x",
) -> dict[str, Any] | None:
    """Vary a known identity into a candidate conjecture.

    Sound moves apply the same transformation to both sides —
    differentiate, scale, shift the variable, square, exponentiate,
    add a shared term — so the candidate inherits truth and the
    oracle usually confirms it. Free moves perturb one side only;
    the oracle usually refutes them with a counterexample. Both
    outcomes are knowledge: sound moves extend what is proven, free
    moves map where the identity's boundary actually sits.

    Returns {"lhs", "rhs", "method", "sound"} or None when sympy is
    absent, a side won't parse, or the result is too large to be a
    useful claim.
    """
    sp = _sympy()
    if sp is None:
        return None
    rng = rng or random.Random()
    try:
        left = _parse(lhs, sp)
        right = _parse(rhs, sp)
    except Exception:  # noqa: BLE001 — unparseable input is just unusable
        return None
    var = sp.Symbol(variable)
    c = rng.randint(1, 3)
    sound_moves = [
        ("differentiate both sides", lambda: (sp.diff(left, var), sp.diff(right, var))),
        ("add the same term", lambda: (left + c * var, right + c * var)),
        ("multiply both sides", lambda: (c * left, c * right)),
        ("shift the variable", lambda: (left.subs(var, var + c), right.subs(var, var + c))),
        ("square both sides", lambda: (left**2, right**2)),
        ("exponentiate both sides", lambda: (sp.exp(left), sp.exp(right))),
    ]
    free_moves = [
        ("perturbed the right side", lambda: (left, right + var)),
        ("perturbed the left side", lambda: (left * (c + 1), right)),
        ("perturbed a shared term", lambda: (left + c, right)),
    ]
    moves = [(m, True) for m in sound_moves] * 3 + [(m, False) for m in free_moves]
    (method, fn), sound = rng.choice(moves)
    try:
        new_lhs, new_rhs = fn()
        out_lhs, out_rhs = sp.sstr(new_lhs), sp.sstr(new_rhs)
    except Exception:  # noqa: BLE001 — transforms can fail on odd trees
        return None
    if len(out_lhs) + len(out_rhs) > 280:
        return None
    return {"lhs": out_lhs, "rhs": out_rhs, "method": method, "sound": sound}
