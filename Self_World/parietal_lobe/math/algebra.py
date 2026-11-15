"""Algebra — the parietal faculty's formal manipulation.

Where ``arithmetic.py`` computes a number, this module keeps the
unknown: expressions over variables parse into trees, simplify to
canonical form, expand by distribution, and solve equations. The
manipulation is endogenous — rules applied to structure, no
``eval()``, no host-language evaluation, no external CAS.
``symbolic.py`` remains the oracle: what this faculty produces is
verified there, never asserted.

The tree:

    Num(v)            a constant
    Var(name)         the unknown (or a named constant — pi, e)
    Neg(x)            unary negation
    Add(terms)        a sum, flattened and canonically ordered
    Mul(factors)      a product, flattened and canonically ordered
    Pow(base, exp)    a power
    Call(name, arg)   an elementary function application

Canonical form is how "the same expression" is recognized: sums
collect like terms, products collect like factors, and both sort
their children — so equality of structure is equality of math.

References:
    - the derivation rules are the standard elementary table —
      Strang, *Calculus*, ch. 1–3.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

# ─── The expression tree ────────────────────────────────────────
#
# One immutable node per shape; every transform returns a new tree.


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Neg:
    x: Expr


@dataclass(frozen=True)
class Add:
    terms: tuple[Expr, ...]


@dataclass(frozen=True)
class Mul:
    factors: tuple[Expr, ...]


@dataclass(frozen=True)
class Pow:
    base: Expr
    exp: Expr


@dataclass(frozen=True)
class Call:
    name: str
    arg: Expr


Expr = Num | Var | Neg | Add | Mul | Pow | Call

# Named constants — parse as Var so rendering keeps the symbol;
# evaluation and the calculus rules treat them as numbers.
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_FUNCS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
}


@dataclass(frozen=True)
class AlgebraResult:
    """The outcome of a manipulation — the operation, the input as
    given, the output rendered, the ok flag, and the failure wording
    when ok is false. ``data`` carries structured extras (roots)."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, object] | None = None


# ─── Scanner ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"""
    \d+\.\d+|\d+|\.\d+      # numbers
    | \*\*|//               # multi-char operators first
    | [+\-*/^%(),=<>!]      # symbols
    | [a-zA-Z_][a-zA-Z_0-9]* # identifiers
    | \S                    # anything else, so errors are visible
    """,
    re.VERBOSE,
)


def _scan(text: str) -> list[str]:
    """Split the math text into tokens — the surface dialect is
    digits, symbols, and identifiers; words belong to arithmetic."""
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


class ParseError(ValueError):
    """The text was not a well-formed expression."""


# ─── Parser ─────────────────────────────────────────────────────
#
#     expr    ::=  term   ( ( "+" | "-" )            term   )*
#     term    ::=  factor ( ( "*" | "/" | IMPLICIT ) factor )*
#     factor  ::=  unary  ( ("**" | "^") factor )?
#     unary   ::=  ( "-" | "+" ) unary  |  postfix
#     postfix ::=  atom   ( "!" )*
#     atom    ::=  NUM | IDENT | IDENT "(" expr ")" | "(" expr ")"
#
# IMPLICIT multiplication is juxtaposition — "2x", "3(x + 1)",
# "(x + 1)(x - 1)", "x sin(x)" — the way algebra is written.


class _Parser:
    """Recursive-descent parser — one method per grammar level,
    descent order is precedence."""

    def __init__(self, tokens: list[str]) -> None:
        self.toks = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def next(self) -> str | None:
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def parse(self) -> Expr:
        value = self._expr()
        if self.pos != len(self.toks):
            raise ParseError(f"trailing tokens: {' '.join(self.toks[self.pos :])}")
        return value

    def _expr(self) -> Expr:
        """Addition and subtraction — the sum level."""
        terms = [self._term()]
        neg_flags = [False]
        while self.peek() in ("+", "-"):
            neg_flags.append(self.next() == "-")
            terms.append(self._term())
        parts = [Neg(t) if neg else t for t, neg in zip(terms, neg_flags, strict=True)]
        return parts[0] if len(parts) == 1 else Add(tuple(parts))

    def _term(self) -> Expr:
        """Multiplication, division, and juxtaposition — the product
        level. A factor that begins with a number, an identifier, or
        an open paren after one already parsed multiplies silently."""
        factors = [self._factor()]
        while True:
            op = self.peek()
            if op in ("*", "/"):
                self.next()
                rhs = self._factor()
                if op == "/":
                    rhs = Pow(rhs, Num(-1.0))
                factors.append(rhs)
                continue
            if op is not None and (
                re.fullmatch(r"\d+\.\d+|\d+|\.\d+", op)
                or op == "("
                or re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", op)
            ):
                factors.append(self._factor())
                continue
            break
        return factors[0] if len(factors) == 1 else Mul(tuple(factors))

    def _factor(self) -> Expr:
        """Right-associative exponentiation."""
        base = self._unary()
        if self.peek() in ("**", "^"):
            self.next()
            return Pow(base, self._factor())
        return base

    def _unary(self) -> Expr:
        """Signs ahead of the operand."""
        tok = self.peek()
        if tok == "-":
            self.next()
            return Neg(self._unary())
        if tok == "+":
            self.next()
            return self._unary()
        return self._postfix()

    def _postfix(self) -> Expr:
        """Factorial — the one postfix operator."""
        value = self._atom()
        while self.peek() == "!":
            self.next()
            value = Call("factorial", value)
        return value

    def _atom(self) -> Expr:
        tok = self.next()
        if tok is None:
            raise ParseError("unexpected end of expression")
        if tok == "(":
            value = self._expr()
            if self.next() != ")":
                raise ParseError("missing closing parenthesis")
            return value
        if re.fullmatch(r"\d+\.\d+|\d+|\.\d+", tok):
            return Num(float(tok))
        if re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", tok):
            if self.peek() == "(":
                self.next()
                arg = self._expr()
                if self.next() != ")":
                    raise ParseError("missing closing parenthesis")
                if tok not in _FUNCS:
                    raise ParseError(f"unknown function: {tok}")
                return Call(tok, arg)
            return Var(tok)
        raise ParseError(f"unexpected token: {tok}")


def parse(text: str) -> Expr:
    """Parse expression text into the tree — the entry the whole
    faculty builds on."""
    tokens = _scan(text)
    if not tokens:
        raise ParseError("empty expression")
    return _Parser(tokens).parse()


def parse_equation(text: str) -> tuple[Expr, Expr]:
    """Parse an equation ``lhs = rhs`` — the shape a solve takes."""
    tokens = _scan(text)
    if "=" not in tokens:
        raise ParseError("no equation — expected '='")
    left = _Parser(tokens[: tokens.index("=")]).parse()
    right = _Parser(tokens[tokens.index("=") + 1 :]).parse()
    return left, right


# ─── Rendering ──────────────────────────────────────────────────


def _num_text(v: float) -> str:
    """Render a number the way a person writes it — no trailing .0,
    and a clean rational rather than float dust when one exists."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    frac = Fraction(v).limit_denominator(1000)
    if frac.denominator > 1 and float(frac) == v:
        return f"{frac.numerator}/{frac.denominator}"
    return repr(v)


def render(expr: Expr) -> str:
    """Canonical infix text for a tree — ``**`` powers, explicit
    ``*`` products, parenthesized where precedence needs it."""
    if isinstance(expr, Num):
        return _num_text(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Neg):
        inner = render(expr.x)
        need = isinstance(expr.x, (Add,))
        return f"-({inner})" if need else f"-{inner}"
    if isinstance(expr, Add):
        out = render(expr.terms[0])
        for t in expr.terms[1:]:
            neg, pos = _signed_parts(t)
            out += f" - {render(pos)}" if neg else f" + {render(t)}"
        return out
    if isinstance(expr, Mul):
        parts = []
        for f in expr.factors:
            s = render(f)
            if isinstance(f, Add):
                s = f"({s})"
            parts.append(s)
        return "*".join(parts)
    if isinstance(expr, Pow):
        base = render(expr.base)
        exp = render(expr.exp)
        if isinstance(expr.base, (Add, Mul, Neg)):
            base = f"({base})"
        if isinstance(expr.exp, (Add, Mul, Neg)):
            exp = f"({exp})"
        return f"{base}**{exp}"
    if isinstance(expr, Call):
        if expr.name == "factorial":
            inner = render(expr.arg)
            if isinstance(expr.arg, (Add, Mul, Neg)):
                inner = f"({inner})"
            return f"{inner}!"
        return f"{expr.name}({render(expr.arg)})"
    raise TypeError(f"unknown node: {expr!r}")


def _signed_parts(term: Expr) -> tuple[bool, Expr]:
    """Split a term's leading sign for sum rendering — ``x + (-2)``
    and ``x + (-2*y)`` are both subtractions on the page."""
    if isinstance(term, Neg):
        return True, term.x
    if isinstance(term, Num) and term.value < 0:
        return True, Num(-term.value)
    if isinstance(term, Mul) and isinstance(term.factors[0], Num) and term.factors[0].value < 0:
        rest = (Num(-term.factors[0].value), *term.factors[1:])
        pos = rest[0] if len(rest) == 1 else Mul(rest)
        return True, pos
    return False, term


def _degree(expr: Expr) -> float:
    """The polynomial degree of a term — how a canonical sum orders
    itself: higher degree first, constants last."""
    if isinstance(expr, Num):
        return 0.0
    if isinstance(expr, Var):
        return 0.0 if expr.name in _CONSTANTS else 1.0
    if isinstance(expr, Neg):
        return _degree(expr.x)
    if isinstance(expr, Pow):
        if isinstance(expr.exp, Num):
            return _degree(expr.base) * expr.exp.value
        return 0.0
    if isinstance(expr, Mul):
        return sum(_degree(f) for f in expr.factors)
    if isinstance(expr, Add):
        return max((_degree(t) for t in expr.terms), default=0.0)
    return 0.0


# ─── Canonical form ─────────────────────────────────────────────
#
# simplify() is the heart: a transform to the canonical shape where
# like terms are collected, like factors are powered, and every
# child is ordered — the reason "x + x" and "2x" meet.


def _canonical_key(expr: Expr) -> str:
    """Sort key for canonical ordering — variables and structure
    first, bare numbers last in sums and first in products."""
    if isinstance(expr, Num):
        return "zzzz_num"
    if isinstance(expr, Var):
        return f"a_var_{expr.name}"
    if isinstance(expr, Pow):
        return f"b_pow_{_canonical_key(expr.base)}"
    return f"c_{render(expr)}"


def _term_parts(expr: Expr) -> tuple[float, Expr | None]:
    """Split a term into (coefficient, non-numeric base).

    ``2*x**2`` → (2.0, x**2); ``x`` → (1.0, x); ``5`` → (5.0, None).
    The base is what like terms share — combining is adding
    coefficients over the same base.
    """
    if isinstance(expr, Num):
        return expr.value, None
    if isinstance(expr, Neg):
        c, inner_base = _term_parts(expr.x)
        return -c, inner_base
    if isinstance(expr, Mul):
        coef = 1.0
        rest: list[Expr] = []
        for f in expr.factors:
            if isinstance(f, Num):
                coef *= f.value
            elif isinstance(f, Neg) and isinstance(f.x, Num):
                coef *= -f.x.value
            else:
                rest.append(f)
        term_base: Expr | None = None
        if rest:
            term_base = rest[0] if len(rest) == 1 else Mul(tuple(rest))
        return coef, term_base
    return 1.0, expr


def _product_parts(expr: Expr) -> tuple[Expr, float]:
    """Split a factor into (base, exponent) — ``x**2`` → (x, 2),
    ``x`` → (x, 1); non-numeric exponents stay whole."""
    if isinstance(expr, Pow) and isinstance(expr.exp, Num):
        return expr.base, expr.exp.value
    return expr, 1.0


def simplify(expr: Expr) -> Expr:
    """Reduce a tree to canonical form — the faculty's tidying.

    Constant subtrees evaluate; identities erase (x + 0, x*1, x**0,
    x**1); sums and products flatten, combine like parts, and sort.
    Applied bottom-up, so each node meets simplified children.
    """
    if isinstance(expr, (Num, Var)):
        return expr
    if isinstance(expr, Neg):
        x = simplify(expr.x)
        if isinstance(x, Num):
            return Num(-x.value)
        if isinstance(x, Neg):
            return x.x
        return Neg(x)
    if isinstance(expr, Pow):
        base = simplify(expr.base)
        exp = simplify(expr.exp)
        if isinstance(exp, Num):
            if exp.value == 0.0:
                return Num(1.0)
            if exp.value == 1.0:
                return base
        if isinstance(base, Num) and isinstance(exp, Num):
            try:
                v = base.value**exp.value
                if math.isfinite(v) and not isinstance(v, complex):
                    return Num(v)
            except (OverflowError, ValueError, ZeroDivisionError):
                return Pow(base, exp)
            return Pow(base, exp)
        if isinstance(base, Pow):
            return Pow(base.base, simplify(Mul((base.exp, exp))))
        return Pow(base, exp)
    if isinstance(expr, Add):
        return _simplify_add(expr)
    if isinstance(expr, Mul):
        return _simplify_mul(expr)
    if isinstance(expr, Call):
        arg = simplify(expr.arg)
        if expr.name == "factorial":
            if isinstance(arg, Num) and arg.value >= 0 and arg.value == int(arg.value):
                return Num(float(math.factorial(int(arg.value))))
            return Call("factorial", arg)
        if expr.name == "abs" and isinstance(arg, Num):
            return Num(abs(arg.value))
        if expr.name == "abs":
            return Call("abs", arg)
        if isinstance(arg, Num):
            try:
                v = _FUNCS[expr.name](arg.value)
                if math.isfinite(v):
                    return Num(v)
            except (ValueError, OverflowError):
                return Call(expr.name, arg)
        return Call(expr.name, arg)
    return expr


def _simplify_add(expr: Add) -> Expr:
    """The sum's canonical form: flattened, like terms combined by
    coefficient over the same base, ordered by degree then name."""
    flat: list[Expr] = []
    for t in expr.terms:
        t = simplify(t)
        if isinstance(t, Add):
            flat.extend(t.terms)
        else:
            flat.append(t)
    # Combine like terms: group coefficient by canonical base.
    groups: dict[str, tuple[float, Expr | None]] = {}
    for t in flat:
        coef, tbase = _term_parts(t)
        key = "" if tbase is None else render(tbase)
        prev = groups.get(key, (0.0, tbase))
        groups[key] = (prev[0] + coef, tbase)
    terms = [_unterm(coef, base) for coef, base in groups.values() if coef != 0.0]
    if not terms:
        return Num(0.0)
    terms.sort(key=lambda t: (-_degree(t), render(t)))
    return terms[0] if len(terms) == 1 else Add(tuple(terms))


def _simplify_mul(expr: Mul) -> Expr:
    """The product's canonical form: flattened, like factors merged
    into powers, the numeric coefficient folded to the front."""
    flat: list[Expr] = []
    for f in expr.factors:
        f = simplify(f)
        if isinstance(f, Mul):
            flat.extend(f.factors)
        else:
            flat.append(f)
    coef = 1.0
    powers: dict[str, tuple[Expr, float]] = {}
    negs = 0
    for f in flat:
        if isinstance(f, Num):
            coef *= f.value
        elif isinstance(f, Neg):
            negs += 1
            inner = f.x
            if isinstance(inner, Num):
                coef *= inner.value
            else:
                b, e = _product_parts(inner)
                key = render(b)
                p = powers.get(key, (b, 0.0))
                powers[key] = (b, p[1] + e)
        else:
            b, e = _product_parts(f)
            key = render(b)
            p = powers.get(key, (b, 0.0))
            powers[key] = (b, p[1] + e)
    if coef == 0.0:
        return Num(0.0)
    factors: list[Expr] = []
    for b, e in powers.values():
        if e == 0.0:
            continue
        if e == 1.0:
            factors.append(b)
        else:
            factors.append(Pow(b, Num(e)))
    factors.sort(key=_canonical_key)
    if coef != 1.0 or not factors:
        factors.insert(0, Num(coef))
    out = factors[0] if len(factors) == 1 else Mul(tuple(factors))
    return Neg(out) if negs % 2 else out


def _unterm(coef: float, base: Expr | None) -> Expr:
    """Rebuild (coef, base) into a term — sign rides the coefficient."""
    if base is None:
        return Num(coef)
    if coef == 1.0:
        return base
    if coef == -1.0:
        return Neg(base)
    return Mul((Num(coef), base))


def expand(expr: Expr) -> Expr:
    """Multiply everything out — distribute products over sums and
    powers over the repeated product, then simplify the result."""
    return simplify(_expand(simplify(expr)))


def _expand(expr: Expr) -> Expr:
    """The distribution pass — bottom-up: children expand first."""
    if isinstance(expr, Add):
        return Add(tuple(_expand(t) for t in expr.terms))
    if isinstance(expr, Mul):
        # Fold flat and distribute each factor over a running sum of
        # terms — the product of sums is the sum of products.
        sums: list[list[Expr]] = []
        for f in expr.factors:
            f = _expand(f)
            sums.append(list(f.terms) if isinstance(f, Add) else [f])
        out: list[Expr] = [Num(1.0)]
        for terms in sums:
            out = [
                Mul((*p.factors, t)) if isinstance(p, Mul) else Mul((p, t))
                for p in out
                for t in terms
            ]
        return out[0] if len(out) == 1 else Add(tuple(out))
    if isinstance(expr, Pow):
        base = _expand(expr.base)
        exp = _expand(expr.exp)
        # (a + b + …)**n for small integer n → repeated product,
        # which the Mul branch distributes on the next pass.
        if (
            isinstance(exp, Num)
            and isinstance(base, Add)
            and exp.value == int(exp.value)
            and 2 <= exp.value <= 12
        ):
            acc: Expr = base
            for _ in range(int(exp.value) - 1):
                acc = _expand(Mul((acc, base)))
            return acc
        return Pow(base, exp)
    if isinstance(expr, Neg):
        return Neg(_expand(expr.x))
    if isinstance(expr, Call):
        return Call(expr.name, _expand(expr.arg))
    return expr


def evaluate(expr: Expr, env: dict[str, float] | None = None) -> float:
    """Evaluate the tree numerically — constants resolve, the env
    binds variables, and the function table computes calls."""
    env = env or {}
    if isinstance(expr, Num):
        return expr.value
    if isinstance(expr, Var):
        if expr.name in env:
            return env[expr.name]
        if expr.name in _CONSTANTS:
            return _CONSTANTS[expr.name]
        raise ValueError(f"unbound variable: {expr.name}")
    if isinstance(expr, Neg):
        return -evaluate(expr.x, env)
    if isinstance(expr, Add):
        return sum(evaluate(t, env) for t in expr.terms)
    if isinstance(expr, Mul):
        v = 1.0
        for f in expr.factors:
            v *= evaluate(f, env)
        return v
    if isinstance(expr, Pow):
        return evaluate(expr.base, env) ** evaluate(expr.exp, env)
    if isinstance(expr, Call):
        arg = evaluate(expr.arg, env)
        if expr.name == "factorial":
            if arg < 0 or arg != int(arg):
                raise ValueError("factorial needs a non-negative integer")
            return float(math.factorial(int(arg)))
        if expr.name not in _FUNCS:
            raise ValueError(f"unknown function: {expr.name}")
        return _FUNCS[expr.name](arg)
    raise TypeError(f"unknown node: {expr!r}")


def variables(expr: Expr) -> set[str]:
    """The non-constant variable names in a tree."""
    out: set[str] = set()

    def walk(e: Expr) -> None:
        if isinstance(e, Var):
            if e.name not in _CONSTANTS:
                out.add(e.name)
        elif isinstance(e, Neg):
            walk(e.x)
        elif isinstance(e, Add):
            for t in e.terms:
                walk(t)
        elif isinstance(e, Mul):
            for f in e.factors:
                walk(f)
        elif isinstance(e, Pow):
            walk(e.base)
            walk(e.exp)
        elif isinstance(e, Call):
            walk(e.arg)

    walk(expr)
    return out


# ─── Solving ────────────────────────────────────────────────────


def _poly_coeffs(expr: Expr, var: str) -> dict[int, float] | None:
    """Collect a tree's polynomial coefficients in ``var``.

    Returns ``{degree: coefficient}`` when every term is ``coef *
    var^d`` with non-negative integer d — or ``None`` when the tree
    is not a polynomial in that variable.
    """
    coeffs: dict[int, float] = {}

    def add_term(e: Expr, sign: float) -> bool:
        e = simplify(e)
        if isinstance(e, Neg):
            return add_term(e.x, -sign)
        if isinstance(e, Num):
            coeffs[0] = coeffs.get(0, 0.0) + sign * e.value
            return True
        if isinstance(e, Var):
            if e.name == var:
                coeffs[1] = coeffs.get(1, 0.0) + sign
                return True
            return e.name in _CONSTANTS and False
        if isinstance(e, Pow) and isinstance(e.base, Var) and e.base.name == var:
            if isinstance(e.exp, Num) and e.exp.value == int(e.exp.value) and e.exp.value >= 0:
                d = int(e.exp.value)
                coeffs[d] = coeffs.get(d, 0.0) + sign
                return True
            return False
        if isinstance(e, Mul):
            coef = 1.0
            degree = 0
            for f in e.factors:
                f = simplify(f)
                if isinstance(f, Num):
                    coef *= f.value
                elif isinstance(f, Var) and f.name == var:
                    degree += 1
                elif (
                    isinstance(f, Pow)
                    and isinstance(f.base, Var)
                    and f.base.name == var
                    and isinstance(f.exp, Num)
                    and f.exp.value == int(f.exp.value)
                    and f.exp.value >= 0
                ):
                    degree += int(f.exp.value)
                else:
                    return False
            coeffs[degree] = coeffs.get(degree, 0.0) + sign * coef
            return True
        if isinstance(e, Add):
            return all(add_term(t, sign) for t in e.terms)
        if isinstance(e, Call):
            return False
        return False

    return coeffs if add_term(expr, 1.0) else None


def _numeric_roots(expr: Expr, var: str) -> list[float]:
    """Approximate roots by sign-change scan + bisection polish —
    the fallback when closed form is out of reach (degree ≥ 3 or a
    transcendental mix). Deterministic: fixed grid, fixed bisection."""
    lo, hi = -1000.0, 1000.0
    n = 4000
    roots: list[float] = []

    def f(x: float) -> float:
        return evaluate(expr, {var: x})

    prev_x = lo
    try:
        prev_y = f(prev_x)
    except (ValueError, OverflowError, ZeroDivisionError):
        prev_y = float("nan")
    for i in range(1, n + 1):
        x = lo + (hi - lo) * i / n
        try:
            y = f(x)
        except (ValueError, OverflowError, ZeroDivisionError):
            prev_x, prev_y = x, float("nan")
            continue
        if math.isnan(y) or math.isinf(y):
            prev_x, prev_y = x, y
            continue
        if y == 0.0:
            roots.append(x)
        elif math.isfinite(prev_y) and prev_y * y < 0:
            a, b, fa = prev_x, x, prev_y
            for _ in range(80):
                m = (a + b) / 2
                fm = f(m)
                if fm == 0.0 or (b - a) < 1e-12:
                    break
                if fa * fm < 0:
                    b = m
                else:
                    a, fa = m, fm
            roots.append((a + b) / 2)
        prev_x, prev_y = x, y
    # Dedupe roots found twice at grid boundaries.
    roots.sort()
    out: list[float] = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1e-7:
            out.append(r)
    return out


def solve(text: str) -> AlgebraResult:
    """Solve an equation — the faculty's whole point.

    Closed forms where they exist (linear, quadratic); a numeric
    root-finding pass where they do not (higher degree,
    transcendental). The unknown is the equation's only non-constant
    variable; ``x`` is assumed when there is more than one.
    """
    try:
        left, right = parse_equation(text)
    except ParseError as e:
        return AlgebraResult("solve", text, "", False, str(e))
    diff = simplify(Add((left, Neg(right))))
    unknowns = sorted(variables(diff))
    if not unknowns:
        v = evaluate(diff)
        if v == 0.0:
            return AlgebraResult("solve", text, "true for all x", True, data={"roots": "all"})
        return AlgebraResult("solve", text, "no solution", True, data={"roots": []})
    var = "x" if "x" in unknowns else unknowns[0]
    if len(unknowns) > 1:
        return AlgebraResult(
            "solve", text, "", False, f"more than one unknown: {', '.join(unknowns)}"
        )

    coeffs = _poly_coeffs(diff, var)
    rendered = f"{render(diff)} = 0"
    if coeffs is not None:
        degree = max(coeffs)
        if degree == 0:
            return AlgebraResult(
                "solve",
                text,
                "no solution" if coeffs[0] != 0 else "true for all x",
                True,
                data={"roots": []},
            )
        if degree == 1:
            root = -coeffs.get(0, 0.0) / coeffs[1]
            return AlgebraResult(
                "solve",
                text,
                f"{var} = {_num_text(root)}",
                True,
                data={"roots": [root]},
            )
        if degree == 2:
            a, b, c = coeffs[2], coeffs.get(1, 0.0), coeffs.get(0, 0.0)
            disc = b * b - 4 * a * c
            if disc > 0:
                r1 = (-b + math.sqrt(disc)) / (2 * a)
                r2 = (-b - math.sqrt(disc)) / (2 * a)
                s1 = _surd_root(-b, disc, 2 * a)
                if s1 is not None:
                    shown = f"{var} = {s1} or {var} = {_negate_surd(s1)}"
                else:
                    shown = f"{var} = {_num_text(_dust(r1))} or {var} = {_num_text(_dust(r2))}"
                return AlgebraResult(
                    "solve",
                    text,
                    shown,
                    True,
                    data={"roots": sorted([r1, r2])},
                )
            if disc == 0:
                r = -b / (2 * a)
                return AlgebraResult(
                    "solve", text, f"{var} = {_num_text(r)}", True, data={"roots": [r]}
                )
            re = -b / (2 * a)
            im = math.sqrt(-disc) / (2 * a)
            return AlgebraResult(
                "solve",
                text,
                f"{var} = {_num_text(_dust(re))} ± {_num_text(_dust(abs(im)))}i",
                True,
                data={"roots": [], "complex": (re, im)},
            )

    return _finish_solve(diff, var, text, rendered)


def _dust(v: float) -> float:
    """Round float dust at the 12th digit — ``1.4142135623730951``
    is honest arithmetic but not how a root is read."""
    r = round(v, 12)
    return 0.0 if r == 0 else r


def _surd_root(num: float, disc: float, den: float) -> str | None:
    """Render ``(num + √disc)/den`` exactly when the parts are whole —
    ``(-0 + √8)/2`` is ``sqrt(2)``, not ``1.4142135623730951``.
    Returns None when the surd is not exact."""
    if not (
        num == int(num) and den == int(den) and disc == int(disc) and 0 < disc < 1e12 and den != 0
    ):
        return None
    n, q, d = int(num), int(disc), int(den)
    k = math.isqrt(q)
    if k * k == q:
        return None  # rational root — _num_text already said it
    # Pull the largest square factor out of the radical.
    m = math.isqrt(q)
    sq = 1
    for i in range(m, 1, -1):
        if q % (i * i) == 0:
            sq = i
            break
    k, q2 = sq, q // (sq * sq)
    g = math.gcd(math.gcd(abs(n), k), abs(d))
    n, k, d = n // g, k // g, d // g
    radical = f"{k}*sqrt({q2})" if k != 1 else f"sqrt({q2})"
    top = f"{n} + {radical}" if n else radical
    if d == 1:
        return top
    return f"({top})/{d}"


def _negate_surd(s: str) -> str:
    """Flip a surd rendering's sign — ``sqrt(2)`` ↔ ``-sqrt(2)``,
    ``(2 + sqrt(3))/4`` ↔ ``(2 - sqrt(3))/4``."""
    if " + " in s:
        return s.replace(" + ", " - ", 1)
    return f"-{s}"


def _finish_solve(diff: Expr, var: str, text: str, rendered: str) -> AlgebraResult:
    """The numeric tail of solve — sign-change scan when the closed
    forms do not apply."""
    roots = _numeric_roots(diff, var)
    if not roots:
        return AlgebraResult("solve", rendered, "no real roots found", True, data={"roots": []})
    text_roots = " or ".join(f"{var} ≈ {_num_text(round(r, 8))}" for r in roots)
    return AlgebraResult("solve", rendered, text_roots, True, data={"roots": roots})


def simplify_text(text: str) -> AlgebraResult:
    """Simplify expression text to canonical form."""
    try:
        out = simplify(parse(text))
    except ParseError as e:
        return AlgebraResult("simplify", text, "", False, str(e))
    return AlgebraResult("simplify", text, render(out), True)


def expand_text(text: str) -> AlgebraResult:
    """Expand expression text — distribute products over sums."""
    try:
        out = expand(parse(text))
    except ParseError as e:
        return AlgebraResult("expand", text, "", False, str(e))
    return AlgebraResult("expand", text, render(out), True)


def factor_text(text: str) -> AlgebraResult:
    """Factor a polynomial into linear/quadratic pieces — the
    inverse of expand. Rational roots are found by the rational
    root theorem; what remains unfactored is kept whole."""
    try:
        expr = simplify(parse(text))
    except ParseError as e:
        return AlgebraResult("factor", text, "", False, str(e))
    unknowns = sorted(variables(expr))
    if len(unknowns) != 1:
        return AlgebraResult("factor", text, render(expr), True)
    var = unknowns[0]
    coeffs = _poly_coeffs(expr, var)
    if coeffs is None:
        return AlgebraResult("factor", text, render(expr), True)
    degree = max(coeffs, default=0)
    if degree < 2:
        return AlgebraResult("factor", text, render(expr), True)

    # Deflate by each rational root found — candidates are p/q with
    # p dividing the constant term and q the leading coefficient.
    remaining = dict(coeffs)
    factors: list[Expr] = []
    c0 = remaining.get(0, 0.0)
    an = remaining.get(max(remaining), 1.0)

    def eval_poly(x: float) -> float:
        return sum(c * x**d for d, c in remaining.items())

    if c0 == int(c0) and an == int(an) and c0 != 0:
        ps = {abs(p) for i in range(1, int(abs(c0)) + 1) if int(c0) % i == 0 for p in (i,)}
        qs = {i for i in range(1, int(abs(an)) + 1) if int(an) % i == 0}
        candidates = sorted(
            {s * Fraction(p, q) for p in ps for q in qs for s in (1, -1)}, key=float
        )
    else:
        candidates = []
    for cand in candidates:
        r = float(cand)
        while max(remaining, default=0) >= 1 and abs(eval_poly(r)) < 1e-9:
            # Synthetic division — deflate by (x - r); missing degrees
            # carry a zero coefficient through the chain.
            new: dict[int, float] = {}
            carry = 0.0
            for d in range(max(remaining), -1, -1):
                carry = remaining.get(d, 0.0) + carry * r
                if d > 0:
                    new[d - 1] = carry
            if abs(carry) > 1e-7:
                break
            remaining = new
            factors.append(Add((Var(var), Num(-r))))
        if max(remaining, default=0) < 2:
            break
    degree_left = max(remaining, default=0)
    if degree_left == 1:
        # A leftover ax + b is a·(x + b/a) — the a joins the lead.
        a1 = remaining[1]
        factors.append(Add((Var(var), Num(remaining.get(0, 0.0) / a1))))
        lead = a1
    elif degree_left >= 2:
        leftover = Add(
            tuple(
                Mul((Num(c), Pow(Var(var), Num(float(d)))))
                for d, c in sorted(remaining.items(), reverse=True)
                if c != 0.0
            )
        )
        factors.append(simplify(leftover))
        lead = 1.0  # the leftover factor carries its own coefficient
    else:
        lead = remaining.get(0, 1.0)
    # Rebuild: leading coefficient × product of factors.
    parts: list[Expr] = []
    if not factors:
        return AlgebraResult("factor", text, render(expr), True)
    if abs(lead - 1.0) > 1e-12:
        parts.append(Num(lead))
    parts.extend(factors)
    out = simplify(parts[0] if len(parts) == 1 else Mul(tuple(parts)))
    return AlgebraResult("factor", text, render(out), True)
