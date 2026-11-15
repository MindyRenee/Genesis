"""Calculus — the parietal faculty's mathematics of change.

Differentiation and integration over the expression trees of
``algebra.py``: the rules are the standard elementary table applied
to structure — power, product, quotient, chain, and the elementary
function derivatives — plus Simpson integration when no closed
antiderivative is found. ``symbolic.py`` remains the oracle; these
results are hers, verifiable, not asserted.
"""

from __future__ import annotations

import math

from .algebra import (
    Add,
    AlgebraResult,
    Call,
    Expr,
    Mul,
    Neg,
    Num,
    ParseError,
    Pow,
    Var,
    _num_text,
    evaluate,
    parse,
    render,
    simplify,
    variables,
)

# ─── Differentiation ────────────────────────────────────────────
#
# The elementary rule table, applied to structure. Every rule is
# chain-rule aware: the operand's own derivative multiplies in.


def _is_const(expr: Expr, var: str) -> bool:
    """Whether the tree holds no trace of ``var`` — named constants
    (pi, e, tau) count as constants by name."""
    return var not in variables(expr)


def differentiate(expr: Expr, var: str = "x") -> Expr:
    """The derivative of a tree with respect to ``var``.

    The rules, in structure order:

        d(c)      = 0                     d(x)      = 1
        d(u + v)  = u' + v'               d(-u)     = -u'
        d(uv)     = u'v + uv'             d(u**n)   = n u**(n-1) u'
        d(u**v)   = u**v (v' ln u + v u'/u)
        d(sin u)  =  cos u · u'           d(cos u)  = -sin u · u'
        d(tan u)  =  sec²u · u'           d(exp u)  = exp u · u'
        d(ln u)   =  u' / u               d(sqrt u) = u' / (2 sqrt u)
    """
    if isinstance(expr, Num):
        return Num(0.0)
    if isinstance(expr, Var):
        if expr.name == var:
            return Num(1.0)
        return Num(0.0)  # other variables and named constants
    if isinstance(expr, Neg):
        return Neg(differentiate(expr.x, var))
    if isinstance(expr, Add):
        return Add(tuple(differentiate(t, var) for t in expr.terms))
    if isinstance(expr, Mul):
        # n-ary product rule: the derivative of a product is the sum
        # over factors of (that factor's derivative times the rest).
        terms = []
        for i, f in enumerate(expr.factors):
            rest = expr.factors[:i] + expr.factors[i + 1 :]
            df = differentiate(f, var)
            if isinstance(df, Num) and df.value == 0.0:
                continue
            terms.append(Mul((df, *rest)) if rest else df)
        if not terms:
            return Num(0.0)
        return terms[0] if len(terms) == 1 else Add(tuple(terms))
    if isinstance(expr, Pow):
        b, e = expr.base, expr.exp
        if isinstance(e, Num) or _is_const(e, var):
            # Constant exponent: d(u**n) = n u**(n-1) u'
            n = e
            coeff = Mul((n, Pow(b, Add((n, Num(-1.0)))), differentiate(b, var)))
            return simplify(coeff)
        if isinstance(b, Num) or _is_const(b, var):
            # Constant base: d(a**v) = a**v · ln a · v'
            return simplify(Mul((expr, Call("ln", b), differentiate(e, var))))
        # General: d(u**v) = u**v · (v' ln u + v u'/u)
        return simplify(
            Mul(
                (
                    expr,
                    Add(
                        (
                            Mul((differentiate(e, var), Call("ln", b))),
                            Mul((e, differentiate(b, var), Pow(b, Num(-1.0)))),
                        )
                    ),
                )
            )
        )
    if isinstance(expr, Call):
        u = expr.arg
        du = differentiate(u, var)
        if isinstance(du, Num) and du.value == 0.0:
            return Num(0.0)
        name = expr.name
        if name == "sin":
            return simplify(Mul((Call("cos", u), du)))
        if name == "cos":
            return simplify(Neg(Mul((Call("sin", u), du))))
        if name == "tan":
            # sec²u = 1/cos²u — render as cos(u)**(-2).
            return simplify(Mul((Pow(Call("cos", u), Num(-2.0)), du)))
        if name == "exp":
            return simplify(Mul((Call("exp", u), du)))
        if name == "ln":
            return simplify(Mul((du, Pow(u, Num(-1.0)))))
        if name == "sqrt":
            return simplify(Mul((du, Num(0.5), Pow(u, Num(-0.5)))))
        if name == "log":
            # log10 u = ln u / ln 10
            return simplify(Mul((du, Pow(u, Num(-1.0)), Num(1.0 / math.log(10.0)))))
        if name == "asin":
            # 1/sqrt(1 - u²)
            return simplify(Mul((du, Pow(Add((Num(1.0), Neg(Pow(u, Num(2.0))))), Num(-0.5)))))
        if name == "acos":
            return simplify(Neg(Mul((du, Pow(Add((Num(1.0), Neg(Pow(u, Num(2.0))))), Num(-0.5))))))
        if name == "atan":
            return simplify(Mul((du, Pow(Add((Num(1.0), Pow(u, Num(2.0)))), Num(-1.0)))))
        if name == "sinh":
            return simplify(Mul((Call("cosh", u), du)))
        if name == "cosh":
            return simplify(Mul((Call("sinh", u), du)))
        if name == "tanh":
            # sech²u = 1/cosh²u
            return simplify(Mul((Pow(Call("cosh", u), Num(-2.0)), du)))
        if name == "abs":
            return simplify(Mul((du, Call("abs", u), Pow(u, Num(-1.0)))))
        raise ValueError(f"no derivative rule for: {name}")
    raise TypeError(f"unknown node: {expr!r}")


# ─── Integration ────────────────────────────────────────────────
#
# The antiderivative table: powers, the elementary functions with a
# linear inner term (the "reverse chain" covering a*x+b inside), and
# linearity — sums integrate termwise, constants factor out. What
# the table does not cover falls to ``integrate_between``'s numeric
# pass rather than a wrong closed form.


def _linear_inner(expr: Expr, var: str) -> tuple[float, float] | None:
    """If ``expr`` is a*x + b in ``var``, return (a, b)."""
    e = simplify(expr)
    if isinstance(e, Var) and e.name == var:
        return 1.0, 0.0
    if isinstance(e, Mul) and len(e.factors) == 2:
        (c, v) = e.factors if isinstance(e.factors[0], Num) else (e.factors[1], e.factors[0])
        if isinstance(c, Num) and isinstance(v, Var) and v.name == var:
            return c.value, 0.0
    if isinstance(e, Add):
        a, b = 0.0, 0.0
        for t in e.terms:
            t = simplify(t)
            lin = _linear_inner(t, var)
            if lin is not None:
                a += lin[0]
                b += lin[1]
                continue
            if _is_const(t, var):
                b += evaluate(t)
                continue
            return None
        return a, b
    if isinstance(e, Neg):
        lin = _linear_inner(e.x, var)
        if lin is not None:
            return -lin[0], -lin[1]
    return None


def integrate(expr: Expr, var: str = "x") -> Expr | None:
    """The antiderivative, or None when the table has no rule —
    the caller then falls back to numeric integration."""
    e = simplify(expr)
    if isinstance(e, Num):
        return Mul((e, Var(var)))
    if isinstance(e, Var):
        if e.name == var:
            return Mul((Num(0.5), Pow(e, Num(2.0))))
        return Mul((e, Var(var)))  # a different variable is constant
    if isinstance(e, Neg):
        inner = integrate(e.x, var)
        return Neg(inner) if inner is not None else None
    if isinstance(e, Add):
        parts = [integrate(t, var) for t in e.terms]
        if any(p is None for p in parts):
            return None
        return Add(tuple(p for p in parts if p is not None))
    if isinstance(e, Mul):
        const_factors = tuple(f for f in e.factors if _is_const(f, var))
        var_factors = tuple(f for f in e.factors if not _is_const(f, var))
        if not var_factors:
            return Mul((*e.factors, Var(var)))
        if len(var_factors) == 1:
            inner = integrate(var_factors[0], var)
            if inner is None:
                return None
            return Mul((*const_factors, inner)) if const_factors else inner
        return None
    if isinstance(e, Pow):
        lin = _linear_inner(e.base, var)
        if isinstance(e.exp, Num):
            n = e.exp.value
            if lin is not None:
                a, _b = lin
                if a == 0.0:
                    return Mul((e, Var(var)))
                if n == -1.0:
                    return Mul((Num(1.0 / a), Call("ln", Call("abs", e.base))))
                return Mul((Num(1.0 / (a * (n + 1.0))), Pow(e.base, Num(n + 1.0))))
        return None
    if isinstance(e, Call):
        lin = _linear_inner(e.arg, var)
        if lin is None or lin[0] == 0.0:
            return None
        a = lin[0]
        u = e.arg
        if e.name == "sin":
            return Mul((Num(-1.0 / a), Call("cos", u)))
        if e.name == "cos":
            return Mul((Num(1.0 / a), Call("sin", u)))
        if e.name == "exp":
            return Mul((Num(1.0 / a), Call("exp", u)))
        if e.name == "sinh":
            return Mul((Num(1.0 / a), Call("cosh", u)))
        if e.name == "cosh":
            return Mul((Num(1.0 / a), Call("sinh", u)))
        if e.name == "tan":
            # ∫tan(ax+b) = -ln|cos(ax+b)|/a
            return Mul((Num(-1.0 / a), Call("ln", Call("abs", Call("cos", u)))))
        return None
    return None


# ─── Numeric integration — Simpson's rule ───────────────────────

_SIMPSON_SLICES = 4096


def integrate_between(expr: Expr, var: str, lo: float, hi: float) -> float:
    """Definite integral by composite Simpson's rule — the honest
    numeric path when the antiderivative table is silent. A fixed
    even slice count keeps the estimate deterministic."""
    n = _SIMPSON_SLICES + (_SIMPSON_SLICES % 2)
    h = (hi - lo) / n

    def f(x: float) -> float:
        return evaluate(expr, {var: x})

    total = f(lo) + f(hi)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(lo + i * h)
    return total * h / 3


# ─── Text front ends — the faculty's surface ────────────────────


def derivative_text(text: str, var: str = "x") -> AlgebraResult:
    """Differentiate expression text; render the simplified result."""
    try:
        d = simplify(differentiate(parse(text), var))
    except (ParseError, ValueError) as e:
        return AlgebraResult("derivative", text, "", False, str(e))
    return AlgebraResult("derivative", text, render(d), True)


def derivative_at_text(text: str, var: str, at: float) -> AlgebraResult:
    """The derivative evaluated at a point — the slope there."""
    try:
        d = simplify(differentiate(parse(text), var))
        v = evaluate(d, {var: at})
    except (ParseError, ValueError, OverflowError, ZeroDivisionError) as e:
        return AlgebraResult("derivative_at", text, "", False, str(e))
    return AlgebraResult("derivative_at", text, _num_text(v), True, data={"value": v})


def integral_text(text: str, var: str = "x") -> AlgebraResult:
    """The antiderivative — '+ C' is the truth, said plainly."""
    try:
        anti = integrate(parse(text), var)
    except (ParseError, ValueError) as e:
        return AlgebraResult("integral", text, "", False, str(e))
    if anti is None:
        return AlgebraResult("integral", text, "", False, "no closed antiderivative found")
    return AlgebraResult("integral", text, f"{render(simplify(anti))} + C", True)


def integral_between_text(text: str, var: str, lo: float, hi: float) -> AlgebraResult:
    """The definite integral — the closed form when the table has
    it, Simpson's estimate when it does not."""
    try:
        e = parse(text)
        anti = integrate(e, var)
        if anti is not None:
            v = evaluate(anti, {var: hi}) - evaluate(anti, {var: lo})
            exact = True
        else:
            v = integrate_between(e, var, lo, hi)
            exact = False
    except (ParseError, ValueError, OverflowError, ZeroDivisionError) as e:
        return AlgebraResult("definite_integral", text, "", False, str(e))
    mark = "=" if exact else "≈"
    return AlgebraResult(
        "definite_integral",
        text,
        f"{_num_text(v)}",
        True,
        data={"value": v, "exact": exact, "mark": mark},
    )


def taylor(expr: Expr, var: str, at: float, order: int) -> Expr:
    """The Taylor polynomial — f(a) + f'(a)(x-a) + f''(a)/2! (x-a)²
    + … to ``order``. Each term differentiates the last — the series
    is the faculty composing itself."""
    terms: list[Expr] = []
    current = expr
    da = Add((Var(var), Num(-at)))
    for k in range(order + 1):
        coeff = evaluate(simplify(current), {var: at}) / math.factorial(k)
        if coeff != 0.0:
            term: Expr = Num(coeff) if k == 0 else Mul((Num(coeff), Pow(da, Num(float(k)))))
            terms.append(term)
        current = differentiate(current, var)
    return simplify(Add(tuple(terms)))
